"""Audit 2026-10-02 DETECT-08 (item 95 residual): the ENDING energy gate vs a flat standby.

Once the quiet timer passes the fallback wait, the energy gate adds "mean sub-stop
power <= end_energy_threshold x 3600 / off_delay": 1.0 W on stock settings. A
stock washer (min_power 2 W -> stop 1.2 W) holding a flat 1.1 W standby therefore
never ended unmatched - 8 h cap, `force_stopped`, 421 min after the wash - and a
matched-but-ambiguous run waited for the 2x-expected hard finalize (60 min late).

The gate is not deleted: its one measured catch is a dishwasher's passive drying
that flickers 0-0.5 W for ~25 min before the final pump-out (Hatton ECO
`44d35b4ca01e`, off_delay 1800 -> a 0.1 W bar). Only a FLAT, non-zero window
skips it: just under stop after the full un-shortened max(off_delay,
min_off_gap), anything lower after 2 h, because flat phases at about half of stop
do resume in real cycles (an 18.9 min washer soak, an 80.8 min dishwasher drying
phase - both reproduced below).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from custom_components.ha_washdata.const import TerminationReason
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _ambiguous(expected: float = 3600.0):
    return ("P", 0.8, expected, None, False, True, False, False, None, None, None, 0.0, None)


class _Run:
    def __init__(self, options: dict, device_type: str, matcher=None) -> None:
        self.cfg = build_detector_config(options, {}, device_type)
        self.ends: list[dict] = []
        self.det = CycleDetector(
            self.cfg, lambda *_: None, self.ends.append, profile_matcher=matcher
        )
        self.t = 0.0
        self._clock = patch(
            "homeassistant.util.dt.now",
            side_effect=lambda: T0 + timedelta(seconds=self.t),
        )

    def __enter__(self) -> "_Run":
        self._clock.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._clock.stop()

    def feed(self, powers, until: float, step: float) -> None:
        """Feed ``powers`` (a float, or a cycle of floats) until ``until`` seconds."""
        seq = powers if isinstance(powers, (list, tuple)) else (powers,)
        i = 0
        while self.t < until and not self.ends:
            self.det.process_reading(seq[i % len(seq)], T0 + timedelta(seconds=self.t))
            self.t += step
            i += 1


@pytest.mark.parametrize("matched", [False, True], ids=["unmatched", "matched-ambiguous"])
def test_flat_standby_just_under_stop_ends_the_cycle(matched):
    with _Run({"min_power": 2.0}, "washing_machine",
              (lambda _r: _ambiguous()) if matched else None) as run:
        assert run.cfg.stop_threshold_w == pytest.approx(1.2)
        run.feed(500.0, 3600, 10.0)
        run.feed(1.1, 3600 + 9 * 3600, 30.0)
    assert len(run.ends) == 1
    end = run.ends[0]
    # Before the fix: unmatched closed 421 min after the wash (force_stopped);
    # matched-ambiguous 60 min after it (the 2x hard finalize).
    assert end["status"] == "completed"
    assert end["termination_reason"] != TerminationReason.FORCE_STOPPED
    assert run.t - 3600 <= 15 * 60
    assert end["duration"] <= 61 * 60


def test_standby_with_jitter_still_counts_as_flat():
    with _Run({"min_power": 2.0}, "washing_machine") as run:
        run.feed(500.0, 3600, 10.0)
        run.feed([1.05, 1.1, 1.15, 1.1], 3600 + 9 * 3600, 30.0)
    assert [e["status"] for e in run.ends] == ["completed"]
    assert run.t - 3600 <= 15 * 60


def test_flat_standby_waits_the_full_min_off_gap():
    opts = {"min_power": 2.0, "min_off_gap": 1800}
    with _Run(opts, "washing_machine") as run:
        run.feed(500.0, 3600, 10.0)
        run.feed(1.1, 3600 + 1700, 30.0)
        assert run.ends == []
        run.feed(1.1, 3600 + 9 * 3600, 30.0)
    assert [e["status"] for e in run.ends] == ["completed"]
    assert run.t - 3600 <= 35 * 60


def test_flickering_drying_phase_still_holds_until_the_pump_out():
    """The Hatton ECO shape: drying flickers 0/0.5 W under a 1.5 W stop, then pumps."""
    opts = {
        "min_power": 2.0, "stop_threshold_w": 1.5, "off_delay": 1800,
        "min_off_gap": 3600,
    }
    with _Run(opts, "dishwasher") as run:
        run.feed(1800.0, 3 * 3600, 10.0)
        # Two hours of flicker, well past max(off_delay, min_off_gap).
        run.feed([0.0, 0.5, 0.0, 0.4], 3 * 3600 + 2 * 3600, 10.0)
        assert run.ends == [], run.ends and run.ends[0]["termination_reason"]
        run.feed(17.0, 3 * 3600 + 2 * 3600 + 60, 5.0)   # pump-out
        run.feed(0.0, 3 * 3600 + 5 * 3600, 30.0)
    assert len(run.ends) == 1
    assert run.ends[0]["duration"] >= 5 * 3600 - 120


def test_flat_window_with_a_zero_reading_is_not_flat():
    opts = {"min_power": 2.0, "stop_threshold_w": 1.5, "off_delay": 1800,
            "min_off_gap": 3600}
    with _Run(opts, "dishwasher") as run:
        run.feed(1800.0, 3 * 3600, 10.0)
        run.feed([0.5, 0.5, 0.5, 0.0], 3 * 3600 + 3 * 3600, 30.0)
    assert run.ends == []


def test_flat_standby_far_under_stop_ends_after_two_hours():
    # min_power 10 W -> stop 6 W: the bar is still 1.0 W, so a 2.8 W standby pinned
    # the gate until the 8 h cap.
    with _Run({"min_power": 10.0}, "washing_machine") as run:
        assert run.cfg.stop_threshold_w == pytest.approx(6.0)
        run.feed(500.0, 3600, 10.0)
        run.feed(2.8, 3600 + 9 * 3600, 30.0)
    assert [e["status"] for e in run.ends] == ["completed"]
    assert 115 * 60 <= run.t - 3600 <= 125 * 60
    assert run.ends[0]["duration"] <= 61 * 60


def test_a_19_minute_soak_at_half_of_stop_does_not_split():
    """Kenroy Morgan `b51a6b32c258`: 16 min heating, 19 min at 2.7-3.0 W, 25 min wash."""
    opts = {"min_power": 10.0, "off_delay": 300, "min_off_gap": 480}
    with _Run(opts, "washing_machine") as run:
        run.feed(500.0, 1000, 10.0)
        run.feed([2.8, 2.9, 3.0, 2.7], 1000 + 19 * 60, 30.0)
        run.feed(500.0, 1000 + 19 * 60 + 1500, 10.0)
        run.feed(0.0, 1000 + 19 * 60 + 1500 + 3600, 30.0)
    assert len(run.ends) == 1
    assert run.ends[0]["duration"] >= 1000 + 19 * 60 + 1400


def test_an_80_minute_flat_drying_phase_does_not_split_an_unmatched_dishwasher():
    """James Hayes Eco `2a49888eb800`: 80.8 min at 0.7-0.9 W on a 1.5 W stop, then pumps."""
    opts = {"min_power": 2.5, "stop_threshold_w": 1.5, "off_delay": 1800, "min_off_gap": 2000}
    with _Run(opts, "dishwasher") as run:
        run.feed(1800.0, 6000, 10.0)
        run.feed([0.8, 0.9, 0.7, 0.8], 6000 + 81 * 60, 30.0)
        run.feed(30.0, 6000 + 81 * 60 + 120, 10.0)   # pump-out
        run.feed(0.0, 6000 + 81 * 60 + 120 + 3 * 3600, 30.0)
    assert len(run.ends) == 1
    assert run.ends[0]["duration"] >= 6000 + 81 * 60
