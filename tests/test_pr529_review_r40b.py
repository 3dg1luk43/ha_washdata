"""Review of PR #529, round 40b, each reproduced.

* The end-gate guard read a wait of 0 (off_delay and min_off_gap both 0) as no
  gate, so Apply accepted it; live, the first quiet reading in ENDING completes
  the cycle, splitting it at any pause that gets there. An off_delay of 0 is a
  one-reading energy window, which nothing holds.
* The kept standby tail stopped at a synthetic reading at working power, so a
  real 3 W tail behind a 20 W keepalive was cut and the 6 W after it read as the
  idle level: two such cycles raised "still drawing" at a 5 W stop.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from custom_components.ha_washdata.const import (
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_OFF_GAP,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.suggestion_engine import (
    detect_standby_above_stop,
    end_gate_would_split,
)
from custom_components.ha_washdata.time_utils import SYNTHETIC_OFFSETS_KEY

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _paused_trace(pause_w: float, pause_s: int, step: int = 10) -> list[tuple[float, float]]:
    """300 W for 20 min, a pause at ``pause_w``, 300 W for 20 min, then 0 W."""
    pts: list[tuple[float, float]] = []
    t = 0
    for _ in range(1200 // step):
        pts.append((float(t), 300.0))
        t += step
    for _ in range(pause_s // step):
        pts.append((float(t), pause_w))
        t += step
    for _ in range(1200 // step):
        pts.append((float(t), 300.0))
        t += step
    for _ in range(2000 // step):
        pts.append((float(t), 0.0))
        t += step
    return pts


def _replay(options: dict[str, Any], pts: list[tuple[float, float]], device_type: str) -> list[Any]:
    """The cycles the shipped detector stores for ``pts`` under ``options``."""
    cfg = build_detector_config(options, {}, device_type)
    ends: list[Any] = []
    det = CycleDetector(cfg, lambda *_a: None, ends.append)
    for offset, watts in pts:
        det.process_reading(watts, T0 + timedelta(seconds=offset))
    return ends


# ── A wait of 0 is a gate ────────────────────────────────────────────────────


_ZERO_OPTS = {
    CONF_STOP_THRESHOLD_W: 2.0, CONF_START_THRESHOLD_W: 5.0, "min_power": 3.0,
    CONF_END_ENERGY_THRESHOLD: 0.05,
}


def test_a_zero_wait_splits_live_and_the_guard_says_so() -> None:
    """off_delay 0 and min_off_gap 0: the detector stores two cycles for one with a
    5 min pause at 1 W, and the guard reports it. 600 s each bridges it, in both."""
    pts = _paused_trace(1.0, 300)
    zero = {**_ZERO_OPTS, CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 0}
    assert len(_replay(zero, pts, "washing_machine")) == 2
    assert end_gate_would_split(pts, 2.0, 0, 0, 0.05)
    bridged = {**_ZERO_OPTS, CONF_OFF_DELAY: 600, CONF_MIN_OFF_GAP: 600}
    assert len(_replay(bridged, pts, "washing_machine")) == 1
    assert not end_gate_would_split(pts, 2.0, 600, 600, 0.05)


def test_a_zero_off_delay_holds_no_energy() -> None:
    """off_delay 0 under a 60 s gap: the one-reading window passes the energy gate,
    so the 300 s pause splits live; under a 600 s gap it does not, in both."""
    pts = _paused_trace(1.0, 300)
    short = {**_ZERO_OPTS, CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 60}
    assert len(_replay(short, pts, "washing_machine")) == 2
    assert end_gate_would_split(pts, 2.0, 0, 60, 0.05)
    long_gap = {**_ZERO_OPTS, CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 600}
    assert len(_replay(long_gap, pts, "washing_machine")) == 1
    assert not end_gate_would_split(pts, 2.0, 0, 600, 0.05)


def test_a_zero_wait_without_a_pause_splits_nothing() -> None:
    """No pause, only the trailing quiet: a wait of 0 splits nothing."""
    pts = _paused_trace(1.0, 0)
    assert len(_replay({**_ZERO_OPTS, CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 0}, pts, "washing_machine")) == 1
    assert not end_gate_would_split(pts, 2.0, 0, 0, 0.05)


# ── A synthetic reading does not cut the kept tail ───────────────────────────


def _stopped_by_hand(cid: str) -> dict[str, Any]:
    """200 W for 10 min, 30 min at 3 W, a 20 W keepalive WashData wrote, then
    two minutes at 6 W until the user stopped it."""
    rows = [[float(t), 200.0] for t in range(0, 600, 10)]
    rows += [[float(t), 3.0] for t in range(600, 2400, 10)]
    rows.append([2400.0, 20.0])
    rows += [[float(t), 6.0] for t in range(2410, 2530, 10)]
    return {
        "id": cid, "status": "interrupted", "termination_reason": "user",
        "power_data": rows, SYNTHETIC_OFFSETS_KEY: [2400.0],
    }


def test_a_synthetic_reading_does_not_cut_the_standby_tail() -> None:
    """At a 5 W stop the tail settled at 3 W, below it; cut at the keepalive it
    read 6 W, and two such cycles raised the advisory."""
    cycles = [_stopped_by_hand("a"), _stopped_by_hand("b")]
    assert detect_standby_above_stop(cycles, 5.0) is None
    # A real 20 W reading still ends the tail: the 6 W after it is the evidence.
    for cycle in cycles:
        cycle[SYNTHETIC_OFFSETS_KEY] = []
    found = detect_standby_above_stop(cycles, 5.0)
    assert found is not None and found["idle_w"] == 6.0
