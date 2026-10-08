"""Audit 2026-10-02 DETECT-03: a standby plateau that begins in ENDING was never closed.

The standby-band finalize (#296/#445) ran only in RUNNING. A washer that dropped to
0 W long enough to reach ENDING and THEN settled on a flat standby above
stop_threshold kept every reading "high": after 120 s in ENDING each one was kept
as a terminal spike and returned, resetting the quiet timer, so neither the
fallback nor the plateau check could end it - the 8 h force-stop did, stored
`force_stopped` and 480 min long.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from custom_components.ha_washdata.const import STATE_ENDING, TerminationReason
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
EXPECTED = 3600.0
PLATEAU_W = 2.0  # above the 1.2 W stop threshold min_power 2 W gives


def _match(tail: float | None = PLATEAU_W):
    return ("Quick", 0.8, EXPECTED, None, False, False, False, False, tail, None, None, 0.0, None)


def _run(gap_s: float, *, user_paused: bool = False):
    """60 min wash, ``gap_s`` at 0 W, then a 2 W standby for up to 8.3 h."""
    cfg = build_detector_config({"min_power": 2.0}, {}, "washing_machine")
    ends: list[dict] = []
    det = CycleDetector(cfg, lambda *_: None, ends.append, profile_matcher=lambda _r: _match())
    t = [0.0]
    state_at_plateau = None
    with patch("homeassistant.util.dt.now", side_effect=lambda: T0 + timedelta(seconds=t[0])):
        def feed(power: float, until: float) -> None:
            while t[0] < until and not ends:
                det.process_reading(power, T0 + timedelta(seconds=t[0]))
                t[0] += 10.0

        feed(500.0, 3550)
        feed(0.0, 3550 + gap_s)
        state_at_plateau = det.state
        if user_paused:
            det.set_user_paused(True)
            det.set_verified_pause(True)
        feed(PLATEAU_W, 3550 + gap_s + 30000)
    return ends, state_at_plateau, t[0], det


def test_plateau_after_a_dip_into_ending_is_finalised_like_one_from_running():
    direct, _, direct_t, _ = _run(0)
    ends, state, closed_t, _ = _run(60)
    assert state == STATE_ENDING
    assert len(ends) == 1
    end = ends[0]
    # Before the fix: closed at 480 min, status force_stopped, duration 480.2 min.
    assert end["status"] == "completed", end["termination_reason"]
    assert end["termination_reason"] == TerminationReason.TIMEOUT
    assert closed_t <= 75 * 60
    # The plateau (and the 0 W dip before it) is trimmed off the stored record.
    assert end["duration"] <= 60 * 60
    assert closed_t == direct_t and end["duration"] == direct[0]["duration"]


def test_longer_dip_into_ending_also_closes():
    ends, state, closed_t, _ = _run(200)
    assert state == STATE_ENDING
    assert [e["status"] for e in ends] == ["completed"]
    assert closed_t <= 75 * 60


def test_a_user_pause_still_holds_the_plateau_in_ending():
    ends, state, _, _det = _run(60, user_paused=True)
    assert state == STATE_ENDING
    # Only the 8 h cap may end it; the standby band honours the pause.
    assert [e["termination_reason"] for e in ends] in ([], [TerminationReason.FORCE_STOPPED])
