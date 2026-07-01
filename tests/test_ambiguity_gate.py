"""Gate predictive end (Smart Termination) when the live match is ambiguous.

When top-1 and top-2 profiles score within MATCH_AMBIGUITY_MARGIN, the matched
profile's expected duration is unreliable, so the predictive Smart Termination
must NOT fire - the cycle should fall through to the power-based fallback.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    STATE_ENDING,
)
from custom_components.ha_washdata.const import DEVICE_TYPE_WASHING_MACHINE, TerminationReason

BASE = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


def _ts(s: float) -> datetime:
    return BASE + timedelta(seconds=s)


def _det_in_ending(*, ambiguous: bool, expected: float = 3600.0, conf: float = 0.7):
    cfg = CycleDetectorConfig(
        min_power=2.0, off_delay=180, stop_threshold_w=2.0, start_threshold_w=5.0,
        device_type=DEVICE_TYPE_WASHING_MACHINE, min_off_gap=300, completion_min_seconds=60,
        end_energy_threshold=0.05, start_energy_threshold=0.05, start_duration_threshold=1.0,
    )
    completed: list[dict] = []
    det = CycleDetector(config=cfg, on_state_change=lambda a, b: None,
                        on_cycle_end=lambda d: completed.append(d))
    # Drive the match through update_match so the ambiguity flag is set the
    # same way the manager sets it (6-element tuple).
    det.update_match(("Cotton", conf, expected, None, False, ambiguous))
    det._state = STATE_ENDING
    det._current_cycle_start = _ts(0)
    det._state_enter_time = _ts(3300)
    det._time_in_state = 200.0            # > WM smart_debounce (120s)
    det._time_below_threshold = 0.0       # below off_delay -> fallback won't fire
    det._last_reading_time = _ts(3564 - 30)
    return det, completed


def test_smart_termination_fires_when_not_ambiguous() -> None:
    det, completed = _det_in_ending(ambiguous=False)
    # A low reading at 99% of expected duration -> predictive end should fire.
    det.process_reading(0.0, _ts(3600 * 0.99))
    assert completed, "Smart Termination should end the cycle for a confident, non-ambiguous match"
    assert completed[0].get("termination_reason") == TerminationReason.SMART


def test_smart_termination_suppressed_when_ambiguous() -> None:
    det, completed = _det_in_ending(ambiguous=True)
    det.process_reading(0.0, _ts(3600 * 0.99))
    # Predictive end is gated off; power-based fallback can't fire yet
    # (time_below_threshold < off_delay), so the cycle stays open.
    assert not completed, (
        "Smart Termination must be suppressed for an ambiguous match; "
        "the cycle should wait for the power-based end instead"
    )


def test_update_match_sets_ambiguous_flag() -> None:
    det, _ = _det_in_ending(ambiguous=True)
    assert det._match_ambiguous is True
    det.update_match(("Cotton", 0.7, 3600.0, None, False, False))
    assert det._match_ambiguous is False
    # Backward compatible: a 5-element tuple defaults ambiguity to False.
    det.update_match(("Cotton", 0.7, 3600.0, None, False))
    assert det._match_ambiguous is False
