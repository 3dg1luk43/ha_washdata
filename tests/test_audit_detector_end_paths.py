"""Audit 2026-10-02 DETECT-04 / -05 / -06 / -09: four end-path defects in the detector.

04: after > 120 s in ENDING a resumed wash stays in ENDING, so Smart
    Termination's debounce (`_time_in_state`) was pre-paid by washing time and the
    next short dip ended the cycle - the final spin became a second cycle.
05: Smart Termination was the one finisher that ignored a user pause; the pause
    itself carried elapsed time past the ratio, so a paused washer was finished.
06: `user_stop()` kept the whole post-appliance wait: Stop pressed 30 min after a
    60 min wash ended stored 90 min, which then fed avg_duration.
09: the state snapshot dropped the match confidence, so after a restart Smart
    Termination was blocked as low-confidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import TerminationReason

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _cfg(device_type: str = "washing_machine", min_power: float = 2.0) -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=float(min_power),
        off_delay=int(C.resolve_off_delay_default(device_type)),
        device_type=device_type,
        interrupted_min_seconds=150,
        completion_min_seconds=C.DEVICE_COMPLETION_THRESHOLDS.get(
            device_type, C.DEFAULT_COMPLETION_MIN_SECONDS
        ),
        start_duration_threshold=C.resolve_start_duration_default(device_type),
        min_off_gap=C.resolve_min_off_gap_default(device_type),
        start_energy_threshold=C.DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE.get(
            device_type, 0.2
        ),
        end_energy_threshold=C.DEFAULT_END_ENERGY_THRESHOLD,
        start_threshold_w=float(min_power) + max(1.0, 0.1 * float(min_power)),
        stop_threshold_w=float(min_power) * 0.6,
        match_interval=C.DEFAULT_PROFILE_MATCH_INTERVAL,
        match_confidence_threshold=C.DEFAULT_PROFILE_MATCH_THRESHOLD,
        smart_termination_duration_ratio=(
            C.resolve_smart_termination_duration_ratio_default(device_type)
        ),
    )


def _match(name: str, conf: float, expected: float):
    return (name, conf, expected, None, False, False, False, False, None, None, None, 0.0, None)


class _Run:
    """A detector fed at a 10 s cadence, with the host clock following the data."""

    def __init__(self, device_type: str = "washing_machine", expected: float = 3600.0):
        self.ends: list[dict] = []
        self.det = CycleDetector(
            _cfg(device_type), lambda *_: None, self.ends.append,
            profile_matcher=lambda _r: _match("Quick", 0.8, expected),
        )
        self.t = 0.0
        self._clock = patch(
            "homeassistant.util.dt.now", side_effect=lambda: T0 + timedelta(seconds=self.t)
        )
        self._clock.start()

    def feed(self, power: float, until: float, step: float = 10.0) -> None:
        while self.t < until:
            self.det.process_reading(power, T0 + timedelta(seconds=self.t))
            self.t += step

    def close(self) -> None:
        self._clock.stop()


def _wash_with_soak(soak_s: float) -> list[dict]:
    run = _Run()
    try:
        run.feed(500.0, 3000)
        if soak_s:
            run.feed(0.0, 3000 + soak_s)
        base = 3000 + soak_s
        run.feed(500.0, base + 560)   # rinse resumes, ~9 min, inside ENDING
        run.feed(0.0, base + 580)     # 20 s dip between rinse and final spin
        run.feed(800.0, base + 700)   # final spin, 2 min
        run.feed(0.0, base + 3000)    # real end
        return run.ends
    finally:
        run.close()


def test_a_wash_resumed_inside_ending_is_not_split_on_the_next_dip():
    assert len(_wash_with_soak(0)) == 1
    ends = _wash_with_soak(200)
    assert len(ends) == 1, [(e["duration"], e["termination_reason"]) for e in ends]
    # The cycle must run through the final spin (3200 + 700 s = 65 min). Ended on
    # the 20 s dip instead, it stored 62.5 min and the spin was cut off.
    assert ends[0]["duration"] >= 64 * 60


def test_smart_termination_honours_a_user_pause():
    for device_type in ("washing_machine", "dryer"):
        run = _Run(device_type)
        try:
            run.feed(500.0, 3560)
            # The user presses Pause: the manager sets both flags (and re-asserts the
            # verified pause on every match).
            run.det.set_user_paused(True)
            run.det.set_verified_pause(True)
            run.feed(0.0, 3560 + 3600, step=30)
            assert run.ends == [], (device_type, run.ends[0]["termination_reason"])
        finally:
            run.close()


def test_user_stop_long_after_the_wash_ended_does_not_bank_the_wait():
    run = _Run(expected=7200.0)  # so Smart Termination stays out of the way
    try:
        run.feed(500.0, 3600)
        run.feed(0.0, 3600 + 120)  # a little quiet, not enough to end
        run.t = 3600 + 1800        # the user presses Stop 30 min after the wash ended
        run.det.user_stop()
    finally:
        run.close()
    assert len(run.ends) == 1
    assert run.ends[0]["termination_reason"] == TerminationReason.USER
    assert run.ends[0]["duration"] < 3600 + 600


def test_user_stop_while_still_running_keeps_done_now():
    run = _Run(expected=7200.0)
    try:
        run.feed(500.0, 3600)
        run.det.user_stop()
    finally:
        run.close()
    assert run.ends[0]["duration"] >= 3590


def test_the_snapshot_keeps_the_match_confidence():
    run = _Run()
    try:
        run.feed(500.0, 900)
        run.det.update_match(_match("Quick", 0.83, 3600.0))
        snap = run.det.get_state_snapshot()
    finally:
        run.close()
    restored = CycleDetector(_cfg(), lambda *_: None, lambda _d: None)
    restored.restore_state_snapshot(snap)
    assert restored._last_match_confidence == 0.83


def test_a_short_pump_run_is_completed_not_interrupted():
    """DETECT-07: the flat 150 s interrupted floor overrode the pump's 5 s floor."""
    from custom_components.ha_washdata.detector_config import build_detector_config

    cfg = build_detector_config({}, {}, "pump")
    assert cfg.interrupted_min_seconds <= cfg.completion_min_seconds == 5
    ends: list[dict] = []
    det = CycleDetector(cfg, lambda *_: None, ends.append)
    for t in range(0, 60, 2):
        det.process_reading(400.0, T0 + timedelta(seconds=t))
    for t in range(60, 60 + 900, 10):
        det.process_reading(0.0, T0 + timedelta(seconds=t))
    assert ends and ends[0]["status"] == "completed"


def test_washer_floors_are_unchanged():
    from custom_components.ha_washdata.detector_config import build_detector_config

    assert build_detector_config({}, {}, "washing_machine").interrupted_min_seconds == 150
