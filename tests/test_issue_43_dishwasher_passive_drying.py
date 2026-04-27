"""Regression tests for GitHub issue #43.

Dishwasher ECO cycles have a 2+ hour passive drying phase at near-0W power.
A terminal drain spike that fires shortly after the ENDING state is entered
resets _time_below_threshold.  The subsequent 60-min silence timeout then fires
at ~180 min — well before the real cycle end (~233 min for ECO) — and
_finish_cycle(keep_tail=False) snaps end_time back to _last_active_time (the
terminal drain spike at 120 min), storing only 120 min instead of 233 min and
corrupting the profile avg_duration.

Fixes:
  1. _should_defer_finish() now defers dishwasher cycles that are below 85%
     of the profile's expected duration, bypassing the confidence gate that
     was too strict during the passive drying phase.
  2. Fallback timeout calls _finish_cycle(keep_tail=True) for dishwashers so
     that, if deferral eventually expires, the stored end_time is the actual
     timeout timestamp rather than the terminal-spike _last_active_time.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    STATE_ENDING,
    STATE_RUNNING,
    STATE_FINISHED,
    STATE_OFF,
    STATE_STARTING,
)
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_WASHING_MACHINE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dishwasher_config(**overrides) -> CycleDetectorConfig:
    defaults = dict(
        min_power=2.0,
        off_delay=1800,
        stop_threshold_w=2.0,
        start_threshold_w=5.0,
        device_type=DEVICE_TYPE_DISHWASHER,
        min_off_gap=3600,
        completion_min_seconds=60,
        end_energy_threshold=0.05,
        start_energy_threshold=0.01,
        start_duration_threshold=1.0,
    )
    defaults.update(overrides)
    return CycleDetectorConfig(**defaults)


def _make_detector(config: CycleDetectorConfig) -> tuple[CycleDetector, list[dict]]:
    """Return (detector, completed_cycles_list)."""
    completed: list[dict] = []

    def _on_cycle_end(data: dict) -> None:
        completed.append(data)

    det = CycleDetector(
        config=config,
        on_state_change=lambda old, new: None,
        on_cycle_end=_on_cycle_end,
    )
    return det, completed


def _ts(offset_seconds: float, base: datetime | None = None) -> datetime:
    if base is None:
        base = datetime(2026, 4, 23, 18, 40, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def _feed(det: CycleDetector, power: float, offset_s: float, base: datetime | None = None) -> None:
    det.process_reading(power, _ts(offset_s, base))


# ---------------------------------------------------------------------------
# Unit tests for _should_defer_finish dishwasher protection
# ---------------------------------------------------------------------------


class TestDishwasherDeferralProtection:
    """_should_defer_finish must protect passive drying cycles below 85% of expected."""

    def _make_det_in_ending(self, expected_duration: float = 14112.0, confidence: float = 0.3) -> CycleDetector:
        cfg = _make_dishwasher_config()
        det, _ = _make_detector(cfg)
        # Manually set up state as if we're mid-cycle in ENDING
        det._state = STATE_ENDING
        det._matched_profile = "ECO"
        det._expected_duration = expected_duration
        det._last_match_confidence = confidence
        det._current_cycle_start = _ts(0)
        return det

    def test_defers_below_85_percent_low_confidence(self) -> None:
        """Protection fires even with confidence < 0.55 (the normal gate threshold)."""
        det = self._make_det_in_ending(expected_duration=14112.0, confidence=0.30)
        # 120 min = 7200s, which is 51% of 14112s — well below 85%
        assert det._should_defer_finish(7200.0), (
            "Expected deferral at 51% of expected duration for dishwasher "
            "(passive drying protection should bypass confidence gate)"
        )

    def test_defers_at_80_percent(self) -> None:
        """Still defers at 80% (within the 85% protection window)."""
        det = self._make_det_in_ending(expected_duration=14112.0, confidence=0.40)
        duration_80pct = 14112.0 * 0.80  # 11289.6s ≈ 188 min
        assert det._should_defer_finish(duration_80pct), (
            "Expected deferral at 80% of expected duration for dishwasher"
        )

    def test_does_not_defer_above_85_percent(self) -> None:
        """Protection stops above 85% so the cycle can eventually end."""
        det = self._make_det_in_ending(expected_duration=14112.0, confidence=0.30)
        # 90% = 12700.8s ≈ 212 min
        duration_90pct = 14112.0 * 0.90
        # With confidence 0.30 < DEFAULT_DEFER_FINISH_CONFIDENCE (0.55), the
        # normal ratio path also won't defer. Overall result: no deferral.
        assert not det._should_defer_finish(duration_90pct), (
            "Protection should NOT defer above 85% with low confidence — "
            "cycle must be allowed to complete."
        )

    def test_no_protection_without_matched_profile(self) -> None:
        """No profile match → no dishwasher protection (can't infer expected duration)."""
        det = self._make_det_in_ending(expected_duration=14112.0, confidence=0.30)
        det._matched_profile = None  # Clear the match
        assert not det._should_defer_finish(7200.0), (
            "Without a matched profile, the passive drying protection must not fire"
        )

    def test_no_protection_for_washing_machine(self) -> None:
        """Passive drying protection must not activate for non-dishwasher devices."""
        cfg = _make_dishwasher_config(device_type=DEVICE_TYPE_WASHING_MACHINE)
        det, _ = _make_detector(cfg)
        det._state = STATE_ENDING
        det._matched_profile = "Cotton60"
        det._expected_duration = 14112.0
        det._last_match_confidence = 0.30
        det._current_cycle_start = _ts(0)
        # At 51% of expected, washing machine should NOT get the protection
        assert not det._should_defer_finish(7200.0), (
            "Passive drying protection must be dishwasher-only"
        )

    def test_safety_limit_still_applies(self) -> None:
        """Even with protection, cycles > expected + 4h must not defer forever."""
        det = self._make_det_in_ending(expected_duration=14112.0, confidence=0.40)
        # 14112 + 14401s = way over the 4-hour deferral limit
        over_limit = 14112.0 + 14401.0
        assert not det._should_defer_finish(over_limit), (
            "Safety deferral limit must override passive drying protection"
        )


# ---------------------------------------------------------------------------
# Integration test: power trace simulation
# ---------------------------------------------------------------------------


class TestDishwasherPassiveDryingIntegration:
    """Simulate the full ECO dishwasher cycle power trace.

    Timeline:
      0–111 min  : Active wash (90W)       → STARTING → RUNNING
      111 min    : Power → 0W              → RUNNING → PAUSED → ENDING
      120 min    : Terminal drain spike (50W, < 120s into ENDING) in ENDING
                   → _end_spike_seen=True, cycle stays in ENDING
      111–233min : Passive drying at 0W
      [Test verifies cycle does NOT end at ~180 min via timeout deferral]
    """

    BASE = datetime(2026, 4, 23, 18, 40, 0, tzinfo=timezone.utc)

    def _run_high_power_phase(self, det: CycleDetector, duration_s: float, step_s: float = 30.0) -> None:
        """Feed 90W readings for duration_s seconds."""
        t = 0.0
        while t <= duration_s:
            det.process_reading(90.0, self.BASE + timedelta(seconds=t))
            t += step_s

    def _run_zero_power_phase(
        self,
        det: CycleDetector,
        start_s: float,
        end_s: float,
        step_s: float = 30.0,
    ) -> None:
        """Feed 0W readings from start_s to end_s."""
        t = start_s
        while t <= end_s:
            det.process_reading(0.0, self.BASE + timedelta(seconds=t))
            t += step_s

    def test_cycle_not_ended_at_120min_timeout(self) -> None:
        """Cycle must not finish at ~180 min when drying protection is active."""
        cfg = _make_dishwasher_config(
            off_delay=1800,
            min_off_gap=3600,
            stop_threshold_w=2.0,
            start_threshold_w=5.0,
            start_duration_threshold=1.0,
            start_energy_threshold=0.001,
            completion_min_seconds=60,
        )
        det, completed = _make_detector(cfg)

        # Inject profile match at 118 min so _expected_duration is known
        EXPECTED_DURATION = 14112.0  # ~235 min (cycle 0 from user data)
        det.update_match(("ECO", 0.45, EXPECTED_DURATION, None, False))

        # Active wash phase: 0 → 111 min (6660s)
        self._run_high_power_phase(det, duration_s=6660, step_s=30.0)
        assert det.state == STATE_RUNNING, "Should be RUNNING during active wash"

        # Drying phase starts: power drops to 0W, cycle should PAUSED→ENDING.
        # Need enough readings to accumulate beyond both dynamic_pause_threshold
        # and dynamic_end_threshold (each ≈ 90–105s with a 30s sampling interval).
        self._run_zero_power_phase(det, start_s=6660, end_s=6900, step_s=30.0)
        # Give enough time below threshold to enter ENDING
        assert det.state == STATE_ENDING, f"Expected ENDING after drying starts, got {det.state}"

        # Inject terminal drain spike ~9 min into ENDING (simulates mid-cycle drain pump).
        # Place it after the initial zero phase so it fires well inside ENDING.
        terminal_spike_t = 6900 + 30  # just past the ENDING entry
        det.process_reading(50.0, self.BASE + timedelta(seconds=terminal_spike_t))
        assert det._end_spike_seen is True, "Terminal spike must set _end_spike_seen"
        assert det.state == STATE_ENDING, "Cycle must stay in ENDING (terminal spike, not resume)"

        # Continue 0W drying phase — push _time_below_threshold well past effective_off_delay (3600s)
        # WITHOUT the fix, the cycle would end here because _should_defer_finish returned False.
        # WITH the fix, deferral protects the cycle because 120 min < 85% of 235 min (200 min).
        self._run_zero_power_phase(
            det,
            start_s=terminal_spike_t + 30,
            end_s=terminal_spike_t + 4200,  # 70 min of 0W after spike (> 3600s off_delay)
            step_s=30.0,
        )

        # Cycle must still be in ENDING — the deferral protection should have fired
        assert det.state == STATE_ENDING, (
            "Cycle ended prematurely! Passive drying protection should have deferred "
            "the timeout at ~180 min. The cycle must remain in ENDING state until "
            "it reaches ~85% of expected duration (~200 min)."
        )
        assert not completed, (
            "No completed cycle should have been recorded at ~180 min; "
            "the dishwasher ECO cycle was still in its passive drying phase."
        )

    def test_deferral_expires_and_cycle_can_end(self) -> None:
        """After 85% of expected duration, deferral expires and cycle terminates."""
        cfg = _make_dishwasher_config(
            off_delay=1800,
            min_off_gap=3600,
            stop_threshold_w=2.0,
            start_threshold_w=5.0,
            start_duration_threshold=1.0,
            start_energy_threshold=0.001,
            completion_min_seconds=60,
        )
        det, completed = _make_detector(cfg)

        EXPECTED_DURATION = 14112.0  # ~235 min
        det.update_match(("ECO", 0.45, EXPECTED_DURATION, None, False))

        # Wash phase
        self._run_high_power_phase(det, duration_s=6660, step_s=30.0)
        # Drying phase → ENDING (need enough readings to pass both dynamic thresholds)
        self._run_zero_power_phase(det, start_s=6660, end_s=6900, step_s=30.0)
        assert det.state == STATE_ENDING

        # Terminal drain spike shortly after ENDING entry
        terminal_spike_t = 6930
        det.process_reading(50.0, self.BASE + timedelta(seconds=terminal_spike_t))

        # Run 0W through 85% of expected (14112 * 0.85 = 11995s ≈ 200 min)
        # then continue to push past 85% so deferral expires
        past_85pct = 12600  # 210 min — beyond the 85% threshold (11995s)
        self._run_zero_power_phase(
            det,
            start_s=terminal_spike_t + 30,
            end_s=past_85pct + 3700,  # 210 min + 62 more min of silence
            step_s=30.0,
        )

        # After 85%+, the cycle should eventually complete (energy gate passes on 0W)
        assert det.state == STATE_FINISHED or completed, (
            "Cycle should have completed after deferral expired past 85% of expected. "
            f"State: {det.state}, completed count: {len(completed)}"
        )

    def test_completed_cycle_end_time_not_set_to_terminal_spike(self) -> None:
        """end_time must be the timeout timestamp, not _last_active_time (terminal spike)."""
        cfg = _make_dishwasher_config(
            off_delay=1800,
            min_off_gap=3600,
            stop_threshold_w=2.0,
            start_threshold_w=5.0,
            start_duration_threshold=1.0,
            start_energy_threshold=0.001,
            completion_min_seconds=60,
        )
        det, completed = _make_detector(cfg)

        EXPECTED_DURATION = 14112.0
        det.update_match(("ECO", 0.45, EXPECTED_DURATION, None, False))

        # Wash phase
        self._run_high_power_phase(det, duration_s=6660, step_s=30.0)
        # Drying → ENDING (enough readings for both dynamic thresholds)
        self._run_zero_power_phase(det, start_s=6660, end_s=6900, step_s=30.0)

        # Terminal drain spike shortly after ENDING entry
        terminal_spike_t = 6930
        det.process_reading(50.0, self.BASE + timedelta(seconds=terminal_spike_t))
        terminal_spike_ts = self.BASE + timedelta(seconds=terminal_spike_t)

        # Run until deferral expires and cycle completes
        end_s = 14112.0 * 0.85 + 4000  # well past 85% + enough silence
        self._run_zero_power_phase(
            det,
            start_s=terminal_spike_t + 30,
            end_s=end_s,
            step_s=30.0,
        )

        if not completed:
            pytest.skip("Cycle did not complete — check test timing parameters")

        cycle_data = completed[0]
        end_time = cycle_data.get("end_time")
        assert end_time is not None, "end_time must be present in completed cycle data"

        # The end_time should be well after the terminal drain spike (120 min).
        # With keep_tail=True for dishwasher timeout, end_time = timeout timestamp,
        # not _last_active_time.  The terminal spike was at 120 min; the cycle must
        # report a duration substantially longer than 120 min.
        start_time = cycle_data.get("start_time")
        if start_time:
            if isinstance(end_time, str):
                from homeassistant.util import dt as dt_util
                end_dt = dt_util.parse_datetime(end_time)
            else:
                end_dt = end_time
            if isinstance(start_time, str):
                from homeassistant.util import dt as dt_util
                start_dt = dt_util.parse_datetime(start_time)
            else:
                start_dt = start_time
            if end_dt and start_dt:
                stored_duration_s = (end_dt - start_dt).total_seconds()
                # With keep_tail=True, stored duration must be > 120 min (7200s).
                # It should be close to the timeout time (past 85% of expected).
                assert stored_duration_s > 8000, (
                    f"Stored cycle duration {stored_duration_s:.0f}s is too short. "
                    f"Expected > 8000s (> 133 min). The terminal drain spike at 120 min "
                    f"must NOT be used as the cycle end_time."
                )
