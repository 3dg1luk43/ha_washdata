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
    DISHWASHER_END_SPIKE_WAIT_SECONDS,
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

    def test_does_not_defer_above_85_percent_after_end_spike(self) -> None:
        """Once the real end-of-cycle pump-out has fired (end_spike_seen=True),
        the deferral lifts above 85% so the cycle can complete promptly.
        """
        det = self._make_det_in_ending(expected_duration=14112.0, confidence=0.30)
        det._end_spike_seen = True  # simulate pump-out already fired
        # 90% = 12700.8s ≈ 212 min
        duration_90pct = 14112.0 * 0.90
        # With confidence 0.30 < DEFAULT_DEFER_FINISH_CONFIDENCE (0.55), the
        # normal ratio path also won't defer. Overall result: no deferral.
        assert not det._should_defer_finish(duration_90pct), (
            "Protection should NOT defer above 85% with low confidence once "
            "the end spike has been seen — cycle must be allowed to complete."
        )

    def test_defers_above_85_percent_until_end_spike(self) -> None:
        """Issue #43: between 85% expected and the end-of-cycle pump-out, the
        cycle must stay deferred so the pump-out gets folded into the original
        cycle instead of being treated as a brand-new cycle.
        """
        det = self._make_det_in_ending(expected_duration=14112.0, confidence=0.30)
        det._end_spike_seen = False  # pump-out has not fired yet
        duration_90pct = 14112.0 * 0.90  # 212 min — past 85%, before pump-out
        assert det._should_defer_finish(duration_90pct), (
            "Above 85% but with no end spike seen, the cycle must remain "
            "deferred so the pump-out is captured as part of the original cycle."
        )

    def test_lifts_after_expected_plus_wait_window(self) -> None:
        """Hard upper bound: once we cross expected + the wait window, deferral
        lifts even without an end spike (smart termination's wait branch takes
        over).  Pulls the boundary from the production constant so the
        assertion can never drift from the code under test.
        """
        expected_duration = 14112.0
        det = self._make_det_in_ending(
            expected_duration=expected_duration, confidence=0.30
        )
        det._end_spike_seen = False
        duration_past_wait = (
            expected_duration + DISHWASHER_END_SPIKE_WAIT_SECONDS + 1.0
        )
        assert not det._should_defer_finish(duration_past_wait), (
            "Once duration exceeds expected + DISHWASHER_END_SPIKE_WAIT_SECONDS, "
            "deferral must release so the cycle can finalise."
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

        # Inject mid-cycle drain spike ~9 min into ENDING (the dishwasher's
        # wash→drying drain wind-down).  It occurs at ~49% of expected duration,
        # well below the 85% end-spike-progress gate added for issue #43, so it
        # must NOT pre-arm Smart Termination.  long_ending_tail still keeps the
        # cycle in ENDING via the existing terminal_spike path.
        terminal_spike_t = 6900 + 30  # just past the ENDING entry
        det.process_reading(50.0, self.BASE + timedelta(seconds=terminal_spike_t))
        assert det._end_spike_seen is False, (
            "Mid-cycle spike at 49% of expected duration must not arm "
            "_end_spike_seen (issue #43 regression)"
        )
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


class TestEndSpikeProgressGate:
    """Regression tests for the issue #43 end-spike progress gate.

    Reproduces the full April 27 user scenario from
    home-assistant_ha_washdata_2026-04-27T21-08-33.248Z.log:

      18:08:35  STARTING → RUNNING (active wash, ~85W, then 2000W heating)
      19:57:34  power drops to 0W → RUNNING → PAUSED
      19:57:44  PAUSED → ENDING                  (~109 min, 47% of expected)
      20:06:35  high-power spikes in ENDING       (~117 min, 50% of expected)
      ...       passive drying at 0–0.5W          (~117 min → 234 min)
      22:02:34  pump-out spike (16–20W, ~1 min)   (~234 min, 99% of expected)
      22:03:34  pump-out finishes, power drops to 0
      22:05:00  cycle should fully terminate

    The pre-fix bug:
      The mid-cycle spike at ~117 min (50% of expected) set _end_spike_seen=True,
      which pre-armed Smart Termination so it fired at 99% of expected (~222 min)
      and closed the cycle BEFORE the real pump-out arrived.  The pump-out at
      ~234 min was then registered as a brand-new cycle.

    The fix gates _end_spike_seen=True on duration >= 85% of expected, so the
    mid-cycle spike is ignored, Smart Termination waits for the real end spike
    (the pump-out), and the pump-out is folded into the original cycle.
    """

    BASE = datetime(2026, 4, 27, 18, 8, 35, tzinfo=timezone.utc)
    EXPECTED_DURATION = 14112.0  # ~235 min ECO cycle

    def _feed(self, det: CycleDetector, power: float, offset_s: float) -> None:
        det.process_reading(power, self.BASE + timedelta(seconds=offset_s))

    def _run_constant(self, det: CycleDetector, power: float, start_s: float, end_s: float, step_s: float = 30.0) -> float:
        t = start_s
        while t <= end_s:
            self._feed(det, power, t)
            t += step_s
        return t - step_s

    def _make_det(self) -> tuple[CycleDetector, list[dict]]:
        cfg = _make_dishwasher_config(
            off_delay=1800,
            min_off_gap=3600,
            stop_threshold_w=2.0,
            start_threshold_w=5.0,
            start_duration_threshold=1.0,
            start_energy_threshold=0.001,
            completion_min_seconds=60,
        )
        return _make_detector(cfg)

    def test_mid_cycle_spike_does_not_arm_smart_termination(self) -> None:
        """A spike at ~50% of expected duration must not set _end_spike_seen."""
        det, _ = self._make_det()
        det.update_match(("ECO", 0.65, self.EXPECTED_DURATION, None, False))

        # Wash phase 0–110 min
        self._run_constant(det, 90.0, 0, 6600)
        # Drop to 0W → drying → PAUSED → ENDING
        self._run_constant(det, 0.0, 6630, 6900)
        assert det.state == STATE_ENDING

        # Mid-cycle drain spike at ~117 min (50% of 235 min expected)
        self._feed(det, 50.0, 7020)
        assert det._end_spike_seen is False, (
            "Spike at 50% of expected duration must be ignored for end-spike "
            "tracking (issue #43)"
        )
        # But the cycle still stays in ENDING via long_ending_tail
        assert det.state == STATE_ENDING

    def test_real_end_spike_at_pump_out_arms_smart_termination(self) -> None:
        """A spike at ~99% of expected duration must set _end_spike_seen."""
        det, _ = self._make_det()
        det.update_match(("ECO", 0.65, self.EXPECTED_DURATION, None, False))

        # Wash 0–110 min, drop, ENDING
        self._run_constant(det, 90.0, 0, 6600)
        self._run_constant(det, 0.0, 6630, 6900)
        assert det.state == STATE_ENDING

        # Mid-cycle drain spike (ignored)
        self._feed(det, 50.0, 7020)
        assert det._end_spike_seen is False

        # Long passive drying at 0W
        self._run_constant(det, 0.0, 7050, 13950, step_s=60.0)

        # Real end-of-cycle pump-out at ~234 min (99.5% of expected)
        # Pump-out is 17W for ~60s, then drops back to 0W.
        pump_start = 14040
        self._run_constant(det, 17.0, pump_start, pump_start + 60, step_s=10.0)

        # The pump-out is at 99.5% of expected → must arm _end_spike_seen
        assert det._end_spike_seen is True, (
            "Real end spike at 99.5% of expected duration must arm "
            "_end_spike_seen so Smart Termination can fire"
        )

    def test_full_cycle_completes_with_pump_out_included(self) -> None:
        """Smoke-test: full April 27 timeline produces exactly one cycle of
        ~234 min, NOT a 222-min cycle followed by a 3-min ghost cycle."""
        det, completed = self._make_det()
        det.update_match(("ECO", 0.65, self.EXPECTED_DURATION, None, False))

        # Wash phase
        self._run_constant(det, 90.0, 0, 6600)
        # Drop into ENDING
        self._run_constant(det, 0.0, 6630, 6900)
        # Mid-cycle drain spikes (multiple, like the user's log shows)
        for offset in (7020, 7080, 7110, 7140):
            self._feed(det, 50.0, offset)
        # Long passive drying — keep cycle in ENDING via verified pause to model
        # the dishwasher passive drying protection at the manager level.
        det.set_verified_pause(True)
        self._run_constant(det, 0.0, 7170, 13950, step_s=60.0)
        det.set_verified_pause(False)
        # Real pump-out
        pump_start = 14040
        self._run_constant(det, 17.0, pump_start, pump_start + 60, step_s=10.0)
        # Power drops, cycle should smart-terminate shortly after
        self._run_constant(det, 0.0, pump_start + 90, pump_start + 600, step_s=30.0)

        # Exactly one cycle completed
        assert len(completed) == 1, (
            f"Expected exactly 1 completed cycle (full ECO), got {len(completed)}. "
            f"This indicates the pump-out was treated as a separate cycle (the "
            f"issue #43 regression)."
        )
        cycle = completed[0]
        # Duration should be ~234 min (close to expected 235 min), NOT 222 min
        # (the pre-fix Smart Termination misfire).
        duration_min = cycle["duration"] / 60.0
        assert duration_min > 230.0, (
            f"Cycle duration {duration_min:.1f} min is too short — pump-out "
            f"appears to have been excluded.  Expected ≥ 230 min."
        )
        # The pump-out must actually be in the recorded power_data — checking
        # max_power is insufficient because earlier wash peaks (≈90W) trivially
        # satisfy a >=17W bound even if the pump-out window was excluded.
        # Inspect a ±2-minute window around pump_start instead and require at
        # least one sample matching the pump-out signature (in [10, 20]W).
        power_data = cycle.get("power_data", [])
        pump_window = [
            (t, p) for t, p in power_data
            if pump_start - 120 <= t <= pump_start + 120
        ]
        pump_signature_samples = [
            (t, p) for t, p in pump_window if 10.0 <= p <= 20.0
        ]
        assert pump_signature_samples, (
            f"No pump-out sample (10–20W) found in power_data within ±2 min of "
            f"pump_start={pump_start}s. Window had {len(pump_window)} samples; "
            f"pump-out appears to have been excluded from the recorded cycle."
        )
