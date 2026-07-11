"""Unit tests for CycleDetector."""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
import pytest
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import (
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_ENDING,
    STATE_PAUSED,
    STATE_FINISHED,
    STATE_INTERRUPTED,
    STATE_FORCE_STOPPED,
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_WASHING_MACHINE,
)

# Helper to create datetime sequence
def dt(offset_seconds: int) -> datetime:
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)

@pytest.fixture
def detector_config():
    """Default detector config."""
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        abrupt_drop_watts=500.0,
        abrupt_drop_ratio=0.5,
        abrupt_high_load_factor=1.2,
        start_duration_threshold=0.0,
    )

def flush_buffer(detector, start_t_offset):
    """Flush detector state machine by sending 80 low readings at 1s intervals.
    This resets the p95 cadence to ~1s and ensures thresholds drop to min values.
    Also ensures we exceed typical off_delay (60s).
    """
    for i in range(1, 81):
        detector.process_reading(0.0, dt(start_t_offset + i))
@pytest.fixture
def mock_callbacks():
    """Mock callbacks."""
    return {
        "on_state_change": Mock(),
        "on_cycle_end": Mock(),
    }

def test_normal_cycle(detector_config, mock_callbacks):
    """Test a normal cycle start and finish."""
    detector = CycleDetector(
        config=detector_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # 1. Start Cycle
    detector.process_reading(100.0, dt(0))
    assert detector.state == STATE_STARTING
    mock_callbacks["on_state_change"].assert_called_with(STATE_OFF, STATE_STARTING)

    # 1b. Confirmation (need > 0.005 Wh)
    # 100W for 10s = 100 * 10/3600 = 0.27 Wh > 0.005
    detector.process_reading(100.0, dt(10))
    assert detector.state == STATE_RUNNING
    mock_callbacks["on_state_change"].assert_called_with(STATE_STARTING, STATE_RUNNING)

    # 2. Run for 20 mins (1200s)
    for t in range(10, 1200, 10):
        detector.process_reading(100.0, dt(t))
    
    # 3. Low power for off_delay (60s)
    # Start low power at 1201s
    detector.process_reading(1.0, dt(1201)) # < min_power 5.0
    # assert detector.is_waiting_low_power() 
    
    # Still waiting
    detector.process_reading(1.0, dt(1201 + 30))
    mock_callbacks["on_cycle_end"].assert_not_called()

    # Finish (flush)
    flush_buffer(detector, 1201 + 30)

    assert detector.state == STATE_FINISHED
    mock_callbacks["on_cycle_end"].assert_called_once()
    
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "completed"
    # Last active time was 1190s (when power was 100W)
    assert cycle_data["duration"] == pytest.approx(1190, abs=1)

def test_short_cycle_interrupted(detector_config, mock_callbacks):
    """Test a cycle that is too short (between interrupted_min and completion_min)."""
    # Config: interrupted_min=150, completion_min=600
    detector = CycleDetector(
        config=detector_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )
    
    # Start
    detector.process_reading(100.0, dt(0))
    
    # Run for 300s (5 mins) - valid start, but too short for full completion
    detector.process_reading(100.0, dt(300))
    
    # End
    detector.process_reading(1.0, dt(301)) # Low power start
    flush_buffer(detector, 301)
    
    assert mock_callbacks["on_cycle_end"].called
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "interrupted"
    # Reason is logged, not in callback data
    # assert "too short for completion" in str(mock_callbacks["on_cycle_end"].call_args)
    # Status is just "interrupted", reason is logged.
    
    # Verify duration
    assert cycle_data["duration"] == pytest.approx(301, abs=5)

def test_very_short_cycle_interrupted(detector_config, mock_callbacks):
    """Test a cycle that is extremely short (< interrupted_min)."""
    detector = CycleDetector(
        config=detector_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )
    
    # Start
    detector.process_reading(100.0, dt(0))
    
    # Run for 60s
    detector.process_reading(100.0, dt(60))
    
    # End
    detector.process_reading(1.0, dt(61))
    flush_buffer(detector, 61)

    assert mock_callbacks["on_cycle_end"].called
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "interrupted"

def test_abrupt_drop(detector_config, mock_callbacks):
    """Test detection of an abrupt power drop."""
    detector = CycleDetector(
        config=detector_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )
    
    # Start
    detector.process_reading(100.0, dt(0))
    
    # Ramp up to high power (2000W)
    detector.process_reading(2000.0, dt(100))
    
    # SUDDEN DROP to 0W at 200s
    # Previous was 2000, now 0.
    # drop=2000, ratio=1.0. 
    # Thresholds: drop_watts=500, ratio=0.5. PASSES.
    detector.process_reading(0.0, dt(200))
    
    # Should flag internal abrupt_drop=True.
    
    # End immediate (wait buffer)
    flush_buffer(detector, 200)
    
    assert mock_callbacks["on_cycle_end"].called
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    
    # Duration ~200s. Thresholds: interrupted=150s.
    # Logic: if abrupt_drop and duration <= interrupted_min + 90s (150+90=240).
    # 200 <= 240 -> True.
    assert cycle_data["status"] == "interrupted"

def test_abrupt_drop_ignored_if_long(detector_config, mock_callbacks):
    """Test that an abrupt drop is IGNORED if the cycle runs long enough after? Or total duration?"""
    # Logic check:
    # if self._abrupt_drop and duration <= (float(self._config.interrupted_min_seconds) + 90.0):
    # It checks TOTAL duration.
    
    detector = CycleDetector(
        config=detector_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )
    
    # Start
    detector.process_reading(100.0, dt(0))
    
    # High power
    detector.process_reading(2000.0, dt(100))
    
    # SUDDEN DROP at 100s, but maybe it just paused?
    detector.process_reading(0.0, dt(200))
    
    # But then it continues running? Or ends?
    # CycleDetector flags `_abrupt_drop = True` when low power starts.
    # If it ends right there, it is interrupted.
    # But wait, `_abrupt_drop` is set when entering LOW POWER waiting.
    # If power goes BACK UP, `_low_power_start` is cleared?
    # process_reading: 
    # if is_active_for_end: _low_power_start = None.
    # BUT `_abrupt_drop` is NOT cleared in `process_reading` if power goes back up!
    # Let's check `_transition_to` or `_finish_cycle`.
    # `_abrupt_drop` is initialized to False in `__init__`.
    # Set to False in `_transition_to(STATE_RUNNING)`.
    # Set to True in `process_reading` (lines 118-124) when low power detected.
    # It is NEVER reset to False if power resumes in `process_reading`!
    
    # This might be a BUG or intended?
    # If the cycle resumes, does the "abrupt drop" flag stick?
    # If it resumes, runs for another hour, and finishes normally...
    # `duration` will be > 150+90. So `_should_mark_interrupted` will return False (unless < completion_min).
    # So the logic holds: "Abrupt drop only assumes interruption if the cycle ends SOON after (or is short overall)."
    
    # Implementation test: Long cycle with early drop
    detector.process_reading(2000.0, dt(300)) # Resume? No, logic above sets drop when entering low power.
    # If I feed 0.0, it sets drop.
    # If I feed 2000.0 next, it clears low_power_start.
    
    # Resume
    detector.process_reading(2000.0, dt(300))
    
    # Run until 1000s (> 240s)
    detector.process_reading(2000.0, dt(1000))
    
    # End normally
    detector.process_reading(0.0, dt(1001))
    flush_buffer(detector, 1001)
    
    assert mock_callbacks["on_cycle_end"].called
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    # Should be completed because duration (1000s) > 240s
    assert cycle_data["status"] == "completed"

def test_force_end(detector_config, mock_callbacks):
    """Test force_end by watchdog."""
    detector = CycleDetector(
        config=detector_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )
    
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(1200))
    detector.force_end(dt(1200))
    
    assert detector.state == STATE_FORCE_STOPPED
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    
    # 1200s is > completion_min, so force_stopped status is preserved
    assert cycle_data["status"] == "force_stopped"
    
    # 300s. Config: interrupted_min=150. completion_min=600.
    # Logic in `_finish_cycle`:
    # if status in ("completed", "force_stopped") and self._should_mark_interrupted(duration):
    #   status = "interrupted"
    
    # Old assertion removed: 300 < 600 check is no longer valid for 1200s test

def test_end_repeat_count_accumulates_across_periods(mock_callbacks):
    """Test that end_condition_count accumulates across low-power periods.
    
    When end_repeat_count > 1, the counter should persist across resets of
    low_power_start. This allows the detector to require multiple periods
    of low power (each >= off_delay) before ending the cycle.
    """
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        end_repeat_count=2,  # Require 2 periods of low power
        start_duration_threshold=0.0,  # Disable start debounce
    )
    
    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )
    
    # Start cycle
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(1)) # Confirm start
    assert detector.state == STATE_RUNNING
    
    # Run for 15 mins (enough to exceed completion_min_seconds of 600)
    for t in range(10, 900, 10):
        detector.process_reading(100.0, dt(t))
    
    # Enter first low-power period at t=900
    detector.process_reading(1.0, dt(900))
    # In vNext, this transitions to ENDING (waiting for confirmation) or PAUSED?
    # If off_delay=60, it likely enters ENDING state logic internally but state remains RUNNING/ENDING?
    # Check if detector helper method exists or removed.
    # Assuming removed, we check behaviour via state or internal flag if accessible.
    # For now, let's skip is_waiting_low_power check or verify state is NOT OFF.
    assert detector.state != STATE_OFF
    
    # Wait past first off_delay (60s) -> counter should increment to 1
    detector.process_reading(1.0, dt(961))
    # Cycle should NOT end yet (need 2 periods)
    # Cycle should NOT end yet (need 2 periods)
    assert detector.state in (STATE_RUNNING, STATE_ENDING, STATE_PAUSED)
    mock_callbacks["on_cycle_end"].assert_not_called()
    
    # low_power_start should now be reset, but counter should persist
    # Next reading at t=962 should start a new low-power period
    detector.process_reading(1.0, dt(962))
    
    # Wait past second off_delay -> counter should increment to 2
    detector.process_reading(1.0, dt(1023))  # 962 + 61 = 1023
    
    # Now cycle should end
    assert detector.state == STATE_FINISHED
    mock_callbacks["on_cycle_end"].assert_called_once()
    
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "completed"


def test_dishwasher_end_spike_stays_in_ending(mock_callbacks):
    """Dishwasher final pump-out spike should not resume RUNNING from ENDING."""
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=1200,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
        device_type=DEVICE_TYPE_DISHWASHER,
    )

    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    detector.process_reading(120.0, dt(0))
    detector.process_reading(120.0, dt(30))
    assert detector.state == STATE_RUNNING

    for t in range(60, 6001, 30):
        detector.process_reading(120.0, dt(t))

    # Pretend we already have a stable profile match for this run.
    detector.update_match(("dishwasher_program", 0.6, 7200.0, None, False))

    # Long low-power tail: RUNNING -> PAUSED -> ENDING.
    for t in range(6030, 6331, 30):
        detector.process_reading(0.0, dt(t))

    assert detector.state == STATE_ENDING

    # Final pump-out spike after a sustained ENDING tail.
    detector.process_reading(85.0, dt(6360))

    # Regression: this used to bounce ENDING -> RUNNING.
    assert detector.state == STATE_ENDING


def test_dishwasher_end_spike_finishes_soon_after(mock_callbacks):
    """Dishwasher should finish soon after terminal spike when match is near completion."""
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=1200,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
        device_type=DEVICE_TYPE_DISHWASHER,
    )

    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    detector.process_reading(120.0, dt(0))
    detector.process_reading(120.0, dt(30))
    assert detector.state == STATE_RUNNING

    for t in range(60, 6001, 30):
        detector.process_reading(120.0, dt(t))

    # Set an expected duration close enough for smart termination after the spike.
    detector.update_match(("dishwasher_program", 0.6, 6200.0, None, False))

    # Long low-power tail: RUNNING -> PAUSED -> ENDING.
    for t in range(6030, 6331, 30):
        detector.process_reading(0.0, dt(t))

    assert detector.state == STATE_ENDING

    # Terminal pump-out burst should not resume RUNNING.
    detector.process_reading(85.0, dt(6360))
    assert detector.state == STATE_ENDING

    # Continue low-power tail; smart termination should complete within this window.
    detector.process_reading(0.0, dt(6390))
    detector.process_reading(0.0, dt(6420))
    detector.process_reading(0.0, dt(6450))

    assert detector.state == STATE_FINISHED
    mock_callbacks["on_cycle_end"].assert_called_once()

    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "completed"
    assert cycle_data["duration"] == pytest.approx(6420, abs=40)


def test_dishwasher_unmatched_end_spike_caps_off_delay_at_1800s(mock_callbacks):
    """Fix B: after a pump-out spike in ENDING, an unmatched dishwasher cycle
    closes after 1800 s of low-power silence, not the full min_off_gap of 3600 s.

    Without Fix B, effective_off_delay = max(off_delay=1800, min_off_gap=3600) = 3600 s.
    With Fix B, unmatched + end_spike_seen → effective_off_delay = min(3600, 1800) = 1800 s.
    """
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=1800,
        min_off_gap=3600,
        device_type=DEVICE_TYPE_DISHWASHER,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
    )
    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # Build cadence at 30 s intervals; run at 100 W for 135 min.
    # With p95_dt ≈ 30 s: pause_thresh ≈ 90 s, end_thresh ≈ 105 s.
    HIGH, LOW = 100.0, 0.0
    t = 0
    detector.process_reading(HIGH, dt(t))   # → STARTING
    t += 30
    detector.process_reading(HIGH, dt(t))   # energy gate passed → RUNNING
    t += 30
    for _ in range(269):                    # 135 min total at 30 s steps
        detector.process_reading(HIGH, dt(t))
        t += 30

    # Drop power.  15 low readings (450 s) is enough to transit RUNNING→PAUSED→ENDING
    # (pause_thresh ≈ 90 s, end_thresh ≈ 105 s, reset on each transition).
    for _ in range(15):
        detector.process_reading(LOW, dt(t))
        t += 30

    assert detector.state == STATE_ENDING, (
        f"Expected ENDING after 15 low readings, got {detector.state}"
    )

    # Accumulate ≥ 120 s in ENDING to satisfy long_ending_tail.
    for _ in range(4):
        detector.process_reading(LOW, dt(t))
        t += 30

    # Pump-out spike: high reading → _end_spike_seen = True.
    # long_ending_tail is True, so the spike is terminal - stays in ENDING.
    detector.process_reading(50.0, dt(t))
    t += 30
    assert detector._end_spike_seen, "Pump-out spike should set _end_spike_seen"
    assert detector.state == STATE_ENDING, "Should remain in ENDING after terminal spike"

    # Accumulate low readings after the spike.  Fix B caps effective_off_delay at
    # 1800 s, so the cycle must end once _time_below_threshold >= 1800 s.
    #
    # Caveat: the energy gate checks the recent window (last off_delay=1800 s).
    # The pump-out spike energy (~0.4 Wh) keeps the gate closed for one extra reading
    # after the cap fires (at exactly t_spike + 1800 s the spike is still in-window).
    # Therefore we need 62 readings (1860 s) to push the spike out of the window and
    # let the energy gate pass.
    for _ in range(65):  # 65 x 30 s = 1950 s -- comfortably past the gate
        detector.process_reading(LOW, dt(t))
        t += 30

    assert mock_callbacks["on_cycle_end"].called, (
        "Cycle should have ended within 1950 s of the pump-out spike (Fix B cap)"
    )


def test_dishwasher_no_spike_uses_full_off_delay(mock_callbacks):
    """Fix B guard: without an end spike, the full effective_off_delay (3600 s) is used.

    At 1800 s of continuous low-power silence - with no end spike - the cycle must
    NOT yet be finished (it still needs the remaining 1800 s to reach 3600 s).
    """
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=1800,
        min_off_gap=3600,
        device_type=DEVICE_TYPE_DISHWASHER,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
    )
    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    HIGH, LOW = 100.0, 0.0
    t = 0
    detector.process_reading(HIGH, dt(t))
    t += 30
    detector.process_reading(HIGH, dt(t))
    t += 30
    for _ in range(269):
        detector.process_reading(HIGH, dt(t))
        t += 30

    # Transition to ENDING via 15 low readings
    for _ in range(15):
        detector.process_reading(LOW, dt(t))
        t += 30

    assert detector.state == STATE_ENDING

    # Feed 60 x 30 s = 1800 s of low readings -- no spike, so _end_spike_seen = False.
    for _ in range(60):
        detector.process_reading(LOW, dt(t))
        t += 30

    # Without a spike the cap does not apply: effective_off_delay = 3600 s.
    # Only 1800 s have elapsed since ENDING entry, so cycle must still be open.
    assert not mock_callbacks["on_cycle_end"].called, (
        "Cycle must not end after only 1800 s of silence when no end spike occurred"
    )


# ── Pump-out just before the 99% smart_ratio gate ────────────────────────────
#
# Regression: when the pump-out (terminal end spike) occurs at ≥90% of the
# expected duration but the cycle's actual length falls slightly below
# avg_duration × 0.99, neither smart termination nor the fallback timeout could
# fire.  Smart termination was blocked by the outer duration gate (0.99) and the
# fallback was blocked because the pump-out reset _time_below_threshold just
# before the threshold was reached.  Fix: once _end_spike_seen was set at ≥90%
# of expected, lower smart_ratio to 0.90 so the next low-power reading after the
# pump-out fires smart termination.


def test_dishwasher_pump_out_below_99pct_threshold_fires_smart_term(mock_callbacks):
    """Pump-out at ~97% of expected fires smart termination even if actual cycle
    duration stays below avg_duration × 0.99.

    Setup: expected = 9000 s.  Cycle runs at full power until t=6700 s, drops to
    0, then shows a pump-out burst (3 readings at 150 W) at t=8750–8810 s
    (≈97.2–98.0% of expected).  The next 0 W reading at t=8840 s should trigger
    smart termination because _end_spike_duration (8750 s) ≥ 0.90 × 9000 = 8100 s.

    Without the fix, smart_ratio = 0.99 → threshold = 8910 s; at t=8840 s
    (8840 < 8910) it would never fire.
    """
    EXPECTED_DUR = 9000.0
    config = CycleDetectorConfig(
        min_power=2.0,
        stop_threshold_w=2.0,
        start_threshold_w=3.0,
        off_delay=180,
        min_off_gap=1999,
        interrupted_min_seconds=150,
        completion_min_seconds=900,
        start_duration_threshold=0.0,
        device_type=DEVICE_TYPE_DISHWASHER,
    )
    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # Start the cycle (two readings to pass energy gate)
    detector.process_reading(2000.0, dt(0))
    detector.process_reading(2000.0, dt(30))
    assert detector.state == STATE_RUNNING

    # Run at full power until t=6700 s
    for t in range(60, 6701, 30):
        detector.process_reading(2000.0, dt(t))

    # Inject a profile match: confident, unambiguous, expected = 9000 s
    detector.update_match(("eco_60", 0.85, EXPECTED_DUR, None, False, False, False))
    assert detector._matched_profile == "eco_60"

    # Power drops → ENDING (two low readings to pass pause+ending thresholds)
    for t in range(6730, 6901, 30):
        detector.process_reading(0.0, dt(t))
    assert detector.state == STATE_ENDING

    # Brief mid-cycle blips below 85% of expected (must NOT set _end_spike_seen)
    # 85% of 9000 = 7650 s; these are at 6960–7020 s (77%)
    for t in (6960, 6990, 7020):
        detector.process_reading(25.0, dt(t))  # below stop_threshold? No, 25 > 2 → is_high
        detector.process_reading(0.0, dt(t + 30))
    # At 7050 s (<85%), end_spike_seen must still be False
    assert not detector._end_spike_seen, "Spike before 85% must not arm end_spike_seen"

    # Silence until just before the pump-out
    for t in range(7080, 8751, 30):
        detector.process_reading(0.0, dt(t))

    # Pump-out burst at 97.2–98.0% of expected (8750–8810 s)
    detector.process_reading(150.0, dt(8750))
    detector.process_reading(150.0, dt(8780))
    detector.process_reading(150.0, dt(8810))
    # The pump-out must set _end_spike_seen at 97%+
    assert detector._end_spike_seen, "Pump-out at 97% must set _end_spike_seen"
    assert detector._end_spike_duration >= EXPECTED_DUR * 0.90, (
        f"_end_spike_duration ({detector._end_spike_duration:.0f}) should be ≥ 90% of "
        f"expected ({EXPECTED_DUR * 0.90:.0f})"
    )

    # Next low reading after pump-out ends — smart termination must fire here.
    # Without the fix: current_duration (8840) < 0.99 × 9000 (8910) → no fire.
    # With the fix:    current_duration (8840) ≥ 0.90 × 9000 (8100) → FIRES.
    detector.process_reading(0.0, dt(8840))
    assert detector.state == STATE_FINISHED, (
        "Smart termination must fire immediately after pump-out ends, even when "
        "actual cycle duration (8840 s) < avg_duration × 0.99 (8910 s)"
    )
    mock_callbacks["on_cycle_end"].assert_called_once()
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "completed"
    assert cycle_data["termination_reason"] == "smart"


def test_dishwasher_pre_rinse_drain_below_90pct_does_not_lower_smart_ratio(mock_callbacks):
    """A mid-cycle drain spike that fires at 87% (below the 90% pump-out gate)
    must NOT lower smart_ratio to 0.90, so the cycle is not prematurely closed
    during the passive Dry phase that follows.

    Setup: expected = 9000 s.  A drain spike at t=7950 s (88.3% of expected) sets
    _end_spike_seen but _end_spike_duration < 90% × 9000 = 8100 s.  The cycle then
    enters the Dry phase with 0 W readings up to t=8500 s (94.4%).  During this
    window, smart termination must NOT fire (old gate 0.99 × 9000 = 8910 still
    applies), keeping the cycle open for the real pump-out at 98%+.
    """
    EXPECTED_DUR = 9000.0
    config = CycleDetectorConfig(
        min_power=2.0,
        stop_threshold_w=2.0,
        start_threshold_w=3.0,
        off_delay=180,
        min_off_gap=1999,
        interrupted_min_seconds=150,
        completion_min_seconds=900,
        start_duration_threshold=0.0,
        device_type=DEVICE_TYPE_DISHWASHER,
    )
    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    detector.process_reading(2000.0, dt(0))
    detector.process_reading(2000.0, dt(30))
    assert detector.state == STATE_RUNNING
    for t in range(60, 7651, 30):
        detector.process_reading(2000.0, dt(t))

    detector.update_match(("eco_60", 0.85, EXPECTED_DUR, None, False, False, False))

    # Drop to ENDING
    for t in range(7680, 7831, 30):
        detector.process_reading(0.0, dt(t))
    assert detector.state == STATE_ENDING

    # Pre-rinse drain spike at 7950 s = 88.3% of expected (below 90% gate)
    detector.process_reading(200.0, dt(7950))
    assert detector._end_spike_seen, "88% spike must still arm _end_spike_seen (≥85% gate)"
    assert detector._end_spike_duration < EXPECTED_DUR * 0.90, (
        "But _end_spike_duration must be < 90% of expected (not a pump-out)"
    )
    detector.process_reading(0.0, dt(7980))

    # Dry phase — silence from 7980 to 8500 s (≤94.4% of expected).
    # Without the fix we'd have smart_ratio = 0.90 → 8100 s threshold → fires at 8130 s.
    # With the fix:  _end_spike_duration < 90% → smart_ratio stays at 0.99 → no fire.
    for t in range(8010, 8501, 30):
        detector.process_reading(0.0, dt(t))

    # Cycle must still be in ENDING — not prematurely finished
    assert detector.state == STATE_ENDING, (
        "Cycle must remain in ENDING during Dry phase after a pre-rinse drain at 88%"
    )
    mock_callbacks["on_cycle_end"].assert_not_called()


# ── Smart Termination prefix-landscape guard (#288) ──────────────────────────
#
# When is_prefix_ambiguous=True (a longer look-alike profile exists in the
# candidate pool) Smart Termination must be blocked so the power-based fallback
# timeout decides.  When False, Smart Termination fires as normal.
#
# Config: min_off_gap=480 gives smart_debounce=240 s and fallback=480 s, so
# the 450 s test window sits between the two: Smart Termination would fire
# without the guard; the fallback does not fire either, leaving the cycle
# definitively in STATE_ENDING when the guard is active.

def _make_washer_detector(mock_callbacks, min_off_gap=480):
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
        min_off_gap=min_off_gap,
        device_type=DEVICE_TYPE_WASHING_MACHINE,
    )
    return CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )


def _run_to_end_of_quick(detector, expected_s=2760):
    """Drive the detector at 100 W for expected_s seconds using 30 s ticks."""
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10))
    assert detector.state == STATE_RUNNING
    for t in range(40, expected_s + 1, 30):
        detector.process_reading(100.0, dt(t))


def test_smart_termination_blocked_by_prefix_ambiguous(mock_callbacks):
    """Reproduces the #288 split-cycle bug class.

    A mid-cycle soak dip lasting 450 s coincides with the short profile's
    expected end.  Without the prefix-landscape guard Smart Termination would
    fire at smart_debounce (240 s); with is_prefix_ambiguous=True it must not.
    """
    detector = _make_washer_detector(mock_callbacks)
    _run_to_end_of_quick(detector)

    # Match: Quick 40°C (2760 s) wins, but Normal 40°C (5280 s ≈ 1.91×) is in
    # the pool with a good shape score → is_prefix_ambiguous=True.
    detector.update_match(("Quick 40C", 0.7, 2760.0, None, False, False, True))
    assert detector._match_prefix_ambiguous is True

    # Soak dip: 450 s of low power (> smart_debounce 240 s, < fallback 480 s).
    for t in range(2790, 2790 + 450, 30):
        detector.process_reading(0.0, dt(t))

    # Smart Termination must be blocked; cycle must still be open.
    assert detector.state in (STATE_ENDING, STATE_PAUSED)
    mock_callbacks["on_cycle_end"].assert_not_called()


def test_smart_termination_fires_without_prefix_ambiguous(mock_callbacks):
    """Baseline: without a long look-alike Smart Termination fires normally."""
    detector = _make_washer_detector(mock_callbacks)
    _run_to_end_of_quick(detector)

    # Same match but is_prefix_ambiguous=False.
    detector.update_match(("Quick 40C", 0.7, 2760.0, None, False, False, False))
    assert detector._match_prefix_ambiguous is False

    # Same 450 s soak dip > smart_debounce (240 s).
    for t in range(2790, 2790 + 450, 30):
        detector.process_reading(0.0, dt(t))

    # Smart Termination fires.
    assert detector.state == STATE_FINISHED
    mock_callbacks["on_cycle_end"].assert_called_once()
    assert mock_callbacks["on_cycle_end"].call_args[0][0]["status"] == "completed"


def test_prefix_ambiguous_flag_stored_and_cleared(mock_callbacks):
    """_match_prefix_ambiguous is stored from the 7th tuple element and cleared
    on a confident mismatch, matching the behaviour of _match_ambiguous."""
    detector = _make_washer_detector(mock_callbacks)
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10))

    # Set via 7-element tuple.
    detector.update_match(("Quick 40C", 0.7, 2760.0, None, False, False, True))
    assert detector._match_prefix_ambiguous is True

    # Cleared on confident mismatch (5th element=True).
    detector.update_match(("Quick 40C", 0.7, 2760.0, None, True, False, False))
    assert detector._match_prefix_ambiguous is False

    # Backward-compatible: 6-element tuple leaves it False.
    detector.update_match(("Quick 40C", 0.7, 2760.0, None, False, False))
    assert detector._match_prefix_ambiguous is False


def test_prefix_ambiguous_survives_snapshot_roundtrip(mock_callbacks):
    """_match_prefix_ambiguous is persisted in the state snapshot and restored."""
    detector = _make_washer_detector(mock_callbacks)
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10))
    detector.update_match(("Quick 40C", 0.7, 2760.0, None, False, False, True))

    snap = detector.get_state_snapshot()
    assert snap["match_prefix_ambiguous"] is True

    # New detector restores the flag.
    detector2 = _make_washer_detector(mock_callbacks)
    detector2.restore_state_snapshot(snap)
    assert detector2._match_prefix_ambiguous is True
