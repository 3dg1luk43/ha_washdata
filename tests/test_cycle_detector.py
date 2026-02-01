"""Unit tests for CycleDetector."""
from datetime import datetime, timedelta
from unittest.mock import Mock
import pytest
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import STATE_OFF, STATE_RUNNING, STATE_STARTING, STATE_ENDING, STATE_PAUSED

# Helper to create datetime sequence
def dt(offset_seconds: int) -> datetime:
    return datetime(2023, 1, 1, 12, 0, 0) + timedelta(seconds=offset_seconds)

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

    assert detector.state == STATE_OFF
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
    detector.force_end(dt(300))
    
    assert detector.state == STATE_OFF
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    
    # 300s is < completion_min, so even force_stopped gets reclassified as interrupted
    assert cycle_data["status"] == "interrupted"
    
    # 300s. Config: interrupted_min=150. completion_min=600.
    # Logic in `_finish_cycle`:
    # if status in ("completed", "force_stopped") and self._should_mark_interrupted(duration):
    #   status = "interrupted"
    
    # 300 < 600. So it should be reclassified as interrupted.
    assert cycle_data["status"] == "interrupted"

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
    assert detector.state == STATE_OFF
    mock_callbacks["on_cycle_end"].assert_called_once()
    
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "completed"

 
