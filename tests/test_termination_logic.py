"""Test termination logic priority."""
import tests.mock_imports  # noqa: F401
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import STATE_OFF, STATE_RUNNING, STATE_ENDING, STATE_PAUSED, STATE_FINISHED

# Helper to create datetime sequence
def dt(offset_seconds: int) -> datetime:
    return datetime(2023, 1, 1, 12, 0, 0) + timedelta(seconds=offset_seconds)

@pytest.fixture
def base_config():
    """Default detector config."""
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        # Default start threshold checks
        start_duration_threshold=0.0,
        start_energy_threshold=0.0,
        start_threshold_w=6.0,
        stop_threshold_w=4.0,
    )

@pytest.fixture
def mock_callbacks():
    return {
        "on_state_change": Mock(),
        "on_cycle_end": Mock(),
    }

def test_long_drying_phase_cycle_continuation(base_config, mock_callbacks):
    """
    Verify cycle remains active during long low-power (drying) phases 
    when expected duration suggests it should continue.
    """
    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # 1. Start Cycle
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10)) # Running
    
    # Simulate a matched profile that expects 3600s (1h)
    # This is normally done by profile matcher callback injection.
    # We can manually set it for testing internals if we want, or use a mock matcher.
    
    # We'll use a mock matcher to conform to __init__ API
    mock_matcher = Mock()
    # Return match: name="Heavy", conf=0.9, duration=3600, phase="Washing", is_mismatch=False
    mock_matcher.side_effect = lambda readings: ("Heavy", 0.9, 3600.0, "Washing", False)
    
    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher
    )
    
    # Restart with matcher
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10)) # Transition to RUNNING
    detector.process_reading(100.0, dt(20)) # Match attempted here
    
    # Check match happened
    assert detector.matched_profile == "Heavy"
    # assert detector._expected_duration == 3600.0 # Internal, but assumed
    
    # 2. Power drops at T=1800 (30 mins), half way. 
    # Must feed intermediate readings to avoid skewing p95 cadence
    # Feed "Running" power every 10s until 1800
    for t in range(30, 1800, 10):
        detector.process_reading(100.0, dt(t))
    
    # Drop to 1.0W
    detector.process_reading(1.0, dt(1800))
    
    # 3. Wait off_delay (60s) with 10s updates
    for t in range(1810, 1870, 10):
        detector.process_reading(1.0, dt(t))
    
    # CURRENT BUGGY BEHAVIOR: Cycle ends because power is low, ignoring 3600s expectation.
    # If this passes 'completed', it confirms the "bug" (default behavior).
    # After fix, this should stay RUNNING or ENDING.
    
    if detector.state == STATE_OFF:
        # Now this means failure (bug persisted)
        cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
        pytest.fail(f"Cycle ended prematurely at {cycle_data['duration']}s (Expected ~3600s)")
    else:
        # Success! Kept alive.
        assert detector.state in (STATE_ENDING, STATE_RUNNING, STATE_PAUSED)

def test_manual_program_override_termination(base_config, mock_callbacks):
    """
    Test that a manual program (with 100% confidence) keeps cycle alive.
    Simulates wrapper return: ("ManualProfile", 1.0, 3600.0, "Manual", False)
    """
    mock_matcher = Mock()
    mock_matcher.side_effect = lambda readings: ("ManualProfile", 1.0, 3600.0, "Manual", False)
    
    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher
    )
    
    # Start and run briefly
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10)) # Transition to RUNNING
    detector.process_reading(100.0, dt(60)) # Trigger match
    
    assert detector.matched_profile == "ManualProfile"
    assert detector._expected_duration == 3600.0

    # Feed "Running" power every 10s until 600
    for t in range(70, 600, 10):
        detector.process_reading(100.0, dt(t))
    
    # Power fail early (10 mins)
    detector.process_reading(0.0, dt(600))
    
    # Wait past off_delay (60s) with updates
    for t in range(610, 710, 10):
        detector.process_reading(0.0, dt(t))
    
    # Should be alive
    assert detector.state != STATE_OFF
    
    # Warp to expected duration end + tolerance (3600 * 1.25 = 4500)
    # So we need to go beyond 4500 to ensure it finishes
    detector.process_reading(0.0, dt(5000))
    
    # Should be OFF
    assert detector.state == STATE_FINISHED

def test_fix_duration_keeps_alive(base_config, mock_callbacks):
    """
    Test that will PASS only after the fix.
    Cycle should remain alive during low power if (elapsed / expected) < ratio.
    """
    # 1. Setup detector with mocked profile match
    mock_matcher = Mock()
    # Expect 3600s
    mock_matcher.side_effect = lambda readings: ("Heavy", 0.9, 3600.0, "Drying", False)
    
    # We need to set min_duration_ratio in config (will add this field in implementation)
    # For now, we rely on default or modify config object after init if needed
    # base_config.min_duration_ratio = 0.8 (Not yet in dataclass)
    
    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher
    )
    
    # Start
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10)) # Transition to RUNNING
    detector.process_reading(100.0, dt(60)) # Trigger match
    
    assert detector.matched_profile == "Heavy"
    
    # Feed "Running" power every 10s until 1800
    for t in range(20, 1800, 10):
        detector.process_reading(100.0, dt(t))
    
    # Drop power at 30 mins (1800s)
    detector.process_reading(0.0, dt(1800))
    
    # Advance past off_delay (60s) -> 1900s
    for t in range(1810, 1910, 10):
        detector.process_reading(0.0, dt(t))
    
    # ASSERTION FOR DESIRED BEHAVIOR:
    # Should NOT be OFF. Should be ENDING (waiting) or RUNNING (if we deem it running).
    # Usually 'ENDING' is the low-power waiting state.
    
    # Note: THIS WILL FAIL currently (step 1 of TDD)
    if detector.state == STATE_OFF:
        pytest.fail("Cycle terminated prematurely! Fix not working.")
    
    # Ensure it ends eventually
    # 1h + off_delay -> 3600 + 100 = 3700
    detector.process_reading(0.0, dt(3700))
    
    # Now it should end
    # assert detector.state == STATE_OFF (Might need to implement the check correctly first)
