
import pytest
import datetime
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig, STATE_RUNNING

# Mock profile matcher
def mock_matcher(readings):
    # Match if we have >= 3 readings
    if len(readings) >= 3:
        return ("TestProfile", 0.95, 1000.0, None)
    return (None, 0.0, 0.0, None)

@pytest.fixture
def detector():
    config = CycleDetectorConfig(
        min_power=10.0,
        off_delay=60,
        smoothing_window=1
    )
    return CycleDetector(config, lambda s, ss: None, lambda c: None, profile_matcher=mock_matcher)

def test_restart_loses_profile(detector):
    """Test that restarting (snapshot/restore) loses the profile if not explicitly persisted."""
    
    start_time = datetime.datetime.now()
    
    # 1. Run cycle, get it to match
    # High power start
    detector.process_reading(100.0, start_time) 
    detector.process_reading(100.0, start_time + datetime.timedelta(seconds=10))
    detector.process_reading(100.0, start_time + datetime.timedelta(seconds=20))
    
    # Low power to trigger matcher
    detector.process_reading(5.0, start_time + datetime.timedelta(seconds=30))
    
    # Verify we matched
    # We expect the detector to store the matched profile
    assert getattr(detector, "_matched_profile", None) == "TestProfile"
    
    # 2. Go back to High Power
    detector.process_reading(100.0, start_time + datetime.timedelta(seconds=40))
    
    # 3. Snapshot
    snap = detector.get_state_snapshot()
    
    # 4. Create NEW detector and restore
    config = CycleDetectorConfig(min_power=10.0, off_delay=60, smoothing_window=1)
    new_detector = CycleDetector(config, lambda s, ss: None, lambda c: None, profile_matcher=mock_matcher)
    new_detector.restore_state_snapshot(snap)
    
    # 5. Verify state
    assert new_detector._state == STATE_RUNNING
    # Power readings should be restored
    assert len(new_detector._power_readings) > 0
    
    # Power readings restored check
    assert new_detector._power_readings[-1][1] == 100.0

    # 6. Verify profile is restored IMMEDIATELY
    assert getattr(new_detector, "_matched_profile", None) == "TestProfile", "Profile should be restored from snapshot"


if __name__ == "__main__":
    pass
