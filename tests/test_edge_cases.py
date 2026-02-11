
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import STATE_OFF, STATE_RUNNING, STATE_ENDING, STATE_STARTING, STATE_PAUSED
from tests.utils.synthesizer import CycleSynthesizer

@pytest.fixture
def base_config():
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=10.0,
        start_energy_threshold=0.01,
        start_threshold_w=6.0,
        stop_threshold_w=4.0,
    )

@pytest.fixture
def detector(base_config):
    on_state_change = Mock()
    on_cycle_end = Mock()
    return CycleDetector(base_config, on_state_change, on_cycle_end)

def test_boot_spike_rejection(detector):
    """Test that short high-power spikes are rejected by duration."""
    synth = CycleSynthesizer()
    synth.add_boot_spike(2000.0, 5.0) # 2000W for 5s (threshold is 10s)
    synth.add_gap(60.0)
    
    readings = synth.generate(sample_interval=1.0)
    
    for ts, p in readings:
        detector.process_reading(p, ts)
        
    # Should have entered STARTING but reverted to OFF
    assert detector.state == STATE_OFF
    assert detector._on_state_change.call_args_list[0][0][1] == STATE_STARTING
    # Reverted to OFF
    assert detector._on_state_change.call_args_list[-1][0][1] == STATE_OFF

def test_low_energy_rejection(detector):
    """Test that low-power hum is rejected by energy threshold."""
    # Threshold is 0.01 Wh
    # Hum at 10W for 20s = 10 * (20/3600) = 0.055 Wh -> wait, hum should be lower
    # Hum at 1W for 60s = 1 * (60/3600) = 0.00027 Wh
    
    # Let's set hum just above start_threshold_w (6.0) but low enough to fail energy
    # Hum at 7W for 20s = 7 * (20/3600) = 0.038 Wh... no, 0.01 Wh is small.
    # 0.01 Wh = 36 Joules. 7W * 5s = 35 Joules.
    
    synth = CycleSynthesizer()
    synth.add_phase(7.0, 5.0) # 7W for 5s (Duration threshold is 10s)
    synth.add_gap(60.0)
    
    readings = synth.generate(sample_interval=1.0)
    
    for ts, p in readings:
        detector.process_reading(p, ts)
        
    assert detector.state == STATE_OFF

def test_long_drying_phase_robustness(detector):
    """Test that a long low-power phase doesn't prematurely end if deferred."""
    # Configure detector with a matched profile expectation (normally done via callback)
    detector._matched_profile = "Dishwasher Eco"
    detector._expected_duration = 7200.0 # 2h
    detector._last_match_confidence = 0.8 # New requirement for deferral
    
    synth = CycleSynthesizer()
    synth.add_phase(2000.0, 1800.0) # 30 min wash
    synth.add_phase(1.0, 3600.0)    # 1h drying (low power but above 0)
    synth.add_phase(50.0, 30.0)     # End spike
    
    readings = synth.generate(sample_interval=30.0)
    
    for ts, p in readings:
        detector.process_reading(p, ts)
        # Verify it stays active during drying
        elapsed = (ts - readings[0][0]).total_seconds()
        if 1800 < elapsed < 5400:
            assert detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
            
    assert detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)

def test_irregular_sampling_stability(detector):
    """Test robustness against highly irregular sampling."""
    synth = CycleSynthesizer()
    synth.add_phase(500.0, 3600.0)
    
    # Generate with large jitter and high drop rate
    readings = synth.generate(sample_interval=10.0, jitter=5.0, drop_rate=0.2)
    
    for ts, p in readings:
        detector.process_reading(p, ts)
        
    assert detector.state == STATE_RUNNING
    
    # Add a huge gap (e.g. 5 minutes)
    last_ts = readings[-1][0]
    detector.process_reading(500.0, last_ts + timedelta(minutes=5))
    
    # Should still be running (it was high power before and after)
    assert detector.state == STATE_RUNNING
