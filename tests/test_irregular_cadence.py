
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import STATE_RUNNING, STATE_PAUSED, STATE_ENDING

def dt(seconds):
    return datetime(2023, 1, 1, 10, 0, 0) + timedelta(seconds=seconds)

def test_slow_cadence_prevents_premature_pause():
    # Cadence 60s. 
    # Current logic (15s threshold) would pause immediately on first low reading (dt=60).
    # Desired: Wait ~3 samples (180s).
    
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=600, # Large off delay so we don't finish
        device_type='washing_machine'
    )
    callbacks = {"on_state_change": Mock(), "on_cycle_end": Mock()}
    
    detector = CycleDetector(config, callbacks["on_state_change"], callbacks["on_cycle_end"])
    
    # Start (High power)
    detector.process_reading(100.0, dt(0)) # STARTING
    detector.process_reading(100.0, dt(60)) # RUNNING (Confirms start)
    assert detector.state == STATE_RUNNING
    
    # 1st Low reading at 120 (dt=60)
    # Ideally should NOT switch to PAUSED yet if we follow "3 * interval" rule.
    # Because 60s gap implies we only know "it is low NOW", it might have been low for 1s or 60s.
    # But strictly accumulating dt=60 means "time_below=60".
    # If dynamic threshold is used: threshold = max(15, 3*60) = 180.
    # So 60 < 180 -> Stay RUNNING (or interpret as 'active waiting'?)
    
    detector.process_reading(0.0, dt(120))
    
    # If logic is fixed, this should be RUNNING.
    # If logic is broken (current), this will be PAUSED (since 60 > 15).
    assert detector.state == STATE_RUNNING 
    
    # 2nd Low reading (dt=60, total=120)
    detector.process_reading(0.0, dt(180))
    assert detector.state == STATE_RUNNING
    
    # 3rd Low reading (dt=60, total=180). Now we might pause.
    detector.process_reading(0.0, dt(240))
    # Threshold reached?
    # If >= 180, maybe pause now.
    
    # Note: State machine might check transition conditions.
    
def test_fast_cadence_uses_default_thresholds():
    # Cadence 1s.
    # Default threshold 15s should apply (max(15, 3*1) = 15).
    
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=600,
        device_type='washing_machine'
    )
    callbacks = {"on_state_change": Mock(), "on_cycle_end": Mock()}
    detector = CycleDetector(config, callbacks["on_state_change"], callbacks["on_cycle_end"])
    
    # Start - Ensure we confirm start (need > 5s high)
    for i in range(6):
        detector.process_reading(100.0, dt(i))
    assert detector.state == STATE_RUNNING
    
    # 10s of low power (10 samples) -> Should stay RUNNING
    for i in range(10):
        detector.process_reading(0.0, dt(2 + i))
    
    assert detector.state == STATE_RUNNING
    
    # 6 more seconds (total 16s) -> Should PAUSE
    for i in range(6):
        detector.process_reading(0.0, dt(12 + i))
        
    assert detector.state == STATE_PAUSED
