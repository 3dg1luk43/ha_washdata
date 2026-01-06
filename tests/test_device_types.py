
import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DRYER, DEVICE_TYPE_COFFEE_MACHINE,
    STATE_RUNNING, STATE_OFF, STATE_PAUSED,
    DEFAULT_OFF_DELAY, DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE,
    DEVICE_COMPLETION_THRESHOLDS
)

def dt(seconds):
    return datetime(2023, 1, 1, 10, 0, 0) + timedelta(seconds=seconds)

def flush_buffer(detector, start_t_offset):
    for i in range(1, 81):
        detector.process_reading(0.0, dt(start_t_offset + i))

def test_dryer_thresholds():
    # Verify defaults
    start_thresh = DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE[DEVICE_TYPE_DRYER] # 0.5
    completion_min = DEVICE_COMPLETION_THRESHOLDS[DEVICE_TYPE_DRYER] # 600
    
    config = CycleDetectorConfig(
        device_type=DEVICE_TYPE_DRYER,
        min_power=5.0,
        off_delay=60,
        start_energy_threshold=start_thresh,
        completion_min_seconds=completion_min,
    )
    
    callbacks = {"on_state_change": Mock(), "on_cycle_end": Mock()}
    detector = CycleDetector(config, callbacks["on_state_change"], callbacks["on_cycle_end"])
    
    # 1. Attempt Start with Low Energy (0.4 Wh)
    # 0.4 Wh = 1440 Ws.
    # 100W for 14.4s?
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(14)) 
    # Energy approx: 100 * 14 / 3600 = 0.38 Wh < 0.5.
    # Should NOT be RUNNING yet (should be STARTING or OFF if timed out).
    # Wait, 14s is > start_duration_threshold (default 0? No, usually small).
    # If energy not met, it stays in STARTING or resets?
    # CycleDetector reset:
    # "Abort if power drops below threshold before confirmation"
    # If we keep power high, it stays in STARTING until energy met?
    
    assert detector.state != STATE_RUNNING 
    
    # Continue to 20s (Energy > 0.5)
    detector.process_reading(100.0, dt(20))
    # 100 * 20 / 3600 = 0.55 Wh.
    assert detector.state == STATE_RUNNING
    
    # 2. Short Run (300s) -> Interrupted
    detector.process_reading(100.0, dt(300))
    detector.process_reading(0.0, dt(301))
    flush_buffer(detector, 301)
    
    assert callbacks["on_cycle_end"].called
    cycle_data = callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "interrupted"

def test_coffee_thresholds():
    # Verify defaults
    start_thresh = DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE[DEVICE_TYPE_COFFEE_MACHINE] # 0.05
    completion_min = DEVICE_COMPLETION_THRESHOLDS[DEVICE_TYPE_COFFEE_MACHINE] # 60
    
    config = CycleDetectorConfig(
        device_type=DEVICE_TYPE_COFFEE_MACHINE,
        min_power=5.0,
        off_delay=30,
        start_energy_threshold=start_thresh,
        completion_min_seconds=completion_min,
        interrupted_min_seconds=10, 
    )
    
    callbacks = {"on_state_change": Mock(), "on_cycle_end": Mock()}
    detector = CycleDetector(config, callbacks["on_state_change"], callbacks["on_cycle_end"])
    
    # 1. Start with very small energy (0.06 Wh)
    # 1000W for > 5s (Default start_duration_threshold)
    detector.process_reading(1000.0, dt(0))
    for i in range(1, 7):
        detector.process_reading(1000.0, dt(i))
        
    assert detector.state == STATE_RUNNING
    
    # 2. Run for 70s -> Completed
    detector.process_reading(100.0, dt(70))
    detector.process_reading(0.0, dt(71))
    flush_buffer(detector, 71)
    
    assert callbacks["on_cycle_end"].called
    cycle_data = callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "completed"

def test_coffee_very_short():
    # 30s run -> Interrupted (if 60s min)
    start_thresh = 0.05
    completion_min = 60
    config = CycleDetectorConfig(
        device_type=DEVICE_TYPE_COFFEE_MACHINE,
        min_power=5.0,
        off_delay=30,
        start_energy_threshold=start_thresh,
        completion_min_seconds=completion_min,
    )
    callbacks = {"on_state_change": Mock(), "on_cycle_end": Mock()}
    detector = CycleDetector(config, callbacks["on_state_change"], callbacks["on_cycle_end"])
    
    detector.process_reading(1000.0, dt(0))
    detector.process_reading(1000.0, dt(30))
    detector.process_reading(0.0, dt(31))
    flush_buffer(detector, 31)
    
    cycle_data = callbacks["on_cycle_end"].call_args[0][0]
    # If completion_min_seconds is 60, then 30s is interrupted.
    assert cycle_data["status"] == "interrupted"
