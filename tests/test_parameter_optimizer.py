import pytest
import numpy as np
from datetime import datetime, timedelta
from tests.benchmarks.parameter_optimizer import ParameterOptimizer

def test_analyze_power_thresholds():
    # Synthetic cycle: starts at 100W, stays above 10W, dips to 8W minimum active
    # We expect stop_threshold to be around 8 * 0.8 = 6.4
    # and start_threshold to be around 8 * 1.2 = 9.6
    cycle1 = {
        "power_data": [
            [0, 100], [60, 120], [120, 8], [180, 50], [240, 10]
        ]
    }
    # Another cycle with higher minimum active power
    cycle2 = {
        "power_data": [
            [0, 200], [60, 20], [120, 15], [180, 20]
        ]
    }
    
    optimizer = ParameterOptimizer([cycle1, cycle2])
    thresholds = optimizer.analyze_power_thresholds()
    
    # 5th percentile of [8, 15] is 8.35? 
    # Let's check: min_active_p05 = np.percentile([8, 15], 5)
    # np.percentile([8, 15], 5) -> 8 + (15-8)*0.05 = 8.35
    # suggested_stop = 8.35 * 0.8 = 6.68
    # suggested_start = 8.35 * 1.2 = 10.02
    
    assert "suggested_stop_threshold_w" in thresholds
    assert "suggested_start_threshold_w" in thresholds
    assert 6.0 < thresholds["suggested_stop_threshold_w"] < 7.0
    assert 9.5 < thresholds["suggested_start_threshold_w"] < 10.5

def test_analyze_energy_thresholds():
    # cycle with a "false end" (pause)
    # Power data: [time, power]
    # Pause between 120s and 240s
    power_data = [
        [0, 100], [60, 100], [119, 100], 
        [120, 1], [180, 1], [239, 1],  # Pause
        [240, 100], [300, 100]
    ]
    # dt = 120s = 1/30 hours. avg_power = 1W. Energy = 1/30 Wh = 0.0333 Wh
    # end_energy_threshold should be > 0.0333
    
    cycle = {"power_data": power_data}
    optimizer = ParameterOptimizer([cycle])
    
    # We need to provide a stop_threshold that allows identifying the pause
    results = optimizer.analyze_energy_thresholds(stop_threshold=2.0)
    
    assert "suggested_end_energy_threshold" in results
    assert results["suggested_end_energy_threshold"] > 0.0333
    
    # Start energy: first 60s
    # [0, 100], [60, 100] -> avg 100W for 60s -> 100 * (60/3600) = 1.666 Wh
    # suggested_start_energy_threshold = min_start_energy * 0.5 = 0.8333
    assert "suggested_start_energy_threshold" in results
    assert pytest.approx(results["suggested_start_energy_threshold"], 0.01) == 0.8333

def test_analyze_timing_parameters():
    # Two cycles with a gap
    # cycle1 ends at T1, cycle2 starts at T2
    cycle1 = {
        "start_time": "2026-02-05T10:00:00",
        "end_time": "2026-02-05T11:00:00"
    }
    cycle2 = {
        "start_time": "2026-02-05T11:10:00", # 10 min gap = 600s
        "end_time": "2026-02-05T12:00:00"
    }
    
    optimizer = ParameterOptimizer([cycle1, cycle2])
    timing = optimizer.analyze_timing_parameters()
    
    # 600s * 0.5 = 300s -> capped at 300s
    assert timing["suggested_min_off_gap"] == 300
    
    # Running dead zone: early dip
    cycle3 = {
        "power_data": [
            [0, 100], [10, 1], [20, 100] # Dip at 10s
        ]
    }
    optimizer = ParameterOptimizer([cycle3])
    timing = optimizer.analyze_timing_parameters()
    assert timing["suggested_running_dead_zone"] == 10

def test_data_loader(tmp_path):
    # Create a dummy JSON file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cycle_file = data_dir / "cycle1.json"
    cycle_data = {
        "start_time": "2026-02-05T10:00:00",
        "end_time": "2026-02-05T11:00:00",
        "power_data": [[0, 100], [3600, 100]]
    }
    with open(cycle_file, "w") as f:
        import json
        json.dump(cycle_data, f)
    
    from tests.benchmarks.parameter_optimizer import DataLoader
    loader = DataLoader([str(data_dir)])
    cycles = loader.load_data()
    
    assert len(cycles) == 1
    assert cycles[0]["start_time"] == "2026-02-05T10:00:00"

def test_data_loader_config_entry(tmp_path):
    # Test loading from config entry dump
    data_dir = tmp_path / "data_ce"
    data_dir.mkdir()
    ce_file = data_dir / "config_entry.json"
    ce_data = {
        "data": {
            "store_data": {
                "past_cycles": [
                    {"start_time": "2026-02-05T10:00:00", "power_data": []},
                    {"start_time": "2026-02-05T12:00:00", "power_data": []}
                ]
            }
        }
    }
    with open(ce_file, "w") as f:
        import json
        json.dump(ce_data, f)
    
    from tests.benchmarks.parameter_optimizer import DataLoader
    loader = DataLoader([str(data_dir)])
    cycles = loader.load_data()
    assert len(cycles) == 2

def test_scorer():
    from tests.benchmarks.parameter_optimizer import Scorer
    scorer = Scorer()
    
    actual = [{
        "start_time": "2026-02-05T10:00:00",
        "end_time": "2026-02-05T11:00:00"
    }]
    # Perfect match
    detected = [{
        "start_time": "2026-02-05T10:00:00",
        "end_time": "2026-02-05T11:00:00"
    }]
    report = scorer.score(actual, detected, [])
    assert report["total"] == 1.0
    
    # Instability penalty
    state_changes = [
        (datetime.now(), "running", "paused"),
        (datetime.now(), "paused", "running")
    ]
    report = scorer.score(actual, detected, state_changes)
    assert report["total"] == 0.9 # 10% penalty
    
    # Missed cycle
    report = scorer.score(actual, [], [])
    assert report["total"] == 0.0

def test_cycle_simulator():
    from tests.benchmarks.parameter_optimizer import CycleSimulator
    from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
    from homeassistant.util import dt as dt_util
    
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=30,
        start_threshold_w=10.0,
        stop_threshold_w=5.0,
        start_duration_threshold=1.0,
        completion_min_seconds=5
    )
    simulator = CycleSimulator(config)
    
    now = dt_util.now()
    readings = []
    # Idle
    for i in range(5):
        readings.append((now + timedelta(seconds=i), 0.0))
    # Start
    for i in range(5, 15):
        readings.append((now + timedelta(seconds=i), 100.0))
    # End
    for i in range(15, 100):
        readings.append((now + timedelta(seconds=i), 0.0))
    
    cycles = simulator.run(readings)
    assert len(cycles) == 1
    assert any(c[2] == "starting" for c in simulator.state_changes)

def test_run_sweep():
    cycle = {
        "power_data": [[0, 100], [60, 100], [120, 8], [180, 100]],
        "start_time": "2026-02-05T10:00:00",
        "end_time": "2026-02-05T11:00:00"
    }
    optimizer = ParameterOptimizer([cycle])
    results = optimizer.run_sweep({})
    assert "suggested_stop_threshold_w" in results
    assert "suggested_start_energy_threshold" in results
    assert "suggested_min_off_gap" in results


