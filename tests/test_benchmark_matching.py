"""Benchmark for profile matching performance."""
import time
import json
import os
import pytest
from datetime import datetime, timedelta
import numpy as np
from custom_components.ha_washdata.profile_store import ProfileStore, MatchResult

# Path to real data
REAL_DATA_PATH = "/root/ha_washdata/cycle_data/me/washing_machine/real_export.json"

def load_real_data():
    """Load real data from JSON export."""
    if not os.path.exists(REAL_DATA_PATH):
        return None
        
    with open(REAL_DATA_PATH, "r") as f:
        data = json.load(f)
        
    return data.get("data", {})

def generate_current_readings_from_real(cycle_power_data, offset=0, noise_std=5.0):
    """Generate 'current' readings from a real cycle trace."""
    current = []
    # Start now
    start_ts = datetime.now()
    
    # Cycle power data is [[offset, power], ...]
    if not cycle_power_data:
        return []

    for offset_s, power in cycle_power_data:
        ts = start_ts + timedelta(seconds=offset_s + offset)
        noisy_power = max(0, power + np.random.normal(0, noise_std))
        current.append((ts.isoformat(), noisy_power))
        
    return current

@pytest.fixture
def store(mock_hass):
    """Create a profile store populated with real data."""
    store = ProfileStore(mock_hass, "test_entry_real")
    
    real_data = load_real_data()
    if not real_data:
        pytest.skip("Real data file not found at " + REAL_DATA_PATH)
        
    # Populate store
    store._data["profiles"] = real_data.get("profiles", {})
    store._data["past_cycles"] = real_data.get("past_cycles", [])
    
    # Ensure cache is cold
    if hasattr(store, "_cached_sample_segments"):
         store._cached_sample_segments = {}
         
    return store

def test_benchmark_match_profile_real(store):
    """Benchmark the match_profile method with real data."""
    
    past_cycles = store._data["past_cycles"]
    if not past_cycles:
        pytest.skip("No past cycles in real data")
        
    # Pick a cycle that belongs to a profile (ideally a long one)
    target_cycle = None
    # Try to find a cycle with a profile name that has significant data
    for c in reversed(past_cycles):
        if c.get("profile_name") and len(c.get("power_data", [])) > 100:
            target_cycle = c
            break
            
    if not target_cycle:
        # Fallback to any cycle
        target_cycle = past_cycles[-1]
        
    print(f"\nBenchmarking with cycle ID: {target_cycle.get('id')} ({len(target_cycle['power_data'])} samples)")
    
    # Simulate current readings from this cycle
    current_readings = generate_current_readings_from_real(target_cycle["power_data"])
    current_duration = target_cycle["duration"]
    
    # Warmup
    store.match_profile(current_readings, current_duration)
    
    start_time = time.perf_counter()
    iterations = 20 # Real matching is slower, reduce iterations
    
    for _ in range(iterations):
        store.match_profile(current_readings, current_duration)
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    print(f"\n--- Benchmark Result (Real Data) ---")
    print(f"Total time for {iterations} matches: {total_time:.4f}s")
    print(f"Average time per match: {avg_time*1000:.2f} ms")
    print(f"------------------------------------\n")
    
    # Flexible assertion
    assert avg_time < 1.0, f"Matching is too slow! {avg_time*1000:.2f}ms > 1000ms"
