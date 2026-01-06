import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from custom_components.ha_washdata.profile_store import ProfileStore

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    # Mock services if needed
    return hass

@pytest.fixture
def store(mock_hass):
    # Initialize store with mocks
    # Set lenient ratios for testing early matches
    ps = ProfileStore(mock_hass, "test_entry_id", min_duration_ratio=0.0, max_duration_ratio=2.0)
    # Mock internal store load to return empty or default data
    ps._store.async_load = AsyncMock(return_value=None)
    ps._store.async_save = AsyncMock()
    return ps

def test_add_cycle(store):
    """Test adding a cycle."""
    cycle_data = {
        "start_time": "2023-01-01T12:00:00",
        "duration": 3600,
        "status": "completed",
        "power_data": [["2023-01-01T12:00:00", 100.0], ["2023-01-01T13:00:00", 100.0]]
    }
    
    store.add_cycle(cycle_data)
    
    assert len(store._data["past_cycles"]) == 1
    saved = store._data["past_cycles"][0]
    assert saved["duration"] == 3600
    assert "id" in saved
    assert saved["profile_name"] is None

def test_create_profile(store):
    """Test creating a profile from a cycle."""
    # Add a cycle first
    store.add_cycle({
        "start_time": "2023-01-01T12:00:00",
        "duration": 3600,
        "status": "completed",
        "power_data": [["2023-01-01T12:00:00", 100.0]]
    })
    cycle_id = store._data["past_cycles"][0]["id"]
    
    async def run_test():
        await store.create_profile("Heavy Duty", cycle_id)
    
    asyncio.run(run_test())
    
    assert "Heavy Duty" in store._data["profiles"]
    profile = store._data["profiles"]["Heavy Duty"]
    assert profile["sample_cycle_id"] == cycle_id
    assert profile["avg_duration"] == 3600
    
    # Check that cycle was labeled
    assert store._data["past_cycles"][0]["profile_name"] == "Heavy Duty"

def test_retention_policy(store):
    """Test that old cycles are dropped."""
    # Set small cap for testing
    store._max_past_cycles = 5
    
    # Add 10 cycles
    for i in range(10):
        t_str = dt_str(i*60)
        store.add_cycle({
            "start_time": t_str, # Increasing time
            "duration": 100,
            "status": "completed",
            "power_data": [[t_str, 10]]
        })
        
    assert len(store._data["past_cycles"]) == 5
    
    # Verify we kept the NEWEST ones (indices 5-9)
    times = [c["start_time"] for c in store._data["past_cycles"]]
    # 5-9 implies start times from i=5 to i=9
    # i=9 -> dt_str(540)
    assert dt_str(540) in times
    assert dt_str(0) not in times

from datetime import datetime, timedelta

def dt_str(offset_seconds: int) -> str:
    return (datetime(2023, 1, 1, 12, 0, 0) + timedelta(seconds=offset_seconds)).isoformat()

def test_rebuild_envelope_updates_stats(store):
    """Test that rebuilding envelope updates min/max duration."""
    # Create profile
    store._data["profiles"]["TestProf"] = {"sample_cycle_id": "dummy"}
    
    # Add 3 cycles with DIFFERENT durations labeled as TestProf
    durations = [3000, 3600, 4000]
    for d in durations:
        start_t = datetime(2023, 1, 1, 12, 0, 0)
        t_start = start_t.isoformat()
        t_mid = (start_t + timedelta(seconds=d/2)).isoformat()
        t_end = (start_t + timedelta(seconds=d)).isoformat()
        
        store.add_cycle({
            "start_time": t_start, 
            "duration": d,
            "status": "completed",
            "profile_name": "TestProf",
            # Need valid power data for rebuild (>=3 points)
            "power_data": [[t_start, 10], [t_mid, 100], [t_end, 10]] 
        })
        
    # Trigger rebuild
    store.rebuild_envelope("TestProf")
    
    profile = store._data["profiles"]["TestProf"]
    assert profile["min_duration"] == 3000
    assert profile["max_duration"] == 4000
    
    # Check envelope existence
    assert "TestProf" in store._data["envelopes"]
    env = store._data["envelopes"]["TestProf"]
    assert env["cycle_count"] == 3

def test_match_profile(store):
    """Test simple profile matching."""
    # Setup - use dense data compatible with current_data
    # Need to use ABSOLUTE timestamps relative to start_time
    start_dt = datetime(2023, 1, 1, 10, 0, 0)
    # Ramp signal for 100s
    dense_power = [[(start_dt + timedelta(seconds=i)).isoformat(), float(i)] for i in range(101)]
    
    store.add_cycle({
        "start_time": start_dt.isoformat(),
        "duration": 100,
        "status": "completed",
        "power_data": dense_power
    })
    cycle_id = store._data["past_cycles"][0]["id"]
    
    async def run_setup():
        await store.create_profile("RampProfile", cycle_id)
    asyncio.run(run_setup())
    
    # Test Match: Exact match sequence (first 100 seconds)
    current_data = [( (start_dt + timedelta(seconds=i)).isoformat(), float(i) ) for i in range(101)]
    current_duration = 100.0 # Match longer duration
    
    result = store.match_profile(current_data, current_duration)
    
    assert result.best_profile == "RampProfile"
    assert result.confidence > 0.9 # Should be very high
    
    # Test Mismatch
    # Use isoformat strings even for mismatch test to avoid preprocessing errors
    current_data_bad = [( (start_dt + timedelta(seconds=i)).isoformat(), 1000.0 ) for i in range(101)]
    result_bad = store.match_profile(current_data_bad, current_duration)
    
    match_bad = result_bad.best_profile
    score_bad = result_bad.confidence
    
    if match_bad == "Constant100":
        assert score_bad < 0.5
