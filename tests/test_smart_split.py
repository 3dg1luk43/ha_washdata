import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from custom_components.ha_washdata.profile_store import ProfileStore
import logging

# Setup Logger
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    # Mock executor job to return result directly (sync) for testing
    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)
        
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    
    # Mock async_create_task
    def mock_create_task(coro):
        return None 
    hass.async_create_task = MagicMock(side_effect=mock_create_task)
    
    return hass

@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        store = ProfileStore(mock_hass, "test_entry")
        # Mock internal storage
        store._data = {"past_cycles": [], "profiles": {}, "envelopes": {}}
        store.async_save = AsyncMock()
        store.async_rebuild_envelope = AsyncMock()
        return store

@pytest.mark.skip(reason="Smart split API has been deprecated - split is now manual via Interactive Editor")
@pytest.mark.asyncio
async def test_async_split_cycles_smart(store):
    """Test splitting a cycle into two parts."""
    
    # 1. Setup Data
    start_time = datetime(2023, 1, 1, 12, 0, 0)
    # Cycle with a big gap:
    # 0-10 mins: Active
    # 10-40 mins: Idle (30m gap)
    # 40-50 mins: Active
    
    points = []
    # Part 1: 0-600s
    for i in range(0, 601, 60):
        points.append([ (start_time + timedelta(seconds=i)).isoformat(), 100.0 ])
        
    # Gap: 600s to 2400s (Idle) - added sparsely
    points.append([ (start_time + timedelta(seconds=1200)).isoformat(), 1.0 ])
    points.append([ (start_time + timedelta(seconds=2000)).isoformat(), 1.0 ])
    
    # Part 2: 2400s-3000s
    for i in range(2400, 3001, 60):
        points.append([ (start_time + timedelta(seconds=i)).isoformat(), 100.0 ])
        
    duration = 3000.0
    
    cycle = {
        "id": "original_cycle",
        "start_time": start_time.isoformat(),
        "duration": duration,
        "status": "completed",
        "power_data": points,
        "profile_name": "TestProfile"
    }
    
    store._data["past_cycles"].append(cycle)
    
    # 2. Mock _analyze_split_sync to return the split ranges
    # We simulate that analysis found two segments: 0-600 and 2400-3000
    expected_segments = [(0.0, 600.0), (2400.0, 3000.0)]
    
    # Mock _decompress_power_data to avoid dependency on decompression logic
    # We return the points but casting time to STR as expected by the new logic in async_split_cycles_smart
    # Actually wait, our points in test setup are already ISO strings.
    # The new logic calls `self._decompress_power_data(cycle)` and expects list[tuple[str, float]]
    
    decompressed_output = [(p[0], float(p[1])) for p in points]
    
    with patch.object(store, '_analyze_split_sync', return_value=expected_segments) as mock_analyze, \
         patch.object(store, '_decompress_power_data', return_value=decompressed_output) as mock_decompress:
        
        # 3. Valid Profile data for reference
        store._data["profiles"]["TestProfile"] = {
            "sample_cycle_id": "original_cycle",
            "avg_duration": 3000
        }
        
        # 4. Run Split
        new_ids = await store.async_split_cycles_smart("original_cycle", min_gap_s=900)
        
        # 5. Verify Results
        assert len(new_ids) == 2
        
        cycles = store._data["past_cycles"]
        assert len(cycles) == 2
        
        c1 = cycles[0]
        c2 = cycles[1]
        
        # Verify Cycle 1
        assert c1["duration"] == 600.0
        assert c1["start_time"] == start_time.isoformat()
        
        # Verify Cycle 2
        assert c2["duration"] == 600.0
        # Start time should be start_time + 2400s
        expected_c2_start = (start_time + timedelta(seconds=2400)).isoformat()
        assert c2["start_time"] == expected_c2_start
        
        # Verify Profile Update (Should point to longest new cycle, i.e. either)
        # Assuming code picks one.
        profile = store._data["profiles"]["TestProfile"]
        assert profile["sample_cycle_id"] in new_ids
        
        # Verify Envelope Rebuild was triggered
        store.async_rebuild_envelope.assert_called_with("TestProfile")
