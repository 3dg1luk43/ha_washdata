
import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.profile_store import ProfileStore, MatchResult

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    return hass

@pytest.fixture
def store(mock_hass):
    with patch("homeassistant.helpers.storage.Store") as MockStore:
        store_instance = ProfileStore(mock_hass, "test_entry")
        store_instance._store = MockStore.return_value
        store_instance._store.async_load = AsyncMock(return_value=None)
        store_instance._store.async_save = AsyncMock()
        return store_instance

@pytest.mark.asyncio
async def test_standalone_profile_creation(store):
    """Test creating a profile without a reference cycle."""
    await store.create_profile_standalone("Standalone", avg_duration=3600.0)
    
    profiles = store.get_profiles()
    assert "Standalone" in profiles
    assert profiles["Standalone"]["avg_duration"] == 3600.0
    assert profiles["Standalone"].get("sample_cycle_id") is None

@pytest.mark.asyncio
async def test_envelope_rebuild_lifecycle(store):
    """Test the full cycle of adding data and rebuilding envelopes."""
    # 1. Add some cycles for a profile
    start_base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    for i in range(3):
        cycle_data = {
            "start_time": (start_base + timedelta(days=i)).isoformat(),
            "duration": 600.0,
            "max_power": 100.0,
            "status": "completed",
            "power_data": [
                [(start_base + timedelta(days=i)).isoformat(), 0.0],
                [(start_base + timedelta(days=i, seconds=100)).isoformat(), 100.0],
                [(start_base + timedelta(days=i, seconds=200)).isoformat(), 150.0],
                [(start_base + timedelta(days=i, seconds=300)).isoformat(), 100.0],
                [(start_base + timedelta(days=i, seconds=600)).isoformat(), 0.0]
            ],
            "profile_name": "TestProfile"
        }
        await store.async_add_cycle(cycle_data)
        
    # 2. Rebuild envelope
    success = await store.async_rebuild_envelope("TestProfile")
    assert success is True
    
    envelope = store.get_envelope("TestProfile")
    assert envelope is not None
    assert "avg" in envelope
    assert len(envelope["avg"]) > 0
    assert envelope["cycle_count"] == 3

@pytest.mark.asyncio
async def test_profile_matching_logic(store):
    """Test profile matching with synthetic current data."""
    # 1. Setup a reference profile
    start_ref = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    ref_cycle = {
        "id": "ref_cycle",
        "start_time": start_ref.isoformat(),
        "duration": 1000.0,
        "max_power": 500.0,
        "status": "completed",
        "power_data": [
            [start_ref.isoformat(), 0.0],
            [(start_ref + timedelta(seconds=200)).isoformat(), 400.0],
            [(start_ref + timedelta(seconds=400)).isoformat(), 500.0],
            [(start_ref + timedelta(seconds=600)).isoformat(), 450.0],
            [(start_ref + timedelta(seconds=800)).isoformat(), 300.0],
            [(start_ref + timedelta(seconds=1000)).isoformat(), 0.0]
        ],
        "profile_name": "PowerProfile"
    }
    await store.async_add_cycle(ref_cycle)
    await store.create_profile_standalone("PowerProfile", reference_cycle_id="ref_cycle")
    
    # 2. Attempt match with similar data
    now_ts = datetime.now(timezone.utc)
    current_data = []
    for i in range(21): # 20 segments of 25s = 500s
        ts = (now_ts + timedelta(seconds=i*25)).isoformat()
        # Pattern: 0 -> 400 -> 500 -> 450 -> 300 -> 0 (approx)
        if i == 0 or i == 20: p = 0.0
        elif i < 5: p = 200.0 + i*40.0
        elif i < 10: p = 400.0 + (i-5)*20.0
        elif i < 15: p = 500.0 - (i-10)*40.0
        else: p = 300.0 - (i-15)*60.0
        current_data.append((ts, p))
    
    result = await store.async_match_profile(current_data, 500.0)
    
    # Since DTW is involved, we expect some match
    assert result.best_profile == "PowerProfile"
    assert result.confidence > 0.1 # Low because data is sparse, but should be best

@pytest.mark.asyncio
async def test_retention_enforcement(store):
    """Test that retention limits are respected."""
    store._max_past_cycles = 5
    
    start_base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(10):
        cycle_data = {
            "start_time": (start_base + timedelta(hours=i)).isoformat(),
            "duration": 100.0,
            "status": "completed"
        }
        await store.async_add_cycle(cycle_data)
        
    assert len(store.get_past_cycles()) == 5
    # Should be the 5 newest
    newest_start = max(c["start_time"] for c in store.get_past_cycles())
    assert newest_start == (start_base + timedelta(hours=9)).isoformat()

@pytest.mark.asyncio
async def test_profile_rename_and_assignment(store):
    """Test renaming a profile and assigning it to cycles."""
    # 1. Create profile and add cycle
    await store.create_profile_standalone("OldName", avg_duration=1000.0)
    cycle_data = {
        "start_time": "2025-01-01T12:00:00+00:00",
        "duration": 1000.0,
        "status": "completed",
        "profile_name": "OldName"
    }
    await store.async_add_cycle(cycle_data)
    cid = store.get_past_cycles()[0]["id"]
    
    # 2. Rename profile
    await store.update_profile("OldName", "NewName")
    assert "NewName" in store.get_profiles()
    assert "OldName" not in store.get_profiles()
    assert store.get_past_cycles()[0]["profile_name"] == "NewName"
    
    # 3. Assign to None
    await store.assign_profile_to_cycle(cid, None)
    assert store.get_past_cycles()[0]["profile_name"] is None
    
    # 4. Assign to NewName
    await store.assign_profile_to_cycle(cid, "NewName")
    assert store.get_past_cycles()[0]["profile_name"] == "NewName"

@pytest.mark.asyncio
async def test_maintenance_and_orphans(store):
    """Test cleaning up orphaned profiles and maintenance."""
    # 1. Create a profile referencing a non-existent cycle
    store._data["profiles"]["Orphaned"] = {"sample_cycle_id": "non_existent"}
    
    removed = store.cleanup_orphaned_profiles()
    assert removed == 1
    assert "Orphaned" not in store.get_profiles()
    
    # 2. Run full maintenance (should handle empty data gracefully)
    stats = await store.async_run_maintenance()
    assert "orphaned_profiles" in stats

@pytest.mark.asyncio
async def test_match_result_serialization():
    """Test MatchResult to_dict serialization."""
    res = MatchResult(
        best_profile="Test",
        confidence=0.95,
        expected_duration=3600.0,
        matched_phase="Wash",
        candidates=[{"name": "Test", "score": 0.95, "current": [1, 2, 3]}],
        is_ambiguous=False,
        ambiguity_margin=0.5
    )
    
    d = res.to_dict()
    assert d["best_profile"] == "Test"
    assert d["confidence"] == 0.95
    # Should exclude heavy arrays
    assert "current" not in d["candidates"][0]
