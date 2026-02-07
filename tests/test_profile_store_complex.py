import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.profile_store import ProfileStore

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
async def test_repair_profile_samples(store):
    """Test repairing profiles with missing or invalid sample references."""
    # 1. Setup: Profile with missing sample_cycle_id
    store._data["profiles"]["NoSample"] = {"sample_cycle_id": None}
    
    # 2. Setup: Cycle with power_data
    cycle_with_power = {
        "id": "c_power",
        "profile_name": None,
        "power_data": [[0, 0], [10, 100], [20, 0]],
        "duration": 20,
        "start_time": "2025-01-01T10:00:00+00:00"
    }
    store._data["past_cycles"].append(cycle_with_power)
    
    # 3. Repair
    with patch.object(store, "async_rebuild_envelope", AsyncMock(return_value=True)):
        stats = await store.async_repair_profile_samples()
        
    assert stats["profiles_repaired"] == 1
    assert store._data["profiles"]["NoSample"]["sample_cycle_id"] == "c_power"
    assert cycle_with_power["profile_name"] == "NoSample"

@pytest.mark.asyncio
async def test_import_data(store):
    """Test importing data payload."""
    payload = {
        "profiles": {"Imported": {"avg_duration": 1234}},
        "past_cycles": [{"id": "imported_c", "duration": 1234}]
    }
    
    with patch.object(store, "async_rebuild_envelope", AsyncMock(return_value=True)):
        # async_import_data currently returns None despite type hint
        await store.async_import_data(payload)
        
    assert "Imported" in store.get_profiles()
    assert len(store.get_past_cycles()) == 1

@pytest.mark.asyncio
async def test_delete_cycle_variants(store):
    """Test various cycle deletion scenarios."""
    # Setup some cycles
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": "P1"},
        {"id": "c2", "profile_name": "P1"},
        {"id": "c3", "profile_name": "P2"}
    ]
    store._data["profiles"] = {"P1": {}, "P2": {}}
    
    # 1. Delete single cycle
    await store.delete_cycle("c1")
    assert len(store.get_past_cycles()) == 2
    
    # 2. Delete profile (unlabel cycles)
    await store.delete_profile("P1", unlabel_cycles=True)
    assert "P1" not in store.get_profiles()
    # Cycle c2 should now have None as profile_name
    c2 = next(c for c in store.get_past_cycles() if c["id"] == "c2")
    assert c2["profile_name"] is None
    
    # 3. Clear all
    await store.clear_all_data()
    assert len(store.get_past_cycles()) == 0
    assert len(store.get_profiles()) == 0

@pytest.mark.asyncio
async def test_suggestions_management(store):
    """Test setting and getting suggestions."""
    store.set_suggestion("test_key", 100, "Because reasons")
    
    s = store.get_suggestions()
    assert "test_key" in s
    assert s["test_key"]["value"] == 100
    assert s["test_key"]["reason"] == "Because reasons"
    
    # Update same value - should not change updated_at if we tracked it (internal detail)
    store.set_suggestion("test_key", 100, "New reason")
    assert store.get_suggestions()["test_key"]["reason"] == "New reason"

@pytest.mark.asyncio
async def test_active_cycle_persistence(store):
    """Test saving and loading active cycle."""
    snapshot = {"state": "RUNNING", "power": 100}
    await store.async_save_active_cycle(snapshot)
    
    loaded = store.get_active_cycle()
    assert loaded == snapshot
    assert store.get_last_active_save() is not None
    
    await store.async_clear_active_cycle()
    assert store.get_active_cycle() is None