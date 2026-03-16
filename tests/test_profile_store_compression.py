import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.profile_store import ProfileStore, compress_power_data, decompress_power_data

@pytest.fixture
def store(mock_hass):
    with patch("homeassistant.helpers.storage.Store") as MockStore:
        store_instance = ProfileStore(mock_hass, "test_entry")
        store_instance._store = MockStore.return_value
        store_instance._store.async_load = AsyncMock(return_value=None)
        store_instance._store.async_save = AsyncMock()
        return store_instance

def test_compression_decompression(store):
    """Test cycle power data compression and decompression."""
    raw_data = [
        ["2025-01-01T10:00:00+00:00", 0.0],
        ["2025-01-01T10:00:10+00:00", 100.5],
        ["2025-01-01T10:00:20+00:00", 0.0]
    ]
    
    cycle = {
        "start_time": "2025-01-01T10:00:00+00:00",
        "power_data": raw_data
    }
    
    # Compress (Global function)
    compressed = compress_power_data(cycle)
    assert isinstance(compressed, list)
    # Check format: [offset, power]
    assert compressed[0] == [0.0, 0.0]
    assert compressed[1] == [10.0, 100.5]
    
    # Decompress (Global function)
    cycle_compressed = {"start_time": cycle["start_time"], "power_data": compressed}
    decompressed = decompress_power_data(cycle_compressed)
    
    assert len(decompressed) == 3
    # Use fromisoformat and check total seconds since start to be timezone agnostic
    start_dt = datetime.fromisoformat(cycle["start_time"])
    p1_dt = datetime.fromisoformat(decompressed[1][0])
    
    # We expect 10s offset
    assert (p1_dt.timestamp() - start_dt.timestamp()) == 10.0
    assert decompressed[1][1] == 100.5

@pytest.mark.asyncio
async def test_migration_to_compressed(store):
    """Test migrating v1 (full ISO strings) to v2 (compressed offsets)."""
    raw_data = [
        ["2025-01-01T10:00:00+00:00", 0.0],
        ["2025-01-01T10:00:10+00:00", 100.0]
    ]
    store._data["past_cycles"] = [
        {"id": "c1", "start_time": "2025-01-01T10:00:00+00:00", "power_data": raw_data}
    ]
    
    count = await store.async_migrate_cycles_to_compressed()
    assert count == 1
    
    migrated_cycle = store.get_past_cycles()[0]
    # Should be offset based now
    assert migrated_cycle["power_data"][1] == [10.0, 100.0]

def test_envelope_extraction(store):
    """Test extracting envelope data for UI."""
    store._data["envelopes"]["Test"] = {
        "avg": [[0, 0], [10, 100], [20, 0]],
        "min": [[0, 0], [10, 80], [20, 0]],
        "max": [[0, 0], [10, 120], [20, 0]],
        "cycle_count": 5
    }
    
    env = store.get_envelope("Test")
    assert env["cycle_count"] == 5
    assert len(env["avg"]) == 3
    
    # Missing profile
    assert store.get_envelope("Missing") is None