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
    # decompress_power_data now returns (offset_seconds, power) tuples
    offset_1, power_1 = decompressed[1]
    assert offset_1 == pytest.approx(10.0, abs=0.1), "Expected 10s offset"
    assert power_1 == pytest.approx(100.5)


def test_decompress_iso_power_data_with_numeric_start_time_does_not_fail(store):
    """Decompression must tolerate historical non-string start_time values."""
    cycle = {
        "start_time": 1735725600.0,
        "power_data": [
            ["2025-01-01T10:00:00+00:00", 0.0],
            ["2025-01-01T10:00:10+00:00", 50.0],
            ["2025-01-01T10:00:20+00:00", 0.0],
        ],
    }

    decompressed = decompress_power_data(cycle)
    assert len(decompressed) == 3
    assert decompressed[0][0] == pytest.approx(0.0, abs=0.1)
    assert decompressed[1][0] == pytest.approx(10.0, abs=0.1)


def test_compress_power_data_with_numeric_start_time(store):
    """Compression should work when start_time is already a numeric timestamp."""
    cycle = {
        "start_time": 1735725600.0,
        "power_data": [
            ["2025-01-01T10:00:00+00:00", 0.0],
            ["2025-01-01T10:00:10+00:00", 100.5],
            ["2025-01-01T10:00:20+00:00", 0.0],
        ],
    }

    compressed = compress_power_data(cycle)
    assert isinstance(compressed, list)
    assert compressed[0] == [0.0, 0.0]
    assert compressed[1] == [10.0, 100.5]


@pytest.mark.asyncio
async def test_async_add_cycle_with_numeric_start_time(store):
    """Adding a cycle should not raise when start_time is numeric."""
    cycle_data = {
        "start_time": 1735725600.0,
        "duration": 20,
        "status": "completed",
        "power_data": [
            [1735725600.0, 0.0],
            [1735725610.0, 90.0],
            [1735725620.0, 0.0],
        ],
    }

    await store.async_add_cycle(cycle_data)

    assert len(store._data["past_cycles"]) == 1
    saved = store._data["past_cycles"][0]
    assert saved["power_data"][0][0] == pytest.approx(10.0, abs=0.1)
    assert saved["power_data"][1][0] == pytest.approx(20.0, abs=0.1)

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