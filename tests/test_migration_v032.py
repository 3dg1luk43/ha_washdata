
"""Test for migration from v0.3.2 (Storage v1) to v2."""
import pytest
import json
import os
from unittest.mock import MagicMock, patch

# Ensure mocks are loaded before importing custom_components
import tests.mock_imports
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import STORAGE_KEY, STORAGE_VERSION
from custom_components.ha_washdata.profile_store import WashDataStore, ProfileStore

@pytest.mark.asyncio
async def test_migration_v1_to_v2_logic(mock_hass):
    """Test that loading v1 data triggers migration to v2 and computes signatures."""
    
    # 1. Setup: Create a fake storage file with v1 data (no signatures)
    entry_id = "test_entry_v032"
    store_key = f"{STORAGE_KEY}.{entry_id}"
    
    # Sample power data (v1 format was valid list of [offset, power])
    # Migration requires > 10 points to compute signature
    raw_power_data = [
        [0.0, 0.0], [10.0, 100.0], [20.0, 200.0], [30.0, 150.0],
        [40.0, 100.0], [50.0, 50.0], [60.0, 0.0], [70.0, 0.0],
        [80.0, 20.0], [90.0, 0.0], [100.0, 0.0], [110.0, 0.0]
    ]
    
    v1_data = {
        "version": 1,
        "key": store_key,
        "data": {
            "past_cycles": [
                {
                    "id": "cycle_1",
                    "start_time": "2023-01-01T12:00:00+00:00",
                    "end_time": "2023-01-01T12:02:00+00:00",
                    "duration": 120.0,
                    "power_data": raw_power_data,
                    # NO signature provided in v1
                    "profile_name": "Standard"
                },
                {
                    "id": "cycle_2_no_power",
                    "start_time": "2023-01-02T12:00:00+00:00",
                    "end_time": "2023-01-02T12:05:00+00:00",
                    "duration": 300.0,
                    "power_data": [], 
                    # Should handle gracefully
                }
            ],
            "profiles": {
                "Standard": {
                    "avg_duration": 120.0
                }
            }
        }
    }

    # Mock the Store to read our v1 data
    # We can't easily adhere to the 'private' _load method of generic Store, 
    # so we'll write a real file to the temp directory used by hass fixture if possible,
    # OR we can mock the `load` result.
    # Writing a file is more robust for checking the full stack including version check.
    
    # Helper to write json to the storage path
    def write_storage(data):
        path = mock_hass.config.path(".storage", store_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
            
    write_storage(v1_data)

    # 2. Execution: Load the ProfileStore
    # This initializes WashDataStore which inherits from Store
    store = ProfileStore(mock_hass, entry_id)
    await store.async_load()

    # 3. Assertion: Check In-Memory Data
    cycles = store.get_past_cycles()
    assert len(cycles) == 2, "Should preserve all cycles"
    
    c1 = next(c for c in cycles if c["id"] == "cycle_1")
    assert "signature" in c1, "Migration should compute signature for cycle_1"
    assert c1["signature"]["max_power"] == 200.0, "Signature should reflect max power"
    assert c1["signature"]["total_energy"] >= 0.0
    assert c1["power_data"] == raw_power_data, "Raw power data should be preserved"
    
    c2 = next(c for c in cycles if c["id"] == "cycle_2_no_power")
    assert "signature" not in c2, "Cycle with no power data should not have signature"
    
    # 4. Assertion: Check Persisted Data (Version Update)
    # We need to force a save to see if the version updates on disk
    # The migration logic in WashDataStore returns the migrated data to the caller,
    # but the caller (Store.async_load) doesn't automatically save it back to disk 
    # UNLESS we explicitly save. However, typically Manager calls save periodically.
    # Let's verify `store._data` has the updates.
    
    # Simulate a save to verify the file version updates
    await store.async_save()
    
    # Read the file back directly
    path = mock_hass.config.path(".storage", store_key)
    with open(path, "r") as f:
        saved_data = json.load(f)
        
    assert saved_data["version"] == 2, "Storage version should be updated to 2"
    assert saved_data["data"]["past_cycles"][0].get("signature") is not None
