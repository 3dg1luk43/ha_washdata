
import pytest
import json
import logging
import numpy as np
import glob
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
sys.path.append(os.path.abspath("/root/ha_washdata/custom_components"))

from ha_washdata.profile_store import ProfileStore

_LOGGER = logging.getLogger(__name__)

# Directory containing the data files
DATA_DIR = os.path.join(os.path.dirname(__file__), "../cycle_data")

def get_test_files():
    """Find all JSON config entry exports in cycle_data."""
    abs_data_dir = os.path.abspath(DATA_DIR)
    files = glob.glob(os.path.join(abs_data_dir, "**", "*.json"), recursive=True)
    return sorted(files)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    
    # Mock executor job to return result immediately (simulated async)
    import inspect
    import asyncio
    async def mock_executor_job(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)
        
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    return hass

@pytest.fixture
def store(mock_hass):
    # Mock the Store
    with patch("homeassistant.helpers.storage.Store") as MockStore:
        store_instance = ProfileStore(mock_hass, "test_entry")
        store_instance._store = MockStore.return_value
        store_instance._store.async_load = AsyncMock(return_value=None)
        store_instance._store.async_save = AsyncMock()
        return store_instance

@pytest.mark.parametrize("data_file", get_test_files())
@pytest.mark.asyncio
async def test_envelope_alignment_with_user_data(store, data_file):
    # Load Real Data
    with open(data_file, 'r') as f:
        full_data = json.load(f)
    
    wash_data = full_data.get("data", {}).get("store_data", {})
    if not wash_data:
        print(f"DEBUG: No store_data in {data_file}. Keys: {full_data.keys()}")
        if "profiles" in full_data:
            wash_data = full_data
            print("DEBUG: Using full_data as wash_data")
        else:
            pytest.skip(f"No store_data found in {data_file}")

    # Inject data into store
    store._data = wash_data
    
    # Identify a profile and associated cycles
    profiles = wash_data.get("profiles", {})
    past_cycles = wash_data.get("past_cycles", [])
    
    if not profiles or not past_cycles:
        pytest.skip(f"Insufficient data in {data_file}")

    # Check if any are already labeled
    profile_name = next((c.get("profile_name") for c in past_cycles if c.get("profile_name") in profiles), None)
    
    if not profile_name:
        # Manually assign the first profile to the first cycle for testing purposes
        profile_name = list(profiles.keys())[0]
        past_cycles[0]["profile_name"] = profile_name
        print(f"DEBUG: Manually assigned {profile_name} to first cycle in {data_file}")
    
    # Run Rebuild Envelope
    _LOGGER.info("Rebuilding envelope for %s from %s...", profile_name, os.path.basename(data_file))
    await store.async_rebuild_envelope(profile_name)
    
    envelope = store.get_envelope(profile_name)
    assert envelope is not None
    
    # Check Stats
    std_curve = np.array(envelope["std"])
    avg_std = np.mean(std_curve)
    max_std = np.max(std_curve)
    
    _LOGGER.info("Envelope Stats - Avg STD: %.2f W, Max STD: %.2f W", avg_std, max_std)
    
    # Basic Sanity Checks
    assert len(envelope["avg"]) > 10
