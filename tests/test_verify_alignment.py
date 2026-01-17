
import pytest
import json
import logging
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
sys.path.append(os.path.abspath("/root/ha_washdata/custom_components"))

from ha_washdata.profile_store import ProfileStore

_LOGGER = logging.getLogger(__name__)

DATA_PATH = "/root/ha_washdata/cycle_data/me/testmachine/test-data-envelope-shift.json"

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
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

@pytest.mark.asyncio
async def test_envelope_alignment_with_user_data(store):
    # Load Real Data
    with open(DATA_PATH, 'r') as f:
        full_data = json.load(f)
    
    # Extract just the washdata part
    # Structure seems to be a full HA diagnostics dump?
    # Let's inspect structure from previous `view_file`.
    # It has "data": { "entry": ..., "store_data": { "profiles": ..., "past_cycles": ... } }
    
    wash_data = full_data.get("data", {}).get("store_data", {})
    if not wash_data:
        # Maybe it's directly the store data?
        # Check keys
        pass

    # Inject data into store
    store._data = wash_data
    
    # Identify the problematic profile
    profile_name = "1:37 bavlna"
    assert profile_name in store._data["profiles"], "Profile not found in test data"
    
    # Run Rebuild Envelope
    _LOGGER.info("Rebuilding envelope for %s...", profile_name)
    result = store.rebuild_envelope(profile_name)
    assert result is True, "Rebuild failed"
    
    envelope = store.get_envelope(profile_name)
    assert envelope is not None
    
    # Check Stats
    std_curve = np.array(envelope["std"])
    avg_std = np.mean(std_curve)
    max_std = np.max(std_curve)
    
    _LOGGER.info("Envelope Stats - Avg STD: %.2f W, Max STD: %.2f W", avg_std, max_std)
    
    # Basic Sanity Checks
    # If alignment works, the envelope shouldn't be "empty" or "all zeros"
    assert len(envelope["avg"]) > 50
    assert max_std < 2000  # Sanity check, max power is ~2000W
    
    # Verify durations list was computed (my fix)
    # It's an internal variable, but we can check if 'duration_std_dev' is present in envelope
    assert "duration_std_dev" in envelope
