
import pytest
import logging
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Ensure the custom_components directory is in the path
sys.path.append(os.path.abspath("/root/ha_washdata/custom_components"))

from ha_washdata.profile_store import ProfileStore

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    
    # Mock executor job to return result immediately
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
async def test_verify_alignment_with_legacy_envelope(store):
    """
    Test that async_verify_alignment handles legacy envelopes where 'avg' 
    is a list of floats instead of a list of [time, power] pairs.
    """
    profile_name = "LegacyProfile"
    
    # Mock legacy envelope data
    legacy_envelope = {
        "avg": [10.0, 15.0, 5.0, 2.0, 1.0],  # Floats only (legacy)
        "time_grid": [0.0, 60.0, 120.0, 180.0, 240.0],
        "target_duration": 240.0
    }
    
    # Inject into store
    store._data["envelopes"] = {profile_name: legacy_envelope}
    store._data["profiles"] = {profile_name: {}}
    
    # Mock power data (trace)
    current_power_data = [
        ("2026-02-06T12:00:00", 11.0),
        ("2026-02-06T12:01:00", 14.0),
        ("2026-02-06T12:02:00", 4.0),
    ]
    
    # This should trigger the TypeError: 'float' object is not subscriptable
    # in profile_store.py at: env_time = [p[0] for p in env_avg]
    try:
        await store.async_verify_alignment(profile_name, current_power_data)
    except TypeError as e:
        assert "'float' object is not subscriptable" in str(e)
        return

    pytest.fail("Should have raised TypeError: 'float' object is not subscriptable")

@pytest.mark.asyncio
async def test_verify_alignment_with_malformed_envelope(store):
    """
    Test that async_verify_alignment handles cases where 'avg' contains mixed garbage.
    """
    profile_name = "MalformedProfile"
    
    # Mock malformed envelope data
    malformed_envelope = {
        "avg": [ [0.0, 10.0], 15.0, [120.0, 5.0] ],  # Mixed pairs and floats
        "time_grid": [0.0, 60.0, 120.0],
        "target_duration": 120.0
    }
    
    # Inject into store
    store._data["envelopes"] = {profile_name: malformed_envelope}
    store._data["profiles"] = {profile_name: {}}
    
    # Mock power data (trace)
    current_power_data = [
        ("2026-02-06T12:00:00", 11.0),
        ("2026-02-06T12:01:00", 14.0),
    ]
    
    try:
        await store.async_verify_alignment(profile_name, current_power_data)
    except TypeError as e:
        assert "object is not subscriptable" in str(e)
        return

    pytest.fail("Should have raised TypeError")
