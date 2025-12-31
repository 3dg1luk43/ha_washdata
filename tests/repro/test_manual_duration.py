import pytest
from unittest.mock import MagicMock
from homeassistant.core import HomeAssistant
from custom_components.ha_washdata.profile_store import ProfileStore

@pytest.mark.asyncio
async def test_manual_duration_creation(mock_hass: HomeAssistant):
    """Test creating a profile with manual duration."""
    store = ProfileStore(mock_hass, MagicMock(), "test_entry")
    # Mock internal data structure
    store._data = {"profiles": {}, "past_cycles": []}
    
    # Mock async_save on the instance to bypass storage
    store.async_save = MagicMock(side_effect=lambda: None)
    async def noop_save(): return
    store.async_save = noop_save
    
    # Create profile with manual duration
    await store.create_profile_standalone(
        name="Manual 30m",
        avg_duration=1800.0  # 30 minutes in seconds
    )
    
    profiles = store.get_profiles()
    assert "Manual 30m" in profiles
    profile = profiles["Manual 30m"]
    assert profile["avg_duration"] == 1800.0
    
    # Create profile WITHOUT manual duration
    await store.create_profile_standalone(
        name="Empty Profile"
    )
    
    profiles = store.get_profiles()
    assert "Empty Profile" in profiles
    empty_profile = profiles["Empty Profile"]
    assert "avg_duration" not in empty_profile
