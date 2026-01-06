"""Tests for config entry migration."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from homeassistant.const import CONF_DEVICE_ID
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from custom_components.ha_washdata.const import (
    DOMAIN,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_DEVICE_TYPE,
    CONF_POWER_SENSOR,
    CONF_NOTIFY_SERVICE,
)
from custom_components.ha_washdata import async_migrate_entry

@pytest.fixture
def mock_config_entry():
    entry = MagicMock()
    entry.domain = DOMAIN
    entry.title = "Test Washer"
    entry.entry_id = "test_entry_id"
    entry.version = 1
    entry.data = {
        CONF_MIN_POWER: 5.0,
        CONF_OFF_DELAY: 120,
        CONF_DEVICE_TYPE: "Washing Machine",
        CONF_POWER_SENSOR: "sensor.washer_power",
        CONF_NOTIFY_SERVICE: "notify.mobile_app",
        "some_other_key": "some_value" # Should remain
    }
    entry.options = {}
    return entry

@pytest.mark.asyncio
async def test_migration_v1_to_v3(mock_hass, mock_config_entry):
    """Test migration from version 1 to 3 moves data to options."""
    
    # Run migration
    mock_hass.config_entries.async_update_entry = AsyncMock()
    
    result = await async_migrate_entry(mock_hass, mock_config_entry)
    
    assert result is True
    
    # Verify update called with version 3
    args = mock_hass.config_entries.async_update_entry.call_args
    assert args is not None
    _, kwargs = args
    
    assert kwargs["version"] == 3
    
    # Verify data cleanup
    new_data = kwargs["data"]
    assert CONF_MIN_POWER not in new_data
    assert CONF_OFF_DELAY not in new_data
    assert CONF_DEVICE_TYPE not in new_data
    assert CONF_POWER_SENSOR not in new_data
    assert CONF_NOTIFY_SERVICE not in new_data
    assert "some_other_key" in new_data # Should be preserved if not in removal list
    
    # Verify options population
    new_options = kwargs["options"]
    assert new_options[CONF_MIN_POWER] == 5.0
    assert new_options[CONF_OFF_DELAY] == 120
    assert new_options[CONF_DEVICE_TYPE] == "Washing Machine"
    assert new_options[CONF_POWER_SENSOR] == "sensor.washer_power"
    # CONF_NOTIFY_SERVICE might not be moved if it wasn't in list?
    # Wait, I added it to remove list, but did I add logic to move it?
    # I should check __init__.py again.
