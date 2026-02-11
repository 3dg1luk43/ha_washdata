import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.sensor import async_setup_entry, WasherTotalDurationSensor
from custom_components.ha_washdata.const import DOMAIN, STATE_RUNNING

@pytest.mark.asyncio
async def test_total_duration_sensor_registered(hass, mock_config_entry):
    """Verify that the total duration sensor is registered."""
    manager = MagicMock()
    hass.data[DOMAIN] = {mock_config_entry.entry_id: manager}
    
    async_add_entities = MagicMock()
    
    await async_setup_entry(hass, mock_config_entry, async_add_entities)
    
    # Get the list of entities passed to async_add_entities
    added_entities = async_add_entities.call_args[0][0]
    
    # Extract keys from entity descriptions
    keys = [entity.entity_description.key for entity in added_entities]
    
    assert "total_duration" in keys

@pytest.mark.asyncio
async def test_total_duration_sensor_value(hass, mock_config_entry):
    """Verify the total duration sensor value."""
    manager = MagicMock()
    manager.check_state = STATE_RUNNING
    # Elapsed = 10 mins (600s), Remaining = 20 mins (1200s)
    manager.get_elapsed_seconds.return_value = 600.0
    manager.time_remaining = 1200.0
    
    sensor = WasherTotalDurationSensor(manager, mock_config_entry)
    
    # Value should be 30 mins
    assert sensor.native_value == 30
    assert sensor.native_unit_of_measurement == "min"
    assert sensor.device_class == "duration"

@pytest.mark.asyncio
async def test_total_duration_sensor_unknown_if_not_matched(hass, mock_config_entry):
    """Verify the total duration sensor is None if no match or not running."""
    manager = MagicMock()
    manager.check_state = STATE_RUNNING
    manager.time_remaining = None
    
    sensor = WasherTotalDurationSensor(manager, mock_config_entry)
    assert sensor.native_value is None
    
    manager.check_state = "off"
    assert sensor.native_value is None