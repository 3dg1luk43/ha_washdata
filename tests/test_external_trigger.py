"""Unit tests for External Cycle End Trigger."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime, timezone
from homeassistant.core import Event, State
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_POWER_SENSOR, STATE_RUNNING, STATE_OFF,
    CONF_EXTERNAL_END_TRIGGER_ENABLED, CONF_EXTERNAL_END_TRIGGER,
    CONF_EXTERNAL_END_TRIGGER_INVERTED
)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.config_entries.async_get_entry = MagicMock()
    return hass

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_POWER_SENSOR: "sensor.test_power",
        CONF_EXTERNAL_END_TRIGGER_ENABLED: True,
        CONF_EXTERNAL_END_TRIGGER: "binary_sensor.test_trigger",
        CONF_EXTERNAL_END_TRIGGER_INVERTED: False,
    }
    entry.data = {}
    return entry

@pytest.fixture
def manager(mock_hass, mock_entry):
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"), \
         patch("custom_components.ha_washdata.manager.CycleRecorder"), \
         patch("custom_components.ha_washdata.manager.LearningManager"):
        
        mgr = WashDataManager(mock_hass, mock_entry)
        # Direct override of detector to ensure we have a clean mock for verification
        mgr.detector = MagicMock()
        type(mgr.detector).state = PropertyMock(return_value=STATE_RUNNING)
        return mgr

def create_state(state_val: str):
    """Helper to create a mock state object."""
    s = MagicMock(spec=State)
    s.state = state_val
    return s

def test_external_trigger_normal_logic(manager):
    """Test trigger fires on ON transition when not inverted."""
    manager.config_entry.options[CONF_EXTERNAL_END_TRIGGER_INVERTED] = False
    
    # 1. Transition from OFF to ON (Should trigger)
    event = MagicMock(spec=Event)
    event.data = {
        "entity_id": "binary_sensor.test_trigger",
        "new_state": create_state("on"),
        "old_state": create_state("off"),
    }
    manager._handle_external_trigger_change(event)
    manager.detector.user_stop.assert_called_once()
    manager.detector.user_stop.reset_mock()

    # 2. Transition from ON to OFF (Should NOT trigger)
    event.data = {
        "entity_id": "binary_sensor.test_trigger",
        "new_state": create_state("off"),
        "old_state": create_state("on"),
    }
    manager._handle_external_trigger_change(event)
    manager.detector.user_stop.assert_not_called()

def test_external_trigger_inverted_logic(manager):
    """Test trigger fires on OFF transition when inverted."""
    manager.config_entry.options[CONF_EXTERNAL_END_TRIGGER_INVERTED] = True
    
    # 1. Transition from ON to OFF (Should trigger)
    event = MagicMock(spec=Event)
    event.data = {
        "entity_id": "binary_sensor.test_trigger",
        "new_state": create_state("off"),
        "old_state": create_state("on"),
    }
    manager._handle_external_trigger_change(event)
    manager.detector.user_stop.assert_called_once()
    manager.detector.user_stop.reset_mock()

    # 2. Transition from OFF to ON (Should NOT trigger)
    event.data = {
        "entity_id": "binary_sensor.test_trigger",
        "new_state": create_state("on"),
        "old_state": create_state("off"),
    }
    manager._handle_external_trigger_change(event)
    manager.detector.user_stop.assert_not_called()

def test_external_trigger_no_state_change(manager):
    """Test trigger does not fire if state hasn't changed."""
    manager.config_entry.options[CONF_EXTERNAL_END_TRIGGER_INVERTED] = False
    
    event = MagicMock(spec=Event)
    event.data = {
        "entity_id": "binary_sensor.test_trigger",
        "new_state": create_state("on"),
        "old_state": create_state("on"),
    }
    manager._handle_external_trigger_change(event)
    manager.detector.user_stop.assert_not_called()