"""Unit tests for WashDataManager."""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS, CONF_NOTIFY_BEFORE_END_MINUTES
)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.components.persistent_notification.async_create = MagicMock()
    return hass

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_COMPLETION_MIN_SECONDS: 600,
        CONF_NOTIFY_BEFORE_END_MINUTES: 5,
        "power_sensor": "sensor.test_power"
    }
    return entry

@pytest.fixture
def manager(mock_hass, mock_entry):
    # Patch ProfileStore and CycleDetector to avoid disk/logic issues
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store._data = {"profiles": {"Heavy Duty": {"avg_duration": 3600}}}
        return mgr

def test_init(manager, mock_entry):
    """Test initialization."""
    assert manager.entry_id == "test_entry"
    assert manager._config.completion_min_seconds == 600
    assert manager._notify_before_end_minutes == 5

def test_set_manual_program(manager):
    """Test setting manual program."""
    # Mock profile data
    manager.profile_store._data["profiles"] = {
        "Heavy Duty": {"avg_duration": 3600}
    }
    
    manager.set_manual_program("Heavy Duty")
    
    assert manager.current_program == "Heavy Duty"
    assert manager.manual_program_active is True
    assert manager._matched_profile_duration == 3600

def test_set_manual_program_invalid(manager):
    """Test setting invalid manual program."""
    manager.profile_store._data["profiles"] = {}
    manager.set_manual_program("Ghost")
    
    # Initially state is 'off', so current_program returns 'off'
    assert manager.current_program == "off"
    assert manager.manual_program_active is False

def test_check_pre_completion_notification(manager, mock_hass):
    """Test the pre-completion notification trigger."""
    manager._time_remaining = 240 # 4 minutes remaining
    manager._notify_before_end_minutes = 5
    manager._notified_pre_completion = False
    manager._cycle_progress = 90
    
    manager._check_pre_completion_notification()
    
    assert manager._notified_pre_completion is True
    # Verify persistent notification called since no notify_service configured
    mock_hass.components.persistent_notification.async_create.assert_called_once()
    args = mock_hass.components.persistent_notification.async_create.call_args[0]
    assert "5 minutes remaining" in args[0]

def test_check_pre_completion_notification_already_sent(manager, mock_hass):
    """Test it doesn't send twice."""
    manager._time_remaining = 240
    manager._notify_before_end_minutes = 5
    manager._notified_pre_completion = True
    
    manager._check_pre_completion_notification()
    
    # Still 1 from previous turn if it was persistent, but here we expect no NEW call
    assert mock_hass.components.persistent_notification.async_create.call_count == 0

def test_check_pre_completion_disabled(manager, mock_hass):
    """Test disabled notification."""
    manager._notify_before_end_minutes = 0
    manager._time_remaining = 60
    manager._check_pre_completion_notification()
    assert mock_hass.components.persistent_notification.async_create.call_count == 0
