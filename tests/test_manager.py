"""Unit tests for WashDataManager."""
from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS, CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_POWER_SENSOR, STATE_RUNNING, STATE_OFF
)

@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    # Prevent 'coroutine was never awaited' warnings when code schedules tasks.
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()  # type: ignore[misc]
    )
    hass.components.persistent_notification.async_create = MagicMock()
    return hass

@pytest.fixture
def mock_entry() -> Any:
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
def manager(mock_hass: Any, mock_entry: Any) -> WashDataManager:
    # Patch ProfileStore and CycleDetector to avoid disk/logic issues
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store._data = {"profiles": {"Heavy Duty": {"avg_duration": 3600}}}
        return mgr

def test_init(manager: WashDataManager, mock_entry: Any) -> None:
    """Test initialization."""
    assert manager.entry_id == "test_entry"
    assert manager._config.completion_min_seconds == 600
    assert manager._notify_before_end_minutes == 5

def test_set_manual_program(manager: WashDataManager) -> None:
    """Test setting manual program."""
    # Mock profile data
    manager.profile_store._data["profiles"] = {
        "Heavy Duty": {"avg_duration": 3600}
    }
    
    manager.set_manual_program("Heavy Duty")
    
    assert manager.current_program == "Heavy Duty"
    assert manager.manual_program_active is True
    assert manager._matched_profile_duration == 3600

def test_set_manual_program_invalid(manager: WashDataManager) -> None:
    """Test setting invalid manual program."""
    manager.profile_store._data["profiles"] = {}
    manager.set_manual_program("Ghost")
    
    # Initially state is 'off', so current_program returns 'off'
    assert manager.current_program == "off"
    assert manager.manual_program_active is False

def test_check_pre_completion_notification(manager: WashDataManager, mock_hass: Any) -> None:
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

def test_check_pre_completion_notification_already_sent(manager: WashDataManager, mock_hass: Any) -> None:
    """Test it doesn't send twice."""
    manager._time_remaining = 240
    manager._notify_before_end_minutes = 5
    manager._notified_pre_completion = True
    
    manager._check_pre_completion_notification()
    
    # Still 1 from previous turn if it was persistent, but here we expect no NEW call
    assert mock_hass.components.persistent_notification.async_create.call_count == 0

def test_check_pre_completion_disabled(manager: WashDataManager, mock_hass: Any) -> None:
    """Test disabled notification."""
    manager._notify_before_end_minutes = 0
    manager._time_remaining = 60
    manager._check_pre_completion_notification()
    assert mock_hass.components.persistent_notification.async_create.call_count == 0


def test_cycle_end_requests_feedback(manager: WashDataManager, mock_hass: Any) -> None:
    """Cycle end should request feedback (event + persistent notification) before state is cleared."""
    # Arrange: pretend we had a confident match
    manager.profile_store._data["profiles"] = {"Heavy Duty": {"avg_duration": 3600}}
    manager._current_program = "Heavy Duty"
    manager._matched_profile_duration = 3600
    manager._last_match_confidence = 0.80
    manager._learning_confidence = 0.70
    manager._auto_label_confidence = 0.95

    cycle_data = {
        "start_time": "2025-12-21T10:00:00",
        "end_time": "2025-12-21T11:00:00",
        "duration": 3600,
        "max_power": 500,
        "power_data": [[0.0, 5.0], [60.0, 200.0], [120.0, 50.0]],
        "status": "completed",
    }

    # Act
    manager._on_cycle_end(dict(cycle_data))

    # Assert: feedback event fired and notification created
    fired_events = [c[0][0] for c in mock_hass.bus.async_fire.call_args_list]
    assert "ha_washdata_feedback_requested" in fired_events
    assert mock_hass.components.persistent_notification.async_create.call_count >= 1


def test_cycle_end_auto_labels_high_confidence(manager: WashDataManager, mock_hass: Any) -> None:
    """High-confidence matches should auto-label and not request user feedback."""
    manager.profile_store._data["profiles"] = {"Heavy Duty": {"avg_duration": 3600}}
    manager._current_program = "Heavy Duty"
    manager._matched_profile_duration = 3600
    manager._last_match_confidence = 0.98
    manager._learning_confidence = 0.70
    manager._auto_label_confidence = 0.95

    manager.learning_manager.auto_label_high_confidence = MagicMock(return_value=True)
    manager.learning_manager.request_cycle_verification = MagicMock()

    cycle_data = {
        "start_time": "2025-12-21T10:00:00",
        "end_time": "2025-12-21T11:00:00",
        "duration": 3600,
        "max_power": 500,
        "power_data": [[0.0, 5.0], [60.0, 200.0], [120.0, 50.0]],
        "status": "completed",
    }

    manager._on_cycle_end(dict(cycle_data))

    manager.learning_manager.auto_label_high_confidence.assert_called_once()
    manager.learning_manager.request_cycle_verification.assert_not_called()

    fired_events = [c[0][0] for c in mock_hass.bus.async_fire.call_args_list]
    assert "ha_washdata_feedback_requested" not in fired_events
    # No feedback prompt should be created in auto-label path.
    assert mock_hass.components.persistent_notification.async_create.call_count == 0


def test_cycle_end_skips_feedback_low_confidence(manager: WashDataManager, mock_hass: Any) -> None:
    """Low-confidence matches should neither auto-label nor request user feedback."""
    manager.profile_store._data["profiles"] = {"Heavy Duty": {"avg_duration": 3600}}
    manager._current_program = "Heavy Duty"
    manager._matched_profile_duration = 3600
    manager._last_match_confidence = 0.40
    manager._learning_confidence = 0.70
    manager._auto_label_confidence = 0.95

    manager.learning_manager.auto_label_high_confidence = MagicMock(return_value=False)
    manager.learning_manager.request_cycle_verification = MagicMock()

    cycle_data = {
        "start_time": "2025-12-21T10:00:00",
        "end_time": "2025-12-21T11:00:00",
        "duration": 3600,
        "max_power": 500,
        "power_data": [[0.0, 5.0], [60.0, 200.0], [120.0, 50.0]],
        "status": "completed",
    }

    manager._on_cycle_end(dict(cycle_data))

    manager.learning_manager.auto_label_high_confidence.assert_not_called()
    manager.learning_manager.request_cycle_verification.assert_not_called()

    fired_events = [c[0][0] for c in mock_hass.bus.async_fire.call_args_list]
    assert "ha_washdata_feedback_requested" not in fired_events
    assert mock_hass.components.persistent_notification.async_create.call_count == 0


@pytest.mark.asyncio
async def test_async_reload_config_blocks_sensor_change_during_active_cycle(
    manager: WashDataManager, mock_entry: Any, mock_hass: Any
) -> None:
    """Test that power sensor changes are blocked when a cycle is active."""
    # Setup: simulate an active cycle
    manager.detector.state = STATE_RUNNING
    original_sensor = manager.power_sensor_entity_id

    # Create a new config entry with a different power sensor
    new_entry = MagicMock()
    new_entry.entry_id = "test_entry"
    new_entry.options = {
        CONF_POWER_SENSOR: "sensor.new_power",
        CONF_MIN_POWER: 2.0,
        CONF_COMPLETION_MIN_SECONDS: 600,
        CONF_NOTIFY_BEFORE_END_MINUTES: 5,
    }
    new_entry.data = {}
    
    # Act: try to reload config with new sensor while cycle is active
    await manager.async_reload_config(new_entry)
    
    # Assert: power sensor should NOT have changed
    assert manager.power_sensor_entity_id == original_sensor
    assert manager.power_sensor_entity_id != "sensor.new_power"


@pytest.mark.asyncio
async def test_async_reload_config_allows_sensor_change_when_idle(
    mock_hass: Any, mock_entry: Any
) -> None:
    """Test that power sensor changes are allowed when no cycle is active."""
    # Setup: create manager with patched dependencies
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"), \
         patch("custom_components.ha_washdata.manager.async_track_state_change_event") as mock_track:
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store.get_duration_ratio_limits = MagicMock(return_value=(0.7, 1.3))
        mgr.profile_store.set_duration_ratio_limits = MagicMock()
        mgr.profile_store.get_active_cycle = MagicMock(return_value=None)
        mgr.profile_store.async_clear_active_cycle = AsyncMock()
        mgr._setup_maintenance_scheduler = AsyncMock()
        
        # Simulate idle state
        mgr.detector.state = STATE_OFF
        original_sensor = mgr.power_sensor_entity_id
        
        # Mock the hass.states.get to return a valid state
        mock_state = MagicMock()
        mock_state.state = "10.5"
        mock_hass.states.get = MagicMock(return_value=mock_state)
        
        # Create a new config entry with a different power sensor
        new_entry = MagicMock()
        new_entry.entry_id = "test_entry"
        new_entry.options = {
            CONF_POWER_SENSOR: "sensor.new_power",
            CONF_MIN_POWER: 2.0,
            CONF_COMPLETION_MIN_SECONDS: 600,
            CONF_NOTIFY_BEFORE_END_MINUTES: 5,
        }
        new_entry.data = {}
        
        # Act: reload config with new sensor while idle
        await mgr.async_reload_config(new_entry)
        
        # Assert: power sensor should have changed
        assert mgr.power_sensor_entity_id == "sensor.new_power"
        assert mgr.power_sensor_entity_id != original_sensor
        # Verify new listener was attached
        mock_track.assert_called()

