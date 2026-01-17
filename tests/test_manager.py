"""Unit tests for WashDataManager."""
from __future__ import annotations

import pytest
from tests import mock_imports
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import timedelta, datetime, timezone
from homeassistant.util import dt as dt_util
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
    # Mock config entries lookups
    hass.config_entries.async_get_entry = MagicMock()
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
    # Setup mock_hass to return our mock_entry
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    
    # Ensure dt_util.now returns real datetimes for comparisons
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)

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


@pytest.mark.asyncio
async def test_cycle_end_requests_feedback(manager: WashDataManager, mock_hass: Any) -> None:
    """Cycle end should request feedback (event + persistent notification) before state is cleared."""
    # Arrange: pretend we had a confident match
    manager.profile_store._data["profiles"] = {"Heavy Duty": {"avg_duration": 3600}}
    manager._current_program = "Heavy Duty"
    manager._matched_profile_duration = 3600
    manager._last_match_confidence = 0.80
    manager._learning_confidence = 0.70
    manager._auto_label_confidence = 0.95

    # Mock async methods called in _async_process_cycle_end
    # Create a mock MatchResult
    mock_res = MagicMock()
    mock_res.best_profile = "Heavy Duty"
    mock_res.confidence = 0.80
    mock_res.ranking = []
    mock_res.debug_details = {}
    mock_res.is_ambiguous = False
    
    manager.profile_store.async_match_profile = AsyncMock(return_value=mock_res)
    manager.profile_store.async_add_cycle = AsyncMock()
    manager.profile_store.async_rebuild_envelope = AsyncMock()
    manager.profile_store.async_clear_active_cycle = AsyncMock()
    manager._run_post_cycle_processing = AsyncMock()

    cycle_data = {
        "start_time": "2025-12-21T10:00:00",
        "end_time": "2025-12-21T11:00:00",
        "duration": 3600,
        "max_power": 500,
        "power_data": [[0.0, 5.0], [60.0, 200.0], [120.0, 50.0]],
        "status": "completed",
    }

    # Act: call async method directly
    await manager._async_process_cycle_end(dict(cycle_data))

    # Assert: feedback event fired and notification created
    # Check that service call was made
    mock_hass.services.async_call.assert_called()
    # verify one of the calls was for persistent_notification.create
    found = False
    for call in mock_hass.services.async_call.call_args_list:
        if call.args[0] == "persistent_notification" and call.args[1] == "create":
            found = True
            break
    assert found



@pytest.mark.asyncio
async def test_cycle_end_auto_labels_high_confidence(manager: WashDataManager, mock_hass: Any) -> None:
    """High-confidence matches should auto-label and not request user feedback."""
    manager.profile_store._data["profiles"] = {"Heavy Duty": {"avg_duration": 3600}}
    manager._current_program = "Heavy Duty"
    manager._matched_profile_duration = 3600
    manager._last_match_confidence = 0.98
    manager._learning_confidence = 0.70
    manager._auto_label_confidence = 0.95

    manager.learning_manager.auto_label_high_confidence = MagicMock(return_value=True)
    manager.learning_manager.request_cycle_verification = MagicMock()

    # Mocks
    mock_res = MagicMock()
    mock_res.best_profile = "Heavy Duty"
    mock_res.confidence = 0.98
    
    manager.profile_store.async_match_profile = AsyncMock(return_value=mock_res)
    manager.profile_store.async_add_cycle = AsyncMock()
    manager.profile_store.async_rebuild_envelope = AsyncMock()
    manager.profile_store.async_clear_active_cycle = AsyncMock()
    manager._run_post_cycle_processing = AsyncMock()

    cycle_data = {
        "start_time": "2025-12-21T10:00:00",
        "end_time": "2025-12-21T11:00:00",
        "duration": 3600,
        "max_power": 500,
        "power_data": [[0.0, 5.0], [60.0, 200.0], [120.0, 50.0]],
        "status": "completed",
    }

    await manager._async_process_cycle_end(dict(cycle_data))

    manager.learning_manager.auto_label_high_confidence.assert_called_once()
    manager.learning_manager.request_cycle_verification.assert_not_called()

    # assert "ha_washdata_feedback_requested" not in fired_events
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

    # assert "ha_washdata_feedback_requested" not in fired_events
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
        mgr.profile_store.get_active_cycle = MagicMock(return_value={"manual_program": False})
        mgr.profile_store.get_past_cycles = MagicMock(return_value=[])
        mgr.profile_store.get_last_active_save = MagicMock(return_value=None)
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


def test_cycle_start_time_exposed(manager: WashDataManager) -> None:
    """Test that cycle_start_time is correctly exposed from detector."""
    # Since detector is mocked in the manager fixture, we can just set the property on the mock
    import datetime
    now = datetime.datetime(2025, 1, 1, 12, 0, 0)
    
    # Configure the mock to return a value for the property
    type(manager.detector).current_cycle_start = PropertyMock(return_value=now)
    assert manager.cycle_start_time == now
    
    # Test when None
    type(manager.detector).current_cycle_start = PropertyMock(return_value=None)
    assert manager.cycle_start_time is None

@pytest.mark.asyncio
async def test_restore_active_cycle_paused(manager: WashDataManager) -> None:
    """Test restoring a cycle that was in PAUSED state."""
    # Setup mocks
    manager.profile_store.async_clear_active_cycle = AsyncMock()
    
    # When restore is called, update the mock state to 'paused'
    def restore_side_effect(snapshot: dict) -> None:
        type(manager.detector).state = PropertyMock(return_value=snapshot["state"])
        type(manager.detector).matched_profile = PropertyMock(return_value=snapshot.get("matched_profile"))
    
    manager.detector.restore_state_snapshot.side_effect = restore_side_effect
    # Initial state
    type(manager.detector).state = PropertyMock(return_value="off")
    type(manager.detector).matched_profile = PropertyMock(return_value=None)

    # Setup snapshot with 'paused' state
    now = dt_util.now()
    snapshot = {
        "state": "paused",
        "sub_state": "Pausing",
        "current_cycle_start": (now - timedelta(minutes=30)).isoformat(),
        "accumulated_energy_wh": 0.5,
        "matched_profile": "Heavy Duty",
        "manual_program": True,
    }
    
    manager.profile_store.get_active_cycle = MagicMock(return_value=snapshot)
    manager.profile_store.get_last_active_save = MagicMock(return_value=now - timedelta(minutes=5))
    
    # Act
    await manager._attempt_state_restoration()
    
    # Assert
    assert manager.detector.state == "paused"
    assert manager.current_program == "Heavy Duty"
    assert manager.manual_program_active is True
    # Should start watchdog
    assert manager._remove_watchdog is not None

@pytest.mark.asyncio
async def test_restore_active_cycle_ending(manager: WashDataManager) -> None:
    """Test restoring a cycle that was in ENDING state."""
    # Setup mocks
    manager.profile_store.async_clear_active_cycle = AsyncMock()
    
    def restore_side_effect(snapshot: dict) -> None:
        type(manager.detector).state = PropertyMock(return_value=snapshot["state"])
        type(manager.detector).matched_profile = PropertyMock(return_value=snapshot.get("matched_profile"))
    
    manager.detector.restore_state_snapshot.side_effect = restore_side_effect
    type(manager.detector).state = PropertyMock(return_value="off")
    type(manager.detector).matched_profile = PropertyMock(return_value=None)

    # Setup snapshot with 'ending' state
    now = dt_util.now()
    snapshot = {
        "state": "ending",
        "sub_state": "Spinning Down",
        "current_cycle_start": (now - timedelta(minutes=60)).isoformat(),
        "accumulated_energy_wh": 1.2,
        "matched_profile": "Normal",
    }
    
    manager.profile_store.get_active_cycle = MagicMock(return_value=snapshot)
    manager.profile_store.get_last_active_save = MagicMock(return_value=now - timedelta(minutes=2))
    
    # Act
    await manager._attempt_state_restoration()
    
    # Assert
    assert manager.detector.state == "ending"
    assert manager.current_program == "Normal"
    # Should start watchdog
    assert manager._remove_watchdog is not None

@pytest.mark.asyncio
async def test_cycle_end_auto_labels_unmatched_cycle(manager: WashDataManager, mock_hass: Any) -> None:
    """Test that _on_cycle_end attempts to auto-label an unmatched cycle."""
    manager._auto_label_confidence = 0.8
    
    # Mock profile store behavior
    from custom_components.ha_washdata.profile_store import MatchResult
    match_result = MatchResult(
        best_profile="DerivedProfile",
        confidence=0.9,
        expected_duration=3600.0,
        matched_phase=None,
        candidates=[],
        is_ambiguous=False, 
        ambiguity_margin=0.0
    )
    # Use AsyncMock for async_match_profile
    manager.profile_store.async_match_profile = AsyncMock(return_value=match_result)
    manager.profile_store.async_add_cycle = AsyncMock()
    manager.profile_store.async_save = AsyncMock()
    manager.profile_store.async_rebuild_envelope = AsyncMock()
    manager.profile_store.async_clear_active_cycle = AsyncMock()
    # Mock _run_post_cycle_processing to avoid errors
    manager._run_post_cycle_processing = AsyncMock()

    cycle_data = {
        "start_time": "2025-01-01T12:00:00",
        "duration": 3600,
        "power_data": [("2025-01-01T12:00:00", 1.0)], # minimal data
        "profile_name": None # Initially None
    }
    
    await manager._async_process_cycle_end(cycle_data)
    
    # Verify async_match_profile was called
    manager.profile_store.async_match_profile.assert_called_once()
    
    # Verify cycle_data was updated BEFORE add_cycle
    args = manager.profile_store.async_add_cycle.call_args[0]
    added_cycle = args[0]
    assert added_cycle["profile_name"] == "DerivedProfile"

