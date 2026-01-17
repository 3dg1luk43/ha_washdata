
"""Integration tests for Recorder feature."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from homeassistant.data_entry_flow import FlowResultType

from custom_components.ha_washdata.const import DOMAIN

@pytest.mark.asyncio
async def test_record_flow_menu(mock_hass, mock_config_entry):
    """Test Record Mode menu logic."""
    # Setup manager mock
    manager = MagicMock()
    manager.recorder = MagicMock()
    manager.recorder.is_recording = False
    manager.recorder.last_run = None
    manager.async_start_recording = AsyncMock()
    
    mock_hass.data[DOMAIN] = {mock_config_entry.entry_id: manager}
    
    # Init options flow
    # We need to test OptionsFlowHandler directly or via hass.config_entries.options.async_init
    # Testing handler directly is easier if we can instantiate it.
    
    from custom_components.ha_washdata.config_flow import OptionsFlowHandler
    flow = OptionsFlowHandler(mock_config_entry)
    flow.hass = mock_hass
    
    # Step: record_cycle (Initial menu check)
    result = await flow.async_step_record_cycle()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "record_cycle"
    
    # Should see "Start New Recording" in options
    # schema = result["data_schema"]
    # schema.schema is {Required('action'): In(...)}
    # We can inspect schema structure if needed, or just assert we got a form.
    # But let's verify options if we can easily inspect vol.In
    
    # Simulate User clicks "Start Recording"
    result = await flow.async_step_record_cycle(user_input={"action": "start_recording"})
    
    # Should call start_recording and loop back to menu
    assert manager.async_start_recording.called
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "record_cycle"

@pytest.mark.asyncio
async def test_record_flow_stop_process(mock_hass, mock_config_entry):
    """Test stopping and processing flow."""
    manager = MagicMock()
    manager.recorder = MagicMock()
    # State: Recording Active
    manager.recorder.is_recording = True
    manager.recorder.last_run = {
        "start_time": "2023-01-01T12:00:00+00:00",
        "end_time": "2023-01-01T12:01:00+00:00",
        "data": [("2023-01-01T12:00:00+00:00", 10.0)]
    }
    manager.async_stop_recording = AsyncMock()
    
    # Setup mock methods
    manager.recorder.get_trim_suggestions.return_value = (5.0, 5.0, 1.0)
    manager.recorder.clear_last_run = AsyncMock()
    manager.profile_store.get_profiles.return_value = {"Existing": {}}
    manager.profile_store.create_profile_standalone = AsyncMock()
    manager.profile_store.async_add_cycle = AsyncMock()
    manager.profile_store.async_rebuild_envelope = AsyncMock()
    manager.profile_store.async_save = AsyncMock()
    manager.profile_store.generate_preview_svg = MagicMock(return_value="<svg></svg>")
    
    mock_hass.data[DOMAIN] = {mock_config_entry.entry_id: manager}
    
    from custom_components.ha_washdata.config_flow import OptionsFlowHandler
    flow = OptionsFlowHandler(mock_config_entry)
    flow.hass = mock_hass
    
    # 1. Stop Recording
    result = await flow.async_step_record_cycle(user_input={"action": "stop_recording"})
    assert manager.async_stop_recording.called
    
    # Should transition directly to process step
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "record_process"
    
    # Verify manager state simulation (stop should have produced last_run)
    # Since we mocked async_stop_recording, it returns None.
    # But current logic calls last_run.
    # We need to ensure last_run is available.
    manager.recorder.last_run = {
        "start_time": "2023-01-01T12:00:00+00:00",
        "end_time": "2023-01-01T12:01:00+00:00",
        "data": [("2023-01-01T12:00:00+00:00", 10.0)]
    }
    
    # 2. Submit Processing Form (Trim & Save)
    user_input = {
        "head_trim": 10.0,
        "tail_trim": 5.0,
        "save_mode": "new_profile",
        "profile_name": "New Profile Name"
    }
    
    result = await flow.async_step_record_process(user_input=user_input)
    
    assert result["type"] == FlowResultType.CREATE_ENTRY
    
    # Verify calls
    manager.profile_store.create_profile_standalone.assert_called_with("New Profile Name")
    assert manager.profile_store.async_add_cycle.called
    manager.profile_store.async_rebuild_envelope.assert_called_with("New Profile Name")
    assert manager.recorder.clear_last_run.called

