import pytest
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, AsyncMock
from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.const import (
    DOMAIN, STATE_RUNNING, STATE_OFF, 
    EVENT_CYCLE_STARTED, EVENT_CYCLE_ENDED,
    CONF_NOTIFY_FIRE_EVENTS
)
from custom_components.ha_washdata.manager import WashDataManager
from tests.utils.synthesizer import CycleSynthesizer

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_integration_entry"
    entry.title = "Integration Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        "min_power": 5.0,
        "off_delay": 30,
        "min_off_gap": 30, # Small value for fast test
        "end_energy_threshold": 0.05,
        "device_type": "washing_machine",
        "notify_events": []
    }
    entry.data = {}
    return entry

@pytest.mark.asyncio
async def test_end_to_end_integration(hass: HomeAssistant, mock_entry):
    """Test full integration flow using synthesized data."""
    # logging.getLogger("custom_components.ha_washdata").setLevel(logging.DEBUG)
    
    # 1. Setup Synthesizer for a quick cycle
    synth = CycleSynthesizer()
    synth.add_phase(100.0, 30.0)  # Start/Starting phase
    synth.add_phase(500.0, 60.0)  # Running phase
    synth.add_gap(300.0)          # End phase (should trigger Paused -> Ending -> Off)
    
    readings = synth.generate(sample_interval=10.0)
    
    # 2. Setup Manager with mocked dependencies
    with patch("custom_components.ha_washdata.manager.ProfileStore") as MockStore:
        manager = WashDataManager(hass, mock_entry)
        store = MockStore.return_value
        store.get_profiles.return_value = {}
        store.get_active_cycle.return_value = None
        store.get_past_cycles.return_value = []
        store.get_last_active_save.return_value = None
        store.async_load = AsyncMock()
        store.async_save = AsyncMock()
        store.async_add_cycle = AsyncMock()
        store.async_clear_active_cycle = AsyncMock()
        store.async_repair_profile_samples = AsyncMock(return_value={})
        store.async_migrate_cycles_to_compressed = AsyncMock()
        store.async_run_maintenance = AsyncMock(return_value={})
        store.get_suggestions.return_value = {}
        
        # Mock MatchResult to avoid complex DTW in this integration test
        mock_result = MagicMock()
        mock_result.confidence = 0.0
        mock_result.best_profile = None
        mock_result.is_ambiguous = False
        mock_result.expected_duration = 0.0
        mock_result.matched_phase = None
        mock_result.candidates = []
        mock_result.is_confident_mismatch = False
        store.async_match_profile = AsyncMock(return_value=mock_result)
        
        await manager.async_setup()
        
        # 3. Listen for events
        events_started = []
        events_ended = []
        
        from homeassistant.core import callback
        
        @callback
        def handle_started(event):
            events_started.append(event)
            
        @callback
        def handle_ended(event):
            events_ended.append(event)
            
        hass.bus.async_listen(EVENT_CYCLE_STARTED, handle_started)
        hass.bus.async_listen(EVENT_CYCLE_ENDED, handle_ended)
        
        # 4. Replay synthesized readings
        for ts, power in readings:
            with patch("homeassistant.util.dt.now", return_value=ts):
                # Simulate HA state change
                event = MagicMock()
                event.data = {"new_state": MagicMock(state=str(power))}
                manager._async_power_changed(event)

        # 5. Finalize and verify
        await hass.async_block_till_done()
        # Give post-cycle tasks a moment
        await asyncio.sleep(0.1)
        await hass.async_block_till_done()
        
        assert len(events_started) >= 1, "EVENT_CYCLE_STARTED never fired"
        assert len(events_ended) >= 1, "EVENT_CYCLE_ENDED never fired"
        assert manager.detector.state in (STATE_OFF, "finished", "interrupted", "force_stopped"), f"Unexpected final state: {manager.detector.state}"
        
        # Verify store interactions
        assert store.async_add_cycle.called
        assert store.async_clear_active_cycle.called


@pytest.mark.asyncio
async def test_notify_fire_events_disabled_suppresses_both_events(hass: HomeAssistant):
    """Test that notify_fire_events=False suppresses both start and end events end-to-end."""
    # Create entry with notify_fire_events=False
    entry = MagicMock()
    entry.entry_id = "test_no_events"
    entry.title = "No Events Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        "min_power": 5.0,
        "off_delay": 30,
        "min_off_gap": 30,
        "end_energy_threshold": 0.05,
        "device_type": "washing_machine",
        "notify_events": [],
        CONF_NOTIFY_FIRE_EVENTS: False,  # Key option: disable events
    }
    entry.data = {}
    
    # Setup Synthesizer for a complete cycle
    synth = CycleSynthesizer()
    synth.add_phase(100.0, 30.0)  # Starting phase
    synth.add_phase(500.0, 60.0)  # Running phase
    synth.add_gap(300.0)          # End phase
    readings = synth.generate(sample_interval=10.0)
    
    # Setup Manager with mocked dependencies
    with patch("custom_components.ha_washdata.manager.ProfileStore") as MockStore:
        manager = WashDataManager(hass, entry)
        store = MockStore.return_value
        store.get_profiles.return_value = {}
        store.get_active_cycle.return_value = None
        store.get_past_cycles.return_value = []
        store.get_last_active_save.return_value = None
        store.async_load = AsyncMock()
        store.async_save = AsyncMock()
        store.async_add_cycle = AsyncMock()
        store.async_clear_active_cycle = AsyncMock()
        store.async_repair_profile_samples = AsyncMock(return_value={})
        store.async_migrate_cycles_to_compressed = AsyncMock()
        store.async_run_maintenance = AsyncMock(return_value={})
        store.get_suggestions.return_value = {}
        
        mock_result = MagicMock()
        mock_result.confidence = 0.0
        mock_result.best_profile = None
        mock_result.is_ambiguous = False
        mock_result.expected_duration = 0.0
        mock_result.matched_phase = None
        mock_result.candidates = []
        mock_result.is_confident_mismatch = False
        store.async_match_profile = AsyncMock(return_value=mock_result)
        
        await manager.async_setup()
        
        # Listen for events (should NOT fire)
        events_started = []
        events_ended = []
        
        @callback
        def handle_started(event):
            events_started.append(event)
            
        @callback
        def handle_ended(event):
            events_ended.append(event)
            
        hass.bus.async_listen(EVENT_CYCLE_STARTED, handle_started)
        hass.bus.async_listen(EVENT_CYCLE_ENDED, handle_ended)
        
        # Replay synthesized readings
        for ts, power in readings:
            with patch("homeassistant.util.dt.now", return_value=ts):
                event = MagicMock()
                event.data = {"new_state": MagicMock(state=str(power))}
                manager._async_power_changed(event)

        # Finalize
        await hass.async_block_till_done()
        await asyncio.sleep(0.1)
        await hass.async_block_till_done()
        
        # CRITICAL ASSERTIONS: Events should NOT have been fired
        assert len(events_started) == 0, f"EVENT_CYCLE_STARTED should NOT fire when notify_fire_events=False, but {len(events_started)} events fired"
        assert len(events_ended) == 0, f"EVENT_CYCLE_ENDED should NOT fire when notify_fire_events=False, but {len(events_ended)} events fired"
        
        # Verify cycle still completed internally (store interactions should still occur)
        assert store.async_add_cycle.called, "Cycle should still be recorded even when events are disabled"
        assert store.async_clear_active_cycle.called


@pytest.mark.asyncio
async def test_notify_fire_events_enabled_fires_both_events(hass: HomeAssistant):
    """Test that notify_fire_events=True (default) fires both start and end events end-to-end."""
    # Create entry with notify_fire_events=True (explicit)
    entry = MagicMock()
    entry.entry_id = "test_with_events"
    entry.title = "With Events Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        "min_power": 5.0,
        "off_delay": 30,
        "min_off_gap": 30,
        "end_energy_threshold": 0.05,
        "device_type": "washing_machine",
        "notify_events": [],
        CONF_NOTIFY_FIRE_EVENTS: True,  # Key option: enable events
    }
    entry.data = {}
    
    # Setup Synthesizer for a complete cycle
    synth = CycleSynthesizer()
    synth.add_phase(100.0, 30.0)  # Starting phase
    synth.add_phase(500.0, 60.0)  # Running phase
    synth.add_gap(300.0)          # End phase
    readings = synth.generate(sample_interval=10.0)
    
    # Setup Manager with mocked dependencies
    with patch("custom_components.ha_washdata.manager.ProfileStore") as MockStore:
        manager = WashDataManager(hass, entry)
        store = MockStore.return_value
        store.get_profiles.return_value = {}
        store.get_active_cycle.return_value = None
        store.get_past_cycles.return_value = []
        store.get_last_active_save.return_value = None
        store.async_load = AsyncMock()
        store.async_save = AsyncMock()
        store.async_add_cycle = AsyncMock()
        store.async_clear_active_cycle = AsyncMock()
        store.async_repair_profile_samples = AsyncMock(return_value={})
        store.async_migrate_cycles_to_compressed = AsyncMock()
        store.async_run_maintenance = AsyncMock(return_value={})
        store.get_suggestions.return_value = {}
        
        mock_result = MagicMock()
        mock_result.confidence = 0.0
        mock_result.best_profile = None
        mock_result.is_ambiguous = False
        mock_result.expected_duration = 0.0
        mock_result.matched_phase = None
        mock_result.candidates = []
        mock_result.is_confident_mismatch = False
        store.async_match_profile = AsyncMock(return_value=mock_result)
        
        await manager.async_setup()
        
        # Listen for events (SHOULD fire)
        events_started = []
        events_ended = []
        
        @callback
        def handle_started(event):
            events_started.append(event)
            
        @callback
        def handle_ended(event):
            events_ended.append(event)
            
        hass.bus.async_listen(EVENT_CYCLE_STARTED, handle_started)
        hass.bus.async_listen(EVENT_CYCLE_ENDED, handle_ended)
        
        # Replay synthesized readings
        for ts, power in readings:
            with patch("homeassistant.util.dt.now", return_value=ts):
                event = MagicMock()
                event.data = {"new_state": MagicMock(state=str(power))}
                manager._async_power_changed(event)

        # Finalize
        await hass.async_block_till_done()
        await asyncio.sleep(0.1)
        await hass.async_block_till_done()
        
        # CRITICAL ASSERTIONS: Events SHOULD have been fired
        assert len(events_started) >= 1, "EVENT_CYCLE_STARTED should fire when notify_fire_events=True"
        assert len(events_ended) >= 1, "EVENT_CYCLE_ENDED should fire when notify_fire_events=True"
        
        # Verify cycle completed and was recorded
        assert store.async_add_cycle.called
        assert store.async_clear_active_cycle.called

