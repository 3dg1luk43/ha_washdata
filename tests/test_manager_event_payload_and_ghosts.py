"""Focused tests for cycle-end event payload and ghost-cycle detection."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant, callback

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_EVENTS,
    EVENT_CYCLE_ENDED,
    NOTIFY_EVENT_FINISH,
)
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH],
    }
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)

    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        return mgr


@pytest.mark.asyncio
async def test_cycle_end_event_payload_excludes_large_fields(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Cycle ended event should strip heavy fields to stay within HA event limits."""
    manager._notify_fire_events = True
    manager._auto_label_confidence = 0.0
    manager.profile_store.get_profiles = MagicMock(return_value={})
    manager.profile_store.async_add_cycle = AsyncMock()
    manager.profile_store.async_clear_active_cycle = AsyncMock()
    manager.profile_store.async_rebuild_envelope = AsyncMock()
    manager._run_post_cycle_processing = AsyncMock()

    cycle_data = {
        "id": "cycle-1",
        "start_time": "2026-01-01T10:00:00+00:00",
        "duration": 1200,
        "status": "completed",
        "power_data": [[0.0, 50.0], [60.0, 200.0]],
        "debug_data": {"large": "blob"},
        "power_trace": [1, 2, 3],
    }

    fired_events: list[dict[str, Any]] = []

    @callback
    def _handle_cycle_ended(event: Any) -> None:
        fired_events.append(event.data)

    hass.bus.async_listen(EVENT_CYCLE_ENDED, _handle_cycle_ended)

    await manager._async_process_cycle_end(dict(cycle_data))
    await hass.async_block_till_done()

    assert fired_events
    event_payload = fired_events[-1]

    event_cycle_data = event_payload["cycle_data"]
    assert "power_data" not in event_cycle_data
    assert "debug_data" not in event_cycle_data
    assert "power_trace" not in event_cycle_data
    assert event_cycle_data["duration"] == 1200
    assert event_cycle_data["device_type"] == manager.device_type


@pytest.mark.asyncio
async def test_short_low_energy_cycle_is_marked_noise(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Short + low-energy cycles should trigger ghost/noise handling."""
    manager._handle_noise_cycle = MagicMock()
    manager._async_process_cycle_end = AsyncMock()

    cycle_data = {
        "duration": 30,
        "max_power": 25,
        "power_data": [[0.0, 5.0], [10.0, 5.0], [20.0, 5.0]],
    }

    manager._on_cycle_end(cycle_data)
    await hass.async_block_till_done()

    manager._handle_noise_cycle.assert_called_once_with(25)


@pytest.mark.asyncio
async def test_short_high_energy_cycle_is_not_marked_noise(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Short cycles with meaningful energy should not be treated as ghost cycles."""
    manager._handle_noise_cycle = MagicMock()
    manager._async_process_cycle_end = AsyncMock()

    cycle_data = {
        "duration": 30,
        "max_power": 3000,
        "power_data": [[0.0, 2000.0], [10.0, 2000.0], [20.0, 2000.0]],
    }

    manager._on_cycle_end(cycle_data)
    await hass.async_block_till_done()

    manager._handle_noise_cycle.assert_not_called()
