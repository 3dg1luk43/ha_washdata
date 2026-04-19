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


@pytest.mark.asyncio
async def test_dishwasher_pump_out_is_suppressed(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Dishwasher pump-out (short, low-energy cycle shortly after main cycle) is suppressed."""
    from datetime import datetime, timezone, timedelta

    manager._handle_noise_cycle = MagicMock()
    manager._async_process_cycle_end = AsyncMock()
    manager.device_type = "dishwasher"

    # Simulate a real cycle that ended 4 minutes ago
    prev_end = datetime(2026, 3, 18, 13, 59, 54, tzinfo=timezone.utc)
    manager._last_cycle_end_time = prev_end

    # Pump-out: starts 4 min 7s after main cycle, lasts 105 s, energy ~0.47 Wh
    pump_start = prev_end + timedelta(seconds=247)
    cycle_data = {
        "start_time": pump_start.isoformat(),
        "duration": 105,
        "max_power": 16.1,
        "status": "interrupted",
        # 16 W for 105 s → ~0.47 Wh
        "power_data": [
            [0.0, 6.6],
            [20.0, 16.1],
            [40.0, 15.5],
            [100.0, 15.5],
            [105.0, 0.5],
        ],
    }

    manager._on_cycle_end(cycle_data)
    await hass.async_block_till_done()

    manager._handle_noise_cycle.assert_called_once_with(16.1)
    manager._async_process_cycle_end.assert_not_called()


@pytest.mark.asyncio
async def test_dishwasher_pump_out_outside_window_is_stored(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """A dishwasher cycle that starts more than 10 min after the previous is stored normally."""
    from datetime import datetime, timezone, timedelta

    manager._handle_noise_cycle = MagicMock()
    manager._async_process_cycle_end = AsyncMock()
    manager.device_type = "dishwasher"

    prev_end = datetime(2026, 3, 18, 13, 59, 54, tzinfo=timezone.utc)
    manager._last_cycle_end_time = prev_end

    # Starts 15 minutes after previous - outside the 10-minute suppression window
    cycle_start = prev_end + timedelta(seconds=900)
    cycle_data = {
        "start_time": cycle_start.isoformat(),
        "duration": 105,
        "max_power": 16.1,
        "status": "interrupted",
        "power_data": [
            [0.0, 6.6],
            [20.0, 16.1],
            [100.0, 15.5],
            [105.0, 0.5],
        ],
    }

    manager._on_cycle_end(cycle_data)
    await hass.async_block_till_done()

    manager._handle_noise_cycle.assert_not_called()
    manager._async_process_cycle_end.assert_called_once()


@pytest.mark.asyncio
async def test_dishwasher_pump_out_high_energy_is_stored(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """A dishwasher cycle within the window but with high energy is stored as a real cycle."""
    from datetime import datetime, timezone, timedelta

    manager._handle_noise_cycle = MagicMock()
    manager._async_process_cycle_end = AsyncMock()
    manager.device_type = "dishwasher"

    prev_end = datetime(2026, 3, 18, 13, 59, 54, tzinfo=timezone.utc)
    manager._last_cycle_end_time = prev_end

    # Starts 3 minutes after previous (within window) but uses 2 Wh - a real load
    cycle_start = prev_end + timedelta(seconds=180)
    cycle_data = {
        "start_time": cycle_start.isoformat(),
        "duration": 120,
        "max_power": 500.0,
        "status": "interrupted",
        # ~500 W for 120 s → ~16.7 Wh → well above 1 Wh threshold
        "power_data": [[0.0, 500.0], [60.0, 500.0], [120.0, 0.0]],
    }

    manager._on_cycle_end(cycle_data)
    await hass.async_block_till_done()

    manager._handle_noise_cycle.assert_not_called()
    manager._async_process_cycle_end.assert_called_once()


# ---------------------------------------------------------------------------
# Fix A: unconditional low_power_floor for unmatched dishwasher cycles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dishwasher_unmatched_survives_7200s_silence(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Fix A: an unmatched dishwasher cycle must NOT be force-ended at 7200s silence.

    Before Fix A the 14400s device floor was only applied when a profile was matched.
    An unmatched cycle (program == 'detecting...') used only the 3600s base timeout,
    so 7200s of silence triggered a force-end during the passive drying phase.
    After Fix A the floor is applied unconditionally: max(14400, 3600) = 14400s.
    """
    from datetime import datetime, timezone, timedelta

    now = datetime(2026, 3, 18, 14, 0, 0, tzinfo=timezone.utc)

    manager.device_type = "dishwasher"
    manager._current_program = "detecting..."  # Unmatched
    manager._low_power_no_update_timeout = 3600.0
    # Set no_update_active_timeout high so the keepalive injection at step 3a
    # does not fire before the staleness check we are testing.
    manager._no_update_active_timeout = 14400.0

    manager.detector.state = "ending"
    manager.detector.is_waiting_low_power.return_value = True
    manager.detector.current_cycle_start = None
    manager.detector.get_elapsed_seconds.return_value = 7200
    manager.detector.expected_duration_seconds = 0
    manager.detector._verified_pause = False

    # Silence for 7200s (above 3600s default, below 14400s floor)
    manager._last_reading_time = now - timedelta(seconds=7200)
    manager._last_real_reading_time = now - timedelta(seconds=7200)
    manager._last_cycle_end_time = None  # Not within suspicious window

    await manager._watchdog_check_stuck_cycle(now)

    manager.detector.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_dishwasher_unmatched_force_ended_after_14400s_silence(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Fix A: an unmatched dishwasher cycle IS force-ended after 14400s silence."""
    from datetime import datetime, timezone, timedelta

    now = datetime(2026, 3, 18, 14, 0, 0, tzinfo=timezone.utc)

    manager.device_type = "dishwasher"
    manager._current_program = "detecting..."
    manager._low_power_no_update_timeout = 3600.0
    manager._no_update_active_timeout = 14400.0

    manager.detector.state = "ending"
    manager.detector.is_waiting_low_power.return_value = True
    manager.detector.current_cycle_start = None
    manager.detector.get_elapsed_seconds.return_value = 15000
    manager.detector.expected_duration_seconds = 0
    manager.detector._verified_pause = False

    # Silence for 15000s (above 14400s floor)
    manager._last_reading_time = now - timedelta(seconds=15000)
    manager._last_real_reading_time = now - timedelta(seconds=15000)
    manager._last_cycle_end_time = None

    await manager._watchdog_check_stuck_cycle(now)

    manager.detector.force_end.assert_called_once()
