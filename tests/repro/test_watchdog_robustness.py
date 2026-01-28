
import asyncio
import logging
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import STATE_RUNNING, STATE_OFF

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    # Execute tasks immediately for testing simplicity
    def _create_task(coro):
        return asyncio.create_task(coro)
    hass.async_create_task = _create_task
    async def _executor(target, *args):
        return target(*args)
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    hass.states.get = MagicMock(return_value=None)
    return hass

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_wd"
    entry.options = {
        "device_type": "dishwasher",
        "min_power": 5.0,
        "off_delay": 120, # 2 mins
        "no_update_active_timeout": 300, # 5 mins
        "low_power_no_update_timeout": 3600, # 60 mins
        "smoothing_window": 1,
        "completion_min_seconds": 60,
    }
    return entry

@pytest.mark.asyncio
async def test_watchdog_low_power_survival(mock_hass, mock_entry):
    """Test that watchdog keeps cycle alive during low-power silence."""
    
    # 1. Setup Manager with Real Dependencies (mocked IO)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        manager = WashDataManager(mock_hass, mock_entry)
        # Mock detector internal state to be 'waiting low power'
        # We can force this by setting detector state and time_below_threshold
        
        manager.detector._state = STATE_RUNNING
        manager.detector._time_below_threshold = 1.0 # Simulate low power start
        
        # Initial State: High Power
        import datetime as dt_module
        start_time = dt_module.datetime.now(dt_module.timezone.utc)
        
        with patch("homeassistant.util.dt.now", return_value=start_time):
             manager.detector.process_reading(100.0, start_time)
             manager._last_real_reading_time = start_time
             manager._last_reading_time = start_time
             
    # 2. Simulate dropping to Low Power (0W)
    t_low = start_time + timedelta(seconds=10)
    with patch("homeassistant.util.dt.now", return_value=t_low):
        manager.detector.process_reading(0.0, t_low)
        manager._last_real_reading_time = t_low
        manager._last_reading_time = t_low
        
    assert manager.detector.is_waiting_low_power()
    
    # 3. Fast forward 10 minutes (Silence)
    # This exceeds off_delay (2m) but is less than low_power_timeout (60m)
    t_check = t_low + timedelta(minutes=10)
    
    with patch("homeassistant.util.dt.now", return_value=t_check):
        await manager._watchdog_check_stuck_cycle(t_check)
        
    # Assertions
    # Should NOT have ended
    assert manager.detector.state == STATE_RUNNING, "Watchdog killed waiting cycle too early!"
    
    # Check if we injected injection (logic: process_reading called with 0W)
    # We can check if _last_reading_time was updated to t_check
    assert manager._last_reading_time == t_check, "Watchdog did not update last_reading_time (injection failed?)"
    # But _last_real_reading_time should remain old
    assert manager._last_real_reading_time == t_low, "Watchdog incorrectly updated real reading time!"


@pytest.mark.asyncio
async def test_watchdog_low_power_termination(mock_hass, mock_entry):
    """Test that watchdog kills cycle after low_power_timeout."""
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        manager = WashDataManager(mock_hass, mock_entry)
        manager.detector._state = STATE_RUNNING
        import datetime as dt_module
        start_time = dt_module.datetime.now(dt_module.timezone.utc)
        manager.detector._time_below_threshold = 1.0 # Force low power state tracking
        
        # Last real reading was 61 minutes ago
        old_time = start_time - timedelta(minutes=61)
        manager._last_real_reading_time = old_time
        manager._last_reading_time = old_time # Or even recent synthetic updates
        
    # Check
    with patch("homeassistant.util.dt.now", return_value=start_time):
        await manager._watchdog_check_stuck_cycle(start_time)
        
    assert manager.detector.state == STATE_OFF, "Watchdog failed to kill stale cycle!"    
