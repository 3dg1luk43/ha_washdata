import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import timedelta, datetime, timezone
import sys
import os

# Ensure the custom_components directory is in the path
sys.path.append(os.path.abspath("/root/ha_washdata/custom_components"))

from ha_washdata.manager import WashDataManager
from ha_washdata.const import (
    CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS, CONF_POWER_SENSOR, 
    CONF_OFF_DELAY, STATE_RUNNING, STATE_OFF, CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_DEVICE_TYPE, CONF_NOTIFY_EVENTS
)
from homeassistant.util import dt as dt_util

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    # Prevent 'coroutine was never awaited' warnings
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.components.persistent_notification.async_create = MagicMock()
    hass.config_entries.async_get_entry = MagicMock()
    return hass

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Dishwasher"
    entry.options = {
        CONF_MIN_POWER: 5.0,
        CONF_OFF_DELAY: 60,
        CONF_COMPLETION_MIN_SECONDS: 600,
        CONF_NO_UPDATE_ACTIVE_TIMEOUT: 1800, # 30 minutes
        CONF_DEVICE_TYPE: "dishwasher",
        "power_sensor": "sensor.test_power",
        CONF_NOTIFY_EVENTS: [],
        "start_duration_threshold": 0,
        "start_energy_threshold": 0,
    }
    return entry

@pytest.mark.asyncio
async def test_repro_long_drying_pause_split(mock_hass, mock_entry):
    """
    Test that a cycle with 1h wash and 2h drying (0W) splits if expected_duration is too short
    or if watchdog kill logic is too aggressive.
    """
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    
    with patch("custom_components.ha_washdata.manager.ProfileStore") as mock_ps_cls:
        mock_ps = mock_ps_cls.return_value
        mock_ps.get_suggestions.return_value = {}
        mock_ps.get_duration_ratio_limits.return_value = (0.5, 1.5)
        mock_ps.async_match_profile = AsyncMock()
        # Mock successful alignment verification
        mock_ps.async_verify_alignment = AsyncMock(return_value=(True, 3600.0, 0.0))
        mock_ps.get_profile.return_value = {"avg_duration": 10800}
        
        manager = WashDataManager(mock_hass, mock_entry)
        
        # Real detector
        detector = manager.detector
        print(f"DEBUG: state={detector.state}, duration_thresh={detector.config.start_duration_threshold}, energy_thresh={detector.config.start_energy_threshold}")
        
        # Start at T=0
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # 1. Start cycle (Wash Phase - 1 hour)
        with patch("homeassistant.util.dt.now", return_value=now):
            # Reading 1: 500W
            print(f"DEBUG: T=0, power=500")
            manager._async_power_changed(MagicMock(data={"new_state": MagicMock(state="500.0")}))
            print(f"DEBUG: After Reading 1, state={detector.state}")
            
        # Advance 60s to reach RUNNING
        now += timedelta(seconds=60)
        with patch("homeassistant.util.dt.now", return_value=now):
            print(f"DEBUG: T=60, power=500")
            manager._async_power_changed(MagicMock(data={"new_state": MagicMock(state="500.0")}))
            print(f"DEBUG: After Reading 2, state={detector.state}")
            assert detector.state == STATE_RUNNING
            
        # 2. Advance to T=1h, power drops to 0W (Drying Phase starts)
        t_drying_start = now + timedelta(hours=1)
        with patch("homeassistant.util.dt.now", return_value=t_drying_start):
            manager._async_power_changed(MagicMock(data={"new_state": MagicMock(state="0.0")}))
            # Should be in RUNNING but waiting for off_delay
            assert detector.state == STATE_RUNNING
            
        # 3. Force Profile Match (Profile says 3 hours total)
        manager._current_program = "Long Dishwasher Program"
        detector._matched_profile = "Long Dishwasher Program"
        detector._expected_duration = 10800 # 3 hours
        manager._matched_profile_duration = 10800
        # Manually set verified_pause to True (simulating async_verify_alignment success)
        detector.set_verified_pause(True)
        
        # 4. Advance 45 minutes into drying (T=1h45m)
        # Silence = 45 mins (2700s).
        t_silence = t_drying_start + timedelta(minutes=45)
        with patch("homeassistant.util.dt.now", return_value=t_silence):
            await manager._watchdog_check_stuck_cycle(t_silence)
            assert detector.state == STATE_RUNNING, "Watchdog killed cycle too early (45m silence)"
            
        # 5. Advance 2 hours 10 minutes into drying (T=3h10m)
        # Total duration = 3h10m (11400s). Expected = 3h (10800s).
        # Silence = 2h10m (7800s).
        
        t_end_drying = t_drying_start + timedelta(hours=2, minutes=10)
        with patch("homeassistant.util.dt.now", return_value=t_end_drying):
            # This is where we expect failure in the current version
            await manager._watchdog_check_stuck_cycle(t_end_drying)
            
            # If the bug exists, state will be OFF
            assert detector.state == STATE_RUNNING, f"Cycle was killed by watchdog after {7800}s silence even though it's a verified pause"