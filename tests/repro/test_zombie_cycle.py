"""Reproduction test for zombie cycle and stuck power entity."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import timedelta, datetime, timezone
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS, CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_POWER_SENSOR, CONF_OFF_DELAY, STATE_RUNNING, STATE_OFF, 
    CONF_NO_UPDATE_ACTIVE_TIMEOUT
)
from homeassistant.util import dt as dt_util

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
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
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 5.0,
        CONF_OFF_DELAY: 60, # Short off delay for testing
        CONF_COMPLETION_MIN_SECONDS: 600,
        CONF_NO_UPDATE_ACTIVE_TIMEOUT: 600, # 10 minutes default
        "power_sensor": "sensor.test_power",
    }
    entry.data = {}
    return entry

@pytest.fixture
def manager(mock_hass, mock_entry):
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    
    # Mock ProfileStore and CycleDetector
    with patch("custom_components.ha_washdata.manager.ProfileStore") as mock_ps_cls, \
         patch("custom_components.ha_washdata.manager.CycleDetector") as mock_cd_cls:
        
        mock_ps = mock_ps_cls.return_value
        mock_ps.get_suggestions.return_value = {}
        mock_ps.get_duration_ratio_limits.return_value = (0.1, 1.3)
        mock_ps.async_match_profile = AsyncMock()
        
        mock_cd = mock_cd_cls.return_value
        # Default state
        mock_cd.state = STATE_OFF
        mock_cd.config = MagicMock()
        mock_cd.config.min_power = 5.0
        mock_cd.config.off_delay = 60
        
        mgr = WashDataManager(mock_hass, mock_entry)
        
        # Manually wire up the detector state property to a local variable we can change
        # to simulate state changes
        mgr.detector = mock_cd
        
        return mgr

@pytest.mark.asyncio
async def test_repro_stuck_power_after_watchdog_kill(manager):
    """
    Reproduction: When watchdog kills a cycle due to silence (e.g. smart plug dies),
    manager.current_power remains at the last high value.
    """
    # Setup: Cycle running, high power
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Patch dt_util.now to return our 'now'
    with patch("homeassistant.util.dt.now", return_value=now):
        manager.detector.state = STATE_RUNNING
        manager.detector.current_cycle_start = now
        manager.detector.is_waiting_low_power.return_value = False # Force High Power branch
        manager._current_program = "detecting..."
        
        # Simulate a reading
        manager._async_power_changed(MagicMock(data={"new_state": MagicMock(state="500.0")}))
    
    # Verify current power is 500
    assert manager.current_power == 500.0
    assert manager._last_reading_time == now
    
    # Fast forward time beyond NO_UPDATE_ACTIVE_TIMEOUT (600s) AND the high-power keepalive limit (4h)
    # We go to 5 hours (18000s)
    future = now + timedelta(hours=5)
    
    # Mock elapsed seconds to exceed the high power limit (18000s)
    manager.detector.get_elapsed_seconds.return_value = 18000
    manager.detector.expected_duration_seconds = 0
    
    # Run watchdog
    await manager._watchdog_check_stuck_cycle(future)
    
    # Assert force_end was called
    manager.detector.force_end.assert_called_once()
    
    # ISSUE: current_power should be reset to 0.0 so the sensor entity reports 0
    # With fix, it should be 0.0
    assert manager.current_power == 0.0 

@pytest.mark.asyncio
async def test_repro_premature_kill_during_expected_pause(manager):
    """
    Reproduction: Watchdog kills cycle during a legitimate long pause that matches
    the profile (e.g. soak cycle), because it ignores profile look-ahead.
    """
    # Setup: Cycle running, matched to "Long Soak"
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    manager.detector.state = STATE_RUNNING
    manager._current_program = "Long Soak"
    manager._matched_profile_duration = 7200 # 2 hours
    
    # Setup ProfileStore to return a profile with a long pause
    # (Mocking async_match_profile isn't enough, we need the logic in watchdog to check it)
    # The fix will likely involve checking 'expected_duration' or similar.
    # Currently watchdog just checks `time_since_any_update > timeout`.
    
    # Simulate last reading 11 minutes ago
    last_reading = now - timedelta(minutes=11)
    manager._last_reading_time = last_reading
    manager._last_real_reading_time = last_reading
    manager._current_power = 200.0 # Was running high power
    
    # Watchdog check at 'now'
    # Timeout is 600s (10m). Silence is 11m.
    # Current logic: High power (200 > 5) -> "High Power Handling"
    # "if time_since_any_update > self._no_update_active_timeout:" (True)
    # It checks expected_duration:
    # "expected = getattr(self.detector, 'expected_duration_seconds', 0)"
    
    # Mock detector attributes used by current logic
    manager.detector.get_elapsed_seconds.return_value = 1000 # 16 mins elapsed
    manager.detector.expected_duration_seconds = 7200 
    
    # CURRENT LOGIC in manager.py:
    # if self._current_power >= min_power: (200 >= 5) True
    #    limit = expected + 7200 (7200+7200 = 14400)
    #    if elapsed < limit: (1000 < 14400) True
    #        Inject Refresh.
    
    # Wait, the current logic SEEMS to handle "High Power" silence by injecting refresh?
    # Let's re-read manager.py _watchdog_check_stuck_cycle.
    
    """
    if time_since_any_update > self._no_update_active_timeout:
        
        # Check if high power (running)
        if self._current_power >= self.detector.config.min_power:
            # Allow extended silence if within reasonable cycle bounds
            expected = getattr(self.detector, "expected_duration_seconds", 0)
            elapsed = self.detector.get_elapsed_seconds()
            limit = (expected + 7200) if expected > 0 else 14400 # 4h default
            
            if elapsed < limit:
                # ... Inject refresh ...
                return
    """
    
    # So if power is High, it injects refresh.
    # BUT, what if power dropped to 0 (Pause) but the 0W update was LOST?
    # Then _current_power is still 200.0. So it injects refresh (200W).
    # So the cycle continues "Running" at 200W (fake).
    # This is also a zombie! It's reporting 200W when it should be 0W (Pause).
    # If the profile says "At minute 15, we expect 0W", and we are at minute 16 reporting 200W (injected),
    # that's wrong.
    
    # However, the "Premature Kill" usually refers to:
    # We are in a "Low Power" state (e.g. 0W received), waiting for next phase.
    # Power is 0W. `is_waiting_low_power()` is True.
    # `effective_low_power_timeout` checks.
    # If `time_since_real_update > effective_low_power_timeout` (default 3600s/1h).
    # If we have a soak that is 2 hours.
    # The watchdog will kill it after 1 hour.
    
    # Let's test THAT scenario.
    
    manager._current_power = 0.0
    manager.detector.is_waiting_low_power.return_value = True
    manager._low_power_no_update_timeout = 3600 # 1 hour
    
    # Simulate silence of 61 minutes
    future = now
    manager._last_reading_time = now - timedelta(minutes=61)
    manager._last_real_reading_time = now - timedelta(minutes=61)
    
    # Mock expected duration to be 2 hours (7200s)
    manager.detector.expected_duration_seconds = 7200
    # Mock elapsed to be small (e.g. 1 hour)
    manager.detector.get_elapsed_seconds.return_value = 3660 

    await manager._watchdog_check_stuck_cycle(future)
    
    # With profile-aware logic, it should NOT be force ended
    manager.detector.force_end.assert_not_called()
    
    # This proves premature kill for > 1h pauses.

