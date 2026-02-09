"""Reproduction test for state expiry."""
from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import timedelta, datetime, timezone
from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS,
    STATE_FINISHED, STATE_OFF, CONF_PROGRESS_RESET_DELAY
)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    return hass

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_PROGRESS_RESET_DELAY: 150,
    }
    return entry

async def test_finished_state_expiry(mock_hass, mock_entry):
    """Test that Finished state expires even without new readings."""
    # Ensure dt_util.now() returns a consistent time
    now = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
    
    with patch("homeassistant.util.dt.now", return_value=now), \
         patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector") as mock_detector_class:
        
        mock_detector = mock_detector_class.return_value
        # Initially in FINISHED state
        mock_detector.state = STATE_FINISHED
        
        manager = WashDataManager(mock_hass, mock_entry)
        
        # Simulate cycle completed 31 minutes ago
        manager._cycle_completed_time = now - timedelta(minutes=31)
        manager._cycle_progress = 100.0
        
        # Manually trigger the check (simulating async_track_time_interval callback)
        await manager._handle_state_expiry(now)
        
        # Check if progress was reset (YES)
        assert manager._cycle_progress == 0.0
        
        # Check if detector.reset(STATE_OFF) was called
        mock_detector.reset.assert_called_once_with(STATE_OFF)

async def test_expiry_timer_cancelled_on_new_cycle(mock_hass, mock_entry):
    """Test that starting a new cycle cancels the expiry timer."""
    now = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
    
    with patch("homeassistant.util.dt.now", return_value=now), \
         patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector") as mock_detector_class:
        
        mock_detector = mock_detector_class.return_value
        manager = WashDataManager(mock_hass, mock_entry)
        
        # Start the expiry timer
        manager._cycle_completed_time = now
        manager._start_state_expiry_timer()
        assert manager._remove_state_expiry_timer is not None
        
        # Simulate new cycle start
        # In manager.py, _on_state_change(STATE_OFF, STATE_RUNNING) calls _stop_state_expiry_timer
        manager._on_state_change(STATE_OFF, "running")
        
        assert manager._remove_state_expiry_timer is None

async def test_expiry_no_completed_time(mock_hass, mock_entry):
    """Test that expiry handles missing completed time."""
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        manager = WashDataManager(mock_hass, mock_entry)
        manager._cycle_completed_time = None
        await manager._handle_state_expiry(dt_util.now())
        assert manager._cycle_progress == 0 # Default

async def test_expiry_during_running(mock_hass, mock_entry):
    """Test that expiry does nothing if cycle is running."""
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector") as mock_detector_class:
        mock_detector = mock_detector_class.return_value
        mock_detector.state = STATE_RUNNING
        manager = WashDataManager(mock_hass, mock_entry)
        manager._cycle_completed_time = dt_util.now() - timedelta(minutes=31)
        manager._cycle_progress = 50.0
        await manager._handle_state_expiry(dt_util.now())
        assert manager._cycle_progress == 50.0