# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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
async def test_repro_premature_kill_during_expected_pause(manager):
    """
    Reproduction: Watchdog kills cycle during a legitimate long pause that matches
    the profile (e.g. soak cycle), because it ignores profile look-ahead.

    A low-power wait silent for 61 min on a 60 min timeout, one hour into a 2 h
    profile: only the look-ahead (expected - elapsed + 30 min) keeps it alive.
    """
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    manager.detector.state = STATE_RUNNING
    manager._current_program = "Long Soak"
    manager._matched_profile_duration = 7200  # 2 hours
    # Not a verified pause: that extension alone would cover 4 h + 30 min and
    # hide the look-ahead (a bare MagicMock attribute is truthy).
    manager.detector._verified_pause = False

    manager._current_power = 0.0
    manager.detector.is_waiting_low_power.return_value = True
    manager._low_power_no_update_timeout = 3600  # 1 hour
    manager._last_reading_time = now - timedelta(minutes=61)
    manager._last_real_reading_time = now - timedelta(minutes=61)
    manager.detector.expected_duration_seconds = 7200
    manager.detector.get_elapsed_seconds.return_value = 3660

    await manager._watchdog_check_stuck_cycle(now)

    # With profile-aware logic, it should NOT be force ended
    manager.detector.force_end.assert_not_called()
    
    # This proves premature kill for > 1h pauses.

