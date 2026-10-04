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
"""Audit MANAGER-15: two small reload / manual-program gaps.

1. ``_noise_events_threshold`` was read only in the constructor, so an options
   reload kept the old value (the last item-388c-style reload gap).
2. ``clear_manual_program`` compared the state with the literal "running", so
   clearing a pin in PAUSED or ENDING set the program to "off" mid-cycle.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    STATE_ENDING,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.p"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_duration_ratio_limits = MagicMock(return_value=(0.1, 1.8))
    mgr._maybe_arm_door_end_dwell_if_open = MagicMock()
    mgr._notify_update = MagicMock()
    return mgr


async def test_noise_threshold_follows_an_options_reload(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    assert manager._noise_events_threshold == 3  # the default
    manager.detector.state = STATE_OFF
    entry = manager.config_entry
    entry.options = {"power_sensor": "sensor.p", CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD: 7}
    await manager.async_reload_config(entry)
    assert manager._noise_events_threshold == 7


@pytest.mark.parametrize("state", [STATE_PAUSED, STATE_ENDING])
def test_clearing_a_pin_while_paused_or_ending_returns_to_detection(
    manager: WashDataManager, state: str
) -> None:
    manager.detector.state = state
    manager._manual_program_active = True
    manager._current_program = "Cotton 60"
    manager._matched_profile_duration = 7200.0
    manager._update_remaining_only = MagicMock()
    manager._update_estimates = MagicMock()

    manager.clear_manual_program()

    assert manager._current_program == "detecting..."
    assert manager._matched_profile_duration is None
    manager._update_remaining_only.assert_called_once()
    manager._update_estimates.assert_not_called()


def test_clearing_a_pin_while_running_reestimates(manager: WashDataManager) -> None:
    manager.detector.state = STATE_RUNNING
    manager._manual_program_active = True
    manager._current_program = "Cotton 60"
    manager._update_estimates = MagicMock()

    manager.clear_manual_program()

    assert manager._current_program == "detecting..."
    manager._update_estimates.assert_called_once()


def test_clearing_a_pin_when_idle_shows_off(manager: WashDataManager) -> None:
    manager.detector.state = STATE_OFF
    manager._manual_program_active = True
    manager._current_program = "Cotton 60"

    manager.clear_manual_program()

    assert manager._current_program == "off"
