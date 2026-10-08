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
"""Audit MANAGER-11: a power-sensor change saved mid-cycle was dropped.

The reload skips the swap while a cycle is under way, and nothing re-applied it
when the cycle ended, so the manager kept listening to the old plug until the
next save or restart while the panel already showed the new one.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    STATE_ANTI_WRINKLE,
    STATE_ENDING,
    STATE_OFF,
    STATE_RUNNING,
)
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.old"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_duration_ratio_limits = MagicMock(return_value=(0.1, 1.8))
    mgr._maybe_arm_door_end_dwell_if_open = MagicMock()
    mgr._notify_update = MagicMock()
    mgr._subscribe_power_sensor()
    return mgr


async def _save_sensor_mid_cycle(hass: HomeAssistant, mgr: WashDataManager) -> None:
    mgr.detector.state = STATE_RUNNING
    entry = mgr.config_entry
    entry.options = {"power_sensor": "sensor.new"}
    await mgr.async_reload_config(entry)
    await hass.async_block_till_done()


async def test_swap_saved_mid_cycle_lands_when_the_cycle_ends(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    await _save_sensor_mid_cycle(hass, manager)
    assert manager.power_sensor_entity_id == "sensor.old"  # not while running

    manager.detector.state = STATE_OFF
    manager._on_state_change(STATE_ENDING, STATE_OFF)
    # Not inside the callback: it runs within the detector's process_reading, and
    # the swap feeds a reading back into it (HA starts tasks eagerly).
    assert manager.power_sensor_entity_id == "sensor.old"
    await hass.async_block_till_done()
    assert manager.power_sensor_entity_id == "sensor.new"

    # ...and the listener really moved: the new plug drives the detector, the old
    # one no longer does.
    manager.detector.process_reading.reset_mock()
    hass.states.async_set("sensor.old", "900")
    await hass.async_block_till_done()
    manager.detector.process_reading.assert_not_called()
    hass.states.async_set("sensor.new", "850")
    await hass.async_block_till_done()
    assert manager.detector.process_reading.called


async def test_swap_waits_out_anti_wrinkle(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    await _save_sensor_mid_cycle(hass, manager)

    manager.detector.state = STATE_ANTI_WRINKLE
    manager._on_state_change(STATE_ENDING, STATE_ANTI_WRINKLE)
    await hass.async_block_till_done()
    assert manager.power_sensor_entity_id == "sensor.old"

    manager.detector.state = STATE_OFF
    manager._on_state_change(STATE_ANTI_WRINKLE, STATE_OFF)
    await hass.async_block_till_done()
    assert manager.power_sensor_entity_id == "sensor.new"


async def test_reverting_the_sensor_before_the_end_cancels_the_swap(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    await _save_sensor_mid_cycle(hass, manager)
    entry = manager.config_entry
    entry.options = {"power_sensor": "sensor.old"}
    await manager.async_reload_config(entry)

    manager.detector.state = STATE_OFF
    manager._on_state_change(STATE_ENDING, STATE_OFF)
    await hass.async_block_till_done()
    assert manager.power_sensor_entity_id == "sensor.old"
