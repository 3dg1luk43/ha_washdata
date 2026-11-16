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
"""Issue #524: a front loader's door is locked while it runs, so a door held open
means the cycle is over.

"Door Opens Automatically At End" (#342) already finalised on a held door-open,
but only in RUNNING and ENDING. After the final spin the power drops and the
detector sits in PAUSED for minutes before ENDING, which is exactly when the
laundry is taken out: the door-open there set the sticky user pause instead, and
the cycle waited out the whole off delay (or longer, paused).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import (
    DOMAIN,
    STATE_PAUSED,
    STATE_RUNNING,
    TerminationReason,
)

POWER = "sensor.washer_power"
DOOR = "binary_sensor.washer_door"
OPTS = {
    "min_power": 2.0,
    "stop_threshold_w": 4.0,
    "start_threshold_w": 5.0,
    "off_delay": 1800,  # long, so only the door can end the cycle in this test
    "min_off_gap": 900,
    "completion_min_seconds": 300,
    "door_sensor_entity": DOOR,
    "door_opens_at_end": True,
    "door_end_dwell_seconds": 60,
}


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int, watts: float | None) -> None:
    freezer.tick(timedelta(seconds=seconds))
    if watts is not None:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"})
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _paused_washer(hass: HomeAssistant, setup_washdata_entry, freezer, options) -> tuple[Any, list]:
    hass.states.async_set(DOOR, "off")
    entry = await setup_washdata_entry(power_sensor=POWER, options=options)
    manager = hass.data[DOMAIN][entry.entry_id]
    ends: list[dict[str, Any]] = []
    original = manager.detector._on_cycle_end

    def _spy(cycle_data: dict[str, Any]) -> None:
        ends.append(dict(cycle_data))
        original(cycle_data)

    manager.detector._on_cycle_end = _spy
    for i in range(40):  # 20 min of washing
        await _tick(hass, freezer, 30, 300.0 + i % 2)
    assert manager.detector.state == STATE_RUNNING
    for _ in range(40):  # the final spin is over: quiet until the detector pauses
        await _tick(hass, freezer, 30, 0.5)
        if manager.detector.state == STATE_PAUSED:
            break
    assert manager.detector.state == STATE_PAUSED
    return manager, ends


async def test_door_held_open_while_paused_ends_the_cycle(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)

    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager._is_user_paused is False  # not the sticky pause
    await _tick(hass, freezer, 30, None)
    assert ends == []  # still inside the dwell
    await _tick(hass, freezer, 40, None)

    assert len(ends) == 1, ends
    assert ends[0]["status"] == "completed"
    assert ends[0]["termination_reason"] == TerminationReason.USER


async def test_brief_open_while_paused_does_not_end_it(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)

    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    await _tick(hass, freezer, 20, None)
    hass.states.async_set(DOOR, "off")
    await hass.async_block_till_done()
    await _tick(hass, freezer, 120, None)

    assert ends == []
    assert manager._is_user_paused is False


async def test_without_the_option_a_paused_door_open_still_pauses(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Unchanged: with the option off, opening the door is a pause."""
    manager, ends = await _paused_washer(
        hass, setup_washdata_entry, freezer, {**OPTS, "door_opens_at_end": False}
    )
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    await _tick(hass, freezer, 120, None)

    assert ends == []
    assert manager._is_user_paused is True
