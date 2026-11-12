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
"""Issue #509: the watchdog's ghost suppressor force-ended a cycle drawing real power.

The suppressor ends an unmatched cycle that started within 180 s (a dishwasher
600 s) of the previous end once it is 10 min old and the plug has been silent for
5 min (a dishwasher: 3 min old, 1 min silent). It never read the power, so a
change-only plug holding a steady draw - which sends nothing while the draw stays
flat - had its next run killed mid-cycle as a "ghost".
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import (
    DOMAIN,
    STATE_FINISHED,
    STATE_RUNNING,
)

POWER = "sensor.plug_power"
# Detector end gates well past the suppressor's own clock, so only the watchdog
# can close a cycle inside the windows below.
FAST = {
    "min_power": 2.0,
    "stop_threshold_w": 4.0,
    "start_threshold_w": 5.0,
    "off_delay": 60,
    "min_off_gap": 900,
    "completion_min_seconds": 300,
    "watchdog_interval": 30,
}
# (device type, elapsed the suppressor needs, silence it needs), in s.
WASHER = ("washing_machine", 600, 300)
DISHWASHER = ("dishwasher", 180, 60)


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int, watts: float | None) -> None:
    freezer.tick(timedelta(seconds=seconds))
    if watts is not None:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"})
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _boot(hass: HomeAssistant, setup_washdata_entry, device_type: str) -> tuple[Any, list]:
    entry = await setup_washdata_entry(
        power_sensor=POWER, device_type=device_type, options=FAST
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    ends: list[tuple[str, float]] = []
    original = manager.detector._on_cycle_end

    def _spy(cycle_data: dict[str, Any]) -> None:
        ends.append((cycle_data.get("status"), manager._current_power))
        original(cycle_data)

    manager.detector._on_cycle_end = _spy
    return manager, ends


async def _first_cycle(hass: HomeAssistant, freezer: Any, manager: Any, ends: list) -> None:
    for i in range(30):  # 15 min of a reporting 400 W run
        await _tick(hass, freezer, 30, 400.0 + i % 2)
    await _tick(hass, freezer, 30, 0.0)
    for _ in range(60):
        if ends:
            break
        await _tick(hass, freezer, 30, None)
    assert [s for s, _p in ends] == ["completed"]
    assert manager.detector.state == STATE_FINISHED


@pytest.mark.parametrize("device", [WASHER, DISHWASHER], ids=lambda d: d[0])
async def test_a_cycle_drawing_power_on_a_silent_plug_is_not_a_ghost(
    hass: HomeAssistant, setup_washdata_entry, freezer, device
) -> None:
    """Back-to-back run 60 s after the last end, steady draw, plug silent."""
    device_type, need_elapsed, need_silence = device
    manager, ends = await _boot(hass, setup_washdata_entry, device_type)
    await _first_cycle(hass, freezer, manager, ends)

    await _tick(hass, freezer, 60, 400.0)  # the next load, 60 s after the end
    for i in range(need_elapsed // 30):  # reports until the suppressor's age
        await _tick(hass, freezer, 30, 380.0 + i % 2)
    assert manager.detector.state == STATE_RUNNING
    for _ in range(2 * need_silence // 30):  # flat: a change-only plug sends nothing
        await _tick(hass, freezer, 30, None)

    assert ends[1:] == [], f"a cycle drawing {manager._current_power} W was ended: {ends}"
    assert manager.detector.state == STATE_RUNNING


@pytest.mark.parametrize("device", [WASHER, DISHWASHER], ids=lambda d: d[0])
async def test_a_ghost_quiet_at_zero_is_still_suppressed(
    hass: HomeAssistant, setup_washdata_entry, freezer, device, caplog
) -> None:
    """Guard (passes before and after): what the suppressor exists for, a blip
    after the end (pump-out, spin-down) left a cycle open on a plug quiet at 0 W."""
    device_type, need_elapsed, need_silence = device
    manager, ends = await _boot(hass, setup_washdata_entry, device_type)
    await _first_cycle(hass, freezer, manager, ends)

    await _tick(hass, freezer, 60, 200.0)
    for i in range(need_elapsed // 30):
        await _tick(hass, freezer, 30, 200.0 + i % 2)
    await _tick(hass, freezer, 30, 0.0)
    for _ in range(need_silence // 30 + 3):
        await _tick(hass, freezer, 30, None)
    # Force-ended by the suppressor (a short one is then stored as interrupted).
    assert "Ghost cycle suppressed" in caplog.text
    assert len(ends) == 2 and ends[1][0] in ("force_stopped", "interrupted"), ends
