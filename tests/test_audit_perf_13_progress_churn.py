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
"""Audit PERF-13 (extra): the progress sensor wrote a recorder row every estimate.

Its state was the raw float progress (45.13, 45.21, ...) and its projected
energy/cost attributes moved with every reading, so each 5 s estimate was a
``state_changed`` event, i.e. one recorder row. The state is a whole percent now
and the projection refreshes when that percent moves. Counted on a real HA bus.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    MockEntityPlatform,
)

from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.sensor import WasherProgressSensor, _whole_percent


async def _sensor(hass: HomeAssistant) -> tuple[MagicMock, WasherProgressSensor]:
    entry = MockConfigEntry(domain=DOMAIN, title="Washer", entry_id="e1")
    manager = MagicMock()
    manager.cycle_progress = 45.1
    manager.projected_energy_wh = 812.4
    manager.projected_cost = 0.244
    sensor = WasherProgressSensor(manager, entry)
    platform = MockEntityPlatform(hass, domain="sensor", platform_name=DOMAIN)
    await platform.async_add_entities([sensor])
    await hass.async_block_till_done()
    return manager, sensor


async def test_estimates_within_one_percent_write_no_row(hass: HomeAssistant) -> None:
    manager, sensor = await _sensor(hass)
    changed: list[object] = []
    hass.bus.async_listen(EVENT_STATE_CHANGED, changed.append)

    for i in range(12):  # a minute of 5 s estimates inside the same percent
        manager.cycle_progress = 45.1 + i * 0.03
        manager.projected_energy_wh = 812.4 + i * 1.7  # moves with every reading
        manager.projected_cost = 0.244 + i * 0.0006
        sensor.async_write_ha_state()
    await hass.async_block_till_done()

    assert changed == [], f"{len(changed)} state_changed events inside one percent"
    state = hass.states.get(sensor.entity_id)
    assert state is not None and state.state == "45"
    assert state.attributes["projected_energy_kwh"] == 0.812

    # The next percent is one row, and it carries the projection of that moment.
    manager.cycle_progress = 45.6
    manager.projected_energy_wh = 840.0
    sensor.async_write_ha_state()
    await hass.async_block_till_done()
    assert len(changed) == 1
    state = hass.states.get(sensor.entity_id)
    assert state.state == "46"
    assert state.attributes["projected_energy_kwh"] == 0.84


async def test_a_projection_that_appears_or_goes_is_written_at_once(
    hass: HomeAssistant,
) -> None:
    manager, sensor = await _sensor(hass)
    manager.projected_energy_wh = None
    manager.projected_cost = None
    sensor.async_write_ha_state()
    await hass.async_block_till_done()
    state = hass.states.get(sensor.entity_id)
    assert state.state == "45"
    assert "projected_energy_kwh" not in state.attributes

    manager.projected_energy_wh = 900.0
    sensor.async_write_ha_state()
    await hass.async_block_till_done()
    state = hass.states.get(sensor.entity_id)
    assert state.attributes["projected_energy_kwh"] == 0.9
    assert "projected_cost" not in state.attributes


def test_whole_percent_rounds_half_up_like_the_card() -> None:
    assert _whole_percent(None) is None
    assert _whole_percent(float("nan")) is None
    assert _whole_percent(0.0) == 0
    assert _whole_percent(44.5) == 45
    assert _whole_percent(45.49) == 45
    assert _whole_percent(99.6) == 100
    assert _whole_percent(100.0) == 100
