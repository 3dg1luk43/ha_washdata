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
"""Audit PERF-13: the total-duration sensor wrote a recorder row every estimate.

Its `last_updated` attribute was stamped on every 5 s estimate, so each state
write was a `state_changed` event (the recorder stores one row per such event)
even while the whole-minute value did not move. Counted on a real HA bus.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    MockEntityPlatform,
)

from custom_components.ha_washdata.const import DOMAIN, STATE_RUNNING
from custom_components.ha_washdata.sensor import WasherTotalDurationSensor


async def test_repeated_estimates_with_the_same_minute_write_one_row(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(domain=DOMAIN, title="Washer", entry_id="e1")
    manager = MagicMock()
    manager.check_state.return_value = STATE_RUNNING
    manager.total_duration = 5400.4
    sensor = WasherTotalDurationSensor(manager, entry)
    platform = MockEntityPlatform(hass, domain="sensor", platform_name=DOMAIN)
    await platform.async_add_entities([sensor])
    await hass.async_block_till_done()

    changed: list[object] = []
    hass.bus.async_listen(EVENT_STATE_CHANGED, changed.append)

    t0 = dt_util.utcnow()
    for i in range(12):  # one minute of 5 s estimates, same 90 min total
        manager.total_duration = 5400.4 + i  # the seconds move, the minute does not
        manager.last_total_duration_update = t0 + timedelta(seconds=5 * i)
        sensor.async_write_ha_state()
    await hass.async_block_till_done()

    assert changed == [], f"{len(changed)} state_changed events for an unchanged value"
    state = hass.states.get(sensor.entity_id)
    assert state is not None and state.state == "90"
    assert "last_updated" not in state.attributes

    # The value moving is still one row, as it should be.
    manager.total_duration = 5460.0
    sensor.async_write_ha_state()
    await hass.async_block_till_done()
    assert len(changed) == 1
