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
"""#517: notification actions dropped a notification sent while the last one ran.

The actions ran through one `Script` in Home Assistant's default `single` mode, so
a second run while the first still awaited its notify call was rejected ("Already
running"). Notifications go out in bursts, so the Live Activity clear after the
finished alert, or one of the start alert and the first live card, was lost.
Driven through a real `Script` against a service that blocks until released.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant, ServiceCall

from custom_components.ha_washdata.manager import WashDataManager

ACTIONS = [{"action": "test.slow_notify", "data": {"message": "{{ message }}"}}]


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "entry_517"
    entry.title = "Lavasciuga"
    entry.options = {"power_sensor": "sensor.test_power", "notify_actions": ACTIONS}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        return mgr


async def test_a_burst_of_two_notifications_delivers_both_in_order(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    release = asyncio.Event()
    delivered: list[str] = []

    async def _slow_notify(call: ServiceCall) -> None:
        await release.wait()
        delivered.append(call.data["message"])

    hass.services.async_register("test", "slow_notify", _slow_notify)

    # The finished alert, then the clear that ends the Live Activity, while the
    # first is still waiting on its notify service.
    assert manager._run_notification_actions({"message": "finished"})  # noqa: SLF001
    await asyncio.sleep(0)
    assert manager._run_notification_actions({"message": "clear_notification"})  # noqa: SLF001
    await asyncio.sleep(0)
    release.set()
    await hass.async_block_till_done()

    assert delivered == ["finished", "clear_notification"]


async def test_the_script_is_queued_with_a_bound(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    hass.services.async_register("test", "slow_notify", lambda call: None)
    manager._run_notification_actions({"message": "x"})  # noqa: SLF001
    await hass.async_block_till_done()
    script: Any = manager._notify_script  # noqa: SLF001
    assert script.script_mode == "queued"
    assert script.max_runs == 20
