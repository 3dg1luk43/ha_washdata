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
"""Audit MANAGER-16: an HA restart lost every held notification.

HA does not unload config entries on a stop, so ``async_shutdown`` never ran on a
restart: the quiet-hours and presence queues (a deferred "finished", a milestone)
lived only in memory and were gone, and the active snapshot was up to 60 s old.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    NOTIFY_EVENT_CLEAN,
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
    NOTIFY_EVENT_START,
    NOTIFY_EVENT_TIMER,
    STATE_OFF,
    STATE_RUNNING,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.time_utils import utc_now

_KEY = "ha_washdata.e.notify_queue"


def _make(hass: HomeAssistant) -> WashDataManager:
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
    mgr.profile_store.async_save = AsyncMock()
    mgr.profile_store.async_flush_saves = AsyncMock()
    mgr.profile_store.async_save_active_cycle = AsyncMock()
    mgr._notify_update = MagicMock()
    mgr.detector.state = STATE_OFF
    return mgr


def _held(event_type: str, message: str) -> dict[str, Any]:
    return {
        "message": message, "title": "WashData", "icon": None,
        "event_type": event_type, "extra_vars": {"tag": "t", "program": "Cotton"},
    }


def _fill_queues(mgr: WashDataManager) -> None:
    mgr._quiet_pending_notifications = [
        _held(NOTIFY_EVENT_FINISH, "W finished"),
        _held(NOTIFY_EVENT_CLEAN, "laundry inside"),
        _held("pre_complete", "10 min left"),
    ]
    mgr._pending_notifications = [
        _held(NOTIFY_EVENT_FINISH, "100th cycle"),
        _held(NOTIFY_EVENT_LIVE, "live 40%"),
        _held(NOTIFY_EVENT_TIMER, "softener"),
        _held(NOTIFY_EVENT_START, "W started"),
    ]


async def _stop(hass: HomeAssistant, mgr: WashDataManager) -> None:
    mgr._remove_ha_stop_listener = hass.bus.async_listen(
        EVENT_HOMEASSISTANT_STOP, mgr._async_on_ha_stop
    )
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()


async def test_stop_keeps_the_held_notifications_and_a_fresh_snapshot(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    mgr = _make(hass)
    _fill_queues(mgr)
    mgr.detector.state = STATE_RUNNING
    mgr._augment_active_snapshot = MagicMock(return_value={"state": STATE_RUNNING})

    await _stop(hass, mgr)

    mgr.profile_store.async_save_active_cycle.assert_awaited_once_with(
        {"state": STATE_RUNNING}
    )
    saved = hass_storage[_KEY]["data"]
    assert [e["message"] for e in saved["quiet"]] == ["W finished", "10 min left"]
    assert [e["message"] for e in saved["presence"]] == ["100th cycle", "W started"]


async def test_the_next_start_redispatches_them_once(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    first = _make(hass)
    _fill_queues(first)
    await _stop(hass, first)

    second = _make(hass)
    sent: list[str] = []
    second._dispatch_notification = MagicMock(
        side_effect=lambda msg, **_kw: sent.append(msg)
    )
    second._schedule_notify_queue_restore(hass)
    await hass.async_block_till_done()

    # The cycle is over, so its start / pre-complete are not replayed.
    assert sent == ["W finished", "100th cycle"]
    assert _KEY not in hass_storage  # read once

    third = _make(hass)
    third._dispatch_notification = MagicMock()
    await third._async_restore_notification_queues()
    third._dispatch_notification.assert_not_called()


async def test_a_queue_from_days_ago_is_dropped(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    hass_storage[_KEY] = {
        "version": 1, "minor_version": 1, "key": _KEY,
        "data": {
            "saved_at": (utc_now() - timedelta(days=3)).isoformat(),
            "quiet": [_held(NOTIFY_EVENT_FINISH, "W finished")],
            "presence": [],
        },
    }
    mgr = _make(hass)
    mgr._dispatch_notification = MagicMock()
    await mgr._async_restore_notification_queues()
    mgr._dispatch_notification.assert_not_called()
    assert _KEY not in hass_storage


async def test_an_entry_reload_keeps_them_too(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    mgr = _make(hass)
    _fill_queues(mgr)
    await mgr.async_shutdown()
    assert [e["message"] for e in hass_storage[_KEY]["data"]["quiet"]] == [
        "W finished", "10 min left",
    ]


async def test_nothing_held_writes_nothing(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    mgr = _make(hass)
    await _stop(hass, mgr)
    await mgr.async_shutdown()
    assert _KEY not in hass_storage
