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
"""Audit MANAGER-13: store-touching tasks bypassed ``_spawn_tracked``.

The learning manager's saves and suggestion passes, the periodic active-cycle
snapshot and the noise auto-tune were bare ``hass.async_create_task`` calls. An
unload could not cancel them, so after an entry reload the old ProfileStore,
which shares the Store key, could write last and clobber the new one.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.time_utils import utc_now


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
    mgr.profile_store.async_save = AsyncMock()
    mgr.profile_store.async_flush_saves = AsyncMock()
    mgr._notify_update = MagicMock()
    return mgr


def _blocked() -> tuple[asyncio.Event, Any]:
    gate = asyncio.Event()

    async def _never_finishes(*_a: Any, **_k: Any) -> None:
        await gate.wait()

    return gate, _never_finishes


async def _assert_one_tracked_task_cancelled_on_unload(
    hass: HomeAssistant, mgr: WashDataManager, before: set[Any]
) -> None:
    await asyncio.sleep(0)
    new = set(mgr._background_tasks) - before
    assert len(new) == 1, "the task was not tracked, so an unload cannot cancel it"
    (task,) = new
    await mgr.async_shutdown()
    assert task.cancelled()
    # Its save never ran, so the unload writes the store once in its place.
    mgr.profile_store.async_save.assert_awaited_once()


async def test_learning_manager_tasks_are_tracked(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    _gate, slow = _blocked()
    lm = manager.learning_manager
    lm._async_run_detection_suggestions = slow
    before = set(manager._background_tasks)
    lm._update_detection_suggestions()
    await _assert_one_tracked_task_cancelled_on_unload(hass, manager, before)


async def test_periodic_snapshot_save_is_tracked(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    _gate, slow = _blocked()
    manager.profile_store.async_save_active_cycle = slow
    manager._augment_active_snapshot = MagicMock(return_value={})
    manager._last_state_save = None
    before = set(manager._background_tasks)
    manager._check_state_save(utc_now())
    await _assert_one_tracked_task_cancelled_on_unload(hass, manager, before)


async def test_noise_auto_tune_is_tracked(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    _gate, slow = _blocked()
    manager._tune_threshold = slow
    manager._noise_events_threshold = 1
    before = set(manager._background_tasks)
    manager._handle_noise_cycle(12.0)
    await _assert_one_tracked_task_cancelled_on_unload(hass, manager, before)


async def test_unload_with_nothing_in_flight_does_not_rewrite_the_store(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    await manager.async_shutdown()
    manager.profile_store.async_save.assert_not_awaited()
    manager.profile_store.async_flush_saves.assert_awaited_once()
