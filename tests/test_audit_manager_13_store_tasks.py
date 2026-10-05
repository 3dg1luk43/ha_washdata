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
"""Audit MANAGER-13 residue (register item 481): the suggestion save was untracked.

``SuggestionEngine.apply_suggestions`` saved the store with a bare
``hass.async_create_task``. The learning manager's own tasks already went through
the manager's ``_spawn_tracked``, but the engine it owns did not, so an unload could
neither cancel nor await that save and the old ProfileStore could write last after
a reload. (The other two bare tasks, the sync ``ProfileStore.add_cycle`` and the
per-profile rebuild in ``async_enforce_retention``, were unreachable since item 463
and are removed.)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine


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
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    mgr.profile_store.async_flush_saves = AsyncMock()
    mgr._notify_update = MagicMock()
    return mgr


async def test_the_suggestion_save_is_tracked_and_cancelled_on_unload(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    gate = asyncio.Event()
    saves: list[int] = []

    async def _save() -> None:
        saves.append(1)
        if len(saves) == 1:  # the suggestion save never finishes on its own
            await gate.wait()

    manager.profile_store.async_save = _save
    before = set(manager._background_tasks)
    manager.learning_manager.suggestion_engine.apply_suggestions(
        {"off_delay": {"value": 240, "reason": "test"}}
    )
    await asyncio.sleep(0)
    new = set(manager._background_tasks) - before
    assert len(new) == 1, "the suggestion save was not tracked, so an unload cannot cancel it"
    (task,) = new

    await manager.async_shutdown()
    assert task.cancelled()
    # The cancelled save is replaced by the unload's own single write.
    assert len(saves) == 2


async def test_without_a_spawner_the_engine_still_schedules_its_save(mock_hass) -> None:
    mock_hass.async_create_task = MagicMock()
    store = MagicMock()
    store.get_suggestions = MagicMock(return_value={})
    store.async_save = MagicMock(return_value="save-coro")
    eng = SuggestionEngine(mock_hass, "e", store, device_type="washing_machine")
    eng.apply_suggestions({"off_delay": {"value": 240, "reason": "test"}})
    mock_hass.async_create_task.assert_called_once_with("save-coro")


async def test_a_spawner_receives_the_save_and_hass_does_not(mock_hass) -> None:
    mock_hass.async_create_task = MagicMock()
    store = MagicMock()
    store.get_suggestions = MagicMock(return_value={})
    store.async_save = MagicMock(return_value="save-coro")
    spawned: list[object] = []
    eng = SuggestionEngine(
        mock_hass, "e", store, device_type="washing_machine", spawn=spawned.append
    )
    # A per-job copy (executor dispatch) keeps the spawner.
    eng.for_job({}).apply_suggestions({"off_delay": {"value": 240, "reason": "test"}})
    assert spawned == ["save-coro"]
    mock_hass.async_create_task.assert_not_called()
