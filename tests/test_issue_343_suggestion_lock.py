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

"""Issue #343: a per-setting lock so the auto-tuner stops re-suggesting values the
user has rejected (e.g. thresholds that break an anti-crease-tuned device).

Locking a setting persists the choice, drops any pending suggestion for it, and
prevents the surface path from ever re-surfacing it - until the user unlocks it.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.learning import LearningManager


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def mock_executor_job(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        s = ProfileStore(mock_hass, "test_entry", min_duration_ratio=0.0, max_duration_ratio=2.0)
        s._store.async_load = AsyncMock(return_value=None)
        s._store.async_save = AsyncMock()
        return s


# ── store ──────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_lock_adds_key_and_drops_pending_suggestion(store):
    store._data["suggestions"] = {"start_threshold_w": {"value": 3.6}}
    await store.set_suggestion_locked("start_threshold_w", True)
    assert "start_threshold_w" in store.get_locked_suggestions()
    # The pending suggestion for a locked key is removed immediately.
    assert "start_threshold_w" not in store.get_suggestions()


@pytest.mark.asyncio
async def test_unlock_removes_key(store):
    await store.set_suggestion_locked("stop_threshold_w", True)
    assert "stop_threshold_w" in store.get_locked_suggestions()
    await store.set_suggestion_locked("stop_threshold_w", False)
    assert "stop_threshold_w" not in store.get_locked_suggestions()


def test_get_locked_defaults_empty(store):
    assert store.get_locked_suggestions() == []


# ── learning surface filter ──────────────────────────────────────────────────
@pytest.fixture
def learning(mock_hass):
    entry = MagicMock()
    entry.data = {}
    entry.options = {"start_threshold_w": 10.0, "stop_threshold_w": 5.0}
    mock_hass.config_entries.async_get_entry.return_value = entry
    ps = MagicMock()
    ps.get_past_cycles.return_value = []
    ps.get_suggestion_apply_cycle_count.return_value = 0
    ps.get_locked_suggestions.return_value = ["start_threshold_w"]
    ps.get_suggestions.return_value = {"start_threshold_w": {"value": 3.6}}
    ps.delete_suggestion = MagicMock()
    mgr = LearningManager(mock_hass, "test_entry", ps, device_type="washing_machine")
    mgr.suggestion_engine.apply_suggestions = MagicMock()
    return mgr


def test_locked_suggestion_is_not_surfaced(learning):
    learning._apply_suggestions_and_notify({
        "start_threshold_w": {"value": 3.6},   # locked -> must be dropped
        "stop_threshold_w": {"value": 2.4},    # not locked -> surfaced
    })
    learning.profile_store.delete_suggestion.assert_any_call("start_threshold_w")
    # apply_suggestions received only the unlocked key.
    assert learning.suggestion_engine.apply_suggestions.call_count == 1
    surfaced = learning.suggestion_engine.apply_suggestions.call_args.args[0]
    assert "start_threshold_w" not in surfaced
    assert "stop_threshold_w" in surfaced


def test_all_locked_triggers_save(mock_hass):
    """When every incoming suggestion is locked the in-memory deletions must be
    persisted: delete_suggestion only mutates _data in memory, so without an
    async_save the pruned keys re-appear from disk on HA restart (regression guard)."""
    entry = MagicMock()
    entry.data = {}
    entry.options = {}
    mock_hass.config_entries.async_get_entry.return_value = entry
    ps = MagicMock()
    ps.get_past_cycles.return_value = []
    ps.get_suggestion_apply_cycle_count.return_value = 0
    ps.get_locked_suggestions.return_value = ["start_threshold_w", "stop_threshold_w"]
    ps.get_suggestions.return_value = {"start_threshold_w": {"value": 3.6}, "stop_threshold_w": {"value": 2.4}}
    ps.delete_suggestion = MagicMock()
    ps.async_save = AsyncMock()
    mgr = LearningManager(mock_hass, "test_entry", ps, device_type="washing_machine")
    mgr.suggestion_engine.apply_suggestions = MagicMock()

    created_tasks = []
    mock_hass.async_create_task = MagicMock(side_effect=lambda coro, *a: created_tasks.append(coro))

    mgr._apply_suggestions_and_notify({
        "start_threshold_w": {"value": 3.6},
        "stop_threshold_w": {"value": 2.4},
    })

    # Both keys deleted in-memory.
    assert ps.delete_suggestion.call_count == 2
    # apply_suggestions NOT called (nothing to surface).
    mgr.suggestion_engine.apply_suggestions.assert_not_called()
    # A save task was scheduled so the deletions reach disk.
    assert len(created_tasks) == 1
