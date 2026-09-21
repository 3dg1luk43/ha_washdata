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
"""Issue #442: option writes that bypassed the settings changelog.

``ws_set_options`` recorded a diff before saving and was the only writer that
did, so "Apply all", an import and both store-download paths changed
``entry.options`` invisibly - and the per-setting revert had nothing to revert
to. The reporter's "Apply all" changed nine tunables at once; their previous
values survived only because they happened to have an older diagnostics dump.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN

pytestmark = pytest.mark.asyncio


def _entry(options: dict[str, Any] | None = None) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {"power_sensor": "sensor.power"}
    entry.options = options if options is not None else {"off_delay": 1780, "min_power": 1.2}
    return entry


def _hass() -> tuple[MagicMock, MagicMock]:
    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()
    manager.profile_store.get_past_cycles.return_value = []
    manager.profile_store.clear_suggestions = AsyncMock()
    manager.profile_store.get_suggestions.return_value = {}
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}
    return hass, manager


def _recorded(manager: MagicMock) -> dict[str, dict[str, Any]]:
    """key -> entry, across every recording call."""
    out: dict[str, dict[str, Any]] = {}
    for call in manager.profile_store.async_record_settings_changes.call_args_list:
        for ch in call.args[0]:
            out[ch["key"]] = ch
    return out


# ---------------------------------------------------------------------------
# The reported path
# ---------------------------------------------------------------------------


async def test_apply_all_suggestions_is_recorded_and_revertible() -> None:
    """The reporter's case, with their own before/after values."""
    entry = _entry({"off_delay": 1780, "min_power": 1.2, "smoothing_window": 2})
    hass, manager = _hass()
    manager.profile_store.get_suggestions.return_value = {
        "off_delay": {"value": 1327},
        "min_power": {"value": 1.0},
        "smoothing_window": {"value": 7},
    }

    ws_fn = ws_api.ws_apply_suggestions.__wrapped__
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_fn(
            hass,
            MagicMock(),
            {
                "id": 1,
                "entry_id": "e1",
                "keys": ["off_delay", "min_power", "smoothing_window"],
            },
        )

    rec = _recorded(manager)
    assert set(rec) == {"off_delay", "min_power", "smoothing_window"}
    # old/new must both be present, or "revert to previous value" has no target.
    assert rec["off_delay"]["old"] == 1780 and rec["off_delay"]["new"] == 1327
    assert rec["min_power"]["old"] == 1.2 and rec["min_power"]["new"] == 1.0
    assert rec["smoothing_window"]["old"] == 2 and rec["smoothing_window"]["new"] == 7
    for ch in rec.values():
        assert ch["timestamp"]


async def test_recording_happens_before_the_entry_update() -> None:
    """async_update_entry schedules a reload that rebuilds the store, so a diff
    recorded afterwards can be lost."""
    entry = _entry({"off_delay": 1780})
    hass, manager = _hass()
    manager.profile_store.get_suggestions.return_value = {"off_delay": {"value": 1327}}

    order: list[str] = []
    manager.profile_store.async_record_settings_changes = AsyncMock(
        side_effect=lambda ch: order.append("record")
    )
    hass.config_entries.async_update_entry = MagicMock(
        side_effect=lambda *a, **k: order.append("update")
    )

    ws_fn = ws_api.ws_apply_suggestions.__wrapped__
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_fn(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "keys": ["off_delay"]}
        )

    assert order.index("record") < order.index("update"), order


async def test_a_changelog_failure_does_not_block_the_apply() -> None:
    """Recording is best-effort: the write it describes must still happen."""
    entry = _entry({"off_delay": 1780})
    hass, manager = _hass()
    manager.profile_store.get_suggestions.return_value = {"off_delay": {"value": 1327}}
    manager.profile_store.async_record_settings_changes = AsyncMock(
        side_effect=RuntimeError("disk full")
    )

    ws_fn = ws_api.ws_apply_suggestions.__wrapped__
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_fn(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "keys": ["off_delay"]}
        )

    saved = hass.config_entries.async_update_entry.call_args.kwargs["options"]
    assert saved["off_delay"] == 1327


# ---------------------------------------------------------------------------
# The helper itself, which the other three writers share
# ---------------------------------------------------------------------------


async def test_helper_records_only_genuine_changes() -> None:
    entry = _entry({"off_delay": 180, "min_power": 2.0})
    hass, manager = _hass()

    await ws_api._record_option_changes(
        hass, entry, {"off_delay": 180, "min_power": 1.0}, "test"
    )

    rec = _recorded(manager)
    assert set(rec) == {"min_power"}, "an unchanged value must not be recorded"
    assert rec["min_power"]["old"] == 2.0 and rec["min_power"]["new"] == 1.0


async def test_helper_reads_through_to_entry_data() -> None:
    """The effective value can come from entry.data, not options."""
    entry = _entry({})
    entry.data = {"power_sensor": "sensor.power", "min_power": 5.0}
    hass, manager = _hass()

    await ws_api._record_option_changes(hass, entry, {"min_power": 1.0}, "test")

    rec = _recorded(manager)
    assert rec["min_power"]["old"] == 5.0


async def test_helper_is_a_noop_on_empty_updates() -> None:
    entry = _entry()
    hass, manager = _hass()
    await ws_api._record_option_changes(hass, entry, {}, "test")
    manager.profile_store.async_record_settings_changes.assert_not_called()


async def test_helper_never_raises_without_a_manager() -> None:
    entry = _entry()
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    await ws_api._record_option_changes(hass, entry, {"min_power": 1.0}, "test")


async def test_every_option_writer_is_wired_to_the_helper() -> None:
    """Guard against a fifth writer being added without a changelog call.

    The count is the contract: ws_set_options records inline (it also handles the
    null-strip and title), the other four go through the helper.
    """
    import inspect

    src = inspect.getsource(ws_api)
    assert src.count("async_update_entry(") == 5, (
        "a new entry.options writer appeared; wire it to _record_option_changes"
    )
    assert src.count("await _record_option_changes(") == 4
