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
    # (The reporter's third key, smoothing_window, is no longer suggested - it sizes
    # a buffer nothing reads, audit SUGGEST-12 - so min_off_gap stands in for it.)
    entry = _entry({"off_delay": 1780, "min_power": 1.2, "min_off_gap": 2000})
    hass, manager = _hass()
    manager.profile_store.get_suggestions.return_value = {
        "off_delay": {"value": 1327},
        "min_power": {"value": 1.0},
        "min_off_gap": {"value": 2400},
    }

    ws_fn = ws_api.ws_apply_suggestions.__wrapped__
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_fn(
            hass,
            MagicMock(),
            {
                "id": 1,
                "entry_id": "e1",
                "keys": ["off_delay", "min_power", "min_off_gap"],
            },
        )

    rec = _recorded(manager)
    assert set(rec) == {"off_delay", "min_power", "min_off_gap"}
    # old/new must both be present, or "revert to previous value" has no target.
    assert rec["off_delay"]["old"] == 1780 and rec["off_delay"]["new"] == 1327
    assert rec["min_power"]["old"] == 1.2 and rec["min_power"]["new"] == 1.0
    assert rec["min_off_gap"]["old"] == 2000 and rec["min_off_gap"]["new"] == 2400
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


async def _drive_set_options(hass, manager, entry):
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_set_options.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "options": {"min_power": 3.5}}
        )


async def _drive_apply_suggestions(hass, manager, entry):
    manager.profile_store.get_suggestions.return_value = {"min_power": {"value": 3.5}}
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_apply_suggestions.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "keys": ["min_power"]}
        )


async def _drive_import(hass, manager, entry):
    manager.profile_store.async_import_data = AsyncMock(
        return_value={"entry_options": {"min_power": 3.5}}
    )
    with patch.object(ws_api, "_get_manager", return_value=manager), \
            patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_import_config.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "json_data": "{}"}
        )


async def _drive_selective_import(hass, manager, entry):
    manager.profile_store.async_import_data_selective = AsyncMock(
        return_value={"settings": {"min_power": 3.5}}
    )
    manager.device_type = "washing_machine"
    with patch.object(ws_api, "_get_manager", return_value=manager), \
            patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_import_config_selective.__wrapped__(
            hass, MagicMock(),
            {"id": 1, "entry_id": "e1", "json_data": "{}", "selection": {},
             "mode": "merge", "conflict_resolutions": {},
             "cycle_destination": "reference", "apply_settings": True},
        )


async def _drive_undo_import(hass, manager, entry):
    """Register item 195: the undo puts the pre-import options back."""
    manager.profile_store.async_restore_pre_import_snapshot = AsyncMock(
        return_value={"restored_from": None, "counts": {},
                      "entry_options": {"min_power": 3.5}}
    )
    with patch.object(ws_api, "_get_manager", return_value=manager), \
            patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_undo_import.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1"}
        )


async def _drive_store_download(hass, manager, entry):
    with patch.object(ws_api, "_get_entry", return_value=entry):
        applied = await ws_api._apply_store_settings(
            hass, "e1", {"settings": {"min_power": 3.5}}, True
        )
    assert applied == 1


@pytest.mark.parametrize(
    "drive",
    [
        _drive_set_options,
        _drive_apply_suggestions,
        _drive_import,
        _drive_selective_import,
        _drive_undo_import,
        _drive_store_download,
    ],
    ids=["set_options", "apply_suggestions", "import", "selective_import", "undo_import",
         "store_download"],
)
async def test_every_option_writer_records_before_it_writes(drive) -> None:
    """Each of the six `entry.options` writers, driven: the change is recorded
    (old and new) BEFORE `async_update_entry` schedules the reload that rebuilds
    the store. Two of them (store download, selective import) had no test of it
    until this replaced a source-text count (audit TESTING-13)."""
    entry = _entry({"min_power": 1.2})
    hass, manager = _hass()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    order: list[str] = []
    manager.profile_store.async_record_settings_changes = AsyncMock(
        side_effect=lambda ch: order.append("record")
    )
    hass.config_entries.async_update_entry = MagicMock(
        side_effect=lambda *a, **k: order.append("update")
    )

    await drive(hass, manager, entry)

    assert "update" in order, "the writer did not write"
    assert "record" in order, "the write was not recorded in the changelog"
    assert order.index("record") < order.index("update"), order
    rec = _recorded(manager)
    assert rec["min_power"]["old"] == 1.2
    assert rec["min_power"]["new"] == 3.5


async def test_no_seventh_option_writer_bypasses_the_helper() -> None:
    """Structural lint, kept on purpose: the behaviour test above covers the six
    known writers, but a NEW writer cannot be found by driving known handlers.

    The count is the contract, and since the PR #448 round-15 review it is a
    clean one: ALL six writers go through `_record_option_changes` (the sixth is
    the import undo, register item 195).
    """
    import inspect

    src = inspect.getsource(ws_api)
    assert src.count("async_update_entry(") == 6, (
        "a new entry.options writer appeared; wire it to _record_option_changes "
        "and add it to test_every_option_writer_records_before_it_writes"
    )
    assert src.count("await _record_option_changes(") == 6, (
        "every option writer records through the helper; no inline copies"
    )
