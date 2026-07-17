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
"""Tests for Group D backend features (D3 + D7).

D3 — Cycles pagination: the ``get_device_cycles`` WS command accepts an
     ``offset`` and returns ``total`` / ``has_more`` alongside the page.
D7 — Settings change history: ``ProfileStore.async_record_settings_changes`` /
     ``get_settings_changelog`` plus the ``ws_set_options`` diff logic and the
     new ``get_settings_changelog`` WS command.

Fast, pure-unit tests (no HA boot, no file I/O).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_NAME,
    CONF_POWER_SENSOR,
    CONF_START_THRESHOLD_W,
    DOMAIN,
)
from custom_components.ha_washdata.profile_store import ProfileStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> ProfileStore:
    """Return a real ProfileStore with storage save stubbed out."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
    ps.async_save = AsyncMock()
    return ps


def _make_cycles(n: int) -> list[dict]:
    """Oldest-first list of simple cycles (matches storage order)."""
    return [{"id": f"c{i}", "status": "completed"} for i in range(n)]


def _call_get_device_cycles(cycles: list[dict], *, limit=50, offset=0):
    """Invoke ws_get_device_cycles with a mock manager and return the result dict."""
    manager = MagicMock()
    manager.profile_store.get_past_cycles.return_value = cycles

    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}

    connection = MagicMock()
    msg = {"id": 1, "entry_id": "e1", "limit": limit, "offset": offset}
    ws_api.ws_get_device_cycles(hass, connection, msg)

    connection.send_result.assert_called_once()
    return connection.send_result.call_args[0][1]


# ---------------------------------------------------------------------------
# D3 — pagination
# ---------------------------------------------------------------------------

def test_d3_offset_zero_matches_legacy_behavior():
    """offset=0 returns the newest `limit` cycles, most-recent-first (legacy)."""
    cycles = _make_cycles(10)
    result = _call_get_device_cycles(cycles, limit=5, offset=0)

    # Legacy behaviour was reversed(raw[-limit:]).
    expected = [ws_api._strip_cycle(c) for c in reversed(cycles[-5:])]
    assert result["cycles"] == expected
    assert [c["id"] for c in result["cycles"]] == ["c9", "c8", "c7", "c6", "c5"]
    assert result["total"] == 10
    assert result["has_more"] is True
    assert result["entry_id"] == "e1"


def test_d3_offset_returns_next_window():
    """A non-zero offset pages further back into history."""
    cycles = _make_cycles(10)
    result = _call_get_device_cycles(cycles, limit=5, offset=5)

    assert [c["id"] for c in result["cycles"]] == ["c4", "c3", "c2", "c1", "c0"]
    assert result["total"] == 10
    assert result["has_more"] is False


def test_d3_partial_last_page_has_more_false():
    """A window that reaches the end reports has_more False even if < limit."""
    cycles = _make_cycles(7)
    result = _call_get_device_cycles(cycles, limit=5, offset=5)

    assert [c["id"] for c in result["cycles"]] == ["c1", "c0"]
    assert result["total"] == 7
    assert result["has_more"] is False


def test_d3_offset_beyond_end_returns_empty():
    """An offset past the last cycle returns an empty page with has_more False."""
    cycles = _make_cycles(10)
    result = _call_get_device_cycles(cycles, limit=5, offset=20)

    assert result["cycles"] == []
    assert result["total"] == 10
    assert result["has_more"] is False


def test_d3_has_more_true_when_full_page_and_more_remain():
    cycles = _make_cycles(100)
    result = _call_get_device_cycles(cycles, limit=50, offset=0)
    assert len(result["cycles"]) == 50
    assert result["has_more"] is True

    result2 = _call_get_device_cycles(cycles, limit=50, offset=50)
    assert len(result2["cycles"]) == 50
    assert result2["has_more"] is False


def test_d3_schema_accepts_offset():
    """The command schema declares offset with a default of 0 and min 0."""
    schema = ws_api.ws_get_device_cycles._ws_schema
    # Validating without offset applies the default.
    validated = schema(
        {"id": 1, "type": "ha_washdata/get_device_cycles", "entry_id": "e1"}
    )
    assert validated["offset"] == 0
    # Negative offsets are rejected.
    with pytest.raises(vol.Invalid):
        schema(
            {"id": 2, "type": "ha_washdata/get_device_cycles", "entry_id": "e1", "offset": -1}
        )


def test_d3_missing_manager_sends_error():
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    connection = MagicMock()
    msg = {"id": 1, "entry_id": "nope", "limit": 5, "offset": 0}
    ws_api.ws_get_device_cycles(hass, connection, msg)
    connection.send_error.assert_called_once()
    connection.send_result.assert_not_called()


# ---------------------------------------------------------------------------
# D7 — settings changelog store methods
# ---------------------------------------------------------------------------

async def test_d7_changelog_empty_by_default():
    store = _make_store()
    assert store.get_settings_changelog() == []


async def test_d7_record_appends_most_recent_first():
    store = _make_store()
    await store.async_record_settings_changes(
        [{"key": "a", "old": 1, "new": 2, "timestamp": "t0"}]
    )
    await store.async_record_settings_changes(
        [{"key": "b", "old": 3, "new": 4, "timestamp": "t1"}]
    )
    log = store.get_settings_changelog()
    assert [e["key"] for e in log] == ["b", "a"]  # newest first
    assert log[0]["old"] == 3 and log[0]["new"] == 4
    assert store.async_save.await_count == 2


async def test_d7_record_noop_on_empty():
    store = _make_store()
    await store.async_record_settings_changes([])
    assert store.get_settings_changelog() == []
    store.async_save.assert_not_awaited()


async def test_d7_record_skips_malformed_entries():
    store = _make_store()
    # No "key" -> skipped; the whole call becomes a no-op (no save).
    await store.async_record_settings_changes([{"old": 1, "new": 2}, "junk"])
    assert store.get_settings_changelog() == []
    store.async_save.assert_not_awaited()


async def test_d7_record_caps_at_50_newest_kept():
    store = _make_store()
    for i in range(60):
        await store.async_record_settings_changes(
            [{"key": f"k{i}", "old": i, "new": i + 1, "timestamp": f"t{i}"}]
        )
    log = store.get_settings_changelog()
    assert len(log) == 50
    # Newest kept, oldest 10 dropped.
    assert log[0]["key"] == "k59"
    assert log[-1]["key"] == "k10"
    assert "k9" not in {e["key"] for e in log}


async def test_d7_record_batch_caps_and_normalizes_timestamp():
    store = _make_store()
    batch = [{"key": f"k{i}", "old": i, "new": i + 1} for i in range(60)]
    await store.async_record_settings_changes(batch)
    log = store.get_settings_changelog()
    assert len(log) == 50
    # Missing timestamps are normalized to a real ISO string.
    assert all(isinstance(e["timestamp"], str) and e["timestamp"] for e in log)


# ---------------------------------------------------------------------------
# D7 — diff helper
# ---------------------------------------------------------------------------

def test_d7_diff_records_only_changed_submitted_keys():
    old = {"min_power": 5.0, "start_threshold_w": 10.0, "other": "x"}
    submitted = {"min_power": 5.0, "start_threshold_w": 15.0}
    changes = ws_api._diff_option_changes(old, submitted)
    assert len(changes) == 1
    c = changes[0]
    assert c["key"] == "start_threshold_w"
    assert c["old"] == 10.0 and c["new"] == 15.0
    assert isinstance(c["timestamp"], str) and c["timestamp"]


def test_d7_diff_records_first_set_from_none():
    old = {}
    submitted = {"min_power": 5.0}
    changes = ws_api._diff_option_changes(old, submitted)
    assert len(changes) == 1
    assert changes[0]["old"] is None and changes[0]["new"] == 5.0


def test_d7_diff_skips_none_to_none_and_name():
    old = {"door_sensor_entity": None}
    submitted = {"door_sensor_entity": None, CONF_NAME: "Renamed"}
    changes = ws_api._diff_option_changes(old, submitted)
    assert changes == []  # None->None unchanged; name is skip-listed


def test_d7_diff_coerces_nonserializable_values():
    class Weird:
        def __repr__(self) -> str:
            return "weird"

    old = {"k": Weird()}
    submitted = {"k": "clean"}
    changes = ws_api._diff_option_changes(old, submitted)
    assert changes[0]["old"] == "weird"  # coerced via str()
    assert changes[0]["new"] == "clean"


# ---------------------------------------------------------------------------
# D7 — ws_set_options integration + ws_get_settings_changelog command
# ---------------------------------------------------------------------------

async def test_d7_set_options_records_changed_keys():
    ws_fn = ws_api.ws_set_options.__wrapped__

    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {CONF_POWER_SENSOR: "sensor.power"}
    entry.options = {CONF_MIN_POWER: 5.0, CONF_START_THRESHOLD_W: 10.0}

    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()

    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}

    with patch.object(ws_api, "_get_entry", return_value=entry):
        connection = MagicMock()
        msg = {
            "id": 1,
            "entry_id": "e1",
            "options": {
                CONF_MIN_POWER: 5.0,          # unchanged -> not recorded
                CONF_START_THRESHOLD_W: 15.0,  # changed   -> recorded
                CONF_NAME: "New Name",         # skip-listed
            },
        }
        await ws_fn(hass, connection, msg)

    manager.profile_store.async_record_settings_changes.assert_awaited_once()
    recorded = manager.profile_store.async_record_settings_changes.await_args[0][0]
    assert {c["key"] for c in recorded} == {CONF_START_THRESHOLD_W}
    hass.config_entries.async_update_entry.assert_called_once()
    connection.send_result.assert_called_once()


async def test_d7_set_options_no_changes_no_record():
    ws_fn = ws_api.ws_set_options.__wrapped__

    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {CONF_POWER_SENSOR: "sensor.power"}
    entry.options = {CONF_MIN_POWER: 5.0}

    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()

    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}

    with patch.object(ws_api, "_get_entry", return_value=entry):
        connection = MagicMock()
        msg = {"id": 1, "entry_id": "e1", "options": {CONF_MIN_POWER: 5.0}}
        await ws_fn(hass, connection, msg)

    manager.profile_store.async_record_settings_changes.assert_not_awaited()
    hass.config_entries.async_update_entry.assert_called_once()


async def test_d7_set_options_changelog_failure_does_not_block_save():
    """A changelog write error must not prevent the options from being saved."""
    ws_fn = ws_api.ws_set_options.__wrapped__

    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {CONF_POWER_SENSOR: "sensor.power"}
    entry.options = {CONF_MIN_POWER: 5.0}

    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock(
        side_effect=RuntimeError("boom")
    )

    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}

    with patch.object(ws_api, "_get_entry", return_value=entry):
        connection = MagicMock()
        msg = {"id": 1, "entry_id": "e1", "options": {CONF_MIN_POWER: 9.0}}
        await ws_fn(hass, connection, msg)

    hass.config_entries.async_update_entry.assert_called_once()
    connection.send_result.assert_called_once()


async def test_d7_get_settings_changelog_command():
    ws_fn = ws_api.ws_get_settings_changelog.__wrapped__

    manager = MagicMock()
    manager.profile_store.get_settings_changelog.return_value = [
        {"key": "a", "old": 1, "new": 2, "timestamp": "t"}
    ]
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}

    connection = MagicMock()
    msg = {"id": 1, "entry_id": "e1"}
    await ws_fn(hass, connection, msg)

    connection.send_result.assert_called_once()
    payload = connection.send_result.call_args[0][1]
    assert payload["changelog"][0]["key"] == "a"


async def test_d7_get_settings_changelog_missing_manager():
    ws_fn = ws_api.ws_get_settings_changelog.__wrapped__
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    connection = MagicMock()
    msg = {"id": 1, "entry_id": "nope"}
    await ws_fn(hass, connection, msg)
    connection.send_error.assert_called_once()
    connection.send_result.assert_not_called()


def test_d7_command_registered():
    """The new command must be wired into the registration block."""
    import inspect

    src = inspect.getsource(ws_api.async_register_commands)
    assert "ws_get_settings_changelog" in src
