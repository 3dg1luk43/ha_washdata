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
"""Playground setting presets: named per-device sandbox snapshots.

Removed with the 0.5.8 UI removals and restored afterwards. A preset stores only
keys the Playground offers today; a preset saved before 0.5.8 can still carry the
Stage 2-4 matcher weights, which are filtered out when the list is read instead
of failing or being deleted.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import playground, ws_api
from custom_components.ha_washdata.const import (
    PLAYGROUND_PRESET_MAX,
    PLAYGROUND_PRESET_NAME_MAX,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import ProfileStore, WashDataStore

# A record as a pre-0.5.8 build wrote it: real options plus the matcher weights
# the Playground no longer offers.
_LEGACY_RECORD = {
    "values": {
        "off_delay": 300,
        "min_off_gap": 240,
        "profile_match_max_duration_ratio": 1.6,
        "corr_weight": 0.6,
        "dtw_bandwidth": 0.25,
        "duration_weight": 0.3,
    },
    "created_at": "2026-08-01T10:00:00+00:00",
    "updated_at": "2026-08-02T10:00:00+00:00",
}


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e1", min_duration_ratio=0.0, max_duration_ratio=2.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        yield ps


# ─── Store ────────────────────────────────────────────────────────────────────


async def test_save_load_overwrite_and_delete(store):
    """A preset round-trips through the store and never touches live config."""
    assert store.get_playground_presets() == {}

    rec = await store.async_save_playground_preset("Quiet nights", {"off_delay": 300})
    assert rec["values"] == {"off_delay": 300}
    assert rec["created_at"] and rec["updated_at"]
    assert list(store.get_playground_presets()) == ["Quiet nights"]

    # Overwriting keeps the original creation timestamp.
    again = await store.async_save_playground_preset("Quiet nights", {"off_delay": 420})
    assert again["created_at"] == rec["created_at"]
    assert store.get_playground_presets()["Quiet nights"]["values"] == {"off_delay": 420}

    assert await store.async_delete_playground_preset("Quiet nights") is True
    assert store.get_playground_presets() == {}
    # Deleting a name that is gone is a no-op, not an error.
    assert await store.async_delete_playground_preset("Quiet nights") is False


async def test_rejects_empty_input_and_enforces_cap(store):
    with pytest.raises(ValueError):
        await store.async_save_playground_preset("   ", {"off_delay": 300})
    with pytest.raises(ValueError):
        await store.async_save_playground_preset("Empty", {})

    for i in range(PLAYGROUND_PRESET_MAX):
        await store.async_save_playground_preset(f"p{i}", {"off_delay": 100 + i})
    with pytest.raises(ValueError):
        await store.async_save_playground_preset("one too many", {"off_delay": 999})
    # An existing name may still be overwritten at the cap.
    await store.async_save_playground_preset("p0", {"off_delay": 111})
    assert len(store.get_playground_presets()) == PLAYGROUND_PRESET_MAX


async def test_overlong_name_is_deletable_and_has_no_trailing_space(store):
    """Save and delete derive the key the same way, clamp first, strip after."""
    long_name = "x" * (PLAYGROUND_PRESET_NAME_MAX + 25)
    await store.async_save_playground_preset(long_name, {"off_delay": 300})
    assert [len(k) for k in store.get_playground_presets()] == [PLAYGROUND_PRESET_NAME_MAX]
    assert await store.async_delete_playground_preset(long_name) is True

    name = "y" * (PLAYGROUND_PRESET_NAME_MAX - 1) + "   tail"
    await store.async_save_playground_preset(name, {"off_delay": 300})
    key = next(iter(store.get_playground_presets()))
    assert key == key.strip()
    assert await store.async_delete_playground_preset(name) is True


async def test_corrupt_store_key_self_heals(store):
    store._data["playground_presets"] = ["not", "a", "dict"]
    assert store.get_playground_presets() == {}
    await store.async_save_playground_preset("ok", {"off_delay": 300})
    assert list(store.get_playground_presets()) == ["ok"]


async def test_wipe_all_data_clears_presets(store):
    await store.async_save_playground_preset("ok", {"off_delay": 300})
    await store.clear_all_data()
    assert store.get_playground_presets() == {}


async def test_load_keeps_presets_stored_by_an_older_build(store):
    """The 0.5.8 build that hid presets left them in storage; they load as they are."""
    store._store.async_load = AsyncMock(return_value={
        "profiles": {}, "past_cycles": [],
        "playground_presets": {"Legacy": dict(_LEGACY_RECORD)},
    })
    await store.async_load()
    assert store.get_playground_presets()["Legacy"] == _LEGACY_RECORD


@pytest.mark.parametrize("old_version", [9, 12, 15, STORAGE_VERSION - 1, STORAGE_VERSION])
async def test_storage_migration_keeps_presets(old_version):
    """No storage migration step may drop the presets key or rewrite a record."""
    hass = MagicMock()
    hass.config.path = lambda *a: "/tmp/" + "/".join(a)
    wds = WashDataStore(hass, STORAGE_VERSION, f"{STORAGE_KEY}.test")
    presets = {"Legacy": dict(_LEGACY_RECORD), "Night": {"values": {"off_delay": 90}}}
    data = {"profiles": {}, "past_cycles": [], "playground_presets": presets}
    out = await wds._async_migrate_func(old_version, 1, data)  # noqa: SLF001
    assert out["playground_presets"] == {
        "Legacy": _LEGACY_RECORD, "Night": {"values": {"off_delay": 90}},
    }


# ─── WebSocket ────────────────────────────────────────────────────────────────


def _manager(store):
    manager = MagicMock()
    manager.profile_store = store
    manager.learning_manager = SimpleNamespace(suggestion_engine=None)
    manager.detector = SimpleNamespace(
        config=CycleDetectorConfig(min_power=2.0, off_delay=300, device_type="washing_machine")
    )
    manager._resolve_energy_price = MagicMock(return_value=None)
    return manager


async def _call(handler, manager, mock_hass, msg):
    conn = MagicMock()
    entry = SimpleNamespace(entry_id="e1", options={"device_type": "washing_machine"}, data={})
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await handler.__wrapped__(mock_hass, conn, {"id": 1, "entry_id": "e1", **msg})
    return conn


def _result(conn):
    conn.send_error.assert_not_called()
    return conn.send_result.call_args[0][1]


async def test_ws_save_keeps_only_real_options(store, mock_hass):
    conn = await _call(ws_api.ws_save_playground_preset, _manager(store), mock_hass, {
        "name": "  Night  ",
        "values": {
            "off_delay": "222",          # the panel's inputs yield strings
            "anti_wrinkle_enabled": True,
            "corr_weight": 0.6,          # removed matcher weight
            "min_off_gap": -5,           # a physical quantity cannot be negative
            "totally_unknown": 1,
        },
    })
    out = _result(conn)
    assert ws_api._validate_ws_contract("save_playground_preset", out) == []  # noqa: SLF001
    assert out["success"] is True
    assert out["presets"][0]["name"] == "Night"
    assert out["presets"][0]["values"] == {"off_delay": 222, "anti_wrinkle_enabled": True}
    assert store.get_playground_presets()["Night"]["values"] == {
        "off_delay": 222, "anti_wrinkle_enabled": True,
    }


async def test_ws_save_with_nothing_storable_is_an_error(store, mock_hass):
    conn = await _call(ws_api.ws_save_playground_preset, _manager(store), mock_hass, {
        "name": "Only weights", "values": {"corr_weight": 0.6},
    })
    conn.send_result.assert_not_called()
    assert conn.send_error.call_args[0][1] == "invalid_format"
    assert store.get_playground_presets() == {}


async def test_ws_save_reports_a_failed_write(store, mock_hass):
    store._store.async_save = AsyncMock(side_effect=OSError("disk full"))
    conn = await _call(ws_api.ws_save_playground_preset, _manager(store), mock_hass, {
        "name": "x", "values": {"off_delay": 100},
    })
    assert conn.send_error.call_args[0][1] == "unknown_error"


async def test_ws_delete_returns_the_remaining_list(store, mock_hass):
    await store.async_save_playground_preset("a", {"off_delay": 100})
    await store.async_save_playground_preset("b", {"off_delay": 200})
    conn = await _call(ws_api.ws_delete_playground_preset, _manager(store), mock_hass, {"name": "a"})
    out = _result(conn)
    assert ws_api._validate_ws_contract("delete_playground_preset", out) == []  # noqa: SLF001
    assert out == {"success": True, "presets": [
        {"name": "b", "values": {"off_delay": 200},
         "created_at": out["presets"][0]["created_at"],
         "updated_at": out["presets"][0]["updated_at"]},
    ]}
    conn = await _call(ws_api.ws_delete_playground_preset, _manager(store), mock_hass, {"name": "a"})
    assert _result(conn)["success"] is False


async def test_ws_unknown_device_is_not_found(mock_hass):
    conn = MagicMock()
    with patch.object(ws_api, "_get_manager", return_value=None):
        await ws_api.ws_save_playground_preset.__wrapped__(
            mock_hass, conn, {"id": 1, "entry_id": "x", "name": "n", "values": {"off_delay": 1}}
        )
    assert conn.send_error.call_args[0][1] == "not_found"


async def test_settings_list_presets_sorted_and_filtered_to_current_keys(store, mock_hass):
    """A pre-0.5.8 preset still loads: removed keys are dropped from the view only."""
    store._data["playground_presets"] = {
        "zeta": {"values": {"off_delay": 60}},
        "Legacy": dict(_LEGACY_RECORD),
        "broken": "not a record",
    }
    conn = await _call(ws_api.ws_get_playground_settings, _manager(store), mock_hass,
                       {"include_suggestions": False})
    out = _result(conn)
    assert ws_api._validate_ws_contract("get_playground_settings", out) == []  # noqa: SLF001
    assert out["preset_limit"] == PLAYGROUND_PRESET_MAX
    assert [p["name"] for p in out["presets"]] == ["Legacy", "zeta"]
    legacy = out["presets"][0]
    assert legacy["values"] == {
        "off_delay": 300, "min_off_gap": 240, "profile_match_max_duration_ratio": 1.6,
    }
    assert set(legacy["values"]) <= playground.SETTING_KEYS
    assert legacy["created_at"] == _LEGACY_RECORD["created_at"]
    # The stored record itself is untouched.
    assert store.get_playground_presets()["Legacy"] == _LEGACY_RECORD


# ─── Wiring / access ──────────────────────────────────────────────────────────


def _rbac_hass(level: str) -> MagicMock:
    hass = MagicMock()
    hass.data = {ws_api._PANEL_DATA_KEY: {"data": {"rbac": {  # noqa: SLF001
        "enabled": True, "default_level": level, "users": {},
    }}}}
    return hass


def _allowed(hass, cmd: str) -> bool:
    conn = MagicMock()
    conn.user = SimpleNamespace(id="u1", is_admin=False)
    return ws_api._rbac_ok(hass, conn, {"id": 1, "type": f"ha_washdata/{cmd}", "entry_id": "e1"})  # noqa: SLF001


@pytest.mark.parametrize("cmd", ["save_playground_preset", "delete_playground_preset"])
def test_preset_writes_need_edit_access(cmd):
    """Presets are stored data: a read user can load them but not change them."""
    assert not _allowed(_rbac_hass("read"), cmd)
    assert _allowed(_rbac_hass("edit"), cmd)
    assert _allowed(_rbac_hass("read"), "get_playground_settings")
    assert cmd not in ws_api._READ_WRITE_COMMANDS  # noqa: SLF001
    assert cmd not in ws_api._ADMIN_COMMANDS  # noqa: SLF001


def test_preset_commands_are_registered():
    registered = []
    with patch.object(ws_api.websocket_api, "async_register_command",
                      side_effect=lambda _h, handler: registered.append(handler._ws_command)):
        ws_api.async_register_commands(MagicMock())
    assert "ha_washdata/save_playground_preset" in registered
    assert "ha_washdata/delete_playground_preset" in registered
