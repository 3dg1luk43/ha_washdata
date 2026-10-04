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
"""Audit TESTING-21: the WebSocket authorization matrix, behaviourally.

Every other WS test calls a handler directly or through ``__wrapped__``, so the
``_guard`` that ``async_register_commands`` wraps around each command never ran
in any test, and ``_rbac_ok`` was 3% covered. Three layers here:

* a role x RBAC x command-class x level matrix over ``_rbac_ok`` (pure);
* every command in HA's REAL websocket registry, after a real config-entry boot,
  refuses a caller with no user - which only holds if ``_guard`` wraps it;
* an end-to-end ``hass_ws_client`` session with a non-admin token.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from homeassistant.components import websocket_api

import custom_components.ha_washdata  # noqa: F401  (import before the loader: phcc ships its own custom_components)
from custom_components.ha_washdata import task_registry, ws_api
from custom_components.ha_washdata.const import DOMAIN

LEVELS = ("none", "read", "edit", "full")
RANK = {"none": 0, "read": 1, "edit": 2, "full": 3}

# One representative per command class, and the level a non-admin needs for it
# once RBAC is on. "admin" is never reachable by a non-admin; "open" always is.
CLASS_EXAMPLES = {
    "admin": ("wipe_history", "store_connect", "set_panel_config", "get_logs"),
    "open": ("get_constants", "get_panel_config", "set_user_prefs"),
    "read_write": ("set_program", "start_playground_history", "store_search_devices"),
    "get": ("get_profiles", "get_options", "get_cycle_power_data"),
    "edit": ("label_cycle", "set_options", "delete_cycle", "store_refresh_catalog"),
    "full": ("trigger_ml_training", "revert_ml_models", "set_lifetime_cycle_count"),
}
REQUIRED = {"open": "none", "read_write": "read", "get": "read", "edit": "edit", "full": "full"}

# Security-critical commands pinned by hand, independent of ws_api's own sets: a
# rename or a set edit that drops one of these from the admin gate must fail here.
MUST_BE_ADMIN_ONLY = {
    "set_panel_config", "get_logs", "wipe_history", "import_config", "export_config",
    "import_config_selective", "export_config_selective", "undo_import",
    "reprocess_history", "clear_debug_data", "store_connect", "store_disconnect",
    "store_upload_cycle", "store_upload_device", "history_import_begin",
    "apply_history_import",
}


def _hass(*, rbac: bool, level: str = "none", users: dict | None = None) -> SimpleNamespace:
    """`_rbac_ok` reads only ``hass.data``; anything else it touched would raise."""
    return SimpleNamespace(data={ws_api._PANEL_DATA_KEY: {"data": {"rbac": {  # noqa: SLF001
        "enabled": rbac, "default_level": level, "users": users or {},
    }}}})


def _conn(role: str) -> MagicMock:
    conn = MagicMock()
    conn.user = {
        "no_user": None,
        "admin": SimpleNamespace(id="admin", is_admin=True),
        "user": SimpleNamespace(id="u1", is_admin=False),
    }[role]
    return conn


def _check(hass, role: str, cmd: str, **msg) -> tuple[bool, str | None]:
    conn = _conn(role)
    ok = ws_api._rbac_ok(hass, conn, {"id": 7, "type": f"{DOMAIN}/{cmd}", **msg})  # noqa: SLF001
    if ok:
        conn.send_error.assert_not_called()
        return True, None
    conn.send_error.assert_called_once()
    assert conn.send_error.call_args.args[0] == 7
    return False, conn.send_error.call_args.args[1]


def _expected(role: str, rbac: bool, cls: str, level: str) -> tuple[bool, str | None]:
    if role == "no_user":
        return False, "unauthorized"
    if role == "admin":
        return True, None
    if cls == "admin":
        return False, "forbidden"
    if cls == "open" or not rbac:
        return True, None
    if RANK[level] >= RANK[REQUIRED[cls]]:
        return True, None
    return False, "forbidden"


@pytest.mark.parametrize("rbac", [False, True], ids=["rbac_off", "rbac_on"])
@pytest.mark.parametrize("role", ["no_user", "admin", "user"])
def test_rbac_matrix(role, rbac) -> None:
    """Every command class x access level for one role and RBAC setting."""
    wrong = []
    for (cls, cmds), level in itertools.product(CLASS_EXAMPLES.items(), LEVELS):
        hass = _hass(rbac=rbac, level=level)
        for cmd in cmds:
            got = _check(hass, role, cmd, entry_id="e1")
            want = _expected(role, rbac, cls, level)
            if got != want:
                wrong.append(f"{cls}/{cmd} at {level}: got {got}, want {want}")
    assert not wrong, "\n".join(wrong)


def test_the_class_examples_are_classified_as_the_matrix_assumes() -> None:
    """Guards the matrix itself: a moved command must move its row too."""
    for cmd in CLASS_EXAMPLES["admin"]:
        assert cmd in ws_api._ADMIN_COMMANDS  # noqa: SLF001
    for cmd in CLASS_EXAMPLES["open"]:
        assert cmd in ws_api._OPEN_COMMANDS  # noqa: SLF001
    for cmd in CLASS_EXAMPLES["read_write"]:
        assert cmd in ws_api._READ_WRITE_COMMANDS  # noqa: SLF001
    for cmd in CLASS_EXAMPLES["full"]:
        assert cmd in ws_api._FULL_COMMANDS and cmd not in ws_api._ADMIN_COMMANDS  # noqa: SLF001


def test_a_per_user_device_grant_overrides_the_default() -> None:
    users = {"u1": {"default": "none", "devices": {"e1": "edit"}}}
    hass = _hass(rbac=True, level="full", users=users)
    assert _check(hass, "user", "label_cycle", entry_id="e1") == (True, None)
    assert _check(hass, "user", "label_cycle", entry_id="e2") == (False, "forbidden")
    assert _check(hass, "user", "trigger_ml_training", entry_id="e1") == (False, "forbidden")


def test_security_critical_commands_are_admin_only_even_with_rbac_off() -> None:
    assert MUST_BE_ADMIN_ONLY <= ws_api._ADMIN_COMMANDS  # noqa: SLF001
    hass = _hass(rbac=False)
    for cmd in sorted(MUST_BE_ADMIN_ONLY):
        assert _check(hass, "user", cmd, entry_id="e1") == (False, "forbidden"), cmd


def test_every_rbac_set_names_a_real_command() -> None:
    """A renamed command silently drops out of its gate; catch the stale name."""
    declared = {
        cmd.split("/", 1)[1]
        for obj in vars(ws_api).values()
        if isinstance(cmd := getattr(obj, "_ws_command", None), str)
    }
    for name in ("_ADMIN_COMMANDS", "_OPEN_COMMANDS", "_FULL_COMMANDS",
                 "_READ_WRITE_COMMANDS", "_TASK_COMMANDS"):
        stale = getattr(ws_api, name) - declared
        assert not stale, f"ws_api.{name} names commands that do not exist: {sorted(stale)}"
    assert not ws_api._ADMIN_COMMANDS & ws_api._OPEN_COMMANDS  # noqa: SLF001


@pytest.mark.parametrize("level", LEVELS)
def test_task_commands_without_a_device_are_scoped_by_the_task(level) -> None:
    """Under RBAC a task id alone must not reach another device's task."""
    hass = _hass(rbac=True, level=level)
    task = task_registry.get_registry(hass).create("e1", "pg_history", "x")
    allowed = RANK[level] >= RANK["read"]
    assert _check(hass, "user", "subscribe_tasks") == (False, "forbidden")
    assert _check(hass, "user", "get_task_result", task_id="missing") == (False, "forbidden")
    assert _check(hass, "user", "get_task_result", task_id=task.id) == (
        (True, None) if allowed else (False, "forbidden")
    )
    # RBAC off: the task block is skipped, exactly as before RBAC existed.
    assert _check(_hass(rbac=False), "user", "subscribe_tasks") == (True, None)


# ─── The guard is wired on registration (real Home Assistant) ──────────────────


async def test_every_registered_command_is_guarded(hass, setup_washdata_entry) -> None:
    """Call each handler HA actually registered with a caller that has no user.

    The guard answers "unauthorized" before the handler body runs. A command
    registered without ``_guard`` would run its body instead (and answer with a
    result, a different error, or schedule work), so this fails for it.
    """
    entry = await setup_washdata_entry()
    registry = hass.data[websocket_api.DOMAIN]
    ours = sorted(cmd for cmd in registry if cmd.startswith(f"{DOMAIN}/"))
    assert len(ours) > 100

    unguarded = []
    for cmd in ours:
        handler, _schema = registry[cmd]
        conn = MagicMock()
        conn.user = None
        result = handler(hass, conn, {"id": 1, "type": cmd, "entry_id": entry.entry_id})
        if (
            result is not None
            or conn.send_error.call_args_list != [((1, "unauthorized", "No authenticated user"),)]
            or conn.send_result.called
            or conn.send_message.called
        ):
            unguarded.append(cmd)
    await hass.async_block_till_done()
    assert not unguarded, f"registered without the RBAC guard: {unguarded}"


async def test_a_non_admin_session_end_to_end(
    hass, setup_washdata_entry, hass_ws_client, hass_access_token, hass_read_only_access_token
) -> None:
    """A real WS session with a non-admin token, RBAC off and then on."""
    entry = await setup_washdata_entry()
    admin = await hass_ws_client(hass, hass_access_token)
    user = await hass_ws_client(hass, hass_read_only_access_token)

    async def _send(client, payload):
        await client.send_json_auto_id(payload)
        return await client.receive_json()

    # RBAC off (default): open and per-device commands work, admin-only ones do not.
    assert (await _send(user, {"type": f"{DOMAIN}/get_constants"}))["success"]
    assert (await _send(user, {"type": f"{DOMAIN}/get_options", "entry_id": entry.entry_id}))["success"]
    denied = await _send(user, {"type": f"{DOMAIN}/wipe_history", "entry_id": entry.entry_id})
    assert not denied["success"] and denied["error"]["code"] == "forbidden"

    # RBAC on, read by default: reads pass, edits are refused.
    on = await _send(admin, {"type": f"{DOMAIN}/set_panel_config",
                             "rbac": {"enabled": True, "default_level": "read"}})
    assert on["success"], on
    assert (await _send(user, {"type": f"{DOMAIN}/get_options", "entry_id": entry.entry_id}))["success"]
    edit = await _send(user, {"type": f"{DOMAIN}/label_cycle", "entry_id": entry.entry_id,
                              "cycle_id": "nope", "profile_name": None})
    assert not edit["success"] and edit["error"]["code"] == "forbidden"

    # RBAC on, no access by default: even a read is refused.
    assert (await _send(admin, {"type": f"{DOMAIN}/set_panel_config",
                                "rbac": {"enabled": True, "default_level": "none"}}))["success"]
    read = await _send(user, {"type": f"{DOMAIN}/get_options", "entry_id": entry.entry_id})
    assert not read["success"] and read["error"]["code"] == "forbidden"
    # The admin is never restricted.
    assert (await _send(admin, {"type": f"{DOMAIN}/get_options", "entry_id": entry.entry_id}))["success"]
