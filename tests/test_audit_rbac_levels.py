"""Audit 2026-10-02 PLATFORM-11: RBAC levels match what the commands do.

A read user could cancel an admin's history import or reprocess, and any non-admin
could publish to the community store under the admin's install-wide account.
(set_program stays read-level by the maintainer's decision.)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from custom_components.ha_washdata import task_registry, ws_api


def _hass(level: str = "read") -> MagicMock:
    hass = MagicMock()
    hass.data = {ws_api._PANEL_DATA_KEY: {"data": {"rbac": {  # noqa: SLF001
        "enabled": True, "default_level": level, "users": {},
    }}}}
    return hass


def _ok(hass, cmd: str, **msg) -> bool:
    conn = MagicMock()
    conn.user = SimpleNamespace(id="u1", is_admin=False)
    return ws_api._rbac_ok(hass, conn, {"id": 1, "type": f"ha_washdata/{cmd}", **msg})  # noqa: SLF001


def test_picking_the_program_stays_open_to_read_users() -> None:
    """A deliberate exception: the maintainer keeps the program pick at read."""
    assert _ok(_hass("read"), "set_program", entry_id="e1")


def test_cancelling_a_task_needs_the_level_that_started_it() -> None:
    hass = _hass("read")
    reg = task_registry.get_registry(hass)
    pg = reg.create("e1", "pg_history", "x")
    imp = reg.create("e1", "history_import", "x")
    lab = reg.create("e1", "auto_label", "x")
    assert _ok(hass, "cancel_task", task_id=pg.id)       # a read user's own what-if
    assert not _ok(hass, "cancel_task", task_id=imp.id)  # an admin's import
    assert not _ok(hass, "cancel_task", task_id=lab.id)  # an editor's job
    editor = _hass("edit")
    job = task_registry.get_registry(editor).create("e1", "auto_label", "y")
    assert _ok(editor, "cancel_task", task_id=job.id)


def test_store_publishing_is_admin_only_even_with_rbac_off() -> None:
    hass = MagicMock()
    hass.data = {}
    for cmd in ("store_upload_cycle", "store_upload_device", "store_rate_device",
                "store_confirm_device"):
        assert not _ok(hass, cmd, entry_id="e1")
