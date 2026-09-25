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
"""PR #448 review round 27: an import re-armed the banked-tail repair and then
nothing ran it.

Both import paths set ``BANKED_TAIL_REPAIR_KEY`` for a payload old enough to
carry banked tails, because an import bypasses ``_async_migrate_func``. Only
``manager.async_setup`` ever spawned the repair, and an import reloads the entry
only when the payload brings options with it - ``async_update_entry`` is not
called for a cycles-only import, nor for a selective one with
``apply_settings=False``. So the imported tails kept feeding ``avg_duration``,
the ETA and Smart Termination until the user next restarted Home Assistant.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import BANKED_TAIL_REPAIR_KEY


class _Mgr:
    """The scheduler in isolation: it touches only the store and _spawn_tracked."""

    def __init__(self, pending: bool):
        from custom_components.ha_washdata.manager import WashDataManager

        self.profile_store = MagicMock()
        self.profile_store.banked_tail_repair_pending.return_value = pending
        self._banked_tail_repair_task = None
        self.spawned: list = []
        self._schedule = WashDataManager.async_schedule_banked_tail_repair.__get__(self)
        self._repair_calls = 0

    def _spawn_tracked(self, coro):
        coro.close()  # never awaited here; keep the loop clean
        self.spawned.append(coro)
        task = MagicMock()
        task.done.return_value = False
        return task

    async def _async_repair_banked_tails(self):
        self._repair_calls += 1


def test_an_armed_marker_schedules_the_repair() -> None:
    mgr = _Mgr(pending=True)
    mgr._schedule()
    assert len(mgr.spawned) == 1


def test_a_clear_marker_schedules_nothing() -> None:
    """The marker is the whole test, so the common case costs one dict lookup."""
    mgr = _Mgr(pending=False)
    mgr._schedule()
    assert mgr.spawned == []


def test_a_second_import_does_not_start_a_second_walk() -> None:
    """The repair clears the marker only at the END of its run, so the pending
    check alone would let two imports in quick succession each start a full walk
    of the history and rebuild the same envelopes concurrently."""
    mgr = _Mgr(pending=True)
    mgr._schedule()
    mgr._schedule()
    assert len(mgr.spawned) == 1


def test_a_finished_run_does_not_block_a_later_import() -> None:
    """The in-flight guard must not become a one-shot latch: a later import that
    re-arms the marker has to be able to schedule again."""
    mgr = _Mgr(pending=True)
    mgr._schedule()
    mgr._banked_tail_repair_task.done.return_value = True
    mgr._schedule()
    assert len(mgr.spawned) == 2


@pytest.mark.parametrize(
    "module, handler",
    [
        ("ws_api", "ws_import_config"),
        ("ws_api", "ws_import_config_selective"),
    ],
)
def test_both_ws_import_handlers_schedule_the_repair(module: str, handler: str) -> None:
    """Asserted on the source: driving these needs a live connection, a manager
    and the per-entry write lock. What matters is that neither handler relies on
    the options write below it to reload the entry."""
    import importlib
    import inspect

    mod = importlib.import_module(f"custom_components.ha_washdata.{module}")
    src = inspect.getsource(getattr(mod, handler))
    assert "async_schedule_banked_tail_repair()" in src


def test_the_import_config_service_schedules_the_repair() -> None:
    """The legacy `import_config` service reads a file straight into
    `async_import_data`, and writes the entry only when the payload carried
    settings."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "__init__.py"
    ).read_text()
    assert "manager.async_schedule_banked_tail_repair()" in src


def test_an_old_selective_import_arms_the_marker() -> None:
    """The half this test file exists to connect to: the store really does set
    the marker on an old payload, so the scheduling above is not hypothetical."""
    from custom_components.ha_washdata.profile_store import (
        _export_predates_banked_tail_repair,
    )

    assert _export_predates_banked_tail_repair({"version": 12}) is True
    assert _export_predates_banked_tail_repair({"version": 13}) is False
