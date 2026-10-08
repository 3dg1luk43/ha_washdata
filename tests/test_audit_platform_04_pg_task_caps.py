"""Audit PLATFORM-04 (+ PLAYGROUND-12, PLAYGROUND-18 backend halves).

Nothing stopped N concurrent Playground history/sweep tasks per device - each is
minutes of executor CPU on a Pi - and a superseded Simulate kept replaying to the
end. Now a second batch run (Test on history or Optimize) on a busy device is
refused with ``task_busy``, a new Simulate cancels the one it supersedes, and
"Last N" reaches the backend as a count capped at ``MAX_BATCH_CYCLES``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import voluptuous as vol

from custom_components.ha_washdata import playground, task_registry, ws_api

sys.path.insert(0, str(Path(__file__).parent))
from test_ws_history_import import _conn, _entry, _hass, _manager  # noqa: E402


def _ctx(hass):
    manager, entry = _manager(hass), _entry()
    return (
        patch.object(ws_api, "_get_manager", return_value=manager),
        patch.object(ws_api, "_get_entry", return_value=entry),
        # The tasks themselves are not under test here: keep them parked.
        patch.object(hass, "async_create_task", lambda coro, *a: coro.close()),
    )


def _start(handler, hass, conn, msg_id, **extra):
    handler(hass, conn, {"id": msg_id, "entry_id": "e", **extra})


@pytest.mark.parametrize("first, second", [
    ("history", "history"), ("history", "sweep"), ("sweep", "history"), ("sweep", "sweep"),
])
def test_a_second_batch_run_on_a_busy_device_is_refused(first, second):
    hass, conn = _hass(), _conn()
    handlers = {
        "history": (ws_api.ws_start_playground_history, {"cycle_ids": [], "settings_override": {}}),
        "sweep": (ws_api.ws_start_playground_sweep, {"param": "off_delay", "values": [60.0], "objective": "end_lag"}),
    }
    a, b, c = _ctx(hass)
    with a, b, c:
        _start(handlers[first][0], hass, conn, 1, **handlers[first][1])
        assert conn.send_result.called and not conn.send_error.called
        _start(handlers[second][0], hass, conn, 2, **handlers[second][1])
    assert conn.send_error.call_args.args[:2] == (2, "task_busy")
    reg = task_registry.get_registry(hass)
    assert len([t for t in reg.snapshot("e") if t["state"] == "running"]) == 1


def test_a_finished_run_frees_the_device_and_other_devices_are_independent():
    hass, conn = _hass(), _conn()
    a, b, c = _ctx(hass)
    with a, b, c:
        _start(ws_api.ws_start_playground_history, hass, conn, 1, cycle_ids=[], settings_override={})
        reg = task_registry.get_registry(hass)
        first = reg.get(conn.send_result.call_args.args[1]["task_id"])
        reg.finish(first)
        _start(ws_api.ws_start_playground_history, hass, conn, 2, cycle_ids=[], settings_override={})
        assert not conn.send_error.called
        # Another entry's run does not count against this one.
        other = reg.create("other", "pg_sweep", "x")
        assert other.state == task_registry.STATE_RUNNING
        reg.finish(reg.get(conn.send_result.call_args.args[1]["task_id"]))
        _start(ws_api.ws_start_playground_history, hass, conn, 3, cycle_ids=[], settings_override={})
        assert not conn.send_error.called


def test_a_new_simulate_cancels_the_one_it_supersedes():
    hass, conn = _hass(), _conn()
    a, b, c = _ctx(hass)
    with a, b, c:
        _start(ws_api.ws_start_playground_cycle_detail, hass, conn, 1, cycle_id="c1", settings_override={})
        reg = task_registry.get_registry(hass)
        first = reg.get(conn.send_result.call_args.args[1]["task_id"])
        _start(ws_api.ws_start_playground_cycle_detail, hass, conn, 2, cycle_id="c2", settings_override={})
        second = reg.get(conn.send_result.call_args.args[1]["task_id"])
    assert first.cancel_requested is True
    assert second.cancel_requested is False
    assert not conn.send_error.called


# ─── PLAYGROUND-18: "Last N" ──────────────────────────────────────────────────


class _Past:
    def __init__(self, n):
        self.cycles = [{"id": f"c{i:03d}"} for i in range(n)]

    def get_past_cycles(self):
        return self.cycles


def test_last_n_is_the_n_most_recent_newest_first_and_capped():
    store = _Past(80)
    picked = playground._select_cycles(store, None, 30)  # noqa: SLF001
    assert [c["id"] for c in picked[:2]] == ["c079", "c078"] and len(picked) == 30
    assert len(playground._select_cycles(store, None, 500)) == playground.MAX_BATCH_CYCLES  # noqa: SLF001
    # No count, no ids: the most recent 20, as before.
    assert len(playground._select_cycles(store, None)) == playground.DEFAULT_RECENT_CYCLES  # noqa: SLF001
    # Explicit ids win.
    assert [c["id"] for c in playground._select_cycles(store, ["c001", "zzz"], 30)] == ["c001"]  # noqa: SLF001


@pytest.mark.parametrize("command", [
    "ha_washdata/start_playground_history", "ha_washdata/start_playground_sweep",
])
def test_the_commands_take_a_count_up_to_the_batch_cap(command):
    handler = {
        "ha_washdata/start_playground_history": ws_api.ws_start_playground_history,
        "ha_washdata/start_playground_sweep": ws_api.ws_start_playground_sweep,
    }[command]
    schema = handler._ws_schema  # noqa: SLF001
    base = {"type": command, "id": 1, "entry_id": "e"}
    if command.endswith("sweep"):
        base.update(param="off_delay", values=[60.0], objective="end_lag")
    assert schema({**base, "count": 50})["count"] == 50
    with pytest.raises(vol.Invalid):
        schema({**base, "count": 51})
