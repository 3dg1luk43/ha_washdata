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
"""Register item 513 (discussion #463): recorder context around a stored cycle.

The cycle chart opens at the first reading over the start threshold, so a cycle
looks cut off and the user cannot see what preceded it. ``get_cycle_context``
returns the power sensor's recorder history for a window before the stored start
and after the trace end, as offsets on the chart's own axis, for the panel to draw
greyed. It is display only: it reads, never writes, and is refused for cycles this
plug did not measure (community-store references, imported history).
"""
from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api

START = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)
TRACE = [(0.0, 112.0), (60.0, 350.0), (1800.0, 700.0), (3600.0, 1.0)]


def _cycle(**extra):
    return {"id": "c1", "start_time": START.isoformat(), "duration": 3600.0,
            "power_data": [[o, p] for o, p in TRACE], **extra}


def _manager(cycle, origin="past", sensor="sensor.washer_power"):
    store = MagicMock()
    store.find_stored_cycle.return_value = (cycle, origin if cycle else "")
    store.get_cycle_power_data.return_value = list(TRACE) if cycle else []
    return SimpleNamespace(profile_store=store, power_sensor_entity_id=sensor)


class _Recorder:
    """Stands in for ``_recorder_power``: serves rows from one synthetic history."""

    def __init__(self, rows):
        self.rows = rows  # [(unix_ts, watts | None)]
        self.calls = []

    async def __call__(self, hass, entity_id, start_dt, *, end_dt=None, keep_unavailable=False):
        self.calls.append((entity_id, start_dt, end_dt, keep_unavailable))
        lo = start_dt.timestamp()
        hi = end_dt.timestamp()
        out = [(max(lo, ts), w) for ts, w in self.rows if ts <= hi]
        # include_start_time_state: only the newest row at/before the window start
        # survives, clamped to it, as the real helper does.
        before = [r for r in out if r[0] == lo]
        return before[-1:] + [r for r in out if r[0] > lo]


def _history():
    s = START.timestamp()
    return [
        (s - 7200, 0.2),          # standby long before: the state in force
        (s - 400, None),          # an unavailable row
        (s - 300, 0.4),
        (s - 95, 9.5),            # a door-lock blip below the start threshold
        (s - 60, 0.6),
        (s, 112.0),               # the stored start reading: not context
        (s + 1800, 700.0),        # inside the cycle: never queried
        (s + 3600, 1.0),          # the trace end
        (s + 3700, 0.3),
        (s + 9000, 0.0),          # outside the after window
    ]


async def _call(manager, recorder, now=None, **params):
    sent = {}

    def _capture(_conn, _mid, command, data):
        sent["command"], sent["data"] = command, data

    msg = {"id": 1, "type": "ha_washdata/get_cycle_context", "entry_id": "e1", "cycle_id": "c1",
           "before_s": 600.0, "after_s": 600.0, **params}
    now = now or START + timedelta(days=1)
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_recorder_power", new=recorder), \
         patch.object(ws_api, "_send_result", side_effect=_capture), \
         patch.object(ws_api.dt_util, "utcnow", return_value=now):
        # hass only reaches the two patched helpers: a bare namespace, not a
        # MagicMock that would accept anything (audit TESTING-07).
        await ws_api.ws_get_cycle_context.__wrapped__(SimpleNamespace(), MagicMock(), msg)
    assert sent["command"] == "get_cycle_context"
    return sent["data"]


@pytest.mark.asyncio
async def test_windows_either_side_of_the_trace_on_the_chart_axis():
    recorder = _Recorder(_history())
    out = await _call(_manager(_cycle()), recorder)

    assert ws_api._validate_ws_contract("get_cycle_context", out) == []
    assert out["available"] is True and out["reason"] is None
    assert out["entity_id"] == "sensor.washer_power"
    # Two bounded queries: [start - 600, start] and [trace end, trace end + 600].
    (_e, b_lo, b_hi, b_keep), (_e2, a_lo, a_hi, a_keep) = recorder.calls
    assert (b_lo, b_hi) == (START - timedelta(seconds=600), START)
    assert (a_lo, a_hi) == (START + timedelta(seconds=3600), START + timedelta(seconds=4200))
    assert b_keep and a_keep
    # Offsets from the stored start; nothing at or after the start in `before`.
    assert out["before"] == [[-600.0, 0.2], [-400.0, None], [-300.0, 0.4], [-95.0, 9.5], [-60.0, 0.6]]
    assert out["after"] == [[3600.0, 1.0], [3700.0, 0.3]]
    assert out["trace_end_s"] == 3600.0 and out["after_end_s"] == 4200.0


@pytest.mark.asyncio
async def test_after_window_stops_at_now():
    out = await _call(_manager(_cycle()), _Recorder(_history()),
                      now=START + timedelta(seconds=3720))
    assert out["after_end_s"] == 3720.0
    assert out["after"] == [[3600.0, 1.0], [3700.0, 0.3]]


@pytest.mark.parametrize("origin", ["reference", "backfill"])
@pytest.mark.asyncio
async def test_a_cycle_this_plug_did_not_measure_is_not_read(origin):
    recorder = _Recorder(_history())
    out = await _call(_manager(_cycle(), origin=origin), recorder)
    assert out["available"] is False and out["reason"] == "not_live"
    assert recorder.calls == []
    assert out["before"] == [] and out["after"] == []


@pytest.mark.asyncio
async def test_unknown_cycle_no_sensor_and_empty_recorder():
    recorder = _Recorder(_history())
    assert (await _call(_manager(None), recorder))["reason"] == "not_found"
    assert (await _call(_manager(_cycle(), sensor=None), recorder))["reason"] == "no_sensor"
    assert recorder.calls == []
    out = await _call(_manager(_cycle()), _Recorder([]))
    assert out["available"] is False and out["reason"] == "no_history"
    # Only unavailable rows: nothing to draw either.
    out = await _call(_manager(_cycle()), _Recorder([(START.timestamp() - 30, None)]))
    assert out["available"] is False and out["reason"] == "no_history"


@pytest.mark.asyncio
async def test_window_lengths_are_clamped():
    recorder = _Recorder(_history())
    out = await _call(_manager(_cycle()), recorder, before_s=99999.0, after_s=-5.0)
    assert out["before_s"] == 3600.0 and out["after_s"] == 0.0
    assert len(recorder.calls) == 1  # no after query for a zero window
    assert recorder.calls[0][1] == START - timedelta(seconds=3600)
    out = await _call(_manager(_cycle()), _Recorder(_history()), before_s=math.nan, after_s=math.inf)
    assert out["before_s"] == 0.0 and out["after_s"] == 0.0


@pytest.mark.asyncio
async def test_display_only_the_cycle_is_never_touched():
    cycle = _cycle()
    snapshot = copy.deepcopy(cycle)
    manager = _manager(cycle)
    await _call(manager, _Recorder(_history()))
    assert cycle == snapshot
    called = {name for name, _a, _k in manager.profile_store.method_calls}
    assert called <= {"find_stored_cycle", "get_cycle_power_data"}


def test_context_points_thin_each_run_and_keep_the_gap():
    base = 1_000_000.0
    rows = [(base + i, 1.0 + (i % 7)) for i in range(1000)]
    rows[500] = (base + 500, None)
    rows[250] = (base + 250, 900.0)  # a one-sample spike must survive thinning
    pts = ws_api._cycle_context_points(rows, base, base, base + 2000, hi_inclusive=False, max_points=100)
    nulls = [p for p in pts if p[1] is None]
    assert nulls == [[500.0, None]]
    assert len(pts) < 130
    assert [250.0, 900.0] in pts
    gap = pts.index([500.0, None])
    assert all(p[0] < 500 for p in pts[:gap]) and all(p[0] > 500 for p in pts[gap + 1:])
    # Window edges: lo inclusive, hi exclusive unless asked.
    assert ws_api._cycle_context_points([(5.0, 1.0), (10.0, 2.0)], 0.0, 5.0, 10.0, hi_inclusive=False) == [[5.0, 1.0]]
    assert ws_api._cycle_context_points([(5.0, 1.0), (10.0, 2.0)], 0.0, 5.0, 10.0, hi_inclusive=True) == [[5.0, 1.0], [10.0, 2.0]]


@pytest.mark.asyncio
async def test_recorder_power_keeps_unavailable_rows_only_when_asked():
    start = START
    states = [
        SimpleNamespace(state="0.3", last_changed=start - timedelta(hours=5)),
        SimpleNamespace(state="unavailable", last_changed=start + timedelta(seconds=10)),
        SimpleNamespace(state="12.5", last_changed=start + timedelta(seconds=20)),
    ]
    hass = SimpleNamespace()  # passed through to the patched recorder only

    class _Instance:
        async def async_add_executor_job(self, fn, *args):
            return fn(*args)

    with patch("homeassistant.components.recorder.history.state_changes_during_period",
               return_value={"sensor.p": states}), \
         patch("homeassistant.components.recorder.get_instance", return_value=_Instance()):
        plain = await ws_api._recorder_power(hass, "sensor.p", start, end_dt=start + timedelta(minutes=1))
        kept = await ws_api._recorder_power(hass, "sensor.p", start, end_dt=start + timedelta(minutes=1),
                                            keep_unavailable=True)
    s = start.timestamp()
    assert plain == [(s, 0.3), (s + 20, 12.5)]
    assert kept == [(s, 0.3), (s + 10, None), (s + 20, 12.5)]


def _conn(uid: str):
    c = MagicMock()
    c.user = SimpleNamespace(id=uid)
    return c


@pytest.mark.asyncio
async def test_cycle_context_minutes_pref_is_validated_and_merged_per_device():
    hass = SimpleNamespace(data={ws_api._PANEL_DATA_KEY: {"data": {"prefs": {}}}})

    async def _set(prefs):
        with patch.object(ws_api, "_save_panel_data", new=AsyncMock()), \
             patch.object(ws_api, "_send_result", MagicMock()):
            await ws_api.ws_set_user_prefs.__wrapped__(hass, _conn("u1"), {"id": 1, "prefs": prefs})

    stored = lambda: hass.data[ws_api._PANEL_DATA_KEY]["data"]["prefs"]["u1"].get("cycle_context_min")  # noqa: E731
    await _set({"cycle_context_min": {"entryA": 30}})
    assert stored() == {"entryA": 30}
    # Merged: another device keeps the first one's value.
    await _set({"cycle_context_min": {"entryB": 0}})
    assert stored() == {"entryA": 30, "entryB": 0}
    # Values outside the offered list, a bool, a non-number and a bad key are dropped.
    await _set({"cycle_context_min": {"entryA": 7, "entryC": True, "entryD": "10", "": 5, "x" * 65: 5}})
    assert stored() == {"entryA": 30, "entryB": 0}
    await _set({"cycle_context_min": "everything"})
    assert stored() == {"entryA": 30, "entryB": 0}


def test_lead_in_eval_separates_probe_prelude_and_lost_head():
    """devtools/lead_in_eval.py: the diagnosis the item-513 design rests on."""
    import importlib.util  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    path = Path(__file__).resolve().parent.parent / "devtools" / "lead_in_eval.py"
    spec = importlib.util.spec_from_file_location("wd_test_lead_in_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t = lambda s: START + timedelta(seconds=s)  # noqa: E731
    idle = [(t(-3000), 0.0)]
    # A sub-threshold run-up (2 W, between stop 1.2 W and start 3 W) from -200 s, an
    # aborted probe at -60 s (8 W, chained into the start), then the stored start.
    history = idle + [(t(-200), 2.0), (t(-60), 8.0), (t(-50), 2.0), (t(0), 112.0), (t(60), 350.0)]
    row = mod.lead_in(history, START, 3.0, 1.2)
    assert row["probe_s"] == 60.0
    assert row["prelude_s"] == 140.0 and row["prelude_max_w"] == 2.0
    assert row["window_unavailable"] == 0.0
    # A head lost to an HA restart: a running load, an unavailable row, then the
    # stored start; the walk stops at the gap and the window shows the energy.
    lost = [(t(-1700), 1600.0), (t(-120), None), (t(0), 40.0)]
    row = mod.lead_in(lost, START, 3.0, 1.2)
    assert row["probe_s"] == 0.0 and row["prelude_s"] == 0.0
    assert row["window_unavailable"] == 1.0 and row["window_max_w"] == 1600.0
    assert row["window_wh"] > 200.0
