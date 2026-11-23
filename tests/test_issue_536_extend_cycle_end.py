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
"""#536: Trim can move a cycle's end past its stored trace, from the recorder.

A cycle ended too early (a dishwasher's drying, #531/#487) keeps going on the
power sensor's history after its stored end. Trim's end handle may now go into
that history: the readings up to it are appended (with the keepalives the live
watchdog would have stored across a silence), and duration, energy, cost and the
meter reading move with them. Only a run this device observed live, not over the
next stored run.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import task_registry, ws_api
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    cycle_extension_points,
    cycle_report_energy_wh,
)
from custom_components.ha_washdata.signal_processing import integrate_cycle_wh
from custom_components.ha_washdata.time_utils import SYNTHETIC_OFFSETS_KEY

POWER = "sensor.washer_power"


# --- the pure builder -------------------------------------------------------

def test_appends_rows_and_snaps_to_the_last_real_reading() -> None:
    rows = [(3600.0, 3.3), (3700.0, 3.4), (3800.0, 6.3), (3840.0, 3.3), (5000.0, 0.0)]
    pts, syn = cycle_extension_points(3600.0, 3.3, rows, 3900.0, keepalive_s=0)
    assert pts == [[3700.0, 3.4], [3800.0, 6.3], [3840.0, 3.3]]
    assert syn == []
    # Within the whole-second tolerance a reading just past the end is kept.
    pts, _ = cycle_extension_points(3600.0, 3.3, rows, 3839.2, keepalive_s=0)
    assert pts[-1] == [3840.0, 3.3]


def test_a_silence_gets_the_live_watchdog_keepalives() -> None:
    # A change-only plug: nothing for 20 min at 3.3 W, then the beep.
    rows = [(3600.0, 3.3), (4800.0, 6.3), (4840.0, 3.3)]
    pts, syn = cycle_extension_points(3600.0, 2.0, rows, 4840.0, keepalive_s=300)
    # The state in force at the trace's end (3.3 W), held every 300 s.
    assert pts[:3] == [[3900.0, 3.3], [4200.0, 3.3], [4500.0, 3.3]]
    assert pts[3:] == [[4800.0, 6.3], [4840.0, 3.3]]
    assert syn == [3900.0, 4200.0, 4500.0]


def test_nothing_is_held_across_an_outage() -> None:
    rows = [(3600.0, 3.3), (3650.0, None), (4800.0, 3.2)]
    pts, syn = cycle_extension_points(3600.0, 3.3, rows, 4800.0, keepalive_s=300)
    assert pts == [[4800.0, 3.2]]
    assert syn == []


def test_no_reading_after_the_end_is_no_extension() -> None:
    assert cycle_extension_points(3600.0, 3.3, [(3600.0, 3.3)], 4000.0, 30) is None
    assert cycle_extension_points(3600.0, 3.3, [(3650.0, None)], 4000.0, 30) is None
    assert cycle_extension_points(3600.0, 3.3, [(3700.0, 1.0)], 3500.0, 30) is None


# --- the store ---------------------------------------------------------------

def _cycle(start, trace: list[list[float]], **extra: Any) -> dict[str, Any]:
    end = trace[-1][0]
    cycle = {
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=end)).isoformat(),
        "duration": end,
        "status": "completed",
        "power_data": [list(p) for p in trace],
        **extra,
    }
    cycle["energy_wh"] = round(integrate_cycle_wh(
        [p[0] for p in trace], [p[1] for p in trace]), 3)
    return cycle


TRACE = [[0.0, 2000.0], [1800.0, 2000.0], [3600.0, 30.0]]


async def test_store_extends_duration_energy_cost_and_marks_the_edit(hass: HomeAssistant) -> None:
    store = ProfileStore(hass, "e536")
    start = dt_util.utcnow() - timedelta(hours=5)
    await store.async_add_cycle(_cycle(start, TRACE, energy_price=0.25, energy_price_mode="fixed"))
    cycle = store.get_past_cycles()[0]
    cycle["cost"] = round(cycle_report_energy_wh(cycle) / 1000.0 * 0.25, 4)
    cycle[SYNTHETIC_OFFSETS_KEY] = [1800.0]
    before_wh = float(cycle["energy_wh"])

    ext = cycle_extension_points(
        3600.0, 30.0, [(3600.0, 3.3), (4800.0, 6.3), (4840.0, 3.3)], 4840.0, 300)
    assert await store.trim_cycle_power_data(cycle["id"], 0.0, 4840.0, extension=ext)

    cycle = store.get_past_cycles()[0]
    assert cycle["duration"] == 4840.0
    assert cycle["power_data"][-2:] == [[4800.0, 6.3], [4840.0, 3.3]]
    assert cycle["end_time"] == (start + timedelta(seconds=4840)).isoformat()
    assert float(cycle["energy_wh"]) > before_wh
    assert cycle["cost"] == round(cycle_report_energy_wh(cycle) / 1000.0 * 0.25, 4)
    assert cycle["meta"]["edited"] is True
    assert cycle["meta"]["extend"] == [3600.0, 4840.0]
    # The keepalives join the trace's own synthetic rows.
    assert cycle[SYNTHETIC_OFFSETS_KEY] == [1800.0, 3900.0, 4200.0, 4500.0]


async def test_meter_reading_moves_with_the_added_stretch(hass: HomeAssistant) -> None:
    store = ProfileStore(hass, "e536m")
    start = dt_util.utcnow() - timedelta(hours=5)
    await store.async_add_cycle(_cycle(start, TRACE, energy_meter_wh=1500.0))
    cid = store.get_past_cycles()[0]["id"]
    ext = cycle_extension_points(3600.0, 30.0, [(4200.0, 30.0)], 4200.0, 0)
    assert await store.trim_cycle_power_data(cid, 0.0, 4200.0, extension=ext)
    # 600 s held at 30 W = 5 Wh.
    assert store.get_past_cycles()[0]["energy_meter_wh"] == 1505.0


async def test_never_over_the_next_run_or_onto_a_reference(hass: HomeAssistant) -> None:
    store = ProfileStore(hass, "e536o")
    start = dt_util.utcnow() - timedelta(hours=8)
    await store.async_add_cycle(_cycle(start, TRACE))
    await store.async_add_cycle(_cycle(start + timedelta(seconds=4000), TRACE))
    first = min(store.get_past_cycles(), key=lambda c: c["start_time"])
    snapshot = dict(first)
    ext = cycle_extension_points(3600.0, 30.0, [(4500.0, 2000.0)], 4500.0, 0)
    assert not await store.trim_cycle_power_data(first["id"], 0.0, 4500.0, extension=ext)
    assert store.find_stored_cycle(first["id"])[0]["duration"] == snapshot["duration"]


# --- the trim task, through a real boot --------------------------------------

class _Recorder:
    def __init__(self, rows: list[tuple[float, float | None]]) -> None:
        self.rows = rows
        self.calls: list[tuple[Any, Any]] = []

    async def __call__(self, hass, entity_id, start_dt, *, end_dt=None, keep_unavailable=False,
                       scale=1.0):
        self.calls.append((start_dt, end_dt))
        lo, hi = start_dt.timestamp(), end_dt.timestamp()
        out = [(max(lo, ts), w) for ts, w in self.rows if ts <= hi]
        held = [r for r in out if r[0] == lo]
        return held[-1:] + [r for r in out if r[0] > lo]


async def _run_trim(hass, entry, cid, start_s, end_s, recorder) -> Any:
    reg = task_registry.get_registry(hass)
    task = reg.create(entry.entry_id, "trim", "Trimming cycle")
    with patch.object(ws_api, "_recorder_power", new=recorder):
        await ws_api._trim_task(hass, task, entry.entry_id, cid, start_s, end_s)  # noqa: SLF001
    return task


@pytest.fixture
async def booted(hass: HomeAssistant, setup_washdata_entry):
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    start = dt_util.utcnow() - timedelta(hours=6)
    await manager.profile_store.async_add_cycle(_cycle(start, TRACE))
    cid = manager.profile_store.get_past_cycles()[0]["id"]
    return entry, manager, start, cid


async def test_trim_past_the_end_reads_the_recorder(hass: HomeAssistant, booted) -> None:
    entry, manager, start, cid = booted
    s = start.timestamp()
    recorder = _Recorder([(s + 3600, 30.0), (s + 3700, 3.3), (s + 7000, 6.3), (s + 7040, 3.2)])
    task = await _run_trim(hass, entry, cid, 0.0, 7040.0, recorder)
    assert task.state == task_registry.STATE_DONE, task.error
    cycle = manager.profile_store.find_stored_cycle(cid)[0]
    assert cycle["duration"] == 7040.0
    assert cycle["meta"]["extend"] == [3600.0, 7040.0]
    # One bounded query from the trace's end, just past the new end.
    ((lo, hi),) = recorder.calls
    assert lo.timestamp() == pytest.approx(s + 3600)
    assert hi.timestamp() == pytest.approx(s + 7041)
    # The silence between 3700 and 7000 s carries the live watchdog's keepalives.
    gaps = [b[0] - a[0] for a, b in zip(cycle["power_data"], cycle["power_data"][1:])]
    assert max(gaps[2:]) <= manager._watchdog_interval + 0.1  # noqa: SLF001


async def test_a_plain_trim_never_reads_the_recorder(hass: HomeAssistant, booted) -> None:
    entry, manager, _start, cid = booted
    recorder = _Recorder([])
    task = await _run_trim(hass, entry, cid, 0.0, 1800.0, recorder)
    assert task.state == task_registry.STATE_DONE
    assert recorder.calls == []
    assert manager.profile_store.find_stored_cycle(cid)[0]["duration"] == 1800.0


@pytest.mark.parametrize(
    ("end_s", "rows", "error"),
    [
        (3600.0 + 4 * 3600.0 + 60, [], "extend_too_far"),
        (5000.0, [], "extend_no_history"),
        (5000.0, [(3650.0, None)], "extend_no_history"),
    ],
)
async def test_refusals_leave_the_cycle_alone(
    hass: HomeAssistant, booted, end_s, rows, error
) -> None:
    entry, manager, start, cid = booted
    s = start.timestamp()
    task = await _run_trim(hass, entry, cid, 0.0, end_s, _Recorder([(s + o, w) for o, w in rows]))
    assert task.state == task_registry.STATE_ERROR
    assert task.error == error
    assert manager.profile_store.find_stored_cycle(cid)[0]["duration"] == 3600.0


async def test_the_next_run_bounds_the_end(hass: HomeAssistant, booted) -> None:
    entry, manager, start, cid = booted
    await manager.profile_store.async_add_cycle(_cycle(start + timedelta(seconds=5000), TRACE))
    s = start.timestamp()
    recorder = _Recorder([(s + 4000, 3.3), (s + 5200, 2000.0)])
    task = await _run_trim(hass, entry, cid, 0.0, 5200.0, recorder)
    assert task.error == "extend_overlaps"
    assert recorder.calls == []
    # Up to just before it is fine.
    task = await _run_trim(hass, entry, cid, 0.0, 4000.0, recorder)
    assert task.state == task_registry.STATE_DONE, task.error
    assert manager.profile_store.find_stored_cycle(cid)[0]["duration"] == 4000.0
