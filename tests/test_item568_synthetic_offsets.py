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
"""Register item 568 (deferred parity finding): a replay could not tell a keepalive
from a plug re-reporting an unchanged value.

Since 0.5.7 the live watchdog's keepalives are recorded in the stored trace at the
sensor's last value, so the replay's ghost suppressor read every repeated value as
silence (``REPLAY_REPEATS_ARE_SILENCE``). On a plug that re-reports an unchanged low
value the replay then force-ended a cycle as a ghost while live, which counts those
reports, kept it running. A cycle the manager records now lists the rows it injected
(``synthetic_offsets``); the replay reads every other row as a report, and the list
follows the trace through every edit.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata import playground, trace_codec
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_WASHING_MACHINE,
    DOMAIN,
    STATE_FINISHED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.time_utils import (
    SYNTHETIC_OFFSETS_KEY,
    offset_key,
    synthetic_offset_keys,
)

POWER = "sensor.plug_power"
# The #509 options with a 400 s Min Off Gap: the end gate lands just after the ghost
# suppressor's clock (10 min old, 5 min silent) would; watchdog every 30 s.
OPTIONS = {
    "min_power": 2.0, "stop_threshold_w": 4.0, "start_threshold_w": 5.0, "off_delay": 60,
    "min_off_gap": 400, "completion_min_seconds": 300, "watchdog_interval": 30,
}
T0 = datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc)


def _rows(cycle: dict[str, Any]) -> dict[int, float]:
    return {offset_key(o): float(p) for o, p in cycle["power_data"]}


def _assert_names_rows(cycle: dict[str, Any]) -> None:
    rows = _rows(cycle)
    assert all(offset_key(o) in rows for o in cycle[SYNTHETIC_OFFSETS_KEY]), cycle


# --- (a) live: the manager records its keepalives ----------------------------------


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int, watts: float | None) -> None:
    freezer.tick(timedelta(seconds=seconds))
    if watts is not None:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"})
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _boot(hass: HomeAssistant, setup_washdata_entry) -> tuple[Any, list, list]:
    entry = await setup_washdata_entry(
        power_sensor=POWER, device_type=DEVICE_TYPE_WASHING_MACHINE, options=OPTIONS
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    ends: list[dict[str, Any]] = []
    injected: list[datetime] = []
    original_end = manager.detector._on_cycle_end
    original_process = manager.detector.process_reading

    def _end(cycle_data: dict[str, Any]) -> None:
        ends.append(cycle_data)
        original_end(cycle_data)

    def _process(power, timestamp, synthetic=False, observed=True):
        if synthetic:
            injected.append(dt_util.as_utc(timestamp))
        return original_process(power, timestamp, synthetic=synthetic, observed=observed)

    manager.detector._on_cycle_end = _end
    manager.detector.process_reading = _process
    return manager, ends, injected


async def test_a_live_cycle_stores_the_offsets_of_its_watchdog_keepalives(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    manager, ends, injected = await _boot(hass, setup_washdata_entry)
    for i in range(12):  # 6 min of a reporting 400 W run
        await _tick(hass, freezer, 30, 400.0 + i % 2)
    await _tick(hass, freezer, 30, 0.5)  # a 3 min drum pause on a silent plug
    for _ in range(6):
        await _tick(hass, freezer, 30, None)
    assert injected, "the watchdog injected no keepalive in the silent pause"
    for i in range(6):
        await _tick(hass, freezer, 30, 400.0 + i % 2)
    await _tick(hass, freezer, 30, 0.0)
    for _ in range(30):
        if ends:
            break
        await _tick(hass, freezer, 30, None)
    assert len(ends) == 1 and manager.detector.state == STATE_FINISHED
    await hass.async_block_till_done()

    stored = manager.profile_store.get_past_cycles()[-1]
    start = dt_util.parse_datetime(stored["start_time"])
    in_trace = {
        offset_key((ts - start).total_seconds()) for ts in injected if ts >= start
    } & set(_rows(stored))
    assert in_trace, "the pause keepalives sit inside the stored trace"
    assert synthetic_offset_keys(stored) == in_trace
    # Each one carries the sensor's last value, which is what made it ambiguous.
    rows = _rows(stored)
    assert {rows[k] for k in in_trace} == {0.5}
    # ...and a replay of the stored cycle reads exactly those rows as keepalives.
    sim = playground._DetailSim(  # noqa: SLF001
        stored, build_detector_config(OPTIONS, {}, DEVICE_TYPE_WASHING_MACHINE),
        None, None, OPTIONS, None,
    )
    flagged = {
        offset_key((ts - sim.base).total_seconds())
        for (ts, _p), syn in zip(sim.readings, sim.reading_synthetic) if syn
    }
    assert flagged == in_trace
    assert sim.reading_synthetic.count(False) == len(sim.readings) - len(in_trace)


async def test_a_plug_re_reporting_an_unchanged_value_is_no_ghost_live(
    hass: HomeAssistant, setup_washdata_entry, freezer, caplog
) -> None:
    """Guard (passes before and after): what the replay now matches. Live counts
    unchanged re-reports (state_reported), so a back-to-back run settled on a
    periodic plug's 0.8 W is ended by its own gate, never as a ghost."""
    manager, ends, _injected = await _boot(hass, setup_washdata_entry)
    for i in range(20):
        await _tick(hass, freezer, 30, 400.0 + i % 2)
    await _tick(hass, freezer, 30, 0.0)
    for _ in range(30):
        if ends:
            break
        await _tick(hass, freezer, 30, None)
    assert len(ends) == 1
    await _tick(hass, freezer, 60, 200.0)  # the next run, inside the 180 s window
    for i in range(22):
        await _tick(hass, freezer, 30, 200.0 + i % 2)
    assert manager.detector.state == STATE_RUNNING
    for _ in range(40):  # 0.8 W re-reported every 20 s
        await _tick(hass, freezer, 20, 0.8)
        if len(ends) > 1:
            break
    assert "Ghost cycle suppressed" not in caplog.text
    assert len(ends) == 2 and ends[1]["status"] == "completed", ends[1:]


def _sensor_state(det: CycleDetector, start: datetime) -> dict[str, Any]:
    """What the detector knows of the SENSOR: its cadence window, its last real
    reading, the flat run and the idle display (synthetic readings feed none)."""
    def off(ts: datetime | None) -> float | None:
        return None if ts is None else round((ts - start).total_seconds(), 1)

    return {
        "dts": [round(d, 1) for d in det._recent_dts],
        "p95": round(det._p95_dt, 1),
        "gate": round(det._gate_cadence, 1),
        "last_real": off(det._last_real_reading_time),
        "flat": [off(ts) for ts in det._flat_run._ts],
        "idle": (det._idle_shown, off(det._idle_candidate_since), off(det._idle_off_since)),
    }


async def test_a_replayed_keepalive_leaves_the_sensor_state_as_live(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """A recorded keepalive reaches the replay's detector as the watchdog fed it
    (``synthetic=True``): the cadence the pause/end gates are sized from, the last
    real reading, the flat run and the idle display match live's. Fed as a report,
    the 30 s keepalives trained the 10 s plug's cadence (#424)."""
    manager, ends, injected = await _boot(hass, setup_washdata_entry)
    for i in range(25):  # a plug reporting every 10 s
        await _tick(hass, freezer, 10, 400.0 + i % 2)
    await _tick(hass, freezer, 10, 0.5)  # then silent at 0.5 W for 150 s
    for _ in range(15):
        await _tick(hass, freezer, 10, None)
    live_at = dt_util.utcnow()
    assert len(injected) >= 3
    det = manager.detector
    start = det.current_cycle_start
    live = _sensor_state(det, start)
    for i in range(6):
        await _tick(hass, freezer, 10, 400.0 + i % 2)
    await _tick(hass, freezer, 10, 0.0)
    for _ in range(30):
        if ends:
            break
        await _tick(hass, freezer, 30, None)
    assert len(ends) == 1
    await hass.async_block_till_done()
    stored = manager.profile_store.get_past_cycles()[-1]
    assert dt_util.parse_datetime(stored["start_time"]) == start

    sim = playground._DetailSim(  # noqa: SLF001
        stored, build_detector_config(OPTIONS, {}, DEVICE_TYPE_WASHING_MACHINE),
        None, None, OPTIONS, None,
    )
    upto = sum(1 for ts, _p in sim.readings if ts <= live_at)
    paused = [ts for ts in injected if ts <= live_at]
    assert sum(1 for syn in sim.reading_synthetic[:upto] if syn) == len(paused)
    sim.step(0, upto)
    assert _sensor_state(sim.detector, start) == live
    assert live["gate"] == 10.0  # the plug's cadence, not the watchdog's


def _feeds(cycle: dict[str, Any]) -> list[tuple[float, float, bool, bool]]:
    """Every reading the replay hands its detector: (offset, W, synthetic, observed)."""
    cfg = build_detector_config(OPTIONS, {}, DEVICE_TYPE_WASHING_MACHINE)
    sim = playground._DetailSim(cycle, cfg, None, None, OPTIONS, None)  # noqa: SLF001
    fed: list[tuple[float, float, bool, bool]] = []
    original = sim.detector.process_reading

    def _spy(power, timestamp, synthetic=False, observed=True):
        fed.append((round((timestamp - sim.base).total_seconds(), 1), power, synthetic, observed))
        return original(power, timestamp, synthetic=synthetic, observed=observed)

    sim.detector.process_reading = _spy
    sim.step(0, sim.n_readings)
    return fed


def test_recorded_keepalives_are_fed_once_and_as_the_watchdog_fed_them() -> None:
    """The pause's first keepalive came 57 s (1.9 ticks) after the drop, the last one
    100 s after the previous (a late tick). The emulation (item 390) used to add one
    at 45 s; recorded keepalives cover the gap, so only they are fed, synthetic,
    and the late one unobserved, as ``manager._watchdog_check_stuck_cycle`` passed."""
    pts = [[float(t), 400.0 + (t // 10) % 2] for t in range(0, 600, 10)]
    pause = [600.0 + d for d in (57.0, 87.0, 117.0, 217.0)]
    pts += [[600.0, 0.5]] + [[t, 0.5] for t in pause]
    pts += [[float(t), 400.0] for t in range(830, 900, 10)] + [[900.0, 0.0]]
    base = {"id": "c568f", "start_time": T0.isoformat(), "power_data": pts}

    fed = _feeds({**base, SYNTHETIC_OFFSETS_KEY: list(pause)})
    in_pause = [f for f in fed if 600.0 < f[0] < 830.0]
    assert in_pause == [(57.0 + 600, 0.5, True, True), (87.0 + 600, 0.5, True, True),
                        (117.0 + 600, 0.5, True, True), (217.0 + 600, 0.5, True, False)]

    # Without the list: as before, rows are reports and the emulation fills the gaps.
    legacy = [f for f in _feeds(dict(base)) if 600.0 < f[0] < 830.0]
    assert [f for f in legacy if not f[2]] == [(t, 0.5, False, True) for t in pause]
    assert [f[0] for f in legacy if f[2]] == [645.0, 162.0 + 600, 192.0 + 600]


# --- (b) replay: a re-reporting plug is not a silent one -----------------------------


def _replay(cycle: dict[str, Any]) -> Any:
    cfg = build_detector_config(OPTIONS, {}, DEVICE_TYPE_WASHING_MACHINE)
    sim = playground._DetailSim(cycle, cfg, None, None, OPTIONS, None)  # noqa: SLF001
    sim.step(0, sim.n_readings)
    sim.run_tail()
    return sim


def _re_reporting_trace() -> tuple[list[list[float]], float]:
    """A 10 min run, then the next one 60 s after its end (inside the suspicious
    window): 11 min at 200 W, then 0.8 W re-reported every 20 s for 15 min (the live
    guard above). Returns the trace and where the 0.8 W stretch begins."""
    pts = [[float(t), 400.0 + (t // 30) % 2] for t in range(0, 600, 30)] + [[600.0, 0.0]]
    end = _replay({"id": "probe", "start_time": T0.isoformat(), "power_data": pts}).cycle_ends[0]["t"]
    start = end + 60.0
    pts += [[start + t, 200.0 + (t // 30) % 2] for t in range(0, 660, 30)]
    low = start + 660.0
    pts += [[low + t, 0.8] for t in range(0, 900, 20)]
    return pts, low


def test_a_periodic_plug_re_reporting_a_low_value_is_not_ended_as_a_ghost() -> None:
    pts, low = _re_reporting_trace()
    base = {"id": "c568r", "start_time": T0.isoformat(), "power_data": pts}

    # Without the list (every cycle recorded before it): the repeats read as silence
    # and the suppressor ends the run 5 min after the last changed value.
    legacy = _replay(dict(base))
    assert legacy.captured[1]["termination_reason"] == "force_stopped"
    assert 300 < legacy.cycle_ends[1]["t"] - low <= 330

    # Recorded with it, no keepalive in the stretch: every 20 s row is a report, so
    # the run ends by its own gate (the 400 s Min Off Gap), as live does.
    recorded = _replay({**base, SYNTHETIC_OFFSETS_KEY: []})
    assert len(recorded.captured) == 2 and recorded.discarded == [None, None]
    assert recorded.captured[1]["termination_reason"] == "timeout"
    assert recorded.captured[1]["status"] == "completed"
    assert recorded.cycle_ends[1]["t"] - low > 330
    # The first cycle is replayed identically either way.
    assert recorded.captured[0] == legacy.captured[0]


def test_listed_keepalives_stay_silence() -> None:
    """The same stretch made of keepalives (listed) still reads as silence."""
    pts, low = _re_reporting_trace()
    offsets = [t for t, _p in pts if t > low]
    sim = _replay({
        "id": "c568k", "start_time": T0.isoformat(), "power_data": pts,
        SYNTHETIC_OFFSETS_KEY: offsets,
    })
    assert sim.captured[1]["termination_reason"] == "force_stopped"
    assert sim.cycle_ends[1]["t"] - low <= 330


# --- the detector: what it lists, across a restart ---------------------------------


def _detector(record: bool = True) -> tuple[CycleDetector, list[dict[str, Any]]]:
    cfg = build_detector_config(OPTIONS, {}, DEVICE_TYPE_WASHING_MACHINE)
    ends: list[dict[str, Any]] = []
    det = CycleDetector(
        cfg, lambda _o, _n: None, ends.append, record_synthetic_offsets=record
    )
    return det, ends


def _feed_pause_cycle(det: CycleDetector, until: int = 2400) -> list[float]:
    """Run, a silent pause filled by keepalives, run again; returns their offsets."""
    keepalives: list[float] = []
    for t in range(0, until, 30):
        ts = T0 + timedelta(seconds=t)
        if 600 <= t < 900:
            if t == 600:
                det.process_reading(0.5, ts)
            else:
                det.process_reading(0.5, ts, synthetic=True)
                keepalives.append(float(t))
        elif t < 1200:
            det.process_reading(400.0 + (t // 30) % 2, ts)
        elif t == 1200:
            det.process_reading(0.0, ts)
    return keepalives


def test_the_detector_lists_its_keepalives_only_when_asked() -> None:
    det, ends = _detector()
    keepalives = _feed_pause_cycle(det)
    det.force_end(T0 + timedelta(seconds=2400))
    assert ends and ends[0][SYNTHETIC_OFFSETS_KEY] == keepalives
    _assert_names_rows(ends[0])

    quiet, quiet_ends = _detector(record=False)  # a replay / history import
    _feed_pause_cycle(quiet)
    quiet.force_end(T0 + timedelta(seconds=2400))
    assert SYNTHETIC_OFFSETS_KEY not in quiet_ends[0]


def test_the_list_survives_a_restart_and_an_older_snapshot_leaves_it_unknown() -> None:
    det, _ends = _detector()
    keepalives = _feed_pause_cycle(det, until=1050)
    snapshot = det.get_state_snapshot()

    restored, ends = _detector()
    assert restored.restore_state_snapshot(snapshot)
    _feed_pause_cycle_tail(restored)
    assert ends[0][SYNTHETIC_OFFSETS_KEY] == keepalives

    older = {k: v for k, v in snapshot.items() if k != "synthetic_readings"}
    restored_old, old_ends = _detector()
    assert restored_old.restore_state_snapshot(older)
    _feed_pause_cycle_tail(restored_old)
    assert SYNTHETIC_OFFSETS_KEY not in old_ends[0]
    # The next cycle is recorded from its start, so it says again.
    _feed_pause_cycle(restored_old)
    restored_old.force_end(T0 + timedelta(seconds=2400))
    assert SYNTHETIC_OFFSETS_KEY in old_ends[1]


def _feed_pause_cycle_tail(det: CycleDetector) -> None:
    for t in range(1050, 1200, 30):
        det.process_reading(400.0, T0 + timedelta(seconds=t))
    det.process_reading(0.0, T0 + timedelta(seconds=1200))
    det.force_end(T0 + timedelta(seconds=1500))


# --- (c) the list follows every edit of the trace ------------------------------------


def _two_loads() -> tuple[list[list[float]], list[float]]:
    """1 h at 2000 W, a 10 min pause held by keepalives, 1 h at 1000 W (30 s rows)."""
    pts: list[list[float]] = []
    synthetic: list[float] = []
    for t in range(0, 7800 + 1, 30):
        if t < 3600:
            p = 2000.0
        elif t < 4200:
            p = 0.5
            if t > 3600:
                synthetic.append(float(t))
        else:
            p = 1000.0
        pts.append([float(t), p])
    return pts, synthetic


def _cycle(cid: str, start: datetime, power: list[list[float]], **extra: Any) -> dict[str, Any]:
    dur = power[-1][0]
    return {
        "id": cid, "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=dur)).isoformat(), "duration": dur,
        "status": "completed", "power_data": power, "profile_name": None, **extra,
    }


@pytest.fixture
async def store(hass) -> ProfileStore:
    st = ProfileStore(hass, "test_568_synthetic")
    st._data = {"past_cycles": [], "profiles": {}, "envelopes": {}}
    st.async_save = AsyncMock()
    st.async_rebuild_envelope = AsyncMock()
    return st


def _absolute(cycle: dict[str, Any]) -> set[float]:
    start = dt_util.parse_datetime(cycle["start_time"]).timestamp()
    return {round(start + o, 1) for o in cycle[SYNTHETIC_OFFSETS_KEY]}


async def test_a_trim_rebases_the_offsets_and_drops_the_cut_rows(store: ProfileStore) -> None:
    pts, synthetic = _two_loads()
    parent = _cycle("c1", T0, pts, **{SYNTHETIC_OFFSETS_KEY: list(synthetic)})
    store._data["past_cycles"] = [parent]
    before = {s for s in _absolute(parent) if s >= T0.timestamp() + 3900.0}

    assert await store.trim_cycle_power_data("c1", 3900.0, 7800.0)
    trimmed = store._data["past_cycles"][0]
    assert trimmed[SYNTHETIC_OFFSETS_KEY][0] == 0.0  # the row at 3900 s, now t=0
    assert _absolute(trimmed) == before
    _assert_names_rows(trimmed)
    rows = _rows(trimmed)
    assert {rows[offset_key(o)] for o in trimmed[SYNTHETIC_OFFSETS_KEY]} == {0.5}


async def test_a_split_hands_each_piece_its_own_rows(store: ProfileStore) -> None:
    pts, synthetic = _two_loads()
    # Off the 30 s grid: the second piece opens on a point the split invents.
    parent = _cycle("c1", T0, pts, **{SYNTHETIC_OFFSETS_KEY: list(synthetic)})
    store._data["past_cycles"] = [parent]
    ids = await store.apply_split_interactive(
        "c1", [{"start": 0.0, "end": 1815.0}, {"start": 1815.0, "end": 7800.0}]
    )
    by_id = {c["id"]: c for c in store._data["past_cycles"]}
    first, second = (by_id[i] for i in ids)
    for piece in (first, second):
        _assert_names_rows(piece)
    assert first[SYNTHETIC_OFFSETS_KEY] == []
    assert second[SYNTHETIC_OFFSETS_KEY] == [0.0] + [o - 1815.0 for o in synthetic]

    # On a row: the piece opens on that report, which is no keepalive.
    store._data["past_cycles"] = [
        _cycle("c2", T0, [list(p) for p in pts], **{SYNTHETIC_OFFSETS_KEY: list(synthetic)})
    ]
    ids = await store.apply_split_interactive(
        "c2", [{"start": 0.0, "end": 1800.0}, {"start": 1800.0, "end": 7800.0}]
    )
    second = next(c for c in store._data["past_cycles"] if c["id"] == ids[1])
    assert second[SYNTHETIC_OFFSETS_KEY] == [o - 1800.0 for o in synthetic]


async def test_a_merge_keeps_both_parts_rows_and_flags_its_gap_fill(
    store: ProfileStore,
) -> None:
    pts, synthetic = _two_loads()
    first = _cycle("a", T0, [list(p) for p in pts], **{SYNTHETIC_OFFSETS_KEY: list(synthetic)})
    later = T0 + timedelta(seconds=9000)
    second = _cycle("b", later, [[0.0, 900.0], [30.0, 900.0], [60.0, 0.4]],
                    **{SYNTHETIC_OFFSETS_KEY: [60.0]})
    store._data["past_cycles"] = [first, second]
    expected = _absolute(first) | _absolute(second)
    new_id = await store.apply_merge_interactive(["a", "b"], None)
    merged = next(c for c in store._data["past_cycles"] if c["id"] == new_id)
    _assert_names_rows(merged)
    gap = {round(T0.timestamp() + 7800.1, 1), round(later.timestamp() - 0.1, 1)}
    assert _absolute(merged) == expected | gap

    # A part that does not say makes the merged cycle not say either.
    store._data["past_cycles"] = [
        _cycle("c", T0, [list(p) for p in pts], **{SYNTHETIC_OFFSETS_KEY: list(synthetic)}),
        _cycle("d", later, [[0.0, 900.0], [30.0, 900.0], [60.0, 0.4]]),
    ]
    new_id = await store.apply_merge_interactive(["c", "d"], None)
    merged = next(c for c in store._data["past_cycles"] if c["id"] == new_id)
    assert SYNTHETIC_OFFSETS_KEY not in merged


async def test_a_repaired_duration_drops_the_cut_rows_and_flags_its_terminal_point(
    store: ProfileStore,
) -> None:
    pts, synthetic = _two_loads()
    cycle = _cycle("c1", T0, pts, **{SYNTHETIC_OFFSETS_KEY: list(synthetic)})
    store._apply_repaired_duration(cycle, 3700.0)  # noqa: SLF001
    _assert_names_rows(cycle)
    # Rows past 3700 s are gone, and the point added at 3700 s is no report.
    assert cycle[SYNTHETIC_OFFSETS_KEY] == [o for o in synthetic if o <= 3700.0] + [3700.0]


async def test_add_cycle_keeps_only_rows_it_stores(store: ProfileStore) -> None:
    """A trimmed leading row (``async_add_cycle`` drops leading zeros) leaves the list."""
    pts = [[0.0, 0.0], [30.0, 0.0], [60.0, 900.0], [90.0, 0.5], [120.0, 900.0]]
    cycle = _cycle("x", T0, pts, **{SYNTHETIC_OFFSETS_KEY: [30.0, 90.0]})
    await store.async_add_cycle(cycle)
    stored = store._data["past_cycles"][-1]
    assert stored[SYNTHETIC_OFFSETS_KEY] == [90.0]
    without = _cycle("y", T0 + timedelta(hours=1), [list(p) for p in pts])
    await store.async_add_cycle(without)
    assert SYNTHETIC_OFFSETS_KEY not in store._data["past_cycles"][-1]


def test_the_list_is_stored_compact_and_reloads_identical() -> None:
    data = {"past_cycles": [{"id": "a", "power_data": [[0.0, 5.0], [30.1, 0.5]],
                             SYNTHETIC_OFFSETS_KEY: [30.1]},
                            {"id": "b", "power_data": [[0.0, 5.0]], SYNTHETIC_OFFSETS_KEY: []}]}
    disk = trace_codec.encode_store_payload(data)
    assert isinstance(disk["past_cycles"][0][SYNTHETIC_OFFSETS_KEY], str)
    assert disk["past_cycles"][1][SYNTHETIC_OFFSETS_KEY] == []  # known, and empty
    trace_codec.decode_store_payload(disk)
    assert disk == data


def test_an_oversized_offset_is_skipped_not_raised() -> None:
    from custom_components.ha_washdata.time_utils import offset_key

    assert offset_key(1e308) is None
    assert offset_key(12.34) == 123


def test_without_a_start_time_synthetic_rows_share_the_traces_anchor() -> None:
    """An ISO trace starting at 00:00 with its first keepalive at 00:01: converted on
    its own, that keepalive came out at 0 s and named the trace's first row."""
    from custom_components.ha_washdata.time_utils import normalize_synthetic_offsets

    trace = [["2026-10-01T00:00:00+00:00", 500.0], ["2026-10-01T00:01:00+00:00", 2.0],
             ["2026-10-01T00:02:00+00:00", 2.0]]
    assert normalize_synthetic_offsets(["2026-10-01T00:01:00+00:00"], trace, None) == [60.0]
    assert normalize_synthetic_offsets(
        ["2026-10-01T00:01:00+00:00"], trace, "2026-10-01T00:00:00+00:00") == [60.0]


def test_an_unusable_start_time_also_takes_the_traces_anchor() -> None:
    from custom_components.ha_washdata.time_utils import normalize_synthetic_offsets

    trace = [[1700000000.0, 500.0], [1700000060.0, 2.0]]
    assert normalize_synthetic_offsets([1700000060.0], trace, "not a time") == [60.0]
    assert normalize_synthetic_offsets([1700000060.0], trace, None) == [60.0]
