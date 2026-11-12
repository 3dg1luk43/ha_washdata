# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Storage v19: compact traces on disk (``trace_codec``).

The store file holds each cycle's ``power_data`` / ``debug_data`` as one string of
compact JSON text; memory, exports and the WS API keep the lists. "Lossless" is
measured against what the store held before: the parse of the indented file Home
Assistant's Store writer produces, which is what ``hass_storage`` holds (the mock
writes through the real serialiser and parses it back).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import struct
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from homeassistant.const import EVENT_HOMEASSISTANT_FINAL_WRITE
from homeassistant.core import CoreState, HomeAssistant
from homeassistant.helpers.json import _orjson_default_encoder
from homeassistant.util import dt as dt_util
from homeassistant.util.json import json_loads
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata import trace_codec
from custom_components.ha_washdata.const import STORAGE_KEY, STORAGE_VERSION
from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    WashDataStore,
    decompress_power_data,
    unwrap_import_payload,
)

ENTRY = "01KV19COMPACTTRACES000000"
MAIN = f"{STORAGE_KEY}.{ENTRY}"


def _as_stored(value: Any) -> Any:
    """What a load of the old (indented) file returns for ``value``."""
    return json_loads(_orjson_default_encoder(value))


def _bits(value: Any) -> Any:
    """``value`` with every float as its 64-bit pattern and every type spelled out."""
    if isinstance(value, float):
        return ("f", struct.pack("<d", value))
    if isinstance(value, bool):
        return ("b", value)
    if isinstance(value, int):
        return ("i", value)
    if isinstance(value, list):
        return ("l", [_bits(v) for v in value])
    if isinstance(value, dict):
        return ("d", {k: _bits(v) for k, v in value.items()})
    return (type(value).__name__, value)


def _trace(n: int = 40, step: float = 30.0, watts: float = 412.3) -> list[list[float]]:
    return [[round(i * step, 1), round(watts + (i % 7) * 13.7, 1)] for i in range(n)]


def _cycle(cid: str, **extra: Any) -> dict[str, Any]:
    base = {
        "id": cid, "profile_name": "Cotton", "status": "completed", "duration": 1170.0,
        "start_time": "2026-10-01T08:00:00+00:00", "end_time": "2026-10-01T08:19:30+00:00",
        "energy_wh": 140.2, "sampling_interval": 30.0, "power_data": _trace(),
    }
    base.update(extra)
    return base


def _envelope(base: float) -> dict[str, Any]:
    """A profile envelope shaped like a real one (full-precision floats, nested refs)."""
    grid = [i * 40.61732673267327 for i in range(30)]
    return {
        "time_grid": grid,
        "target_duration": 1170.0,
        "min": [[round(t, 1), round(base * 0.5 + i, 1)] for i, t in enumerate(grid)],
        "max": [[round(t, 1), round(base * 1.5 + i, 1)] for i, t in enumerate(grid)],
        "avg": [[round(t, 1), round(base + i, 1)] for i, t in enumerate(grid)],
        "std": [[round(t, 1), round(base / 7 + i / 3, 1)] for i, t in enumerate(grid)],
        "cycle_count": 12, "shape_cycle_count": 12,
        "avg_energy": 0.38660216666666664, "duration_std_dev": 1026.4176080916666,
        "updated": "2026-11-04T00:00:02.427691+01:00",
        "energy_ref": {"n": 12, "median_wh": 388.089},
        "spin_ref": {"v": 1, "n": 12, "events": [{"heat": [[624.9, 685.3, 60.4]], "final": None}]},
        "tail_ref": {"n": 12, "with_tail": 9, "median_s": 157.3},
        "member_prog_s": {"92f0a85f960c": 7952.0, "5eb517f16806": -0.0},
    }


def _v18_store() -> dict[str, Any]:
    """A v18 store as it sits on disk: every list, the odd shapes a real one has."""
    odd = [[-0.0, 0.0], [0.1, 1e-7], [3.0, 2], [6.0, 123456789.123], [9.0, -4.5]]
    legacy_iso = [["2026-10-02T08:00:00+00:00", 10.0], ["2026-10-02T08:00:30+00:00", 600.0]]
    debug = {
        "ranking": [{"name": "Cotton", "score": 0.7750123090355376,
                     "current": [11.0, 30.68839779005513], "sample": [23.7, 15.75263157894741]}],
        "details": {}, "ambiguous": False,
    }
    return _as_stored({
        "profiles": {"Cotton": {"avg_duration": 1170.0, "sample_cycle_id": "p1"}},
        "past_cycles": [
            _cycle("p1", debug_data=debug),
            _cycle("p2", power_data=odd),
            _cycle("p3", power_data=legacy_iso, start_time="2026-10-02T08:00:00+00:00"),
            _cycle("p4", power_data=[]),
            {k: v for k, v in _cycle("p5").items() if k != "power_data"},
        ],
        "reference_cycles": [_cycle("r1", label_source="store")],
        "backfill_cycles": [_cycle("b1", profile_name=None)],
        "envelopes": {"Cotton": _envelope(412.3), "Eco 40": _envelope(97.1), "Empty": {}},
        "lifetime_cycle_count": 5,
        "lifetime_energy_wh": 700.0,
    })


_COMPACT_KEYS = (*trace_codec.STORED_CYCLE_LISTS, trace_codec.ENVELOPES_KEY)


def _cycles(data: dict[str, Any]) -> dict[str, Any]:
    """The parts of a store the codec writes: the three cycle lists and the envelopes."""
    return {k: data.get(k) for k in _COMPACT_KEYS}


def _migrator(hass: HomeAssistant) -> WashDataStore:
    return WashDataStore(hass, STORAGE_VERSION, f"{STORAGE_KEY}.test")


# --- the codec -------------------------------------------------------------------


def test_round_trip_is_bit_identical_to_the_indented_file() -> None:
    values = [
        _trace(),
        [[-0.0, 0.0], [0.1, 0.2], [1e-300, 5e-324], [1.7976931348623157e308, -2.5]],
        [[0.0, 7], [1, 2.0], [2.5, True], [3, None], ["2026-10-01T08:00:00+00:00", 9.9]],
        [[0.0, 1.0, 2.0], [], [5.0]],
        {"ranking": [{"score": 1 / 3, "current": [math.pi, math.e]}], 3: "int key"},
    ]
    for value in values:
        text = trace_codec.encode_value(value)
        assert isinstance(text, str) and text[0] in "[{"
        decoded = trace_codec.decode_value(text, type(value))
        assert _bits(decoded) == _bits(_as_stored(value))


def test_nan_and_inf_decode_as_the_indented_file_stores_them() -> None:
    value = [[0.0, float("nan")], [1.0, float("inf")]]
    decoded = trace_codec.decode_value(trace_codec.encode_value(value), list)
    assert decoded == _as_stored(value) == [[0.0, None], [1.0, None]]


def test_what_cannot_or_need_not_be_encoded_is_left_alone() -> None:
    assert trace_codec.encode_value([]) is None
    assert trace_codec.encode_value({}) is None
    assert trace_codec.encode_value("x") is None
    assert trace_codec.encode_value([[0.0, object()]]) is None
    for bad in ("not json", "{]", "42", '"text"', "{}"):
        assert trace_codec.decode_value(bad, list) is bad
    assert trace_codec.decode_value([[0.0, 1.0]], list) == [[0.0, 1.0]]


def test_encode_never_changes_the_store_it_was_given() -> None:
    data = _v18_store()
    before = copy.deepcopy(data)
    lists = {k: data[k] for k in _COMPACT_KEYS}
    out = trace_codec.encode_store_payload(data)
    assert data == before
    assert all(data[k] is lists[k] for k in lists)
    assert isinstance(out["envelopes"]["Cotton"], str)
    assert out["envelopes"]["Empty"] == {}  # nothing to gain
    assert out["profiles"] is data["profiles"]
    assert isinstance(out["past_cycles"][0]["power_data"], str)
    assert isinstance(out["past_cycles"][0]["debug_data"], str)
    assert out["past_cycles"][3]["power_data"] == []  # nothing to gain
    assert out["past_cycles"][4] is data["past_cycles"][4]  # no trace: not copied


def test_decode_is_copy_on_write_and_idempotent() -> None:
    disk = _as_stored(trace_codec.encode_store_payload(_v18_store()))
    shared = disk["past_cycles"]
    shared_first = shared[0]
    shared_envelopes = disk["envelopes"]
    assert trace_codec.decode_store_payload(disk) == 8  # 5 traces, 1 debug_data, 2 envelopes
    assert shared[0] is shared_first and isinstance(shared_first["power_data"], str)
    assert isinstance(shared_envelopes["Cotton"], str) and disk["envelopes"] is not shared_envelopes
    assert _bits(_cycles(disk)) == _bits(_cycles(_v18_store()))
    again = copy.deepcopy(disk)
    assert trace_codec.decode_store_payload(again) == 0
    assert again == disk


def test_file_is_at_least_four_times_smaller_per_sample() -> None:
    data = {"past_cycles": [_cycle(f"c{i}", power_data=_trace(n=400)) for i in range(5)]}
    old = len(_orjson_default_encoder({"data": data}))
    new = len(_orjson_default_encoder({"data": trace_codec.encode_store_payload(data)}))
    assert old / new > 4


# --- the migration ---------------------------------------------------------------


async def test_v18_to_v19_keeps_every_cycle_bit_identical_and_is_idempotent(
    hass: HomeAssistant,
) -> None:
    store = _migrator(hass)
    before = _v18_store()
    once = await store._async_migrate_func(18, 1, copy.deepcopy(before))  # noqa: SLF001
    assert _bits(_cycles(once)) == _bits(_cycles(before))
    assert once["envelopes"]["Cotton"] == before["envelopes"]["Cotton"]
    assert {k: v for k, v in once.items() if k not in _COMPACT_KEYS} == {
        k: v for k, v in before.items() if k not in _COMPACT_KEYS
    }
    twice = await store._async_migrate_func(18, 1, copy.deepcopy(once))  # noqa: SLF001
    assert _bits(twice) == _bits(once)


async def test_migration_decodes_text_an_older_build_wrote_back(hass: HomeAssistant) -> None:
    # An older build loads a v19 file without decoding it and saves it as its own
    # version: every later step must still read lists.
    store = _migrator(hass)
    round_tripped = _as_stored(trace_codec.encode_store_payload(_v18_store()))
    out = await store._async_migrate_func(17, 1, round_tripped)  # noqa: SLF001
    for key in trace_codec.STORED_CYCLE_LISTS:
        for cycle in out[key]:
            assert not isinstance(cycle.get("power_data"), str)
            assert not isinstance(cycle.get("debug_data"), str)
    assert all(isinstance(e, dict) for e in out["envelopes"].values())
    assert out["past_cycles"][0]["power_data"] == _v18_store()["past_cycles"][0]["power_data"]
    assert _bits(out["envelopes"]) == _bits(_v18_store()["envelopes"])


# --- the real Store ----------------------------------------------------------------


async def test_v18_file_migrates_to_compact_v19_and_reloads_identical(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    before = _v18_store()
    hass_storage[MAIN] = {"version": 18, "minor_version": 1, "key": MAIN, "data": copy.deepcopy(before)}

    ps = ProfileStore(hass, ENTRY)
    await ps.async_load()
    assert _bits(_cycles(ps._data)) == _bits(_cycles(before))  # noqa: SLF001

    # Migrated and written back compact; decoding the file gives the same cycles.
    disk = hass_storage[MAIN]
    assert disk["version"] == STORAGE_VERSION == 19
    assert isinstance(disk["data"]["past_cycles"][0]["power_data"], str)
    assert isinstance(disk["data"]["past_cycles"][0]["debug_data"], str)
    assert isinstance(disk["data"]["reference_cycles"][0]["power_data"], str)
    assert isinstance(disk["data"]["backfill_cycles"][0]["power_data"], str)
    assert isinstance(disk["data"]["envelopes"]["Cotton"], str)
    assert isinstance(disk["data"]["envelopes"]["Eco 40"], str)
    file_cycles = copy.deepcopy(disk["data"])
    trace_codec.decode_store_payload(file_cycles)
    assert _bits(_cycles(file_cycles)) == _bits(_cycles(before))

    # Save, load in a fresh store: identical again, and every reader sees lists.
    await ps.async_save()
    fresh = ProfileStore(hass, ENTRY)
    await fresh.async_load()
    assert _bits(fresh._data) == _bits(ps._data)  # noqa: SLF001
    for cycle in fresh.iter_stored_cycles():
        assert not isinstance(cycle.get("power_data"), str)
        assert decompress_power_data(cycle) == decompress_power_data(
            next(c for c in ps.iter_stored_cycles() if c["id"] == cycle["id"])
        )
    assert fresh.get_cycle_power_data("p1") == [tuple(r) for r in before["past_cycles"][0]["power_data"]]
    assert fresh.get_envelope("Cotton") == before["envelopes"]["Cotton"]
    assert all(isinstance(e, dict) for e in fresh._data["envelopes"].values())  # noqa: SLF001


async def test_saves_leave_memory_decoded_and_the_debounced_write_encodes_late(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps = ProfileStore(hass, ENTRY)
    await ps.async_load()
    ps._data = _v18_store()  # noqa: SLF001
    traces = [c.get("power_data") for c in ps.iter_stored_cycles()]
    await ps.async_save()
    assert [c.get("power_data") for c in ps.iter_stored_cycles()] == traces
    assert all(a is b for a, b in zip(traces, [c.get("power_data") for c in ps.iter_stored_cycles()]))

    # A debounced write encodes the store as it is when it fires, not when asked.
    ps.coalesce_saves()
    await ps.async_save()
    ps._data["past_cycles"][0]["power_data"] = _trace(n=3)  # noqa: SLF001
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=11))
    await hass.async_block_till_done()
    disk = copy.deepcopy(hass_storage[MAIN]["data"])
    assert isinstance(disk["past_cycles"][0]["power_data"], str)
    trace_codec.decode_store_payload(disk)
    assert disk["past_cycles"][0]["power_data"] == _trace(n=3)
    assert isinstance(ps._data["past_cycles"][0]["power_data"], list)  # noqa: SLF001


async def test_a_save_while_stopping_still_writes_the_store_as_it_is_at_the_final_write(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps = ProfileStore(hass, ENTRY)
    await ps.async_load()
    ps._data = _v18_store()  # noqa: SLF001
    hass.set_state(CoreState.stopping)
    try:
        await ps.async_save()  # deferred by HA to the final write
        ps._data["past_cycles"][0]["power_data"] = _trace(n=4)  # noqa: SLF001
        hass.bus.async_fire(EVENT_HOMEASSISTANT_FINAL_WRITE)
        await hass.async_block_till_done()
    finally:
        hass.set_state(CoreState.running)
    disk = copy.deepcopy(hass_storage[MAIN]["data"])
    assert isinstance(disk["past_cycles"][0]["power_data"], str)
    trace_codec.decode_store_payload(disk)
    assert disk["past_cycles"][0]["power_data"] == _trace(n=4)


# --- exports and imports -------------------------------------------------------------


async def test_export_keeps_the_readable_shape(hass: HomeAssistant) -> None:
    ps = ProfileStore(hass, ENTRY)
    await ps.async_load()
    ps._data = _v18_store()  # noqa: SLF001
    await ps.async_save()
    export = json.loads(json.dumps(ps.export_data()))
    assert export["version"] == STORAGE_VERSION
    assert export["data"]["past_cycles"][0]["power_data"] == _trace()
    assert isinstance(export["data"]["past_cycles"][0]["debug_data"], dict)
    assert isinstance(export["data"]["envelopes"]["Cotton"]["avg"], list)
    assert _bits(_cycles(export["data"])) == _bits(_cycles(_v18_store()))


def test_import_unwrap_accepts_old_exports_and_compact_store_files() -> None:
    old_export = {"version": 18, "data": _v18_store()}
    data_old, _ = unwrap_import_payload(old_export)
    assert _bits(_cycles(data_old)) == _bits(_cycles(_v18_store()))

    store_file = _as_stored({
        "version": 19, "minor_version": 1, "key": MAIN,
        "data": trace_codec.encode_store_payload(_v18_store()),
    })
    payload = copy.deepcopy(store_file)
    data_new, meta = unwrap_import_payload(payload)
    assert meta["version"] == 19
    assert _bits(_cycles(data_new)) == _bits(_cycles(_v18_store()))
    assert payload == store_file  # the payload itself is not changed


async def test_import_of_old_and_new_exports_stores_identical_traces(hass: HomeAssistant) -> None:
    src = ProfileStore(hass, ENTRY)
    await src.async_load()
    src._data = _v18_store()  # noqa: SLF001
    new_export = json.loads(json.dumps(src.export_data()))
    old_export = {"version": 18, "data": _v18_store()}
    raw_file = _as_stored({"version": 19, "data": trace_codec.encode_store_payload(_v18_store())})

    results = []
    envelopes: list[Any] = []
    for i, payload in enumerate((old_export, new_export, raw_file)):
        dst = ProfileStore(hass, f"01KV19IMPORT{i:014d}")
        await dst.async_load()
        await dst.async_import_data(payload)
        results.append({c["id"]: c.get("power_data") for c in dst.iter_stored_cycles()})
        envelopes.append(dst._data["envelopes"])  # noqa: SLF001
    assert results[0] == results[1] == results[2]
    assert results[0]["p1"] == _trace()
    assert all(not isinstance(v, str) for r in results for v in r.values())
    assert envelopes[0] == envelopes[1] == envelopes[2]
    assert all(isinstance(e, dict) for m in envelopes for e in m.values())


async def test_restore_point_is_compact_on_disk_and_restores_identical(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps = ProfileStore(hass, ENTRY)
    await ps.async_load()
    ps._data = _v18_store()  # noqa: SLF001
    await ps.async_save()
    before = copy.deepcopy(ps._data)  # noqa: SLF001
    assert await ps.async_save_pre_import_snapshot("test")
    slot = next(v for k, v in hass_storage.items() if k.endswith(".pre_import"))
    assert isinstance(slot["data"]["data"]["past_cycles"][0]["power_data"], str)
    assert isinstance(slot["data"]["data"]["envelopes"]["Cotton"], str)

    ps._data = {"profiles": {}, "past_cycles": [], "reference_cycles": [], "backfill_cycles": []}  # noqa: SLF001
    await ps.async_restore_pre_import_snapshot()
    assert _bits(_cycles(ps._data)) == _bits(_cycles(before))  # noqa: SLF001


def test_harness_loader_reads_a_raw_v19_store_file() -> None:
    # devtools/eval.py's `_unwrap` feeds every corpus harness; a live store file
    # handed to one must read the same as its export.
    path = Path(__file__).resolve().parents[1] / "devtools" / "eval.py"
    spec = importlib.util.spec_from_file_location("wd_eval_harness_v19", path)
    ev = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ev  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(ev)

    export = {"version": 19, "data": _v18_store(), "entry_options": {"device_type": "washing_machine"}}
    store_file = _as_stored({
        "version": 19, "minor_version": 1, "key": MAIN,
        "data": trace_codec.encode_store_payload(_v18_store()),
    })
    from_export, from_file = ev._unwrap(export), ev._unwrap(store_file)  # noqa: SLF001
    assert from_export[3] == from_file[3] == "export"
    assert _bits(_cycles(from_file[0])) == _bits(_cycles(from_export[0]))


def test_a_field_of_the_wrong_kind_is_written_as_it_is() -> None:
    """``debug_data`` is a dict; a list there (an import) was encoded, and decoding,
    which accepts only a dict, left it as JSON text after the next load."""
    data = {"past_cycles": [{"id": "a", "debug_data": [1, 2], "power_data": [[0.0, 5.0]]}]}
    disk = trace_codec.encode_store_payload(data)
    assert disk["past_cycles"][0]["debug_data"] == [1, 2]
    assert isinstance(disk["past_cycles"][0]["power_data"], str)
    loaded = json_loads(json.dumps(disk))
    trace_codec.decode_store_payload(loaded)
    assert loaded["past_cycles"][0]["debug_data"] == [1, 2]
    assert loaded["past_cycles"][0]["power_data"] == [[0.0, 5.0]]
