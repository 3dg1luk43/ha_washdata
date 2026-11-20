"""Review of PR #529, round 41b, each reproduced.

* A skipped write of a retired store (its entry unloaded) logged through the
  module logger, without the appliance it belongs to.
* An import or an Undo Import the unload retired the store under (its final
  save skipped, debounced or not) reported success; Undo also deleted its
  restore point. The save they report raises now, before either.
* An import resumed after a reload wrote its restore point over the one a newer
  import on the replacement had saved, and Undo restored the older data.
* A profile whose stored ``avg_duration`` was not a number (an import can carry
  anything) raised in the matcher's candidate builder: no profile was scored.
* A trim of one cycle while a live match resampled its uncached templates in
  the executor discarded the whole batch, and the build resampled every one of
  them on the loop. Only the changed cycle's result goes now; a cache reset
  collects the misses again under the same budget, a bounded number of times.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import profile_store as ps_mod
from custom_components.ha_washdata.const import (
    DOMAIN,
    PRE_IMPORT_STORE_SUFFIX,
    STORAGE_KEY,
)
from custom_components.ha_washdata.profile_store import StoreRetiredError

from .test_pr529_review_r40a import (
    STEP,
    _query,
    _ranking,
    _single_cycle_store,
    _trace_of,
)


def _trace(scale: float = 1.0) -> list[list[float]]:
    return [[i * 10.0, scale * (2000.0 if 20 <= i < 60 else 150.0)] for i in range(200)]


def _cycle(cid: str, profile: str, scale: float = 1.0) -> dict[str, Any]:
    pts = _trace(scale)
    return {
        "id": cid, "profile_name": profile, "status": "completed",
        "start_time": f"2026-11-0{int(cid[-1]) + 1}T08:00:00+00:00",
        "duration": pts[-1][0], "power_data": pts,
    }


async def _seeded(hass: HomeAssistant, setup_washdata_entry: Any) -> Any:
    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["profiles"] = {"Cotton": {"avg_duration": 1990.0}}  # noqa: SLF001
    store._data["past_cycles"] = [_cycle(f"c{i}", "Cotton") for i in range(3)]  # noqa: SLF001
    await store.async_save()
    return entry, store


@asynccontextmanager
async def _reloaded_inside(
    hass: HomeAssistant, entry: Any, store: Any, method: str, *, coalesce: bool = False
) -> AsyncIterator[Any]:
    """Patch ``store.<method>`` so the first call waits; the body starts the
    operation as a task; on exit the entry is reloaded while it waits (the unload
    retires ``store``), then it resumes. Yields a holder for the task."""
    real = getattr(store, method)
    started, release = asyncio.Event(), asyncio.Event()

    async def gated(*args: Any, **kwargs: Any) -> Any:
        out = await real(*args, **kwargs)
        if not started.is_set():
            started.set()
            await release.wait()
        return out

    holder: dict[str, Any] = {}
    with patch.object(store, method, side_effect=gated):
        yield holder
        await asyncio.wait_for(started.wait(), 5)
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        assert hass.data[DOMAIN][entry.entry_id].profile_store is not store
        assert store._store.retired  # noqa: SLF001
        if coalesce:
            store.coalesce_saves()
        release.set()
        await asyncio.wait({holder["task"]}, timeout=5)

# ── A retired store's skipped write names its appliance ──────────────────────


async def test_a_skipped_write_of_a_retired_store_names_the_appliance(
    hass: HomeAssistant, setup_washdata_entry: Any, caplog: pytest.LogCaptureFixture
) -> None:
    entry = await setup_washdata_entry(title="Bosch Upstairs")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    assert await hass.config_entries.async_unload(entry.entry_id)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="custom_components.ha_washdata"):
        await store.async_save()
        store.coalesce_saves()
        await store.async_save()
    skipped = [r for r in caplog.records if "unloaded entry's store skipped" in r.getMessage()]
    assert len(skipped) == 2, caplog.text
    assert all(r.getMessage().startswith("[Bosch Upstairs] ") for r in skipped)
    assert all(getattr(r, "wd_device", None) == "Bosch Upstairs" for r in skipped)


# ── An import or undo the unload retired the store under is no success ───────


@pytest.mark.parametrize("coalesce", [False, True])
async def test_an_import_the_unload_retired_the_store_under_is_no_success(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any,
    hass_storage: dict[str, Any], coalesce: bool,
) -> None:
    """A wholesale import of Cotton's definition rebuilds Cotton from the kept
    cycles; the entry reloads meanwhile. Its final save was skipped (inside a
    save burst too) and the reply said success. Now the reply is an error and the
    file holds nothing of the import."""
    entry, store = await _seeded(hass, setup_washdata_entry)
    payload = {"version": 19, "data": {"profiles": {"Cotton": {"avg_duration": 1990.0, "mark": 1}}}}
    client = await hass_ws_client(hass)
    async with _reloaded_inside(
        hass, entry, store, "async_rebuild_envelope", coalesce=coalesce
    ) as held:
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/import_config", "entry_id": entry.entry_id,
            "json_data": json.dumps(payload),
        })
        held["task"] = asyncio.ensure_future(client.receive_json())
    res = held["task"].result()
    assert res["success"] is False, res
    assert "reloaded" in res["error"]["message"]
    on_disk = hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"]["data"]
    assert "mark" not in on_disk["profiles"]["Cotton"]


async def test_a_selective_import_the_unload_retired_the_store_under_raises(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_storage: dict[str, Any]
) -> None:
    entry, store = await _seeded(hass, setup_washdata_entry)
    payload = {"version": 19, "data": {
        "profiles": {"Eco": {"avg_duration": 1990.0}},
        "past_cycles": [_cycle("x1", "Eco", 0.5)],
    }}
    async with _reloaded_inside(hass, entry, store, "async_rebuild_envelope") as held:
        held["task"] = asyncio.ensure_future(store.async_import_data_selective(
            payload, selection={"categories": ["profiles", "real_cycles"]},
            local_device_type="washing_machine",
        ))
    with pytest.raises(StoreRetiredError):
        held["task"].result()
    assert "Eco" not in hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"]["data"]["profiles"]


async def test_an_undo_the_unload_retired_the_store_under_keeps_its_restore_point(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any,
    hass_storage: dict[str, Any],
) -> None:
    """Undo resumed after a reload: the restored store was never saved, the reply
    said success and the restore point was deleted. Now the reply is an error and
    the replacement still offers the undo."""
    entry, store = await _seeded(hass, setup_washdata_entry)
    await store.async_import_data(
        {"version": 19, "data": {"profiles": {"Eco": {}}, "past_cycles": [_cycle("x1", "Eco")]}}
    )
    point = f"{STORAGE_KEY}.{entry.entry_id}.{PRE_IMPORT_STORE_SUFFIX}"
    assert point in hass_storage
    client = await hass_ws_client(hass)
    async with _reloaded_inside(
        hass, entry, store, "_async_load_pre_import_snapshot"
    ) as held:
        await client.send_json_auto_id({"type": f"{DOMAIN}/undo_import", "entry_id": entry.entry_id})
        held["task"] = asyncio.ensure_future(client.receive_json())
    res = held["task"].result()
    assert res["success"] is False, res
    assert point in hass_storage
    assert "Cotton" not in hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"]["data"]["profiles"]
    new = hass.data[DOMAIN][entry.entry_id].profile_store
    snap = await new.async_get_pre_import_snapshot()
    assert snap is not None and snap["counts"]["real_cycles"] == 3


# ── An import resumed after a reload keeps the newer restore point ───────────


async def test_an_import_resumed_after_a_reload_keeps_the_newer_restore_point(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_storage: dict[str, Any]
) -> None:
    """Import A captured the store, the entry reloaded, import B on the
    replacement saved its restore point; A then wrote its own over it. Now A is
    refused before its write, and a retired store's discard leaves the slot."""
    entry, old = await _seeded(hass, setup_washdata_entry)
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    new = hass.data[DOMAIN][entry.entry_id].profile_store
    await new.async_import_data(
        {"version": 19, "data": {"profiles": {"Eco": {}}, "past_cycles": [_cycle("x1", "Eco")]}},
        source="newer",
    )
    point = f"{STORAGE_KEY}.{entry.entry_id}.{PRE_IMPORT_STORE_SUFFIX}"
    assert hass_storage[point]["data"]["source"] == "newer"

    with pytest.raises(StoreRetiredError):
        await old.async_import_data(
            {"version": 19, "data": {"profiles": {"Old": {}}, "past_cycles": [_cycle("y1", "Old")]}},
            source="older",
        )
    assert hass_storage[point]["data"]["source"] == "newer"
    await old.async_discard_pre_import_snapshot()
    assert hass_storage[point]["data"]["source"] == "newer"
    assert (await new.async_restore_pre_import_snapshot())["counts"]["real_cycles"] == 3
    assert set(new.get_profiles()) == {"Cotton"}


# ── A non-numeric stored profile length costs no match ───────────────────────


@pytest.mark.parametrize("bad", ["abc", [1], "nan", -5.0])
async def test_a_non_numeric_profile_length_costs_no_match(
    hass: HomeAssistant, bad: Any
) -> None:
    """P2's ``avg_duration`` unusable: it is absent, so its sample cycle's
    length stands in (the same here) and the ranking is the clean store's."""
    query = _query()
    clean = await _single_cycle_store(hass).async_match_profile(
        query, query[-1][0], in_progress=True
    )
    store = _single_cycle_store(hass)
    store._data["profiles"]["P2"]["avg_duration"] = bad  # noqa: SLF001
    result = await store.async_match_profile(query, query[-1][0], in_progress=True)
    assert result.best_profile == "P2"
    assert _ranking(result) == _ranking(clean)
    assert store.resolve_profile_duration("P2") == clean.expected_duration


def test_an_envelope_template_with_a_non_numeric_length_falls_back_to_its_span(
    hass: HomeAssistant,
) -> None:
    store = _single_cycle_store(hass)
    curve = [[i * 30.0, 150.0 + (i % 7) * 100.0] for i in range(101)]
    store._data["envelopes"]["P0"] = {"avg": curve, "shape_cycle_count": 2}  # noqa: SLF001
    store._data["profiles"]["P0"]["avg_duration"] = "abc"  # noqa: SLF001
    snaps = {s["name"]: s for s in store.build_match_snapshots(STEP)}
    assert set(snaps) == {f"P{k}" for k in range(5)}
    assert snaps["P0"]["avg_duration"] == 3000.0


# ── A change during the template job sends only what changed to the loop ─────


def _recording_jobs(mock_hass: Any, during: Any) -> dict[str, Any]:
    """Run executor jobs inline, calling ``during(n)`` after the n-th; record
    each template resampling as in or outside a job (outside = the loop)."""
    seen: dict[str, Any] = {"jobs": 0, "in_job": False, "loop": 0}

    async def job(target: Any, *args: Any) -> Any:
        seen["in_job"] = True
        try:
            out = target(*args)
        finally:
            seen["in_job"] = False
        seen["jobs"] += 1
        during(seen["jobs"])
        return out

    mock_hass.async_add_executor_job = job
    return seen


def _counting(seen: dict[str, Any]) -> Any:
    real = ps_mod.resample_uniform

    def counted(*args: Any, **kwargs: Any) -> Any:
        if not seen["in_job"]:
            seen["loop"] += 1
        return real(*args, **kwargs)

    return counted


async def test_a_trim_during_the_job_sends_only_that_cycle_to_the_loop(mock_hass: Any) -> None:
    store = _single_cycle_store(mock_hass)
    cycle = store._data["past_cycles"][0]  # noqa: SLF001
    trimmed = cycle["power_data"][:600]

    def trim(n: int) -> None:
        if n == 1:
            cycle["power_data"] = trimmed
            store._drop_cached_sample_segments("c0")  # noqa: SLF001

    seen = _recording_jobs(mock_hass, trim)
    with patch.object(ps_mod, "resample_uniform", side_effect=_counting(seen)):
        snaps = await store._async_match_snapshots(STEP)  # noqa: SLF001
    assert seen["jobs"] == 1 and seen["loop"] == 1
    by_name = {s["name"]: s["sample_power"] for s in snaps}
    assert by_name["P0"] == ps_mod._resample_sample_trace(trimmed, STEP).power.tolist()  # noqa: SLF001
    for k in range(1, 5):
        expected = ps_mod._resample_sample_trace(_trace_of(k), STEP)  # noqa: SLF001
        assert by_name[f"P{k}"] == expected.power.tolist()


@pytest.mark.parametrize(("resets", "jobs", "loop"), [(1, 2, 0), (99, 3, 5)])
async def test_a_cache_reset_during_the_job_collects_the_misses_again(
    mock_hass: Any, resets: int, jobs: int, loop: int
) -> None:
    """Reset once (an import): a second job, nothing on the loop. Reset under
    every job: three jobs, then the loop, as before."""
    store = _single_cycle_store(mock_hass)

    def reset(n: int) -> None:
        if n <= resets:
            store._cached_sample_segments = {}  # noqa: SLF001

    seen = _recording_jobs(mock_hass, reset)
    with patch.object(ps_mod, "resample_uniform", side_effect=_counting(seen)):
        snaps = await store._async_match_snapshots(STEP)  # noqa: SLF001
    assert (seen["jobs"], seen["loop"]) == (jobs, loop)
    assert [s["name"] for s in snaps] == [f"P{k}" for k in range(5)]
