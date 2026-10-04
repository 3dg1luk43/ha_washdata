"""Audit 2026-10-02 ML-08 / ML-09 / ML-12 / PLAYGROUND-22: envelope and cache robustness.

ML-08: two overlapping rebuilds of one profile - the older finished last and
overwrote the newer, and a rebuild straddling a delete wrote an envelope under the
dead name.
ML-09: deleting a profile left its envelope behind, and nothing pruned orphans.
ML-12: the resampled sample-segment cache grew without bound (the grid step drifts
on change-only plugs, and register item 463 keeps every cycle).
PLAYGROUND-22: the Playground inserts into that cache from an executor thread while
loop code walked it live ("dictionary changed size during iteration").
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.profile_store import (
    SAMPLE_SEGMENT_CACHE_MIN,
    ProfileStore,
)

_BASE = dt_util.parse_datetime("2026-09-01T08:00:00+00:00")


def _trace(k: int) -> list[list[float]]:
    """A 30 min wash: heat block, agitation, spin; ``k`` shifts the level a little."""
    pts: list[list[float]] = []
    for t in range(0, 1800 + 1, 30):
        if t < 600:
            w = 1800.0 + 20 * k
        elif t < 1500:
            w = 150.0 + 40 * ((t // 120) % 3)
        else:
            w = 500.0 if (t // 60) % 2 else 50.0
        pts.append([float(t), w])
    return pts


def _seed(ps: ProfileStore, name: str = "A", n: int = 3) -> None:
    assert _BASE is not None
    last: dict[str, Any] = {}
    for k in range(n):
        pts = _trace(k)
        start = _BASE + timedelta(days=k)
        last = {
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(seconds=pts[-1][0])).isoformat(),
            "duration": pts[-1][0], "status": "completed",
            "profile_name": name, "power_data": pts,
        }
        ps._add_cycle_data(last)  # noqa: SLF001
    ps.get_profiles()[name] = {"avg_duration": last["duration"], "sample_cycle_id": last["id"]}


class _GatedHass:
    """The real hass, except that each envelope build waits for the test to release it."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.gates: list[asyncio.Event] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def async_add_executor_job(self, fn: Any, *args: Any) -> Any:
        if getattr(fn, "__name__", "") != "_rebuild_envelope_sync":
            return await self._real.async_add_executor_job(fn, *args)
        result = fn(*args)
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        return result


async def _until(cond: Any) -> None:
    for _ in range(200):
        if cond():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true")


# --------------------------------------------------------------------------- ML-08


async def test_an_older_envelope_build_finishing_last_does_not_overwrite_the_newer(hass):
    ps = ProfileStore(hass, "ml08a")
    _seed(ps, n=3)
    gated = _GatedHass(hass)
    ps.hass = gated

    first = asyncio.create_task(ps.async_rebuild_envelope("A"))  # sees 3 cycles
    await _until(lambda: len(gated.gates) == 1)
    ps.get_past_cycles()[0]["profile_name"] = None  # the user unlabels one
    second = asyncio.create_task(ps.async_rebuild_envelope("A"))  # sees 2
    await _until(lambda: len(gated.gates) == 2)

    gated.gates[1].set()
    assert await second is True
    gated.gates[0].set()
    await first

    assert ps.get_envelope("A")["cycle_count"] == 2


async def test_a_build_straddling_a_delete_does_not_resurrect_the_envelope(hass):
    ps = ProfileStore(hass, "ml08b")
    _seed(ps, n=3)
    gated = _GatedHass(hass)
    ps.hass = gated

    build = asyncio.create_task(ps.async_rebuild_envelope("A"))
    await _until(lambda: len(gated.gates) == 1)
    await ps.delete_profile("A", unlabel_cycles=False)
    gated.gates[0].set()
    await build

    assert ps.get_envelope("A") is None
    assert "A" not in ps._data.get("envelopes", {})  # noqa: SLF001


# --------------------------------------------------------------------------- ML-09


async def test_deleting_a_profile_removes_its_envelope(hass):
    ps = ProfileStore(hass, "ml09a")
    _seed(ps, n=3)
    assert await ps.async_rebuild_envelope("A") is True
    assert ps.get_envelope("A") is not None

    await ps.delete_profile("A")

    assert "A" not in ps._data.get("envelopes", {})  # noqa: SLF001


async def test_maintenance_prunes_envelopes_of_missing_profiles(hass):
    ps = ProfileStore(hass, "ml09b")
    _seed(ps, n=3)
    assert await ps.async_rebuild_envelope("A") is True
    # Left behind by a delete from before the fix.
    ps._data["envelopes"]["Gone"] = dict(ps.get_envelope("A"))  # noqa: SLF001

    stats = await ps.async_run_maintenance()

    assert stats["pruned_envelopes"] == 1
    assert set(ps._data["envelopes"]) == {"A"}  # noqa: SLF001


# --------------------------------------------------------------------------- ML-12


def test_the_sample_segment_cache_is_bounded_and_keeps_what_is_used(hass):
    ps = ProfileStore(hass, "ml12")
    cycle = {"id": "c1", "power_data": _trace(0)}
    hot = ps._get_cached_sample_segment(cycle, 5.0)  # noqa: SLF001
    assert hot is not None

    # Live matching re-grids on every drift of the plug's cadence.
    for i in range(1, 300):
        assert ps._get_cached_sample_segment(cycle, 5.0 + 0.05 * i) is not None  # noqa: SLF001
        assert ps._get_cached_sample_segment(cycle, 5.0) is hot  # noqa: SLF001

    cache = ps._cached_sample_segments  # noqa: SLF001
    assert len(cache) == SAMPLE_SEGMENT_CACHE_MIN
    assert ("c1", 5.0) in cache


def test_the_cache_bound_grows_with_the_profile_count(hass):
    ps = ProfileStore(hass, "ml12b")
    for i in range(100):
        ps.get_profiles()[f"P{i}"] = {}
    assert ps._sample_segment_cache_cap() == 200  # noqa: SLF001


# --------------------------------------------------------------------- PLAYGROUND-22


def _racing_cache(ps: ProfileStore) -> dict[Any, Any]:
    """A cache whose first key lands an insert while it is inspected.

    Deterministic stand-in for the Playground's executor thread inserting a
    segment while loop code is part-way through walking the cache.
    """

    class _InsertingKey(tuple):
        def __getitem__(self, i: Any) -> Any:
            ps._cached_sample_segments[("other", 9.99)] = "inserted"  # noqa: SLF001
            return tuple.__getitem__(self, i)

    return {_InsertingKey(("c1", 5.0)): "stale", ("c2", 5.0): "keep"}


def test_repairing_a_cycle_survives_a_concurrent_cache_insert(hass):
    ps = ProfileStore(hass, "pg22a")
    ps._cached_sample_segments = _racing_cache(ps)  # noqa: SLF001
    cycle = {"id": "c1", "duration": 1800.0, "power_data": _trace(0)}

    ps._apply_repaired_duration(cycle, 1500.0)  # noqa: SLF001

    assert set(ps._cached_sample_segments) == {("c2", 5.0), ("other", 9.99)}  # noqa: SLF001


async def test_trimming_a_cycle_survives_a_concurrent_cache_insert(hass):
    ps = ProfileStore(hass, "pg22b")
    _seed(ps, n=1)
    cid = ps.get_past_cycles()[0]["id"]
    racing = _racing_cache(ps)
    ps._cached_sample_segments = {  # noqa: SLF001
        type(next(iter(racing)))((cid, 5.0)): "stale", ("c2", 5.0): "keep",
    }

    assert await ps.trim_cycle_power_data(cid, 0.0, 1500.0) is True

    assert set(ps._cached_sample_segments) == {("c2", 5.0), ("other", 9.99)}  # noqa: SLF001
