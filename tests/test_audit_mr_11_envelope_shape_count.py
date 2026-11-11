"""Audit MR-11: the envelope template is gated on the cycles that SHAPED it.

``async_rebuild_envelope`` stores ``cycle_count = len(real cycles)`` as soon as any
reference or backfill cycle exists: a usage figure. ``build_match_snapshots`` used
the envelope only when that count reached 2, so a profile built from backfilled
history (#344) or store downloads, with 0-1 real cycles, matched against one sample
cycle and never against the envelope its cycles built. The gate (and its mirrors in
``_stage1_duration_for`` and ``unmatchable_profiles``) now reads
``shape_cycle_count``: the cycles actually in the curve.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata.profile_store import ProfileStore, envelope_shape_count


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.1, max_duration_ratio=1.8)
        ps._store.async_save = AsyncMock()  # noqa: SLF001
        yield ps


def _cycle(cid: str, name: str, k: int) -> dict:
    t = np.arange(0.0, 3600.0 + 1, 30.0)
    w = np.where(t < 900, 2000.0 + 40 * k, np.where(t < 3000, 180.0 + 10 * np.sin(t / 50 + k), 600.0))
    return {
        "id": cid, "profile_name": name, "status": "completed", "duration": 3600.0,
        "start_time": f"2026-01-0{k + 1}T08:00:00+00:00",
        "power_data": [[float(a), float(b)] for a, b in zip(t, w)],
    }


async def _backfill_only(store: ProfileStore, n_real: int) -> None:
    store._data["profiles"]["Eco"] = {"avg_duration": 3600.0, "sample_cycle_id": "b0"}  # noqa: SLF001
    store._data["backfill_cycles"] = [_cycle(f"b{k}", "Eco", k) for k in range(3)]  # noqa: SLF001
    store._data["past_cycles"] = [_cycle(f"r{k}", "Eco", 5 + k) for k in range(n_real)]  # noqa: SLF001
    assert await store.async_rebuild_envelope("Eco")


@pytest.mark.parametrize("n_real", [0, 1])
async def test_backfill_built_profile_matches_its_envelope(store: ProfileStore, n_real: int) -> None:
    await _backfill_only(store, n_real)
    env = store._data["envelopes"]["Eco"]  # noqa: SLF001
    assert env["cycle_count"] == n_real          # usage stays real-only
    assert env["shape_cycle_count"] == 3 + n_real
    assert envelope_shape_count(env) == 3 + n_real

    snap = next(s for s in store.build_match_snapshots(30.0) if s["name"] == "Eco")
    template = store._envelope_template_on_grid("Eco", env, 30.0)  # noqa: SLF001
    # The envelope's average curve, not the sample cycle b0 re-gridded.
    assert snap["sample_power"] == template


async def test_one_shaping_cycle_still_uses_the_sample(store: ProfileStore) -> None:
    store._data["profiles"]["Solo"] = {"avg_duration": 3600.0, "sample_cycle_id": "s0"}  # noqa: SLF001
    store._data["backfill_cycles"] = [_cycle("s0", "Solo", 0)]  # noqa: SLF001
    store._data["past_cycles"] = [_cycle("r0", "Other", 1), _cycle("r1", "Other", 2)]  # noqa: SLF001
    assert await store.async_rebuild_envelope("Solo")
    assert store._data["envelopes"]["Solo"]["shape_cycle_count"] == 1  # noqa: SLF001
    snap = next(s for s in store.build_match_snapshots(30.0) if s["name"] == "Solo")
    env = store._data["envelopes"]["Solo"]  # noqa: SLF001
    assert snap["sample_power"] != store._envelope_template_on_grid("Solo", env, 30.0)  # noqa: SLF001


def test_an_envelope_from_before_the_field_falls_back_to_cycle_count():
    assert envelope_shape_count({"cycle_count": 4}) == 4
    assert envelope_shape_count({"cycle_count": 1, "shape_cycle_count": 5}) == 5
    assert envelope_shape_count(None) == 0
    assert envelope_shape_count({"cycle_count": "x"}) == 0


async def test_the_template_cache_holds_only_the_current_envelope(store: ProfileStore) -> None:
    """#521: keyed by the envelope's id and grid step, the cache kept every old
    envelope and one template per jittered step until 512 entries piled up."""
    import weakref

    await _backfill_only(store, 1)

    class _Env(dict):  # a dict cannot be weak-referenced
        pass

    first = _Env(store._data["envelopes"]["Eco"])  # noqa: SLF001
    gone = weakref.ref(first)
    store._envelope_template_on_grid("Eco", first, 30.0)  # noqa: SLF001
    rebuilt = _Env(first, updated="later")
    del first
    for step in (30.0, 30.01, 30.02, 30.03, 30.04):   # a change-only plug's median
        store._envelope_template_on_grid("Eco", rebuilt, step)  # noqa: SLF001
    cache = store._envelope_template_cache  # noqa: SLF001
    assert gone() is None
    assert sorted(k[1] for k in cache if k[0] == "Eco") == [30.03, 30.04]
    assert store._envelope_template_on_grid("Eco", rebuilt, 30.04) is cache[("Eco", 30.04)][2]  # noqa: SLF001

