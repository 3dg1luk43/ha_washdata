"""Stage-5 profile groups: storage CRUD, membership upkeep, cohesion, and
near-duplicate suggestion. Pure-store unit tests (fast suite)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def store():
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        ps._data["profiles"] = {
            "A": {"avg_duration": 1000.0},
            "B": {"avg_duration": 1000.0},
            "C": {"avg_duration": 3000.0},
        }
        yield ps


def _ramp(n=30, scale=1.0, offset=0.0):
    return [[round(i * 10.0, 1), round(offset + scale * i, 3)] for i in range(n)]


# ── CRUD ────────────────────────────────────────────────────────────────────


async def test_group_crud(store):
    await store.create_profile_group("Cotton", ["A", "B", "C", "Ghost"])  # Ghost dropped
    assert set(store.get_profile_groups()["Cotton"]["members"]) == {"A", "B", "C"}

    await store.set_profile_group_members("Cotton", ["A", "B"])
    assert set(store.get_profile_groups()["Cotton"]["members"]) == {"A", "B"}

    await store.rename_profile_group("Cotton", "Cottons")
    groups = store.get_profile_groups()
    assert "Cottons" in groups and "Cotton" not in groups

    await store.set_profile_group_members("Cottons", [])  # emptying deletes it
    assert "Cottons" not in store.get_profile_groups()


async def test_delete_group(store):
    await store.create_profile_group("G", ["A", "B"])
    assert await store.delete_profile_group("G") is True
    assert "G" not in store.get_profile_groups()
    assert await store.delete_profile_group("G") is False


async def test_membership_pruned_when_profile_removed(store):
    await store.create_profile_group("G", ["A", "B"])
    del store._data["profiles"]["B"]
    assert store.get_profile_groups()["G"]["members"] == ["A"]


async def test_rename_profile_updates_group_membership(store):
    await store.create_profile_group("G", ["A", "B"])
    await store.update_profile("A", "A2")
    assert set(store.get_profile_groups()["G"]["members"]) == {"A2", "B"}


# ── cohesion ──────────────────────────────────────────────────────────────


def test_group_cohesion_tight_vs_loose(store):
    store._data["envelopes"] = {
        "A": {"avg": _ramp()},
        "B": {"avg": _ramp(scale=1.2)},        # same shape, scaled (temp variant)
        "C": {"avg": _ramp(scale=-1.0, offset=29.0)},  # reversed -> different shape
    }
    # DTW/peak-normalised similarity: amplitude scaling is tolerated (tight ~1.0);
    # a genuinely different (reversed) shape scores low even after warping.
    assert store.group_cohesion(["A", "B"]) > 0.95
    assert store.group_cohesion(["A", "C"]) < 0.6


def test_group_cohesion_single_member_is_one(store):
    store._data["envelopes"] = {"A": {"avg": _ramp()}}
    assert store.group_cohesion(["A"]) == 1.0


# ── suggestion ───────────────────────────────────────────────────────────────


def test_suggest_clusters_near_duplicates(store):
    store._data["profiles"] = {
        "Eco30": {"avg_duration": 1000.0},
        "Eco60": {"avg_duration": 1050.0},   # same shape + duration -> cluster with Eco30
        "Quick": {"avg_duration": 300.0},    # far shorter -> excluded
    }
    store._data["envelopes"] = {
        "Eco30": {"avg": _ramp()},
        "Eco60": {"avg": _ramp(scale=1.3)},
        "Quick": {"avg": _ramp(n=30, scale=1.0)},
    }
    sug = store.suggest_profile_groups()
    clusters = [set(s["members"]) for s in sug]
    assert any({"Eco30", "Eco60"} <= c for c in clusters)
    # Quick has a very different duration, so it is not grouped with the Eco pair.
    assert not any("Quick" in c and {"Eco30", "Eco60"} <= c for c in clusters)


def _snap(name, power, dur):
    return {"name": name, "avg_duration": float(dur), "sample_power": list(power)}


# ── group-aware matcher transform ──────────────────────────────────────────


def test_grouped_snapshots_collapses_cohesive_group(store):
    store._data["envelopes"] = {"A": {"avg": _ramp()}, "B": {"avg": _ramp(scale=1.2)}}
    store._data["profile_groups"] = {"G": {"members": ["A", "B"]}}
    snaps = [
        _snap("A", [float(i) for i in range(30)], 1000),
        _snap("B", [i * 1.2 for i in range(30)], 1000),
        _snap("C", [100.0] * 30, 3000),
    ]
    out, gm, ms = store._grouped_snapshots(snaps)
    names = [s["name"] for s in out]
    assert "C" in names                       # ungrouped stays individual
    assert "A" not in names and "B" not in names  # cohesive members collapsed
    key = next(n for n in names if n.startswith("__group__"))
    assert set(gm[key]) == {"A", "B"}
    assert set(ms) == {"A", "B"}


def test_grouped_snapshots_skips_loose_group(store):
    # Anti-correlated envelopes -> cohesion below threshold -> NOT collapsed.
    store._data["envelopes"] = {"A": {"avg": _ramp()}, "C": {"avg": _ramp(scale=-1.0, offset=29.0)}}
    store._data["profile_groups"] = {"G": {"members": ["A", "C"]}}
    snaps = [_snap("A", [float(i) for i in range(30)], 1000), _snap("C", [float(29 - i) for i in range(30)], 1000)]
    out, gm, _ms = store._grouped_snapshots(snaps)
    names = [s["name"] for s in out]
    assert "A" in names and "C" in names
    assert gm == {}


def test_grouped_snapshots_no_groups_is_noop(store):
    snaps = [_snap("A", [1.0, 2.0, 3.0, 4.0], 1000)]
    out, gm, ms = store._grouped_snapshots(snaps)
    assert out == snaps and gm == {} and ms == {}


def test_stage5_picks_member_by_mean_power(store):
    ms = {"Hot": _snap("Hot", [2000.0] * 30, 1000), "Cold": _snap("Cold", [500.0] * 30, 1000)}
    chosen, fit, dur = store._stage5_pick_member([1900.0] * 30, 1000.0, ["Hot", "Cold"], ms)
    assert chosen == "Hot"
    assert dur == 1000.0
    assert fit is not None


def test_suggest_skips_already_grouped(store):
    store._data["profiles"] = {"E1": {"avg_duration": 1000.0}, "E2": {"avg_duration": 1010.0}}
    store._data["envelopes"] = {"E1": {"avg": _ramp()}, "E2": {"avg": _ramp(scale=1.1)}}
    store._data["profile_groups"] = {"Eco": {"members": ["E1", "E2"]}}
    # Both already in a group -> nothing new to suggest.
    assert store.suggest_profile_groups() == []
