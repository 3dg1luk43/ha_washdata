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
"""Phase A: imported reference-cycle storage isolation.

Reference cycles live in a separate `reference_cycles` list. They shape the envelope
and can serve as a matching template, but must never touch usage/energy/count stats.
"""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        yield ps


def _iso_trace(watts, n=61, dur=3600):
    step = dur / (n - 1)
    return [((BASE + timedelta(seconds=i * step)).isoformat(), float(watts)) for i in range(n)]


def _offset_trace(watts, n=61, dur=3600):
    step = dur / (n - 1)
    return [[i * step, float(watts)] for i in range(n)]


async def _add_real(store, profile, watts, dur=3600):
    await store.async_add_cycle({
        "start_time": BASE.isoformat(),
        "duration": dur,
        "status": "completed",
        "profile_name": profile,
        "power_data": _iso_trace(watts, dur=dur),
    })


# --- A2 ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_reference_cycle_is_isolated(store):
    base_kwh = store.get_lifetime_energy_wh()
    base_past = len(store.get_past_cycles())
    await store.add_reference_cycle(
        "Cotton 40", _offset_trace(2000), {"store_cycle_id": "x1", "sampling_interval": 60}
    )
    assert store.get_lifetime_energy_wh() == base_kwh           # lifetime untouched
    assert len(store.get_past_cycles()) == base_past            # NOT in past_cycles
    refs = store.get_reference_cycles()
    assert len(refs) == 1
    assert refs[0]["meta"]["source"] == "store:x1"
    assert refs[0]["ml_review"]["golden"] is True
    assert refs[0]["status"] == "completed"


# --- A3 ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reference_shapes_envelope_but_not_energy_or_count(store):
    await _add_real(store, "Cotton 40", 1000)
    await store.async_rebuild_envelope("Cotton 40")
    env0 = store.get_envelope("Cotton 40")
    assert env0 and env0["cycle_count"] == 1

    # A reference cycle with a very different level (3000 W) - must not inflate energy/count.
    await store.add_reference_cycle("Cotton 40", _offset_trace(3000), {"store_cycle_id": "r1"})
    env1 = store.get_envelope("Cotton 40")

    assert env1["cycle_count"] == 1                              # real-only count
    assert env1["avg_energy"] < 1.6                              # ~1 kWh real, NOT ~2 (mean) / ~3 (ref)
    assert env1["avg"] != env0["avg"]                            # shape DID move (ref included)


# --- A4 ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_import_only_profile_matches(store):
    # No real cycles: seed a profile purely from an imported reference cycle.
    ramp = [[float(i), float(i)] for i in range(101)]  # 0..100 W over 100 s
    await store.add_reference_cycle("Eco 50", ramp, {"store_cycle_id": "r2"})
    assert len(store.get_past_cycles()) == 0
    assert store.get_envelope("Eco 50") is not None

    current = [((BASE + timedelta(seconds=i)).isoformat(), float(i)) for i in range(101)]
    result = await store.async_match_profile(current, 100.0)
    assert result.best_profile == "Eco 50"


# --- A5 ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reference_cycles_excluded_from_past_cycle_analytics(store):
    from custom_components.ha_washdata.suggestion_engine import select_clean_cycles
    await store.add_reference_cycle("Cotton 40", _offset_trace(2000), {"store_cycle_id": "r3"})
    # Anything that reads past_cycles must not see the reference cycle.
    assert store.get_past_cycles() == []
    clean = select_clean_cycles(store.get_past_cycles())
    clean_list = clean[0] if isinstance(clean, tuple) else clean
    assert clean_list == []


@pytest.mark.asyncio
async def test_export_import_round_trips_reference_cycles(store):
    await store.add_reference_cycle("Cotton 40", _offset_trace(2000), {"store_cycle_id": "r4"})
    exported = dict(store._data)  # store export is the raw data dict
    payload = {"version": 2, "data": exported}
    # Fresh store imports it
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps2 = ProfileStore(store.hass, "e2", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps2._store.async_load = AsyncMock(return_value=None)
        ps2._store.async_save = AsyncMock()
        await ps2.async_import_data(payload)
    assert len(ps2.get_reference_cycles()) == 1
    assert ps2.get_reference_cycles()[0]["meta"]["source"] == "store:r4"


# --- Panel Cycles-tab exposure (view + delete a bad import) ------------------

@pytest.mark.asyncio
async def test_get_cycle_power_data_finds_reference_cycle(store):
    cid = await store.add_reference_cycle("Cotton 40", _offset_trace(2000), {"store_cycle_id": "v1"})
    # Not in past_cycles, but the inspector must still be able to load its curve.
    assert store.get_past_cycles() == []
    samples = store.get_cycle_power_data(cid)
    assert samples and len(samples) > 1
    assert all(len(p) == 2 for p in samples)


@pytest.mark.asyncio
async def test_delete_cycle_removes_reference_cycle_and_rebuilds(store):
    # A real cycle plus an imported one on the same profile.
    await _add_real(store, "Cotton 40", 1000)
    await store.async_rebuild_envelope("Cotton 40")
    cid = await store.add_reference_cycle("Cotton 40", _offset_trace(3000), {"store_cycle_id": "d1"})
    env_before = store.get_envelope("Cotton 40")["avg"]
    assert len(store.get_reference_cycles()) == 1

    # delete_cycle routes an unknown id to the reference list.
    ok = await store.delete_cycle(cid)
    assert ok is True
    assert store.get_reference_cycles() == []
    # Real cycle untouched; envelope rebuilt back toward the real-only shape.
    assert len(store.get_past_cycles()) == 1
    assert store.get_envelope("Cotton 40")["avg"] != env_before


@pytest.mark.asyncio
async def test_delete_cycle_unknown_id_returns_false(store):
    assert await store.delete_cycle("nope-not-here") is False


@pytest.mark.asyncio
async def test_relabel_reference_cycle_moves_template_and_stays_isolated(store):
    # Two profiles exist (add_reference_cycle auto-creates the profile entry).
    await store.add_reference_cycle("Eco 50", _offset_trace(900), {"store_cycle_id": "l0"})
    cid = await store.add_reference_cycle("Cotton 40", _offset_trace(2000), {"store_cycle_id": "l1"})
    moved = next(c for c in store.get_reference_cycles() if c["id"] == cid)
    assert moved["profile_name"] == "Cotton 40"

    # Bulk relabel routes through assign_profile_to_cycle; reference cycles move too.
    await store.assign_profile_to_cycle(cid, "Eco 50")
    refs = store.get_reference_cycles()
    assert len(refs) == 2                          # still reference cycles, none promoted
    moved = next(c for c in refs if c["id"] == cid)
    assert moved["profile_name"] == "Eco 50"       # moved to the new profile
    assert all(c["id"] != cid for c in store.get_past_cycles())  # never in past_cycles


@pytest.mark.asyncio
async def test_relabel_reference_cycle_unknown_profile_raises(store):
    cid = await store.add_reference_cycle("Cotton 40", _offset_trace(2000), {"store_cycle_id": "l2"})
    with pytest.raises(ValueError):
        await store.assign_profile_to_cycle(cid, "Does Not Exist")


# --- Maintenance must not corrupt import-only profiles (review findings #1/#2) ---

@pytest.mark.asyncio
async def test_repair_does_not_steal_real_cycle_into_import_only_profile(store):
    # An UNLABELED real cycle plus an import-only profile seeded from a reference cycle.
    await store.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "power_data": _iso_trace(1000),
    })
    await store.add_reference_cycle("Eco 50", _offset_trace(2000), {"store_cycle_id": "rr1"})
    assert store.get_past_cycles()[0].get("profile_name") in (None, "")
    assert store.get_profiles()["Eco 50"].get("sample_cycle_id")  # sample points at the ref cycle

    stats = await store.async_repair_profile_samples()
    # The imported profile's sample resolves in reference_cycles, so repair leaves the
    # unlabeled real cycle untouched (previously it was stolen into "Eco 50").
    assert store.get_past_cycles()[0].get("profile_name") in (None, "")
    assert stats["cycles_labeled_as_sample"] == 0


@pytest.mark.asyncio
async def test_cleanup_keeps_import_only_profile(store):
    await store.add_reference_cycle("Eco 50", _offset_trace(2000), {"store_cycle_id": "rr2"})
    assert "Eco 50" in store.get_profiles()
    # Import-only profile's sample is a reference cycle -> not an orphan.
    assert store.cleanup_orphaned_profiles() == 0
    assert "Eco 50" in store.get_profiles()


@pytest.mark.asyncio
async def test_delete_last_reference_cycle_removes_empty_profile_no_theft(store):
    # Unlabeled real cycle + an import-only profile whose ONLY cycle is the ref.
    await store.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "power_data": _iso_trace(1000),
    })
    cid = await store.add_reference_cycle("Eco 50", _offset_trace(2000), {"store_cycle_id": "d2"})
    assert "Eco 50" in store.get_profiles()

    # Deleting the last (imported) cycle removes the now-empty profile outright,
    # so no dangling sample survives for maintenance to mishandle.
    assert await store.delete_cycle(cid) is True
    assert "Eco 50" not in store.get_profiles()

    # And repair can no longer steal the unlabeled real cycle into a phantom profile.
    stats = await store.async_repair_profile_samples()
    assert store.get_past_cycles()[0].get("profile_name") in (None, "")
    assert stats["cycles_labeled_as_sample"] == 0


@pytest.mark.asyncio
async def test_playground_snapshots_include_imported_profile(store):
    # An import-only profile must be a Playground match candidate, else auto-detect
    # replays never match a downloaded profile (issue: _build_match_snapshots resolved
    # sample_cycle_id against past_cycles only).
    from custom_components.ha_washdata import playground
    await store.add_reference_cycle("Eco 50", _offset_trace(2000), {"store_cycle_id": "pg1"})
    assert store.get_past_cycles() == []
    snaps, _cfg, _gm, _ms = playground._build_match_snapshots(store)
    assert "Eco 50" in {s["name"] for s in snaps}


@pytest.mark.asyncio
async def test_delete_reference_cycle_keeps_profile_with_other_cycles(store):
    # Profile has a real cycle AND an imported one; deleting the import keeps it.
    await _add_real(store, "Cotton 40", 1000)
    cid = await store.add_reference_cycle("Cotton 40", _offset_trace(3000), {"store_cycle_id": "d3"})
    assert await store.delete_cycle(cid) is True
    assert "Cotton 40" in store.get_profiles()
    assert store.get_reference_cycles() == []


# --- Addendum A: shareable-cycle enumeration -------------------------------

@pytest.mark.asyncio
async def test_get_shareable_cycles_only_golden(store):
    # A recorder-golden cycle, a hand-flagged golden cycle, a plain cycle, and an
    # imported reference cycle. Only the two golden PAST cycles are shareable.
    await store.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "profile_name": "Cotton 40", "power_data": _iso_trace(2000),
        "meta": {"source": "recorder"},
    })
    await store.async_add_cycle({
        "start_time": (BASE + timedelta(days=1)).isoformat(), "duration": 3600, "status": "completed",
        "profile_name": "Eco 60", "power_data": _iso_trace(1800),
        "ml_review": {"golden": True},
    })
    await store.async_add_cycle({
        "start_time": (BASE + timedelta(days=2)).isoformat(), "duration": 2700, "status": "completed",
        "profile_name": "Quick 30", "power_data": _iso_trace(1500),
    })
    await store.add_reference_cycle("Imported", _offset_trace(1200), {"store_cycle_id": "z1"})

    items = store.get_shareable_cycles()
    progs = {i["profile_name"] for i in items}
    assert progs == {"Cotton 40", "Eco 60"}          # plain + imported excluded
    assert all(i.get("id") for i in items)
    # Most-recent-first ordering.
    assert items[0]["profile_name"] == "Eco 60"


# --- Naming a program from an imported cycle (issue #344) --------------------
#
# A historical import (issue #344) writes UNLABELLED cycles straight into
# `reference_cycles` via `_add_cycle_data(target=...)`, then asks the user to name the
# programs it found. That makes `create_profile` from a reference cycle the primary
# path for that feature, where before it raised "Cycle not found".

async def _add_imported(store, watts, *, start=BASE, dur=3600):
    """An unlabelled imported-history cycle, written the way the importer writes them."""
    cycle = {
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=dur)).isoformat(),
        "duration": dur,
        "status": "completed",
        "power_data": _offset_trace(watts, dur=dur),
        "meta": {"source": "history_import"},
    }
    store._add_cycle_data(cycle, target=store._data.setdefault("reference_cycles", []))
    return cycle["id"]


@pytest.mark.asyncio
async def test_create_profile_from_imported_cycle(store):
    cid = await _add_imported(store, 1500)
    assert next(c for c in store.get_reference_cycles() if c["id"] == cid)["profile_name"] is None

    await store.create_profile("Cotton 40", cid)

    profile = store._data["profiles"]["Cotton 40"]
    assert profile["sample_cycle_id"] == cid
    assert profile["avg_duration"] == 3600
    # Labelled in place: still a reference cycle, never promoted into usage stats.
    refs = store.get_reference_cycles()
    assert len(refs) == 1
    assert refs[0]["profile_name"] == "Cotton 40"
    assert store.get_past_cycles() == []
    assert store.get_lifetime_cycle_count() == 0
    assert store.get_lifetime_energy_wh() == 0
    # Usable immediately: the envelope is built here, not left to the next match pass.
    assert store.get_envelope("Cotton 40")


@pytest.mark.asyncio
async def test_create_profile_from_real_cycle_also_builds_the_envelope(store):
    await store.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "power_data": _iso_trace(2000),
    })
    cid = store.get_past_cycles()[0]["id"]

    await store.create_profile("Eco 60", cid)

    assert store.get_past_cycles()[0]["profile_name"] == "Eco 60"
    assert store.get_envelope("Eco 60")


@pytest.mark.asyncio
async def test_create_profile_unknown_cycle_still_raises(store):
    with pytest.raises(ValueError):
        await store.create_profile("Nope", "does-not-exist")


# --- Panel capability flags -------------------------------------------------
#
# `is_reference` used to be computed two different ways: by list membership in the cycle
# list, and from `meta.source.startswith("store")` in the inspector. With a
# `history_import` source those disagreed, and the inspector offered trim/split tools
# whose store functions only know about `past_cycles` and would silently no-op.

def test_cycle_capabilities_are_derived_from_list_membership():
    from custom_components.ha_washdata.ws_api import _cycle_capabilities

    real = _cycle_capabilities({"meta": {"source": "recorder"}}, "past")
    assert real == {"is_reference": False, "labelable": True, "editable": True}

    downloaded = _cycle_capabilities({"meta": {"source": "store:abc"}}, "reference")
    assert downloaded["is_reference"] is True
    assert downloaded["cycle_origin"] == "reference"
    # Non-real cycles are labelable (that is how a program gets named) but not editable.
    assert downloaded["labelable"] is True
    assert downloaded["editable"] is False

    backfilled = _cycle_capabilities({"meta": {"source": "history_import"}}, "backfill")
    assert backfilled["cycle_origin"] == "backfill"
    assert backfilled["labelable"] is True
    assert backfilled["editable"] is False

    # An unknown origin is treated as non-real rather than as an editable real cycle.
    assert _cycle_capabilities({}, "")["cycle_origin"] == "reference"


# --- The third category: backfilled history cycles (issue #344) --------------
#
# `backfill_cycles` holds cycles recovered by replaying raw power history. They are
# auto-detected and unverified, so they are deliberately neither `past_cycles` (lifetime
# stats, ML training labels, feedback queue, retention eviction) nor `reference_cycles`
# (curated store templates, golden by construction). They DO shape envelopes and matching
# once labelled - that is the point of importing them.

async def _add_backfill(store, watts, *, profile=None, start=BASE, dur=3600):
    cycle = {
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=dur)).isoformat(),
        "duration": dur,
        "status": "completed",
        "power_data": _offset_trace(watts, dur=dur),
        "meta": {"source": "history_import"},
    }
    if profile:
        cycle["profile_name"] = profile
    store._add_cycle_data(cycle, target=store._data.setdefault("backfill_cycles", []))
    return cycle["id"]


@pytest.mark.asyncio
async def test_backfill_cycles_are_isolated_from_usage_stats(store):
    await _add_backfill(store, 1800, profile="Cotton 40")
    store._data["profiles"]["Cotton 40"] = {"avg_duration": 3600}

    assert len(store.get_backfill_cycles()) == 1
    assert store.get_past_cycles() == []
    assert store.get_reference_cycles() == []
    assert store.get_lifetime_cycle_count() == 0
    assert store.get_lifetime_energy_wh() == 0
    # Never offered to the community store, whatever its flags say.
    assert store.get_shareable_cycles() == []


@pytest.mark.asyncio
async def test_labelled_backfill_cycle_shapes_the_envelope(store):
    store._data["profiles"]["Cotton 40"] = {"avg_duration": 3600}
    await _add_backfill(store, 1800, profile="Cotton 40")

    assert await store.async_rebuild_envelope("Cotton 40") is True
    envelope = store.get_envelope("Cotton 40")
    assert envelope
    # The curve is built from it, but usage stats stay at zero: cycle_count counts
    # only real observed cycles.
    assert envelope.get("cycle_count") == 0


@pytest.mark.asyncio
async def test_profile_gc_does_not_delete_a_backfill_only_profile(store):
    """The failure mode a forgotten read path causes.

    `cleanup_orphaned_profiles` deletes any profile whose `sample_cycle_id` resolves to
    no cycle. If it does not know about `backfill_cycles`, an import-only profile is
    destroyed on the next maintenance run.
    """
    cid = await _add_backfill(store, 1800)
    await store.create_profile("Cotton 40", cid)

    assert store.cleanup_orphaned_profiles() == 0
    assert "Cotton 40" in store._data["profiles"]


@pytest.mark.asyncio
async def test_sample_repair_does_not_steal_a_real_cycle_into_a_backfill_profile(store):
    cid = await _add_backfill(store, 1800)
    await store.create_profile("Cotton 40", cid)
    # An unlabelled real cycle exists and must stay unlabelled.
    await store.async_add_cycle({
        "start_time": (BASE + timedelta(days=1)).isoformat(), "duration": 3600,
        "status": "completed", "power_data": _iso_trace(2000),
    })

    await store.async_repair_profile_samples()

    assert store._data["profiles"]["Cotton 40"]["sample_cycle_id"] == cid
    assert store.get_past_cycles()[0]["profile_name"] is None


@pytest.mark.asyncio
async def test_backfill_cycle_is_matchable_and_labelable(store):
    cid = await _add_backfill(store, 1800)
    assert store.has_real_profiles is False           # nothing labelled yet

    await store.create_profile("Cotton 40", cid)
    assert store.has_real_profiles is True            # an import-only install can match

    # Relabelling moves it between profiles and keeps it out of past_cycles.
    store._data["profiles"]["Eco 60"] = {"avg_duration": 3600}
    await store.assign_profile_to_cycle(cid, "Eco 60")
    assert store.get_backfill_cycles()[0]["profile_name"] == "Eco 60"
    assert store.get_past_cycles() == []


@pytest.mark.asyncio
async def test_deleting_a_backfill_cycle_drops_its_now_empty_profile(store):
    cid = await _add_backfill(store, 1800)
    await store.create_profile("Cotton 40", cid)

    assert await store.delete_cycle(cid) is True
    assert store.get_backfill_cycles() == []
    # Mirrors the reference-cycle path: an empty profile would otherwise be re-populated
    # by sample repair stealing an unrelated real cycle.
    assert "Cotton 40" not in store._data["profiles"]


@pytest.mark.asyncio
async def test_renaming_and_deleting_a_profile_cascades_to_backfill_cycles(store):
    cid = await _add_backfill(store, 1800)
    await store.create_profile("Cotton 40", cid)

    await store.update_profile("Cotton 40", new_name="Cotton 60")
    assert store.get_backfill_cycles()[0]["profile_name"] == "Cotton 60"

    await store.delete_profile("Cotton 60", unlabel_cycles=True)
    assert store.get_backfill_cycles()[0]["profile_name"] is None


@pytest.mark.asyncio
async def test_backfill_count_is_reported_separately_from_cycle_count(store):
    cid = await _add_backfill(store, 1800)
    await store.create_profile("Cotton 40", cid)

    profile = next(p for p in store.list_profiles() if p["name"] == "Cotton 40")
    assert profile["cycle_count"] == 0        # no real observed cycles
    assert profile["backfill_count"] == 1
    assert profile["is_imported"] is False   # not a community-store template


@pytest.mark.asyncio
async def test_full_export_carries_backfill_cycles(store):
    await _add_backfill(store, 1800, profile="Cotton 40")
    payload = store.export_data()
    assert len(payload["data"]["backfill_cycles"]) == 1


@pytest.mark.asyncio
async def test_clear_all_data_wipes_backfill_cycles(store):
    await _add_backfill(store, 1800)
    await store.clear_all_data()
    assert store.get_backfill_cycles() == []
