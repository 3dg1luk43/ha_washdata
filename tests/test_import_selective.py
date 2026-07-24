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
"""Selective import apply: merge/replace, conflict resolution, cycle destination,
and the reference-vs-past isolation invariant."""
import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


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


def _fresh_store(mock_hass, entry_id="e"):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, entry_id, min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    return ps


@pytest.fixture
def store(mock_hass):
    return _fresh_store(mock_hass)


def _trace(watts, n=61, dur=3600):
    step = dur / (n - 1)
    return [[i * step, float(watts)] for i in range(n)]


def _payload(profiles=None, past=None, refs=None, device_type="washing_machine",
             extra=None, entry_options=None):
    data = {
        "profiles": profiles if profiles is not None else {},
        "past_cycles": past if past is not None else [],
        "reference_cycles": refs if refs is not None else [],
    }
    if extra:
        data.update(extra)
    return {
        "version": 11,
        "device_fingerprint": {"device_type": device_type},
        "data": data,
        "entry_data": {},
        "entry_options": entry_options or {},
    }


def _cyc(cid, profile, watts, dur=3600):
    return {"id": cid, "profile_name": profile, "duration": dur, "status": "completed",
            "start_time": "2023-01-01T10:00:00+00:00", "power_data": _trace(watts, dur=dur)}


# ── merge: nothing local is lost ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_merge_adds_new_profile_keeps_existing(store):
    # Seed a local profile.
    store._data["profiles"]["Local"] = {"avg_duration": 1000}
    payload = _payload(profiles={"Imported": {"avg_duration": 2000}})
    summary = await store.async_import_data_selective(
        payload, selection={"categories": ["profiles"]}, local_device_type="washing_machine"
    )
    assert set(store.get_profiles().keys()) == {"Local", "Imported"}
    assert summary["profiles_imported"] == 1


@pytest.mark.asyncio
async def test_merge_conflict_import_as_copy(store):
    store._data["profiles"]["Cotton 40"] = {"avg_duration": 1000}
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 9999}})
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles"]},
        conflict_resolutions={"Cotton 40": "import_as_copy"},
        local_device_type="washing_machine",
    )
    profs = store.get_profiles()
    assert profs["Cotton 40"]["avg_duration"] == 1000          # local untouched
    assert profs["Cotton 40 (imported)"]["avg_duration"] == 9999  # copy created


@pytest.mark.asyncio
async def test_merge_conflict_keep_mine(store):
    store._data["profiles"]["Cotton 40"] = {"avg_duration": 1000}
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 9999}})
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles"]},
        conflict_resolutions={"Cotton 40": "keep_mine"},
        local_device_type="washing_machine",
    )
    assert store.get_profiles()["Cotton 40"]["avg_duration"] == 1000
    assert "Cotton 40 (imported)" not in store.get_profiles()


@pytest.mark.asyncio
async def test_merge_conflict_overwrite(store):
    store._data["profiles"]["Cotton 40"] = {"avg_duration": 1000}
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 9999}})
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles"]},
        conflict_resolutions={"Cotton 40": "overwrite"},
        local_device_type="washing_machine",
    )
    assert store.get_profiles()["Cotton 40"]["avg_duration"] == 9999
    assert "Cotton 40 (imported)" not in store.get_profiles()


@pytest.mark.asyncio
async def test_copy_reroutes_cycles_to_copied_profile(store):
    store._data["profiles"]["Cotton 40"] = {"avg_duration": 1000}
    payload = _payload(
        profiles={"Cotton 40": {"avg_duration": 3600}},
        refs=[_cyc("r1", "Cotton 40", 2000)],
    )
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles", "reference_cycles"]},
        conflict_resolutions={"Cotton 40": "import_as_copy"},
        local_device_type="washing_machine",
    )
    ref = store.get_reference_cycles()
    assert len(ref) == 1
    assert ref[0]["profile_name"] == "Cotton 40 (imported)"  # cycle follows the rename


# ── cycle destination: reference (default, isolation-safe) ───────────────────

@pytest.mark.asyncio
async def test_real_cycles_to_reference_preserves_isolation(store):
    store._data["lifetime_energy_wh"] = 500.0
    store._data["lifetime_cycle_count"] = 7
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 3600}},
                       past=[_cyc("p1", "Cotton 40", 2000)])
    summary = await store.async_import_data_selective(
        payload, selection={"categories": ["profiles", "real_cycles"]},
        cycle_destination="reference", local_device_type="washing_machine",
    )
    assert store.get_past_cycles() == []                   # never enters usage history
    assert len(store.get_reference_cycles()) == 1
    assert store.get_lifetime_energy_wh() == 500.0         # lifetime untouched
    assert store.get_lifetime_cycle_count() == 7
    assert summary["reference_cycles_imported"] == 1
    assert summary["real_cycles_imported"] == 0


# ── cycle destination: real_history (migration) ──────────────────────────────

@pytest.mark.asyncio
async def test_real_cycles_to_real_history(store):
    store._data["lifetime_energy_wh"] = 500.0
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 3600}},
                       past=[_cyc("p1", "Cotton 40", 2000)])
    summary = await store.async_import_data_selective(
        payload, selection={"categories": ["profiles", "real_cycles"]},
        cycle_destination="real_history", local_device_type="washing_machine",
    )
    assert len(store.get_past_cycles()) == 1
    assert store.get_past_cycles()[0]["profile_name"] == "Cotton 40"
    assert store.get_past_cycles()[0]["meta"]["imported_from"] == "p1"
    assert store.get_lifetime_energy_wh() == 500.0         # lifetime NOT bumped by import
    assert summary["real_cycles_imported"] == 1


@pytest.mark.asyncio
async def test_real_history_dedup_on_reimport(store):
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 3600}},
                       past=[_cyc("p1", "Cotton 40", 2000)])
    sel = {"categories": ["profiles", "real_cycles"]}
    await store.async_import_data_selective(
        payload, selection=sel, cycle_destination="real_history",
        local_device_type="washing_machine")
    await store.async_import_data_selective(
        payload, selection=sel, cycle_destination="real_history",
        local_device_type="washing_machine")
    assert len(store.get_past_cycles()) == 1  # second import is idempotent


# ── device-type gating ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_device_type_mismatch_forces_reference(store):
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 3600}},
                       past=[_cyc("p1", "Cotton 40", 2000)], device_type="dishwasher")
    summary = await store.async_import_data_selective(
        payload, selection={"categories": ["profiles", "real_cycles"]},
        cycle_destination="real_history",  # requested, but must be forced to reference
        local_device_type="washing_machine",
    )
    assert store.get_past_cycles() == []
    assert len(store.get_reference_cycles()) == 1
    assert summary["device_type_match"] is False


@pytest.mark.asyncio
async def test_device_type_mismatch_drops_device_specific_settings(store):
    payload = _payload(
        profiles={"Cotton 40": {"avg_duration": 3600}},
        device_type="dishwasher",
        entry_options={"profile_match_threshold": 0.4},
    )
    summary = await store.async_import_data_selective(
        payload, selection={"categories": ["profiles", "settings"]},
        apply_settings=True, local_device_type="washing_machine",
    )
    assert summary["settings"] == {}  # settings are device-specific -> blocked on mismatch


# ── leaf category merge / replace ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_leaf_list_merge_dedup(store):
    store._data["maintenance_log"] = [{"id": "m1", "type": "clean"}]
    payload = _payload(extra={"maintenance_log": [{"id": "m1", "type": "clean"},
                                                  {"id": "m2", "type": "descale"}]})
    await store.async_import_data_selective(
        payload, selection={"categories": ["maintenance_log"]},
        local_device_type="washing_machine")
    ids = {m["id"] for m in store._data["maintenance_log"]}
    assert ids == {"m1", "m2"}  # m1 not duplicated


@pytest.mark.asyncio
async def test_leaf_dict_merge_keep_mine(store):
    store._data["profile_groups"] = {"g1": {"members": ["Local"]}}
    payload = _payload(extra={"profile_groups": {"g1": {"members": ["Theirs"]},
                                                 "g2": {"members": ["New"]}}})
    await store.async_import_data_selective(
        payload, selection={"categories": ["profile_groups"]},
        local_device_type="washing_machine")
    assert store._data["profile_groups"]["g1"]["members"] == ["Local"]  # keep-mine
    assert store._data["profile_groups"]["g2"]["members"] == ["New"]    # added


@pytest.mark.asyncio
async def test_lifetime_stats_skipped_in_merge_replaced_in_replace(store):
    store._data["lifetime_energy_wh"] = 100.0
    payload = _payload(extra={"lifetime_energy_wh": 999.0, "lifetime_cycle_count": 3})
    # merge: lifetime never summed / replaced
    await store.async_import_data_selective(
        payload, selection={"categories": ["lifetime_stats"]}, mode="merge",
        local_device_type="washing_machine")
    assert store.get_lifetime_energy_wh() == 100.0
    # replace: overwritten
    await store.async_import_data_selective(
        payload, selection={"categories": ["lifetime_stats"]}, mode="replace",
        local_device_type="washing_machine")
    assert store.get_lifetime_energy_wh() == 999.0


# ── replace mode ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_replace_wipes_only_selected_reference_category(store):
    await store.add_reference_cycle("Old", _trace(1000), {"store_cycle_id": "old"})
    store._data["profiles"]["KeepMe"] = {"avg_duration": 100}
    payload = _payload(profiles={"New": {"avg_duration": 3600}},
                       refs=[_cyc("r1", "New", 2000)])
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles", "reference_cycles"]},
        mode="replace", local_device_type="washing_machine")
    ref_profiles = {c["profile_name"] for c in store.get_reference_cycles()}
    assert ref_profiles == {"New"}          # old reference cycle wiped
    assert store.get_past_cycles() == []    # past_cycles never touched by ref replace


@pytest.mark.asyncio
async def test_replace_empty_payload_guarded(store):
    store._data["profiles"]["KeepMe"] = {"avg_duration": 100}
    payload = _payload(profiles={}, past=[])
    with pytest.raises(ValueError):
        await store.async_import_data_selective(
            payload, selection={"categories": ["profiles", "real_cycles"]},
            mode="replace", local_device_type="washing_machine")


@pytest.mark.asyncio
async def test_replace_empty_reference_payload_guarded(store):
    # A reference-only replace with no reference cycles in the file must NOT wipe the
    # locally-curated reference cycles (the guard previously only covered profiles/real).
    await store.add_reference_cycle("KeepMe", _trace(1000), {"store_cycle_id": "keep"})
    payload = _payload(refs=[])
    with pytest.raises(ValueError):
        await store.async_import_data_selective(
            payload, selection={"categories": ["reference_cycles"]},
            mode="replace", local_device_type="washing_machine")
    assert len(store.get_reference_cycles()) == 1  # preserved


@pytest.mark.asyncio
async def test_replace_conflict_overwrites_not_copies(store):
    # In replace mode a name clash must overwrite the local profile in place, not create a
    # copy. Crucially this must hold even though the panel transmits its analyze-time
    # conflict_resolutions (import_as_copy) while hiding the resolver in replace mode — the
    # backend enforces overwrite regardless of what the client sent (real UI path).
    store._data["profiles"]["Cotton 40"] = {"avg_duration": 1000, "marker": "local"}
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 9999, "marker": "from_file"}},
                       refs=[_cyc("r1", "Cotton 40", 2000)])
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles", "reference_cycles"]},
        conflict_resolutions={"Cotton 40": "import_as_copy"},   # what the panel actually sends
        mode="replace", local_device_type="washing_machine")
    profs = store.get_profiles()
    # Definition overwritten in place (marker survives the envelope rebuild that recomputes
    # avg_duration), and no dangling "(imported)" copy left behind.
    assert profs["Cotton 40"].get("marker") == "from_file"
    assert "Cotton 40 (imported)" not in profs
    ref = store.get_reference_cycles()
    assert [c["profile_name"] for c in ref] == ["Cotton 40"]   # cycle routes to the real profile


@pytest.mark.asyncio
async def test_replace_overwrite_carries_file_envelope(store):
    # Definition-only replace of a clashing profile (no cycles routed to it) must adopt the
    # file's envelope, not keep the stale local one that no longer matches the new definition.
    store._data["profiles"]["Eco"] = {"avg_duration": 1000}
    store._data.setdefault("envelopes", {})["Eco"] = {"marker": "LOCAL_ENV"}
    payload = _payload(profiles={"Eco": {"avg_duration": 5000}},
                       extra={"envelopes": {"Eco": {"marker": "FILE_ENV"}}})
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles"]},
        mode="replace", local_device_type="washing_machine")
    assert store._data["envelopes"]["Eco"] == {"marker": "FILE_ENV"}  # file wins


@pytest.mark.asyncio
async def test_replace_real_cycles_to_reference_wipes_reference_list(store):
    # Replace mode + only "real_cycles" ticked + default reference destination routes the
    # imported real cycles into reference_cycles, so that list (the actual destination) must
    # be wiped first -- otherwise replace silently behaves like merge for reference_cycles.
    await store.add_reference_cycle("Old", _trace(1000), {"store_cycle_id": "old"})
    assert len(store.get_reference_cycles()) == 1
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 3600}},
                       past=[_cyc("p1", "Cotton 40", 2000)])
    await store.async_import_data_selective(
        payload, selection={"categories": ["real_cycles"]},
        mode="replace", cycle_destination="reference", local_device_type="washing_machine")
    refs = store.get_reference_cycles()
    assert len(refs) == 1                       # old wiped, one new added (not 2)
    assert refs[0]["profile_name"] == "Cotton 40"
    assert store.get_past_cycles() == []        # real_history untouched


@pytest.mark.asyncio
async def test_reference_reimport_is_idempotent(store):
    # Importing the same reference bundle twice must not duplicate cycles (which would
    # double-weight the envelope and inflate cycle_count).
    payload = _payload(profiles={"Cotton 40": {"avg_duration": 3600}},
                       refs=[_cyc("r1", "Cotton 40", 2000)])
    sel = {"categories": ["profiles", "reference_cycles"]}
    s1 = await store.async_import_data_selective(payload, selection=sel,
                                                 local_device_type="washing_machine")
    s2 = await store.async_import_data_selective(payload, selection=sel,
                                                 local_device_type="washing_machine")
    assert s1["reference_cycles_imported"] == 1
    assert s2["reference_cycles_imported"] == 0        # deduped on re-import
    assert len(store.get_reference_cycles()) == 1


# ── guards + envelopes ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_selection_raises(store):
    payload = _payload(profiles={"A": {"avg_duration": 1}})
    with pytest.raises(ValueError):
        await store.async_import_data_selective(
            payload, selection={"categories": []}, local_device_type="washing_machine")


@pytest.mark.asyncio
async def test_profiles_only_carries_envelope_not_blanked(store):
    # profiles-only import with a carried envelope must keep it (no zero-cycle rebuild).
    payload = {
        "version": 11,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {
            "profiles": {"Cotton 40": {"avg_duration": 3600}},
            "past_cycles": [],
            "reference_cycles": [],
            "envelopes": {"Cotton 40": {"avg": [100, 200], "cycle_count": 3}},
        },
    }
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles"]},
        local_device_type="washing_machine")
    env = store._data["envelopes"].get("Cotton 40")
    assert env is not None and env["avg"] == [100, 200]  # carried, not blanked


@pytest.mark.asyncio
async def test_reference_import_rebuilds_envelope(store):
    payload = _payload(profiles={"Eco 50": {"avg_duration": 3600}},
                       refs=[_cyc("r1", "Eco 50", 1500)])
    await store.async_import_data_selective(
        payload, selection={"categories": ["profiles", "reference_cycles"]},
        local_device_type="washing_machine")
    assert store.get_envelope("Eco 50") is not None  # rebuilt from the imported cycle
