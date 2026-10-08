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
"""Register item 129 (a)(d)(e): imported cycles (reference + backfill) as peers.

(a) ``get_device_cycles`` pages imported cycles on their own cursor instead of
    returning every one of them on page 1.
(d) the setup advisor counts backfilled history as the device's own, so an
    import-only device is neither "unconfigured" nor "from the community".
(e) the selective export/import wizard has a ``backfill_cycles`` category, so a
    backfill-only device round-trips without losing its cycles.
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    DOMAIN,
    EVIDENCE_REAL_CYCLES,
    EVIDENCE_REFERENCE_CYCLES,
)
from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    build_import_manifest,
)
from custom_components.ha_washdata.setup_advisor import compute_setup_phase

_T0 = datetime(2026, 3, 1, 8, 0, tzinfo=timezone.utc)
_NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)


def _hass(config_dir: Path | None = None):
    hass = MagicMock()
    hass.data = {}
    if config_dir is not None:  # an import writes its undo snapshot through HA's Store
        hass.config.path = lambda *a: str(config_dir.joinpath(*a))

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


def _store(hass=None) -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass or _hass(), "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    return ps


def _trace(watts: float, n: int = 61, dur: int = 3600) -> list[list[float]]:
    step = dur / (n - 1)
    return [[i * step, float(watts)] for i in range(n)]


def _cycle(cid: str, day: int, profile: str | None = None, *, watts: float = 500.0,
           source: str | None = None) -> dict:
    out = {
        "id": cid,
        "start_time": (_T0 + timedelta(days=day)).isoformat(),
        "duration": 3600,
        "status": "completed",
        "profile_name": profile,
        "power_data": _trace(watts),
    }
    if source:
        out["meta"] = {"source": source}
    return out


# ─── (a) get_device_cycles pages imported cycles ───────────────────────────────

def _device_cycles(store: ProfileStore, **params) -> dict:
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": SimpleNamespace(profile_store=store)}}
    conn = MagicMock()
    msg = {"id": 1, "entry_id": "e1", "limit": 50, "offset": 0, **params}
    ws_api.ws_get_device_cycles(hass, conn, msg)
    conn.send_result.assert_called_once()
    return conn.send_result.call_args[0][1]


def _imported_store() -> ProfileStore:
    """2 real, 3 reference and 4 backfill cycles; days interleave across the lists."""
    store = _store()
    store._data["past_cycles"] = [_cycle("p0", 0, "A"), _cycle("p1", 1, "A")]
    store._data["reference_cycles"] = [
        _cycle(f"r{d}", d, "A", source="store:x") for d in (2, 5, 8)
    ]
    store._data["backfill_cycles"] = [
        _cycle(f"b{d}", d, None, source="history_import") for d in (3, 4, 6, 7)
    ]
    return store


def test_imported_offset_pages_reference_and_backfill_together():
    store = _imported_store()
    pages, cursor = [], 0
    while True:
        res = _device_cycles(store, limit=3, imported_offset=cursor)
        assert res["imported_total"] == 7
        page = res["reference_cycles"] + res["backfill_cycles"]
        assert len(page) <= 3
        pages.append(page)
        cursor += len(page)
        if not res["imported_has_more"]:
            break
    assert [len(p) for p in pages] == [3, 3, 1]
    flat = [c for p in pages for c in p]
    ids = [c["id"] for c in flat]
    # One newest-start-first list across both categories, no duplicates.
    assert ids == ["r8", "b7", "b6", "r5", "b4", "b3", "r2"]
    for item in flat:
        assert item["is_reference"] is True
        assert item["cycle_origin"] == ("reference" if item["id"][0] == "r" else "backfill")
        assert "power_data" not in item


def test_imported_offset_leaves_real_paging_alone():
    store = _imported_store()
    res = _device_cycles(store, limit=1, offset=0, imported_offset=0)
    assert [c["id"] for c in res["cycles"]] == ["p1"]
    assert res["total"] == 2 and res["has_more"] is True
    # Imported cycles stay out of the real total.
    res = _device_cycles(store, limit=1, offset=1, imported_offset=1)
    assert [c["id"] for c in res["cycles"]] == ["p0"]
    assert res["total"] == 2 and res["has_more"] is False
    assert res["imported_has_more"] is True


def test_legacy_client_without_imported_offset_gets_everything_on_page_one():
    """The panel that predates the cursor keeps working: all imports on offset 0."""
    store = _imported_store()
    first = _device_cycles(store, limit=1, offset=0)
    assert len(first["reference_cycles"]) == 3 and len(first["backfill_cycles"]) == 4
    assert first["imported_total"] == 7 and first["imported_has_more"] is False
    later = _device_cycles(store, limit=1, offset=1)
    assert later["reference_cycles"] == [] and later["backfill_cycles"] == []
    assert later["imported_total"] == 7 and later["imported_has_more"] is False


def test_imported_offset_schema():
    schema = ws_api.ws_get_device_cycles._ws_schema
    ok = schema({"id": 1, "type": "ha_washdata/get_device_cycles", "entry_id": "e",
                 "imported_offset": 25})
    assert ok["imported_offset"] == 25
    assert "imported_offset" not in schema(
        {"id": 1, "type": "ha_washdata/get_device_cycles", "entry_id": "e"}
    )
    with pytest.raises(vol.Invalid):
        schema({"id": 1, "type": "ha_washdata/get_device_cycles", "entry_id": "e",
                "imported_offset": -1})


# ─── (d) setup advisor counts backfilled history ───────────────────────────────

def _phase(**kw):
    args = dict(
        device_type="washing_machine", profile_names=[], past_cycles=[],
        ref_profile_names=set(), coverage_gap=None, suggestions=[],
        skipped_steps={}, now=_NOW,
    )
    args.update(kw)
    return compute_setup_phase(**args)


def test_advisor_labelled_backfill_is_not_phase0_nor_community():
    r = _phase(profile_names=["Cotton"], backfill_cycles=[{"profile_name": "Cotton"}])
    assert r.phase == "phase1a"
    # Reference profiles alongside it do not make it a store-only device.
    r = _phase(profile_names=["Cotton", "Eco"], ref_profile_names={"Eco"},
               backfill_cycles=[{"profile_name": "Cotton"}])
    assert r.phase not in ("phase0", "phase1c")


def test_advisor_backfill_establishes_the_device():
    r = _phase(profile_names=["Cotton"], backfill_cycles=[{"profile_name": "Cotton"}] * 5)
    assert r.phase == "phase4"
    assert r.message_params == {"profile_count": 1}


def test_advisor_unlabelled_backfill_points_at_labelling():
    r = _phase(backfill_cycles=[{"profile_name": None}, {"profile_name": None}])
    assert r.phase == "phase0"
    assert r.message_key == "setup.phase0.generic"
    assert r.cta_action == "open_cycles_unlabeled"
    assert r.secondary_action == "open_recorder"


def test_advisor_without_backfill_is_unchanged():
    r = _phase(past_cycles=[{"profile_name": None}])
    assert (r.phase, r.message_key, r.cta_action) == (
        "phase0", "setup.phase0.washer", "open_recorder"
    )


async def _setup_status(store: ProfileStore) -> dict:
    hass = _hass()
    conn = MagicMock()
    manager = SimpleNamespace(profile_store=store, device_type="washing_machine")
    entry = SimpleNamespace(data={"device_type": "washing_machine"}, options={})
    conn.user = None
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_get_setup_status.__wrapped__(hass, conn, {"id": 1, "entry_id": "e"})
    conn.send_result.assert_called_once()
    return conn.send_result.call_args[0][1]


@pytest.mark.asyncio
async def test_setup_status_backfill_only_device_is_configured():
    store = _store()
    store._data["profiles"] = {"Cotton": {"avg_duration": 3600}}
    store._data["backfill_cycles"] = [_cycle("b1", 1, "Cotton", source="history_import")]
    res = await _setup_status(store)
    assert res["phase"] == "phase1a"


@pytest.mark.asyncio
@pytest.mark.parametrize("sources", [
    None,
    [EVIDENCE_REAL_CYCLES],
    [EVIDENCE_REFERENCE_CYCLES],
    [EVIDENCE_REAL_CYCLES, EVIDENCE_REFERENCE_CYCLES],
])
async def test_setup_status_phase0_is_exactly_no_matchable_profile(sources):
    """The card's phase0 is when the manager skips matching for lack of profiles."""
    for lists in (
        {"backfill_cycles": [_cycle("b1", 1, "Cotton")]},
        {"reference_cycles": [_cycle("r1", 1, "Cotton", source="store:x")]},
        {"past_cycles": [_cycle("p1", 1, "Cotton")]},
    ):
        store = _store()
        if sources is not None:
            store.evidence_sources = sources
        store._data["profiles"] = {"Cotton": {"avg_duration": 3600}}
        store._data.update(lists)
        res = await _setup_status(store)
        assert (res["phase"] == "phase0") == (not store.has_real_profiles), (sources, lists)


# ─── (e) selective export/import carries backfill_cycles ───────────────────────

def _backfill_device() -> ProfileStore:
    store = _store()
    store._data["profiles"] = {"Cotton": {"avg_duration": 3600}}
    store._data["backfill_cycles"] = [
        _cycle("b1", 1, "Cotton", source="history_import"),
        _cycle("b2", 2, "Cotton", source="history_import"),
        _cycle("b3", 3, None, source="history_import"),
    ]
    return store


_OPTS = {"device_type": "washing_machine"}


def _export(store: ProfileStore, selection: dict) -> dict:
    payload = store.export_data(entry_data=_OPTS, entry_options=_OPTS, selection=selection)
    payload["device_fingerprint"] = {"device_type": "washing_machine"}
    return payload


def test_inventory_lists_backfill_category():
    inv = _backfill_device().get_export_inventory(_OPTS)
    cat = inv["backfill_cycles"]
    assert cat["present"] is True and cat["count"] == 3
    groups = {g["profile"]: g["count"] for g in cat["groups"]}
    assert groups == {"Cotton": 2, "": 1}
    assert inv["profiles"]["items"][0]["backfill_cycles"] == 2


def test_selective_export_carries_chosen_backfill_cycles():
    store = _backfill_device()
    payload = _export(store, {"categories": ["backfill_cycles"],
                              "backfill_cycle_ids": ["b1", "b3"]})
    assert [c["id"] for c in payload["data"]["backfill_cycles"]] == ["b1", "b3"]
    whole = _export(store, {"categories": ["profiles", "backfill_cycles"]})
    assert len(whole["data"]["backfill_cycles"]) == 3
    # Cycles ship, so the "profiles empty" envelope carry does not apply.
    assert "envelopes" not in whole["data"]


@pytest.mark.asyncio
async def test_backfill_only_device_round_trips_through_selective_import(tmp_path):
    src = _backfill_device()
    payload = _export(src, {"categories": ["profiles", "backfill_cycles"]})

    dst = _store(_hass(tmp_path))
    summary = await dst.async_import_data_selective(
        payload, selection={"categories": ["profiles", "backfill_cycles"]},
        local_device_type="washing_machine",
    )
    assert summary["backfill_cycles_imported"] == 3
    got = dst.get_backfill_cycles()
    assert sorted(str(c["profile_name"]) for c in got) == ["Cotton", "Cotton", "None"]
    assert {c["meta"]["imported_from"] for c in got} == {"b1", "b2", "b3"}
    assert all(c["meta"]["source"] == "history_import" for c in got)
    # Isolation: backfill never becomes real history or usage stats.
    assert dst.get_past_cycles() == [] and dst.get_reference_cycles() == []
    # The labelled profile is matchable again on the target.
    assert dst.has_real_profiles
    assert "Cotton" in dst._data["envelopes"]

    # Re-importing the same file is a no-op.
    again = await dst.async_import_data_selective(
        payload, selection={"categories": ["profiles", "backfill_cycles"]},
        local_device_type="washing_machine",
    )
    assert again["backfill_cycles_imported"] == 0
    assert len(dst.get_backfill_cycles()) == 3


@pytest.mark.asyncio
async def test_backfill_import_skips_cycles_already_on_record(tmp_path):
    payload = _export(_backfill_device(), {"categories": ["backfill_cycles"]})
    dst = _store(_hass(tmp_path))
    # The same run recorded live, ending a few seconds apart: an overlap, not a new cycle.
    live = _cycle("live", 1, "Cotton")
    live["duration"] = 3590
    dst._data["past_cycles"] = [live]
    dst._data["profiles"] = {"Cotton": {"avg_duration": 3600}}
    summary = await dst.async_import_data_selective(
        payload, selection={"categories": ["backfill_cycles"]},
        local_device_type="washing_machine",
    )
    assert summary["backfill_cycles_imported"] == 2
    assert {c["meta"]["imported_from"] for c in dst.get_backfill_cycles()} == {"b2", "b3"}


@pytest.mark.asyncio
async def test_replace_mode_never_wipes_local_backfill(tmp_path):
    payload = _export(_backfill_device(), {"categories": ["backfill_cycles"],
                                           "backfill_cycle_ids": ["b2"]})
    dst = _store(_hass(tmp_path))
    mine = _cycle("mine", 40, None, source="history_import")
    dst._data["backfill_cycles"] = [mine]
    await dst.async_import_data_selective(
        payload, selection={"categories": ["backfill_cycles"]}, mode="replace",
        local_device_type="washing_machine",
    )
    assert mine in dst.get_backfill_cycles()
    assert len(dst.get_backfill_cycles()) == 2


@pytest.mark.asyncio
async def test_backfill_is_device_specific(tmp_path):
    payload = _export(_backfill_device(), {"categories": ["profiles", "backfill_cycles"]})
    manifest = build_import_manifest(payload, local_device_type="dishwasher",
                                     local_profile_names=[])
    assert manifest["categories"]["backfill_cycles"]["importable"] is False
    dst = _store(_hass(tmp_path))
    summary = await dst.async_import_data_selective(
        payload, selection={"categories": ["profiles", "backfill_cycles"]},
        local_device_type="dishwasher",
    )
    assert summary["backfill_cycles_imported"] == 0
    assert dst.get_backfill_cycles() == []


@pytest.mark.asyncio
async def test_backfill_import_never_golden_and_capped(tmp_path):
    src = _backfill_device()
    src._data["backfill_cycles"][0]["ml_review"] = {"golden": True, "label": "good"}
    payload = _export(src, {"categories": ["backfill_cycles"]})
    dst = _store(_hass(tmp_path))
    with patch("custom_components.ha_washdata.const.HISTORY_IMPORT_MAX_TOTAL_CYCLES", 2):
        summary = await dst.async_import_data_selective(
            payload, selection={"categories": ["backfill_cycles"]},
            local_device_type="washing_machine",
        )
    assert summary["backfill_cycles_imported"] == 2
    assert summary["skipped_cycles"] == 1
    for c in dst.get_backfill_cycles():
        assert "golden" not in (c.get("ml_review") or {})
