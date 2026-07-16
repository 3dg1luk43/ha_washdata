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
"""Phase E: StoreBridge orchestration (connect/import/share/catalog) with a fake client."""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import store_account
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.store import StoreBridge

BASE = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


class FakeClient:
    def __init__(self):
        self.token = "TOK"
        self.uploaded = None
        self.cycle = None
        self.confirmed = None
        self.rated = None
        self.last_refresh_token = None
    async def ensure_id_token(self, rt):
        self.last_refresh_token = rt  # record so tests can assert the token is forwarded
        return self.token
    async def get_cycle(self, cid):
        return self.cycle
    async def search_devices(self, brand, appliance_type, model_query=None, include_pending=False):
        self.last_search = {"brand": brand, "appliance_type": appliance_type, "model_query": model_query, "include_pending": include_pending}
        return [{"id": "d1", "brand": "Bosch", "status": "pending", "confirmCount": 2}]
    async def list_brands(self, q, include_pending=True):
        self.last_brands = {"q": q, "include_pending": include_pending}
        return [{"id": "bosch", "brand": "Bosch", "status": "approved"}]
    async def get_profiles(self, did):
        return [{"id": "p1", "program": "Cotton 40"}]
    async def device_profiles(self, brand, model, appliance_type):
        self.last_device_profiles = {"brand": brand, "model": model, "appliance_type": appliance_type}
        return {"device_id": f"{appliance_type}__{brand.lower()}__{model.lower()}", "items": [{"id": "p1", "program": "Cotton 40"}]}
    async def get_cycles(self, pid):
        return [{"id": "c1"}]
    async def get_device_quality(self, did):
        return {"avg": 4.5, "count": 2}
    async def confirm_device(self, rt, uid, did):
        self.confirmed = {"rt": rt, "uid": uid, "did": did}
        return {"confirmed": True, "confirmCount": 3, "status": "pending"}
    async def rate_device(self, rt, uid, did, rating):
        self.rated = {"rt": rt, "uid": uid, "did": did, "rating": rating}
        return True
    async def upload_reference_cycle(self, rt, uid, name, meta, points, stats, qc):
        self.uploaded = {"uid": uid, "name": name, "meta": meta, "points": points, "stats": stats, "qc": qc}
        return "newcycle"
    async def upload_device_bundle(self, rt, uid, name, device_meta, items):
        self.uploaded_bundle = {"rt": rt, "uid": uid, "name": name, "device_meta": device_meta, "items": items}
        return {"ok": True, "cycle_ids": [f"c{i}" for i in range(len(items))], "errors": []}
    async def get_device_bundle(self, device_id, include_pending=True):
        return getattr(self, "bundle", {"device_id": device_id, "profiles": []})


@pytest.fixture
def bridge():
    hass = MagicMock()
    hass.data = {}
    fake = MagicMock()
    fake.async_save = AsyncMock()
    hass.data[store_account._DATA_KEY] = {"store": fake, "data": {"online_enabled": True, "account": {}}}

    async def _exec(func, *a, **k):
        return await func(*a, **k) if inspect.iscoroutinefunction(func) else func(*a, **k)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        br = StoreBridge(hass, ps)
        br._client = FakeClient()
        yield br, ps, hass


@pytest.mark.asyncio
async def test_connect_persists_account_globally(bridge):
    br, ps, hass = bridge
    res = await br.connect("refresh", "u1", "Alice")
    assert res["connected"] is True and res["uid"] == "u1"
    # connect() must forward the SUPPLIED refresh token to the client for validation.
    assert br._client.last_refresh_token == "refresh"
    # Account lives in the integration-wide store, NOT the per-device ProfileStore.
    assert store_account.get_account(hass)["refresh_token"] == "refresh"
    assert ps.get_store_account() == {}


@pytest.mark.asyncio
async def test_connect_rejects_bad_token(bridge):
    br, ps, hass = bridge
    br._client.token = None
    res = await br.connect("bad", "u1", "Alice")
    assert res == {"error": "token_invalid"}
    assert store_account.get_account(hass) == {}


@pytest.mark.asyncio
async def test_status_reads_global(bridge):
    br, ps, hass = bridge
    await br.connect("refresh", "u1", "Alice")
    st = br.status()
    assert st["enabled"] is True and st["connected"] is True and st["uid"] == "u1"
    assert "refresh_token" not in st


@pytest.mark.asyncio
async def test_list_brands_and_search_passthrough(bridge):
    br, ps, hass = bridge
    await br.list_brands("bo", include_pending=True)
    assert br._client.last_brands == {"q": "bo", "include_pending": True}
    await br.search_devices("bosch", "washer", model_query="wat", include_pending=True)
    assert br._client.last_search["model_query"] == "wat" and br._client.last_search["include_pending"] is True


@pytest.mark.asyncio
async def test_confirm_and_rate_require_connection(bridge):
    br, ps, hass = bridge
    assert await br.confirm_device("d1") == {"error": "not_connected"}
    assert await br.rate_device("d1", 4) == {"error": "not_connected"}


@pytest.mark.asyncio
async def test_confirm_and_rate_when_connected(bridge):
    br, ps, hass = bridge
    await br.connect("refresh", "u1", "Alice")
    res = await br.confirm_device("d1")
    assert res["confirmCount"] == 3 and br._client.confirmed["uid"] == "u1"
    assert await br.rate_device("d1", 4) == {"ok": True}
    assert br._client.rated["rating"] == 4


@pytest.mark.asyncio
async def test_get_device_quality(bridge):
    br, ps, hass = bridge
    assert await br.get_device_quality("d1") == {"avg": 4.5, "count": 2}


@pytest.mark.asyncio
async def test_device_profiles_maps_type(bridge):
    br, ps, hass = bridge
    res = await br.device_profiles("Bosch", "WAT", "washing_machine")
    # HA washing_machine -> catalog washer before resolving the deviceId.
    assert br._client.last_device_profiles["appliance_type"] == "washer"
    assert res["device_id"].startswith("washer__") and res["items"][0]["program"] == "Cotton 40"


@pytest.mark.asyncio
async def test_import_cycle_adds_reference(bridge):
    br, ps, hass = bridge
    br._client.cycle = {
        "id": "storecyc", "program_lc": "cotton-40", "createdAt": "2026-01-01T00:00:00Z",
        "importable": [[0, 2000], [60, 100], [120, 0]],
        "trace": {"points": [[0, 2000]], "sampleIntervalSec": 60},
    }
    res = await br.import_cycle("storecyc", new_profile_name="Cotton 40")
    assert res["profile"] == "Cotton 40"
    refs = ps.get_reference_cycles()
    assert len(refs) == 1 and refs[0]["meta"]["source"] == "store:storecyc"
    assert ps.get_past_cycles() == []  # isolation preserved
    # The full importable waveform must be persisted (not truncated/wrong): all three
    # samples survive the store round-trip.
    stored = ps.get_cycle_power_data(refs[0]["id"])
    assert [[round(o), round(w)] for o, w in stored] == [[0, 2000], [60, 100], [120, 0]]


@pytest.mark.asyncio
async def test_import_cycle_unsupported_schema(bridge):
    br, ps, hass = bridge
    br._client.cycle = {"id": "x", "importable": None}
    res = await br.import_cycle("x")
    assert res == {"error": "unsupported_schema"}


@pytest.mark.asyncio
async def test_share_cycle_requires_connection(bridge):
    br, ps, hass = bridge
    res = await br.share_cycle("local1", "Cotton 40", "Bosch", "WAT", "washer")
    assert res == {"error": "not_connected"}


@pytest.mark.asyncio
async def test_share_cycle_maps_washing_machine_to_washer(bridge):
    br, ps, hass = bridge
    await br.connect("refresh", "u1", "Alice")
    await ps.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "profile_name": "Cotton 40",
        "power_data": [((BASE + timedelta(seconds=i * 60)).isoformat(), 1000.0) for i in range(61)],
    })
    local_id = ps.get_past_cycles()[0]["id"]
    # HA device type is washing_machine; the catalog only knows washer.
    await br.share_cycle(local_id, "Cotton 40", "Bosch", "WAT", "washing_machine", sample_interval_sec=60)
    assert br._client.uploaded["meta"]["applianceType"] == "washer"


@pytest.mark.asyncio
async def test_share_cycle_uploads_with_derived_qc(bridge):
    br, ps, hass = bridge
    await br.connect("refresh", "u1", "Alice")
    await ps.async_add_cycle({
        "start_time": BASE.isoformat(),
        "duration": 3600,
        "status": "completed",
        "profile_name": "Cotton 40",
        "power_data": [((BASE + timedelta(seconds=i * 60)).isoformat(), 1000.0) for i in range(61)],
    })
    local_id = ps.get_past_cycles()[0]["id"]
    res = await br.share_cycle(local_id, "Cotton 40", "Bosch", "WAT28660", "washer", sample_interval_sec=60)
    assert res == {"store_cycle_id": "newcycle"}
    up = br._client.uploaded
    assert up["uid"] == "u1" and up["qc"] == 3  # plain detected golden -> manual
    assert up["meta"]["brand"] == "Bosch" and up["meta"]["program"] == "Cotton 40"
    assert len(up["points"]) >= 2


@pytest.mark.asyncio
async def test_share_device_forwards_items(bridge):
    br, ps, hass = bridge
    await br.connect("refresh", "u1", "Alice")  # sets the global account
    await ps.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "profile_name": "Cotton 40",
        "power_data": [[i * 60.0, 1000.0] for i in range(61)],
    })
    cid = ps.get_past_cycles()[0]["id"]
    res = await br.share_device(
        "Bosch", "WAT28", "washing_machine",
        [{"local_cycle_id": cid, "program": "Cotton 40"}],
    )
    assert res.get("ok") is True
    up = br._client.uploaded_bundle
    # HA washing_machine maps to the store's washer type; item carries the program + trace.
    assert up["device_meta"] == {"applianceType": "washer", "brand": "Bosch", "model": "WAT28"}
    assert len(up["items"]) == 1
    assert up["items"][0]["program"] == "Cotton 40" and up["items"][0]["points"]


@pytest.mark.asyncio
async def test_share_device_requires_connection(bridge):
    br, ps, hass = bridge
    res = await br.share_device("Bosch", "WAT28", "washer", [{"local_cycle_id": "x", "program": "P"}])
    assert res == {"error": "not_connected"}


@pytest.mark.asyncio
async def test_download_device_adopts_bundle(bridge):
    br, ps, hass = bridge
    br._client.bundle = {"device_id": "d1", "profiles": [
        {"id": "p1", "program": "Cotton 40", "cycles": [
            {"id": "c1", "importable": [[0, 2000], [60, 100], [120, 0]], "createdAt": "t",
             "trace": {"sampleIntervalSec": 60}},
        ]},
        {"id": "p2", "program": "Eco 50", "cycles": [
            {"id": "c2", "importable": [[0, 1500], [60, 50], [120, 0]], "createdAt": "t",
             "trace": {"sampleIntervalSec": 60}},
        ]},
    ]}
    res = await br.download_device("d1")
    assert res == {"profiles_adopted": 2, "cycles_imported": 2, "phases_applied": 0, "settings": {}}
    refs = ps.get_reference_cycles()
    assert {r["profile_name"] for r in refs} == {"Cotton 40", "Eco 50"}
    assert ps.get_past_cycles() == []  # real data untouched


@pytest.mark.asyncio
async def test_download_device_is_idempotent(bridge):
    br, ps, hass = bridge
    br._client.bundle = {"device_id": "d1", "profiles": [
        {"id": "p1", "program": "Cotton 40", "cycles": [
            {"id": "c1", "importable": [[0, 2000], [60, 100], [120, 0]], "createdAt": "t",
             "trace": {"sampleIntervalSec": 60}},
        ]},
    ]}
    first = await br.download_device("d1")
    assert first == {"profiles_adopted": 1, "cycles_imported": 1, "phases_applied": 0, "settings": {}}
    # Re-downloading the same device must not duplicate the already-imported cycle.
    second = await br.download_device("d1")
    assert second == {"profiles_adopted": 0, "cycles_imported": 0, "phases_applied": 0, "settings": {}}
    assert len(ps.get_reference_cycles()) == 1


@pytest.mark.asyncio
async def test_share_device_attaches_phases(bridge):
    br, ps, hass = bridge
    await br.connect("refresh", "u1", "Alice")
    await ps.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "profile_name": "Cotton 40",
        "power_data": [[i * 60.0, 1000.0] for i in range(61)],
    })
    cid = ps.get_past_cycles()[0]["id"]
    ps._data.setdefault("profiles", {}).setdefault("Cotton 40", {})
    await ps.async_set_profile_phase_ranges("Cotton 40", [{"name": "Wash", "start": 0, "end": 600}])
    res = await br.share_device(
        "Bosch", "WAT28", "washing_machine",
        [{"local_cycle_id": cid, "program": "Cotton 40"}],
        include_phases=["Cotton 40"],
    )
    assert res.get("ok") is True
    item = br._client.uploaded_bundle["items"][0]
    assert item["phases"] == [{"name": "Wash", "start": 0.0, "end": 600.0}]
    assert item["phaseSourceCycleId"]  # non-empty deterministic id


@pytest.mark.asyncio
async def test_share_device_omits_phases_when_not_requested(bridge):
    br, ps, hass = bridge
    await br.connect("refresh", "u1", "Alice")
    await ps.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "profile_name": "Cotton 40", "power_data": [[i * 60.0, 1000.0] for i in range(61)],
    })
    cid = ps.get_past_cycles()[0]["id"]
    ps._data.setdefault("profiles", {}).setdefault("Cotton 40", {})
    await ps.async_set_profile_phase_ranges("Cotton 40", [{"name": "Wash", "start": 0, "end": 600}])
    await br.share_device("Bosch", "WAT28", "washer", [{"local_cycle_id": cid, "program": "Cotton 40"}])
    assert "phases" not in br._client.uploaded_bundle["items"][0]


@pytest.mark.asyncio
async def test_download_device_applies_phases(bridge):
    br, ps, hass = bridge
    br._client.bundle = {"device_id": "d1", "profiles": [
        {"id": "p1", "program": "Cotton 40", "phases": [{"name": "Rinse", "start": 0, "end": 300}],
         "cycles": [
            {"id": "c1", "importable": [[0, 2000], [60, 100], [120, 0]], "createdAt": "t",
             "trace": {"sampleIntervalSec": 60}},
         ]},
    ]}
    res = await br.download_device("d1", "washer")
    assert res["profiles_adopted"] == 1 and res["phases_applied"] == 1
    ranges = ps.get_profile_phase_ranges("Cotton 40")
    assert ranges == [{"name": "Rinse", "start": 0.0, "end": 300.0, "description": ""}]
    # The label was reconciled into the custom-phase catalog.
    assert any(p.get("name", "").casefold() == "rinse" for p in ps.list_phase_catalog("washer"))


@pytest.mark.asyncio
async def test_share_device_carries_settings(bridge):
    br, ps, hass = bridge
    await br.connect("refresh", "u1", "Alice")
    await ps.async_add_cycle({
        "start_time": BASE.isoformat(), "duration": 3600, "status": "completed",
        "profile_name": "Cotton 40", "power_data": [[i * 60.0, 1000.0] for i in range(61)],
    })
    cid = ps.get_past_cycles()[0]["id"]
    res = await br.share_device(
        "Bosch", "WAT28", "washer", [{"local_cycle_id": cid, "program": "Cotton 40"}],
        settings={"off_delay": 180, "start_threshold_w": 12.0},
    )
    assert res.get("ok") is True
    assert br._client.uploaded_bundle["device_meta"]["settings"] == {"off_delay": 180, "start_threshold_w": 12.0}


@pytest.mark.asyncio
async def test_download_device_returns_settings(bridge):
    br, ps, hass = bridge
    br._client.bundle = {"device_id": "d1", "settings": {"off_delay": 180}, "profiles": [
        {"id": "p1", "program": "Cotton 40", "cycles": [
            {"id": "c1", "importable": [[0, 2000], [60, 100], [120, 0]], "createdAt": "t",
             "trace": {"sampleIntervalSec": 60}},
        ]},
    ]}
    res = await br.download_device("d1", "washer")
    assert res["settings"] == {"off_delay": 180}
    assert res["profiles_adopted"] == 1
