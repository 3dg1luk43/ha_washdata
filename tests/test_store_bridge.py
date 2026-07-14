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
    async def ensure_id_token(self, rt):
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
