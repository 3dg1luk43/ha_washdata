"""Phase E: StoreBridge orchestration (connect/import/share) with a fake client."""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.store import StoreBridge

BASE = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


class FakeClient:
    def __init__(self):
        self.token = "TOK"
        self.uploaded = None
        self.cycle = None
    async def ensure_id_token(self, rt):
        return self.token
    async def get_cycle(self, cid):
        return self.cycle
    async def search_devices(self, brand, at):
        return [{"id": "d1", "brand": "Bosch"}]
    async def get_profiles(self, did):
        return [{"id": "p1", "program": "Cotton 40"}]
    async def get_cycles(self, pid):
        return [{"id": "c1"}]
    async def upload_reference_cycle(self, rt, uid, name, meta, points, stats, qc):
        self.uploaded = {"uid": uid, "name": name, "meta": meta, "points": points, "stats": stats, "qc": qc}
        return "newcycle"


@pytest.fixture
def bridge():
    hass = MagicMock()

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
        yield br, ps


@pytest.mark.asyncio
async def test_connect_persists_account(bridge):
    br, ps = bridge
    res = await br.connect("refresh", "u1", "Alice")
    assert res["connected"] is True and res["uid"] == "u1"
    assert ps.get_store_account()["refresh_token"] == "refresh"


@pytest.mark.asyncio
async def test_connect_rejects_bad_token(bridge):
    br, ps = bridge
    br._client.token = None
    res = await br.connect("bad", "u1", "Alice")
    assert res == {"error": "token_invalid"}
    assert ps.get_store_account() == {}


@pytest.mark.asyncio
async def test_import_cycle_adds_reference(bridge):
    br, ps = bridge
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
    br, ps = bridge
    br._client.cycle = {"id": "x", "importable": None}
    res = await br.import_cycle("x")
    assert res == {"error": "unsupported_schema"}


@pytest.mark.asyncio
async def test_share_cycle_requires_connection(bridge):
    br, ps = bridge
    res = await br.share_cycle("local1", "Cotton 40", "Bosch", "WAT", "washer")
    assert res == {"error": "not_connected"}


@pytest.mark.asyncio
async def test_share_cycle_uploads_with_derived_qc(bridge):
    br, ps = bridge
    await br.connect("refresh", "u1", "Alice")
    # a local golden cycle to share
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
