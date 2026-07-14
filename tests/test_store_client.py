"""Phase C: store_client - id parity, decode, token exchange, reads, upload shape."""
import json
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import store_client as sc
from custom_components.ha_washdata.store_client import StoreClient


# ── fake aiohttp session ───────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, body=None):
        self.status = status
        self._body = body if body is not None else {}
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def json(self):
        return self._body
    async def text(self):
        return json.dumps(self._body) if not isinstance(self._body, str) else self._body


class _Session:
    def __init__(self):
        self.posts = []  # (url, kwargs)
        self.gets = []
        self._post_queue = []
        self._get_queue = []
    def queue_post(self, resp):
        self._post_queue.append(resp)
    def queue_get(self, resp):
        self._get_queue.append(resp)
    def post(self, url, **kw):
        self.posts.append((url, kw))
        return self._post_queue.pop(0) if self._post_queue else _Resp(200, {})
    def get(self, url, **kw):
        self.gets.append((url, kw))
        return self._get_queue.pop(0) if self._get_queue else _Resp(200, {})


def _client(session):
    return StoreClient(MagicMock(), project_id="washdata-store", api_key="KEY", session=session)


def _cycle_write(session):
    """Find the cycle-create commit write among the upload's :commit posts."""
    for url, kw in session.posts:
        if not url.endswith(":commit"):
            continue
        for w in kw.get("json", {}).get("writes", []):
            if "/cycles/" in w.get("update", {}).get("name", ""):
                return w
    raise AssertionError("no cycle-create commit found")


# ── id parity with lib/ids.js ──────────────────────────────────────────────────

def test_normalize_token_parity():
    assert sc.normalize_token("  Serie 6  WAT28660GB/01 ") == "serie-6-wat28660gb-01"
    assert sc.normalize_token("Bosch") == "bosch"


def test_device_and_profile_id_parity():
    d = sc.device_id("washer", "Bosch", "WAT 28660")
    assert d == "washer__bosch__wat-28660"
    assert sc.profile_id(d, "Cotton 40") == "washer__bosch__wat-28660__cotton-40"
    assert sc.brand_id("Bosch") == "bosch"


def test_typed_decode():
    doc = {"name": "projects/p/databases/(default)/documents/cycles/abc",
           "fields": {"qc": {"integerValue": "2"}, "brand_lc": {"stringValue": "bosch"},
                      "trace": {"mapValue": {"fields": {"points": {"arrayValue": {"values": [
                          {"arrayValue": {"values": [{"integerValue": "0"}, {"doubleValue": 5.0}]}}]}}}}}}}
    out = sc._decode_doc(doc)
    assert out["id"] == "abc" and out["qc"] == 2 and out["brand_lc"] == "bosch"
    assert out["trace"]["points"] == [[0, 5.0]]


# ── token exchange ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_token_exchange_and_cache():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "TОK", "expires_in": "3600"}))
    c = _client(s)
    tok = await c.ensure_id_token("refresh123")
    assert tok == "TОK"
    # cached: no second network call
    tok2 = await c.ensure_id_token("refresh123")
    assert tok2 == "TОK" and len(s.posts) == 1
    assert "securetoken" in s.posts[0][0] and "key=KEY" in s.posts[0][0]


# ── reads ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_devices_decodes():
    s = _Session()
    s.queue_post(_Resp(200, [{"document": {"name": ".../devices/d1", "fields": {
        "brand": {"stringValue": "Bosch"}, "status": {"stringValue": "approved"}}}}]))
    c = _client(s)
    items = await c.search_devices(brand="Bosch")
    assert items == [{"brand": "Bosch", "status": "approved", "id": "d1"}]


@pytest.mark.asyncio
async def test_search_devices_include_pending_uses_in_filter():
    s = _Session()
    s.queue_post(_Resp(200, []))
    c = _client(s)
    await c.search_devices(brand="Bosch", include_pending=True)
    where = s.posts[-1][1]["json"]["structuredQuery"]["where"]
    # First AND clause is the status filter, now an IN over [approved, pending].
    clauses = where["compositeFilter"]["filters"]
    status = next(f for f in clauses if f["fieldFilter"]["field"]["fieldPath"] == "status")
    assert status["fieldFilter"]["op"] == "IN"
    vals = status["fieldFilter"]["value"]["arrayValue"]["values"]
    assert {v["stringValue"] for v in vals} == {"approved", "pending"}


@pytest.mark.asyncio
async def test_search_devices_model_query_filters_client_side():
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../devices/d1", "fields": {"model_lc": {"stringValue": "wat28"}}}},
        {"document": {"name": ".../devices/d2", "fields": {"model_lc": {"stringValue": "smv"}}}},
    ]))
    c = _client(s)
    items = await c.search_devices(brand="Bosch", model_query="wat")
    assert [i["id"] for i in items] == ["d1"]


@pytest.mark.asyncio
async def test_list_brands_prefix_filter():
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
        {"document": {"name": ".../brands/miele", "fields": {"brand_lc": {"stringValue": "miele"}}}},
    ]))
    c = _client(s)
    items = await c.list_brands(q="bo")
    assert [i["id"] for i in items] == ["bosch"]
    assert s.posts[-1][1]["json"]["structuredQuery"]["from"] == [{"collectionId": "brands"}]


@pytest.mark.asyncio
async def test_device_profiles_resolves_id_and_includes_pending():
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../profiles/p1", "fields": {"program": {"stringValue": "Cotton 40"}, "status": {"stringValue": "pending"}}}},
    ]))
    c = _client(s)
    res = await c.device_profiles("Bosch", "WAT 28660", "washer")
    assert res["device_id"] == "washer__bosch__wat-28660"
    assert res["items"][0]["program"] == "Cotton 40"
    # include_pending -> status IN [approved, pending]
    clauses = s.posts[-1][1]["json"]["structuredQuery"]["where"]["compositeFilter"]["filters"]
    status = next(f for f in clauses if f["fieldFilter"]["field"]["fieldPath"] == "status")
    assert status["fieldFilter"]["op"] == "IN"


@pytest.mark.asyncio
async def test_get_device_quality_decodes_aggregation():
    s = _Session()
    s.queue_post(_Resp(200, [{"result": {"aggregateFields": {
        "cnt": {"integerValue": "3"}, "avg": {"doubleValue": 4.25}}}}]))
    c = _client(s)
    q = await c.get_device_quality("washer__bosch__wat")
    assert q == {"avg": 4.25, "count": 3}


@pytest.mark.asyncio
async def test_get_cycles_includes_pending_and_attaches_rating():
    s = _Session()
    # 1: the cycles list query (one v1 cycle so it hydrates an importable trace).
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../cycles/c1", "fields": {
            "status": {"stringValue": "pending"},
            "downloads": {"integerValue": "4"},
            "confirmCount": {"integerValue": "1"},
            "cycleSchemaVersion": {"integerValue": "1"},
            "trace": {"mapValue": {"fields": {"points": {"arrayValue": {"values": [
                {"mapValue": {"fields": {"o": {"integerValue": "0"}, "w": {"doubleValue": 5.0}}}},
                {"mapValue": {"fields": {"o": {"integerValue": "60"}, "w": {"doubleValue": 0.0}}}},
            ]}}}}},
        }}},
    ]))
    # 2: the per-cycle rating aggregation.
    s.queue_post(_Resp(200, [{"result": {"aggregateFields": {
        "cnt": {"integerValue": "2"}, "avg": {"doubleValue": 4.5}}}}]))
    c = _client(s)
    items = await c.get_cycles("washer__bosch__wat__cotton-40")
    assert len(items) == 1
    # include_pending default -> status IN [approved, pending]
    clauses = s.posts[0][1]["json"]["structuredQuery"]["where"]["compositeFilter"]["filters"]
    status = next(f for f in clauses if f["fieldFilter"]["field"]["fieldPath"] == "status")
    assert status["fieldFilter"]["op"] == "IN"
    # rating summary attached + map points hydrated to [[o, w]] pairs
    assert items[0]["rating"] == {"avg": 4.5, "count": 2}
    assert items[0]["importable"] == [[0, 5.0], [60, 0.0]]


@pytest.mark.asyncio
async def test_confirm_device_batch_shape_no_promote():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))  # token
    s.queue_post(_Resp(200, {}))                                        # commit (confirm)
    s.queue_get(_Resp(200, {"name": ".../devices/d1", "fields": {
        "confirmCount": {"integerValue": "3"}, "status": {"stringValue": "pending"}}}))
    s.queue_get(_Resp(200, {"name": ".../config/site", "fields": {"confirmThreshold": {"integerValue": "5"}}}))
    c = _client(s)
    res = await c.confirm_device("refresh", "u1", "d1")
    assert res == {"confirmed": True, "confirmCount": 3, "status": "pending"}
    writes = s.posts[-1][1]["json"]["writes"]  # the confirm commit
    assert writes[0]["currentDocument"] == {"exists": False}
    assert writes[0]["update"]["fields"]["uid"] == {"stringValue": "u1"}
    assert writes[1]["transform"]["fieldTransforms"][0]["fieldPath"] == "confirmCount"
    assert writes[1]["transform"]["fieldTransforms"][0]["increment"] == {"integerValue": "1"}


@pytest.mark.asyncio
async def test_confirm_device_promotes_at_threshold():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))  # token
    s.queue_post(_Resp(200, {}))                                        # commit (confirm)
    s.queue_get(_Resp(200, {"name": ".../devices/d1", "fields": {
        "confirmCount": {"integerValue": "5"}, "status": {"stringValue": "pending"}}}))
    s.queue_get(_Resp(200, {"name": ".../config/site", "fields": {"confirmThreshold": {"integerValue": "5"}}}))
    s.queue_post(_Resp(200, {}))                                        # commit (promote)
    c = _client(s)
    res = await c.confirm_device("refresh", "u1", "d1")
    assert res["status"] == "approved"
    promote = s.posts[-1][1]["json"]["writes"][0]
    assert promote["updateMask"] == {"fieldPaths": ["status"]}
    assert promote["update"]["fields"]["status"] == {"stringValue": "approved"}


@pytest.mark.asyncio
async def test_rate_device_shape():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))  # token
    s.queue_post(_Resp(200, {}))                                        # commit
    c = _client(s)
    ok = await c.rate_device("refresh", "u1", "d1", 4)
    assert ok is True
    write = s.posts[-1][1]["json"]["writes"][0]
    assert write["update"]["fields"]["rating"] == {"integerValue": "4"}
    assert {"fieldPath": "updatedAt", "setToServerValue": "REQUEST_TIME"} in write["updateTransforms"]


@pytest.mark.asyncio
async def test_rate_device_rejects_out_of_range():
    s = _Session()
    c = _client(s)
    assert await c.rate_device("refresh", "u1", "d1", 9) is False
    assert len(s.posts) == 0  # no network for an invalid rating


@pytest.mark.asyncio
async def test_get_config_decodes():
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../config/site", "fields": {
        "maintenance": {"booleanValue": False}, "confirmThreshold": {"integerValue": "7"}}}))
    c = _client(s)
    cfg = await c.get_config()
    assert cfg["confirmThreshold"] == 7 and cfg["maintenance"] is False


@pytest.mark.asyncio
async def test_upload_encodes_points_as_maps_not_nested_arrays():
    # Firestore forbids directly-nested arrays; trace.points must be an array of maps.
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))
    for _ in range(4):
        s.queue_post(_Resp(200, {}))
    c = _client(s)
    await c.upload_reference_cycle(
        "refresh", "uid", "Alice",
        {"applianceType": "washer", "brand": "Bosch", "model": "WAT", "program": "Cotton 40", "sampleIntervalSec": 60},
        [[0, 2000], [60, 100], [120, 0]],
        {"duration": 3600}, 3,
    )
    write = _cycle_write(s)
    vals = write["update"]["fields"]["trace"]["mapValue"]["fields"]["points"]["arrayValue"]["values"]
    assert len(vals) == 3
    assert all("mapValue" in v for v in vals), "points must be maps, not nested arrays"
    f0 = vals[0]["mapValue"]["fields"]
    assert "o" in f0 and "w" in f0 and "arrayValue" not in f0["o"]


@pytest.mark.asyncio
async def test_get_cycle_unpacks_map_points_to_pairs():
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../cycles/c1", "fields": {
        "cycleSchemaVersion": {"integerValue": "1"},
        "trace": {"mapValue": {"fields": {"points": {"arrayValue": {"values": [
            {"mapValue": {"fields": {"o": {"integerValue": "0"}, "w": {"integerValue": "2000"}}}},
            {"mapValue": {"fields": {"o": {"integerValue": "60"}, "w": {"integerValue": "100"}}}},
        ]}}}}}}}))
    c = _client(s)
    cyc = await c.get_cycle("c1")
    assert cyc["importable"] == [[0, 2000], [60, 100]]
    assert cyc["trace"]["points"] == [[0, 2000], [60, 100]]


@pytest.mark.asyncio
async def test_get_cycle_skips_unsupported_schema():
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../cycles/c9", "fields": {
        "cycleSchemaVersion": {"integerValue": "99"},
        "trace": {"mapValue": {"fields": {"points": {"arrayValue": {"values": []}}}}}}}))
    c = _client(s)
    cyc = await c.get_cycle("c9")
    assert cyc["importable"] is None  # unknown schema -> not importable


@pytest.mark.asyncio
async def test_get_cycle_v1_importable():
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../cycles/c1", "fields": {
        "cycleSchemaVersion": {"integerValue": "1"},
        "trace": {"mapValue": {"fields": {"points": {"arrayValue": {"values": [
            {"arrayValue": {"values": [{"integerValue": "0"}, {"integerValue": "100"}]}}]}}}}}}}))
    c = _client(s)
    cyc = await c.get_cycle("c1")
    assert cyc["importable"] == [[0, 100]]


# ── upload shape ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_reference_cycle_shape():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))  # token exchange
    for _ in range(4):  # brand, device, profile, cycle commits
        s.queue_post(_Resp(200, {}))
    c = _client(s)
    cid = await c.upload_reference_cycle(
        "refresh", "uid42", "Alice",
        {"applianceType": "washer", "brand": "Bosch", "model": "WAT28660",
         "program": "Cotton 40", "sampleIntervalSec": 60, "description": "eco"},
        [[0, 2000], [60, 100], [120, 0]],
        {"duration": 3600, "energy_wh": 800, "peak_w": 2000, "mean_w": 200, "signature": {}},
        2,
    )
    assert cid and isinstance(cid, str)
    # Locate the cycle-create commit among the upload's :commit posts.
    cyc_post = next(p for p in s.posts if p[0].endswith(":commit")
                    and any("/cycles/" in w.get("update", {}).get("name", "") for w in p[1]["json"]["writes"]))
    kw = cyc_post[1]
    write = kw["json"]["writes"][0]
    assert kw["headers"]["Authorization"] == "Bearer T"
    assert write["currentDocument"] == {"exists": False}
    assert {"fieldPath": "createdAt", "setToServerValue": "REQUEST_TIME"} in write["updateTransforms"]
    fields = write["update"]["fields"]
    assert fields["qc"] == {"integerValue": "2"}
    assert fields["status"] == {"stringValue": "pending"}
    assert fields["uploaderUid"] == {"stringValue": "uid42"}
    assert fields["deviceId"] == {"stringValue": "washer__bosch__wat28660"}
    # points encoded as an array of {o,w} maps (Firestore forbids nested arrays)
    pts = fields["trace"]["mapValue"]["fields"]["points"]["arrayValue"]["values"]
    assert len(pts) == 3
