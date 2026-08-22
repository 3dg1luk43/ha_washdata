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
    # the exchange must send the refresh-token grant with the supplied token
    assert s.posts[0][1]["data"]["grant_type"] == "refresh_token"
    assert s.posts[0][1]["data"]["refresh_token"] == "refresh123"
    # a DIFFERENT refresh token must not reuse the cached id_token (issue: cache
    # was not keyed to the token that produced it)
    s.queue_post(_Resp(200, {"id_token": "TOK2", "expires_in": "3600"}))
    tok3 = await c.ensure_id_token("other-refresh")
    assert tok3 == "TOK2" and len(s.posts) == 2


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


# ── read cache (catalog + config) ───────────────────────────────────────────────
# The community catalog is public and slow-changing; caching brand/device/config reads
# in memory keeps the panel re-opening a tab from re-querying Firestore (the store's #1
# free-tier read source). See StoreClient class docstring.

@pytest.mark.asyncio
async def test_list_brands_caches_across_calls():
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
        {"document": {"name": ".../brands/miele", "fields": {"brand_lc": {"stringValue": "miele"}}}},
    ]))
    c = _client(s)
    first = await c.list_brands()
    assert [b["id"] for b in first] == ["bosch", "miele"]
    assert len(s.posts) == 1
    # Second call is served from cache -- no new Firestore query.
    second = await c.list_brands()
    assert [b["id"] for b in second] == ["bosch", "miele"]
    assert len(s.posts) == 1
    # A prefix search reuses the same cached rows (filtered in memory), still no query.
    filtered = await c.list_brands(q="mi")
    assert [b["id"] for b in filtered] == ["miele"]
    assert len(s.posts) == 1


@pytest.mark.asyncio
async def test_search_devices_caches_per_key():
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../devices/d1", "fields": {"model_lc": {"stringValue": "wat28"}}}},
    ]))
    c = _client(s)
    await c.search_devices(brand="Bosch", appliance_type="washer")
    assert len(s.posts) == 1
    # Same brand/type/include_pending/page_size -> cached, and a model prefix filters in memory.
    hit = await c.search_devices(brand="Bosch", appliance_type="washer", model_query="wat")
    assert [d["id"] for d in hit] == ["d1"]
    assert len(s.posts) == 1
    # A different brand is a different cache key -> a new query.
    s.queue_post(_Resp(200, []))
    await c.search_devices(brand="Miele", appliance_type="washer")
    assert len(s.posts) == 2


@pytest.mark.asyncio
async def test_concurrent_misses_coalesce_into_one_query():
    # Single-flight: two concurrent cache-misses for the same key share one Firestore read.
    import asyncio
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
    ]))
    c = _client(s)
    r1, r2 = await asyncio.gather(c.list_brands(), c.list_brands())
    assert [b["id"] for b in r1] == ["bosch"]
    assert [b["id"] for b in r2] == ["bosch"]
    assert len(s.posts) == 1  # coalesced: a second query would have hit the empty default resp


@pytest.mark.asyncio
async def test_get_config_caches_and_never_caches_failure():
    s = _Session()
    # First read fails (non-200) -> {} and NOT cached.
    s.queue_get(_Resp(500, {}))
    c = _client(s)
    assert await c.get_config() == {}
    assert len(s.gets) == 1
    # A later successful read is fetched (failure was not cached) and then cached.
    s.queue_get(_Resp(200, {"name": ".../config/site", "fields": {"confirmThreshold": {"integerValue": "5"}}}))
    assert (await c.get_config())["confirmThreshold"] == 5
    assert len(s.gets) == 2
    assert (await c.get_config())["confirmThreshold"] == 5
    assert len(s.gets) == 2  # served from cache


@pytest.mark.asyncio
async def test_catalog_cache_invalidated_on_create():
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
    ]))
    c = _client(s)
    await c.list_brands()
    await c.list_brands()
    assert len(s.posts) == 1  # cached
    # An upload creates the brand/device docs -> catalog cache must be dropped so the
    # newly-contributed entry appears immediately for this user.
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))  # token
    for _ in range(4):  # brand/device/profile/cycle creates (200 => created)
        s.queue_post(_Resp(200, {}))
    meta = {"applianceType": "washer", "brand": "Miele", "model": "WWV", "program": "Cotton", "sampleIntervalSec": 60}
    await c.upload_reference_cycle("refresh", "u1", "Alice", meta, [[0, 2000], [60, 100]], {}, 2)
    posts_after_upload = len(s.posts)
    # The next brand list re-queries (the cache was invalidated on create).
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
    ]))
    await c.list_brands()
    assert len(s.posts) == posts_after_upload + 1


def test_read_cache_get_put_and_expiry():
    c = _client(_Session())
    c._cache_put("k", [1, 2], ttl=100.0)
    assert c._cache_get("k") == [1, 2]
    # An already-expired entry reads as a miss and is evicted.
    c._cache_put("k2", [3], ttl=-1.0)
    assert c._cache_get("k2") is None
    assert "k2" not in c._read_cache



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


@pytest.mark.asyncio
async def test_upload_device_bundle_uploads_each_cycle():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))  # token exchange (cached after)
    c = _client(s)
    device_meta = {"applianceType": "washer", "brand": "Bosch", "model": "WAT28660"}
    items = [
        {"program": "Cotton 40", "points": [[0, 2000], [60, 100], [120, 0]],
         "stats": {"duration": 3600, "energy_wh": 800, "peak_w": 2000}, "qc": 2, "sampleIntervalSec": 60},
        {"program": "Eco 50", "points": [[0, 1500], [60, 50], [120, 0]],
         "stats": {"duration": 5400, "energy_wh": 600, "peak_w": 1500}, "qc": 2, "sampleIntervalSec": 60},
    ]
    res = await c.upload_device_bundle("refresh", "uid1", "Alice", device_meta, items)
    assert res["ok"] is True
    assert len(res["cycle_ids"]) == 2 and all(res["cycle_ids"])
    commits = [w for (url, kw) in s.posts if url.endswith(":commit")
               for w in kw.get("json", {}).get("writes", [])]
    cyc = [w for w in commits if "/cycles/" in w.get("update", {}).get("name", "")]
    prof_ids = {w["update"]["name"].rsplit("/", 1)[-1] for w in commits
                if "/profiles/" in w.get("update", {}).get("name", "")}
    assert len(cyc) == 2  # one reference cycle created per item
    d = sc.device_id("washer", "Bosch", "WAT28660")
    assert sc.profile_id(d, "Cotton 40") in prof_ids
    assert sc.profile_id(d, "Eco 50") in prof_ids


@pytest.mark.asyncio
async def test_get_device_bundle_groups_cycles_under_profiles():
    s = _Session()
    d = sc.device_id("washer", "Bosch", "WAT28660")
    p = sc.profile_id(d, "Cotton 40")
    # 1: profiles query for the device
    s.queue_post(_Resp(200, [
        {"document": {"name": f".../profiles/{p}", "fields": {
            "program": {"stringValue": "Cotton 40"}, "deviceId": {"stringValue": d},
            "status": {"stringValue": "approved"}}}},
    ]))
    # 2: cycles query for that profile (one v1 cycle -> hydrates importable pairs)
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../cycles/c1", "fields": {
            "status": {"stringValue": "approved"}, "cycleSchemaVersion": {"integerValue": "1"},
            "trace": {"mapValue": {"fields": {"points": {"arrayValue": {"values": [
                {"mapValue": {"fields": {"o": {"integerValue": "0"}, "w": {"doubleValue": 5.0}}}},
                {"mapValue": {"fields": {"o": {"integerValue": "60"}, "w": {"doubleValue": 0.0}}}},
            ]}}}}}}}},
    ]))
    # 3: rating aggregation for c1
    s.queue_post(_Resp(200, [{"result": {"aggregateFields": {"cnt": {"integerValue": "0"}}}}]))
    c = _client(s)
    bundle = await c.get_device_bundle(d)
    assert bundle["device_id"] == d
    assert len(bundle["profiles"]) == 1
    prof = bundle["profiles"][0]
    assert prof["program"] == "Cotton 40"
    assert len(prof["cycles"]) == 1
    assert prof["cycles"][0]["importable"] == [[0, 5.0], [60, 0.0]]


@pytest.mark.asyncio
async def test_upload_reference_cycle_is_idempotent():
    """Same trace -> same deterministic id; a re-upload is refused server-side and
    reported as created=False (not a new doc)."""
    pts = [[0, 2000], [60, 100], [120, 0]]
    meta = {"applianceType": "washer", "brand": "Bosch", "model": "WAT28660",
            "program": "Cotton 40", "sampleIntervalSec": 60}
    expected = sc.trace_hash(sc.profile_id(sc.device_id("washer", "Bosch", "WAT28660"), "Cotton 40"), pts)

    # First upload: token + 4 creates (brand/device/profile/cycle) all 200.
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))
    for _ in range(4):
        s.queue_post(_Resp(200, {}))
    c = _client(s)
    r1 = await c.upload_reference_cycle("refresh", "u1", "Alice", meta, pts, {}, 2, return_status=True)
    assert r1 == {"id": expected, "created": True}
    cyc = _cycle_write(s)
    assert cyc["update"]["name"].endswith(f"/cycles/{expected}")
    assert cyc["update"]["fields"]["traceHash"] == {"stringValue": expected}

    # Second upload of the SAME trace: cycle create returns ALREADY_EXISTS -> no new doc.
    s2 = _Session()
    s2.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))
    for _ in range(3):
        s2.queue_post(_Resp(200, {}))  # brand/device/profile upserts
    s2.queue_post(_Resp(409, {"error": {"status": "ALREADY_EXISTS"}}))  # cycle already there
    c2 = _client(s2)
    r2 = await c2.upload_reference_cycle("refresh", "u1", "Alice", meta, pts, {}, 2, return_status=True)
    assert r2 == {"id": expected, "created": False}
    # Bare-id default return is preserved for the single-cycle share path.
    assert isinstance(await c2.upload_reference_cycle("refresh", "u1", "Alice", meta, pts, {}, 2), str)


@pytest.mark.asyncio
async def test_upload_device_bundle_counts_new_vs_duplicate():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))  # token
    # item 1: brand/device/profile + cycle all created (200)
    for _ in range(4):
        s.queue_post(_Resp(200, {}))
    # item 2: brand/device/profile upserts (200) then the cycle already exists (409)
    for _ in range(3):
        s.queue_post(_Resp(200, {}))
    s.queue_post(_Resp(409, {"error": {"status": "ALREADY_EXISTS"}}))
    c = _client(s)
    device_meta = {"applianceType": "washer", "brand": "Bosch", "model": "WAT28660"}
    items = [
        {"program": "Cotton 40", "points": [[0, 2000], [60, 100], [120, 0]], "stats": {}, "qc": 2, "sampleIntervalSec": 60},
        {"program": "Eco 50", "points": [[0, 1500], [60, 50], [120, 0]], "stats": {}, "qc": 2, "sampleIntervalSec": 60},
    ]
    res = await c.upload_device_bundle("refresh", "u1", "Alice", device_meta, items)
    assert res["ok"] is True
    assert res["created"] == 1 and res["duplicates"] == 1
    assert len(res["cycle_ids"]) == 2 and res["errors"] == []


@pytest.mark.asyncio
async def test_upload_reference_cycle_attaches_phases():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))
    for _ in range(4):  # brand/device/profile/cycle
        s.queue_post(_Resp(200, {}))
    c = _client(s)
    meta = {"applianceType": "washer", "brand": "Bosch", "model": "WAT28660",
            "program": "Cotton 40", "sampleIntervalSec": 60,
            "phases": [{"name": "Wash", "start": 0, "end": 600}, {"name": "Spin", "start": 600, "end": 900}],
            "phaseSourceCycleId": "abc123"}
    await c.upload_reference_cycle("refresh", "u1", "Alice", meta, [[0, 2000], [60, 100]], {}, 2)
    # Locate the profile-create commit.
    prof = next(w for (url, kw) in s.posts if url.endswith(":commit")
                for w in kw["json"]["writes"] if "/profiles/" in w.get("update", {}).get("name", ""))
    fields = prof["update"]["fields"]
    assert fields["phasesSchemaVersion"] == {"integerValue": "1"}
    assert fields["phaseSourceCycleId"] == {"stringValue": "abc123"}
    vals = fields["phases"]["arrayValue"]["values"]
    assert len(vals) == 2
    assert vals[0]["mapValue"]["fields"]["name"] == {"stringValue": "Wash"}


@pytest.mark.asyncio
async def test_upload_reference_cycle_no_phases_when_absent():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))
    for _ in range(4):
        s.queue_post(_Resp(200, {}))
    c = _client(s)
    meta = {"applianceType": "washer", "brand": "Bosch", "model": "WAT28660", "program": "Cotton 40"}
    await c.upload_reference_cycle("refresh", "u1", "Alice", meta, [[0, 2000], [60, 100]], {}, 2)
    prof = next(w for (url, kw) in s.posts if url.endswith(":commit")
                for w in kw["json"]["writes"] if "/profiles/" in w.get("update", {}).get("name", ""))
    assert "phases" not in prof["update"]["fields"]


@pytest.mark.asyncio
async def test_upload_reference_cycle_attaches_settings():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))
    for _ in range(4):
        s.queue_post(_Resp(200, {}))
    c = _client(s)
    meta = {"applianceType": "washer", "brand": "Bosch", "model": "WAT28660", "program": "Cotton 40",
            "settings": {"start_threshold_w": 12.0, "off_delay": 180}}
    await c.upload_reference_cycle("refresh", "u1", "Alice", meta, [[0, 2000], [60, 100]], {}, 2)
    dev = next(w for (url, kw) in s.posts if url.endswith(":commit")
               for w in kw["json"]["writes"] if "/devices/" in w.get("update", {}).get("name", ""))
    setmap = dev["update"]["fields"]["settings"]["mapValue"]["fields"]
    assert setmap["start_threshold_w"] == {"doubleValue": 12.0}
    assert setmap["off_delay"] == {"integerValue": "180"}


@pytest.mark.asyncio
async def test_get_device_bundle_includes_settings():
    s = _Session()
    d = sc.device_id("washer", "Bosch", "WAT28660")
    p = sc.profile_id(d, "Cotton 40")
    # 1: device GET (carries settings)
    s.queue_get(_Resp(200, {"name": f".../devices/{d}", "fields": {
        "settings": {"mapValue": {"fields": {"start_threshold_w": {"doubleValue": 12.0}}}}}}))
    # 2: profiles query
    s.queue_post(_Resp(200, [
        {"document": {"name": f".../profiles/{p}", "fields": {
            "program": {"stringValue": "Cotton 40"}, "deviceId": {"stringValue": d},
            "status": {"stringValue": "approved"}}}},
    ]))
    # 3: cycles query for that profile
    s.queue_post(_Resp(200, []))
    c = _client(s)
    bundle = await c.get_device_bundle(d)
    assert bundle["settings"] == {"start_threshold_w": 12.0}
    assert bundle["profiles"][0]["program"] == "Cotton 40"


# ── read-budget reductions (see the StoreClient class docstring) ────────────────
# The store runs on Firebase's free tier (50k document reads/day, shared by every
# install). These tests pin the behaviours that keep the catalog off that budget:
# resolving a known id with a point read, answering a prefix search server-side, and
# never re-querying for a subset of rows already held in memory.

def _q(session):
    """The structuredQuery of the most recent :runQuery POST."""
    return session.posts[-1][1]["json"]["structuredQuery"]


@pytest.mark.asyncio
async def test_list_brands_prefix_uses_range_query_not_full_list():
    # A prefix search must NOT download the whole collection to filter it in memory:
    # a brand_lc range query reads only the matching docs (measured on the live
    # catalog: 3 docs for "bo" against 84 for the full list).
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
    ]))
    c = _client(s)
    rows = await c.list_brands(q="bo")
    assert [r["id"] for r in rows] == ["bosch"]
    ops = {
        f["fieldFilter"]["op"]
        for f in _q(s)["where"]["compositeFilter"]["filters"]
        if f["fieldFilter"]["field"]["fieldPath"] == "brand_lc"
    }
    assert ops == {"GREATER_THAN_OR_EQUAL", "LESS_THAN_OR_EQUAL"}


@pytest.mark.asyncio
async def test_list_brands_narrowing_prefix_served_from_cached_prefix():
    # Typing extends the prefix one character at a time. A result set for "bo" already
    # contains every brand starting with "bos", so only the first keystroke may query.
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
        {"document": {"name": ".../brands/bomann", "fields": {"brand_lc": {"stringValue": "bomann"}}}},
    ]))
    c = _client(s)
    await c.list_brands(q="bo")
    assert len(s.posts) == 1
    assert [r["id"] for r in await c.list_brands(q="bos")] == ["bosch"]
    assert [r["id"] for r in await c.list_brands(q="bosch")] == ["bosch"]
    assert len(s.posts) == 1  # both narrower prefixes answered from memory


@pytest.mark.asyncio
async def test_full_brand_list_answers_later_prefix_searches():
    # Opening the dropdown (no q) fetches everything; every subsequent search is free.
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
        {"document": {"name": ".../brands/miele", "fields": {"brand_lc": {"stringValue": "miele"}}}},
    ]))
    c = _client(s)
    await c.list_brands()
    assert [r["id"] for r in await c.list_brands(q="mi")] == ["miele"]
    assert len(s.posts) == 1


@pytest.mark.asyncio
async def test_approved_only_served_from_cached_pending_superset():
    # A pending-inclusive result already contains every approved row, so an
    # approved-only caller must be filtered out of it rather than issuing a second,
    # narrower query for a strict subset.
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../devices/d1", "fields": {
            "model_lc": {"stringValue": "wat28"}, "status": {"stringValue": "approved"}}}},
        {"document": {"name": ".../devices/d2", "fields": {
            "model_lc": {"stringValue": "wat29"}, "status": {"stringValue": "pending"}}}},
    ]))
    c = _client(s)
    await c.search_devices(brand="Bosch", appliance_type="washer", include_pending=True)
    assert len(s.posts) == 1
    approved = await c.search_devices(brand="Bosch", appliance_type="washer", include_pending=False)
    assert [d["id"] for d in approved] == ["d1"]
    assert len(s.posts) == 1


@pytest.mark.asyncio
async def test_approved_only_is_not_upgraded_to_the_superset_query():
    # The reverse must NOT happen: widening an approved-only request to the
    # pending-inclusive query would raise its read count (on the live catalog, ~6
    # approved devices of one type against ~140 pending-inclusive).
    s = _Session()
    s.queue_post(_Resp(200, []))
    c = _client(s)
    await c.search_devices(brand="Bosch", appliance_type="washer", include_pending=False)
    status = next(
        f for f in _q(s)["where"]["compositeFilter"]["filters"]
        if f["fieldFilter"]["field"]["fieldPath"] == "status"
    )
    assert status["fieldFilter"]["op"] == "EQUAL"
    assert status["fieldFilter"]["value"]["stringValue"] == "approved"


@pytest.mark.asyncio
async def test_catalog_entry_uses_two_point_reads_and_no_query():
    # The settings-form status badges: two GETs by deterministic id, replacing the
    # brand list + device list this used to be a by-product of.
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../brands/bosch", "fields": {
        "brand": {"stringValue": "Bosch"}, "status": {"stringValue": "approved"}}}))
    s.queue_get(_Resp(200, {"name": ".../devices/washer__bosch__wat28660", "fields": {
        "model": {"stringValue": "WAT28660"}, "status": {"stringValue": "pending"},
        "confirmCount": {"integerValue": "2"}}}))
    c = _client(s)
    res = await c.catalog_entry("Bosch", "WAT28660", "washer")
    assert res["device_id"] == "washer__bosch__wat28660"
    assert res["brand"]["status"] == "approved"
    assert res["device"]["confirmCount"] == 2
    assert len(s.gets) == 2
    assert not s.posts  # no list query at all
    # Both documents are cached, so re-rendering the form costs nothing.
    await c.catalog_entry("Bosch", "WAT28660", "washer")
    assert len(s.gets) == 2


@pytest.mark.asyncio
async def test_catalog_entry_reports_missing_entries_as_none():
    s = _Session()
    s.queue_get(_Resp(404, {}))
    s.queue_get(_Resp(404, {}))
    c = _client(s)
    res = await c.catalog_entry("Nope", "X1", "washer")
    assert res["brand"] is None and res["device"] is None
    # A miss is NOT cached -- it flips as soon as the user contributes the entry.
    s.queue_get(_Resp(200, {"name": ".../brands/nope", "fields": {"status": {"stringValue": "pending"}}}))
    s.queue_get(_Resp(404, {}))
    assert (await c.catalog_entry("Nope", "X1", "washer"))["brand"]["status"] == "pending"


@pytest.mark.asyncio
async def test_catalog_entry_skips_lookups_for_blank_identity():
    s = _Session()
    c = _client(s)
    res = await c.catalog_entry("", "", "washer")
    assert res == {"device_id": "", "brand": None, "device": None}
    assert not s.gets and not s.posts


@pytest.mark.asyncio
async def test_point_reads_percent_encode_the_document_id():
    # brand_id() is just brand.lower() with no normalisation, so the live catalog holds
    # ids like "aeg lavamat" and "fisher & paykel". Splicing those into a URL raw
    # produces a malformed request, not a 404.
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../brands/aeg lavamat", "fields": {}}))
    c = _client(s)
    await c.get_brand("AEG Lavamat")
    assert s.gets[-1][0].endswith("/brands/aeg%20lavamat")


@pytest.mark.asyncio
async def test_catalog_invalidation_drops_point_read_cache_too():
    # Contributing a brand writes the very document the badge resolves, so a stale hit
    # would keep showing "not in the catalog" for an entry the user just added.
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../brands/bosch", "fields": {"status": {"stringValue": "approved"}}}))
    c = _client(s)
    await c.get_brand("Bosch")
    assert len(s.gets) == 1
    c._invalidate_catalog_cache()
    s.queue_get(_Resp(200, {"name": ".../brands/bosch", "fields": {"status": {"stringValue": "approved"}}}))
    await c.get_brand("Bosch")
    assert len(s.gets) == 2


@pytest.mark.asyncio
async def test_refresh_catalog_forces_a_requery():
    s = _Session()
    s.queue_post(_Resp(200, [
        {"document": {"name": ".../brands/bosch", "fields": {"brand_lc": {"stringValue": "bosch"}}}},
    ]))
    c = _client(s)
    await c.list_brands()
    await c.list_brands()
    assert len(s.posts) == 1
    c.refresh_catalog()
    s.queue_post(_Resp(200, []))
    await c.list_brands()
    assert len(s.posts) == 2


@pytest.mark.asyncio
async def test_list_queries_project_only_the_fields_the_ui_reads():
    # A projection does not change the billed read count, but these lists are relayed
    # verbatim to the panel, and device docs carry a ~25-key settings map no list view
    # reads (measured: one brand's device list 54.4 KB -> 17 KB).
    s = _Session()
    s.queue_post(_Resp(200, []))
    c = _client(s)
    await c.list_brands()
    assert [f["fieldPath"] for f in _q(s)["select"]["fields"]] == list(sc._BRAND_LIST_FIELDS)
    s.queue_post(_Resp(200, []))
    await c.search_devices(brand="Bosch", appliance_type="washer")
    assert [f["fieldPath"] for f in _q(s)["select"]["fields"]] == list(sc._DEVICE_LIST_FIELDS)
    assert "settings" not in sc._DEVICE_LIST_FIELDS


@pytest.mark.asyncio
async def test_bundle_skips_the_per_cycle_rating_fanout():
    # Ratings are browse-only decoration the adopt path never reads, but they cost one
    # aggregation request per cycle.
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../devices/d1", "fields": {}}))          # get_device
    s.queue_post(_Resp(200, [                                                  # get_profiles
        {"document": {"name": ".../profiles/p1", "fields": {"program": {"stringValue": "Cotton"}}}},
    ]))
    s.queue_post(_Resp(200, [                                                  # get_cycles
        {"document": {"name": ".../cycles/c1", "fields": {}}},
        {"document": {"name": ".../cycles/c2", "fields": {}}},
    ]))
    c = _client(s)
    bundle = await c.get_device_bundle("d1")
    assert [cyc["id"] for cyc in bundle["profiles"][0]["cycles"]] == ["c1", "c2"]
    assert not [u for u, _ in s.posts if "runAggregationQuery" in u]
    # ...while the browse path still attaches them.
    s.queue_post(_Resp(200, [{"document": {"name": ".../cycles/c1", "fields": {}}}]))
    s.queue_post(_Resp(200, [{"result": {"aggregateFields": {
        "cnt": {"integerValue": "2"}, "avg": {"doubleValue": 4.5}}}}]))
    cycles = await c.get_cycles("p1")
    assert cycles[0]["rating"] == {"avg": 4.5, "count": 2}


def test_shared_client_is_one_per_install():
    # A StoreBridge exists per appliance, but the catalog it reads is public and
    # device-agnostic: a per-bridge client made an N-appliance install issue N
    # cold-cache copies of the same queries.
    hass = MagicMock()
    hass.data = {}
    first = sc.get_client(hass)
    assert sc.get_client(hass) is first


# ── review regressions (2026-08-20) ─────────────────────────────────────────────
# Caching the point read that confirm_device uses to read its own write back is the
# kind of bug that only shows up as "community approval never happens".

@pytest.mark.asyncio
async def test_confirm_device_does_not_read_its_own_write_from_cache():
    s = _Session()
    s.queue_post(_Resp(200, {"id_token": "T", "expires_in": "3600"}))     # token
    # The settings badge warms devices/<id> BEFORE the confirm (this is the normal order).
    s.queue_get(_Resp(200, {"name": ".../devices/d1", "fields": {
        "status": {"stringValue": "pending"}, "confirmCount": {"integerValue": "4"}}}))
    c = _client(s)
    assert (await c.get_device("d1"))["confirmCount"] == 4

    s.queue_post(_Resp(200, {}))                                          # the confirm commit
    # Post-write state: threshold reached. If the cached pre-write doc were reused, count
    # would still read 4, `count >= threshold` would be False, and the device would never
    # be promoted.
    s.queue_get(_Resp(200, {"name": ".../devices/d1", "fields": {
        "status": {"stringValue": "pending"}, "confirmCount": {"integerValue": "5"}}}))
    s.queue_get(_Resp(200, {"name": ".../config/site", "fields": {
        "confirmThreshold": {"integerValue": "5"}}}))
    s.queue_post(_Resp(200, {}))                                          # the promote commit
    res = await c.confirm_device("refresh", "u1", "d1")
    assert res["confirmCount"] == 5
    assert res["status"] == "approved"


@pytest.mark.asyncio
async def test_point_read_spanning_an_invalidation_is_not_cached():
    """A read in flight when a write lands must not re-pin the pre-write document."""
    s = _Session()
    s.queue_get(_Resp(200, {"name": ".../brands/bosch", "fields": {
        "status": {"stringValue": "pending"}}}))
    c = _client(s)

    real_get = c._sess().get
    def _get_and_invalidate(url, **kw):
        c._invalidate_catalog_cache()   # a write lands mid-flight
        return real_get(url, **kw)
    c._session.get = _get_and_invalidate

    assert (await c.get_brand("Bosch"))["status"] == "pending"
    c._session.get = real_get
    # Not cached, so the next read goes back to the store and sees the new state.
    s.queue_get(_Resp(200, {"name": ".../brands/bosch", "fields": {
        "status": {"stringValue": "approved"}}}))
    assert (await c.get_brand("Bosch"))["status"] == "approved"
    assert len(s.gets) == 2


@pytest.mark.asyncio
async def test_truncated_superset_does_not_answer_an_approved_only_request():
    """A list capped at page_size may be missing approved rows past the cap, so it
    cannot be filtered down to answer a narrower question."""
    s = _Session()
    # page_size=2 and exactly 2 rows back => possibly truncated.
    def row(lc, status):
        return {"document": {"name": f".../brands/{lc}", "fields": {
            "brand_lc": {"stringValue": lc}, "status": {"stringValue": status}}}}
    s.queue_post(_Resp(200, [row("aaa", "pending"), row("bbb", "pending")]))
    c = _client(s)
    assert len(await c.list_brands(include_pending=True, page_size=2)) == 2
    assert len(s.posts) == 1
    # Must re-query rather than concluding "no approved brands" from a truncated page.
    s.queue_post(_Resp(200, [row("zzz", "approved")]))
    assert [b["id"] for b in await c.list_brands(include_pending=False, page_size=2)] == ["zzz"]
    assert len(s.posts) == 2
