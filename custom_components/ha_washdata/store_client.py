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
"""Async Firestore-REST client for the WashData community store (v2 hierarchy).

Reads (approved brands/devices/profiles/cycles) are public and need no token. Writes
(upload a reference cycle) use the signed-in user's Firebase ID token, obtained by
exchanging the refresh token handed over by the store's connect page.

No Firebase SDK, no new dependency: plain aiohttp via Home Assistant's shared session.
Never raises into the event loop - failures return ``None``/empty and are logged.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    SHAREABLE_SETTING_KEYS,
    STORE_API_KEY,
    STORE_PROJECT_ID,
    SUPPORTED_CYCLE_SCHEMA_VERSIONS,
)

_LOGGER = logging.getLogger(__name__)

_APPLIANCE_TYPES = {"washer", "dryer", "dishwasher", "washer_dryer"}

# Max concurrent per-cycle rating aggregations when listing a profile's cycles.
_RATING_FANOUT_LIMIT = 8

# Max profiles hydrated concurrently when downloading a whole-device bundle. One query
# each (the bundle skips the per-cycle rating fan-out), kept small to stay well under the
# store's rate limiter on devices that carry many profiles.
_BUNDLE_HYDRATE_LIMIT = 4

# Upper bound for a Firestore "starts with" range on a string field: U+F8FF sits above
# every character that realistically appears in a brand name, so [p, p + _PREFIX_MAX]
# selects exactly the values beginning with p. Written as an escape on purpose -- the
# literal glyph is invisible in an editor and trivially lost to a copy/paste.
_PREFIX_MAX = "\uf8ff"

# Field projections for the catalog list queries (Firestore `select`). A projection does
# NOT reduce the billed document count -- Firestore bills per document read regardless --
# but it does cut the wire payload substantially, which matters because these lists are
# relayed verbatim to the panel over the HA WebSocket. Measured on the live catalog: the
# brand list drops 64.7 KB -> 33 KB, and a single brand's device list 54.4 KB -> 17 KB
# (device docs carry a ~25-key `settings` map that no list view reads; the whole-device
# bundle fetches the full doc via get_device instead).
#
# These are ALLOW-lists: a field missing here is absent from the decoded row, so keep
# them in sync with what the panel + StoreBridge actually consume. `id` is derived from
# the document name and is always present.
_BRAND_LIST_FIELDS = ("brand", "brand_lc", "status")
_DEVICE_LIST_FIELDS = (
    "brand", "brand_lc", "model", "model_lc", "applianceType", "status",
    "confirmCount", "favoriteCount", "manualUrl", "createdByName",
    # Content markers for the Store browse list. Only ~30% of catalog entries carry any
    # shared program (168 of 564 measured), so "does this entry actually have anything?"
    # is the most useful thing a row can say. NB these are contributor-maintained
    # counters and are known to under-report where a best-effort increment was denied,
    # so the UI shows a chip when the count is positive and says nothing when it is
    # zero/absent -- never "this is empty", which a stale zero would make a lie.
    "profileCount", "cycleCount",
)

# Domain-scoped key for the process-wide shared client (see get_client).
_CLIENT_KEY = f"{DOMAIN}_store_client"


def get_client(hass: HomeAssistant) -> "StoreClient":
    """The one shared StoreClient for this HA install.

    The community catalog is public, device-agnostic data, but a StoreBridge is created
    per config entry -- so a per-bridge client made an N-appliance install pay N times
    over for the exact same brand/device lists, each with its own cold cache. Hanging a
    single instance off ``hass.data`` collapses that to one, and (unlike a bridge) it is
    deliberately NOT removed on unload so the read cache also survives an entry reload.
    Mirrors the global-state pattern in ``store_account``.
    """
    client = hass.data.get(_CLIENT_KEY)
    if not isinstance(client, StoreClient):
        client = StoreClient(hass)
        hass.data[_CLIENT_KEY] = client
    return client


def _seg(value: Any) -> str:
    """Percent-encode one REST path segment.

    Store document ids are raw user text: ``brand_id`` is just ``brand.lower()`` with no
    normalisation, so the live catalog really does contain ids like ``aeg lavamat``,
    ``fisher & paykel`` and ``ok.``. Interpolating those into a URL unencoded produces a
    malformed request rather than a 404, so every id-in-path must go through this.
    """
    return quote(str(value if value is not None else ""), safe="")


# ── deterministic ids (must match the store's lib/ids.js exactly) ──────────────

def normalize_token(s: Any) -> str:
    """lowercase -> NFKD -> collapse non-alphanumerics to '-' -> trim '-'."""
    text = unicodedata.normalize("NFKD", str(s if s is not None else "").lower())
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def device_id(appliance_type: str, brand: str, model: str) -> str:
    return "__".join((normalize_token(appliance_type), normalize_token(brand), normalize_token(model)))


def profile_id(dev_id: str, program: str) -> str:
    return f"{dev_id}__{normalize_token(program)}"


def brand_id(brand: str) -> str:
    return str(brand or "").lower()


# ── typed-value encode/decode (Firestore REST) ─────────────────────────────────

def _encode(v: Any) -> dict[str, Any]:
    if v is None:
        return {"nullValue": None}
    if isinstance(v, bool):
        return {"booleanValue": v}
    if isinstance(v, int):
        return {"integerValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_encode(x) for x in v]}}
    if isinstance(v, dict):
        return {"mapValue": {"fields": {k: _encode(x) for k, x in v.items()}}}
    return {"stringValue": str(v)}


def _decode(v: dict[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "integerValue" in v:
        return int(v["integerValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "booleanValue" in v:
        return v["booleanValue"]
    if "nullValue" in v:
        return None
    if "timestampValue" in v:
        return v["timestampValue"]
    if "arrayValue" in v:
        return [_decode(x) for x in v["arrayValue"].get("values", [])]
    if "mapValue" in v:
        return {k: _decode(x) for k, x in v["mapValue"].get("fields", {}).items()}
    return None


def _decode_doc(doc: dict[str, Any]) -> dict[str, Any]:
    out = {k: _decode(x) for k, x in doc.get("fields", {}).items()}
    name = doc.get("name", "")
    out["id"] = name.rsplit("/", 1)[-1] if "/" in name else name
    return out


# Firestore forbids directly-nested arrays, so a trace can't be stored as
# [[offset, watts], ...]. On the wire we store an array of {o, w} maps and convert
# to/from [[offset, watts], ...] pairs at the boundary (matches lib/trace.js).
def pack_points(pairs: list[list[float]]) -> list[dict[str, float]]:
    return [{"o": float(p[0]), "w": float(p[1])} for p in pairs if len(p) >= 2]


def unpack_points(points: Any) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(points, list):
        return out
    for p in points:
        if isinstance(p, dict):
            out.append([p.get("o", 0), p.get("w", 0)])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([p[0], p[1]])
    return out


def trace_hash(profile_id_: str, pts: list[list[float]]) -> str:
    """Deterministic content hash for a reference-cycle trace, scoped to its profile.

    Used as the store cycle's document id so an identical trace re-uploaded to the
    same program collides on the same id and is refused server-side (the create
    precondition), making share idempotent. Two DIFFERENT recordings of the same
    program hash differently, so genuine multi-instance contributions are preserved.
    Offsets are rounded to whole seconds and watts to 1 decimal so trivial float
    formatting differences do not change the hash.
    """
    norm = [[int(round(float(p[0]))), round(float(p[1]), 1)] for p in pts if len(p) >= 2]
    payload = f"{profile_id_}|{json.dumps(norm, separators=(',', ':'))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StoreClient:
    """Read/write client for the store. One per manager; safe to keep for the entry."""

    _FS = "https://firestore.googleapis.com/v1"
    _TOKEN = "https://securetoken.googleapis.com/v1/token"

    # The community catalog (brands + device searches) is public and changes rarely, but
    # the panel re-queries it every time the Settings/Store tab is (re-)opened. On the
    # store's Firebase free tier (50k document reads/day) that brand-list query alone was
    # the single largest read source. Cache these reads in memory for a short TTL so a
    # burst of panel opens collapses to at most one Firestore query per key per window.
    # The client is a process-wide singleton (see get_client) so the cache survives panel
    # reloads AND config-entry reloads, and is shared by every appliance. Writes that add
    # brands/devices invalidate it (see _commit_create) so a freshly-contributed entry
    # still appears immediately for the user who added it.
    # The catalog changes a few times a week (a contributor adds a brand, an admin
    # approves one), so a 15-minute window was re-reading a near-static ~650-document
    # catalog dozens of times a day. An hour keeps a browsing session on one query while
    # staying fresh enough to notice someone else's contribution, and it is not the only
    # freshness path: a local create invalidates immediately (_commit_create) and
    # refresh_catalog() force-drops everything on demand.
    _CATALOG_CACHE_TTL_S = 3600.0  # brands + device searches (1 h)
    _CONFIG_CACHE_TTL_S = 3600.0   # config/site (maintenance flag + confirm threshold)
    # Hard cap on distinct cached read keys. Device searches key on the brand term, so a
    # long-lived session that issues many distinct searches would otherwise accumulate an
    # unbounded number of (mostly expired) entries. When the cap is exceeded we evict the
    # soonest-to-expire entries, keeping the live TTL guarantee while bounding memory.
    _MAX_READ_CACHE_ENTRIES = 256

    def __init__(
        self,
        hass: HomeAssistant,
        project_id: str = STORE_PROJECT_ID,
        api_key: str = STORE_API_KEY,
        session: Any | None = None,
    ) -> None:
        self._hass = hass
        self._pid = project_id
        self._key = api_key
        self._session = session
        self._id_token: str | None = None
        self._id_token_exp: float = 0.0
        self._id_token_rt: str | None = None  # refresh token that produced the cached id_token
        self._last_error: str | None = None  # short reason for the last failed write, for the UI
        self._base = f"{self._FS}/projects/{project_id}/databases/(default)/documents"
        # key -> (expiry_epoch, value). Read-only catalog/config responses; see class docstring.
        self._read_cache: dict[str, tuple[float, Any]] = {}
        # Bumped on every catalog invalidation; a read captures it before its query and only
        # caches the result if it is unchanged afterwards, so an in-flight read that spans an
        # invalidation cannot re-cache a pre-write snapshot.
        self._cache_gen: int = 0
        # Single-flight: one in-flight (generation, task) per cache key so concurrent panel
        # misses share a single Firestore read instead of each issuing one (matters on the
        # free-tier budget). The generation is stored so a query started before an
        # invalidation is not joined by a post-invalidation caller (which would receive a
        # pre-write snapshot); such a caller starts a fresh query instead.
        self._inflight: dict[str, "tuple[int, asyncio.Future[list[dict[str, Any]]]]"] = {}
        # Guards the write-then-read-_last_error sequence; see the write_lock property.
        self._write_lock = asyncio.Lock()

    def last_error(self) -> str | None:
        return self._last_error

    @property
    def write_lock(self) -> asyncio.Lock:
        """Serialises a write with the ``last_error()`` read that interprets it.

        ``_last_error`` is a single slot cleared at the start of each upload, and this
        client is shared by every appliance in the install (see get_client), so two
        concurrent shares could otherwise interleave clear/set and make one device report
        the other's failure reason. Holding this across the whole upload-then-read
        sequence keeps that attribution correct; it also naturally rate-limits concurrent
        uploads, which the store's free tier appreciates.

        Every caller that can reach a ``_last_error`` writer must hold it, not just the
        ones that read it back: ``ensure_id_token`` also writes the slot, so connect /
        confirm / rate take it too (see ``store.WashDataStore``). It is NOT reentrant --
        take it around a client call, never inside one.
        """
        return self._write_lock

    def _sess(self) -> Any:
        if self._session is None:
            self._session = async_get_clientsession(self._hass)
        return self._session

    # ── read cache (public catalog/config; see class docstring) ─────────────────

    def _cache_get(self, key: str) -> Any | None:
        ent = self._read_cache.get(key)
        if ent is None:
            return None
        expiry, value = ent
        # Monotonic clock: immune to NTP/wall-clock steps that could otherwise extend or
        # truncate the TTL (put + get must use the same clock).
        if time.monotonic() >= expiry:
            self._read_cache.pop(key, None)
            return None
        return value

    def _cache_put(self, key: str, value: Any, ttl: float) -> None:
        self._read_cache[key] = (time.monotonic() + ttl, value)
        if len(self._read_cache) > self._MAX_READ_CACHE_ENTRIES:
            # Drop the soonest-to-expire entries (expired ones first) to stay bounded.
            overflow = len(self._read_cache) - self._MAX_READ_CACHE_ENTRIES
            for k, _ in sorted(self._read_cache.items(), key=lambda kv: kv[1][0])[:overflow]:
                self._read_cache.pop(k, None)

    def _invalidate_catalog_cache(self) -> None:
        """Drop cached brand/device catalog reads (call after a create/upload/promote write so
        a just-contributed or newly-approved entry appears immediately, not after the TTL).

        Covers both the list queries (``brands:``/``devices:``) and the single-document
        lookups behind the identity badges (``brand:``/``device:``) -- a create writes the
        very document those resolve, so a stale hit there would show "not in the catalog"
        for an entry the user just added.
        """
        self._cache_gen += 1
        for key in [
            k for k in self._read_cache
            if k.startswith(("brands:", "devices:", "brand:", "device:"))
        ]:
            self._read_cache.pop(key, None)

    def refresh_catalog(self) -> None:
        """Force the next catalog read to hit Firestore (public wrapper for the panel's
        "refresh catalog" action). With a 1-hour TTL a user who is told by a friend that a
        brand was just approved needs a way to see it without waiting out the window."""
        self._invalidate_catalog_cache()

    # ── auth ──────────────────────────────────────────────────────────────────

    async def ensure_id_token(self, refresh_token: str) -> str | None:
        """Exchange the refresh token for a (cached) Firebase ID token."""
        now = time.time()
        # The cache is only valid for the same refresh token that produced it -- after
        # a disconnect/reconnect (or a different global account) the previous account's
        # token must not be returned even if it is still unexpired.
        if (
            self._id_token
            and self._id_token_rt == refresh_token
            and now < self._id_token_exp - 60
        ):
            return self._id_token
        try:
            async with self._sess().post(
                f"{self._TOKEN}?key={self._key}",
                data={"grant_type": "refresh_token", "refresh_token": refresh_token},
                timeout=15,
            ) as resp:
                if resp.status != 200:
                    _LOGGER.warning("Store token exchange failed: HTTP %s", resp.status)
                    self._last_error = f"sign-in expired (HTTP {resp.status}) - reconnect GitHub in the gear"
                    return None
                body = await resp.json()
        except Exception as exc:  # noqa: BLE001 - never raise into the loop
            _LOGGER.warning("Store token exchange error: %s", exc)
            self._last_error = "could not reach the sign-in service"
            return None
        self._id_token = body.get("id_token")
        self._id_token_rt = refresh_token
        try:
            self._id_token_exp = now + float(body.get("expires_in", 3600))
        except (TypeError, ValueError):
            self._id_token_exp = now + 3600
        return self._id_token

    # ── reads (public, no token) ────────────────────────────────────────────────

    async def _run_query(self, sq: dict[str, Any], parent: str = "") -> list[dict[str, Any]] | None:
        """Run a structured query. Returns the decoded rows on success (possibly an empty
        list), or ``None`` on any HTTP error / network failure so callers can distinguish a
        genuinely-empty result from a transient failure and avoid caching the latter."""
        url = f"{self._base}/{parent}:runQuery" if parent else f"{self._base}:runQuery"
        try:
            async with self._sess().post(url, json={"structuredQuery": sq}, timeout=15) as resp:
                if resp.status != 200:
                    try:
                        body = await resp.json()
                        _LOGGER.warning("Store query HTTP %s: %s", resp.status, body)
                    except Exception:
                        _LOGGER.warning("Store query HTTP %s (no body)", resp.status)
                    return None
                rows = await resp.json()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Store query error: %s", exc)
            return None
        return [_decode_doc(r["document"]) for r in rows if isinstance(r, dict) and "document" in r]

    async def _cached_catalog_query(
        self, key: str, build_sq: "Callable[[], dict[str, Any]]"
    ) -> list[dict[str, Any]]:
        """Shared cache-get -> (on miss) run query -> conditionally-cache flow for the public
        catalog reads. ``build_sq`` is called only on a cache miss (kept lazy). A successful
        result is cached only if no invalidation happened while the query was in flight (the
        generation guard), so an in-flight read cannot re-cache a pre-write snapshot; a
        transient failure (None) is never cached. Callers apply their own in-memory prefix
        filter to the returned full list."""
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        # Single-flight: join an in-flight query for the CURRENT generation; otherwise start
        # a fresh one. An entry from an older generation (started before an invalidation) is
        # deliberately not joined -- it may carry a pre-write snapshot.
        entry = self._inflight.get(key)
        if entry is None or entry[0] != self._cache_gen:
            gen = self._cache_gen
            entry = (gen, asyncio.ensure_future(self._fetch_and_cache(key, gen, build_sq)))
            self._inflight[key] = entry
        # Shield so one waiter's cancellation neither cancels the shared query nor the other
        # waiters coalesced onto it.
        return await asyncio.shield(entry[1])

    async def _fetch_and_cache(
        self, key: str, gen: int, build_sq: "Callable[[], dict[str, Any]]"
    ) -> list[dict[str, Any]]:
        """The shared single-flight body: run the query, cache only a successful, non-
        superseded result, and clear the in-flight slot when done (if it is still ours)."""
        try:
            fetched = await self._run_query(build_sq())
            result = fetched if fetched is not None else []
            if fetched is not None and gen == self._cache_gen:
                self._cache_put(key, result, self._CATALOG_CACHE_TTL_S)
            return result
        finally:
            cur = self._inflight.get(key)
            if cur is not None and cur[1] is asyncio.current_task():
                self._inflight.pop(key, None)

    @staticmethod
    def _field_filter(field: str, op: str, value: Any) -> dict[str, Any]:
        return {"fieldFilter": {"field": {"fieldPath": field}, "op": op, "value": _encode(value)}}

    def _where(self, filters: list[dict[str, Any]]) -> dict[str, Any]:
        if len(filters) == 1:
            return filters[0]
        return {"compositeFilter": {"op": "AND", "filters": filters}}

    def _status_filter(self, include_pending: bool) -> dict[str, Any]:
        """status == approved, or status IN [approved, pending] when browsing the
        community catalog (pending entries are publicly readable, shown with a tag)."""
        if include_pending:
            return {"fieldFilter": {
                "field": {"fieldPath": "status"}, "op": "IN",
                "value": _encode(["approved", "pending"]),
            }}
        return self._field_filter("status", "EQUAL", "approved")

    @staticmethod
    def _approved_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Narrow a pending-inclusive result to the approved rows, in memory."""
        return [r for r in rows if str(r.get("status") or "") == "approved"]

    def _serve_from_superset(
        self, superset_key: str, include_pending: bool, page_size: int
    ) -> list[dict[str, Any]] | None:
        """A warm pending-inclusive cache entry already contains every approved row, so an
        approved-only caller can be served from it instead of issuing a second, narrower
        query for a strict subset of rows we are already holding. Returns None when the
        request wants pending rows anyway, or when the superset is not cached.

        This is why ``include_pending`` stays in the cache key rather than being collapsed
        into one always-pending-inclusive fetch: upgrading an approved-only request to the
        superset query would *raise* its read count (on the live catalog, approved devices
        of one appliance type number ~6 against ~140 pending-inclusive). Sharing downwards
        is free; sharing upwards is not.
        """
        if include_pending:
            return None
        cached = self._cache_get(superset_key)
        if cached is None:
            return None
        # A result capped at page_size may be missing approved rows past the cap, so it
        # cannot answer a narrower question. Mirrors the same guard in
        # _cached_brand_superset.
        if len(cached) >= page_size:
            return None
        return self._approved_only(cached)

    async def search_devices(
        self, brand: str | None = None, appliance_type: str | None = None,
        model_query: str | None = None, *, include_pending: bool = False, page_size: int = 500,
    ) -> list[dict[str, Any]]:
        # Cache the full per-brand/type device list (the model_query prefix filter is applied
        # in memory below), so one Firestore query serves every model prefix. page_size is a
        # full-catalog ceiling, not a UI page size: the in-memory filter can only match what
        # was fetched, so caching a truncated list would hide models past the limit for the
        # whole TTL. A query reads only the docs that exist, so the high ceiling adds no reads
        # for today's catalog while staying complete as it grows.
        base = f"devices:{(brand or '').lower()}:{appliance_type or ''}"
        key = f"{base}:{int(include_pending)}:{page_size}"

        def _finish(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if model_query:
                p = model_query.lower()
                rows = [r for r in rows if str(r.get("model_lc", "")).startswith(p)]
            return rows

        shared = self._serve_from_superset(f"{base}:1:{page_size}", include_pending, page_size)
        if shared is not None:
            return _finish(shared)

        def _build() -> dict[str, Any]:
            filters = [self._status_filter(include_pending)]
            if appliance_type:
                filters.append(self._field_filter("applianceType", "EQUAL", appliance_type))
            if brand:
                filters.append(self._field_filter("brand_lc", "EQUAL", brand.lower()))
            return {
                "from": [{"collectionId": "devices"}],
                "select": {"fields": [{"fieldPath": f} for f in _DEVICE_LIST_FIELDS]},
                "where": self._where(filters),
                "orderBy": [{"field": {"fieldPath": "favoriteCount"}, "direction": "DESCENDING"}],
                "limit": page_size,
            }

        return _finish(await self._cached_catalog_query(key, _build))

    def _cached_brand_superset(self, q: str, include_pending: bool, page_size: int) -> list[dict[str, Any]] | None:
        """Find a cached brand result that provably contains every match for prefix ``q``.

        Checked in order of breadth: the pending-inclusive full list, the same-status full
        list, then the longest cached prefix that is itself a prefix of ``q`` (a result set
        for "bo" contains every brand starting with "bos"). A cached entry is only reusable
        when it was not truncated at ``page_size`` -- a capped result may be missing rows
        that a narrower query would have returned.
        """
        candidates = [f"brands:1:{page_size}"]
        if not include_pending:
            candidates.append(f"brands:0:{page_size}")
        # Longest usable prefix first: fewer rows to filter, same answer.
        candidates += [
            f"brands:p:{int(include_pending)}:{q[:n]}:{page_size}"
            for n in range(len(q), 0, -1)
        ]
        for cand in candidates:
            cached = self._cache_get(cand)
            if cached is not None and len(cached) < page_size:
                return cached if include_pending else self._approved_only(cached)
        return None

    async def list_brands(
        self, q: str | None = None, *, include_pending: bool = True, page_size: int = 500,
    ) -> list[dict[str, Any]]:
        """Brands for the picker. With ``q``, resolved server-side as a prefix range query
        unless a broad-enough cached result can answer it in memory.

        Downloading all ~84 brand documents to render a filtered dropdown was the single
        largest read source in the store. A ``brand_lc`` range query rides the existing
        (status, brand_lc) composite index and reads only the matching documents -- a
        2-character prefix costs single digits against 84 -- and because a prefix result is
        cached, every subsequent keystroke narrowing that prefix is answered from memory.
        The unfiltered list (no ``q``) still fetches everything: it is the "show me the
        dropdown" case, and truncating it would make brands past the cap unfindable.
        """
        prefix = (q or "").strip().lower()
        select = {"fields": [{"fieldPath": f} for f in _BRAND_LIST_FIELDS]}
        order = [{"field": {"fieldPath": "brand_lc"}, "direction": "ASCENDING"}]

        if prefix:
            cached = self._cached_brand_superset(prefix, include_pending, page_size)
            if cached is not None:
                return [r for r in cached if str(r.get("brand_lc", "")).startswith(prefix)]
            key = f"brands:p:{int(include_pending)}:{prefix}:{page_size}"
            rows = await self._cached_catalog_query(key, lambda: {
                "from": [{"collectionId": "brands"}],
                "select": select,
                "where": self._where([
                    self._status_filter(include_pending),
                    self._field_filter("brand_lc", "GREATER_THAN_OR_EQUAL", prefix),
                    self._field_filter("brand_lc", "LESS_THAN_OR_EQUAL", prefix + _PREFIX_MAX),
                ]),
                "orderBy": order,
                "limit": page_size,
            })
            # The range is inclusive of exactly the prefix matches, so no further filter is
            # needed -- but stay defensive in case a cached entry pre-dates this path.
            return [r for r in rows if str(r.get("brand_lc", "")).startswith(prefix)]

        key = f"brands:{int(include_pending)}:{page_size}"
        shared = self._serve_from_superset(f"brands:1:{page_size}", include_pending, page_size)
        if shared is not None:
            return shared
        return await self._cached_catalog_query(key, lambda: {
            "from": [{"collectionId": "brands"}],
            "select": select,
            "where": self._where([self._status_filter(include_pending)]),
            "orderBy": order,
            "limit": page_size,
        })

    async def _get_doc(self, path: str, *, cache_key: str | None = None) -> dict[str, Any] | None:
        """Fetch one document by path. ``None`` for missing/forbidden or on any failure.

        A point read costs exactly one document, so resolving a known id this way is
        vastly cheaper than the list query that used to be used to find it. Only
        successful lookups are cached (a miss may simply mean "not contributed yet",
        which flips as soon as the user contributes it).
        """
        if cache_key is not None:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
        # Capture the generation BEFORE the request: if an invalidation lands while this
        # read is in flight, caching its result would re-pin a pre-write document for the
        # full TTL. Same discipline as _fetch_and_cache.
        gen = self._cache_gen
        try:
            async with self._sess().get(f"{self._base}/{path}", timeout=15) as resp:
                if resp.status != 200:
                    if resp.status not in (403, 404):
                        _LOGGER.debug("Store get %s HTTP %s", path, resp.status)
                    return None
                doc = await resp.json()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store get %s error: %s", path, exc)
            return None
        out = _decode_doc(doc)
        if cache_key is not None and gen == self._cache_gen:
            self._cache_put(cache_key, out, self._CATALOG_CACHE_TTL_S)
        return out

    async def get_device(self, device_id: str) -> dict[str, Any] | None:
        return await self._get_doc(f"devices/{_seg(device_id)}", cache_key=f"device:{device_id}")

    async def get_brand(self, brand: str) -> dict[str, Any] | None:
        """The brand document for a brand name (doc id is the lowercased name)."""
        b_id = brand_id(brand)
        if not b_id:
            return None
        return await self._get_doc(f"brands/{_seg(b_id)}", cache_key=f"brand:{b_id}")

    async def catalog_entry(
        self, brand: str, model: str, appliance_type: str,
    ) -> dict[str, Any]:
        """Resolve just the catalog *identity* of one appliance: its brand and device
        documents, by deterministic id.

        This backs the brand/model status badges in the settings form, which previously
        forced a download of the entire brand list plus that brand's whole device list
        (measured: 128 documents, 119 KB) purely to locate two rows the caller could
        already name. Two point reads answer it exactly, and they are issued
        concurrently so the round-trip cost is one request deep, not two.
        """
        async def _brand() -> dict[str, Any] | None:
            return await self.get_brand(brand) if brand else None

        async def _device() -> dict[str, Any] | None:
            return await self.get_device(dev_id) if dev_id else None

        dev_id = device_id(appliance_type, brand, model) if (brand and model and appliance_type) else ""
        brand_doc, device_doc = await asyncio.gather(_brand(), _device())
        return {"device_id": dev_id, "brand": brand_doc, "device": device_doc}

    async def get_config(self) -> dict[str, Any]:
        """Public config/site (maintenance flag + confirmThreshold). {} on failure.

        Cached for a long TTL: this is read on every confirm_device to resolve the
        confirmThreshold, which changes almost never. Only successful reads are cached,
        so a transient failure never pins an empty config."""
        cached = self._cache_get("config:site")
        if cached is not None:
            return cached
        try:
            async with self._sess().get(f"{self._base}/config/site", timeout=15) as resp:
                if resp.status != 200:
                    return {}
                cfg = _decode_doc(await resp.json())
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store get_config error: %s", exc)
            return {}
        self._cache_put("config:site", cfg, self._CONFIG_CACHE_TTL_S)
        return cfg

    async def _rating_agg(self, parent_path: str) -> dict[str, Any]:
        """count + average over the `ratings` subcollection under ``parent_path``.

        Public (unauthenticated) aggregation -- ratings are world-readable. Returns
        ``{"avg": float|None, "count": int}`` and never raises.
        """
        body = {"structuredAggregationQuery": {
            "structuredQuery": {"from": [{"collectionId": "ratings"}]},
            "aggregations": [
                {"alias": "cnt", "count": {}},
                # NB: the Firestore aggregation operator is `avg`, NOT `average` --
                # the wrong name 400s the whole query and silently zeroes ratings.
                {"alias": "avg", "avg": {"field": {"fieldPath": "rating"}}},
            ],
        }}
        try:
            async with self._sess().post(
                f"{self._base}/{parent_path}:runAggregationQuery",
                json=body, timeout=15,
            ) as resp:
                if resp.status != 200:
                    return {"avg": None, "count": 0}
                rows = await resp.json()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store rating aggregation error (%s): %s", parent_path, exc)
            return {"avg": None, "count": 0}
        agg = next((r["result"]["aggregateFields"] for r in rows if isinstance(r, dict) and "result" in r), None)
        if not agg:
            return {"avg": None, "count": 0}
        cnt = _decode(agg["cnt"]) if "cnt" in agg else 0
        avg = _decode(agg["avg"]) if ("avg" in agg and "nullValue" not in agg["avg"]) else None
        return {"avg": avg if (cnt and avg is not None) else None, "count": cnt or 0}

    async def get_device_quality(self, device_id: str) -> dict[str, Any]:
        """count + average of the device's 5-star quality ratings (info only)."""
        return await self._rating_agg(f"devices/{_seg(device_id)}")

    async def cycle_rating(self, cycle_id: str) -> dict[str, Any]:
        """count + average of a reference cycle's 5-star ratings (info only)."""
        return await self._rating_agg(f"cycles/{_seg(cycle_id)}")

    async def get_profiles(self, dev_id: str, include_pending: bool = False, page_size: int = 100) -> list[dict[str, Any]]:
        sq = {
            "from": [{"collectionId": "profiles"}],
            "where": self._where([
                self._field_filter("deviceId", "EQUAL", dev_id),
                self._status_filter(include_pending),
            ]),
            "orderBy": [{"field": {"fieldPath": "createdAt"}, "direction": "DESCENDING"}],
            "limit": page_size,
        }
        return await self._run_query(sq) or []

    async def device_profiles(self, brand: str, model: str, appliance_type: str) -> dict[str, Any]:
        """Resolve the store deviceId from brand/model/type and return its profiles
        (approved + the caller's own pending), for the Share dialog's profile picker."""
        dev_id = device_id(appliance_type, brand, model)
        items = await self.get_profiles(dev_id, include_pending=True)
        return {"device_id": dev_id, "items": items}

    async def get_device_bundle(self, dev_id: str, include_pending: bool = True) -> dict[str, Any]:
        """Whole-device package for download: the device's shareable ``settings`` (from
        the device doc) + its profiles, each with its reference cycles nested under
        ``cycles`` (hydrated by get_cycles). One device GET + one profiles query + one
        cycles query per profile. Never raises.

        Star ratings are deliberately skipped here. They are browse-only decoration and
        the adopt path (``StoreBridge.download_device``) never reads them, but they cost
        one aggregation request *per cycle*: a 15-profile device turned a ~17-request
        download into ~60, all on the user's critical path.
        """
        device = await self.get_device(dev_id) or {}
        settings = device.get("settings") if isinstance(device.get("settings"), dict) else {}
        profiles = await self.get_profiles(dev_id, include_pending=include_pending)

        # Bound the per-profile fan-out: an unbounded gather over a device with many
        # profiles could burst hundreds of concurrent requests and trip the store's rate
        # limiter. A shared semaphore caps how many profiles hydrate at once.
        sem = asyncio.Semaphore(_BUNDLE_HYDRATE_LIMIT)

        async def _cycles_for(p: dict[str, Any]) -> list[dict[str, Any]]:
            pid = p.get("id")
            if not pid:
                return []
            async with sem:
                return await self.get_cycles(pid, include_pending=include_pending, include_ratings=False)

        # Fetch profiles' cycles concurrently (bounded) rather than one at a time.
        cycle_lists = await asyncio.gather(*(_cycles_for(p) for p in profiles))
        for p, cycles in zip(profiles, cycle_lists):
            p["cycles"] = cycles
        return {"device_id": dev_id, "settings": settings, "profiles": profiles}

    async def get_cycles(
        self, prof_id: str, include_pending: bool = True, page_size: int = 50,
        *, include_ratings: bool = True,
    ) -> list[dict[str, Any]]:
        """Reference cycles for a profile, most-recent-first.

        ``include_pending`` (default True) also returns still-awaiting-approval
        recordings so they can be browsed/imported before the community votes them
        in (they are publicly readable, shown with an "awaiting approval" tag).
        Each cycle gets a ``rating`` = ``{"avg", "count"}`` summary attached, unless
        ``include_ratings`` is False -- one extra aggregation request per cycle that
        only the browse UI displays (see get_device_bundle).
        """
        sq = {
            "from": [{"collectionId": "cycles"}],
            "where": self._where([
                self._field_filter("profileId", "EQUAL", prof_id),
                self._status_filter(include_pending),
            ]),
            "orderBy": [{"field": {"fieldPath": "createdAt"}, "direction": "DESCENDING"}],
            "limit": page_size,
        }
        cycles = [self._with_decoded_trace(c) for c in (await self._run_query(sq) or [])]
        if not include_ratings:
            return cycles
        # Attach each cycle's 5-star rating summary (info-only; the aggregation lives
        # in a subcollection so it can't ride the list query). Bound concurrency with
        # a semaphore so a large page can't fan out into dozens of simultaneous
        # aggregation requests.
        sem = asyncio.Semaphore(_RATING_FANOUT_LIMIT)
        async def _rate(cyc: dict[str, Any]) -> dict[str, Any]:
            cid = cyc.get("id")
            if not cid:
                return {"avg": None, "count": 0}
            async with sem:
                return await self.cycle_rating(cid)
        summaries = await asyncio.gather(*(_rate(c) for c in cycles), return_exceptions=True)
        for cyc, summary in zip(cycles, summaries):
            cyc["rating"] = summary if isinstance(summary, dict) else {"avg": None, "count": 0}
        return cycles

    async def get_cycle(self, cycle_id: str) -> dict[str, Any] | None:
        # Not cached: cycle documents carry the full trace, so they are the one read here
        # worth fetching fresh rather than pinning in memory.
        doc = await self._get_doc(f"cycles/{_seg(cycle_id)}")
        return None if doc is None else self._with_decoded_trace(doc)

    @staticmethod
    def _with_decoded_trace(cycle: dict[str, Any]) -> dict[str, Any]:
        """Attach ``importable`` = trace points when the cycleSchemaVersion is supported."""
        ver = cycle.get("cycleSchemaVersion", 1)
        trace = cycle.get("trace")
        if ver in SUPPORTED_CYCLE_SCHEMA_VERSIONS and isinstance(trace, dict) and isinstance(trace.get("points"), list):
            pairs = unpack_points(trace["points"])
            trace["points"] = pairs  # hydrate to [[offset, watts]] for the panel sparkline
            cycle["importable"] = pairs
        else:
            cycle["importable"] = None
        return cycle

    # ── write: upload a reference cycle (authed) ────────────────────────────────

    async def _commit_create(self, id_token: str, path: str, fields: dict[str, Any], server_ts_field: str = "createdAt") -> bool:
        """Create-if-missing. Returns True on create OR if it already exists; False on
        real failure. Thin wrapper over :meth:`_commit_create_ex` (drops the created flag).
        """
        ok, _created = await self._commit_create_ex(id_token, path, fields, server_ts_field)
        return ok

    async def _commit_create_ex(
        self, id_token: str, path: str, fields: dict[str, Any], server_ts_field: str = "createdAt"
    ) -> tuple[bool, bool]:
        """Create a document if it does not already exist, stamping ``server_ts_field``
        with the server request time (so the store rules' ``createdAt == request.time``
        holds). Returns ``(ok, created)``: ``created=False`` means the doc already
        existed (a benign no-op that supports idempotent re-upload); ``ok=False`` is a
        real failure.
        """
        write: dict[str, Any] = {
            "update": {
                "name": f"projects/{self._pid}/databases/(default)/documents/{path}",
                "fields": {k: _encode(v) for k, v in fields.items()},
            },
            "currentDocument": {"exists": False},
            "updateTransforms": [
                {"fieldPath": server_ts_field, "setToServerValue": "REQUEST_TIME"}
            ],
        }
        try:
            async with self._sess().post(
                f"{self._base}:commit",
                json={"writes": [write]},
                headers={"Authorization": f"Bearer {id_token}"},
                timeout=15,
            ) as resp:
                if resp.status == 200:
                    # A newly-created brand/device changes the catalog listing; drop the
                    # cached brand/device reads so it appears immediately for this user.
                    if path.split("/", 1)[0] in ("brands", "devices"):
                        self._invalidate_catalog_cache()
                    return (True, True)
                body = await resp.text()
                # Precondition failure => the doc already exists; that is fine (no-op).
                if resp.status == 409 or "ALREADY_EXISTS" in body or "FAILED_PRECONDITION" in body:
                    return (True, False)
                _LOGGER.warning("Store create %s failed: HTTP %s %s", path, resp.status, body[:300])
                coll = path.split("/", 1)[0]
                if resp.status == 403 or "PERMISSION_DENIED" in body:
                    self._last_error = f"{coll} rejected by the store rules (HTTP 403) - the community catalog rules may be out of date"
                else:
                    self._last_error = f"{coll} create failed (HTTP {resp.status})"
                return (False, False)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Store create %s error: %s", path, exc)
            self._last_error = f"{path.split('/', 1)[0]} create error: {exc}"
            return (False, False)

    async def upload_reference_cycle(
        self, refresh_token: str, uid: str, uploader_name: str | None, meta: dict[str, Any],
        points: list[list[float]], stats: dict[str, Any], qc: int, return_status: bool = False,
    ) -> str | None | dict[str, Any]:
        """Ensure brand/device/profile docs exist, then create the reference cycle.

        The cycle's document id is a deterministic content hash of its trace (scoped
        to the profile), so re-uploading an identical trace collides on the same id
        and the create is refused server-side -- share is idempotent. Returns the
        cycle id (on create OR already-exists), or None on real failure. With
        ``return_status=True`` returns ``{"id": str|None, "created": bool}`` where
        ``created=False`` means the trace was already in the store. All writes authed.
        """
        def _out(cid: str | None, created: bool) -> str | None | dict[str, Any]:
            return {"id": cid, "created": created} if return_status else cid

        self._last_error = None
        token = await self.ensure_id_token(refresh_token)
        if not token:
            return _out(None, False)

        # Preserve the documented never-raise contract: malformed metadata/points must
        # return a failure marker (with _last_error set), not propagate an exception to
        # the no-raise StoreBridge caller.
        if not isinstance(meta, dict):
            self._last_error = "invalid upload metadata"
            return _out(None, False)
        # Reject (don't coerce) missing/blank required metadata: str(None) -> "None"
        # would otherwise pollute the catalog with a literal "None" brand/model/etc.
        required: dict[str, str] = {}
        for _key in ("applianceType", "brand", "model", "program"):
            _val = meta.get(_key)
            if not isinstance(_val, str) or not _val.strip():
                self._last_error = f"invalid or missing upload metadata: {_key}"
                return _out(None, False)
            required[_key] = _val.strip()
        appliance = required["applianceType"]
        brand = required["brand"]
        model = required["model"]
        program = required["program"]
        try:
            interval = float(meta.get("sampleIntervalSec") or 0)
        except (TypeError, ValueError):
            interval = 0.0
        if appliance not in _APPLIANCE_TYPES:
            _LOGGER.warning("Store upload: invalid applianceType %r", appliance)
            self._last_error = f"unsupported appliance type {appliance!r} (only washer/dryer/dishwasher/washer_dryer)"
            return _out(None, False)

        b_id = brand_id(brand)
        d_id = device_id(appliance, brand, model)
        p_id = profile_id(d_id, program)
        qc_code = qc if qc in (1, 2, 3) else 3
        try:
            pts = [[float(p[0]), float(p[1])] for p in (points or [])[:10000] if len(p) >= 2]
        except (TypeError, ValueError):
            self._last_error = "malformed trace points"
            return _out(None, False)
        if len(pts) < 2:
            self._last_error = "empty or too-short trace"
            return _out(None, False)

        # 1-3: brand/device/profile (create-if-missing; rules deny updating existing).
        ok = await self._commit_create(token, f"brands/{b_id}", {
            "brand": brand, "brand_lc": b_id, "status": "pending", "createdByUid": uid,
        })
        device_fields: dict[str, Any] = {
            "applianceType": appliance, "brand": brand, "brand_lc": b_id,
            "model": model, "model_lc": model.lower(), "status": "pending",
            "createdByUid": uid, "createdByName": None, "manualUrl": None,
            "favoriteCount": 0, "confirmCount": 0,
        }
        # Stage 3: bundle the device's recognition/matching settings (allow-listed,
        # numeric only) onto the device doc when supplied. Create rule allows extra
        # fields, so no rules change; settings attach at create time (owner update is
        # Stage 5).
        settings = meta.get("settings")
        if isinstance(settings, dict) and settings:
            # Defense in depth at the store boundary: keep only allow-listed, numeric
            # settings (never trust the caller to have filtered) so nothing arbitrary is
            # ever written to the shared device doc.
            filtered = {
                str(k): v for k, v in settings.items()
                if k in SHAREABLE_SETTING_KEYS
                and isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if filtered:
                device_fields["settings"] = filtered
        ok = ok and await self._commit_create(token, f"devices/{d_id}", device_fields)
        profile_fields: dict[str, Any] = {
            "deviceId": d_id, "applianceType": appliance, "program": program,
            "program_lc": program.lower(), "description": meta.get("description", ""),
            "status": "pending", "createdByUid": uid,
        }
        # Stage 2: bundle the program's phase map onto the profile doc when the caller
        # supplies it. The profile create rule allows extra fields, so this needs no
        # rules change; phases attach at create time (updating an existing profile's
        # phases is an owner action -> Stage 5).
        phases = meta.get("phases")
        if isinstance(phases, list) and phases:
            def _valid_phase(p: Any) -> dict[str, Any] | None:
                # Drop a phase with non-numeric start/end rather than coercing it to
                # 0.0 (which would ship a bogus zero-length phase to the catalog).
                try:
                    return {"name": str(p.get("name", "")), "start": float(p["start"]), "end": float(p["end"])}
                except (KeyError, TypeError, ValueError):
                    return None
            valid_phases = [
                vp for vp in (_valid_phase(p) for p in phases if isinstance(p, dict)) if vp is not None
            ]
            if valid_phases:
                profile_fields["phases"] = valid_phases
                profile_fields["phaseSourceCycleId"] = str(meta.get("phaseSourceCycleId") or "")
                profile_fields["phasesSchemaVersion"] = 1
        ok = ok and await self._commit_create(token, f"profiles/{p_id}", profile_fields)
        if not ok:
            return _out(None, False)

        # 4: the reference cycle. Its id is a deterministic content hash of the trace
        # (scoped to the profile), so an identical re-upload collides on the same id
        # and the create precondition refuses it -> idempotent share (no duplicate).
        cyc_id = trace_hash(p_id, pts)
        cycle_fields = {
            "profileId": p_id, "deviceId": d_id, "brand_lc": b_id,
            "program_lc": program.lower(), "applianceType": appliance,
            "uploaderUid": uid, "uploaderName": uploader_name,
            "status": "pending", "rejectionReason": None,
            "traceHash": cyc_id,
            # Firestore rejects nested arrays -> store points as {o,w} maps.
            "trace": {"points": pack_points(pts), "sampleIntervalSec": interval},
            "stats": stats if isinstance(stats, dict) else {},
            "cycleSchemaVersion": 1, "downloads": 0, "commentCount": 0, "confirmCount": 0, "qc": qc_code,
        }
        cyc_ok, created = await self._commit_create_ex(token, f"cycles/{cyc_id}", cycle_fields)
        if not cyc_ok:
            return _out(None, False)
        # NB: cycle/profile counts are CALCULATED on the store (COUNT aggregation over
        # approved+pending), not maintained as a running total here -- a best-effort
        # increment that a rule denied is what left the browse counters stuck at 0.
        return _out(cyc_id, created)

    async def upload_device_bundle(
        self, refresh_token: str, uid: str, uploader_name: str | None,
        device_meta: dict[str, Any], items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Upload a whole-device bundle: one item per selected reference cycle.

        ``device_meta`` = ``{applianceType, brand, model}``; each ``item`` =
        ``{program, points, stats, qc, sampleIntervalSec}``. Reuses
        ``upload_reference_cycle`` per item, which idempotently upserts the
        brand/device/profile chain (existing ancestors are treated as success) and
        creates the cycle. Returns ``{ok, cycle_ids, created, duplicates, errors}``:
        ``created`` counts newly-uploaded cycles, ``duplicates`` counts ones whose
        identical trace was already in the store (both still land in ``cycle_ids``).
        Never raises.
        """
        cycle_ids: list[str] = []
        errors: list[str] = []
        created = 0
        duplicates = 0
        token = await self.ensure_id_token(refresh_token)
        if not token:
            return {"ok": False, "cycle_ids": [], "created": 0, "duplicates": 0,
                    "errors": [self._last_error or "not_connected"]}
        for it in items or []:
            meta = {
                "applianceType": device_meta.get("applianceType"),
                "brand": device_meta.get("brand"),
                "model": device_meta.get("model"),
                "program": it.get("program"),
                "sampleIntervalSec": it.get("sampleIntervalSec"),
                # Stage 2: optional phase map for the profile doc (create-time).
                "phases": it.get("phases"),
                "phaseSourceCycleId": it.get("phaseSourceCycleId"),
                # Stage 3: optional device-level settings (attach to the device doc).
                "settings": device_meta.get("settings"),
            }
            res = await self.upload_reference_cycle(
                refresh_token, uid, uploader_name, meta,
                it.get("points") or [], it.get("stats") or {}, int(it.get("qc") or 3),
                return_status=True,
            )
            cid = res.get("id") if isinstance(res, dict) else res
            if cid:
                cycle_ids.append(cid)
                if isinstance(res, dict) and res.get("created"):
                    created += 1
                else:
                    duplicates += 1
            else:
                errors.append(self._last_error or f"failed to upload {it.get('program')!r}")
        return {"ok": not errors, "cycle_ids": cycle_ids,
                "created": created, "duplicates": duplicates, "errors": errors}

    # ── community catalog: confirm + rate a device (authed) ──────────────────────

    async def _commit(self, id_token: str, writes: list[dict[str, Any]]) -> tuple[bool, str]:
        """Post a batched :commit. Returns (ok, response_body_text)."""
        try:
            async with self._sess().post(
                f"{self._base}:commit",
                json={"writes": writes},
                headers={"Authorization": f"Bearer {id_token}"},
                timeout=15,
            ) as resp:
                return (resp.status == 200, await resp.text())
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Store commit error: %s", exc)
            return (False, str(exc))

    def _doc_path(self, rel: str) -> str:
        return f"projects/{self._pid}/databases/(default)/documents/{rel}"

    async def confirm_device(self, refresh_token: str, uid: str, device_id: str) -> dict[str, Any] | None:
        """Confirm a device (one per user). Bumps the honest confirmCount in the same
        batch that creates confirmations/{uid}, then best-effort promotes to approved
        once the threshold is reached (the rule is the real guard). Returns state."""
        token = await self.ensure_id_token(refresh_token)
        if not token:
            return None
        dev_path = self._doc_path(f"devices/{device_id}")
        conf_path = self._doc_path(f"devices/{device_id}/confirmations/{uid}")
        writes = [
            {
                "update": {"name": conf_path, "fields": {"uid": _encode(uid)}},
                "currentDocument": {"exists": False},
                "updateTransforms": [{"fieldPath": "createdAt", "setToServerValue": "REQUEST_TIME"}],
            },
            {
                "transform": {
                    "document": dev_path,
                    "fieldTransforms": [{"fieldPath": "confirmCount", "increment": _encode(1)}],
                },
            },
        ]
        ok, body = await self._commit(token, writes)
        # A precondition failure means this user already confirmed - not an error.
        if not ok and "ALREADY_EXISTS" not in body and "FAILED_PRECONDITION" not in body:
            _LOGGER.warning("Store confirm_device failed: %s", body[:200])
            return None
        # This write just changed the very document the next line reads, and get_device
        # is a CACHED point read that the settings badge has almost certainly already
        # warmed. Without dropping it here the read-back returns the pre-increment doc,
        # so the count shown is one behind and `count >= threshold` never becomes true --
        # community auto-promotion would silently never fire.
        self._invalidate_catalog_cache()
        dev = await self.get_device(device_id) or {}
        count = int(dev.get("confirmCount") or 0)
        status = dev.get("status")
        try:
            threshold = int((await self.get_config()).get("confirmThreshold") or 5)
        except (TypeError, ValueError):
            threshold = 5
        if status == "pending" and count >= threshold:
            promote = [{
                "update": {"name": dev_path, "fields": {"status": _encode("approved")}},
                "updateMask": {"fieldPaths": ["status"]},
                "currentDocument": {"exists": True},
            }]
            if (await self._commit(token, promote))[0]:
                status = "approved"
                # Promotion changes catalog visibility (pending -> approved); drop cached
                # listings so the newly-approved device shows in approved-only searches now.
                self._invalidate_catalog_cache()
        return {"confirmed": True, "confirmCount": count, "status": status}

    async def rate_device(self, refresh_token: str, uid: str, device_id: str, rating: int) -> bool:
        """Set this user's 5-star quality rating for a device (info only)."""
        if rating not in (1, 2, 3, 4, 5):
            return False
        token = await self.ensure_id_token(refresh_token)
        if not token:
            return False
        path = self._doc_path(f"devices/{device_id}/ratings/{uid}")
        writes = [{
            "update": {"name": path, "fields": {"uid": _encode(uid), "rating": _encode(rating)}},
            "updateTransforms": [{"fieldPath": "updatedAt", "setToServerValue": "REQUEST_TIME"}],
        }]
        ok, body = await self._commit(token, writes)
        if not ok:
            _LOGGER.warning("Store rate_device failed: %s", body[:200])
        return ok

    async def bump_downloads(self, cycle_ids: list[str]) -> None:
        """Best-effort +1 to each cycle's public ``downloads`` counter. The store rule
        allows an anonymous ``downloads++`` (unauthenticated), so this needs no token --
        it mirrors the website's per-download bump. Chunked to stay under Firestore's
        500-writes-per-commit limit. Never raises."""
        # Deduplicate (order-preserving): Firestore :commit rejects a batch (HTTP 400)
        # if it contains two writes to the same document, which a bundle referencing
        # the same shared cycle across profiles can produce.
        ids = list(dict.fromkeys(c for c in (cycle_ids or []) if c))
        for start in range(0, len(ids), 400):
            writes = [
                {"transform": {
                    "document": self._doc_path(f"cycles/{cid}"),
                    "fieldTransforms": [{"fieldPath": "downloads", "increment": _encode(1)}],
                }}
                for cid in ids[start:start + 400]
            ]
            try:
                async with self._sess().post(
                    f"{self._base}:commit", json={"writes": writes}, timeout=15,
                ) as resp:
                    if resp.status != 200:
                        _LOGGER.debug("Store bump_downloads HTTP %s: %s", resp.status, (await resp.text())[:200])
            except Exception as exc:  # noqa: BLE001
                _LOGGER.debug("Store bump_downloads error: %s", exc)

    async def bump_analytics(self, field: str, n: int = 1) -> None:
        """Best-effort +n to an aggregate usage counter the store's admin dashboard reads
        (``analytics/totals`` + ``analytics/daily_YYYYMMDD``). This is how integration
        downloads become a real, community-wide usage metric -- the website has no
        download action, so the integration is the source of truth for "someone adopted
        this". Unauthenticated (the analytics rule allows anonymous counter writes) and
        never raises. The daily doc id is UTC to line up with the website's own writes.
        """
        if not field or n <= 0:
            return
        day = dt_util.utcnow().strftime("%Y%m%d")  # UTC to match the website's daily docs
        writes = [
            {"transform": {
                "document": self._doc_path(f"analytics/daily_{day}"),
                "fieldTransforms": [{"fieldPath": field, "increment": _encode(n)}],
            }},
            {"transform": {
                "document": self._doc_path("analytics/totals"),
                "fieldTransforms": [{"fieldPath": field, "increment": _encode(n)}],
            }},
        ]
        try:
            async with self._sess().post(
                f"{self._base}:commit", json={"writes": writes}, timeout=15,
            ) as resp:
                if resp.status != 200:
                    _LOGGER.debug("Store bump_analytics HTTP %s: %s", resp.status, (await resp.text())[:200])
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store bump_analytics error: %s", exc)
