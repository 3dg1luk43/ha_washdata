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
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    SHAREABLE_SETTING_KEYS,
    STORE_API_KEY,
    STORE_PROJECT_ID,
    SUPPORTED_CYCLE_SCHEMA_VERSIONS,
)

_LOGGER = logging.getLogger(__name__)

_APPLIANCE_TYPES = {"washer", "dryer", "dishwasher", "washer_dryer"}

# Max concurrent per-cycle rating aggregations when listing a profile's cycles.
_RATING_FANOUT_LIMIT = 8


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

    def last_error(self) -> str | None:
        return self._last_error

    def _sess(self) -> Any:
        if self._session is None:
            self._session = async_get_clientsession(self._hass)
        return self._session

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

    async def _run_query(self, sq: dict[str, Any], parent: str = "") -> list[dict[str, Any]]:
        url = f"{self._base}/{parent}:runQuery" if parent else f"{self._base}:runQuery"
        try:
            async with self._sess().post(url, json={"structuredQuery": sq}, timeout=15) as resp:
                if resp.status != 200:
                    try:
                        body = await resp.json()
                        _LOGGER.warning("Store query HTTP %s: %s", resp.status, body)
                    except Exception:
                        _LOGGER.warning("Store query HTTP %s (no body)", resp.status)
                    return []
                rows = await resp.json()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Store query error: %s", exc)
            return []
        return [_decode_doc(r["document"]) for r in rows if isinstance(r, dict) and "document" in r]

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

    async def search_devices(
        self, brand: str | None = None, appliance_type: str | None = None,
        model_query: str | None = None, include_pending: bool = False, page_size: int = 60,
    ) -> list[dict[str, Any]]:
        filters = [self._status_filter(include_pending)]
        if appliance_type:
            filters.append(self._field_filter("applianceType", "EQUAL", appliance_type))
        if brand:
            filters.append(self._field_filter("brand_lc", "EQUAL", brand.lower()))
        sq = {
            "from": [{"collectionId": "devices"}],
            "where": self._where(filters),
            "orderBy": [{"field": {"fieldPath": "favoriteCount"}, "direction": "DESCENDING"}],
            "limit": page_size,
        }
        rows = await self._run_query(sq)
        if model_query:
            p = model_query.lower()
            rows = [r for r in rows if str(r.get("model_lc", "")).startswith(p)]
        return rows

    async def list_brands(self, q: str | None = None, include_pending: bool = True, page_size: int = 60) -> list[dict[str, Any]]:
        sq = {
            "from": [{"collectionId": "brands"}],
            "where": self._where([self._status_filter(include_pending)]),
            "orderBy": [{"field": {"fieldPath": "brand_lc"}, "direction": "ASCENDING"}],
            "limit": page_size,
        }
        rows = await self._run_query(sq)
        if q:
            p = q.lower()
            rows = [r for r in rows if str(r.get("brand_lc", "")).startswith(p)]
        return rows

    async def get_device(self, device_id: str) -> dict[str, Any] | None:
        try:
            async with self._sess().get(f"{self._base}/devices/{device_id}", timeout=15) as resp:
                if resp.status in (403, 404):
                    return None
                if resp.status != 200:
                    return None
                doc = await resp.json()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store get_device error: %s", exc)
            return None
        return _decode_doc(doc)

    async def get_config(self) -> dict[str, Any]:
        """Public config/site (maintenance flag + confirmThreshold). {} on failure."""
        try:
            async with self._sess().get(f"{self._base}/config/site", timeout=15) as resp:
                if resp.status != 200:
                    return {}
                return _decode_doc(await resp.json())
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store get_config error: %s", exc)
            return {}

    async def _rating_agg(self, parent_path: str) -> dict[str, Any]:
        """count + average over the `ratings` subcollection under ``parent_path``.

        Public (unauthenticated) aggregation -- ratings are world-readable. Returns
        ``{"avg": float|None, "count": int}`` and never raises.
        """
        body = {"structuredAggregationQuery": {
            "structuredQuery": {"from": [{"collectionId": "ratings"}]},
            "aggregations": [
                {"alias": "cnt", "count": {}},
                {"alias": "avg", "average": {"field": {"fieldPath": "rating"}}},
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
        return await self._rating_agg(f"devices/{device_id}")

    async def cycle_rating(self, cycle_id: str) -> dict[str, Any]:
        """count + average of a reference cycle's 5-star ratings (info only)."""
        return await self._rating_agg(f"cycles/{cycle_id}")

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
        return await self._run_query(sq)

    async def device_profiles(self, brand: str, model: str, appliance_type: str) -> dict[str, Any]:
        """Resolve the store deviceId from brand/model/type and return its profiles
        (approved + the caller's own pending), for the Share dialog's profile picker."""
        dev_id = device_id(appliance_type, brand, model)
        items = await self.get_profiles(dev_id, include_pending=True)
        return {"device_id": dev_id, "items": items}

    async def get_device_bundle(self, dev_id: str, include_pending: bool = True) -> dict[str, Any]:
        """Whole-device package for download: the device's shareable ``settings`` (from
        the device doc) + its profiles, each with its reference cycles nested under
        ``cycles`` (hydrated + rating-summarised by get_cycles). One device GET + one
        profiles query + one cycles query per profile. Never raises.
        """
        device = await self.get_device(dev_id) or {}
        settings = device.get("settings") if isinstance(device.get("settings"), dict) else {}
        profiles = await self.get_profiles(dev_id, include_pending=include_pending)

        async def _cycles_for(p: dict[str, Any]) -> list[dict[str, Any]]:
            pid = p.get("id")
            return await self.get_cycles(pid, include_pending=include_pending) if pid else []

        # Fetch every profile's cycles concurrently rather than one query at a time.
        cycle_lists = await asyncio.gather(*(_cycles_for(p) for p in profiles))
        for p, cycles in zip(profiles, cycle_lists):
            p["cycles"] = cycles
        return {"device_id": dev_id, "settings": settings, "profiles": profiles}

    async def get_cycles(
        self, prof_id: str, include_pending: bool = True, page_size: int = 50
    ) -> list[dict[str, Any]]:
        """Reference cycles for a profile, most-recent-first.

        ``include_pending`` (default True) also returns still-awaiting-approval
        recordings so they can be browsed/imported before the community votes them
        in (they are publicly readable, shown with an "awaiting approval" tag).
        Each cycle gets a ``rating`` = ``{"avg", "count"}`` summary attached.
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
        cycles = [self._with_decoded_trace(c) for c in await self._run_query(sq)]
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
        try:
            async with self._sess().get(f"{self._base}/cycles/{cycle_id}", timeout=15) as resp:
                if resp.status in (403, 404):
                    return None
                if resp.status != 200:
                    _LOGGER.debug("Store get_cycle HTTP %s", resp.status)
                    return None
                doc = await resp.json()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store get_cycle error: %s", exc)
            return None
        return self._with_decoded_trace(_decode_doc(doc))

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

        appliance = meta["applianceType"]
        brand = meta["brand"]
        model = meta["model"]
        program = meta["program"]
        interval = float(meta.get("sampleIntervalSec") or 0)
        if appliance not in _APPLIANCE_TYPES:
            _LOGGER.warning("Store upload: invalid applianceType %r", appliance)
            self._last_error = f"unsupported appliance type {appliance!r} (only washer/dryer/dishwasher/washer_dryer)"
            return _out(None, False)

        b_id = brand_id(brand)
        d_id = device_id(appliance, brand, model)
        p_id = profile_id(d_id, program)
        qc_code = qc if qc in (1, 2, 3) else 3
        pts = [[float(p[0]), float(p[1])] for p in points[:10000] if len(p) >= 2]

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
            profile_fields["phases"] = [
                {"name": str(p.get("name", "")), "start": float(p.get("start", 0)), "end": float(p.get("end", 0))}
                for p in phases if isinstance(p, dict)
            ]
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
