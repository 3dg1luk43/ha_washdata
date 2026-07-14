"""Async Firestore-REST client for the WashData community store (v2 hierarchy).

Reads (approved brands/devices/profiles/cycles) are public and need no token. Writes
(upload a reference cycle) use the signed-in user's Firebase ID token, obtained by
exchanging the refresh token handed over by the store's connect page.

No Firebase SDK, no new dependency: plain aiohttp via Home Assistant's shared session.
Never raises into the event loop - failures return ``None``/empty and are logged.
"""
from __future__ import annotations

import logging
import re
import secrets
import time
import unicodedata
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import STORE_API_KEY, STORE_PROJECT_ID, SUPPORTED_CYCLE_SCHEMA_VERSIONS

_LOGGER = logging.getLogger(__name__)

_APPLIANCE_TYPES = {"washer", "dryer", "dishwasher", "washer_dryer"}


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
        if self._id_token and now < self._id_token_exp - 60:
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
                    _LOGGER.debug("Store query HTTP %s", resp.status)
                    return []
                rows = await resp.json()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store query error: %s", exc)
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

    async def get_device_quality(self, device_id: str) -> dict[str, Any]:
        """count + average of the device's 5-star quality ratings (info only)."""
        body = {"structuredAggregationQuery": {
            "structuredQuery": {"from": [{"collectionId": "ratings"}]},
            "aggregations": [
                {"alias": "cnt", "count": {}},
                {"alias": "avg", "average": {"field": {"fieldPath": "rating"}}},
            ],
        }}
        try:
            async with self._sess().post(
                f"{self._base}/devices/{device_id}:runAggregationQuery",
                json=body, timeout=15,
            ) as resp:
                if resp.status != 200:
                    return {"avg": None, "count": 0}
                rows = await resp.json()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Store get_device_quality error: %s", exc)
            return {"avg": None, "count": 0}
        agg = next((r["result"]["aggregateFields"] for r in rows if isinstance(r, dict) and "result" in r), None)
        if not agg:
            return {"avg": None, "count": 0}
        cnt = _decode(agg["cnt"]) if "cnt" in agg else 0
        avg = _decode(agg["avg"]) if ("avg" in agg and "nullValue" not in agg["avg"]) else None
        return {"avg": avg if (cnt and avg is not None) else None, "count": cnt or 0}

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

    async def get_cycles(self, prof_id: str, page_size: int = 50) -> list[dict[str, Any]]:
        sq = {
            "from": [{"collectionId": "cycles"}],
            "where": self._where([
                self._field_filter("profileId", "EQUAL", prof_id),
                self._field_filter("status", "EQUAL", "approved"),
            ]),
            "orderBy": [{"field": {"fieldPath": "createdAt"}, "direction": "DESCENDING"}],
            "limit": page_size,
        }
        return [self._with_decoded_trace(c) for c in await self._run_query(sq)]

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
        """Create a document if it does not already exist, stamping ``server_ts_field``
        with the server request time (so the store rules' ``createdAt == request.time``
        holds). Returns True on create or if it already exists; False on real failure.
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
                    return True
                body = await resp.text()
                # Precondition failure => the doc already exists; that is fine.
                if resp.status == 409 or "ALREADY_EXISTS" in body or "FAILED_PRECONDITION" in body:
                    return True
                _LOGGER.warning("Store create %s failed: HTTP %s %s", path, resp.status, body[:300])
                coll = path.split("/", 1)[0]
                if resp.status == 403 or "PERMISSION_DENIED" in body:
                    self._last_error = f"{coll} rejected by the store rules (HTTP 403) - the community catalog rules may be out of date"
                else:
                    self._last_error = f"{coll} create failed (HTTP {resp.status})"
                return False
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Store create %s error: %s", path, exc)
            self._last_error = f"{path.split('/', 1)[0]} create error: {exc}"
            return False

    async def upload_reference_cycle(
        self, refresh_token: str, uid: str, uploader_name: str | None, meta: dict[str, Any],
        points: list[list[float]], stats: dict[str, Any], qc: int,
    ) -> str | None:
        """Ensure brand/device/profile docs exist, then create the reference cycle.
        Returns the new cycle id, or None on failure. All writes are authed.
        """
        self._last_error = None
        token = await self.ensure_id_token(refresh_token)
        if not token:
            return None

        appliance = meta["applianceType"]
        brand = meta["brand"]
        model = meta["model"]
        program = meta["program"]
        interval = float(meta.get("sampleIntervalSec") or 0)
        if appliance not in _APPLIANCE_TYPES:
            _LOGGER.warning("Store upload: invalid applianceType %r", appliance)
            self._last_error = f"unsupported appliance type {appliance!r} (only washer/dryer/dishwasher/washer_dryer)"
            return None

        b_id = brand_id(brand)
        d_id = device_id(appliance, brand, model)
        p_id = profile_id(d_id, program)
        qc_code = qc if qc in (1, 2, 3) else 3
        pts = [[float(p[0]), float(p[1])] for p in points[:3000] if len(p) >= 2]

        # 1-3: brand/device/profile (create-if-missing; rules deny updating existing).
        ok = await self._commit_create(token, f"brands/{b_id}", {
            "brand": brand, "brand_lc": b_id, "status": "pending", "createdByUid": uid,
        })
        ok = ok and await self._commit_create(token, f"devices/{d_id}", {
            "applianceType": appliance, "brand": brand, "brand_lc": b_id,
            "model": model, "model_lc": model.lower(), "status": "pending",
            "createdByUid": uid, "createdByName": None, "manualUrl": None,
            "profileCount": 0, "favoriteCount": 0, "confirmCount": 0,
        })
        ok = ok and await self._commit_create(token, f"profiles/{p_id}", {
            "deviceId": d_id, "applianceType": appliance, "program": program,
            "program_lc": program.lower(), "description": meta.get("description", ""),
            "status": "pending", "createdByUid": uid, "cycleCount": 0,
        })
        if not ok:
            return None

        # 4: the reference cycle (client-generated id so it is a create).
        cyc_id = secrets.token_hex(10)
        cycle_fields = {
            "profileId": p_id, "deviceId": d_id, "brand_lc": b_id,
            "program_lc": program.lower(), "applianceType": appliance,
            "uploaderUid": uid, "uploaderName": uploader_name,
            "status": "pending", "rejectionReason": None,
            # Firestore rejects nested arrays -> store points as {o,w} maps.
            "trace": {"points": pack_points(pts), "sampleIntervalSec": interval},
            "stats": stats if isinstance(stats, dict) else {},
            "cycleSchemaVersion": 1, "downloads": 0, "commentCount": 0, "confirmCount": 0, "qc": qc_code,
        }
        if not await self._commit_create(token, f"cycles/{cyc_id}", cycle_fields):
            return None
        # Bump the profile's cycleCount for the browse count (best-effort).
        try:
            await self._commit(token, [{
                "transform": {
                    "document": self._doc_path(f"profiles/{p_id}"),
                    "fieldTransforms": [{"fieldPath": "cycleCount", "increment": _encode(1)}],
                },
            }])
        except Exception:  # noqa: BLE001 - counter is best-effort
            pass
        return cyc_id

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
