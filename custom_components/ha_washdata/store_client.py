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
        self._base = f"{self._FS}/projects/{project_id}/databases/(default)/documents"

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
                    return None
                body = await resp.json()
        except Exception as exc:  # noqa: BLE001 - never raise into the loop
            _LOGGER.warning("Store token exchange error: %s", exc)
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

    async def search_devices(self, brand: str | None = None, appliance_type: str | None = None, page_size: int = 60) -> list[dict[str, Any]]:
        filters = [self._field_filter("status", "EQUAL", "approved")]
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
        return await self._run_query(sq)

    async def get_profiles(self, dev_id: str, page_size: int = 100) -> list[dict[str, Any]]:
        sq = {
            "from": [{"collectionId": "profiles"}],
            "where": self._where([
                self._field_filter("deviceId", "EQUAL", dev_id),
                self._field_filter("status", "EQUAL", "approved"),
            ]),
            "orderBy": [{"field": {"fieldPath": "createdAt"}, "direction": "DESCENDING"}],
            "limit": page_size,
        }
        return await self._run_query(sq)

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
            cycle["importable"] = trace["points"]
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
                _LOGGER.warning("Store create %s failed: HTTP %s %s", path, resp.status, body[:200])
                return False
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Store create %s error: %s", path, exc)
            return False

    async def upload_reference_cycle(
        self, refresh_token: str, uid: str, uploader_name: str | None, meta: dict[str, Any],
        points: list[list[float]], stats: dict[str, Any], qc: int,
    ) -> str | None:
        """Ensure brand/device/profile docs exist, then create the reference cycle.
        Returns the new cycle id, or None on failure. All writes are authed.
        """
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
            "createdByUid": uid, "profileCount": 0, "favoriteCount": 0,
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
            "trace": {"points": pts, "sampleIntervalSec": interval},
            "stats": stats if isinstance(stats, dict) else {},
            "cycleSchemaVersion": 1, "downloads": 0, "commentCount": 0, "qc": qc_code,
        }
        if not await self._commit_create(token, f"cycles/{cyc_id}", cycle_fields):
            return None
        return cyc_id
