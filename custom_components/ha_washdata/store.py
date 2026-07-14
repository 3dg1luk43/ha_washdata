"""Community-store bridge: gating + provenance + import/share/catalog orchestration.

Pure/near-pure glue between ``store_client`` (network) and ``profile_store`` (local),
plus the integration-wide account/online flag in ``store_account``. The GitHub
connection and the online-features switch are device-agnostic (one per HA install);
brand/model stay per-device. Nothing here runs unless online features are enabled.
"""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant

from . import store_account
from .const import QC_EDITED, QC_MANUAL, QC_RECORDING
from .store_client import StoreClient

_LOGGER = logging.getLogger(__name__)


def online_features_enabled(hass: HomeAssistant) -> bool:
    """True when online store features are enabled integration-wide (default off)."""
    return store_account.online_enabled(hass)


# The community catalog only knows washer/dryer/dishwasher/washer_dryer; HA's
# washing_machine device type maps to washer. Keep this in sync with the panel's
# _storeApplianceType() so search, create and share all resolve the same deviceId.
_STORE_APPLIANCE_TYPE = {"washing_machine": "washer"}


def store_appliance_type(device_type: str) -> str:
    return _STORE_APPLIANCE_TYPE.get(device_type, device_type)


def derive_qc(cycle: dict[str, Any]) -> int:
    """Derive the obfuscated provenance code for a cycle being uploaded.

    QC_RECORDING - a pure recorder capture.
    QC_EDITED    - trimmed/edited from a detected cycle.
    QC_MANUAL    - a plain detected cycle the user flagged golden by hand.
    Never raises.
    """
    meta = cycle.get("meta") if isinstance(cycle.get("meta"), dict) else {}
    if meta.get("source") == "recorder" or "original_samples" in meta:
        return QC_RECORDING
    if meta.get("edited"):
        return QC_EDITED
    return QC_MANUAL


def _downsample(points: list[list[float]], max_n: int = 3000) -> list[list[float]]:
    if len(points) <= max_n:
        return [[float(p[0]), float(p[1])] for p in points]
    step = len(points) / max_n
    return [[float(points[int(i * step)][0]), float(points[int(i * step)][1])] for i in range(max_n)]


class StoreBridge:
    """Orchestrates store browse/import/share/catalog against a ProfileStore.

    All methods no-op-safe: they return an ``{"error": ...}`` marker rather than raising.
    Callers must gate on ``online_features_enabled`` first. The account/online flag are
    global (via ``store_account``); import/share target this bridge's ProfileStore.
    """

    def __init__(self, hass: Any, profile_store: Any) -> None:
        self._hass = hass
        self._ps = profile_store
        self._client = StoreClient(hass)

    # ── account / status (global) ───────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {"enabled": store_account.online_enabled(self._hass), **store_account.get_identity(self._hass)}

    async def connect(self, refresh_token: str, uid: str, name: str | None) -> dict[str, Any]:
        # Validate the refresh token by exchanging it once before persisting.
        if not await self._client.ensure_id_token(refresh_token):
            return {"error": "token_invalid"}
        await store_account.async_set_account(self._hass, {"refresh_token": refresh_token, "uid": uid, "name": name})
        return store_account.get_identity(self._hass)

    async def disconnect(self) -> dict[str, Any]:
        await store_account.async_clear_account(self._hass)
        return {"connected": False}

    # ── catalog browse (reads) ───────────────────────────────────────────────────

    async def list_brands(self, query: str | None = None, include_pending: bool = True) -> list[dict[str, Any]]:
        return await self._client.list_brands(query, include_pending=include_pending)

    async def search_devices(
        self, brand: str | None, appliance_type: str | None,
        model_query: str | None = None, include_pending: bool = False,
    ) -> list[dict[str, Any]]:
        return await self._client.search_devices(
            brand, appliance_type, model_query=model_query, include_pending=include_pending,
        )

    async def get_profiles(self, device_id: str) -> list[dict[str, Any]]:
        return await self._client.get_profiles(device_id)

    async def device_profiles(self, brand: str, model: str, appliance_type: str) -> dict[str, Any]:
        """Profiles for the appliance identified by brand/model/type (for the Share
        dialog's profile picker). Maps the HA device type to the catalog type first."""
        return await self._client.device_profiles(brand, model, store_appliance_type(appliance_type))

    async def get_cycles(self, profile_id: str) -> list[dict[str, Any]]:
        return await self._client.get_cycles(profile_id)

    async def get_device_quality(self, device_id: str) -> dict[str, Any]:
        return await self._client.get_device_quality(device_id)

    # ── community actions (authed writes) ────────────────────────────────────────

    async def confirm_device(self, device_id: str) -> dict[str, Any]:
        acct = store_account.get_account(self._hass)
        if not acct.get("refresh_token"):
            return {"error": "not_connected"}
        res = await self._client.confirm_device(acct["refresh_token"], acct.get("uid", ""), device_id)
        return res if res else {"error": "confirm_failed"}

    async def rate_device(self, device_id: str, rating: int) -> dict[str, Any]:
        acct = store_account.get_account(self._hass)
        if not acct.get("refresh_token"):
            return {"error": "not_connected"}
        ok = await self._client.rate_device(acct["refresh_token"], acct.get("uid", ""), device_id, rating)
        return {"ok": True} if ok else {"error": "rate_failed"}

    # ── import / share (target this device's ProfileStore) ───────────────────────

    async def import_cycle(
        self, cycle_id: str, target_profile: str | None = None, new_profile_name: str | None = None
    ) -> dict[str, Any]:
        cyc = await self._client.get_cycle(cycle_id)
        if not cyc:
            return {"error": "not_found"}
        pts = cyc.get("importable")
        if not pts:
            return {"error": "unsupported_schema"}
        profile = (new_profile_name or target_profile or cyc.get("program_lc") or "Imported").strip()
        local_id = await self._ps.add_reference_cycle(profile, pts, {
            "store_cycle_id": cyc.get("id"),
            "store_uploaded_at": cyc.get("createdAt"),
            "sampling_interval": (cyc.get("trace") or {}).get("sampleIntervalSec"),
        })
        return {"profile": profile, "cycle_id": local_id}

    async def share_cycle(
        self, local_cycle_id: str, program: str, brand: str, model: str, appliance_type: str,
        sample_interval_sec: float = 0.0, description: str = "",
    ) -> dict[str, Any]:
        acct = store_account.get_account(self._hass)
        if not acct.get("refresh_token"):
            return {"error": "not_connected"}
        pts = self._ps.get_cycle_power_data(local_cycle_id)
        if not pts:
            return {"error": "cycle_not_found"}
        cyc = next((c for c in self._ps.get_past_cycles() if c.get("id") == local_cycle_id), {})
        vals = [float(p[1]) for p in pts]
        stats = {
            "duration": float(cyc.get("duration") or (pts[-1][0] - pts[0][0])),
            "energy_wh": float(cyc.get("energy_wh") or 0.0),
            "peak_w": max(vals) if vals else 0.0,
            "mean_w": (sum(vals) / len(vals)) if vals else 0.0,
            "signature": cyc.get("signature") if isinstance(cyc.get("signature"), dict) else {},
        }
        meta = {
            "applianceType": store_appliance_type(appliance_type), "brand": brand, "model": model, "program": program,
            "sampleIntervalSec": float(sample_interval_sec or cyc.get("sampling_interval") or 0.0),
            "description": description,
        }
        new_id = await self._client.upload_reference_cycle(
            acct["refresh_token"], acct.get("uid", ""), acct.get("name"),
            meta, _downsample([[p[0], p[1]] for p in pts]), stats, derive_qc(cyc),
        )
        if not new_id:
            return {"error": "upload_failed", "detail": self._client.last_error()}
        return {"store_cycle_id": new_id}
