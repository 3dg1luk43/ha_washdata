"""Community-store bridge: gating + provenance + import/share orchestration.

Pure/near-pure glue between ``store_client`` (network) and ``profile_store`` (local).
Nothing here runs unless online features are enabled.
"""
from __future__ import annotations

import logging
from typing import Any

from .const import (
    CONF_ENABLE_ONLINE_FEATURES,
    DEFAULT_ENABLE_ONLINE_FEATURES,
    QC_EDITED,
    QC_MANUAL,
    QC_RECORDING,
)
from .store_client import StoreClient

_LOGGER = logging.getLogger(__name__)


def online_features_enabled(options: dict[str, Any] | None) -> bool:
    """True when the user has opted into online store features (default off)."""
    if not options:
        return DEFAULT_ENABLE_ONLINE_FEATURES
    return bool(options.get(CONF_ENABLE_ONLINE_FEATURES, DEFAULT_ENABLE_ONLINE_FEATURES))


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
    """Orchestrates store browse/import/share against a ProfileStore.

    All methods no-op-safe: they return an ``{"error": ...}`` marker rather than raising.
    Callers must gate on ``online_features_enabled`` first.
    """

    def __init__(self, hass: Any, profile_store: Any) -> None:
        self._hass = hass
        self._ps = profile_store
        self._client = StoreClient(hass)

    def status(self, options: dict[str, Any] | None) -> dict[str, Any]:
        return {"enabled": online_features_enabled(options), **self._ps.get_store_identity()}

    async def connect(self, refresh_token: str, uid: str, name: str | None) -> dict[str, Any]:
        # Validate the refresh token by exchanging it once before persisting.
        if not await self._client.ensure_id_token(refresh_token):
            return {"error": "token_invalid"}
        await self._ps.set_store_account({"refresh_token": refresh_token, "uid": uid, "name": name})
        return self._ps.get_store_identity()

    async def disconnect(self) -> dict[str, Any]:
        await self._ps.clear_store_account()
        return {"connected": False}

    async def search_devices(self, brand: str | None, appliance_type: str | None) -> list[dict[str, Any]]:
        return await self._client.search_devices(brand, appliance_type)

    async def get_profiles(self, device_id: str) -> list[dict[str, Any]]:
        return await self._client.get_profiles(device_id)

    async def get_cycles(self, profile_id: str) -> list[dict[str, Any]]:
        return await self._client.get_cycles(profile_id)

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
        acct = self._ps.get_store_account()
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
            "applianceType": appliance_type, "brand": brand, "model": model, "program": program,
            "sampleIntervalSec": float(sample_interval_sec or cyc.get("sampling_interval") or 0.0),
            "description": description,
        }
        new_id = await self._client.upload_reference_cycle(
            acct["refresh_token"], acct.get("uid", ""), acct.get("name"),
            meta, _downsample([[p[0], p[1]] for p in pts]), stats, derive_qc(cyc),
        )
        if not new_id:
            return {"error": "upload_failed"}
        return {"store_cycle_id": new_id}
