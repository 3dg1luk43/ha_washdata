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
    # Evenly spaced indices INCLUDING the last sample, so the uploaded trace keeps
    # its terminal transition (the old int(i*step) never reached points[-1]).
    n = len(points)
    return [
        [float(points[j][0]), float(points[j][1])]
        for j in (round(i * (n - 1) / (max_n - 1)) for i in range(max_n))
    ]


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
        # The name comes from the caller (localized) or the store's program label;
        # never fall back to an inline English string. Require a non-empty name.
        raw_profile = new_profile_name or target_profile or cyc.get("program_lc")
        profile = raw_profile.strip() if isinstance(raw_profile, str) else ""
        if not profile:
            return {"error": "profile_name_required"}
        local_id = await self._ps.add_reference_cycle(profile, pts, {
            "store_cycle_id": cyc.get("id"),
            "store_uploaded_at": cyc.get("createdAt"),
            "sampling_interval": (cyc.get("trace") or {}).get("sampleIntervalSec"),
        })
        if not local_id:  # trace failed validation in add_reference_cycle
            return {"error": "invalid_trace"}
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

    async def share_device(
        self, brand: str, model: str, appliance_type: str, items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Share a device bundle. ``items`` = ``[{local_cycle_id, program}]`` (the
        panel's tree selection). Resolves each local cycle's trace + stats and uploads
        the whole set via ``upload_device_bundle``. Returns ``{ok, cycle_ids, errors}``
        or ``{error}``.
        """
        acct = store_account.get_account(self._hass)
        if not acct.get("refresh_token"):
            return {"error": "not_connected"}
        by_id = {
            c.get("id"): c
            for c in (list(self._ps.get_past_cycles()) + list(self._ps.get_reference_cycles()))
        }
        bundle_items: list[dict[str, Any]] = []
        for it in items or []:
            cid = it.get("local_cycle_id")
            program = str(it.get("program") or "").strip()
            if not cid or not program:
                continue
            pts = self._ps.get_cycle_power_data(cid)
            if not pts:
                continue
            cyc = by_id.get(cid, {})
            vals = [float(p[1]) for p in pts]
            bundle_items.append({
                "program": program,
                "points": _downsample([[p[0], p[1]] for p in pts]),
                "stats": {
                    "duration": float(cyc.get("duration") or (pts[-1][0] - pts[0][0])),
                    "energy_wh": float(cyc.get("energy_wh") or 0.0),
                    "peak_w": max(vals) if vals else 0.0,
                    "mean_w": (sum(vals) / len(vals)) if vals else 0.0,
                    "signature": cyc.get("signature") if isinstance(cyc.get("signature"), dict) else {},
                },
                "qc": derive_qc(cyc),
                "sampleIntervalSec": float(cyc.get("sampling_interval") or 0.0),
            })
        if not bundle_items:
            return {"error": "nothing_to_share"}
        device_meta = {"applianceType": store_appliance_type(appliance_type), "brand": brand, "model": model}
        res = await self._client.upload_device_bundle(
            acct["refresh_token"], acct.get("uid", ""), acct.get("name"), device_meta, bundle_items,
        )
        # Return the raw bundle result ({ok, cycle_ids, errors}) so the caller can
        # tell a partial upload (some cycle_ids present) from a total failure.
        # Only a pre-flight gate short-circuits with an {"error": ...} marker above.
        if not res.get("ok") and not res.get("cycle_ids"):
            res = {**res, "detail": self._client.last_error()}
        return res

    async def download_device(self, device_id: str) -> dict[str, Any]:
        """Adopt a whole-device bundle: for each downloaded profile, import its
        reference cycles into ``reference_cycles`` (merge/upsert; real past_cycles are
        never touched). Returns ``{profiles_adopted, cycles_imported}``.

        Idempotent: a store cycle already imported locally (``meta.source ==
        "store:<id>"``) is skipped, so re-downloading the same device does not
        accumulate duplicate reference cycles (a full clear-then-import lands with
        phases in Stage 2).
        """
        bundle = await self._client.get_device_bundle(device_id)
        already = {
            str((c.get("meta") or {}).get("source") or "")
            for c in self._ps.get_reference_cycles()
        }
        profiles_adopted = 0
        cycles_imported = 0
        for prof in bundle.get("profiles", []) or []:
            program = str(prof.get("program") or prof.get("program_lc") or "").strip()
            if not program:
                continue
            adopted_any = False
            for cyc in prof.get("cycles", []) or []:
                pts = cyc.get("importable")
                if not pts:
                    continue
                store_cid = cyc.get("id")
                if store_cid and f"store:{store_cid}" in already:
                    continue  # already imported on a previous download
                local_id = await self._ps.add_reference_cycle(program, pts, {
                    "store_cycle_id": store_cid,
                    "store_uploaded_at": cyc.get("createdAt"),
                    "sampling_interval": (cyc.get("trace") or {}).get("sampleIntervalSec"),
                })
                if local_id:
                    cycles_imported += 1
                    adopted_any = True
            if adopted_any:
                profiles_adopted += 1
        return {"profiles_adopted": profiles_adopted, "cycles_imported": cycles_imported}
