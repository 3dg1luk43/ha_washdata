"""Diagnostics support for WashData."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .manager import WashDataManager


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]

    # Build a safe, whitelisted diagnostics summary from store export metadata.
    exported = manager.profile_store.export_data()
    exported_data = exported.get("data", {}) if isinstance(exported, dict) else {}
    profiles = exported_data.get("profiles", {}) if isinstance(exported_data, dict) else {}
    past_cycles = (
        exported_data.get("past_cycles", []) if isinstance(exported_data, dict) else []
    )
    envelopes = exported_data.get("envelopes", {}) if isinstance(exported_data, dict) else {}

    last_cycle_summary: dict[str, Any] | None = None
    if isinstance(past_cycles, list) and past_cycles and isinstance(past_cycles[-1], dict):
        last_cycle = past_cycles[-1]
        last_cycle_summary = {
            "id": last_cycle.get("id"),
            "has_profile_name": bool(last_cycle.get("profile_name")),
            "status": last_cycle.get("status"),
            "start_time": last_cycle.get("start_time"),
            "end_time": last_cycle.get("end_time"),
            "duration": last_cycle.get("duration"),
            "energy_wh": last_cycle.get("energy_wh"),
        }

    store_data = {
        "version": exported.get("version") if isinstance(exported, dict) else None,
        "exported_at": exported.get("exported_at") if isinstance(exported, dict) else None,
        "profile_count": len(profiles) if isinstance(profiles, dict) else 0,
        "past_cycle_count": len(past_cycles) if isinstance(past_cycles, list) else 0,
        "envelope_count": len(envelopes) if isinstance(envelopes, dict) else 0,
        "last_cycle_summary": last_cycle_summary,
        "feature_flags": {
            "auto_maintenance": bool(getattr(manager, "_auto_maintenance", False)),
            "save_debug_traces": bool(getattr(manager, "_save_debug_traces", False)),
            "notify_fire_events": bool(getattr(manager, "_notify_fire_events", False)),
        },
    }

    _SENSITIVE_KEYS = {
        # Common ConfigEntry metadata and flow/user identifiers.
        "auth",
        "entry_id",
        "flow_id",
        "flow_title",
        "handler",
        "name",
        "source",
        "title",
        "unique_id",
        "user_id",
        # config-entry data / options fields that can include personal identifiers
        "notify_service",
        "notify_people",
        "notify_actions",
        "power_sensor",
        "external_end_trigger",
    }

    _STORE_REDACT_KEYS = {
        "profile_name",
    }

    def _redact(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: "**REDACTED**" if k in _SENSITIVE_KEYS else _redact(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_redact(v) for v in obj]
        return obj

    def _sanitize_store_data(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: "<redacted_profile_name>" if k in _STORE_REDACT_KEYS else _sanitize_store_data(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_sanitize_store_data(v) for v in obj]
        return obj

    raw_entry = entry.as_dict()

    return {
        "entry": _redact(raw_entry),
        "manager_state": {
            "current_state": manager.check_state(),
            "current_program": manager.current_program,
            "time_remaining": manager.time_remaining,
            "cycle_progress": manager.cycle_progress,
            "sample_interval_stats": dict(manager.sample_interval_stats) if isinstance(manager.sample_interval_stats, dict) else {},
            "profile_sample_repair_stats": manager.profile_sample_repair_stats,
            "suggestions": manager.profile_store.get_suggestions(),
        },
        "store_data": _sanitize_store_data(store_data),
    }
