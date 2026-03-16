"""Diagnostics support for WashData."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .manager import WashDataManager

# Keys that can identify the user or their home network — redacted in all contexts.
_SENSITIVE_KEYS = {
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
    # HA entity / service references that reveal home topology.
    "notify_service",
    "notify_people",
    "notify_actions",
    "power_sensor",
    "external_end_trigger",
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


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]

    # Full store export — same payload as the export_config service, but the
    # entry_data / entry_options pass through the redactor to strip personal keys.
    exported: dict[str, Any] = manager.profile_store.export_data(
        entry_data=dict(entry.data),
        entry_options=dict(entry.options),
    )

    return {
        "entry": _redact(entry.as_dict()),
        "manager_state": {
            "current_state": manager.check_state(),
            "current_program": manager.current_program,
            "time_remaining": manager.time_remaining,
            "cycle_progress": manager.cycle_progress,
            "sample_interval_stats": (
                dict(manager.sample_interval_stats)
                if isinstance(manager.sample_interval_stats, dict)
                else {}
            ),
            "profile_sample_repair_stats": manager.profile_sample_repair_stats,
            "suggestions": manager.profile_store.get_suggestions(),
            "feature_flags": {
                "auto_maintenance": bool(getattr(manager, "_auto_maintenance", False)),
                "save_debug_traces": bool(getattr(manager, "_save_debug_traces", False)),
                "notify_fire_events": bool(getattr(manager, "_notify_fire_events", False)),
            },
        },
        "store_export": _redact(exported),
    }
