"""Diagnostics support for HA WashData."""

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

    # Get internal data from the store
    # We access via public export_data() for diagnostics dump
    store_data = manager.profile_store.export_data().get("data", {})

    return {
        "entry": entry.as_dict(),
        "manager_state": {
            "current_state": manager.check_state,
            "current_program": manager.current_program,
            "time_remaining": manager.time_remaining,
            "cycle_progress": manager.cycle_progress,
            "sample_interval_stats": manager.sample_interval_stats,
            "profile_sample_repair_stats": manager.profile_sample_repair_stats,
            "suggestions": manager.profile_store.get_suggestions(),
        },
        "store_data": store_data,
    }
