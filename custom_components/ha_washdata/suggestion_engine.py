"""Suggestion engine for HA WashData."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

import numpy as np
from homeassistant.core import HomeAssistant
import homeassistant.util.dt as dt_util

from .const import (
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_DURATION_TOLERANCE,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_START_ENERGY_THRESHOLD,
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_OFF_GAP,
    CONF_RUNNING_DEAD_ZONE,
)

if TYPE_CHECKING:
    from .profile_store import ProfileStore

_LOGGER = logging.getLogger(__name__)

class SuggestionEngine:
    """Refined engine for generating data-driven parameter suggestions."""

    def __init__(
        self, hass: HomeAssistant, entry_id: str, profile_store: "ProfileStore"
    ) -> None:
        """Initialize the suggestion engine."""
        self.hass = hass
        self.entry_id = entry_id
        self.profile_store = profile_store

    def generate_operational_suggestions(self, p95_dt: float, median_dt: float) -> dict[str, Any]:
        """Generate suggestions for operational parameters based on cadence."""
        suggestions = {}

        # 1. Watchdog Interval
        suggested_watchdog = int(max(30, p95_dt * 10))
        suggestions[CONF_WATCHDOG_INTERVAL] = {
            "value": suggested_watchdog,
            "reason": f"Based on observed update cadence (p95={p95_dt:.1f}s) * 10 (min 30s buffer)."
        }

        # 2. No Update Timeout
        suggested_timeout = int(max(60, p95_dt * 20))
        suggestions[CONF_NO_UPDATE_ACTIVE_TIMEOUT] = {
            "value": suggested_timeout,
            "reason": f"Based on observed update cadence (p95={p95_dt:.1f}s) * 20 (min 60s)."
        }

        # 3. Off Delay
        suggested_off_delay = int(max(60, p95_dt * 5))
        suggestions[CONF_OFF_DELAY] = {
            "value": suggested_off_delay,
            "reason": f"Based on observed update cadence (p95={p95_dt:.1f}s) * 5 (min 60s)."
        }

        # 4. Profile Match Interval
        suggested_match = int(max(10, median_dt * 10))
        suggestions[CONF_PROFILE_MATCH_INTERVAL] = {
            "value": suggested_match,
            "reason": f"Based on observed update cadence (median={median_dt:.1f}s) * 10."
        }

        return suggestions

    def generate_model_suggestions(self) -> dict[str, Any]:
        """Generate suggestions for model parameters based on past cycles."""
        suggestions = {}
        
        cycles = self.profile_store.get_past_cycles()[-100:]
        profiles = self.profile_store.get_profiles()
        
        ratios = []
        for c in cycles:
            if not c.get("profile_name") or c.get("status") == "interrupted":
                continue
            prof = profiles.get(c["profile_name"])
            if not prof:
                continue
            avg = prof.get("avg_duration", 0)
            dur = c.get("duration", 0)
            if avg > 60 and dur > 60:
                ratios.append(dur / avg)

        if len(ratios) >= 10:
            arr = np.array(ratios)
            deviations = np.abs(arr - 1.0)
            p95_dev = float(np.percentile(deviations, 95))
            
            suggested_tol = min(0.50, max(0.10, round(p95_dev + 0.05, 2)))
            reason_tol = f"Based on duration variance of {len(ratios)} recent labeled cycles (p95 dev={p95_dev:.2f})."
            
            suggestions[CONF_DURATION_TOLERANCE] = {"value": suggested_tol, "reason": reason_tol}
            suggestions[CONF_PROFILE_DURATION_TOLERANCE] = {"value": suggested_tol, "reason": reason_tol}

            p05_ratio = float(np.percentile(arr, 5))
            p95_ratio = float(np.percentile(arr, 95))
            
            min_r = max(0.1, round(p05_ratio - 0.1, 2))
            max_r = min(3.0, round(p95_ratio + 0.1, 2))
            
            if min_r < max_r - 0.2:
                suggestions[CONF_PROFILE_MATCH_MIN_DURATION_RATIO] = {
                    "value": min_r,
                    "reason": f"Based on labeled cycle durations (p05={p05_ratio:.2f})."
                }
                suggestions[CONF_PROFILE_MATCH_MAX_DURATION_RATIO] = {
                    "value": max_r,
                    "reason": f"Based on labeled cycle durations (p95={p95_ratio:.2f})."
                }

        return suggestions

    def run_simulation(self, cycle_data: dict[str, Any]) -> dict[str, Any]:
        """Replay a cycle with varied parameters to find optimal settings."""
        # This will be implemented in Phase 3
        return {}

    def apply_suggestions(self, suggestions: dict[str, Any]) -> None:
        """Persist suggestions to the profile store."""
        for key, data in suggestions.items():
            self.profile_store.set_suggestion(key, data["value"], reason=data["reason"])
        
        if self.hass and suggestions:
            self.hass.async_create_task(self.profile_store.async_save())
