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
from .cycle_detector import CycleDetector, CycleDetectorConfig

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
        power_data = cycle_data.get("power_data", [])
        if not power_data or len(power_data) < 10:
            return {}

        # Convert [(iso_str, power), ...] back to [(datetime, power), ...]
        try:
            readings = []
            for ts_str, power in power_data:
                ts = dt_util.parse_datetime(ts_str)
                if ts:
                    readings.append((ts, power))
        except Exception as e:
            _LOGGER.error("Failed to parse power data for simulation: %s", e)
            return {}

        if not readings:
            return {}

        # 1. Base suggestions from actual trace data (Offline Heuristics)
        # Note: We reuse logic from parameter_optimizer.py but simplified for runtime
        powers = np.array([p[1] for p in readings])
        active_powers = powers[powers > 0.5]
        
        if len(active_powers) < 5:
            return {}

        min_active = np.min(active_powers)
        
        suggested_stop = round(min_active * 0.8, 2)
        suggested_start = round(min_active * 1.2, 2)
        
        # Energy suggestions
        # Simplified: Use 0.05Wh as default end gate
        suggested_end_energy = 0.05
        
        # Timing suggestions (Aggressive as per user feedback)
        # We can't really do gap analysis on a single cycle, 
        # but we can look for early dips for dead zone.
        dead_zone = 0
        for i, (ts, p) in enumerate(readings):
            elapsed = (ts - readings[0][0]).total_seconds()
            if elapsed > 300:
                break
            if p < 5.0 and elapsed > 5.0:
                dead_zone = int(elapsed)
        
        suggested_dead_zone = min(300, dead_zone) if dead_zone > 0 else 60

        new_suggestions = {
            CONF_STOP_THRESHOLD_W: {
                "value": suggested_stop,
                "reason": f"Based on minimum active power ({min_active:.1f}W) observed in last cycle."
            },
            CONF_START_THRESHOLD_W: {
                "value": suggested_start,
                "reason": f"Based on minimum active power ({min_active:.1f}W) observed in last cycle."
            },
            CONF_END_ENERGY_THRESHOLD: {
                "value": suggested_end_energy,
                "reason": "Default recommended baseline for end-of-cycle noise gate."
            },
            CONF_RUNNING_DEAD_ZONE: {
                "value": suggested_dead_zone,
                "reason": f"Based on early power dip detected at {suggested_dead_zone}s."
            }
        }

        return new_suggestions

    def apply_suggestions(self, suggestions: dict[str, Any]) -> None:
        """Persist suggestions to the profile store."""
        for key, data in suggestions.items():
            self.profile_store.set_suggestion(key, data["value"], reason=data["reason"])
        
        if self.hass and suggestions:
            self.hass.async_create_task(self.profile_store.async_save())
