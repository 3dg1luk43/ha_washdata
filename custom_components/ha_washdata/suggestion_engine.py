"""Suggestion engine for WashData."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from homeassistant.core import HomeAssistant

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
    CONF_END_ENERGY_THRESHOLD,
    CONF_RUNNING_DEAD_ZONE,
    DEFAULT_OFF_DELAY_BY_DEVICE,
    DEFAULT_OFF_DELAY,
)
from .time_utils import power_data_to_offsets

if TYPE_CHECKING:
    from .profile_store import ProfileStore

_LOGGER = logging.getLogger(__name__)

class SuggestionEngine:
    """Refined engine for generating data-driven parameter suggestions."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        profile_store: "ProfileStore",
        device_type: str | None = None,
    ) -> None:
        """Initialize the suggestion engine."""
        self.hass = hass
        self.entry_id = entry_id
        self.profile_store = profile_store
        self.device_type = device_type

    def generate_operational_suggestions(self, p95_dt: float, median_dt: float) -> dict[str, Any]:
        """Generate suggestions for operational parameters based on cadence."""
        suggestions: dict[str, dict[str, Any]] = {}

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
        # Use device-specific default as floor to prevent splitting cycles with long pauses
        device_floor = (
            DEFAULT_OFF_DELAY_BY_DEVICE.get(self.device_type, DEFAULT_OFF_DELAY)
            if self.device_type is not None
            else DEFAULT_OFF_DELAY
        )
        suggested_off_delay = int(max(device_floor, p95_dt * 5))

        reason_off = f"Based on observed update cadence (p95={p95_dt:.1f}s) * 5"
        if suggested_off_delay == device_floor:
            if self.device_type and self.device_type in DEFAULT_OFF_DELAY_BY_DEVICE:
                reason_off = (
                    f"Used device-specific safe minimum for {self.device_type} ({device_floor}s)."
                )
            else:
                reason_off = f"Used generic safe minimum ({DEFAULT_OFF_DELAY}s)."

        suggestions[CONF_OFF_DELAY] = {
            "value": suggested_off_delay,
            "reason": reason_off
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
        suggestions: dict[str, dict[str, Any]] = {}

        cycles = self.profile_store.get_past_cycles()[-100:]
        profiles = self.profile_store.get_profiles()

        ratios: list[float] = []
        for c in cycles:
            if not isinstance(c, dict):
                continue
            profile_name = c.get("profile_name")
            if not isinstance(profile_name, str) or c.get("status") == "interrupted":
                continue
            prof = profiles.get(profile_name)
            if not isinstance(prof, dict):
                continue
            try:
                avg = float(prof.get("avg_duration") or 0.0)
                dur = float(c.get("duration") or 0.0)
            except (TypeError, ValueError):
                continue
            if avg > 60 and dur > 60:
                ratios.append(dur / avg)

        if len(ratios) >= 10:
            arr: np.ndarray[Any, np.dtype[np.float64]] = np.array(ratios, dtype=float)
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
        power_data_raw: Any = cycle_data.get("power_data", [])
        if not isinstance(power_data_raw, list):
            return {}
        power_data = cast(list[list[float] | tuple[Any, float]], power_data_raw)
        if len(power_data) < 10:
            return {}

        start_time_raw = cycle_data.get("start_time")
        start_time_iso = (
            start_time_raw if isinstance(start_time_raw, str) and start_time_raw else None
        )

        # Normalise power_data to [[offset_sec, power], ...] regardless of source format.
        readings_list = power_data_to_offsets(power_data, start_time_iso)

        readings: list[tuple[float, float]] = [
            (float(offset), float(power)) for offset, power in readings_list
        ]

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
        for ts_offset, p in readings:
            elapsed = ts_offset
            if elapsed > 300:
                break
            if p < 5.0 and elapsed > 5.0:
                dead_zone = int(elapsed)

        suggested_dead_zone = min(300, dead_zone) if dead_zone > 0 else 60

        new_suggestions: dict[str, dict[str, Any]] = {
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
