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
    CONF_MIN_OFF_GAP,
    DEFAULT_OFF_DELAY_BY_DEVICE,
    DEFAULT_OFF_DELAY,
    DEFAULT_MIN_OFF_GAP_BY_DEVICE,
    DEFAULT_MIN_OFF_GAP,
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

        # Min-off-gap: derived from observed inter-cycle gaps
        min_off_gap = self._suggest_min_off_gap(cycles)
        if min_off_gap is not None:
            suggestions[CONF_MIN_OFF_GAP] = min_off_gap

        return suggestions

    def _suggest_min_off_gap(
        self, cycles: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Derive a min_off_gap suggestion from observed inter-cycle gaps."""
        # Only consider completed, labeled cycles with valid timestamps
        timed_cycles: list[tuple[float, float]] = []
        for c in cycles:
            if not isinstance(c, dict):
                continue
            if c.get("status") not in ("completed", "force_stopped"):
                continue
            label = c.get("profile_name") or c.get("label")
            if not label or label == "noise":
                continue
            try:
                start = float(c["start_time"]) if isinstance(c.get("start_time"), (int, float)) and not isinstance(c.get("start_time"), bool) else None
                end = float(c["end_time"]) if isinstance(c.get("end_time"), (int, float)) and not isinstance(c.get("end_time"), bool) else None
                if start is None or end is None:
                    # Try ISO string parsing
                    from datetime import datetime
                    def _parse_ts(v: Any) -> float | None:
                        if isinstance(v, str):
                            try:
                                return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
                            except ValueError:
                                return None
                        return None
                    start = _parse_ts(c.get("start_time"))
                    end = _parse_ts(c.get("end_time"))
                if start is None or end is None or end <= start:
                    continue
                timed_cycles.append((start, end))
            except (TypeError, ValueError, KeyError):
                continue

        if len(timed_cycles) < 3:
            return None

        timed_cycles.sort(key=lambda x: x[0])
        gaps: list[float] = []
        for i in range(1, len(timed_cycles)):
            gap = timed_cycles[i][0] - timed_cycles[i - 1][1]
            if 30 <= gap <= 86400:  # Only gaps between 30s and 1 day
                gaps.append(gap)

        if len(gaps) < 3:
            return None

        gaps_arr = np.array(gaps)
        # Use the 5th-percentile gap as the safe minimum, with device-type floor
        p05_gap = float(np.percentile(gaps_arr, 5))
        device_floor = (
            DEFAULT_MIN_OFF_GAP_BY_DEVICE.get(self.device_type, DEFAULT_MIN_OFF_GAP)
            if self.device_type is not None
            else DEFAULT_MIN_OFF_GAP
        )
        # Add a 20% safety margin so we never split a real gap into two cycles
        suggested = int(max(device_floor, min(p05_gap * 0.8, 3600)))
        reason = (
            f"Based on {len(gaps)} observed inter-cycle gaps "
            f"(p05={p05_gap:.0f}s). Device floor: {device_floor}s."
        )
        return {"value": suggested, "reason": reason}

    def run_simulation(self, cycle_data: dict[str, Any]) -> dict[str, Any]:
        """Replay a single cycle with varied parameters to find optimal settings.

        For richer, multi-cycle suggestions use :meth:`run_batch_simulation`.
        """
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

        powers = np.array([p[1] for p in readings])
        active_powers = powers[powers > 0.5]

        if len(active_powers) < 5:
            return {}

        min_active = float(np.min(active_powers))

        suggested_stop = round(min_active * 0.8, 2)
        suggested_start = round(min_active * 1.2, 2)

        # Energy suggestions
        suggested_end_energy = 0.05

        # Dead zone: look for early dips in the first 5 minutes
        dead_zone = 0
        for ts_offset, p in readings:
            elapsed = ts_offset
            if elapsed > 300:
                break
            if p < 5.0 and elapsed > 5.0:
                dead_zone = int(elapsed)

        suggested_dead_zone = min(300, dead_zone) if dead_zone > 0 else 60

        return {
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
            },
        }

    def run_batch_simulation(self, cycles: list[dict[str, Any]]) -> dict[str, Any]:
        """Derive parameter suggestions from a collection of labeled cycles.

        Unlike :meth:`run_simulation` (single-cycle heuristics), this method
        aggregates statistics across *multiple* cycles for robustness:

        - Power thresholds from the 5th-percentile minimum active power.
        - Dead zone from the 75th-percentile of early dips across cycles.
        - End-energy threshold from the maximum false-end energy seen.
        - Min-off-gap from the 5th-percentile inter-cycle gap.

        Returns an empty dict when fewer than ``_BATCH_MIN_CYCLES`` valid
        cycles are provided.
        """
        _BATCH_MIN_CYCLES = 5

        valid_cycles: list[list[tuple[float, float]]] = []
        for c in cycles:
            if not isinstance(c, dict):
                continue
            label = c.get("label") or c.get("profile_name")
            if not isinstance(label, str) or not label:
                continue
            if not (
                isinstance(c.get("end_time"), str)
                or c.get("state") == "completed"
                or c.get("status") in ("completed", "force_stopped")
            ):
                continue
            raw = c.get("power_data")
            if not isinstance(raw, list) or len(raw) < 5:
                continue
            start_iso = c.get("start_time") if isinstance(c.get("start_time"), str) else None
            readings_list = power_data_to_offsets(
                cast(list[list[float] | tuple[Any, float]], raw), start_iso
            )
            readings = [(float(o), float(p)) for o, p in readings_list]
            if len(readings) >= 5:
                valid_cycles.append(readings)

        if len(valid_cycles) < _BATCH_MIN_CYCLES:
            return {}

        # --- Power thresholds ---
        lowest_active: list[float] = []
        false_end_energies: list[float] = []
        dead_zone_candidates: list[int] = []

        for readings in valid_cycles:
            powers = np.array([p for _, p in readings])
            active = powers[powers > 0.5]
            if len(active) > 0:
                lowest_active.append(float(np.min(active)))

            # Dead zone: first dip below 5 W within the first 5 minutes
            for ts_offset, p in readings:
                if ts_offset > 300:
                    break
                if p < 5.0 and ts_offset > 5.0:
                    dead_zone_candidates.append(int(ts_offset))
                    break

            # False-end energies: low-power segments that resumed
            in_pause = False
            pause_energy = 0.0
            stop_w = 2.0
            for i in range(1, len(readings)):
                t0, p0 = readings[i - 1]
                t1, p1 = readings[i]
                avg_p = (p0 + p1) / 2.0
                dt_h = (t1 - t0) / 3600.0
                if avg_p < stop_w:
                    if not in_pause:
                        in_pause = True
                        pause_energy = 0.0
                    pause_energy += avg_p * dt_h
                elif in_pause:
                    false_end_energies.append(pause_energy)
                    in_pause = False

        suggestions: dict[str, dict[str, Any]] = {}

        if lowest_active:
            p05_min = float(np.percentile(lowest_active, 5))
            suggested_stop = round(p05_min * 0.8, 2)
            suggested_start = round(max(suggested_stop + 0.1, p05_min * 1.2), 2)
            n = len(lowest_active)
            suggestions[CONF_STOP_THRESHOLD_W] = {
                "value": suggested_stop,
                "reason": (
                    f"Based on p05 of minimum active power across {n} cycles "
                    f"({p05_min:.1f}W)."
                ),
            }
            suggestions[CONF_START_THRESHOLD_W] = {
                "value": suggested_start,
                "reason": (
                    f"Based on p05 of minimum active power across {n} cycles "
                    f"({p05_min:.1f}W)."
                ),
            }

        if false_end_energies:
            max_false = float(np.max(false_end_energies))
            suggested_end = round(max(0.05, max_false * 1.2), 4)
        else:
            suggested_end = 0.05
        suggestions[CONF_END_ENERGY_THRESHOLD] = {
            "value": suggested_end,
            "reason": (
                f"Based on maximum false-end energy "
                f"({float(np.max(false_end_energies)) if false_end_energies else 0:.4f}Wh) "
                f"across {len(valid_cycles)} cycles."
            ),
        }

        if dead_zone_candidates:
            # Use the 75th percentile to cover most cycles without being overly generous
            p75_dz = int(np.percentile(dead_zone_candidates, 75))
            suggested_dz = min(300, p75_dz)
            suggestions[CONF_RUNNING_DEAD_ZONE] = {
                "value": suggested_dz,
                "reason": (
                    f"Based on p75 of early power dips across "
                    f"{len(dead_zone_candidates)} cycles ({suggested_dz}s)."
                ),
            }

        min_off_gap = self._suggest_min_off_gap(cycles)
        if min_off_gap is not None:
            suggestions[CONF_MIN_OFF_GAP] = min_off_gap

        return suggestions

    def apply_suggestions(self, suggestions: dict[str, Any]) -> None:
        """Persist suggestions to the profile store."""
        for key, data in suggestions.items():
            self.profile_store.set_suggestion(key, data["value"], reason=data["reason"])

        if self.hass and suggestions:
            self.hass.async_create_task(self.profile_store.async_save())
