"""Profile storage and matching logic for HA WashData."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, TypeAlias, cast
import json

import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    STORAGE_KEY,
    STORAGE_VERSION,
    DEFAULT_MAX_PAST_CYCLES,
    DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
    DEFAULT_MAX_FULL_TRACES_UNLABELED,
    DEFAULT_DTW_BANDWIDTH,
)
from .features import compute_signature
from .signal_processing import resample_uniform, resample_adaptive, Segment
from . import analysis

_LOGGER = logging.getLogger(__name__)

JSONDict: TypeAlias = dict[str, Any]
CycleDict: TypeAlias = dict[str, Any]


def profile_sort_key(name: str) -> tuple[int, int, str]:
    """Sort key for profile names: numeric-prefixed first (by number), then alphabetically."""
    match = re.match(r'^(\d+)', name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)




def trim_zero_power_data(
    data: list[list[float]], 
    threshold: float = 0.5
) -> list[list[float]]:
    """Trim leading/trailing zero/near-zero power readings from stored data.
    
    Args:
        data: List of [offset, power] pairs
        threshold: Power values <= this are considered "zero"
        
    Returns:
        Trimmed list with leading/trailing zeros removed
    """
    if not data:
        return data
    
    # Find first non-zero reading
    start_idx = 0
    for i, point in enumerate(data):
        if point[1] > threshold:
            start_idx = i
            break
    else:
        # All readings are zero - keep at least one
        return data[:1] if data else []
    
    # Find last non-zero reading
    end_idx = len(data) - 1
    for i in range(len(data) - 1, -1, -1):
        if data[i][1] > threshold:
            end_idx = i
            break
    
    # Return trimmed slice (inclusive of end)
    return data[start_idx:end_idx + 1]


@dataclasses.dataclass
class SVGCurve:
    """Definition for a curve in the SVG chart."""
    points: list[tuple[float, float]]  # (x, y)
    color: str
    opacity: float = 1.0
    stroke_width: int = 2
    dasharray: str | None = None


def _generate_generic_svg(
    title: str,
    curves: list[SVGCurve],
    width: int = 800,
    height: int = 400,
    max_x_override: float | None = None,
    max_y_override: float | None = None,
    markers: list[dict[str, Any]] | None = None, # {x, label, color}
) -> str:
    """Generate a generic time-series SVG chart."""
    if not curves:
        return ""

    padding_x = 50
    padding_y = 40
    graph_w = width - 2 * padding_x
    graph_h = height - 2 * padding_y

    # Determine bounds
    all_x = [p[0] for c in curves for p in c.points]
    all_y = [p[1] for c in curves for p in c.points]

    if not all_x:
        return ""

    max_x = max_x_override if max_x_override is not None else max(all_x)
    max_y = max_y_override if max_y_override is not None else max(all_y, default=1.0)

    # Headroom
    max_y = max(max_y, 10.0) * 1.05
    max_x = max(max_x, 1.0) # Ensure no div by zero

    def to_x(t: float) -> float:
        return padding_x + (t / max_x) * graph_w

    def to_y(p: float) -> float:
        return height - padding_y - (p / max_y) * graph_h

    # Build Paths
    paths = []
    for c in curves:
        if not c.points:
            continue

        pts = []
        # Optimization: verify step size if huge data
        for x_val, y_val in c.points:
            pts.append(f"{to_x(x_val):.1f},{to_y(y_val):.1f}")

        path_d = " ".join(pts)
        style = f'stroke="{c.color}" stroke-width="{c.stroke_width}" stroke-opacity="{c.opacity}" fill="none"'
        if c.dasharray:
            style += f' stroke-dasharray="{c.dasharray}"'

        paths.append(f'<polyline points="{path_d}" {style} />')

    # Build Markers
    marker_svgs = []
    if markers:
        for m in markers:
            mx = m["x"]
            if 0 <= mx <= max_x:
                screen_x = to_x(mx)
                color = m.get("color", "#aaa")
                label = m.get("label", "")
                marker_svgs.append(
                    f'<line x1="{screen_x:.1f}" y1="{padding_y}" x2="{screen_x:.1f}" y2="{height - padding_y}" '
                    f'stroke="{color}" stroke-dasharray="4" stroke-width="1" />'
                )
                if label:
                    marker_svgs.append(
                        f'<text x="{screen_x:.1f}" y="{height - padding_y + 15}" '
                        f'fill="{color}" font-size="12" text-anchor="middle">{label}</text>'
                    )

    # Grid & Axes (border + mid lines)
    grid = f"""
    <rect x="0" y="0" width="{width}" height="{height}" fill="#1c1c1c" />
    <line x1="{padding_x}" y1="{height - padding_y}" x2="{width - padding_x}" y2="{height - padding_y}" stroke="#444" stroke-width="2" />
    <line x1="{padding_x}" y1="{padding_y}" x2="{padding_x}" y2="{height - padding_y}" stroke="#444" stroke-width="2" />
    <text x="{padding_x}" y="{padding_y - 15}" fill="#aaa" font-size="16">{int(max_y)}W</text>
    <text x="{width - padding_x}" y="{height - 10}" fill="#aaa" font-size="16" text-anchor="middle">{int(max_x)}s</text>
    <text x="{width / 2}" y="{padding_y - 15}" fill="#fff" font-size="20" text-anchor="middle" font-weight="bold">{title}</text>
    """

    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'style="background-color: #1c1c1c; font-family: sans-serif;">'
    )

    return header + grid + "".join(paths) + "".join(marker_svgs) + "</svg>"



@dataclasses.dataclass
class MatchResult:
    """Result of a profile matching attempt."""

    best_profile: str | None
    confidence: float
    expected_duration: float
    matched_phase: str | None
    candidates: list[dict[str, Any]]
    is_ambiguous: bool
    ambiguity_margin: float
    ranking: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    debug_details: dict[str, Any] = dataclasses.field(default_factory=dict)
    is_confident_mismatch: bool = False
    mismatch_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with JSON-serializable types, excluding heavy arrays."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            if isinstance(obj, np.ndarray):
                # Fallback for unexpected arrays: just describe shape
                return f"<array shape={obj.shape}>"
            if isinstance(obj, dict):
                # Exclude huge raw data arrays from cycle candidates
                return {
                    k: _convert(v)
                    for k, v in obj.items()
                    if k not in ("current", "sample", "metrics", "warping_path")
                }
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            if dataclasses.is_dataclass(obj):
                return text_type_safe_asdict(obj)
            return obj

        def text_type_safe_asdict(d_obj: Any) -> dict[str, Any]:
            return {f.name: _convert(getattr(d_obj, f.name)) for f in dataclasses.fields(d_obj)}

        return text_type_safe_asdict(self)



def decompress_power_data(cycle: CycleDict) -> list[tuple[str, float]]:
    """Decompress cycle power data for matching (Module-level helper)."""
    compressed_raw = cycle.get("power_data", [])
    if not isinstance(compressed_raw, list) or not compressed_raw:
        return []

    compressed: list[Any] = cast(list[Any], compressed_raw)

    # Handle missing start_time gracefully
    if "start_time" not in cycle:
        return []

    try:
        start_time = datetime.fromisoformat(cycle["start_time"])
    except ValueError:
        return []

    result: list[tuple[str, float]] = []

    for item in compressed:
        if not isinstance(item, (list, tuple)):
            continue
        try:
            offset_seconds, power = cast(tuple[Any, Any], item)
        except (TypeError, ValueError):
            continue
        if isinstance(offset_seconds, (int, float)) and isinstance(power, (int, float)):
            timestamp = start_time.timestamp() + float(offset_seconds)
            result.append((datetime.fromtimestamp(timestamp).isoformat(), float(power)))

    return result


def compress_power_data(cycle: CycleDict) -> list[Any] | None:
    """Compress cycle power data to [offset, power] format (Module-level helper).
    
    Returns the compressed list structure or None if compression failed/not needed.
    """
    raw_data = cycle.get("power_data")
    if not isinstance(raw_data, list) or not raw_data:
        return None

    # Check if already compressed (first element is number or mixed format)
    first = raw_data[0]
    if isinstance(first, (int, float)):
        # Already flat list (very old format?) or specific compression
        return None 
    if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
        # Already compressed [offset, power]
        return None

    # Proceed with compression from [iso_string, power]
    if "start_time" not in cycle:
        return None

    try:
        start_ts = datetime.fromisoformat(cycle["start_time"]).timestamp()
        compressed: list[list[float]] = []
        
        last_saved_p = -999.0
        last_saved_t = -999.0
        
        for i, entry in enumerate(raw_data):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                t_str, p_val_raw = entry
                try:
                    # Handle both ISO string and potential timestamp float
                    if isinstance(t_str, str):
                        t = datetime.fromisoformat(t_str).timestamp()
                    else:
                        t = float(t_str)
                    
                    p_val = float(p_val_raw)
                    offset = round(t - start_ts, 1)
                    if offset < 0:
                        offset = 0.0
                        
                    # Save first and last
                    is_endpoint = i == 0 or i == len(raw_data) - 1

                    # Downsample: change > 1W or gap > 60s
                    if (
                        is_endpoint
                        or abs(p_val - last_saved_p) > 1.0
                        or (offset - last_saved_t) > 60
                    ):
                        compressed.append([offset, round(p_val, 1)])
                        last_saved_p = p_val
                        last_saved_t = offset
                        
                except (ValueError, TypeError):
                    continue
        return compressed
    except Exception:
        return None


class WashDataStore(Store[JSONDict]):
    """Store implementation with migration support."""

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,  # pylint: disable=unused-argument
        old_data: JSONDict,
    ) -> JSONDict:
        """Migrate data to the new version."""
        if old_major_version < 2:
            _LOGGER.info("Migrating storage from v%s to v2", old_major_version)
            # Logic moved from ProfileStore._migrate_v1_to_v2
            cycles = old_data.get("past_cycles", [])
            migrated_cycles = 0
            for cycle in cycles:
                if "signature" not in cycle and cycle.get("power_data"):
                    try:
                        # Decompress using helper
                        tuples = decompress_power_data(cycle)
                        if tuples and len(tuples) > 10:
                            # Convert to relative time arrays for signature computation
                            start = datetime.fromisoformat(
                                cycle["start_time"]
                            ).timestamp()
                            ts_arr = []
                            p_arr = []
                            for t_str, p in tuples:
                                t = datetime.fromisoformat(t_str).timestamp()
                                ts_arr.append(t - start)
                                p_arr.append(p)

                            sig = compute_signature(np.array(ts_arr), np.array(p_arr))
                            cycle["signature"] = dataclasses.asdict(sig)
                            migrated_cycles += 1
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        _LOGGER.warning(
                            "Failed to migrate signature for cycle %s: %s", cycle.get("id"), e
                        )

            _LOGGER.info(
                "Migration v1->v2: Computed signatures for %s cycles", migrated_cycles
            )

        if old_major_version < 3:
            _LOGGER.info("Migrating storage from v%s to v3", old_major_version)
            cycles = old_data.get("past_cycles", [])
            profiles = old_data.get("profiles", {})
            migrated_count = 0
            
            # 1. Compress Power Data & Ensure Status
            for cycle in cycles:
                # Ensure status
                if "status" not in cycle:
                    cycle["status"] = "completed"
                
                # Compress power data if needed
                # (Check if it looks like old list of lists/tuples)
                if cycle.get("power_data") and isinstance(cycle["power_data"], list):
                    first_elem = cycle["power_data"][0] if cycle["power_data"] else None
                    # If it's a list/tuple [offset, power], it's uncompressed (v2)
                    # If it's [start_ts, power, dt, encoding], it's compressed (v3)
                    if isinstance(first_elem, (list, tuple)) and len(first_elem) == 2:
                        try:
                            # Use helper to compress in-place (returns compressed dict structure, we need to adapt)
                            # Actually, compress_power_data returns the list of points. 
                            # We need to manually apply compression logic here or use a helper that MUTATES the cycle.
                            # The helper `compress_power_data` takes a cycle dict and returns the compressed list structure.
                            # Let's verify what compress_power_data does.
                            # It takes (cycle_data: dict) -> list[Any] (the compressed structure)
                            compressed = compress_power_data(cycle)
                            if compressed:
                                cycle["power_data"] = compressed
                                migrated_count += 1
                        except Exception as e:
                             _LOGGER.warning(
                                "Failed to compress data for cycle %s: %s", cycle.get("id"), e
                            )

            # 2. Ensure Device Type in Profiles
            for profile in profiles.values():
                if "device_type" not in profile:
                    profile["device_type"] = "washing_machine"

            _LOGGER.info(
                "Migration v2->v3: Compressed data for %s cycles", migrated_count
            )

        return old_data

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage usage statistics."""
        data = self._data  # pylint: disable=protected-access
        if not data:
            data = await self.async_load() or {}

        # Rough file size estimation if possible, else 0
        file_size_kb = 0
        try:
            path = self.path  # pylint: disable=no-member
            if os.path.exists(path):
                file_size_kb = os.path.getsize(path) / 1024
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        cycles = data.get("past_cycles", [])
        profiles = data.get("profiles", {})

        debug_traces_count = sum(1 for c in cycles if c.get("debug_data"))

        return {
            "file_size_kb": round(file_size_kb, 1),
            "total_cycles": len(cycles),
            "total_profiles": len(profiles),
            "debug_traces_count": debug_traces_count,
        }

    async def async_clear_debug_data(self) -> int:
        """Clear granular debug data from all cycles to free space."""
        if not self._data:
            await self.async_load()

        cycles = self._data.get("past_cycles", [])
        count = 0
        for cycle in cycles:
            if "debug_data" in cycle:
                del cycle["debug_data"]
                count += 1

        if count > 0:
            await self.async_save(self._data)
            _LOGGER.info("Cleared debug data from %s cycles", count)

        return count


class ProfileStore:
    """Manages storage of washer profiles and past cycles."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        min_duration_ratio: float = 0.50,
        max_duration_ratio: float = 1.50,
        save_debug_traces: bool = False,
        match_threshold: float = 0.4,
        unmatch_threshold: float = 0.35,
    ) -> None:
        """Initialize the profile store."""
        self.hass = hass
        self.entry_id = entry_id
        self._min_duration_ratio = min_duration_ratio
        self._max_duration_ratio = max_duration_ratio
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold
        self.dtw_bandwidth: float = DEFAULT_DTW_BANDWIDTH
        self._save_debug_traces = save_debug_traces

        # Cache for resampled sample segments: key=(cycle_id, dt)
        self._cached_sample_segments: dict[tuple[str, float], Segment] = {}
        # Profile duration tolerance (set by manager; reserved for duration-based heuristics)
        self._duration_tolerance: float = 0.25
        # Retention policy: cap total cycles and number of full-resolution traces per profile
        self._max_past_cycles = DEFAULT_MAX_PAST_CYCLES
        self._max_full_traces_per_profile = DEFAULT_MAX_FULL_TRACES_PER_PROFILE
        self._max_full_traces_unlabeled = DEFAULT_MAX_FULL_TRACES_UNLABELED
        # Separate store for each entry to avoid giant files
        # Use WashDataStore to handle migration
        self._store: Store[JSONDict] = WashDataStore(
            hass, STORAGE_VERSION, f"{STORAGE_KEY}.{entry_id}"
        )
        self._data: JSONDict = {
            "profiles": {},
            "past_cycles": [],
            "envelopes": {},  # Cached statistical envelopes per profile
            "auto_adjustments": [],  # Log of automatic setting changes
            "suggestions": {},  # Suggested settings (do NOT change user options)
            "feedback_history": {},  # Persisted user feedback (cycle_id -> record)
            "pending_feedback": {},  # Persisted pending feedback requests
        }
        



    def set_suggestion(self, key: str, value: Any, reason: str | None = None) -> None:
        """Store a suggested setting value without changing config entry options."""
        suggestions: JSONDict = self._data.setdefault("suggestions", {})
        suggestions[key] = {
            "value": value,
            "reason": reason,
            "updated": dt_util.now().isoformat(),
        }

    def get_suggestions(self) -> dict[str, Any]:
        """Return current suggestion map."""
        raw = self._data.get("suggestions")
        if isinstance(raw, dict):
            suggestions = cast(JSONDict, raw)
            return suggestions.copy()
        return {}

    def get_feedback_history(self) -> dict[str, dict[str, Any]]:
        """Return mutable feedback history mapping (cycle_id -> record)."""
        raw = self._data.setdefault("feedback_history", {})
        if isinstance(raw, dict):
            return cast(dict[str, dict[str, Any]], raw)
        return {}

    def get_pending_feedback(self) -> dict[str, dict[str, Any]]:
        """Return mutable pending feedback mapping (cycle_id -> request)."""
        raw = self._data.setdefault("pending_feedback", {})
        if isinstance(raw, dict):
            return cast(dict[str, dict[str, Any]], raw)
        return {}

    def add_pending_feedback(self, cycle_id: str, request_data: dict[str, Any]) -> None:
        """Add a pending feedback request (sync wrapper, does not save immediately)."""
        feedbacks = self.get_pending_feedback()
        feedbacks[cycle_id] = request_data
        # Caller must ensure save is called eventually

    def remove_pending_feedback(self, cycle_id: str) -> None:
        """Remove a pending feedback request."""
        feedbacks = self.get_pending_feedback()
        if cycle_id in feedbacks:
            del feedbacks[cycle_id]


    def get_profile(self, name: str) -> JSONDict | None:
        """Return a single profile by name with calculated stats (via list_profiles)."""
        # Reuse list_profiles logic to ensure consistency and avoid duplication
        all_profiles = self.list_profiles()
        return next((p for p in all_profiles if p["name"] == name), None)

    def get_profiles(self) -> dict[str, JSONDict]:
        """Return mutable profiles mapping (profile_name -> profile data)."""
        raw = self._data.setdefault("profiles", {})
        if isinstance(raw, dict):
            return cast(dict[str, JSONDict], raw)
        return {}

    def get_past_cycles(self) -> list[CycleDict]:
        """Return mutable list of stored cycles."""
        raw = self._data.setdefault("past_cycles", [])
        if isinstance(raw, list):
            return cast(list[CycleDict], raw)
        return []

    def set_duration_tolerance(self, tolerance: float) -> None:
        """Set the profile duration tolerance used by matching heuristics."""
        try:
            self._duration_tolerance = float(tolerance)
        except (TypeError, ValueError):
            return

    def set_retention_limits(
        self,
        *,
        max_past_cycles: int,
        max_full_traces_per_profile: int,
        max_full_traces_unlabeled: int,
    ) -> None:
        """Set retention caps for stored cycles and full-resolution traces."""
        try:
            self._max_past_cycles = int(max_past_cycles)
            self._max_full_traces_per_profile = int(max_full_traces_per_profile)
            self._max_full_traces_unlabeled = int(max_full_traces_unlabeled)
        except (TypeError, ValueError):
            return

    def get_duration_ratio_limits(self) -> tuple[float, float]:
        """Return (min_duration_ratio, max_duration_ratio) used for duration matching."""
        return (float(self._min_duration_ratio), float(self._max_duration_ratio))

    def set_duration_ratio_limits(self, *, min_ratio: float, max_ratio: float) -> None:
        """Update duration ratio bounds used for duration matching."""
        try:
            self._min_duration_ratio = float(min_ratio)
            self._max_duration_ratio = float(max_ratio)
        except (TypeError, ValueError):
            return

    async def async_load(self) -> None:
        """Load data from storage with migration."""
        # WashDataStore handles migration internally via _async_migrate_func
        data = await self._store.async_load()
        if data:
            self._data = data

    # _migrate_v1_to_v2 and _decompress_power_from_raw removed; logic moved to WashDataStore

    def _decompress_power_from_raw(
        self, cycle: CycleDict
    ) -> list[tuple[float, float, float]] | None:
        # Helper not needed if we use _decompress_power_data
        pass

    async def async_repair_profile_samples(self) -> dict[str, int]:
        """Repair profile sample references after retention or migrations.

        Ensures each profile's sample_cycle_id points to an existing cycle that still
        has full-resolution power_data. If missing, picks the newest available cycle
        with power_data and assigns it as the sample (and labels that cycle to the
        profile if it was unlabeled).

        Returns stats dict.
        """
        stats = {
            "profiles_checked": 0,
            "profiles_repaired": 0,
            "cycles_labeled_as_sample": 0,
        }

        profiles: dict[str, dict[str, Any]] = self._data.get("profiles", {}) or {}
        cycles: list[dict[str, Any]] = self._data.get("past_cycles", []) or []
        if not profiles or not cycles:
            return stats

        by_id: dict[str, dict[str, Any]] = {c["id"]: c for c in cycles if c.get("id")}

        def newest_unlabeled_with_power_data() -> dict[str, Any] | None:
            candidates: list[dict[str, Any]] = [
                c for c in cycles if c.get("power_data") and not c.get("profile_name")
            ]
            if not candidates:
                return None
            try:
                return max(candidates, key=lambda c: c.get("start_time", ""))
            except Exception:  # pylint: disable=broad-exception-caught
                return candidates[-1]

        for profile_name, profile in profiles.items():
            stats["profiles_checked"] += 1
            sample_id = profile.get("sample_cycle_id")
            sample = by_id.get(sample_id) if sample_id else None

            # Sample is valid only if it exists and still has power_data
            if sample and sample.get("power_data"):
                continue

            # Prefer newest already-labeled cycle for this profile that still has power_data
            labeled_candidates = [
                c
                for c in cycles
                if c.get("profile_name") == profile_name and c.get("power_data")
            ]
            if labeled_candidates:
                try:
                    chosen = max(
                        labeled_candidates, key=lambda c: c.get("start_time", "")
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    chosen = labeled_candidates[-1]
            else:
                # Fallback: pick newest UNLABELED cycle with power_data
                chosen = newest_unlabeled_with_power_data()

            if not chosen:
                continue

            profile["sample_cycle_id"] = chosen.get("id")
            if chosen.get("duration"):
                profile["avg_duration"] = chosen["duration"]

            # If chosen cycle is unlabeled, label it to this profile to bootstrap matching
            if not chosen.get("profile_name"):
                chosen["profile_name"] = profile_name
                stats["cycles_labeled_as_sample"] += 1

            stats["profiles_repaired"] += 1
            try:
                await self.async_rebuild_envelope(profile_name)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        return stats

    async def async_save(self) -> None:
        """Save data to storage."""
        await self._store.async_save(self._data)

    async def async_save_active_cycle(self, detector_snapshot: JSONDict) -> None:
        """Save the active cycle state to storage (throttled by Manager)."""
        self._data["active_cycle"] = detector_snapshot
        self._data["last_active_save"] = dt_util.now().isoformat()
        await self._store.async_save(self._data)

    def get_active_cycle(self) -> JSONDict | None:
        """Get the saved active cycle."""
        raw = self._data.get("active_cycle")
        if isinstance(raw, dict):
            return cast(JSONDict, raw)
        return None

    def get_last_active_save(self) -> datetime | None:
        """Return the last time the active cycle snapshot was persisted."""
        raw = self._data.get("last_active_save")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            return dt_util.parse_datetime(raw)
        except ValueError:
            return None

    async def async_clear_active_cycle(self) -> None:
        """Clear the active cycle snapshot from storage."""
        if "active_cycle" in self._data:
            del self._data["active_cycle"]
            await self._store.async_save(self._data)

    def add_cycle(self, cycle_data: CycleDict) -> None:
        """Add a completed cycle to history (sync wrapper, schedules async tasks)."""
        self._add_cycle_data(cycle_data)
        self.hass.async_create_task(self.async_enforce_retention())

    async def async_add_cycle(self, cycle_data: CycleDict) -> None:
        """Add a completed cycle to history asynchronously."""
        self._add_cycle_data(cycle_data)
        await self.async_enforce_retention()

    def _add_cycle_data(self, cycle_data: CycleDict) -> None:
        """Internal logic to add cycle data to storage."""
        # Generate SHA256 ID
        unique_str = f"{cycle_data['start_time']}_{cycle_data['duration']}"
        cycle_data["id"] = hashlib.sha256(unique_str.encode()).hexdigest()[:12]

        # Preserve profile_name if already set by manager; default to None otherwise
        if "profile_name" not in cycle_data:
            cycle_data["profile_name"] = None  # Initially unknown

        # Store power data at native sampling resolution
        # Format: [seconds_offset, power] preserves actual sample rate from device
        # (e.g., 3s intervals from test socket, 60s intervals from real socket)
        raw_data: list[Any] = cycle_data.get("power_data", []) or []
        _LOGGER.debug("add_cycle: raw_data has %s points", len(raw_data))

        if raw_data:
            start_ts = datetime.fromisoformat(cycle_data["start_time"]).timestamp()
            stored: list[list[float]] = []
            offsets: list[float] = []

            for point in raw_data:
                if not isinstance(point, (list, tuple)):
                    continue
                point_any = cast(list[Any] | tuple[Any, ...], point)
                try:
                    ts_raw = point_any[0]
                    p_raw = point_any[1]
                except IndexError:
                    continue

                if isinstance(ts_raw, str):
                    try:
                        t_val = datetime.fromisoformat(ts_raw).timestamp()
                    except ValueError:
                        continue
                elif isinstance(ts_raw, (int, float)):
                    t_val = float(ts_raw)
                else:
                    continue

                try:
                    p_val = float(p_raw)
                except (TypeError, ValueError):
                    continue

                # Store as [offset_seconds, power] for consistency
                offset = round(t_val - start_ts, 1)
                offsets.append(offset)
                stored.append([offset, round(p_val, 1)])

            # Calculate average sampling interval (in seconds)
            if len(offsets) > 1:
                intervals = np.diff(offsets)
                sampling_interval = float(np.median(intervals[intervals > 0]))
            else:
                sampling_interval = 1.0  # Default fallback

            # Trim leading/trailing zero readings for cleaner data
            # SKIP for completed cycles to preserve end spike data
            if cycle_data.get("status") == "completed":
                # Only trim leading zeros for completed cycles, keep trailing data
                start_idx = 0
                for i, point in enumerate(stored):
                    if point[1] > 1.5:
                        start_idx = i
                        break
                stored = stored[start_idx:]
                _LOGGER.debug("add_cycle: Skipping trailing trim for completed cycle")
            else:
                stored = trim_zero_power_data(stored, threshold=1.5)
            
            cycle_data["power_data"] = stored
            cycle_data["sampling_interval"] = round(sampling_interval, 1)

            # Helper to get arrays for signature (use stored data for consistency)
            ts_arr = np.array([t for t, _ in stored])
            p_arr = np.array([p for _, p in stored])

            # Compute and store signature
            if len(ts_arr) > 1 and len(ts_arr) == len(p_arr):
                sig = compute_signature(ts_arr, p_arr)
                cycle_data["signature"] = dataclasses.asdict(sig)

            _LOGGER.debug(
                "add_cycle: stored %s samples at %.1fs intervals",
                len(stored),
                sampling_interval,
            )

        # 4. Handle Debug Data (Strip if not enabled)
        if hasattr(self, "_save_debug_traces") and not self._save_debug_traces:
            if "debug_data" in cycle_data:
                del cycle_data["debug_data"]

        self._data["past_cycles"].append(cycle_data)
        # Apply retention after adding


    async def async_enforce_retention(self) -> None:
        """Apply retention policy asynchronously."""
        affected = self._enforce_retention_data()
        for p in affected:
            try:
                # Use async rebuild task
                self.hass.async_create_task(self.async_rebuild_envelope(p))
            except Exception as e: # pylint: disable=broad-exception-caught
                _LOGGER.warning("Failed to schedule envelope rebuild for %s: %s", p, e)

    def _enforce_retention_data(self) -> set[str]:
        """Internal retention logic (data operations only).
        Returns set of affected profile names."""
        raw_cycles = self._data.get("past_cycles", [])
        cycles: list[CycleDict] = (
            cast(list[CycleDict], raw_cycles) if isinstance(raw_cycles, list) else []
        )
        if not cycles:
            return set()

        def _start_time(cycle: CycleDict) -> str:
            return str(cycle.get("start_time", ""))

        affected_profiles: set[str] = set()

        # 1) Cap total cycles
        if len(cycles) > self._max_past_cycles:
            # Sort by start_time and drop oldest beyond cap
            try:
                cycles.sort(key=_start_time)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            drop_count = len(cycles) - self._max_past_cycles
            to_drop = cycles[:drop_count]

            # Maintain profile sample references when dropping
            sample_refs = {
                name: p.get("sample_cycle_id")
                for name, p in self._data.get("profiles", {}).items()
            }
            for cy in to_drop:
                # Track affected profile
                p_name = cy.get("profile_name")
                if p_name:
                    affected_profiles.add(p_name)

                cy_id = cy.get("id")
                # If a profile sample points here, try to move to most recent cycle of that profile
                for name, ref_id in list(sample_refs.items()):
                    if ref_id == cy_id:
                        # find newest cycle for that profile
                        newest = next(
                            (
                                c
                                for c in reversed(cycles)
                                if c.get("profile_name") == name and c not in to_drop
                            ),
                            None,
                        )
                        if newest:
                            self._data["profiles"][name]["sample_cycle_id"] = (
                                newest.get("id")
                            )
                        else:
                            # No replacement available
                            self._data["profiles"][name].pop("sample_cycle_id", None)
            # Actually drop
            del cycles[:drop_count]

        # 2) Strip older full traces per profile
        by_profile: dict[str | None, list[CycleDict]] = {}
        for cy in cycles:
            key_any = cy.get("profile_name")  # None for unlabeled
            key: str | None = key_any if isinstance(key_any, str) and key_any else None
            by_profile.setdefault(key, []).append(cy)

        for key, group in by_profile.items():
            # newest first based on start_time
            try:
                group.sort(key=_start_time)
            except Exception: # pylint: disable=broad-exception-caught
                pass
            # determine cap
            cap = (
                self._max_full_traces_unlabeled
                if key
                in (
                    None,
                    "",
                )
                else self._max_full_traces_per_profile
            )
            # count existing full traces
            full_indices = [i for i, c in enumerate(group) if c.get("power_data")]
            if len(full_indices) > cap:
                # preserve last 'cap' full traces (newest at end after sort), strip older ones
                keep_set = set(full_indices[-cap:])

                # Get sample cycle ID for this profile
                sample_id: str | None = None
                if key and key in self._data.get("profiles", {}):
                    sample_id = self._data["profiles"][key].get("sample_cycle_id")

                for i, c in enumerate(group):
                    if i in keep_set:
                        continue

                    # EXEMPTION: Never strip power data from the profile's sample cycle!
                    if sample_id and c.get("id") == sample_id:
                        continue

                    if c.get("power_data"):
                        c.pop("power_data", None)
                        c.pop("sampling_interval", None)
                        if key:
                            affected_profiles.add(key)

        return affected_profiles



    def cleanup_orphaned_profiles(self) -> int:
        """Remove profiles that reference non-existent cycles.
        Returns number of profiles removed."""
        cycle_ids = {c["id"] for c in self._data.get("past_cycles", [])}
        orphaned = []
        for name, profile in self._data["profiles"].items():
            ref = profile.get("sample_cycle_id")
            # Only delete if it references a non-existent cycle ID (Broken Link)
            # Creating a profile without a sample (None) is allowed (Pending State)
            if ref and ref not in cycle_ids:
                orphaned.append(name)

        for name in orphaned:
            del self._data["profiles"][name]
            _LOGGER.info(
                "Cleaned up orphaned profile '%s' (cycle no longer exists)", name
            )

        return len(orphaned)

    async def async_run_maintenance(self) -> dict[str, int]:
        """Run full maintenance: cleanup orphans, merge fragments, trim old cycles.

        Also rebuilds envelopes. Returns stats dict with counts of actions taken.
        """
        stats = {
            "orphaned_profiles": 0,
            "merged_cycles": 0,
            "split_cycles": 0,
            "rebuilt_envelopes": 0,
        }

        # 1. Clean up orphaned profiles
        stats["orphaned_profiles"] = self.cleanup_orphaned_profiles()

        # 2. Auto-Label missed cycles (retroactive matching)
        # Use overwrite=False to respect existing manual/confident labels
        label_stats = await self.auto_label_cycles(confidence_threshold=0.75, overwrite=False)
        stats["labeled_cycles"] = label_stats.get("labeled", 0)

        # 2. Smart Process History (Merge/Split/Rebuild)
        proc_stats = await self.async_smart_process_history()
        stats["merged_cycles"] = proc_stats.get("merged", 0)
        stats["split_cycles"] = proc_stats.get("split", 0)
        stats["rebuilt_envelopes"] = len(self._data.get("profiles", {})) # Approximation of rebuilt count

        # 4. Save if any changes made (smart process saves internally if needed, but explicit save safe)
        if any(stats.values()):
            await self.async_save()
            _LOGGER.info("Maintenance completed: %s", stats)

        return stats

    def _reprocess_all_data_sync(self) -> int:
        """Synchronous implementation of reprocessing logic (run in executor)."""
        cycles = self._data.get("past_cycles", [])
        if not cycles:
            return 0

        processed_count = 0

        # 1. Update Signatures & Optimize Data
        for cycle in cycles:
            # Data Optimization: Trim leading/trailing zeros (0W)
            # Only apply to compressed data to avoid breaking legacy format
            p_data = cycle.get("power_data")
            if (
                p_data 
                and isinstance(p_data, list) 
                and p_data 
                and isinstance(p_data[0], (list, tuple)) 
                and len(p_data[0]) == 2 
                and isinstance(p_data[0][0], (int, float))
            ):
                # Apply trim helper
                original_len = len(p_data)
                trimmed = trim_zero_power_data(p_data, threshold=1.0) # Conservative 1W threshold

                if trimmed and len(trimmed) < original_len:
                    # Data was trimmed - check for start time shift
                    first_offset = trimmed[0][0]
                    
                    if first_offset > 0:
                        # Leading zeros removed - Must shift start_time forward
                        try:
                            start_dt = datetime.fromisoformat(cycle["start_time"])
                            new_start = start_dt + timedelta(seconds=first_offset)
                            cycle["start_time"] = new_start.isoformat()
                            
                            # Re-normalize offsets to 0
                            shifted_data = []
                            for row in trimmed:
                                # row is [offset, power]
                                shifted_data.append([round(row[0] - first_offset, 1), row[1]])
                            cycle["power_data"] = shifted_data
                            processed_count += 1
                        except (ValueError, TypeError) as e:
                            _LOGGER.warning("Failed to shift start_time for trimmed cycle: %s", e)
                    else:
                        # Only trailing trimmed or no shift needed
                        cycle["power_data"] = trimmed
                        processed_count += 1
                    
                    # Update duration to match new data length
                    if cycle.get("power_data"):
                         cycle["duration"] = cycle["power_data"][-1][0]

            if cycle.get("power_data"):
                try:
                    tuples = decompress_power_data(cycle)
                    if tuples and len(tuples) > 10:
                        start_ts = datetime.fromisoformat(cycle["start_time"]).timestamp()
                        ts_arr = []
                        p_arr = []
                        for t_str, p in tuples:
                            t = datetime.fromisoformat(t_str).timestamp()
                            ts_arr.append(t - start_ts)
                            p_arr.append(p)

                        sig = compute_signature(np.array(ts_arr), np.array(p_arr))
                        cycle["signature"] = dataclasses.asdict(sig)
                        processed_count += 1
                except Exception as e: # pylint: disable=broad-exception-caught
                    _LOGGER.warning("Failed to reprocess signature: %s", e)

        # 2. Rebuild Envelopes


        return processed_count

    async def async_reprocess_all_data(self) -> int:
        """Reprocess all historical data to update signatures and rebuild envelopes.

        This is a non-destructive operation for raw cycle data. It:
        1. Recalculates signatures for ALL past cycles using current logic.
        2. Rebuilds all profile envelopes from scratch.
        3. Updates global stats.

        Returns total number of cycles processed.
        """
        _LOGGER.info("Starting reprocessing (offloaded)...")

        # Offload heavy synchronous work
        processed_count = await self.hass.async_add_executor_job(
            self._reprocess_all_data_sync
        )

        # 2. Rebuild Envelopes (Using new async infrastructure)
        await self.async_rebuild_all_envelopes()

        await self.async_save()

        return processed_count

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage usage stats."""
        cycles = self._data.get("past_cycles", [])
        profiles = self._data.get("profiles", {})
        debug_traces_count = sum(1 for c in cycles if c.get("debug_data"))

        file_size_kb = 0
        try:
            # Attempt to get real file size from store
            if hasattr(self._store, "path") and os.path.exists(self._store.path):
                file_size_kb = os.path.getsize(self._store.path) / 1024
            else:
                # Fallback: estimate
                file_size_kb = len(json.dumps(self._data, default=str)) / 1024
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        return {
            "file_size_kb": round(file_size_kb, 1),
            "total_cycles": len(cycles),
            "total_profiles": len(profiles),
            "debug_traces_count": debug_traces_count,
        }

    async def async_clear_debug_data(self) -> int:
        """Clear debug data from all cycles."""
        cycles = self._data.get("past_cycles", [])
        count = 0
        for cycle in cycles:
            if "debug_data" in cycle:
                del cycle["debug_data"]
                count += 1
        
        if count > 0:
            await self.async_save()
            _LOGGER.info("Cleared debug data from %s cycles", count)
            
        return count



    def _rebuild_envelope_sync(
        self, labeled_cycles: list[CycleDict]
    ) -> tuple[Any, list[float]] | None:
        """Sync worker to parse data and build envelope (run in executor)."""
        raw_cycles_data = []
        durations = []

        for cycle in labeled_cycles:
            power_data_raw = cycle.get("power_data", [])
            if not isinstance(power_data_raw, list) or len(power_data_raw) < 3:
                continue

            # Parse pairs
            pairs = []
            for item in power_data_raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        pairs.append((float(item[0]), float(item[1])))
                    except (ValueError, TypeError):
                        continue

            if len(pairs) < 3:
                continue

            offsets = [p[0] for p in pairs]
            values = [p[1] for p in pairs]

            raw_cycles_data.append((offsets, values))
            
            # Use manual duration if available (e.g. from feedback correction)
            man_dur = cycle.get("manual_duration")
            if man_dur:
                durations.append(float(man_dur))
            else:
                durations.append(offsets[-1])

        if not raw_cycles_data:
            return None

        # Run Heavy Computation
        result = analysis.compute_envelope_worker(
            raw_cycles_data,
            self.dtw_bandwidth
        )
        
        if not result:
            return None
            
        return result, durations

    async def async_rebuild_all_envelopes(self) -> int:
        """Rebuild envelopes for all profiles. Returns count of envelopes rebuilt."""
        count = 0
        for profile_name in list(self._data["profiles"].keys()):
            if await self.async_rebuild_envelope(profile_name):
                count += 1
        return count

    async def async_rebuild_envelope(self, profile_name: str) -> bool:
        """
        Build/rebuild statistical envelope for a profile asynchronously.
        Offloads heavy DTW/normalization to executor.
        """
        # 1. Gather Data (Main Thread)
        labeled_cycles = [
            c
            for c in self._data["past_cycles"]
            if c.get("profile_name") == profile_name
            and c.get("status") in ("completed", "force_stopped")
            and c.get("duration", 0) > 60
        ]

        if not labeled_cycles:
            if profile_name in self._data.get("envelopes", {}):
                del self._data["envelopes"][profile_name]
            return False

        # 2. Run Heavy Computation in Executor (Parsing + DTW)
        result_pkg = await self.hass.async_add_executor_job(
            self._rebuild_envelope_sync,
            labeled_cycles
        )

        if not result_pkg:
            if profile_name in self._data.get("envelopes", {}):
                del self._data["envelopes"][profile_name]
            return False

        result, durations = result_pkg

        # Update profile stats in storage (Fast metadata update)
        if durations and profile_name in self._data.get("profiles", {}):
            min_duration = float(np.min(durations))
            max_duration = float(np.max(durations))
            avg_duration = float(np.mean(durations))
            self._data["profiles"][profile_name]["min_duration"] = min_duration
            self._data["profiles"][profile_name]["max_duration"] = max_duration
            self._data["profiles"][profile_name]["avg_duration"] = avg_duration

        if not result:
            if profile_name in self._data.get("envelopes", {}):
                del self._data["envelopes"][profile_name]
            return False

        time_grid, min_curve, max_curve, avg_curve, std_curve, target_duration = result

        # 3. Update Storage
        # Convert to list of points [[x, y], ...]
        def to_points(y_vals: list[float]) -> list[list[float]]:
            return [[round(t, 1), round(y, 1)] for t, y in zip(time_grid, y_vals)]

        # Calculate scalar stats
        duration_std_dev = float(np.std(durations)) if durations else 0.0
        
        # Calculate Energy from Average Curve (Trapezoidal Integration)
        avg_energy = 0.0
        if len(time_grid) > 1:
            # P(W) * dt(h) = Wh
            # avg_curve is in Watts, time_grid is in Seconds
            dt_h = np.diff(time_grid) / 3600.0
            avg_p = (np.array(avg_curve[:-1]) + np.array(avg_curve[1:])) / 2.0
            avg_energy = float(np.sum(avg_p * dt_h)) / 1000.0 # Convert to kWh for display? No, config flow expects kWh?
            # Config flow line 1552: f"{envelope.get('avg_energy', 0):.2f}"
            # If line 1552 says "kwh", then we should store as kWh or Wh?
            # Config flow label says "Energy ... kWh" in table row (line 1587).
            # Let's check config flow usage again.
            # line 1552: kwh = f"{envelope.get('avg_energy', 0):.2f}"
            # line 1587: ... | {kwh} kWh | ...
            # So if we store 1.5, it displays "1.50 kWh".
            # My calculation above gives Wh. So divide by 1000.
            # avg_energy is already in kWh from line above.
            pass

        envelope_data = {
            "time_grid": time_grid,  # Time grid used by manager for phase estimation
            "target_duration": target_duration,  # Target duration for phase estimation
            "min": to_points(min_curve),
            "max": to_points(max_curve),
            "avg": to_points(avg_curve),
            "std": to_points(std_curve),
            "cycle_count": len(durations),
            "avg_energy": avg_energy,
            "duration_std_dev": duration_std_dev,
            "updated": dt_util.now().isoformat(),
        }

        if "envelopes" not in self._data:
            self._data["envelopes"] = {}
        self._data["envelopes"][profile_name] = envelope_data

        return True




    def generate_profile_svg(self, profile_name: str) -> str | None:
        """Generate an SVG string for the profile's power envelope."""
        envelope = self.get_envelope(profile_name)
        if not envelope or not envelope.get("time_grid"):
            return None

        try:
            time_grid = envelope["time_grid"]
            # Envelope curves are stored as list of [t, y] points.
            # Extract Y values for SVG generation logic.
            avg_curve = [p[1] for p in envelope["avg"]]
            min_curve = [p[1] for p in envelope["min"]]
            max_curve = [p[1] for p in envelope["max"]]

            # Canvas configuration (Scaled up 50% for High DPI)
            width, height = 1200, 450
            padding_x, padding_y = 60, 45
            graph_w = width - 2 * padding_x
            graph_h = height - 2 * padding_y

            max_time = time_grid[-1]
            # Add 5% headroom for power
            max_power = max(*max_curve, 10.0) * 1.05

            def to_x(t: float) -> float:
                return padding_x + (t / max_time) * graph_w

            def to_y(p: float) -> float:
                return height - padding_y - (p / max_power) * graph_h

            # Generate polygon points for min/max band
            # Top edge (max) forward, Bottom edge (min) backward
            points_max = []
            points_min = []
            points_avg = []

            for i, t in enumerate(time_grid):
                x = to_x(t)
                points_max.append(f"{x},{to_y(max_curve[i])}")
                points_min.append(f"{x},{to_y(min_curve[i])}")
                points_avg.append(f"{x},{to_y(avg_curve[i])}")

            # Band path: Max curve -> Reverse Min curve -> Close
            band_path = " ".join(points_max + list(reversed(points_min)))
            avg_path = " ".join(points_avg)

            # Metadata text
            avg_energy = envelope.get("avg_energy", 0)
            avg_duration = envelope.get("target_duration", 0) / 60.0
            title = f"{profile_name} ({avg_duration:.0f} min, ~{avg_energy:.2f} kWh)"

            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background-color: #1c1c1c; font-family: sans-serif;">
            <!-- Grid & Axes -->
            <rect x="0" y="0" width="{width}" height="{height}" fill="#1c1c1c" />
            <line x1="{padding_x}" y1="{height - padding_y}" x2="{width - padding_x}" y2="{height - padding_y}" stroke="#444" stroke-width="3" />
            <line x1="{padding_x}" y1="{padding_y}" x2="{padding_x}" y2="{height - padding_y}" stroke="#444" stroke-width="3" />

            <!-- Axis Labels -->
            <text x="{padding_x}" y="{padding_y - 15}" fill="#aaa" font-size="18">{int(max_power)}W</text>
            <text x="{width - padding_x}" y="{height - 10}" fill="#aaa" font-size="18" text-anchor="middle">{int(max_time / 60)}m</text>
            <text x="{width / 2}" y="{padding_y - 15}" fill="#fff" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>

            <!-- Envelope Band (Min/Max) -->
            <polygon points="{band_path}" fill="#3498db" fill-opacity="0.3" stroke="none" />

            <!-- Average Line -->
            <polyline points="{avg_path}" fill="none" stroke="#3498db" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
            </svg>"""

            return svg

        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.error("Error generating SVG for %s: %s", profile_name, e)
            return None



    def generate_profile_spaghetti_svg(
        self, profile_name: str
    ) -> tuple[str | None, dict[str, str]]:
        """
        Generate a 'Spaghetti Plot' SVG showing ALL individual cycles for a profile.
        Returns (svg_string, cycle_metadata_map).
        """
        # Get ALL completed cycles labeled with this profile
        labeled_cycles = [
            c
            for c in self._data["past_cycles"]
            if c.get("profile_name") == profile_name
            and c.get("status") in ("completed", "force_stopped")
        ]

        if not labeled_cycles:
            return None, {}

        # Sort by date
        labeled_cycles.sort(key=lambda x: x["start_time"])

        palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
            "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
        ]

        cycle_metadata: dict[str, str] = {}
        svg_curves: list[SVGCurve] = []

        for i, cycle in enumerate(labeled_cycles):
            power_data_raw = cycle.get("power_data", [])
            cid = cycle["id"]

            # Decompress
            pairs: list[tuple[float, float]] = []
            if isinstance(power_data_raw, list):
                for item in power_data_raw:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            pairs.append((float(item[0]), float(item[1])))
                        except (ValueError, TypeError):
                            continue

            if len(pairs) < 3:
                continue

            offsets = [p[0] for p in pairs]
            values = [p[1] for p in pairs]

            if not offsets:
                continue

            # Assign color
            color = palette[i % len(palette)]
            cycle_metadata[cid] = color

            # Subsample for rendering performance
            step = max(1, len(pairs) // 500)
            subsampled_points = [(offsets[j], values[j]) for j in range(0, len(pairs), step)]

            svg_curves.append(SVGCurve(
                points=subsampled_points,
                color=color,
                opacity=0.8,
                stroke_width=2
            ))

        if not svg_curves:
            return None, {}

        svg_content = _generate_generic_svg(
            title=f"{profile_name} (Overview)",
            curves=svg_curves,
            width=1000,
            height=400
        )

        return svg_content, cycle_metadata

    def generate_preview_svg(
        self, power_data: list[tuple[str, float]], head_trim: float, tail_trim: float
    ) -> str:
        """
        Generate a preview SVG for a recorded cycle, highlighting trimmed areas.
        Blue = Keep, Red = Trim.
        """
        if not power_data:
            return ""

        # Parse data
        points: list[tuple[float, float]] = []
        try:
            start_ts = dt_util.parse_datetime(power_data[0][0]).timestamp()
            for t_str, p in power_data:
                t = dt_util.parse_datetime(t_str).timestamp() - start_ts
                points.append((t, float(p)))
        except (ValueError, TypeError, IndexError):
            return ""

        if not points:
            return ""

        total_duration = points[-1][0]

        keep_start = head_trim
        keep_end = max(keep_start, total_duration - tail_trim)

        # Prepare curves
        curves: list[SVGCurve] = []

        # 1. Background (All Red)
        curves.append(SVGCurve(
            points=points,
            color="#e6194b",
            opacity=0.5,
            stroke_width=2
        ))

        # 2. Keep (Blue)
        keep_points = [pt for pt in points if keep_start <= pt[0] <= keep_end]
        if keep_points:
            curves.append(SVGCurve(
                points=keep_points,
                color="#4363d8",
                opacity=1.0,
                stroke_width=2
            ))

        # Markers
        markers = [
            {"x": keep_start, "label": "Trim Start", "color": "#e6194b"},
            {"x": keep_end, "label": "Trim End", "color": "#e6194b"},
        ]

        return _generate_generic_svg(
            title="Recording Preview",
            curves=curves,
            width=800,
            height=400,
            markers=markers
        )

    def get_envelope(self, profile_name: str) -> JSONDict | None:
        """Get cached envelope for a profile, or None if not available."""
        envelopes = self._data.get("envelopes", {})
        if isinstance(envelopes, dict):
            envelopes_map = cast(dict[str, Any], envelopes)
            env = envelopes_map.get(profile_name)
            return cast(JSONDict, env) if isinstance(env, dict) else None
        return None

    def _get_cached_sample_segment(
        self, sample_cycle: dict[str, Any], dt: float
    ) -> Segment | None:
        """Get or compute resampled segment for a sample cycle, using cache."""
        cycle_id = sample_cycle.get("id")
        if not cycle_id:
            return None

        # Round dt to avoid float cache misses
        dt_key = float(round(dt, 2))
        key = (cycle_id, dt_key)

        if key in self._cached_sample_segments:
            return self._cached_sample_segments[key]

        # Miss: Compute
        sample_data = sample_cycle.get("power_data")
        if not sample_data:
            return None

        try:
            if len(sample_data) > 0 and isinstance(sample_data[0], (list, tuple)):
                s_ts = np.array([x[0] for x in sample_data])
                s_p = np.array([x[1] for x in sample_data])
            else:
                return None

            s_segments = resample_uniform(s_ts, s_p, dt_s=dt, gap_s=300.0)
            if not s_segments:
                return None

            sample_seg = max(s_segments, key=lambda s: len(s.power))

            # Store
            self._cached_sample_segments[key] = sample_seg
            return sample_seg
        except Exception as e: # pylint: disable=broad-exception-caught
            _LOGGER.warning("Error caching sample segment %s: %s", cycle_id, e)
            return None

    async def async_match_profile(
        self,
        current_power_data: list[tuple[str, float]] | list[tuple[datetime, float]],
        current_duration: float,
    ) -> MatchResult:
        """Run profile matching asynchronously in executor."""
        # 1. Prepare data in main thread (Access ProfileStore state safely)

        # Convert to list of floats for current power (uniform resampling)
        if not current_power_data:
            return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

        # Pre-process current data
        try:
            # Normalize input format
            if isinstance(current_power_data[0][0], datetime):
                t_start = cast(datetime, current_power_data[0][0]).timestamp()
                ts_arr = np.array([(cast(datetime, x[0]).timestamp() - t_start) for x in current_power_data])
            else:
                t_start = datetime.fromisoformat(cast(str, current_power_data[0][0])).timestamp()
                ts_arr = np.array([(datetime.fromisoformat(cast(str, x[0])).timestamp() - t_start) for x in current_power_data])

            p_arr = np.array([float(x[1]) for x in current_power_data])

            # Resample current
            segments, used_dt = resample_adaptive(ts_arr, p_arr, min_dt=5.0, gap_s=300.0)
            if not segments:
                return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)
            current_seg = max(segments, key=lambda s: len(s.power))
            if len(current_seg.power) < 12:
                return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

            current_power_list = current_seg.power.tolist()

            # Prepare Snapshots
            snapshots = []
            skipped_profiles = []
            for name, profile in self._data["profiles"].items():
                # Try sample_cycle_id first, fall back to any labeled cycle
                sample_id = profile.get("sample_cycle_id")
                sample_cycle = None
                if sample_id:
                    sample_cycle = next(
                        (c for c in self._data["past_cycles"] if c["id"] == sample_id),
                        None
                    )
                # Fallback: find ANY completed cycle labeled with this profile
                if not sample_cycle:
                    sample_cycle = next(
                        (c for c in self._data["past_cycles"]
                          if c.get("profile_name") == name
                          and c.get("status") in ("completed", "force_stopped")
                          and c.get("power_data")),
                        None
                    )
                if not sample_cycle:
                    skipped_profiles.append(
                        f"{name}: no sample cycle (sample_id={sample_id})"
                    )
                    continue

                # Prepare sample segment (using cache)
                sample_seg = self._get_cached_sample_segment(sample_cycle, used_dt)
                if not sample_seg:
                    skipped_profiles.append(
                        f"{name}: failed to resample cycle {sample_cycle.get('id')}"
                    )
                    continue
                snapshots.append({
                    "name": name,
                    "avg_duration": profile.get(
                        "avg_duration", sample_cycle.get("duration", 0)
                    ),
                    "sample_power": sample_seg.power.tolist(),
                    "sample_dt": used_dt
                })

            if skipped_profiles:
                _LOGGER.debug(
                    "Profile matching skipped %d profiles: %s",
                    len(skipped_profiles),
                    "; ".join(skipped_profiles)
                )

            config = {
                "min_duration_ratio": self._min_duration_ratio,
                "max_duration_ratio": self._max_duration_ratio,
                "dtw_bandwidth": self.dtw_bandwidth
            }

        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.error("Preparation for async match failed: %s", e)
            return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

        # 2. Run Heavy Logic in Executor
        candidates = await self.hass.async_add_executor_job(
            analysis.compute_matches_worker,
            current_power_list,
            current_duration,
            snapshots,
            config
        )

        # 3. Process Result (Main Thread)
        if not candidates:
            profiles_count = len(self._data.get("profiles", {}))
            snapshots_count = len(snapshots) if 'snapshots' in dir() else 0
            _LOGGER.debug(
                "No profile match candidates: profiles=%d, snapshots=%d, "
                "duration=%.0fs. Possible reasons: duration ratio filter, "
                "no labeled cycles, or no profiles defined.",
                profiles_count,
                snapshots_count,
                current_duration
            )
            return MatchResult(None, 0.0, 0.0, None, [], False, 0.0, [], {}, is_confident_mismatch=True, mismatch_reason="all_rejected")

        best = candidates[0]

        # Reconstruct MatchResult
        # Need to handle margin/ambiguity
        margin = 1.0
        if len(candidates) > 1:
            margin = best["score"] - candidates[1]["score"]

        is_ambiguous = margin < 0.05

        # Phase Detection (Sync on main thread, fast enough? Phase check is O(N) but simple bounds check)
        # We can run check_phase_match logic here or defer it.
        # Let's run it here since we have the data.
        # But check_phase_match uses wrappers.
        matched_phase = None
        if best["score"] > 0.6: # Threshold
            # We need logic from check_phase_match but customized
            matched_phase = self.check_phase_match(best["name"], current_duration)

        return MatchResult(
            best["name"],
            best["score"],
            best["profile_duration"],
            matched_phase,
            candidates[:5], # Ranking
            is_ambiguous,
            margin,
            # Extra fields...
        )

    def match_profile(
        self, power_data: list[tuple[str, float]], duration: float
    ) -> MatchResult:
        """Synchronous wrapper for matching (for use in executor tasks)."""
        # Convert to list for worker
        p_list = [p[1] for p in power_data]

        # Prepare snapshots safely
        snapshots = []
        # Accessing self._data in thread is generally safe for reads if not modifying
        for name, profile in self._data["profiles"].items():
            sample_id = profile.get("sample_cycle_id")
            sample_cycle = next((c for c in self._data["past_cycles"] if c["id"] == sample_id), None)
            if not sample_cycle:
                continue

            # Decompress sample data
            sample_p_data = self._decompress_power_data(sample_cycle)
            if not sample_p_data:
                continue

            snapshots.append({
                "name": name,
                "avg_duration": profile.get("avg_duration", sample_cycle.get("duration", 0)),
                "sample_power": [x[1] for x in sample_p_data],
            })

        config = {
            "min_duration_ratio": self._min_duration_ratio,
            "max_duration_ratio": self._max_duration_ratio,
            "dtw_bandwidth": self.dtw_bandwidth
        }

        candidates = analysis.compute_matches_worker(
            p_list, duration, snapshots, config
        )

        if not candidates:
            return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

        best = candidates[0]

        # Calculate ambiguity
        margin = 1.0
        if len(candidates) > 1:
            margin = best["score"] - candidates[1]["score"]

        is_ambiguous = margin < 0.05

        return MatchResult(
            best["name"],
            best["score"],
            best["profile_duration"],
            None,
            candidates,
            is_ambiguous,
            margin,
            ranking=candidates,
        )

    async def async_verify_alignment(
        self, profile_name: str, current_power_data: list[tuple[str, float]]
    ) -> tuple[bool, float, float]:
        """
        Verify if the current power trace aligns with an expected low-power region in the envelope.
        Returns: (is_confirmed_low_power, mapped_envelope_time, mapped_envelope_power)
        """
        envelope = self.get_envelope(profile_name)
        if not envelope or not envelope.get("avg") or not current_power_data:
            return False, 0.0, 9999.0

        # Extract envelope curves
        # "avg" is list of [t, p]
        env_avg = envelope["avg"]
        env_time = [p[0] for p in env_avg]
        env_power = [p[1] for p in env_avg]
        
        # Prepare current power (floats)
        try:
             # Normalize input format
            if isinstance(current_power_data[0][0], datetime):
                # Should not happen if passed from detector readings directly but handled
                current_power_list = [float(x[1]) for x in current_power_data]
            else:
                current_power_list = [float(x[1]) for x in current_power_data]
        except Exception:  # pylint: disable=broad-exception-caught
            return False, 0.0, 9999.0

        # Offload to worker
        mapped_time, mapped_power, score = await self.hass.async_add_executor_job(
            analysis.verify_profile_alignment_worker,
            current_power_list,
            env_power,
            env_time,
            self.dtw_bandwidth
        )
        
        # Verify if mapped power and alignment score indicate an expected low-power region.
        # Thresholds: Expected power < 15W, Alignment score > 0.4
        is_confirmed = (mapped_power < 15.0) and (score > 0.4)
        
        return is_confirmed, mapped_time, mapped_power


    # match_profile (sync) removed in favor of async_match_profile

    def check_phase_match(self, profile_name: str, duration: float) -> str | None:
        """
        Check if the current duration aligns with a known phase in the profile.
        Returns the phase name (e.g., 'Rinse', 'Spin') or None.
        """
        profile = self._data["profiles"].get(profile_name)
        if not profile:
            return None

        phases = profile.get("phases", [])
        if not phases:
            return None

        for phase in phases:
            p_start = phase.get("start", 0)
            p_end = phase.get("end", 0)
            if p_start <= duration <= p_end:
                return str(phase.get("name", "Unknown"))

        return None



    async def create_profile(self, name: str, source_cycle_id: str) -> None:
        """Create a new profile from a past cycle."""
        cycle = next(
            (c for c in self._data["past_cycles"] if c["id"] == source_cycle_id), None
        )
        if not cycle:
            raise ValueError("Cycle not found")

        cycle["profile_name"] = name

        self._data.setdefault("profiles", {})[name] = {
            "avg_duration": cycle["duration"],
            "sample_cycle_id": source_cycle_id,
        }

        # Save to persist the label
        await self.async_save()

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all profiles with metadata."""
        profiles: list[JSONDict] = []
        raw_profiles = self._data.get("profiles", {})
        profiles_map = (
            cast(dict[str, Any], raw_profiles) if isinstance(raw_profiles, dict) else {}
        )
        for name, data in profiles_map.items():
            profile_meta = cast(JSONDict, data) if isinstance(data, dict) else {}
            
            # Calculate count and last_run
            p_cycles = [
                c for c in self._data.get("past_cycles", []) 
                if c.get("profile_name") == name
            ]
            cycle_count = len(p_cycles)
            
            last_run = None
            if p_cycles:
                last_c = max(p_cycles, key=lambda x: x.get("start_time", ""))
                last_run = last_c.get("start_time")

            # Fetch envelope stats
            envelope = self.get_envelope(name)
            avg_energy = envelope.get("avg_energy") if envelope else None

            profiles.append(
                {
                    "name": name,
                    "avg_duration": profile_meta.get("avg_duration", 0),
                    "min_duration": profile_meta.get("min_duration", 0),
                    "max_duration": profile_meta.get("max_duration", 0),
                    "sample_cycle_id": profile_meta.get("sample_cycle_id"),
                    "cycle_count": cycle_count,
                    "last_run": last_run,
                    "avg_energy": avg_energy,
                }
            )
        return sorted(profiles, key=lambda p: profile_sort_key(p.get("name", "")))

    async def create_profile_standalone(
        self,
        name: str,
        reference_cycle_id: str | None = None,
        avg_duration: float | None = None,
    ) -> None:
        """Create a profile without immediately labeling a cycle.
        If reference_cycle_id is provided, use that cycle's characteristics.
        If avg_duration is provided (and no reference cycle), use it as baseline."""
        if name in self._data.get("profiles", {}):
            raise ValueError(f"Profile '{name}' already exists")

        profile_data: JSONDict = {}
        if reference_cycle_id:
            cycle = next(
                (c for c in self._data["past_cycles"] if c["id"] == reference_cycle_id),
                None,
            )
            if cycle:
                profile_data = {
                    "avg_duration": cycle["duration"],
                    "sample_cycle_id": reference_cycle_id,
                }
        elif avg_duration is not None and avg_duration > 0:
            profile_data = {
                "avg_duration": float(avg_duration),
            }

        # Create profile with minimal data (will be updated when cycles are labeled)
        self._data.setdefault("profiles", {})[name] = profile_data
        await self.async_save()
        _LOGGER.info("Created standalone profile '%s'", name)

    async def update_profile(
        self, old_name: str, new_name: str, avg_duration: float | None = None
    ) -> int:
        """Update a profile's name and/or average duration.
        Returns number of cycles updated (if renamed)."""
        profiles = self._data.get("profiles", {})
        if old_name not in profiles:
            raise ValueError(f"Profile '{old_name}' not found")

        # Handle Rename
        renamed = False
        if new_name != old_name:
            if new_name in profiles:
                raise ValueError(f"Profile '{new_name}' already exists")

            # Rename in profiles dict
            profiles[new_name] = profiles.pop(old_name)

            # Rename in envelopes
            if "envelopes" in self._data and old_name in self._data["envelopes"]:
                self._data["envelopes"][new_name] = self._data["envelopes"].pop(
                    old_name
                )

            renamed = True

        target_name = new_name if renamed else old_name

        # Handle Duration Update
        if avg_duration is not None and avg_duration > 0:
            profiles[target_name]["avg_duration"] = float(avg_duration)
            # If there's an envelope, we ideally update its target_duration too,
            # but envelope is usually rebuilt from data.
            # However, for manual profiles, envelope might be empty or theoretical.
            # Let's log it.
            _LOGGER.info(
                "Updated baseline duration for '%s' to %ss",
                target_name,
                avg_duration,
            )

        # Update cycles if renamed
        count = 0
        if renamed:
            for cycle in self._data.get("past_cycles", []):
                if cycle.get("profile_name") == old_name:
                    cycle["profile_name"] = new_name
                    count += 1
            _LOGGER.info(
                "Renamed profile '%s' to '%s', updated %s cycles",
                old_name,
                new_name,
                count,
            )

        await self.async_save()
        return count

    async def delete_profile(self, name: str, unlabel_cycles: bool = True) -> int:
        """Delete a profile.
        If unlabel_cycles=True, removes profile label from cycles.
        If unlabel_cycles=False, cycles keep the label (orphaned).
        Returns number of cycles affected."""
        if name not in self._data.get("profiles", {}):
            raise ValueError(f"Profile '{name}' not found")

        # Delete profile
        del self._data["profiles"][name]

        # Handle cycles
        count = 0
        for cycle in self._data.get("past_cycles", []):
            if cycle.get("profile_name") == name:
                if unlabel_cycles:
                    cycle["profile_name"] = None
                count += 1

        await self.async_save()
        action = "unlabeled" if unlabel_cycles else "orphaned"
        _LOGGER.info("Deleted profile '%s', %s %s cycles", name, action, count)
        return count

    async def clear_all_data(self) -> None:
        """Clear all profiles and cycle data."""
        self._data["past_cycles"] = []
        self._data["profiles"] = {}
        await self.async_save()
        _LOGGER.info("Cleared all WashData storage")

    async def assign_profile_to_cycle(
        self, cycle_id: str, profile_name: str | None
    ) -> None:
        """Assign an existing profile to a cycle. Rebuilds envelope."""
        old_profile = None
        cycle = next(
            (c for c in self._data["past_cycles"] if c["id"] == cycle_id), None
        )
        if not cycle:
            raise ValueError(f"Cycle {cycle_id} not found")

        # Track old profile for envelope rebuild
        old_profile = cycle.get("profile_name")

        if profile_name and profile_name not in self._data.get("profiles", {}):
            raise ValueError(f"Profile '{profile_name}' not found. Create it first.")

        # Update cycle
        cycle["profile_name"] = profile_name if profile_name else None

        # Update profile metadata if this is the first cycle
        if profile_name:
            profile = self._data["profiles"][profile_name]
            if not profile.get("sample_cycle_id"):
                profile["sample_cycle_id"] = cycle_id
                profile["avg_duration"] = cycle["duration"]

        # Rebuild envelopes for affected profiles
        if old_profile and old_profile != profile_name:
            await self.async_rebuild_envelope(old_profile)  # Old profile lost a cycle
        if profile_name:
            await self.async_rebuild_envelope(profile_name)  # New profile gained a cycle
            # Apply retention after labeling, in case profile now exceeds cap
            await self.async_enforce_retention()

        await self.async_save()
        _LOGGER.info("Assigned profile '%s' to cycle %s", profile_name, cycle_id)
        # Trigger smart processing to potentially merge now-labeled cycle
        await self.async_smart_process_history()

    async def auto_label_cycles(
        self, confidence_threshold: float = 0.75, overwrite: bool = False
    ) -> dict[str, int]:
        """Auto-label cycles retroactively using profile matching.

        Args:
            confidence_threshold: Min confidence to apply a label.
            overwrite: If True, re-evaluates already labeled cycles.

        Returns stats: {labeled: int, relabeled: int, skipped: int, total: int}
        """
        stats = {"labeled": 0, "relabeled": 0, "skipped": 0, "total": 0}

        cycles = self._data.get("past_cycles", [])

        # Filter down if not overwriting
        if not overwrite:
            target_cycles = [c for c in cycles if not c.get("profile_name")]
        else:
            target_cycles = cycles

        stats["total"] = len(target_cycles)

        for cycle in target_cycles:
            # Reconstruct power data for matching
            power_data = self._decompress_power_data(cycle)
            if not power_data or len(power_data) < 10:
                stats["skipped"] += 1
                continue

            # Try to match
            result = await self.async_match_profile(power_data, cycle["duration"])

            if result.best_profile and result.confidence >= confidence_threshold:
                current_label = cycle.get("profile_name")

                # If overwriting, check if new match is different and better/valid
                if current_label:
                    if current_label != result.best_profile:
                        cycle["profile_name"] = result.best_profile
                        stats["relabeled"] += 1
                        _LOGGER.info(
                            "Relabeled cycle %s: '%s' -> '%s' (confidence: %.2f)",
                            cycle["id"],
                            current_label,
                            result.best_profile,
                            result.confidence,
                        )
                else:
                    cycle["profile_name"] = result.best_profile
                    stats["labeled"] += 1
                    _LOGGER.info(
                        "Auto-labeled cycle %s as '%s' (confidence: %.2f)",
                        cycle["id"],
                        result.best_profile,
                        result.confidence,
                    )
            else:
                stats["skipped"] += 1

        if stats["labeled"] > 0 or stats["relabeled"] > 0:
            await self.async_save()
            # Trigger smart processing after bulk labeling
            await self.async_smart_process_history()

        _LOGGER.info(
            "Auto-labeling complete: %s labeled, %s relabeled, %s skipped",
            stats["labeled"],
            stats["relabeled"],
            stats["skipped"],
        )
        return stats

    def _decompress_power_data(self, cycle: CycleDict) -> list[tuple[str, float]]:
        """Decompress cycle power data for matching (wrapper)."""
        return decompress_power_data(cycle)

    async def async_save_cycle(self, cycle_data: dict[str, Any]) -> None:
        """Add and save a cycle. Rebuilds envelope if cycle is labeled."""
        self.add_cycle(cycle_data)

        # If cycle has a profile, rebuild that profile's envelope
        profile_name = cycle_data.get("profile_name")
        if profile_name:
            await self.async_rebuild_envelope(profile_name)

        await self.async_save()
        # Trigger smart processing on new cycle
        await self.async_smart_process_history()

    async def async_migrate_cycles_to_compressed(self) -> int:
        """
        Migrate all cycles to the compressed format.
        Ensures all cycles use [offset_seconds, power] format.
        Returns number of cycles migrated.
        """
        raw_cycles = self._data.get("past_cycles", [])
        cycles: list[CycleDict] = (
            cast(list[CycleDict], raw_cycles) if isinstance(raw_cycles, list) else []
        )
        migrated = 0

        for cycle in cycles:
            raw_data: list[Any] = cycle.get("power_data", []) or []
            if not raw_data:
                continue

            # Check if already compressed (first element is number or mixed format)
            first_elem = raw_data[0][0]
            if isinstance(first_elem, (int, float)):
                # Already in offset format
                continue

            # Old format: ISO timestamp strings. Convert to compressed offsets.
            try:
                compressed = compress_power_data(cycle)
                if compressed:
                    cycle["power_data"] = compressed
                    migrated += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                _LOGGER.warning("Failed to migrate cycle %s: %s", cycle.get("id"), e)
                continue

        if migrated > 0:
            _LOGGER.info("Migrated %s cycles to compressed format", migrated)
            await self.async_save()

        return migrated



    async def async_split_cycles_smart(
        self, cycle_id: str, min_gap_s: int = 900, idle_power: float = 2.0
    ) -> list[str]:
        """Scan a cycle for significant idle gaps and split if parts match better (offloaded)."""
        cycles = cast(list[CycleDict], self._data.get("past_cycles", []))
        idx = next((i for i, c in enumerate(cycles) if c.get("id") == cycle_id), -1)

        if idx == -1:
            return []

        cycle = cycles[idx]

        # Offload analysis
        seg_ranges = await self.hass.async_add_executor_job(
            self.analyze_split_sync, cycle, min_gap_s, idle_power
        )

        if not seg_ranges:
            return [cycle_id]

        # Apply Split (Main Thread)
        cycles.pop(idx)
        new_ids = []
        original_profile = cycle.get("profile_name")
        start_dt_base_parsed = dt_util.parse_datetime(cycle["start_time"])
        if not start_dt_base_parsed:
            # Should not happen as analyze checked it, but safety
            _LOGGER.warning("Failed to parse start time during split apply for %s", cycle_id)
            return [cycle_id]

        start_dt_base: datetime = start_dt_base_parsed
        start_ts = start_dt_base.timestamp()

        # Use decompress_power_data which handles all format variations
        p_data_tuples = self._decompress_power_data(cycle)

        if not p_data_tuples:
            _LOGGER.warning("Failed to decompress data during split for %s", cycle_id)
            return [cycle_id]

        # Convert to relative seconds for array logic
        # decompress_power_data returns list[tuple[str, float]] (ISO strings)
        # We need relative seconds for the `seg_ranges` which are inputs in seconds.

        points = []
        for t_str, val in p_data_tuples:
            dt_p = dt_util.parse_datetime(t_str)
            if dt_p:
                rel = dt_p.timestamp() - start_ts
                points.append((rel, float(val)))

        for seg_start, seg_end in seg_ranges:
            # Construct new cycle logic
            seg_dur = seg_end - seg_start
            new_cycle_start = start_dt_base + timedelta(seconds=seg_start)
            new_cycle_start_ts = new_cycle_start.timestamp()

            # Extract points
            p_data_abs = []
            state_val = 0.0
            for t, p in points:
                if t <= seg_start:
                    state_val = p
                else:
                    break
            p_data_abs.append([round(new_cycle_start_ts, 1), state_val])

            for t, p in points:
                if seg_start < t <= seg_end:
                    if start_dt_base:
                        p_data_abs.append([round(start_dt_base.timestamp() + t, 1), p])

            new_cycle = {
                "start_time": new_cycle_start.isoformat(),
                "end_time": (new_cycle_start + timedelta(seconds=seg_dur)).isoformat(),
               "duration": round(seg_dur, 1),
               "status": "completed",
               "power_data": p_data_abs,
               "profile_name": None
            }
            self.add_cycle(new_cycle)
            new_ids.append(new_cycle["id"])

        # Fix profile refs (same as original logic)
        original_sample_id = cycle.get("id")
        best_replacement_id = None
        longest_dur = 0
        new_cycles_objs = [c for c in cycles if c["id"] in new_ids]

        for c in new_cycles_objs:
            d = c.get("duration", 0)
            if d > longest_dur:
                longest_dur = d
                best_replacement_id = c["id"]

        if best_replacement_id and original_profile:
            p_data = self._data["profiles"].get(original_profile)
            if p_data and p_data.get("sample_cycle_id") == original_sample_id:
                p_data["sample_cycle_id"] = best_replacement_id
            
            # Rebuild envelope because dataset changed
            await self.async_rebuild_envelope(original_profile)

        await self.async_save()
        return new_ids

    async def async_smart_process_history(
        self
    ) -> dict[str, int]:
        # Orchestrate smart history processing: Cleanup, Retention.
        # Split/Merge is now manual via Interactive Editor.
        stats = {"cleaned_profiles": 0}

        # 1. Cleanup
        _LOGGER.debug("Running maintenance: cleanup_orphaned_profiles")
        stats["cleaned_profiles"] = self.cleanup_orphaned_profiles()

        # 2. Retention
        _LOGGER.debug("Running maintenance: async_enforce_retention")
        await self.async_enforce_retention()

        # 3. Save
        _LOGGER.debug("Maintenance complete, saving")
        await self.async_save()

        return stats



    def log_adjustment(
        self, setting_name: str, old_value: Any, new_value: Any, reason: str
    ) -> None:
        # Log an automatic adjustment to a setting.
        if old_value == new_value:
            return
        adjustment: JSONDict = {
            "timestamp": dt_util.now().isoformat(),
            "setting": setting_name,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }
        self._data.setdefault("auto_adjustments", []).append(adjustment)
        # Keep last 50 adjustments
        if len(self._data["auto_adjustments"]) > 50:
            self._data["auto_adjustments"] = self._data["auto_adjustments"][-50:]
        _LOGGER.info(
            "Auto-adjustment: %s changed from %s to %s (%s)",
            setting_name,
            old_value,
            new_value,
            reason,
        )

    def export_data(
        self, entry_data: JSONDict | None = None, entry_options: JSONDict | None = None
    ) -> JSONDict:
        # Return a serializable snapshot of the store for backup/export.
        # Includes config entry data/options so users can transfer fine-tuned settings.
        return {
            "version": STORAGE_VERSION,
            "entry_id": self.entry_id,
            "exported_at": dt_util.now().isoformat(),
            "data": self._data,
            "entry_data": entry_data or {},
            "entry_options": entry_options or {},
        }

    async def async_import_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Import data from JSON payload (migration aware).
        version = payload.get("version", 1)

        # Handle v1 format (flat structure) - convert to v2
        if version == 1 or "data" not in payload:
            # V1 format had profiles/past_cycles at top level
            data_dict = {
                "profiles": payload.get("profiles", {}),
                "past_cycles": payload.get("past_cycles", []),
                "envelopes": payload.get("envelopes", {}),
            }
            _LOGGER.info(
                "Importing v1 format: %s cycles", len(data_dict.get("past_cycles", []))
            )
        else:
            # V2 format with nested "data" key
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError(
                    "Invalid export payload (missing or invalid 'data' key)"
                )
            data_dict = cast(JSONDict, data)
            _LOGGER.info(
                "Importing v2 format: %s cycles", len(data_dict.get("past_cycles", []))
            )

        # Validate and repair structure
        if not isinstance(data_dict.get("profiles"), dict):
            data_dict["profiles"] = {}
        if not isinstance(data_dict.get("past_cycles"), list):
            data_dict["past_cycles"] = []
        data_dict.setdefault("envelopes", {})

        self._data = data_dict
        await self.async_save()


    async def delete_cycle(self, cycle_id: str) -> bool:
        """Delete a cycle by ID."""
        cycles = cast(list[CycleDict], self._data.get("past_cycles", []))
        initial_len = len(cycles)
        cycle_to_delete = next((c for c in cycles if c.get("id") == cycle_id), None)
        if not cycle_to_delete:
            return False
            
        profile_name = cycle_to_delete.get("profile_name")
        self._data["past_cycles"] = [c for c in cycles if c.get("id") != cycle_id]
        
        if len(self._data["past_cycles"]) < initial_len:
            # Check profile references
            for _p_name, p_data in self.get_profiles().items():
                if p_data.get("sample_cycle_id") == cycle_id:
                    p_data["sample_cycle_id"] = None
            
            # Rebuild envelope if cycle belonged to a profile
            if profile_name:
                await self.async_rebuild_envelope(profile_name)

            await self.async_save()
            return True
        return False

    def analyze_split_sync(
        self, cycle: CycleDict, min_gap_s: int = 900, idle_power: float = 2.0
    ) -> list[tuple[float, float]]:
        """Analyze cycle for potential splits (sync for executor)."""
        p_data = self._decompress_power_data(cycle)
        if not p_data:
            return []
            
        start_ts = dt_util.parse_datetime(cycle["start_time"]).timestamp()
        
        # Parse all points to (rel_t, power)
        points = []
        for t_str, val in p_data:
            dt_p = dt_util.parse_datetime(t_str)
            if dt_p:
                points.append((dt_p.timestamp() - start_ts, float(val)))
               
        if not points:
            return []
            
        valid_segments = []
        seg_start = 0.0
        
        for i in range(1, len(points)):
            t, _ = points[i]
            prev_t, prev_p = points[i-1]
            
            # Detect idle gap
            gap = t - prev_t
            if prev_p < idle_power and gap > min_gap_s:
                # Segment ending at prev_t
                if (prev_t - seg_start) > 60:
                    valid_segments.append((seg_start, prev_t))
                seg_start = t

        # Last segment
        last_t = points[-1][0]
        if (last_t - seg_start) > 60:
            valid_segments.append((seg_start, last_t))
             
        if len(valid_segments) < 2:
            return []
            
        _LOGGER.debug(
            "Analyzed split for %s: found %d segments",
            cycle.get("id"),
            len(valid_segments)
        )
        return valid_segments

    async def apply_split_interactive(
        self, cycle_id: str, segments: list[dict[str, Any]]
    ) -> list[str]:
        """Apply a manual split config.
        
        segments format: [{"start": float, "end": float, "profile": str|None}]
        Returns list of new cycle IDs.
        """
        cycles = cast(list[CycleDict], self._data.get("past_cycles", []))
        idx = next((i for i, c in enumerate(cycles) if c.get("id") == cycle_id), -1)

        if idx == -1:
            return []

        cycle = cycles[idx]
        cycles.pop(idx) # Remove original
        
        new_ids = []
        original_profile = cycle.get("profile_name")
        start_dt_base_parsed = dt_util.parse_datetime(cycle["start_time"])
        if not start_dt_base_parsed:
            return []
            
        start_ts = start_dt_base_parsed.timestamp()

        # Decompress original data
        p_data_tuples = self._decompress_power_data(cycle)
        if not p_data_tuples:
            return []

        # Prepare points (relative seconds)
        points = []
        for t_str, val in p_data_tuples:
            dt_p = dt_util.parse_datetime(t_str)
            if dt_p:
                rel = dt_p.timestamp() - start_ts
                points.append((rel, float(val)))

        # Create new cycles
        for seg in segments:
            if isinstance(seg, (list, tuple)):
                seg_start = float(seg[0])
                seg_end = float(seg[1])
                seg_profile = None
            else:
                seg_start = float(seg["start"])
                seg_end = float(seg["end"])
                seg_profile = seg.get("profile")
            
            seg_dur = seg_end - seg_start
            new_cycle_start = start_dt_base_parsed + timedelta(seconds=seg_start)
            new_cycle_start_ts = new_cycle_start.timestamp()

            # Extract points for this segment
            p_data_abs = []
            
            # Find closest state before/at start to ensure continuity?
            # Or just take points strictly inside?
            # Generally better to capture the state at start 0.
            state_val = 0.0
            for t, p in points:
                if t <= seg_start:
                    state_val = p
                else:
                    break
            
            # Start point (t=0 relative to new cycle)
            p_data_abs.append([round(new_cycle_start_ts, 1), state_val])

            for t, p in points:
                if seg_start < t <= seg_end:
                    p_data_abs.append([round(start_ts + t, 1), p])
            
            # Create Cycle Record
            new_cycle = {
                "start_time": new_cycle_start.isoformat(),
                "end_time": (new_cycle_start + timedelta(seconds=seg_dur)).isoformat(),
                "duration": round(seg_dur, 1),
                "status": "completed",
                "power_data": p_data_abs,
                "profile_name": seg_profile
            }
            self.add_cycle(new_cycle)
            new_ids.append(new_cycle["id"])

        # Fix profile refs (handle original sample cycle logic)
        original_sample_id = cycle.get("id")
        best_replacement_id = None
        longest_dur = 0
        new_cycles_objs = [c for c in cycles if c["id"] in new_ids] # 'cycles' is mutated by add_cycle

        for c in new_cycles_objs:
            d = c.get("duration", 0)
            if d > longest_dur:
                longest_dur = d
                best_replacement_id = c["id"]

        if best_replacement_id and original_profile:
            p_data = self._data["profiles"].get(original_profile)
            if p_data and p_data.get("sample_cycle_id") == original_sample_id:
                p_data["sample_cycle_id"] = best_replacement_id
            
            # Rebuild envelope because dataset changed
            await self.async_rebuild_envelope(original_profile)

        await self.async_save()
        _LOGGER.info("Interactive Split Applied to %s -> %s", cycle_id, new_ids)
        return new_ids

    async def apply_merge_interactive(
        self, cycle_ids: list[str], target_profile: str | None
    ) -> str | None:
        """Apply a manual merge of multiple cycles with gap filling.
        
        Returns the new merged cycle ID.
        """
        if len(cycle_ids) < 2:
            return None

        cycles = self.get_past_cycles()
        target_cycles = [c for c in cycles if c.get("id") in cycle_ids]
        
        if len(target_cycles) != len(cycle_ids):
            return None

        # Sort by time
        target_cycles.sort(key=lambda c: str(c.get("start_time", "")))
        
        # Collect affected profiles for envelope rebuild
        affected_profiles = set()
        for c in target_cycles:
            if c.get("profile_name"):
                affected_profiles.add(c["profile_name"])
        if target_profile:
            affected_profiles.add(target_profile)
        
        # We modify the first cycle (c1) to become the merged one
        c1 = target_cycles[0]
        ids_to_remove = [c["id"] for c in target_cycles[1:]]
        
        # Base setup
        c1_start_dt = dt_util.parse_datetime(c1["start_time"])
        if not c1_start_dt:
            return None

        # Helper to get parsed points from a cycle
        def get_points(cy: CycleDict) -> list[tuple[float, float, float]]:
            # content: [(timestamp, offset, power)]
            raw = self._decompress_power_data(cy)
            res = []
            if not raw:
                return []
            base_t = dt_util.parse_datetime(cy["start_time"]).timestamp()
            for t_str, val in raw:
                dt_pt = dt_util.parse_datetime(t_str)
                if dt_pt:
                    t_abs = dt_pt.timestamp()
                    res.append((t_abs, t_abs - base_t, float(val)))
            return res

        # Start with C1 points
        merged_points_abs: list[list[float]] = [] # [timestamp, power]
        
        # Add C1 points
        c1_pts = get_points(c1)
        for t_abs, _, p in c1_pts:
            merged_points_abs.append([t_abs, p])
        
        last_t_abs = c1_pts[-1][0] if c1_pts else c1_start_dt.timestamp()

        # Iterate others
        max_power = c1.get("max_power", 0)
        
        for next_c in target_cycles[1:]:
            c_start_dt = dt_util.parse_datetime(next_c["start_time"])
            if not c_start_dt:
                continue
            
            c_pts = get_points(next_c)
            if not c_pts:
                continue
            
            current_start_ts = c_pts[0][0]
            
            # --- GAP FILLING ---
            gap = current_start_ts - last_t_abs
            # If gap > 1s, inject 0W points to ensure graph drops to 0
            if gap > 1.0:
                merged_points_abs.append([last_t_abs + 0.1, 0.0])
                merged_points_abs.append([current_start_ts - 0.1, 0.0])
            
            # Append points
            for t_abs, _, p in c_pts:
                merged_points_abs.append([t_abs, p])
                last_t_abs = t_abs
            
            max_power = max(max_power, next_c.get("max_power", 0))

        # Update C1 metadata
        final_end_dt = dt_util.utc_from_timestamp(last_t_abs)
        new_dur = (final_end_dt - c1_start_dt).total_seconds()
        
        c1["end_time"] = final_end_dt.isoformat()
        c1["duration"] = round(new_dur, 1)
        c1["max_power"] = max_power
        c1["profile_name"] = target_profile
        
        # Generate new compressed power_data [offset, power]
        new_power_data = []
        c1_start_ts = c1_start_dt.timestamp()
        
        for t_abs, p in merged_points_abs:
            offset = round(t_abs - c1_start_ts, 1)
            new_power_data.append([offset, float(p)])
        
        c1["power_data"] = new_power_data
        
        # New Hash ID
        new_id = hashlib.sha256(f"{c1['start_time']}_{c1['duration']}".encode()).hexdigest()[:12]
        old_c1_id = c1["id"]
        c1["id"] = new_id
        
        # Update references in profiles
        all_removed_ids = ids_to_remove + [old_c1_id]
        for p_data in self.get_profiles().values():
            if p_data.get("sample_cycle_id") in all_removed_ids:
                p_data["sample_cycle_id"] = new_id

        # Remove consumed cycles
        self._data["past_cycles"] = [
            c for c in cycles if c.get("id") not in ids_to_remove
        ]
        
        # Update signature
        try:
            ts_arr = np.array([pt[0] for pt in new_power_data])
            p_arr = np.array([pt[1] for pt in new_power_data])
            if len(ts_arr) > 1:
                sig = compute_signature(ts_arr, p_arr)
                c1["signature"] = dataclasses.asdict(sig)
        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.warning("Failed to update signature for merged cycle %s: %s", new_id, e)

        await self.async_save()
        _LOGGER.info("Interactive Merge Applied: %s -> %s", cycle_ids, new_id)
        
        # Rebuild envelopes for all affected profiles
        for p_name in affected_profiles:
            await self.async_rebuild_envelope(p_name)
            
        return new_id

    def generate_interactive_split_svg(
        self, cycle_id: str, segments: list[tuple[float, float]], width: int = 600, height: int = 300
    ) -> str:
        """Generate SVG for split preview."""
        cycle = next((c for c in self.get_past_cycles() if c["id"] == cycle_id), None)
        if not cycle:
            return ""

        p_data = self._decompress_power_data(cycle)
        if not p_data:
            return ""
        
        start_ts = dt_util.parse_datetime(cycle["start_time"]).timestamp()
        points = []
        for t_str, val in p_data:
            t = dt_util.parse_datetime(t_str)
            if t:
                points.append((t.timestamp() - start_ts, float(val)))
        
        curves = [SVGCurve(points=points, color="#9E9E9E", opacity=0.5)] # Base ghost
        markers = []

        # Highlight Segments
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, (seg_start, seg_end) in enumerate(segments):
            seg_pts = [(t, p) for t, p in points if seg_start <= t <= seg_end]
            if seg_pts:
                color = colors[i % len(colors)]
                curves.append(SVGCurve(points=seg_pts, color=color, stroke_width=2))
                markers.append({"x": seg_start, "label": f"S{i+1}", "color": color})
        
        return _generate_generic_svg(f"Split Preview: {cycle.get('profile_name') or 'Unlabeled'}", curves, width, height, markers=markers)

    def generate_interactive_merge_svg(
        self, cycle_ids: list[str], width: int = 600, height: int = 300
    ) -> str:
        """Generate SVG for merge preview."""
        cycles = [c for c in self.get_past_cycles() if c["id"] in cycle_ids]
        cycles.sort(key=lambda c: str(c.get("start_time", "")))
        
        if not cycles:
            return ""
        
        first_start = dt_util.parse_datetime(cycles[0]["start_time"]).timestamp()
        curves = []
        
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
        
        for i, c in enumerate(cycles):
            p_data = self._decompress_power_data(c)
            if not p_data:
                continue
            points = []
            for t_str, val in p_data:
                t = dt_util.parse_datetime(t_str)
                if t:
                    rel_t = t.timestamp() - first_start
                    points.append((rel_t, float(val)))
            
            if points:
                curves.append(SVGCurve(points=points, color=colors[i % len(colors)], stroke_width=2))

        return _generate_generic_svg("Merge Preview", curves, width, height)


