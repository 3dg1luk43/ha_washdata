"""Profile storage and matching logic for WashData."""

from __future__ import annotations

import dataclasses
import hashlib
import html
import logging
import os
import re
import uuid
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
from .time_utils import (
    migrate_power_data_to_offsets,
    power_data_to_offsets,
)
from .phase_catalog import (
    DEFAULT_PHASES_BY_DEVICE,
    _builtin_phase_id,
    get_builtin_phase_by_id,
    merge_phase_catalog,
    normalize_phase_name,
)
from .log_utils import DeviceLoggerAdapter

_LOGGER = logging.getLogger(__name__)

JSONDict: TypeAlias = dict[str, Any]
CycleDict: TypeAlias = dict[str, Any]


def _empty_ranking() -> list[dict[str, Any]]:
    """Typed default factory for ranking entries."""
    return []


def _empty_debug_details() -> dict[str, Any]:
    """Typed default factory for debug details."""
    return {}


def _value_to_timestamp(value: Any) -> float | None:
    """Parse supported datetime-like values into unix seconds."""
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        parsed = dt_util.parse_datetime(value)
        if parsed is not None:
            return parsed.timestamp()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


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


def filter_duration_outliers(durations: list[float]) -> list[float]:
    """Return a robust duration set with extreme outliers removed.

    Uses Tukey IQR fences for normal spread and falls back to a MAD-based
    filter when IQR collapses (common when most cycles are identical).
    """
    if len(durations) < 4:
        return durations

    arr = np.array(durations, dtype=float)

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    if iqr > 0:
        lower = max(60.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower) & (arr <= upper)]
    else:
        # Degenerate spread (e.g. many identical durations); keep values
        # close to median and drop only extreme anomalies.
        median = float(np.median(arr))
        abs_dev = np.abs(arr - median)
        mad = float(np.median(abs_dev))
        if mad == 0:
            tol = max(300.0, median * 0.15)
            filtered = arr[np.abs(arr - median) <= tol]
        else:
            robust_z = abs_dev / (1.4826 * mad)
            filtered = arr[robust_z <= 3.5]

    # Guardrail: do not over-filter sparse datasets.
    if len(filtered) >= max(3, int(len(arr) * 0.6)):
        return filtered.astype(float).tolist()

    return durations


@dataclasses.dataclass
class SVGCurve:
    """Definition for a curve in the SVG chart."""
    points: list[tuple[float, float]]  # (x, y)
    color: str
    opacity: float = 1.0
    stroke_width: int = 2
    dasharray: str | None = None
    is_polygon: bool = False


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
    paths: list[str] = []
    for c in curves:
        if not c.points:
            continue

        pts: list[str] = []
        # Optimization: verify step size if huge data
        for x_val, y_val in c.points:
            pts.append(f"{to_x(x_val):.1f},{to_y(y_val):.1f}")

        path_d = " ".join(pts)
        if c.is_polygon:
            style = f'fill="{c.color}" fill-opacity="{c.opacity}" stroke="none"'
            paths.append(f'<polygon points="{path_d}" {style} />')
        else:
            style = f'stroke="{c.color}" stroke-width="{c.stroke_width}" stroke-opacity="{c.opacity}" fill="none"'
            if c.dasharray:
                style += f' stroke-dasharray="{c.dasharray}"'
            paths.append(f'<polyline points="{path_d}" {style} />')

    # Build Markers
    marker_svgs: list[str] = []
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
    ranking: list[dict[str, Any]] = dataclasses.field(default_factory=_empty_ranking)
    debug_details: dict[str, Any] = dataclasses.field(default_factory=_empty_debug_details)
    is_confident_mismatch: bool = False
    mismatch_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with JSON-serializable types, excluding heavy arrays."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.generic):
                return cast(np.generic, obj).item()
            if isinstance(obj, np.ndarray):
                # Fallback for unexpected arrays: just describe shape
                arr = cast(np.ndarray[Any, Any], obj)
                return f"<array shape={arr.shape}>"
            if isinstance(obj, dict):
                # Exclude huge raw data arrays from cycle candidates
                obj_dict = cast(dict[str, Any], obj)
                return {
                    k: _convert(v)
                    for k, v in obj_dict.items()
                    if k not in ("current", "sample", "metrics", "warping_path")
                }
            if isinstance(obj, list):
                obj_list = cast(list[Any], obj)
                return [_convert(v) for v in obj_list]
            if dataclasses.is_dataclass(obj):
                return text_type_safe_asdict(obj)
            return obj

        def text_type_safe_asdict(d_obj: Any) -> dict[str, Any]:
            return {f.name: _convert(getattr(d_obj, f.name)) for f in dataclasses.fields(d_obj)}

        return text_type_safe_asdict(self)



def decompress_power_data(cycle: CycleDict) -> list[tuple[float, float]]:
    """Return power data as ``[(offset_seconds, power), ...]`` for a cycle.

    Handles both the current canonical ``[offset_float, power]`` format and the
    legacy ``(iso_str, power)`` format transparently.  Returns an empty list if
    data is missing or malformed.
    """
    raw = cycle.get("power_data", [])
    if not isinstance(raw, list) or not raw:
        return []

    start_time_raw = cycle.get("start_time")
    start_time_iso: str | None = start_time_raw if isinstance(start_time_raw, str) and start_time_raw else None
    if isinstance(start_time_raw, datetime):
        start_time_iso = start_time_raw.isoformat()
    offsets = power_data_to_offsets(cast(list[list[Any] | tuple[Any, ...]], raw), start_time_iso)
    return [(float(o), float(p)) for o, p in offsets]


def compress_power_data(cycle: CycleDict) -> list[Any] | None:
    """Compress cycle power data to [offset, power] format (Module-level helper).

    Returns the compressed list structure or None if compression failed/not needed.
    """
    raw_data_raw = cycle.get("power_data")
    if not isinstance(raw_data_raw, list) or not raw_data_raw:
        return None
    raw_data = cast(list[Any], raw_data_raw)

    # Check if already compressed (first element is number or mixed format)
    first = raw_data[0]
    if isinstance(first, (int, float)):
        # Already flat list (very old format?) or specific compression
        return None
    if isinstance(first, (list, tuple)):
        first_seq = cast(list[Any] | tuple[Any, ...], first)
        if len(first_seq) == 2 and isinstance(first_seq[0], (int, float)):
            # Already compressed [offset, power]
            return None

    # Proceed with compression from [iso_string, power]
    if "start_time" not in cycle:
        return None

    try:
        start_ts = _value_to_timestamp(cycle.get("start_time"))
        if start_ts is None:
            return None
        compressed: list[list[float]] = []

        last_saved_p = -999.0
        last_saved_t = -999.0

        for i, entry in enumerate(raw_data):
            if isinstance(entry, (list, tuple)):
                entry_seq = cast(list[Any] | tuple[Any, ...], entry)
                if len(entry_seq) != 2:
                    continue
                t_str, p_val_raw = entry_seq
                try:
                    # Handle both ISO string and potential timestamp float
                    t = _value_to_timestamp(t_str)
                    if t is None:
                        continue

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
            cycles_raw = old_data.get("past_cycles", [])
            cycles = cast(list[dict[str, Any]], cycles_raw) if isinstance(cycles_raw, list) else []
            migrated_cycles = 0
            for cycle in cycles:
                if "signature" not in cycle and cycle.get("power_data"):
                    try:
                        # decompress_power_data now returns [(offset_seconds, power), ...]
                        tuples = decompress_power_data(cycle)
                        if tuples and len(tuples) > 10:
                            ts_arr = np.array([t for t, _ in tuples])
                            p_arr = np.array([p for _, p in tuples])
                            sig = compute_signature(ts_arr, p_arr)
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
            cycles_raw = old_data.get("past_cycles", [])
            profiles_raw = old_data.get("profiles", {})
            cycles = cast(list[dict[str, Any]], cycles_raw) if isinstance(cycles_raw, list) else []
            profiles = cast(dict[str, dict[str, Any]], profiles_raw) if isinstance(profiles_raw, dict) else {}
            migrated_count = 0

            # 1. Migrate Power Data to canonical offset format & ensure status
            for cycle in cycles:
                if "status" not in cycle:
                    cycle["status"] = "completed"

                if cycle.get("power_data") and isinstance(cycle["power_data"], list):
                    try:
                        if migrate_power_data_to_offsets(cycle):
                            migrated_count += 1
                    except Exception as e:
                        _LOGGER.warning(
                            "Failed to migrate power data for cycle %s: %s",
                            cycle.get("id"),
                            e,
                        )

            # 2. Ensure Device Type in Profiles
            for profile in profiles.values():
                if "device_type" not in profile:
                    profile["device_type"] = "washing_machine"

            _LOGGER.info(
                "Migration v2->v3: Migrated power data for %s cycles", migrated_count
            )

        if old_major_version < 4:
            _LOGGER.info("Migrating storage from v%s to v4", old_major_version)
            profiles = old_data.get("profiles", {})
            if isinstance(profiles, dict):
                for profile in cast(dict[str, dict[str, Any]], profiles).values():
                    phases = profile.get("phases")
                    if not isinstance(phases, list):
                        profile["phases"] = []

            custom = old_data.get("custom_phases")
            if not isinstance(custom, dict):
                old_data["custom_phases"] = {}

        if old_major_version < 5:
            _LOGGER.info("Migrating storage from v%s to v5", old_major_version)
            custom = old_data.get("custom_phases")
            if isinstance(custom, list):
                normalized: list[dict[str, Any]] = []
                seen: set[tuple[str, str]] = set()
                for item in cast(list[Any], custom):
                    if not isinstance(item, dict):
                        continue
                    item_dict = cast(dict[str, Any], item)
                    name = str(item_dict.get("name", "")).strip()
                    if not name:
                        continue
                    device_type = str(item_dict.get("device_type", "")).strip()
                    key = (name.casefold(), device_type.casefold())
                    if key in seen:
                        continue
                    seen.add(key)
                    normalized.append(
                        {
                            "name": name,
                            "description": str(item_dict.get("description", "")).strip(),
                            "device_type": device_type,
                            "created_at": item_dict.get("created_at") or dt_util.now().isoformat(),
                        }
                    )
                old_data["custom_phases"] = normalized
            elif isinstance(custom, dict):
                normalized: list[dict[str, Any]] = []
                seen: set[tuple[str, str]] = set()
                custom_dict = cast(dict[str, Any], custom)
                for legacy_device_type, phase_list in custom_dict.items():
                    if not isinstance(phase_list, list):
                        continue
                    for item in cast(list[Any], phase_list):
                        if not isinstance(item, dict):
                            continue
                        item_dict = cast(dict[str, Any], item)
                        name = str(item_dict.get("name", "")).strip()
                        if not name:
                            continue
                        device_type = str(legacy_device_type or "").strip()
                        key = (name.casefold(), device_type.casefold())
                        if key in seen:
                            continue
                        seen.add(key)
                        normalized.append(
                            {
                                "name": name,
                                "description": str(item_dict.get("description", "")).strip(),
                                "device_type": device_type,
                                "created_at": item_dict.get("created_at") or dt_util.now().isoformat(),
                            }
                        )
                old_data["custom_phases"] = normalized
            else:
                old_data["custom_phases"] = []

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

        if self._data is None:
            return 0

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
        device_name: str = "",
    ) -> None:
        """Initialize the profile store."""
        self.hass = hass
        self.entry_id = entry_id
        self._logger = DeviceLoggerAdapter(_LOGGER, device_name)
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
            "custom_phases": [],  # Shared custom phase catalog
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

    def _get_shared_custom_phases(self) -> list[dict[str, Any]]:
        """Return mutable shared custom phase list with legacy flattening."""
        raw = self._data.setdefault("custom_phases", [])
        if isinstance(raw, list):
            return cast(list[dict[str, Any]], raw)

        # Legacy format: {device_type: [phase, ...]}. Flatten to shared list.
        flattened: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        if isinstance(raw, dict):
            raw_dict = cast(dict[str, Any], raw)
            for legacy_device_type, phase_list in raw_dict.items():
                if not isinstance(phase_list, list):
                    continue
                for item in cast(list[Any], phase_list):
                    if not isinstance(item, dict):
                        continue
                    item_dict = cast(dict[str, Any], item)
                    name = str(item_dict.get("name", "")).strip()
                    if not name:
                        continue
                    device_type = str(legacy_device_type or "").strip()
                    key = (name.casefold(), device_type.casefold())
                    if key in seen:
                        continue
                    seen.add(key)
                    flattened.append(
                        {
                            "name": name,
                            "description": str(item_dict.get("description", "")).strip(),
                            "device_type": device_type,
                            "created_at": item_dict.get("created_at") or dt_util.now().isoformat(),
                        }
                    )

        self._data["custom_phases"] = flattened
        return self._data["custom_phases"]

    def list_custom_phases(self, device_type: str) -> list[dict[str, Any]]:
        """Return shared custom phases relevant to the requested device type."""

        def applies_to_device(item_device_type: str, target_device_type: str) -> bool:
            if not item_device_type:
                return True
            return item_device_type == target_device_type

        target = str(device_type or "").strip()
        phases = self._get_shared_custom_phases()
        return [
            {
                "id": str(p.get("id", "")),
                "name": str(p.get("name", "")).strip(),
                "description": str(p.get("description", "")).strip(),
                "device_type": str(p.get("device_type", "")).strip(),
                "is_default": False,
            }
            for p in phases
            if p.get("name")
            and applies_to_device(str(p.get("device_type", "")).strip(), target)
        ]

    def list_phase_catalog(self, device_type: str) -> list[dict[str, Any]]:
        """Return merged shared default + custom phase catalog."""
        return merge_phase_catalog(device_type, self.list_custom_phases(device_type))

    async def async_create_custom_phase(
        self,
        device_type: str,
        phase_name: str,
        description: str = "",
    ) -> None:
        """Create a custom phase in the shared catalog."""
        target_device_type = str(device_type or "").strip()
        name = normalize_phase_name(phase_name)
        desc = str(description or "").strip()
        catalog = self.list_phase_catalog(target_device_type)
        if any(str(p.get("name", "")).casefold() == name.casefold() for p in catalog):
            raise ValueError("duplicate_phase")

        self._get_shared_custom_phases().append(
            {
                "id": str(uuid.uuid4()),
                "name": name,
                "description": desc,
                "device_type": target_device_type,
                "created_at": dt_util.now().isoformat(),
            }
        )
        await self.async_save()

    async def async_update_custom_phase(
        self,
        phase_id: str,
        new_name: str,
        description: str = "",
    ) -> None:
        """Update a phase by id, propagating rename to profile assignments.

        If phase_id matches a built-in, a custom override is created using the
        built-in's id so the merge can replace it in-place.
        """
        target_name = normalize_phase_name(new_name)
        desc = str(description or "").strip()
        phases = self._get_shared_custom_phases()

        # Look for an existing custom entry with this id.
        found: dict[str, Any] | None = next(
            (p for p in phases if str(p.get("id", "")) == phase_id), None
        )
        creating_new = found is None

        if creating_new:
            builtin = get_builtin_phase_by_id(phase_id)
            if builtin is None:
                raise ValueError("phase_not_found")
            candidate: dict[str, Any] = {
                "id": phase_id,
                "name": str(builtin.get("name", "")),
                "description": "",
                "device_type": str(builtin.get("device_type", "")),
                "created_at": dt_util.now().isoformat(),
            }
        else:
            candidate = found  # type: ignore[assignment]

        old_name = str(candidate.get("name", ""))
        target_device_type = str(candidate.get("device_type", "")).strip()

        # Duplicate-name check before any mutation.
        for p in self.list_phase_catalog(target_device_type):
            pname = str(p.get("name", ""))
            if pname.casefold() == target_name.casefold() and pname.casefold() != old_name.casefold():
                raise ValueError("duplicate_phase")

        if creating_new:
            phases.append(candidate)
            found = candidate

        found["name"] = target_name  # type: ignore[index]
        found["description"] = desc  # type: ignore[index]

        # Propagate rename to profile assignments.
        for profile in self.get_profiles().values():
            profile_device_type = str(profile.get("device_type", "")).strip()
            if target_device_type and profile_device_type != target_device_type:
                continue
            phases_assigned = profile.get("phases", [])
            if not isinstance(phases_assigned, list):
                continue
            for assigned in cast(list[dict[str, Any]], phases_assigned):
                if str(assigned.get("name", "")).casefold() == old_name.casefold():
                    assigned["name"] = target_name

        await self.async_save()

    def count_phase_usage(self, phase_name: str) -> int:
        """Count how many profile assignments use a phase name."""
        used = 0
        for profile in self.get_profiles().values():
            phases_assigned = profile.get("phases", [])
            if not isinstance(phases_assigned, list):
                continue
            assigned_list = cast(list[dict[str, Any]], phases_assigned)
            used += sum(
                1
                for phase in assigned_list
                if str(phase.get("name", "")).casefold() == phase_name.casefold()
            )
        return used

    async def async_delete_custom_phase(self, phase_id: str) -> int:
        """Delete a custom phase by id and remove matching profile assignments.

        Returns number of removed assignments.
        Raises ValueError('phase_not_found') if no custom phase has this id.
        Raises ValueError('cannot_delete_builtin') if the id is a built-in phase.
        """
        phases = self._get_shared_custom_phases()
        found = next((p for p in phases if str(p.get("id", "")) == phase_id), None)
        if found is None:
            raise ValueError("phase_not_found")
        if get_builtin_phase_by_id(phase_id) is not None:
            raise ValueError("cannot_delete_builtin")

        phase_name = str(found.get("name", ""))
        phase_scope = str(found.get("device_type", "")).strip()
        self._data["custom_phases"] = [p for p in phases if str(p.get("id", "")) != phase_id]

        removed_assignments = 0
        for profile in self.get_profiles().values():
            profile_device_type = str(profile.get("device_type", "")).strip()
            if phase_scope and profile_device_type != phase_scope:
                continue
            assigned = profile.get("phases", [])
            if not isinstance(assigned, list):
                continue
            assigned_list = cast(list[dict[str, Any]], assigned)
            before = len(assigned_list)
            profile["phases"] = [
                p for p in assigned_list
                if str(p.get("name", "")).casefold() != phase_name.casefold()
            ]
            removed_assignments += before - len(profile["phases"])

        await self.async_save()
        return removed_assignments

    def get_profile_phase_ranges(self, profile_name: str) -> list[dict[str, Any]]:
        """Return assigned phase ranges for a profile."""
        profiles = self._data.get("profiles", {})
        if not isinstance(profiles, dict):
            return []
        profile_raw = cast(dict[str, Any], profiles).get(profile_name)
        profile = cast(dict[str, Any], profile_raw) if isinstance(profile_raw, dict) else None
        if not isinstance(profile, dict):
            return []
        phases = profile.get("phases", [])
        if not isinstance(phases, list):
            return []
        phases_list = cast(list[dict[str, Any]], phases)
        cleaned: list[dict[str, Any]] = []
        for phase in phases_list:
            try:
                start = float(phase.get("start", 0.0))
                end = float(phase.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            cleaned.append(
                {
                    "name": str(phase.get("name", "")).strip(),
                    "start": start,
                    "end": end,
                    "description": str(phase.get("description", "")).strip(),
                }
            )
        return sorted(cleaned, key=lambda x: (x["start"], x["end"], x["name"]))

    def get_profile_phase_ranges_for_device(
        self, profile_name: str, device_type: str
    ) -> list[dict[str, Any]]:
        """Return assigned ranges enriched with catalog descriptions."""
        ranges = self.get_profile_phase_ranges(profile_name)
        catalog = self.list_phase_catalog(device_type)
        desc_map = {
            str(p.get("name", "")).casefold(): str(p.get("description", "")).strip()
            for p in catalog
        }
        enriched: list[dict[str, Any]] = []
        for row in ranges:
            name = str(row.get("name", "")).strip()
            enriched.append(
                {
                    "name": name,
                    "start": float(row.get("start", 0.0)),
                    "end": float(row.get("end", 0.0)),
                    "description": desc_map.get(name.casefold(), ""),
                }
            )
        return enriched

    async def async_set_profile_phase_ranges(
        self, profile_name: str, ranges: list[dict[str, Any]]
    ) -> None:
        """Replace assigned phase ranges for a profile."""
        profile = self._data.get("profiles", {}).get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError("profile_not_found")

        normalized: list[dict[str, Any]] = []
        for item in ranges:
            name = normalize_phase_name(str(item.get("name", "")))
            try:
                start = float(item.get("start", 0.0))
                end = float(item.get("end", 0.0))
            except (TypeError, ValueError) as e:
                raise ValueError("invalid_phase_range") from e
            if end <= start:
                raise ValueError("invalid_phase_range")
            normalized.append({"name": name, "start": start, "end": end})

        normalized.sort(key=lambda x: (x["start"], x["end"]))
        prev_end = None
        for row in normalized:
            if prev_end is not None and row["start"] < prev_end:
                raise ValueError("overlapping_phase_ranges")
            prev_end = row["end"]

        profile["phases"] = normalized
        await self.async_save()

    def set_duration_tolerance(self, tolerance: float) -> None:
        """Set the profile duration tolerance used by matching heuristics."""
        try:
            self._duration_tolerance = float(tolerance)
        except (TypeError, ValueError):
            pass

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
            pass

    def get_duration_ratio_limits(self) -> tuple[float, float]:
        """Return (min_duration_ratio, max_duration_ratio) used for duration matching."""
        return (float(self._min_duration_ratio), float(self._max_duration_ratio))

    def set_duration_ratio_limits(self, *, min_ratio: float, max_ratio: float) -> None:
        """Update duration ratio bounds used for duration matching."""
        try:
            self._min_duration_ratio = float(min_ratio)
            self._max_duration_ratio = float(max_ratio)
        except (TypeError, ValueError):
            pass

    def _migrate_phase_ids(self) -> bool:
        """Assign ids to any custom phase missing one. Returns True if anything changed."""
        phases = self._data.get("custom_phases", [])
        if not isinstance(phases, list):
            return False
        changed = False
        for phase in cast(list[dict[str, Any]], phases):
            if phase.get("id"):
                continue
            dt = str(phase.get("device_type", "")).strip()
            name = str(phase.get("name", "")).strip()
            matched_id: str | None = None
            for bdt, bphases in DEFAULT_PHASES_BY_DEVICE.items():
                if dt and bdt != dt:
                    continue
                for bp in bphases:
                    if str(bp.get("name", "")).strip().casefold() == name.casefold():
                        matched_id = _builtin_phase_id(bdt, str(bp.get("name", "")))
                        break
                if matched_id:
                    break
            phase["id"] = matched_id if matched_id else str(uuid.uuid4())
            changed = True
        return changed

    async def async_load(self) -> None:
        """Load data from storage with migration."""
        # WashDataStore handles migration internally via _async_migrate_func
        data = await self._store.async_load()
        if data:
            self._data = data
        # Ensure legacy custom phase formats are normalized in-memory.
        self._get_shared_custom_phases()
        # Assign ids to any custom phase missing one.
        if self._migrate_phase_ids():
            await self.async_save()
        # Repair cycles whose power_data was corrupted by the double-subtract bug.
        if self.repair_corrupted_power_data():
            await self.async_save()
            await self.async_rebuild_all_envelopes()
            await self.async_save()

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
        self._logger.debug("add_cycle: raw_data has %s points", len(raw_data))

        if raw_data:
            start_time_raw = cycle_data.get("start_time")
            start_time_iso: str | None = None
            if isinstance(start_time_raw, str) and start_time_raw:
                parsed_dt = dt_util.parse_datetime(start_time_raw)
                if parsed_dt is not None:
                    start_time_iso = start_time_raw
                else:
                    try:
                        ts = float(start_time_raw)
                        start_time_iso = dt_util.utc_from_timestamp(ts).isoformat()
                    except (ValueError, OSError):
                        self._logger.debug(
                            "add_cycle: unparseable string start_time %r, falling back",
                            start_time_raw,
                        )
            elif isinstance(start_time_raw, datetime):
                start_time_iso = start_time_raw.isoformat()
            elif isinstance(start_time_raw, (int, float)):
                try:
                    start_time_iso = dt_util.utc_from_timestamp(float(start_time_raw)).isoformat()
                except (ValueError, OSError):
                    pass
            if start_time_iso is not None:
                cycle_data["start_time"] = start_time_iso
            if start_time_iso is None and _value_to_timestamp(start_time_raw) is None:
                self._logger.debug("add_cycle: invalid start_time %r, skipping power_data normalization", start_time_raw)
                if hasattr(self, "_save_debug_traces") and not self._save_debug_traces:
                    cycle_data.pop("debug_data", None)
                self._data["past_cycles"].append(cycle_data)
                return

            # Use unified normalizer: handles offset, ISO-string, and datetime formats
            pairs = power_data_to_offsets(
                cast(list[list[Any] | tuple[Any, ...]], raw_data), start_time_iso
            )
            stored: list[list[float]] = [[round(p[0], 1), round(p[1], 1)] for p in pairs]
            offsets: list[float] = [p[0] for p in stored]

            # Calculate average sampling interval (in seconds)
            if len(offsets) > 1:
                intervals = np.diff(offsets)
                positive_intervals = intervals[intervals > 0]
                sampling_interval = float(np.median(positive_intervals)) if positive_intervals.size > 0 else 0.0
            else:
                sampling_interval = 1.0  # Default fallback

            # Trim leading/trailing zero readings for cleaner data
            # SKIP for completed cycles to preserve end spike data
            if cycle_data.get("status") in ("completed", "force_stopped"):
                # Only trim leading zeros for completed cycles, keep trailing data
                start_idx = 0
                for i, point in enumerate(stored):
                    if point[1] > 1.0:
                        start_idx = i
                        break
                stored = stored[start_idx:]
                self._logger.debug("add_cycle: Skipping trailing trim for completed cycle")
            else:
                stored = trim_zero_power_data(stored, threshold=1.0)

            cycle_data["power_data"] = stored
            cycle_data["sampling_interval"] = round(sampling_interval, 1)

            # Helper to get arrays for signature (use stored data for consistency)
            ts_arr = np.array([t for t, _ in stored])
            p_arr = np.array([p for _, p in stored])

            # Compute and store signature
            if len(ts_arr) > 1 and len(ts_arr) == len(p_arr):
                sig = compute_signature(ts_arr, p_arr)
                cycle_data["signature"] = dataclasses.asdict(sig)

            self._logger.debug(
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
                self._logger.warning("Failed to schedule envelope rebuild for %s: %s", p, e)

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

        # Collect cycle IDs that have pending feedback — never strip their power_data
        pending_feedback_ids: set[str] = set(self._data.get("pending_feedback", {}).keys())

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

                    # EXEMPTION: Never strip power data from cycles awaiting feedback review
                    if c.get("id") in pending_feedback_ids:
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
        orphaned: list[str] = []
        for name, profile in self._data["profiles"].items():
            ref = profile.get("sample_cycle_id")
            # Only delete if it references a non-existent cycle ID (Broken Link)
            # Creating a profile without a sample (None) is allowed (Pending State)
            if ref and ref not in cycle_ids:
                orphaned.append(name)

        for name in orphaned:
            del self._data["profiles"][name]
            self._logger.info(
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
            self._logger.info("Maintenance completed: %s", stats)

        return stats

    def _reprocess_all_data_sync(self) -> int:
        """Synchronous implementation of reprocessing logic (run in executor)."""
        cycles_raw = self._data.get("past_cycles", [])
        cycles = cast(list[CycleDict], cycles_raw) if isinstance(cycles_raw, list) else []
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
            ):
                first_point = cast(list[Any] | tuple[Any, ...], p_data[0])
                # Only trim offset-format data (numeric offsets). Legacy ISO-format
                # cycles skip trimming but still reach the signature block below.
                if len(first_point) == 2 and isinstance(first_point[0], (int, float)):
                    p_data_list = cast(list[list[float]], p_data)
                    # Apply trim helper
                    original_len = len(p_data_list)

                    # Logic: For completed cycles, only trim leading zeros.
                    # For others, trim both ends.
                    if cycle.get("status") in ("completed", "force_stopped"):
                        # Only trim leading
                        start_idx = 0
                        for i, point in enumerate(p_data_list):
                            if point[1] > 1.0: # Match threshold below
                                start_idx = i
                                break
                        trimmed: list[list[float]] = p_data_list[start_idx:]
                    else:
                        trimmed = trim_zero_power_data(p_data_list, threshold=1.0) # Conservative 1W threshold

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
                                shifted_data: list[list[float]] = []
                                for row in trimmed:
                                    # row is [offset, power]
                                    shifted_data.append([round(row[0] - first_offset, 1), row[1]])
                                cycle["power_data"] = shifted_data
                                processed_count += 1
                            except (ValueError, TypeError) as e:
                                self._logger.warning("Failed to shift start_time for trimmed cycle: %s", e)
                        else:
                            # Only trailing trimmed or no shift needed
                            cycle["power_data"] = trimmed
                            processed_count += 1

                        # Update duration to match new data length
                        # If we only trimmed the head, the new duration is old_duration - first_offset
                        # This preserves trailing silence.
                        if cycle.get("power_data"):
                            old_dur = float(cycle.get("duration", 0.0) or 0.0)
                            # If we shifted (first_offset > 0), new duration is old_dur - first_offset
                            # Otherwise if we only trimmed tail, we might want to snap,
                            # but for completed cycles we don't trim tail in this loop.
                            if first_offset > 0:
                                cycle["duration"] = max(0.0, old_dur - first_offset)
                            else:
                                # Only trailing was trimmed (not expected for completed cycles here)
                                # or no trim happened.
                                # If trailing was trimmed, we SHOULD snap.
                                if len(trimmed) < original_len:
                                    cycle["duration"] = cycle["power_data"][-1][0]

            if cycle.get("power_data"):
                try:
                    tuples = decompress_power_data(cycle)
                    if tuples and len(tuples) > 10:
                        ts_arr: list[float] = []
                        p_arr: list[float] = []
                        for offset_sec, p in tuples:
                            ts_arr.append(float(offset_sec))
                            p_arr.append(float(p))

                        sig = compute_signature(np.array(ts_arr, dtype=float), np.array(p_arr, dtype=float))
                        cycle["signature"] = dataclasses.asdict(sig)
                        processed_count += 1
                except Exception as e: # pylint: disable=broad-exception-caught
                    self._logger.warning("Failed to reprocess signature: %s", e)

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
        self._logger.info("Starting reprocessing (offloaded)...")

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
            self._logger.info("Cleared debug data from %s cycles", count)

        return count



    def _rebuild_envelope_sync(
        self, labeled_cycles: list[CycleDict]
    ) -> tuple[Any, list[float]] | None:
        """Sync worker to parse data and build envelope (run in executor)."""
        raw_cycles_data: list[tuple[list[float], list[float], float]] = []
        durations: list[float] = []

        for cycle in labeled_cycles:
            # Use the shared decompressor so both legacy ISO-timestamp format
            # and the current offset-float format are handled transparently.
            pairs = self._decompress_power_data(cycle)

            if len(pairs) < 3:
                continue

            offsets: list[float] = [p[0] for p in pairs]
            values: list[float] = [p[1] for p in pairs]

            stored_dur = float(cycle.get("duration", 0.0) or 0.0)
            authoritative_dur = float(max(offsets[-1], stored_dur))

            # Use manual duration if available (e.g. from feedback correction)
            man_dur = cycle.get("manual_duration")
            if man_dur:
                final_dur = float(man_dur)
            else:
                final_dur = authoritative_dur

            raw_cycles_data.append((offsets, values, final_dur))
            durations.append(final_dur)

        if not raw_cycles_data:
            return None

        # Run Heavy Computation
        result = analysis.compute_envelope_worker(
            cast(Any, raw_cycles_data),
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

    def repair_corrupted_power_data(self) -> int:
        """Fix cycles whose power_data offsets were corrupted by the double-subtract bug.

        The bug caused ``offset = small_float - unix_timestamp`` to be stored instead of
        just ``small_float``.  Corrupted cycles have a first-offset < -1e8 (a value that
        can never occur for a real appliance cycle offset).  Recovery: add ``start_ts``
        back to every offset in the affected cycle.

        Returns the number of cycles repaired.
        """
        repaired = 0
        for cycle in self._data.get("past_cycles", []):
            power_data = cycle.get("power_data")
            if not isinstance(power_data, list) or not power_data:
                continue
            first = power_data[0]
            if not isinstance(first, (list, tuple)) or len(first) < 2:
                continue
            first_offset = first[0]
            if not isinstance(first_offset, (int, float)) or first_offset > -1e8:
                continue  # Not corrupted

            start_ts = _value_to_timestamp(cycle.get("start_time"))
            if start_ts is None:
                continue

            repaired_rows: list[list[float]] = []
            for pt in power_data:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                try:
                    repaired_rows.append([round(float(pt[0]) + start_ts, 1), round(float(pt[1]), 1)])
                except (TypeError, ValueError):
                    continue
            if not repaired_rows:
                continue  # all rows malformed — leave original trace untouched
            cycle["power_data"] = repaired_rows
            repaired += 1
            repaired_data = cycle["power_data"]
            if len(repaired_data) > 1:
                r_offsets = [pt[0] for pt in repaired_data]
                r_intervals = np.diff(r_offsets)
                r_pos = r_intervals[r_intervals > 0]
                r_si = float(np.median(r_pos)) if len(r_pos) > 0 else 1.0
                cycle["sampling_interval"] = round(r_si, 1)
                # duration = last sample offset from cycle start (not span between
                # first and last sample, which would be wrong when leading zeros
                # were trimmed before storage)
                r_duration = round(r_offsets[-1], 1)
                cycle["duration"] = r_duration
                cycle["end_time"] = dt_util.utc_from_timestamp(
                    start_ts + r_duration
                ).isoformat()
                r_ts = np.array(r_offsets, dtype=float)
                r_p = np.array([pt[1] for pt in repaired_data], dtype=float)
                r_sig = compute_signature(r_ts, r_p)
                cycle["signature"] = dataclasses.asdict(r_sig)
            elif len(repaired_data) == 1:
                cycle["sampling_interval"] = 1.0
                cycle["duration"] = 0.0
                cycle["end_time"] = dt_util.utc_from_timestamp(start_ts).isoformat()
                cycle["signature"] = None

        if repaired:
            self._logger.warning(
                "Repaired corrupted power_data offsets in %d cycle(s)", repaired
            )
        return repaired

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
            stats_durations = filter_duration_outliers(durations)
            raw_arr = np.array(durations, dtype=float)
            # min/max reflect the actual observed range (including outliers)
            # avg uses the outlier-filtered set for a robust representative value
            min_duration = float(np.min(raw_arr))
            max_duration = float(np.max(raw_arr))
            avg_duration = float(np.mean(stats_durations))
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

        envelope_data: dict[str, Any] = {
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
            time_grid = cast(list[float], envelope["time_grid"])
            # Envelope curves are stored as list of [t, y] points.
            # Extract Y values for SVG generation logic.
            avg_curve = [float(p[1]) for p in cast(list[list[Any] | tuple[Any, ...]], envelope["avg"])]
            min_curve = [float(p[1]) for p in cast(list[list[Any] | tuple[Any, ...]], envelope["min"])]
            max_curve = [float(p[1]) for p in cast(list[list[Any] | tuple[Any, ...]], envelope["max"])]

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
            points_max: list[str] = []
            points_min: list[str] = []
            points_avg: list[str] = []

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
            self._logger.error("Error generating SVG for %s: %s", profile_name, e)
            return None



    def generate_profile_spaghetti_svg(
        self, profile_name: str, overview_suffix: str = "Overview"
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
                for item in cast(list[Any], power_data_raw):
                    if isinstance(item, (list, tuple)):
                        item_seq = cast(list[Any] | tuple[Any, ...], item)
                        if len(item_seq) < 2:
                            continue
                        try:
                            pairs.append((float(item_seq[0]), float(item_seq[1])))
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
            title=f"{profile_name} ({overview_suffix})",
            curves=svg_curves,
            width=1000,
            height=400
        )

        return svg_content, cycle_metadata

    def generate_preview_svg(
        self,
        power_data: list[tuple[str, float]],
        head_trim: float,
        tail_trim: float,
        title: str = "Recording Preview",
        trim_start_label: str = "Trim Start",
        trim_end_label: str = "Trim End",
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
            start_dt = dt_util.parse_datetime(power_data[0][0])
            if start_dt is None:
                return ""
            start_ts = start_dt.timestamp()
            for t_str, p in power_data:
                parsed = dt_util.parse_datetime(t_str)
                if parsed is None:
                    continue
                t = parsed.timestamp() - start_ts
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
        markers: list[dict[str, Any]] = [
            {"x": keep_start, "label": trim_start_label, "color": "#e6194b"},
            {"x": keep_end, "label": trim_end_label, "color": "#e6194b"},
        ]

        return _generate_generic_svg(
            title=title,
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

    def generate_feedback_comparison_svg(
        self, profile_name: str, actual_cycle: CycleDict
    ) -> str | None:
        """Generate SVG comparing expected profile envelope with actual recorded cycle.

        Displays:
        - Light blue band: min/max envelope from all labeled cycles
        - Darker blue line: average expected profile
        - Orange line: actual recorded power data from the cycle

        Args:
            profile_name: Name of the detected/expected profile
            actual_cycle: CycleDict with power_data and duration

        Returns:
            SVG string or None if data unavailable
        """
        try:
            # Get envelope for the profile
            envelope = self.get_envelope(profile_name)
            if not envelope or not envelope.get("time_grid"):
                return None

            # Decompress actual cycle power data (handles both ISO-timestamp and offset formats)
            actual_pairs = decompress_power_data(actual_cycle)

            if len(actual_pairs) < 3:
                return None

            # Extract envelope curves (already have [t, y] format)
            time_grid = envelope["time_grid"]
            avg_curve = envelope.get("avg", [])
            min_curve = envelope.get("min", [])
            max_curve = envelope.get("max", [])

            if not avg_curve or not min_curve or not max_curve:
                return None

            # Build envelope curves for SVG
            avg_points = [(p[0], p[1]) for p in avg_curve]
            min_points = [(p[0], p[1]) for p in min_curve]
            max_points = [(p[0], p[1]) for p in max_curve]

            # For the expected envelope band, we'll create a special visualization
            # Canvas configuration (same as profile stats)
            width, height = 1200, 450

            # Use max time from actual data or envelope, whichever is larger
            max_time_envelope = time_grid[-1] if time_grid else 1.0
            max_time_actual = actual_pairs[-1][0] if actual_pairs else 1.0
            max_time = max(max_time_envelope, max_time_actual)

            # Determine max power for scaling
            all_power = (
                [p[1] for p in min_curve] +
                [p[1] for p in avg_curve] +
                [p[1] for p in max_curve] +
                [p[1] for p in actual_pairs]
            )
            max_power = max(all_power, default=1.0) * 1.05

            # Build SVG curves
            svg_curves: list[SVGCurve] = []

            # 1. Envelope band (min/max as polygon fill)
            envelope_band_points = (
                max_points +
                list(reversed(min_points))
            )
            svg_curves.append(SVGCurve(
                points=envelope_band_points,
                color="#3498db",
                opacity=0.3,
                stroke_width=0,
                is_polygon=True,
            ))

            # 2. Average curve (darker blue line)
            svg_curves.append(SVGCurve(
                points=avg_points,
                color="#3498db",
                opacity=1.0,
                stroke_width=4
            ))

            # 3. Actual cycle (orange line)
            svg_curves.append(SVGCurve(
                points=actual_pairs,
                color="#f39c12",
                opacity=0.95,
                stroke_width=3
            ))

            # Get profile info for title
            profile = self.get_profile(profile_name)
            avg_duration = (
                profile.get("avg_duration", 0) / 60.0
                if profile
                else max_time / 60.0
            )
            avg_energy = (
                profile.get("avg_energy")
                if profile
                else envelope.get("avg_energy", 0)
            )

            title = (
                f"Power Profile Comparison: {profile_name} "
                f"({avg_duration:.0f}m, ~{avg_energy:.2f}kWh)"
            )

            # Create SVG using generic generator
            svg = _generate_generic_svg(
                title=title,
                curves=svg_curves,
                width=width,
                height=height,
                max_x_override=max_time,
                max_y_override=max_power
            )

            # Add a single-row legend below the chart
            if svg:
                legend_height = 34
                total_height = height + legend_height
                svg = svg.replace(
                    f'viewBox="0 0 {width} {height}"',
                    f'viewBox="0 0 {width} {total_height}"',
                    1
                )
                ly = height + 22  # Vertical mid-line for all legend items
                legend = (
                    f'<!-- Legend (single row) -->\n'
                    f'<g>\n'
                    # Item 1: band swatch
                    f'  <rect x="65" y="{ly - 11}" width="28" height="16" '
                    f'fill="#3498db" fill-opacity="0.35" stroke="#3498db" stroke-width="1.5" />\n'
                    f'  <text x="102" y="{ly + 4}" fill="#aaa" font-size="18">'
                    f'Expected range</text>\n'
                    # Item 2: avg line
                    f'  <line x1="390" y1="{ly}" x2="418" y2="{ly}" '
                    f'stroke="#3498db" stroke-width="4" />\n'
                    f'  <text x="428" y="{ly + 4}" fill="#aaa" font-size="18">'
                    f'Average profile</text>\n'
                    # Item 3: actual line
                    f'  <line x1="720" y1="{ly}" x2="748" y2="{ly}" '
                    f'stroke="#f39c12" stroke-width="3" />\n'
                    f'  <text x="758" y="{ly + 4}" fill="#aaa" font-size="18">'
                    f'This cycle (actual)</text>\n'
                    f'</g>\n'
                )
                return svg.replace("</svg>", legend + "</svg>", 1)

            return svg

        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.exception("Error generating feedback comparison SVG")
            return None

    def generate_feedback_multi_profile_svg(
        self,
        profile_names: list[str],
        detected_profile: str,
        actual_cycle: CycleDict,
        chart_title_prefix: str = "Profile Comparison",
        actual_cycle_label: str = "This cycle (actual)",
    ) -> str | None:
        """Generate a single SVG overlaying all profiles' avg curves with the actual cycle.

        The detected profile also shows a min/max envelope band.
        Each profile gets a distinct colour; the actual cycle is orange.
        A compact multi-column legend is appended below the chart.
        """
        try:
            # Colours: orange (#f39c12) is reserved for the actual cycle
            palette = [
                "#3498db",  # blue   – detected profile (matches envelope tint)
                "#2ecc71",  # green
                "#9b59b6",  # purple
                "#e74c3c",  # red
                "#1abc9c",  # teal
                "#f1c40f",  # yellow
                "#36a2eb",  # sky-blue
                "#8e44ad",  # dark purple
                "#16a085",  # dark teal
                "#c0392b",  # dark red
            ]

            # Load envelope data for every profile that has one
            profile_envs: dict[str, JSONDict] = {}
            for pname in profile_names:
                env = self.get_envelope(pname)
                if env and env.get("time_grid") and env.get("avg"):
                    profile_envs[pname] = env

            if not profile_envs:
                return None

            # Decompress actual cycle power data (handles both ISO-timestamp and offset formats)
            actual_pairs = decompress_power_data(actual_cycle)

            if len(actual_pairs) < 3:
                return None

            # Global bounds
            max_time = actual_pairs[-1][0]
            for env in profile_envs.values():
                tg = env.get("time_grid", [])
                if tg:
                    max_time = max(max_time, tg[-1])

            all_power: list[float] = [p[1] for p in actual_pairs]
            for env in profile_envs.values():
                all_power += [p[1] for p in env.get("max", [])]
                all_power += [p[1] for p in env.get("avg", [])]
            max_power = max(all_power, default=1.0) * 1.05

            # Canvas
            width, height = 1200, 450
            padding_x, padding_y = 60, 45
            graph_w = width - 2 * padding_x
            graph_h = height - 2 * padding_y

            def _x(t: float) -> str:
                return f"{padding_x + (t / max_time) * graph_w:.1f}" if max_time > 0 else str(padding_x)

            def _y(p: float) -> str:
                return f"{height - padding_y - (p / max_power) * graph_h:.1f}" if max_power > 0 else str(height - padding_y)

            # Assign colours; detected profile always gets palette[0]
            colors: dict[str, str] = {}
            color_idx = 1
            if detected_profile in profile_envs:
                colors[detected_profile] = palette[0]
            for pname in profile_names:
                if pname in profile_envs and pname != detected_profile:
                    colors[pname] = palette[color_idx % len(palette)]
                    color_idx += 1

            elems: list[str] = []

            # Background + axes
            elems.append(
                f'<rect x="0" y="0" width="{width}" height="{height}" fill="#1c1c1c" />'
            )
            elems.append(
                f'<line x1="{padding_x}" y1="{height - padding_y}" '
                f'x2="{width - padding_x}" y2="{height - padding_y}" stroke="#444" stroke-width="2" />'
            )
            elems.append(
                f'<line x1="{padding_x}" y1="{padding_y}" '
                f'x2="{padding_x}" y2="{height - padding_y}" stroke="#444" stroke-width="2" />'
            )
            elems.append(
                f'<text x="{padding_x}" y="{padding_y - 15}" fill="#aaa" font-size="20">{int(max_power)}W</text>'
            )
            elems.append(
                f'<text x="{width - padding_x}" y="{height - 10}" fill="#aaa" font-size="20" '
                f'text-anchor="middle">{int(max_time / 60)}m</text>'
            )
            elems.append(
                f'<text x="{width / 2:.0f}" y="{padding_y - 15}" fill="#fff" font-size="26" '
                f'text-anchor="middle" font-weight="bold">{chart_title_prefix}: {detected_profile}</text>'
            )

            # Detected-profile envelope band (drawn first, behind all lines)
            if detected_profile in profile_envs:
                env = profile_envs[detected_profile]
                max_c = env.get("max", [])
                min_c = env.get("min", [])
                if max_c and min_c:
                    fwd = " ".join(f"{_x(p[0])},{_y(p[1])}" for p in max_c)
                    rev = " ".join(f"{_x(p[0])},{_y(p[1])}" for p in reversed(min_c))
                    band_color = colors.get(detected_profile, palette[0])
                    elems.append(
                        f'<polygon points="{fwd} {rev}" fill="{band_color}" fill-opacity="0.2" stroke="none" />'
                    )

            # Average lines for every profile
            for pname in profile_names:
                if pname not in profile_envs:
                    continue
                avg_c = profile_envs[pname].get("avg", [])
                if not avg_c:
                    continue
                color = colors.get(pname, "#aaa")
                pts = " ".join(f"{_x(p[0])},{_y(p[1])}" for p in avg_c)
                sw = 4 if pname == detected_profile else 2
                elems.append(
                    f'<polyline points="{pts}" fill="none" stroke="{color}" '
                    f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" />'
                )

            # Actual cycle on top
            actual_pts = " ".join(f"{_x(p[0])},{_y(p[1])}" for p in actual_pairs)
            elems.append(
                f'<polyline points="{actual_pts}" fill="none" stroke="#f39c12" '
                f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />'
            )

            # Legend (compact multi-column below the chart)
            legend_items: list[tuple[str, str, int]] = []  # (color, label, stroke_width)
            for pname in profile_names:
                if pname not in profile_envs:
                    continue
                color = colors.get(pname, "#aaa")
                label = f"\u2605 {pname}" if pname == detected_profile else pname
                legend_items.append((color, label, 4 if pname == detected_profile else 2))
            legend_items.append(("#f39c12", actual_cycle_label, 3))

            items_per_row = 3
            col_w = (width - 2 * padding_x) // items_per_row
            row_h = 34
            n_rows = (len(legend_items) + items_per_row - 1) // items_per_row
            legend_h = n_rows * row_h + 22
            total_height = height + legend_h

            leg_elems: list[str] = []
            for i, (color, label, sw) in enumerate(legend_items):
                col = i % items_per_row
                row = i // items_per_row
                lx = padding_x + col * col_w
                ly = height + 26 + row * row_h
                leg_elems.append(
                    f'<line x1="{lx}" y1="{ly}" x2="{lx + 32}" y2="{ly}" '
                    f'stroke="{color}" stroke-width="{sw}" />'
                )
                max_chars = 22
                display = label[:max_chars] + "\u2026" if len(label) > max_chars else label
                leg_elems.append(
                    f'<text x="{lx + 42}" y="{ly + 6}" fill="#aaa" font-size="22">{display}</text>'
                )

            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {total_height}" '
                'style="background-color: #1c1c1c; font-family: sans-serif;">\n'
                + "\n".join(elems)
                + "\n<!-- Legend -->\n"
                + "\n".join(leg_elems)
                + "\n</svg>"
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Error generating multi-profile comparison SVG: %s", e)
            return None

    def get_match_candidates_summary(
        self, match_result: MatchResult, limit: int = 3
    ) -> list[dict[str, Any]]:
        """Extract top candidates from query result for UI display.

        Args:
            match_result: MatchResult from profile matching
            limit: Number of top candidates to return

        Returns:
            List of dicts with keys: profile_name, confidence_pct, mae, correlation, duration_ratio
        """
        candidates: list[dict[str, Any]] = []

        for candidate in match_result.ranking[:limit]:
            try:
                confidence_pct = round(candidate.get("score", 0.0) * 100, 1)
                metrics = candidate.get("metrics", {})
                mae = round(metrics.get("mae", 0.0), 2)
                corr = round(metrics.get("corr", 0.0), 3)

                profile_duration = candidate.get("profile_duration", 0.0)
                actual_duration = match_result.expected_duration
                duration_ratio = (
                    round((actual_duration / profile_duration - 1.0) * 100, 1)
                    if profile_duration > 0
                    else 0.0
                )

                candidates.append({
                    "profile_name": candidate.get("name", "Unknown"),
                    "confidence_pct": confidence_pct,
                    "mae": mae,
                    "correlation": corr,
                    "duration_ratio": duration_ratio,  # ±% from expected
                })
            except (TypeError, ValueError, KeyError):
                continue

        return candidates

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

            s_segments = resample_uniform(s_ts, s_p, dt_s=dt, gap_s=21600.0)
            if not s_segments:
                return None

            sample_seg = max(s_segments, key=lambda s: len(s.power))

            # Store
            self._cached_sample_segments[key] = sample_seg
            return sample_seg
        except Exception as e: # pylint: disable=broad-exception-caught
            self._logger.warning("Error caching sample segment %s: %s", cycle_id, e)
            return None

    async def async_match_profile(
        self,
        current_power_data: list[tuple[str, float]] | list[tuple[datetime, float]] | list[tuple[float, float]] | list[list[float]],
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
            first_elem = current_power_data[0][0]
            if isinstance(first_elem, datetime):
                # datetime objects: compute relative timestamps
                t_start = first_elem.timestamp()
                ts_arr = np.array([(x[0].timestamp() - t_start) for x in cast(list[tuple[datetime, float]], current_power_data)])
            elif isinstance(first_elem, (int, float)):
                # Already offset timestamps (from compressed format)
                ts_arr = np.array([float(x[0]) for x in cast(list[tuple[float, float]], current_power_data)])
            else:
                # ISO format strings
                t_start = datetime.fromisoformat(first_elem).timestamp()
                ts_arr = np.array(
                    [
                        (datetime.fromisoformat(x[0]).timestamp() - t_start)
                        for x in cast(list[tuple[str, float]], current_power_data)
                    ]
                )

            p_arr = np.array([float(x[1]) for x in current_power_data])

            # Resample current
            segments, used_dt = resample_adaptive(ts_arr, p_arr, min_dt=5.0, gap_s=21600.0)
            if not segments:
                return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)
            current_seg = max(segments, key=lambda s: len(s.power))
            if len(current_seg.power) < 12:
                return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

            current_power_list = current_seg.power.tolist()

            # Prepare Snapshots
            snapshots: list[dict[str, Any]] = []
            skipped_profiles: list[str] = []
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
                self._logger.debug(
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
            self._logger.error("Preparation for async match failed: %s", e)
            return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

        # 2. Run Heavy Logic in Executor
        candidates = await self.hass.async_add_executor_job(
            analysis.compute_matches_worker,
            current_power_list,
            current_duration,
            cast(Any, snapshots),
            config
        )

        # 3. Process Result (Main Thread)
        if not candidates:
            profiles_count = len(self._data.get("profiles", {}))
            snapshots_count = len(snapshots) if 'snapshots' in dir() else 0
            self._logger.debug(
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
        if best.get("name"):
            # Always resolve phase for the matched profile so phase sensors can
            # show user-assigned phase names even when confidence is moderate.
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
        snapshots: list[dict[str, Any]] = []
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
            p_list, duration, cast(Any, snapshots), config
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
        self,
        profile_name: str,
        current_power_data: list[list[float]] | list[tuple[Any, ...]],
    ) -> tuple[bool, float, float]:
        """
        Verify if the current power trace aligns with an expected low-power region in the envelope.
        Returns: (is_confirmed_low_power, mapped_envelope_time, mapped_envelope_power)
        """
        envelope = self.get_envelope(profile_name)
        if not envelope or not envelope.get("avg") or not current_power_data:
            return False, 0.0, 9999.0

        # Extract envelope curves
        # "avg" can be list of [t, p] (new) or [p, ...] (legacy)
        env_avg_raw = envelope.get("avg", [])
        if not env_avg_raw:
            return False, 0.0, 9999.0

        try:
            # Handle both formats: [[t, y], ...] (new) or [y, ...] (legacy)
            if isinstance(env_avg_raw[0], (list, tuple)) and len(env_avg_raw[0]) >= 2:
                # New format: [[t, y], ...]
                env_points = cast(list[list[Any] | tuple[Any, ...]], env_avg_raw)
                env_time = [float(p[0]) for p in env_points]
                env_power = [float(p[1]) for p in env_points]
            else:
                # Legacy format: [y, ...]
                env_values = cast(list[float | int], env_avg_raw)
                env_power = [float(p) for p in env_values]
                # Reconstruct time grid from envelope if available, or assume 60s intervals
                env_time_raw = envelope.get("time_grid")
                env_time = cast(list[float], env_time_raw) if isinstance(env_time_raw, list) else None
                if not env_time or len(env_time) != len(env_power):
                    target_dur = float(envelope.get("target_duration", 0.0) or 0.0)
                    if target_dur > 0:
                        env_time = cast(list[float], np.linspace(0, target_dur, len(env_power)).tolist())
                    else:
                        env_time = [float(i * 60) for i in range(len(env_power))]
        except (TypeError, ValueError, IndexError) as e:
            first_type_name = type(env_avg_raw[0]).__name__ if env_avg_raw else "None"
            self._logger.error(
                "Malformed envelope 'avg' data for %s. Type: %s, Length: %d, Error: %s",
                profile_name, first_type_name, len(env_avg_raw), e
            )
            return False, 0.0, 9999.0

        try:
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

        phases_sorted = sorted(
            phases,
            key=lambda p: float(p.get("start", 0)),
        )

        for phase in phases_sorted:
            p_start = phase.get("start", 0)
            p_end = phase.get("end", 0)
            if p_start <= duration <= p_end:
                return str(phase.get("name", "Unknown"))

        # If duration is outside explicit bounds, keep a phase label anyway so
        # entities avoid falling back to generic running/starting states.
        if phases_sorted:
            if duration < float(phases_sorted[0].get("start", 0)):
                return str(phases_sorted[0].get("name", "Unknown"))
            return str(phases_sorted[-1].get("name", "Unknown"))

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
            duration_std_dev = envelope.get("duration_std_dev") if envelope else None

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
                    "duration_std_dev": duration_std_dev,
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
                # Label the reference cycle with the new profile so that
                # statistics are immediately populated after creation.
                if not cycle.get("profile_name"):
                    cycle["profile_name"] = name
        elif avg_duration is not None and avg_duration > 0:
            profile_data = {
                "avg_duration": float(avg_duration),
            }

        # Create profile with minimal data (will be updated when cycles are labeled)
        profile_data.setdefault("phases", [])
        self._data.setdefault("profiles", {})[name] = profile_data

        # Build the envelope from any already-labeled cycles (e.g. reference cycle above)
        await self.async_rebuild_envelope(name)

        await self.async_save()
        self._logger.info("Created standalone profile '%s'", name)

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
            self._logger.info(
                "Updated baseline duration for '%s' to %ss",
                target_name,
                avg_duration,
            )

        # Update cycles and feedback if renamed
        count = 0
        if renamed:
            # 1. Update past cycles
            for cycle in self._data.get("past_cycles", []):
                if cycle.get("profile_name") == old_name:
                    cycle["profile_name"] = new_name
                    count += 1

            # 2. Update pending feedback
            pending = self.get_pending_feedback()
            for req in pending.values():
                if req.get("detected_profile") == old_name:
                    req["detected_profile"] = new_name

            # 3. Update feedback history
            history = self.get_feedback_history()
            for record in history.values():
                if record.get("original_detected_profile") == old_name:
                    record["original_detected_profile"] = new_name
                if record.get("corrected_profile") == old_name:
                    record["corrected_profile"] = new_name

            self._logger.info(
                "Renamed profile '%s' to '%s', updated %s cycles and associated feedback",
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
        self._logger.info("Deleted profile '%s', %s %s cycles", name, action, count)
        return count

    async def clear_all_data(self) -> None:
        """Clear all profiles and cycle data."""
        self._data["past_cycles"] = []
        self._data["profiles"] = {}
        await self.async_save()
        self._logger.info("Cleared all WashData storage")

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
        self._logger.info("Assigned profile '%s' to cycle %s", profile_name, cycle_id)
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
                        self._logger.info(
                            "Relabeled cycle %s: '%s' -> '%s' (confidence: %.2f)",
                            cycle["id"],
                            current_label,
                            result.best_profile,
                            result.confidence,
                        )
                else:
                    cycle["profile_name"] = result.best_profile
                    stats["labeled"] += 1
                    self._logger.info(
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

        self._logger.info(
            "Auto-labeling complete: %s labeled, %s relabeled, %s skipped",
            stats["labeled"],
            stats["relabeled"],
            stats["skipped"],
        )
        return stats

    def _decompress_power_data(self, cycle: CycleDict) -> list[tuple[float, float]]:
        """Decompress cycle power data for matching (wrapper)."""
        return [(float(offset), float(power)) for offset, power in decompress_power_data(cycle)]

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
                self._logger.warning("Failed to migrate cycle %s: %s", cycle.get("id"), e)
                continue

        if migrated > 0:
            self._logger.info("Migrated %s cycles to compressed format", migrated)
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
        new_ids: list[str] = []
        original_profile = cycle.get("profile_name")
        start_dt_base_parsed = dt_util.parse_datetime(cycle["start_time"])
        if not start_dt_base_parsed:
            # Should not happen as analyze checked it, but safety
            self._logger.warning("Failed to parse start time during split apply for %s", cycle_id)
            return [cycle_id]

        start_dt_base: datetime = start_dt_base_parsed
        # Use decompress_power_data which handles all format variations
        p_data_tuples = self._decompress_power_data(cycle)

        if not p_data_tuples:
            self._logger.warning("Failed to decompress data during split for %s", cycle_id)
            return [cycle_id]

        # Convert to relative seconds for array logic.
        # _decompress_power_data returns (offset_seconds, power).

        points: list[tuple[float, float]] = []
        for offset_seconds, val in p_data_tuples:
            points.append((float(offset_seconds), float(val)))

        for seg_start, seg_end in seg_ranges:
            # Construct new cycle logic
            seg_dur = seg_end - seg_start
            new_cycle_start = start_dt_base + timedelta(seconds=seg_start)
            new_cycle_start_ts = new_cycle_start.timestamp()

            # Extract points
            p_data_abs: list[list[float]] = []
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

            new_cycle: dict[str, Any] = {
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
        self._logger.debug("Running maintenance: cleanup_orphaned_profiles")
        stats["cleaned_profiles"] = self.cleanup_orphaned_profiles()

        # 2. Retention
        self._logger.debug("Running maintenance: async_enforce_retention")
        await self.async_enforce_retention()

        # 3. Save
        self._logger.debug("Maintenance complete, saving")
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
        self._logger.info(
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
            self._logger.info(
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
            self._logger.info(
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

        return {
            "entry_data": payload.get("entry_data", {}),
            "entry_options": payload.get("entry_options", {}),
        }


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

    def get_cycle_power_data(self, cycle_id: str) -> list[tuple[float, float]]:
        """Return decompressed power data for a cycle as [(offset_s, watts), ...].

        Returns an empty list if the cycle is not found or has no power data.
        """
        cycle = next(
            (c for c in self.get_past_cycles() if c.get("id") == cycle_id), None
        )
        if cycle is None:
            return []
        return self._decompress_power_data(cycle)

    async def trim_cycle_power_data(
        self,
        cycle_id: str,
        new_start_s: float,
        new_end_s: float,
    ) -> bool:
        """Trim a cycle's power_data to the window [new_start_s, new_end_s].

        Offsets are renormalized so the kept segment starts at 0.0.
        The cycle's ``duration``, ``signature``, and ``sampling_interval`` are
        recomputed from the trimmed data.

        Returns True if successful, False if the cycle was not found or the
        resulting data is empty.
        """
        cycles = cast(list[CycleDict], self._data.get("past_cycles", []))
        cycle = next((c for c in cycles if c.get("id") == cycle_id), None)
        if cycle is None:
            return False

        p_data = self._decompress_power_data(cycle)
        if not p_data:
            return False

        new_start_s = max(0.0, float(new_start_s))
        new_end_s = float(new_end_s)

        kept = sorted(
            (
                (offset, power)
                for offset, power in p_data
                if new_start_s <= offset <= new_end_s
            ),
            key=lambda x: x[0],
        )
        if not kept:
            return False

        # Re-normalize offsets so the trimmed segment starts at 0.0
        base = kept[0][0]
        renorm: list[list[float]] = [
            [round(offset - base, 2), power] for offset, power in kept
        ]

        # Advance start_time when trimming from the front
        if base > 0:
            start_ts = _value_to_timestamp(cycle.get("start_time"))
            if start_ts is not None:
                cycle["start_time"] = dt_util.utc_from_timestamp(
                    start_ts + base
                ).isoformat()

        # Recompute sampling interval
        if len(renorm) > 1:
            offsets_arr = np.array([r[0] for r in renorm], dtype=float)
            intervals = np.diff(offsets_arr)
            pos = intervals[intervals > 0]
            sampling_interval = float(np.median(pos)) if len(pos) > 0 else 1.0
        else:
            sampling_interval = 1.0

        # Recompute signature
        ts_arr = np.array([r[0] for r in renorm], dtype=float)
        p_arr = np.array([r[1] for r in renorm], dtype=float)
        if len(ts_arr) > 1:
            sig = compute_signature(ts_arr, p_arr)
            cycle["signature"] = dataclasses.asdict(sig)
        else:
            cycle["signature"] = None

        new_duration = round(renorm[-1][0], 1) if renorm else 0.0
        cycle["power_data"] = renorm
        cycle["sampling_interval"] = round(sampling_interval, 1)
        cycle["duration"] = new_duration

        # Keep end_time consistent with the updated start_time and duration
        new_start_ts = _value_to_timestamp(cycle.get("start_time"))
        if new_start_ts is not None:
            cycle["end_time"] = dt_util.utc_from_timestamp(
                new_start_ts + new_duration
            ).isoformat()

        # Clear manual_duration override — trimmed duration is now authoritative
        cycle.pop("manual_duration", None)

        # Invalidate cached sample segments for this cycle so future lookups
        # are recomputed from the trimmed data
        stale_keys = [k for k in self._cached_sample_segments if k[0] == cycle_id]
        for k in stale_keys:
            del self._cached_sample_segments[k]

        # Rebuild envelope for the associated profile
        profile_name = cycle.get("profile_name")
        if profile_name:
            await self.async_rebuild_envelope(profile_name)

        await self.async_save()
        return True

    def analyze_split_sync(
        self, cycle: CycleDict, min_gap_s: int = 900, idle_power: float = 2.0
    ) -> list[tuple[float, float]]:
        """Analyze cycle for potential splits (sync for executor)."""
        p_data = self._decompress_power_data(cycle)
        if not p_data:
            return []

        # Parse all points to (rel_t, power)
        points: list[tuple[float, float]] = []
        for offset_seconds, val in p_data:
            points.append((float(offset_seconds), float(val)))

        if not points:
            return []

        valid_segments: list[tuple[float, float]] = []
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

        self._logger.debug(
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

        new_ids: list[str] = []
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
        points: list[tuple[float, float]] = []
        for offset_seconds, val in p_data_tuples:
            points.append((float(offset_seconds), float(val)))

        # Create new cycles
        for seg in segments:
            if isinstance(seg, (list, tuple)):
                seg_tuple = cast(tuple[Any, ...] | list[Any], seg)
                seg_start = float(seg_tuple[0])
                seg_end = float(seg_tuple[1])
                seg_profile = None
            else:
                seg_start = float(seg["start"])
                seg_end = float(seg["end"])
                seg_profile = seg.get("profile")

            seg_dur = seg_end - seg_start
            new_cycle_start = start_dt_base_parsed + timedelta(seconds=seg_start)
            new_cycle_start_ts = new_cycle_start.timestamp()

            # Extract points for this segment
            p_data_abs: list[list[float]] = []

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
            new_cycle: dict[str, Any] = {
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
        self._logger.info("Interactive Split Applied to %s -> %s", cycle_id, new_ids)
        return new_ids

    async def apply_merge_interactive(
        self, cycle_ids: list[str], target_profile: str | None
    ) -> str | None:
        """
        Merge multiple past cycles into a single cycle record, filling gaps between traces with short zero-power segments.
        
        Parameters:
            cycle_ids (list[str]): Unordered set of past-cycle IDs to merge; at least two IDs are required.
                The function internally sorts cycles by start_time and mutates the chronologically earliest cycle.
            target_profile (str | None): Profile name to assign to the merged cycle, or `None` to leave unlabeled.

        Description:
            When successful, this sorts the provided cycles by start_time and replaces the earliest cycle with the merged cycle,
            removing the other consumed cycles and updating related metadata.
        
            Side effects:
            - Updates the store's past_cycles (removes consumed cycles and replaces the first cycle with the merged record).
            - Clears any `manual_duration` override on the resulting cycle.
            - Updates `sample_cycle_id` references in profiles that pointed to removed cycles.
            - Attempts to recompute and store the merged cycle's signature.
            - Persists changes to storage and triggers envelope rebuilds for affected profiles.
        
        Returns:
            merged_id (str | None): The new merged cycle's ID if the merge was applied, `None` if the merge could not be performed.
        """
        if len(cycle_ids) < 2:
            return None

        cycles = self.get_past_cycles()
        target_cycles = [c for c in cycles if c.get("id") in cycle_ids]

        if len(target_cycles) != len(cycle_ids):
            return None

        # Sort by time — use timestamp comparison to handle mixed timezone offsets correctly
        def _cycle_start_ts(c: CycleDict) -> float:
            """
            Return the cycle's start time as a POSIX timestamp, or positive infinity if missing or unparsable.
            
            Parameters:
                c (CycleDict): Cycle dictionary; expected to contain a 'start_time' value (ISO string or datetime) parseable by dt_util.parse_datetime.
            
            Returns:
                float: Seconds since the epoch for the cycle's start time, or float('inf') when the start time is absent or cannot be parsed.
            """
            dt = dt_util.parse_datetime(str(c.get("start_time", "")))
            return dt.timestamp() if dt is not None else float("inf")

        target_cycles.sort(key=_cycle_start_ts)

        # Collect affected profiles for envelope rebuild
        affected_profiles: set[str] = set()
        for c in target_cycles:
            if c.get("profile_name"):
                affected_profiles.add(c["profile_name"])
        if target_profile:
            affected_profiles.add(target_profile)

        # We modify the first cycle (c1) to become the merged one
        c1 = target_cycles[0]
        ids_to_remove: list[str] = []

        # Base setup
        c1_start_dt = dt_util.parse_datetime(c1["start_time"])
        if not c1_start_dt:
            return None

        # Helper to get parsed points from a cycle
        def get_points(cy: CycleDict) -> list[tuple[float, float, float]]:
            # content: [(timestamp, offset, power)]
            raw = self._decompress_power_data(cy)
            res: list[tuple[float, float, float]] = []
            if not raw:
                return []
            base_dt = dt_util.parse_datetime(cy["start_time"])
            if base_dt is None:
                return []
            base_t = base_dt.timestamp()
            for offset_seconds, val in raw:
                t_abs = base_t + float(offset_seconds)
                res.append((t_abs, float(offset_seconds), float(val)))
            return res

        # Start with C1 points
        merged_points_abs: list[list[float]] = [] # [timestamp, power]

        # Add C1 points
        c1_pts = get_points(c1)
        for t_abs, _, p in c1_pts:
            merged_points_abs.append([t_abs, p])

        # Use the maximum t_abs seen so far (guards against out-of-order or corrupted points)
        last_t_abs = max((pt[0] for pt in c1_pts), default=c1_start_dt.timestamp())

        # Iterate others
        max_power = c1.get("max_power", 0)

        for next_c in target_cycles[1:]:
            c_start_dt = dt_util.parse_datetime(str(next_c.get("start_time", "")))
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

            # Append points; track the running maximum to guard against reversed/corrupt data
            for t_abs, _, p in c_pts:
                merged_points_abs.append([t_abs, p])
                if t_abs > last_t_abs:
                    last_t_abs = t_abs

            max_power = max(max_power, next_c.get("max_power", 0))
            ids_to_remove.append(next_c["id"])

        # Derive merged end time from power data when available; otherwise fall back to
        # the end_time field of the last cycle (handles cycles without recorded power data).
        if merged_points_abs:
            # Use the maximum absolute timestamp from all collected data points
            last_t_abs = max(pt[0] for pt in merged_points_abs)
            final_end_dt = dt_util.utc_from_timestamp(last_t_abs)
        else:
            last_cycle = target_cycles[-1]
            fallback_end_dt = dt_util.parse_datetime(str(last_cycle.get("end_time", "")))
            if fallback_end_dt is not None:
                final_end_dt = fallback_end_dt
            else:
                final_end_dt = c1_start_dt

        new_dur = (final_end_dt - c1_start_dt).total_seconds()

        c1["end_time"] = final_end_dt.isoformat()
        c1["duration"] = round(new_dur, 1)
        c1["max_power"] = max_power
        c1["profile_name"] = target_profile
        # Remove manual_duration override so the freshly computed duration is shown
        c1.pop("manual_duration", None)

        # Generate new compressed power_data [offset, power]
        new_power_data: list[list[float]] = []
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
            ts_arr = np.array([pt[0] for pt in new_power_data], dtype=float)
            p_arr = np.array([pt[1] for pt in new_power_data], dtype=float)
            if len(ts_arr) > 1:
                sig = compute_signature(ts_arr, p_arr)
                c1["signature"] = dataclasses.asdict(sig)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.warning("Failed to update signature for merged cycle %s: %s", new_id, e)

        await self.async_save()
        self._logger.info("Interactive Merge Applied: %s -> %s", cycle_ids, new_id)

        # Rebuild envelopes for all affected profiles
        for p_name in affected_profiles:
            await self.async_rebuild_envelope(p_name)

        return new_id

    def generate_interactive_split_svg(
        self,
        cycle_id: str,
        segments: list[tuple[float, float]],
        width: int = 600,
        height: int = 300,
        title_prefix: str = "Split Preview",
        unlabeled_text: str = "Unlabeled",
    ) -> str:
        """Generate SVG for split preview."""
        cycle = next((c for c in self.get_past_cycles() if c["id"] == cycle_id), None)
        if not cycle:
            return ""

        p_data = self._decompress_power_data(cycle)
        if not p_data:
            return ""

        start_dt = dt_util.parse_datetime(cycle["start_time"])
        if start_dt is None:
            return ""
        points: list[tuple[float, float]] = []
        for offset_seconds, val in p_data:
            points.append((float(offset_seconds), float(val)))

        curves: list[SVGCurve] = [SVGCurve(points=points, color="#9E9E9E", opacity=0.5)] # Base ghost
        markers: list[dict[str, Any]] = []

        # Highlight Segments
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, (seg_start, seg_end) in enumerate(segments):
            seg_pts = [(t, p) for t, p in points if seg_start <= t <= seg_end]
            if seg_pts:
                color = colors[i % len(colors)]
                curves.append(SVGCurve(points=seg_pts, color=color, stroke_width=2))
                markers.append({"x": seg_start, "label": f"S{i+1}", "color": color})

        return _generate_generic_svg(
            f"{title_prefix}: {cycle.get('profile_name') or unlabeled_text}",
            curves,
            width,
            height,
            markers=markers,
        )

    def generate_interactive_merge_svg(
        self,
        cycle_ids: list[str],
        width: int = 600,
        height: int = 300,
        title: str = "Merge Preview",
        no_data_label: str | None = None,
    ) -> str:
        """
        Generate an SVG preview that overlays power traces from the specified past cycles to illustrate the result of merging them.
        
        Cycles are ordered by their parsed start_time and each cycle's power data is aligned to the earliest cycle start to form overlaid curves.
        
        Parameters:
            cycle_ids (list[str]): IDs of past cycles to include in the preview.
            width (int): Width of the generated SVG in pixels.
            height (int): Height of the generated SVG in pixels.
            title (str): Title text shown in the SVG header.
            no_data_label (str | None): Message rendered in the placeholder SVG when cycles are
                present but contain no recorded power data. Defaults to None (empty message).

        Returns:
            str: SVG markup for the merge preview. Returns an empty string if no valid cycles or
            if the first cycle's start_time cannot be parsed. If cycles are present but none
            contain power data, returns a placeholder SVG using no_data_label as the descriptive
            message instead of a fixed string.
        """
        cycles = [c for c in self.get_past_cycles() if c["id"] in cycle_ids]

        def _sort_ts(c: CycleDict) -> float:
            """
            Provide a numeric sort key for a cycle by converting its `start_time` to a UNIX timestamp.
            
            Parameters:
                c (CycleDict): Cycle mapping that may contain a `start_time` value in any parseable datetime form.
            
            Returns:
                float: UNIX timestamp in seconds parsed from `start_time`, or `float('inf')` when `start_time` is missing or cannot be parsed so the cycle sorts after valid-dated cycles.
            """
            dt = dt_util.parse_datetime(str(c.get("start_time", "")))
            return dt.timestamp() if dt is not None else float("inf")

        cycles.sort(key=_sort_ts)

        if not cycles:
            return ""

        first_start_dt = dt_util.parse_datetime(str(cycles[0].get("start_time", "")))
        if first_start_dt is None:
            return ""
        first_start = first_start_dt.timestamp()
        curves: list[SVGCurve] = []

        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

        for i, c in enumerate(cycles):
            p_data = self._decompress_power_data(c)
            if not p_data:
                continue
            points: list[tuple[float, float]] = []
            cycle_start_raw = c.get("start_time")
            if not isinstance(cycle_start_raw, str):
                continue
            cycle_start_dt = dt_util.parse_datetime(cycle_start_raw)
            if cycle_start_dt is None:
                continue
            cycle_start = cycle_start_dt.timestamp()
            for offset_seconds, val in p_data:
                rel_t = (cycle_start + float(offset_seconds)) - first_start
                points.append((rel_t, float(val)))

            if points:
                curves.append(SVGCurve(points=points, color=colors[i % len(colors)], stroke_width=2))

        if not curves:
            # No power data available — return a placeholder SVG with a message
            safe_title = html.escape(title)
            safe_label = html.escape(no_data_label or "")
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
                f'style="background-color: #1c1c1c; font-family: sans-serif;">'
                f'<rect x="0" y="0" width="{width}" height="{height}" fill="#1c1c1c" />'
                f'<text x="{width // 2}" y="{height // 2 - 10}" fill="#aaa" font-size="16" '
                f'text-anchor="middle">{safe_title}</text>'
                f'<text x="{width // 2}" y="{height // 2 + 14}" fill="#666" font-size="13" '
                f'text-anchor="middle">{safe_label}</text>'
                f'</svg>'
            )

        return _generate_generic_svg(html.escape(title), curves, width, height)
