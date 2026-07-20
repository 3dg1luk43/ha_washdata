# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Profile storage and matching logic for WashData."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import os
import re
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, TypeAlias, cast

import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    CLUSTER_RESAMPLE_N,
    CLUSTER_SHAPE_SIMILARITY_THRESHOLD,
    GROUP_MIN_COHESION,
    MAINTENANCE_EVENT_TYPES,
    MAINTENANCE_RECENT_SUPPRESS_DAYS,
    MATCH_AMBIGUITY_MARGIN,
    PHASE_CONSISTENCY_MIN_CYCLES,
    PHASE_PROFILE_MIN_CYCLES,
    PHASE_HEAT_CV_WARN,
    PHASE_HEAT_OCC_MIXED_LO,
    PHASE_HEAT_OCC_MIXED_HI,
    REFERENCE_PROFILE_CURVE_POINTS,
    SHAPE_DRIFT_MIN_CYCLES,
    SHAPE_DRIFT_RESAMPLE_N,
    SHAPE_DRIFT_THRESHOLD,
    SMART_TERM_LANDSCAPE_RATIO,
    SMART_TERM_LANDSCAPE_MIN_SHAPE,
    STORAGE_KEY,
    STORAGE_VERSION,
    DEFAULT_MAX_PAST_CYCLES,
    DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
    DEFAULT_MAX_FULL_TRACES_UNLABELED,
    DEFAULT_DTW_BANDWIDTH,
)
from .features import compute_signature
from .signal_processing import resample_uniform, resample_adaptive, Segment, integrate_wh, energy_gap_threshold_s
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
from .phase_segmenter import phase_matching_live_supported, phase_model_for, segment_cycle
from .phase_match import (
    build_phase_profile,
    match_phase_profiles,
    phase_eta,
    phase_profile_from_dict,
    phase_profile_to_dict,
)
from .log_utils import DeviceLoggerAdapter

_LOGGER = logging.getLogger(__name__)

# Absolute peak-power floor below which a cycle trace is considered a
# mis-capture (e.g. the power sensor wasn't reporting), regardless of profile.
# Used together with a relative (10% of median peak) test in envelope rebuild
# and reference selection so degenerate cycles never become the matching template.
_DEGENERATE_POWER_FLOOR = 15.0  # watts

JSONDict: TypeAlias = dict[str, Any]
CycleDict: TypeAlias = dict[str, Any]


def _is_recorded_cycle(cycle: dict[str, Any]) -> bool:
    """True when a cycle was produced by the manual recorder.

    Two independent signals, because the explicit marker did not always exist:

    1. **Explicit meta marker** (recent recorder builds): ``meta.source ==
       "recorder"`` and, equivalently, ``meta.original_samples`` (the pre-trim
       sample count). (In exported diagnostics ``meta.source`` is redacted, but
       the real stored value is ``"recorder"`` — the migration runs against the
       real store, not the export.)

    2. **Structural signature** (old recordings, ``meta`` is ``None``): the
       recorder builds the cycle dict directly (``ws_process_recording``) and
       never sets ``max_power`` or ``termination_reason``, whereas *every*
       auto-detected cycle goes through ``CycleDetector._finish_cycle`` which
       stamps ``max_power`` unconditionally (in every version — ``termination_reason``
       was added later, so old *auto* cycles can lack it but never lack
       ``max_power``). So a **completed** cycle missing *both* fields is a
       recording that predates the meta marker. Verified against real exports:
       this recovers old recordings that carry only ``meta: None`` — which the
       marker-only check silently missed.
    """
    meta = cycle.get("meta")
    if isinstance(meta, dict) and (
        meta.get("source") == "recorder" or "original_samples" in meta
    ):
        return True
    return (
        cycle.get("status") == "completed"
        and "max_power" not in cycle
        and "termination_reason" not in cycle
    )


def _flag_recorded_cycles_golden(cycles: list[dict[str, Any]]) -> int:
    """Mark manually-recorded cycles as golden references (recorded == golden).

    A recorded cycle (see :func:`_is_recorded_cycle`) is, by definition, a
    hand-picked clean example, so it becomes the profile's golden reference. The
    flag lives in the single ``ml_review`` dict (no duplicate ``recorded`` field).
    Idempotent: already-golden cycles are skipped, so it is safe to re-run from the
    migration and the diagnostics processing trigger.

    Returns the number of cycles newly flagged.
    """
    flagged = 0
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        review = cycle.get("ml_review")
        if isinstance(review, dict) and review.get("golden"):
            continue
        if not _is_recorded_cycle(cycle):
            continue
        review = dict(review) if isinstance(review, dict) else {}
        review["golden"] = True
        if not review.get("quality"):
            review["quality"] = "good"  # recorded cycles are clean by definition
        if not review.get("reviewed_at"):
            review["reviewed_at"] = dt_util.now().isoformat()
        cycle["ml_review"] = review
        flagged += 1
    return flagged


def _empty_ranking() -> list[dict[str, Any]]:
    """Typed default factory for ranking entries."""
    return []


def _parse_start_dt(value: Any) -> datetime | None:
    """Parse a start_time value (ISO string or numeric timestamp) into a datetime.

    Handles the case where legacy cycles stored numeric unix timestamps instead
    of ISO-formatted strings, which dt_util.parse_datetime cannot handle.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=dt_util.UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str) and value:
        parsed = dt_util.parse_datetime(value)
        if parsed is not None:
            return parsed
        try:
            return datetime.fromtimestamp(float(value), tz=dt_util.UTC)
        except (TypeError, ValueError):
            return None
    return None


def _parse_maintenance_dt(value: Any) -> datetime | None:
    """Parse a maintenance-log date (ISO date or datetime) into an aware datetime.

    Accepts both full ISO datetimes (as written by ``dt_util.now().isoformat()``)
    and bare ISO dates (``"2026-07-01"``) that the user may enter. Naive results are
    localised so recency/comparison math stays timezone-consistent. Never raises.
    """
    dt = _parse_start_dt(value)
    if dt is None and isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            dt = None
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=dt_util.DEFAULT_TIME_ZONE or dt_util.UTC)
    return dt


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
    is_prefix_ambiguous: bool = False

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



def _safe_file_size_kb(path: str) -> float:
    """Return the size of ``path`` in KiB, or 0.0 if it cannot be read.

    Runs blocking os.path calls; intended to be offloaded to the executor.
    """
    try:
        return os.path.getsize(path) / 1024 if os.path.exists(path) else 0.0
    except OSError:
        return 0.0


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


def earliest_sustained_quiet_offset(
    cycles: list[CycleDict],
    stop_threshold_w: float,
    min_quiet_span_s: float,
    min_clean_cycles: int,
) -> float | None:
    """Smallest elapsed offset (s) at which any *completed* cycle first shows a
    sustained (>= ``min_quiet_span_s``) near-zero (< ``stop_threshold_w``) span.

    This is the device's learned "we have never legitimately been quiet before
    this many seconds into a cycle" baseline, used by the opt-in terminal-drop
    detector: a hard cliff-to-~0 that begins *earlier* than this is anomalous and
    almost certainly a real stop rather than a soak pause.

    Only ``completed`` cycles seed the baseline - interrupted / force-stopped /
    terminal-drop cycles are exactly the anomalies we are trying to catch, so
    including them would poison the baseline and suppress detection. Returns
    ``None`` when there are fewer than ``min_clean_cycles`` completed cycles with
    usable traces, so the caller keeps the proven slow end-detection.

    Using the strict *minimum* (rather than a higher percentile) is deliberately
    conservative: one cycle that happened to go quiet early only lowers the
    baseline, making the detector fire *less* often - never more.
    """
    if not cycles:
        return None
    earliest: float | None = None
    clean = 0
    for cycle in cycles:
        if cycle.get("status") != "completed":
            continue
        points = decompress_power_data(cycle)
        if not points:
            continue
        clean += 1
        span_start: float | None = None
        for offset, power in points:
            if power < stop_threshold_w:
                if span_start is None:
                    span_start = offset
                elif (offset - span_start) >= min_quiet_span_s:
                    if earliest is None or span_start < earliest:
                        earliest = span_start
                    break
            else:
                span_start = None
    if clean < min_clean_cycles:
        return None
    return earliest


def device_active_peak_range(
    cycles: list[CycleDict], min_clean_cycles: int
) -> tuple[float, float] | None:
    """(min_peak, max_peak) across the device's **completed** cycles.

    The peak of a cycle is its maximum power.  This is the device's learned
    "power levels we have produced before" band, used by the terminal-drop
    familiarity gate: a running cycle peaking outside this range (widened by a
    tolerance) is drawing power unlike anything in its history and may be a new
    program, so an early drop on it is deferred rather than assumed terminal.

    Returns ``None`` below ``min_clean_cycles`` completed cycles with usable
    traces (same guard as the quiet-offset baseline)."""
    peaks: list[float] = []
    for cycle in cycles:
        if cycle.get("status") != "completed":
            continue
        points = decompress_power_data(cycle)
        if not points:
            continue
        peaks.append(max(p for _, p in points))
    if len(peaks) < min_clean_cycles:
        return None
    return (min(peaks), max(peaks))


def terminal_drop_baseline(
    cycles: list[CycleDict],
    stop_threshold_w: float,
    min_quiet_span_s: float,
    min_clean_cycles: int,
) -> tuple[float | None, tuple[float, float] | None]:
    """Combined ``(earliest_quiet_offset, peak_range)`` baseline.

    Pure (no store / no I/O beyond decompressing the passed-in cycle traces), so
    the manager can offload the whole per-cycle scan to an executor thread in one
    hop instead of decompressing every trace on the event loop (issue #311)."""
    earliest = earliest_sustained_quiet_offset(
        cycles, stop_threshold_w, min_quiet_span_s, min_clean_cycles
    )
    peak_range = device_active_peak_range(cycles, min_clean_cycles)
    return earliest, peak_range


def is_terminal_drop(
    points: list[tuple[float, float]],
    earliest_quiet: float | None,
    peak_range: tuple[float, float] | None,
    stop_threshold_w: float,
    earliness_ratio: float,
    min_peak_ratio: float,
    peak_familiar_tol: float,
) -> bool:
    """Whether a running cycle's current low-power event is an anomalously-early
    terminal drop (plug pulled / cancelled) rather than a soak pause.

    Pure decision logic (no store / no I/O) so it is unit-testable in isolation;
    the manager wires the learned ``earliest_quiet`` / ``peak_range`` baselines
    and gates on the ML opt-in. All three checks must hold:

    * **Clearly ON** - the cycle peaked >= ``min_peak_ratio`` x stop-threshold
      (a low-power idling device can't trip it).
    * **Familiar** - the peak is within the device's historical peak range
      widened by ``peak_familiar_tol``.  A peak outside that band means the cycle
      is drawing power unlike anything seen before (possibly a new program), so
      we DEFER (return ``False``) rather than assume a stop.
    * **Anomalous** - the trailing sub-threshold span began at an offset
      ``< earliest_quiet * earliness_ratio`` - earlier than this device has ever
      legitimately gone quiet.

    Returns ``False`` (keep the proven slow path) whenever a baseline is missing
    or any check fails."""
    if not points or earliest_quiet is None or peak_range is None:
        return False
    peak = max(p for _, p in points)
    if peak < min_peak_ratio * max(stop_threshold_w, 1.0):
        return False
    lo, hi = peak_range
    if not (lo * (1.0 - peak_familiar_tol) <= peak <= hi * (1.0 + peak_familiar_tol)):
        return False
    drop_start: float | None = None
    for offset, power in points:
        if power < stop_threshold_w:
            if drop_start is None:
                drop_start = offset
        else:
            drop_start = None
    if drop_start is None:
        return False
    return drop_start < earliest_quiet * earliness_ratio


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

        if old_major_version < 6:
            _LOGGER.info("Migrating storage from v%s to v6", old_major_version)
            cycles_raw = old_data.get("past_cycles", [])
            cycles = cast(list[dict[str, Any]], cycles_raw) if isinstance(cycles_raw, list) else []
            flagged = _flag_recorded_cycles_golden(cycles)
            _LOGGER.info(
                "Migration v5->v6: flagged %s recorded cycle(s) as golden references", flagged
            )

        if old_major_version < 7:
            # The v6 step only ran when upgrading from below v6, so recorded
            # cycles on installs already at v6 (e.g. recorded by an older build
            # that didn't set the golden flag) were never tagged. Re-run the now
            # broadened backfill once so every historical recorded cycle is
            # recognised. Idempotent: already-golden cycles are skipped.
            _LOGGER.info("Migrating storage from v%s to v7", old_major_version)
            cycles_raw = old_data.get("past_cycles", [])
            cycles = cast(list[dict[str, Any]], cycles_raw) if isinstance(cycles_raw, list) else []
            flagged = _flag_recorded_cycles_golden(cycles)
            _LOGGER.info(
                "Migration v6->v7: tagged %s previously-unmarked recorded cycle(s) as golden",
                flagged,
            )

        if old_major_version < 8:
            # The v6->v7 backfill identified recordings only by the explicit
            # ``meta.source``/``original_samples`` marker, which OLD recordings
            # (made before that marker existed, ``meta: None``) do not carry — so
            # they stayed untagged. _is_recorded_cycle now also recognises them by
            # their structural signature (completed + no max_power/termination_reason).
            # Re-run the backfill once more so those old recordings are finally
            # tagged. Idempotent: already-golden cycles are skipped.
            _LOGGER.info("Migrating storage from v%s to v8", old_major_version)
            cycles_raw = old_data.get("past_cycles", [])
            cycles = cast(list[dict[str, Any]], cycles_raw) if isinstance(cycles_raw, list) else []
            flagged = _flag_recorded_cycles_golden(cycles)
            _LOGGER.info(
                "Migration v7->v8: tagged %s old recorded cycle(s) (no meta marker) as golden",
                flagged,
            )

        if old_major_version < 9:
            # Pre-initialize additive top-level keys so they are present from the
            # first load rather than appearing lazily on first use. Idempotent:
            # setdefault leaves existing values untouched.
            _LOGGER.info("Migrating storage from v%s to v9", old_major_version)
            old_data.setdefault("lifetime_energy_wh", 0.0)
            # Seed the monotonic lifetime cycle counter from the retained history so an
            # existing install does not start at 0 and re-fire past milestones (50/100/...)
            # on cycles it already completed. Matches the cycle_count fallback semantics.
            old_data.setdefault(
                "lifetime_cycle_count", len(old_data.get("past_cycles") or [])
            )
            old_data.setdefault("settings_changelog", [])
            old_data.setdefault("maintenance_log", [])

        if old_major_version < 10:
            # Reference cycles imported from the community store live in their own list,
            # never in past_cycles, so they can feed the envelope/matcher but can never
            # touch usage/energy stats. Additive + idempotent.
            _LOGGER.info("Migrating storage from v%s to v10", old_major_version)
            old_data.setdefault("reference_cycles", [])

        if old_major_version < 11:
            # Marker-only bump. Per-phase profiles (envelope["phase_profile"], used
            # by phase-segmented matching / phase-resolved ETA) are DERIVED CACHE
            # built by async_rebuild_envelope, not stored data - so there is nothing
            # to migrate. They self-populate on the next envelope rebuild (which
            # runs on every cycle end / label change); until then consumers fall
            # back to the existing estimator via lazy absent-key handling. No data
            # is added, removed, or altered here.
            _LOGGER.info("Migrating storage from v%s to v11 (phase-profile cache marker)",
                         old_major_version)

        return old_data

def _ambiguity_from_candidates(candidates: list[dict]) -> tuple[float, bool]:
    """Top1-vs-top2 score margin and whether the match is ambiguous.

    Single source for the ambiguity rule shared by both match paths:
    ``is_ambiguous = margin < MATCH_AMBIGUITY_MARGIN``. ``margin`` defaults to
    1.0 when there is only one candidate.
    """
    margin = 1.0
    if len(candidates) > 1:
        margin = candidates[0]["score"] - candidates[1]["score"]
    return margin, margin < MATCH_AMBIGUITY_MARGIN


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
        # Cache for group cohesion scores to avoid re-running DTW on the event loop
        # every 5 minutes.  Keyed by sorted-members tuple; invalidated when profile_groups
        # content changes (tracked by a simple generation counter).
        self._cohesion_cache: dict[tuple[str, ...], float] = {}
        self._cohesion_cache_generation: int = 0
        self._cohesion_cache_generation_checked: int = -1
        # group_cohesion runs in executor threads (live matching + Playground can
        # touch the same store concurrently); serialize the read-clear-compute-store
        # sequence so a concurrent call can't clear a half-populated cache or
        # duplicate the DTW work.  A plain thread lock, never held across an await.
        self._cohesion_cache_lock = threading.Lock()
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
            "reference_cycles": [],  # Imported store cycles: envelope/matcher only, never usage stats
            "envelopes": {},  # Cached statistical envelopes per profile
            "auto_adjustments": [],  # Log of automatic setting changes
            "suggestions": {},  # Suggested settings (do NOT change user options)
            "feedback_history": {},  # Persisted user feedback (cycle_id -> record)
            "pending_feedback": {},  # Persisted pending feedback requests
            "custom_phases": [],  # Shared custom phase catalog
            "ml_model_versions": {},  # On-device trained model specs (Stage 4)
            "profile_groups": {},  # Named groups of near-duplicate profiles (Stage 5)
            "maintenance_log": [],  # User-logged maintenance events (Group E)
        }




    def set_suggestion(
        self,
        key: str,
        value: Any,
        reason: str | None = None,
        reason_key: str | None = None,
        reason_params: dict[str, Any] | None = None,
    ) -> None:
        """Store a suggested setting value without changing config entry options.

        ``reason`` is the English fallback text; ``reason_key`` + ``reason_params``
        are the localization key + interpolation values the panel resolves via
        ``_t(reason_key, reason_params, reason)``. Both localization fields are
        optional so old callers / persisted entries keep working.
        """
        suggestions: JSONDict = self._data.setdefault("suggestions", {})
        entry: dict[str, Any] = {
            "value": value,
            "reason": reason,
            "updated": dt_util.now().isoformat(),
        }
        # Store localization sidecars only when present so we never overwrite an
        # existing key with a stale ``None`` (keeps the shape lean for old data).
        if reason_key is not None:
            entry["reason_key"] = reason_key
        if reason_params is not None:
            entry["reason_params"] = reason_params
        suggestions[key] = entry

    def get_suggestions(self) -> dict[str, Any]:
        """Return current suggestion map."""
        raw = self._data.get("suggestions")
        if isinstance(raw, dict):
            suggestions = cast(JSONDict, raw)
            return suggestions.copy()
        return {}

    def delete_suggestion(self, key: str) -> None:
        """Remove a single suggestion entry by key."""
        suggestions: JSONDict = self._data.setdefault("suggestions", {})
        suggestions.pop(key, None)

    async def clear_suggestions(self) -> None:
        """Clear all pending suggestions and persist."""
        self._data["suggestions"] = {}
        await self.async_save()

    def get_suggestion_apply_cycle_count(self) -> int:
        """Return the total cycle count recorded at the last user suggestion apply."""
        return int(self._data.get("suggestion_apply_cycle_count", 0))

    def set_suggestion_apply_cycle_count(self, count: int) -> None:
        """Record the total cycle count at the moment the user applies suggestions."""
        self._data["suggestion_apply_cycle_count"] = count

    # ─── Settings changelog (Group D7) ─────────────────────────────────────────

    #: Maximum settings-changelog entries retained per device (newest kept).
    SETTINGS_CHANGELOG_MAX = 50

    def get_settings_changelog(self) -> list[dict[str, Any]]:
        """Return the recorded settings-change history (most-recent-first).

        Never raises: returns an empty list when no changelog exists yet.
        """
        try:
            raw = self._data.get("settings_changelog", [])
            if isinstance(raw, list):
                return raw
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return []

    async def async_record_settings_changes(self, changes: list[dict]) -> None:
        """Append settings-change entries and persist (capped at the newest 50).

        Each entry is normalized to ``{"key", "old", "new", "timestamp"}`` and
        prepended so the list stays most-recent-first. No-op on empty/invalid
        input.
        """
        if not changes:
            return
        log = self._data.setdefault("settings_changelog", [])
        if not isinstance(log, list):
            log = []
            self._data["settings_changelog"] = log

        now_iso = dt_util.now().isoformat()
        added = False
        for ch in changes:
            if not isinstance(ch, dict) or "key" not in ch:
                continue
            log.insert(
                0,
                {
                    "key": str(ch.get("key")),
                    "old": ch.get("old"),
                    "new": ch.get("new"),
                    "timestamp": ch.get("timestamp") or now_iso,
                },
            )
            added = True

        if not added:
            return

        # Keep only the newest entries (list is most-recent-first).
        if len(log) > self.SETTINGS_CHANGELOG_MAX:
            del log[self.SETTINGS_CHANGELOG_MAX:]
        self._data["settings_changelog"] = log
        await self.async_save()

    # ─── On-device ML model versions (Stage 4) ────────────────────────────────

    def get_ml_model_versions(self) -> dict[str, Any]:
        """Return the map of on-device trained model specs (capability -> record).

        Each record is ``{spec, trained_at, cycle_count, metrics, baseline_auc}``
        where ``spec`` is a NumPy-only standardized-logistic model produced by
        :mod:`.ml.trainer`. Empty when nothing has been trained on-device.
        """
        raw = self._data.get("ml_model_versions")
        if isinstance(raw, dict):
            return cast(JSONDict, raw).copy()
        return {}

    async def set_ml_model_version(self, capability: str, record: dict[str, Any]) -> None:
        """Persist a trained model record for a capability (quality/live_match/end)."""
        versions: JSONDict = self._data.setdefault("ml_model_versions", {})
        versions[capability] = record
        await self.async_save()

    async def clear_ml_model_versions(self) -> None:
        """Drop all on-device trained models (reverts to embedded baselines)."""
        self._data["ml_model_versions"] = {}
        await self.async_save()

    def get_ml_last_training_run(self) -> str | None:
        """ISO timestamp of the last *completed* on-device training run, or None.

        Distinct from each model's ``trained_at`` (the promotion time): this
        advances on every run even when no model beat the baseline, so the panel's
        "Last trained" reflects that training actually ran.
        """
        ts = self._data.get("ml_last_training_run")
        return ts if isinstance(ts, str) else None

    async def set_ml_last_training_run(self, iso: str) -> None:
        """Record when on-device training last completed (regardless of promotion)."""
        self._data["ml_last_training_run"] = iso
        await self.async_save()

    def get_ml_training_history(self) -> dict[str, list[dict[str, Any]]]:
        """Per-capability held-out-score history across training runs.

        ``{capability: [{"ts": iso, "score": float, "higher_better": bool}, ...]}``
        oldest-first. Lets the panel show whether a model's fit is improving,
        steady, or declining over time. Empty until training has run.
        """
        raw = self._data.get("ml_training_history")
        return cast(dict[str, list[dict[str, Any]]], raw) if isinstance(raw, dict) else {}

    async def append_ml_training_history(self, run_iso: str, results: list[dict[str, Any]]) -> None:
        """Append each capability's held-out score from a training run.

        ``results`` is ``train_from_cycles``'s per-capability records; classifiers
        report ``new_auc`` (higher is better), regressors ``model_mae`` (lower is
        better). Capabilities without a held-out metric this run (insufficient
        data) are skipped. Retains at most ``ML_TRAINING_HISTORY_MAX`` runs each.
        """
        from .const import ML_TRAINING_HISTORY_MAX  # noqa: PLC0415

        hist: JSONDict = self._data.setdefault("ml_training_history", {})
        changed = False
        for rec in results or []:
            if not isinstance(rec, dict):
                continue
            cap = rec.get("capability")
            if not isinstance(cap, str) or not cap:
                continue
            if rec.get("new_auc") is not None:
                score, higher_better = float(rec["new_auc"]), True
            elif rec.get("model_mae") is not None:
                score, higher_better = float(rec["model_mae"]), False
            else:
                continue
            series = hist.setdefault(cap, [])
            if not isinstance(series, list):
                series = hist[cap] = []
            series.append({"ts": run_iso, "score": round(score, 5), "higher_better": higher_better})
            del series[:-ML_TRAINING_HISTORY_MAX]
            changed = True
        if changed:
            await self.async_save()

    # ─── On-device matching-config tuning (Stage 4/5, opt-in) ──────────────────

    #: Only these bounded scoring weights (all in [0, 1]) may be overridden
    #: on-device, so a stored record can never change structural matching
    #: behaviour. Mirrors ml.matching_tuner.OVERRIDE_KEYS.
    _MATCHING_OVERRIDE_KEYS = ("corr_weight", "duration_weight", "energy_weight", "dtw_ensemble_w")

    def get_matching_config(self) -> dict[str, Any]:
        """Return the on-device tuned matcher scoring-weight record, if any.

        Record shape: ``{config, trained_at, cycle_count, baseline_test_top1,
        tuned_test_top1}`` where ``config`` holds the bounded scoring weights
        (``corr_weight``/``duration_weight``/``energy_weight``). Empty when the
        matcher is using the shipped defaults.
        """
        raw = self._data.get("matching_config")
        if isinstance(raw, dict):
            return cast(JSONDict, raw).copy()
        return {}

    async def set_matching_config(self, record: dict[str, Any]) -> None:
        """Persist the tuned matcher scoring-weight override + metadata."""
        self._data["matching_config"] = record
        await self.async_save()

    async def clear_matching_config(self) -> None:
        """Revert the matcher to the shipped default scoring weights."""
        self._data.pop("matching_config", None)
        await self.async_save()

    def _matching_overrides(self) -> dict[str, float]:
        """Bounded scoring-weight overrides to merge into the matcher config.

        Only the whitelisted keys are honoured, each clamped to ``[0, 1]``, so a
        stored record can never alter structural matching behaviour - only the
        emphasis between shape, level and energy agreement.
        """
        rec = self._data.get("matching_config")
        cfg = rec.get("config") if isinstance(rec, dict) else None
        out: dict[str, float] = {}
        if isinstance(cfg, dict):
            for k in self._MATCHING_OVERRIDE_KEYS:
                try:
                    out[k] = min(1.0, max(0.0, float(cfg[k])))
                except (KeyError, TypeError, ValueError):
                    continue
        return out

    async def set_cycle_review(
        self,
        cycle_id: str,
        *,
        quality: str | None = None,
        golden: bool | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> bool:
        """Attach an ML-Lab review to a cycle (Stage 4b).

        The review (``quality``/``golden``/``tags``/``notes``) is stored under the
        cycle's ``ml_review`` key and becomes a strong training label for the
        on-device quality model. Only the provided fields are updated. Returns
        True when the cycle was found and updated.
        """
        cycle = next(
            (c for c in self._data.get("past_cycles", []) if c.get("id") == cycle_id),
            None,
        )
        if not cycle:
            raise ValueError(f"Cycle {cycle_id} not found")
        prev_review = cycle.get("ml_review")
        prev_golden = bool(prev_review.get("golden")) if isinstance(prev_review, dict) else False
        review = dict(prev_review or {})
        if quality is not None:
            review["quality"] = quality
        if golden is not None:
            review["golden"] = bool(golden)
        if tags is not None:
            review["tags"] = list(tags)
        if notes is not None:
            review["notes"] = notes
        review["reviewed_at"] = dt_util.now().isoformat()
        cycle["ml_review"] = review
        # Golden cycles seed the profile's matching reference/envelope. When the
        # golden flag actually flips (either direction), rebuild the affected
        # profile so its envelope + reference cycle + cohesion cache reflect the
        # new golden set. Only when the value changed — an unchanged save stays
        # idempotent and skips the (executor-bound) rebuild.
        golden_changed = golden is not None and bool(golden) != prev_golden
        profile_name = cycle.get("profile_name")
        if golden_changed and isinstance(profile_name, str) and profile_name:
            await self.async_rebuild_envelope(profile_name)
        await self.async_save()
        # Don't log the full review (notes/tags are user-entered) — only field names.
        _changed = [
            n for n, v in (("quality", quality), ("golden", golden), ("tags", tags), ("notes", notes))
            if v is not None
        ]
        self._logger.info(
            "Recorded ML review for cycle %s (updated: %s)", cycle_id, ", ".join(_changed) or "none"
        )
        return True

    async def async_backfill_recorded_golden(self) -> int:
        """Flag manually-recorded cycles (meta.source == 'recorder') as golden
        references. Recorded == golden: a single stored flag, no duplicate field.
        Idempotent; saves only when something changed. Returns the count flagged.
        """
        cycles = self._data.get("past_cycles", [])
        if not isinstance(cycles, list):
            return 0
        # Snapshot each cycle's golden state (aligned by list position, which
        # _flag_recorded_cycles_golden never reorders) so we can rebuild exactly
        # the profiles whose cycles get newly flagged — golden cycles seed the
        # matching reference/envelope.
        before_golden = [
            bool(c.get("ml_review", {}).get("golden"))
            if isinstance(c, dict) and isinstance(c.get("ml_review"), dict)
            else False
            for c in cycles
        ]
        flagged = _flag_recorded_cycles_golden(cycles)
        if flagged:
            affected: set[str] = set()
            for was_golden, cycle in zip(before_golden, cycles):
                if was_golden or not isinstance(cycle, dict):
                    continue
                review = cycle.get("ml_review")
                if isinstance(review, dict) and review.get("golden"):
                    pname = cycle.get("profile_name")
                    if isinstance(pname, str) and pname:
                        affected.add(pname)
            for pname in affected:
                await self.async_rebuild_envelope(pname)
            await self.async_save()
            self._logger.info("Backfilled golden flag on %s recorded cycle(s)", flagged)
        return flagged

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

    def prune_orphaned_feedback(self) -> int:
        """Drop pending-feedback entries whose cycle no longer exists.

        A cycle that was deleted/merged/trimmed can leave its pending feedback
        behind, so the device's "needs review" count stays >0 while the Cycles
        review filter (which matches feedback ids against real cycles) shows
        nothing. Returns the number pruned; never raises. Caller saves.
        """
        try:
            pending = self._data.get("pending_feedback")
            if not isinstance(pending, dict) or not pending:
                return 0
            live_ids = {c.get("id") for c in self.get_past_cycles()}
            orphans = [cid for cid in list(pending) if cid not in live_ids]
            for cid in orphans:
                pending.pop(cid, None)
            return len(orphans)
        except Exception:  # noqa: BLE001
            return 0


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

    # ── Community-store account (connect handoff) ─────────────────────────────
    def get_store_account(self) -> dict[str, Any]:
        """Full persisted store account incl. the refresh token (credential)."""
        raw = self._data.get("store_account")
        return dict(raw) if isinstance(raw, dict) else {}

    def get_store_identity(self) -> dict[str, Any]:
        """Safe account view for status/UI - never includes the refresh token."""
        a = self.get_store_account()
        return {
            "connected": bool(a.get("refresh_token")),
            "uid": a.get("uid"),
            "name": a.get("name"),
            "brand": a.get("brand"),
            "model": a.get("model"),
        }

    async def set_store_account(self, account: dict[str, Any]) -> None:
        """Persist/merge the store account (refresh_token, uid, name, brand, model)."""
        cur = self.get_store_account()
        cur.update({k: v for k, v in account.items() if v is not None})
        self._data["store_account"] = cur
        await self.async_save()

    async def clear_store_account(self) -> None:
        self._data.pop("store_account", None)
        await self.async_save()

    def get_reference_cycles(self) -> list[CycleDict]:
        """Return the imported-store reference cycles.

        These are NOT in ``past_cycles``: they feed only the envelope shape and the
        matcher template, never usage/energy/count/trend stats.
        """
        raw = self._data.setdefault("reference_cycles", [])
        if isinstance(raw, list):
            return cast(list[CycleDict], raw)
        return []

    def get_shareable_cycles(self) -> list[dict[str, Any]]:
        """Recorded/golden reference cycles eligible to share to the community store.

        A cycle qualifies when it is hand-flagged golden (``ml_review.golden``) or a
        manual recording (``meta.source == "recorder"``) and carries a program label.
        Imported ``reference_cycles`` are intentionally excluded (you do not re-share
        what you downloaded). Returns light summaries (no traces), most-recent-first;
        the whole set (not a page) so the panel's share tree can offer all of them.
        Never raises.
        """
        out: list[dict[str, Any]] = []
        for c in self.get_past_cycles():
            if not isinstance(c, dict):
                continue
            program = str(c.get("profile_name") or "").strip()
            if not program:
                continue
            review = c.get("ml_review") if isinstance(c.get("ml_review"), dict) else {}
            meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
            if not (review.get("golden") or meta.get("source") == "recorder"):
                continue
            out.append({
                "id": c.get("id"),
                "profile_name": program,
                "start_time": c.get("start_time"),
                "duration": c.get("duration"),
                "source": meta.get("source"),
            })
        out.sort(key=lambda r: r.get("start_time") or "", reverse=True)
        return out

    async def add_reference_cycle(
        self, profile_name: str, points: list[list[float]], meta: dict[str, Any]
    ) -> str:
        """Import a reference cycle downloaded from the store into ``reference_cycles``.

        ``points`` is a raw trace of ``[offset_seconds, watts]`` pairs. ``meta`` may carry
        ``store_cycle_id`` (-> ``meta.source = "store:<id>"``), ``store_uploaded_at`` and
        ``sampling_interval``. The cycle is stamped with import-time timestamps (its real
        run time is meaningless locally), forced ``status="completed"`` and
        ``ml_review.golden=True`` so it seeds the envelope shape, then the envelope is
        rebuilt. Never accumulates lifetime energy or touches ``past_cycles``.
        """
        # Validate the trace BEFORE creating any persistent state (profile entry /
        # reference cycle): drop non-finite/non-numeric samples, require >= 2 points
        # and a positive time span. A garbage trace returns "" and mutates nothing.
        pairs: list[list[float]] = []
        for p in (points or []):
            if len(p) < 2:
                continue
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                pairs.append([x, y])
        if len(pairs) < 2:
            return ""
        pairs.sort(key=lambda q: q[0])  # defensive: normalize any out-of-order offsets
        duration = float(pairs[-1][0] - pairs[0][0])
        if duration <= 0:
            return ""
        # Re-base to offset 0 so envelope reconstruction and DTW work correctly.
        if pairs[0][0] != 0.0:
            origin = pairs[0][0]
            pairs = [[p[0] - origin, p[1]] for p in pairs]
        now = dt_util.now()
        store_id = str(meta.get("store_cycle_id") or "")
        cycle: CycleDict = {
            "profile_name": profile_name,
            "power_data": pairs,
            "start_time": now.isoformat(),
            # Keep the timestamp interval consistent with the imported duration
            # (both were previously "now", giving a zero-length interval).
            "end_time": (now + timedelta(seconds=duration)).isoformat(),
            "duration": duration,
            "status": "completed",
            "ml_review": {"golden": True},
            "meta": {
                "source": f"store:{store_id}" if store_id else "store",
                "store_uploaded_at": meta.get("store_uploaded_at"),
            },
        }
        if meta.get("sampling_interval"):
            cycle["sampling_interval"] = float(meta["sampling_interval"])
        # A reference cycle implies its program exists locally; create a minimal profile
        # entry if absent so the matcher iterates it and the rebuild can set its template.
        profiles = self._data.setdefault("profiles", {})
        if profile_name not in profiles:
            profiles[profile_name] = {"avg_duration": duration}
        self._add_cycle_data(cycle, target=self._data.setdefault("reference_cycles", []))
        await self.async_rebuild_envelope(profile_name)
        await self.async_save()
        return str(cycle.get("id", ""))

    # ── Profile groups (Stage 5: near-duplicate variants) ──────────────────────

    def get_profile_groups(self) -> dict[str, JSONDict]:
        """Return mutable profile-groups mapping (group_name -> {members, created_at}).

        A group bundles near-duplicate profiles (e.g. the same program at
        different temperature/spin). The matcher treats the group as one
        aggregate candidate and only picks the exact member once the group wins.
        Groups with fewer than 2 present members are ignored by the matcher.
        """
        raw = self._data.setdefault("profile_groups", {})
        if not isinstance(raw, dict):
            self._data["profile_groups"] = {}
            return self._data["profile_groups"]
        # Drop members that no longer exist as profiles (kept consistent lazily).
        profiles = self.get_profiles()
        for g in raw.values():
            if isinstance(g, dict) and isinstance(g.get("members"), list):
                g["members"] = [m for m in g["members"] if m in profiles]
        return cast(dict[str, JSONDict], raw)

    def _members_in_other_groups(self, members: list[str], exclude: str | None) -> dict[str, str]:
        """Map any member already assigned to a DIFFERENT group -> that group name."""
        owner: dict[str, str] = {}
        for gname, g in self.get_profile_groups().items():
            if gname == exclude or not isinstance(g, dict):
                continue
            for m in (g.get("members") or []):
                owner.setdefault(m, gname)
        return {m: owner[m] for m in members if m in owner}

    async def create_profile_group(self, name: str, members: list[str]) -> bool:
        """Create (or overwrite) a group with the given member profile names."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Group name is required")
        profiles = self.get_profiles()
        members = [m for m in dict.fromkeys(members or []) if m in profiles]
        # A profile may belong to at most one group (else _grouped_snapshots would
        # collapse it inconsistently). Reject up front — no partial mutation.
        conflicts = self._members_in_other_groups(members, exclude=name)
        if conflicts:
            raise ValueError(
                "Already in another group: "
                + ", ".join(f"{m} ({g})" for m, g in conflicts.items())
            )
        groups = self.get_profile_groups()
        groups[name] = {"members": members, "created_at": dt_util.now().isoformat()}
        self._cohesion_cache_generation += 1
        await self.async_save()
        self._logger.info("Created profile group %r with %d members", name, len(members))
        return True

    async def set_profile_group_members(self, name: str, members: list[str]) -> bool:
        """Replace a group's member list. Deletes the group if left empty."""
        groups = self.get_profile_groups()
        if name not in groups:
            raise ValueError(f"Group {name} not found")
        profiles = self.get_profiles()
        members = [m for m in dict.fromkeys(members or []) if m in profiles]
        conflicts = self._members_in_other_groups(members, exclude=name)
        if conflicts:
            raise ValueError(
                "Already in another group: "
                + ", ".join(f"{m} ({g})" for m, g in conflicts.items())
            )
        if members:
            groups[name]["members"] = members
        else:
            groups.pop(name, None)
        self._cohesion_cache_generation += 1
        await self.async_save()
        return True

    async def rename_profile_group(self, name: str, new_name: str) -> bool:
        groups = self.get_profile_groups()
        new_name = (new_name or "").strip()
        if name not in groups or not new_name:
            raise ValueError("Group not found or new name empty")
        if new_name != name:
            if new_name in groups:
                raise ValueError(f"A group named {new_name!r} already exists")
            groups[new_name] = groups.pop(name)
        self._cohesion_cache_generation += 1
        await self.async_save()
        return True

    async def delete_profile_group(self, name: str) -> bool:
        groups = self.get_profile_groups()
        if groups.pop(name, None) is None:
            return False
        self._cohesion_cache_generation += 1
        await self.async_save()
        self._logger.info("Deleted profile group %r", name)
        return True

    def _profile_curve(self, name: str, n: int = 150) -> np.ndarray | None:
        """A profile's envelope average resampled to n points, or None."""
        env = self._data.get("envelopes", {}).get(name) if isinstance(self._data.get("envelopes"), dict) else None
        avg = env.get("avg") if isinstance(env, dict) else None
        if not avg or not isinstance(avg[0], (list, tuple)):
            return None
        ys = np.asarray([float(p[1]) for p in avg], dtype=float)
        if ys.size < 4:
            return None
        return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, ys.size), ys)

    @staticmethod
    def _shape_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Duration- and amplitude-tolerant shape similarity in (0,1].

        Peak-normalises both curves (so temperature/spin amplitude differences
        don't count against membership) then aligns them with a Sakoe-Chiba-band
        DTW (so a longer heating or draining phase at a different setting warps
        into alignment instead of being penalised as a different shape). 1.0 =
        identical shape; genuinely different programs score low even after warping.
        """
        na = a / (float(a.max()) or 1.0)
        nb = b / (float(b.max()) or 1.0)
        dist = analysis.compute_dtw_lite(na, nb, band_width_ratio=0.2)
        norm = dist / max(len(na), 1)  # mean per-sample distance on [0,1] curves
        return 1.0 / (1.0 + norm / 0.15)

    def group_cohesion(self, members: list[str]) -> float:
        """Minimum pairwise shape similarity among a group's members (1.0 =
        identical shapes; see _shape_similarity for the DTW/peak-normalised
        metric). Low cohesion means the members are not really the same program,
        so their averaged aggregate would be too generic and could out-match
        unrelated profiles - the matcher refuses to aggregate below
        GROUP_MIN_COHESION and the UI warns the user.

        Results are cached per member-set and invalidated via
        ``_cohesion_cache_generation`` (incremented by group mutation methods) so
        the DTW pairwise comparison does not run on the event loop every 5 minutes.
        """
        key = tuple(sorted(members))
        with self._cohesion_cache_lock:
            if self._cohesion_cache_generation != self._cohesion_cache_generation_checked:
                self._cohesion_cache.clear()
                self._cohesion_cache_generation_checked = self._cohesion_cache_generation
            if key in self._cohesion_cache:
                return self._cohesion_cache[key]
            curves = [c for c in (self._profile_curve(m) for m in members) if c is not None]
            if len(members) < 2:
                # A genuinely single-member group is trivially cohesive (and is never
                # collapsed anyway — nothing to aggregate).
                result = 1.0
            elif len(curves) < 2:
                # Multi-member but too few built curves -> insufficient evidence, treat as
                # NOT cohesive so the group isn't collapsed into a blurry aggregate yet.
                result = 0.0
            else:
                result = 1.0
                for i in range(len(curves)):
                    for j in range(i + 1, len(curves)):
                        result = min(result, self._shape_similarity(curves[i], curves[j]))
            self._cohesion_cache[key] = result
            return result

    def _grouped_snapshots(
        self, snapshots: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, dict[str, Any]]]:
        """Collapse each *cohesive* group into one aggregate candidate snapshot.

        Returns (snapshots, group_members, member_snaps). Groups with <2 present
        members, or cohesion below GROUP_MIN_COHESION (too generic an aggregate),
        are left as individual member snapshots. Aggregate name is prefixed
        ``__group__`` and mapped to its member profile names in group_members.
        """
        groups = self.get_profile_groups()
        if not groups:
            return snapshots, {}, {}
        by_name = {s["name"]: s for s in snapshots}
        member_to_group: dict[str, str] = {}
        for gname, g in groups.items():
            present = [m for m in (g.get("members") or []) if m in by_name]
            if len(present) < 2 or self.group_cohesion(present) < GROUP_MIN_COHESION:
                continue  # keep members individual
            for m in present:
                member_to_group[m] = gname
        if not member_to_group:
            return snapshots, {}, {}
        n = 200
        agg_curves: dict[str, list[np.ndarray]] = {}
        agg_durs: dict[str, list[float]] = {}
        member_snaps: dict[str, dict[str, Any]] = {}
        out: list[dict[str, Any]] = []
        for s in snapshots:
            g = member_to_group.get(s["name"])
            if g is None:
                out.append(s)
                continue
            member_snaps[s["name"]] = s
            sp = s.get("sample_power") or []
            if len(sp) >= 2:
                arr = np.asarray(sp, dtype=float)
                agg_curves.setdefault(g, []).append(
                    np.interp(np.linspace(0, 1, n), np.linspace(0, 1, arr.size), arr)
                )
                agg_durs.setdefault(g, []).append(float(s.get("avg_duration") or 0.0))
        group_members: dict[str, list[str]] = {}
        for g, curves in agg_curves.items():
            key = f"__group__{g}"
            durs = [d for d in agg_durs[g] if d > 0]
            out.append({
                "name": key,
                "avg_duration": float(np.mean(durs)) if durs else 0.0,
                "sample_power": np.mean(np.array(curves), axis=0).tolist(),
            })
            group_members[key] = [m for m in member_to_group if member_to_group[m] == g and m in member_snaps]
        return out, group_members, member_snaps

    def _stage5_pick_member(
        self, current_power: list[float], current_duration: float,
        members: list[str], member_snaps: dict[str, dict[str, Any]],
    ) -> tuple[str, float | None, float | None]:
        """Within a winning group, pick the member whose duration + mean power +
        peak best match the cycle (temperature -> mean power, spin -> peak).
        Returns (member_name, individual_fit_score, member_avg_duration). The fit
        score is the chosen member's own alignment score, used as a sanity check."""
        cur = np.asarray(current_power, dtype=float)
        if cur.size == 0 or not members:
            return (members[0] if members else ""), None, None
        cur_mp = float(cur.mean()); cur_pk = float(cur.max())

        def agree(a: float, b: float, scale: float) -> float:
            if a <= 0 or b <= 0:
                return 0.0
            return 1.0 / (1.0 + abs(math.log(a / b)) / scale)

        best_m, best_sc, best_dur = members[0], -1.0, None
        for m in members:
            snap = member_snaps.get(m)
            if not snap:
                continue
            sp = np.asarray(snap.get("sample_power") or [], dtype=float)
            if sp.size == 0:
                continue
            md = float(snap.get("avg_duration") or 0.0)
            sc = (agree(current_duration, md, 0.15)
                  * agree(cur_mp, float(sp.mean()), 0.20)
                  * agree(cur_pk, float(sp.max()), 0.20))
            if sc > best_sc:
                best_sc, best_m, best_dur = sc, m, md
        fit = None
        snap = member_snaps.get(best_m)
        if snap and snap.get("sample_power"):
            try:
                fit = float(analysis.find_best_alignment(current_power, snap["sample_power"], 1.0)[0])
            except Exception:  # pylint: disable=broad-exception-caught
                fit = None
        return best_m, fit, best_dur

    def suggest_profile_groups(self, dur_tol: float = 0.60, sim_min: float = 0.85) -> list[dict[str, Any]]:
        """Detect clusters of near-duplicate profiles not already fully grouped.

        Two profiles cluster when their durations are within a (loose) ``dur_tol``
        sanity bound AND their DTW/peak-normalised shape similarity exceeds
        ``sim_min`` (same program shape; they may differ in temperature/spin and in
        phase length, which the DTW alignment tolerates). The duration bound is
        loose because higher-temp/lower-rpm variants legitimately run longer - it
        only rules out grouping, say, a 20-min rinse with a 3-hour cotton.
        Returns {"members": [...], "existing_group": name|None} suggestions the
        user confirms. Never mutates state.
        """
        profiles = self.get_profiles()
        envelopes = self._data.get("envelopes", {})
        # Build per-profile (avg curve resampled to N, avg duration).
        N = 150
        reps: dict[str, tuple[np.ndarray, float]] = {}
        for name, prof in profiles.items():
            env = envelopes.get(name) if isinstance(envelopes, dict) else None
            avg = env.get("avg") if isinstance(env, dict) else None
            dur = float(prof.get("avg_duration") or 0.0)
            if not avg or dur <= 0 or not isinstance(avg[0], (list, tuple)):
                continue
            ys = np.asarray([float(p[1]) for p in avg], dtype=float)
            if ys.size < 4:
                continue
            curve = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, ys.size), ys)
            reps[name] = (curve, dur)
        names = list(reps)
        # Union-find near-duplicates.
        parent = {n: n for n in names}
        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        lim = math.log(1.0 + dur_tol)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                (ca, da), (cb, db) = reps[a], reps[b]
                if da <= 0 or db <= 0 or abs(math.log(da / db)) > lim:
                    continue
                if self._shape_similarity(ca, cb) > sim_min:
                    parent[find(a)] = find(b)
        clusters: dict[str, list[str]] = {}
        for n in names:
            clusters.setdefault(find(n), []).append(n)
        # Only clusters of >=2; annotate with any existing group overlap; skip
        # clusters already fully contained in one group.
        out: list[dict[str, Any]] = []
        groups = self.get_profile_groups()
        for members in clusters.values():
            if len(members) < 2:
                continue
            existing = None
            for gname, g in groups.items():
                gmembers = set(g.get("members") or [])
                if gmembers & set(members):
                    existing = gname
                    if set(members) <= gmembers:
                        members = []  # already grouped
                    break
            if members:
                out.append({"members": sorted(members), "existing_group": existing})
        return out

    # ------------------------------------------------------------------
    # A1: Underrun anomaly helpers
    # ------------------------------------------------------------------

    def get_profile_median_duration(self, profile_name: str) -> float | None:
        """Median cycle duration (s) for this profile across all labeled cycles. Never raises."""
        try:
            durations = [
                float(c["duration"])
                for c in self.get_past_cycles()
                if c.get("profile_name") == profile_name and c.get("duration")
            ]
            return float(np.median(durations)) if len(durations) >= 2 else None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # A2: Energy anomaly helpers
    # ------------------------------------------------------------------

    def get_profile_energy_stats(self, profile_name: str) -> dict[str, float] | None:
        """Energy stats {avg_wh, std_wh, n} for this profile. None if fewer than 3 cycles. Never raises."""
        try:
            energies = [
                float(c["energy_wh"])
                for c in self.get_past_cycles()
                if c.get("profile_name") == profile_name and c.get("energy_wh")
            ]
            if len(energies) < 3:
                return None
            arr = np.asarray(energies, dtype=float)
            return {"avg_wh": float(np.mean(arr)), "std_wh": float(np.std(arr)), "n": len(energies)}
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # A4: Profile warm-up mode helper
    # ------------------------------------------------------------------

    def get_profile_labeled_count(self, profile_name: str) -> int:
        """Number of labeled cycles for this profile. Never raises."""
        try:
            return sum(1 for c in self.get_past_cycles() if c.get("profile_name") == profile_name)
        except Exception:  # noqa: BLE001
            return 0

    def profile_has_reference_cycles(self, profile_name: str) -> bool:
        """True if this profile was seeded from imported store reference cycle(s).

        Such a profile is a trusted, community-shared template that the user
        downloaded to match immediately, so it is exempt from the local warm-up
        gate (which exists to stabilise profiles built from a few local cycles).
        """
        try:
            return any(
                c.get("profile_name") == profile_name
                for c in self.get_reference_cycles()
            )
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # B1: Lifetime energy accumulator (HA Energy dashboard)
    # ------------------------------------------------------------------

    def get_lifetime_energy_wh(self) -> float:
        """Total accumulated lifetime energy (Wh). Pure getter — never persists. Never raises."""
        try:
            return float(self._data.get("lifetime_energy_wh", 0.0))
        except Exception:  # noqa: BLE001
            return 0.0

    async def async_add_lifetime_energy_wh(self, wh: float) -> None:
        """Add *wh* (clamped >= 0) to the lifetime energy total and persist.

        Called exactly once per completed cycle so it never double-counts. Does
        not backfill from history — the meter starts at 0 and accumulates forward.
        Ignores non-numeric input.
        """
        try:
            add = max(0.0, float(wh))
        except (ValueError, TypeError):
            return
        base = self.get_lifetime_energy_wh()
        self._data["lifetime_energy_wh"] = round(base + add, 3)
        await self.async_save()

    def get_lifetime_cycle_count(self) -> int:
        """Persisted monotonic lifetime completed-cycle count.

        Only ever increments (never regresses when history is trimmed/merged), so it
        is the correct basis for milestone crossings. Pure getter - never persists.
        Never raises; returns 0 when unset.
        """
        try:
            return int(self._data.get("lifetime_cycle_count", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    def set_lifetime_cycle_count(self, count: int) -> None:
        """Set the lifetime cycle count in memory WITHOUT persisting.

        The cycle-end path batches this with the immediately-following lifetime
        energy save (:meth:`async_add_lifetime_energy_wh`) so the store is written
        once per cycle. Encapsulates the storage key so callers need not touch
        ``_data`` directly. Ignores non-integer input.
        """
        try:
            self._data["lifetime_cycle_count"] = int(count)
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # E1: Maintenance log & predictive-maintenance reminders (Group E)
    # ------------------------------------------------------------------

    def get_maintenance_log(self) -> list[dict[str, Any]]:
        """Return logged maintenance events, most-recent-first. Never raises.

        Each entry is ``{"id", "date", "event_type", "notes"}``. Entries with an
        unparseable date sort last.
        """
        try:
            raw = self._data.get("maintenance_log", [])
            if not isinstance(raw, list):
                return []
            entries = [dict(e) for e in raw if isinstance(e, dict)]
            _floor = datetime.min.replace(tzinfo=dt_util.UTC)
            entries.sort(
                key=lambda e: _parse_maintenance_dt(e.get("date")) or _floor,
                reverse=True,
            )
            return entries
        except Exception:  # noqa: BLE001
            return []

    async def async_add_maintenance_event(
        self, event_type: str, date: str | None = None, notes: str = ""
    ) -> dict[str, Any]:
        """Append a maintenance event and persist. Returns the created entry.

        ``event_type`` must be one of :data:`MAINTENANCE_EVENT_TYPES` (else raises
        ``ValueError``). ``date`` defaults to the current timestamp; a short unique
        id is generated for the entry.
        """
        if event_type not in MAINTENANCE_EVENT_TYPES:
            raise ValueError(f"Unknown maintenance event_type: {event_type!r}")
        entry: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "date": date if isinstance(date, str) and date else dt_util.now().isoformat(),
            "event_type": event_type,
            "notes": str(notes or ""),
        }
        log = self._data.setdefault("maintenance_log", [])
        if not isinstance(log, list):
            log = []
            self._data["maintenance_log"] = log
        log.append(entry)
        await self.async_save()
        return entry

    async def async_delete_maintenance_event(self, event_id: str) -> bool:
        """Remove a maintenance event by id, persist, and report whether removed."""
        log = self._data.get("maintenance_log")
        if not isinstance(log, list):
            return False
        remaining = [e for e in log if not (isinstance(e, dict) and e.get("id") == event_id)]
        if len(remaining) == len(log):
            return False
        self._data["maintenance_log"] = remaining
        await self.async_save()
        return True

    def cycles_since_maintenance(self, event_type: str) -> int:
        """Count completed cycles since the most recent maintenance event of a type.

        Counts completed cycles (``status == "completed"``) whose ``start_time`` is
        after the most recent maintenance event of ``event_type``. If no such event
        was ever logged, returns the total completed-cycle count. Never raises.
        """
        try:
            completed = [
                c for c in self.get_past_cycles()
                if isinstance(c, dict) and c.get("status") == "completed"
            ]
            latest_dt: datetime | None = None
            for e in self._data.get("maintenance_log", []) or []:
                if not isinstance(e, dict) or e.get("event_type") != event_type:
                    continue
                dt = _parse_maintenance_dt(e.get("date"))
                if dt is not None and (latest_dt is None or dt > latest_dt):
                    latest_dt = dt
            if latest_dt is None:
                return len(completed)
            count = 0
            for c in completed:
                start = _parse_start_dt(c.get("start_time"))
                if start is not None and start > latest_dt:
                    count += 1
            return count
        except Exception:  # noqa: BLE001
            return 0

    def get_maintenance_due(self, reminder_cfg: dict[str, Any] | None) -> list[str]:
        """Return event types whose cycles-since threshold has been reached.

        For each event type with a positive integer threshold in ``reminder_cfg``,
        includes it when ``cycles_since_maintenance(event_type) >= threshold``.
        Never raises.
        """
        try:
            if not isinstance(reminder_cfg, dict):
                return []
            due: list[str] = []
            for event_type, threshold in reminder_cfg.items():
                try:
                    thr = int(threshold)
                except (TypeError, ValueError):
                    continue
                if thr <= 0:
                    continue
                if self.cycles_since_maintenance(str(event_type)) >= thr:
                    due.append(str(event_type))
            return due
        except Exception:  # noqa: BLE001
            return []

    def has_recent_maintenance(
        self, event_type: str, days: int = MAINTENANCE_RECENT_SUPPRESS_DAYS
    ) -> bool:
        """True if a matching maintenance event was logged within ``days``. Never raises."""
        try:
            cutoff = dt_util.now() - timedelta(days=max(0, int(days)))
            for e in self._data.get("maintenance_log", []) or []:
                if not isinstance(e, dict) or e.get("event_type") != event_type:
                    continue
                dt = _parse_maintenance_dt(e.get("date"))
                if dt is not None and dt >= cutoff:
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False

    def compute_profile_health(self) -> dict[str, dict[str, Any]]:
        """Compute per-profile health indicators from labeled cycle history.

        For each profile that has at least 3 labeled cycles, returns a dict with:
          ``cycle_count``      – number of labeled cycles used in the calculation
          ``confidence_mean``  – mean match_confidence across those cycles (0–1)
          ``duration_cv``      – coefficient of variation of cycle durations (lower = consistent)
          ``health_score``     – composite 0–1 health score (1 = healthy, 0 = needs attention)
          ``health_status``    – "healthy" / "fair" / "poor" / "unknown"
          ``shape_drift``      – True when the early/recent envelope correlation is below
                                 SHAPE_DRIFT_THRESHOLD (only present when >= SHAPE_DRIFT_MIN_CYCLES
                                 labeled cycles with power_data exist)
          ``shape_drift_correlation`` – Pearson r between early and recent average envelopes

        Profiles with fewer than 3 labeled cycles return ``health_status="unknown"``.
        Never raises — returns an empty dict on any error.
        """
        try:
            cycles = self.get_past_cycles()
            from collections import defaultdict  # pylint: disable=import-outside-toplevel
            by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for c in cycles:
                name = c.get("profile_name")
                if name:
                    by_profile[str(name)].append(c)

            result: dict[str, dict[str, Any]] = {}
            for name, pcy in by_profile.items():
                count = len(pcy)
                durations = [float(c["duration"]) for c in pcy if c.get("duration")]
                confidences = [
                    float(c["match_confidence"])
                    for c in pcy
                    if c.get("match_confidence") is not None  # keep genuine 0.0, drop only absent
                ]

                if count < 3 or not durations:
                    result[name] = {"cycle_count": count, "health_status": "unknown"}
                    continue

                dur_arr = np.asarray(durations, dtype=float)
                dur_mean = float(np.mean(dur_arr))
                dur_cv = float(np.std(dur_arr) / dur_mean) if dur_mean > 0 else 0.0

                conf_mean = float(np.mean(confidences)) if confidences else 0.5

                # consistency: 1 at CV=0, linearly decays to 0 at CV=0.5+
                consistency = max(0.0, 1.0 - dur_cv / 0.5)
                health_score = round(0.5 * consistency + 0.5 * conf_mean, 3)

                if health_score >= 0.65:
                    status = "healthy"
                elif health_score >= 0.40:
                    status = "fair"
                else:
                    status = "poor"

                # A5: Shape drift — compare early vs recent power curve envelopes.
                _sd: dict[str, Any] = {}
                _traced = [c for c in pcy if c.get("power_data")]
                if len(_traced) >= SHAPE_DRIFT_MIN_CYCLES:
                    _third = len(_traced) // 3
                    _early = _traced[:_third]
                    _recent = _traced[-_third:]
                    try:
                        from .signal_processing import resample_to_n as _resamp  # noqa: PLC0415

                        def _avg_env(cycles: list[dict[str, Any]]) -> "np.ndarray | None":
                            _traces = []
                            for _c in cycles:
                                _raw = decompress_power_data(_c)
                                if not _raw:
                                    continue
                                _pwr = [float(_p) for _, _p in _raw]
                                if len(_pwr) < 5:
                                    continue
                                _t = np.asarray(_resamp(_pwr, SHAPE_DRIFT_RESAMPLE_N), dtype=float)
                                _mx = _t.max()
                                if _mx > 0:
                                    _t = _t / _mx
                                _traces.append(_t)
                            if len(_traces) < 2:
                                return None
                            return np.mean(np.stack(_traces), axis=0)

                        _early_env = _avg_env(_early)
                        _rec_env = _avg_env(_recent)
                        if _early_env is not None and _rec_env is not None:
                            _corr = float(np.corrcoef(_early_env, _rec_env)[0, 1])
                            if not np.isfinite(_corr):
                                _corr = 1.0
                            _sd = {
                                "shape_drift": _corr < SHAPE_DRIFT_THRESHOLD,
                                "shape_drift_correlation": round(_corr, 3),
                            }
                    except Exception:  # noqa: BLE001
                        pass

                result[name] = {
                    "cycle_count": count,
                    "confidence_mean": round(conf_mean, 3),
                    "duration_cv": round(dur_cv, 3),
                    "health_score": health_score,
                    "health_status": status,
                    **_sd,   # merges shape_drift and shape_drift_correlation when available
                }
            return result
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    # Coverage gap detection (unmatched cycle clustering)
    # ------------------------------------------------------------------

    def suggest_coverage_gaps(
        self,
        recent_window: int = 30,
        min_unmatched: int = 5,
        min_unmatched_rate: float = 0.20,
        low_confidence_threshold: float = 0.40,
        duration_bucket_s: float = 900.0,
    ) -> dict[str, Any]:
        """Detect gaps in profile coverage from recent unlabelled cycles.

        Looks at the ``recent_window`` most recent cycles.  When there are
        enough unmatched (no ``profile_name``) or low-confidence cycles,
        returns a record with:

          ``unmatched_count``       – number of cycles with no profile label
          ``low_confidence_count``  – number of labelled cycles below the confidence
                                      threshold (potential mis-matches)
          ``unmatched_rate``        – fraction of recent cycles that are unmatched
          ``suggest_create``        – True when unmatched_count >= min_unmatched
                                      AND unmatched_rate >= min_unmatched_rate
          ``duration_clusters``     – list of duration-cluster dicts for unmatched
                                      cycles: ``{"duration_bucket_min": int,
                                      "count": int, "avg_duration_s": float}``
                                      sorted descending by count. Only clusters
                                      with ≥ 2 members are returned.

        Returns an empty dict when there are fewer than ``min_unmatched``
        unmatched cycles or on any error (never raises).
        """
        try:
            cycles = self.get_past_cycles()
            recent = cycles[-recent_window:] if len(cycles) > recent_window else cycles
            if not recent:
                return {}

            unmatched: list[dict[str, Any]] = []
            low_conf: list[dict[str, Any]] = []
            for c in recent:
                name = c.get("profile_name")
                if not name:
                    unmatched.append(c)
                else:
                    conf = c.get("match_confidence")
                    if (
                        isinstance(conf, (int, float))
                        and not isinstance(conf, bool)
                        and conf < low_confidence_threshold
                    ):
                        low_conf.append(c)

            n_unmatched = len(unmatched)
            n_total = len(recent)
            rate = n_unmatched / n_total if n_total > 0 else 0.0

            if n_unmatched < min_unmatched:
                return {}

            # Cluster unmatched cycles by duration bucket (simple histogram)
            bucket_dur: dict[int, list[float]] = {}
            for c in unmatched:
                dur = c.get("duration")
                if isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur > 0:
                    bucket = int(dur // duration_bucket_s)
                    bucket_dur.setdefault(bucket, []).append(float(dur))

            clusters = []
            for bucket, durs in sorted(bucket_dur.items(), key=lambda kv: -len(kv[1])):
                if len(durs) >= 2:
                    clusters.append({
                        "duration_bucket_min": int(bucket * duration_bucket_s / 60),
                        "count": len(durs),
                        "avg_duration_s": round(float(np.mean(durs)), 1),
                    })

            # A3: Shape-similarity clustering within duration buckets
            profile_suggestions: list[dict[str, Any]] = []
            for bucket, durs in sorted(bucket_dur.items(), key=lambda kv: -len(kv[1])):
                if len(durs) < 2:
                    continue
                bucket_cycles = [
                    c for c in unmatched
                    if c.get("power_data")
                    and c.get("duration")
                    and int(float(c["duration"]) // duration_bucket_s) == bucket
                ]
                if len(bucket_cycles) < 2:
                    continue
                try:
                    from .signal_processing import resample_to_n  # noqa: PLC0415
                    traces: list[np.ndarray] = []
                    ids: list[str] = []
                    for c in bucket_cycles[:5]:  # cap at 5 per bucket for performance
                        raw = decompress_power_data(c)
                        if not raw:
                            continue
                        pwr = [float(p) for _, p in raw]
                        if len(pwr) < 5:
                            continue
                        t = np.asarray(resample_to_n(pwr, CLUSTER_RESAMPLE_N), dtype=float)
                        mx = t.max()
                        if mx > 0:
                            t = t / mx
                        traces.append(t)
                        ids.append(str(c.get("id", "")))
                    if len(traces) < 2:
                        continue
                    corrs = [
                        float(np.corrcoef(traces[i], traces[j])[0, 1])
                        for i in range(len(traces))
                        for j in range(i + 1, len(traces))
                    ]
                    valid_corrs = [r for r in corrs if np.isfinite(r)]
                    if not valid_corrs:
                        continue
                    avg_corr = float(np.mean(valid_corrs))
                    if avg_corr >= CLUSTER_SHAPE_SIMILARITY_THRESHOLD:
                        avg_dur_s = float(np.mean(durs))
                        profile_suggestions.append({
                            "suggested_name": f"~{int(avg_dur_s // 60)} min program",
                            "cycle_ids": [cid for cid in ids if cid],
                            "avg_duration_s": round(avg_dur_s, 1),
                            "count": len(ids),
                            "similarity": round(avg_corr, 3),
                        })
                except Exception:  # noqa: BLE001
                    continue

            # Most-recent unmatched cycle id, so the setup advisor's phase-2
            # "unmatched" nudge can deep-link straight to it (open_cycle:<id>)
            # instead of always falling back to the whole unlabelled list.
            last_unmatched_cycle_id = None
            for c in reversed(unmatched):
                cid = c.get("id")
                if cid:
                    last_unmatched_cycle_id = cid
                    break

            return {
                "unmatched_count": n_unmatched,
                "low_confidence_count": len(low_conf),
                "unmatched_rate": round(rate, 3),
                "suggest_create": bool(n_unmatched >= min_unmatched and rate >= min_unmatched_rate),
                "duration_clusters": clusters,
                "profile_suggestions": profile_suggestions,   # NEW (A3)
                "last_unmatched_cycle_id": last_unmatched_cycle_id,
            }
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    # Profile trend analysis (pure stats, appliance performance drift)
    # ------------------------------------------------------------------

    def compute_profile_trends(
        self,
        min_cycles: int = 12,
        recent_window: int = 8,
        slope_threshold_pct: float = 0.08,
    ) -> dict[str, dict[str, Any]]:
        """Detect per-profile duration / energy drift using ordinary least-squares.

        Uses all labeled cycles in chronological order.  For each profile with at
        least ``min_cycles`` labeled cycles:

        * Fits a linear trend (OLS via NumPy) to the most-recent N=``len(cycles)``
          values (not just the window — the window is used only for the *recent
          mean* display; the slope is computed on all data so short histories don't
          give noisy slopes).
        * Expresses the slope as percent-change-per-cycle relative to the mean,
          so the magnitude is comparable across low- and high-duration profiles.
        * Classifies as ``"up"`` / ``"down"`` / ``"stable"`` per metric when the
          normalized slope exceeds ``slope_threshold_pct`` (default 8%/cycle).

        Returns a dict mapping profile_name → trend record:
          ``duration_trend``            – "up" / "down" / "stable"
          ``duration_slope_pct``        – normalized slope (%/cycle)
          ``duration_recent_mean_s``    – mean duration of the last N cycles (s)
          ``energy_trend``              – "up" / "down" / "stable" (if available)
          ``energy_slope_pct``          – normalized slope (%/cycle)
          ``energy_recent_mean_wh``     – mean energy of the last N cycles (Wh)
          ``cycle_count``               – number of labeled cycles used
          ``recent_window``             – how many cycles the recent means cover

        Profiles with fewer than ``min_cycles`` labeled cycles are omitted.
        Never raises — returns ``{}`` on any error.
        """
        try:
            cycles = self.get_past_cycles()
            from collections import defaultdict  # pylint: disable=import-outside-toplevel
            by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for c in cycles:
                name = c.get("profile_name")
                if name:
                    by_profile[str(name)].append(c)

            result: dict[str, dict[str, Any]] = {}
            for name, pcy in by_profile.items():
                if len(pcy) < min_cycles:
                    continue
                # Preserve insertion order (chronological from get_past_cycles)
                durations = [float(c["duration"]) for c in pcy if c.get("duration")]
                energies = [float(c["energy_wh"]) for c in pcy if c.get("energy_wh")]

                if len(durations) < min_cycles:
                    continue

                def _slope_pct(values: list[float]) -> float:
                    """OLS slope as % of mean per cycle."""
                    arr = np.asarray(values, dtype=float)
                    n = len(arr)
                    x = np.arange(n, dtype=float)
                    x -= x.mean()
                    denom = float(np.dot(x, x))
                    if denom < 1e-9:
                        return 0.0
                    slope = float(np.dot(x, arr)) / denom
                    mean_val = float(np.mean(arr))
                    return (slope / mean_val) if abs(mean_val) > 1e-9 else 0.0

                def _classify(slope_pct: float) -> str:
                    if slope_pct > slope_threshold_pct:
                        return "up"
                    if slope_pct < -slope_threshold_pct:
                        return "down"
                    return "stable"

                dur_slope = _slope_pct(durations)
                dur_recent = float(np.mean(durations[-recent_window:]))

                rec: dict[str, Any] = {
                    "cycle_count": len(pcy),
                    "recent_window": min(recent_window, len(durations)),
                    "duration_trend": _classify(dur_slope),
                    "duration_slope_pct": round(dur_slope * 100, 2),
                    "duration_recent_mean_s": round(dur_recent, 1),
                }
                if len(energies) >= min_cycles:
                    en_slope = _slope_pct(energies)
                    en_recent = float(np.mean(energies[-recent_window:]))
                    rec["energy_trend"] = _classify(en_slope)
                    rec["energy_slope_pct"] = round(en_slope * 100, 2)
                    rec["energy_recent_mean_wh"] = round(en_recent, 1)
                result[name] = rec
            return result
        except Exception:  # noqa: BLE001
            return {}

    def compute_profile_advisories(self) -> list[dict[str, Any]]:
        """Actionable per-profile maintenance advisories from existing signals.

        Pure statistics (no ML): consolidates :meth:`compute_profile_health` and
        :meth:`compute_profile_trends` into a small ranked list of human-readable,
        actionable recommendations (e.g. "durations trending longer -> re-record").
        These surface as recommendations in the Profiles tab - never a
        notification. Each item is ``{profile, severity, code, message}``; severity
        is ``"warning"`` or ``"info"``. Returns ``[]`` on error (never raises).
        """
        try:
            advisories: list[dict[str, Any]] = []
            health = self.compute_profile_health() or {}
            trends = self.compute_profile_trends() or {}

            for name, h in health.items():
                if h.get("health_status") == "poor":
                    advisories.append({
                        "profile": name, "severity": "warning", "code": "poor_health",
                        "message": (
                            f"'{name}' has a low fit score - its recent cycles vary "
                            "a lot or match weakly. Review its cycles or re-record "
                            "the profile so matching and time estimates stay accurate."
                        ),
                        # Localization: panel renders _t(message_key, message_params,
                        # message). The English `message` above is the fallback.
                        "message_key": "msg.advisory_poor_health",
                        "message_params": {"name": name},
                    })
                elif h.get("shape_drift"):
                    corr = h.get("shape_drift_correlation")
                    corr_str = f" (correlation {corr:.2f})" if corr is not None else ""
                    advisories.append({
                        "profile": name, "severity": "info", "code": "shape_drift",
                        "message": (
                            f"'{name}' has drifted significantly from its original "
                            f"power shape{corr_str}. The appliance may have changed "
                            "behaviour over time (e.g. limescale, wear). Consider "
                            "re-recording this profile with recent cycles."
                        ),
                        "message_key": (
                            "msg.advisory_shape_drift_corr"
                            if corr is not None
                            else "msg.advisory_shape_drift"
                        ),
                        "message_params": (
                            {"name": name, "corr": f"{corr:.2f}"}
                            if corr is not None
                            else {"name": name}
                        ),
                    })

            for name, t in trends.items():
                # Skip profiles already flagged poor to avoid double advice.
                if health.get(name, {}).get("health_status") == "poor":
                    continue
                if t.get("duration_trend") == "up":
                    # duration_slope_pct is already expressed as %/cycle by
                    # compute_profile_trends (dur_slope * 100); do NOT re-scale.
                    pct = abs(float(t.get("duration_slope_pct") or 0.0))
                    advisories.append({
                        "profile": name, "severity": "info", "code": "duration_trend_up",
                        "message": (
                            f"'{name}' cycles are running progressively longer "
                            f"(about +{pct:.0f}% per cycle). If the appliance's "
                            "behaviour changed, re-record or rebuild this profile."
                        ),
                        "message_key": "msg.advisory_duration_trend_up",
                        "message_params": {"name": name, "pct": f"{pct:.0f}"},
                    })
                elif t.get("energy_trend") == "up":
                    # energy_slope_pct is already %/cycle (en_slope * 100); no re-scale.
                    pct = abs(float(t.get("energy_slope_pct") or 0.0))
                    advisories.append({
                        "profile": name, "severity": "info", "code": "energy_trend_up",
                        "message": (
                            f"'{name}' is drawing progressively more energy "
                            f"(about +{pct:.0f}% per cycle) - worth checking the "
                            "appliance if that is unexpected."
                        ),
                        "message_key": "msg.advisory_energy_trend_up",
                        "message_params": {"name": name, "pct": f"{pct:.0f}"},
                    })

            # Phase-structure consistency (phase-matching device types only). Uses
            # the cached per-role phase profile: a profile whose member cycles heat
            # for wildly different times, or where only some cycles heat at all,
            # most likely mixes different programs/temperatures under one label -
            # which hurts both matching and the phase-resolved ETA. Advisory only
            # (Profiles tab); no relabeling (phase matching does not label better).
            _sd = getattr(self, "_data", None)
            _envs = _sd.get("envelopes") if isinstance(_sd, dict) else None
            for pname, penv in (_envs if isinstance(_envs, dict) else {}).items():
                if health.get(pname, {}).get("health_status") == "poor":
                    continue  # avoid double advice
                pp = penv.get("phase_profile") if isinstance(penv, dict) else None
                if not isinstance(pp, dict):
                    continue
                try:
                    if int(pp.get("n_cycles") or 0) < PHASE_CONSISTENCY_MIN_CYCLES:
                        continue
                    heat = (pp.get("roles") or {}).get("heating") or {}
                    heat_mean = float(heat.get("dur_mean") or 0.0)
                    heat_std = float(heat.get("dur_std") or 0.0)
                    heat_occ = float(heat.get("occurrence") or 0.0)
                    heat_cv = (heat_std / heat_mean) if heat_mean > 60.0 else 0.0
                    mixed_temp = heat_cv > PHASE_HEAT_CV_WARN
                    mixed_prog = PHASE_HEAT_OCC_MIXED_LO <= heat_occ <= PHASE_HEAT_OCC_MIXED_HI
                    if not (mixed_temp or mixed_prog):
                        continue
                    advisories.append({
                        "profile": pname, "severity": "warning",
                        "code": "phase_inconsistent",
                        "message": (
                            f"'{pname}' looks like it mixes different programs or "
                            "temperatures - its cycles heat for very different lengths "
                            "of time. Splitting it into separate profiles (e.g. per "
                            "temperature) will improve matching and time estimates."
                        ),
                        "message_key": "msg.advisory_phase_inconsistent",
                        "message_params": {"name": pname},
                    })
                except (TypeError, ValueError):
                    continue

            # E1: suppress the "needs maintenance" nag (duration-trending-longer /
            # shape-drift/poor-fit) when the user recently logged a descale, filter
            # clean, or drum clean — any recent maintenance clears the reminder.
            try:
                # ``is True`` (not truthiness) so a mocked store whose method
                # returns a MagicMock does not accidentally suppress advisories.
                if any(
                    self.has_recent_maintenance(evt) is True
                    for evt in ("descale", "filter_clean", "drum_clean")
                ):
                    _nag_codes = {"duration_trend_up", "poor_health", "shape_drift"}
                    advisories = [a for a in advisories if a.get("code") not in _nag_codes]
            except Exception:  # noqa: BLE001
                pass

            order = {"warning": 0, "info": 1}
            advisories.sort(key=lambda a: order.get(a["severity"], 2))
            return advisories
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # Match ranking history (per-cycle ML commit snapshots)
    # ------------------------------------------------------------------

    def record_match_ranking_snapshot(
        self,
        start_time_iso: str,
        features: dict[str, float],
        top1_profile: str,
        top1_score: float,
        top2_score: float | None,
        candidate_count: int,
        cycle_id: str = "",
    ) -> None:
        """Append a live-match ranking snapshot for the active cycle.

        Snapshots are keyed by ``start_time_iso`` (and optionally ``cycle_id``) so
        that ``confirm_match_ranking_snapshots`` can back-fill the confirmed label
        when the cycle ends.  Pre-computed feature scalars are stored (not raw traces)
        to keep footprint small.  The store is NOT persisted here — the caller must
        schedule ``async_save``.
        """
        from .const import MATCH_RANKING_HISTORY_MAX  # pylint: disable=import-outside-toplevel
        history: list[dict[str, Any]] = self._data.setdefault("match_ranking_history", [])
        history.append({
            "start_time_iso": start_time_iso,
            "cycle_id": str(cycle_id) if cycle_id else "",
            "features": dict(features),
            "top1_profile": str(top1_profile),
            "top1_score": round(float(top1_score), 4),
            "top2_score": round(float(top2_score), 4) if top2_score is not None else None,
            "candidate_count": int(candidate_count),
            "confirmed_label": None,
        })
        # Trim to the most recent N snapshots.
        if len(history) > MATCH_RANKING_HISTORY_MAX:
            del history[: len(history) - MATCH_RANKING_HISTORY_MAX]

    def confirm_match_ranking_snapshots(
        self,
        start_time_iso: str,
        confirmed_label: str,
        cycle_id: str = "",
    ) -> int:
        """Back-fill the confirmed label for all snapshots belonging to a cycle.

        When ``cycle_id`` is provided (and the snapshot was recorded with one), matches
        by ``cycle_id`` to avoid cross-contamination between cycles that share the same
        second-resolution ``start_time_iso``.  Falls back to ``start_time_iso`` matching
        for snapshots recorded without a cycle_id (backward compatibility).
        Returns the number of snapshots updated.
        """
        history = self._data.get("match_ranking_history")
        if not isinstance(history, list):
            return 0
        updated = 0
        for snap in history:
            if not isinstance(snap, dict):
                continue
            snap_cid = snap.get("cycle_id") or ""
            if cycle_id and snap_cid:
                # Both sides have an ID — match by ID for precision.
                if snap_cid == cycle_id:
                    snap["confirmed_label"] = str(confirmed_label)
                    updated += 1
            else:
                # Legacy path: match by timestamp (no cycle_id on one or both sides).
                if snap.get("start_time_iso") == start_time_iso:
                    snap["confirmed_label"] = str(confirmed_label)
                    updated += 1
        return updated

    def get_match_ranking_history(self) -> list[dict[str, Any]]:
        """Return all ranking snapshots (confirmed and pending).

        Callers that build a training dataset should filter to snapshots where
        ``confirmed_label`` is not None.
        """
        raw = self._data.get("match_ranking_history")
        return list(raw) if isinstance(raw, list) else []

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
        # Prune pending feedback whose cycle no longer exists, so the device's
        # "needs review" count can't disagree with the Cycles review filter.
        if self.prune_orphaned_feedback():
            await self.async_save()

    # _migrate_v1_to_v2 and _decompress_power_from_raw removed; logic moved to WashDataStore

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
        ref_cycles: list[dict[str, Any]] = self._data.get("reference_cycles", []) or []
        if not profiles or not cycles:
            return stats

        # Sample validity must recognise imported reference cycles: an import-only
        # profile legitimately points its sample at a reference cycle. Without this,
        # such a sample looks "missing" and the repair below would steal an unrelated
        # unlabeled real cycle into the imported profile.
        by_id: dict[str, dict[str, Any]] = {
            c["id"]: c for c in list(cycles) + list(ref_cycles) if c.get("id")
        }

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

    def _add_cycle_data(self, cycle_data: CycleDict, target: list[CycleDict] | None = None) -> None:
        """Internal logic to add cycle data to storage.

        ``target`` defaults to ``past_cycles``; ``add_reference_cycle`` passes the
        separate ``reference_cycles`` list so imported cycles never enter usage stats.
        """
        dest = self._data["past_cycles"] if target is None else target
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
            if start_time_raw is not None:
                parsed_dt = _parse_start_dt(start_time_raw)
                if parsed_dt is not None:
                    start_time_iso = parsed_dt.isoformat()
                    # Keep original ISO string as-is if it was already a valid ISO string
                    if isinstance(start_time_raw, str) and dt_util.parse_datetime(start_time_raw) is not None:
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
                dest.append(cycle_data)
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

            # Compute and store energy (Wh) if not already set (e.g. by manager).
            # Shared trapezoidal integrator + data-driven outage gap (single source
            # with manager._on_cycle_end).
            if "energy_wh" not in cycle_data and len(ts_arr) > 1:
                sort_idx = np.argsort(ts_arr)
                ts_s = ts_arr[sort_idx]
                p_s = p_arr[sort_idx]
                cycle_data["energy_wh"] = round(
                    integrate_wh(ts_s, p_s, max_gap_s=energy_gap_threshold_s(ts_s)), 3
                )

            self._logger.debug(
                "add_cycle: stored %s samples at %.1fs intervals",
                len(stored),
                sampling_interval,
            )

        # 4. Handle Debug Data (Strip if not enabled)
        if hasattr(self, "_save_debug_traces") and not self._save_debug_traces:
            if "debug_data" in cycle_data:
                del cycle_data["debug_data"]

        dest.append(cycle_data)
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

        # Collect cycle IDs that have pending feedback - never strip their power_data
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

                    # EXEMPTION: Never strip power data from user-pinned "golden" cycles.
                    # The matcher uses a golden trace as the sharp single-cycle template
                    # (has_golden in the snapshot builder), and golden_profiles membership
                    # is gated on the trace still being present — trimming it would flip
                    # has_golden false and silently drop the profile back to the smeared
                    # envelope average.
                    rev = c.get("ml_review")
                    if isinstance(rev, dict) and rev.get("golden"):
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
        # Imported reference cycles are valid sample targets too (an import-only
        # profile points its sample there), so include them or such profiles would
        # be wrongly deleted as orphans.
        cycle_ids = {c["id"] for c in self._data.get("past_cycles", [])}
        cycle_ids |= {
            c["id"] for c in self._data.get("reference_cycles", []) if c.get("id")
        }
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

        # 3. Smart Process History (Merge/Split/Rebuild)
        proc_stats = await self.async_smart_process_history()
        stats["merged_cycles"] = proc_stats.get("merged", 0)
        stats["split_cycles"] = proc_stats.get("split", 0)

        # 4. Rebuild every profile's envelope so bands stay fresh, and report the
        # real count of successful rebuilds (not an approximation).
        rebuilt = 0
        for profile_name in list(self._data.get("profiles", {}).keys()):
            try:
                # Count only real rebuilds; a no-op/failed rebuild returns False.
                if await self.async_rebuild_envelope(profile_name):
                    rebuilt += 1
            except Exception:  # pylint: disable=broad-exception-caught
                self._logger.debug(
                    "Envelope rebuild failed for %s during maintenance", profile_name, exc_info=True
                )
        stats["rebuilt_envelopes"] = rebuilt

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

                    # Self-heal duration/end_time drift.  A freshly-finalized cycle
                    # always has last_offset == duration (the finalizer appends a
                    # terminal point at end_time, and add_cycle keeps the tail for
                    # completed cycles), so this is a no-op for healthy records.  It
                    # repairs older / manually-edited cycles whose trace was trimmed
                    # without updating duration + end_time - e.g. a legacy record
                    # whose drying tail was dropped while duration kept the pre-trim
                    # value (duration says 8820s but the trace and end_time end at
                    # 6845s).  Snap both to the trace so the three always agree.
                    pd_now = cycle.get("power_data")
                    if (
                        isinstance(pd_now, list)
                        and pd_now
                        and isinstance(pd_now[-1], (list, tuple))
                        and len(pd_now[-1]) == 2
                    ):
                        trace_end = float(pd_now[-1][0])
                        stored_dur = float(cycle.get("duration", 0.0) or 0.0)
                        si = float(cycle.get("sampling_interval", 0.0) or 0.0)
                        # Tolerance above rounding + one sampling gap; the drifts we
                        # repair are large (minutes), so this never churns healthy data.
                        tol = max(5.0, 2.0 * si)
                        if trace_end > 0 and abs(stored_dur - trace_end) > tol:
                            cycle["duration"] = round(trace_end, 1)
                            start_ts = _value_to_timestamp(cycle.get("start_time"))
                            if start_ts is not None:
                                cycle["end_time"] = dt_util.utc_from_timestamp(
                                    start_ts + trace_end
                                ).isoformat()
                            processed_count += 1
                            self._logger.info(
                                "Reprocess self-heal: cycle %s duration %.0fs -> %.0fs "
                                "(snapped to trace; end_time realigned)",
                                cycle.get("id"),
                                stored_dur,
                                trace_end,
                            )

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
        """Get storage usage stats.

        The on-disk size lookup uses blocking os.path calls, so it is offloaded
        to the executor. We deliberately never json.dumps(self._data) as a
        fallback: with power_data included the dataset can be many megabytes and
        serialising it on the event loop previously stalled the loop long enough
        that the diagnostics view appeared to hang on a spinner.
        """
        cycles = self._data.get("past_cycles", [])
        profiles = self._data.get("profiles", {})
        debug_traces_count = sum(1 for c in cycles if c.get("debug_data"))

        file_size_kb = 0.0
        path = getattr(self._store, "path", None)
        if path:
            file_size_kb = await self.hass.async_add_executor_job(
                _safe_file_size_kb, path
            )

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
        """Sync worker to parse data and build envelope (run in executor).

        Degenerate cycles (a near-flat trace whose peak is a tiny fraction of the
        profile's typical peak - e.g. a run where the power sensor wasn't
        reporting) are excluded so they cannot pollute the envelope average that
        the live matcher scores against, or drag ``avg_duration`` around.
        User-pinned golden cycles are always kept.
        """
        # First pass: decompress everything and record each cycle's peak so we
        # can judge degeneracy relative to the profile (works for both a 2000W
        # dishwasher and a low-power pump).
        parsed: list[tuple[list[float], list[float], float, bool, float]] = []
        for cycle in labeled_cycles:
            pairs = decompress_power_data(cycle)
            if len(pairs) < 3:
                continue
            offsets = [p[0] for p in pairs]
            values = [p[1] for p in pairs]
            stored_dur = float(cycle.get("duration", 0.0) or 0.0)
            authoritative_dur = float(max(offsets[-1], stored_dur))
            man_dur = cycle.get("manual_duration")
            final_dur = float(man_dur) if man_dur else authoritative_dur
            review = cycle.get("ml_review")
            is_golden = bool(review.get("golden")) if isinstance(review, dict) else False
            peak = max(values) if values else 0.0
            parsed.append((offsets, values, final_dur, is_golden, peak))

        if not parsed:
            return None

        # Degeneracy floor: below max(_DEGENERATE_POWER_FLOOR, 10% of the median
        # peak) a cycle is treated as a mis-capture and dropped (unless golden).
        peaks = sorted(p[4] for p in parsed)
        median_peak = peaks[len(peaks) // 2] if peaks else 0.0
        degen_floor = max(_DEGENERATE_POWER_FLOOR, 0.10 * median_peak)

        raw_cycles_data: list[tuple[list[float], list[float], float]] = []
        durations: list[float] = []
        golden_mask: list[bool] = []
        dropped = 0
        for offsets, values, final_dur, is_golden, peak in parsed:
            if peak < degen_floor and not is_golden:
                dropped += 1
                continue
            raw_cycles_data.append((offsets, values, final_dur))
            durations.append(final_dur)
            golden_mask.append(is_golden)

        # Never drop everything: if the filter removed all cycles (e.g. a truly
        # low-power profile misjudged), fall back to using them all.
        if not raw_cycles_data:
            for offsets, values, final_dur, is_golden, _peak in parsed:
                raw_cycles_data.append((offsets, values, final_dur))
                durations.append(final_dur)
                golden_mask.append(is_golden)
        elif dropped:
            self._logger.debug(
                "Envelope rebuild: excluded %d degenerate cycle(s) below %.0fW "
                "(median peak %.0fW)", dropped, degen_floor, median_peak,
            )

        if not raw_cycles_data:
            return None

        # Run Heavy Computation. When the profile has user-verified "golden"
        # cycles, they define the reference shape (see compute_envelope_worker).
        result = analysis.compute_envelope_worker(
            cast(Any, raw_cycles_data),
            self.dtw_bandwidth,
            reference_mask=golden_mask if any(golden_mask) else None,
        )

        if not result:
            return None

        return result, durations

    def _cycle_peak(self, cycle: CycleDict) -> float:
        """Peak power of a cycle's trace (0.0 if it has none)."""
        pairs = decompress_power_data(cycle)
        return max((p[1] for p in pairs), default=0.0) if pairs else 0.0

    def _select_reference_cycle_id(
        self, profile_name: str, target_duration: float | None = None
    ) -> str | None:
        """Choose the best reference cycle for a profile's matching template.

        Preference order: user-pinned golden cycles first, then non-degenerate
        cycles (peak not a tiny fraction of the profile's median peak - this
        rejects mis-captured near-flat traces), and among those the one closest
        to the profile's representative duration (so a truncated half-cycle is
        not chosen). Returns a cycle id, or None if there are no usable cycles.
        """
        cands = [
            c for c in list(self._data.get("past_cycles", [])) + list(self._data.get("reference_cycles", []))
            if c.get("profile_name") == profile_name
            and c.get("status") in ("completed", "force_stopped")
            and isinstance(c.get("power_data"), list) and len(c["power_data"]) >= 3
        ]
        if not cands:
            return None
        peaks = {c["id"]: self._cycle_peak(c) for c in cands}
        med_peak = sorted(peaks.values())[len(peaks) // 2] if peaks else 0.0
        degen_floor = max(_DEGENERATE_POWER_FLOOR, 0.10 * med_peak)

        golden = [
            c for c in cands
            if isinstance(c.get("ml_review"), dict) and c["ml_review"].get("golden")
        ]
        if golden:
            pool = golden
        else:
            pool = [c for c in cands if peaks.get(c["id"], 0.0) >= degen_floor] or cands

        if target_duration and target_duration > 0:
            best = min(pool, key=lambda c: abs(float(c.get("duration") or 0.0) - target_duration))
        else:
            # No duration hint: prefer the longest (avoids truncated half-cycles).
            best = max(pool, key=lambda c: float(c.get("duration") or 0.0))
        return best.get("id")

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
                continue  # all rows malformed - leave original trace untouched
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
        # A rebuild changes this profile's curve, which feeds group cohesion, so
        # invalidate the cohesion cache (not only on group mutations) to avoid stale
        # cohesion approving/rejecting a collapse against outdated shapes.
        self._cohesion_cache_generation += 1
        # 1. Gather Data (Main Thread)
        def _eligible(seq: list[CycleDict]) -> list[CycleDict]:
            return [
                c
                for c in seq
                if c.get("profile_name") == profile_name
                and c.get("status") in ("completed", "force_stopped")
                and c.get("duration", 0) > 60
            ]

        # Real cycles drive usage stats (energy/count). Imported reference cycles
        # additionally shape the curves + matching duration, but never usage stats.
        real_cycles = _eligible(self._data["past_cycles"])
        ref_cycles = _eligible(self._data.get("reference_cycles", []))
        shape_cycles = real_cycles + ref_cycles

        if not shape_cycles:
            if profile_name in self._data.get("envelopes", {}):
                del self._data["envelopes"][profile_name]
            return False

        # Kept for the fallback path + duration-stat code below (behaviour is
        # byte-identical to before when there are no reference cycles).
        labeled_cycles = shape_cycles

        # 2. Run Heavy Computation in Executor (Parsing + DTW)
        result_pkg = await self.hass.async_add_executor_job(
            self._rebuild_envelope_sync,
            shape_cycles
        )

        if not result_pkg:
            # Envelope shape couldn't be built (no power data / too few points).
            # Still update profile min/max/avg from raw cycle durations so that
            # a duration correction via feedback is immediately reflected in stats.
            if labeled_cycles and profile_name in self._data.get("profiles", {}):
                raw_durs = [
                    float(c.get("manual_duration") or c.get("duration", 0))
                    for c in labeled_cycles
                ]
                raw_durs = [d for d in raw_durs if d > 60]
                if raw_durs:
                    raw_arr_fallback = np.array(raw_durs, dtype=float)
                    self._data["profiles"][profile_name]["min_duration"] = float(np.min(raw_arr_fallback))
                    self._data["profiles"][profile_name]["max_duration"] = float(np.max(raw_arr_fallback))
                    self._data["profiles"][profile_name]["avg_duration"] = float(np.mean(raw_arr_fallback))
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

            # Re-point the matching reference to a golden / non-degenerate cycle
            # near the representative duration, so a mis-captured (flat) or
            # truncated cycle can never remain the profile's template.
            ref_id = self._select_reference_cycle_id(profile_name, avg_duration)
            if ref_id and self._data["profiles"][profile_name].get("sample_cycle_id") != ref_id:
                old = self._data["profiles"][profile_name].get("sample_cycle_id")
                self._data["profiles"][profile_name]["sample_cycle_id"] = ref_id
                self._logger.debug(
                    "Re-selected reference cycle for %s: %s -> %s",
                    profile_name, old, ref_id,
                )

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

        # Usage stats (avg_energy, cycle_count) come from REAL cycles only, so imported
        # reference cycles can shape the curve without ever inflating energy/count.
        # When there are no reference cycles this is byte-identical to the prior behaviour
        # (avg_energy from the avg curve, cycle_count = len(durations)).
        if ref_cycles:
            real_energies = [
                float(c["energy_wh"])
                for c in real_cycles
                if isinstance(c.get("energy_wh"), (int, float))
            ]
            avg_energy = (sum(real_energies) / len(real_energies) / 1000.0) if real_energies else 0.0
            cycle_count = len(real_cycles)
        else:
            # Average-curve energy (kWh) via the shared trapezoidal integrator.
            # avg_curve is in Watts, time_grid in seconds; integrate_wh returns Wh.
            avg_energy = integrate_wh(time_grid, avg_curve) / 1000.0
            cycle_count = len(durations)

        envelope_data: dict[str, Any] = {
            "time_grid": time_grid,  # Time grid used by manager for phase estimation
            "target_duration": target_duration,  # Target duration for phase estimation
            "min": to_points(min_curve),
            "max": to_points(max_curve),
            "avg": to_points(avg_curve),
            "std": to_points(std_curve),
            "cycle_count": cycle_count,
            "avg_energy": avg_energy,
            "duration_std_dev": duration_std_dev,
            "updated": dt_util.now().isoformat(),
        }

        # Derived cache: per-phase profile (per-role duration/energy priors) used by
        # phase-segmented matching / phase-resolved ETA. Built only for device types
        # phase matching is live-supported for; absent otherwise (consumers fall back
        # to the whole-cycle pipeline). Pure/cheap - segmentation is O(samples).
        device_type = str(
            self._data.get("profiles", {}).get(profile_name, {}).get("device_type") or ""
        )
        # Offload the per-cycle segmentation to the executor (it can be tens of ms
        # for very long traces; keep it off the event loop, like the envelope DTW).
        phase_profile = await self.hass.async_add_executor_job(
            self._compute_phase_profile, profile_name, shape_cycles, device_type
        )
        if phase_profile is not None:
            envelope_data["phase_profile"] = phase_profile

        if "envelopes" not in self._data:
            self._data["envelopes"] = {}
        self._data["envelopes"][profile_name] = envelope_data

        return True

    def _compute_phase_profile(
        self, profile_name: str, cycles: list[CycleDict], device_type: str
    ) -> dict[str, Any] | None:
        """Segment each member cycle and aggregate a per-role phase profile.

        Returns a JSON-safe dict for ``envelope["phase_profile"]`` or ``None`` when
        phase matching is not live-supported for this device type or no cycle could
        be segmented. Never raises (phase support must never break envelope rebuild).
        """
        try:
            if not phase_matching_live_supported(device_type):
                return None
            model = phase_model_for(device_type)
            if model is None:
                return None
            segmented: list = []
            for cycle in cycles:
                offsets = power_data_to_offsets(cycle.get("power_data") or [])
                if len(offsets) < 4:
                    continue
                t = [float(o) for o, _ in offsets]
                w = [float(p) for _, p in offsets]
                segs = segment_cycle(t, w, model)
                if segs:
                    segmented.append(segs)
            if not segmented:
                return None
            profile = build_phase_profile(profile_name, segmented)
            return phase_profile_to_dict(profile) if profile is not None else None
        except Exception:  # noqa: BLE001 - phase caching must never break rebuild
            self._logger.debug("phase-profile build failed for %s", profile_name, exc_info=True)
            return None

    def _group_scope(self, program: str) -> set[str] | None:
        """Phase-narrowing scope for the matched ``program``:

        * If ``program`` is in a group with >= 2 members, return that group's
          members - narrow WITHIN the family (design §9). This is both coherent
          (same program family as the displayed program) and accurate (picks the
          right temperature/spin variant among siblings).
        * Otherwise return ``None`` = no scope filter (consider ALL of the
          device's phase profiles). The Phase-0 gate showed that constraining an
          UNGROUPED cycle to only the whole-cycle-matched program regresses the
          ETA whenever that match is wrong (common on mislabeled data): the best
          ETA comes from letting the phase matcher pick the best-fitting profile,
          bounded by the ambiguity gate + cold-start floor. Grouping variants is
          the recommended workflow and restores full coherence.
        """
        try:
            for grp in self.get_profile_groups().values():
                members = grp.get("members") if isinstance(grp, dict) else None
                if isinstance(members, list) and program in members:
                    sib = {m for m in members if isinstance(m, str)}
                    if len(sib) >= 2:
                        return sib
        except Exception:  # noqa: BLE001
            self._logger.debug("_group_scope failed for %r", program, exc_info=True)
        return None

    def _candidate_phase_profiles(self, scope: set[str] | None = None) -> list:
        """Cached per-profile PhaseProfiles (from envelope['phase_profile']).

        Restricted to ``scope`` (profile names) when given, and always filtered to
        profiles with >= ``PHASE_PROFILE_MIN_CYCLES`` member cycles so a noisy
        single-cycle prior can't drive the ETA (cold-start floor).
        """
        out = []
        for name, env in (self._data.get("envelopes") or {}).items():
            if scope is not None and name not in scope:
                continue
            if isinstance(env, dict):
                pp = phase_profile_from_dict(env.get("phase_profile"))
                if pp is not None and pp.n_cycles >= PHASE_PROFILE_MIN_CYCLES:
                    out.append(pp)
        return out

    def phase_remaining(
        self,
        power_data: list,
        device_type: str,
        program: str | None = None,
    ) -> dict[str, Any] | None:
        """Phase-resolved remaining-time for a running cycle. Never raises.

        Segments the observed-so-far trace and matches it against the matched
        ``program``'s phase profile (and its group siblings, design §9), returning
        the winning member's per-role budget remaining. Returns ``None`` (caller
        keeps the current estimate) when: phase matching is not live-supported for
        the device type; ``program`` is unknown / has no cached phase profile
        (or too few cycles - cold-start floor); segmentation is degenerate; or the
        top two candidates are within ``MATCH_AMBIGUITY_MARGIN`` (ambiguous - do
        not commit a variant, design §7).

        This is the *phase* half of the blended ETA; the blend with the current
        estimator lives in ``progress.compute_progress`` (single source of truth).
        Pure and cheap (segmentation + per-role agreement, no DTW) - safe to call
        inline from the async matching path.
        """
        try:
            if not phase_matching_live_supported(device_type):
                return None
            model = phase_model_for(device_type)
            if model is None or not program:
                return None
            candidates = self._candidate_phase_profiles(self._group_scope(program))
            if not candidates:
                return None
            offsets = power_data_to_offsets(power_data or [])
            if len(offsets) < 4:
                return None
            t = [float(o) for o, _ in offsets]
            w = [float(p) for _, p in offsets]
            segs = segment_cycle(t, w, model, partial=True)
            if not segs:
                return None
            ranked = match_phase_profiles(segs, candidates, {})
            if not ranked:
                return None
            # Ambiguity gate: a near-tie among group members is not a confident
            # variant call - fall back rather than swing the ETA between budgets.
            if (len(ranked) >= 2
                    and (ranked[0].score - ranked[1].score) < MATCH_AMBIGUITY_MARGIN):
                return None
            best = next((c for c in candidates if c.name == ranked[0].name), None)
            remaining = phase_eta(segs, best) if best is not None else None
            if remaining is None:
                return None
            return {
                "remaining_s": float(remaining),
                "matched": ranked[0].name,
                "score": float(ranked[0].score),
            }
        except Exception:  # noqa: BLE001 - phase ETA must never break the estimate
            self._logger.debug("phase_remaining failed", exc_info=True)
            return None






    def get_envelope(self, profile_name: str) -> JSONDict | None:
        """Get cached envelope for a profile, or None if not available."""
        envelopes = self._data.get("envelopes", {})
        if isinstance(envelopes, dict):
            envelopes_map = cast(dict[str, Any], envelopes)
            env = envelopes_map.get(profile_name)
            return cast(JSONDict, env) if isinstance(env, dict) else None
        return None

    def reference_curve(
        self, profile_name: str, n: int = REFERENCE_PROFILE_CURVE_POINTS
    ) -> JSONDict | None:
        """Compact, read-only reference power curve for a matched profile.

        Downsamples the profile envelope's average power-over-time shape to at
        most ``n`` points so it can be exposed as an entity attribute for
        consumers (e.g. home energy managers) that want the *forward-looking
        load shape* rather than a scalar ETA - to anticipate, say, a heating
        spike later in the cycle. Shape::

            {
                "points": [[offset_s, watts], ...],  # <= n samples
                "duration_s": float,                 # profile target duration
                "cycle_count": int,                  # cycles behind the average
            }

        Offsets are absolute seconds from cycle start (0 .. ``duration_s``); a
        consumer derives the *remaining* curve by slicing from the live progress
        position (already exposed as the progress sensor). The curve is static
        per profile - it only changes when the profile is re-learned - so it can
        be surfaced as an attribute without recorder churn.

        Pure statistics (no ML); never raises - returns ``None`` when the
        envelope is missing or too short to be meaningful.
        """
        try:
            env = self.get_envelope(profile_name)
            if not isinstance(env, dict):
                return None
            avg = env.get("avg")
            if (
                not isinstance(avg, list)
                or len(avg) < 2
                or not isinstance(avg[0], (list, tuple))
                or len(avg[0]) < 2
            ):
                return None
            ts = np.asarray([float(p[0]) for p in avg], dtype=float)
            ws = np.asarray([float(p[1]) for p in avg], dtype=float)
            if ts.size < 2 or not (np.all(np.isfinite(ts)) and np.all(np.isfinite(ws))):
                return None
            if ts[-1] <= ts[0]:
                return None
            duration = float(env.get("target_duration") or 0.0)
            if duration <= 0:
                duration = float(ts[-1])
            count = max(2, min(int(n), ts.size))
            grid = np.linspace(float(ts[0]), float(ts[-1]), count)
            vals = np.interp(grid, ts, ws)
            points = [
                [int(round(float(g))), round(float(v), 1)]
                for g, v in zip(grid, vals)
            ]
            return {
                "points": points,
                "duration_s": round(duration, 1),
                "cycle_count": int(env.get("cycle_count") or 0),
            }
        except Exception:  # pragma: no cover - defensive; never break the sensor
            return None

    def get_profile_power_profile(
        self, profile_name: str, interval_s: float = 900.0
    ) -> list[float]:
        """Average power (W) per fixed interval across a profile's learned shape.

        Resamples the profile envelope's average power-over-time curve into
        consecutive ``interval_s`` buckets (default 15 min) and returns the mean
        watts in each, e.g. ``[2200, 2200, 800, 800, 1500, 500, 400, 200]`` - the
        flat per-slot array external planners such as tibber_prices'
        ``power_profile`` consume to pick the cheapest window to run the appliance
        (issue #272). Unlike :meth:`reference_curve` (a downsampled time/watt shape
        surfaced on the running-program sensor), this is a fixed-interval average
        exposed per profile so it can be read for planning before a cycle starts.

        The final bucket is averaged only over the part of the cycle that actually
        falls inside it. Pure statistics; never raises. Returns an empty list when
        the profile has no learned envelope yet.
        """
        try:
            if interval_s <= 0:
                return []
            env = self.get_envelope(profile_name)
            if not isinstance(env, dict):
                return []
            avg = env.get("avg")
            if (
                not isinstance(avg, list)
                or len(avg) < 2
                or not isinstance(avg[0], (list, tuple))
                or len(avg[0]) < 2
            ):
                return []
            ts = np.asarray([float(p[0]) for p in avg], dtype=float)
            ws = np.asarray([float(p[1]) for p in avg], dtype=float)
            if ts.size < 2 or not (np.all(np.isfinite(ts)) and np.all(np.isfinite(ws))):
                return []
            if ts[-1] <= ts[0]:
                return []
            total = float(env.get("target_duration") or 0.0)
            if total <= 0:
                total = float(ts[-1])
            if total <= 0:
                return []
            n_buckets = int(math.ceil(total / interval_s))
            out: list[float] = []
            for k in range(n_buckets):
                lo = k * interval_s
                hi = min(lo + interval_s, total)
                if hi <= lo:
                    break
                # Time-average the curve over the half-open slot [lo, hi) on a fine
                # interpolated grid, so the result is independent of the envelope's
                # grid spacing. endpoint=False keeps a slot boundary from being
                # double-counted into the adjacent slot.
                fine = np.linspace(lo, hi, 16, endpoint=False)
                out.append(round(float(np.mean(np.interp(fine, ts, ws))), 1))
            return out
        except Exception:  # pragma: no cover - defensive; never break the sensor
            return []

    def compute_envelope_conformance(
        self,
        profile_name: str,
        points: list[tuple[float, float]],
    ) -> dict[str, Any] | None:
        """Score how well the current power trace conforms to the profile envelope band.

        Resamples ``points`` to the envelope's time grid (scaling by the ratio of
        the current elapsed time to the envelope duration) and computes the fraction
        of samples that land within the [lower, upper] band.  Returns a dict:

          ``conformance``     – fraction of samples inside the envelope band (0–1)
          ``outside_frac``    – fraction outside the band (1 - conformance)
          ``samples``         – number of samples compared
          ``envelope_name``   – profile_name used

        Returns ``None`` if the envelope or points are unavailable / too short.
        Never raises.

        This is a complementary signal to match confidence: confidence measures
        shape correlation, while conformance measures absolute level/spread.  A
        high-confidence but low-conformance score may indicate the cycle was
        mis-detected (merged cycles, offset start) or the appliance is running
        anomalously.
        """
        try:
            if not points or len(points) < 4:
                return None
            env = self.get_envelope(profile_name)
            if not env:
                return None
            time_grid = env.get("time_grid")
            lower_raw = env.get("min")
            upper_raw = env.get("max")
            if not time_grid or not lower_raw or not upper_raw:
                return None

            tg = np.asarray(time_grid, dtype=float)
            # Envelope curves are stored as [[t, y], ...]; extract the y column.
            lo = np.asarray(
                [p[1] if isinstance(p, (list, tuple)) else p for p in lower_raw],
                dtype=float,
            )
            hi = np.asarray(
                [p[1] if isinstance(p, (list, tuple)) else p for p in upper_raw],
                dtype=float,
            )
            env_duration = float(tg[-1]) if len(tg) > 1 else 1.0

            # Normalise observed trace to [0, env_duration] time range
            t_obs = np.asarray([t for t, _ in points], dtype=float)
            p_obs = np.asarray([p for _, p in points], dtype=float)
            obs_duration = float(t_obs[-1]) if len(t_obs) > 1 else 1.0
            t_scaled = t_obs * (env_duration / obs_duration) if obs_duration > 0 else t_obs

            # Interpolate envelope bounds at scaled observed time points (clamp ends)
            t_clamped = np.clip(t_scaled, tg[0], tg[-1])
            lo_interp = np.interp(t_clamped, tg, lo)
            hi_interp = np.interp(t_clamped, tg, hi)

            inside = np.sum((p_obs >= lo_interp) & (p_obs <= hi_interp))
            n = len(p_obs)
            conformance = float(inside) / n if n > 0 else 0.0

            return {
                "conformance": round(conformance, 3),
                "outside_frac": round(1.0 - conformance, 3),
                "samples": n,
                "envelope_name": profile_name,
            }
        except Exception:  # noqa: BLE001
            return None

    def detect_cycle_artifacts(
        self,
        profile_name: str,
        points: list[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Detect transient artifacts in a cycle by comparing it to the profile band.

        Walks the trace against the matched profile's envelope and flags contiguous
        segments that deviate from what the program normally does:

          * ``pause``  – power fell to ~0 where the profile expects activity and then
            resumed (e.g. the door was opened mid-cycle to add an item, or a manual
            pause); excluded at the very end (that's just the cycle finishing).
          * ``dip``    – ran sustainedly below the usual power band.
          * ``spike``  – ran sustainedly above the usual power band.

        Returns a chronological list of ``{type, start_s, end_s, detail, severity}``
        in the trace's own time offsets (so the panel can mark them on the graph),
        capped to the most significant few. ``[]`` when unavailable. Pure statistics
        (no ML), never raises — and the events double as candidate labels for a
        future supervised anomaly model.
        """
        try:
            if not points or len(points) < 6:
                return []
            env = self.get_envelope(profile_name)
            if not env:
                return []
            tg = np.asarray(env.get("time_grid") or [], dtype=float)
            # Envelope curves are stored as [[t, y], ...]; extract the y column.
            def _extract_y(raw: list) -> np.ndarray:
                if not raw:
                    return np.array([], dtype=float)
                return np.asarray(
                    [p[1] if isinstance(p, (list, tuple)) else p for p in raw],
                    dtype=float,
                )
            lo = _extract_y(env.get("min") or [])
            hi = _extract_y(env.get("max") or [])
            avg = _extract_y(env.get("avg") or [])
            if tg.size < 2 or lo.size != tg.size or hi.size != tg.size:
                return []

            t_obs = np.asarray([t for t, _ in points], dtype=float)
            p_obs = np.asarray([max(0.0, float(p)) for _, p in points], dtype=float)
            obs_dur = float(t_obs[-1]) if t_obs.size > 1 else 1.0
            env_dur = float(tg[-1]) if tg.size > 1 else 1.0
            if obs_dur <= 0 or env_dur <= 0:
                return []

            # Map each observed time to the envelope grid (same scaling as
            # compute_envelope_conformance) so the band is aligned to progress.
            t_scaled = np.clip(t_obs * (env_dur / obs_dur), tg[0], tg[-1])
            lo_i = np.interp(t_scaled, tg, lo)
            hi_i = np.interp(t_scaled, tg, hi)
            avg_i = np.interp(t_scaled, tg, avg) if avg.size == tg.size else (lo_i + hi_i) / 2.0

            peak = max(float(np.max(hi)), 1.0)
            active_thr = max(5.0, 0.05 * peak)   # profile expects real power here
            pause_thr = max(2.0, 0.03 * peak)    # observed effectively off
            margin = max(10.0, 0.12 * peak)      # band slack to ignore edge noise

            # Pre-check: if the cycle's linear-resampled alignment is too poor
            # (e.g. duration or phase structure doesn't match the profile), the
            # envelope comparison produces unreliable artifacts.  Compute tight
            # conformance (no margin) over the active region; bail out when more
            # than 45 % of expected-active samples are outside the raw band.
            active_mask = avg_i > active_thr
            n_active = int(np.sum(active_mask))
            if n_active > 10:
                outside_tight = int(np.sum(
                    (p_obs[active_mask] < lo_i[active_mask]) |
                    (p_obs[active_mask] > hi_i[active_mask])
                ))
                if outside_tight / n_active > 0.45:
                    return []

            states: list[str] = []
            for i in range(len(p_obs)):
                expects_power = avg_i[i] > active_thr
                if expects_power and p_obs[i] <= pause_thr:
                    states.append("pause")
                elif expects_power and p_obs[i] < lo_i[i] - margin:
                    states.append("dip")
                elif p_obs[i] > hi_i[i] + margin:
                    states.append("spike")
                else:
                    states.append("ok")

            min_dur = {"pause": 25.0, "dip": 45.0, "spike": 30.0}
            sev_scale = {"pause": 300.0, "dip": 600.0, "spike": 300.0}
            events: list[dict[str, Any]] = []
            n = len(states)
            i = 0
            while i < n:
                s = states[i]
                if s == "ok":
                    i += 1
                    continue
                j = i
                while j + 1 < n and states[j + 1] == s:
                    j += 1
                start_s, end_s = float(t_obs[i]), float(t_obs[j])
                dur = end_s - start_s
                resumes = (j + 1 < n) and float(np.max(p_obs[j + 1:])) > active_thr
                if dur >= min_dur[s] and (s != "pause" or resumes):
                    if s == "pause":
                        detail = (f"Power dropped to near zero for ~{int(dur)}s then resumed — "
                                  "likely the door was opened mid-cycle or the cycle was paused.")
                        detail_key = "msg.artifact_pause_detail"
                    elif s == "dip":
                        detail = f"Drew below the usual power band for ~{int(dur)}s."
                        detail_key = "msg.artifact_dip_detail"
                    else:
                        detail = f"Drew above the usual power band for ~{int(dur)}s."
                        detail_key = "msg.artifact_spike_detail"
                    events.append({
                        "type": s,
                        "start_s": round(start_s, 1),
                        "end_s": round(end_s, 1),
                        "detail": detail,
                        "detail_key": detail_key,
                        "detail_params": {"n": int(dur)},
                        "severity": round(min(1.0, dur / sev_scale[s]), 3),
                    })
                i = j + 1

            events.sort(key=lambda e: -e["severity"])
            events = events[:6]
            events.sort(key=lambda e: e["start_s"])
            return events
        except Exception:  # noqa: BLE001
            return []

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
        group_members: dict[str, list[str]] = {}
        member_snaps: dict[str, dict[str, Any]] = {}

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

            # Prepare Snapshots. Imported reference cycles are eligible as matching
            # templates alongside real cycles (so an import-only profile can match).
            all_cycles = list(self._data["past_cycles"]) + list(self._data.get("reference_cycles", []))
            # Precompute per-profile lookups ONCE so the loop below is O(profiles),
            # not O(profiles x cycles). Rescanning all_cycles with next()/any() for
            # every profile made matching quadratic and stalled low-power hosts on
            # auto-label (many matches x many cycles) - issue #311. Selections are
            # byte-identical: cycles_by_id keeps the FIRST occurrence (== next()),
            # labeled_by_profile keeps the first eligible cycle in all_cycles order,
            # and golden_profiles mirrors the any(...) golden test.
            cycles_by_id: dict[str, CycleDict] = {}
            labeled_by_profile: dict[str, CycleDict] = {}
            golden_profiles: set[str] = set()
            for c in all_cycles:
                cid = c.get("id")
                if cid is not None and cid not in cycles_by_id:
                    cycles_by_id[cid] = c
                pname = c.get("profile_name")
                if not pname or not c.get("power_data"):
                    continue
                if (
                    pname not in labeled_by_profile
                    and c.get("status") in ("completed", "force_stopped")
                ):
                    labeled_by_profile[pname] = c
                rev = c.get("ml_review")
                if isinstance(rev, dict) and rev.get("golden"):
                    golden_profiles.add(pname)

            snapshots: list[dict[str, Any]] = []
            skipped_profiles: list[str] = []
            for name, profile in self._data["profiles"].items():
                # Try sample_cycle_id first, fall back to any labeled cycle
                sample_id = profile.get("sample_cycle_id")
                sample_cycle = cycles_by_id.get(sample_id) if sample_id else None
                # Fallback: find ANY completed cycle labeled with this profile
                if not sample_cycle:
                    sample_cycle = labeled_by_profile.get(name)
                # Boost user-pinned "golden" cycles: when a profile has one, use
                # its sharp single-cycle trace as the matching template instead
                # of the envelope average. The envelope average smears the
                # wash-phase peaks (each cycle's spikes land at slightly
                # different times), which hurts correlation for sharply-shaped
                # programs; a trusted golden cycle preserves that shape.
                has_golden = name in golden_profiles

                # Prefer envelope avg curve when ≥2 labeled cycles have been
                # confirmed - it gives a more representative reference signal
                # than the original sample alone, so confidence improves over
                # time as the user keeps confirming correct detections. Skipped
                # when a golden cycle is pinned (see above).
                envelope = self._data.get("envelopes", {}).get(name)
                _env_avg = envelope.get("avg") if envelope else None
                if (
                    not has_golden
                    and envelope
                    and envelope.get("cycle_count", 0) >= 2
                    and _env_avg
                    and isinstance(_env_avg[0], (list, tuple))
                    and len(_env_avg[0]) >= 2
                ):
                    avg_y = [float(p[1]) for p in _env_avg]
                    _env_ts_duration = (
                        float(_env_avg[-1][0]) - float(_env_avg[0][0])
                        if len(_env_avg) > 1 else 0.0
                    )
                    avg_duration = (
                        envelope.get("target_duration") or
                        profile.get("avg_duration") or
                        _env_ts_duration or
                        None
                    )
                    if not avg_duration:
                        skipped_profiles.append(
                            f"{name}: no valid duration (envelope has no target_duration, avg_duration, or timestamp span)"
                        )
                        continue
                    snapshots.append({
                        "name": name,
                        "avg_duration": float(avg_duration),
                        "sample_power": avg_y,
                    })
                    continue

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
                # avg_duration preference order:
                #   1. profile["avg_duration"] (rolling average, most accurate)
                #   2. sample_cycle["duration"] (raw cycle field)
                #   3. timestamp span of sample_seg (estimate from the resampled data)
                # Profiles created before avg_duration tracking was added may have
                # 0 or a missing value; falling back to the segment estimate prevents
                # update_match() from always seeing expected_duration=0, which
                # silences time-remaining estimates and logs a misleading warning.
                _seg_ts_duration = (
                    float(sample_seg.timestamps[-1]) - float(sample_seg.timestamps[0])
                    if len(sample_seg.timestamps) > 1 else 0.0
                )
                avg_dur = (
                    profile.get("avg_duration") or
                    sample_cycle.get("duration") or
                    _seg_ts_duration
                )
                if not avg_dur:
                    skipped_profiles.append(
                        f"{name}: no valid duration (avg_duration, cycle duration, and timestamp span all zero/missing)"
                    )
                    continue
                snapshots.append({
                    "name": name,
                    "avg_duration": float(avg_dur),
                    "sample_power": sample_seg.power.tolist(),
                    "sample_dt": used_dt
                })

            if skipped_profiles:
                self._logger.debug(
                    "Profile matching skipped %d profiles: %s",
                    len(skipped_profiles),
                    "; ".join(skipped_profiles)
                )

            # Stage 5: collapse cohesive near-duplicate groups into one aggregate
            # candidate each (loose groups stay individual). No-op without groups.
            snapshots, group_members, member_snaps = self._grouped_snapshots(snapshots)

            config = {
                "min_duration_ratio": self._min_duration_ratio,
                "max_duration_ratio": self._max_duration_ratio,
                "dtw_bandwidth": self.dtw_bandwidth,
                # On-device tuned scoring weights (opt-in); empty = shipped defaults.
                **self._matching_overrides(),
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

        # Reconstruct MatchResult. Top-level ambiguity (safeguard #1): a close
        # group-vs-runner-up call is flagged so it goes to the uncertain/feedback
        # path, not a confident commit.
        margin, is_ambiguous = _ambiguity_from_candidates(candidates)

        best_name = best["name"]
        best_duration = best["profile_duration"]
        # Stage 5: if a group won, pick the best-fitting member.
        if best_name in group_members:
            chosen, member_fit, member_dur = self._stage5_pick_member(
                current_power_list, current_duration, group_members[best_name], member_snaps
            )
            best_name = chosen
            if member_dur:
                best_duration = member_dur
            # Safeguard #2: the group aggregate matched but if the chosen member
            # does not individually fit reasonably (vs the group score), the real
            # program may be a different single profile -> treat as uncertain.
            # member_fit is a Stage-2-only score; best["score"] includes DTW-blend +
            # duration/energy agreement (typically 25-30% higher). Use 0.55× to avoid
            # the threshold being effectively too strict for DTW-boosted groups.
            if member_fit is not None and best["score"] > 0 and member_fit < 0.55 * best["score"]:
                is_ambiguous = True
            # Overrun guard: if the cycle has already outlasted the chosen member's
            # expected duration we may be running the longer group member — downgrade
            # to ambiguous so Smart Termination falls back to the power timeout.
            if best_duration and current_duration > best_duration * 1.05:
                is_ambiguous = True
            # Relabel the winning candidate for the ranking / diagnostics.
            candidates = [{**best, "name": best_name, "profile_duration": best_duration}, *candidates[1:]]

        matched_phase = None
        if best_name:
            # Always resolve phase for the matched profile so phase sensors can
            # show user-assigned phase names even when confidence is moderate.
            matched_phase = self.check_phase_match(best_name, current_duration)

        # Detect "prefix ambiguity": a non-winning candidate whose duration is
        # significantly longer than the matched profile AND whose shape matched
        # well before Stage-4 penalised its duration. When this is true the
        # current trace may be a prefix of that longer program, not a complete
        # short cycle. Signal cycle_detector to block Smart Termination; the
        # power-based fallback timeout will decide instead.
        best_dur = best_duration or 0.0
        is_prefix_ambiguous = best_dur > 0 and any(
            float(c.get("profile_duration") or 0) > best_dur * SMART_TERM_LANDSCAPE_RATIO
            and float(c.get("shape_score", c.get("score", 0))) >= SMART_TERM_LANDSCAPE_MIN_SHAPE
            for c in candidates[1:]
        )

        return MatchResult(
            best_name,
            best["score"],
            best_duration,
            matched_phase,
            candidates[:5],
            is_ambiguous,
            margin,
            ranking=candidates[:5],  # populate ranking (consumed for training snapshots)
            is_prefix_ambiguous=is_prefix_ambiguous,
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
            sample_p_data = decompress_power_data(sample_cycle)
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
            "dtw_bandwidth": self.dtw_bandwidth,
            # On-device tuned scoring weights (opt-in); empty = shipped defaults.
            **self._matching_overrides(),
        }

        candidates = analysis.compute_matches_worker(
            p_list, duration, cast(Any, snapshots), config
        )

        if not candidates:
            return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

        best = candidates[0]

        margin, is_ambiguous = _ambiguity_from_candidates(candidates)

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

    @property
    def has_real_profiles(self) -> bool:
        """True if at least one stored profile is backed by a real cycle.

        A profile counts as "real" when it has a labelled cycle in ``past_cycles``
        OR an imported ``reference_cycle`` (store-adopted templates that the matcher
        treats as eligible snapshots — see the snapshot builder in async_match). An
        import-only install has zero past_cycles but is fully matchable, so it must
        pass this gate too, otherwise matching and the setup notifications are
        skipped for it entirely.
        """
        profile_names = self._data.get("profiles", {}).keys()
        if not profile_names:
            return False
        assigned = {
            c.get("profile_name")
            for c in self._data.get("past_cycles", [])
            if c.get("profile_name")
        }
        if assigned.intersection(profile_names):
            return True
        ref_assigned = {
            c.get("profile_name")
            for c in self._data.get("reference_cycles", [])
            if c.get("profile_name")
        }
        return bool(ref_assigned.intersection(profile_names))

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

            # Per-profile cost aggregates from frozen per-cycle costs. Never raises;
            # both default to None when no cycle carries a cost.
            avg_cost: float | None = None
            total_cost: float | None = None
            try:
                costs = [float(c["cost"]) for c in p_cycles if c.get("cost") is not None]
                if costs:
                    total_cost = round(sum(costs), 4)
                    avg_cost = round(sum(costs) / len(costs), 4)
            except Exception:  # noqa: BLE001
                avg_cost = None
                total_cost = None

            # Downsampled power signature (the profile's real average power curve),
            # for the profile-card mini graph. Small (<=40 pts), power values only.
            sig_curve: list[float] = []
            try:
                avg_pts = (envelope or {}).get("avg") or []
                ps = [
                    float(p[1]) for p in avg_pts
                    if isinstance(p, (list, tuple)) and len(p) >= 2
                ]
                if not ps:  # legacy flat [p, ...] envelopes
                    ps = [float(x) for x in avg_pts if isinstance(x, (int, float))]
                if len(ps) > 40:
                    step = (len(ps) - 1) / 39.0
                    sig_curve = [ps[round(i * step)] for i in range(40)]
                else:
                    sig_curve = ps
            except Exception:  # noqa: BLE001
                sig_curve = []

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
                    "avg_cost": avg_cost,
                    "total_cost": total_cost,
                    "signature_curve": sig_curve,
                    "is_imported": self.profile_has_reference_cycles(name),
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
            # 1. Update past + imported reference cycles (imports carry profile_name
            #    too; leaving them under the old name orphans them from the matcher).
            for cycle in (
                list(self._data.get("past_cycles", []))
                + list(self._data.get("reference_cycles", []))
            ):
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

            # 4. Update group membership (rename the member in any group)
            for g in self._data.get("profile_groups", {}).values():
                mems = g.get("members") if isinstance(g, dict) else None
                if isinstance(mems, list) and old_name in mems:
                    g["members"] = [new_name if m == old_name else m for m in mems]

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

        # Handle cycles (past + imported reference; both carry profile_name, so an
        # imported cycle would otherwise keep a dangling label for a deleted profile).
        count = 0
        for cycle in (
            list(self._data.get("past_cycles", []))
            + list(self._data.get("reference_cycles", []))
        ):
            if cycle.get("profile_name") == name:
                if unlabel_cycles:
                    cycle["profile_name"] = None
                count += 1

        await self.async_save()
        action = "unlabeled" if unlabel_cycles else "orphaned"
        self._logger.info("Deleted profile '%s', %s %s cycles", name, action, count)
        return count

    async def clear_all_data(self) -> None:
        """Clear all profiles, cycle data, and derived state."""
        self._data["past_cycles"] = []
        self._data["reference_cycles"] = []
        self._data["profiles"] = {}
        self._data["envelopes"] = {}
        self._data["suggestions"] = {}
        self._data["feedback_history"] = {}
        self._data["pending_feedback"] = {}
        self._data["auto_adjustments"] = []
        self._data["active_cycle"] = None
        self._data["last_active_save"] = None
        # Newer persisted state must also be wiped, else a "wipe all" leaves trained
        # models, groups, matcher tuning, histories, and counters behind.
        self._data["custom_phases"] = []
        self._data["ml_model_versions"] = {}
        self._data["profile_groups"] = {}
        self._data["maintenance_log"] = []
        self._data["matching_config"] = {}
        self._data["match_ranking_history"] = []
        self._data["ml_last_training_run"] = None
        self._data["ml_training_history"] = {}
        self._data["lifetime_energy_wh"] = 0.0
        self._data["lifetime_cycle_count"] = 0
        self._data["settings_changelog"] = []
        self._data["suggestion_apply_cycle_count"] = 0
        self._cached_sample_segments = {}
        self._cohesion_cache = {}
        self._cohesion_cache_generation += 1
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
            # Imported reference recording (separate list, never in usage stats):
            # relabelling just moves which profile's template it seeds.
            ref = next(
                (c for c in self.get_reference_cycles() if c.get("id") == cycle_id),
                None,
            )
            if ref is not None:
                await self._assign_reference_cycle_profile(ref, profile_name)
                return
            raise ValueError(f"Cycle {cycle_id} not found")

        # Track old profile for envelope rebuild
        old_profile = cycle.get("profile_name")

        if profile_name and profile_name not in self._data.get("profiles", {}):
            raise ValueError(f"Profile '{profile_name}' not found. Create it first.")

        # Preserve original auto-assigned label before first manual relabeling
        if profile_name and old_profile and not cycle.get("original_auto_label"):
            orig_src = cycle.get("label_source", "")
            if orig_src in ("auto_match", "auto_label_post", "auto_label_service"):
                cycle["original_auto_label"] = old_profile

        # Update cycle
        cycle["profile_name"] = profile_name if profile_name else None
        cycle["label_source"] = "manual" if profile_name else None

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

    async def _assign_reference_cycle_profile(
        self, ref: CycleDict, profile_name: str | None
    ) -> None:
        """Reassign an imported reference recording to a different profile.

        The cycle stays in ``reference_cycles`` (out of usage stats); only which
        profile envelope it seeds changes. Rebuilds the old and new envelopes.
        ``profile_name=None`` clears the label (the recording then seeds nothing).
        """
        if profile_name and profile_name not in self._data.get("profiles", {}):
            raise ValueError(f"Profile '{profile_name}' not found. Create it first.")
        old_profile = ref.get("profile_name")
        ref_id = ref.get("id")
        ref["profile_name"] = profile_name if profile_name else None
        if old_profile and old_profile != profile_name:
            # The moved cycle may have been the old profile's sample. Clear that
            # stale pointer so the old profile can't resolve the moved trace (now
            # another profile's) by id, and drop the old profile if it is now empty
            # (mirrors _delete_reference_cycle; prevents repair adopting a real cycle).
            op = self._data.get("profiles", {}).get(old_profile)
            if op is not None and op.get("sample_cycle_id") == ref_id:
                op["sample_cycle_id"] = None
            old_has_cycles = any(
                c.get("profile_name") == old_profile
                for c in list(self._data.get("past_cycles", []))
                + list(self._data.get("reference_cycles", []))
            )
            if old_has_cycles:
                await self.async_rebuild_envelope(old_profile)
            else:
                self._data.get("profiles", {}).pop(old_profile, None)
                self._data.get("envelopes", {}).pop(old_profile, None)
        if profile_name:
            await self.async_rebuild_envelope(profile_name)
        await self.async_save()
        self._logger.info(
            "Reassigned imported reference cycle %s to profile '%s'",
            ref.get("id"), profile_name,
        )

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
            power_data = decompress_power_data(cycle)
            if not power_data or len(power_data) < 10:
                stats["skipped"] += 1
                continue

            # Try to match
            result = await self.async_match_profile(power_data, cycle["duration"])

            # Honor the ambiguity safeguard: never auto-label a close/ambiguous match.
            if result.best_profile and result.confidence >= confidence_threshold and not result.is_ambiguous:
                current_label = cycle.get("profile_name")
                # Sanitize: strip heavy current/sample arrays before persisting.
                ranking_top5 = [
                    {
                        "name": c.get("name"),
                        "score": round(float(c.get("score", 0.0)), 3),
                        "profile_duration": c.get("profile_duration"),
                    }
                    for c in (getattr(result, "ranking", []) or [])[:5]
                ]

                # If overwriting, check if new match is different and better/valid
                if current_label:
                    if current_label != result.best_profile:
                        # Preserve original label before first auto-service relabeling
                        if not cycle.get("original_auto_label"):
                            orig_src = cycle.get("label_source", "")
                            if orig_src in ("auto_match", "auto_label_post", "auto_label_service"):
                                cycle["original_auto_label"] = current_label
                        cycle["profile_name"] = result.best_profile
                        cycle["match_confidence"] = float(result.confidence)
                        cycle["label_source"] = "auto_label_service"
                        if ranking_top5:
                            cycle["match_ranking_top5"] = ranking_top5
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
                    cycle["match_confidence"] = float(result.confidence)
                    cycle["label_source"] = "auto_label_service"
                    if ranking_top5:
                        cycle["match_ranking_top5"] = ranking_top5
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

    async def async_backfill_match_confidence(self) -> int:
        """Populate match_confidence for labeled cycles that predate the field.

        Runs the matcher once per cycle with profile_name set but no
        match_confidence, and persists the resulting confidence if the same
        profile is returned. Returns the number of cycles updated. Safe to
        call repeatedly - already-backfilled cycles are skipped.
        """
        cycles = self._data.get("past_cycles", []) or []
        updated = 0
        for cycle in cycles:
            if cycle.get("match_confidence") is not None:
                continue
            profile_name = cycle.get("profile_name")
            if not profile_name:
                continue
            power_data = decompress_power_data(cycle)
            if not power_data or len(power_data) < 10:
                continue
            try:
                result = await self.async_match_profile(
                    power_data, cycle.get("duration", 0)
                )
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if result.best_profile == profile_name and result.confidence > 0:
                cycle["match_confidence"] = float(result.confidence)
                updated += 1
        if updated:
            await self.async_save()
            self._logger.info("Backfilled match_confidence on %d cycles", updated)
        return updated

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



    def export_data(
        self, entry_data: JSONDict | None = None, entry_options: JSONDict | None = None
    ) -> JSONDict:
        # Return a serializable snapshot of the store for backup/export.
        # Includes config entry data/options so users can transfer fine-tuned settings.
        opts = entry_options or {}
        data = entry_data or {}
        _FINGERPRINT_KEYS = (
            "off_delay", "min_power", "sampling_interval",
            "start_power_threshold", "idle_power_threshold",
            "min_cycle_duration", "max_cycle_duration",
            "delay_start_detect_enabled",
        )
        device_fingerprint: JSONDict = {
            "device_type": data.get("device_type") or opts.get("device_type", "unknown"),
        }
        for key in _FINGERPRINT_KEYS:
            if key in opts:
                device_fingerprint[key] = opts[key]
        # Never let a backup/diagnostics export carry the GitHub refresh token: the
        # (legacy, now-global) per-device store account may still hold it. Shallow-copy
        # so we can drop the credential without mutating live state, and so a caller
        # can't alias-mutate self._data through the returned snapshot.
        export_store = dict(self._data)
        export_store.pop("store_account", None)
        return {
            "version": STORAGE_VERSION,
            "entry_id": self.entry_id,
            "exported_at": dt_util.now().isoformat(),
            "device_fingerprint": device_fingerprint,
            "data": export_store,
            "entry_data": data,
            "entry_options": opts,
        }

    async def async_import_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Import data from JSON payload (migration aware).
        # Unwrap HA diagnostics download file (outer HA wrapper: {home_assistant, data, ...})
        if "home_assistant" in payload and "data" in payload:
            payload = payload["data"]
            self._logger.info("Detected HA diagnostics file wrapper, unwrapping 'data'")

        # Unwrap our integration's diagnostics format ({entry, manager_state, store_export, ...})
        if "store_export" in payload:
            payload = payload["store_export"]
            self._logger.info("Detected diagnostics store_export format, unwrapping")

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
        if not isinstance(data_dict.get("reference_cycles"), list):
            data_dict["reference_cycles"] = []
        data_dict.setdefault("envelopes", {})

        if not data_dict.get("profiles") and not data_dict.get("past_cycles"):
            raise ValueError(
                "Import payload contains no profiles or cycles — aborting to prevent data loss"
            )
        self._data = data_dict
        self._cached_sample_segments = {}
        await self.async_save()

        # Strip diagnostic redaction sentinels so they don't overwrite real settings
        def _strip_redacted(d: dict) -> dict:
            if not isinstance(d, dict):
                return {}
            return {k: v for k, v in d.items() if v != "**REDACTED**"}

        return {
            "entry_data": _strip_redacted(payload.get("entry_data", {})),
            "entry_options": _strip_redacted(payload.get("entry_options", {})),
        }


    async def delete_cycle(self, cycle_id: str) -> bool:
        """Delete a cycle by ID."""
        cycles = cast(list[CycleDict], self._data.get("past_cycles", []))
        initial_len = len(cycles)
        cycle_to_delete = next((c for c in cycles if c.get("id") == cycle_id), None)
        if not cycle_to_delete:
            # Not a real cycle -- it may be an imported store recording, which
            # lives in the separate reference_cycles list (never in usage stats).
            return await self._delete_reference_cycle(cycle_id)

        profile_name = cycle_to_delete.get("profile_name")
        self._data["past_cycles"] = [c for c in cycles if c.get("id") != cycle_id]

        if len(self._data["past_cycles"]) < initial_len:
            # Check profile references
            for _p_name, p_data in self.get_profiles().items():
                if p_data.get("sample_cycle_id") == cycle_id:
                    p_data["sample_cycle_id"] = None

            # Drop any pending feedback for this cycle so it can't orphan the
            # "needs review" count (the count would say 1 while the cycle is gone).
            self._data.get("pending_feedback", {}).pop(cycle_id, None)

            # Rebuild envelope if cycle belonged to a profile
            if profile_name:
                await self.async_rebuild_envelope(profile_name)

            await self.async_save()
            return True
        return False

    async def _delete_reference_cycle(self, cycle_id: str) -> bool:
        """Delete a single imported store recording from ``reference_cycles``.

        Rebuilds the affected profile's envelope so removing a bad import
        immediately stops influencing the matcher template. Returns False when
        no reference cycle carries that id.
        """
        refs = cast(list[CycleDict], self._data.get("reference_cycles", []))
        cycle = next((c for c in refs if c.get("id") == cycle_id), None)
        if cycle is None:
            return False
        profile_name = cycle.get("profile_name")
        self._data["reference_cycles"] = [c for c in refs if c.get("id") != cycle_id]
        # Clear any profile that sampled this now-deleted reference cycle, mirroring
        # the real-cycle path in delete_cycle, so no sample id is left dangling.
        for _p_name, p_data in self.get_profiles().items():
            if p_data.get("sample_cycle_id") == cycle_id:
                p_data["sample_cycle_id"] = None
        if profile_name:
            # If removing this recording leaves the profile with no cycles at all
            # (no real, no imported), drop the now-empty profile. Otherwise a
            # sampleless import-only profile would be re-populated by
            # async_repair_profile_samples stealing an unlabeled real cycle into it.
            remaining = any(
                c.get("profile_name") == profile_name
                for c in list(self._data.get("past_cycles", []))
                + list(self._data.get("reference_cycles", []))
            )
            if remaining:
                await self.async_rebuild_envelope(profile_name)
            else:
                self._data.get("profiles", {}).pop(profile_name, None)
                self._data.get("envelopes", {}).pop(profile_name, None)
        await self.async_save()
        return True

    def get_cycle_power_data(self, cycle_id: str) -> list[tuple[float, float]]:
        """Return decompressed power data for a cycle as [(offset_s, watts), ...].

        Returns an empty list if the cycle is not found or has no power data.
        """
        cycle = next(
            (c for c in self.get_past_cycles() if c.get("id") == cycle_id), None
        )
        if cycle is None:
            # Imported store recordings live in a separate list; the panel opens
            # them from the same Cycles table, so look them up here too.
            cycle = next(
                (c for c in self.get_reference_cycles() if c.get("id") == cycle_id),
                None,
            )
        if cycle is None:
            return []
        return decompress_power_data(cycle)

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

        p_data = decompress_power_data(cycle)
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

        # Clear manual_duration override - trimmed duration is now authoritative
        cycle.pop("manual_duration", None)

        # Mark the cycle as edited so store-upload provenance (qc) can distinguish a
        # trimmed cycle from a plain detected one.
        meta = cycle.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["edited"] = True
            meta["trim"] = [round(new_start_s, 1), round(new_end_s, 1)]

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
        p_data = decompress_power_data(cycle)
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

    def build_split_segments_from_offsets(
        self,
        cycle: CycleDict,
        split_offsets_s: list[float],
        min_segment_s: float = 60.0,
    ) -> list[tuple[float, float]]:
        """Build segments for a manual split from explicit offsets (seconds from cycle start).

        Returns adjacent [(start, end)] segments covering the cycle window, split at the
        given offsets. Offsets are sorted and deduplicated; offsets outside the cycle window
        or producing a sub-`min_segment_s` slice are dropped. Returns [] if fewer than two
        segments would result.
        """
        p_data = decompress_power_data(cycle)
        if not p_data:
            return []

        last_t = float(p_data[-1][0])
        if last_t <= 0:
            return []

        unique_offsets = sorted(
            {round(float(o), 3) for o in split_offsets_s if 0.0 < float(o) < last_t}
        )
        if not unique_offsets:
            return []

        filtered_offsets: list[float] = []
        for offset in unique_offsets:
            if offset <= min_segment_s:
                continue
            if offset >= (last_t - min_segment_s):
                continue
            if filtered_offsets and (offset - filtered_offsets[-1]) < min_segment_s:
                continue
            filtered_offsets.append(offset)

        if not filtered_offsets:
            return []

        boundaries = [0.0, *filtered_offsets, last_t]
        segments: list[tuple[float, float]] = []
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            if (seg_end - seg_start) >= min_segment_s:
                segments.append((seg_start, seg_end))

        if len(segments) < 2:
            return []

        self._logger.debug(
            "Built manual split for %s: %d segments at offsets %s",
            cycle.get("id"),
            len(segments),
            filtered_offsets,
        )
        return segments

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
        start_dt_base_parsed = _parse_start_dt(cycle["start_time"])
        if not start_dt_base_parsed:
            return []

        start_ts = start_dt_base_parsed.timestamp()

        # Decompress original data
        p_data_tuples = decompress_power_data(cycle)
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

        # Rebuild envelopes ONLY for the profiles whose dataset actually changed:
        # the original profile (it lost the parent cycle) plus any profile a labeled
        # segment was assigned to. This replaces a blanket rebuild-all-envelopes in
        # the caller, which re-scanned every profile serially and stalled low-power
        # hosts on a split (issue #311 follow-up).
        touched: set[str] = set()
        if original_profile:
            touched.add(original_profile)
        for seg in segments:
            if isinstance(seg, dict):
                seg_prof = seg.get("profile")
                if seg_prof:
                    touched.add(seg_prof)
        for name in touched:
            await self.async_rebuild_envelope(name)

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

        # Sort by time - use timestamp comparison to handle mixed timezone offsets correctly
        def _cycle_start_ts(c: CycleDict) -> float:
            ts = _value_to_timestamp(c.get("start_time"))
            return ts if ts is not None else float("inf")

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
        c1_start_dt = _parse_start_dt(c1["start_time"])
        if not c1_start_dt:
            return None

        # Helper to get parsed points from a cycle
        def get_points(cy: CycleDict) -> list[tuple[float, float, float]]:
            # content: [(timestamp, offset, power)]
            raw = decompress_power_data(cy)
            res: list[tuple[float, float, float]] = []
            if not raw:
                return []
            base_dt = _parse_start_dt(cy["start_time"])
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
            c_start_dt = _parse_start_dt(next_c.get("start_time"))
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
            fallback_end_dt = _parse_start_dt(last_cycle.get("end_time"))
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

