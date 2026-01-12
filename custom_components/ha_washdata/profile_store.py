"""Profile storage and matching logic for HA WashData."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import Any, TypeAlias, cast

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
)
from .features import compute_signature
from .signal_processing import resample_uniform, resample_adaptive

_LOGGER = logging.getLogger(__name__)

JSONDict: TypeAlias = dict[str, Any]
CycleDict: TypeAlias = dict[str, Any]


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
                    if k not in ("current", "sample")
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

    # Handle missing start_time gracefully?
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
        self._save_debug_traces = save_debug_traces
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold
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

    def repair_profile_samples(self) -> dict[str, int]:
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
                self.rebuild_envelope(profile_name)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        return stats

    async def async_save(self) -> None:
        """Save data to storage."""
        await self._store.async_save(self._data)

    async def async_save_active_cycle(self, detector_snapshot: JSONDict) -> None:
        """Save the active cycle state separately (or in main data)."""
        # We can store it in the main store, but we need to ensure we don't wear out flash
        # if this is called often.
        # Home Assistant's Store helper writes atomically.
        # Let's put it in _data but only save if significant change?
        # Actually Manager throttles this call.
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

    def clear_active_cycle(self) -> None:
        """Clear active cycle data."""
        if "active_cycle" in self._data:
            del self._data["active_cycle"]
            # We don't necessarily need to save immediately, can wait for next save
            # But safer to save.
            # to be safe let's just schedule a save task in manager?
            # Or just save here.
            # Since this happens once per cycle end, it's fine.
            # We must be async though.
            raise NotImplementedError("Use async_clear_active_cycle")

    async def async_clear_active_cycle(self) -> None:
        """Clear the active cycle snapshot from storage."""
        if "active_cycle" in self._data:
            del self._data["active_cycle"]
            await self._store.async_save(self._data)

    def add_cycle(self, cycle_data: CycleDict) -> None:
        """Add a completed cycle to history."""
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

            cycle_data["power_data"] = stored
            cycle_data["sampling_interval"] = round(sampling_interval, 1)

            # Helper to get arrays for signature
            ts_arr = np.array(offsets)
            p_arr = np.array([p for _, p in stored])

            # Compute and store signature
            if len(ts_arr) > 1:
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
        self._enforce_retention()

    def _enforce_retention(self) -> None:
        """Apply retention policy:
        - Keep at most _max_past_cycles cycles (oldest removed)
        - For each profile, keep only the last N cycles with full power_data; strip older power_data
        - Keep a reasonable number of unlabeled full traces to allow auto-labeling
        - Update envelopes for affected profiles
        """
        raw_cycles = self._data.get("past_cycles", [])
        cycles: list[CycleDict] = (
            cast(list[CycleDict], raw_cycles) if isinstance(raw_cycles, list) else []
        )
        if not cycles:
            return

        def _start_time(cycle: CycleDict) -> str:
            return str(cycle.get("start_time", ""))

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
                cy_id = cy.get("id")
                # If a profile sample points here, try to move to most recent cycle of that profile
                for name, ref_id in list(sample_refs.items()):
                    if ref_id == cy_id:
                        # find newest cycle for that profile
                        newest = next(
                            (
                                c
                                for c in reversed(cycles)
                                if c.get("profile_name") == name
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

        affected_profiles: set[str] = set()
        for key, group in by_profile.items():
            # newest first based on start_time
            try:
                group.sort(key=_start_time)
            except Exception:  # pylint: disable=broad-exception-caught
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

        # 3) Rebuild envelopes for affected profiles
        for p in affected_profiles:
            try:
                self.rebuild_envelope(p)
            except Exception as e:  # pylint: disable=broad-exception-caught
                _LOGGER.debug(
                    "Envelope rebuild skipped for '%s' during retention: %s", p, e
                )

    async def delete_cycle(self, cycle_id: str) -> bool:
        """Delete a cycle by ID. Returns True if deleted, False if not found.
        Also removes any profiles that reference this cycle."""
        cycles = self._data["past_cycles"]
        for i, cycle in enumerate(cycles):
            if cycle.get("id") == cycle_id:
                profile_name = cycle.get("profile_name")
                cycles.pop(i)
                # Clean up any profiles referencing this cycle
                orphaned_profiles = [
                    name
                    for name, profile in self._data["profiles"].items()
                    if profile.get("sample_cycle_id") == cycle_id
                ]
                for name in orphaned_profiles:
                    del self._data["profiles"][name]
                    _LOGGER.info(
                        "Removed orphaned profile '%s' (referenced deleted cycle %s)",
                        name,
                        cycle_id,
                    )

                # If cycle was labeled (and profile not orphaned/deleted), rebuild statistics
                if (
                    profile_name
                    and profile_name not in orphaned_profiles
                    and profile_name in self._data["profiles"]
                ):
                    self.rebuild_envelope(profile_name)
                    _LOGGER.info(
                        "Rebuilt statistics for '%s' after cycle deletion", profile_name
                    )

                await self.async_save()
                _LOGGER.info("Deleted cycle %s", cycle_id)
                # Trigger smart processing after delete
                await self.async_smart_process_history()
                return True
        _LOGGER.warning("Cycle %s not found for deletion", cycle_id)
        return False

    def cleanup_orphaned_profiles(self) -> int:
        """Remove profiles that reference non-existent cycles.
        Returns number of profiles removed."""
        cycle_ids = {c["id"] for c in self._data.get("past_cycles", [])}
        orphaned = [
            name
            for name, profile in self._data["profiles"].items()
            if profile.get("sample_cycle_id") not in cycle_ids
        ]

        for name in orphaned:
            del self._data["profiles"][name]
            _LOGGER.info(
                "Cleaned up orphaned profile '%s' (cycle no longer exists)", name
            )

        return len(orphaned)

    async def async_run_maintenance(
        self, lookback_hours: int = 24, gap_seconds: int = 3600
    ) -> dict[str, int]:
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
        proc_stats = await self.async_smart_process_history(
            hours=lookback_hours, gap_seconds=gap_seconds
        )
        stats["merged_cycles"] = proc_stats.get("merged", 0)
        stats["split_cycles"] = proc_stats.get("split", 0)
        stats["rebuilt_envelopes"] = len(self._data.get("profiles", {})) # Approximation of rebuilt count

        # 4. Save if any changes made (smart process saves internally if needed, but explicit save safe)
        if any(stats.values()):
            await self.async_save()
            _LOGGER.info("Maintenance completed: %s", stats)

        return stats

    async def async_reprocess_all_data(self) -> int:
        """Reprocess all historical data to update signatures and rebuild envelopes.

        This is a non-destructive operation for raw cycle data. It:
        1. Recalculates signatures for ALL past cycles using current logic.
        2. Rebuilds all profile envelopes from scratch.
        3. Updates global stats.

        Returns total number of cycles processed.
        """
        cycles = self._data.get("past_cycles", [])
        if not cycles:
            return 0

        _LOGGER.info("Starting reprocessing of %s cycles...", len(cycles))
        processed_count = 0

        # 1. Update Signatures for all cycles
        for cycle in cycles:
            # We process ALL cycles, not just those without signatures,
            # to ensure they use the latest algorithm.
            if cycle.get("power_data"):
                try:
                    # Decompress using helper
                    tuples = decompress_power_data(cycle)
                    if tuples and len(tuples) > 10:
                        start_ts = datetime.fromisoformat(
                            cycle["start_time"]
                        ).timestamp()
                        ts_arr = []
                        p_arr = []
                        for t_str, p in tuples:
                            t = datetime.fromisoformat(t_str).timestamp()
                            ts_arr.append(t - start_ts)
                            p_arr.append(p)

                        # Compute fresh signature
                        sig = compute_signature(np.array(ts_arr), np.array(p_arr))
                        cycle["signature"] = dataclasses.asdict(sig)
                        processed_count += 1
                except Exception as e:  # pylint: disable=broad-exception-caught
                    _LOGGER.warning(
                        "Failed to reprocess signature for cycle %s: %s",
                        cycle.get("id"),
                        e,
                    )

        # 2. Rebuild all envelopes (clearing old ones first is implied by rebuild_envelope logic if we iterate all)
        # But to be safe and clean, let's force a full rebuild.
        # rebuilding_all_envelopes does iterate known profiles.
        self.rebuild_all_envelopes()

        # 3. Save changes
        await self.async_save()
        _LOGGER.info("Reprocessing complete. Updated %s cycles.", processed_count)
        return processed_count

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage usage stats."""
        # Proxy to internal store
        return await self._store.get_storage_stats()

    async def async_clear_debug_data(self) -> int:
        """Clear debug data."""
        return await self._store.async_clear_debug_data()

    def rebuild_all_envelopes(self) -> int:
        """Rebuild envelopes for all profiles. Returns count of envelopes rebuilt."""
        count = 0
        for profile_name in list(self._data["profiles"].keys()):
            if self.rebuild_envelope(profile_name):
                count += 1
        return count

    def rebuild_envelope(self, profile_name: str) -> bool:
        """
        Build/rebuild statistical envelope for a profile from all labeled cycles.

        Creates min/max/avg/std power curves by normalizing all cycles to same
        TIME DURATION (not sample count), accounting for different sampling rates
        (e.g., 3s intervals vs 60s intervals).

        Returns True if envelope was built, False if insufficient data.
        """
        # Get ALL completed cycles labeled with this profile
        labeled_cycles = [
            c
            for c in self._data["past_cycles"]
            if c.get("profile_name") == profile_name
            and c.get("status") in ("completed", "force_stopped")
            # Filter out noise/debounce (< 60s) to preventing poisoning
            # We trust the user if they manually forced it, but general rule applies.
            and c.get("duration", 0) > 60
        ]

        if not labeled_cycles or len(labeled_cycles) < 1:
            # Clear envelope if it exists
            if profile_name in self._data.get("envelopes", {}):
                del self._data["envelopes"][profile_name]
            return False

        # Extract and normalize all power curves
        normalized_curves: list[tuple[np.ndarray, np.ndarray]] = []
        sampling_rates: list[float] = []

        for cycle in labeled_cycles:
            power_data_raw = cycle.get("power_data", [])
            if not isinstance(power_data_raw, list):
                continue

            power_data_items: list[Any] = cast(list[Any], power_data_raw)
            if len(power_data_items) < 3:
                continue

            # Extract power values from [offset, power] pairs
            pairs: list[tuple[float, float]] = []
            for item in power_data_items:
                if not isinstance(item, (list, tuple)):
                    continue
                try:
                    a, b = cast(tuple[Any, Any], item)
                except (TypeError, ValueError):
                    continue
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    pairs.append((float(a), float(b)))
            if len(pairs) < 3:
                continue

            offsets = np.array([o for o, _ in pairs], dtype=float)
            values = np.array([p for _, p in pairs], dtype=float)

            if len(values) >= 3:
                # Use TIME as x-axis, not sample index
                # This accounts for different sampling rates (3s vs 60s intervals)
                normalized_curves.append((offsets, values))

                # Track sampling interval for diagnostics
                if len(offsets) > 1:
                    intervals = np.diff(offsets)
                    sampling_rate = float(np.median(intervals[intervals > 0]))
                    sampling_rates.append(sampling_rate)

        if not normalized_curves:
            if profile_name in self._data.get("envelopes", {}):
                del self._data["envelopes"][profile_name]
            return False

        # Find common time range (0 to max_time_duration)
        max_times: list[float] = [
            float(offsets[-1]) for offsets, _ in normalized_curves
        ]
        target_duration = float(np.median(max_times))  # Use median duration

        # Resample all curves to same TIME axis
        # Create uniform time grid from 0 to target_duration
        avg_sample_rate = float(np.median(sampling_rates)) if sampling_rates else 1.0
        num_points = max(50, int(target_duration / avg_sample_rate))  # ~50-300 points
        time_grid = np.linspace(0.0, target_duration, num_points)

        # Use the actual max offset from each curve as duration
        durations: list[float] = [
            float(offsets[-1]) for offsets, _ in normalized_curves
        ]

        min_duration = float(np.min(durations))
        max_duration = float(np.max(durations))

        # Update profile stats in storage
        if profile_name in self._data.get("profiles", {}):
            self._data["profiles"][profile_name]["min_duration"] = min_duration
            self._data["profiles"][profile_name]["max_duration"] = max_duration
            # avg_duration is usually updated elsewhere but let's ensure it's consistent
            # self._data["profiles"][profile_name]["avg_duration"] = float(np.mean(durations))

        resampled: list[np.ndarray] = []
        for offsets, values in normalized_curves:
            # Interpolate this cycle to the common time grid
            curve_resampled = np.interp(time_grid, offsets, values)
            resampled.append(curve_resampled)

        # Stack into 2D array and calculate statistics
        curves_array = np.array(resampled)

        # Calculate Signature Stats (from signatures of contributing cycles)
        sig_distribution = {}
        contributing_sigs = [
            c.get("signature") for c in labeled_cycles if c.get("signature")
        ]

        if contributing_sigs:
            # Aggregate keys
            keys = contributing_sigs[0].keys()
            for k in keys:
                vals = [s[k] for s in contributing_sigs if k in s]
                if vals:
                    sig_distribution[k] = {
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "avg": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "med": float(np.median(vals)),  # Medoid approx?
                    }

        envelope: JSONDict = {
            "min": np.min(curves_array, axis=0).tolist(),
            "max": np.max(curves_array, axis=0).tolist(),
            "avg": np.mean(curves_array, axis=0).tolist(),
            "std": np.std(curves_array, axis=0).tolist(),
            "time_grid": time_grid.tolist(),  # Store time axis for reference
            "cycle_count": len(resampled),
            "target_duration": float(target_duration),
            "sampling_rates": list(sampling_rates),
            "updated_at": dt_util.now().isoformat(),
        }

        # Calculate Energy and Consistency Metrics
        try:
            # Duration Consistency
            duration_std_dev = float(np.std(durations)) if durations else 0.0
            envelope["duration_std_dev"] = duration_std_dev

            # Energy (kWh)
            energy_values = []
            max_powers = []
            for offsets, values in normalized_curves:
                # Integrate Power(W) over Time(s) = Joules
                # Use np.trapezoid (NumPy 2.0+) or fallback to np.trapz (legacy)
                if hasattr(np, "trapezoid"):
                    joules = np.trapezoid(values, offsets)
                else:
                    joules = getattr(np, "trapz")(values, offsets)

                kwh = joules / 3600000.0
                energy_values.append(kwh)
                max_powers.append(np.max(values) if len(values) > 0 else 0)

            envelope["avg_energy"] = (
                float(np.mean(energy_values)) if energy_values else 0.0
            )
            envelope["energy_std_dev"] = (
                float(np.std(energy_values)) if energy_values else 0.0
            )
            envelope["avg_peak_power"] = (
                float(np.mean(max_powers)) if max_powers else 0.0
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.warning(
                "Failed to calculate advanced stats for %s: %s", profile_name, e
            )
            envelope["avg_energy"] = 0.0
            envelope["duration_std_dev"] = 0.0

        # Cache in storage
        if "envelopes" not in self._data:
            self._data["envelopes"] = {}
        self._data["envelopes"][profile_name] = envelope
        if sig_distribution:
            self._data["envelopes"][profile_name]["signature_stats"] = sig_distribution

        _LOGGER.debug(
            "Rebuilt envelope for '%s': %s cycles, duration=%.0fs, avg_sample_rate=%.1fs, "
            "normalized_to=%s time-aligned points",
            profile_name,
            len(resampled),
            target_duration,
            avg_sample_rate,
            num_points,
        )

        return True

    def generate_profile_svg(self, profile_name: str) -> str | None:
        """Generate an SVG string for the profile's power envelope."""
        envelope = self.get_envelope(profile_name)
        if not envelope or not envelope.get("time_grid"):
            return None

        try:
            time_grid = envelope["time_grid"]
            avg_curve = envelope["avg"]
            min_curve = envelope["min"]
            max_curve = envelope["max"]

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

    def get_envelope(self, profile_name: str) -> JSONDict | None:
        """Get cached envelope for a profile, or None if not available."""
        envelopes = self._data.get("envelopes", {})
        if isinstance(envelopes, dict):
            envelopes_map = cast(dict[str, Any], envelopes)
            env = envelopes_map.get(profile_name)
            return cast(JSONDict, env) if isinstance(env, dict) else None
        return None

    def match_profile(
        self, current_power_data: list[tuple[str, float]], current_duration: float
    ) -> MatchResult:
        """
        Attempt to match current running cycle to a known profile.
        Returns MatchResult object.

        Pipeline:
        1. Fast Reject (Duration/Energy)
        2. Core Similarity (MAE, Correlation, Peak) on uniform grid
        3. DTW-lite (Tie-breaking or Confirmation)
        """
        if not current_power_data or len(current_power_data) < 10:
            return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

        # 1. Pre-process Current Data
        try:
            # We assume current_power_data is sorted by time
            # Parse first/last to get relative time
            start_ts = datetime.fromisoformat(current_power_data[0][0]).timestamp()
            timestamps = []
            power_values = []

            for t_str, p in current_power_data:
                ts = datetime.fromisoformat(t_str).timestamp()
                timestamps.append(ts - start_ts)
                power_values.append(p)

            ts_arr = np.array(timestamps)
            p_arr = np.array(power_values)

            # Resample to uniform grid ADJUSTED to sensor cadence
            # default min_dt=5.0, gap=300?
            # Use somewhat large gap to bridge occasional misses
            segments, used_dt = resample_adaptive(
                ts_arr, p_arr, min_dt=5.0, gap_s=300.0
            )

            if not segments:
                return (None, 0.0, 0.0, None)

            # Use the longest segment for matching
            current_seg = max(segments, key=lambda s: len(s.power))

            # If segment is too short, abort
            if len(current_seg.power) < 12:  # < 1 min
                return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.warning("Match preprocessing failed: %s", e)
            return MatchResult(None, 0.0, 0.0, None, [], False, 0.0)

        candidates = []

        _LOGGER.debug(
            "Profile matching checking %s profiles: %s",
            len(self._data.get("profiles", {})),
            list(self._data.get("profiles", {}).keys()),
        )

        for name, profile in self._data["profiles"].items():
            sample_id = profile.get("sample_cycle_id")
            sample_cycle = next(
                (c for c in self._data["past_cycles"] if c["id"] == sample_id), None
            )

            if not sample_cycle:
                _LOGGER.debug(
                    "Skip %s: sample cycle %s not found in storage", name, sample_id
                )
                continue

            # --- STAGE 1: Fast Reject ---

            # Duration Check
            profile_duration = profile.get(
                "avg_duration", sample_cycle.get("duration", 0)
            )
            if profile_duration > 0:
                duration_ratio = current_duration / profile_duration
                # Allow wide range for running cycles (e.g. 7% to 150%)
                if (
                    duration_ratio < self._min_duration_ratio
                    or duration_ratio > self._max_duration_ratio
                ):
                    _LOGGER.debug(
                        "Reject %s: duration ratio %.2f (need %.2f-%.2f)",
                        name,
                        duration_ratio,
                        self._min_duration_ratio,
                        self._max_duration_ratio,
                    )
                    continue

            # Load Sample Data (Lazy)
            sample_data = sample_cycle.get("power_data")
            if not sample_data:
                _LOGGER.debug(
                    "Skip %s: sample cycle %s has no power_data", name, sample_id
                )
                continue

            # Extract sample values (assuming sample is typically good quality)
            # Need to resample sample too!
            if len(sample_data) > 0 and isinstance(sample_data[0], (list, tuple)):
                # [offset, power] or (offset, power)
                s_ts = np.array([x[0] for x in sample_data])
                s_p = np.array([x[1] for x in sample_data])
            else:
                _LOGGER.debug(
                    "Skip %s: sample data format invalid, first element type: %s",
                    name,
                    type(sample_data[0]) if sample_data else "empty",
                )
                continue

            # Resample sample to matched grid (used_dt)
            # This ensures we compare same resolution
            s_segments = resample_uniform(s_ts, s_p, dt_s=used_dt, gap_s=300.0)
            if not s_segments:
                _LOGGER.debug("Skip %s: sample resample failed (no segments)", name)
                continue
            sample_seg = max(s_segments, key=lambda s: len(s.power))

            # --- STAGE 2: Core Similarity (Iterative Alignment) ---
            # Replaced single-pass robust calc with coarse-to-fine search
            score, metrics, best_offset = self._find_best_alignment(
                current_seg.power, sample_seg.power, used_dt
            )
            
            # Log significant shifts
            if abs(best_offset) > 1:
                _LOGGER.debug(
                    "Profile %s aligned with offset %d (score improved)", name, best_offset
                )

            _LOGGER.debug(
                "Profile %s: score=%.3f, mae=%.1f, corr=%.3f",
                name,
                score,
                metrics.get("mae", 0),
                metrics.get("corr", 0),
            )

            if score > 0.1:  # Relaxed to allow tracking of weak matches/unmatches
                candidates.append(
                    {
                        "name": name,
                        "score": score,
                        "metrics": metrics,
                        "profile_duration": profile_duration,
                        "current": current_seg.power,
                        "sample": sample_seg.power,
                    }
                )
            else:
                _LOGGER.debug("Reject %s: score %.3f < 0.1 threshold", name, score)

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        if not candidates:
            # All rejected by fast reject
            return MatchResult(
                None,
                0.0,
                0.0,
                None,
                [],
                False,
                0.0,
                [],
                {},
                is_confident_mismatch=True,
                mismatch_reason="all_rejected",
            )

        # Use configured threshold
        current_best = candidates[0]
        # Start assuming best match is valid, refine below
        best = current_best
        best_name = best["name"]
        best_score = best["score"]
        is_ambiguous = False
        margin = 1.0


        # --- STAGE 3: DTW Tie-Break (Ambiguity Check) ---
        # If top 2 are close (margin < 0.1), use DTW on them

        if len(candidates) > 1:
            second = candidates[1]
            margin = best_score - second["score"]

            if margin < 0.15 and best_score > 0.6:
                # Ambiguous! Run DTW-lite on top candidate(s)
                is_ambiguous = True
                _LOGGER.info(
                    "Ambiguity detected (%s=%.2f vs %s=%.2f). Running DTW...",
                    best_name,
                    best_score,
                    second["name"],
                    second["score"],
                )

                for cand in candidates[:2]:
                    # Normalize arrays first? Power is already W.
                    # DTW distance depends on magnitude.
                    dtw_dist = self._compute_dtw_lite(
                        cand["current"], cand["sample"], band_width_ratio=0.1
                    )
                    # Normalize distance by length
                    norm_dist = dtw_dist / len(cand["current"])
                    # Map distance to score penalty
                    dtw_score = 1.0 / (1.0 + norm_dist / 50.0)

                    # Update score: blend Core and DTW
                    cand["original_score"] = cand["score"]
                    cand["score"] = 0.5 * cand["score"] + 0.5 * dtw_score
                    cand["dtw_dist"] = norm_dist

                # Re-sort
                candidates.sort(key=lambda x: x["score"], reverse=True)
                best = candidates[0]
                best_name = best["name"]
                best_score = best["score"]
                _LOGGER.info(
                    "Post-DTW: %s (%.2f), %s (%.2f)",
                    best_name,
                    best_score,
                    candidates[1]["name"],
                    candidates[1]["score"],
                )

                # Re-calculate margin after DTW
                if len(candidates) > 1:
                    margin = best_score - candidates[1]["score"]
                    # If margin still small, flag remains ambiguous?
                    # Plan says: "Ambiguity explicit"
                    if margin < 0.05:
                        is_ambiguous = True
                    else:
                        is_ambiguous = False

        # Compile debug details for top candidate
        debug_details = {}
        if best:
            debug_details = {
                "final_score": best["score"],
                "metrics": best.get("metrics", {}),
                "dtw_dist": best.get("dtw_dist"),
                "original_score": best.get("original_score"),
                "duration_ratio": (
                    best.get("profile_duration", 0) / current_duration
                    if current_duration > 0
                    else 0
                ),
            }

        # Build clean ranking list for storage/display
        ranking = []
        for c in candidates:  # Candidates are already sorted
            entry = {
                "name": c["name"],
                "score": round(c["score"], 3),
            }
            if "dtw_dist" in c:
                entry["dtw"] = round(c["dtw_dist"], 3)
            ranking.append(entry)

        # Check for confident mismatch at the end (after DTW potential updates)
        is_confident_mismatch = False
        mismatch_reason = None
        if best_score < self._unmatch_threshold:
            is_confident_mismatch = True
            mismatch_reason = f"low_confidence_{best_score:.2f}"

        return MatchResult(
            best_profile=best_name,
            confidence=best_score,
            expected_duration=best.get("profile_duration", 0.0),
            matched_phase=None,
            candidates=candidates,
            is_ambiguous=is_ambiguous,
            ambiguity_margin=margin,
            ranking=ranking,
            debug_details=debug_details,
            is_confident_mismatch=is_confident_mismatch,
            mismatch_reason=mismatch_reason,
        )

    def _find_best_alignment(
        self, current: np.ndarray, sample: np.ndarray, used_dt: float
    ) -> tuple[float, dict, int]:
        """
        Perform iterative coarse-to-fine alignment search.
        Returns: (best_score, best_metrics, best_offset_index)
        """
        # Coarse Scan: [-60s, -30s, 0s, +30s, +60s]
        # Assuming index map is 1-to-1 with 'used_dt' (e.g. 5s or 10s steps?).
        # ProfileStore uses `used_dt` (typically 5s or dynamic).
        # We perform shifting by array slicing.

        len_cur = len(current)
        len_sam = len(sample)
        
        # We need a shared length for comparison.
        # Logic: We slide 'sample' over 'current' (or vice versa).
        # Simpler: We crop to the OVERLAP.
        
        def get_score_at_offset(offset: int) -> tuple[float, dict]:
            """
            Offset > 0: Shift Sample RIGHT (Sample starts later)
                        Compare current[offset:] with sample[:-offset]
            Offset < 0: Shift Sample LEFT (Sample starts earlier)
                        Compare current[:offset] with sample[-offset:]
            """
            if offset == 0:
                c = current
                s = sample
            elif offset > 0:
                # Sample shifted right relative to current
                # current: |...[........]
                # sample:      [........]...|
                # Compare overlap
                if offset >= len_cur:
                    return 0.0, {}
                
                # We align start of sample with index 'offset' of current
                # intersection length = min(len_cur - offset, len_sam)
                length = min(len_cur - offset, len_sam)
                
                # Minimum Overlap Constraint
                if length < len_sam * self._min_duration_ratio:
                    return 0.0, {}
                     
                c = current[offset : offset + length]
                s = sample[:length]
            else:
                # offset < 0
                # Sample shifted left
                # current:      [........]...|
                # sample:  |...[........]
                abs_off = abs(offset)
                if abs_off >= len_cur or abs_off >= len_sam:
                    return 0.0, {}
                
                # Align start of current with index 'abs_off' of sample
                length = min(len_cur, len_sam - abs_off)
                
                # Minimum Overlap Constraint
                if length < len_sam * self._min_duration_ratio:
                    return 0.0, {}

                c = current[:length]
                s = sample[abs_off : abs_off + length]

            return self._calculate_similarity_robust(c, s)

        if used_dt <= 0:
            used_dt = 5.0
            
        # Hierarchical Search (Coarse -> Fine)
        # Goal: Find alignment within ±30 minutes (1800s)
        
        # Pass 1: Global Coarse Scan
        # Window: ±30 minutes
        # Step: 60 seconds
        coarse_window_s = 1800.0
        coarse_step_s = 60.0
        
        coarse_radius_idx = int(coarse_window_s / used_dt)
        step_idx = max(1, int(coarse_step_s / used_dt))
        
        # Scan points: 0, ±step, ±2*step ... until radius
        # We use a set to track visited offsets to avoid re-calculating in Fine pass
        visited_offsets = set()
        
        best_score = -1.0
        best_metrics = {}
        best_offset = 0
        
        # Generate coarse offsets
        coarse_offsets = [0]
        curr = step_idx
        while curr <= coarse_radius_idx:
            coarse_offsets.append(curr)
            coarse_offsets.append(-curr)
            curr += step_idx
            
        for off in coarse_offsets:
            visited_offsets.add(off)
            s, m = get_score_at_offset(off)
            if s > best_score:
                best_score = s
                best_metrics = m
                best_offset = off
                
        # Pass 2: Local Fine Scan
        # Window: ±2 minutes (120s) around best_offset
        # Step: 1 index (used_dt)
        fine_window_s = 120.0
        fine_radius_idx = int(fine_window_s / used_dt)
        
        start_fine = best_offset - fine_radius_idx
        end_fine = best_offset + fine_radius_idx
        
        # Clamp to meaningful bounds if desired, but array slicing handles out-of-bounds gracefully (returns empty)
        # We just need to ensure loop is correct
        
        for off in range(start_fine, end_fine + 1):
            if off in visited_offsets:
                continue
            visited_offsets.add(off)
            
            s, m = get_score_at_offset(off)
            if s > best_score:
                best_score = s
                best_metrics = m
                best_offset = off
                
        return best_score, best_metrics, best_offset

    def _calculate_similarity_robust(
        self, current: np.ndarray, sample: np.ndarray
    ) -> tuple[float, dict]:
        """Core similarity with robust scaling."""
        len_cur = len(current)
        len_sam = len(sample)
        
        # Ensure minimal overlap
        if len_cur < 5 or len_sam < 5:
            return 0.0, {}
            
        # Truncate to min length
        n = min(len_cur, len_sam)
        c = current[:n]
        s = sample[:n]

        # 1. MAE (Mean Absolute Error)
        diff = np.abs(c - s)
        mae = np.mean(diff)

        # 2. Correlation
        if np.std(c) > 1e-3 and np.std(s) > 1e-3:
            corr = np.corrcoef(c, s)[0, 1]
        else:
            corr = 0.0

        # 3. Peak Match
        peak_cur = np.max(c)
        peak_sam = np.max(s)
        peak_diff = abs(peak_cur - peak_sam)

        # Scoring
        mae_score = 1.0 / (1.0 + mae / 50.0)  # 50W characteristic scale
        corr_score = max(0.0, corr)
        peak_score = 1.0 / (1.0 + peak_diff / 100.0)

        # Weighted Sum
        final = 0.4 * mae_score + 0.4 * corr_score + 0.2 * peak_score

        # Boost for strong correlation
        if corr > 0.9:
            final = min(1.0, final * 1.1)

        return float(final), {"mae": mae, "corr": corr, "peak_diff": peak_diff}

    def _compute_dtw_lite(
        self, x: np.ndarray, y: np.ndarray, band_width_ratio: float = 0.1
    ) -> float:
        """
        Compute DTW distance with Sakoe-Chiba band constraint.
        Numpy implementation. O(N*W).

        Args:
            x, y: Input arrays (1D power values)
            band_width_ratio: constraint window as fraction of length (e.g. 0.1 = 10%)
        """
        n, m = len(x), len(y)
        if n == 0 or m == 0:
            return float("inf")

        # Band width
        w = max(1, int(min(n, m) * band_width_ratio))

        # Cost matrix: use 2 rows to save memory?
        # Standard DP: D[i, j] = dist(i, j) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        # We can implement full matrix since N is small (5s grid -> 1h = 720 points).
        # 720x720 = 500k floats = 4MB. Fine.

        dtw = np.full((n + 1, m + 1), float("inf"))
        dtw[0, 0] = 0

        # Vectorized implementation of band constraint is hard in pure numpy without loop.
        # But we can limit the loop range.

        # Euclidean distance sq or abs? ABS is more robust for power.

        for i in range(1, n + 1):
            center = i * (m / n)
            start_j = max(1, int(center - w))
            end_j = min(m, int(center + w) + 1)

            for j in range(start_j, end_j + 1):
                cost = abs(x[i - 1] - y[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        # Normalization: Return average distance per step
        # Path length is roughly max(n, m) to n+m.
        # Standard: divide by (n + m).
        return dtw[n, m] / (n + m)

    def _calculate_similarity(self, current: np.ndarray, sample: np.ndarray) -> float:
        score, _ = self._calculate_similarity_robust(current, sample)
        return score

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
            # Count cycles using this profile
            cycle_count = sum(
                1
                for c in self._data.get("past_cycles", [])
                if c.get("profile_name") == name
            )
            profiles.append(
                {
                    "name": name,
                    "avg_duration": profile_meta.get("avg_duration", 0),
                    "min_duration": profile_meta.get("min_duration", 0),
                    "max_duration": profile_meta.get("max_duration", 0),
                    "sample_cycle_id": profile_meta.get("sample_cycle_id"),
                    "cycle_count": cycle_count,
                }
            )
        return sorted(profiles, key=lambda p: str(p.get("name", "")))

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
            self.rebuild_envelope(old_profile)  # Old profile lost a cycle
        if profile_name:
            self.rebuild_envelope(profile_name)  # New profile gained a cycle
            # Apply retention after labeling, in case profile now exceeds cap
            self._enforce_retention()

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
            result = self.match_profile(power_data, cycle["duration"])

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
            self.rebuild_envelope(profile_name)

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
                start_ts = datetime.fromisoformat(cycle["start_time"]).timestamp()
                compressed: list[list[float]] = []

                last_saved_p = -999.0
                last_saved_t = -999.0

                for i, point in enumerate(raw_data):
                    # Parse timestamp
                    if isinstance(point[0], str):
                        t_val = datetime.fromisoformat(point[0]).timestamp()
                    else:
                        t_val = float(point[0])

                    p_val = float(point[1])
                    offset = round(t_val - start_ts, 1)

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

                cycle["power_data"] = compressed
                migrated += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                _LOGGER.warning("Failed to migrate cycle %s: %s", cycle.get("id"), e)
                continue

        if migrated > 0:
            _LOGGER.info("Migrated %s cycles to compressed format", migrated)
            await self.async_save()

        return migrated

    async def async_merge_cycles_smart(self, hours: int = 24, gap_threshold: int = 3600) -> int:
        """
        Smartly merge fragmented cycles using profile matching validation.
        gap_threshold: max seconds between cycles to consider merging.
        Returns number of merges performed.
        """
        limit = dt_util.now().timestamp() - (hours * 3600)
        cycles = cast(list[CycleDict], self._data.get("past_cycles", []))
        if not cycles:
            return 0

        # Sort by start time
        cycles.sort(key=lambda c: str(c.get("start_time", "")))

        merged_count = 0
        i = 0
        while i < len(cycles) - 1:
            c1 = cycles[i]
            c2 = cycles[i + 1]

            # Parse times
            # Parse times
            try:
                t1_end_dt = dt_util.parse_datetime(c1["end_time"])
                t2_start_dt = dt_util.parse_datetime(c2["start_time"])
                if not t1_end_dt or not t2_start_dt:
                    i += 1
                    continue
                t1_end = t1_end_dt.timestamp()
                t2_start = t2_start_dt.timestamp()
            except (ValueError, TypeError):
                i += 1
                continue

            # Skip if both are too old
            if (
                t1_end < limit
                and datetime.fromisoformat(c2["end_time"]).timestamp() < limit
            ):
                i += 1
                continue

            gap = t2_start - t1_end
            if gap < 0 or gap > gap_threshold:
                i += 1
                continue

            # --- SMART MERGE VALIDATION ---
            # 1. Evaluate individual cycles
            # 2. Evaluate merged candidate
            # 3. Compare scores

            # Helper to get score
            def get_score(cycle_data: CycleDict) -> float:
                p_data = self._decompress_power_data(cycle_data)
                if not p_data:
                    return 0.0
                res = self.match_profile(p_data, cycle_data["duration"])
                return res.confidence

            score1 = get_score(c1)
            score2 = get_score(c2)

            # Construct Candidate (Simulation)
            # We strictly emulate the data merge without mutating c1 yet
            c1_start_dt = dt_util.parse_datetime(c1["start_time"])
            if not c1_start_dt:
                i += 1
                continue

            c1_power = decompress_power_data(c1)  # [(iso, p), ...]
            c2_power = decompress_power_data(c2)

            # Merge power data (list of tuples)
            merged_power = list(c1_power)
            # c2 explicitly shifted?
            # decompress_power_data returns absolute ISO timestamps.
            # So just appending is fine if they are sorted?
            # Yes, match_profile converts them to relative.
            merged_power.extend(c2_power)

            # Recalculate duration
            c2_end_dt = dt_util.parse_datetime(c2["end_time"])
            if not c2_end_dt:
                i += 1
                continue
            new_dur = (c2_end_dt - c1_start_dt).total_seconds()

            # Score candidate
            res_merged = self.match_profile(merged_power, new_dur)
            score_merged = res_merged.confidence
            best_candidate_profile = res_merged.best_profile

            # --- DECISION LOGIC ---
            should_merge = False

            # Rule 1: Merging creates a significantly better match than the parts
            # e.g. Part A (0.3), Part B (0.3) -> Merged (0.8)
            current_max = max(score1, score2)
            if score_merged >= self._match_threshold and score_merged > (
                current_max + 0.1
            ):
                should_merge = True
                _LOGGER.info(
                    "Smart Merge: %s & %s -> Better Match (%.2f vs %.2f/%.2f)",
                    c1["id"],
                    c2["id"],
                    score_merged,
                    score1,
                    score2,
                )

            # Rule 2: Rescue "Noise" / Debounce
            # If C1 is tiny/noise and C2 is good (or vice versa), and merging kept the score high.
            # "Noise" definition: < 2 mins OR score very low
            is_c1_noise = c1["duration"] < 120 or score1 < self._unmatch_threshold
            if is_c1_noise and score_merged >= self._match_threshold:
                # Ensure we didn't degrade the good cycle significantly
                if score_merged >= (score2 - 0.05):
                    should_merge = True
                    _LOGGER.info(
                        "Smart Merge: Rescuing fragment %s into %s (Score %.2f)",
                        c1["id"],
                        c2["id"],
                        score_merged,
                    )

            # Anti-Rule: Distinct Profiles
            # If both match DIFFERENT profiles with high confidence, DO NOT MERGE unless gap is tiny (< 60s)?
            # Actually, score comparison handles this. If both match diff profiles, merged score will likely be low
            # (unless it matches a third "super profile").

            if should_merge:
                # EXECUTE MERGE (Ported form old logic)
                c1["end_time"] = c2["end_time"]
                c1["duration"] = new_dur

                # Merge compressed data
                # We need to do this carefully on local compressed data
                # We already calculated shift relative to c1 start above:
                # Let's use the explicit logic from old method but simpler

                c2_raw = c2.get("power_data", [])
                shift_seconds = (
                    t2_start - dt_util.parse_datetime(c1["start_time"]).timestamp()
                )

                shifted_c2 = []
                for item in c2_raw:
                    # item is [offset, power]
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        shifted_c2.append(
                            [
                                round(float(item[0]) + shift_seconds, 1),
                                float(item[1]),
                            ]
                        )

                c1.setdefault("power_data", []).extend(shifted_c2)

                c1["max_power"] = max(c1.get("max_power", 0), c2.get("max_power", 0))

                # Inherit profile if C1 didn't have one but we found a better one
                if best_candidate_profile:
                    c1["profile_name"] = best_candidate_profile
                elif c2.get("profile_name") and not c1.get("profile_name"):
                    c1["profile_name"] = c2.get("profile_name")

                # Update References
                new_id = hashlib.sha256(
                    f"{c1['start_time']}_{c1['duration']}".encode()
                ).hexdigest()[:12]
                old_c1_id = c1["id"]
                old_c2_id = c2["id"]
                c1["id"] = new_id

                # Update profile samples
                for _, p_data in self._data["profiles"].items():
                    if p_data.get("sample_cycle_id") in (old_c1_id, old_c2_id):
                        p_data["sample_cycle_id"] = new_id

                # Remove c2
                cycles.pop(i + 1)
                merged_count += 1
                # Don't increment i
            else:
                i += 1

        return merged_count

    async def async_split_cycles_smart(
        self, cycle_id: str, min_gap_s: int = 900, idle_power: float = 2.0
    ) -> list[str]:
        """
        Scan a cycle for significant idle gaps and split if parts match better than whole.
        Returns list of resulting cycle IDs.
        """
        cycles = cast(list[CycleDict], self._data.get("past_cycles", []))
        idx = next((i for i, c in enumerate(cycles) if c.get("id") == cycle_id), -1)

        if idx == -1:
            return []

        cycle = cycles[idx]
        power_data = cycle.get("power_data", [])

        if not power_data or len(power_data) < 2:
            return [cycle_id]
        
        # 1. Evaluate Original
        p_data_tuples = self._decompress_power_data(cycle)
        if not p_data_tuples:
            return [cycle_id]

        res_orig = self.match_profile(p_data_tuples, cycle["duration"])
        score_orig = res_orig.confidence
        
        # If original is a strong match, we trust it (Dishwasher drying pause etc)
        # UNLESS the gap is massive (e.g. > 2 hours), but let's trust profile logic mostly.
        if score_orig >= 0.8: # Strong match
            _LOGGER.debug("Skipping split for %s: Strong match %.2f to %s", cycle_id, score_orig, res_orig.best_profile)
            return [cycle_id]

        # 2. Identify Potential Split Points (Gaps)
        points = []
        for p in power_data:
            points.append((float(p[0]), float(p[1])))

        splits: list[tuple[float, float]] = []
        last_t = points[0][0]
        last_p = points[0][1]
        current_idle_start: float | None = None

        if last_p <= idle_power:
            current_idle_start = last_t

        for i in range(1, len(points)):
            t, p = points[i]
            if last_p <= idle_power:
                if current_idle_start is None:
                    current_idle_start = last_t
                if p > idle_power:
                    duration = t - current_idle_start
                    if duration >= min_gap_s:
                        splits.append((current_idle_start, t))
                    current_idle_start = None
            else:
                if p <= idle_power:
                    current_idle_start = t
            last_t = t
            last_p = p

        if not splits:
            return [cycle_id]
            
        # 3. Simulate Splits and Score Parts
        # We need to test if splitting improves the situation.
        # Simple heuristic: Split at the LARGEST gap first? Or all gaps?
        # Let's assume splitting all gaps for now, effectively reducing to "Island detection"
        # Validate if ANY of the resulting islands is a valid profile match.
        
        cycle_start_iso = cycle["start_time"]
        start_dt_base = dt_util.parse_datetime(cycle_start_iso)
        if not start_dt_base:
            return [cycle_id]
        seg_ranges: list[tuple[float, float]] = []
        prev_end = 0.0
        for gap_start, gap_end in splits:
            if gap_start > prev_end:
                # Check if segment is long enough to bother (e.g. > 2 mins)
                if (gap_start - prev_end) > 120:
                    seg_ranges.append((prev_end, gap_start))
            prev_end = gap_end
        
        total_dur = cycle.get("duration", points[-1][0])
        if (total_dur - prev_end) > 120:
            seg_ranges.append((prev_end, total_dur))
            
        if len(seg_ranges) < 2:
            # Only 1 substantial segment found? Maybe tail was trimmed.
            # If effectively only 1 part, treat as "trimming" the cycle.
            # Trimming logic: if score improves?
            # For now, let's focus on Splitting (1 -> 2+).
            return [cycle_id]

        # Check scores of proposed parts
        valid_part_found = False
        parts_data = [] # List of (segment_points_tuples, duration)
        
        start_ts = start_dt_base.timestamp()
        
        for seg_start, seg_end in seg_ranges:
            # Extract points and shift to relative
            seg_points = []
            # Find state val
            state_val = 0.0
            for t, p in points:
                if t <= seg_start:
                    state_val = p
                else:
                    break
            seg_points.append((datetime.fromtimestamp(start_ts + seg_start).isoformat(), state_val)) # Point 0

            for t, p in points:
                if t > seg_start and t <= seg_end:
                    seg_points.append((datetime.fromtimestamp(start_ts + t).isoformat(), p))

            seg_dur = seg_end - seg_start
            if len(seg_points) < 5: continue

            # Score it
            res_part = self.match_profile(seg_points, seg_dur)
            if res_part.confidence >= self._match_threshold:
                valid_part_found = True

            parts_data.append(res_part.confidence)
        
        # DECISION:
        # If the original score was low (< threshold) AND at least one part matches well
        # OR original was "okay" (0.5) but distinct parts match "excellent" (0.9)
        # We split.

        should_split = False
        if score_orig < self._match_threshold and valid_part_found:
            should_split = True
            _LOGGER.info(
                "Smart Split: Splitting %s (Score %.2f) -> Found valid part(s) %s",
                cycle_id,
                score_orig,
                parts_data,
            )
        elif score_orig < 0.7:
            # Even if score was decent, if parts are remarkably better...
            if any(s > (score_orig + 0.2) for s in parts_data):
                should_split = True
                _LOGGER.info(
                    "Smart Split: Splitting %s (Score %.2f) -> Parts are much better %s",
                    cycle_id,
                    score_orig,
                    parts_data,
                )

        if not should_split:
            return [cycle_id]

        # EXECUTE SPLIT (Ported from old logic)
        cycles.pop(idx)
        new_ids = []
        original_profile = cycle.get("profile_name")

        for i, (seg_start, seg_end) in enumerate(seg_ranges):
            # Re-extract and format for adding
            # ... (Same extract logic as before but simpler construction)
            # Need normalized compressed list [offset, power] for add_cycle

            seg_compressed = []
            # determine start state
            state_val = 0.0
            for t, p in points:
                if t <= seg_start:
                    state_val = p
                else:
                    break
            seg_compressed.append([0.0, state_val])

            for t, p in points:
                if t > seg_start and t <= seg_end:
                    seg_compressed.append([round(t - seg_start, 1), p])

            seg_dur = seg_end - seg_start

            new_cycle_start = start_dt_base + timedelta(seconds=seg_start)
            new_cycle_end = new_cycle_start + timedelta(seconds=seg_dur)
            new_cycle_start_ts = new_cycle_start.timestamp()

            # Convert to absolute for add_cycle
            p_data_abs = []
            for off, val in seg_compressed:
                p_data_abs.append([round(new_cycle_start_ts + off, 1), val])

            new_cycle = {
                "start_time": new_cycle_start.isoformat(),
                "end_time": new_cycle_end.isoformat(),
                "duration": round(seg_dur, 1),
                "status": "completed",
                "power_data": p_data_abs,
                "profile_name": None # Reset profile, let auto-labeler fix it
            }

            self.add_cycle(new_cycle)
            new_ids.append(new_cycle["id"])
             
        # Cleanup orphaned profile refs (Original logic)
        original_sample_id = cycle.get("id")
        best_replacement_id = None
        longest_dur = 0
        new_cycles_objs = [c for c in cycles if c["id"] in new_ids] # Need to fetch fresh
        
        for c in new_cycles_objs:
            d = c.get("duration", 0)
            if d > longest_dur:
                longest_dur = d
                best_replacement_id = c["id"]

        if best_replacement_id and original_profile:
            # Only update the profile that relied on this cycle
            p_data = self._data["profiles"].get(original_profile)
            if p_data and p_data.get("sample_cycle_id") == original_sample_id:
                p_data["sample_cycle_id"] = best_replacement_id
                self.rebuild_envelope(original_profile)

        await self.async_save()
        return new_ids

    async def async_smart_process_history(
        self, hours: int = 24, gap_seconds: int = 3600
    ) -> dict[str, int]:
        """
        Orchestrate smart history processing: Merge fragments, Split joins, Rebuild envelopes.
        Should be called after major history changes (cycle end, delete, label).
        """
        stats = {"merged": 0, "split": 0}

        # 1. Smart Merge (Combine fragments)
        stats["merged"] = await self.async_merge_cycles_smart(
            hours=hours, gap_threshold=gap_seconds
        )

        # 2. Smart Split (Separate joined cycles)
        # Scan recent cycles
        limit = dt_util.now().timestamp() - (hours * 3600)
        cycles = cast(list[CycleDict], self._data.get("past_cycles", []))

        # Snapshot IDs to avoid modification issues
        candidates = []
        for c in cycles:
            try:
                end_dt = dt_util.parse_datetime(str(c.get("end_time")))
                if end_dt and end_dt.timestamp() > limit:
                    candidates.append(c["id"])
            except (ValueError, TypeError):
                continue

        for cid in candidates:
            # Check if cycle still exists
            if not any(c["id"] == cid for c in cycles):
                continue

            new_ids = await self.async_split_cycles_smart(cid, min_gap_s=900)
            if len(new_ids) > 1:
                stats["split"] += 1

        # 3. Use the cleaner data to rebuild envelopes
        self.rebuild_all_envelopes()

        if stats["merged"] > 0 or stats["split"] > 0:
            await self.async_save()
            _LOGGER.info("Smart Process History: Merged %s, Split %s", stats["merged"], stats["split"])

        return stats
    def log_adjustment(
        self, setting_name: str, old_value: Any, new_value: Any, reason: str
    ) -> None:
        """Log an automatic setting adjustment (auto-tune, auto-label changes)."""
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
        """Return a serializable snapshot of the store for backup/export.
        Includes config entry data/options so users can transfer fine-tuned settings."""
        return {
            "version": STORAGE_VERSION,
            "entry_id": self.entry_id,
            "exported_at": dt_util.now().isoformat(),
            "data": self._data,
            "entry_data": entry_data or {},
            "entry_options": entry_options or {},
        }

    async def async_import_data(self, payload: JSONDict) -> dict[str, JSONDict]:
        """Load store data from an export payload and persist it.

        Supports both v1 (flat) and v2 (nested data) export formats.
        Returns dict with 'entry_data' and 'entry_options' keys for updating the config entry.
        """
        version = payload.get("version", 1)

        # Handle v1 format (flat structure) - convert to v2
        if version == 1 or "data" not in payload:
            # V1 format had profiles/past_cycles at top level
            data_dict = {
                "profiles": payload.get("profiles", {}),
                "past_cycles": payload.get("past_cycles", []),
                "envelopes": payload.get("envelopes", {}),
            }
            entry_data = payload.get("entry_data", {})
            entry_options = payload.get("entry_options", {})
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
            entry_data = payload.get("entry_data", {})
            entry_options = payload.get("entry_options", {})
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

        # Rebuild all envelopes after import to ensure consistency
        self.rebuild_all_envelopes()

        _LOGGER.info(
            "Import complete: %s profiles, %s cycles",
            len(data_dict.get("profiles", {})),
            len(data_dict.get("past_cycles", [])),
        )

        # Return config data/options for caller to apply
        return {
            "entry_data": (
                cast(JSONDict, entry_data) if isinstance(entry_data, dict) else {}
            ),
            "entry_options": (
                cast(JSONDict, entry_options) if isinstance(entry_options, dict) else {}
            ),
        }
