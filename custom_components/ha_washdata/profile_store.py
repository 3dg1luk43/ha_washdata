"""Profile storage and matching logic for HA WashData."""
from __future__ import annotations

import json
import logging
import os
import hashlib
from datetime import datetime
from typing import Any
import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    DOMAIN,
    STORAGE_KEY,
    STORAGE_VERSION,
    DEFAULT_MAX_PAST_CYCLES,
    DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
    DEFAULT_MAX_FULL_TRACES_UNLABELED,
)

_LOGGER = logging.getLogger(__name__)

class ProfileStore:
    """Manages storage of washer profiles and past cycles."""

    def __init__(
        self, 
        hass: HomeAssistant, 
        entry_id: str,
        min_duration_ratio: float = 0.50,
        max_duration_ratio: float = 1.50,
    ) -> None:
        """Initialize the profile store."""
        self.hass = hass
        self.entry_id = entry_id
        self._min_duration_ratio = min_duration_ratio
        self._max_duration_ratio = max_duration_ratio
        # Retention policy: cap total cycles and number of full-resolution traces per profile
        self._max_past_cycles = DEFAULT_MAX_PAST_CYCLES
        self._max_full_traces_per_profile = DEFAULT_MAX_FULL_TRACES_PER_PROFILE
        self._max_full_traces_unlabeled = DEFAULT_MAX_FULL_TRACES_UNLABELED
        # Separate store for each entry to avoid giant files
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}.{entry_id}")
        self._data: dict[str, Any] = {
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
        suggestions: dict[str, Any] = self._data.setdefault("suggestions", {})
        suggestions[key] = {
            "value": value,
            "reason": reason,
            "updated": datetime.now().isoformat(),
        }

    def get_suggestions(self) -> dict[str, Any]:
        """Return current suggestion map."""
        raw = self._data.get("suggestions") or {}
        if isinstance(raw, dict):
            return dict(raw)
        return {}

    async def async_load(self) -> None:
        """Load data from storage."""
        data = await self._store.async_load()
        if data:
            self._data = data

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
            except Exception:
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
                c for c in cycles
                if c.get("profile_name") == profile_name and c.get("power_data")
            ]
            if labeled_candidates:
                try:
                    chosen = max(labeled_candidates, key=lambda c: c.get("start_time", ""))
                except Exception:
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
            except Exception:
                pass

        return stats

    async def async_save(self) -> None:
        """Save data to storage."""
        await self._store.async_save(self._data)

    async def async_save_active_cycle(self, detector_snapshot: dict) -> None:
        """Save the active cycle state separately (or in main data)."""
        # We can store it in the main store, but we need to ensure we don't wear out flash
        # if this is called often.
        # Home Assistant's Store helper writes atomically.
        # Let's put it in _data but only save if significant change? 
        # Actually Manager throttles this call.
        self._data["active_cycle"] = detector_snapshot
        self._data["last_active_save"] = datetime.now().isoformat()
        await self._store.async_save(self._data)
        
    def get_active_cycle(self) -> dict | None:
        """Get the saved active cycle."""
        return self._data.get("active_cycle")
    
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
        if "active_cycle" in self._data:
            del self._data["active_cycle"]
            await self._store.async_save(self._data)

    def add_cycle(self, cycle_data: dict[str, Any]) -> None:
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
        raw_data = cycle_data.get("power_data", [])
        _LOGGER.debug(f"add_cycle: raw_data has {len(raw_data)} points")
        
        if raw_data:
            start_ts = datetime.fromisoformat(cycle_data["start_time"]).timestamp()
            stored = []
            offsets = []
            
            # Helper to parse time
            def get_ts(item):
                if isinstance(item[0], str):
                    return datetime.fromisoformat(item[0]).timestamp()
                return float(item[0])

            for point in raw_data:
                t_val = get_ts(point)
                p_val = point[1]
                
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
            
            _LOGGER.debug(
                f"add_cycle: stored {len(stored)} samples at {sampling_interval:.1f}s intervals"
            )

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
        cycles = self._data.get("past_cycles", [])
        if not cycles:
            return

        # 1) Cap total cycles
        if len(cycles) > self._max_past_cycles:
            # Sort by start_time and drop oldest beyond cap
            try:
                cycles.sort(key=lambda c: c.get("start_time", ""))
            except Exception:
                pass
            drop_count = len(cycles) - self._max_past_cycles
            to_drop = cycles[:drop_count]
            # Maintain profile sample references when dropping
            sample_refs = {name: p.get("sample_cycle_id") for name, p in self._data.get("profiles", {}).items()}
            for cy in to_drop:
                cy_id = cy.get("id")
                # If a profile sample points here, try to move to most recent cycle of that profile
                for name, ref_id in list(sample_refs.items()):
                    if ref_id == cy_id:
                        # find newest cycle for that profile
                        newest = next((c for c in reversed(cycles) if c.get("profile_name") == name), None)
                        if newest:
                            self._data["profiles"][name]["sample_cycle_id"] = newest.get("id")
                        else:
                            # No replacement available
                            self._data["profiles"][name].pop("sample_cycle_id", None)
            # Actually drop
            del cycles[:drop_count]

        # 2) Strip older full traces per profile
        by_profile: dict[Any, list[dict]] = {}
        for cy in cycles:
            key = cy.get("profile_name")  # None for unlabeled
            by_profile.setdefault(key, []).append(cy)

        affected_profiles: set[str] = set()
        for key, group in by_profile.items():
            # newest first based on start_time
            try:
                group.sort(key=lambda c: c.get("start_time", ""))
            except Exception:
                pass
            # determine cap
            cap = self._max_full_traces_unlabeled if key in (None, "",) else self._max_full_traces_per_profile
            # count existing full traces
            full_indices = [i for i, c in enumerate(group) if c.get("power_data")]
            if len(full_indices) > cap:
                # preserve last 'cap' full traces (newest at end after sort), strip older ones
                keep_set = set(full_indices[-cap:])
                
                # Get sample cycle ID for this profile
                sample_id = None
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
            except Exception as e:
                _LOGGER.debug(f"Envelope rebuild skipped for '{p}' during retention: {e}")

    def delete_cycle(self, cycle_id: str) -> bool:
        """Delete a cycle by ID. Returns True if deleted, False if not found.
        Also removes any profiles that reference this cycle."""
        cycles = self._data["past_cycles"]
        for i, cycle in enumerate(cycles):
            if cycle.get("id") == cycle_id:
                cycles.pop(i)
                # Clean up any profiles referencing this cycle
                orphaned_profiles = [
                    name for name, profile in self._data["profiles"].items()
                    if profile.get("sample_cycle_id") == cycle_id
                ]
                for name in orphaned_profiles:
                    del self._data["profiles"][name]
                    _LOGGER.info(f"Removed orphaned profile '{name}' (referenced deleted cycle {cycle_id})")
                _LOGGER.info(f"Deleted cycle {cycle_id}")
                return True
        _LOGGER.warning(f"Cycle {cycle_id} not found for deletion")
        return False

    def cleanup_orphaned_profiles(self) -> int:
        """Remove profiles that reference non-existent cycles.
        Returns number of profiles removed."""
        cycle_ids = {c["id"] for c in self._data.get("past_cycles", [])}
        orphaned = [
            name for name, profile in self._data["profiles"].items()
            if profile.get("sample_cycle_id") not in cycle_ids
        ]
        
        for name in orphaned:
            del self._data["profiles"][name]
            _LOGGER.info(f"Cleaned up orphaned profile '{name}' (cycle no longer exists)")
        
        return len(orphaned)

    async def async_run_maintenance(self, lookback_hours: int = 24, gap_seconds: int = 1800) -> dict[str, int]:
        """Run full maintenance: cleanup orphans, merge fragments, trim old cycles, rebuild envelopes.
        Returns stats dict with counts of actions taken."""
        stats = {
            "orphaned_profiles": 0,
            "merged_cycles": 0,
            "rebuilt_envelopes": 0,
        }
        
        # 1. Clean up orphaned profiles
        stats["orphaned_profiles"] = self.cleanup_orphaned_profiles()
        
        # 2. Merge fragmented cycles (recent history)
        stats["merged_cycles"] = self.merge_cycles(hours=lookback_hours, gap_threshold=gap_seconds)
        
        # 3. Rebuild all envelopes (they may be stale after merges/deletions)
        stats["rebuilt_envelopes"] = self.rebuild_all_envelopes()
        
        # 4. Save if any changes made
        if any(stats.values()):
            await self.async_save()
            _LOGGER.info(f"Maintenance completed: {stats}")
        
        return stats

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
            c for c in self._data["past_cycles"]
            if c.get("profile_name") == profile_name and c.get("status") in ("completed", "force_stopped")
        ]
        
        if not labeled_cycles or len(labeled_cycles) < 1:
            # Clear envelope if it exists
            if profile_name in self._data.get("envelopes", {}):
                del self._data["envelopes"][profile_name]
            return False
        
        # Extract and normalize all power curves
        normalized_curves = []
        sampling_rates = []
        
        for cycle in labeled_cycles:
            power_data = cycle.get("power_data", [])
            if not power_data or len(power_data) < 3:
                continue
            
            # Extract power values from [offset, power] pairs
            offsets = np.array([o for o, _ in power_data])
            values = np.array([p for _, p in power_data])
            
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
        max_times = [offsets[-1] for offsets, _ in normalized_curves]
        target_duration = np.median(max_times)  # Use median duration
        
        # Resample all curves to same TIME axis
        # Create uniform time grid from 0 to target_duration
        num_points = max(50, int(target_duration / np.median(sampling_rates)))  # ~50-300 points
        time_grid = np.linspace(0, target_duration, num_points)
        
        # Calculate duration stats (min/max/avg)
        durations = [len(offsets) * (sampling_rates[i] if i < len(sampling_rates) else avg_sample_rate) 
                     for i, (offsets, _) in enumerate(normalized_curves)]
        # Better: use the actual max offset from each curve as duration
        durations = [offsets[-1] for offsets, _ in normalized_curves]
        
        min_duration = float(np.min(durations))
        max_duration = float(np.max(durations))
        
        # Update profile stats in storage
        if profile_name in self._data.get("profiles", {}):
            self._data["profiles"][profile_name]["min_duration"] = min_duration
            self._data["profiles"][profile_name]["max_duration"] = max_duration
            # avg_duration is usually updated elsewhere but let's ensure it's consistent
            # self._data["profiles"][profile_name]["avg_duration"] = float(np.mean(durations))
        
        resampled = []
        for offsets, values in normalized_curves:
            # Interpolate this cycle to the common time grid
            curve_resampled = np.interp(time_grid, offsets, values)
            resampled.append(curve_resampled)
        
        # Stack into 2D array and calculate statistics
        curves_array = np.array(resampled)
        
        envelope = {
            "min": np.min(curves_array, axis=0).tolist(),
            "max": np.max(curves_array, axis=0).tolist(),
            "avg": np.mean(curves_array, axis=0).tolist(),
            "std": np.std(curves_array, axis=0).tolist(),
            "time_grid": time_grid.tolist(),  # Store time axis for reference
            "cycle_count": len(resampled),
            "target_duration": float(target_duration),
            "sampling_rates": list(sampling_rates),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Cache in storage
        if "envelopes" not in self._data:
            self._data["envelopes"] = {}
        self._data["envelopes"][profile_name] = envelope
        
        avg_sample_rate = np.median(sampling_rates) if sampling_rates else 1.0
        _LOGGER.debug(
            f"Rebuilt envelope for '{profile_name}': {len(resampled)} cycles, "
            f"duration={target_duration:.0f}s, avg_sample_rate={avg_sample_rate:.1f}s, "
            f"normalized_to={num_points} time-aligned points"
        )
        
        return True

    def get_envelope(self, profile_name: str) -> dict | None:
        """Get cached envelope for a profile, or None if not available."""
        return self._data.get("envelopes", {}).get(profile_name)

    def match_profile(self, current_power_data: list[tuple[str, float]], current_duration: float) -> tuple[str | None, float]:
        """
        Attempt to match current running cycle to a known profile using NumPy.
        Returns (profile_name, confidence).
        Prefers complete cycles over interrupted ones.
        
        Note: Both current and stored sample cycles are full-resolution uncompressed,
        ensuring fair apples-to-apples comparison.
        """
        if not current_power_data or len(current_power_data) < 10:
            return (None, 0.0)

        best_match = None
        best_score = 0.0

        # Extract just the power values from the current cycle
        current_values = np.array([p for _, p in current_power_data])
        
        for name, profile in self._data["profiles"].items():
            # Get the sample cycle data
            sample_id = profile.get("sample_cycle_id")
            sample_cycle = next((c for c in self._data["past_cycles"] if c["id"] == sample_id), None)
            
            if not sample_cycle:
                continue
            
            # Extract power values from [offset, power] pairs (full resolution)
            sample_data = sample_cycle["power_data"]
            if not sample_data:
                continue
            
            sample_values = np.array([p for _, p in sample_data])
            
            if len(sample_values) == 0:
                continue

            # Check duration mismatch for running cycles
            # For running cycles, we need more lenient matching since we don't know final duration
            # Use configurable duration range (default: 50%-150% of profile duration)
            profile_duration = profile.get("avg_duration", sample_cycle.get("duration", 0))
            if profile_duration > 0:
                duration_ratio = current_duration / profile_duration
                if duration_ratio < self._min_duration_ratio or duration_ratio > self._max_duration_ratio:
                    _LOGGER.debug(f"Profile {name}: duration mismatch (current={current_duration:.0f}s, expected={profile_duration:.0f}s, ratio={duration_ratio:.2f}, range={self._min_duration_ratio:.2f}-{self._max_duration_ratio:.2f})")
                    continue

            # Calculate similarity
            score = self._calculate_similarity(current_values, sample_values)
            _LOGGER.debug(f"Profile {name}: similarity={score:.3f} (samples: current={len(current_values)}, sample={len(sample_values)})")
            
            # Apply status penalty: prefer complete cycles
            status = sample_cycle.get("status", "completed")
            if status in ("completed", "force_stopped"):
                score *= 1.0  # No penalty (both are natural completions)
            elif status == "resumed":
                score *= 0.85  # 15% penalty for resumed
            elif status == "interrupted":
                score *= 0.7  # 30% penalty for interrupted (user stopped)
            
            if score > best_score:
                best_score = score
                best_match = name

        return (best_match, best_score)

    def _calculate_similarity(self, current: np.ndarray, sample: np.ndarray) -> float:
        """Calculate similarity score (0-1) between two power curves."""
        len_cur = len(current)
        len_sam = len(sample)
        
        # Need at least 10% of profile to make reasonable comparison
        if len_cur < max(3, len_sam * 0.1):
            return 0.0
        
        # Compare prefix of current cycle to same-length prefix of sample
        if len_cur > len_sam:
            # Current is longer than sample - compare against full sample
            compare_sample = sample
            compare_current = current[:len_sam]
        else:
            # Current is shorter - compare against prefix of sample
            compare_sample = sample[:len_cur]
            compare_current = current
        
        # Calculate normalized similarity using multiple metrics
        try:
            # 1. Mean absolute error (MAE) in watts
            mae = np.mean(np.abs(compare_current - compare_sample))
            
            # 2. Correlation coefficient (shape similarity, -1 to 1)
            if len(compare_current) > 1 and np.std(compare_current) > 0 and np.std(compare_sample) > 0:
                correlation = np.corrcoef(compare_current, compare_sample)[0, 1]
            else:
                correlation = 0.0
            
            # 3. Peak power similarity
            peak_cur = np.max(compare_current) if len(compare_current) > 0 else 0
            peak_sam = np.max(compare_sample) if len(compare_sample) > 0 else 0
            peak_diff = abs(peak_cur - peak_sam)
            peak_score = 1.0 / (1.0 + peak_diff / 100.0)  # Normalize by 100W
            
            # Combine scores (weighted):
            # - MAE score: 40% weight (lower error = better)
            # - Correlation: 40% weight (shape similarity)
            # - Peak similarity: 20% weight
            mae_score = 1.0 / (1.0 + mae / 50.0)  # 50W is "acceptable" error
            corr_score = max(0.0, correlation)  # Clamp negative to 0
            
            final_score = 0.4 * mae_score + 0.4 * corr_score + 0.2 * peak_score
            
            _LOGGER.debug(f"Similarity calc: mae={mae:.1f}W, corr={correlation:.3f}, peak_diff={peak_diff:.1f}W, final={final_score:.3f}")
            
            return float(final_score)
            
        except Exception as e:
            _LOGGER.warning(f"Similarity calculation failed: {e}")
            return 0.0

    async def create_profile(self, name: str, source_cycle_id: str) -> None:
        """Create a new profile from a past cycle."""
        cycle = next((c for c in self._data["past_cycles"] if c["id"] == source_cycle_id), None)
        if not cycle:
             raise ValueError("Cycle not found")
        
        cycle["profile_name"] = name
        
        self._data.setdefault("profiles", {})[name] = {
            "avg_duration": cycle["duration"],
            "sample_cycle_id": source_cycle_id
        }
        
        # Save to persist the label
        await self.async_save()

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all profiles with metadata."""
        profiles = []
        for name, data in self._data.get("profiles", {}).items():
            # Count cycles using this profile
            cycle_count = sum(1 for c in self._data.get("past_cycles", []) if c.get("profile_name") == name)
            profiles.append({
                "name": name,
                "avg_duration": data.get("avg_duration", 0),
                "min_duration": data.get("min_duration", 0),
                "max_duration": data.get("max_duration", 0),
                "sample_cycle_id": data.get("sample_cycle_id"),
                "cycle_count": cycle_count,
            })
        return sorted(profiles, key=lambda p: p["name"])

    async def create_profile_standalone(self, name: str, reference_cycle_id: str = None) -> None:
        """Create a profile without immediately labeling a cycle.
        If reference_cycle_id is provided, use that cycle's characteristics."""
        if name in self._data.get("profiles", {}):
            raise ValueError(f"Profile '{name}' already exists")
        
        profile_data = {}
        if reference_cycle_id:
            cycle = next((c for c in self._data["past_cycles"] if c["id"] == reference_cycle_id), None)
            if cycle:
                profile_data = {
                    "avg_duration": cycle["duration"],
                    "sample_cycle_id": reference_cycle_id
                }
        
        # Create profile with minimal data (will be updated when cycles are labeled)
        self._data.setdefault("profiles", {})[name] = profile_data
        await self.async_save()
        _LOGGER.info(f"Created standalone profile '{name}'")

    async def rename_profile(self, old_name: str, new_name: str) -> int:
        """Rename a profile and update all cycles using it.
        Returns number of cycles updated."""
        if old_name not in self._data.get("profiles", {}):
            raise ValueError(f"Profile '{old_name}' not found")
        if new_name == old_name:
            return 0
        if new_name in self._data.get("profiles", {}):
            raise ValueError(f"Profile '{new_name}' already exists")
        
        # Rename in profiles dict
        self._data["profiles"][new_name] = self._data["profiles"].pop(old_name)
        
        # Rename corresponding envelope if it exists
        if "envelopes" in self._data and old_name in self._data["envelopes"]:
            self._data["envelopes"][new_name] = self._data["envelopes"].pop(old_name)

        # Update all cycles
        count = 0
        for cycle in self._data.get("past_cycles", []):
            if cycle.get("profile_name") == old_name:
                cycle["profile_name"] = new_name
                count += 1
        
        await self.async_save()
        _LOGGER.info(f"Renamed profile '{old_name}' to '{new_name}', updated {count} cycles")
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
        _LOGGER.info(f"Deleted profile '{name}', {action} {count} cycles")
        return count

    async def assign_profile_to_cycle(self, cycle_id: str, profile_name: str) -> None:
        """Assign an existing profile to a cycle. Rebuilds envelope."""
        old_profile = None
        cycle = next((c for c in self._data["past_cycles"] if c["id"] == cycle_id), None)
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
        _LOGGER.info(f"Assigned profile '{profile_name}' to cycle {cycle_id}")

    async def auto_label_unlabeled_cycles(self, confidence_threshold: float = 0.7) -> dict[str, int]:
        """Retroactively auto-label unlabeled cycles using profile matching.
        Returns stats: {labeled: int, skipped: int, total: int}"""
        stats = {"labeled": 0, "skipped": 0, "total": 0}
        
        unlabeled = [c for c in self._data.get("past_cycles", []) if not c.get("profile_name")]
        stats["total"] = len(unlabeled)
        
        for cycle in unlabeled:
            # Reconstruct power data for matching
            power_data = self._decompress_power_data(cycle)
            if not power_data or len(power_data) < 10:
                stats["skipped"] += 1
                continue
            
            # Try to match
            matched_profile, confidence = self.match_profile(power_data, cycle["duration"])
            
            if matched_profile and confidence >= confidence_threshold:
                cycle["profile_name"] = matched_profile
                stats["labeled"] += 1
                _LOGGER.info(f"Auto-labeled cycle {cycle['id']} as '{matched_profile}' (confidence: {confidence:.2f})")
            else:
                stats["skipped"] += 1
        
        if stats["labeled"] > 0:
            await self.async_save()
        
        _LOGGER.info(f"Auto-labeling complete: {stats['labeled']} labeled, {stats['skipped']} skipped")
        return stats

    def _decompress_power_data(self, cycle: dict) -> list[tuple[str, float]]:
        """Decompress cycle power data for matching."""
        compressed = cycle.get("power_data", [])
        if not compressed:
            return []
        
        start_time = datetime.fromisoformat(cycle["start_time"])
        result = []
        
        for item in compressed:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                offset_seconds, power = item
                timestamp = (start_time.timestamp() + offset_seconds)
                result.append((datetime.fromtimestamp(timestamp).isoformat(), power))
        
        return result

    async def async_save_cycle(self, cycle_data: dict[str, Any]) -> None:
        """Add and save a cycle. Rebuilds envelope if cycle is labeled."""
        self.add_cycle(cycle_data)
        
        # If cycle has a profile, rebuild that profile's envelope
        profile_name = cycle_data.get("profile_name")
        if profile_name:
            self.rebuild_envelope(profile_name)
        
        await self.async_save()

    async def async_migrate_cycles_to_compressed(self) -> int:
        """
        Migrate all cycles to the compressed format.
        Ensures all cycles use [offset_seconds, power] format.
        Returns number of cycles migrated.
        """
        cycles = self._data.get("past_cycles", [])
        migrated = 0
        
        for cycle in cycles:
            raw_data = cycle.get("power_data", [])
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
                compressed = []
                
                last_saved_p = -999.0
                last_saved_t = -999.0
                
                for i, point in enumerate(raw_data):
                    # Parse timestamp
                    if isinstance(point[0], str):
                        t_val = datetime.fromisoformat(point[0]).timestamp()
                    else:
                        t_val = float(point[0])
                    
                    p_val = point[1]
                    offset = round(t_val - start_ts, 1)
                    
                    # Save first and last
                    is_endpoint = (i == 0 or i == len(raw_data) - 1)
                    
                    # Downsample: change > 1W or gap > 60s
                    if is_endpoint or abs(p_val - last_saved_p) > 1.0 or (offset - last_saved_t) > 60:
                        compressed.append([offset, round(p_val, 1)])
                        last_saved_p = p_val
                        last_saved_t = offset
                
                cycle["power_data"] = compressed
                migrated += 1
            except Exception as e:
                _LOGGER.warning(f"Failed to migrate cycle {cycle.get('id')}: {e}")
                continue
        
        if migrated > 0:
            _LOGGER.info(f"Migrated {migrated} cycles to compressed format")
            await self.async_save()
        
        return migrated

    def merge_cycles(self, hours: int = 24, gap_threshold: int = 1800) -> int:
        """
        Merge fragmented cycles within the last X hours.
        gap_threshold: max seconds between cycles to consider them one (default 30m).
        Returns number of merges performed.
        """
        # Use timezone-aware now to match stored timestamps
        from homeassistant.util import dt as dt_util
        limit = dt_util.now().timestamp() - (hours * 3600)
        cycles = self._data["past_cycles"]
        if not cycles:
            return 0
        
        # Sort by start time just in case
        cycles.sort(key=lambda x: x["start_time"])
        
        merged_count = 0
        i = 0
        while i < len(cycles) - 1:
            c1 = cycles[i]
            c2 = cycles[i+1]
            
            # Parse times
            try:
                # Isoformat handles T separator? My code produces it.
                t1_end = datetime.fromisoformat(c1["end_time"]).timestamp()
                t2_start = datetime.fromisoformat(c2["start_time"]).timestamp()
            except ValueError:
                i += 1
                continue
            
            # Check time window (only touch if at least one is in range)
            # If both are old, skip
            if t1_end < limit and datetime.fromisoformat(c2["end_time"]).timestamp() < limit:
                i += 1
                continue
                
            gap = t2_start - t1_end
            
            if 0 <= gap <= gap_threshold:
                # MERGE c2 into c1
                _LOGGER.info(f"Merging cycle {c2['id']} into {c1['id']} (Gap: {gap}s)")
                
                # Update c1 duration and end time
                t2_end = datetime.fromisoformat(c2["end_time"]).timestamp()
                t1_start = datetime.fromisoformat(c1["start_time"]).timestamp()
                
                c1["end_time"] = c2["end_time"]
                c1["duration"] = t2_end - t1_start
                
                # Merge power data
                # Since stored data is now relative offsets [offset, power]
                # We need to shift c2's offsets by the time difference (gap + c1_duration_before_merge?)
                # Actually, c2 offsets are relative to c2 start.
                # New offsets must be relative to c1 start.
                shift = t2_start - t1_start
                
                # Check format of c2/c1. If old format (string timestamps), we can't easily math it here without parsing.
                # Assuming new format if we are here (or we should check).
                # To be safe, let's try to detect.
                
                c2_data = c2["power_data"]
                if c2_data and isinstance(c2_data[0][0], (int, float)):
                    # Shift it
                    shifted_c2 = [[round(x[0] + shift, 1), x[1]] for x in c2_data]
                    c1["power_data"].extend(shifted_c2)
                else:
                    # fallback for old data (ISO strings) - just append, though it will be messy
                    c1["power_data"].extend(c2_data)
                
                # If c2 had a max power higher, take it
                c1["max_power"] = max(c1.get("max_power", 0), c2.get("max_power", 0))
                
                # PRESERVE PROFILE
                # If c1 is unlabeled but c2 has a label, take c2's label
                if not c1.get("profile_name") and c2.get("profile_name"):
                    c1["profile_name"] = c2["profile_name"]
                
                # Track old IDs for profile update
                old_c1_id = c1["id"]
                old_c2_id = c2["id"]
                
                # Regenerate ID
                unique_str = f"{c1['start_time']}_{c1['duration']}"
                new_id = hashlib.sha256(unique_str.encode()).hexdigest()[:12]
                c1["id"] = new_id
                
                # UPDATE PROFILE REFERENCES
                # If any profile pointed to old_c1_id or old_c2_id, update to new_id
                for p_name, p_data in self._data["profiles"].items():
                    if p_data.get("sample_cycle_id") in (old_c1_id, old_c2_id):
                        if c1.get("profile_name") == p_name:
                             # Only update if this cycle is actually the one named p_name?
                             # Or just update generically?
                             # If we merged them, this new cycle is the best representative now.
                             p_data["sample_cycle_id"] = new_id
                             # Also update avg duration? Maybe later.
                
                # Remove c2
                cycles.pop(i+1)
                
                merged_count += 1
                # Do NOT increment i, so we can check if the NEW c1 merges with c3
            else:
                i += 1
                
        return merged_count

    def log_adjustment(self, setting_name: str, old_value: Any, new_value: Any, reason: str) -> None:
        """Log an automatic setting adjustment (auto-tune, auto-label changes)."""
        adjustment = {
            "timestamp": datetime.now().isoformat(),
            "setting": setting_name,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }
        self._data.setdefault("auto_adjustments", []).append(adjustment)
        # Keep last 50 adjustments
        if len(self._data["auto_adjustments"]) > 50:
            self._data["auto_adjustments"] = self._data["auto_adjustments"][-50:]
        _LOGGER.info(f"Auto-adjustment: {setting_name} changed from {old_value} to {new_value} ({reason})")

    def export_data(self, entry_data: dict = None, entry_options: dict = None) -> dict[str, Any]:
        """Return a serializable snapshot of the store for backup/export.
        Includes config entry data/options so users can transfer fine-tuned settings."""
        return {
            "version": STORAGE_VERSION,
            "entry_id": self.entry_id,
            "exported_at": datetime.now().isoformat(),
            "data": self._data,
            "entry_data": entry_data or {},
            "entry_options": entry_options or {},
        }

    async def async_import_data(self, payload: dict[str, Any]) -> dict[str, dict]:
        """Load store data from an export payload and persist it.
        Returns dict with 'entry_data' and 'entry_options' keys for updating the config entry."""
        if not isinstance(payload, dict):
            raise ValueError("Invalid export payload (not a dict)")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("Invalid export payload (missing data)")

        # Basic shape repair to avoid key errors
        data.setdefault("profiles", {})
        data.setdefault("past_cycles", [])

        self._data = data
        await self.async_save()
        
        # Return config data/options for caller to apply
        return {
            "entry_data": payload.get("entry_data", {}),
            "entry_options": payload.get("entry_options", {}),
        }
