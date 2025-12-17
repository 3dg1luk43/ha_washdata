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

from .const import DOMAIN, STORAGE_KEY, STORAGE_VERSION

_LOGGER = logging.getLogger(__name__)

class ProfileStore:
    """Manages storage of washer profiles and past cycles."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize the profile store."""
        self.hass = hass
        self.entry_id = entry_id
        # Separate store for each entry to avoid giant files
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}.{entry_id}")
        self._data: dict[str, Any] = {
            "profiles": {},
            "past_cycles": []
        }

    async def async_load(self) -> None:
        """Load data from storage."""
        data = await self._store.async_load()
        if data:
            self._data = data

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
        
        cycle_data["profile"] = None  # Initially unknown
        
        # Compress power data (Smart Downsampling + Relative Time)
        # Convert to [seconds_offset, power] to save massive space (recoverable)
        raw_data = cycle_data.get("power_data", [])
        _LOGGER.debug(f"add_cycle: raw_data has {len(raw_data)} points, first={raw_data[0] if raw_data else 'none'}")
        if raw_data:
            start_ts = datetime.fromisoformat(cycle_data["start_time"]).timestamp()
            compressed = []
            
            # Helper to parse time
            def get_ts(item):
                # item is [iso_str, power] (usually)
                if isinstance(item[0], str):
                    return datetime.fromisoformat(item[0]).timestamp()
                return float(item[0]) # Start timestamp relative? No, usually iso.

            last_saved_p = -999.0
            last_saved_t = -999.0
            
            for i, point in enumerate(raw_data):
                t_val = get_ts(point)
                p_val = point[1]
                
                # Relative offset rounded to 1 decimal place
                offset = round(t_val - start_ts, 1)
                
                # Save first and last point always
                is_endpoint = (i == 0 or i == len(raw_data) - 1)
                
                # Filter: Only save if power changed > 1.0W OR time gap > 60s
                # AND always save endpoints
                if is_endpoint or abs(p_val - last_saved_p) > 1.0 or (offset - last_saved_t) > 60:
                    # Append [offset, power]
                    compressed.append([offset, round(p_val, 1)])
                    last_saved_p = p_val
                    last_saved_t = offset
            
            _LOGGER.debug(f"add_cycle: compressed to {len(compressed)} points, first={compressed[0] if compressed else 'none'}")
            cycle_data["power_data"] = compressed

        self._data["past_cycles"].append(cycle_data)
        # Keep last 50 cycles
        if len(self._data["past_cycles"]) > 50:
             self._data["past_cycles"].pop(0)

    def delete_cycle(self, cycle_id: str) -> bool:
        """Delete a cycle by ID. Returns True if deleted, False if not found."""
        cycles = self._data["past_cycles"]
        for i, cycle in enumerate(cycles):
            if cycle.get("id") == cycle_id:
                cycles.pop(i)
                _LOGGER.info(f"Deleted cycle {cycle_id}")
                return True
        _LOGGER.warning(f"Cycle {cycle_id} not found for deletion")
        return False

    def match_profile(self, current_power_data: list[tuple[str, float]], current_duration: float) -> tuple[str | None, float]:
        """
        Attempt to match current running cycle to a known profile using NumPy.
        Returns (profile_name, confidence).
        Prefers complete cycles over interrupted ones.
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
            
            # Handle both compressed [offset, power] and old [iso_timestamp, power] formats
            sample_data = sample_cycle["power_data"]
            if not sample_data:
                continue
            
            # Check format by inspecting first element
            if isinstance(sample_data[0][0], (int, float)):
                # Compressed format: [offset_seconds, power]
                sample_values = np.array([p for _, p in sample_data])
            else:
                # Old format: [iso_timestamp, power]
                sample_values = np.array([p for _, p in sample_data])
            
            if len(sample_values) == 0:
                continue

            # Check duration mismatch - reject if too different from profile's expected duration
            profile_duration = profile.get("avg_duration", sample_cycle.get("duration", 0))
            if profile_duration > 0:
                duration_ratio = current_duration / profile_duration
                # Reject if current is < 50% or > 150% of profile duration
                if duration_ratio < 0.5 or duration_ratio > 1.5:
                    continue

            # Calculate similarity
            score = self._calculate_similarity(current_values, sample_values)
            
            # Apply status penalty: prefer complete cycles
            status = sample_cycle.get("status", "completed")
            if status == "completed":
                score *= 1.0  # No penalty
            elif status == "resumed":
                score *= 0.85  # 15% penalty for resumed
            elif status in ("interrupted", "force_stopped"):
                score *= 0.7  # 30% penalty for interrupted/stopped
            
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
        
        cycle["profile"] = name
        
        self._data["profiles"][name] = {
            "avg_duration": cycle["duration"],
            "sample_cycle_id": source_cycle_id
        }
        
        # Save to persist the label
        await self.async_save()

    async def async_save_cycle(self, cycle_data: dict[str, Any]) -> None:
        """Add and save a cycle."""
        self.add_cycle(cycle_data)
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
                if not c1.get("profile") and c2.get("profile"):
                    c1["profile"] = c2["profile"]
                
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
                        if c1.get("profile") == p_name:
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
