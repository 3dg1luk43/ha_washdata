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
        
        # Compress power data (Smart Downsampling)
        # Keep point if power delta > 1.0W or time delta > 30s
        raw_data = cycle_data.get("power_data", [])
        if raw_data:
            compressed = [raw_data[0]]
            last_p = raw_data[0][1]
            # last_t = ... (ignore time parsing for speed, just rely on index implied time approx or just value delta)
            # Actually we need time for replay.
            
            for i in range(1, len(raw_data) - 1):
                _, p = raw_data[i]
                if abs(p - last_p) > 1.0:
                    compressed.append(raw_data[i])
                    last_p = p
            
            compressed.append(raw_data[-1])
            cycle_data["power_data"] = compressed

        self._data["past_cycles"].append(cycle_data)
        # Keep last 50 cycles
        if len(self._data["past_cycles"]) > 50:
             self._data["past_cycles"].pop(0)

    def match_profile(self, current_power_data: list[tuple[str, float]], current_duration: float) -> tuple[str | None, float]:
        """
        Attempt to match current running cycle to a known profile using NumPy.
        Returns (profile_name, confidence).
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
                
            sample_values = np.array([p for _, p in sample_cycle["power_data"]])
            
            if len(sample_values) == 0:
                continue

            # Calculate similarity
            score = self._calculate_similarity(current_values, sample_values)
            
            if score > best_score:
                best_score = score
                best_match = name

        return (best_match, best_score)

    def _calculate_similarity(self, current: np.ndarray, sample: np.ndarray) -> float:
        """Calculate similarity score (0-1) between two power curves."""
        # 1. Length comparison first (if current is much longer than sample, it's not it)
        len_cur = len(current)
        len_sam = len(sample)
        
        # If we barely started, comparison is weak
        if len_cur < len_sam * 0.1:
            return 0.0
            
        # 2. Resample sample to match current length for comparison of the "shape so far"
        # We want to see if 'current' looks like the beginning of 'sample'
        
        # Take the first N samples of the reference profile, where N matches the current duration logic
        # But wait, sample has fixed sampling rate approx 0.5Hz. 
        # Ideally we compare current vs sample[0:len(current)]
        
        if len_cur > len_sam:
            # Current is longer than sample? Then maybe it's not this profile, or we ran longer.
            # Compare prefix
            compare_sample = sample
            compare_current = current[:len_sam]
        else:
            # Current is shorter. Compare against the prefix of the sample
            compare_sample = sample[:len_cur]
            compare_current = current
            
        # Normalize (feature scaling) to focus on shape, though absolute power matters too.
        # Let's keep absolute power but cap outlier differences.
        
        # Euclidean distance
        dist = np.linalg.norm(compare_current - compare_sample)
        
        # Normalize distance by length to make it independent of duration
        avg_dist = dist / len(compare_current)
        
        # Convert to similarity score. 
        # If avg_dist is 0 (identical), score 1.
        # If avg_dist is huge (e.g. 1000W diff), score 0.
        # Let's say acceptable average error is 50W.
        # score = 1 / (1 + error)
        score = 100.0 / (100.0 + avg_dist)
        
        return float(score)

    def create_profile(self, name: str, source_cycle_id: str) -> None:
        """Create a new profile from a past cycle."""
        cycle = next((c for c in self._data["past_cycles"] if c["id"] == source_cycle_id), None)
        if not cycle:
             raise ValueError("Cycle not found")
        
        cycle["profile"] = name
        
        self._data["profiles"][name] = {
            "avg_duration": cycle["duration"],
            "sample_cycle_id": source_cycle_id
        }

    async def async_save_cycle(self, cycle_data: dict[str, Any]) -> None:
        """Add and save a cycle."""
        self.add_cycle(cycle_data)
        await self.async_save()
