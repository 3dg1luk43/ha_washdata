"""Profile storage and matching logic for HA WashData."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

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

    def add_cycle(self, cycle_data: dict[str, Any]) -> None:
        """Add a completed cycle to history."""
        # Add ID to cycle
        cycle_data["id"] = str(len(self._data["past_cycles"]) + 1)
        cycle_data["profile"] = None  # Initially unknown
        self._data["past_cycles"].append(cycle_data)
        # Keep last 50 cycles
        if len(self._data["past_cycles"]) > 50:
            self._data["past_cycles"].pop(0)

    def match_profile(self, current_power_data: list[tuple[str, float]], current_duration: float) -> tuple[str | None, float]:
        """
        Attempt to match current running cycle to a known profile.
        Returns (profile_name, confidence).
        """
        best_match = None
        best_score = 0.0

        # Very basic matching: compare duration and average power of specific segments
        # This is a placeholder for the "heuristic/ML-lite" requirement.
        # Real implementation would do dynamic time warping or similar.
        
        # For now, let's just use duration if we have profiles with durations
        for name, profile in self._data["profiles"].items():
            expected_duration = profile.get("avg_duration")
            if not expected_duration:
                continue
            
            # If we are within 20% of expected duration, it's a weak hint
            # But really we want to match active shape.
            
            # Let's simple check: if we are near the end of expected duration, might be it?
            # No, we need early detection.
            
            # Simple heuristic: compare average power of first N minutes if enough data
            pass

        return (None, 0.0)

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
