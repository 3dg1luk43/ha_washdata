"""Recorder for raw cycle data in HA WashData."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import STORAGE_VERSION, STORAGE_KEY

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY_RECORDER = f"{STORAGE_KEY}.recorder"



class RecorderStore(Store):
    """Store for recorder data with migration support."""

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
        old_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Migrate data to the new version."""
        _LOGGER.info(
            "Migrating recorder storage from v%s to v%s",
            old_major_version,
            STORAGE_VERSION,
        )
        # Recorder data schema hasn't changed, simple pass-through is safe
        return old_data


class CycleRecorder:
    """Records raw power data without interference from detection logic."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize the recorder."""
        self.hass = hass
        self.entry_id = entry_id
        self._store = RecorderStore(hass, STORAGE_VERSION, f"{STORAGE_KEY_RECORDER}.{entry_id}")
        
        # State
        self._is_recording = False
        self._start_time: datetime | None = None
        self._buffer: list[tuple[str, float]] = []  # stored as (iso_str, power) for easy json
        self._last_save: datetime | None = None
        self._last_run: dict[str, Any] | None = None

    @property
    def is_recording(self) -> bool:
        """Return True if recording is active."""
        return self._is_recording

    @property
    def start_time(self) -> datetime | None:
        """Return recording start time."""
        return self._start_time

    @property
    def current_duration(self) -> float:
        """Return current recording duration in seconds."""
        if self._start_time:
            return (dt_util.now() - self._start_time).total_seconds()
        return 0.0

    async def async_load(self) -> None:
        """Load state from storage."""
        data = await self._store.async_load()
        if data:
            self._is_recording = data.get("is_recording", False)
            start_iso = data.get("start_time")
            if start_iso:
                try:
                    self._start_time = dt_util.parse_datetime(start_iso)
                except ValueError:
                    self._start_time = None
            self._buffer = data.get("buffer", [])
            self._last_run = data.get("last_run")  # Restore last run
            _LOGGER.info(
                "Loaded recorder state: recording=%s, samples=%d, has_last_run=%s",
                self._is_recording,
                len(self._buffer),
                self._last_run is not None,
            )

    async def stop_recording(self) -> dict[str, Any]:
        """Stop recording and save data for processing."""
        if not self._is_recording:
            return {}

        _LOGGER.info("Stopping cycle recording. Total samples: %d", len(self._buffer))
        self._is_recording = False
        
        # Create output packet
        result = {
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": dt_util.now().isoformat(),
            "data": list(self._buffer),  # Copy
        }
        
        # Save as last run (persisted)
        self._last_run = result
        
        # Clear active state
        self._start_time = None
        self._buffer = []
        await self._async_save()
        
        return result

    @property
    def last_run(self) -> dict[str, Any] | None:
        """Return the last recorded cycle data."""
        return getattr(self, "_last_run", None)

    async def clear_last_run(self) -> None:
        """Clear the last recorded run."""
        self._last_run = None
        await self._async_save()

    async def _async_save(self) -> None:
        """Save state to storage."""
        data = {
            "is_recording": self._is_recording,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "buffer": self._buffer,
            "last_run": getattr(self, "_last_run", None),
        }
        await self._store.async_save(data)
        self._last_save = dt_util.now()

    async def start_recording(self) -> None:
        """Start a new recording."""
        if self._is_recording:
            _LOGGER.warning("Recording already in progress")
            return

        _LOGGER.info("Starting new cycle recording")
        # Previous recordings are kept until explicitly cleared or overwritten
        
        self._is_recording = True
        self._start_time = dt_util.now()
        self._buffer = []
        await self._async_save()

    def process_reading(self, power: float) -> None:
        """Process a power reading (synchronous to avoid blocking loop)."""
        if not self._is_recording:
            return

        now = dt_util.now()
        # Append to buffer
        self._buffer.append((now.isoformat(), float(power)))

        # Periodic save every 60s to ensure data persistence
        # Better safe than sorry: save if last save was > 1 minute ago
        if self._last_save and (now - self._last_save).total_seconds() > 60:
            self.hass.add_job(self._async_save)
        elif not self._last_save:
            self.hass.add_job(self._async_save)

    def get_trim_suggestions(
        self,
        data: list[tuple[str, float]],
        recording_start: datetime | None = None,
        recording_end: datetime | None = None,
    ) -> tuple[float, float, float]:
        """Analyze data to propose trims.

        Args:
            data: List of (iso_timestamp, power)
            recording_start: Actual start time of recording (for head trim relative to start)
            recording_end: Actual end time of recording (for tail trim relative to end)

        Returns: (head_trim_seconds, tail_trim_seconds, average_noise_power)
        """
        if not data:
            # No data found - return full recording duration as trim
            if recording_start and recording_end:
                dur = (recording_end - recording_start).total_seconds()
                return 0.0, dur, 0.0
            return 0.0, 0.0, 0.0

        # Parse timestamps and powers
        parsed = []
        for t_str, p in data:
            t = dt_util.parse_datetime(t_str)
            if t:
                parsed.append((t.timestamp(), p))

        if not parsed:
            return 0.0, 0.0, 0.0

        data_start_ts = parsed[0][0]
        data_end_ts = parsed[-1][0]

        # Use provided bounds or fallback to data bounds
        rec_start_ts = recording_start.timestamp() if recording_start else data_start_ts
        rec_end_ts = recording_end.timestamp() if recording_end else data_end_ts

        # Ensure bounds cover data
        rec_start_ts = min(rec_start_ts, data_start_ts)
        rec_end_ts = max(rec_end_ts, data_end_ts)

        threshold = 2.0  # W

        first_active_idx = -1
        last_active_idx = -1

        for i, (_, p) in enumerate(parsed):
            if p > threshold:
                if first_active_idx == -1:
                    first_active_idx = i
                last_active_idx = i

        if first_active_idx == -1:
            # No activity found
            total_dur = rec_end_ts - rec_start_ts
            return 0.0, round(total_dur, 1), 0.0

        head_ts = parsed[first_active_idx][0]
        tail_ts = parsed[last_active_idx][0]

        if len(parsed) > 1:
            dts = [t - s for (t, _), (s, _) in zip(parsed[1:], parsed[:-1])]
            # Median calculation without numpy
            dts.sort()
            mid = len(dts) // 2
            avg_dt = dts[mid]
            if avg_dt <= 0:
                avg_dt = 1.0  # Fallback
        else:
            avg_dt = 1.0

        # 1. Head Trim
        # Time from recording start to first active sample
        raw_head_trim = max(0.0, head_ts - rec_start_ts)

        # Align to sampling rate (floor to keep buffer)
        # Example: raw=19s, dt=10s -> trim 10s. Buffer=9s.
        # Example: raw=21s, dt=10s -> trim 20s. Buffer=1s.
        # To ensure we don't cut active sample if jitter:
        # We start at rec_start_ts. We want start_time + trim <= head_ts
        # floor ensures this.

        steps_head = int(raw_head_trim / avg_dt)
        # Align trim to sampling rate (floor to keep buffer before active sample)

        # However, if using the "floor" logic makes it 0, that's fine.
        head_trim = steps_head * avg_dt

        # 2. Tail Trim
        # Time from last active sample to recording end
        raw_tail_trim = max(0.0, rec_end_ts - tail_ts)
        steps_tail = int(raw_tail_trim / avg_dt)
        tail_trim = steps_tail * avg_dt

        return round(head_trim, 1), round(tail_trim, 1), round(avg_dt, 1)
