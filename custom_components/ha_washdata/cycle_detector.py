"""Cycle detection logic for HA WashData."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .const import STATE_OFF, STATE_RUNNING, STATE_IDLE

_LOGGER = logging.getLogger(__name__)

@dataclass
class CycleDetectorConfig:
    """Configuration for cycle detection."""
    min_power: float
    off_delay: int


class CycleDetector:
    """Detects washing machine cycles based on power usage."""

    def __init__(
        self,
        config: CycleDetectorConfig,
        on_state_change: Callable[[str, str], None],
        on_cycle_end: Callable[[dict], None],
    ) -> None:
        """Initialize the cycle detector."""
        self._config = config
        self._on_state_change = on_state_change
        self._on_cycle_end = on_cycle_end

        self._state = STATE_OFF
        self._power_readings: list[tuple[datetime, float]] = []
        self._current_cycle_start: datetime | None = None
        self._last_active_time: datetime | None = None
        self._cycle_max_power: float = 0.0
        self._ma_buffer: list[float] = []

    @property
    def state(self) -> str:
        """Return current state."""
        return self._state

    def process_reading(self, power: float, timestamp: datetime) -> None:
        """Process a new power reading."""
        # Update raw history for graph
        # But we use SMOOTHED power for state logic to filter noise
        
        # Buffer for moving average (last 5 readings)
        self._ma_buffer = getattr(self, "_ma_buffer", [])
        self._ma_buffer.append(power)
        if len(self._ma_buffer) > 5:
            self._ma_buffer.pop(0)
            
        avg_power = sum(self._ma_buffer) / len(self._ma_buffer)
        
        # Use smoothed power for threshold check
        is_active = avg_power >= self._config.min_power

        if self._state == STATE_OFF:
            if is_active:
                self._transition_to(STATE_RUNNING, timestamp)
                self._current_cycle_start = timestamp
                self._power_readings = [(timestamp, power)]
                self._last_active_time = timestamp
                self._cycle_max_power = power

        elif self._state == STATE_RUNNING:
            self._power_readings.append((timestamp, power))
            # Track max of RAW power
            if power > self._cycle_max_power:
                self._cycle_max_power = power
            
            if is_active:
                 self._last_active_time = timestamp
            else:
                 # Check if we should conclude the cycle
                 if self._last_active_time and (timestamp - self._last_active_time).total_seconds() > self._config.off_delay:
                     self._finish_cycle(timestamp)

    def _transition_to(self, new_state: str, timestamp: datetime) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        _LOGGER.debug("Transition: %s -> %s at %s", old_state, new_state, timestamp)
        self._on_state_change(old_state, new_state)

    def _finish_cycle(self, timestamp: datetime) -> None:
        """Finalize the current cycle."""
        self._transition_to(STATE_OFF, timestamp)
        
        if not self._current_cycle_start:
            return

        duration = (self._last_active_time - self._current_cycle_start).total_seconds()
        
        cycle_data = {
            "start_time": self._current_cycle_start.isoformat(),
            "end_time": self._last_active_time.isoformat(),
            "duration": duration,
            "max_power": self._cycle_max_power,
            "power_data": [(t.isoformat(), p) for t, p in self._power_readings],
        }
        
        self._on_cycle_end(cycle_data)
        
        # Cleanup
        self._power_readings = []
        self._current_cycle_start = None
        self._current_cycle_start = None
        self._last_active_time = None
        self._ma_buffer = []
    def get_state_snapshot(self) -> dict:
        """Return a snapshot of current state."""
        return {
            "state": self._state,
            "current_cycle_start": self._current_cycle_start.isoformat() if self._current_cycle_start else None,
            "last_active_time": self._last_active_time.isoformat() if self._last_active_time else None,
            "cycle_max_power": self._cycle_max_power,
            "power_readings": [(t.isoformat(), p) for t, p in self._power_readings],
        }

    def restore_state_snapshot(self, snapshot: dict) -> None:
        """Restore state from snapshot."""
        try:
            self._state = snapshot.get("state", STATE_OFF)
            
            start = snapshot.get("current_cycle_start")
            self._current_cycle_start = datetime.fromisoformat(start) if start else None
            
            last = snapshot.get("last_active_time")
            self._last_active_time = datetime.fromisoformat(last) if last else None
            
            self._cycle_max_power = snapshot.get("cycle_max_power", 0.0)
            
            readings = snapshot.get("power_readings", [])
            self._power_readings = [(datetime.fromisoformat(t), p) for t, p in readings]
            
            _LOGGER.info(f"Restored CycleDetector state: {self._state}, {len(self._power_readings)} readings")
            
        except Exception as e:
            _LOGGER.error(f"Failed to restore CycleDetector state: {e}")
            self._state = STATE_OFF
            self._power_readings = []
