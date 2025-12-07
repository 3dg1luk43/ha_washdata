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

    @property
    def state(self) -> str:
        """Return current state."""
        return self._state

    def process_reading(self, power: float, timestamp: datetime) -> None:
        """Process a new power reading."""
        # Simple threshold logic for now
        is_active = power >= self._config.min_power

        if self._state == STATE_OFF:
            if is_active:
                self._transition_to(STATE_RUNNING, timestamp)
                self._current_cycle_start = timestamp
                self._power_readings = [(timestamp, power)]
                self._last_active_time = timestamp

        elif self._state == STATE_RUNNING:
            self._power_readings.append((timestamp, power))
            
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
            "power_data": [(t.isoformat(), p) for t, p in self._power_readings],
        }
        
        self._on_cycle_end(cycle_data)
        
        # Cleanup
        self._power_readings = []
        self._current_cycle_start = None
        self._last_active_time = None
