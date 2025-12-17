"""Cycle detection logic for HA WashData."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from homeassistant.util import dt as dt_util

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
        self._low_power_start: datetime | None = None  # Track when we entered low-power waiting
        self._cycle_max_power: float = 0.0
        self._ma_buffer: list[float] = []
        self._cycle_status: str | None = None  # Track how cycle ended

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
        _LOGGER.debug(f"process_reading: power={power}W, avg={avg_power:.1f}W, is_active={is_active}, state={self._state}, min_power={self._config.min_power}W")

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
            
            # Safety check: if cycle has been running for > 4 hours, force end (prevents infinite cycles)
            if self._current_cycle_start and (timestamp - self._current_cycle_start).total_seconds() > 14400:
                import logging
                logging.getLogger(__name__).warning(f"Force-ending cycle after 4+ hours (likely stuck)")
                self._finish_cycle(timestamp, status="force_stopped")
                return
            
            if is_active:
                 self._last_active_time = timestamp
                 self._low_power_start = None  # Reset low-power timer
            else:
                 # Track when low power started
                 if not self._low_power_start:
                     self._low_power_start = timestamp
                     _LOGGER.debug(f"Low power detected, starting completion timer")
                 
                 # Check if we should conclude the cycle
                 low_duration = (timestamp - self._low_power_start).total_seconds()
                 _LOGGER.debug(f"Low power: duration={low_duration:.1f}s, off_delay={self._config.off_delay}s, will_end={low_duration > self._config.off_delay}")
                 if low_duration > self._config.off_delay:
                     # Power has been low for the configured delay - cycle is done NATURALLY
                     _LOGGER.info(f"Ending cycle: power below {self._config.min_power}W for {low_duration:.0f}s (threshold: {self._config.off_delay}s)")
                     self._finish_cycle(timestamp, status="completed")

    def _transition_to(self, new_state: str, timestamp: datetime) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        _LOGGER.debug("Transition: %s -> %s at %s", old_state, new_state, timestamp)
        self._on_state_change(old_state, new_state)

    def _finish_cycle(self, timestamp: datetime, status: str | None = None) -> None:
        """Finalize the current cycle."""
        self._transition_to(STATE_OFF, timestamp)
        
        if not self._current_cycle_start:
            return

        # Ensure timestamps are valid - if _last_active_time is invalid, use provided timestamp
        if not self._last_active_time or self._last_active_time < self._current_cycle_start:
            self._last_active_time = timestamp

        duration = (self._last_active_time - self._current_cycle_start).total_seconds()
        if duration < 0:
            # Clock issue or bad state - use current timestamp as end
            self._last_active_time = self._current_cycle_start
            duration = 0
        
        _LOGGER.info(f"Cycle finished: {int(duration/60)}m, max_power={self._cycle_max_power}W, samples={len(self._power_readings)}, status={status or 'completed'}")
        
        # Default to completed if not specified
        if status is None:
            status = "completed"
        
        cycle_data = {
            "start_time": self._current_cycle_start.isoformat(),
            "end_time": self._last_active_time.isoformat(),
            "duration": duration,
            "max_power": self._cycle_max_power,
            "status": status,
            "power_data": [(t.isoformat(), p) for t, p in self._power_readings],
        }
        
        self._on_cycle_end(cycle_data)
        
        # Cleanup
        self._power_readings = []
        self._current_cycle_start = None
        self._last_active_time = None
        self._low_power_start = None
        self._ma_buffer = []

    def force_end(self, timestamp: datetime) -> None:
        """Force-finish the current cycle (used by watchdog when sensor stops sending data)."""
        if self._state != STATE_RUNNING:
            return

        # Check if we're in low-power waiting state (natural completion)
        if self._low_power_start:
            elapsed = (timestamp - self._low_power_start).total_seconds()
            if elapsed >= self._config.off_delay:
                _LOGGER.info(f"Watchdog: completing cycle naturally (low power for {elapsed:.0f}s)")
                self._finish_cycle(timestamp, status="completed")
                return
        
        # Not in low-power state or haven't waited long enough - this is a forced stop
        _LOGGER.warning(f"Watchdog: force-stopping cycle (no data received)")
        self._finish_cycle(timestamp, status="force_stopped")

    def get_power_trace(self) -> list[tuple[datetime, float]]:
        """Return a copy of the current raw power trace."""
        return list(self._power_readings)

    def get_elapsed_seconds(self) -> float:
        """Return elapsed seconds in the current cycle based on readings."""
        if not self._current_cycle_start or not self._power_readings:
            return 0.0
        return (self._power_readings[-1][0] - self._current_cycle_start).total_seconds()
    def get_state_snapshot(self) -> dict:
        """Return a snapshot of current state."""
        return {
            "state": self._state,
            "current_cycle_start": self._current_cycle_start.isoformat() if self._current_cycle_start else None,
            "last_active_time": self._last_active_time.isoformat() if self._last_active_time else None,
            "low_power_start": self._low_power_start.isoformat() if self._low_power_start else None,
            "cycle_max_power": self._cycle_max_power,
            "power_readings": [(t.isoformat(), p) for t, p in self._power_readings],
            "ma_buffer": getattr(self, "_ma_buffer", []),
        }

    def restore_state_snapshot(self, snapshot: dict) -> None:
        """Restore state from snapshot."""
        try:
            self._state = snapshot.get("state", STATE_OFF)
            
            start = snapshot.get("current_cycle_start")
            if start:
                parsed = dt_util.parse_datetime(start)
                # Ensure timezone-aware - if naive, assume local timezone
                self._current_cycle_start = dt_util.as_local(parsed) if parsed else None
            else:
                self._current_cycle_start = None
            
            last = snapshot.get("last_active_time")
            if last:
                parsed = dt_util.parse_datetime(last)
                # Ensure timezone-aware - if naive, assume local timezone
                self._last_active_time = dt_util.as_local(parsed) if parsed else None
            else:
                self._last_active_time = None
            
            # Restore low_power_start if present (tracking when we entered low-power waiting)
            low_start = snapshot.get("low_power_start")
            if low_start:
                parsed = dt_util.parse_datetime(low_start)
                self._low_power_start = dt_util.as_local(parsed) if parsed else None
            else:
                self._low_power_start = None
            
            self._cycle_max_power = snapshot.get("cycle_max_power", 0.0)
            
            readings = snapshot.get("power_readings", [])
            self._power_readings = []
            for t, p in readings:
                parsed = dt_util.parse_datetime(t)
                if parsed:
                    # Ensure timezone-aware
                    self._power_readings.append((dt_util.as_local(parsed), p))
            
            # Restore buffer if present, else rebuild from last few readings? 
            # Or just start empty? If we start empty, average is 0 -> isActive False.
            # If actual power is high, next reading will fix it.
            # But if actual power is low/fluctuating...
            # Best to restore.
            self._ma_buffer = snapshot.get("ma_buffer", [])
            
            _LOGGER.info(f"Restored CycleDetector state: {self._state}, {len(self._power_readings)} readings")
            
        except Exception as e:
            _LOGGER.error(f"Failed to restore CycleDetector state: {e}")
            self._state = STATE_OFF
            self._power_readings = []
