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
    smoothing_window: int = 5
    interrupted_min_seconds: int = 150
    abrupt_drop_watts: float = 500.0
    abrupt_drop_ratio: float = 0.6
    abrupt_drop_watts: float = 500.0
    abrupt_drop_ratio: float = 0.6
    abrupt_high_load_factor: float = 5.0
    completion_min_seconds: int = 600


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
        self._last_power: float | None = None  # Track previous raw power reading
        self._abrupt_drop: bool = False  # Flag abrupt drop events

    @property
    def state(self) -> str:
        """Return current state."""
        return self._state

    def process_reading(self, power: float, timestamp: datetime) -> None:
        """Process a new power reading."""
        prev_power = self._last_power
        # Update raw history for graph
        # But we use SMOOTHED power for state logic to filter noise
        
        # Buffer for moving average (configurable window)
        self._ma_buffer = getattr(self, "_ma_buffer", [])
        self._ma_buffer.append(power)
        if len(self._ma_buffer) > max(1, int(self._config.smoothing_window)):
            self._ma_buffer.pop(0)
            
        avg_power = sum(self._ma_buffer) / len(self._ma_buffer)
        
        # Use smoothed power for START detection (prevents false starts from spikes)
        # Use RAW power for END detection (immediate response to power drops)
        is_active_for_start = avg_power >= self._config.min_power
        is_active_for_end = power >= self._config.min_power
        
        _LOGGER.debug(
            f"process_reading: power={power}W, avg={avg_power:.1f}W, "
            f"buffer={self._ma_buffer}, active_start={is_active_for_start}, "
            f"active_end={is_active_for_end}, state={self._state}, "
            f"min_power={self._config.min_power}W, off_delay={self._config.off_delay}s, "
            f"smoothing={self._config.smoothing_window}"
        )

        if self._state == STATE_OFF:
            if is_active_for_start:
                self._transition_to(STATE_RUNNING, timestamp)
                self._current_cycle_start = timestamp
                self._power_readings = [(timestamp, power)]
                self._last_active_time = timestamp
                self._cycle_max_power = power
                self._abrupt_drop = False
                self._last_power = power

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
            
            if is_active_for_end:
                 self._last_active_time = timestamp
                 self._low_power_start = None  # Reset low-power timer
            else:
                 # Track when low power started
                 if not self._low_power_start:
                     self._low_power_start = timestamp
                     _LOGGER.debug(f"Low power detected (raw power < {self._config.min_power}W), starting completion timer")
                     # If we had a steep drop from a high load, flag as abrupt
                     if prev_power is not None:
                         drop = prev_power - power
                         drop_ratio = drop / max(prev_power, 1.0)
                         high_load = prev_power >= (self._config.min_power * self._config.abrupt_high_load_factor)
                         if (high_load and drop_ratio >= self._config.abrupt_drop_ratio) or drop >= self._config.abrupt_drop_watts:
                             self._abrupt_drop = True
                             _LOGGER.debug(
                                 "Abrupt drop detected: prev=%.1fW, now=%.1fW, ratio=%.2f, high_load=%s",
                                 prev_power,
                                 power,
                                 drop_ratio,
                                 high_load,
                             )
                 
                 # Check if we should conclude the cycle
                 low_duration = (timestamp - self._low_power_start).total_seconds()
                 _LOGGER.debug(f"Low power: duration={low_duration:.1f}s, off_delay={self._config.off_delay}s, will_end={low_duration >= self._config.off_delay}")
                 if low_duration >= self._config.off_delay:
                     # Power has been low for the configured delay - cycle is done NATURALLY
                     _LOGGER.info(f"Ending cycle: power below {self._config.min_power}W for {low_duration:.0f}s (threshold: {self._config.off_delay}s)")
                     self._finish_cycle(timestamp, status="completed")

            # Update last power for next iteration
            self._last_power = power

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

        # Re-classify abnormal endings as interrupted (user abort/power cut)
        if status in ("completed", "force_stopped") and self._should_mark_interrupted(duration):
            reason = self._get_interruption_reason(duration)
            _LOGGER.info(f"Cycle marked as interrupted: {reason}")
            status = "interrupted"
        
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
        self._last_power = None
        self._abrupt_drop = False

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

    def _should_mark_interrupted(self, duration: float) -> bool:
        """Return True if the cycle looks abnormal/aborted."""
        # Treat extremely short runs as interruptions (likely accidental start/stop or power loss)
        if duration < float(self._config.interrupted_min_seconds):
            return True

        # Treat runs shorter than the "completion minimum" as interrupted too
        # This catches "4 minute" cycles that finished naturally but are too short to be real
        if duration < float(self._config.completion_min_seconds):
            return True

        # If we saw an abrupt power cliff from a high load, flag as interrupted
        # Use <= to catch cycles right at the boundary
        if self._abrupt_drop and duration <= (float(self._config.interrupted_min_seconds) + 90.0):
            return True

        return False

    def _get_interruption_reason(self, duration: float) -> str:
        """Return human-readable reason why cycle was marked as interrupted."""
        if duration < float(self._config.interrupted_min_seconds):
            return f"too short ({int(duration)}s < {self._config.interrupted_min_seconds}s min)"
        
        if duration < float(self._config.completion_min_seconds):
            return f"too short for completion ({int(duration)}s < {self._config.completion_min_seconds}s)"
        
        if self._abrupt_drop:
            return f"abrupt power drop detected (duration={int(duration)}s, threshold={self._config.interrupted_min_seconds + 90}s)"
        
        return "unknown reason"

    def get_power_trace(self) -> list[tuple[datetime, float]]:
        """Return a copy of the current raw power trace."""
        return list(self._power_readings)

    def get_elapsed_seconds(self) -> float:
        """Return elapsed seconds in the current cycle based on readings."""
        if not self._current_cycle_start or not self._power_readings:
            return 0.0
        return (self._power_readings[-1][0] - self._current_cycle_start).total_seconds()

    def is_waiting_low_power(self) -> bool:
        """Return True if we're in low-power waiting window to complete."""
        return self._state == STATE_RUNNING and self._low_power_start is not None

    def low_power_elapsed(self, now: datetime) -> float:
        """Seconds elapsed since entering low-power waiting (0 if not waiting)."""
        if not self._low_power_start:
            return 0.0
        return (now - self._low_power_start).total_seconds()
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
