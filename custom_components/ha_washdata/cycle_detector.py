"""Cycle detection logic for HA WashData."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, cast

from homeassistant.util import dt as dt_util

from .const import STATE_OFF, STATE_RUNNING

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
    abrupt_high_load_factor: float = 5.0
    completion_min_seconds: int = 600
    start_duration_threshold: float = 5.0
    running_dead_zone: int = 0  # Seconds after start to ignore power dips
    end_repeat_count: int = 1  # Number of times end condition must be met consecutively


class CycleDetector:
    """Detects washing machine cycles based on power usage."""

    def __init__(
        self,
        config: CycleDetectorConfig,
        on_state_change: Callable[[str, str], None],
        on_cycle_end: Callable[[dict[str, Any]], None],
        profile_matcher: Callable[[list[tuple[datetime, float]]], tuple[str | None, float, float, str | None]] | None = None,
    ) -> None:
        """Initialize the cycle detector."""
        self._config = config
        self._on_state_change = on_state_change
        self._on_cycle_end = on_cycle_end
        self._profile_matcher = profile_matcher

        self._state = STATE_OFF
        self._sub_state: str | None = None  # Additional status info (e.g. "Running (Eco Wash)")
        self._power_readings: list[tuple[datetime, float]] = []
        self._current_cycle_start: datetime | None = None
        self._last_active_time: datetime | None = None
        self._low_power_start: datetime | None = None  # Track when we entered low-power waiting
        self._cycle_max_power: float = 0.0
        self._ma_buffer: list[float] = []
        self._cycle_status: str | None = None  # Track how cycle ended
        self._last_power: float | None = None  # Track previous raw power reading
        self._abrupt_drop: bool = False  # Flag abrupt drop events
        self._potential_start_time: datetime | None = None  # For start debounce
        self._end_condition_count: int = 0  # Track consecutive end conditions met
        self._extension_count: int = 0 # Track how many times we extended due to profile match
        self._dynamic_min_duration: float | None = None  # Smart Cycle Extension: min duration to enforce
        self._matched_profile: str | None = None # Persist the detected profile name

    def reset(self) -> None:
        """Force reset the detector state to OFF."""
        self._state = STATE_OFF
        self._sub_state = None
        self._power_readings = []
        self._current_cycle_start = None
        self._last_active_time = None
        self._low_power_start = None
        self._cycle_max_power = 0.0
        self._ma_buffer = []
        self._cycle_status = None
        self._last_power = None
        self._abrupt_drop = False
        self._potential_start_time = None
        self._end_condition_count = 0
        self._extension_count = 0
        self._dynamic_min_duration = None
        self._matched_profile = None
        self._on_state_change(self._state, "Force Stopped")

    @property
    def state(self) -> str:
        """Return current state."""
        return self._state

    @property
    def sub_state(self) -> str | None:
        """Return current sub-state (phase info)."""
        return self._sub_state

    @property
    def config(self) -> CycleDetectorConfig:
        """Return current configuration."""
        return self._config

    def set_min_duration(self, seconds: float | None) -> None:
        """Set a dynamic minimum duration for the current cycle (Smart Cycle Extension)."""
        self._dynamic_min_duration = seconds
        if seconds:
            _LOGGER.info(f"Smart Cycle Extension: Enforcing minimum duration of {int(seconds)}s")
        else:
            _LOGGER.debug("Smart Cycle Extension: clear minimum duration")

    @property
    def matched_profile(self) -> str | None:
        """Return the confirmed matched profile name if any."""
        return self._matched_profile

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
                if not self._potential_start_time:
                    self._potential_start_time = timestamp
                
                elapsed = (timestamp - self._potential_start_time).total_seconds()
                if elapsed >= self._config.start_duration_threshold:
                    self._transition_to(STATE_RUNNING, timestamp)
                    self._current_cycle_start = timestamp
                    self._power_readings = [(timestamp, power)]
                    self._last_active_time = timestamp
                    self._cycle_max_power = power
                    self._abrupt_drop = False
                    self._last_power = power
                    self._potential_start_time = None
                    self._sub_state = "Starting"
                    self._extension_count = 0
                    self._matched_profile = None
                    self._dynamic_min_duration = None # Clear any previous dynamic min duration
            else:
                self._potential_start_time = None

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
            
            # Check if we're in the dead zone period (ignore power dips during startup)
            cycle_elapsed = (timestamp - self._current_cycle_start).total_seconds() if self._current_cycle_start else 0
            in_dead_zone = cycle_elapsed < self._config.running_dead_zone
            
            # Check if we should extend the cycle due to dynamic min duration (Smart Cycle Extension)
            force_extension = False
            if self._dynamic_min_duration and cycle_elapsed < self._dynamic_min_duration:
                force_extension = True
            
            if is_active_for_end or force_extension:
                 if force_extension and not is_active_for_end:
                     # Only log periodically to avoid spam
                     if int(cycle_elapsed) % 60 == 0:
                         _LOGGER.debug(f"Smart Cycle Extension active: power low but elapsed {int(cycle_elapsed)}s < min {int(self._dynamic_min_duration)}s")
                 
                 self._last_active_time = timestamp
                 self._low_power_start = None  # Reset low-power timer
                 self._end_condition_count = 0  # Reset end condition counter when power goes back up (or forced extended)
                 # Reset extension count if we recover power
                 if self._extension_count > 0:
                     self._extension_count = 0
                 
                 # Basic running state, update later if we match profile
                 if self._sub_state and "detecting" not in self._sub_state.lower():
                     self._sub_state = "Running"

            elif not in_dead_zone:  # Only check end conditions if NOT in dead zone
                # Track when low power started
                if not self._low_power_start:
                    self._low_power_start = timestamp
                    self._sub_state = "Low Power"
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
                
                # Determine effective off_delay (standard vs predictive shortened)
                effective_off_delay = self._config.off_delay
                predictive_end = False
                
                # Run profile matcher once for this low-power evaluation (if available)
                matched_profile_name: str | None = None
                matched_confidence: float = 0.0
                matched_expected_duration: float = 0.0
                matched_phase_name: str | None = None
                if self._profile_matcher:
                    matched_profile_name, matched_confidence, matched_expected_duration, matched_phase_name = self._profile_matcher(self._power_readings)
                
                # Check predictive end to potentially shorten the delay
                if (
                    matched_profile_name
                    and matched_confidence >= 0.90
                    and matched_expected_duration > 0
                    and matched_phase_name is None
                ):
                    pct = cycle_elapsed / matched_expected_duration
                    # If > 98% complete, we trust the profile heavily
                    if pct >= 0.98:
                        # Shorten delay to 30s or half of configured delay
                        effective_off_delay = min(30.0, self._config.off_delay / 2.0)
                        if low_duration >= effective_off_delay:
                            _LOGGER.info(
                                f"Predictive End: Profile '{matched_profile_name}' matched "
                                f"(conf={matched_confidence:.2f}), progress={pct*100:.1f}%. "
                                f"Threshold shortened to {effective_off_delay}s."
                            )
                            predictive_end = True
                
                # Persist matched profile if confident, so we can restore it on restart
                if matched_profile_name and matched_confidence >= 0.70:
                    self._matched_profile = matched_profile_name

                _LOGGER.debug(f"Low power: duration={low_duration:.1f}s, threshold={effective_off_delay}s, end_count={self._end_condition_count}/{self._config.end_repeat_count}")
                
                if low_duration >= effective_off_delay:
                    # Check Profile Match extension logic (only if not already predictive ended)
                    extended = False
                    if not predictive_end and matched_profile_name and matched_confidence >= 0.70 and matched_expected_duration > 0:
                        pct_complete = (cycle_elapsed / matched_expected_duration)
                        should_extend = pct_complete < 0.95 or (matched_phase_name is not None)
                        
                        if should_extend:
                            _LOGGER.info(
                                f"Profile match '{matched_profile_name}' (conf={matched_confidence:.2f}): "
                                f"elapsed {cycle_elapsed:.0f}s < expected {matched_expected_duration:.0f}s ({pct_complete*100:.0f}%). "
                                f"Phase='{matched_phase_name}'. Extending cycle (ignoring low power)."
                            )
                            if matched_phase_name:
                                self._sub_state = f"Running ({matched_phase_name})"
                            else:
                                self._sub_state = f"Running ({matched_profile_name} - {int(pct_complete*100)}%)"
                                
                            self._low_power_start = timestamp # Reset low power timer
                            self._extension_count += 1
                            extended = True
                    
                    if not extended:
                        # SIMPLIFIED REPEAT CHECK LOGIC
                        check_passes = False
                        
                        if predictive_end:
                            check_passes = True
                        else:
                            # If this is the FIRST time we hit the limit, increment and schedule next check
                            if self._end_condition_count == 0:
                                self._end_condition_count = 1
                                self._verification_next_check = timestamp.timestamp() + 15.0  # Check again in 15s
                                _LOGGER.debug(f"End condition met 1/{self._config.end_repeat_count}, verifying... (next check in 15s)")
                            else:
                                # We are in verification mode
                                now_ts = timestamp.timestamp()
                                if now_ts >= getattr(self, "_verification_next_check", 0):
                                    self._end_condition_count += 1
                                    self._verification_next_check = now_ts + 15.0
                                    _LOGGER.debug(f"End condition met {self._end_condition_count}/{self._config.end_repeat_count}, verifying...")
                            
                            if self._end_condition_count >= self._config.end_repeat_count:
                                check_passes = True
                        
                        if check_passes:
                            # Cycle is done NATURALLY
                            _LOGGER.info(f"Ending cycle: power below {self._config.min_power}W for {low_duration:.0f}s (threshold: {effective_off_delay:.0f}s), end count: {self._end_condition_count}")
                            self._finish_cycle(timestamp, status="completed")
                        else:
                            pass
            else:
                # In dead zone - log but don't start end detection
                if not is_active_for_end:
                    _LOGGER.debug(f"Low power during dead zone ({cycle_elapsed:.0f}s < {self._config.running_dead_zone}s), ignoring")

            # Update last power for next iteration
            self._last_power = power

    def _transition_to(self, new_state: str, timestamp: datetime) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        if new_state == STATE_OFF:
            self._sub_state = None
            self._dynamic_min_duration = None # Clear dynamic min duration on cycle end
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
        
        cycle_data: dict[str, Any] = {
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
        self._extension_count = 0
        self._sub_state = None
        self._dynamic_min_duration = None

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

    def user_stop(self) -> None:
        """Manually stop the cycle (user request), treating it as a natural completion."""
        if self._state != STATE_RUNNING:
            return
        # We treat this as "completed" so it goes through standard saving/filtering logic
        _LOGGER.info("User requested manual stop - finishing cycle")
        self._finish_cycle(dt_util.now(), status="completed")

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
    def get_state_snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current state."""
        return {
            "state": self._state,
            "sub_state": self._sub_state,
            "current_cycle_start": self._current_cycle_start.isoformat() if self._current_cycle_start else None,
            "last_active_time": self._last_active_time.isoformat() if self._last_active_time else None,
            "low_power_start": self._low_power_start.isoformat() if self._low_power_start else None,
            "cycle_max_power": self._cycle_max_power,
            "power_readings": [(t.isoformat(), p) for t, p in self._power_readings],
            "ma_buffer": getattr(self, "_ma_buffer", []),
            "end_condition_count": self._end_condition_count,
            "extension_count": self._extension_count,
            "dynamic_min_duration": self._dynamic_min_duration,
            "matched_profile": self._matched_profile,
        }

    def restore_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        try:
            self._state = snapshot.get("state", STATE_OFF)
            self._sub_state = snapshot.get("sub_state")
            
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
            if isinstance(readings, list):
                for item in cast(list[Any], readings):
                    if not isinstance(item, (list, tuple)):
                        continue
                    try:
                        t, p = cast(tuple[Any, Any], item)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(t, str):
                        continue
                    parsed = dt_util.parse_datetime(t)
                    if parsed:
                        # Ensure timezone-aware
                        try:
                            self._power_readings.append((dt_util.as_local(parsed), float(p)))
                        except (TypeError, ValueError):
                            continue
            

            self._ma_buffer = snapshot.get("ma_buffer", [])
            
            self._matched_profile = snapshot.get("matched_profile")
            
            # Restore end condition counter
            self._end_condition_count = snapshot.get("end_condition_count", 0)
            self._extension_count = snapshot.get("extension_count", 0)
            
            # Restore dynamic min duration
            self._dynamic_min_duration = snapshot.get("dynamic_min_duration")
            
            _LOGGER.info(f"Restored CycleDetector state: {self._state}, {len(self._power_readings)} readings")
            
        except Exception as e:
            _LOGGER.error(f"Failed to restore CycleDetector state: {e}")
            self._state = STATE_OFF
            self._power_readings = []
