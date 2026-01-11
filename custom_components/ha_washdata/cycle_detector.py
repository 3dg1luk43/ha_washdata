"""Cycle detection logic for HA WashData."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
import numpy as np

from homeassistant.util import dt as dt_util

from .const import (
    STATE_OFF,
    STATE_STARTING,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_ENDING,
    STATE_UNKNOWN,
    DEVICE_TYPE_WASHING_MACHINE,
)
from .signal_processing import integrate_wh

_LOGGER = logging.getLogger(__name__)


@dataclass
class CycleDetectorConfig:
    """Configuration for cycle detection."""

    min_power: float
    off_delay: int
    device_type: str = DEVICE_TYPE_WASHING_MACHINE
    smoothing_window: int = 5
    interrupted_min_seconds: int = 150
    abrupt_drop_watts: float = 500.0
    abrupt_drop_ratio: float = 0.6
    abrupt_high_load_factor: float = 5.0
    completion_min_seconds: int = 600
    start_duration_threshold: float = 5.0
    start_energy_threshold: float = 0.005  # 5 Wh default
    end_energy_threshold: float = 0.05  # 50 Wh threshold for "still active"
    running_dead_zone: int = 0
    end_repeat_count: int = 1
    min_off_gap: int = 60
    start_threshold_w: float = 2.0
    stop_threshold_w: float = 2.0


@dataclass
class CycleDetectorState:
    """Internal state storage for save/restore."""

    state: str = STATE_OFF
    sub_state: str | None = None
    accumulated_energy_wh: float = 0.0
    # Add other fields as needed


class CycleDetector:
    """Detects washing machine cycles based on power usage.

    Implements a robust state machine:
    OFF -> STARTING -> RUNNING <-> PAUSED -> ENDING -> OFF
    """

    def __init__(
        self,
        config: CycleDetectorConfig,
        on_state_change: Callable[[str, str], None],
        on_cycle_end: Callable[[dict[str, Any]], None],
        profile_matcher: (
            Callable[
                [list[tuple[datetime, float]]],
                tuple[str | None, float, float, str | None],
            ]
            | None
        ) = None,
    ) -> None:
        """Initialize the cycle detector."""
        self._config = config
        self._on_state_change = on_state_change
        self._on_cycle_end = on_cycle_end
        self._profile_matcher = profile_matcher

        # State
        self._state = STATE_OFF
        self._sub_state: str | None = None

        # Data
        self._power_readings: list[tuple[datetime, float]] = []  # (time, raw_power)
        self._current_cycle_start: datetime | None = None
        self._last_active_time: datetime | None = None
        self._cycle_max_power: float = 0.0

        # Accumulators (dt-aware)
        self._energy_since_idle_wh: float = 0.0
        self._time_above_threshold: float = 0.0
        self._time_below_threshold: float = 0.0
        self._last_process_time: datetime | None = None

        # New State Machine trackers
        self._state_enter_time: datetime | None = None
        self._matched_profile: str | None = None

        self._abrupt_drop: bool = False
        self._last_power: float | None = None

        # Smoothing buffer
        self._ma_buffer: list[float] = []

        # Adaptive Sampling Tracker
        self._recent_dts: list[float] = []  # Track last 20 dt values
        self._p95_dt: float = 1.0  # Default assumption

        # Profile Matching Tracker
        self._last_match_time: datetime | None = None
        self._last_state_change: datetime | None = None
        self._expected_duration: float = 0.0
        self._match_interval_s: float = 60.0  # Default match interval

    @property
    def _dynamic_pause_threshold(self) -> float:
        """Calculate dynamic pause threshold based on sampling cadence."""
        # User requirement: T_pause >= 3 * p95_update_interval
        # Default 15s or 3 * p95
        return max(15.0, 3.0 * self._p95_dt)

    @property
    def _dynamic_end_threshold(self) -> float:
        """Calculate dynamic end candidate threshold."""
        # Default 30s or 3 * p95 + buffer
        # Let's ensure it's strictly greater than pause threshold to define state progression
        base = max(30.0, 3.0 * self._p95_dt)
        # Ensure at least 15s gap from pause?
        return max(base, self._dynamic_pause_threshold + 15.0)

    def _update_cadence(self, dt: float) -> None:
        """Update rolling cadence statistics."""
        if dt <= 0.1:
            return
        self._recent_dts.append(dt)
        if len(self._recent_dts) > 20:
            self._recent_dts.pop(0)

        # Calculate p95 if enough samples
        if len(self._recent_dts) >= 5:
            self._p95_dt = float(np.percentile(self._recent_dts, 95))
        else:
            self._p95_dt = max(dt, 1.0)

    def _try_profile_match(self, timestamp: datetime, force: bool = False) -> None:
        """Attempt to invoke the profile matcher if conditions are met.

        Args:
            timestamp: Current timestamp.
            force: If True, run match immediately regardless of interval.
        """
        if not self._profile_matcher:
            return
        if not self._power_readings:
            return

        # Rate limiting
        if not force and self._last_match_time:
            elapsed = (timestamp - self._last_match_time).total_seconds()
            if elapsed < self._match_interval_s:
                return

        self._last_match_time = timestamp

        # Call the matcher
        try:
            result = self._profile_matcher(self._power_readings)
            # Unpack 5 elements (or 4 for backward compatibility if needed, but wrapper is updated)
            if result:
                # wrapper returns (name, confidence, duration, phase, is_mismatch)
                if len(result) >= 5:
                    (
                        match_name,
                        _,
                        expected_duration,
                        phase_name,
                        is_mismatch,
                    ) = result[:5]
                else:
                    # Fallback for old signature
                    (match_name, _, expected_duration, phase_name) = result[:4]
                    is_mismatch = False

                if is_mismatch and self._matched_profile:
                    # Confident non-match - revert to detecting if previously matched
                    self._matched_profile = None
                    
                elif match_name:
                    self._matched_profile = match_name
                    # Sub-state can be set from phase_name if available
                    if phase_name:
                        self._sub_state = phase_name
                    # Wrapper provides it
                    self._expected_duration = expected_duration

        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Profile match failed: %s", e)

    def reset(self) -> None:
        """Force reset the detector state to OFF."""
        self._transition_to(STATE_OFF, dt_util.now())
        self._power_readings = []
        self._current_cycle_start = None
        self._last_active_time = None
        self._cycle_max_power = 0.0
        self._ma_buffer = []
        self._energy_since_idle_wh = 0.0
        self._time_above_threshold = 0.0
        self._time_below_threshold = 0.0
        self._last_match_time = None
        self._matched_profile = None
        self._on_state_change(self._state, "Force Stopped")

    @property
    def state(self) -> str:
        """Return current state."""
        return self._state

    @property
    def sub_state(self) -> str | None:
        """Return current sub-state."""
        return self._sub_state

    @property
    def config(self) -> CycleDetectorConfig:
        """Return current configuration."""
        return self._config

    @property
    def matched_profile(self) -> str | None:
        """Return the name of the matched profile, if any."""
        return self._matched_profile

    @property
    def current_cycle_start(self) -> datetime | None:
        """Return the start timestamp of the current cycle."""
        return self._current_cycle_start

    def process_reading(self, power: float, timestamp: datetime) -> None:
        """Process a new power reading using robust dt-aware logic."""

        # Calculate dt
        dt = 0.0
        if self._last_process_time:
            dt = (timestamp - self._last_process_time).total_seconds()

        # Sanity check for negative dt
        if dt < 0:
            self._last_process_time = timestamp
            return

        self._update_cadence(dt)
        self._last_process_time = timestamp

        # 1. Smoothing (Legacy buffer for debug/display, logic uses raw + time accumulators)
        self._ma_buffer.append(power)
        if len(self._ma_buffer) > self._config.smoothing_window:
            self._ma_buffer.pop(0)

        # 2. Accumulators Update
        # Hysteresis Logic
        if self._state in (STATE_OFF, STATE_STARTING, STATE_UNKNOWN):
            threshold = self._config.start_threshold_w
        else:
            threshold = self._config.stop_threshold_w

        is_high = power >= threshold

        if is_high:
            self._time_above_threshold += dt
            self._time_below_threshold = 0.0
            # Energy integration (trapezoidal approx for this single step)
            # prev_p = self._last_power if self._last_power is not None else power
            # step_wh = ((power + prev_p) / 2.0) * (dt / 3600.0)
            # Simplified: just P * dt for short steps is fine,
            # or call integrate_wh on buffer if needed.
            # Let's use simple rect/trapz here for running sum
            step_wh = power * (dt / 3600.0)
            self._energy_since_idle_wh += step_wh
            self._last_active_time = timestamp
        else:
            self._time_below_threshold += dt
            self._time_above_threshold = 0.0

        self._last_power = power

        # 3. State Machine

        if self._state == STATE_OFF:
            if is_high:
                # Transition to STARTING
                self._transition_to(STATE_STARTING, timestamp)
                self._current_cycle_start = timestamp
                self._power_readings = [(timestamp, power)]
                self._energy_since_idle_wh = power * (dt / 3600.0) if dt > 0 else 0.0
                self._cycle_max_power = power
                self._abrupt_drop = False

        elif self._state == STATE_STARTING:
            self._power_readings.append((timestamp, power))
            self._cycle_max_power = max(self._cycle_max_power, power)

            if self._time_above_threshold >= self._config.start_duration_threshold:
                if self._energy_since_idle_wh >= self._config.start_energy_threshold:
                    self._transition_to(STATE_RUNNING, timestamp)

            # Abort if power drops below threshold before confirmation
            if not is_high and self._time_below_threshold > 1.0:  # 1s grace?
                # False start
                _LOGGER.debug(
                    "False start detected: power dropped after %.2fs",
                    self._time_above_threshold,
                )
                self._transition_to(STATE_OFF, timestamp)

        elif self._state == STATE_RUNNING:
            self._power_readings.append((timestamp, power))
            self._cycle_max_power = max(self._cycle_max_power, power)

            # Use dynamic threshold
            thresh = self._dynamic_pause_threshold
            if self._time_below_threshold >= thresh:
                self._try_profile_match(timestamp, force=True)  # Refine match on pause
                self._transition_to(STATE_PAUSED, timestamp)

            # Periodic profile matching
            self._try_profile_match(timestamp)

            # Max duration safety
            if (
                self._current_cycle_start
                and (timestamp - self._current_cycle_start).total_seconds() > 28800
            ):  # 8h safety
                self._finish_cycle(timestamp, status="force_stopped")

        elif self._state == STATE_PAUSED:
            self._power_readings.append((timestamp, power))

            if is_high:
                # Resume to RUNNING
                self._transition_to(STATE_RUNNING, timestamp)
            else:
                # Periodic profile matching during pause
                self._try_profile_match(timestamp)

                thresh = self._dynamic_end_threshold
                if self._time_below_threshold >= thresh:
                    self._transition_to(STATE_ENDING, timestamp)

        elif self._state == STATE_ENDING:
            self._power_readings.append((timestamp, power))

            if is_high:
                # Resume -> RUNNING
                self._transition_to(STATE_RUNNING, timestamp)
            else:
                # Periodic profile matching during ending
                self._try_profile_match(timestamp)

                # Rule: To separate cycles, we must wait at least min_off_gap.
                effective_off_delay = max(
                    self._config.off_delay, self._config.min_off_gap
                )

                if self._time_below_threshold >= effective_off_delay:

                    recent_window = [
                        r
                        for r in self._power_readings
                        if (timestamp - r[0]).total_seconds() <= self._config.off_delay
                    ]
                    if not recent_window:
                        self._finish_cycle(timestamp, status="completed")
                        return

                    # Compute energy in recent window
                    recent_ts = np.array([r[0].timestamp() for r in recent_window])
                    recent_p = np.array([r[1] for r in recent_window])
                    recent_e = integrate_wh(recent_ts, recent_p)

                    if recent_e <= self.config.end_energy_threshold:
                        self._finish_cycle(timestamp, status="completed")
                    else:

                        _LOGGER.debug(
                            "Cycle ending prevented by energy gate: %.4fWh > %.4fWh",
                            recent_e,
                            self._config.end_energy_threshold,
                        )

    def _transition_to(self, new_state: str, timestamp: datetime) -> None:
        """Handle state transitions."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._state_enter_time = timestamp
        self._sub_state = new_state.capitalize()  # Default substate

        # Reset specific accumulators on transitions?
        if new_state == STATE_OFF:
            self._energy_since_idle_wh = 0.0

        _LOGGER.debug("Transition: %s -> %s at %s", old_state, new_state, timestamp)
        self._on_state_change(old_state, new_state)

    def _finish_cycle(self, timestamp: datetime, status: str = "completed") -> None:
        """Finalize cycle."""

        # Capture data before reset
        end_time = self._last_active_time or timestamp
        if not self._current_cycle_start:
            self.reset()
            return

        duration = (end_time - self._current_cycle_start).total_seconds()

        # "Interrupted" logic (short cycle etc)
        if duration < self._config.interrupted_min_seconds:
            status = "interrupted"
        elif duration < self._config.completion_min_seconds:
            status = "interrupted"
        elif self._abrupt_drop and duration < (
            self._config.interrupted_min_seconds + 90
        ):
            status = "interrupted"

        cycle_data = {
            "start_time": self._current_cycle_start.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": duration,
            "max_power": self._cycle_max_power,
            "status": status,
            "power_data": [(t.isoformat(), p) for t, p in self._power_readings],
        }

        _LOGGER.info("Cycle Finished: %s, %.1f min", status, duration / 60)
        self._on_cycle_end(cycle_data)
        self.reset()

    # Stub methods for compatibility or simpler logic
    def force_end(self, timestamp: datetime) -> None:
        """Force the cycle to end immediately."""
        if self._state != STATE_OFF:
            self._finish_cycle(timestamp, status="force_stopped")

    def user_stop(self) -> None:
        """Handle user-initiated stop."""
        if self._state != STATE_OFF:
            self._finish_cycle(dt_util.now(), status="completed")

    def get_power_trace(self) -> list[tuple[datetime, float]]:
        """Return the current power trace."""
        return list(self._power_readings)

    def get_state_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the current state for persistence."""
        return {
            "state": self._state,
            "sub_state": self._sub_state,
            "current_cycle_start": (
                self._current_cycle_start.isoformat()
                if self._current_cycle_start
                else None
            ),
            "power_readings": [(t.isoformat(), p) for t, p in self._power_readings],
            "accumulated_energy_wh": self._energy_since_idle_wh,
            "time_above": self._time_above_threshold,
            "time_below": self._time_below_threshold,
            "cycle_max_power": self._cycle_max_power,
            "last_active_time": (
                self._last_active_time.isoformat() if self._last_active_time else None
            ),
            "matched_profile": self._matched_profile,
        }

    def get_elapsed_seconds(self) -> float:
        """Return seconds elapsed in current cycle."""
        if self._current_cycle_start:
            return (dt_util.now() - self._current_cycle_start).total_seconds()
        return 0.0

    def is_waiting_low_power(self) -> bool:
        """Return True if we are pending end/pause due to low power."""
        return (
            self._state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
            and self._time_below_threshold > 0
        )

    def low_power_elapsed(self, now: datetime) -> float:
        """Return duration of current low power spell including time since last process."""
        if self._time_below_threshold > 0 and self._last_process_time:
            # Add time since last processing
            return (
                self._time_below_threshold
                + (now - self._last_process_time).total_seconds()
            )
        return self._time_below_threshold

    def restore_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        try:
            self._state = snapshot.get("state", STATE_OFF)
            self._sub_state = snapshot.get("sub_state")
            self._energy_since_idle_wh = snapshot.get("accumulated_energy_wh", 0.0)
            self._time_above_threshold = snapshot.get("time_above", 0.0)
            self._time_below_threshold = snapshot.get("time_below", 0.0)
            self._cycle_max_power = snapshot.get("cycle_max_power", 0.0)
            self._matched_profile = snapshot.get("matched_profile")

            start = snapshot.get("current_cycle_start")
            if start:
                try:
                    self._current_cycle_start = dt_util.parse_datetime(start)
                except Exception:  # pylint: disable=broad-exception-caught
                    self._current_cycle_start = None

            readings = snapshot.get("power_readings", [])
            self._power_readings = []
            for r in readings:
                if isinstance(r, (list, tuple)) and len(r) == 2:
                    t = dt_util.parse_datetime(r[0])
                    if t:
                        self._power_readings.append((t, float(r[1])))

            # Restore last active
            last_active = snapshot.get("last_active_time")
            if last_active:
                self._last_active_time = dt_util.parse_datetime(last_active)
            else:
                self._last_active_time = self._current_cycle_start

        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.error("Failed restore: %s", e)
            self.reset()
