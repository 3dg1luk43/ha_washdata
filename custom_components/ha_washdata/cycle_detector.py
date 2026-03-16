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
    STATE_FINISHED,
    STATE_INTERRUPTED,
    STATE_FORCE_STOPPED,
    STATE_UNKNOWN,
    DEVICE_TYPE_WASHING_MACHINE,
    DEFAULT_MAX_DEFERRAL_SECONDS,
    DEFAULT_DEFER_FINISH_CONFIDENCE,
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
    min_duration_ratio: float = 0.8  # Default deferred finish ratio
    match_interval: int = 300  # Default profile match interval
    profile_duration_tolerance: float = 0.25  # Default tolerance (±25%)


@dataclass
class CycleDetectorState:
    """Internal state storage for save/restore."""

    state: str = STATE_OFF
    sub_state: str | None = None
    accumulated_energy_wh: float = 0.0
    # Add other fields as needed


def trim_zero_readings(
    readings: list[tuple[datetime, float]],
    threshold: float = 0.5,
    trim_start: bool = True,
    trim_end: bool = True,
) -> list[tuple[datetime, float]]:
    """Trim continuous zero/near-zero readings from start and end of cycle.

    Args:
        readings: List of (timestamp, power) tuples
        threshold: Power values below this are considered "zero"
        trim_start: Whether to trim zeros from the beginning
        trim_end: Whether to trim zeros from the end

    Returns:
        Trimmed list
    """
    if not readings:
        return readings

    start_idx = 0
    if trim_start:
        for i, (_, power) in enumerate(readings):
            if power > threshold:
                start_idx = i
                break
        else:
            # All readings are zero - return single point if list not empty
            return readings[:1] if readings else []

    end_idx = len(readings) - 1
    if trim_end:
        # Find last non-zero reading
        found_end = False
        for i in range(len(readings) - 1, -1, -1):
            if readings[i][1] > threshold:
                end_idx = i
                found_end = True
                break

        if not found_end and trim_start:
            # If all zeros and trim_start was checked, it would return early.
            # But if safety fallback needed:
            end_idx = start_idx
        elif not found_end and not trim_start:
             # Trimming end but not start, and all zeros?
             # Keep first point
            end_idx = 0

    # Return trimmed slice (inclusive of end)
    return readings[start_idx : end_idx + 1]


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
        self._ignore_power_until_idle: bool = False

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
        self._verified_pause: bool = False

        self._abrupt_drop: bool = False
        self._last_power: float | None = None
        self._time_in_state: float = 0.0

        # Smoothing buffer
        self._ma_buffer: list[float] = []

        # Adaptive Sampling Tracker
        self._recent_dts: list[float] = []  # Track last 20 dt values
        self._p95_dt: float = 1.0  # Default assumption

        # Profile Matching Tracker
        self._last_match_time: datetime | None = None
        self._expected_duration: float = 0.0
        self._last_match_confidence: float = 0.0
        self._end_spike_seen: bool = False

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
        # Ensure end threshold is at least 15s greater than pause threshold
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
            if elapsed < self._config.match_interval:
                return

        self._last_match_time = timestamp

        # Call the matcher
        try:
            result = self._profile_matcher(self._power_readings)
            # If synchronous result returned, process it.
            # If None returned (async offload), the matcher is responsible for
            # calling update_match later.
            if result:
                self.update_match(result)

        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Profile match failed: %s", e)

    def update_match(self, result: tuple | Any) -> None:
        """Process a match result (synchronously).

        Can be called by the matcher callback directly or asynchronously.
        """
        # Unpack 5 elements (or 4 for backward compatibility if needed, but wrapper is updated)
        # wrapper returns (name, confidence, duration, phase, is_mismatch)
        # Or MatchResult object if refactored, but currently wrapper returns tuple.

        is_match_mismatch = False
        match_name = None
        phase_name = None

        if isinstance(result, (list, tuple)):
            if len(result) >= 5:
                (
                    match_name,
                    confidence,
                    expected_duration,
                    phase_name,
                    is_match_mismatch,
                ) = result[:5]
            else:
                # Fallback for old signature
                (match_name, confidence, expected_duration, phase_name) = result[:4]
                is_match_mismatch = False

            # Store confidence for Smart Termination checks
            self._last_match_confidence = confidence or 0.0
        else:
            # Assume MatchResult object or similar (future proofing)
            # But for now wrapper returns tuple
            return

        if is_match_mismatch and self._matched_profile:
            # Confident non-match - revert to detecting if previously matched
            self._matched_profile = None

        elif match_name:
            self._matched_profile = match_name
            # Sub-state can be set from phase_name if available
            if phase_name:
                self._sub_state = phase_name
            # Wrapper provides it
            self._expected_duration = expected_duration

    def set_verified_pause(self, verified: bool) -> None:
        """Set or clear the verified pause flag."""
        self._verified_pause = verified

    def reset(self, target_state: str = STATE_OFF) -> None:
        """Force reset the detector state to target state."""
        self._transition_to(target_state, dt_util.now())
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
        self._ignore_power_until_idle = False  # Reset lockout

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

    @property
    def samples_recorded(self) -> int:
        """Return the number of power samples recorded in current cycle."""
        return len(self._power_readings)

    @property
    def expected_duration_seconds(self) -> float:
        """Return the expected duration of the current cycle in seconds."""
        return self._expected_duration

    def process_reading(self, power: float, timestamp: datetime) -> None:
        """Process a new power reading using robust dt-aware logic."""

        # Manual Stop Lockout:
        # If user forced a stop, ignore high power readings until machine goes idle.
        if self._ignore_power_until_idle:
            if power < self._config.start_threshold_w:
                self._ignore_power_until_idle = False
                _LOGGER.debug(
                    "Power dropped below start threshold. Manual stop lockout cleared."
                )
            else:
                # Still high after manual stop - ignore reading
                return

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

        self._time_in_state += dt

        self._last_power = power

        # 3. State Machine

        if self._state in (STATE_OFF, STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED):
            if is_high:
                # Transition to STARTING
                self._transition_to(STATE_STARTING, timestamp)
                self._current_cycle_start = timestamp
                self._power_readings = [(timestamp, power)]
                self._energy_since_idle_wh = power * (dt / 3600.0) if dt > 0 else 0.0
                self._cycle_max_power = power
                self._abrupt_drop = False
            elif self._state != STATE_OFF:
                # Auto-expire terminal states after 30 minutes
                if self._state_enter_time and (timestamp - self._state_enter_time).total_seconds() > 1800:
                    self._transition_to(STATE_OFF, timestamp)

        elif self._state == STATE_STARTING:
            self._power_readings.append((timestamp, power))
            self._cycle_max_power = max(self._cycle_max_power, power)

            if self._time_above_threshold >= self._config.start_duration_threshold:
                if self._energy_since_idle_wh >= self._config.start_energy_threshold:
                    self._transition_to(STATE_RUNNING, timestamp)

            # Abort if power drops below threshold before confirmation
            if not is_high and self._time_below_threshold > 1.0:  # 1s grace period
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
                # End spike detected! Mark it
                self._end_spike_seen = True
                _LOGGER.debug("End spike detected (power high in ENDING state)")
                
                # Check if we're past expected duration - if so, DON'T resume to RUNNING
                # This prevents the cycle from bouncing forever on pump-out spikes
                start_time = self._current_cycle_start or timestamp
                current_duration = (timestamp - start_time).total_seconds()
                
                # Sanity check: if expected_duration is unreasonable (>6 hours), use fallback
                max_reasonable = 21600.0  # 6 hours
                effective_expected = self._expected_duration
                
                if effective_expected <= 0 or effective_expected > max_reasonable:
                    # Fallback: use current duration + buffer if we've run > 3 hours
                    # (Assumes any cycle over 3 hours running is near completion when in ENDING)
                    if current_duration > 10800:  # 3 hours
                        effective_expected = current_duration * 0.99  # Always past threshold
                        _LOGGER.debug(
                            "End spike check using fallback: expected_duration=%ds is unreasonable, "
                            "using current_duration=%ds as reference",
                            int(self._expected_duration), int(current_duration)
                        )
                
                past_expected = (
                    effective_expected > 0 
                    and current_duration >= (effective_expected * 0.98)
                )
                
                if past_expected:
                    _LOGGER.debug(
                        "End spike ignored for state transition (past expected duration %.0fs/%.0fs)",
                        current_duration, effective_expected
                    )
                    # Stay in ENDING, the spike is recorded but doesn't resume cycle
                else:
                    # Resume -> RUNNING (spike is genuine mid-cycle activity)
                    self._transition_to(STATE_RUNNING, timestamp)
            else:
                # Periodic profile matching during ending
                self._try_profile_match(timestamp)

                # --- SMART TERMINATION CHECK ---
                # If we have a confident profile match and duration meets expectations,
                # we terminate early (after appropriate debounce), ignoring long arbitrary timeouts.
                if self._matched_profile:
                    start_time = self._current_cycle_start or timestamp
                    current_duration = (timestamp - start_time).total_seconds()

                    # --- ROBUSTNESS UPGRADE ---
                    # 1. Require higher duration ratio for Smart path
                    # 2. Require debounce to be measured FROM entry into ENDING state

                    if self._config.device_type == "dishwasher":
                        smart_ratio = (
                            0.99  # Very conservative for dishwashers to catch end spikes
                        )
                    else:
                        smart_ratio = 0.98

                    is_confident_match = (
                        getattr(self, "_last_match_confidence", 0.0) >= 0.4
                    )

                    if (
                        current_duration >= (self._expected_duration * smart_ratio)
                        and is_confident_match
                    ):
                        # Dynamic confirmation window
                        if self._config.device_type == "dishwasher":
                            smart_debounce = max(300.0, self._config.off_delay * 0.25)
                        else:
                            smart_debounce = 120.0

                        if self._time_in_state >= smart_debounce:
                            # --- END SPIKE WAIT PERIOD (Dishwashers) ---
                            # If we are a dishwasher and haven't seen a high-power spike since entering ENDING,
                            # wait up to 5 extra minutes past expected_duration for the end spike.
                            end_spike_wait = 300.0  # 5 minutes
                            end_spike_seen = getattr(self, "_end_spike_seen", False)
                            past_wait_period = current_duration >= (
                                self._expected_duration + end_spike_wait
                            )

                            if (
                                self._config.device_type == "dishwasher"
                                and not end_spike_seen
                                and not past_wait_period
                            ):
                                _LOGGER.debug(
                                    "Waiting for end spike (duration %.0fs, expected %.0fs + "
                                    "%.0fs wait)",
                                    current_duration,
                                    self._expected_duration,
                                    end_spike_wait,
                                )
                                return  # Don't finish yet, wait for spike or timeout

                            _LOGGER.info(
                                "Smart Termination: Profile '%s' match confirmed (duration %.0fs, "
                                "conf %.2f, spike_seen=%s), ending.",
                                self._matched_profile,
                                current_duration,
                                getattr(self, "_last_match_confidence", 0.0),
                                end_spike_seen,
                            )
                            # Keep tail when smart terminating (matches profile duration)
                            self._finish_cycle(
                                timestamp,
                                status="completed",
                                termination_reason="smart",
                                keep_tail=True,
                            )
                            return

                # --- FALLBACK TIMEOUT CHECK ---
                # Rule: To separate cycles, we must wait at least min_off_gap.
                effective_off_delay = max(self._config.off_delay, self._config.min_off_gap)

                if self._time_below_threshold >= effective_off_delay:

                    recent_window = [
                        r
                        for r in self._power_readings
                        if (timestamp - r[0]).total_seconds() <= self._config.off_delay
                    ]

                    if not recent_window:
                        # Check deferred finish for matched profiles
                        start_time = self._current_cycle_start or timestamp
                        current_duration = (timestamp - start_time).total_seconds()

                        if self._should_defer_finish(current_duration):
                            return

                        self._finish_cycle(timestamp, status="completed")
                        return

                    # Compute energy in recent window
                    recent_ts = np.array([r[0].timestamp() for r in recent_window])
                    recent_p = np.array([r[1] for r in recent_window])
                    recent_e = integrate_wh(recent_ts, recent_p)

                    if recent_e <= self.config.end_energy_threshold:
                        start_time = self._current_cycle_start or timestamp
                        current_duration = (timestamp - start_time).total_seconds()

                        if self._should_defer_finish(current_duration):
                            return

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
        self._time_in_state = 0.0
        self._sub_state = new_state.capitalize()  # Default substate

        # Reset energy accumulator on transition to OFF
        if new_state == STATE_OFF:
            self._energy_since_idle_wh = 0.0

        # Reset end spike tracker when entering ENDING state
        if new_state == STATE_ENDING:
            self._end_spike_seen = False

        _LOGGER.debug("Transition: %s -> %s at %s", old_state, new_state, timestamp)
        self._on_state_change(old_state, new_state)

    def should_defer_for_profile(self) -> bool:
        """Check if we should defer termination for profile matching (public)."""
        start_time = self._current_cycle_start
        if not self._matched_profile or self._expected_duration <= 0 or not start_time:
            return False

        current_duration = (dt_util.now() - start_time).total_seconds()
        return self._should_defer_finish(current_duration)

    def _should_defer_finish(self, duration: float) -> bool:
        """Check if we should defer termination based on expected duration."""
        # Check explicit verified pause override from manager
        if getattr(self, "_verified_pause", False):
            _LOGGER.debug("Deferring cycle finish: Verified pause active")
            return True

        if not self._matched_profile or self._expected_duration <= 0:
            return False

        # Safety: Don't defer forever
        if duration > (self._expected_duration + DEFAULT_MAX_DEFERRAL_SECONDS):
            _LOGGER.warning(
                "Deferral limit exceeded (%.0fs > expected %.0f + %s), allowing finish",
                duration,
                self._expected_duration,
                DEFAULT_MAX_DEFERRAL_SECONDS,
            )
            return False

        # If matched profile, enforce min duration ratio
        ratio = self._config.min_duration_ratio

        # --- STRICTER DEFERRAL ---
        # If we are NOT in a verified pause, but power has been low for a long time (ENDING state),
        # we only defer if we are VERY confident this profile is correct.
        # This prevents hanging on too-long profiles that matched early but are now diverging.
        if self._last_match_confidence < DEFAULT_DEFER_FINISH_CONFIDENCE:
            _LOGGER.debug(
                "Not deferring finish: confidence %.2f too low for unverified pause (profile: %s)",
                self._last_match_confidence,
                self._matched_profile,
            )
            return False

        # Also use profile tolerance to handle variable cycle lengths (e.g. long drying)
        # Allow deferral up to Expected * (1 + tolerance)
        upper_threshold = self._expected_duration * (
            1.0 + self._config.profile_duration_tolerance
        )

        # Primary check: Is duration significantly below expectation?
        if duration < (self._expected_duration * ratio):
            _LOGGER.debug(
                "Deferring cycle finish: duration %.0fs < %.0f%% of expected %.0fs (profile: %s, confidence %.2f)",
                duration,
                ratio * 100,
                self._expected_duration,
                self._matched_profile,
                self._last_match_confidence,
            )
            return True

        # Secondary check: If within valid completion window (ratio to tolerance), allow finish.
        if duration <= upper_threshold:
            return False

        # Tertiary check: If duration exceeded max tolerance, allow finish (failsafe).
        return False

    def _finish_cycle(
        self,
        timestamp: datetime,
        status: str = "completed",
        termination_reason: str = "timeout",
        keep_tail: bool = False,
    ) -> None:
        """Finalize cycle.

        Args:
            timestamp: Time of completion
            status: Cycle status string
            termination_reason: Reason for termination
            keep_tail: If True, use current timestamp as end time and preserve
                       trailing zero readings (e.g. Smart Termination).
                       If False (default), snap back to last active time and trim
                       trailing zeros (e.g. Timeout).
        """

        # Capture data before reset
        if keep_tail:
            end_time = timestamp
        else:
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

        # Trim leading/trailing zero readings for cleaner data
        # If we keep tail, we explicitly do NOT trim end zeros
        trimmed_readings = trim_zero_readings(
            self._power_readings,
            threshold=self._config.stop_threshold_w,
            trim_end=not keep_tail,
        )

        # Ensure power_data covers the full duration until end_time
        # (especially important for manual recordings or drying phases with no sensor updates)
        final_readings = list(trimmed_readings)
        if final_readings:
            last_t, last_p = final_readings[-1]
            if last_t < end_time:
                final_readings.append((end_time, last_p))

        cycle_data = {
            "start_time": self._current_cycle_start.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": duration,
            "max_power": self._cycle_max_power,
            "status": status,
            "termination_reason": termination_reason,
            "power_data": [(t.isoformat(), p) for t, p in final_readings],
        }

        _LOGGER.info("Cycle Finished: %s, %.1f min", status, duration / 60)
        self._on_cycle_end(cycle_data)
        
        target = STATE_FINISHED
        if status == "interrupted":
            target = STATE_INTERRUPTED
        elif status == "force_stopped":
            target = STATE_FORCE_STOPPED
            
        self.reset(target_state=target)

    # Stub methods for compatibility or simpler logic
    def force_end(self, timestamp: datetime) -> None:
        """Force the cycle to end immediately."""
        if self._state != STATE_OFF:
            self._finish_cycle(
                timestamp,
                status="force_stopped",
                termination_reason="force_stopped",
                keep_tail=False,  # Force stop usually implies snap back to reality
            )
            self._ignore_power_until_idle = False

    def user_stop(self) -> None:
        """Handle user-initiated stop."""
        if self._state != STATE_OFF:
            self._finish_cycle(
                dt_util.now(),
                status="completed",
                termination_reason="user",
                keep_tail=True,  # User implies "Done Now"
            )
            # Prevent immediate restart if power is still high
            self._ignore_power_until_idle = True


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
            "expected_duration": self._expected_duration,
            "matched_profile": self._matched_profile,
            "state_enter_time": (
                self._state_enter_time.isoformat() if self._state_enter_time else None
            ),
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
            self._expected_duration = snapshot.get("expected_duration", 0.0)

            # Restore state enter time
            enter_time = snapshot.get("state_enter_time")
            if enter_time:
                try:
                    self._state_enter_time = dt_util.parse_datetime(enter_time)
                except Exception: # pylint: disable=broad-exception-caught
                     _LOGGER.warning("Failed to parse state enter time")

            start = snapshot.get("current_cycle_start")
            self._current_cycle_start = None
            if start:
                try:
                    dt_start = dt_util.parse_datetime(start)
                    if dt_start and dt_start.tzinfo is None:
                        # Fix Naive Timestamp (Legacy Data)
                        dt_start = dt_start.replace(tzinfo=dt_util.now().tzinfo)
                        _LOGGER.warning("Restored Naive start_time, assuming local: %s", dt_start)
                    self._current_cycle_start = dt_start
                except Exception:  # pylint: disable=broad-exception-caught
                    _LOGGER.warning("Failed to parse start time: %s", start)

            readings = snapshot.get("power_readings", [])
            self._power_readings = []

            # Detect naive readings once
            has_naive_readings = False

            for r in readings:
                if isinstance(r, (list, tuple)) and len(r) == 2:
                    t = dt_util.parse_datetime(r[0])
                    if t:
                        if t.tzinfo is None:
                            t = t.replace(tzinfo=dt_util.now().tzinfo)
                            has_naive_readings = True
                        self._power_readings.append((t, float(r[1])))

            if has_naive_readings:
                _LOGGER.warning(
                    "Restored %d power readings with Naive timestamps (fixed to local)",
                    len(self._power_readings),
                )

            # Restore last active
            last_active = snapshot.get("last_active_time")
            if last_active:
                dt_last = dt_util.parse_datetime(last_active)
                if dt_last and dt_last.tzinfo is None:
                    dt_last = dt_last.replace(tzinfo=dt_util.now().tzinfo)
                self._last_active_time = dt_last
            else:
                self._last_active_time = self._current_cycle_start

        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.error("Failed restore: %s", e)
            self.reset()
