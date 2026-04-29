"""Cycle detection logic for WashData."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, cast
import numpy as np

from homeassistant.util import dt as dt_util

from .log_utils import DeviceLoggerAdapter
from .const import (
    STATE_OFF,
    STATE_DELAY_WAIT,
    STATE_STARTING,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_ANTI_WRINKLE,
    STATE_INTERRUPTED,
    STATE_FORCE_STOPPED,
    STATE_UNKNOWN,
    DEVICE_TYPE_WASHING_MACHINE,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_WASHER_DRYER,
    DEFAULT_MAX_DEFERRAL_SECONDS,
    DEFAULT_DEFER_FINISH_CONFIDENCE,
    DISHWASHER_END_SPIKE_MIN_PROGRESS,
    DISHWASHER_END_SPIKE_WAIT_SECONDS,
)

# The dishwasher end-spike wait window is shared between two code paths
# (Smart Termination's wait branch and _should_defer_finish's no-end-spike
# branch).  They MUST release the cycle at the same instant — sanity-check
# that the constants module loaded a sensible value rather than allowing the
# paths to silently drift if one was changed and the other forgotten.
assert DISHWASHER_END_SPIKE_WAIT_SECONDS > 0, (
    "DISHWASHER_END_SPIKE_WAIT_SECONDS must be positive"
)
assert 0 < DISHWASHER_END_SPIKE_MIN_PROGRESS < 1, (
    "DISHWASHER_END_SPIKE_MIN_PROGRESS must be a fraction in (0, 1)"
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
    anti_wrinkle_enabled: bool = False
    anti_wrinkle_max_power: float = 400.0
    anti_wrinkle_max_duration: float = 60.0
    anti_wrinkle_exit_power: float = 0.8
    delay_detect_enabled: bool = False
    delay_drain_min_power: float = 10.0
    delay_drain_max_power: float = 80.0
    delay_drain_max_duration: float = 60.0
    delay_timeout_seconds: float = 28800.0


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
        device_name: str = "",
    ) -> None:
        """Initialize the cycle detector."""
        self._logger = DeviceLoggerAdapter(_LOGGER, device_name)
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

        # Anti-wrinkle tracking (dryers only)
        self._anti_wrinkle_candidate_start: datetime | None = None
        self._anti_wrinkle_candidate_peak: float = 0.0
        self._anti_wrinkle_candidate_start_power: float = 0.0
        self._anti_wrinkle_idle_time: float = 0.0  # Track time spent below exit_power while in ANTI_WRINKLE
        self._anti_wrinkle_idle_timeout: float = 120.0

        # Delayed-start drain candidate tracking
        self._delay_drain_start: datetime | None = None
        self._delay_drain_peak: float = 0.0
        self._delay_wait_true_off_seconds: float = 0.0

    @property
    def _dynamic_pause_threshold(self) -> float:
        """Calculate dynamic pause threshold based on sampling cadence."""
        # User requirement: T_pause >= 3 * p95_update_interval
        # Default 15s or 3 * p95
        return max(15.0, 3.0 * self._p95_dt)

    @property
    def _dynamic_end_threshold(self) -> float:
        """Calculate dynamic end candidate threshold."""
        # Keep this generic for pause->ending transitions across all device types.
        base = 3.0 * self._p95_dt
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
            self._logger.debug("Profile match failed: %s", e)

    # Maximum reasonable cycle duration accepted by the detector.  Anything
    # longer is rejected as corrupted data and replaced with the
    # _SANITIZE_INVALID_SENTINEL so downstream gates fall through to the
    # unmatched / no-expected-duration path.
    _SANITIZE_MAX_EXPECTED_DURATION = 6 * 3600.0  # 6 hours
    _SANITIZE_INVALID_SENTINEL = 0.0  # 0 == "no valid expected_duration"

    def _sanitize_expected_duration(
        self, raw: Any, *, source: str = "update_match"
    ) -> float:
        """Coerce ``raw`` into a finite float in (0, 6h] or return 0.0.

        The class invariant is that ``self._expected_duration`` is either a
        finite, strictly positive float ≤ 6 hours, or 0.0 meaning "no valid
        expected duration".  Every code path that assigns ``_expected_duration``
        (live profile-match callbacks AND restored snapshots) routes through
        this helper so the gates in STATE_ENDING and ``_should_defer_finish``
        can trust the value without re-validating.

        Emits a DEBUG log line distinguishing the rejection reason — the
        ``<= 0`` and ``> 6h`` markers are part of issue #197's regression
        contract and tests assert on them.
        """
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self._logger.debug(
                "%s: invalid raw_expected_duration %r, defaulting to 0.0",
                source, raw,
            )
            return self._SANITIZE_INVALID_SENTINEL
        if not math.isfinite(value):
            self._logger.debug(
                "%s: invalid raw_expected_duration %r, defaulting to 0.0",
                source, raw,
            )
            return self._SANITIZE_INVALID_SENTINEL
        if value <= 0:
            self._logger.debug(
                "%s: invalid raw_expected_duration %r (<= 0), defaulting to 0.0",
                source, raw,
            )
            return self._SANITIZE_INVALID_SENTINEL
        if value > self._SANITIZE_MAX_EXPECTED_DURATION:
            self._logger.debug(
                "%s: invalid raw_expected_duration %r (> 6h), defaulting to 0.0",
                source, raw,
            )
            return self._SANITIZE_INVALID_SENTINEL
        return value

    def update_match(self, result: tuple[Any, ...] | list[Any] | Any) -> None:  # type: ignore[misc]
        """Process a match result (synchronously).

        Can be called by the matcher callback directly or asynchronously.
        """
        # Unpack 5 elements (or 4 for backward compatibility if needed, but wrapper is updated)
        # wrapper returns (name, confidence, duration, phase, is_mismatch)
        # Or MatchResult object if refactored, but currently wrapper returns tuple.

        is_match_mismatch = False
        match_name: str | None = None
        phase_name: str | None = None
        confidence: float = 0.0
        expected_duration: float = 0.0

        if isinstance(result, (list, tuple)):  # type: ignore[misc]
            result_seq = cast(tuple[Any, ...] | list[Any], result)
            if len(result_seq) >= 5:
                (
                    raw_name,
                    raw_confidence,
                    raw_expected_duration,
                    raw_phase_name,
                    raw_mismatch,
                ) = result_seq[:5]
                match_name = str(raw_name) if raw_name is not None else None
                try:
                    confidence = float(raw_confidence)
                    if not math.isfinite(confidence):
                        confidence = 0.0
                        self._logger.debug("update_match: invalid raw_confidence %r, defaulting to 0.0", raw_confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                    self._logger.debug("update_match: invalid raw_confidence %r, defaulting to 0.0", raw_confidence)
                expected_duration = self._sanitize_expected_duration(
                    raw_expected_duration, source="update_match"
                )
                phase_name = str(raw_phase_name) if raw_phase_name is not None else None
                is_match_mismatch = raw_mismatch if isinstance(raw_mismatch, bool) else bool(raw_mismatch)
            else:
                # Fallback for old signature
                if len(result_seq) >= 4:
                    (
                        raw_name,
                        raw_confidence,
                        raw_expected_duration,
                        raw_phase_name,
                    ) = result_seq[:4]
                    match_name = str(raw_name) if raw_name is not None else None
                    try:
                        confidence = float(raw_confidence)
                        if not math.isfinite(confidence):
                            confidence = 0.0
                            self._logger.debug("update_match: invalid raw_confidence %r, defaulting to 0.0", raw_confidence)
                    except (TypeError, ValueError):
                        confidence = 0.0
                        self._logger.debug("update_match: invalid raw_confidence %r, defaulting to 0.0", raw_confidence)
                    expected_duration = self._sanitize_expected_duration(
                        raw_expected_duration, source="update_match"
                    )
                    phase_name = (
                        str(raw_phase_name) if raw_phase_name is not None else None
                    )
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
        # Only reset time_below_threshold if not transitioning to ANTI_WRINKLE
        # (ANTI_WRINKLE needs to track idle time to determine true-off)
        if target_state != STATE_ANTI_WRINKLE:
            self._time_below_threshold = 0.0
        self._last_match_time = None
        self._matched_profile = None
        self._ignore_power_until_idle = False  # Reset lockout
        self._anti_wrinkle_candidate_start = None
        self._anti_wrinkle_candidate_peak = 0.0
        self._anti_wrinkle_candidate_start_power = 0.0
        # Reset idle time tracker for anti-wrinkle
        self._anti_wrinkle_idle_time = 0.0
        # Reset delayed-start tracking
        self._delay_drain_start = None
        self._delay_drain_peak = 0.0
        self._delay_wait_true_off_seconds = 0.0

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
                self._logger.debug(
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
        if self._state in (STATE_OFF, STATE_DELAY_WAIT, STATE_STARTING, STATE_UNKNOWN):
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

        anti_wrinkle_active = (
            self._config.anti_wrinkle_enabled
            and self._config.device_type in (DEVICE_TYPE_DRYER, DEVICE_TYPE_WASHER_DRYER)
        )

        # 3. State Machine

        if self._state in (
            STATE_OFF,
            STATE_FINISHED,
            STATE_INTERRUPTED,
            STATE_FORCE_STOPPED,
            STATE_ANTI_WRINKLE,
        ):
            started_from_anti_wrinkle = False
            if anti_wrinkle_active and self._state == STATE_ANTI_WRINKLE and is_high:
                if self._anti_wrinkle_candidate_start is None:
                    self._anti_wrinkle_candidate_start = timestamp
                    self._anti_wrinkle_candidate_peak = power
                    self._anti_wrinkle_candidate_start_power = power
                else:
                    self._anti_wrinkle_candidate_peak = max(
                        self._anti_wrinkle_candidate_peak, power
                    )

                candidate_duration = (
                    timestamp - self._anti_wrinkle_candidate_start
                ).total_seconds()
                exceeds = (
                    self._anti_wrinkle_candidate_peak
                    > self._config.anti_wrinkle_max_power
                    or power > self._config.anti_wrinkle_max_power
                    or candidate_duration > self._config.anti_wrinkle_max_duration
                )

                if exceeds:
                    candidate_start = self._anti_wrinkle_candidate_start
                    candidate_peak = self._anti_wrinkle_candidate_peak
                    candidate_start_power = self._anti_wrinkle_candidate_start_power
                    self._anti_wrinkle_candidate_start = None
                    self._anti_wrinkle_candidate_peak = 0.0
                    self._anti_wrinkle_candidate_start_power = 0.0
                    self._transition_to(STATE_STARTING, timestamp)
                    started_from_anti_wrinkle = True
                    self._current_cycle_start = candidate_start or timestamp

                    # Preserve the anti-wrinkle candidate window instead of dropping ramp-up samples.
                    if candidate_start and candidate_start < timestamp:
                        start_power = candidate_start_power if candidate_start_power > 0 else power
                        self._power_readings = [(candidate_start, start_power), (timestamp, power)]
                        interval_s = (timestamp - candidate_start).total_seconds()
                        avg_power = (start_power + power) / 2.0
                        self._energy_since_idle_wh = max(0.0, avg_power * (interval_s / 3600.0))
                    else:
                        self._power_readings = [(timestamp, power)]
                        self._energy_since_idle_wh = power * (dt / 3600.0) if dt > 0 else 0.0

                    self._cycle_max_power = max(candidate_peak, power)
                    self._abrupt_drop = False
            elif self._state != STATE_ANTI_WRINKLE:
                self._anti_wrinkle_candidate_start = None
                self._anti_wrinkle_candidate_peak = 0.0
                self._anti_wrinkle_candidate_start_power = 0.0

            if self._state == STATE_ANTI_WRINKLE:
                # Track time in idle (below exit_power threshold)
                effective_exit = max(self._config.anti_wrinkle_exit_power, self._config.stop_threshold_w)
                if power < effective_exit:
                    # Low-power gap invalidates any burst candidate collected while in anti-wrinkle.
                    self._anti_wrinkle_candidate_start = None
                    self._anti_wrinkle_candidate_peak = 0.0
                    self._anti_wrinkle_candidate_start_power = 0.0
                    self._anti_wrinkle_idle_time += dt
                    anti_wrinkle_end_threshold = max(
                        self._dynamic_end_threshold,
                        self._anti_wrinkle_idle_timeout,
                    )
                    if self._anti_wrinkle_idle_time >= anti_wrinkle_end_threshold:
                        self._transition_to(STATE_OFF, timestamp)
                        return
                else:
                    # Reset idle timer when power rises (burst detected)
                    self._anti_wrinkle_idle_time = 0.0

                # Exit conditions:
                # 1. Idle duration exceeded (handled above), OR
                # 2. Safety timeout (2 hours in anti-wrinkle), OR
                # 3. External trigger (user_stop, external triggers handled by manager)
                if (
                    self._state_enter_time
                    and (timestamp - self._state_enter_time).total_seconds() > 7200
                ):
                    # Safety timeout: 2 hours in anti-wrinkle
                    self._transition_to(STATE_OFF, timestamp)
                return

            # Delayed-start drain detection (only from STATE_OFF, not terminal states)
            if (
                self._config.delay_detect_enabled
                and self._state == STATE_OFF
                and not started_from_anti_wrinkle
            ):
                drain_min = self._config.delay_drain_min_power
                effective_drain_max = min(
                    self._config.delay_drain_max_power, self._config.start_threshold_w
                )
                if drain_min < effective_drain_max and drain_min <= power < effective_drain_max:
                    # Power is in the drain-spike window: track it
                    if self._delay_drain_start is None:
                        self._delay_drain_start = timestamp
                        self._delay_drain_peak = power
                    else:
                        self._delay_drain_peak = max(self._delay_drain_peak, power)
                    # Don't fall through to the normal start logic
                    return
                elif power < drain_min and self._delay_drain_start is not None:
                    # Power dropped out of drain window: check if spike was short enough
                    drain_duration = (timestamp - self._delay_drain_start).total_seconds()
                    if drain_duration <= self._config.delay_drain_max_duration:
                        self._logger.info(
                            "Delayed start detected: drain spike %.1fW for %.0fs → DELAY_WAIT",
                            self._delay_drain_peak,
                            drain_duration,
                        )
                        self._transition_to(STATE_DELAY_WAIT, timestamp)
                        return
                    else:
                        # Drain lasted too long — not a real drain, reset
                        self._delay_drain_start = None
                        self._delay_drain_peak = 0.0
                elif power >= effective_drain_max:
                    # Power shot above drain ceiling → real start, clear candidate
                    self._delay_drain_start = None
                    self._delay_drain_peak = 0.0

            if is_high and not started_from_anti_wrinkle:
                # Transition to STARTING
                self._transition_to(STATE_STARTING, timestamp)
                self._current_cycle_start = timestamp
                self._power_readings = [(timestamp, power)]
                self._energy_since_idle_wh = power * (dt / 3600.0) if dt > 0 else 0.0
                self._cycle_max_power = power
                self._abrupt_drop = False
            elif self._state != STATE_OFF:
                # Auto-expire terminal states after 30 minutes
                if (
                    self._state_enter_time
                    and (timestamp - self._state_enter_time).total_seconds() > 1800
                ):
                    self._transition_to(STATE_OFF, timestamp)

        elif self._state == STATE_DELAY_WAIT:
            if power >= self._config.start_threshold_w:
                # Real cycle started
                self._logger.info(
                    "Delayed start: cycle starting (power %.1fW ≥ %.1fW threshold)",
                    power,
                    self._config.start_threshold_w,
                )
                self._transition_to(STATE_STARTING, timestamp)
                self._current_cycle_start = timestamp
                self._power_readings = [(timestamp, power)]
                self._energy_since_idle_wh = power * (dt / 3600.0) if dt > 0 else 0.0
                self._cycle_max_power = power
                self._abrupt_drop = False
            else:
                if power < self._config.stop_threshold_w:
                    # Power near zero: machine turned off, not just waiting
                    self._delay_wait_true_off_seconds += dt
                    if self._delay_wait_true_off_seconds >= 30.0:
                        self._logger.info(
                            "Delayed start cancelled: power dropped to off (%.1fW) for %.0fs",
                            power,
                            self._delay_wait_true_off_seconds,
                        )
                        self._transition_to(STATE_OFF, timestamp)
                        return
                else:
                    self._delay_wait_true_off_seconds = 0.0

                # Safety timeout
                if (
                    self._state_enter_time
                    and (timestamp - self._state_enter_time).total_seconds()
                    >= self._config.delay_timeout_seconds
                ):
                    self._logger.info(
                        "Delayed start timeout after %.0fh → OFF",
                        self._config.delay_timeout_seconds / 3600.0,
                    )
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
                self._logger.debug(
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
                start_time = self._current_cycle_start or timestamp
                current_duration = (timestamp - start_time).total_seconds()

                is_dishwasher = self._config.device_type == "dishwasher"

                # Issue #43: only treat this as a *terminal* end spike (which then
                # pre-arms Smart Termination) when it occurs near the end of the
                # expected cycle.  Mid-cycle spikes - e.g. the dishwasher
                # wash→drying drain wind-down at ~50% of expected duration - must
                # not arm smart termination, otherwise the cycle finishes at 99%
                # of expected *before* the real end-of-cycle pump-out, and that
                # pump-out is then misread as a brand-new cycle.  Without a
                # matched profile (expected==0) the gating is bypassed so the
                # legacy "any spike counts" behaviour is preserved for unmatched
                # cycles (relied on by the dishwasher unmatched-cap path).
                if (
                    self._expected_duration <= 0
                    or current_duration
                    >= self._expected_duration * DISHWASHER_END_SPIKE_MIN_PROGRESS
                ):
                    self._end_spike_seen = True
                    self._logger.debug(
                        "End spike detected (power high in ENDING state, "
                        "%.0fs/%.0fs)",
                        current_duration,
                        self._expected_duration,
                    )
                else:
                    self._logger.debug(
                        "Mid-cycle spike in ENDING ignored for end-spike "
                        "tracking (%.0fs < %.0f%% of expected %.0fs)",
                        current_duration,
                        DISHWASHER_END_SPIKE_MIN_PROGRESS * 100,
                        self._expected_duration,
                    )

                # Sanity check: if expected_duration is unreasonable (>6 hours), use fallback
                max_reasonable = 21600.0  # 6 hours
                effective_expected = self._expected_duration

                if effective_expected <= 0 or effective_expected > max_reasonable:
                    # Fallback: use current duration + buffer if we've run > 3 hours
                    # (Assumes any cycle over 3 hours running is near completion when in ENDING)
                    if current_duration > 10800:  # 3 hours
                        effective_expected = current_duration * 0.99  # Always past threshold
                        self._logger.debug(
                            "End spike check using fallback: expected_duration=%ds is unreasonable, "
                            "using current_duration=%ds as reference",
                            int(self._expected_duration), int(current_duration)
                        )

                past_expected = (
                    effective_expected > 0
                    and current_duration >= (effective_expected * 0.98)
                )

                # If ENDING has already lasted long enough, treat any power burst as
                # terminal (applies to all device types). Dishwashers additionally check
                # proximity to the expected duration.
                long_ending_tail = self._time_in_state >= 120.0
                terminal_spike = long_ending_tail

                if is_dishwasher:
                    near_expected = (
                        effective_expected > 0
                        and current_duration >= (effective_expected * 0.90)
                    )
                    terminal_spike = near_expected or long_ending_tail

                if terminal_spike:
                    self._logger.debug(
                        "End spike kept in ENDING (duration %.0fs/%.0fs, time_in_ending %.0fs)",
                        current_duration,
                        effective_expected,
                        self._time_in_state,
                    )
                    return

                if past_expected:
                    self._logger.debug(
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
                            # If we are a dishwasher and haven't seen a high-power
                            # spike since entering ENDING, wait up to
                            # DISHWASHER_END_SPIKE_WAIT_SECONDS past
                            # expected_duration for the end spike.  Shared with
                            # _should_defer_finish so the two paths release the
                            # cycle at the same instant (issue #43).
                            end_spike_seen = getattr(self, "_end_spike_seen", False)
                            past_wait_period = current_duration >= (
                                self._expected_duration
                                + DISHWASHER_END_SPIKE_WAIT_SECONDS
                            )

                            if (
                                self._config.device_type == "dishwasher"
                                and not end_spike_seen
                                and not past_wait_period
                            ):
                                self._logger.debug(
                                    "Waiting for end spike (duration %.0fs, expected %.0fs + "
                                    "%.0fs wait)",
                                    current_duration,
                                    self._expected_duration,
                                    DISHWASHER_END_SPIKE_WAIT_SECONDS,
                                )
                                return  # Don't finish yet, wait for spike or timeout

                            self._logger.info(
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

                # Energy gate always looks back off_delay seconds by default;
                # overridden below for the dishwasher cap case so the window
                # is consistent with the shortened effective_off_delay.
                gate_window = self._config.off_delay

                # Dishwasher-specific: after a terminal end spike (pump-out), an
                # unmatched cycle doesn't need to wait the full min_off_gap (up to
                # 9000s) before closing. Cap at 30 min so cycle 3 ends cleanly
                # ~30 min after the pump-out rather than sitting open for hours.
                if (
                    self._config.device_type == "dishwasher"
                    and not self._matched_profile
                    and self._end_spike_seen
                ):
                    effective_off_delay = min(effective_off_delay, 1800)
                    gate_window = effective_off_delay

                if self._time_below_threshold >= effective_off_delay:

                    recent_window = [
                        r
                        for r in self._power_readings
                        if (timestamp - r[0]).total_seconds() <= gate_window
                    ]

                    if not recent_window:
                        # Check deferred finish for matched profiles
                        start_time = self._current_cycle_start or timestamp
                        current_duration = (timestamp - start_time).total_seconds()

                        if self._should_defer_finish(current_duration):
                            return

                        # For dishwashers, use the timeout timestamp as end_time
                        # (keep_tail=True) so that the stored cycle duration includes
                        # the passive drying phase.  Without this, end_time snaps back
                        # to _last_active_time which may be set by a terminal drain
                        # spike mid-ENDING, producing a falsely short cycle duration.
                        keep_tail = self._config.device_type == "dishwasher"
                        self._finish_cycle(timestamp, status="completed", keep_tail=keep_tail)
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

                        keep_tail = self._config.device_type == "dishwasher"
                        self._finish_cycle(timestamp, status="completed", keep_tail=keep_tail)
                    else:

                        self._logger.debug(
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
            # Also reset idle time tracker when leaving ANTI_WRINKLE
            self._anti_wrinkle_idle_time = 0.0
            self._delay_drain_start = None
            self._delay_drain_peak = 0.0
            self._delay_wait_true_off_seconds = 0.0

        # Reset end spike tracker when entering ENDING state
        if new_state == STATE_ENDING:
            self._end_spike_seen = False
        elif new_state == STATE_DELAY_WAIT:
            self._delay_drain_start = None
            self._delay_drain_peak = 0.0
            self._delay_wait_true_off_seconds = 0.0
            self._sub_state = "Waiting to Start"
        elif new_state == STATE_ANTI_WRINKLE:
            self._anti_wrinkle_candidate_start = None
            self._anti_wrinkle_candidate_peak = 0.0
            self._anti_wrinkle_candidate_start_power = 0.0
            self._anti_wrinkle_idle_time = 0.0  # Reset idle time when entering ANTI_WRINKLE
            self._sub_state = "Anti-Wrinkle"
        elif new_state == STATE_STARTING:
            # Reset idle time if exiting ANTI_WRINKLE to STARTING (high-power burst resumed)
            self._anti_wrinkle_idle_time = 0.0

        self._logger.debug("Transition: %s -> %s at %s", old_state, new_state, timestamp)
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
            self._logger.debug("Deferring cycle finish: Verified pause active")
            return True

        if not self._matched_profile or self._expected_duration <= 0:
            return False

        # Safety: Don't defer forever
        if duration > (self._expected_duration + DEFAULT_MAX_DEFERRAL_SECONDS):
            self._logger.warning(
                "Deferral limit exceeded (%.0fs > expected %.0f + %s), allowing finish",
                duration,
                self._expected_duration,
                DEFAULT_MAX_DEFERRAL_SECONDS,
            )
            return False

        # Dishwasher passive drying protection:
        # Dishwashers can have 2+ hour passive drying phases at near-0W.  A terminal
        # drain spike that fires early in the ENDING state (e.g. at 120 min of a
        # 233-min ECO cycle) resets _time_below_threshold, and the subsequent 60-min
        # silence timeout would otherwise end the cycle at ~180 min — well before the
        # real finish.  Defer until the cycle reaches the late-phase threshold (the
        # same one used by the end-spike arm gate, so both move together) so that
        # smart termination can catch the true end (~99% of expected) instead.
        # Confidence may be low this early, so the normal confidence gate is
        # bypassed here.
        if (
            self._config.device_type == "dishwasher"
            and self._matched_profile
            and self._expected_duration > 0
            and duration
            < (self._expected_duration * DISHWASHER_END_SPIKE_MIN_PROGRESS)
        ):
            self._logger.debug(
                "Deferring cycle finish: dishwasher drying phase protection "
                "(%.0fs < %.0f%% of expected %.0fs, profile: %s, conf %.2f)",
                duration,
                DISHWASHER_END_SPIKE_MIN_PROGRESS * 100,
                self._expected_duration,
                self._matched_profile,
                self._last_match_confidence,
            )
            return True

        # Issue #43: dishwasher end-spike wait protection.  After 85% of expected
        # the passive-drying gate above releases, but if we have not yet seen the
        # real end-of-cycle pump-out (_end_spike_seen=False) we must keep the
        # cycle in ENDING for a little longer — otherwise the fallback timeout
        # would close it at ~85% of expected, BEFORE the pump-out fires, and the
        # pump-out would then register as a brand-new cycle.  Defer until either
        # the spike fires (which sets _end_spike_seen=True via the gated logic
        # in STATE_ENDING) or we cross the smart-termination wait window
        # (expected + DISHWASHER_END_SPIKE_WAIT_SECONDS), at which point Smart
        # Termination's own wait branch takes over and finalises the cycle.
        # Shares the same constant so the two paths cannot drift.
        if (
            self._config.device_type == "dishwasher"
            and self._matched_profile
            and self._expected_duration > 0
            and not self._end_spike_seen
            and duration
            < (self._expected_duration + DISHWASHER_END_SPIKE_WAIT_SECONDS)
        ):
            self._logger.debug(
                "Deferring cycle finish: dishwasher waiting for end-of-cycle "
                "pump-out (%.0fs < expected %.0fs + %.0fs wait, profile: %s)",
                duration,
                self._expected_duration,
                DISHWASHER_END_SPIKE_WAIT_SECONDS,
                self._matched_profile,
            )
            return True

        # If matched profile, enforce min duration ratio
        ratio = self._config.min_duration_ratio

        # --- STRICTER DEFERRAL ---
        # If we are NOT in a verified pause, but power has been low for a long time (ENDING state),
        # we only defer if we are VERY confident this profile is correct.
        # This prevents hanging on too-long profiles that matched early but are now diverging.
        if self._last_match_confidence < DEFAULT_DEFER_FINISH_CONFIDENCE:
            self._logger.debug(
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
            self._logger.debug(
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

        start_ts = self._current_cycle_start.timestamp()
        cycle_data: dict[str, Any] = {
            "start_time": self._current_cycle_start.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": duration,
            "max_power": self._cycle_max_power,
            "status": status,
            "termination_reason": termination_reason,
            "power_data": [[round(t.timestamp() - start_ts, 1), p] for t, p in final_readings],
        }

        self._logger.info("Cycle Finished: %s, %.1f min", status, duration / 60)
        self._on_cycle_end(cycle_data)

        target = STATE_FINISHED
        if status == "interrupted":
            target = STATE_INTERRUPTED
        elif status == "force_stopped":
            target = STATE_FORCE_STOPPED
        elif (
            status == "completed"
            and termination_reason in {"timeout", "smart"}
            and self._config.anti_wrinkle_enabled
            and self._config.device_type in (DEVICE_TYPE_DRYER, DEVICE_TYPE_WASHER_DRYER)
        ):
            target = STATE_ANTI_WRINKLE

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
            "end_spike_seen": self._end_spike_seen,
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
            # Sanitize via the same helper as update_match so the class
            # invariant on _expected_duration holds across restarts and the
            # gates in STATE_ENDING / _should_defer_finish can trust the value.
            self._expected_duration = self._sanitize_expected_duration(
                snapshot.get("expected_duration", 0.0),
                source="restore_state_snapshot",
            )
            self._end_spike_seen = snapshot.get("end_spike_seen", False)

            # Restore state enter time and recompute time_in_state from it
            enter_time = snapshot.get("state_enter_time")
            if enter_time:
                try:
                    self._state_enter_time = dt_util.parse_datetime(enter_time)
                    if self._state_enter_time:
                        elapsed = (dt_util.now() - self._state_enter_time).total_seconds()
                        self._time_in_state = max(0.0, elapsed)
                except Exception: # pylint: disable=broad-exception-caught
                    self._logger.warning("Failed to parse state enter time")

            start = snapshot.get("current_cycle_start")
            self._current_cycle_start = None
            if start:
                try:
                    dt_start = dt_util.parse_datetime(start)
                    if dt_start and dt_start.tzinfo is None:
                        # Fix Naive Timestamp (Legacy Data)
                        dt_start = dt_start.replace(tzinfo=dt_util.now().tzinfo)
                        self._logger.warning("Restored Naive start_time, assuming local: %s", dt_start)
                    self._current_cycle_start = dt_start
                except Exception:  # pylint: disable=broad-exception-caught
                    self._logger.warning("Failed to parse start time: %s", start)

            readings = snapshot.get("power_readings", [])
            self._power_readings = []

            # Detect naive readings once
            has_naive_readings = False

            for r in readings:
                if isinstance(r, (list, tuple)):
                    reading = cast(list[Any] | tuple[Any, ...], r)
                    if len(reading) < 2:
                        continue
                    try:
                        t = dt_util.parse_datetime(str(reading[0]))
                        if t:
                            if t.tzinfo is None:
                                t = t.replace(tzinfo=dt_util.now().tzinfo)
                                has_naive_readings = True
                            value = float(reading[1])
                            if math.isfinite(value):
                                self._power_readings.append((t, value))
                    except (TypeError, ValueError) as exc:
                        self._logger.debug("Skipping malformed power reading %s: %s", r, exc)

            if has_naive_readings:
                self._logger.warning(
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
            self._logger.error("Failed restore: %s", e)
            self.reset()