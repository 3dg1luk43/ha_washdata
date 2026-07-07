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
    ANTI_WRINKLE_ELIGIBLE_REASONS,
    TerminationReason,
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
    DISHWASHER_MIN_CYCLE_DURATION_S,
    TERMINAL_DROP_OFF_DELAY_SECONDS,
)

# The dishwasher end-spike wait window is shared between two code paths
# (Smart Termination's wait branch and _should_defer_finish's no-end-spike
# branch).  They MUST release the cycle at the same instant - sanity-check
# that the constants module loaded a sensible value rather than allowing the
# paths to silently drift if one was changed and the other forgotten.
assert DISHWASHER_END_SPIKE_WAIT_SECONDS > 0, (
    "DISHWASHER_END_SPIKE_WAIT_SECONDS must be positive"
)

# Opt-in ML end-detection guard (Stage 6). When the manager injects an
# end-confidence provider (only when the user enabled ML models for the device),
# the cycle-end model can defer a *normal* completion if it judges the current
# low-power event to be a pause rather than the true end. This is intentionally
# asymmetric: it can only *delay* a completion, never end a cycle early, and it
# is bounded, so a wrong model can slow a finish but can neither stop one early
# nor hang the cycle. Force-stop / smart-termination / user paths never consult
# it. Overridable emphasis lives here rather than const.py to keep the guard
# self-contained (it is detector-internal policy, not user configuration).
ML_END_GUARD_MIN_CONFIDENCE = 0.5        # P(true end) below this -> treat as a likely pause
ML_END_GUARD_MAX_DEFER_SECONDS = 1800.0  # cap the extra wait the guard may add (30 min)
assert 0 < DISHWASHER_END_SPIKE_MIN_PROGRESS < 1, (
    "DISHWASHER_END_SPIKE_MIN_PROGRESS must be a fraction in (0, 1)"
)
from .signal_processing import integrate_wh

_LOGGER = logging.getLogger(__name__)

# After a user/external stop the manual-stop lockout swallows the machine's
# spin-down/drain so it is not logged as a fresh cycle. The lockout normally
# clears as soon as power drops to idle. As a safety net, if power instead stays
# at or above the start threshold for longer than any plausible spin-down, the
# device is running a genuinely new (back-to-back) load: release the lockout so
# the new cycle is detected instead of being pinned until the progress-reset
# window expires (issue #267).
STOP_LOCKOUT_RELEASE_SECONDS = 180.0


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
    start_energy_threshold: float = 0.005
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
    # Sustained seconds power must stay in the standby band (between
    # stop_threshold_w and start_threshold_w) before DELAY_WAIT engages.
    # Tuned to filter out brief menu-navigation peaks at the start of a
    # delayed program.
    delay_confirm_seconds: float = 60.0
    delay_timeout_seconds: float = 28800.0


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
        end_confidence_provider: (
            Callable[[list[tuple[float, float]], float], float | None] | None
        ) = None,
        terminal_drop_provider: (
            Callable[[list[tuple[float, float]], float], bool | None] | None
        ) = None,
    ) -> None:
        """Initialize the cycle detector."""
        self._logger = DeviceLoggerAdapter(_LOGGER, device_name)
        self._config = config
        self._on_state_change = on_state_change
        self._on_cycle_end = on_cycle_end
        self._profile_matcher = profile_matcher
        # Opt-in ML end-guard: (points, expected_duration) -> P(true end) or None.
        # Injected by the manager; None disables the guard (existing behavior).
        self._end_confidence_provider = end_confidence_provider
        # Opt-in terminal-drop detector: (points, expected_duration) -> bool.
        # True means the current low-power event is an anomalously-early hard
        # cliff-to-0 (never seen this early on this device), so the cycle may be
        # finalized without waiting out the full soak-bridging min_off_gap.
        # Injected by the manager; None disables it (existing behavior). Opposite
        # asymmetry to the end-guard: it can only ever *shorten* the end wait.
        self._terminal_drop_provider = terminal_drop_provider
        # Cycle duration (s) at which the ML guard first deferred the current
        # ending episode; bounds how long the guard may keep deferring.
        self._ml_defer_start_duration: float | None = None

        # State
        self._state = STATE_OFF
        self._sub_state: str | None = None
        self._ignore_power_until_idle: bool = False
        # Sustained high-power time accrued while the stop lockout is armed; used
        # to release the lockout for a genuinely new back-to-back load (#267).
        self._lockout_high_seconds: float = 0.0

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
        self._match_ambiguous: bool = False  # last live match was ambiguous (gates predictive end)
        self._match_prefix_ambiguous: bool = False  # longer candidate with good shape exists (prefix guard)

        # Anti-wrinkle tracking (dryers only)
        self._anti_wrinkle_candidate_start: datetime | None = None
        self._anti_wrinkle_candidate_peak: float = 0.0
        self._anti_wrinkle_candidate_start_power: float = 0.0
        self._anti_wrinkle_idle_time: float = 0.0  # Track time spent below exit_power while in ANTI_WRINKLE
        self._anti_wrinkle_idle_timeout: float = 120.0

        # Delayed-start band tracking.
        # _delay_band_start anchors the first reading in the standby band
        # [stop_threshold_w, start_threshold_w) while still in STATE_OFF.
        # _delay_band_seconds mirrors the anchored elapsed time for
        # diagnostics and tests.
        self._delay_band_start: datetime | None = None
        self._delay_band_seconds: float = 0.0
        # _delay_band_peak is purely diagnostic - surfaced in the log line
        # when the transition fires so users can see what their machine's
        # actual standby plateau looked like.
        self._delay_band_peak: float = 0.0
        # _delay_wait_true_off_seconds tracks sustained "true off" (power
        # below stop_threshold_w) inside DELAY_WAIT, so we can drop back to
        # OFF only when the machine has clearly been switched off rather
        # than briefly dipped.
        self._delay_wait_true_off_seconds: float = 0.0
        # _delay_wait_high_start anchors the first high-power reading
        # observed inside DELAY_WAIT.  We only transition to STARTING
        # when the high-power streak has lasted at least
        # start_duration_threshold real seconds - measured between two
        # consecutive high readings, not from the dt to the previous
        # (low) reading.  This prevents a single isolated spike from
        # tripping STARTING just because the sampling interval is long.
        self._delay_wait_high_start: datetime | None = None
        self._delay_wait_high_power: float | None = None
        # Preserve a delayed-start candidate across a false STARTING probe
        # that drops back into the standby band without the machine truly
        # turning off.
        self._preserve_delay_band_on_off: bool = False

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

        Emits a DEBUG log line distinguishing the rejection reason - the
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
        ambiguous: bool = False

        if isinstance(result, (list, tuple)):  # type: ignore[misc]
            result_seq = cast(tuple[Any, ...] | list[Any], result)
            # Optional 6th element: whether the live match is ambiguous
            # (top-1 vs top-2 within MATCH_AMBIGUITY_MARGIN). Used to gate the
            # predictive Smart Termination below.
            if len(result_seq) >= 6:
                ambiguous = bool(result_seq[5])
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

            # Store confidence + ambiguity for Smart Termination checks
            self._last_match_confidence = confidence or 0.0
            self._match_ambiguous = ambiguous
            self._match_prefix_ambiguous = bool(result_seq[6]) if len(result_seq) >= 7 else False
        else:
            # Assume MatchResult object or similar (future proofing)
            # But for now wrapper returns tuple
            return

        if is_match_mismatch and self._matched_profile:
            # Confident non-match - revert to detecting if previously matched
            self._matched_profile = None
            self._match_ambiguous = False
            self._match_prefix_ambiguous = False

        elif match_name:
            # If sanitization rejected the expected_duration, treat the match
            # as invalid: setting _matched_profile while _expected_duration is
            # the 0.0 sentinel would let Smart Termination fire on the
            # `current_duration >= 0` always-true comparison.  Drop both so
            # the cycle stays in detecting/unmatched mode.
            if expected_duration == self._SANITIZE_INVALID_SENTINEL:
                self._logger.debug(
                    "update_match: match %r ignored - expected_duration "
                    "sanitized to invalid sentinel; treating as unmatched",
                    match_name,
                )
                self._matched_profile = None
                self._expected_duration = self._SANITIZE_INVALID_SENTINEL
            else:
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
        self._lockout_high_seconds = 0.0
        self._anti_wrinkle_candidate_start = None
        self._anti_wrinkle_candidate_peak = 0.0
        self._anti_wrinkle_candidate_start_power = 0.0
        # Reset idle time tracker for anti-wrinkle
        self._anti_wrinkle_idle_time = 0.0
        # Reset delayed-start tracking
        self._delay_band_seconds = 0.0
        self._delay_band_peak = 0.0
        self._delay_wait_true_off_seconds = 0.0
        self._delay_wait_high_start = None

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

        # Calculate dt (needed by the stop lockout below and the state machine).
        dt = 0.0
        if self._last_process_time:
            dt = (timestamp - self._last_process_time).total_seconds()

        # Sanity check for negative dt
        if dt < 0:
            self._last_process_time = timestamp
            return

        # Manual Stop Lockout:
        # If user/external stop forced an end, ignore the machine's spin-down so
        # it is not logged as a new cycle. The lockout clears the moment power
        # drops to idle. As a safety net, if power instead stays high far longer
        # than any plausible spin-down, treat it as a genuinely new back-to-back
        # load and release the lockout so the cycle is detected immediately
        # rather than pinned until the progress-reset window expires (#267).
        if self._ignore_power_until_idle:
            if power < self._config.start_threshold_w:
                self._ignore_power_until_idle = False
                self._lockout_high_seconds = 0.0
                self._logger.debug(
                    "Power dropped below start threshold. Manual stop lockout cleared."
                )
            else:
                self._lockout_high_seconds += dt
                if self._lockout_high_seconds < STOP_LOCKOUT_RELEASE_SECONDS:
                    # Still within the spin-down window - ignore reading.
                    self._last_process_time = timestamp
                    return
                self._ignore_power_until_idle = False
                self._lockout_high_seconds = 0.0
                self._logger.info(
                    "Manual stop lockout released after sustained power "
                    "(>= %.1fs at/above start threshold): treating as a new "
                    "cycle (#267).",
                    STOP_LOCKOUT_RELEASE_SECONDS,
                )
                # Fall through: the state machine will start a new cycle.

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
            and self._config.device_type in (
                DEVICE_TYPE_WASHING_MACHINE,
                DEVICE_TYPE_DRYER,
                DEVICE_TYPE_WASHER_DRYER,
            )
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

            # Delayed-start "standby band" detection (only from STATE_OFF).
            #
            # A machine in delayed-start mode sits in a power band between
            # the off-noise floor (stop_threshold_w) and the cycle-start
            # threshold (start_threshold_w) - display, electronics, the
            # occasional anti-damp tumble - for minutes to hours.  We
            # track anchored elapsed time while power is in that band; once
            # it crosses delay_confirm_seconds we transition to DELAY_WAIT.
            #
            # Brief high-power excursions (menu navigation, button presses)
            # don't break the candidate: they fall through to the normal
            # start logic below, and unless they sustain for
            # start_duration_threshold they get aborted as a false start
            # and we re-enter the band on the next reading.  Excursions
            # below stop_threshold_w (machine momentarily idle on the noise
            # floor) DO reset the candidate, because that's the same
            # signal we use to define "off".
            if (
                self._config.delay_detect_enabled
                and self._state == STATE_OFF
                and not started_from_anti_wrinkle
                and self._config.stop_threshold_w < self._config.start_threshold_w
            ):
                in_band = (
                    self._config.stop_threshold_w
                    <= power
                    < self._config.start_threshold_w
                )
                if in_band:
                    if self._delay_band_start is None:
                        self._delay_band_start = timestamp
                        self._delay_band_seconds = 0.0
                    else:
                        self._delay_band_seconds = (
                            timestamp - self._delay_band_start
                        ).total_seconds()
                    self._delay_band_peak = max(self._delay_band_peak, power)
                    if self._delay_band_seconds >= self._config.delay_confirm_seconds:
                        self._logger.info(
                            "Delayed start detected: standby band held for %.0fs "
                            "(peak %.1fW, current %.1fW) → DELAY_WAIT",
                            self._delay_band_seconds,
                            self._delay_band_peak,
                            power,
                        )
                        self._transition_to(STATE_DELAY_WAIT, timestamp)
                        return
                    # Stay in OFF while we accumulate evidence - do not
                    # fall through to the high-power start logic, the
                    # reading is below threshold by definition.
                    return
                elif power < self._config.stop_threshold_w:
                    # Machine genuinely idle: forget any band history.
                    self._delay_band_start = None
                    self._delay_band_seconds = 0.0
                    self._delay_band_peak = 0.0
                    self._preserve_delay_band_on_off = False
                # power >= start_threshold_w: fall through to the normal
                # start path below.  If it turns out to be a brief peak,
                # STATE_STARTING will abort it as a false start and we'll
                # re-enter the band check on the next sample without
                # losing accumulated time (we don't reset on a high
                # excursion - most users' "menu navigation" peaks last
                # less than a sample interval anyway).

            if is_high and not started_from_anti_wrinkle:
                # Transition to STARTING
                self._preserve_delay_band_on_off = self._delay_band_start is not None
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
                # Power is in cycle-start territory.  Require at least
                # two consecutive high readings spanning
                # start_duration_threshold real seconds before committing
                # to STARTING, so a single isolated spike (a heavy menu
                # interaction, an anti-damp pulse briefly crossing the
                # threshold) doesn't false-trigger.  We anchor on the
                # FIRST high reading instead of accumulating dt, because
                # dt to the previous (low) reading is unrelated to how
                # long the high power has actually persisted.
                self._delay_wait_true_off_seconds = 0.0
                if self._delay_wait_high_start is None:
                    self._delay_wait_high_start = timestamp
                    self._delay_wait_high_power = power
                else:
                    elapsed_high = (
                        timestamp - self._delay_wait_high_start
                    ).total_seconds()
                    if elapsed_high >= self._config.start_duration_threshold:
                        self._logger.info(
                            "Delayed start: cycle starting (power %.1fW sustained ≥ %.1fW for %.0fs)",
                            power,
                            self._config.start_threshold_w,
                            elapsed_high,
                        )
                        self._transition_to(STATE_STARTING, timestamp)
                        start_timestamp = self._delay_wait_high_start or timestamp
                        start_power = self._delay_wait_high_power or power
                        self._current_cycle_start = start_timestamp
                        self._power_readings = [(start_timestamp, start_power)]
                        elapsed_from_anchor = (timestamp - start_timestamp).total_seconds()
                        self._energy_since_idle_wh = (
                            start_power * (elapsed_from_anchor / 3600.0)
                            if elapsed_from_anchor > 0
                            else 0.0
                        )
                        if timestamp != start_timestamp:
                            self._power_readings.append((timestamp, power))
                        self._cycle_max_power = max(start_power, power)
                        self._abrupt_drop = False
            else:
                # Power dropped back below start threshold - clear the
                # high-power streak anchor so the next high reading
                # starts a fresh confirmation window.
                self._delay_wait_high_start = None
                self._delay_wait_high_power = None
                if power < self._config.stop_threshold_w:
                    # Power near zero: machine genuinely turned off, not
                    # just waiting.
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
                self._delay_band_start = None
                self._delay_band_seconds = 0.0
                self._delay_band_peak = 0.0
                self._preserve_delay_band_on_off = False
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

                    # Gate the predictive end on match certainty.
                    # _match_ambiguous: top-1 vs top-2 score gap is too small to
                    # trust the matched profile's expected duration — fall through
                    # to the power-based fallback timeout instead.
                    # _match_prefix_ambiguous: a longer candidate with a similar
                    # shape score exists in the pool. The current trace may be a
                    # prefix of that longer program (e.g. Quick 46 min matched
                    # while the machine is actually running Normal 88 min and
                    # happens to be in a mid-cycle soak dip at the 46-min mark).
                    # Blocking Smart Termination here means a true Quick cycle
                    # waits for the fallback timeout instead of getting an early
                    # close — an acceptable trade-off against the alternative of
                    # splitting a Normal wash into two separate cycle records.
                    if (
                        current_duration >= (self._expected_duration * smart_ratio)
                        and is_confident_match
                        and not self._match_ambiguous
                        and not self._match_prefix_ambiguous
                    ):
                        # Dynamic confirmation window
                        if self._config.device_type == "dishwasher":
                            smart_debounce = max(300.0, self._config.off_delay * 0.25)
                        elif self._config.device_type in (
                            DEVICE_TYPE_WASHING_MACHINE,
                            DEVICE_TYPE_WASHER_DRYER,
                        ):
                            # Washing machines and washer-dryers have soak and
                            # rinse gaps that can dip for several minutes between
                            # programme phases.  Require quiet time equal to half
                            # the soak-bridging min_off_gap before committing
                            # Smart Termination, so a near-duplicate profile
                            # doesn't cut a long cycle short during a mid-cycle
                            # power trough.
                            smart_debounce = max(180.0, self._config.min_off_gap * 0.5)
                        else:
                            smart_debounce = 120.0

                        if self._time_in_state >= smart_debounce:
                            # --- END SPIKE WAIT PERIOD (Dishwashers) ---
                            # Dishwashers should see the real end-of-cycle
                            # pump-out (which arms _end_spike_seen via the 85%
                            # progress gate) before Smart Termination fires -
                            # otherwise the pump-out arrives AFTER the cycle
                            # has already closed and registers as a brand-new
                            # "ghost" cycle.  User reports (issue #43) showed
                            # the original 5-min past_wait_period escape hatch
                            # closing the cycle ~4 min before the real pump-out
                            # at ~99.5% of expected.  Widen the escape hatch
                            # substantially (DISHWASHER_END_SPIKE_WAIT_SECONDS,
                            # currently 30 min past expected) so it cannot
                            # short-circuit a pump-out that fires within a
                            # reasonable window around expected end, but still
                            # guarantees the cycle terminates eventually for
                            # dishwashers that have no pump-out at all.
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
                                    "Waiting for end spike (duration %.0fs, "
                                    "expected %.0fs + %.0fs wait)",
                                    current_duration,
                                    self._expected_duration,
                                    DISHWASHER_END_SPIKE_WAIT_SECONDS,
                                )
                                return  # Don't finish yet, wait for spike

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
                                termination_reason=TerminationReason.SMART,
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

                # Opt-in terminal-drop fast finalize (asymmetric, shorten-only):
                # a hard cliff-to-~0 sustained for TERMINAL_DROP_OFF_DELAY_SECONDS
                # that began earlier than this device has ever legitimately gone
                # quiet is almost certainly a real stop (plug pulled / cancelled),
                # not a soak.  Finalize now instead of waiting out the full
                # soak-bridging min_off_gap.  Only consulted when there is a longer
                # wait to shorten and the provider is wired (ML/anomaly opt-in);
                # the energy/defer gates are bypassed because the sustained sub-
                # threshold span already proves the appliance is off, and the
                # anomaly check has ruled out a legitimate early pause.
                if (
                    self._terminal_drop_provider is not None
                    and not self._verified_pause
                    and effective_off_delay > TERMINAL_DROP_OFF_DELAY_SECONDS
                    and self._time_below_threshold >= TERMINAL_DROP_OFF_DELAY_SECONDS
                    and self._is_terminal_drop()
                ):
                    start_time = self._current_cycle_start or timestamp
                    current_duration = (timestamp - start_time).total_seconds()
                    self._logger.info(
                        "Terminal drop: anomalously-early power cliff after %.0fs "
                        "(device never quiet this early) - finalizing without the "
                        "full %.0fs soak wait.",
                        current_duration,
                        effective_off_delay,
                    )
                    self._finish_cycle(
                        timestamp,
                        status="interrupted",
                        termination_reason=TerminationReason.TERMINAL_DROP,
                        keep_tail=False,
                    )
                    return

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

        # Bound each ENDING episode's ML-guard deferral independently: clear the
        # tracker whenever we are not in ENDING (e.g. on resume back to RUNNING).
        if new_state != STATE_ENDING:
            self._ml_defer_start_duration = None

        # Reset energy accumulator on transition to OFF
        if new_state == STATE_OFF:
            self._energy_since_idle_wh = 0.0
            # Also reset idle time tracker when leaving ANTI_WRINKLE
            self._anti_wrinkle_idle_time = 0.0
            if not self._preserve_delay_band_on_off:
                self._delay_band_start = None
                self._delay_band_seconds = 0.0
                self._delay_band_peak = 0.0
            self._delay_wait_true_off_seconds = 0.0
            self._delay_wait_high_start = None
            self._delay_wait_high_power = None
            self._preserve_delay_band_on_off = False

        # Reset end spike tracker when entering ENDING state
        if new_state == STATE_ENDING:
            self._end_spike_seen = False
        elif new_state == STATE_DELAY_WAIT:
            # Band-accumulation tracker already played its role getting us
            # here; reset it so a future OFF→band cycle starts fresh.
            self._delay_band_start = None
            self._delay_band_seconds = 0.0
            self._delay_band_peak = 0.0
            self._delay_wait_true_off_seconds = 0.0
            self._delay_wait_high_start = None
            self._delay_wait_high_power = None
            self._sub_state = "Waiting to Start"
            self._preserve_delay_band_on_off = False
        elif new_state == STATE_ANTI_WRINKLE:
            self._anti_wrinkle_candidate_start = None
            self._anti_wrinkle_candidate_peak = 0.0
            self._anti_wrinkle_candidate_start_power = 0.0
            self._anti_wrinkle_idle_time = 0.0  # Reset idle time when entering ANTI_WRINKLE
            self._sub_state = "Anti-Wrinkle"
        elif new_state == STATE_STARTING:
            # Reset idle time if exiting ANTI_WRINKLE to STARTING (high-power burst resumed)
            self._anti_wrinkle_idle_time = 0.0
        elif new_state == STATE_RUNNING:
            self._delay_band_start = None
            self._delay_band_seconds = 0.0
            self._delay_band_peak = 0.0
            self._preserve_delay_band_on_off = False

        self._logger.debug("Transition: %s -> %s at %s", old_state, new_state, timestamp)
        self._on_state_change(old_state, new_state)

    def _ml_end_confidence(self) -> float | None:
        """P(the current low-power event is the true end) from the opt-in ML guard.

        Builds the offset-second trace from the current cycle's readings and asks
        the injected provider. Returns None when there is no provider, no cycle
        start, or the provider declines (ML off / unmatched / model unavailable),
        so the caller keeps the existing power/energy-based behavior.
        """
        provider = self._end_confidence_provider
        if provider is None or self._current_cycle_start is None or not self._power_readings:
            return None
        start = self._current_cycle_start
        points = [
            ((ts - start).total_seconds(), float(power))
            for ts, power in self._power_readings
        ]
        try:
            return provider(points, float(self._expected_duration))
        except Exception:  # noqa: BLE001 - ML must never break detection
            return None

    def _is_terminal_drop(self) -> bool:
        """Whether the current low-power event is an anomalously-early hard drop.

        Mirrors ``_ml_end_confidence``: builds the offset-second trace from the
        current cycle's readings and asks the injected terminal-drop provider.
        Returns ``False`` when there is no provider, no cycle start, or the
        provider declines/raises (ML off / too little history / not anomalous),
        so the caller keeps the proven soak-bridging end-detection.
        """
        provider = self._terminal_drop_provider
        if provider is None or self._current_cycle_start is None or not self._power_readings:
            return False
        start = self._current_cycle_start
        points = [
            ((ts - start).total_seconds(), float(power))
            for ts, power in self._power_readings
        ]
        try:
            return bool(provider(points, float(self._expected_duration)))
        except Exception:  # noqa: BLE001 - ML must never break detection
            return False

    def _should_defer_finish(self, duration: float) -> bool:
        """Check if we should defer termination based on expected duration."""
        # Check explicit verified pause override from manager
        if getattr(self, "_verified_pause", False):
            self._logger.debug("Deferring cycle finish: Verified pause active")
            return True

        # Dishwasher minimum-duration floor: even without a matched profile (e.g.
        # first cycle of a program, or the 5-min matcher hasn't fired yet) a
        # dishwasher cycle should never end before it has crossed the minimum
        # reasonable programme duration.  This prevents a dip during the fill or
        # early wash phase from being read as the end of a complete cycle.
        if (
            self._config.device_type == "dishwasher"
            and duration < DISHWASHER_MIN_CYCLE_DURATION_S
        ):
            self._logger.debug(
                "Deferring dishwasher cycle end: elapsed %.0fs < minimum %.0fs",
                duration,
                DISHWASHER_MIN_CYCLE_DURATION_S,
            )
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

        # Opt-in ML end-guard (asymmetric anti-premature-stop, bounded). If the
        # cycle-end model judges this low-power event to be more likely a pause
        # than the true end, defer the normal completion - but only for a bounded
        # extra window, so a wrong model can delay, never hang, the cycle. As the
        # low-power run lengthens the model's confidence rises, so a genuine end
        # is released once the model agrees or the cap is reached.
        if (
            self._end_confidence_provider is not None
            and self._last_match_confidence >= DEFAULT_DEFER_FINISH_CONFIDENCE
        ):
            confidence = self._ml_end_confidence()
            if confidence is not None and confidence < ML_END_GUARD_MIN_CONFIDENCE:
                if self._ml_defer_start_duration is None:
                    self._ml_defer_start_duration = duration
                if (duration - self._ml_defer_start_duration) < ML_END_GUARD_MAX_DEFER_SECONDS:
                    self._logger.debug(
                        "Deferring cycle finish: ML end-guard (P(true end)=%.2f < %.2f)",
                        confidence,
                        ML_END_GUARD_MIN_CONFIDENCE,
                    )
                    return True
            elif confidence is not None:
                # Model is confident this is the true end -> stop ML-deferring.
                self._ml_defer_start_duration = None

        # Dishwasher passive drying protection:
        # Dishwashers can have 2+ hour passive drying phases at near-0W.  A terminal
        # drain spike that fires early in the ENDING state (e.g. at 120 min of a
        # 233-min ECO cycle) resets _time_below_threshold, and the subsequent 60-min
        # silence timeout would otherwise end the cycle at ~180 min - well before the
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

        # Issue #43: dishwasher end-spike wait protection.  Once past the 85%
        # passive-drying gate above, we still keep the cycle deferred until
        # the real end-of-cycle pump-out fires (sets _end_spike_seen=True via
        # the 85% progress gate in STATE_ENDING) or we cross the
        # smart-termination wait window (expected + 30 min) - whichever comes
        # first.  Shares DISHWASHER_END_SPIKE_WAIT_SECONDS with Smart
        # Termination's wait branch so the two paths release the cycle at the
        # same instant.  Beyond the wait window, Smart Termination's
        # past_wait_period kicks in and finalises; below it, the fallback
        # timeout's energy gate is the safety net for cycles whose pump-out
        # never arrives.
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
        termination_reason: str = TerminationReason.TIMEOUT,
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
            and termination_reason in ANTI_WRINKLE_ELIGIBLE_REASONS
            and self._config.anti_wrinkle_enabled
            and self._config.device_type in (
                DEVICE_TYPE_WASHING_MACHINE,
                DEVICE_TYPE_DRYER,
                DEVICE_TYPE_WASHER_DRYER,
            )
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
                termination_reason=TerminationReason.FORCE_STOPPED,
                keep_tail=False,  # Force stop usually implies snap back to reality
            )
            self._ignore_power_until_idle = False

    def user_stop(self) -> None:
        """Handle user-initiated stop."""
        if self._state != STATE_OFF:
            now = dt_util.now()
            self._finish_cycle(
                now,
                status="completed",
                termination_reason=TerminationReason.USER,
                keep_tail=True,  # User implies "Done Now"
            )
            # Prevent immediate restart if power is still high
            self._ignore_power_until_idle = True
            # Anchor the lockout clock to this stop instant. The next reading's
            # dt is measured from the last processed sample, which predates the
            # stop, so without this the high-power accumulator would count the
            # pre-stop gap and release the lockout early (#267).
            self._lockout_high_seconds = 0.0
            self._last_process_time = now


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
            "match_ambiguous": self._match_ambiguous,
            "match_prefix_ambiguous": self._match_prefix_ambiguous,
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

    def restore_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        try:
            self._state = snapshot.get("state", STATE_OFF)
            self._sub_state = snapshot.get("sub_state")
            self._energy_since_idle_wh = snapshot.get("accumulated_energy_wh", 0.0)
            self._time_above_threshold = snapshot.get("time_above", 0.0)
            self._time_below_threshold = snapshot.get("time_below", 0.0)
            self._cycle_max_power = snapshot.get("cycle_max_power", 0.0)
            # Sanitize via the same helper as update_match so the class
            # invariant on _expected_duration holds across restarts and the
            # gates in STATE_ENDING / _should_defer_finish can trust the value.
            # If sanitization rejects the snapshot's expected_duration, also
            # clear the matched_profile so we don't restore a half-valid state
            # where Smart Termination can fire on _expected_duration == 0.0.
            restored_match = snapshot.get("matched_profile")
            sanitized_expected = self._sanitize_expected_duration(
                snapshot.get("expected_duration", 0.0),
                source="restore_state_snapshot",
            )
            if (
                restored_match is not None
                and sanitized_expected == self._SANITIZE_INVALID_SENTINEL
            ):
                self._logger.debug(
                    "restore_state_snapshot: dropping matched_profile %r "
                    "because expected_duration sanitized to invalid sentinel",
                    restored_match,
                )
                self._matched_profile = None
            else:
                self._matched_profile = restored_match
            self._expected_duration = sanitized_expected
            self._end_spike_seen = snapshot.get("end_spike_seen", False)
            self._match_ambiguous = snapshot.get("match_ambiguous", False)
            self._match_prefix_ambiguous = snapshot.get("match_prefix_ambiguous", False)

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