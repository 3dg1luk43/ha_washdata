"""Manager for HA WashData."""

# pylint: disable=broad-exception-caught

from __future__ import annotations

import logging
import hashlib
import inspect
from datetime import datetime, timedelta
from typing import Any, cast
import numpy as np

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.util import dt as dt_util
import homeassistant.helpers.event as evt

from .const import (
    CONF_POWER_SENSOR,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_EVENTS,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_LOW_POWER_NO_UPDATE_TIMEOUT, # Import new constant
    CONF_SMOOTHING_WINDOW,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_INTERRUPTED_MIN_SECONDS,
    CONF_ABRUPT_DROP_WATTS,
    CONF_ABRUPT_DROP_RATIO,
    CONF_ABRUPT_HIGH_LOAD_FACTOR,
    CONF_PROGRESS_RESET_DELAY,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_AUTO_MAINTENANCE,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_MAX_PAST_CYCLES,
    CONF_MAX_FULL_TRACES_PER_PROFILE,
    CONF_MAX_FULL_TRACES_UNLABELED,
    CONF_WATCHDOG_INTERVAL,
    CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_PROFILE_UNMATCH_THRESHOLD,
    CONF_DEVICE_TYPE,
    CONF_START_DURATION_THRESHOLD,
    CONF_RUNNING_DEAD_ZONE,
    CONF_END_REPEAT_COUNT,
    CONF_MIN_OFF_GAP,
    CONF_START_ENERGY_THRESHOLD,
    CONF_END_ENERGY_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_SAMPLING_INTERVAL,
    CONF_SAVE_DEBUG_TRACES,
    CONF_DTW_BANDWIDTH,
    CONF_EXTERNAL_END_TRIGGER_ENABLED,
    CONF_EXTERNAL_END_TRIGGER,
    SIGNAL_WASHER_UPDATE,
    NOTIFY_EVENT_START,
    NOTIFY_EVENT_FINISH,
    EVENT_CYCLE_STARTED,
    EVENT_CYCLE_ENDED,
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_PROFILE_DURATION_TOLERANCE,
    DEFAULT_INTERRUPTED_MIN_SECONDS,
    DEFAULT_ABRUPT_DROP_WATTS,
    DEFAULT_ABRUPT_DROP_RATIO,
    DEFAULT_ABRUPT_HIGH_LOAD_FACTOR,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_PROFILE_MATCH_THRESHOLD,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
    DEFAULT_SAMPLING_INTERVAL,
    DEFAULT_PROGRESS_RESET_DELAY,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_AUTO_MAINTENANCE,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_MAX_PAST_CYCLES,
    DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
    CONF_NOTIFY_TITLE,
    CONF_NOTIFY_ICON,
    CONF_NOTIFY_START_MESSAGE,
    CONF_NOTIFY_FINISH_MESSAGE,
    CONF_NOTIFY_PRE_COMPLETE_MESSAGE,
    DEFAULT_NOTIFY_TITLE,
    DEFAULT_NOTIFY_START_MESSAGE,
    DEFAULT_NOTIFY_FINISH_MESSAGE,
    DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE,

    DEFAULT_MAX_FULL_TRACES_UNLABELED,
    DEFAULT_DTW_BANDWIDTH,
    DEFAULT_WATCHDOG_INTERVAL,
    CONF_MATCH_PERSISTENCE,
    DEFAULT_MATCH_PERSISTENCE,
    DEFAULT_MATCH_REVERT_RATIO,
    DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_START_DURATION_THRESHOLD,
    DEFAULT_RUNNING_DEAD_ZONE,
    DEFAULT_END_REPEAT_COUNT,
    DEFAULT_MIN_OFF_GAP,
    DEFAULT_MIN_OFF_GAP_BY_DEVICE,
    DEFAULT_MAX_DEFERRAL_SECONDS,
    DEVICE_SMOOTHING_THRESHOLDS,
    DEVICE_COMPLETION_THRESHOLDS,
    STATE_RUNNING,
    STATE_OFF,
    STATE_STARTING,
    STATE_PAUSED,
    STATE_ENDING,
    STATE_IDLE,
    STATE_UNKNOWN,
)
from .cycle_detector import CycleDetector, CycleDetectorConfig
from .learning import LearningManager
from .profile_store import ProfileStore, decompress_power_data
from .recorder import CycleRecorder

_LOGGER = logging.getLogger(__name__)


def _pn_create(
    hass: HomeAssistant,
    message: str,
    *,
    title: str | None = None,
    notification_id: str | None = None,
) -> None:
    """Best-effort persistent notification creation.

    Tests stub out the entire `homeassistant` module, so we can't import
    `homeassistant.components.persistent_notification` here.
    """
    try:
        components = getattr(cast(Any, hass), "components", None)
        pn = getattr(cast(Any, components), "persistent_notification", None)
        if pn is None:
            return
        result = pn.async_create(message, title=title, notification_id=notification_id)
        if inspect.iscoroutine(result):
            hass.async_create_task(result)
    except Exception:
        return


class WashDataManager:
    """Manages a single washing machine instance."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the manager."""
        self.hass = hass
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id

        # Prioritize options -> data for power sensor (allows changing it)
        self.power_sensor_entity_id = config_entry.options.get(
            CONF_POWER_SENSOR, config_entry.data.get(CONF_POWER_SENSOR)
        )
        self.device_type = config_entry.options.get(
            CONF_DEVICE_TYPE,
            config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
        )

        # Initialize attributes to satisfy pylint
        self._off_delay = float(DEFAULT_OFF_DELAY)
        self._no_update_active_timeout = float(DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT)
        self._low_power_no_update_timeout = 3600.0 # Default 1h
        self._notify_before_end_minutes = float(DEFAULT_NOTIFY_BEFORE_END_MINUTES)
        self._notify_service = ""
        self._notify_events = []

        # State
        self._current_power = 0.0
        self._last_reading_time: datetime | None = None
        self._last_real_reading_time: datetime | None = None # Track last real sensor update
        self._noise_events = []
        self._noise_max_powers = []
        self._last_match_result = None
        self._last_phase_estimate_time = None
        self._sample_intervals = []
        self._sample_interval_stats = {}
        self._matching_task = None
        self._last_state_save = 0.0
        self._last_cycle_end_time: datetime | None = None
        self._remove_state_expiry_timer = None

        # Components
        match_threshold = config_entry.options.get(
            CONF_PROFILE_MATCH_THRESHOLD, DEFAULT_PROFILE_MATCH_THRESHOLD
        )
        unmatch_threshold = config_entry.options.get(
            CONF_PROFILE_UNMATCH_THRESHOLD, DEFAULT_PROFILE_UNMATCH_THRESHOLD
        )
        self._unmatch_threshold = unmatch_threshold

        self.profile_store = ProfileStore(
            hass,
            self.entry_id,
            min_duration_ratio=config_entry.options.get(
                CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
            ),
            max_duration_ratio=config_entry.options.get(
                CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
            ),
            save_debug_traces=config_entry.options.get(CONF_SAVE_DEBUG_TRACES, False),
            match_threshold=match_threshold,
            unmatch_threshold=unmatch_threshold,
        )
        self.profile_store.dtw_bandwidth = float(
            config_entry.options.get(CONF_DTW_BANDWIDTH, DEFAULT_DTW_BANDWIDTH)
        )
        self.learning_manager = LearningManager(hass, self.entry_id, self.profile_store)
        self.recorder = CycleRecorder(hass, self.entry_id)

        # Priority: Options > Data > Default
        min_power = config_entry.options.get(
            CONF_MIN_POWER, config_entry.data.get(CONF_MIN_POWER, DEFAULT_MIN_POWER)
        )
        off_delay = config_entry.options.get(
            CONF_OFF_DELAY, config_entry.data.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY)
        )
        progress_reset_delay = config_entry.options.get(
            CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY
        )
        self._no_update_active_timeout = float(
            config_entry.options.get(
                CONF_NO_UPDATE_ACTIVE_TIMEOUT,
                DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
            )
        )
        self._low_power_no_update_timeout = float(
            config_entry.options.get(CONF_LOW_POWER_NO_UPDATE_TIMEOUT, 3600.0)
        )
        self._off_delay = float(config_entry.options.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY))
        self._learning_confidence = config_entry.options.get(
            CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE
        )
        self._duration_tolerance = config_entry.options.get(
            CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE
        )
        self._auto_label_confidence = config_entry.options.get(
            CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE
        )

        self._profile_match_interval = int(
            config_entry.options.get(
                CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
            )
        )
        self._notify_before_end_minutes = int(
            config_entry.options.get(
                CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES
            )
        )

        # Advanced options
        smoothing_window = int(config_entry.options.get("smoothing_window", 5))
        interrupted_min_seconds = int(
            config_entry.options.get("interrupted_min_seconds", 150)
        )
        abrupt_drop_watts = float(config_entry.options.get("abrupt_drop_watts", 500.0))
        abrupt_drop_ratio = float(config_entry.options.get("abrupt_drop_ratio", 0.6))
        abrupt_high_load_factor = float(
            config_entry.options.get("abrupt_high_load_factor", 5.0)
        )

        # Get device specific default for completion threshold
        device_default_completion = DEVICE_COMPLETION_THRESHOLDS.get(
            self.device_type, DEFAULT_COMPLETION_MIN_SECONDS
        )
        completion_min_seconds = int(
            config_entry.options.get(
                CONF_COMPLETION_MIN_SECONDS, device_default_completion
            )
        )

        start_duration_threshold = float(
            config_entry.options.get(
                CONF_START_DURATION_THRESHOLD, DEFAULT_START_DURATION_THRESHOLD
            )
        )
        running_dead_zone = int(
            config_entry.options.get(CONF_RUNNING_DEAD_ZONE, DEFAULT_RUNNING_DEAD_ZONE)
        )
        end_repeat_count = int(
            config_entry.options.get(CONF_END_REPEAT_COUNT, DEFAULT_END_REPEAT_COUNT)
        )

        _LOGGER.info(
            "Manager init: min_power=%sW, off_delay=%ss, type=%s",
            min_power,
            off_delay,
            self.device_type,
        )

        config = CycleDetectorConfig(
            min_power=float(min_power),
            off_delay=int(off_delay),
            smoothing_window=smoothing_window,
            interrupted_min_seconds=interrupted_min_seconds,
            abrupt_drop_watts=abrupt_drop_watts,
            abrupt_drop_ratio=abrupt_drop_ratio,
            abrupt_high_load_factor=abrupt_high_load_factor,
            completion_min_seconds=completion_min_seconds,
            start_duration_threshold=start_duration_threshold,
            running_dead_zone=running_dead_zone,
            end_repeat_count=end_repeat_count,
            min_off_gap=int(
                config_entry.options.get(
                    CONF_MIN_OFF_GAP,
                    DEFAULT_MIN_OFF_GAP_BY_DEVICE.get(
                        self.device_type, DEFAULT_MIN_OFF_GAP
                    ),
                )
            ),
            start_energy_threshold=float(
                config_entry.options.get(CONF_START_ENERGY_THRESHOLD, 0.0)
            ),
            end_energy_threshold=float(
                config_entry.options.get(CONF_END_ENERGY_THRESHOLD, 0.0)
            ),
            start_threshold_w=float(
                config_entry.options.get(
                    CONF_START_THRESHOLD_W,
                    float(min_power) + max(1.0, 0.1 * float(min_power)),
                )
            ),
            stop_threshold_w=float(
                config_entry.options.get(
                    CONF_STOP_THRESHOLD_W,
                    float(min_power) * 0.6 if float(min_power) > 0 else 2.0,
                )
            ),
            min_duration_ratio=float(
                config_entry.options.get(
                    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE.get(
                        self.device_type, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO
                    ),
                )
            ),
            match_interval=int(
                config_entry.options.get(
                    CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
                )
            ),
        )
        self._config = config


        def profile_matcher_wrapper(
            readings: list[tuple[datetime, float]],
        ) -> tuple[str | None, float, float, str | None, bool] | None:
            """Wraps profile store matching logic with detector callback signature.

            Returns: None (async offload)
            """
            # Manual program override
            if self._manual_program_active and self._current_program:
                dur = self._matched_profile_duration or 0.0
                return (self._current_program, 1.0, dur, "Manual", False)

            if not readings:
                return (None, 0.0, 0.0, None, False)

            # Snapshotted for thread safety indirectly by task logic
            # We don't need a wrapper task if we unify with _update_estimates matching
            # but for now let's keep the detector callback as a trigger
            self.hass.async_create_task(self._async_perform_combined_matching(readings))
            return None

        self.detector = CycleDetector(
            config,
            self._on_state_change,
            self._on_cycle_end,
            profile_matcher=profile_matcher_wrapper,
        )

        self._remove_listener = None
        self._remove_external_trigger_listener = None  # External cycle end trigger
        self._remove_watchdog = None
        self._watchdog_interval = int(
            config_entry.options.get(CONF_WATCHDOG_INTERVAL, DEFAULT_WATCHDOG_INTERVAL)
        )
        self._match_persistence = int(
            config_entry.options.get(CONF_MATCH_PERSISTENCE, DEFAULT_MATCH_PERSISTENCE)
        )
        self._sampling_interval = float(
            config_entry.options.get(CONF_SAMPLING_INTERVAL, DEFAULT_SAMPLING_INTERVAL)
        )
        self._noise_events_threshold = int(
            config_entry.options.get(
                CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
                DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
            )
        )
        self._current_program = "off"
        self._time_remaining: float | None = None
        self._total_duration: float | None = None
        self._last_total_duration_update: datetime | None = None
        self._cycle_progress: float = 0.0
        self._smoothed_progress: float = 0.0  # Smoothed progress tracking for EMA
        self._cycle_completed_time: datetime | None = None  # Track when cycle finished
        self._progress_reset_delay: int = int(
            progress_reset_delay
        )  # Reset progress after idle
        self._last_reading_time: datetime | None = None
        self._current_power: float = 0.0
        self._last_estimate_time: datetime | None = None
        self._last_match_ambiguous: bool = False
        self._matched_profile_duration: float | None = None
        self._last_match_confidence: float = 0.0
        # Sample interval tracking (seconds) for adaptive timing
        # Profile matching duration tolerance (0.25 = ±25%)
        self._profile_duration_tolerance: float = float(
            config_entry.options.get("profile_duration_tolerance", 0.25)
        )


        self._remove_maintenance_scheduler = None
        self._profile_sample_repair_stats: dict[str, int] | None = None

        self._last_suggestion_update: datetime | None = None

        self._manual_program_active: bool = False
        self._notified_pre_completion: bool = False
        self._last_match_result: Any = None  # Stores full MatchResult object
        self._score_history: dict[str, list[float]] = {}  # Tracks recent scores for trend analysis
        self._match_persistence_counter: dict[str, int] = {}  # Tracks consecutive matches
        self._unmatch_persistence_counter: int = 0  # Tracks consecutive low-confidence matches
        self._current_match_candidate: str | None = None  # Pending profile name

    async def _async_perform_combined_matching(
        self, readings: list[tuple[datetime, float]]
    ) -> None:
        """PRIMARY matching task: Updates both Manager and Detector using best method."""
        _LOGGER.debug(
            "Matching trigger: readings=%d, task_exists=%s",
            len(readings) if readings else 0,
            getattr(self, "_matching_task", None) is not None
        )
        # Prevent concurrent matching tasks
        if getattr(self, "_matching_task", None) and not self._matching_task.done():
            _LOGGER.debug("Matching skipped: previous task still running")
            return

        try:
            if not readings:
                _LOGGER.debug("Matching skipped: no readings")
                return

            self._matching_task = self.hass.async_create_task(self._async_do_perform_matching(readings))
        except Exception as e:
            _LOGGER.error("Perform combined matching trigger failed: %s", e)

    async def _async_do_perform_matching(self, readings: list[tuple[datetime, float]]) -> None:
        """Inner task to handle actual matching logic."""
        try:
            end_time = readings[-1][0]
            start_time = readings[0][0]
            current_duration = (end_time - start_time).total_seconds()

            # 1. RUN BETTER ASYNC MATCHING
            result = await self.profile_store.async_match_profile(
                 readings, 
                 current_duration
            )

            # 2. UPDATE MANAGER STATE (Estimates, Program Name, etc.)
            self._last_match_result = result
            self._last_match_ambiguous = result.is_ambiguous
            
            profile_name = result.best_profile
            confidence = result.confidence
            matched_duration = result.expected_duration
            phase_name = result.matched_phase
            
            # --- Switching Logic (Temporal Persistence) ---
            should_switch = False
            switch_reason = ""
            
            # Identify candidate score from results
            candidate_score = confidence
            current_program_score = 0.0
            for c in result.candidates:
                if c.get("name") == self._current_program:
                    current_program_score = c.get("score", 0.0)
                    break
            
            # CASE: Divergence Detection (Score Drop)
            # If current matched program has a significant drop from its own peak score,
            # we should consider unmatching it even if it's still the "best" candidate.
            if (
                self._current_program not in ("detecting...", "off", "starting", "unknown")
                and profile_name == self._current_program
            ):
                history = self._score_history.get(self._current_program, [])
                if len(history) > 3:
                    peak_score = max(history)
                    # If score drops by more than 40% from peak AND is below threshold, unmatch.
                    # This catches divergence faster than waiting for fixed unmatch_threshold.
                    if confidence < peak_score * (1.0 - DEFAULT_MATCH_REVERT_RATIO):
                        self._unmatch_persistence_counter += 1
                        if self._unmatch_persistence_counter >= self._match_persistence:
                            self._current_program = "detecting..."
                            self._matched_profile_duration = None
                            self._unmatch_persistence_counter = 0
                            _LOGGER.info(
                                "Divergence detected for profile '%s' (confidence %.3f < 60%% of peak %.3f). "
                                "Reverting to detection.",
                                profile_name, confidence, peak_score
                            )
                            # Reset profile_name so Case 3 doesn't re-trigger
                            profile_name = "detecting..."
                    
            # Update persistence for the best profile
            if profile_name and profile_name != "detecting...":
                self._match_persistence_counter[profile_name] = self._match_persistence_counter.get(profile_name, 0) + 1
                
                # Check if this is the same candidate as before
                if profile_name != self._current_match_candidate:
                    # Reset counter for old candidate if it wasn't locked in
                    self._current_match_candidate = profile_name
                    self._match_persistence_counter[profile_name] = 1
            else:
                self._current_match_candidate = None

            is_persistent = profile_name and self._match_persistence_counter.get(profile_name, 0) >= self._match_persistence

            # Case 1: Initial Match from "detecting..."
            if (
                profile_name
                and confidence >= 0.15
                and not result.is_ambiguous
                and (not self._matched_profile_duration or self._current_program == "detecting...")
            ):
                if is_persistent:
                    should_switch = True
                    switch_reason = f"initial_match (persistent {self._match_persistence_counter[profile_name]}x)"
                else:
                    _LOGGER.debug(
                        "Match persistence: %s at %d/%d matches. Stay at detecting...",
                        profile_name, self._match_persistence_counter.get(profile_name, 0), self._match_persistence
                    )

            # Case 2: Mid-cycle override (different profile)
            elif (
                profile_name
                and self._current_program != profile_name
                and self._current_program not in ("detecting...", "off", "starting", "unknown")
            ):
                # High Confidence Override: Bypass persistence if match is VERY strong
                if confidence > 0.8 and (confidence - current_program_score) > 0.15:
                    should_switch = True
                    switch_reason = f"high_confidence_override ({confidence:.3f} vs {current_program_score:.3f})"
                
                # Normal Switch: Requires persistence AND either better score + trend
                elif is_persistent:
                    if confidence > current_program_score and self._analyze_trend(profile_name):
                        # Add a minimum score gap for mid-cycle switching (0.05) to prevent flapping
                        if (confidence - current_program_score) > 0.05:
                            should_switch = True
                            switch_reason = f"positive_trend_persistent ({confidence:.3f} > {current_program_score:.3f})"

            # Case 3: Unmatching (confidence drop)
            elif (
                self._current_program not in ("detecting...", "off", "starting", "unknown")
                and profile_name == self._current_program
                and confidence < self._unmatch_threshold
            ):
                self._unmatch_persistence_counter += 1
                is_unmatch_persistent = self._unmatch_persistence_counter >= self._match_persistence
                
                if is_unmatch_persistent:
                    self._current_program = "detecting..."
                    self._matched_profile_duration = None
                    self._unmatch_persistence_counter = 0
                    _LOGGER.info(
                        "Unmatched profile '%s' (confidence %.3f < threshold %.3f persistent %dx). "
                        "Reverting to detection.",
                        profile_name,
                        confidence,
                        self._unmatch_threshold,
                        self._match_persistence
                    )
                else:
                    _LOGGER.debug(
                        "Unmatch persistence: %s at %d/%d low-confidence matches. Stay at %s...",
                        profile_name, self._unmatch_persistence_counter, self._match_persistence, profile_name
                    )
            
            # Reset unmatch counter if confidence is healthy 
            # AND we didn't just detect a divergence
            elif (
                profile_name == self._current_program 
                and confidence >= self._unmatch_threshold
                and not (len(self._score_history.get(self._current_program, [])) > 3 and confidence < max(self._score_history[self._current_program]) * (1.0 - DEFAULT_MATCH_REVERT_RATIO))
            ):
                self._unmatch_persistence_counter = 0

            if should_switch:
                self._current_program = profile_name
                self._last_match_confidence = confidence
                self._unmatch_persistence_counter = 0 # Reset on switch
                if profile_name in self._match_persistence_counter:
                    self._match_persistence_counter[profile_name] = self._match_persistence # Lock it in
                
                avg_duration = float(matched_duration)
                self._matched_profile_duration = avg_duration if avg_duration > 0 else None
                _LOGGER.info(
                     "Switching to profile '%s' (reason: %s). Expected duration: %.0fs (%smin)",
                     profile_name, switch_reason, avg_duration, int(avg_duration / 60),
                )
            elif profile_name == self._current_program:
                # Same program, but update confidence for sensors
                self._last_match_confidence = confidence
            elif not self._matched_profile_duration:
                self._current_program = "detecting..."

            self._last_estimate_time = dt_util.now()
            
            # Update score history for all candidates to track trends
            for cand in result.candidates:
                cname = cand.get("name")
                if cname:
                    history = self._score_history.setdefault(cname, [])
                    history.append(float(cand.get("score", 0.0)))
                    if len(history) > 20:
                        history.pop(0)

            # Note: _update_remaining_only() and notify move to end of flow

            # 3. UPDATE DETECTOR (Envelopes, Deferral, State Transitions)
            current_matched = self.detector.matched_profile
            verified_pause = getattr(self.detector, "_verified_pause", False)
            current_power = readings[-1][1] if readings else 0.0

            # --- Envelope Verification for Mismatches & Pauses ---
            # ALWAYS check alignment if we have a match and power is low, 
            # to confirm if this is a legitimate pause or a mismatch.
            stop_thresh = float(self.detector.config.stop_threshold_w)
            if current_matched and current_power < stop_thresh:
                formatted = [(t.isoformat(), p) for t, p in readings]
                try:
                    is_confirmed, mapped_time, _ = (
                        await self.profile_store.async_verify_alignment(current_matched, formatted)
                    )
                except Exception as e: # pylint: disable=broad-exception-caught
                    _LOGGER.error(
                        "Alignment verification crashed for profile %s: %s",
                        current_matched, e, exc_info=True
                    )
                    is_confirmed = False
                    mapped_time = 0.0

                if is_confirmed:
                    if not verified_pause:
                        _LOGGER.info(
                            "Envelope verified expected low power phase for %s. Enabling verified pause.",
                            current_matched
                        )
                    verified_pause = True
                    # Smart Termination within Envelope block
                    try:
                        profile = self.profile_store.get_profile(current_matched)
                        if profile:
                            avg_dur = profile.get("avg_duration", 0)
                            if avg_dur > 0 and (mapped_time / avg_dur) > 0.95:
                                verified_pause = False
                                _LOGGER.info("Smart Termination: Near end of profile. Releasing pause lock.")
                    except Exception as e:
                        _LOGGER.debug("Smart Termination alignment verification failed: %s", e)
                        pass
                else:
                    if verified_pause:
                        _LOGGER.info(
                            "Envelope indicates UNEXPECTED low power for %s. Disabling verified pause.",
                            current_matched
                        )
                    verified_pause = False

            # --- High Power Clear ---
            stop_threshold = getattr(self.detector.config, "stop_threshold_w", 5.0)
                
            if current_power > stop_threshold * 10:
                verified_pause = False

            # --- Consistency Override ---
            # If envelope verified or mismatched, ensure manager program matches
            if profile_name != self._current_program and (verified_pause or result.is_confident_mismatch):
                if profile_name:
                    self._current_program = profile_name
                    self._last_match_confidence = confidence
                    # Try to fetch duration if we switched back to matched
                    try:
                        prof = self.profile_store.get_profile(profile_name)
                        if prof:
                            self._matched_profile_duration = float(prof.get("avg_duration", 0))
                    except Exception as e:
                        _LOGGER.debug("Failed to fetch profile duration on switch: %s", e)
                        pass
                else:
                    self._current_program = "detecting..."
                    self._matched_profile_duration = None

            # --- HEURISTICS (Descriptive Phases) ---
            if not phase_name:
                if self.device_type == "dishwasher" and self.detector.is_waiting_low_power():
                    phase_name = "Drying"
                elif self.device_type == "washing_machine" and current_power > 200:
                    phase_name = "Spinning"
                elif self.device_type == "washing_machine" and self.detector.is_waiting_low_power():
                    phase_name = "Rinsing/Soaking"
                elif self.device_type == "ev":
                    if current_power > 100:
                        phase_name = "Charging"
                    elif self.detector.is_waiting_low_power():
                        phase_name = "Maintenance"

            # Push updates to detector
            self.detector.set_verified_pause(verified_pause)
            self.detector.update_match(
                (profile_name, confidence, matched_duration, phase_name, result.is_confident_mismatch)
            )

            # --- LOGGING (Unified) ---
            _LOGGER.info(
                "Profile match attempt: name=%s, confidence=%.3f, duration=%.0fs, samples=%d",
                profile_name, confidence, current_duration, len(readings),
            )

            self._update_remaining_only()
            self._check_pre_completion_notification()
            self._notify_update()

        except Exception as e:
            _LOGGER.error("Perform combined matching failed: %s", e, exc_info=True)

    @property
    def top_candidates(self) -> list[dict[str, Any]]:
        """Return a lightweight list of top candidates from the last match."""
        if not self._last_match_result:
            return []
            
        # Get raw list from ranking (best) or candidates
        raw_list = []
        if hasattr(self._last_match_result, "ranking") and self._last_match_result.ranking:
            raw_list = self._last_match_result.ranking
        elif hasattr(self._last_match_result, "candidates"):
            raw_list = self._last_match_result.candidates
            
        # SANITIZE: Remove heavy power arrays before sending to Home Assistant attributes
        sanitized = []
        for cand in raw_list[:5]:
            sanitized.append({
                "name": cand.get("name"),
                "score": round(float(cand.get("score", 0.0)), 3),
                "profile_duration": cand.get("profile_duration"),
                # Explicitly exclude "current" and "sample" keys which are big lists
            })
        return sanitized

    @property
    def phase_description(self) -> str:
        """Return a description of the current phase."""
        if self._last_match_result and self._last_match_result.matched_phase:
            return self._last_match_result.matched_phase
        return self.detector.state

    @property
    def match_ambiguity(self) -> bool:
        """Return True if the last match was ambiguous."""
        if self._last_match_result and hasattr(self._last_match_result, "is_ambiguous"):
            return self._last_match_result.is_ambiguous
        return False

    # Note: last_match_details property is defined later in the class
    # It returns MatchResult from _last_match_result
    async def _attempt_state_restoration(self) -> None:
        """Attempt to restore active cycle state from storage."""
        active_snapshot = self.profile_store.get_active_cycle()

        # Check current power state first
        state = self.hass.states.get(self.power_sensor_entity_id)
        current_power = 0.0
        power_is_valid = False

        if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            try:
                current_power = float(state.state)
                power_is_valid = True
            except (ValueError, TypeError):
                # Power sensor state is not numeric during restoration; treat as 0W
                _LOGGER.debug(
                    "Power sensor %s state %r is not numeric during restoration; "
                    "treating as 0W and not restoring by power",
                    self.power_sensor_entity_id,
                    getattr(state, "state", None),
                )

        should_restore = False
        active_snapshot_to_restore = active_snapshot

        # Helper to check if a snapshot is viable
        def is_viable_restore(last_save_time: datetime) -> bool:
            now = dt_util.now()
            # Handle timezone mismatch gracefully
            if last_save_time.tzinfo is None:
                # Assume naive means local system time, convert to aware
                last_save_time = last_save_time.replace(tzinfo=now.tzinfo)

            age = (now - last_save_time).total_seconds()

            # Unconditional restore window (30 mins)
            if age < 1800:
                return True
            # Extended window if power is confirmed HIGH (60 mins)
            if (
                age < 3600
                and power_is_valid
                and current_power >= self._config.min_power
            ):
                return True
            return False

        last_save = self.profile_store.get_last_active_save()
        if last_save and last_save.tzinfo is None:
            # Normalize naive legacy timestamps to system time
            last_save = last_save.replace(tzinfo=dt_util.now().tzinfo)

        if active_snapshot and last_save and is_viable_restore(last_save):
            should_restore = True
            age = (dt_util.now() - last_save).total_seconds()
            age = (dt_util.now() - last_save).total_seconds()
            _LOGGER.info(
                "Found recently saved active cycle (last_save=%s, age=%.0fs), restoring...",
                last_save,
                age
            )
            # strict extension logic unless the user wants to enforce it.
            active_snapshot_to_restore["sub_state"] = (
                active_snapshot_to_restore.get("sub_state") or "Restored"
            )
            # NOTE: We disable dynamic min duration enforcement on recovery since we
            # might have missed data
            active_snapshot_to_restore["dynamic_min_duration"] = None

        # FALLBACK: Resurrection Logic
        if not should_restore:
            past_cycles = self.profile_store.get_past_cycles()
            if past_cycles:
                last_cycle = past_cycles[-1]
                last_end_str = last_cycle.get("end_time")
                if last_end_str:
                    last_end = dt_util.parse_datetime(last_end_str)
                    if last_end:
                        gap = (dt_util.now() - last_end).total_seconds()
                        is_recent = gap < 1200  # 20 mins
                        status = last_cycle.get("status")

                        if is_recent and status != "completed":
                            _LOGGER.info(
                                "Found recent interrupted cycle in history "
                                "(id=%s, gap=%.0fs). Resurrecting...",
                                last_cycle["id"],
                                gap,
                            )
                            try:
                                power_data = decompress_power_data(last_cycle)
                                if power_data:
                                    active_snapshot_to_restore = {
                                        # Reconstruct basic running state
                                        "state": "running",
                                        "sub_state": "Resurrected",
                                        "current_cycle_start": last_cycle["start_time"],
                                        "last_active_time": last_cycle["end_time"],
                                        "low_power_start": None,
                                        "cycle_max_power": (
                                            max([p for _, p in power_data])
                                            if power_data
                                            else 0
                                        ),
                                        "power_readings": power_data,
                                        "ma_buffer": (
                                            [p for _, p in power_data[-10:]]
                                            if power_data
                                            else []
                                        ),
                                        "end_condition_count": 0,
                                        "extension_count": 0,
                                        "dynamic_min_duration": None,
                                        "matched_profile": last_cycle.get(
                                            "profile_name"
                                        ),
                                    }
                                    should_restore = True
                                    past_cycles.pop()
                                    await self.profile_store.async_save()
                            except Exception as e:
                                _LOGGER.error("Failed to resurrect cycle: %s", e)

        if should_restore and active_snapshot_to_restore:
            try:
                self.detector.restore_state_snapshot(active_snapshot_to_restore)

                # Restore if in any active state (Running, Paused, Ending)
                if self.detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
                    # Restore manual program flag if present
                    self._manual_program_active = active_snapshot_to_restore.get(
                        "manual_program", False
                    )

                    # If we restored into a low-power state, ensure we don't
                    # immediately quit. For now we just log this; the cycle
                    # detector's off_delay will handle actual shutdown.
                    if power_is_valid and current_power < self._config.min_power:
                        _LOGGER.debug(
                            "Restored active cycle in low-power state "
                            "(power=%.2fW < min_power=%.2fW); waiting for "
                            "detector off_delay before marking as finished",
                            current_power,
                            self._config.min_power,
                        )

                    if self.detector.matched_profile:
                        self._current_program = self.detector.matched_profile
                        _LOGGER.info(
                            "Restored/Resurrected washer cycle with profile: %s",
                            self._current_program,
                        )
                    else:
                        self._current_program = "detecting..."
                    self._start_watchdog()
                else:
                    await self.profile_store.async_clear_active_cycle()
            except Exception as err:
                _LOGGER.warning("Failed to restore active cycle: %s, clearing", err)
                await self.profile_store.async_clear_active_cycle()
        else:
            if last_save:
                age = (dt_util.now() - last_save).total_seconds()
                _LOGGER.info("Active cycle too stale (age=%.0fs), clearing", age)
            await self.profile_store.async_clear_active_cycle()

    async def async_setup(self) -> None:
        """Set up the manager."""
        await self.profile_store.async_load()
        # Apply configurable duration tolerance to profile store
        try:
            self.profile_store.set_duration_tolerance(self._profile_duration_tolerance)
            self.profile_store.set_retention_limits(
                max_past_cycles=int(
                    self.config_entry.options.get(
                        CONF_MAX_PAST_CYCLES, DEFAULT_MAX_PAST_CYCLES
                    )
                ),
                max_full_traces_per_profile=int(
                    self.config_entry.options.get(
                        CONF_MAX_FULL_TRACES_PER_PROFILE,
                        DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
                    )
                ),
                max_full_traces_unlabeled=int(
                    self.config_entry.options.get(
                        CONF_MAX_FULL_TRACES_UNLABELED,
                        DEFAULT_MAX_FULL_TRACES_UNLABELED,
                    )
                ),
            )
        except Exception:
            pass

        # Repair broken sample_cycle_id references (can happen after aggressive retention)
        try:
            stats = await self.profile_store.async_repair_profile_samples()
            self._profile_sample_repair_stats = stats
            if stats.get("profiles_repaired", 0) or stats.get(
                "cycles_labeled_as_sample", 0
            ):
                _LOGGER.warning(
                    "Repaired profile sample references for %s: %s",
                    self.entry_id,
                    stats,
                )
                await self.profile_store.async_save()
        except Exception:
            _LOGGER.exception(
                "Failed repairing profile sample references for %s", self.entry_id
            )

        # Subscribe to power sensor updates
        self._remove_listener = async_track_state_change_event(
            self.hass, [self.power_sensor_entity_id], self._async_power_changed
        )

        # Attempt to restore state (BEFORE starting listener)
        await self._attempt_state_restoration()

        # Restore last cycle end time to ensure ghost cycle suppression works after restart
        try:
            cycles = self.profile_store.get_past_cycles()
            if cycles:
                # Find last completed cycle with a valid end time
                for cycle in reversed(cycles):
                    if cycle.get("end_time") and cycle.get("status") == "completed":
                        ts = dt_util.parse_datetime(cycle["end_time"])
                        if ts:
                            self._last_cycle_end_time = ts
                            _LOGGER.debug("Restored last cycle end time: %s", ts)
                            break
        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Failed to restore last cycle end time")

        # Load recorder state
        await self.recorder.async_load()

        # Force initial update from current state (in case it's already stable)
        state = self.hass.states.get(self.power_sensor_entity_id)
        if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            try:
                power = float(state.state)
                now = dt_util.now()
                self.detector.process_reading(power, now)
            except (ValueError, TypeError):
                pass

        # Trigger migration/compression of old cycle format
        # This is safe to run repeatedly (it skips already compressed cycles)
        await self.profile_store.async_migrate_cycles_to_compressed()

        # Subscribe to external cycle end trigger (if enabled)
        await self._setup_external_end_trigger()

    async def async_reload_config(self, config_entry: ConfigEntry) -> None:
        """
        Reload configuration options without interrupting running cycle detection.

        Updates detector config in-place.
        Handles Power Sensor entity change by reconnecting listener.
        """
        _LOGGER.info("Reloading configuration for %s", self.entry_id)
        # Replace reference
        self.config_entry = config_entry

        # Check if power sensor changed
        new_sensor = config_entry.options.get(
            CONF_POWER_SENSOR, config_entry.data.get(CONF_POWER_SENSOR)
        )
        if new_sensor and new_sensor != self.power_sensor_entity_id:
            # Block sensor changes when a cycle is active to prevent inconsistent state
            d_state = self.detector.state
            _LOGGER.debug(
                "Reloading config: detector.state=%r (type=%s), RUNNING=%r",
                d_state,
                type(d_state),
                STATE_RUNNING,
            )
            if d_state == STATE_RUNNING:
                _LOGGER.warning(
                    "Cannot change power sensor from %s to %s while a cycle "
                    "is active. Please wait for the current cycle to complete "
                    "before changing the power sensor.",
                    self.power_sensor_entity_id,
                    new_sensor,
                )
                # Skip sensor change but continue with other config updates
                return

            _LOGGER.info(
                "Power sensor changed: %s -> %s", self.power_sensor_entity_id, new_sensor
            )
            self.power_sensor_entity_id = new_sensor
            # Remove old listener
            if self._remove_listener:
                self._remove_listener()
            # Attach new listener
            self._remove_listener = async_track_state_change_event(
                self.hass, [self.power_sensor_entity_id], self._async_power_changed
            )
            # Force update from new sensor
            state = self.hass.states.get(self.power_sensor_entity_id)
            if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                try:
                    power = float(state.state)
                    self.detector.process_reading(power, dt_util.now())
                except ValueError:
                    _LOGGER.debug(
                        "Initial power value for %s after config reload is not numeric: %r",
                        self.power_sensor_entity_id,
                        state.state,
                    )

        # Update device type
        self.device_type = config_entry.options.get(
            CONF_DEVICE_TYPE,
            config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
        )

        # Update detector config in-place
        old_min_power = self.detector.config.min_power
        old_off_delay = self.detector.config.off_delay
        old_smoothing = self.detector.config.smoothing_window
        old_interrupted_min = self.detector.config.interrupted_min_seconds
        old_abrupt_drop_watts = self.detector.config.abrupt_drop_watts
        old_abrupt_drop_ratio = self.detector.config.abrupt_drop_ratio
        old_abrupt_high_load = self.detector.config.abrupt_high_load_factor

        # Get new values from config
        new_min_power = float(
            config_entry.options.get(CONF_MIN_POWER, DEFAULT_MIN_POWER)
        )
        new_off_delay = int(config_entry.options.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY))
        new_smoothing = int(
            config_entry.options.get(CONF_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW)
        )
        new_interrupted_min = int(
            config_entry.options.get(
                CONF_INTERRUPTED_MIN_SECONDS, DEFAULT_INTERRUPTED_MIN_SECONDS
            )
        )
        new_abrupt_drop_watts = float(
            config_entry.options.get(CONF_ABRUPT_DROP_WATTS, DEFAULT_ABRUPT_DROP_WATTS)
        )
        new_abrupt_drop_ratio = float(
            config_entry.options.get(CONF_ABRUPT_DROP_RATIO, DEFAULT_ABRUPT_DROP_RATIO)
        )
        self.detector.config.match_interval = int(
            config_entry.options.get(
                CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
            )
        )
        self.profile_store.dtw_bandwidth = float(
            config_entry.options.get(CONF_DTW_BANDWIDTH, DEFAULT_DTW_BANDWIDTH)
        )
        new_abrupt_high_load = float(
            config_entry.options.get(
                CONF_ABRUPT_HIGH_LOAD_FACTOR, DEFAULT_ABRUPT_HIGH_LOAD_FACTOR
            )
        )

        # Device default
        dev_def = DEVICE_COMPLETION_THRESHOLDS.get(
            self.device_type, DEFAULT_COMPLETION_MIN_SECONDS
        )
        new_completion_min = int(
            config_entry.options.get(CONF_COMPLETION_MIN_SECONDS, dev_def)
        )

        new_start_threshold = float(
            config_entry.options.get(
                CONF_START_DURATION_THRESHOLD, DEFAULT_START_DURATION_THRESHOLD
            )
        )
        new_running_dead_zone = int(
            config_entry.options.get(CONF_RUNNING_DEAD_ZONE, DEFAULT_RUNNING_DEAD_ZONE)
        )
        new_end_repeat_count = int(
            config_entry.options.get(CONF_END_REPEAT_COUNT, DEFAULT_END_REPEAT_COUNT)
        )

        # Power Hysteresis Thresholds
        new_start_threshold_w = float(
            config_entry.options.get(
                CONF_START_THRESHOLD_W,
                float(new_min_power) + max(1.0, 0.1 * float(new_min_power)),
            )
        )
        new_stop_threshold_w = float(
            config_entry.options.get(
                CONF_STOP_THRESHOLD_W,
                max(0.0, float(new_min_power) - max(0.5, 0.1 * float(new_min_power))),
            )
        )

        # Apply all detector config updates
        self.detector.config.min_power = new_min_power
        self.detector.config.off_delay = new_off_delay
        self.detector.config.smoothing_window = new_smoothing
        self.detector.config.interrupted_min_seconds = new_interrupted_min
        self.detector.config.abrupt_drop_watts = new_abrupt_drop_watts
        self.detector.config.abrupt_drop_ratio = new_abrupt_drop_ratio
        self.detector.config.abrupt_high_load_factor = new_abrupt_high_load
        self.detector.config.completion_min_seconds = new_completion_min
        self.detector.config.start_duration_threshold = new_start_threshold
        self.detector.config.running_dead_zone = new_running_dead_zone
        self.detector.config.end_repeat_count = new_end_repeat_count
        self.detector.config.start_threshold_w = new_start_threshold_w
        self.detector.config.stop_threshold_w = new_stop_threshold_w

        if (
            old_min_power != new_min_power
            or old_off_delay != new_off_delay
            or old_smoothing != new_smoothing
            or old_interrupted_min != new_interrupted_min
            or old_abrupt_drop_watts != new_abrupt_drop_watts
            or old_abrupt_drop_ratio != new_abrupt_drop_ratio
            or old_abrupt_high_load != new_abrupt_high_load
        ):
            _LOGGER.info(
                "Updated detector config: min_power %.1fW→%.1fW, off_delay %ds→%ds, "
                "smoothing %d→%d, interrupted_min %ds→%ds, abrupt_drop %.0fW→%.0fW, "
                "abrupt_ratio %.2f→%.2f, high_load %.1f→%.1f",
                old_min_power,
                new_min_power,
                old_off_delay,
                new_off_delay,
                old_smoothing,
                new_smoothing,
                old_interrupted_min,
                new_interrupted_min,
                old_abrupt_drop_watts,
                new_abrupt_drop_watts,
                old_abrupt_drop_ratio,
                new_abrupt_drop_ratio,
                old_abrupt_high_load,
                new_abrupt_high_load,
            )

        # Update profile matching parameters
        old_min_ratio, old_max_ratio = self.profile_store.get_duration_ratio_limits()

        new_min_ratio = float(
            config_entry.options.get(
                CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
            )
        )
        new_max_ratio = float(
            config_entry.options.get(
                CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
            )
        )

        if old_min_ratio != new_min_ratio or old_max_ratio != new_max_ratio:
            self.profile_store.set_duration_ratio_limits(
                min_ratio=new_min_ratio, max_ratio=new_max_ratio
            )
            _LOGGER.info(
                "Updated duration ratios: min %.2f→%.2f, max %.2f→%.2f",
                old_min_ratio,
                new_min_ratio,
                old_max_ratio,
                new_max_ratio,
            )

        # Update match interval
        old_interval = self._profile_match_interval
        new_interval = int(
            config_entry.options.get(
                CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
            )
        )
        if old_interval != new_interval:
            self._profile_match_interval = new_interval
            _LOGGER.info("Updated match interval: %ds→%ds", old_interval, new_interval)

        # Update other configurable options
        self._profile_duration_tolerance = float(
            config_entry.options.get(
                CONF_PROFILE_DURATION_TOLERANCE, DEFAULT_PROFILE_DURATION_TOLERANCE
            )
        )


        # Update notification settings
        self._notify_service = config_entry.options.get(CONF_NOTIFY_SERVICE)
        self._notify_events = config_entry.options.get(CONF_NOTIFY_EVENTS, [])
        self._notify_before_end_minutes = int(
            config_entry.options.get(
                CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES
            )
        )

        # Re-subscribe to external cycle end trigger
        await self._setup_external_end_trigger()

        _LOGGER.info("Configuration reloaded successfully")

        # Trigger entity updates to reflect any changes
        async_dispatcher_send(self.hass, f"ha_washdata_update_{self.entry_id}")

        if self.detector:
            self.detector.config.profile_duration_tolerance = self._profile_duration_tolerance

        # Schedule midnight maintenance if enabled
        await self._setup_maintenance_scheduler()

        # Update sampling interval
        old_sampling = self._sampling_interval
        new_sampling = float(
            config_entry.options.get(CONF_SAMPLING_INTERVAL, DEFAULT_SAMPLING_INTERVAL)
        )
        if old_sampling != new_sampling:
            self._sampling_interval = new_sampling
            _LOGGER.info(
                "Updated sampling interval: %.1fs -> %.1fs", old_sampling, new_sampling
            )

        # RESTORE STATE (only if recent enough, otherwise treat as stale)
        await self._attempt_state_restoration()

        _LOGGER.info("Configuration reloaded successfully")

    async def async_shutdown(self) -> None:
        """Shutdown."""
        if self._remove_listener:
            self._remove_listener()
        if self._remove_external_trigger_listener:
            self._remove_external_trigger_listener()
        if self._remove_watchdog:
            self._remove_watchdog()
        if (
            hasattr(self, "_remove_state_expiry_timer")
            and self._remove_state_expiry_timer
        ):
            self._remove_state_expiry_timer()
        if self._remove_maintenance_scheduler:
            self._remove_maintenance_scheduler()

        # Save active state before shutdown
        if self.detector.state == "running":
            snapshot = self.detector.get_state_snapshot()
            snapshot["manual_program"] = self._manual_program_active
            await self.profile_store.async_save_active_cycle(snapshot)

        self._last_reading_time = None

    async def _setup_external_end_trigger(self) -> None:
        """Set up listener for external cycle end trigger binary sensor."""
        # Remove existing listener if any
        if self._remove_external_trigger_listener:
            self._remove_external_trigger_listener()
            self._remove_external_trigger_listener = None

        # Check if enabled
        enabled = self.config_entry.options.get(
            CONF_EXTERNAL_END_TRIGGER_ENABLED, False
        )
        if not enabled:
            _LOGGER.debug("External cycle end trigger is disabled")
            return

        # Get entity ID
        entity_id = self.config_entry.options.get(CONF_EXTERNAL_END_TRIGGER, "")
        if not entity_id:
            _LOGGER.debug("External cycle end trigger: no entity configured")
            return

        _LOGGER.info(
            "Setting up external cycle end trigger: %s", entity_id
        )

        # Subscribe to state changes
        self._remove_external_trigger_listener = async_track_state_change_event(
            self.hass, [entity_id], self._handle_external_trigger_change
        )

    @callback
    def _handle_external_trigger_change(self, event: Event) -> None:
        """Handle external trigger sensor state change."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")

        if new_state is None:
            return

        # Only trigger on transition to "on"
        new_value = new_state.state
        old_value = old_state.state if old_state else None

        if new_value == "on" and old_value != "on":
            _LOGGER.info(
                "External cycle end trigger activated by %s",
                event.data.get("entity_id")
            )
            # End cycle with "completed" status (not interrupted)
            if self.detector.state != STATE_OFF:
                self.detector.user_stop()
                _LOGGER.info("Cycle completed via external trigger")

    async def _setup_maintenance_scheduler(self) -> None:
        """Set up daily maintenance task at midnight."""
        auto_maintenance = self.config_entry.options.get(
            CONF_AUTO_MAINTENANCE,
            self.config_entry.data.get(CONF_AUTO_MAINTENANCE, DEFAULT_AUTO_MAINTENANCE),
        )

        # Cancel existing scheduler if any
        if self._remove_maintenance_scheduler:
            self._remove_maintenance_scheduler()
            self._remove_maintenance_scheduler = None

        if not auto_maintenance:
            _LOGGER.debug("Auto-maintenance disabled")
            return

        # Calculate next midnight
        now = dt_util.now()
        tomorrow = now + timedelta(days=1)
        next_midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

        # Schedule first run at midnight
        async def run_maintenance(_now: datetime | None = None) -> None:
            """Run maintenance task."""
            _LOGGER.info("Running scheduled maintenance")
            try:
                stats = await self.profile_store.async_run_maintenance()
                _LOGGER.info("Maintenance completed: %s", stats)
            except Exception as err:
                _LOGGER.error("Maintenance failed: %s", err, exc_info=True)

        # Use async_track_point_in_time for midnight, then reschedule daily
        self._remove_maintenance_scheduler = evt.async_track_point_in_time(
            self.hass, run_maintenance, next_midnight
        )

        _LOGGER.info("Scheduled maintenance at %s", next_midnight)

        # Also schedule daily repeat after first run
        async def maintenance_wrapper(_now: datetime) -> None:
            await run_maintenance(_now)
            # Reschedule for next day
            next_run = dt_util.now() + timedelta(days=1)
            next_run = next_run.replace(hour=0, minute=0, second=0, microsecond=0)
            self._remove_maintenance_scheduler = evt.async_track_point_in_time(
                self.hass, maintenance_wrapper, next_run
            )

        self._remove_maintenance_scheduler = evt.async_track_point_in_time(
            self.hass, maintenance_wrapper, next_midnight
        )

    @callback
    def _async_power_changed(self, event: Any) -> None:
        """Handle power sensor state change."""
        event_data = cast(dict[str, Any], getattr(event, "data", {}))
        new_state = cast(State | None, event_data.get("new_state"))
        if new_state is None or new_state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return

        try:
            power = float(new_state.state)
        except ValueError:
            return

        # RECORD MODE INTERCEPTION
        if self.recorder.is_recording:
            self.recorder.process_reading(power)
            self._current_power = power
            self._last_reading_time = dt_util.now()
            self._notify_update()
            return

        now = dt_util.now()

        # Throttle updates to avoid CPU overload on noisy sensors
        # BUT always allow updates if power is below min_power (critical end-of-cycle signal).
        min_p = float(self.detector.config.min_power)
        is_low_power = power < min_p
        
        if (
            not is_low_power
            and self._last_reading_time
            and (now - self._last_reading_time).total_seconds() < self._sampling_interval
        ):
            return

        # Track observed power readings for learning
        self.learning_manager.process_power_reading(power, now, self._last_reading_time)
        self._last_reading_time = now
        self._last_real_reading_time = now # Track real update
        self._current_power = power
        self.detector.process_reading(power, now)

        # If running (or paused/ending), try to match profile and update estimates
        if self.detector.state in (
            STATE_RUNNING,
            STATE_PAUSED,
            STATE_ENDING,
            STATE_STARTING,
        ):
            self._update_estimates()
            # Periodically save state every 60s to avoid flash wear
            # We need a tracker.
            self._check_state_save(now)

        self._notify_update()

    def _check_state_save(self, now: datetime) -> None:
        """Periodically save active state."""
        last_save = getattr(self, "_last_state_save", None)
        if not last_save or (now - last_save).total_seconds() > 60:
            # Fire and forget save task
            # Inject manual program flag into snapshot before saving
            snapshot = self.detector.get_state_snapshot()
            snapshot["manual_program"] = self._manual_program_active

            self.hass.async_create_task(
                self.profile_store.async_save_active_cycle(snapshot)
            )
            self._last_state_save = now

    async def _run_final_match_from_cycle_data(self, cycle_data: dict[str, Any]) -> None:
        """Run final profile match using the cycle's power data before it's saved.

        This is called from _on_cycle_end when _current_program is still 'detecting...'
        to ensure we try matching with complete cycle data before persistence.
        """
        # Cycle data from detector has power_data as [(isoformat_str, power), ...]
        power_data = cycle_data.get("power_data", [])
        duration = cycle_data.get("duration", 0)

        if not power_data or len(power_data) < 10:
            _LOGGER.debug("Insufficient power data for final match (< 10 readings)")
            return

        # power_data is already in [(isoformat_str, power), ...] format for matching
        _LOGGER.info(
            "Running final match from cycle data: %s samples, %.0fs duration",
            len(power_data),
            duration,
        )

        result = await self.profile_store.async_match_profile(power_data, duration)
        profile_name = result.best_profile
        confidence = result.confidence

        # Store result for debug data
        self._last_match_result = result

        # Accept match at lower threshold since cycle is complete
        # Also ignore ambiguity for completed cycles - pick the best match
        if profile_name and confidence >= 0.15:
            _LOGGER.info(
                "Final match from cycle data: '%s' with confidence %.3f",
                profile_name,
                confidence,
            )
            self._current_program = profile_name
            self._last_match_confidence = confidence
        else:
            _LOGGER.info(
                "No confident match from cycle data (best: %s, conf=%.3f)",
                profile_name,
                confidence,
            )

    def _start_watchdog(self) -> None:
        """Start the watchdog timer when a cycle begins."""
        if self._remove_watchdog:
            return  # Already running

        interval = self._watchdog_interval
        _LOGGER.debug(
            "Starting watchdog timer (configured=%ss)",
            self._watchdog_interval,
        )
        self._remove_watchdog = async_track_time_interval(
            self.hass, self._watchdog_check_stuck_cycle, timedelta(seconds=interval)
        )

    def _stop_watchdog(self) -> None:
        """Stop the watchdog timer when cycle ends."""
        if self._remove_watchdog:
            _LOGGER.debug("Stopping watchdog timer")
            self._remove_watchdog()
            self._remove_watchdog = None

    def _start_state_expiry_timer(self) -> None:
        """Start timer to reset state to OFF and progress to 0% after idle period."""
        if not hasattr(self, "_remove_state_expiry_timer"):
            self._remove_state_expiry_timer = None

        if self._remove_state_expiry_timer:
            return  # Already running

        _LOGGER.debug(
            "Starting state expiry timer (will reset after %ss)",
            self._progress_reset_delay,
        )
        self._remove_state_expiry_timer = async_track_time_interval(
            self.hass,
            self._handle_state_expiry,
            timedelta(seconds=60),  # Check every minute
        )

    def _stop_state_expiry_timer(self) -> None:
        """Stop the state expiry timer."""
        if (
            hasattr(self, "_remove_state_expiry_timer")
            and self._remove_state_expiry_timer
        ):
            _LOGGER.debug("Stopping state expiry timer")
            self._remove_state_expiry_timer()
            self._remove_state_expiry_timer = None

    async def _handle_state_expiry(self, now: datetime) -> None:
        """Check if state and progress should be reset (auto-expiration)."""
        if not self._cycle_completed_time or self.detector.state == STATE_RUNNING:
            # Cycle is running or not completed, don't reset
            return

        time_since_complete = (now - self._cycle_completed_time).total_seconds()

        if time_since_complete > self._progress_reset_delay:
            # Auto-expire the "Finished" (or other terminal) state
            _LOGGER.debug(
                "State expiry: cycle idle for %.0fs (threshold: %ss). Resetting to OFF.",
                time_since_complete,
                self._progress_reset_delay,
            )
            self._cycle_progress = 0.0
            self._cycle_completed_time = None
            self.detector.reset(STATE_OFF)
            self._stop_state_expiry_timer()
            self._notify_update()

    async def _watchdog_check_stuck_cycle(self, now: datetime) -> None:
        """Watchdog: check if cycle is stuck (no updates for too long)."""
        if self.detector.state not in (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING):
            return

        if not self._last_reading_time:
            return

        time_since_any_update = (now - self._last_reading_time).total_seconds()
        
        # Calculate time since REAL update (if available, else fallback to any update)
        last_real = self._last_real_reading_time or self._last_reading_time
        time_since_real_update = (now - last_real).total_seconds()
        
        elapsed = self.detector.get_elapsed_seconds()
        expected = getattr(self.detector, "expected_duration_seconds", 0)

        # 0. ZOMBIE KILLER (Hard Limit)
        # If cycle has run significantly longer than expected (200%), kill it.
        # Only applies if we have a profile match.
        if expected > 0 and elapsed > (expected * 2.0) and elapsed > 7200:
            _LOGGER.warning(
                "Watchdog: Zombie cycle detected (%.0fs > 200%% of expected %.0fs). Force-ending.",
                elapsed, expected
            )
            self.detector.force_end(now)
            self._current_power = 0.0  # Force 0W
            self._notify_update()
            return

        # 1. GHOST CYCLE SUPPRESSOR
        # If we are "detecting" for more than 10 minutes and haven't seen an update for 5 minutes, 
        # it's likely a pump-out spike or an accidental start (ghost cycle).
        # We end it aggressively ONLY if it started shortly after another cycle ended (Suspicious Window).
        cycle_start = self.detector.current_cycle_start
        is_suspicious = False
        if cycle_start and self._last_cycle_end_time:
            # If started within 3 minutes of previous cycle finish (prevent back-to-back ghosting)
            if (cycle_start - self._last_cycle_end_time).total_seconds() < 180:
                is_suspicious = True

        if (
            self._current_program == "detecting..."
            and is_suspicious
            and elapsed > 600  # 10 minutes
            and time_since_real_update > 300  # 5 minutes of silence
        ):
            _LOGGER.warning(
                "Watchdog: Ghost cycle suppressed (within suspicious window). Detecting for %.0fs with %.0fs silence.",
                elapsed, time_since_real_update
            )
            self.detector.force_end(now)
            self._current_power = 0.0
            self._notify_update()
            return

        # --- LOW POWER HANDLING ---
        # If we are in a low power state (waiting for off_delay or drying profile),
        # we treat silence leniently. We inject keepalives until the stricter
        # low_power_no_update_timeout is reached.
        
        # Dishwashers can have very long silent drying phases (up to 2h)
        # We use the device-specific timeout as the floor for this effective timeout
        low_power_floor = DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE.get(
            self.device_type, 0
        )
        effective_low_power_timeout = self._low_power_no_update_timeout
        if self._current_program not in ("detecting...", "off"):
            effective_low_power_timeout = max(low_power_floor, effective_low_power_timeout)

        # Profile-Aware Extension:
        # If we have a matched profile, ensure we don't kill during the expected duration.
        if expected > 0 and elapsed < expected:
            # Extend timeout to cover the remaining expected duration + buffer
            remaining = expected - elapsed
            # Allow silence up to remaining + 1800s (30m buffer for drying/pause)
            extended_timeout = remaining + 1800
            if extended_timeout > effective_low_power_timeout:
                effective_low_power_timeout = extended_timeout

        # Verified Pause Extension:
        # If the manager/store has confirmed this is a legitimate pause (e.g. Drying),
        # allow even more leniency up to the global deferral limit.
        if getattr(self.detector, "_verified_pause", False):
            # Allow silence up to DEFAULT_MAX_DEFERRAL_SECONDS (default 2h) + buffer
            pause_limit = DEFAULT_MAX_DEFERRAL_SECONDS + 1800
            if pause_limit > effective_low_power_timeout:
                effective_low_power_timeout = pause_limit
                _LOGGER.debug(
                    "Watchdog: Extending timeout to %.0fs due to verified pause",
                    effective_low_power_timeout
                )
             
        if self.detector.is_waiting_low_power():
            
            # 2. Staleness Check
            if time_since_real_update > effective_low_power_timeout:
                _LOGGER.warning(
                    "Watchdog: Force-ending cycle. Low-power state stale for %.0fs (> %.0fs).",
                    time_since_real_update,
                    effective_low_power_timeout
                )
                self.detector.force_end(now)
                self._last_reading_time = now
                self._current_power = 0.0
                self._notify_update()
                return

            # 3. Injection Check (Keepalive)
            # If silence > _config.off_delay, inject 0W to keep detector clock moving
            # so it can evaluate profile logic (smart termination) or just mark time passing.
            if time_since_any_update > self._config.off_delay:
                _LOGGER.debug(
                    "Watchdog: Low power silence (%.0fs). Injecting 0W keepalive.",
                    time_since_any_update
                )
                # Ensure we handle the injection cleanly
                # Do NOT update _last_real_reading_time here
                self.detector.process_reading(0.0, now)
                self._last_reading_time = now # Resets 'any' timer so we don't spam
                self._current_power = 0.0
                self._notify_update()
                return
                
            return

        # Fallback for old "Case 1.5" logic (Low Power but NOT is_waiting_low_power)
        # Check this BEFORE High Power timeout to prevent trapping "Not Yet Waiting" states
        if (
            time_since_any_update > self._config.off_delay 
            and self._current_power < self.detector.config.min_power
        ):
            # Treating as start of low power wait
            _LOGGER.debug("Watchdog: Silence at low power (%.0fs). Injecting 0W.", time_since_any_update)
            self.detector.process_reading(0.0, now)
            self._last_reading_time = now
            self._current_power = 0.0
            self._notify_update()
            return

        # --- HIGH POWER HANDLING (Normal) ---
        # If power is high, we expect frequent updates.
        
        if time_since_any_update > self._no_update_active_timeout:
            
            # Check if high power (running)
            if self._current_power >= self.detector.config.min_power:
                # Allow extended silence if within reasonable cycle bounds
                expected = getattr(self.detector, "expected_duration_seconds", 0)
                elapsed = self.detector.get_elapsed_seconds()
                limit = (expected + 7200) if expected > 0 else 14400 # 4h default
                
                if elapsed < limit:
                    _LOGGER.info(
                        "Watchdog: High power (%.1fW) stale (%.0fs). Injecting refresh.",
                        self._current_power, time_since_any_update
                    )
                    self.detector.process_reading(self._current_power, now)
                    self._last_reading_time = now
                    self._notify_update()
                    return

            # If we get here, it's truly stuck/offline
            _LOGGER.warning(
                "Watchdog: Force-ending cycle. Active state stale for %.0fs (> timeout).",
                time_since_any_update
            )
            self.detector.force_end(now)
            self._current_power = 0.0  # FIX: Reset current power
            self._notify_update()
            return


    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """Handle state change from detector."""
        _LOGGER.debug("Washer state changed: %s -> %s", old_state, new_state)
        if new_state == STATE_RUNNING:
            # Only reset estimates if we are truly starting a NEW cycle (from off or starting)
            # If we transition from PAUSED or ENDING, it's a resume - keep estimates!
            if old_state in (STATE_OFF, STATE_STARTING, STATE_UNKNOWN):
                self._cycle_completed_time = None
                self._stop_state_expiry_timer()
                
                self._current_program = "detecting..."
                self._manual_program_active = False
                self._notified_pre_completion = False
                self._time_remaining = None
                self._total_duration = None
                self._cycle_progress = 0
                self._matched_profile_duration = None
                self._last_estimate_time = None
                self._score_history = {}  # Reset score history on new cycle
                self._match_persistence_counter = {}  # Reset persistence counter
                self._unmatch_persistence_counter = 0  # Reset unmatch counter
                self._current_match_candidate = None  # Reset candidate
                self._start_watchdog()  # Start watchdog when cycle starts
            else:
                _LOGGER.debug("Cycle resumed from %s, preserving estimates", old_state)
                # Ensure watchdog is running
                self._start_watchdog()
            self.hass.bus.async_fire(
                EVENT_CYCLE_STARTED,
                {
                    "entry_id": self.entry_id,
                    "device_name": self.config_entry.title,
                    "device_type": self.device_type,
                    "program": self._current_program,
                    "start_time": dt_util.now().isoformat(),
                },
            )

            # Send notification if enabled
            events = self.config_entry.options.get(CONF_NOTIFY_EVENTS, [])
            if NOTIFY_EVENT_START in events:
                msg_template = self.config_entry.options.get(CONF_NOTIFY_START_MESSAGE, DEFAULT_NOTIFY_START_MESSAGE)
                msg = msg_template.format(device=self.config_entry.title)
                self._send_notification(msg)
        elif new_state == STATE_OFF and old_state == STATE_RUNNING:
            self._stop_watchdog()  # Stop watchdog when cycle ends

        self._notify_update()

    def _on_cycle_end(self, cycle_data: dict[str, Any]) -> None:
        """Handle cycle end - clear all active timers and state."""
        duration = cycle_data["duration"]
        max_power = cycle_data.get("max_power", 0)

        # IMMEDIATELY stop all active timers when cycle determined to have ended
        self._stop_watchdog()  # Stop active cycle watchdog
        self._stop_state_expiry_timer()  # Cancel any pending progress reset
        self._last_cycle_end_time = dt_util.now()

        # Auto-Tune: Check for ghost cycles (short duration AND low energy)
        # Ghost = duration < 60s AND total energy < 0.05 Wh (avoids killing pump-out spikes)
        power_data = cycle_data.get("power_data", [])
        cycle_energy_wh = 0.0
        if power_data and len(power_data) >= 2:
            try:
                ts = np.array([float(p[0]) if isinstance(p[0], (int, float)) else 0 for p in power_data])
                ps = np.array([float(p[1]) for p in power_data])
                # Simple trapezoidal integration
                dt_h = np.diff(ts) / 3600.0 if ts[0] > 1e9 else np.diff(ts) / 3600.0  # seconds -> hours
                avg_p = (ps[:-1] + ps[1:]) / 2
                cycle_energy_wh = float(np.sum(avg_p * np.abs(dt_h)))
            except Exception:
                cycle_energy_wh = 0.0

        # Ghost cycle: short AND low energy (real cycles have energy even if short)
        if duration < 60 and cycle_energy_wh < 0.05:
            self._handle_noise_cycle(max_power)

        # Schedule heavy post-processing asynchronously
        self.hass.async_create_task(self._async_process_cycle_end(cycle_data))

    async def _async_process_cycle_end(self, cycle_data: dict[str, Any]) -> None:
        """Process cycle completion asynchronously (heavy tasks)."""

        # FINAL PROFILE MATCH: If still detecting, try one last match with complete cycle data
        if self._current_program in ("detecting...", "restored..."):
            await self._run_final_match_from_cycle_data(cycle_data)

        # If we had a runtime match, attach the profile name for persistence
        if (
            self._current_program
            and self._current_program not in ("off", "detecting...", "restored...")
            and self._current_program in self.profile_store.get_profiles()
        ):
            cycle_data["profile_name"] = self._current_program

        # Attach extensive debug data if available (and configured)
        if self._last_match_result:
            cycle_data["debug_data"] = {
                "ranking": getattr(self._last_match_result, "ranking", []),
                "details": getattr(self._last_match_result, "debug_details", {}),
                "ambiguous": getattr(self._last_match_result, "is_ambiguous", False),
            }

        # Post-Cycle Auto-Labeling (if not already matched)
        # Offload this match too if needed
        if not cycle_data.get("profile_name") and self._auto_label_confidence > 0:
            res = await self.profile_store.async_match_profile(
                cycle_data["power_data"], cycle_data["duration"]
            )
            if res.best_profile and res.confidence >= self._auto_label_confidence:
                cycle_data["profile_name"] = res.best_profile
                _LOGGER.info(
                    "Post-cycle auto-labeled as '%s' (confidence: %.2f)",
                    res.best_profile,
                    res.confidence,
                )

        # Add cycle to store immediately (still sync but offloadable parts optimized
        # internally if possible)
        # Note: add_cycle is mostly safe (signature calc is O(N) but fast enough for
        # single cycle).
        # We could offload signature calc to analysis logic if really needed, but let's
        # stick to match profile optimization first.
        try:
            await self.profile_store.async_add_cycle(cycle_data)
            profile_name = cycle_data.get("profile_name")
            if profile_name:
                await self.profile_store.async_rebuild_envelope(profile_name)
        except Exception as e: # pylint: disable=broad-exception-caught
            _LOGGER.error("Failed to add cycle to store: %s", e)

        # Ensure cycle has a stable ID even if store add failed (or did not mutate).
        if not cycle_data.get("id"):
            try:
                unique_str = f"{cycle_data['start_time']}_{cycle_data['duration']}"
                cycle_data["id"] = hashlib.sha256(unique_str.encode()).hexdigest()[:12]
            except Exception:  # noqa: BLE001
                pass

        self.hass.async_create_task(self.profile_store.async_clear_active_cycle())

        # Auto post-process: merge fragmented cycles from last 3 hours
        self.hass.async_create_task(self._run_post_cycle_processing())

        # Prepare cycle data for event (enrich if needed)
        # IMPORTANT: Exclude large fields to prevent exceeding HA's 32KB event data limit
        excluded_fields = {"power_data", "debug_data", "power_trace"}
        event_cycle_data = {
            k: v for k, v in cycle_data.items() if k not in excluded_fields
        }
        event_cycle_data["device_type"] = self.device_type
        # Add program if missing or generic
        if "profile_name" not in event_cycle_data and self._current_program:
            event_cycle_data["profile_name"] = self._current_program

        self.hass.bus.async_fire(
            EVENT_CYCLE_ENDED,
            {
                "entry_id": self.entry_id,
                "device_name": self.config_entry.title,
                "cycle_data": event_cycle_data,
                "program": event_cycle_data.get("profile_name", "unknown"),
                "duration": event_cycle_data.get("duration"),
                "start_time": event_cycle_data.get("start_time"),
                "end_time": dt_util.now().isoformat(),
            },
        )

        # Send notification if enabled
        events = self.config_entry.options.get(CONF_NOTIFY_EVENTS, [])
        if NOTIFY_EVENT_FINISH in events:
            msg_template = self.config_entry.options.get(CONF_NOTIFY_FINISH_MESSAGE, DEFAULT_NOTIFY_FINISH_MESSAGE)
            duration_min = int(cycle_data['duration'] / 60)
            program_name = event_cycle_data.get("profile_name", "unknown")
            msg = msg_template.format(
                device=self.config_entry.title,
                duration=duration_min,
                program=program_name
            )
            self._send_notification(msg)

        # Request user feedback if we had a confident match.
        # AND perform learning analysis on the completed cycle.
        # IMPORTANT: this must happen before we clear match state.
        self.learning_manager.process_cycle_end(
            cycle_data,
            detected_profile=self._current_program,
            confidence=self._last_match_confidence or 0.0,
            predicted_duration=self._matched_profile_duration,
        )

        # Clear all state and timers - zero everything out
        self._current_program = "off"
        self._manual_program_active = False
        self._notified_pre_completion = False
        self._time_remaining = None
        self._matched_profile_duration = None
        self._last_estimate_time = None
        self._cycle_progress = 100.0  # 100% = cycle complete
        self._cycle_completed_time = dt_util.now()

        # Start progress reset timer to go back to 0% after user unload window
        self._start_state_expiry_timer()

        self._notify_update()

    @property
    def profile_sample_repair_stats(self) -> dict[str, int] | None:
        """Return statistics from profile sample repair operation."""
        return self._profile_sample_repair_stats

    @property
    def suggestions(self) -> dict[str, Any]:
        """Suggested settings computed by learning/heuristics (never auto-applied)."""
        return self.profile_store.get_suggestions()

    def _send_notification(self, message: str, title: str | None = None, icon: str | None = None) -> None:
        """Send a notification via configured service."""
        notify_service = self.config_entry.options.get(CONF_NOTIFY_SERVICE)
        
        # Use customized title if not provided explicitly
        if not title:
            title_template = self.config_entry.options.get(CONF_NOTIFY_TITLE, DEFAULT_NOTIFY_TITLE)
            title = title_template.format(device=self.config_entry.title)

        # Use customized icon if configured
        if not icon:
            icon = self.config_entry.options.get(CONF_NOTIFY_ICON)

        data = {}
        if icon:
            data["icon"] = icon

        if notify_service:
            domain, service = (
                notify_service.split(".", 1)
                if "." in notify_service
                else ("notify", notify_service)
            )
            service_data = {"message": message, "title": title}
            if data:
                service_data["data"] = data

            self.hass.async_create_task(
                self.hass.services.async_call(domain, service, service_data)
            )
        else:
            _pn_create(
                self.hass, message, title=title
            )

    def _handle_noise_cycle(self, max_power: float) -> None:
        """Handle a detected noise cycle."""
        # Clean up old noise events > 24h
        now = dt_util.now()
        self._noise_events = [
            t
            for t in getattr(self, "_noise_events", [])
            if (now - t).total_seconds() < 86400
        ]
        self._noise_events.append(now)

        # Track max power of noise
        self._noise_max_powers = getattr(self, "_noise_max_powers", [])
        self._noise_max_powers.append(max_power)

        # If noise events exceed threshold in 24h, trigger tune
        if len(self._noise_events) >= self._noise_events_threshold:
            self.hass.async_create_task(self._tune_threshold())

    async def _tune_threshold(self) -> None:
        """Increase the minimum power threshold."""
        current_min = self.detector.config.min_power

        # Calculate new suggested threshold
        # Max of observed noise * 1.2 safety factor
        noise_max = max(self._noise_max_powers)
        new_min = noise_max * 1.2

        # Cap absolute max to avoid runaway (e.g. 50W)
        if new_min > 50.0:
            new_min = 50.0

        if new_min <= current_min:
            # Clear events so we don't loop try to update
            self._noise_events = []
            self._noise_max_powers = []
            return

        _LOGGER.info(
            "Auto-Tune suggestion: min_power from %.1fW -> %.1fW due to noise",
            current_min,
            new_min,
        )

        # Store a suggestion (do not mutate user-set options)
        self.profile_store.set_suggestion(
            CONF_MIN_POWER,
            float(new_min),
            f"Auto-tune: {len(self._noise_events)} ghost cycles detected in 24h",
        )
        await self.profile_store.async_save()

        # Notify user
        notify_service = self.config_entry.options.get(CONF_NOTIFY_SERVICE)
        message = (
            f"Washing Machine '{self.config_entry.title}' detected ghost cycles. "
            f"Suggested min_power change: {current_min:.1f}W → {new_min:.1f}W "
            f"(not applied automatically)."
        )

        if notify_service:
            # call service notify.<name>
            domain, service = (
                notify_service.split(".", 1)
                if "." in notify_service
                else ("notify", notify_service)
            )
            self.hass.async_create_task(
                self.hass.services.async_call(domain, service, {"message": message})
            )
        else:
            _pn_create(self.hass, message, title="HA WashData Auto-Tune")

        # Reset trackers
        self._noise_events = []
        self._noise_max_powers = []

    def _update_estimates(self) -> None:
        """Update time remaining and profile estimates."""
        if self.detector.state in (STATE_OFF, STATE_UNKNOWN, STATE_IDLE, STATE_STARTING):
            self._current_program = "off"
            self._time_remaining = None
            self._total_duration = None
            self._cycle_progress = 0.0
            self._last_match_result = None
            self._notify_update()
            return

        now = dt_util.now()

        # Throttle heavy matching to configured interval (default: 5 minutes)
        effective_match_interval = self._profile_match_interval
        if (
            self._last_estimate_time
            and (now - self._last_estimate_time).total_seconds()
            < effective_match_interval
        ):
            # Still update remaining/progress if we already have a match
            self._update_remaining_only()
            return

        # SKIP matching if manual program is active
        if self._manual_program_active:
            self._last_estimate_time = now  # touch timestamp to throttle estimates loop
            self._update_remaining_only()
            # Also check notifications in loop
            self._check_pre_completion_notification()
            self._notify_update()
            return

        # No matching task trigger here anymore! 
        # The detector callback handles it.
        # Just update progress/remaining based on existing match.
        self._update_remaining_only()
        self._check_pre_completion_notification()
        self._notify_update()

    # _async_run_matching removed in favor of _async_perform_combined_matching

    def _analyze_trend(self, profile_name: str) -> bool:
        """Analyze score history to detect positive trend.

        Returns True if score has increased in at least 7 of the last 10 intervals.
        Requires at least 5 samples history to make a determination.
        """
        history = self._score_history.get(profile_name, [])
        if len(history) < 5:
            return False

        # Use last 11 points to get 10 intervals (or fewer if history short)
        recent = history[-11:]
        if len(recent) < 2:
            return False

        up_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        total_intervals = len(recent) - 1

        # Proportional threshold (7/10 => 0.7)
        return (up_count / total_intervals) >= 0.70

    def _check_pre_completion_notification(self) -> None:
        """Check and send pre-completion notification."""
        if (
            self._notify_before_end_minutes > 0
            and not self._notified_pre_completion
            and self._time_remaining is not None
            and self._time_remaining <= (self._notify_before_end_minutes * 60)
            and self._cycle_progress < 100
            and not self._last_match_ambiguous
        ):
            # Send notification!
            self._notified_pre_completion = True
            # Send notification!
            self._notified_pre_completion = True
            
            msg_template = self.config_entry.options.get(CONF_NOTIFY_PRE_COMPLETE_MESSAGE, DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE)
            # Safe default if rounding goes weird equivalent to int()
            minutes_left = int(self._time_remaining / 60) + 1 
            
            msg = msg_template.format(
                device=self.config_entry.title,
                minutes=minutes_left
            )
            self._send_notification(msg)
            _LOGGER.info("Sent pre-completion notification: %s", msg)

    def _update_remaining_only(self) -> None:
        """Recompute remaining/progress using phase-aware estimation."""
        # Throttle updates and only clear on truly dead states
        if self.detector.state in (STATE_OFF, STATE_UNKNOWN, STATE_IDLE):
            self._time_remaining = None
            self._total_duration = None
            self._cycle_progress = 0.0
            self._smoothed_progress = 0.0
            return

        now = dt_util.now()
        if (
            self._last_phase_estimate_time
            and (now - self._last_phase_estimate_time).total_seconds() < 5.0
        ):
            return
        self._last_phase_estimate_time = now

        duration_so_far = float(self.detector.get_elapsed_seconds())

        if self._matched_profile_duration and self._matched_profile_duration > 0:
            # Get current power trace for phase analysis
            trace = self.detector.get_power_trace()
            # current_power_data = [(t.isoformat(), p) for t, p in trace]
            # DEPRECATED: avoid O(N) conversion

            # --- PHASE-AWARE ESTIMATION ---
            if len(trace) >= 10 and self._current_program != "detecting...":
                phase_result = self._estimate_phase_progress(
                    trace, duration_so_far, self._current_program
                )
                if phase_result is not None:
                    phase_progress, phase_variance = phase_result

                    # Smoothing: Exponential Moving Average
                    # If this is the first reliable estimate, snap to it.
                    # Otherwise, blend 20% new, 80% old.
                    if self._smoothed_progress == 0.0:
                        self._smoothed_progress = phase_progress
                    else:
                        current_smoothed = self._smoothed_progress
                        # Smart Time Prediction (Variance-Based Locking)
                        # If variance is high (e.g. > 50W std dev), this phase
                        # is unpredictable. DAMP HEAVILY.
                        # If variance is low (< 10W), trust the estimate more.
                        alpha = 0.2  # Default
                        if phase_variance > 100.0:
                            alpha = 0.05  # Very slow updates (mostly locked)
                            _LOGGER.debug(
                                "High variance phase (std=%.1fW), "
                                "locking time estimate (alpha=0.05)",
                                phase_variance,
                            )
                        elif phase_variance > 50.0:
                            alpha = 0.1

                        # Monotonicity check: don't let it jump BACKWARD
                        # significantly unless the profile changed (handled
                        # elsewhere). Allow small fluctuations, but prevent
                        # large drops. Use device-type-specific threshold to
                        # handle different cycle characteristics.
                        smoothing_threshold = DEVICE_SMOOTHING_THRESHOLDS.get(
                            self.device_type, 5.0
                        )
                        if phase_progress < current_smoothed - smoothing_threshold:
                            # Let's damp it heavily (keep mostly old value).
                            self._smoothed_progress = (current_smoothed * 0.95) + (
                                phase_progress * 0.05
                            )
                            _LOGGER.debug(
                                "Progress drop detected (%.1f%% < %.1f%% - %.1f%%), "
                                "applying heavy damping for %s",
                                phase_progress,
                                current_smoothed,
                                smoothing_threshold,
                                self.device_type,
                            )
                        else:
                            # Normal estimate update with dynamic alpha
                            self._smoothed_progress = (
                                self._smoothed_progress * (1.0 - alpha)
                            ) + (phase_progress * alpha)

                    # Ensure we don't exceed 99% until actually finished
                    self._smoothed_progress = min(99.0, self._smoothed_progress)

                    # Update User-Facing Progress from Smoothed Value
                    self._cycle_progress = self._smoothed_progress

                    # Back-calculate "Time Remaining" from the smoothed progress
                    # exact_remaining = duration * (1 - progress)
                    # This prevents "progress says 90% but time says 20 mins" mismatch
                    remaining = self._matched_profile_duration * (
                        1.0 - (self._cycle_progress / 100.0)
                    )
                    self._time_remaining = max(0.0, remaining)
                    self._total_duration = duration_so_far + remaining
                    self._last_total_duration_update = now

                    _LOGGER.debug(
                        "Phase-aware estimate: raw=%.1f%%, smoothed=%.1f%%, remaining=%smin",
                        phase_progress,
                        self._cycle_progress,
                        int(remaining / 60),
                    )
                    return

            # --- LINEAR FALLBACK (if phase analysis unavailable) ---
            matched_dur = float(self._matched_profile_duration)
            remaining = max(matched_dur - duration_so_far, 0.0)
            progress = (duration_so_far / matched_dur) * 100.0

            # Blend linear estimate into smoothed tracker too, to prevent
            # jumps if we lose phase lock
            if self._smoothed_progress > 0:
                # Blend gently
                self._smoothed_progress = (self._smoothed_progress * 0.9) + (
                    progress * 0.1
                )
            else:
                self._smoothed_progress = progress

            self._time_remaining = remaining
            self._total_duration = duration_so_far + remaining
            self._last_total_duration_update = now
            self._cycle_progress = max(0.0, min(self._smoothed_progress, 100.0))
            _LOGGER.debug(
                "Linear estimate: remaining=%smin, progress=%.1f%%",
                int(remaining / 60),
                self._cycle_progress,
            )
        else:
            # No profile matched - don't provide misleading time estimates
            # Just show that we're detecting (no Smart Resume based on history)
            self._time_remaining = None
            self._total_duration = None
            self._cycle_progress = 0.0
            self._smoothed_progress = 0.0
            _LOGGER.debug(
                "No profile matched yet, elapsed=%smin", int(duration_so_far / 60)
            )

    def _estimate_phase_progress(
        self,
        current_power_data: list[tuple[datetime, float]] | list[tuple[str, float]],
        current_duration: float,
        profile_name: str,
    ) -> float | None:
        """
        Estimate cycle progress by analyzing which phase we're in.

        Uses cached statistical envelope built from ALL cycles labeled with
        this profile, normalized by TIME to account for different sampling rates.

        Returns progress percentage (0-100) or None if estimation fails.
        """
        # Get cached envelope (fast - already computed and stored)
        envelope = self.profile_store.get_envelope(profile_name)

        if envelope is None:
            _LOGGER.debug("No envelope cached for profile %s", profile_name)
            return None

        # Convert cached lists back to numpy arrays
        # Envelope curves are stored as [[t, y], ...] points, extract Y values only
        try:
            env_min = envelope.get("min", [])
            env_max = envelope.get("max", [])
            env_avg = envelope.get("avg", [])
            env_std = envelope.get("std", [])

            # Handle both formats: [[t, y], ...] (new) or [y, ...] (legacy)
            def extract_y_values(data: list) -> np.ndarray:
                if not data:
                    return np.array([])
                if isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                    # New format: [[t, y], ...]
                    return np.array([float(pt[1]) for pt in data])
                # Legacy format: [y, ...]
                return np.array(data)

            envelope_arrays = {
                "min": extract_y_values(env_min),
                "max": extract_y_values(env_max),
                "avg": extract_y_values(env_avg),
                "std": extract_y_values(env_std),
            }
            time_grid = np.array(envelope.get("time_grid", []))
            target_duration = envelope.get("target_duration", 0)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            _LOGGER.warning("Invalid envelope format for %s: %s", profile_name, e)
            return None

        if len(time_grid) == 0 or target_duration <= 0:
            if target_duration > 0 and len(envelope_arrays["avg"]) > 0:
                # Reconstruct time_grid if missing (Legacy envelope support)
                count = len(envelope_arrays["avg"])
                time_grid = np.linspace(0, target_duration, count)
                _LOGGER.debug(
                    "Reconstructed missing time_grid for %s (n=%d)",
                    profile_name,
                    count,
                )
            else:
                _LOGGER.debug("Envelope missing time grid/duration, cannot estimate phase")
                return None

        # Extract power values and offsets from current cycle
        # We handle both datetime objects (raw trace) and ISO strings (legacy/converted)
        start_ts: float = 0.0
        if current_power_data:
            first_t = current_power_data[0][0]
            if isinstance(first_t, datetime):
                start_ts = first_t.timestamp()
            elif isinstance(first_t, str):
                start_ts = datetime.fromisoformat(first_t).timestamp()

        current_offsets = np.array(
            [
                (
                    (t.timestamp() - start_ts)
                    if isinstance(t, datetime)
                    else (
                        float(t)
                        if isinstance(t, (int, float))
                        else (datetime.fromisoformat(t).timestamp() - start_ts)
                    )
                )
                for t, _ in current_power_data
            ]
        )
        current_values = np.array([p for _, p in current_power_data])

        # Use sliding window on TIME, not sample count
        # Look at last ~1 minute of data or 25% of expected duration, whichever is smaller
        window_duration = min(60.0, target_duration * 0.25)
        current_time = current_offsets[-1]
        window_start_time = max(0, current_time - window_duration)

        # Get current window (last N seconds of data)
        window_mask = current_offsets >= window_start_time
        current_window_values = current_values[window_mask]

        if len(current_window_values) < 3:
            _LOGGER.debug("Insufficient data in current window for phase estimation")
            return None

        best_progress = None
        best_score = -1.0
        in_bounds = False
        best_time_window_start: float | None = None

        # Search through envelope TIME grid for best matching position
        for i in range(len(time_grid) - 1):
            time_window_start = time_grid[i]

            # Get envelope values for this time window
            envelope_window_start = i
            envelope_window_end = min(
                i + len(current_window_values), len(envelope_arrays["avg"])
            )

            if envelope_window_end <= envelope_window_start:
                continue

            avg_window = envelope_arrays["avg"][
                envelope_window_start:envelope_window_end
            ]
            min_window = envelope_arrays["min"][
                envelope_window_start:envelope_window_end
            ]
            max_window = envelope_arrays["max"][
                envelope_window_start:envelope_window_end
            ]

            # Interpolate envelope to match current window length if needed
            if len(avg_window) != len(current_window_values):
                x_old = np.linspace(0, 1, len(avg_window))
                x_new = np.linspace(0, 1, len(current_window_values))
                avg_window = np.interp(x_new, x_old, avg_window)
                min_window = np.interp(x_new, x_old, min_window)
                max_window = np.interp(x_new, x_old, max_window)

            # Check if current power is within expected bounds (±20% tolerance)
            within_bounds = np.all(
                (current_window_values >= min_window * 0.8)
                & (current_window_values <= max_window * 1.2)
            )
            bounds_score = np.mean(
                (current_window_values >= min_window)
                & (current_window_values <= max_window)
            )

            # Calculate shape similarity to average
            try:
                if np.std(current_window_values) > 0 and np.std(avg_window) > 0:
                    correlation = np.corrcoef(current_window_values, avg_window)[0, 1]
                else:
                    correlation = 0.0

                # MAE against average
                mae = np.mean(np.abs(current_window_values - avg_window))
                max_power = max(np.max(avg_window), np.max(current_window_values), 1.0)
                mae_normalized = 1.0 - min(mae / max_power, 1.0)

                # Combined score: shape + amplitude + bounds compliance
                score = (
                    0.4 * max(correlation, 0.0)  # Shape matching
                    + 0.3 * mae_normalized  # Amplitude matching
                    + 0.3 * bounds_score  # Within expected range
                )

                # Penalize matches that are far from current elapsed time
                # (assume linear progress is roughly correct). This prevents
                # wild jumps in time remaining when patterns repeat
                time_diff = abs(time_window_start - current_duration)
                # Max penalty at 30% duration diff
                time_penalty = min(1.0, time_diff / (target_duration * 0.3))

                # Apply time penalty (reduce score by up to 40%)
                score = score * (1.0 - 0.4 * time_penalty)

                if score > best_score:
                    best_score = score
                    best_progress = (time_window_start / target_duration) * 100.0
                    in_bounds = within_bounds
                    best_time_window_start = float(time_window_start)
            except Exception:  # pylint: disable=broad-exception-caught
                continue

        if best_progress is None or best_score < 0.4:
            _LOGGER.debug("Phase detection failed: best_score=%.3f", best_score)
            return None

        # Calculate variance for the best window (Smart Time Prediction)
        # Low variance = high confidence in timing. High variance = low confidence.
        best_variance = 0.0
        if best_time_window_start is not None:
            # Find index in time_grid again (approx)
            # Optimization: store best_index in loop?
            # Just map time back to index
            idx_start = int((best_time_window_start / target_duration) * len(time_grid))
            idx_end = min(
                idx_start + len(current_window_values), len(envelope_arrays["std"])
            )
            if idx_end > idx_start:
                window_std = envelope_arrays["std"][idx_start:idx_end]
                if len(window_std) > 0:
                    best_variance = float(np.mean(window_std))

        # Cap progress at 99% until actual completion
        best_progress = max(0.0, min(best_progress, 99.0))

        # Log with envelope metadata
        cycle_count = envelope.get("cycle_count", 0)
        avg_sample_rates = envelope.get("sampling_rates", [1.0])
        avg_sample_rate = np.median(avg_sample_rates) if avg_sample_rates else 1.0

        tws = (
            best_time_window_start
            if best_time_window_start is not None
            else float(current_duration)
        )
        if not in_bounds:
            _LOGGER.debug(
                "Phase detection: progress=%.1f%%, score=%.3f, var=%.1fW, "
                "time=%.0f/%.0fs [OUT OF BOUNDS, %s cycles, avg_sample_rate=%.1fs]",
                best_progress,
                best_score,
                best_variance,
                tws,
                target_duration,
                cycle_count,
                avg_sample_rate,
            )
        else:
            _LOGGER.debug(
                "Phase detection: progress=%.1f%%, score=%.3f, var=%.1fW, "
                "time=%.0f/%.0fs [IN BOUNDS, %s cycles, avg_sample_rate=%.1fs]",
                best_progress,
                best_score,
                best_variance,
                tws,
                target_duration,
                cycle_count,
                avg_sample_rate,
            )

        return (best_progress, best_variance)

    def _notify_update(self) -> None:
        """Notify entities of update."""
        async_dispatcher_send(self.hass, SIGNAL_WASHER_UPDATE.format(self.entry_id))

    def notify_update(self) -> None:
        """Public method to notify entities of update."""
        self._notify_update()

    @property
    def check_state(self):
        """Return current detector state."""
        if self.recorder.is_recording:
            return STATE_RUNNING
        return self.detector.state

    @property
    def sub_state(self) -> str | None:
        """Return more granular state info (e.g. current phase)."""
        if self.recorder.is_recording:
            return "Recording"
        return self.detector.sub_state

    @property
    def current_program(self):
        """Return the current program name."""
        return self._current_program

    @property
    def time_remaining(self):
        """Return estimated time remaining in seconds."""
        return self._time_remaining

    @property
    def total_duration(self) -> float | None:
        """Return total predicted duration in seconds."""
        return self._total_duration

    @property
    def last_total_duration_update(self) -> datetime | None:
        """Return when total duration was last refined."""
        return self._last_total_duration_update

    @property
    def cycle_progress(self):
        """Return cycle progress as a percentage."""
        return self._cycle_progress

    @property
    def current_power(self):
        """Return current power reading in watts."""
        return self._current_power

    @property
    def cycle_start_time(self) -> datetime | None:
        """Return the start time of the current cycle."""
        return self.detector.current_cycle_start

    @property
    def last_match_details(self) -> dict[str, Any] | None:
        """Return details of the last profile match."""
        res = getattr(self, "_last_match_result", None)
        return res.to_dict() if res else None

    @property
    def samples_recorded(self):
        """Return the number of power samples recorded in current cycle."""
        return len(self.detector.get_power_trace())

    @property
    def sample_interval_stats(self):
        """Return statistics about sampling intervals."""
        return self._sample_interval_stats

    @property
    def manual_program_active(self) -> bool:
        """Return True if a manual program override is active."""
        return getattr(self, "_manual_program_active", False)

    def set_manual_program(self, profile_name: str) -> None:
        """Manually set the current program."""
        if self.detector.state != "running":
            pass

        profiles_raw: Any = None
        try:
            profiles_raw = self.profile_store.get_profiles()
        except Exception:  # pylint: disable=broad-exception-caught
            profiles_raw = None

        if isinstance(profiles_raw, dict):
            profiles: dict[str, Any] = cast(dict[str, Any], profiles_raw)
        else:
            profiles_fallback = getattr(self.profile_store, "_data", {}).get(
                "profiles", {}
            )
            profiles = (
                cast(dict[str, Any], profiles_fallback)
                if isinstance(profiles_fallback, dict)
                else {}
            )

        if profile_name not in profiles:
            _LOGGER.warning("Cannot set manual program: '%s' not found", profile_name)
            return

        self._current_program = profile_name
        self._manual_program_active = True

        # Update expected duration immediately
        profile = profiles.get(profile_name)
        if profile:
            avg = float(profile.get("avg_duration", 0.0))
            if avg > 0:
                self._matched_profile_duration = avg
                _LOGGER.info(
                    "Manual program set to %s, duration=%.0fs", profile_name, avg
                )

                # Update estimates if running
                if self.detector.state == "running":
                    self._update_estimates()

    async def async_terminate_cycle(self) -> None:
        """Force terminate the current cycle via user request."""
        _LOGGER.warning("Force terminating cycle by user request")

        # Trigger natural cycle end via detector
        # This will call _on_cycle_end callback, which handles:
        # - Saving to profile store
        # - Clearing active cycle persistence
        # - Post-processing/Merging
        # - Notifications
        self.detector.user_stop()

        # We DO NOT clear manager state manually here (e.g. self._current_program)
        # because we want the UI to show the "Clean" state with the just-finished
        # program info. The standard reset timers in _on_cycle_end /
        # _async_power_changed will handle cleanup after delay.

        # Force a state update to reflect the change immediately
        self._notify_update()

    async def async_start_recording(self) -> None:
        """Start manual recording of a cycle."""
        if self.recorder.is_recording:
            _LOGGER.warning("Already recording")
            return

        # Ensure we are in a clean state (stop any running cycle first?)
        # If running, user should probably stop it? Or force stop?
        # Plan said "unregulated", so we just start recording.
        # But if cycle_detector thinks it's running, we should probably "pause" it
        # or just override state. My override in checks_state handles UI.
        # But should we clear current program?
        if self.detector.state != "off":
            _LOGGER.info("Forcing detector reset before recording")
            self.detector.reset()

        await self.recorder.start_recording()
        self._notify_update()

    async def async_stop_recording(self) -> None:
        """Stop manual recording."""
        if not self.recorder.is_recording:
            return

        await self.recorder.stop_recording()
        self._notify_update()

    def clear_manual_program(self) -> None:
        """Clear manual program override."""
        if not self._manual_program_active:
            return

        self._manual_program_active = False
        # If running, revert to detecting so auto-detection can resume?
        if self.detector.state == "running":
            self._current_program = "detecting..."
            self._matched_profile_duration = None
            self._update_estimates()  # Trigger immediate re-detection attempt
        else:
            # If not running, clear the forced program
            self._current_program = None
            self._matched_profile_duration = None

        self._notify_update()
        _LOGGER.info("Manual program cleared, reverting to auto-detection")

    async def _run_post_cycle_processing(self) -> None:
        """Run post-cycle processing (merge fragments, split anomalies)."""
        try:
            # User Feedback: Use 5 hour lookback and configured gap settings
            stats = await self.profile_store.async_run_maintenance()

            # Log significant actions
            merged = stats.get("merged_cycles", 0)
            split = stats.get("split_cycles", 0)
            if merged > 0 or split > 0:
                _LOGGER.info(
                    "Post-cycle processing: Merged %s, Split %s cycle(s)", merged, split
                )

            # Note: async_run_maintenance saves automatically if changes occur
        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOGGER.error("Post-cycle processing failed: %s", e)
