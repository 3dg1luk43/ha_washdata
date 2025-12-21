"""Manager for HA WashData."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.util import dt as dt_util
import homeassistant.helpers.event as evt

from .const import (
    DOMAIN,
    CONF_POWER_SENSOR,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_EVENTS,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_SMOOTHING_WINDOW,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_AUTO_MERGE_LOOKBACK_HOURS,
    CONF_AUTO_MERGE_GAP_SECONDS,
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
    SIGNAL_WASHER_UPDATE,
    NOTIFY_EVENT_START,
    NOTIFY_EVENT_FINISH,
    EVENT_CYCLE_STARTED,
    EVENT_CYCLE_ENDED,
    FEEDBACK_REQUEST_EVENT,
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_PROFILE_DURATION_TOLERANCE,
    DEFAULT_AUTO_MERGE_LOOKBACK_HOURS,
    DEFAULT_AUTO_MERGE_GAP_SECONDS,
    DEFAULT_INTERRUPTED_MIN_SECONDS,
    DEFAULT_ABRUPT_DROP_WATTS,
    DEFAULT_ABRUPT_DROP_RATIO,
    DEFAULT_ABRUPT_HIGH_LOAD_FACTOR,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_PROGRESS_RESET_DELAY,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_AUTO_MAINTENANCE,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_MAX_PAST_CYCLES,
    DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
    DEFAULT_MAX_FULL_TRACES_UNLABELED,
    DEFAULT_WATCHDOG_INTERVAL,
    DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    STATE_RUNNING,
    STATE_OFF,
)
from .cycle_detector import CycleDetector, CycleDetectorConfig
from .learning import LearningManager
from .profile_store import ProfileStore

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
        pn.async_create(message, title=title, notification_id=notification_id)
    except Exception:
        return

class WashDataManager:
    """Manages a single washing machine instance."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the manager."""
        self.hass = hass
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        
        self.power_sensor_entity_id = config_entry.data[CONF_POWER_SENSOR]
        
        # Components
        self.profile_store = ProfileStore(
            hass, 
            self.entry_id,
            config_entry.options.get(CONF_PROFILE_MATCH_MIN_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO),
            config_entry.options.get(CONF_PROFILE_MATCH_MAX_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO),
        )
        self.learning_manager = LearningManager(hass, self.entry_id, self.profile_store)
        
        # Priority: Options > Data > Default
        min_power = config_entry.options.get(CONF_MIN_POWER, config_entry.data.get(CONF_MIN_POWER, DEFAULT_MIN_POWER))
        off_delay = config_entry.options.get(CONF_OFF_DELAY, config_entry.data.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY))
        progress_reset_delay = config_entry.options.get(CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY)
        self._no_update_active_timeout = int(
            config_entry.options.get(
                CONF_NO_UPDATE_ACTIVE_TIMEOUT,
                DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
            )
        )
        self._learning_confidence = config_entry.options.get(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE)
        self._duration_tolerance = config_entry.options.get(CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE)
        self._auto_label_confidence = config_entry.options.get(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE)
        self._auto_label_confidence = config_entry.options.get(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE)
        self._profile_match_interval = int(config_entry.options.get(CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL))
        self._notify_before_end_minutes = int(config_entry.options.get(CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES))
        
        # Advanced options
        smoothing_window = int(config_entry.options.get("smoothing_window", 5))
        interrupted_min_seconds = int(config_entry.options.get("interrupted_min_seconds", 150))
        abrupt_drop_watts = float(config_entry.options.get("abrupt_drop_watts", 500.0))
        abrupt_drop_ratio = float(config_entry.options.get("abrupt_drop_ratio", 0.6))
        abrupt_high_load_factor = float(config_entry.options.get("abrupt_high_load_factor", 5.0))
        completion_min_seconds = int(config_entry.options.get(CONF_COMPLETION_MIN_SECONDS, DEFAULT_COMPLETION_MIN_SECONDS))

        _LOGGER.info(f"Manager init: min_power={min_power}W, off_delay={off_delay}s (from options={CONF_MIN_POWER in config_entry.options}, defaults={DEFAULT_MIN_POWER}W, {DEFAULT_OFF_DELAY}s)")
        
        config = CycleDetectorConfig(
            min_power=float(min_power),
            off_delay=int(off_delay),
            smoothing_window=smoothing_window,
            interrupted_min_seconds=interrupted_min_seconds,
            abrupt_drop_watts=abrupt_drop_watts,
            abrupt_drop_ratio=abrupt_drop_ratio,
            abrupt_high_load_factor=abrupt_high_load_factor,
            completion_min_seconds=completion_min_seconds,
        )
        self._config = config
        self.detector = CycleDetector(
            config,
            self._on_state_change,
            self._on_cycle_end
        )
        
        self._remove_listener = None
        self._remove_watchdog = None
        self._watchdog_interval = int(config_entry.options.get(CONF_WATCHDOG_INTERVAL, DEFAULT_WATCHDOG_INTERVAL))
        self._noise_events_threshold = int(config_entry.options.get(CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD, DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD))
        self._current_program = "off"
        self._time_remaining: float | None = None
        self._cycle_progress: float = 0.0
        self._cycle_completed_time: datetime | None = None  # Track when cycle finished
        self._progress_reset_delay: int = int(progress_reset_delay)  # Reset progress after idle
        self._last_reading_time: datetime | None = None
        self._current_power: float = 0.0
        self._last_estimate_time: datetime | None = None
        self._matched_profile_duration: float | None = None
        self._last_match_confidence: float = 0.0  # Store confidence for feedback
        # Sample interval tracking (seconds) for adaptive timing
        self._sample_intervals: list[float] = []
        self._sample_interval_stats: dict[str, float | int | None] = {
            "median": None,
            "p95": None,
            "count": 0,
        }
        # Profile matching duration tolerance (0.25 = ±25%)
        self._profile_duration_tolerance: float = float(
            config_entry.options.get("profile_duration_tolerance", 0.25)
        )
        # Auto-merge controls
        self._auto_merge_lookback_hours: int = int(
            config_entry.options.get("auto_merge_lookback_hours", 3)
        )
        self._auto_merge_gap_seconds: int = int(
            config_entry.options.get("auto_merge_gap_seconds", 1800)
        )
        self._remove_maintenance_scheduler = None
        self._profile_sample_repair_stats: dict[str, int] | None = None
        self._remove_maintenance_scheduler = None
        self._profile_sample_repair_stats: dict[str, int] | None = None
        self._last_suggestion_update: datetime | None = None
        
        self._manual_program_active: bool = False
        self._notified_pre_completion: bool = False

    async def async_setup(self) -> None:
        """Set up the manager."""
        await self.profile_store.async_load()
        # Apply configurable duration tolerance to profile store
        try:
            self.profile_store.set_duration_tolerance(self._profile_duration_tolerance)
            self.profile_store.set_retention_limits(
                max_past_cycles=int(self.config_entry.options.get(CONF_MAX_PAST_CYCLES, DEFAULT_MAX_PAST_CYCLES)),
                max_full_traces_per_profile=int(self.config_entry.options.get(CONF_MAX_FULL_TRACES_PER_PROFILE, DEFAULT_MAX_FULL_TRACES_PER_PROFILE)),
                max_full_traces_unlabeled=int(self.config_entry.options.get(CONF_MAX_FULL_TRACES_UNLABELED, DEFAULT_MAX_FULL_TRACES_UNLABELED)),
            )
        except Exception:
            pass

        # Repair broken sample_cycle_id references (can happen after aggressive retention)
        try:
            stats = self.profile_store.repair_profile_samples()
            self._profile_sample_repair_stats = stats
            if stats.get("profiles_repaired", 0) or stats.get("cycles_labeled_as_sample", 0):
                _LOGGER.warning(
                    "Repaired profile sample references for %s: %s",
                    self.entry_id,
                    stats,
                )
                await self.profile_store.async_save()
        except Exception:
            _LOGGER.exception("Failed repairing profile sample references for %s", self.entry_id)
        
        # Subscribe to power sensor updates
        self._remove_listener = async_track_state_change_event(
            self.hass, [self.power_sensor_entity_id], self._async_power_changed
        )
        
        # Force initial update from current state (in case it's already stable)
        state = self.hass.states.get(self.power_sensor_entity_id)
        if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            try:
                power = float(state.state)
                now = dt_util.now()
                self.detector.process_reading(power, now)
            except (ValueError, TypeError):
                pass

    async def async_reload_config(self, config_entry: ConfigEntry) -> None:
        """
        Reload configuration options without interrupting running cycle detection.
        
        Updates detector config in-place so running cycles immediately use new settings.
        This includes min_power, off_delay, smoothing_window, and all abrupt drop parameters.
        """
        _LOGGER.info("Reloading configuration for %s", self.entry_id)
        # Replace reference so future reads use the latest options
        self.config_entry = config_entry
        
        # Update detector config in-place (for running cycle to use new settings immediately)
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
        new_off_delay = int(
            config_entry.options.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY)
        )
        new_smoothing = int(
            config_entry.options.get(CONF_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW)
        )
        new_interrupted_min = int(
            config_entry.options.get(CONF_INTERRUPTED_MIN_SECONDS, DEFAULT_INTERRUPTED_MIN_SECONDS)
        )
        new_abrupt_drop_watts = float(
            config_entry.options.get(CONF_ABRUPT_DROP_WATTS, DEFAULT_ABRUPT_DROP_WATTS)
        )
        new_abrupt_drop_ratio = float(
            config_entry.options.get(CONF_ABRUPT_DROP_RATIO, DEFAULT_ABRUPT_DROP_RATIO)
        )
        new_abrupt_high_load = float(
            config_entry.options.get(CONF_ABRUPT_HIGH_LOAD_FACTOR, DEFAULT_ABRUPT_HIGH_LOAD_FACTOR)
        )
        new_completion_min = int(
            config_entry.options.get(CONF_COMPLETION_MIN_SECONDS, DEFAULT_COMPLETION_MIN_SECONDS)
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
        
        if (old_min_power != new_min_power or old_off_delay != new_off_delay or
            old_smoothing != new_smoothing or old_interrupted_min != new_interrupted_min or
            old_abrupt_drop_watts != new_abrupt_drop_watts or old_abrupt_drop_ratio != new_abrupt_drop_ratio or
            old_abrupt_high_load != new_abrupt_high_load):
            _LOGGER.info(
                "Updated detector config: min_power %.1fW→%.1fW, off_delay %ds→%ds, "
                "smoothing %d→%d, interrupted_min %ds→%ds, abrupt_drop %.0fW→%.0fW, "
                "abrupt_ratio %.2f→%.2f, high_load %.1f→%.1f",
                old_min_power, new_min_power, old_off_delay, new_off_delay,
                old_smoothing, new_smoothing, old_interrupted_min, new_interrupted_min,
                old_abrupt_drop_watts, new_abrupt_drop_watts, old_abrupt_drop_ratio, new_abrupt_drop_ratio,
                old_abrupt_high_load, new_abrupt_high_load
            )
        
        # Update profile matching parameters
        old_min_ratio, old_max_ratio = self.profile_store.get_duration_ratio_limits()
        
        new_min_ratio = float(
            config_entry.options.get(CONF_PROFILE_MATCH_MIN_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO)
        )
        new_max_ratio = float(
            config_entry.options.get(CONF_PROFILE_MATCH_MAX_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO)
        )
        
        if old_min_ratio != new_min_ratio or old_max_ratio != new_max_ratio:
            self.profile_store.set_duration_ratio_limits(min_ratio=new_min_ratio, max_ratio=new_max_ratio)
            _LOGGER.info(
                "Updated duration ratios: min %.2f→%.2f, max %.2f→%.2f",
                old_min_ratio, new_min_ratio, old_max_ratio, new_max_ratio
            )
        
        # Update match interval
        old_interval = self._profile_match_interval
        new_interval = int(
            config_entry.options.get(CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL)
        )
        if old_interval != new_interval:
            self._profile_match_interval = new_interval
            _LOGGER.info("Updated match interval: %ds→%ds", old_interval, new_interval)
        
        # Update other configurable options
        self._profile_duration_tolerance = float(
            config_entry.options.get(CONF_PROFILE_DURATION_TOLERANCE, DEFAULT_PROFILE_DURATION_TOLERANCE)
        )
        self._auto_merge_lookback_hours = int(
            config_entry.options.get(CONF_AUTO_MERGE_LOOKBACK_HOURS, DEFAULT_AUTO_MERGE_LOOKBACK_HOURS)
        )
        self._auto_merge_gap_seconds = int(
            config_entry.options.get(CONF_AUTO_MERGE_GAP_SECONDS, DEFAULT_AUTO_MERGE_GAP_SECONDS)
        )
        
        # Update notification settings
        self._notify_service = config_entry.options.get(CONF_NOTIFY_SERVICE)
        self._notify_events = config_entry.options.get(CONF_NOTIFY_EVENTS, [])
        self._notify_before_end_minutes = int(config_entry.options.get(CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES))
        
        _LOGGER.info("Configuration reloaded successfully")
        
        # Trigger entity updates to reflect any changes
        async_dispatcher_send(self.hass, f"ha_washdata_update_{self.entry_id}")
        
        # Schedule midnight maintenance if enabled
        await self._setup_maintenance_scheduler()
        
        # RESTORE STATE (only if recent enough, otherwise treat as stale)
        active_snapshot = self.profile_store.get_active_cycle()
        if active_snapshot:
            # Check current power state first - if it's off/low, the cycle is definitely not running
            state = self.hass.states.get(self.power_sensor_entity_id)
            current_power = 0.0
            if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                try:
                    current_power = float(state.state)
                except (ValueError, TypeError):
                    pass
            
            # If current power is below threshold, don't restore running state
            if current_power < self._config.min_power:
                _LOGGER.info(f"Current power {current_power}W is below threshold, clearing stale active cycle")
                await self.profile_store.async_clear_active_cycle()
            else:
                # Check if the saved state is recent (within last 30 minutes)
                # If older, it's likely stale from a code update or restart
                try:
                    last_save = self.profile_store.get_last_active_save()
                    if last_save:
                        time_since_save = (datetime.now() - last_save).total_seconds()
                        # Only restore if saved within last 10 minutes
                        if time_since_save < 600:
                            self.detector.restore_state_snapshot(active_snapshot)
                            if self.detector.state == "running":
                                self._current_program = "restored..."
                                self._start_watchdog()  # Resume watchdog for restored cycle
                                _LOGGER.info("Restored interrupted washer cycle.")
                        else:
                            _LOGGER.info(f"Active cycle too stale ({time_since_save}s old), clearing")
                            await self.profile_store.async_clear_active_cycle()
                    else:
                        # No timestamp, clear it to be safe
                        await self.profile_store.async_clear_active_cycle()
                except Exception as err:
                    _LOGGER.warning(f"Failed to restore active cycle: {err}, clearing")
                    await self.profile_store.async_clear_active_cycle()
        
        _LOGGER.info("Configuration reloaded successfully")

    async def async_shutdown(self) -> None:
        """Shutdown."""
        if self._remove_listener:
            self._remove_listener()
        if self._remove_watchdog:
            self._remove_watchdog()
        if hasattr(self, "_remove_progress_reset_timer") and self._remove_progress_reset_timer:
            self._remove_progress_reset_timer()
        if self._remove_maintenance_scheduler:
            self._remove_maintenance_scheduler()
            
        # Try to save state one last time?
        if self.detector.state == "running":
             await self.profile_store.async_save_active_cycle(self.detector.get_state_snapshot())

        self._last_reading_time = None

    async def _setup_maintenance_scheduler(self) -> None:
        """Set up daily maintenance task at midnight."""
        auto_maintenance = self.config_entry.options.get(
            CONF_AUTO_MAINTENANCE,
            self.config_entry.data.get(CONF_AUTO_MAINTENANCE, DEFAULT_AUTO_MAINTENANCE)
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
                stats = await self.profile_store.async_run_maintenance(
                    lookback_hours=self._auto_merge_lookback_hours,
                    gap_seconds=self._auto_merge_gap_seconds,
                )
                _LOGGER.info(f"Maintenance completed: {stats}")
            except Exception as err:
                _LOGGER.error(f"Maintenance failed: {err}", exc_info=True)
        
        # Use async_track_point_in_time for midnight, then reschedule daily
        self._remove_maintenance_scheduler = evt.async_track_point_in_time(
            self.hass, run_maintenance, next_midnight
        )
        
        _LOGGER.info(f"Scheduled maintenance at {next_midnight}")
        
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

        now = dt_util.now()
        # Throttle updates to avoid CPU overload on noisy sensors
        if self._last_reading_time and (now - self._last_reading_time).total_seconds() < 2.0:
            return
            
        # Track observed sample interval for adaptive timing
        self._record_sample_interval(now)
        self._last_reading_time = now
        self._current_power = power
        self.detector.process_reading(power, now)
        
        # If running, try to match profile and update estimates
        if self.detector.state == "running":
            self._update_estimates()
            # Periodically save state (e.g. every minute?)
            # Doing it every reading (2s) is too much for flash storage.
            # Let's do it if 60s has passed since last save?
            # We need a tracker.
            self._check_state_save(now)
        elif self.detector.state == "off" and self._current_program == "detecting...":
            # Cycle just ended with no profile matched - run final match
            self._run_final_profile_match()
             
        self._notify_update()

    def _run_final_profile_match(self) -> None:
        """Run one final profile match after cycle completion if no profile was detected."""
        # Check if we have a completed cycle in storage
        past_cycles = self.profile_store.get_past_cycles()
        if not past_cycles:
            _LOGGER.debug("No past cycles to match against")
            return
        
        # Get the most recent cycle (sort by start_time descending to be safe)
        try:
            latest_cycle = max(
                past_cycles,
                key=lambda c: c.get("start_time", ""),
            )
        except Exception:
            latest_cycle = past_cycles[-1]
        cycle_id = latest_cycle.get("id")
        duration = latest_cycle.get("duration", 0)
        power_data = latest_cycle.get("power_data", [])
        
        if not power_data:
            _LOGGER.debug("Latest cycle has no power data")
            return
        
        # Convert compressed data to time-series format for matching
        start_time_raw = latest_cycle.get("start_time")
        if not isinstance(start_time_raw, str):
            return
        start_time = dt_util.parse_datetime(start_time_raw)
        if not start_time:
            return
        
        time_series = [
            ((start_time + timedelta(seconds=offset)).isoformat(), power)
            for offset, power in power_data
        ]
        
        _LOGGER.info(
            f"Running final profile match for completed cycle {cycle_id}: "
            f"{len(time_series)} samples, {duration:.0f}s duration"
        )
        
        profile_name, confidence = self.profile_store.match_profile(
            time_series,
            duration,
        )
        
        if profile_name and confidence >= 0.15:
            _LOGGER.info(
                f"Final match found: '{profile_name}' with confidence {confidence:.3f}"
            )
            self._current_program = profile_name
            self._last_match_confidence = confidence
            # Don't update duration/progress since cycle is already complete
        else:
            _LOGGER.info(
                f"No confident match in final attempt (best: {profile_name}, conf={confidence:.3f})"
            )

    def _check_state_save(self, now: datetime) -> None:
        """Periodically save active state."""
        last_save = getattr(self, "_last_state_save", None)
        if not last_save or (now - last_save).total_seconds() > 60:
             # Fire and forget save task
             self.hass.async_create_task(
                 self.profile_store.async_save_active_cycle(self.detector.get_state_snapshot())
             )
             self._last_state_save = now

    def _start_watchdog(self) -> None:
        """Start the watchdog timer when a cycle begins."""
        if self._remove_watchdog:
            return  # Already running
        
        interval = self._get_effective_watchdog_interval()
        _LOGGER.debug(
            "Starting watchdog timer (configured=%ss, effective=%.1fs)",
            self._watchdog_interval,
            interval,
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

    def _start_progress_reset_timer(self) -> None:
        """Start timer to reset progress to 0% after idle period (user unload time)."""
        # Use watchdog mechanism but with longer interval for progress reset
        if not hasattr(self, "_remove_progress_reset_timer"):
            self._remove_progress_reset_timer = None
        
        if self._remove_progress_reset_timer:
            return  # Already running
        
        _LOGGER.debug(f"Starting progress reset timer (will reset after {self._progress_reset_delay}s)")
        self._remove_progress_reset_timer = async_track_time_interval(
            self.hass, self._check_progress_reset, timedelta(seconds=10)  # Check every 10s
        )

    def _stop_progress_reset_timer(self) -> None:
        """Stop the progress reset timer."""
        if hasattr(self, "_remove_progress_reset_timer") and self._remove_progress_reset_timer:
            _LOGGER.debug("Stopping progress reset timer")
            self._remove_progress_reset_timer()
            self._remove_progress_reset_timer = None

    async def _check_progress_reset(self, now: datetime) -> None:
        """Check if progress should be reset (user unload timeout)."""
        if not self._cycle_completed_time or self.detector.state == STATE_RUNNING:
            # Cycle is running or not completed, don't reset
            return
        
        time_since_complete = (now - self._cycle_completed_time).total_seconds()
        
        if time_since_complete > self._progress_reset_delay:
            # User has had enough time to unload, reset to 0%
            _LOGGER.debug(f"Progress reset: cycle idle for {time_since_complete:.0f}s (threshold: {self._progress_reset_delay}s)")
            self._cycle_progress = 0.0
            self._cycle_completed_time = None
            self._stop_progress_reset_timer()
            self._notify_update()

    async def _watchdog_check_stuck_cycle(self, now: datetime) -> None:
        """Watchdog: check if cycle is stuck (no updates for too long)."""
        if self.detector.state != STATE_RUNNING:
            return
        
        # Handle publish-on-change sockets: only force-complete quickly if we're already in low-power waiting;
        # otherwise wait for a longer active-timeout before force-stopping.
        if self._last_reading_time:
            time_since_update = (now - self._last_reading_time).total_seconds()

            # Case 1: In low-power waiting and exceeded off_delay → complete naturally via force_end
            if self.detector.is_waiting_low_power() and self.detector.low_power_elapsed(now) >= self._config.off_delay:
                _LOGGER.info(
                    "Watchdog: finalizing cycle after low-power wait (no update for %.0fs)",
                    time_since_update,
                )
                self.detector.force_end(now)
                self._last_reading_time = now
                self._current_power = 0.0
                self._notify_update()
                return

            # Case 1.5: No updates for > off_delay seconds → inject 0W reading to flush buffer
            # This handles publish-on-change sensors that stop sending 0W updates
            if time_since_update > self._config.off_delay and not self.detector.is_waiting_low_power():
                _LOGGER.info(
                    "Watchdog: no updates for %.0fs (> off_delay %.0fs), injecting 0W to flush smoothing buffer",
                    time_since_update,
                    self._config.off_delay,
                )
                self._current_power = 0.0
                self.detector.process_reading(0.0, now)
                self._last_reading_time = now
                # Don't return - let normal cycle end logic handle it
                self._notify_update()
                return

            # Case 2: Not in low power; if no updates for a long time, sensor likely offline → force stop
            if time_since_update > self._no_update_active_timeout:
                _LOGGER.warning(
                    "Watchdog: no power updates for %.0fs while active (timeout %.0fs), force-stopping",
                    time_since_update,
                    self._no_update_active_timeout,
                )
                self.detector.force_end(now)
                self._last_reading_time = now
                self._current_power = 0.0
                self._notify_update()
                if self.detector.state == STATE_RUNNING:
                    _LOGGER.error("Watchdog: cycle still running after forced end, will retry next check")

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """Handle state change from detector."""
        _LOGGER.debug(f"Washer state changed: {old_state} -> {new_state}")
        if new_state == "running":
            # If a cycle finishes and a new one starts within the reset window,
            # treat it as continuation or quick restart (don't reset progress yet)
            self._cycle_completed_time = None
            self._stop_progress_reset_timer()
            
            self._current_program = "detecting..."
            self._manual_program_active = False
            self._notified_pre_completion = False
            self._time_remaining = None
            self._cycle_progress = 0
            self._matched_profile_duration = None
            self._last_estimate_time = None
            self._start_watchdog()  # Start watchdog when cycle starts
            self.hass.bus.async_fire(EVENT_CYCLE_STARTED, {"entry_id": self.entry_id, "device_name": self.config_entry.title})
            
            # Send notification if enabled
            events = self.config_entry.options.get(CONF_NOTIFY_EVENTS, [])
            if NOTIFY_EVENT_START in events:
                self._send_notification(f"{self.config_entry.title} started.")
        elif new_state == STATE_OFF and old_state == STATE_RUNNING:
            self._stop_watchdog()  # Stop watchdog when cycle ends
            
        self._notify_update()

    def _on_cycle_end(self, cycle_data: dict[str, Any]) -> None:
        """Handle cycle end - clear all active timers and state."""
        duration = cycle_data["duration"]
        max_power = cycle_data.get("max_power", 0)
        
        # IMMEDIATELY stop all active timers when cycle determined to have ended
        self._stop_watchdog()  # Stop active cycle watchdog
        self._stop_progress_reset_timer()  # Cancel any pending progress reset
        
        # Auto-Tune: Check for ghost cycles (short duration, lowish power)
        # Definition of noise: duration < 120s
        if duration < 120:
             self._handle_noise_cycle(max_power)
             # We still save it as a cycle for history, or maybe we shouldn't?
             # Let's save it but marked as potential noise? 
             # For now save as normal.
        
        # If we had a runtime match, attach the profile name for persistence
        if (
            self._current_program
            and self._current_program not in ("off", "detecting...", "restored...")
            and self._current_program in self.profile_store.get_profiles()
        ):
            cycle_data["profile_name"] = self._current_program

        # Add cycle to store immediately so it has an ID and can be referenced by
        # feedback/auto-label logic synchronously.
        try:
            self.profile_store.add_cycle(cycle_data)
            profile_name = cycle_data.get("profile_name")
            if profile_name:
                self.profile_store.rebuild_envelope(profile_name)
            self.hass.async_create_task(self.profile_store.async_save())
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Failed to add/save cycle")

        # Ensure cycle has a stable ID even if store add failed (or did not mutate).
        if not cycle_data.get("id"):
            try:
                import hashlib

                unique_str = f"{cycle_data['start_time']}_{cycle_data['duration']}"
                cycle_data["id"] = hashlib.sha256(unique_str.encode()).hexdigest()[:12]
            except Exception:  # noqa: BLE001
                pass

        self.hass.async_create_task(self.profile_store.async_clear_active_cycle())
        
        # Auto post-process: merge fragmented cycles from last 3 hours
        self.hass.async_create_task(self._auto_merge_recent_cycles())
        
        self.hass.bus.async_fire(EVENT_CYCLE_ENDED, {"entry_id": self.entry_id, "device_name": self.config_entry.title, "cycle_data": cycle_data})
        
        # Send notification if enabled
        events = self.config_entry.options.get(CONF_NOTIFY_EVENTS, [])
        if NOTIFY_EVENT_FINISH in events:
             self._send_notification(f"{self.config_entry.title} finished. Duration: {int(duration/60)}m.")
        
        # Request user feedback if we had a confident match.
        # IMPORTANT: this must happen before we clear match state.
        self._maybe_request_feedback(cycle_data)

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
        self._start_progress_reset_timer()
        
        self._notify_update()

    def _record_sample_interval(self, now: datetime) -> None:
        """Record the interval between power updates for adaptive timing."""
        if self._last_reading_time:
            delta = (now - self._last_reading_time).total_seconds()
            if delta > 0.1:  # ignore ultra-small jitter
                self._sample_intervals.append(delta)
                # keep a reasonable window
                if len(self._sample_intervals) > 200:
                    self._sample_intervals = self._sample_intervals[-200:]
                self._compute_sample_interval_stats()

    def _compute_sample_interval_stats(self) -> None:
        """Compute median and p95 sample intervals for diagnostics and adaptation."""
        import numpy as np

        arr = np.array(self._sample_intervals)
        if arr.size == 0:
            self._sample_interval_stats = {"median": None, "p95": None, "count": 0}
            return

        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        self._sample_interval_stats = {
            "median": median,
            "p95": p95,
            "count": int(arr.size),
        }

        # Update suggested settings based on observed sensor cadence (no auto-overrides)
        try:
            self._update_sampling_suggestions(median=median, p95=p95, count=int(arr.size))
        except Exception:
            _LOGGER.exception("Failed to update sampling-based suggestions")

    def _set_suggestion(self, key: str, value: float | int, reason: str) -> bool:
        """Persist a suggested setting value without changing HA options."""
        current_raw = self.profile_store.get_suggestions().get(key)
        current: dict[str, Any] = cast(dict[str, Any], current_raw) if isinstance(current_raw, dict) else {}
        current_value = current.get("value")
        # Avoid churn: only update if materially different
        if isinstance(current_value, (int, float)) and abs(float(current_value) - float(value)) < 0.01:
            return False
        self.profile_store.set_suggestion(key, value, reason=reason)
        self.hass.async_create_task(self.profile_store.async_save())
        return True

    def _update_sampling_suggestions(self, median: float, p95: float, count: int) -> None:
        """Suggest timing-related options based on sensor update cadence.

        Goal: keep detection responsive while avoiding false watchdog triggers for slow or
        publish-on-change sensors.
        """
        if count < 20:
            return

        now = dt_util.now()
        if self._last_suggestion_update and (now - self._last_suggestion_update).total_seconds() < 1800:
            return

        # Suggest watchdog interval ~ p95 cadence (checking faster doesn't help much)
        suggested_watchdog = int(max(1, round(p95)))
        suggested_watchdog = min(suggested_watchdog, int(max(1, self._config.off_delay / 2)))
        self._set_suggestion(
            CONF_WATCHDOG_INTERVAL,
            suggested_watchdog,
            reason=f"Based on observed update cadence (p95={p95:.1f}s).",
        )

        # Suggest match interval to require ~10 samples between matches
        suggested_match = int(max(10, round(median * 10)))
        self._set_suggestion(
            CONF_PROFILE_MATCH_INTERVAL,
            suggested_match,
            reason=f"Based on observed update cadence (median={median:.1f}s).",
        )

        # Suggest no-update timeout to reduce false force-ends on slow sensors
        # Keep it at least one off_delay, and ~4x p95 cadence.
        suggested_no_update = int(max(self._config.off_delay, round(p95 * 4)))
        self._set_suggestion(
            CONF_NO_UPDATE_ACTIVE_TIMEOUT,
            suggested_no_update,
            reason=f"Based on observed update cadence (p95={p95:.1f}s) and off_delay={self._config.off_delay}s.",
        )

        # Suggest duration tolerances based on observed cycle duration variance
        try:
            var_p95 = self._compute_duration_variance_p95()
            if var_p95 is not None:
                suggested_tol = float(min(0.5, max(0.05, round(var_p95 + 0.05, 2))))
                self._set_suggestion(
                    CONF_DURATION_TOLERANCE,
                    suggested_tol,
                    reason=f"Based on recent labeled cycles vs profile avg_duration (p95 variance={var_p95:.2f}).",
                )
                self._set_suggestion(
                    CONF_PROFILE_DURATION_TOLERANCE,
                    suggested_tol,
                    reason=f"Based on recent labeled cycles vs profile avg_duration (p95 variance={var_p95:.2f}).",
                )
        except Exception:
            _LOGGER.exception("Failed to compute duration-variance suggestions")

        # Suggest off_delay based on sensor cadence (ensure enough below-threshold samples)
        try:
            # Ensure at least ~4 p95 intervals (and at least default), cap at 10 minutes
            suggested_off_delay = int(min(600, max(60, round(max(DEFAULT_OFF_DELAY, p95 * 4)))))
            self._set_suggestion(
                CONF_OFF_DELAY,
                suggested_off_delay,
                reason=f"Based on observed update cadence (p95={p95:.1f}s).",
            )
        except Exception:
            _LOGGER.exception("Failed to compute off_delay suggestion")

        # Suggest profile match min/max duration ratios based on observed labeled duration ratios
        try:
            bounds = self._compute_duration_ratio_bounds()
            if bounds is not None:
                min_ratio, max_ratio, sample_count = bounds
                self._set_suggestion(
                    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                    float(round(min_ratio, 2)),
                    reason=f"Based on labeled cycles vs profile avg_duration (n={sample_count}).",
                )
                self._set_suggestion(
                    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                    float(round(max_ratio, 2)),
                    reason=f"Based on labeled cycles vs profile avg_duration (n={sample_count}).",
                )
        except Exception:
            _LOGGER.exception("Failed to compute match duration ratio suggestions")

        # Suggest auto-merge gap based on observed fragmentation gaps
        try:
            gap = self._compute_fragmentation_gap_suggestion_seconds()
            if gap is not None:
                suggested_gap, sample_count = gap
                self._set_suggestion(
                    CONF_AUTO_MERGE_GAP_SECONDS,
                    int(suggested_gap),
                    reason=f"Based on observed fragmented runs (n={sample_count}).",
                )
        except Exception:
            _LOGGER.exception("Failed to compute auto-merge gap suggestion")

        # Suggest auto-label confidence based on persisted feedback confirmation/correction rates
        try:
            suggested = self._compute_auto_label_confidence_suggestion()
            if suggested is not None:
                value, stats = suggested
                self._set_suggestion(
                    CONF_AUTO_LABEL_CONFIDENCE,
                    float(round(value, 2)),
                    reason=stats,
                )
        except Exception:
            _LOGGER.exception("Failed to compute auto-label confidence suggestion")

        self._last_suggestion_update = now

    def _compute_duration_ratio_bounds(self) -> tuple[float, float, int] | None:
        """Compute suggested min/max duration ratios from labeled cycles.

        Uses ratio = duration / profile.avg_duration for labeled, non-interrupted cycles.
        Returns (min_ratio, max_ratio, sample_count) or None.
        """
        profiles = self.profile_store.get_profiles()
        cycles = self.profile_store.get_past_cycles()

        ratios: list[float] = []
        for c in cycles[-140:]:
            try:
                profile_name = c.get("profile_name")
                if not profile_name:
                    continue
                if c.get("status") == "interrupted":
                    continue
                profile = profiles.get(str(profile_name))
                if not isinstance(profile, dict):
                    continue
                avg = float(profile.get("avg_duration") or 0)
                dur = float(c.get("duration") or 0)
                if avg <= 1.0 or dur <= 1.0:
                    continue
                ratios.append(dur / avg)
            except Exception:
                continue

        if len(ratios) < 10:
            return None

        arr = np.array(ratios, dtype=float)
        p05 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))

        # Add a small margin so bounds are not too tight.
        min_ratio = max(0.10, min(1.00, p05 - 0.05))
        max_ratio = max(1.00, min(3.00, p95 + 0.05))

        # Ensure sane ordering
        if max_ratio <= min_ratio:
            max_ratio = min(3.00, min_ratio + 0.25)

        return (min_ratio, max_ratio, len(ratios))

    def _compute_fragmentation_gap_suggestion_seconds(self) -> tuple[int, int] | None:
        """Suggest auto-merge gap threshold from observed likely-fragmentation gaps.

        Heuristic: look for consecutive cycles with a small time gap where the first looks
        abnormal/short and the next completes normally. Returns (suggested_gap_seconds, n).
        """
        cycles = self.profile_store.get_past_cycles()
        if len(cycles) < 3:
            return None

        # Sort by start time
        try:
            ordered = sorted(cycles, key=lambda c: str(c.get("start_time", "")))
        except Exception:
            ordered = list(cycles)

        gaps: list[float] = []
        short_threshold = float(getattr(self._config, "interrupted_min_seconds", 150)) * 2.0

        for prev, nxt in zip(ordered, ordered[1:]):
            try:
                prev_end_raw = prev.get("end_time")
                nxt_start_raw = nxt.get("start_time")
                if not isinstance(prev_end_raw, str) or not isinstance(nxt_start_raw, str):
                    continue
                prev_end = dt_util.parse_datetime(prev_end_raw)
                nxt_start = dt_util.parse_datetime(nxt_start_raw)
                if not prev_end or not nxt_start:
                    continue
                gap_s = (nxt_start - prev_end).total_seconds()
                if gap_s <= 0:
                    continue
                if gap_s > 3600:
                    continue

                prev_status = prev.get("status")
                nxt_status = nxt.get("status")
                prev_dur = float(prev.get("duration") or 0)

                prev_profile = prev.get("profile_name")
                nxt_profile = nxt.get("profile_name")
                same_or_unknown = (
                    (prev_profile and nxt_profile and prev_profile == nxt_profile)
                    or (not prev_profile)
                    or (not nxt_profile)
                )
                if not same_or_unknown:
                    continue

                prev_looks_fragmented = (
                    prev_status in ("interrupted", "force_stopped")
                    or (prev_dur > 0 and prev_dur < short_threshold)
                )
                nxt_looks_complete = nxt_status in ("completed", "force_stopped")
                if not (prev_looks_fragmented and nxt_looks_complete):
                    continue

                gaps.append(gap_s)
            except Exception:
                continue

        if len(gaps) < 3:
            return None

        arr = np.array(gaps, dtype=float)
        p95 = float(np.percentile(arr, 95))
        # Add a 60s cushion, cap to 2 hours (schema max allows 7200)
        suggested = int(min(7200, max(60, round(p95 + 60))))
        return (suggested, len(gaps))

    def _compute_auto_label_confidence_suggestion(self) -> tuple[float, str] | None:
        """Suggest auto-label confidence threshold based on feedback correction rates."""
        history = self.profile_store.get_feedback_history()
        if len(history) < 10:
            return None

        records: list[tuple[float, bool]] = []
        for rec in history.values():
            try:
                conf_raw = rec.get("original_confidence")
                conf = float(conf_raw) if conf_raw is not None else 0.0
                confirmed = bool(rec.get("user_confirmed") is True)
            except Exception:
                continue
            if conf <= 0 or conf > 1.0:
                continue
            records.append((conf, confirmed))

        if len(records) < 10:
            return None

        # Find the lowest threshold where correction rate is acceptably low.
        target_correction = 0.05
        min_samples = 5

        best: float | None = None
        best_stats: str | None = None

        for t in np.arange(0.70, 0.991, 0.01):
            subset = [(c, ok) for (c, ok) in records if c >= float(t)]
            if len(subset) < min_samples:
                continue
            corrections = sum(1 for _, ok in subset if not ok)
            rate = corrections / len(subset)
            if rate <= target_correction:
                best = float(t)
                best_stats = (
                    f"Based on feedback: correction_rate={rate:.0%} at conf≥{t:.2f} (n={len(subset)})."
                )
                break

        if best is None:
            # Fallback: keep it conservative. If corrections are common, stay near default.
            best = float(DEFAULT_AUTO_LABEL_CONFIDENCE)
            total = len(records)
            corrections = sum(1 for _, ok in records if not ok)
            best_stats = (
                f"Based on feedback: overall correction_rate={corrections/total:.0%} (n={total}); keeping conservative default."
            )

        best = min(0.99, max(0.70, best))
        return (best, best_stats or "Based on feedback history.")

    def _compute_duration_variance_p95(self) -> float | None:
        """Compute p95 relative duration variance for labeled cycles.

        Uses current profile avg_duration as the reference. Returns None if insufficient data.
        """
        profiles = self.profile_store.get_profiles()
        cycles = self.profile_store.get_past_cycles()

        errors: list[float] = []
        for c in cycles[-80:]:
            profile_name = c.get("profile_name")
            if not profile_name:
                continue
            profile = profiles.get(str(profile_name))
            if not isinstance(profile, dict):
                continue
            avg = profile.get("avg_duration")
            dur = c.get("duration")
            if avg is None or dur is None:
                continue
            try:
                avg_f = float(avg)
                dur_f = float(dur)
            except Exception:
                continue
            if avg_f <= 1.0 or dur_f <= 1.0:
                continue
            errors.append(abs(dur_f - avg_f) / avg_f)

        if len(errors) < 5:
            return None
        arr = np.array(errors, dtype=float)
        return float(np.percentile(arr, 95))

    def _get_observed_sample_interval(self, default: float = 5.0) -> float:
        """Return median observed sample interval, or a default if unknown."""
        stats = self._sample_interval_stats
        median = stats.get("median")
        if isinstance(median, (int, float)):
            return max(0.5, float(median))
        return default

    def _get_effective_match_interval(self) -> float:
        """Clamp match interval based on observed sample interval (seconds)."""
        observed = self._get_observed_sample_interval(default=self._profile_match_interval)
        floor = observed * 10  # require roughly 10 samples between heavy matches
        effective = max(self._profile_match_interval, floor)
        # Don't let it explode; cap to 2x configured interval
        return min(effective, self._profile_match_interval * 2)

    def _get_effective_watchdog_interval(self) -> float:
        """Clamp watchdog interval so it respects slow sensors but stays responsive."""
        observed = self._get_observed_sample_interval(default=self._watchdog_interval)
        # Never shorter than configured, never shorter than observed cadence
        base = max(self._watchdog_interval, observed)
        # Avoid checking slower than half the off_delay to still conclude cycles
        upper = max(1.0, self._config.off_delay / 2)
        return min(base, upper)

    @property
    def sample_interval_stats(self) -> dict[str, float | int | None]:
        return dict(self._sample_interval_stats)

    @property
    def profile_sample_repair_stats(self) -> dict[str, int] | None:
        return self._profile_sample_repair_stats

    @property
    def suggestions(self) -> dict[str, Any]:
        """Suggested settings computed by learning/heuristics (never auto-applied)."""
        return self.profile_store.get_suggestions()

    def _send_notification(self, message: str) -> None:
        """Send a notification via configured service."""
        notify_service = self.config_entry.options.get(CONF_NOTIFY_SERVICE)
        if notify_service:
            domain, service = notify_service.split('.', 1) if '.' in notify_service else ("notify", notify_service)
            self.hass.async_create_task(
                self.hass.services.async_call(domain, service, {"message": message})
            )
        else:
            # Fallback for Auto-Tune, but maybe we shouldn't spam persistent for normal events?
            # User only selects events if they want them.
            # But if no service is selected, where do they go?
            # Let's assume persistent for now if they enabled the event but no service.
            _pn_create(self.hass, message, title=f"HA WashData: {self.config_entry.title}")

    def _handle_noise_cycle(self, max_power: float) -> None:
        """Handle a detected noise cycle."""
        # Clean up old noise events > 24h
        now = dt_util.now()
        self._noise_events = [t for t in getattr(self, "_noise_events", []) if (now - t).total_seconds() < 86400]
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
            domain, service = notify_service.split('.', 1) if '.' in notify_service else ("notify", notify_service)
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
        if self.detector.state != "running":
            return

        now = dt_util.now()

        # Throttle heavy matching to configured interval (default: 5 minutes)
        effective_match_interval = self._get_effective_match_interval()
        if self._last_estimate_time and (now - self._last_estimate_time).total_seconds() < effective_match_interval:
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

        trace = self.detector.get_power_trace()
        if len(trace) < 3:
            return

        duration_so_far = self.detector.get_elapsed_seconds()
        current_power_data = [(t.isoformat(), p) for t, p in trace]

        _LOGGER.debug(
            "Profile matching using complete cycle data: %d samples spanning %.0fs",
            len(trace), duration_so_far
        )

        profile_name, confidence = self.profile_store.match_profile(
            current_power_data,
            duration_so_far,
        )

        _LOGGER.info(f"Profile match attempt: name={profile_name}, confidence={confidence:.3f}, duration={duration_so_far:.0f}s, samples={len(trace)}")

        if profile_name and confidence >= 0.15:  # Lowered threshold from 0.2 to 0.15
            # Match found - update or keep existing match
            if not self._matched_profile_duration or self._current_program == "detecting...":
                # First match or no previous match
                self._current_program = profile_name
                self._last_match_confidence = confidence  # Store for later feedback
                profile = self.profile_store.get_profiles().get(profile_name, {})
                avg_duration = float(profile.get("avg_duration", 0.0))
                self._matched_profile_duration = avg_duration if avg_duration > 0 else None
                _LOGGER.info(f"Matched profile '{profile_name}' with expected duration {avg_duration:.0f}s ({int(avg_duration/60)}min)")
            # If we already have a match, keep it (don't thrash between profiles)
        elif not self._matched_profile_duration:
            # No match yet and still searching
            self._current_program = "detecting..."
        # else: keep existing match even if current attempt failed (prevents "unknown" flip-flop)
        
        # If manual program is active, we skip the matching logic update to _current_program
        # But we DO want to process the phase estimation below using the manually set program.
        # The logic above only updates _current_program if we are searching ("detecting...").
        # If manual mode is on, _current_program is already set and locked.

        self._last_estimate_time = now
        self._update_remaining_only()
        
        # Check for pre-completion notification
        self._check_pre_completion_notification()

        self._notify_update()

    def _check_pre_completion_notification(self) -> None:
        """Check and send pre-completion notification."""
        if (
            self._notify_before_end_minutes > 0
            and not self._notified_pre_completion
            and self._time_remaining is not None
            and self._time_remaining <= (self._notify_before_end_minutes * 60)
            and self._cycle_progress < 100
        ):
            # Send notification!
            self._notified_pre_completion = True
            msg = f"{self.config_entry.title}: Less than {self._notify_before_end_minutes} minutes remaining."
            self._send_notification(msg)
            _LOGGER.info(f"Sent pre-completion notification: {msg}")

    def _update_remaining_only(self) -> None:
        """Recompute remaining/progress using phase-aware estimation."""
        if self.detector.state != "running":
            self._time_remaining = None
            self._cycle_progress = 0.0
            return

        duration_so_far = self.detector.get_elapsed_seconds()

        if self._matched_profile_duration and self._matched_profile_duration > 0:
            # Get current power trace for phase analysis
            trace = self.detector.get_power_trace()
            current_power_data = [(t.isoformat(), p) for t, p in trace]
            
            # Try phase-aware estimation if we have enough data
            if len(trace) >= 10 and self._current_program != "detecting...":
                phase_progress = self._estimate_phase_progress(
                    current_power_data, 
                    duration_so_far,
                    self._current_program
                )
                if phase_progress is not None:
                    # Adjust time remaining based on phase progress
                    remaining = (self._matched_profile_duration * (1.0 - phase_progress / 100.0))
                    self._time_remaining = max(0.0, remaining)
                    
                    # Recalculate progress to be consistent with elapsed + remaining
                    # This ensures progress doesn't jump wildly if we are running slower/faster than average
                    total_predicted = duration_so_far + self._time_remaining
                    if total_predicted > 0:
                        self._cycle_progress = max(0.0, min(99.0, (duration_so_far / total_predicted) * 100.0))
                    else:
                        self._cycle_progress = 0.0

                    _LOGGER.debug(
                        f"Phase-aware estimate: phase={phase_progress:.1f}%, "
                        f"remaining={int(remaining/60)}min, progress={self._cycle_progress:.1f}%"
                    )
                    return
            
            # Fallback to linear estimation if phase analysis unavailable
            remaining = max(self._matched_profile_duration - duration_so_far, 0.0)
            self._time_remaining = remaining
            progress = (duration_so_far / self._matched_profile_duration) * 100.0
            self._cycle_progress = max(0.0, min(progress, 100.0))
            _LOGGER.debug(f"Linear estimate: remaining={int(remaining/60)}min, progress={progress:.1f}%")
        else:
            self._time_remaining = None
            self._cycle_progress = 0.0
            _LOGGER.debug(f"No profile matched yet, elapsed={int(duration_so_far/60)}min")

    def _estimate_phase_progress(
        self, 
        current_power_data: list[tuple[str, float]], 
        current_duration: float,
        profile_name: str
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
            _LOGGER.debug(f"No envelope cached for profile {profile_name}")
            return None
        
        # Convert cached lists back to numpy arrays
        try:
            envelope_arrays = {
                "min": np.array(envelope["min"]),
                "max": np.array(envelope["max"]),
                "avg": np.array(envelope["avg"]),
                "std": np.array(envelope["std"]),
            }
            time_grid = np.array(envelope.get("time_grid", []))
            target_duration = envelope.get("target_duration", 0)
        except (KeyError, ValueError) as e:
            _LOGGER.warning(f"Invalid envelope format for {profile_name}: {e}")
            return None
        
        if len(time_grid) == 0 or target_duration <= 0:
            _LOGGER.debug("Envelope missing time grid, cannot estimate phase")
            return None
        
        # Extract power values and offsets from current cycle
        current_offsets = np.array([float(t) if isinstance(t, (int, float)) else 
                                     (datetime.fromisoformat(t).timestamp() - 
                                      datetime.fromisoformat(current_power_data[0][0]).timestamp())
                                     for t, _ in current_power_data])
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
            envelope_window_end = min(i + len(current_window_values), len(envelope_arrays["avg"]))
            
            if envelope_window_end <= envelope_window_start:
                continue
            
            avg_window = envelope_arrays["avg"][envelope_window_start:envelope_window_end]
            min_window = envelope_arrays["min"][envelope_window_start:envelope_window_end]
            max_window = envelope_arrays["max"][envelope_window_start:envelope_window_end]
            
            # Interpolate envelope to match current window length if needed
            if len(avg_window) != len(current_window_values):
                x_old = np.linspace(0, 1, len(avg_window))
                x_new = np.linspace(0, 1, len(current_window_values))
                avg_window = np.interp(x_new, x_old, avg_window)
                min_window = np.interp(x_new, x_old, min_window)
                max_window = np.interp(x_new, x_old, max_window)
            
            # Check if current power is within expected bounds (±20% tolerance)
            within_bounds = np.all(
                (current_window_values >= min_window * 0.8) & 
                (current_window_values <= max_window * 1.2)
            )
            bounds_score = np.mean(
                (current_window_values >= min_window) & 
                (current_window_values <= max_window)
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
                    0.4 * max(correlation, 0.0) +      # Shape matching
                    0.3 * mae_normalized +              # Amplitude matching
                    0.3 * bounds_score                  # Within expected range
                )
                
                # Penalize matches that are far from current elapsed time (assume linear progress is roughly correct)
                # This prevents wild jumps in time remaining when patterns repeat
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
            except:
                continue
        
        if best_progress is None or best_score < 0.4:
            _LOGGER.debug(f"Phase detection failed: best_score={best_score:.3f}")
            return None
        
        # Cap progress at 99% until actual completion
        best_progress = max(0.0, min(best_progress, 99.0))
        
        # Log with envelope metadata
        cycle_count = envelope.get("cycle_count", 0)
        avg_sample_rates = envelope.get("sampling_rates", [1.0])
        avg_sample_rate = np.median(avg_sample_rates) if avg_sample_rates else 1.0
        
        tws = best_time_window_start if best_time_window_start is not None else float(current_duration)
        if not in_bounds:
            _LOGGER.debug(
                f"Phase detection: progress={best_progress:.1f}%, "
                f"score={best_score:.3f}, time={tws:.0f}/{target_duration:.0f}s "
                f"[OUT OF BOUNDS, {cycle_count} cycles, avg_sample_rate={avg_sample_rate:.1f}s]"
            )
        else:
            _LOGGER.debug(
                f"Phase detection: progress={best_progress:.1f}%, "
                f"score={best_score:.3f}, time={tws:.0f}/{target_duration:.0f}s "
                f"[IN BOUNDS, {cycle_count} cycles, avg_sample_rate={avg_sample_rate:.1f}s]"
            )
        
        return best_progress

    def _notify_update(self) -> None:
        """Notify entities of update."""
        async_dispatcher_send(self.hass, SIGNAL_WASHER_UPDATE.format(self.entry_id))

    @property
    def check_state(self):
        return self.detector.state
    
    @property
    def current_program(self):
        return self._current_program
    
    @property
    def time_remaining(self):
        return self._time_remaining

    @property
    def cycle_progress(self):
        return self._cycle_progress

    @property
    def current_power(self):
        return self._current_power

    @property
    def samples_recorded(self):
        return len(self.detector.get_power_trace())

    @property
    def manual_program_active(self) -> bool:
        return getattr(self, "_manual_program_active", False)

    def set_manual_program(self, profile_name: str) -> None:
        """Manually set the current program."""
        if self.detector.state != "running":
            # Can we set it before start? Maybe, but usually makes sense during run
            # For now allow it only during run or just label it? 
            # Let's allow setting it, it will be "detecting..." initially but we force it.
            pass
        
        profiles_raw: Any = None
        try:
            profiles_raw = self.profile_store.get_profiles()
        except Exception:
            profiles_raw = None

        if isinstance(profiles_raw, dict):
            profiles: dict[str, Any] = cast(dict[str, Any], profiles_raw)
        else:
            profiles_fallback = getattr(self.profile_store, "_data", {}).get("profiles", {})
            profiles = cast(dict[str, Any], profiles_fallback) if isinstance(profiles_fallback, dict) else {}

        if profile_name not in profiles:
            _LOGGER.warning(f"Cannot set manual program: '{profile_name}' not found")
            return
            
        self._current_program = profile_name
        self._manual_program_active = True
        
        # Update expected duration immediately
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            _LOGGER.warning(f"Cannot set manual program: '{profile_name}' profile is invalid")
            return

        profile_dict = cast(dict[str, Any], profile)
        avg_raw = profile_dict.get("avg_duration")
        try:
            avg = float(avg_raw) if avg_raw is not None else 0.0
        except (TypeError, ValueError):
            avg = 0.0
        self._matched_profile_duration = avg if avg > 0 else None
        
        # Force estimate update
        self._update_remaining_only()
        self._notify_update()
        _LOGGER.info(f"Manual program set to '{profile_name}'")

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
        
        self._notify_update()
        _LOGGER.info("Manual program cleared, reverting to auto-detection")

    async def _auto_merge_recent_cycles(self) -> None:
        """Automatically merge fragmented cycles from the last 3 hours."""
        try:
            count = self.profile_store.merge_cycles(hours=self._auto_merge_lookback_hours, gap_threshold=self._auto_merge_gap_seconds)
            if count > 0:
                _LOGGER.info(f"Auto-merged {count} fragmented cycle(s)")
                await self.profile_store.async_save()
        except Exception as e:
            _LOGGER.error(f"Auto-merge failed: {e}")

    def _maybe_request_feedback(self, cycle_data: dict[str, Any]) -> None:
        """Request user feedback if we made a confident match, or auto-label if very high confidence."""
        if not self._matched_profile_duration or not self._current_program or self._current_program in ("off", "detecting..."):
            # No match was made, don't request feedback
            return

        # Get the cycle ID from the cycle_data
        cycle_id = cycle_data.get("id")
        if not cycle_id:
            _LOGGER.warning("Cycle data missing ID, cannot request feedback")
            return

        # Use stored confidence from matching
        confidence = self._last_match_confidence
        actual_duration = cycle_data.get("duration", 0)

        # Auto-label if very high confidence (configurable)
        if confidence >= self._auto_label_confidence:
            labeled = self.learning_manager.auto_label_high_confidence(
                cycle_id=cycle_id,
                profile_name=self._current_program,
                confidence=confidence,
                confidence_threshold=self._auto_label_confidence,
            )
            if labeled:
                # Persist label and rebuild envelope for the profile
                self.hass.async_create_task(self.profile_store.async_save())
                try:
                    self.profile_store.rebuild_envelope(self._current_program)
                except Exception:
                    pass
                _LOGGER.debug(f"Auto-labeled high-confidence cycle {cycle_id}")
            return

        # Skip low-confidence matches below learning threshold
        if confidence < self._learning_confidence:
            _LOGGER.debug(
                "Skipping feedback for low-confidence match (conf=%.2f < %.2f)",
                confidence,
                self._learning_confidence,
            )
            return

        # Request feedback via learning manager for moderate confidence
        self.learning_manager.request_cycle_verification(
            cycle_id=cycle_id,
            detected_profile=self._current_program,
            confidence=confidence,
            estimated_duration=self._matched_profile_duration,
            actual_duration=actual_duration,
            duration_tolerance=self._duration_tolerance,
        )

        # Persist pending feedback request so it survives restart
        self.hass.async_create_task(self.profile_store.async_save())

        # Emit event so UI/automations can react
        self.hass.bus.async_fire(
            FEEDBACK_REQUEST_EVENT,
            {
                "entry_id": self.entry_id,
                "cycle_id": cycle_id,
                "detected_profile": self._current_program,
                "confidence": confidence,
                "estimated_duration": int(self._matched_profile_duration / 60),
                "actual_duration": int(actual_duration / 60),
            },
        )

        # Also create a user-visible prompt. Without this, feedback requests are easy
        # to miss unless the user has automations listening for the event.
        try:
            notification_id = f"ha_washdata_feedback_{self.entry_id}_{cycle_id}"
            title = f"HA WashData: Verify cycle ({self.config_entry.title})"
            message = (
                f"Detected program: {self._current_program}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Cycle ID: {cycle_id}\n\n"
                f"Confirm/correct using service `{DOMAIN}.submit_cycle_feedback` with:\n"
                f"- entry_id: {self.entry_id}\n"
                f"- cycle_id: {cycle_id}\n"
                f"- user_confirmed: true\n"
                f"Optionally set `corrected_profile` (profile name) or `corrected_duration` (seconds)."
            )
            _pn_create(self.hass, message, title=title, notification_id=notification_id)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Failed to create feedback notification")
