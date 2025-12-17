"""Manager for HA WashData."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    CONF_POWER_SENSOR,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_EVENTS,
    NOTIFY_EVENT_START,
    NOTIFY_EVENT_FINISH,
    EVENT_CYCLE_STARTED,
    EVENT_CYCLE_ENDED,
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
    STATE_RUNNING,
    STATE_OFF,
)
from .cycle_detector import CycleDetector, CycleDetectorConfig
from .profile_store import ProfileStore

_LOGGER = logging.getLogger(__name__)

SIGNAL_WASHER_UPDATE = "ha_washdata_update_{}"

class WashDataManager:
    """Manages a single washing machine instance."""

    def __init__(self, hass: HomeAssistant, config_entry) -> None:
        """Initialize the manager."""
        self.hass = hass
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        
        self.power_sensor_entity_id = config_entry.data[CONF_POWER_SENSOR]
        
        # Components
        self.profile_store = ProfileStore(hass, self.entry_id)
        
        # Priority: Options > Data > Default
        min_power = config_entry.options.get(CONF_MIN_POWER, config_entry.data.get(CONF_MIN_POWER, DEFAULT_MIN_POWER))
        off_delay = config_entry.options.get(CONF_OFF_DELAY, config_entry.data.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY))
        
        _LOGGER.info(f"Manager init: min_power={min_power}W, off_delay={off_delay}s (from options={CONF_MIN_POWER in config_entry.options}, defaults={DEFAULT_MIN_POWER}W, {DEFAULT_OFF_DELAY}s)")
        
        config = CycleDetectorConfig(
            min_power=float(min_power),
            off_delay=int(off_delay),
        )
        self._config = config
        self.detector = CycleDetector(
            config,
            self._on_state_change,
            self._on_cycle_end
        )
        
        self._remove_listener = None
        self._remove_watchdog = None
        self._watchdog_interval = 5  # Check every 5 seconds when running
        self._current_program = "off"
        self._time_remaining: float | None = None
        self._cycle_progress: float = 0.0
        self._last_reading_time: datetime | None = None
        self._current_power: float = 0.0
        self._last_estimate_time: datetime | None = None
        self._matched_profile_duration: float | None = None

    async def async_setup(self) -> None:
        """Set up the manager."""
        await self.profile_store.async_load()
        
        # RESTORE STATE
        active_snapshot = self.profile_store.get_active_cycle()
        if active_snapshot:
            # Check if it's stale
            # For now simply restore. The detector can decide if gaps are too big?
            # Or we check last_active_save here?
            # Let's try to restore.
            self.detector.restore_state_snapshot(active_snapshot)
            if self.detector.state == "running":
                 self._current_program = "restored..."
                 self._start_watchdog()  # Resume watchdog for restored cycle
                 _LOGGER.info("Restored interrupted washer cycle.")
        
        # Subscribe to power sensor updates
        self._remove_listener = async_track_state_change_event(
            self.hass, [self.power_sensor_entity_id], self._async_power_changed
        )
        
        # Watchdog starts disabled, will be enabled when cycle starts
        
        # Force initial update from current state (in case it's already stable)
        state = self.hass.states.get(self.power_sensor_entity_id)
        if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            # Create a mock event or just call logic? _async_power_changed expects event.
            # Easier to just parse power and call process directly or fake event.
            try:
                power = float(state.state)
                # Ensure we don't duplicate logic, calling internal methods or fabricating event
                # Let's call internal logic directly to avoid event object construction overhead/complexity
                self._current_power = power
                self.detector.process_reading(power, dt_util.now())
                
                # If running, update estimates (and potentially finish cycle if expired)
                if self.detector.state == "running":
                    self._update_estimates()
                    
            except ValueError:
                pass

    async def async_shutdown(self) -> None:
        """Shutdown."""
        if self._remove_listener:
            self._remove_listener()
        if self._remove_watchdog:
            self._remove_watchdog()
            
        # Try to save state one last time?
        if self.detector.state == "running":
             await self.profile_store.async_save_active_cycle(self.detector.get_state_snapshot())

        self._last_reading_time = None

    @callback
    def _async_power_changed(self, event) -> None:
        """Handle power sensor state change."""
        new_state = event.data.get("new_state")
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
             
        self._notify_update()

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
        
        _LOGGER.debug(f"Starting watchdog timer (interval: {self._watchdog_interval}s)")
        self._remove_watchdog = async_track_time_interval(
            self.hass, self._watchdog_check_stuck_cycle, timedelta(seconds=self._watchdog_interval)
        )

    def _stop_watchdog(self) -> None:
        """Stop the watchdog timer when cycle ends."""
        if self._remove_watchdog:
            _LOGGER.debug("Stopping watchdog timer")
            self._remove_watchdog()
            self._remove_watchdog = None

    async def _watchdog_check_stuck_cycle(self, now: datetime) -> None:
        """Watchdog: check if cycle is stuck (no updates for too long)."""
        if self.detector.state != STATE_RUNNING:
            return
        
        # If cycle is running but no power update for > off_delay, force end
        # This handles sensor disconnects or stops sending values
        if self._last_reading_time:
            time_since_update = (now - self._last_reading_time).total_seconds()
            timeout_seconds = self._config.off_delay  # Same as cycle end threshold
            
            if time_since_update > timeout_seconds:
                _LOGGER.warning(f"Watchdog: no power update for {time_since_update:.0f}s (threshold: {timeout_seconds}s), forcing end")
                # Force-finish the cycle rather than relying on smoothing/thresholds
                self.detector.force_end(now)
                self._last_reading_time = now
                self._current_power = 0.0  # Reset display to 0
                self._notify_update()  # Update UI immediately
                
                # If it's still running after that, log warning
                if self.detector.state == STATE_RUNNING:
                    _LOGGER.error(f"Watchdog: cycle still running after forced end, will retry next check")

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """Handle state change from detector."""
        _LOGGER.debug(f"Washer state changed: {old_state} -> {new_state}")
        if new_state == "running":
            self._current_program = "detecting..."
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

    def _on_cycle_end(self, cycle_data: dict) -> None:
        """Handle cycle end."""
        duration = cycle_data["duration"]
        max_power = cycle_data.get("max_power", 0)
        
        # Auto-Tune: Check for ghost cycles (short duration, lowish power)
        # Definition of noise: duration < 120s
        if duration < 120:
             self._handle_noise_cycle(max_power)
             # We still save it as a cycle for history, or maybe we shouldn't?
             # Let's save it but marked as potential noise? 
             # For now save as normal.
        
        self.hass.async_create_task(self.profile_store.async_save_cycle(cycle_data))
        self.hass.async_create_task(self.profile_store.async_clear_active_cycle())
        
        # Auto post-process: merge fragmented cycles from last 3 hours
        self.hass.async_create_task(self._auto_merge_recent_cycles())
        
        self.hass.bus.async_fire(EVENT_CYCLE_ENDED, {"entry_id": self.entry_id, "device_name": self.config_entry.title, "cycle_data": cycle_data})
        
        # Send notification if enabled
        events = self.config_entry.options.get(CONF_NOTIFY_EVENTS, [])
        if NOTIFY_EVENT_FINISH in events:
             self._send_notification(f"{self.config_entry.title} finished. Duration: {int(duration/60)}m.")
        
        self._current_program = "off"
        self._time_remaining = None
        self._matched_profile_duration = None
        self._last_estimate_time = None
        self._notify_update()

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
            self.hass.components.persistent_notification.async_create(
                message, title=f"HA WashData: {self.config_entry.title}"
            )

    def _handle_noise_cycle(self, max_power: float) -> None:
        """Handle a detected noise cycle."""
        # Clean up old noise events > 24h
        now = dt_util.now()
        self._noise_events = [t for t in getattr(self, "_noise_events", []) if (now - t).total_seconds() < 86400]
        self._noise_events.append(now)
        
        # Track max power of noise
        self._noise_max_powers = getattr(self, "_noise_max_powers", [])
        self._noise_max_powers.append(max_power)
        
        # If > 3 events in 24h, trigger tune
        if len(self._noise_events) >= 3:
             self._tune_threshold()

    def _tune_threshold(self) -> None:
        """Increase the minimum power threshold."""
        current_min = self.detector._config.min_power
        
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

        _LOGGER.info(f"Auto-Tuning: Increasing min_power from {current_min}W to {new_min}W due to noise.")
        
        # Update config entry options
        new_options = dict(self.config_entry.options)
        new_options[CONF_MIN_POWER] = new_min
        
        self.hass.config_entries.async_update_entry(self.config_entry, options=new_options)
        
        # Notify user
        notify_service = self.config_entry.options.get(CONF_NOTIFY_SERVICE)
        message = (
            f"Washing Machine '{self.config_entry.title}' detected ghost cycles. "
            f"Power threshold auto-adjusted from {current_min:.1f}W to {new_min:.1f}W."
        )
        
        if notify_service:
            # call service notify.<name>
            domain, service = notify_service.split('.', 1) if '.' in notify_service else ("notify", notify_service)
            self.hass.async_create_task(
                self.hass.services.async_call(domain, service, {"message": message})
            )
        else:
            self.hass.components.persistent_notification.async_create(
                message,
                title="HA WashData Auto-Tune"
            )
        
        # Reset trackers
        self._noise_events = []
        self._noise_max_powers = []

    def _update_estimates(self) -> None:
        """Update time remaining and profile estimates."""
        if self.detector.state != "running":
            return

        now = dt_util.now()

        # Throttle heavy matching to every ~5 minutes
        if self._last_estimate_time and (now - self._last_estimate_time).total_seconds() < 300:
            # Still update remaining/progress if we already have a match
            self._update_remaining_only()
            return

        trace = self.detector.get_power_trace()
        if len(trace) < 3:
            return

        duration_so_far = self.detector.get_elapsed_seconds()
        current_power_data = [(t.isoformat(), p) for t, p in trace]

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
                profile = self.profile_store._data["profiles"].get(profile_name, {})
                avg_duration = float(profile.get("avg_duration", 0.0))
                self._matched_profile_duration = avg_duration if avg_duration > 0 else None
                _LOGGER.info(f"Matched profile '{profile_name}' with expected duration {avg_duration:.0f}s ({int(avg_duration/60)}min)")
            # If we already have a match, keep it (don't thrash between profiles)
        elif not self._matched_profile_duration:
            # No match yet and still searching
            self._current_program = "detecting..."
        # else: keep existing match even if current attempt failed (prevents "unknown" flip-flop)

        self._last_estimate_time = now
        self._update_remaining_only()
        self._notify_update()

    def _update_remaining_only(self) -> None:
        """Recompute remaining/progress using last matched profile."""
        if self.detector.state != "running":
            self._time_remaining = None
            self._cycle_progress = 0.0
            return

        duration_so_far = self.detector.get_elapsed_seconds()

        if self._matched_profile_duration and self._matched_profile_duration > 0:
            remaining = max(self._matched_profile_duration - duration_so_far, 0.0)
            self._time_remaining = remaining
            progress = (duration_so_far / self._matched_profile_duration) * 100.0
            self._cycle_progress = max(0.0, min(progress, 100.0))
            _LOGGER.debug(f"Updated estimates: remaining={int(remaining/60)}min, progress={progress:.1f}%")
        else:
            self._time_remaining = None
            self._cycle_progress = 0.0
            _LOGGER.debug(f"No profile matched yet, elapsed={int(duration_so_far/60)}min")
            self._cycle_progress = 0.0

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
        return len(self.detector._power_readings)

    async def _auto_merge_recent_cycles(self) -> None:
        """Automatically merge fragmented cycles from the last 3 hours."""
        try:
            count = self.profile_store.merge_cycles(hours=3, gap_threshold=1800)
            if count > 0:
                _LOGGER.info(f"Auto-merged {count} fragmented cycle(s)")
                await self.profile_store.async_save()
        except Exception as e:
            _LOGGER.error(f"Auto-merge failed: {e}")
