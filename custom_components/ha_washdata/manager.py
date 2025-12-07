"""Manager for HA WashData."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN

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
        min_power = config_entry.options.get(CONF_MIN_POWER, config_entry.data.get(CONF_MIN_POWER, 5.0))
        off_delay = config_entry.options.get(CONF_OFF_DELAY, config_entry.data.get(CONF_OFF_DELAY, 60))
        
        config = CycleDetectorConfig(
            min_power=float(min_power),
            off_delay=int(off_delay),
        )
        self.detector = CycleDetector(
            config,
            self._on_state_change,
            self._on_cycle_end
        )
        
        self._remove_listener = None
        self._current_program = "unknown"
        self._time_remaining: float | None = None
        self._cycle_progress: float = 0.0
        self._last_reading_time: datetime | None = None
        self._current_power: float = 0.0

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
                 _LOGGER.info("Restored interrupted washer cycle.")
        
        # Subscribe to power sensor updates
        self._remove_listener = async_track_state_change_event(
            self.hass, [self.power_sensor_entity_id], self._async_power_changed
        )
        
    async def async_shutdown(self) -> None:
        """Shutdown."""
        if self._remove_listener:
            self._remove_listener()
            
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

        now = datetime.now()
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

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """Handle state change from detector."""
        _LOGGER.debug(f"Washer state changed: {old_state} -> {new_state}")
        if new_state == "running":
            self._current_program = "detecting..."
            self._time_remaining = None
            self._cycle_progress = 0
            self.hass.bus.async_fire(EVENT_CYCLE_STARTED, {"entry_id": self.entry_id, "device_name": self.config_entry.title})
            
            # Send notification if enabled
            events = self.config_entry.options.get(CONF_NOTIFY_EVENTS, [])
            if NOTIFY_EVENT_START in events:
                self._send_notification(f"{self.config_entry.title} started.")
            
        self._notify_update()

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
        self.hass.bus.async_fire(EVENT_CYCLE_ENDED, {"entry_id": self.entry_id, "device_name": self.config_entry.title, "cycle_data": cycle_data})
        
        # Send notification if enabled
        events = self.config_entry.options.get(CONF_NOTIFY_EVENTS, [])
        if NOTIFY_EVENT_FINISH in events:
             self._send_notification(f"{self.config_entry.title} finished. Duration: {int(duration/60)}m.")
        
        self._current_program = "unknown"
        self._time_remaining = None
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
        now = datetime.now()
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
        # This calls match_profile on the store
        # For now, minimal implementation
        pass

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
