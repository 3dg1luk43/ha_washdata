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
        
        config = CycleDetectorConfig(
            min_power=config_entry.data.get(CONF_MIN_POWER, config_entry.options.get(CONF_MIN_POWER)),
            off_delay=config_entry.data.get(CONF_OFF_DELAY, config_entry.options.get(CONF_OFF_DELAY)),
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

    async def async_setup(self) -> None:
        """Set up the manager."""
        await self.profile_store.async_load()
        
        # Subscribe to power sensor updates
        self._remove_listener = async_track_state_change_event(
            self.hass, [self.power_sensor_entity_id], self._async_power_changed
        )
        
    async def async_shutdown(self) -> None:
        """Shutdown."""
        if self._remove_listener:
            self._remove_listener()

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
        self.detector.process_reading(power, now)
        
        # If running, try to match profile and update estimates
        if self.detector.state == "running":
             self._update_estimates()
             
        self._notify_update()

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """Handle state change from detector."""
        _LOGGER.debug(f"Washer state changed: {old_state} -> {new_state}")
        if new_state == "running":
            self._current_program = "detecting..."
            self._time_remaining = None
            self._cycle_progress = 0
            self.hass.bus.async_fire(EVENT_CYCLE_STARTED, {"entry_id": self.entry_id, "device_name": self.config_entry.title})
            
        self._notify_update()

    def _on_cycle_end(self, cycle_data: dict) -> None:
        """Handle cycle end."""
        self.hass.async_create_task(self.profile_store.async_save_cycle(cycle_data))
        self.hass.bus.async_fire(EVENT_CYCLE_ENDED, {"entry_id": self.entry_id, "device_name": self.config_entry.title, "cycle_data": cycle_data})
        self._current_program = "unknown"
        self._time_remaining = None
        self._notify_update()

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
