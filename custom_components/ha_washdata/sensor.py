"""Sensors for HA WashData."""
from __future__ import annotations

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, SIGNAL_WASHER_UPDATE
from .manager import WashDataManager
from homeassistant.util import dt as dt_util

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensors."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]
    
    entities = [
        WasherStateSensor(manager, entry),
        WasherProgramSensor(manager, entry),
        WasherTimeRemainingSensor(manager, entry),
        WasherProgressSensor(manager, entry),
        WasherPowerSensor(manager, entry),
        WasherElapsedTimeSensor(manager, entry),
        WasherDebugSensor(manager, entry),
    ]
    
    async_add_entities(entities)


class WasherBaseSensor(SensorEntity):
    """Base sensor for ha_washdata."""

    _attr_has_entity_name = True

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize."""
        self._manager = manager
        self._entry = entry
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "HA WashData",
        }
        self._attr_unique_id = f"{entry.entry_id}_{self.entity_description.key}"

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_WASHER_UPDATE.format(self._entry.entry_id),
                self._update_callback,
            )
        )

    @callback
    def _update_callback(self) -> None:
        """Update the sensor."""
        self.async_write_ha_state()



class WasherStateSensor(WasherBaseSensor):
    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="washer_state",
            name="State",
            icon="mdi:washing-machine"
        )
        super().__init__(manager, entry)
    
    @property
    def native_value(self):
        return self._manager.check_state

    @property
    def extra_state_attributes(self):
        return {
            "samples_recorded": self._manager.samples_recorded,
            "current_program_guess": self._manager.current_program,
            "sub_state": self._manager.sub_state,
        }


class WasherProgramSensor(WasherBaseSensor):
    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="washer_program",
            name="Program",
            icon="mdi:file-document-outline"
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.current_program


class WasherTimeRemainingSensor(WasherBaseSensor):
    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="time_remaining",
            name="Time Remaining",
            # native_unit_of_measurement="min",  # Removed static unit
            icon="mdi:timer-sand"
        )
        super().__init__(manager, entry)

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit of measurement."""
        if self._manager.check_state == "off":
            return None
        return "min"

    @property
    def native_value(self):
        if self._manager.check_state == "off":
            return "off"
        if self._manager.time_remaining:
            return int(self._manager.time_remaining / 60)
        return None

class WasherProgressSensor(WasherBaseSensor):
    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="cycle_progress",
            name="Progress",
            native_unit_of_measurement="%",
            icon="mdi:progress-clock"
        )
        super().__init__(manager, entry)


    @property
    def native_value(self):
        return self._manager.cycle_progress


class WasherPowerSensor(WasherBaseSensor):
    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="current_power",
            name="Current Power",
            native_unit_of_measurement="W",
            device_class="power",
            icon="mdi:flash"
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.current_power


class WasherElapsedTimeSensor(WasherBaseSensor):
    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="elapsed_time",
            name="Elapsed Time",
            native_unit_of_measurement="s",
            device_class="duration",
            icon="mdi:timer-outline"
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        if self._manager.check_state == "off":
            return 0
        start = self._manager.cycle_start_time
        if start:
            delta = dt_util.now() - start
            return int(delta.total_seconds())
        return 0


class WasherDebugSensor(WasherBaseSensor):
    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="debug_info",
            name="Debug Info",
            icon="mdi:bug",
            entity_registry_enabled_default=False, # Hidden by default
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.check_state

    @property
    def extra_state_attributes(self):
        detector = self._manager.detector
        stats = self._manager.sample_interval_stats
        return {
            "sub_state": detector.sub_state,
            "match_confidence": getattr(self._manager, "_last_match_confidence", 0.0),
            "cycle_id": getattr(detector, "_current_cycle_start", None),
            "samples": detector.samples_recorded,
            "energy_accum": getattr(detector, "_energy_since_idle_wh", 0.0),
            "time_below": getattr(detector, "_time_below_threshold", 0.0),
            "sampling_p95": stats.get("p95"),
            "noise_events": len(getattr(self._manager, "_noise_events", [])),
        }
