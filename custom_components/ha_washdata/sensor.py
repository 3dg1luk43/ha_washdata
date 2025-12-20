"""Sensors for HA WashData."""
from __future__ import annotations

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, STATE_RUNNING, SIGNAL_WASHER_UPDATE
from .manager import WashDataManager

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
            "current_program_guess": self._manager.current_program
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
