"""Binary sensor for HA WashData."""
from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
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
    """Set up the binary sensor."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([WasherRunningBinarySensor(manager, entry)])


class WasherRunningBinarySensor(BinarySensorEntity):
    """Binary sensor indicating if washer is running."""

    _attr_has_entity_name = True
    
    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize."""
        self._manager = manager
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_running"
        self._attr_name = "Running"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "HA WashData",
        }

    @property
    def is_on(self) -> bool:
        """Return true if the binary sensor is on."""
        return self._manager.check_state == STATE_RUNNING

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
