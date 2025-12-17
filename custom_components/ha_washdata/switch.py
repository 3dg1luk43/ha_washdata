"""Switch platform for HA WashData."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, CONF_AUTO_MAINTENANCE, DEFAULT_AUTO_MAINTENANCE

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up switch entities."""
    manager = hass.data[DOMAIN][config_entry.entry_id]
    
    async_add_entities([
        AutoMaintenanceSwitch(manager, config_entry),
    ])


class AutoMaintenanceSwitch(SwitchEntity):
    """Switch to enable/disable automatic nightly maintenance."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:broom"

    def __init__(self, manager, config_entry: ConfigEntry) -> None:
        """Initialize the switch."""
        self._manager = manager
        self._config_entry = config_entry
        self._attr_unique_id = f"{config_entry.entry_id}_auto_maintenance"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": config_entry.title,
            "manufacturer": "HA WashData",
        }
        self._attr_name = "Auto Maintenance"

    @property
    def is_on(self) -> bool:
        """Return true if the switch is on."""
        return self._config_entry.options.get(
            CONF_AUTO_MAINTENANCE,
            self._config_entry.data.get(CONF_AUTO_MAINTENANCE, DEFAULT_AUTO_MAINTENANCE)
        )

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on."""
        new_options = {**self._config_entry.options, CONF_AUTO_MAINTENANCE: True}
        self.hass.config_entries.async_update_entry(self._config_entry, options=new_options)
        self.async_write_ha_state()
        # Re-setup scheduler in manager
        await self._manager._setup_maintenance_scheduler()
        _LOGGER.info("Auto-maintenance enabled")

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off."""
        new_options = {**self._config_entry.options, CONF_AUTO_MAINTENANCE: False}
        self.hass.config_entries.async_update_entry(self._config_entry, options=new_options)
        self.async_write_ha_state()
        # Cancel scheduler in manager
        await self._manager._setup_maintenance_scheduler()
        _LOGGER.info("Auto-maintenance disabled")
