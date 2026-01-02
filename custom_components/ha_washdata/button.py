"""Button platform for HA WashData."""
from __future__ import annotations

import logging
from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .manager import WashDataManager

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the WashData button."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]
    
    async_add_entities([WashDataTerminateButton(manager, entry)])

class WashDataTerminateButton(ButtonEntity):
    """Button to force terminate the current cycle."""

    _attr_has_entity_name = True
    _attr_name = "Force End Cycle"
    _attr_icon = "mdi:stop-circle-outline"

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the button."""
        self._manager = manager
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_force_end"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "HA WashData",
        }

    async def async_press(self) -> None:
        """Handle the button press."""
        await self._manager.async_terminate_cycle()
