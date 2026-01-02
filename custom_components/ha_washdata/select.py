"""Select entity for HA WashData."""
from __future__ import annotations

import logging

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN, SIGNAL_WASHER_UPDATE
from .manager import WashDataManager

_LOGGER = logging.getLogger(__name__)

OPTION_AUTO = "auto_detect"

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the select entity."""
    manager: WashDataManager = hass.data[DOMAIN][config_entry.entry_id]
    
    async_add_entities([WashDataProgramSelect(manager, config_entry)])


class WashDataProgramSelect(SelectEntity):
    """Select entity to manually choose the running program."""

    _attr_has_entity_name = True

    _attr_translation_key = "program_select"

    def __init__(self, manager: WashDataManager, config_entry: ConfigEntry) -> None:
        """Initialize the select entity."""
        self._manager = manager
        self._config_entry = config_entry
        self._attr_unique_id = f"{config_entry.entry_id}_program_select"
        self._attr_name = "Cycle Program"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": config_entry.title,
            "manufacturer": "HA WashData",
        }
        
        # Determine icon based on device type
        dtype = getattr(manager, "device_type", "washing_machine")
        if dtype == "dryer":
            self._attr_icon = "mdi:tumble-dryer"
        elif dtype == "dishwasher":
            self._attr_icon = "mdi:dishwasher"
        elif dtype == "coffee_machine":
            self._attr_icon = "mdi:coffee"
        else:
            self._attr_icon = "mdi:washing-machine" # Default and washing_machine
            
        self._update_options()

    @callback
    def _update_options(self) -> None:
        """Update the list of available options from profiles."""
        profiles = self._manager.profile_store.list_profiles()
        # Sort profiles by name
        profile_names = sorted([p["name"] for p in profiles])
        self._attr_options = [OPTION_AUTO] + profile_names

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_WASHER_UPDATE.format(self._manager.entry_id),
                self._update_state,
            )
        )
        self._update_state()

    @callback
    def _update_state(self) -> None:
        """Update state from manager."""
        # Refresh options in case new profiles were created
        self._update_options()
        
        current = self._manager.current_program
        manual_active = getattr(self._manager, "manual_program_active", False)
        
        if manual_active and current:
            self._attr_current_option = current
        elif current in ("detecting...", "off", "restored..."):
            self._attr_current_option = OPTION_AUTO
        elif current:
             # It detected a program, but it's not "manual override" mode.
             # Should we show the detected program or "Auto"?
             # Showing "Auto" implies "I am in auto mode".
             # But user might want to see what is detected here too?
             # Standard pattern: Select shows target/mode. 
             # If we are in auto mode, show Auto. The sensor shows the detected program.
             self._attr_current_option = OPTION_AUTO
        else:
             self._attr_current_option = OPTION_AUTO
        
        self.async_write_ha_state()

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        if option == OPTION_AUTO:
            self._manager.clear_manual_program()
        else:
            self._manager.set_manual_program(option)
