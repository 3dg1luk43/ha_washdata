"""Button platform for WashData."""

from __future__ import annotations

import logging
from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING
from .manager import WashDataManager

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the WashData button."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]

    async_add_entities([
        WashDataTerminateButton(manager, entry),
        WashDataPauseCycleButton(manager, entry),
        WashDataResumeCycleButton(manager, entry),
    ])


class WashDataTerminateButton(ButtonEntity):
    """Button to force terminate the current cycle."""

    _attr_has_entity_name = True
    _attr_translation_key = "force_end_cycle"
    _attr_icon = "mdi:stop-circle-outline"

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the button."""
        self._manager = manager
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_force_end"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "WashData",
        }

    def press(self) -> None:
        """Handle the button press."""
        raise NotImplementedError()

    async def async_press(self) -> None:
        """Handle the button press."""
        await self._manager.async_terminate_cycle()


class WashDataPauseCycleButton(ButtonEntity):
    """Button to pause the current cycle (user-triggered)."""

    _attr_has_entity_name = True
    _attr_translation_key = "pause_cycle"
    _attr_icon = "mdi:pause-circle-outline"

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the button."""
        self._manager = manager
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_pause_cycle"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "WashData",
        }

    @property
    def available(self) -> bool:
        """Only available when a cycle is active and not already user-paused."""
        return (
            self._manager.check_state() in (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING)
            and not self._manager.is_user_paused
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        await self._manager.async_pause_cycle()


class WashDataResumeCycleButton(ButtonEntity):
    """Button to resume a user-paused cycle."""

    _attr_has_entity_name = True
    _attr_translation_key = "resume_cycle"
    _attr_icon = "mdi:play-circle-outline"

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the button."""
        self._manager = manager
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_resume_cycle"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "WashData",
        }

    @property
    def available(self) -> bool:
        """Only available when the cycle is user-paused."""
        return self._manager.is_user_paused

    async def async_press(self) -> None:
        """Handle the button press."""
        await self._manager.async_resume_cycle()
