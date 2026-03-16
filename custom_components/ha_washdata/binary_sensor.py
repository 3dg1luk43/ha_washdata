"""Binary sensor for WashData."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.const import EntityCategory
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    STATE_RUNNING,
    SIGNAL_WASHER_UPDATE,
    CONF_EXPOSE_DEBUG_ENTITIES,
)
from .manager import WashDataManager
from .sensor import cleanup_orphaned_diagnostic_entities


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the binary sensor."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]
    entities = [WasherRunningBinarySensor(manager, entry)]

    if entry.options.get(CONF_EXPOSE_DEBUG_ENTITIES):
        entities.append(WasherAmbiguitySensor(manager, entry))

    async_add_entities(entities)
    cleanup_orphaned_diagnostic_entities(hass, manager, entry)


class WasherRunningBinarySensor(BinarySensorEntity):
    """Binary sensor indicating if washer is running."""

    _attr_has_entity_name = True

    _attr_translation_key = "running"

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize."""
        self._manager = manager
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_running"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "WashData",
        }

    @property
    def is_on(self) -> bool | None:
        """Return true if the binary sensor is on."""
        return self._manager.check_state() == STATE_RUNNING

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


class WasherAmbiguitySensor(WasherRunningBinarySensor):
    """Binary sensor indicating if current profiling is ambiguous."""

    _attr_translation_key = "match_ambiguity"

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(manager, entry)
        self._attr_unique_id = f"{entry.entry_id}_ambiguity"
        self._attr_icon = "mdi:alert-circle-outline"
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def is_on(self) -> bool:  # type: ignore[override]
        """Return true if match is ambiguous."""
        return self._manager.match_ambiguity

    @property
    def extra_state_attributes(self) -> dict[str, Any]:  # type: ignore[override]
        """Return ambiguous candidate info."""
        details = self._manager.last_match_details
        return {"margin": details.get("ambiguity_margin", 0.0) if details else 0.0}
