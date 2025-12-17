"""The HA WashData integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BINARY_SENSOR]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA WashData from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    from .manager import WashDataManager
    manager = WashDataManager(hass, entry)
    hass.data[DOMAIN][entry.entry_id] = manager
    
    await manager.async_setup()

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    
    # Register service if not already
    if not hass.services.has_service(DOMAIN, "label_cycle"):
        async def handle_label_cycle(call):
            device_id = call.data.get("device_id")
            cycle_id = call.data.get("cycle_id")
            profile_name = call.data.get("profile_name")
            
            # Find the config entry for this device
            dr = hass.helpers.device_registry.async_get(hass)
            device = dr.async_get(device_id)
            if not device:
                raise ValueError("Device not found")
                
            entry_id = next(iter(device.config_entries))
            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this device")
                
            manager = hass.data[DOMAIN][entry_id]
            await manager.profile_store.create_profile(profile_name, cycle_id)
            
        hass.services.async_register(DOMAIN, "label_cycle", handle_label_cycle)

    return True

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        manager = hass.data[DOMAIN].pop(entry.entry_id)
        await manager.async_shutdown()

    return unload_ok
