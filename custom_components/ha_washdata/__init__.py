"""The HA WashData integration."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import device_registry as dr

from .const import (
    DOMAIN,
    SERVICE_SUBMIT_FEEDBACK,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_PROGRESS_RESET_DELAY,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_AUTO_LABEL_CONFIDENCE,
    DEFAULT_PROGRESS_RESET_DELAY,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_SMOOTHING_WINDOW,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_AUTO_MERGE_LOOKBACK_HOURS,
    CONF_AUTO_MERGE_GAP_SECONDS,
    CONF_INTERRUPTED_MIN_SECONDS,
    CONF_ABRUPT_DROP_WATTS,
    CONF_ABRUPT_DROP_RATIO,
    CONF_ABRUPT_HIGH_LOAD_FACTOR,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_PROFILE_DURATION_TOLERANCE,
    DEFAULT_AUTO_MERGE_LOOKBACK_HOURS,
    DEFAULT_AUTO_MERGE_GAP_SECONDS,
    DEFAULT_INTERRUPTED_MIN_SECONDS,
    DEFAULT_ABRUPT_DROP_WATTS,
    DEFAULT_ABRUPT_DROP_RATIO,
    DEFAULT_ABRUPT_HIGH_LOAD_FACTOR,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_MAX_PAST_CYCLES,
    CONF_MAX_FULL_TRACES_PER_PROFILE,
    CONF_MAX_FULL_TRACES_UNLABELED,
    CONF_WATCHDOG_INTERVAL,
    CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_MAX_PAST_CYCLES,
    DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
    DEFAULT_MAX_FULL_TRACES_UNLABELED,
    DEFAULT_WATCHDOG_INTERVAL,
    DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BINARY_SENSOR, Platform.SWITCH, Platform.SELECT]


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    return value

async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry to the latest version while preserving settings."""
    version = entry.version or 1

    data: dict[str, Any] = dict(entry.data)
    options: dict[str, Any] = dict(entry.options)

    # Preserve core settings from data into options if missing
    if CONF_MIN_POWER not in options and CONF_MIN_POWER in data:
        options[CONF_MIN_POWER] = data[CONF_MIN_POWER]
    if CONF_OFF_DELAY not in options and CONF_OFF_DELAY in data:
        options[CONF_OFF_DELAY] = data[CONF_OFF_DELAY]

    options.setdefault(CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY)
    options.setdefault(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE)
    options.setdefault(CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE)
    options.setdefault(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE)
    # New: active no-update timeout for publish-on-change sockets
    options.setdefault(CONF_NO_UPDATE_ACTIVE_TIMEOUT, DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT)
    # Advanced defaults for configurability
    options.setdefault(CONF_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW)
    options.setdefault(CONF_PROFILE_DURATION_TOLERANCE, DEFAULT_PROFILE_DURATION_TOLERANCE)
    options.setdefault(CONF_AUTO_MERGE_LOOKBACK_HOURS, DEFAULT_AUTO_MERGE_LOOKBACK_HOURS)
    options.setdefault(CONF_AUTO_MERGE_GAP_SECONDS, DEFAULT_AUTO_MERGE_GAP_SECONDS)
    options.setdefault(CONF_INTERRUPTED_MIN_SECONDS, DEFAULT_INTERRUPTED_MIN_SECONDS)
    options.setdefault(CONF_ABRUPT_DROP_WATTS, DEFAULT_ABRUPT_DROP_WATTS)
    options.setdefault(CONF_ABRUPT_DROP_RATIO, DEFAULT_ABRUPT_DROP_RATIO)
    options.setdefault(CONF_ABRUPT_HIGH_LOAD_FACTOR, DEFAULT_ABRUPT_HIGH_LOAD_FACTOR)

    # Matching + retention + watchdog defaults
    options.setdefault(CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL)
    options.setdefault(CONF_PROFILE_MATCH_MIN_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO)
    options.setdefault(CONF_PROFILE_MATCH_MAX_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO)
    options.setdefault(CONF_MAX_PAST_CYCLES, DEFAULT_MAX_PAST_CYCLES)
    options.setdefault(CONF_MAX_FULL_TRACES_PER_PROFILE, DEFAULT_MAX_FULL_TRACES_PER_PROFILE)
    options.setdefault(CONF_MAX_FULL_TRACES_UNLABELED, DEFAULT_MAX_FULL_TRACES_UNLABELED)
    options.setdefault(CONF_WATCHDOG_INTERVAL, DEFAULT_WATCHDOG_INTERVAL)
    options.setdefault(CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD, DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD)
    options.setdefault(CONF_COMPLETION_MIN_SECONDS, DEFAULT_COMPLETION_MIN_SECONDS)
    options.setdefault(CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES)

    # Update entry with new version
    hass.config_entries.async_update_entry(
        entry,
        data=data,
        options=options,
        version=3,
    )
    _LOGGER.info("Migrated HA WashData entry from version %s to 3", version)
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA WashData from a config entry."""
    # Guard against duplicate setup during hot-reload
    if entry.entry_id in hass.data.get(DOMAIN, {}):
        _LOGGER.warning("Entry %s already set up, skipping duplicate setup", entry.entry_id)
        return True
    
    hass.data.setdefault(DOMAIN, {})
    
    from .manager import WashDataManager
    manager = WashDataManager(hass, entry)
    hass.data[DOMAIN][entry.entry_id] = manager
    
    await manager.async_setup()

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    
    # Register service if not already
    if not hass.services.has_service(DOMAIN, "label_cycle"):
        async def handle_label_cycle(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            cycle_id = _require_str(call.data.get("cycle_id"), "cycle_id")
            profile_name = call.data.get("profile_name", "").strip()
            
            # Find the config entry for this device
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")
                
            entry_id = next(iter(device.config_entries), None)
            if not entry_id:
                raise ValueError("No config entry found for device")
            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this device")
                
            manager = hass.data[DOMAIN][entry_id]
            
            # Assign existing profile or remove label
            if profile_name:
                await manager.profile_store.assign_profile_to_cycle(cycle_id, profile_name)
            else:
                await manager.profile_store.assign_profile_to_cycle(cycle_id, None)
            
            manager._notify_update()
            
        hass.services.async_register(DOMAIN, "label_cycle", handle_label_cycle)

    # Register create_profile service
    if not hass.services.has_service(DOMAIN, "create_profile"):
        async def handle_create_profile(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            profile_name = _require_str(call.data.get("profile_name"), "profile_name")
            reference_cycle_id = call.data.get("reference_cycle_id")
            
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")
                
            entry_id = next(iter(device.config_entries), None)
            if not entry_id:
                raise ValueError("No config entry found for device")
            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this device")
                
            manager = hass.data[DOMAIN][entry_id]
            await manager.profile_store.create_profile_standalone(profile_name, reference_cycle_id)
            manager._notify_update()
            
        hass.services.async_register(DOMAIN, "create_profile", handle_create_profile)

    # Register delete_profile service
    if not hass.services.has_service(DOMAIN, "delete_profile"):
        async def handle_delete_profile(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            profile_name = _require_str(call.data.get("profile_name"), "profile_name")
            unlabel_cycles = call.data.get("unlabel_cycles", True)
            
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")
                
            entry_id = next(iter(device.config_entries), None)
            if not entry_id:
                raise ValueError("No config entry found for device")
            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this device")
                
            manager = hass.data[DOMAIN][entry_id]
            await manager.profile_store.delete_profile(profile_name, unlabel_cycles)
            manager._notify_update()
            
        hass.services.async_register(DOMAIN, "delete_profile", handle_delete_profile)

    # Register auto_label_cycles service
    if not hass.services.has_service(DOMAIN, "auto_label_cycles"):
        async def handle_auto_label_cycles(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            confidence_threshold = call.data.get("confidence_threshold", 0.70)
            
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")
                
            entry_id = next(iter(device.config_entries), None)
            if not entry_id:
                raise ValueError("No config entry found for device")
            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this device")
                
            manager = hass.data[DOMAIN][entry_id]
            stats = await manager.profile_store.auto_label_unlabeled_cycles(confidence_threshold)
            manager._notify_update()
            
            _LOGGER.info(f"Auto-label complete: {stats['labeled']} labeled, {stats['skipped']} skipped")
            
        hass.services.async_register(DOMAIN, "auto_label_cycles", handle_auto_label_cycles)

    # Register feedback service
    if not hass.services.has_service(DOMAIN, SERVICE_SUBMIT_FEEDBACK.split(".")[-1]):
        async def handle_submit_feedback(call: ServiceCall) -> None:
            entry_id_raw = call.data.get("entry_id")
            device_id_raw = call.data.get("device_id")

            entry_id: str | None = entry_id_raw if isinstance(entry_id_raw, str) and entry_id_raw else None
            if entry_id is None:
                # Prefer device_id for user-facing workflows.
                device_id = _require_str(device_id_raw, "device_id")
                registry = dr.async_get(hass)
                device = registry.async_get(device_id)
                if not device:
                    raise ValueError("Device not found")
                entry_id = next(iter(device.config_entries), None)
                if not entry_id:
                    raise ValueError("No config entry found for device")

            if not entry_id:
                raise ValueError("entry_id or device_id is required")

            cycle_id = _require_str(call.data.get("cycle_id"), "cycle_id")
            user_confirmed = call.data.get("user_confirmed", False)
            corrected_profile = call.data.get("corrected_profile")
            corrected_duration = call.data.get("corrected_duration")  # in seconds
            notes = call.data.get("notes", "")
            
            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this entry")
                
            manager = hass.data[DOMAIN][entry_id]
            success = manager.learning_manager.submit_cycle_feedback(
                cycle_id=cycle_id,
                user_confirmed=user_confirmed,
                corrected_profile=corrected_profile,
                corrected_duration=corrected_duration,
                notes=notes,
            )
            
            if success:
                # Save updated profile data
                await manager.profile_store.async_save()

                # If feedback changed labeling, rebuild envelope so future matching benefits.
                try:
                    if corrected_profile:
                        manager.profile_store.rebuild_envelope(corrected_profile)
                    else:
                        # Rebuild for the detected profile if present on the cycle.
                        cycles = manager.profile_store._data.get("past_cycles", [])
                        cycle = next(
                            (
                                cd
                                for c in cycles
                                if isinstance(c, dict)
                                for cd in (cast(dict[str, Any], c),)
                                if cd.get("id") == cycle_id
                            ),
                            None,
                        )
                        profile_name = cycle.get("profile_name") if cycle else None
                        if isinstance(profile_name, str) and profile_name:
                            manager.profile_store.rebuild_envelope(profile_name)
                except Exception:
                    _LOGGER.exception("Failed to rebuild envelope after feedback")

                # Best-effort dismiss the feedback notification if it exists.
                try:
                    notification_id = f"ha_washdata_feedback_{entry_id}_{cycle_id}"
                    persistent_notification.async_dismiss(hass, notification_id)
                except Exception:
                    pass

                _LOGGER.info(f"Cycle feedback submitted for {cycle_id}")
            else:
                _LOGGER.warning(f"Failed to submit feedback for cycle {cycle_id}")
            
        hass.services.async_register(DOMAIN, SERVICE_SUBMIT_FEEDBACK.split(".")[-1], handle_submit_feedback)

    # Export store to file (per entry/device)
    if not hass.services.has_service(DOMAIN, "export_config"):
        async def handle_export_config(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            file_path = call.data.get("path")

            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")

            entry_id = next(iter(device.config_entries), None)
            if not entry_id:
                raise ValueError("No config entry found for device")
            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this device")

            manager = hass.data[DOMAIN][entry_id]
            entry = hass.config_entries.async_get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Config entry not found: {entry_id}")
            payload = manager.profile_store.export_data(
                entry_data=dict(entry.data),
                entry_options=dict(entry.options),
            )

            target = Path(file_path) if file_path else Path(hass.config.path(f"ha_washdata_export_{entry_id}.json"))
            target = target.resolve()

            # Write export
            target.write_text(json.dumps(payload, indent=2))
            _LOGGER.info("Exported ha_washdata entry %s to %s", entry_id, target)

        hass.services.async_register(DOMAIN, "export_config", handle_export_config)

    # Import store from file into the target entry/device
    if not hass.services.has_service(DOMAIN, "import_config"):
        async def handle_import_config(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            file_path = call.data.get("path")

            if not file_path:
                raise ValueError("path is required for import")

            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")

            entry_id = next(iter(device.config_entries), None)
            if not entry_id:
                raise ValueError("No config entry found for device")
            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this device")

            manager = hass.data[DOMAIN][entry_id]
            entry = hass.config_entries.async_get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Config entry not found: {entry_id}")

            source = Path(file_path).resolve()
            if not source.exists():
                raise ValueError(f"File not found: {source}")

            try:
                payload = json.loads(source.read_text())
            except Exception as err:  # noqa: BLE001
                raise ValueError(f"Failed to read import file: {err}") from err

            config_updates = await manager.profile_store.async_import_data(payload)
            
            # Apply imported settings to config entry if present
            entry_data = config_updates.get("entry_data", {})
            entry_options = config_updates.get("entry_options", {})
            
            if entry_data or entry_options:
                new_data: dict[str, Any] = dict(entry.data)
                new_options: dict[str, Any] = dict(entry.options)
                
                # Only update min_power/off_delay from data (don't overwrite power_sensor/name)
                for key in [CONF_MIN_POWER, CONF_OFF_DELAY]:
                    if key in entry_data:
                        new_data[key] = entry_data[key]
                
                # Update all options from import
                new_options.update(entry_options)
                
                hass.config_entries.async_update_entry(
                    entry,
                    data=new_data,
                    options=new_options,
                )
                _LOGGER.info("Applied imported settings to config entry %s", entry_id)
                
            _LOGGER.info("Imported ha_washdata entry %s from %s", entry_id, source)

        hass.services.async_register(DOMAIN, "import_config", handle_import_config)

    return True

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry - update settings without interrupting running cycles."""
    manager = hass.data[DOMAIN].get(entry.entry_id)
    if manager:
        # Update configuration without interrupting detector
        await manager.async_reload_config(entry)
    else:
        # Full reload if manager not found
        await async_unload_entry(hass, entry)
        await async_setup_entry(hass, entry)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        manager = hass.data[DOMAIN].pop(entry.entry_id)
        await manager.async_shutdown()

    return unload_ok
