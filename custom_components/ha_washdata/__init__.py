# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""The WashData integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr

from .const import (
    DOMAIN,
    SERVICE_SUBMIT_FEEDBACK,
    CONF_LINKED_DEVICE,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_DEVICE_TYPE,
    CONF_POWER_SENSOR,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_EVENTS,
    NOTIFY_EVENT_LIVE,
    CONF_NOTIFY_START_SERVICES,
    CONF_NOTIFY_FINISH_SERVICES,
    CONF_NOTIFY_LIVE_SERVICES,
    CONF_NOTIFY_ACTIONS,
    CONF_NOTIFY_PEOPLE,
    CONF_NOTIFY_ONLY_WHEN_HOME,
    CONF_NOTIFY_FIRE_EVENTS,
    CONF_NOTIFY_LIVE_INTERVAL_SECONDS,
    CONF_NOTIFY_LIVE_OVERRUN_PERCENT,
    CONF_NOTIFY_TIMEOUT_SECONDS,
    CONF_NOTIFY_CHANNEL,
    CONF_NOTIFY_FINISH_CHANNEL,
    CONF_NOTIFY_REMINDER_MESSAGE,
    DEFAULT_NOTIFY_ONLY_WHEN_HOME,
    DEFAULT_NOTIFY_FIRE_EVENTS,
    DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS,
    DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_CHANNEL,
    DEFAULT_NOTIFY_FINISH_CHANNEL,
    DEFAULT_NOTIFY_REMINDER_MESSAGE,
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
    CONF_INTERRUPTED_MIN_SECONDS,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_PROFILE_DURATION_TOLERANCE,
    DEFAULT_INTERRUPTED_MIN_SECONDS,
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
    DEFAULT_DEVICE_TYPE,
    DEVICE_TYPE_OTHER,
    DEFAULT_START_DURATION_THRESHOLD,
    CONF_START_DURATION_THRESHOLD,
)
from .log_utils import DeviceLoggerAdapter

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.SELECT,
    Platform.BUTTON,
]


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key=f"{name}_required",
        )
    return value


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry to the latest version while preserving settings."""
    _log = DeviceLoggerAdapter(_LOGGER, entry.title)
    version = entry.version or 1
    minor_version = entry.minor_version or 1

    if version > 3:
        _log.error(
            "Refusing to migrate unsupported future schema %s.%s", version, minor_version
        )
        return False

    if version == 3 and minor_version >= 7:
        return True

    # 3.6 → 3.7: remove initial_profile stub key from entry.data.
    if version == 3 and minor_version == 6:
        new_data = {k: v for k, v in entry.data.items() if k != "initial_profile"}
        hass.config_entries.async_update_entry(
            entry, data=new_data, minor_version=7
        )
        minor_version = 7
        _log.debug("Migrated WashData entry from 3.6 to 3.7")

    if version == 3 and minor_version >= 7:
        return True

    data: dict[str, Any] = dict(entry.data)
    options: dict[str, Any] = dict(entry.options)

    # Preserve core settings from data into options if missing
    if CONF_MIN_POWER not in options and CONF_MIN_POWER in data:
        options[CONF_MIN_POWER] = data[CONF_MIN_POWER]
    if CONF_OFF_DELAY not in options and CONF_OFF_DELAY in data:
        options[CONF_OFF_DELAY] = data[CONF_OFF_DELAY]
    if CONF_DEVICE_TYPE not in options and CONF_DEVICE_TYPE in data:
        options[CONF_DEVICE_TYPE] = data[CONF_DEVICE_TYPE]
    if CONF_POWER_SENSOR not in options and CONF_POWER_SENSOR in data:
        options[CONF_POWER_SENSOR] = data[CONF_POWER_SENSOR]
    if CONF_NOTIFY_SERVICE not in options and CONF_NOTIFY_SERVICE in data:
        options[CONF_NOTIFY_SERVICE] = data[CONF_NOTIFY_SERVICE]

    # Migrate legacy single CONF_NOTIFY_SERVICE into per-event service lists.
    # Users who configured a notify service before 0.3.x would otherwise lose
    # their notification settings entirely on upgrade.
    legacy_svc = options.get(CONF_NOTIFY_SERVICE) or data.get(CONF_NOTIFY_SERVICE)
    if legacy_svc and isinstance(legacy_svc, str):
        # CONF_NOTIFY_EVENTS is a deprecated list of enabled event types.
        # Only migrate live services when live events were explicitly opted in.
        legacy_events = options.get(CONF_NOTIFY_EVENTS) or data.get(CONF_NOTIFY_EVENTS) or []
        if CONF_NOTIFY_START_SERVICES not in options:
            options[CONF_NOTIFY_START_SERVICES] = [legacy_svc]
        if CONF_NOTIFY_FINISH_SERVICES not in options:
            options[CONF_NOTIFY_FINISH_SERVICES] = [legacy_svc]
        if CONF_NOTIFY_LIVE_SERVICES not in options and NOTIFY_EVENT_LIVE in legacy_events:
            options[CONF_NOTIFY_LIVE_SERVICES] = [legacy_svc]

    options.setdefault(CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY)
    options.setdefault(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE)
    options.setdefault(CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE)
    options.setdefault(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE)
    options.setdefault(CONF_NO_UPDATE_ACTIVE_TIMEOUT, DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT)
    options.setdefault(CONF_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW)
    options.setdefault(
        CONF_PROFILE_DURATION_TOLERANCE, DEFAULT_PROFILE_DURATION_TOLERANCE
    )
    options.setdefault(CONF_INTERRUPTED_MIN_SECONDS, DEFAULT_INTERRUPTED_MIN_SECONDS)

    options.setdefault(
        CONF_DEVICE_TYPE, data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)
    )
    options.setdefault(CONF_START_DURATION_THRESHOLD, DEFAULT_START_DURATION_THRESHOLD)

    options.setdefault(CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL)
    options.setdefault(
        CONF_PROFILE_MATCH_MIN_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO
    )
    options.setdefault(
        CONF_PROFILE_MATCH_MAX_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
    )
    options.setdefault(CONF_MAX_PAST_CYCLES, DEFAULT_MAX_PAST_CYCLES)
    options.setdefault(
        CONF_MAX_FULL_TRACES_PER_PROFILE, DEFAULT_MAX_FULL_TRACES_PER_PROFILE
    )
    options.setdefault(
        CONF_MAX_FULL_TRACES_UNLABELED, DEFAULT_MAX_FULL_TRACES_UNLABELED
    )
    options.setdefault(CONF_WATCHDOG_INTERVAL, DEFAULT_WATCHDOG_INTERVAL)
    options.setdefault(
        CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD, DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD
    )
    options.setdefault(CONF_COMPLETION_MIN_SECONDS, DEFAULT_COMPLETION_MIN_SECONDS)
    options.setdefault(
        CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES
    )

    # Normalize notification options (added in 0.3.2)
    options.setdefault(CONF_NOTIFY_ACTIONS, [])
    options.setdefault(CONF_NOTIFY_PEOPLE, [])
    options.setdefault(CONF_NOTIFY_ONLY_WHEN_HOME, DEFAULT_NOTIFY_ONLY_WHEN_HOME)
    options.setdefault(CONF_NOTIFY_FIRE_EVENTS, DEFAULT_NOTIFY_FIRE_EVENTS)
    options.setdefault(
        CONF_NOTIFY_LIVE_INTERVAL_SECONDS, DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS
    )
    options.setdefault(
        CONF_NOTIFY_LIVE_OVERRUN_PERCENT, DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT
    )

    # 3.5: notification delivery overhaul (lifecycle tag, timeout, per-type channels,
    # distinct reminder message).
    options.setdefault(CONF_NOTIFY_TIMEOUT_SECONDS, DEFAULT_NOTIFY_TIMEOUT_SECONDS)
    options.setdefault(CONF_NOTIFY_CHANNEL, DEFAULT_NOTIFY_CHANNEL)
    options.setdefault(CONF_NOTIFY_FINISH_CHANNEL, DEFAULT_NOTIFY_FINISH_CHANNEL)
    options.setdefault(CONF_NOTIFY_REMINDER_MESSAGE, DEFAULT_NOTIFY_REMINDER_MESSAGE)

    keys_to_remove = [
        CONF_MIN_POWER,
        CONF_OFF_DELAY,
        CONF_DEVICE_TYPE,
        CONF_POWER_SENSOR,
        CONF_NOTIFY_SERVICE,
    ]
    for k in keys_to_remove:
        data.pop(k, None)

    # 3.4: drain-spike delayed-start model replaced by band-based DELAY_WAIT.
    # Strip the obsolete drain knobs so they don't linger in options and
    # confuse anyone inspecting entry.options.
    for k in (
        "delay_drain_min_power",
        "delay_drain_max_power",
        "delay_drain_max_duration",
    ):
        options.pop(k, None)

    # 3.6: the feedback/verify-cycle and ghost-cycle persistent notifications
    # were removed (suggestions and pending reviews are surfaced in the panel),
    # so the now-inert "suppress feedback notifications" toggle is stripped.
    options.pop("suppress_feedback_notifications", None)

    # 3.6: coffee_machine / ev / heat_pump / oven device types were removed.
    # Remap any entry still on one of them to DEVICE_TYPE_OTHER (Threshold Device),
    # preserving all tuned options so no user data is lost.
    _removed_device_types = {"coffee_machine", "ev", "heat_pump", "oven"}
    if (options.get(CONF_DEVICE_TYPE) or data.get(CONF_DEVICE_TYPE)) in _removed_device_types:
        _log.info(
            "Device type %r is no longer supported; migrating to %r (options preserved)",
            options.get(CONF_DEVICE_TYPE), DEVICE_TYPE_OTHER,
        )
        options[CONF_DEVICE_TYPE] = DEVICE_TYPE_OTHER
        # NB: CONF_DEVICE_TYPE was already popped from ``data`` above (keys_to_remove),
        # so no stale removed value can linger there; the flow/manager read it from
        # options (options-first).

    hass.config_entries.async_update_entry(
        entry,
        data=data,
        options=options,
        version=3,
        minor_version=7,
    )
    _log.info(
        "Migrated WashData entry from version %s.%s to 3.7", version, minor_version
    )
    return True


async def _migrate_online_to_global(hass: HomeAssistant, entry: ConfigEntry, manager: Any) -> None:
    """Hoist the (formerly per-device) online-features flag + store account to the
    integration-wide store. Pre-release cleanup.

    The enable flag is hoisted exactly ONCE (guarded by a marker in the global store):
    the stale per-entry option is never cleared, so without the marker a user who later
    turns online off would have it silently re-enabled on the next restart. The account
    hoist stays idempotent (it clears the per-entry copy after moving it)."""
    from . import store_account  # pylint: disable=import-outside-toplevel
    from .const import CONF_ENABLE_ONLINE_FEATURES  # pylint: disable=import-outside-toplevel

    # Best-effort, pre-release migration: a transient store write failure here must
    # never propagate and abort async_setup_entry (it retries on the next restart).
    try:
        await store_account.async_load(hass)
        if not store_account.migration_done(hass):
            any_on = any(
                e.options.get(CONF_ENABLE_ONLINE_FEATURES)
                for e in hass.config_entries.async_entries(DOMAIN)
            )
            if any_on and not store_account.online_enabled(hass):
                await store_account.async_set_online(hass, True)
            await store_account.async_mark_migrated(hass)
    except Exception:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Online-features migration to global store failed", exc_info=True)

    try:
        acct = manager.profile_store.get_store_account()
    except Exception:  # pylint: disable=broad-exception-caught
        acct = {}
    if acct:
        _account_preserved = False
        try:
            if acct.get("refresh_token") and not store_account.get_account(hass).get("refresh_token"):
                await store_account.async_set_account(hass, {
                    "refresh_token": acct.get("refresh_token"),
                    "uid": acct.get("uid"), "name": acct.get("name"),
                })
            _account_preserved = True
        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.warning("Store-account hoist to global store failed", exc_info=True)
        if _account_preserved:
            try:
                await manager.profile_store.clear_store_account()
            except Exception:  # pylint: disable=broad-exception-caught
                pass


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up WashData from a config entry."""
    _log = DeviceLoggerAdapter(_LOGGER, entry.title)
    # Guard against duplicate setup during hot-reload
    if entry.entry_id in hass.data.get(DOMAIN, {}):
        _log.warning(
            "Entry %s already set up, skipping duplicate setup", entry.entry_id
        )
        return True

    hass.data.setdefault(DOMAIN, {})

    # Migration: Remove old auto_maintenance switch entity (now in settings)
    # pylint: disable=import-outside-toplevel
    from homeassistant.helpers import entity_registry as er

    ent_reg = er.async_get(hass)
    old_switch_id = f"{entry.entry_id}_auto_maintenance"
    old_entity = ent_reg.async_get_entity_id("switch", DOMAIN, old_switch_id)
    if old_entity:
        _log.info(
            "Removing deprecated auto_maintenance switch entity: %s", old_entity
        )
        ent_reg.async_remove(old_entity)

    # pylint: disable=import-outside-toplevel
    from .manager import WashDataManager

    manager = WashDataManager(hass, entry)
    hass.data[DOMAIN][entry.entry_id] = manager

    await manager.async_setup()
    await _migrate_online_to_global(hass, entry, manager)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    _apply_device_link(hass, entry)

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
            try:
                if profile_name:
                    await manager.profile_store.assign_profile_to_cycle(
                        cycle_id, profile_name
                    )
                else:
                    await manager.profile_store.assign_profile_to_cycle(cycle_id, None)
            except ValueError as exc:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="assign_profile_failed",
                    translation_placeholders={"error": str(exc)},
                ) from exc

            manager.notify_update()

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
            try:
                await manager.profile_store.create_profile_standalone(
                    profile_name, reference_cycle_id
                )
            except ValueError as exc:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="create_profile_failed",
                    translation_placeholders={"error": str(exc)},
                ) from exc
            manager.notify_update()

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
            manager.notify_update()

        hass.services.async_register(DOMAIN, "delete_profile", handle_delete_profile)

    # Register auto_label_cycles service
    if not hass.services.has_service(DOMAIN, "auto_label_cycles"):

        async def handle_auto_label_cycles(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            confidence_threshold = call.data.get("confidence_threshold", 0.75)

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
            stats = await manager.profile_store.auto_label_cycles(
                confidence_threshold
            )
            manager.notify_update()

            manager._logger.info(
                "Auto-label complete: %s labeled, %s skipped",
                stats["labeled"],
                stats["skipped"],
            )

        hass.services.async_register(
            DOMAIN, "auto_label_cycles", handle_auto_label_cycles
        )

    # Register trim_cycle service
    if not hass.services.has_service(DOMAIN, "trim_cycle"):

        async def handle_trim_cycle(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            cycle_id = _require_str(call.data.get("cycle_id"), "cycle_id")
            trim_start_s = max(0.0, float(call.data.get("trim_start_s", 0)))

            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="device_not_found",
                )

            entry_id = next(iter(device.config_entries), None)
            if not entry_id:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="no_config_entry",
                )
            if entry_id not in hass.data[DOMAIN]:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="integration_not_loaded",
                )

            manager = hass.data[DOMAIN][entry_id]
            store = manager.profile_store

            # Determine trim end - default to full cycle duration if not supplied
            raw_end = call.data.get("trim_end_s")
            # Always check cycle existence first, regardless of which trim path is taken
            p_data = store.get_cycle_power_data(cycle_id)
            if not p_data:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="cycle_not_found_or_no_power",
                )
            if raw_end is not None:
                trim_end_s = max(0.0, float(raw_end))
            else:
                trim_end_s = max(point[0] for point in p_data)

            if trim_end_s <= trim_start_s:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="trim_invalid_range",
                )

            ok = await store.trim_cycle_power_data(cycle_id, trim_start_s, trim_end_s)
            if not ok:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="trim_failed_empty_window",
                )
            manager.notify_update()

        hass.services.async_register(DOMAIN, "trim_cycle", handle_trim_cycle)

    # Register custom card via frontend.py - once per HA instance only.
    if not hass.data.get("ha_washdata_card_registered") and not hass.data.get(
        "ha_washdata_card_deferred"
    ) and not hass.data.get("ha_washdata_card_registering"):
        # pylint: disable=import-outside-toplevel
        from .frontend import (
            CARD_REGISTERED,
            CARD_DEFERRED,
            WashDataCardRegistration,
        )

        card_reg = WashDataCardRegistration(hass)
        hass.data["ha_washdata_card_registering"] = True
        try:
            register_result = await card_reg.async_register()
        except Exception as err:  # pylint: disable=broad-exception-caught
            hass.data["ha_washdata_card_registering"] = False
            _log.warning("Card registration failed, will retry on next setup: %s", err)
        else:
            hass.data["ha_washdata_card_registering"] = False
            if register_result == CARD_REGISTERED:
                hass.data["ha_washdata_card_deferred"] = False
                hass.data["ha_washdata_card_registered"] = True
            elif register_result == CARD_DEFERRED:
                hass.data["ha_washdata_card_deferred"] = True
                hass.data["ha_washdata_card_registered"] = False
            else:
                hass.data["ha_washdata_card_deferred"] = False
                hass.data["ha_washdata_card_registered"] = False
                _log.warning("Card registration failed and was not deferred")

    # Register full-screen sidebar panel - once per HA instance only.
    # pylint: disable=import-outside-toplevel
    from .frontend import async_register_panel, PANEL_REGISTERED_KEY

    if not hass.data.get(PANEL_REGISTERED_KEY):
        await async_register_panel(hass)

    # Register WebSocket API commands for the panel. Re-run on every setup/reload:
    # HA's async_register_command overwrites the handler per command type, so this
    # is idempotent AND means NEW commands become available after an integration
    # reload, not only after a full Home Assistant restart (previously the
    # once-per-instance guard forced a full restart for any newly-added command).
    from .ws_api import (  # pylint: disable=import-outside-toplevel
        async_load_panel_config,
        async_register_commands,
    )

    await async_load_panel_config(hass)  # self-guards; safe to call repeatedly
    from . import store_account  # pylint: disable=import-outside-toplevel
    await store_account.async_load(hass)  # integration-wide online flag + account
    async_register_commands(hass)
    hass.data["ha_washdata_ws_registered"] = True

    # Register conversation intents (e.g. "is my washer done?") - once per HA
    # instance. Intents are domain-global, so guard against re-registration when
    # more than one device is configured.
    if not hass.data.get("ha_washdata_intents_registered"):
        from .intents import async_setup_intents  # pylint: disable=import-outside-toplevel

        async_setup_intents(hass)
        hass.data["ha_washdata_intents_registered"] = True

    # Register feedback service
    if not hass.services.has_service(
        DOMAIN, SERVICE_SUBMIT_FEEDBACK.rsplit(".", maxsplit=1)[-1]
    ):

        async def handle_submit_feedback(call: ServiceCall) -> None:
            entry_id_raw = call.data.get("entry_id")
            device_id_raw = call.data.get("device_id")

            entry_id: str | None = (
                entry_id_raw if isinstance(entry_id_raw, str) and entry_id_raw else None
            )
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
            dismiss = call.data.get("dismiss", False)

            if entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded for this entry")

            manager = hass.data[DOMAIN][entry_id]
            success = await manager.learning_manager.async_submit_cycle_feedback(
                cycle_id=cycle_id,
                user_confirmed=user_confirmed,
                corrected_profile=corrected_profile,
                corrected_duration=corrected_duration,
                notes=notes,
                dismiss=dismiss,
            )
            manager.notify_update()

            if success:
                # Best-effort dismiss the feedback notification if it exists.
                try:
                    notification_id = f"ha_washdata_feedback_{entry_id}_{cycle_id}"
                    persistent_notification.async_dismiss(hass, notification_id)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

                manager._logger.info("Cycle feedback submitted for %s", cycle_id)
            else:
                manager._logger.warning("Failed to submit feedback for cycle %s", cycle_id)

        hass.services.async_register(
            DOMAIN,
            SERVICE_SUBMIT_FEEDBACK.rsplit(".", maxsplit=1)[-1],
            handle_submit_feedback,
        )

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

            target = (
                Path(file_path)
                if file_path
                else Path(hass.config.path(f"ha_washdata_export_{entry_id}.json"))
            )
            target = target.resolve()

            # Restrict caller-supplied paths to HA-allowed dirs (path-traversal /
            # arbitrary-write guard). The default (no path) lands in the config dir.
            if file_path and not hass.config.is_allowed_path(str(target)):
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="path_not_allowed",
                    translation_placeholders={"path": str(target)},
                )

            # Write export (offloaded to executor to avoid blocking the event
            # loop). A caller-supplied path must never silently overwrite an
            # existing file even when is_allowed_path() accepts it; exclusive
            # creation ("x") makes that no-overwrite check atomic (no TOCTOU
            # window). The default generated path may be re-written freely.
            def _dump_and_write():
                text = json.dumps(payload, indent=2)
                try:
                    if file_path:
                        # Exclusive creation ("x") makes the no-overwrite check
                        # atomic; the default generated path may be re-written.
                        with open(target, "x", encoding="utf-8") as handle:
                            handle.write(text)
                    else:
                        target.write_text(text, encoding="utf-8")
                except FileExistsError as exc:
                    # Subclass of OSError -> must be caught first (no-overwrite).
                    raise ServiceValidationError(
                        translation_domain=DOMAIN,
                        translation_key="export_path_exists",
                        translation_placeholders={"path": str(target)},
                    ) from exc
                except OSError as exc:
                    # Disk full / permission denied / bad path: surface a clean
                    # localized error instead of a raw OSError from the executor.
                    raise ServiceValidationError(
                        translation_domain=DOMAIN,
                        translation_key="export_write_failed",
                        translation_placeholders={
                            "path": str(target), "error": str(exc)
                        },
                    ) from exc
            await hass.async_add_executor_job(_dump_and_write)
            manager._logger.info("Exported ha_washdata entry %s to %s", entry_id, target)

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

            # resolve()/exists() hit the filesystem; offload so the event loop is not
            # blocked on I/O during the import service call.
            source = await hass.async_add_executor_job(
                lambda: Path(file_path).resolve()
            )
            # Restrict reads to HA-allowed dirs (path-traversal / arbitrary-read guard).
            if not hass.config.is_allowed_path(str(source)):
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="path_not_allowed",
                    translation_placeholders={"path": str(source)},
                )
            if not await hass.async_add_executor_job(source.exists):
                raise ValueError(f"File not found: {source}")

            try:
                def _read_and_parse():
                    text = source.read_text(encoding="utf-8")
                    return json.loads(text)
                payload = await hass.async_add_executor_job(_read_and_parse)
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
                manager._logger.info("Applied imported settings to config entry %s", entry_id)

            manager._logger.info("Imported ha_washdata entry %s from %s", entry_id, source)

        hass.services.async_register(DOMAIN, "import_config", handle_import_config)

    # Register recorder services
    if not hass.services.has_service(DOMAIN, "record_start"):
        async def handle_record_start(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")
            entry_id = next(iter(device.config_entries), None)
            if not entry_id or entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded")

            manager = hass.data[DOMAIN][entry_id]
            await manager.async_start_recording()

        hass.services.async_register(DOMAIN, "record_start", handle_record_start)

    if not hass.services.has_service(DOMAIN, "record_stop"):
        async def handle_record_stop(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")
            entry_id = next(iter(device.config_entries), None)
            if not entry_id or entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded")

            manager = hass.data[DOMAIN][entry_id]
            await manager.async_stop_recording()

        hass.services.async_register(DOMAIN, "record_stop", handle_record_stop)

    # Register on-device ML training trigger (Stage 4, gated by ENABLE_ML_TRAINING)
    from .const import ENABLE_ML_TRAINING, SERVICE_TRIGGER_ML_TRAINING

    if ENABLE_ML_TRAINING and not hass.services.has_service(
        DOMAIN, SERVICE_TRIGGER_ML_TRAINING
    ):
        async def handle_trigger_ml_training(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ValueError("Device not found")
            entry_id = next(iter(device.config_entries), None)
            if not entry_id or entry_id not in hass.data[DOMAIN]:
                raise ValueError("Integration not loaded")

            manager = hass.data[DOMAIN][entry_id]
            summary = await manager.async_run_ml_training(force=True)
            manager._logger.info("Manual ML training: %s", summary)

        hass.services.async_register(
            DOMAIN, SERVICE_TRIGGER_ML_TRAINING, handle_trigger_ml_training
        )

    # Register pause/resume services
    if not hass.services.has_service(DOMAIN, "pause_cycle"):
        async def handle_pause_cycle(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="device_not_found",
                )
            entry_id = next(
                (eid for eid in device.config_entries if eid in hass.data.get(DOMAIN, {})),
                None,
            )
            if not entry_id:
                if any(eid for eid in device.config_entries):
                    raise ServiceValidationError(
                        translation_domain=DOMAIN,
                        translation_key="integration_not_loaded",
                    )
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="no_config_entry",
                )

            manager = hass.data[DOMAIN][entry_id]
            success = await manager.async_pause_cycle()
            if not success:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="no_active_cycle",
                )

        hass.services.async_register(DOMAIN, "pause_cycle", handle_pause_cycle)

    if not hass.services.has_service(DOMAIN, "resume_cycle"):
        async def handle_resume_cycle(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            registry = dr.async_get(hass)
            device = registry.async_get(device_id)
            if not device:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="device_not_found",
                )
            entry_id = next(
                (eid for eid in device.config_entries if eid in hass.data.get(DOMAIN, {})),
                None,
            )
            if not entry_id:
                if any(eid for eid in device.config_entries):
                    raise ServiceValidationError(
                        translation_domain=DOMAIN,
                        translation_key="integration_not_loaded",
                    )
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="no_config_entry",
                )

            manager = hass.data[DOMAIN][entry_id]
            success = await manager.async_resume_cycle()
            if not success:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="no_active_cycle",
                )

        hass.services.async_register(DOMAIN, "resume_cycle", handle_resume_cycle)

    return True


def _apply_device_link(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Sync the WashData device's via_device link with the configured option.

    When CONF_LINKED_DEVICE points at an existing device (e.g. the smart plug or
    appliance), the WashData device is shown as "Connected via <device>" in the
    HA device registry. Clearing the option removes the link. Stale targets that
    no longer exist are treated as "no link" so the registry never references a
    deleted device.
    """
    registry = dr.async_get(hass)
    washdata_device = registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    if washdata_device is None:
        return

    linked_device_id = entry.options.get(CONF_LINKED_DEVICE) or None
    if linked_device_id and registry.async_get(linked_device_id) is None:
        linked_device_id = None

    if washdata_device.via_device_id != linked_device_id:
        registry.async_update_device(
            washdata_device.id, via_device_id=linked_device_id
        )


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry - update settings without interrupting running cycles."""
    manager = hass.data[DOMAIN].get(entry.entry_id)
    if manager:
        # Update configuration without interrupting detector
        await manager.async_reload_config(entry)
        # Options changes (e.g. linked device) reload in place without
        # recreating entities, so apply the device link explicitly here.
        _apply_device_link(hass, entry)
    else:
        # Full reload if manager not found
        await async_unload_entry(hass, entry)
        await async_setup_entry(hass, entry)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        manager = hass.data[DOMAIN].pop(entry.entry_id)
        await manager.async_shutdown()
        # When the last WashData entry is removed, tear down the shared panel/sidebar
        # so no stale registration flags or sidebar entry linger.
        if not hass.data.get(DOMAIN):
            from .frontend import async_unregister_panel
            await async_unregister_panel(hass)

    return unload_ok
