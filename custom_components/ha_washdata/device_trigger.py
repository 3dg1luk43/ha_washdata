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
"""Device triggers for WashData (#478): the device page's "Automations" card.

The manager sends ``SIGNAL_DEVICE_TRIGGER`` for its config entry with
``(trigger_type, variables)`` at the moment each occurrence happens (see
``WashDataManager._fire_device_trigger``). Here a trigger resolves its device to
that entry, applies its options (extra fields, ``trigger_rules``) and runs the
automation with the variables under ``trigger.*``. The integration triggers in
``trigger.py`` listen to the same signal.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components.device_automation import DEVICE_TRIGGER_BASE_SCHEMA
from homeassistant.components.device_automation.exceptions import (
    InvalidDeviceAutomationConfig,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_DEVICE_ID, CONF_DOMAIN, CONF_PLATFORM, CONF_TYPE
from homeassistant.core import CALLBACK_TYPE, HassJob, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import selector
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.trigger import TriggerActionType, TriggerInfo
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_DEVICE_TYPE,
    DATA_TRIGGER_LISTENERS,
    DEFAULT_DEVICE_TYPE,
    DEVICE_TRIGGER_TYPES,
    DOMAIN,
    SIGNAL_DEVICE_TRIGGER,
)
from .trigger_rules import (
    OPT_FROM,
    OPT_STATUS,
    OPT_TO,
    OPTION_FIELDS,
    describe,
    run_variables,
    trigger_offered,
)

_TYPE_SCHEMA = DEVICE_TRIGGER_BASE_SCHEMA.extend(
    {vol.Required(CONF_TYPE): vol.In(DEVICE_TRIGGER_TYPES)}, extra=vol.ALLOW_EXTRA
)
# Per type: the base keys plus that type's options as top-level extra fields.
_TYPE_SCHEMAS = {
    trigger_type: DEVICE_TRIGGER_BASE_SCHEMA.extend(
        {vol.Required(CONF_TYPE): trigger_type, **OPTION_FIELDS.get(trigger_type, {})}
    )
    for trigger_type in DEVICE_TRIGGER_TYPES
}
# The select fields get a translated selector in the editor; the rest serialize as is.
_SELECT_TRANSLATION_KEYS = {
    OPT_STATUS: "trigger_cycle_status",
    OPT_FROM: "trigger_state",
    OPT_TO: "trigger_state",
}


def TRIGGER_SCHEMA(config: ConfigType) -> ConfigType:  # noqa: N802 - HA's platform name
    """The type, then that type's options (unknown keys rejected)."""
    trigger_type = _TYPE_SCHEMA(config)[CONF_TYPE]
    return _TYPE_SCHEMAS[trigger_type](config)


async def async_validate_trigger_config(
    hass: HomeAssistant, config: ConfigType
) -> ConfigType:
    """Validate a WashData device trigger."""
    return TRIGGER_SCHEMA(config)


@callback
def entry_for_device(hass: HomeAssistant, device_id: str) -> ConfigEntry | None:
    """The WashData config entry behind a device: identifier ``(DOMAIN, entry_id)``."""
    device = dr.async_get(hass).async_get(device_id)
    if device is None:
        return None
    candidates = [ident for domain, ident in device.identifiers if domain == DOMAIN]
    candidates += list(device.config_entries)
    for entry_id in candidates:
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is not None and entry.domain == DOMAIN:
            return entry
    return None


@callback
def async_connect_trigger_signal(
    hass: HomeAssistant, entry_id: str, handler: Any
) -> CALLBACK_TYPE:
    """Listen to one entry's trigger signal, counted in ``DATA_TRIGGER_LISTENERS``
    so the manager knows something is attached (see ``_has_trigger_listeners``)."""
    counts: dict[str, int] = hass.data.setdefault(DATA_TRIGGER_LISTENERS, {})
    counts[entry_id] = counts.get(entry_id, 0) + 1
    unsub = async_dispatcher_connect(hass, SIGNAL_DEVICE_TRIGGER.format(entry_id), handler)
    removed = False

    @callback
    def _remove() -> None:
        nonlocal removed
        if removed:
            return
        removed = True
        unsub()
        counts[entry_id] = max(0, counts.get(entry_id, 1) - 1)

    return _remove


@callback
def washdata_device(registry: dr.DeviceRegistry, entry_id: str) -> dr.DeviceEntry | None:
    """The WashData device of a config entry, or None before its entities exist.

    HA 2026.9+ deprecated ``async_get_device`` because identifiers are no longer
    unique across config entries (#405). This device's identifier is owned by its
    entry, so the by-identifier lookup is unambiguous; the attribute guard keeps the
    older HA the manifest still supports, where that method does not exist yet.
    """
    identifier = (DOMAIN, entry_id)
    if hasattr(registry, "async_get_device_by_identifier"):
        return registry.async_get_device_by_identifier(identifier, entry_id)
    return registry.async_get_device(identifiers={identifier})


@callback
def washdata_device_id(hass: HomeAssistant, entry_id: str) -> str | None:
    """The WashData device's id for a config entry, or None before it exists."""
    device = washdata_device(dr.async_get(hass), entry_id)
    return device.id if device is not None else None


@callback
def _device_type(hass: HomeAssistant, entry: ConfigEntry) -> str:
    """The running manager's device type, else the entry's stored one."""
    manager = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    device_type = getattr(manager, "device_type", None)
    if isinstance(device_type, str):
        return device_type
    return str(
        entry.options.get(
            CONF_DEVICE_TYPE, entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)
        )
    )


async def async_get_triggers(
    hass: HomeAssistant, device_id: str
) -> list[dict[str, Any]]:
    """List the triggers a WashData device offers."""
    entry = entry_for_device(hass, device_id)
    if entry is None:
        return []
    device_type = _device_type(hass, entry)
    return [
        {
            CONF_PLATFORM: "device",
            CONF_DOMAIN: DOMAIN,
            CONF_DEVICE_ID: device_id,
            CONF_TYPE: trigger_type,
        }
        for trigger_type in DEVICE_TRIGGER_TYPES
        if trigger_offered(trigger_type, device_type)
    ]


async def async_get_trigger_capabilities(
    hass: HomeAssistant, config: ConfigType
) -> dict[str, vol.Schema]:
    """The type's options as the editor's extra fields."""
    fields: dict[Any, Any] = {}
    for key, validator in OPTION_FIELDS.get(config.get(CONF_TYPE, ""), {}).items():
        translation_key = _SELECT_TRANSLATION_KEYS.get(str(key))
        if translation_key and isinstance(validator, vol.In):
            validator = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=list(validator.container),
                    translation_key=translation_key,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
        fields[key] = validator
    return {"extra_fields": vol.Schema(fields)} if fields else {}


async def async_attach_trigger(
    hass: HomeAssistant,
    config: ConfigType,
    action: TriggerActionType,
    trigger_info: TriggerInfo,
) -> CALLBACK_TYPE:
    """Run ``action`` each time the device's manager sends this trigger type."""
    device_id: str = config[CONF_DEVICE_ID]
    trigger_type: str = config[CONF_TYPE]
    entry = entry_for_device(hass, device_id)
    if entry is None:
        raise InvalidDeviceAutomationConfig(
            f"Device '{device_id}' is not a WashData device"
        )
    options = {
        str(key): config[str(key)]
        for key in OPTION_FIELDS.get(trigger_type, {})
        if str(key) in config
    }
    trigger_data = trigger_info["trigger_data"]
    job = HassJob(action, f"WashData device trigger {trigger_type}")

    @callback
    def _handle(fired_type: str, variables: dict[str, Any]) -> None:
        if fired_type != trigger_type:
            return
        run = run_variables(trigger_type, options, variables)
        if run is None:
            return
        hass.async_run_hass_job(
            job,
            {
                "trigger": {
                    **run,
                    **trigger_data,
                    CONF_PLATFORM: "device",
                    CONF_DOMAIN: DOMAIN,
                    CONF_DEVICE_ID: device_id,
                    CONF_TYPE: trigger_type,
                    "description": describe(trigger_type, run.get("device_name")),
                }
            },
        )

    return async_connect_trigger_signal(hass, entry.entry_id, _handle)
