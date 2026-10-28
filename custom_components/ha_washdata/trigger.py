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
"""Integration triggers (#478): ``trigger: ha_washdata.<type>``.

Home Assistant's trigger platform, so WashData is listed under "Integrations" in
the add-trigger dialog and its triggers under a WashData device, entity or area::

    triggers:
      - trigger: ha_washdata.finishing_soon
        target:
          device_id: <washdata device>
        options:
          minutes: 10

The target (device, entity, area, floor, label) is resolved to WashData config
entries each time an occurrence is sent, so moving a device to another area
takes effect at once. The manager's ``SIGNAL_DEVICE_TRIGGER`` is the only firing
path, shared with the device triggers; ``trigger_rules.run_variables`` applies
the options. The descriptions and selectors are in ``triggers.yaml``.

Uses only what Home Assistant 2026.2 (the test pin) and 2026.9 share:
``Trigger`` / ``TriggerConfig``, ``async_get_triggers`` and
``helpers.target``. ``async_attach_runner`` takes the ``did_not_trigger``
reporter 2026.9 passes as an optional argument and never calls it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    SIGNAL_CONFIG_ENTRY_CHANGED,
    ConfigEntry,
    ConfigEntryChange,
)
from homeassistant.const import CONF_OPTIONS, CONF_TARGET
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.trigger import Trigger, TriggerActionRunner, TriggerConfig
from homeassistant.helpers.typing import ConfigType

from .const import DEVICE_TRIGGER_TYPES, DOMAIN
from .device_trigger import (
    async_connect_trigger_signal,
    entry_for_device,
    washdata_device_id,
)
from .trigger_rules import describe, options_schema, run_variables

try:  # 2026.2 and 2026.9; TargetSelectorData is its deprecated alias until 2026.12
    from homeassistant.helpers.target import TargetSelection
except ImportError:  # pragma: no cover - an older core
    from homeassistant.helpers.target import (  # type: ignore[attr-defined,no-redef]
        TargetSelectorData as TargetSelection,
    )
from homeassistant.helpers.target import async_extract_referenced_entity_ids

_TARGET_SCHEMA = vol.Schema(
    vol.All(cv.TARGET_FIELDS, cv.has_at_least_one_key(*cv.TARGET_FIELDS))
)


@callback
def async_target_entry_ids(hass: HomeAssistant, target: dict[str, Any]) -> set[str]:
    """The WashData config entries a target reaches: its WashData devices (directly,
    or through an area, floor or label) and the entries of its WashData entities."""
    selected = async_extract_referenced_entity_ids(
        hass, TargetSelection(target), expand_group=False
    )
    entry_ids: set[str] = set()
    for device_id in selected.referenced_devices:
        entry = entry_for_device(hass, device_id)
        if entry is not None:
            entry_ids.add(entry.entry_id)
    ent_reg = er.async_get(hass)
    for entity_id in selected.referenced | selected.indirectly_referenced:
        entity = ent_reg.async_get(entity_id)
        if entity is not None and entity.platform == DOMAIN and entity.config_entry_id:
            entry_ids.add(entity.config_entry_id)
    return entry_ids


@callback
def async_listen_all_entries(
    hass: HomeAssistant, handler: Callable[[str, str, dict[str, Any]], None]
) -> CALLBACK_TYPE:
    """Call ``handler(entry_id, type, variables)`` for every WashData entry's
    trigger signal, including entries added later. The signal is per entry id,
    which survives a reload, so a reload needs nothing here."""
    unsubs: dict[str, CALLBACK_TYPE] = {}

    @callback
    def _add(entry_id: str) -> None:
        if entry_id not in unsubs:
            unsubs[entry_id] = async_connect_trigger_signal(
                hass, entry_id, partial(handler, entry_id)
            )

    for entry in hass.config_entries.async_entries(DOMAIN):
        _add(entry.entry_id)

    @callback
    def _entry_changed(change: ConfigEntryChange, entry: ConfigEntry) -> None:
        if entry.domain != DOMAIN:
            return
        if change == ConfigEntryChange.ADDED:
            _add(entry.entry_id)
        elif change == ConfigEntryChange.REMOVED and entry.entry_id in unsubs:
            unsubs.pop(entry.entry_id)()

    unsub_changes = async_dispatcher_connect(
        hass, SIGNAL_CONFIG_ENTRY_CHANGED, _entry_changed
    )

    @callback
    def _unsubscribe() -> None:
        unsub_changes()
        for unsub in unsubs.values():
            unsub()
        unsubs.clear()

    return _unsubscribe


class WashDataTrigger(Trigger):
    """One WashData trigger type; subclassed per type below."""

    trigger_type: str

    @classmethod
    async def async_validate_config(
        cls, hass: HomeAssistant, config: ConfigType
    ) -> ConfigType:
        """A target is required; the options are the type's, with their defaults."""
        return {
            CONF_TARGET: _TARGET_SCHEMA(config.get(CONF_TARGET)),
            CONF_OPTIONS: options_schema(cls.trigger_type)(config.get(CONF_OPTIONS) or {}),
        }

    def __init__(self, hass: HomeAssistant, config: TriggerConfig) -> None:
        """Keep the target and the options."""
        super().__init__(hass, config)
        self._target = dict(config.target or {})
        self._options = dict(config.options or {})

    async def async_attach_runner(
        self, run_action: TriggerActionRunner, did_not_trigger: Any = None
    ) -> CALLBACK_TYPE:
        """Run the action for each occurrence on a targeted WashData entry."""
        trigger_type = self.trigger_type
        hass = self._hass

        @callback
        def _handle(entry_id: str, fired_type: str, variables: dict[str, Any]) -> None:
            if fired_type != trigger_type:
                return
            if entry_id not in async_target_entry_ids(hass, self._target):
                return
            run = run_variables(trigger_type, self._options, variables)
            if run is None:
                return
            run_action(
                {
                    **run,
                    "type": trigger_type,
                    "device_id": washdata_device_id(hass, entry_id),
                },
                describe(trigger_type, run.get("device_name")),
            )

        return async_listen_all_entries(hass, _handle)


TRIGGERS: dict[str, type[Trigger]] = {
    trigger_type: type(
        f"WashData{''.join(p.title() for p in trigger_type.split('_'))}Trigger",
        (WashDataTrigger,),
        {"trigger_type": trigger_type},
    )
    for trigger_type in DEVICE_TRIGGER_TYPES
}


async def async_get_triggers(hass: HomeAssistant) -> dict[str, type[Trigger]]:
    """The WashData integration triggers."""
    return TRIGGERS
