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
"""Binary sensor for WashData."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.const import EntityCategory
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.translation import async_get_cached_translations

from .const import (
    DOMAIN,
    CYCLE_IN_PROGRESS_STATES,
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
    entities = [
        WasherRunningBinarySensor(manager, entry),
        WasherMaintenanceDueBinarySensor(manager, entry),
    ]

    if entry.options.get(CONF_EXPOSE_DEBUG_ENTITIES):
        entities.append(WasherAmbiguitySensor(manager, entry))

    async_add_entities(entities)
    cleanup_orphaned_diagnostic_entities(hass, manager, entry)


class WasherRunningBinarySensor(BinarySensorEntity):
    """Binary sensor indicating if washer is running."""

    _attr_should_poll = False  # pushed by the manager's update signal (PERF-02)

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
        # Any in-progress state, not only `running`: a soak, a pause or the end
        # wait is not "done" (audit PLATFORM-06).
        return self._manager.check_state() in CYCLE_IN_PROGRESS_STATES

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


# How often a day-based reminder is re-checked when nothing else happens. The cycle
# and log paths refresh the sensor through the update signal; only the passing of
# time needs a clock, and hourly is plenty for an interval counted in days.
_MAINTENANCE_RECHECK = timedelta(hours=1)


class WasherMaintenanceDueBinarySensor(BinarySensorEntity):
    """On while any maintenance task is due (#461): build your own notification.

    WashData itself never notifies about maintenance (no new notification type);
    this entity is the hook for a user's automation. Attributes list the due tasks
    with how far past their interval each is. Built-in task names are translated
    into Home Assistant's language; custom task names are the user's own text.
    """

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_translation_key = "maintenance_due"
    _attr_icon = "mdi:wrench-clock"

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize."""
        self._manager = manager
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_maintenance_due"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "WashData",
        }
        self._attr_is_on = False
        self._attr_extra_state_attributes = {"due_task_ids": [], "due_tasks": []}

    def _builtin_names(self) -> dict[str, str]:
        """Home Assistant's loaded entity strings for this integration ({} if none)."""
        try:
            return async_get_cached_translations(
                self.hass, self.hass.config.language, "entity", DOMAIN
            )
        except Exception:  # noqa: BLE001 - a missing translation is not an error
            return {}

    def _refresh(self) -> bool:
        """Recompute state and attributes; return whether either changed.

        The update signal fires on every power reading, so the state is written
        only when it changes (audit PERF-08 counts entity writes per reading).
        """
        due = [row for row in self._manager.maintenance_status if row.get("due")]
        strings = self._builtin_names() if any(not r.get("custom") for r in due) else {}
        prefix = (
            f"component.{DOMAIN}.entity.binary_sensor.maintenance_due"
            ".state_attributes.due_tasks.state."
        )
        tasks = [
            {
                "id": row["id"],
                "name": (
                    row["name"] if row.get("custom")
                    else strings.get(prefix + row["id"]) or row["id"]
                ),
                "cycles_since": row.get("cycles_since"),
                "cycles_interval": row.get("cycles_interval"),
                "days_since": row.get("days_since"),
                "days_interval": row.get("days_interval"),
            }
            for row in due
        ]
        attrs = {"due_task_ids": [t["id"] for t in tasks], "due_tasks": tasks}
        is_on = bool(tasks)
        if is_on == self._attr_is_on and attrs == self._attr_extra_state_attributes:
            return False
        self._attr_is_on = is_on
        self._attr_extra_state_attributes = attrs
        return True

    async def async_added_to_hass(self) -> None:
        """Register callbacks and compute the first state."""
        self._refresh()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_WASHER_UPDATE.format(self._entry.entry_id),
                self._update_callback,
            )
        )
        self.async_on_remove(
            async_track_time_interval(
                self.hass, self._update_callback, _MAINTENANCE_RECHECK
            )
        )

    @callback
    def _update_callback(self, *_args: Any) -> None:
        """Write the state only when it changed."""
        if self._refresh():
            self.async_write_ha_state()
