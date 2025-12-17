"""Config flow for HA WashData integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    CONF_POWER_SENSOR,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_EVENTS,
    NOTIFY_EVENT_START,
    NOTIFY_EVENT_FINISH,
    DEFAULT_NAME,
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
        vol.Required(CONF_POWER_SENSOR): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor"),
        ),
        vol.Optional(CONF_MIN_POWER, default=DEFAULT_MIN_POWER): vol.Coerce(float),
    }
)

class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA WashData."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle a options flow for HA WashData."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry
        self._selected_cycle_id: str | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["settings", "manage_profiles", "post_process", "migrate_data", "wipe_history"]
        )

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage configuration settings."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Populate notify services
        notify_services = []
        services = self.hass.services.async_services()
        for service in services.get("notify", {}):
            notify_services.append(f"notify.{service}")
        notify_services.sort()
        
        # Ensure current value is in the list (so it doesn't vanish)
        current_notify = self._config_entry.options.get(
            CONF_NOTIFY_SERVICE,
            self._config_entry.data.get(CONF_NOTIFY_SERVICE, "")
        )
        if current_notify and current_notify not in notify_services:
            notify_services.append(current_notify)

        return self.async_show_form(
            step_id="settings",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MIN_POWER,
                        default=self._config_entry.options.get(
                            CONF_MIN_POWER,
                            self._config_entry.data.get(CONF_MIN_POWER, DEFAULT_MIN_POWER),
                        ),
                    ): vol.Coerce(float),
                    vol.Optional(
                        CONF_OFF_DELAY,
                        default=self._config_entry.options.get(
                            CONF_OFF_DELAY,
                            self._config_entry.data.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY),
                        ),
                    ): vol.Coerce(int),
                    vol.Optional(
                        CONF_NOTIFY_SERVICE,
                        default=self._config_entry.options.get(
                            CONF_NOTIFY_SERVICE,
                            self._config_entry.data.get(CONF_NOTIFY_SERVICE, ""),
                        ),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=notify_services,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                            custom_value=True,
                        )
                    ),
                    vol.Optional(
                        CONF_NOTIFY_EVENTS,
                        default=list(self._config_entry.options.get(
                            CONF_NOTIFY_EVENTS,
                            self._config_entry.data.get(CONF_NOTIFY_EVENTS, []),
                        )),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value=NOTIFY_EVENT_START, label="Cycle Start"),
                                selector.SelectOptionDict(value=NOTIFY_EVENT_FINISH, label="Cycle Finish"),
                            ],
                            multiple=True,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
        )

    async def async_step_manage_profiles(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Choose what to do with cycles."""
        if user_input is not None:
            action = user_input["action"]
            if action == "label":
                return await self.async_step_select_cycle_to_label()
            elif action == "delete":
                return await self.async_step_select_cycle_to_delete()

        return self.async_show_form(
            step_id="manage_profiles",
            data_schema=vol.Schema({
                vol.Required("action"): vol.In({
                    "label": "Label a Cycle",
                    "delete": "Delete a Cycle"
                })
            })
        )

    async def async_step_select_cycle_to_label(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select a cycle to label."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        
        # Get last 20 cycles
        cycles = store._data.get("past_cycles", [])[-20:]
        
        # Build readable options with status
        options = []
        for c in reversed(cycles):
            start = c["start_time"].split(".")[0].replace("T", " ")
            duration_min = int(c['duration']/60)
            prof = c.get("profile") or "Unlabeled"
            status = c.get("status", "completed")
            status_icon = "✓" if status == "completed" else "⚠" if status == "resumed" else "✗"
            label = f"[{status_icon}] {start} - {duration_min}m - {prof}"
            options.append(selector.SelectOptionDict(value=c["id"], label=label))
            
        if not options:
            return self.async_abort(reason="no_cycles_found")

        if user_input is not None:
            self._selected_cycle_id = user_input["cycle_id"]
            return await self.async_step_label_cycle()

        return self.async_show_form(
            step_id="select_cycle_to_label",
            data_schema=vol.Schema({
                vol.Required("cycle_id"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                )
            })
        )

    async def async_step_select_cycle_to_delete(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select a cycle to delete."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        
        # Get last 20 cycles
        cycles = store._data.get("past_cycles", [])[-20:]
        
        # Build readable options with status
        options = []
        for c in reversed(cycles):
            start = c["start_time"].split(".")[0].replace("T", " ")
            duration_min = int(c['duration']/60)
            prof = c.get("profile") or "Unlabeled"
            status = c.get("status", "completed")
            status_icon = "✓" if status == "completed" else "⚠" if status == "resumed" else "✗"
            label = f"[{status_icon}] {start} - {duration_min}m - {prof}"
            options.append(selector.SelectOptionDict(value=c["id"], label=label))
            
        if not options:
            return self.async_abort(reason="no_cycles_found")

        if user_input is not None:
            cycle_id = user_input["cycle_id"]
            manager.profile_store.delete_cycle(cycle_id)
            await manager.profile_store.async_save()
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="select_cycle_to_delete",
            data_schema=vol.Schema({
                vol.Required("cycle_id"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                )
            }),
            description_placeholders={"warning": "⚠️ This will permanently delete the selected cycle"}
        )

    async def async_step_label_cycle(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Name the selected cycle."""
        if user_input is not None:
            name = user_input["profile_name"]
            manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
            await manager.profile_store.create_profile(name, self._selected_cycle_id)
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="label_cycle",
            data_schema=vol.Schema({
                vol.Required("profile_name"): str
            })
        )

    async def async_step_post_process(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle post-processing options."""
        if user_input is not None:
             choice = user_input["time_range"]
             manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
             
             if choice == "all":
                 # Merge all cycles (no time limit)
                 count = manager.profile_store.merge_cycles(hours=999999)
             else:
                 hours = int(choice)
                 count = manager.profile_store.merge_cycles(hours=hours)
             
             if count > 0:
                 await manager.profile_store.async_save()
                 
             return self.async_create_entry(
                 title="",
                 data={},
                 description_placeholders={"count": str(count)}
             )

        return self.async_show_form(
            step_id="post_process",
            data_schema=vol.Schema({
                vol.Required("time_range", default="24"): vol.In({
                    "12": "Last 12 Hours",
                    "24": "Last 24 Hours",
                    "48": "Last 48 Hours",
                    "168": "Last 7 Days",
                    "all": "All Data"
                })
            })
        )

    async def async_step_migrate_data(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Migrate/compress all cycle data to the latest format."""
        if user_input is not None:
             manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
             
             # Run migration
             count = await manager.profile_store.async_migrate_cycles_to_compressed()
             
             return self.async_create_entry(
                 title="",
                 data={},
                 description_placeholders={"count": str(count)}
             )

        return self.async_show_form(
            step_id="migrate_data",
            data_schema=vol.Schema({}),
            description_placeholders={"info": "This will re-compress all saved cycle data to ensure it's in the latest format. This is safe and can be run multiple times."}
        )

    async def async_step_wipe_history(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Wipe all stored cycles and profiles for this device (for testing)."""
        if user_input is not None:
             manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
             
             # Clear all cycles and profiles
             manager.profile_store._data["past_cycles"] = []
             manager.profile_store._data["profiles"] = {}
             await manager.profile_store.async_save()
             
             return self.async_create_entry(
                 title="",
                 data={},
                 description_placeholders={"info": "History cleared"}
             )

        return self.async_show_form(
            step_id="wipe_history",
            data_schema=vol.Schema({}),
            description_placeholders={"warning": "⚠️ This will permanently delete ALL stored cycles and profiles for this device. This cannot be undone!"}
        )
