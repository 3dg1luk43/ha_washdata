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
            menu_options=["settings", "manage_profiles"]
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
                        default=self._config_entry.options.get(
                            CONF_NOTIFY_EVENTS,
                            self._config_entry.data.get(CONF_NOTIFY_EVENTS, []),
                        ),
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
        """Select a cycle to label."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        
        # Get last 20 cycles
        cycles = store._data.get("past_cycles", [])[-20:]
        
        # Filter reversed (newest first)
        options = []
        for c in reversed(cycles):
            start = c["start_time"].split(".")[0].replace("T", " ")
            prof = c.get("profile") or "Unlabeled"
            label = f"{start} - {int(c['duration']/60)}m - {prof}"
            options.append(selector.SelectOptionDict(value=c["id"], label=label))
            
        if not options:
            return self.async_abort(reason="no_cycles_found")

        if user_input is not None:
            self._selected_cycle_id = user_input["cycle_id"]
            return await self.async_step_label_cycle()

        return self.async_show_form(
            step_id="manage_profiles",
            data_schema=vol.Schema({
                vol.Required("cycle_id"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                )
            })
        )

    async def async_step_label_cycle(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Name the selected cycle."""
        if user_input is not None:
            name = user_input["profile_name"]
            manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
            manager.profile_store.create_profile(name, self._selected_cycle_id)
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="label_cycle",
            data_schema=vol.Schema({
                vol.Required("profile_name"): str
            })
        )
