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
    DEFAULT_NAME,
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
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

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_MIN_POWER,
                    default=self.config_entry.options.get(
                        CONF_MIN_POWER,
                        self.config_entry.data.get(CONF_MIN_POWER, DEFAULT_MIN_POWER),
                    ),
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_OFF_DELAY,
                    default=self.config_entry.options.get(
                        CONF_OFF_DELAY,
                        self.config_entry.data.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY),
                    ),
                ): vol.Coerce(int),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
