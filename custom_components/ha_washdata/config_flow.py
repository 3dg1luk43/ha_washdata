"""Config flow for WashData integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_POWER_SENSOR,
    CONF_MIN_POWER,
    CONF_DEVICE_TYPE,
    DEFAULT_NAME,
    DEFAULT_MIN_POWER,
    DEFAULT_DEVICE_TYPE,
    DEVICE_TYPES,
)


_LOGGER = logging.getLogger(__name__)


def _device_type_options(
    current: str | None = None,  # pylint: disable=unused-argument
) -> list[str]:
    """Build the device-type dropdown option keys."""
    return list(DEVICE_TYPES)


def _resolve_options_first(
    entry: config_entries.ConfigEntry, key: str, default: Any = None
) -> Any:
    """Resolve a config value options-first (then data), matching WashDataManager.

    Reading data-first would surface the creation-time value and, after the 3.6
    device-type remap (which updates only options), a value no longer in
    DEVICE_TYPES.
    """
    return entry.options.get(key, entry.data.get(key, default))


def _merge_structural_options(
    entry: config_entries.ConfigEntry, user_input: dict[str, Any]
) -> dict[str, Any]:
    """Merge the three structural tunables into the existing options.

    Only the structural fields are merged; ``entry.data`` (identity keys, name)
    is deliberately NOT spread in. Also strip identity/data-only keys that legacy
    entries may have leaked into options (e.g. ``CONF_NAME`` from the old
    data-spread bug) so they don't persist there.
    """
    preserved = {k: v for k, v in entry.options.items() if k != CONF_NAME}
    return {
        **preserved,
        CONF_DEVICE_TYPE: user_input[CONF_DEVICE_TYPE],
        CONF_POWER_SENSOR: user_input[CONF_POWER_SENSOR],
        CONF_MIN_POWER: user_input[CONF_MIN_POWER],
    }



STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
        vol.Required(
            CONF_DEVICE_TYPE, default=DEFAULT_DEVICE_TYPE
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=_device_type_options(),
                mode=selector.SelectSelectorMode.DROPDOWN,
                translation_key="device_type",
            )
        ),
        vol.Required(CONF_POWER_SENSOR): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor"),
        ),
        vol.Optional(CONF_MIN_POWER, default=DEFAULT_MIN_POWER): vol.Coerce(float),
    }
)


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):  # pylint: disable=abstract-method
    """Handle a config flow for WashData."""

    VERSION = 3
    MINOR_VERSION = 6

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._user_input: dict[str, Any] = {}

    def _get_schema(
        self, user_input: dict[str, Any] | None = None  # pylint: disable=unused-argument
    ) -> vol.Schema:
        """Get the configuration schema."""
        return STEP_USER_DATA_SCHEMA

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None  # pylint: disable=unused-argument
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=self._get_schema(), errors=errors
            )

        try:
            if user_input[CONF_MIN_POWER] <= 0:
                errors[CONF_MIN_POWER] = "invalid_power"
        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"

        if errors:
            return self.async_show_form(
                step_id="user", data_schema=self._get_schema(user_input), errors=errors
            )

        self._user_input = user_input
        return await self.async_step_first_profile()

    async def async_step_first_profile(
        self, user_input: dict[str, Any] | None = None  # pylint: disable=unused-argument
    ) -> FlowResult:
        """Step to optionally create the first profile."""

        if user_input is not None:
            profile_name = user_input.get("profile_name", "").strip()
            data = dict(self._user_input)

            if profile_name:
                duration_mins = user_input.get("manual_duration")
                duration_sec = (duration_mins * 60.0) if duration_mins else None
                data["initial_profile"] = {
                    "name": profile_name,
                    "avg_duration": duration_sec,
                }

            return self.async_create_entry(title=data[CONF_NAME], data=data)

        schema = vol.Schema(
            {
                vol.Optional("profile_name"): str,
                vol.Optional("manual_duration", default=120): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=480,
                        unit_of_measurement="min",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }
        )

        return self.async_show_form(step_id="first_profile", data_schema=schema)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle reconfigure flow to change sensor, device type, or name."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                if user_input[CONF_MIN_POWER] <= 0:
                    errors[CONF_MIN_POWER] = "invalid_power"
            except Exception:  # pylint: disable=broad-exception-caught
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

            if not errors:
                new_options = _merge_structural_options(entry, user_input)
                # NB: intentionally do NOT write entry.data here — post-3.6 the
                # structural fields live in options and the display name is carried
                # by the entry title (see test_reconfigure_saves_and_aborts_on_valid_input).
                return self.async_update_reload_and_abort(
                    entry,
                    title=user_input[CONF_NAME],
                    options=new_options,
                )

        # Resolve options-first to mirror WashDataManager (which reads
        # options.get(..., data.get(...))). Reading data-first here would show the
        # creation-time value and, after the 3.6 device-type remap (which updates
        # only options), a value no longer in DEVICE_TYPES.
        current_device_type = _resolve_options_first(
            entry, CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE
        )
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_NAME,
                    default=entry.title,
                ): str,
                vol.Required(
                    CONF_DEVICE_TYPE,
                    default=current_device_type,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=_device_type_options(current_device_type),
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        translation_key="device_type",
                    )
                ),
                vol.Required(
                    CONF_POWER_SENSOR,
                    default=_resolve_options_first(entry, CONF_POWER_SENSOR, ""),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor"),
                ),
                vol.Optional(
                    CONF_MIN_POWER,
                    default=_resolve_options_first(entry, CONF_MIN_POWER, DEFAULT_MIN_POWER),
                ): vol.Coerce(float),
            }
        )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Minimal options flow -- a stub that redirects users to the WashData panel.

    All settings (detection, matching, timing, notifications, profiles, cycles,
    diagnostics) are available in the full-screen WashData panel accessible via
    the panel icon in the HA sidebar. This form handles only the three structural
    fields that are quickest to change via the HA integrations UI.
    """

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show the minimal options form."""
        entry = self._config_entry
        # Resolve options-first to mirror WashDataManager (which reads
        # options.get(..., data.get(...))). Reading data-first here would show the
        # creation-time value and, after the 3.6 device-type remap (which updates
        # only options), a value no longer in DEVICE_TYPES.
        current_device_type = _resolve_options_first(
            entry, CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE
        )

        errors: dict[str, str] = {}

        if user_input is not None:
            if user_input.get(CONF_MIN_POWER, 0) <= 0:
                errors[CONF_MIN_POWER] = "invalid_power"
            if not errors:
                new_name = str(user_input.get(CONF_NAME, "")).strip()
                new_options = _merge_structural_options(entry, user_input)
                if new_name and new_name != entry.title:
                    self.hass.config_entries.async_update_entry(
                        entry,
                        title=new_name,
                        data={**entry.data, CONF_NAME: new_name},
                    )
                return self.async_create_entry(title="", data=new_options)

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_NAME,
                    default=entry.title,
                ): str,
                vol.Required(
                    CONF_DEVICE_TYPE,
                    default=current_device_type,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=_device_type_options(current_device_type),
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        translation_key="device_type",
                    )
                ),
                vol.Required(
                    CONF_POWER_SENSOR,
                    default=_resolve_options_first(entry, CONF_POWER_SENSOR, ""),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor"),
                ),
                vol.Optional(
                    CONF_MIN_POWER,
                    default=_resolve_options_first(entry, CONF_MIN_POWER, DEFAULT_MIN_POWER),
                ): vol.Coerce(float),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)
