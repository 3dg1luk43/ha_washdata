"""Config flow for HA WashData integration."""
from __future__ import annotations

import json
import logging
import os
import time
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
    CONF_OFF_DELAY,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_EVENTS,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_SMOOTHING_WINDOW,
    CONF_START_DURATION_THRESHOLD,
    CONF_DEVICE_TYPE,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_AUTO_MERGE_LOOKBACK_HOURS,
    CONF_AUTO_MERGE_GAP_SECONDS,
    CONF_APPLY_SUGGESTIONS,
    CONF_SHOW_ADVANCED,
    CONF_INTERRUPTED_MIN_SECONDS,
    CONF_ABRUPT_DROP_WATTS,
    CONF_ABRUPT_DROP_RATIO,
    CONF_ABRUPT_HIGH_LOAD_FACTOR,
    CONF_PROGRESS_RESET_DELAY,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_AUTO_MAINTENANCE,
    CONF_MAX_PAST_CYCLES,
    CONF_MAX_FULL_TRACES_PER_PROFILE,
    CONF_MAX_FULL_TRACES_UNLABELED,
    CONF_WATCHDOG_INTERVAL,
    CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_RUNNING_DEAD_ZONE,
    CONF_END_REPEAT_COUNT,
    CONF_SMART_EXTENSION_THRESHOLD,
    NOTIFY_EVENT_START,
    NOTIFY_EVENT_FINISH,
    DEFAULT_NAME,
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_START_DURATION_THRESHOLD,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_PROFILE_DURATION_TOLERANCE,
    DEVICE_TYPES,
    DEFAULT_AUTO_MERGE_LOOKBACK_HOURS,
    DEFAULT_AUTO_MERGE_GAP_SECONDS,
    DEFAULT_INTERRUPTED_MIN_SECONDS,
    DEFAULT_ABRUPT_DROP_WATTS,
    DEFAULT_ABRUPT_DROP_RATIO,
    DEFAULT_ABRUPT_HIGH_LOAD_FACTOR,
    DEFAULT_PROGRESS_RESET_DELAY,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_AUTO_MAINTENANCE,
    DEFAULT_MAX_PAST_CYCLES,
    DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
    DEFAULT_MAX_FULL_TRACES_UNLABELED,
    DEFAULT_WATCHDOG_INTERVAL,
    DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_RUNNING_DEAD_ZONE,
    DEFAULT_END_REPEAT_COUNT,
    DEFAULT_SMART_EXTENSION_THRESHOLD,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
        vol.Required(CONF_DEVICE_TYPE, default=DEFAULT_DEVICE_TYPE): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(value=k, label=v)
                    for k, v in DEVICE_TYPES.items()
                ],
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Required(CONF_POWER_SENSOR): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor"),
        ),
        vol.Optional(CONF_MIN_POWER, default=DEFAULT_MIN_POWER): vol.Coerce(float),
    }
)

class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA WashData."""

    VERSION = 3

    async def async_migrate_entry(self, config_entry: config_entries.ConfigEntry) -> bool:
        """Migrate old entry to the latest version while preserving user settings."""
        data = {**config_entry.data}
        options = {**config_entry.options}

        # Ensure min_power/off_delay from data are preserved into options if missing
        if CONF_MIN_POWER not in options and CONF_MIN_POWER in data:
            options[CONF_MIN_POWER] = data[CONF_MIN_POWER]
        if CONF_OFF_DELAY not in options and CONF_OFF_DELAY in data:
            options[CONF_OFF_DELAY] = data[CONF_OFF_DELAY]

        # Seed new configurable values with defaults if missing
        options.setdefault(CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY)
        options.setdefault(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE)
        options.setdefault(CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE)
        options.setdefault(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE)
        options.setdefault(CONF_AUTO_MAINTENANCE, DEFAULT_AUTO_MAINTENANCE)
        
        # New Feature Defaults (Migration)
        options.setdefault(CONF_DEVICE_TYPE, data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE))
        options.setdefault(CONF_START_DURATION_THRESHOLD, DEFAULT_START_DURATION_THRESHOLD)

        # Seed detector settings if missing
        options.setdefault(CONF_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW)
        options.setdefault(CONF_NO_UPDATE_ACTIVE_TIMEOUT, DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT)
        options.setdefault(CONF_PROFILE_DURATION_TOLERANCE, DEFAULT_PROFILE_DURATION_TOLERANCE)
        options.setdefault(CONF_AUTO_MERGE_LOOKBACK_HOURS, DEFAULT_AUTO_MERGE_LOOKBACK_HOURS)
        options.setdefault(CONF_AUTO_MERGE_GAP_SECONDS, DEFAULT_AUTO_MERGE_GAP_SECONDS)
        options.setdefault(CONF_INTERRUPTED_MIN_SECONDS, DEFAULT_INTERRUPTED_MIN_SECONDS)
        options.setdefault(CONF_ABRUPT_DROP_WATTS, DEFAULT_ABRUPT_DROP_WATTS)
        options.setdefault(CONF_ABRUPT_DROP_RATIO, DEFAULT_ABRUPT_DROP_RATIO)
        options.setdefault(CONF_ABRUPT_HIGH_LOAD_FACTOR, DEFAULT_ABRUPT_HIGH_LOAD_FACTOR)
        options.setdefault(CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL)
        options.setdefault(CONF_PROFILE_MATCH_MIN_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO)
        options.setdefault(CONF_PROFILE_MATCH_MAX_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO)
        # New retention and watchdog defaults
        options.setdefault(CONF_MAX_PAST_CYCLES, DEFAULT_MAX_PAST_CYCLES)
        options.setdefault(CONF_MAX_FULL_TRACES_PER_PROFILE, DEFAULT_MAX_FULL_TRACES_PER_PROFILE)
        options.setdefault(CONF_MAX_FULL_TRACES_UNLABELED, DEFAULT_MAX_FULL_TRACES_UNLABELED)
        options.setdefault(CONF_WATCHDOG_INTERVAL, DEFAULT_WATCHDOG_INTERVAL)
        options.setdefault(CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD, DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD)
        options.setdefault(CONF_COMPLETION_MIN_SECONDS, DEFAULT_COMPLETION_MIN_SECONDS)
        options.setdefault(CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES)
        # New dead zone and end repeat count defaults
        options.setdefault(CONF_RUNNING_DEAD_ZONE, DEFAULT_RUNNING_DEAD_ZONE)
        options.setdefault(CONF_END_REPEAT_COUNT, DEFAULT_END_REPEAT_COUNT)
        options.setdefault(CONF_SMART_EXTENSION_THRESHOLD, DEFAULT_SMART_EXTENSION_THRESHOLD)

        # Bump version and save
        self.hass.config_entries.async_update_entry(
            config_entry,
            data=data,
            options=options,
            version=self.VERSION,
        )
        _LOGGER.info("Migrated HA WashData entry to version %s", self.VERSION)
        return True

    def _get_schema(self, user_input: dict[str, Any] | None = None) -> vol.Schema:
        """Get the configuration schema."""
        return STEP_USER_DATA_SCHEMA

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=self._get_schema(), errors=errors
            )

        # Validate input
        try:
            # Basic validation
            if user_input[CONF_MIN_POWER] <= 0:
                errors[CONF_MIN_POWER] = "invalid_power"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"

        if errors:
            return self.async_show_form(
                step_id="user", data_schema=self._get_schema(user_input), errors=errors
            )

        # Store user input and proceed to profile creation
        self._user_input = user_input
        return await self.async_step_first_profile()

    async def async_step_first_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step to optionally create the first profile."""
        
        if user_input is not None:
            # Check if user wants to create a profile (if name is provided)
            profile_name = user_input.get("profile_name", "").strip()
            
            # Combine initial setup data with profile data if present
            data = dict(self._user_input)
            
            if profile_name:
                duration_mins = user_input.get("manual_duration")
                duration_sec = (duration_mins * 60.0) if duration_mins else None
                
                # Pass as special key to be handled in async_setup_entry
                data["initial_profile"] = {
                    "name": profile_name,
                    "avg_duration": duration_sec
                }

            return self.async_create_entry(title=data[CONF_NAME], data=data)

        # Schema for first profile
        schema = vol.Schema({
            vol.Optional("profile_name"): str,
            vol.Optional("manual_duration", default=120): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, max=480, unit_of_measurement="min", mode=selector.NumberSelectorMode.BOX
                )
            )
        })

        return self.async_show_form(
            step_id="first_profile",
            data_schema=schema
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
        self._suggested_values: dict[str, Any] | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["settings", "manage_cycles", "manage_profiles", "diagnostics"]
        )


    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage configuration settings (Basic Step)."""
        # Initialize or clear stored basic options
        if not hasattr(self, "_basic_options"):
            self._basic_options = {}

        if user_input is not None:
             # Check if user wants to edit advanced settings
            if user_input.get(CONF_SHOW_ADVANCED):
                # Store basic input to merge later
                self._basic_options = user_input
                # Remove the navigation flag from data storage
                self._basic_options.pop(CONF_SHOW_ADVANCED, None)
                return await self.async_step_advanced_settings()

            # Save Basic Settings Only
            # Merge with existing options to preserve settings not shown in this form
            user_input.pop(CONF_SHOW_ADVANCED, None)
            merged_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=merged_options)

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

        current_sensor = self._config_entry.options.get(
            CONF_POWER_SENSOR,
            self._config_entry.data.get(CONF_POWER_SENSOR, "")
        )

        def get_val(key, default):
            return self._config_entry.options.get(key, self._config_entry.data.get(key, default))

        # Base schema with essential options
        schema = {
            # --- Device Configuration (Top Priority) ---
            vol.Required(
                CONF_DEVICE_TYPE,
                default=get_val(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        selector.SelectOptionDict(value=k, label=v)
                        for k, v in DEVICE_TYPES.items()
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_POWER_SENSOR,
                default=current_sensor,
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor")
            ),
            
            vol.Optional(
                CONF_MIN_POWER,
                default=get_val(CONF_MIN_POWER, DEFAULT_MIN_POWER),
            ): vol.Coerce(float),
            vol.Optional(
                CONF_OFF_DELAY,
                default=get_val(CONF_OFF_DELAY, DEFAULT_OFF_DELAY),
            ): vol.Coerce(int),

            # --- Notification Settings ---
            vol.Optional(
                CONF_NOTIFY_SERVICE,
                default=get_val(CONF_NOTIFY_SERVICE, ""),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=notify_services,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_NOTIFY_EVENTS,
                default=list(get_val(CONF_NOTIFY_EVENTS, [])),
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
            vol.Optional(
                CONF_NOTIFY_BEFORE_END_MINUTES,
                default=get_val(CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=60, mode=selector.NumberSelectorMode.BOX)
            ),

            vol.Optional(CONF_SHOW_ADVANCED, default=False): bool,
        }
            
        return self.async_show_form(
            step_id="settings",
            data_schema=vol.Schema(schema),
            description_placeholders={
                "error": "",
            },
        )

    async def async_step_advanced_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage advanced configuration settings (Step 2)."""
        manager = self.hass.data[DOMAIN][self._config_entry.entry_id]
        suggestions = manager.suggestions if manager else {}

        if user_input is not None:
             # If "Apply Suggestions" checkbox was checked, merge suggested values into the input
            if user_input.get(CONF_APPLY_SUGGESTIONS):
                keys_to_apply = [
                    CONF_MIN_POWER,
                    CONF_OFF_DELAY,
                    CONF_WATCHDOG_INTERVAL,
                    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
                    CONF_PROFILE_MATCH_INTERVAL,
                    CONF_AUTO_LABEL_CONFIDENCE,
                    CONF_DURATION_TOLERANCE,
                    CONF_PROFILE_DURATION_TOLERANCE,
                    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                    CONF_AUTO_MERGE_GAP_SECONDS,
                ]
                
                # Create a copy of current options/input to work with
                updated_input = {**user_input}
                # Uncheck it so it doesn't stay checked via recursion (Use recursion only for apply suggestion visual confirmation if needed, but here we can just save)
                # Actually, standard behavior for 'Apply Suggestions' is usually to populate the form and let user review. 
                # But to keep it simple as requested ("moved to advanced options"), we can just apply and save?
                # The user probably expects to SEE the values. 
                # For a wizard, we can reload this step with values filled.
                
                updated_input[CONF_APPLY_SUGGESTIONS] = False
                
                applied_count = 0
                for key in keys_to_apply:
                    entry = suggestions.get(key) if isinstance(suggestions, dict) else None
                    if isinstance(entry, dict) and "value" in entry:
                        val = entry.get("value")
                        if key in (CONF_OFF_DELAY, CONF_WATCHDOG_INTERVAL, CONF_NO_UPDATE_ACTIVE_TIMEOUT, 
                                   CONF_PROFILE_MATCH_INTERVAL, CONF_AUTO_MERGE_GAP_SECONDS):
                            updated_input[key] = int(float(val))
                        else:
                            updated_input[key] = float(val)
                        applied_count += 1
                
                if applied_count > 0:
                    self._suggested_values = updated_input
                    return await self.async_step_advanced_settings(user_input=None)

            # Final Save
            final_options = {**self._config_entry.options, **self._basic_options, **user_input}
            final_options.pop(CONF_APPLY_SUGGESTIONS, None)
            return self.async_create_entry(title="", data=final_options)

        # Helper to get current value
        def get_val(key, default):
            # Prioritize suggested values (if "Apply Suggestions" triggered a reload)
            if self._suggested_values and key in self._suggested_values:
                return self._suggested_values[key]
            # Fallback to basic options (if coming from basic step)
            if key in self._basic_options:
                return self._basic_options[key]
            # Fallback to config options
            return self._config_entry.options.get(key, self._config_entry.data.get(key, default))
        
        # Format suggestions for description
        def _fmt_suggested(key: str) -> str:
            val = (suggestions.get(key) or {}).get("value") if isinstance(suggestions, dict) else None
            if val is None:
                return "—"
            try:
                return str(int(val)) if float(val).is_integer() else f"{float(val):.2f}"
            except Exception:
                return str(val)

        reason_lines: list[str] = []
        for key in [
            CONF_MIN_POWER,
            CONF_WATCHDOG_INTERVAL,
            CONF_NO_UPDATE_ACTIVE_TIMEOUT,
            CONF_PROFILE_MATCH_INTERVAL,
            CONF_DURATION_TOLERANCE,
            CONF_PROFILE_DURATION_TOLERANCE,
        ]:
            entry = suggestions.get(key) if isinstance(suggestions, dict) else None
            if isinstance(entry, dict) and entry.get("reason"):
                reason_lines.append(f"- {key}: {entry['reason']}")
        suggested_reason = "\n".join(reason_lines) if reason_lines else ""

        schema = {
             vol.Optional(CONF_APPLY_SUGGESTIONS, default=False): bool,

             # --- Detection Settings ---
            vol.Optional(
                CONF_START_DURATION_THRESHOLD,
                default=get_val(CONF_START_DURATION_THRESHOLD, DEFAULT_START_DURATION_THRESHOLD),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=60.0, step=0.5, unit_of_measurement="s", mode=selector.NumberSelectorMode.BOX)
            ),
           
            vol.Optional(
                CONF_INTERRUPTED_MIN_SECONDS,
                default=get_val(CONF_INTERRUPTED_MIN_SECONDS, DEFAULT_INTERRUPTED_MIN_SECONDS),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=900, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_COMPLETION_MIN_SECONDS,
                default=get_val(CONF_COMPLETION_MIN_SECONDS, DEFAULT_COMPLETION_MIN_SECONDS),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=3600, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_RUNNING_DEAD_ZONE,
                default=get_val(CONF_RUNNING_DEAD_ZONE, DEFAULT_RUNNING_DEAD_ZONE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=600, step=10, unit_of_measurement="s", mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_END_REPEAT_COUNT,
                default=get_val(CONF_END_REPEAT_COUNT, DEFAULT_END_REPEAT_COUNT),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=1, max=10, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_SMART_EXTENSION_THRESHOLD,
                default=get_val(CONF_SMART_EXTENSION_THRESHOLD, DEFAULT_SMART_EXTENSION_THRESHOLD),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode=selector.NumberSelectorMode.BOX)
            ),
            # --- Learning & Profiles ---
            vol.Optional(
                CONF_LEARNING_CONFIDENCE,
                default=get_val(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.01, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_AUTO_LABEL_CONFIDENCE,
                default=get_val(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.01, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_PROFILE_MATCH_INTERVAL,
                default=get_val(CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                default=get_val(CONF_PROFILE_MATCH_MIN_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.1, max=1.0, step=0.05, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                default=get_val(CONF_PROFILE_MATCH_MAX_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=1.0, max=3.0, step=0.1, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_DURATION_TOLERANCE,
                default=get_val(CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=0.5, step=0.01, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_SMOOTHING_WINDOW,
                default=get_val(CONF_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_PROFILE_DURATION_TOLERANCE,
                default=get_val(CONF_PROFILE_DURATION_TOLERANCE, DEFAULT_PROFILE_DURATION_TOLERANCE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=0.5, step=0.01, mode=selector.NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_AUTO_MERGE_LOOKBACK_HOURS,
                default=get_val(CONF_AUTO_MERGE_LOOKBACK_HOURS, DEFAULT_AUTO_MERGE_LOOKBACK_HOURS),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_AUTO_MERGE_GAP_SECONDS,
                default=get_val(CONF_AUTO_MERGE_GAP_SECONDS, DEFAULT_AUTO_MERGE_GAP_SECONDS),
            ): vol.Coerce(int),

            vol.Optional(
                CONF_ABRUPT_DROP_WATTS,
                default=get_val(CONF_ABRUPT_DROP_WATTS, DEFAULT_ABRUPT_DROP_WATTS),
            ): vol.Coerce(float),
            vol.Optional(
                CONF_ABRUPT_DROP_RATIO,
                default=get_val(CONF_ABRUPT_DROP_RATIO, DEFAULT_ABRUPT_DROP_RATIO),
            ): vol.Coerce(float),
            vol.Optional(
                CONF_ABRUPT_HIGH_LOAD_FACTOR,
                default=get_val(CONF_ABRUPT_HIGH_LOAD_FACTOR, DEFAULT_ABRUPT_HIGH_LOAD_FACTOR),
            ): vol.Coerce(float),

            vol.Optional(
                CONF_MAX_PAST_CYCLES,
                default=get_val(CONF_MAX_PAST_CYCLES, DEFAULT_MAX_PAST_CYCLES),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_MAX_FULL_TRACES_PER_PROFILE,
                default=get_val(CONF_MAX_FULL_TRACES_PER_PROFILE, DEFAULT_MAX_FULL_TRACES_PER_PROFILE),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_MAX_FULL_TRACES_UNLABELED,
                default=get_val(CONF_MAX_FULL_TRACES_UNLABELED, DEFAULT_MAX_FULL_TRACES_UNLABELED),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_WATCHDOG_INTERVAL,
                default=get_val(CONF_WATCHDOG_INTERVAL, DEFAULT_WATCHDOG_INTERVAL),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
                default=get_val(CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD, DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_PROGRESS_RESET_DELAY,
                default=get_val(CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY),
            ): vol.Coerce(int),
            
            vol.Optional(CONF_AUTO_MAINTENANCE, default=get_val(CONF_AUTO_MAINTENANCE, DEFAULT_AUTO_MAINTENANCE)): bool,
        }

        return self.async_show_form(
            step_id="advanced_settings",
            data_schema=vol.Schema(schema),
            description_placeholders={
                "error": "",
                "suggested": suggested_reason or "No suggestions available yet.",
                "suggested_min_power": _fmt_suggested(CONF_MIN_POWER),
                "suggested_off_delay": _fmt_suggested(CONF_OFF_DELAY),
                "suggested_watchdog_interval": _fmt_suggested(CONF_WATCHDOG_INTERVAL),
                "suggested_no_update_active_timeout": _fmt_suggested(CONF_NO_UPDATE_ACTIVE_TIMEOUT),
                "suggested_profile_match_interval": _fmt_suggested(CONF_PROFILE_MATCH_INTERVAL),
                "suggested_auto_label_confidence": _fmt_suggested(CONF_AUTO_LABEL_CONFIDENCE),
                "suggested_duration_tolerance": _fmt_suggested(CONF_DURATION_TOLERANCE),
                "suggested_profile_duration_tolerance": _fmt_suggested(CONF_PROFILE_DURATION_TOLERANCE),
                "suggested_profile_match_min_duration_ratio": _fmt_suggested(CONF_PROFILE_MATCH_MIN_DURATION_RATIO),
                "suggested_profile_match_max_duration_ratio": _fmt_suggested(CONF_PROFILE_MATCH_MAX_DURATION_RATIO),
                "suggested_auto_merge_gap_seconds": _fmt_suggested(CONF_AUTO_MERGE_GAP_SECONDS),
                "suggested_reason": suggested_reason,
            },
        )


    async def async_step_diagnostics(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Diagnostics submenu for maintenance actions."""
        if user_input is not None:
            choice = user_input["action"]
            if choice == "migrate_data":
                return await self.async_step_migrate_data()
            if choice == "wipe_history":
                return await self.async_step_wipe_history()
            if choice == "export_import":
                return await self.async_step_export_import()

        return self.async_show_form(
            step_id="diagnostics",
            data_schema=vol.Schema({
                vol.Required("action"): vol.In({
                    "migrate_data": "Migrate/compress stored data to latest format",
                    "wipe_history": "Wipe ALL data for this device (irreversible)",
                    "export_import": "Export/Import JSON with settings (copy/paste)"
                })
            })
        )

    async def async_step_export_import(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Export or import profile/cycle data via JSON copy/paste."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        export_payload = manager.profile_store.export_data(
            entry_data=dict(self.config_entry.data),
            entry_options=dict(self.config_entry.options),
        )
        export_str = json.dumps(export_payload, indent=2)

        errors: dict[str, str] = {}

        if user_input is not None:
            mode = user_input.get("mode", "export")
            payload_str = user_input.get("json_payload", "")

            # Always preserve existing options unless we explicitly update them
            options_to_return = dict(self._config_entry.options)

            if mode == "import":
                try:
                    payload = json.loads(payload_str)
                    config_updates = await manager.profile_store.async_import_data(payload)
                    
                    # Apply imported settings to config entry if present
                    entry_data = config_updates.get("entry_data", {})
                    entry_options = config_updates.get("entry_options", {})
                    
                    if entry_data or entry_options:
                        # Merge imported options with current data/options
                        new_data = {**self.config_entry.data}
                        new_options = {**self.config_entry.options}
                        
                        # Only update settings that exist in the import (don't overwrite power_sensor/name)
                        for key in [CONF_MIN_POWER, CONF_OFF_DELAY]:
                            if key in entry_data:
                                new_data[key] = entry_data[key]
                        
                        # Update all options from import
                        new_options.update(entry_options)
                        
                        self.hass.config_entries.async_update_entry(
                            self.config_entry,
                            data=new_data,
                            options=new_options,
                        )
                        _LOGGER.info("Applied imported settings to config entry")

                        # Return the merged options so the options flow itself doesn't revert them
                        options_to_return = dict(new_options)
                        
                except Exception:  # noqa: BLE001
                    errors["base"] = "import_failed"
                    # Re-show form with error
                    return self.async_show_form(
                        step_id="export_import",
                        data_schema=vol.Schema({
                            vol.Required("mode", default=mode): selector.SelectSelector(
                                selector.SelectSelectorConfig(
                                    options=[
                                        selector.SelectOptionDict(value="export", label="Export only"),
                                        selector.SelectOptionDict(value="import", label="Import from JSON"),
                                    ]
                                )
                            ),
                            vol.Optional("json_payload", default=payload_str): selector.TextSelector(
                                selector.TextSelectorConfig(multiline=True)
                            ),
                        }),
                        errors=errors,
                    )

            return self.async_create_entry(title="", data=options_to_return)

        return self.async_show_form(
            step_id="export_import",
            data_schema=vol.Schema({
                vol.Required("mode", default="export"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(value="export", label="Export only"),
                            selector.SelectOptionDict(value="import", label="Import from JSON"),
                        ]
                    )
                ),
                vol.Optional("json_payload", default=export_str): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=True)
                ),
            }),
        )

    async def async_step_manage_cycles(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage cycles submenu."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        
        # Build recent cycles list
        recent_cycles = store._data.get("past_cycles", [])[-8:]
        recent_lines = []
        for c in reversed(recent_cycles):
            start = c["start_time"].split(".")[0].replace("T", " ")
            duration_min = int(c["duration"] / 60)
            prof = c.get("profile_name") or "Unlabeled"
            status = c.get("status", "completed")
            status_icon = "✓" if status in ("completed", "force_stopped") else "⚠" if status == "resumed" else "✗"
            recent_lines.append(f"{status_icon} {start} - {duration_min}m - {prof}")
        recent_text = "\n".join(recent_lines) if recent_lines else "No cycles recorded yet."

        if user_input is not None:
            action = user_input["action"]
            if action == "label_cycle":
                return await self.async_step_select_cycle_to_label()
            elif action == "auto_label":
                return await self.async_step_auto_label_cycles()
            elif action == "delete_cycle":
                return await self.async_step_select_cycle_to_delete()
            elif action == "post_process":
                return await self.async_step_post_process()

        return self.async_show_form(
            step_id="manage_cycles",
            data_schema=vol.Schema({
                vol.Required("action"): vol.In({
                    "label_cycle": "🏷️ Label a Cycle",
                    "auto_label": "🤖 Auto-Label History",
                    "delete_cycle": "❌ Delete a Cycle",
                    "post_process": "🧹 Post-Process / Merge",
                })
            }),
            description_placeholders={"recent_cycles": recent_text}
        )

    async def async_step_manage_profiles(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage profiles submenu."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        profiles = store.list_profiles()
        
        # Build profile summary
        summary_lines = []
        for p in profiles:
            count = p["cycle_count"]
            avg = int(p["avg_duration"]/60) if p["avg_duration"] else 0
            summary_lines.append(f"- **{p['name']}**: {count} cycles, {avg}m avg")
        summary_text = "\n".join(summary_lines) if summary_lines else "No profiles created yet."

        if user_input is not None:
            action = user_input["action"]
            if action == "create_profile":
                return await self.async_step_create_profile()
            elif action == "edit_profile":
                return await self.async_step_edit_profile()
            elif action == "delete_profile":
                return await self.async_step_delete_profile_select()
            elif action == "profile_stats":
                return await self.async_step_profile_stats()

        return self.async_show_form(
            step_id="manage_profiles",
            data_schema=vol.Schema({
                vol.Required("action"): vol.In({
                    "create_profile": "➕ Create New Profile",
                    "edit_profile": "✏️ Edit/Rename Profile",
                    "delete_profile": "🗑️ Delete Profile",
                    "profile_stats": "📊 Profile Statistics",
                })
            }),
            description_placeholders={"profile_summary": summary_text}
        )

    async def async_step_profile_stats(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show detailed profile statistics with graphs."""
        if user_input is not None:
             # Back to manage profiles
             return await self.async_step_manage_profiles()

        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        
        # Ensure stats directory exists
        stats_dir = self.hass.config.path("www", "ha_washdata", "profiles")
        await self.hass.async_add_executor_job(
            lambda: os.makedirs(stats_dir, exist_ok=True)
        )
        
        from homeassistant.util import slugify
        
        profiles = store.list_profiles()
        sections = []
        ts = int(time.time())
        
        # Get all cycles to find last run
        cycles = store._data.get("past_cycles", [])
        
        for p in profiles:
            name = p["name"]
            
            # FORCE REFRESH: Rebuild envelope to ensure data is fresh
            # This calculates energy, consistency, etc.
            store.rebuild_envelope(name)
            
            safe_name = slugify(name)
            count = p["cycle_count"]
            avg = int(p["avg_duration"]/60) if p["avg_duration"] else 0
            mn = int(p["min_duration"]/60) if p.get("min_duration") else 0
            mx = int(p["max_duration"]/60) if p.get("max_duration") else 0
            
            # Get envelope for advanced stats
            envelope = store.get_envelope(name)
            kwh = f"{envelope.get('avg_energy', 0):.2f}" if envelope else "-"
            std_dev = envelope.get('duration_std_dev', 0) if envelope else 0
            consistency = f"±{int(std_dev/60)}m" if std_dev > 0 else "-"
            
            # Find last run
            last_run = "-"
            p_cycles = [c for c in cycles if c.get("profile_name") == name]
            if p_cycles:
                last_c = max(p_cycles, key=lambda x: x["start_time"])
                dt = last_c["start_time"].split("T")[0]
                last_run = dt
            
            # Generate and Write SVG
            svg_content = store.generate_profile_svg(name)
            graph_markdown = ""
            if svg_content:
                file_path = f"{stats_dir}/profile_{safe_name}.svg"
                def write_svg():
                    with open(file_path, "w") as f:
                        f.write(svg_content)
                await self.hass.async_add_executor_job(write_svg)
                graph_markdown = f"![{name}](/local/ha_washdata/profiles/profile_{safe_name}.svg?v={ts})"

            # Build Per-Profile Section
            # Headers: Count | Avg | Min | Max | Energy | Consistency | Last Run
            table_header = "| Count | Avg | Min | Max | Energy | Consist. | Last Run |"
            table_sep = "| --- | --- | --- | --- | --- | --- | --- |"
            table_row = f"| {count} | {avg}m | {mn}m | {mx}m | {kwh} kWh | {consistency} | {last_run} |"
            
            legend = "> **Graph Legend**: The blue band represents the minimum and maximum power draw range observed. The line shows the average power curve."
            
            section = f"## {name}\n{table_header}\n{table_sep}\n{table_row}\n\n{graph_markdown}\n\n{legend}"
            sections.append(section)
            
        content = "\n\n---\n\n".join(sections) if sections else "No profiles found."

        return self.async_show_form(
            step_id="profile_stats",
            data_schema=vol.Schema({}),
            # Key 'stats_table' must match the key in translations/strings (description)
            description_placeholders={"stats_table": content}
        )

    async def async_step_create_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Create a new profile."""
        errors = {}
        
        if user_input is not None:
            name = user_input["profile_name"].strip()
            reference_cycle = user_input.get("reference_cycle")
            
            if not name:
                errors["profile_name"] = "empty_name"
            else:
                manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
                manual_duration_mins = user_input.get("manual_duration")
                avg_duration = None
                if manual_duration_mins and float(manual_duration_mins) > 0:
                    avg_duration = float(manual_duration_mins) * 60.0

                try:
                    await manager.profile_store.create_profile_standalone(
                        name, 
                        reference_cycle if reference_cycle != "none" else None,
                        avg_duration=avg_duration
                    )
                    manager._notify_update()
                    return self.async_create_entry(title="", data=dict(self._config_entry.options))
                except ValueError as e:
                    errors["base"] = "profile_exists"
        
        # Build cycle options for reference
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        cycles = store._data.get("past_cycles", [])[-20:]
        
        cycle_options = [selector.SelectOptionDict(value="none", label="(No reference cycle)")]
        for c in reversed(cycles):
            start = c["start_time"].split(".")[0].replace("T", " ")
            duration_min = int(c['duration']/60)
            prof = c.get("profile_name") or "Unlabeled"
            label = f"{start} - {duration_min}m - {prof}"
            cycle_options.append(selector.SelectOptionDict(value=c["id"], label=label))
        
        return self.async_show_form(
            step_id="create_profile",
            data_schema=vol.Schema({
                vol.Required("profile_name"): str,
                vol.Optional("reference_cycle", default="none"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=cycle_options,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),
                vol.Optional("manual_duration"): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, 
                        max=480, 
                        mode=selector.NumberSelectorMode.BOX,
                        unit_of_measurement="min"
                    )
                )
            }),
            errors=errors
        )

    async def async_step_edit_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select profile to edit/rename."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        profiles = store.list_profiles()
        
        if not profiles:
            return self.async_abort(reason="no_profiles_found")
        
        if user_input is not None:
            self._selected_profile = user_input["profile"]
            return await self.async_step_rename_profile()
        
        # Build profile options
        options = []
        for p in profiles:
            count = p["cycle_count"]
            duration_min = int(p["avg_duration"] / 60) if p["avg_duration"] else 0
            label = f"{p['name']} ({count} cycles, ~{duration_min}m avg)"
            options.append(selector.SelectOptionDict(value=p["name"], label=label))
        
        return self.async_show_form(
            step_id="edit_profile",
            data_schema=vol.Schema({
                vol.Required("profile"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                )
            })
        )

    async def async_step_rename_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Edit profile settings (Name and Duration)."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        errors = {}
        
        # Get current profile data
        profiles = manager.profile_store.list_profiles()
        current_data = next((p for p in profiles if p["name"] == self._selected_profile), None)
        current_duration_mins = int(current_data["avg_duration"] / 60) if current_data and current_data["avg_duration"] else 0
        
        if user_input is not None:
            new_name = user_input["new_name"].strip()
            manual_duration_mins = user_input.get("manual_duration")
            
            if not new_name:
                errors["new_name"] = "empty_name"
            else:
                avg_duration = None
                if manual_duration_mins is not None and manual_duration_mins > 0:
                    avg_duration = float(manual_duration_mins) * 60.0

                try:
                    await manager.profile_store.update_profile(
                        self._selected_profile, 
                        new_name,
                        avg_duration=avg_duration
                    )
                    manager._notify_update()
                    return self.async_create_entry(title="", data=dict(self._config_entry.options))
                except ValueError as e:
                    errors["base"] = "rename_failed"
        
        return self.async_show_form(
            step_id="rename_profile",
            data_schema=vol.Schema({
                vol.Required("new_name", default=self._selected_profile): str,
                vol.Optional("manual_duration", default=current_duration_mins): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=480, unit_of_measurement="min", mode=selector.NumberSelectorMode.BOX
                    )
                )
            }),
            errors=errors,
            description_placeholders={
                "current_name": self._selected_profile
            }
        )

    async def async_step_delete_profile_select(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select profile to delete."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        profiles = store.list_profiles()
        
        if not profiles:
            return self.async_abort(reason="no_profiles_found")
        
        if user_input is not None:
            self._selected_profile = user_input["profile"]
            return await self.async_step_delete_profile_confirm()
        
        # Build profile options
        options = []
        for p in profiles:
            count = p["cycle_count"]
            duration_min = int(p["avg_duration"] / 60) if p["avg_duration"] else 0
            label = f"{p['name']} ({count} cycles, ~{duration_min}m avg)"
            options.append(selector.SelectOptionDict(value=p["name"], label=label))
        
        return self.async_show_form(
            step_id="delete_profile_select",
            data_schema=vol.Schema({
                vol.Required("profile"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                )
            })
        )

    async def async_step_delete_profile_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm profile deletion."""
        if user_input is not None:
            unlabel = user_input["unlabel_cycles"]
            manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
            count = await manager.profile_store.delete_profile(self._selected_profile, unlabel)
            manager._notify_update()
            return self.async_create_entry(title="", data=dict(self._config_entry.options))
        
        return self.async_show_form(
            step_id="delete_profile_confirm",
            data_schema=vol.Schema({
                vol.Required("unlabel_cycles", default=True): bool
            }),
            description_placeholders={
                "profile_name": self._selected_profile,
                "warning": "⚠️ This will permanently delete the profile."
            }
        )

    async def async_step_auto_label_cycles(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Auto-label unlabeled cycles retroactively."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        
        # Check if there are any profiles to match against
        profiles = store.list_profiles()
        if not profiles:
            return self.async_abort(reason="no_profiles_for_matching")
        
        # Count unlabeled cycles
        unlabeled_count = sum(1 for c in store._data.get("past_cycles", []) if not c.get("profile_name"))
        if unlabeled_count == 0:
            return self.async_abort(reason="no_unlabeled_cycles")
        
        if user_input is not None:
            threshold = user_input["confidence_threshold"]
            stats = await store.auto_label_unlabeled_cycles(threshold)
            manager._notify_update()
            return self.async_create_entry(
                title="",
                data=dict(self._config_entry.options),
            )
        
        return self.async_show_form(
            step_id="auto_label_cycles",
            data_schema=vol.Schema({
                vol.Required("confidence_threshold", default=0.70): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.50,
                        max=0.95,
                        step=0.05,
                        mode=selector.NumberSelectorMode.BOX
                    )
                )
            }),
            description_placeholders={
                "info": f"Found {unlabeled_count} unlabeled cycles. Profiles: {', '.join(p['name'] for p in profiles)}"
            }
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
            prof = c.get("profile_name") or "Unlabeled"
            status = c.get("status", "completed")
            # ✓ = completed/force_stopped (natural end), ⚠ = resumed, ✗ = interrupted (user stopped)
            status_icon = "✓" if status in ("completed", "force_stopped") else "⚠" if status == "resumed" else "✗"
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
            prof = c.get("profile_name") or "Unlabeled"
            status = c.get("status", "completed")
            # ✓ = completed/force_stopped (natural end), ⚠ = resumed, ✗ = interrupted (user stopped)
            status_icon = "✓" if status in ("completed", "force_stopped") else "⚠" if status == "resumed" else "✗"
            label = f"[{status_icon}] {start} - {duration_min}m - {prof}"
            options.append(selector.SelectOptionDict(value=c["id"], label=label))
            
        if not options:
            return self.async_abort(reason="no_cycles_found")

        if user_input is not None:
            cycle_id = user_input["cycle_id"]
            await manager.profile_store.delete_cycle(cycle_id)
            # await manager.profile_store.async_save() # Handled inside delete_cycle now
            manager._notify_update()
            return self.async_create_entry(title="", data=dict(self._config_entry.options))

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
        """Assign profile to the selected cycle."""
        errors = {}
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        
        if user_input is not None:
            profile_choice = user_input["profile_name"]
            
            # Handle "create new" option
            if profile_choice == "__create_new__":
                new_name = user_input.get("new_profile_name", "").strip()
                if not new_name:
                    errors["new_profile_name"] = "empty_name"
                else:
                    try:
                        # Create profile from this cycle
                        await store.create_profile(new_name, self._selected_cycle_id)
                        manager._notify_update()
                        return self.async_create_entry(title="", data=dict(self._config_entry.options))
                    except ValueError:
                        errors["base"] = "profile_exists"
            elif profile_choice == "__remove_label__":
                # Remove label from cycle
                await store.assign_profile_to_cycle(self._selected_cycle_id, None)
                manager._notify_update()
                return self.async_create_entry(title="", data=dict(self._config_entry.options))
            else:
                # Assign existing profile
                try:
                    await store.assign_profile_to_cycle(self._selected_cycle_id, profile_choice)
                    manager._notify_update()
                    return self.async_create_entry(title="", data=dict(self._config_entry.options))
                except ValueError as e:
                    errors["base"] = "assignment_failed"
        
        # Build profile dropdown options
        profiles = store.list_profiles()
        profile_options = [
            selector.SelectOptionDict(value="__create_new__", label="➕ Create New Profile"),
            selector.SelectOptionDict(value="__remove_label__", label="🗑️ Remove Label"),
        ]
        for p in profiles:
            count = p["cycle_count"]
            duration_min = int(p["avg_duration"] / 60) if p["avg_duration"] else 0
            label = f"{p['name']} ({count} cycles, ~{duration_min}m)"
            profile_options.append(selector.SelectOptionDict(value=p["name"], label=label))
        
        # Get cycle info for display
        cycle = next((c for c in store._data.get("past_cycles", []) if c["id"] == self._selected_cycle_id), None)
        cycle_info = ""
        if cycle:
            start = cycle["start_time"].split(".")[0].replace("T", " ")
            duration_min = int(cycle['duration']/60)
            current_label = cycle.get("profile") or "Unlabeled"
            cycle_info = f"Cycle: {start}, {duration_min}m, Current: {current_label}"
        
        schema = {
            vol.Required("profile_name", default="__create_new__" if not profiles else profiles[0]["name"]): 
                selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=profile_options,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                )
        }
        
        # Add new profile name field (shown when "__create_new__" selected)
        schema[vol.Optional("new_profile_name")] = str
        
        return self.async_show_form(
            step_id="label_cycle",
            data_schema=vol.Schema(schema),
            errors=errors,
            description_placeholders={"cycle_info": cycle_info}
        )

    async def async_step_post_process(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle post-processing options."""
        if user_input is not None:
             choice = user_input["time_range"]
             manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
             
             gap_seconds = self.config_entry.options.get(
                 CONF_AUTO_MERGE_GAP_SECONDS, DEFAULT_AUTO_MERGE_GAP_SECONDS
             )
             
             hours = 999999 if choice >= 999999 else int(choice)
             
             # Use async_run_maintenance to ensure envelopes are rebuilt after merging
             stats = await manager.profile_store.async_run_maintenance(
                 lookback_hours=hours, 
                 gap_seconds=gap_seconds
             )
             count = stats.get("merged_cycles", 0)
                 
             return self.async_create_entry(
                 title="",
                 data=dict(self._config_entry.options),
                 description_placeholders={"count": str(count)}
             )

        return self.async_show_form(
            step_id="post_process",
            data_schema=vol.Schema({
                vol.Required("time_range", default=24): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=1, max=9999, unit_of_measurement="h", mode=selector.NumberSelectorMode.BOX)
                )
            }),
            description_placeholders={"info": "Enter number of past hours to process (or use 999999 for all)"}
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
                 data=dict(self._config_entry.options),
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
             manager._notify_update()
             
             return self.async_create_entry(
                 title="",
                 data=dict(self._config_entry.options),
                 description_placeholders={"info": "History cleared"}
             )

        return self.async_show_form(
            step_id="wipe_history",
            data_schema=vol.Schema({}),
            description_placeholders={"warning": "⚠️ This will permanently delete ALL stored cycles and profiles for this device. This cannot be undone!"}
        )
