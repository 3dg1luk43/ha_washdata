"""Config flow for HA WashData integration."""
# pylint: disable=too-many-lines

from __future__ import annotations

import json
import logging
import os
import time
import base64
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.util import slugify
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    CONF_POWER_SENSOR,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_SAMPLING_INTERVAL,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_EVENTS,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_SMOOTHING_WINDOW,
    CONF_START_DURATION_THRESHOLD,
    CONF_DEVICE_TYPE,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_APPLY_SUGGESTIONS,
    CONF_SHOW_ADVANCED,
    CONF_PROGRESS_RESET_DELAY,
    CONF_DURATION_TOLERANCE,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_EXPOSE_DEBUG_ENTITIES,
    CONF_SAVE_DEBUG_TRACES,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_AUTO_MAINTENANCE,
    CONF_WATCHDOG_INTERVAL,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_RUNNING_DEAD_ZONE,
    CONF_END_REPEAT_COUNT,
    CONF_START_ENERGY_THRESHOLD,
    CONF_END_ENERGY_THRESHOLD,
    CONF_EXTERNAL_END_TRIGGER_ENABLED,
    CONF_EXTERNAL_END_TRIGGER,
    NOTIFY_EVENT_START,
    NOTIFY_EVENT_FINISH,
    DEFAULT_NAME,
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_START_DURATION_THRESHOLD,
    DEFAULT_START_ENERGY_THRESHOLD,
    DEFAULT_END_ENERGY_THRESHOLD,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_PROFILE_DURATION_TOLERANCE,
    DEVICE_TYPES,
    DEFAULT_PROGRESS_RESET_DELAY,
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_AUTO_MAINTENANCE,
    DEFAULT_WATCHDOG_INTERVAL,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_RUNNING_DEAD_ZONE,
    DEFAULT_END_REPEAT_COUNT,
    DEFAULT_MIN_OFF_GAP_BY_DEVICE,
    DEFAULT_MIN_OFF_GAP,
    CONF_MIN_OFF_GAP,
    DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE,
    DEVICE_COMPLETION_THRESHOLDS,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_PROFILE_UNMATCH_THRESHOLD,
    DEFAULT_PROFILE_MATCH_THRESHOLD,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
    DEFAULT_SAMPLING_INTERVAL,
    CONF_NOTIFY_TITLE,
    CONF_NOTIFY_ICON,
    CONF_NOTIFY_START_MESSAGE,
    CONF_NOTIFY_FINISH_MESSAGE,
    CONF_NOTIFY_PRE_COMPLETE_MESSAGE,
    DEFAULT_NOTIFY_TITLE,
    DEFAULT_NOTIFY_START_MESSAGE,
    DEFAULT_NOTIFY_FINISH_MESSAGE,
    DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_OFF_DELAY_BY_DEVICE,
    DEFAULT_SAMPLING_INTERVAL_BY_DEVICE,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE,
)
from .profile_store import profile_sort_key


_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
        vol.Required(
            CONF_DEVICE_TYPE, default=DEFAULT_DEVICE_TYPE
        ): selector.SelectSelector(
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


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):  # pylint: disable=abstract-method
    """Handle a config flow for HA WashData."""

    VERSION = 3

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

        # Validate input
        try:
            # Basic validation
            if user_input[CONF_MIN_POWER] <= 0:
                errors[CONF_MIN_POWER] = "invalid_power"
        except Exception:  # pylint: disable=broad-exception-caught
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
        self, user_input: dict[str, Any] | None = None  # pylint: disable=unused-argument
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
                    "avg_duration": duration_sec,
                }

            return self.async_create_entry(title=data[CONF_NAME], data=data)

        # Schema for first profile
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
        self._selected_profile: str | None = None
        self._suggested_values: dict[str, Any] | None = None
        self._suggested_values: dict[str, Any] | None = None
        self._basic_options: dict[str, Any] = {}
        self._editor_action: str | None = None
        self._editor_selected_ids: list[str] = []
        self._editor_split_gap: int = 900

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None  # pylint: disable=unused-argument
    ) -> FlowResult:
        """Manage the options."""
        # Check for pending feedbacks to show count
        manager = self.hass.data[DOMAIN][self._config_entry.entry_id]
        pending_count = len(manager.profile_store.get_pending_feedback())
        
        feedback_label = "Review Learning Feedbacks"
        if pending_count > 0:
            feedback_label = f"({pending_count}) Review Learning Feedbacks"

        menu_options = {
            "settings": "Settings",
            "manage_cycles": "Manage Cycles",
            "manage_profiles": "Manage Profiles",
            "record_cycle": "Record Cycle (Manual)",
            "learning_feedbacks": feedback_label,
            "diagnostics": "Diagnostics & Maintenance",
        }
        
        # User requested feedback review to be near the bottom (above diagnostics)

        return self.async_show_menu(
            step_id="init",
            menu_options=menu_options,
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
            # Safe merge: Start with data (legacy), override with options, then user input
            # This prevents losing settings if options was empty (legacy migration)
            merged_options = {
                **self.config_entry.data,
                **self.config_entry.options,
                **user_input,
            }
            return self.async_create_entry(title="", data=merged_options)

        # Populate notify services
        notify_services = []
        services = self.hass.services.async_services()
        for service in services.get("notify", {}):
            notify_services.append(f"notify.{service}")
        notify_services.sort()

        # Ensure current value is in the list (so it doesn't vanish)
        current_notify = self.config_entry.options.get(
            CONF_NOTIFY_SERVICE, self.config_entry.data.get(CONF_NOTIFY_SERVICE, "")
        )
        if current_notify and current_notify not in notify_services:
            notify_services.append(current_notify)

        current_sensor = self.config_entry.options.get(
            CONF_POWER_SENSOR, self.config_entry.data.get(CONF_POWER_SENSOR, "")
        )

        def get_val(key, default):
            return self.config_entry.options.get(
                key, self.config_entry.data.get(key, default)
            )

        # Resolve Device Type for Defaults
        current_device_type = self.config_entry.options.get(
            CONF_DEVICE_TYPE,
            self.config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
        )

        # Specialized defaults for Device Type
        default_off_delay = DEFAULT_OFF_DELAY_BY_DEVICE.get(
            current_device_type, DEFAULT_OFF_DELAY
        )

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
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor")),
            # --- Power Thresholds ---
            vol.Optional(
                CONF_MIN_POWER,
                default=get_val(CONF_MIN_POWER, DEFAULT_MIN_POWER),
            ): vol.Coerce(float),

            vol.Optional(
                CONF_OFF_DELAY,
                default=get_val(CONF_OFF_DELAY, default_off_delay),
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
                        selector.SelectOptionDict(
                            value=NOTIFY_EVENT_START, label="Cycle Start"
                        ),
                        selector.SelectOptionDict(
                            value=NOTIFY_EVENT_FINISH, label="Cycle Finish"
                        ),
                    ],
                    multiple=True,
                    mode=selector.SelectSelectorMode.LIST,
                )
            ),
            vol.Optional(
                CONF_NOTIFY_BEFORE_END_MINUTES,
                default=get_val(
                    CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, max=60, mode=selector.NumberSelectorMode.BOX
                )
            ),
             vol.Optional(
                 CONF_NOTIFY_TITLE,
                 default=get_val(CONF_NOTIFY_TITLE, DEFAULT_NOTIFY_TITLE),
             ): selector.TextSelector(),
             vol.Optional(
                 CONF_NOTIFY_ICON,
                 default=get_val(CONF_NOTIFY_ICON, ""),
             ): selector.IconSelector(),
             vol.Optional(
                 CONF_NOTIFY_START_MESSAGE,
                 default=get_val(CONF_NOTIFY_START_MESSAGE, DEFAULT_NOTIFY_START_MESSAGE),
             ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),
             vol.Optional(
                 CONF_NOTIFY_FINISH_MESSAGE,
                 default=get_val(CONF_NOTIFY_FINISH_MESSAGE, DEFAULT_NOTIFY_FINISH_MESSAGE),
             ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),
             vol.Optional(
                 CONF_NOTIFY_PRE_COMPLETE_MESSAGE,
                 default=get_val(CONF_NOTIFY_PRE_COMPLETE_MESSAGE, DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE),
             ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),

            vol.Optional(CONF_SHOW_ADVANCED, default=False): bool,
        }

        return self.async_show_form(
            step_id="settings",
            data_schema=vol.Schema(schema),
            description_placeholders={
                "error": "",
                "device": "{device}",
                "duration": "{duration}",
                "program": "{program}",
                "minutes": "{minutes}",
            },
        )

    async def async_step_advanced_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage advanced configuration settings (Step 2)."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        suggestions = manager.suggestions if manager else {}

        if user_input is not None:
            # If "Apply Suggestions" checkbox was checked, merge suggested values into the input
            if user_input.get(CONF_APPLY_SUGGESTIONS):
                keys_to_apply = [
                    CONF_MIN_POWER,
                    CONF_OFF_DELAY,
                    CONF_WATCHDOG_INTERVAL,
                    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
                    CONF_SAMPLING_INTERVAL,
                    CONF_PROFILE_MATCH_INTERVAL,
                    CONF_AUTO_LABEL_CONFIDENCE,
                    CONF_DURATION_TOLERANCE,
                    CONF_PROFILE_DURATION_TOLERANCE,
                    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                    CONF_MIN_OFF_GAP,
                ]

                updated_input = {**user_input}

                updated_input[CONF_APPLY_SUGGESTIONS] = False

                applied_count = 0
                for key in keys_to_apply:
                    entry = (
                        suggestions.get(key) if isinstance(suggestions, dict) else None
                    )
                    if isinstance(entry, dict) and "value" in entry:
                        val = entry.get("value")
                        if key in (
                            CONF_OFF_DELAY,
                            CONF_WATCHDOG_INTERVAL,
                            CONF_NO_UPDATE_ACTIVE_TIMEOUT,
                            CONF_PROFILE_MATCH_INTERVAL,

                            CONF_MIN_OFF_GAP,
                        ):
                            updated_input[key] = int(float(val))
                        else:
                            updated_input[key] = float(val)
                        applied_count += 1

                if applied_count > 0:
                    self._suggested_values = updated_input
                    return await self.async_step_advanced_settings(user_input=None)

            # Ensure clearing CONF_EXTERNAL_END_TRIGGER translates to None/Empty
            # Handle missing key, empty list [], empty string "", or None
            _trigger_val = user_input.get(CONF_EXTERNAL_END_TRIGGER)
            if not _trigger_val:
                user_input[CONF_EXTERNAL_END_TRIGGER] = None

            # Final Save
            final_options = {
                **self.config_entry.data,
                **self.config_entry.options,
                **self._basic_options,
                **user_input,
            }
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
            return self.config_entry.options.get(
                key, self.config_entry.data.get(key, default)
            )

        # Format suggestions for description
        def _fmt_suggested(key: str) -> str:
            val = (
                (suggestions.get(key) or {}).get("value")
                if isinstance(suggestions, dict)
                else None
            )
            if val is None:
                return "—"
            try:
                return str(int(val)) if float(val).is_integer() else f"{float(val):.2f}"
            except Exception:  # pylint: disable=broad-exception-caught
                return str(val)

        reason_lines: list[str] = []
        for key in [
            CONF_MIN_POWER,
            CONF_WATCHDOG_INTERVAL,
            CONF_NO_UPDATE_ACTIVE_TIMEOUT,
            CONF_SAMPLING_INTERVAL,
            CONF_PROFILE_MATCH_INTERVAL,
            CONF_DURATION_TOLERANCE,
            CONF_PROFILE_DURATION_TOLERANCE,
        ]:
            entry = suggestions.get(key) if isinstance(suggestions, dict) else None
            if isinstance(entry, dict) and entry.get("reason"):
                reason_lines.append(f"- {key}: {entry['reason']}")
        suggested_reason = "\n".join(reason_lines) if reason_lines else ""

        # Resolve Device Type for Defaults
        current_device_type = self.config_entry.options.get(
            CONF_DEVICE_TYPE,
            self.config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
        )
        if CONF_DEVICE_TYPE in self._basic_options:
            current_device_type = self._basic_options[CONF_DEVICE_TYPE]

        # Device-specific defaults
        _default_min_off_gap = DEFAULT_MIN_OFF_GAP_BY_DEVICE.get(
            current_device_type, DEFAULT_MIN_OFF_GAP
        )
        default_start_energy = DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE.get(
            current_device_type, DEFAULT_START_ENERGY_THRESHOLD
        )
        default_completion_min = DEVICE_COMPLETION_THRESHOLDS.get(
            current_device_type, DEFAULT_COMPLETION_MIN_SECONDS
        )
        
        default_sampling = DEFAULT_SAMPLING_INTERVAL_BY_DEVICE.get(
            current_device_type, DEFAULT_SAMPLING_INTERVAL
        )

        # Specialized defaults for Device Type
        default_no_update_timeout = DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE.get(
            current_device_type, DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT
        )
        
        default_min_duration_ratio = DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE.get(
            current_device_type, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO
        )

        schema = {
            vol.Optional(CONF_APPLY_SUGGESTIONS, default=False): bool,
            # --- Detection Settings ---
            vol.Optional(
                CONF_START_DURATION_THRESHOLD,
                default=get_val(
                    CONF_START_DURATION_THRESHOLD, DEFAULT_START_DURATION_THRESHOLD
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=60.0,
                    step=0.5,
                    unit_of_measurement="s",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_START_ENERGY_THRESHOLD,
                default=get_val(CONF_START_ENERGY_THRESHOLD, default_start_energy),
            ): vol.Coerce(float),

            vol.Optional(
                CONF_COMPLETION_MIN_SECONDS,
                default=get_val(CONF_COMPLETION_MIN_SECONDS, default_completion_min),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, max=3600, mode=selector.NumberSelectorMode.BOX
                )
            ),
            # --- Advanced Power Thresholds ---
            vol.Optional(
                CONF_START_THRESHOLD_W,
                default=get_val(
                    CONF_START_THRESHOLD_W,
                    float(get_val(CONF_MIN_POWER, DEFAULT_MIN_POWER)) + 1.0,
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=100.0,
                    step=0.5,
                    unit_of_measurement="W",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_STOP_THRESHOLD_W,
                default=get_val(
                    CONF_STOP_THRESHOLD_W,
                    max(0.0, float(get_val(CONF_MIN_POWER, DEFAULT_MIN_POWER)) - 0.5),
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=100.0,
                    step=0.5,
                    unit_of_measurement="W",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            # --- Deferral Logic ---
            vol.Optional(
                CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                default=get_val(
                    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                    default_min_duration_ratio,
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.1,
                    max=1.0,
                    step=0.01,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_END_ENERGY_THRESHOLD,
                default=get_val(
                CONF_END_ENERGY_THRESHOLD, DEFAULT_END_ENERGY_THRESHOLD
                ),
            ): vol.Coerce(float),
            vol.Optional(
                CONF_RUNNING_DEAD_ZONE,
                default=get_val(CONF_RUNNING_DEAD_ZONE, DEFAULT_RUNNING_DEAD_ZONE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=600,
                    step=10,
                    unit_of_measurement="s",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_END_REPEAT_COUNT,
                default=get_val(CONF_END_REPEAT_COUNT, DEFAULT_END_REPEAT_COUNT),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1, max=10, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_MIN_OFF_GAP,
                default=get_val(CONF_MIN_OFF_GAP, _default_min_off_gap),
            ): vol.Coerce(int),

             vol.Optional(
                CONF_SAMPLING_INTERVAL,
                default=get_val(CONF_SAMPLING_INTERVAL, default_sampling),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1.0,
                    max=60.0,
                    step=0.5,
                    unit_of_measurement="s",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            # --- Learning & Profiles ---

            vol.Optional(
                CONF_PROFILE_MATCH_INTERVAL,
                default=get_val(
                    CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
                ),
            ): vol.Coerce(int),

            vol.Optional(
                CONF_PROFILE_MATCH_THRESHOLD,
                default=get_val(
                    CONF_PROFILE_MATCH_THRESHOLD, DEFAULT_PROFILE_MATCH_THRESHOLD
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.1, max=1.0, step=0.05, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_PROFILE_UNMATCH_THRESHOLD,
                default=get_val(
                    CONF_PROFILE_UNMATCH_THRESHOLD, DEFAULT_PROFILE_UNMATCH_THRESHOLD
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.1, max=1.0, step=0.05, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_DURATION_TOLERANCE,
                default=get_val(CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0, max=0.5, step=0.01, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_SMOOTHING_WINDOW,
                default=get_val(CONF_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_PROFILE_DURATION_TOLERANCE,
                default=get_val(
                    CONF_PROFILE_DURATION_TOLERANCE, DEFAULT_PROFILE_DURATION_TOLERANCE
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0, max=0.5, step=0.01, mode=selector.NumberSelectorMode.BOX
                )
            ),



            vol.Optional(
                CONF_WATCHDOG_INTERVAL,
                default=get_val(CONF_WATCHDOG_INTERVAL, DEFAULT_WATCHDOG_INTERVAL),
            ): vol.Coerce(int),

            vol.Optional(
                 CONF_NO_UPDATE_ACTIVE_TIMEOUT,
                 default=get_val(CONF_NO_UPDATE_ACTIVE_TIMEOUT, default_no_update_timeout),
            ): vol.Coerce(int),

            vol.Optional(
                CONF_PROGRESS_RESET_DELAY,
                default=get_val(
                    CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY
                ),
            ): vol.Coerce(int),
            vol.Optional(
                CONF_AUTO_MAINTENANCE,
                default=get_val(CONF_AUTO_MAINTENANCE, DEFAULT_AUTO_MAINTENANCE),
            ): bool,
            vol.Optional(
                CONF_EXPOSE_DEBUG_ENTITIES,
                default=get_val(CONF_EXPOSE_DEBUG_ENTITIES, False),
            ): bool,
            vol.Optional(
                CONF_SAVE_DEBUG_TRACES, default=get_val(CONF_SAVE_DEBUG_TRACES, False)
            ): bool,
            # --- External Cycle End Trigger ---
            vol.Optional(
                CONF_EXTERNAL_END_TRIGGER_ENABLED,
                default=get_val(CONF_EXTERNAL_END_TRIGGER_ENABLED, False),
            ): bool,
            vol.Optional(
                CONF_EXTERNAL_END_TRIGGER,
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="binary_sensor",
                    multiple=False,
                )
            ),
        }

        data_schema = vol.Schema(schema)
        data_schema = self.add_suggested_values_to_schema(
            data_schema,
            {CONF_EXTERNAL_END_TRIGGER: get_val(CONF_EXTERNAL_END_TRIGGER, None)},
        )

        return self.async_show_form(
            step_id="advanced_settings",
            data_schema=data_schema,
            description_placeholders={
                "error": "",
                "suggested": suggested_reason or "No suggestions available yet.",
                "suggested_min_power": _fmt_suggested(CONF_MIN_POWER),
                "suggested_off_delay": _fmt_suggested(CONF_OFF_DELAY),
                "suggested_watchdog_interval": _fmt_suggested(CONF_WATCHDOG_INTERVAL),
                "suggested_no_update_active_timeout": _fmt_suggested(
                    CONF_NO_UPDATE_ACTIVE_TIMEOUT
                ),
                "suggested_sampling_interval": _fmt_suggested(
                    CONF_SAMPLING_INTERVAL
                ),
                "suggested_profile_match_interval": _fmt_suggested(
                    CONF_PROFILE_MATCH_INTERVAL
                ),
                "suggested_auto_label_confidence": _fmt_suggested(
                    CONF_AUTO_LABEL_CONFIDENCE
                ),
                "suggested_duration_tolerance": _fmt_suggested(CONF_DURATION_TOLERANCE),
                "suggested_profile_duration_tolerance": _fmt_suggested(
                    CONF_PROFILE_DURATION_TOLERANCE
                ),
                "suggested_profile_match_min_duration_ratio": _fmt_suggested(
                    CONF_PROFILE_MATCH_MIN_DURATION_RATIO
                ),
                "suggested_profile_match_max_duration_ratio": _fmt_suggested(
                    CONF_PROFILE_MATCH_MAX_DURATION_RATIO
                ),

                "suggested_reason": suggested_reason,
                # Placeholders for keys in data_description
                "device": "{device}",
                "duration": "{duration}",
                "program": "{program}",
                "minutes": "{minutes}",
            },
        )




    # --- Interactive Editor Steps ---

    async def async_step_interactive_editor(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 1: Select Action (Merge or Split)."""
        if user_input is not None:
            self._editor_action = user_input["action"]
            return await self.async_step_editor_select()

        return self.async_show_form(
            step_id="interactive_editor",
            data_schema=vol.Schema(
                {
                    vol.Required("action"): vol.In(
                        {
                            "split": "✂️ Split a Cycle (Find gaps)",
                            "merge": "🔗 Merge Cycles (Join fragments)",
                        }
                    )
                }
            ),
        )

    async def async_step_editor_select(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 2: Select Cycles."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        
        errors = {}

        if user_input is not None:
            selected = user_input.get("selected_cycles", [])
            self._editor_selected_ids = selected
            
            # Validation
            if self._editor_action == "split":
                if len(selected) != 1:
                    errors["base"] = "select_exactly_one"
                else:
                    return await self.async_step_editor_split_params()
            
            elif self._editor_action == "merge":
                if len(selected) < 2:
                    errors["base"] = "select_at_least_two"
                else:
                    return await self.async_step_editor_configure()

        # Build options (Recent 50 cycles)
        cycles = store.get_past_cycles()[-50:]
        cycles.sort(key=lambda x: x["start_time"], reverse=True)
        
        options = []
        for c in cycles:
            dt = dt_util.parse_datetime(c["start_time"])
            start = dt_util.as_local(dt).strftime("%Y-%m-%d %H:%M") if dt else c["start_time"]
            duration_min = int(c["duration"] / 60)
            prof = c.get("profile_name") or "Unlabeled"
            label = f"{start} - {duration_min}m - {prof}"
            options.append(selector.SelectOptionDict(value=c["id"], label=label))

        return self.async_show_form(
            step_id="editor_select",
            data_schema=vol.Schema(
                {
                    vol.Required("selected_cycles"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                             options=options,
                             mode=selector.SelectSelectorMode.LIST,
                             multiple=True
                        )
                    )
                }
            ),
            errors=errors,
            description_placeholders={
                "info_text": "Select 1 cycle to split, or 2+ cycles to merge."
            }
        )

    async def async_step_editor_split_params(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 2.5: Configure split parameter."""
        if user_input is not None:
            self._editor_split_gap = int(user_input["min_gap_seconds"])
            return await self.async_step_editor_configure()

        return self.async_show_form(
            step_id="editor_split_params",
            data_schema=vol.Schema({
                vol.Required("min_gap_seconds", default=900): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=60, max=3600, mode=selector.NumberSelectorMode.BOX, unit_of_measurement="s"
                    )
                )
            })
        )

    async def async_step_editor_configure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 3: Configure and Preview."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store

        if user_input is not None:
            # Execute
            if self._editor_action == "split":
                # User selected action "apply"
                if user_input.get("confirm_commit"):
                    # We need the segments config. 
                    # For simplicity, we auto-apply the analysis results for now,
                    # as complex per-segment assignment in one form is hard in HA config flow.
                    # We'll rely on auto-labeling or user can label later.
                    
                    # Re-run analysis to get segments
                    cycle = next((c for c in store.get_past_cycles() if c["id"] in self._editor_selected_ids), None)
                    if cycle:

                        segments = await self.hass.async_add_executor_job(
                            store.analyze_split_sync, cycle, self._editor_split_gap, 2.0
                        )
                        if segments:
                            # Apply with profiles from user input
                            final_segments = []
                            for i, seg in enumerate(segments):
                                start_t, end_t = seg
                                prof = user_input.get(f"segment_{i}_profile")
                                if prof == "none": prof = None
                                final_segments.append({"start": start_t, "end": end_t, "profile": prof})

                            await store.apply_split_interactive(cycle["id"], final_segments)
                            # Maintenance: Reprocess envelopes in background to avoid blocking UI
                            self.hass.async_create_task(store.async_rebuild_all_envelopes())
                    
                    return self.async_create_entry(title="", data={})

            elif self._editor_action == "merge":
                if user_input.get("confirm_commit"):
                    target_profile = user_input.get("merged_profile")
                    
                    if target_profile == "create_new":
                        new_name = user_input.get("new_profile_name", "").strip()
                        if new_name:
                            # Create profile
                            try:
                                await store.create_profile_standalone(new_name)
                                target_profile = new_name
                            except ValueError:
                                # Profile exists or other error, fallback to unlabel/existing behavior
                                pass
                    
                    if target_profile in ("none", "create_new"): target_profile = None
                    
                    await store.apply_merge_interactive(self._editor_selected_ids, target_profile)
                    # Maintenance: Reprocess envelopes in background
                    self.hass.async_create_task(store.async_rebuild_all_envelopes())
                    return self.async_create_entry(title="", data={})

        # Generate Preview
        preview_md = ""
        schema = {}

        if self._editor_action == "split":
            cid = self._editor_selected_ids[0]
            cycle = next((c for c in store.get_past_cycles() if c["id"] == cid), None)
            if not cycle:
                return self.async_abort(reason="cycle_not_found")
            
            # Run Analysis

            segments = await self.hass.async_add_executor_job(
                store.analyze_split_sync, cycle, self._editor_split_gap, 2.0
            ) # Split params
            
            if not segments:
                return self.async_abort(reason="no_split_segments_found")

            # Generate SVG
            svg = store.generate_interactive_split_svg(cycle["id"], segments)
            b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
            
            preview_md = f"""
### Split Preview
Found {len(segments)} segments.
![Preview](data:image/svg+xml;base64,{b64})

Click Confirm to split this cycle into {len(segments)} separate cycles.
"""
            schema = {vol.Required("confirm_commit"): bool}
            
            # Add profile pickers for each segment
            profiles = store.list_profiles()
            prof_options = [selector.SelectOptionDict(value="none", label="(Unlabeled)")]
            for p in profiles:
                prof_options.append(selector.SelectOptionDict(value=p["name"], label=p["name"]))

            for i, seg in enumerate(segments):
                # seg_dur = int(seg[1] - seg[0])
                # label = f"New Cycle {i+1} ({seg_dur}s)" # Unused currently
                schema[vol.Optional(f"segment_{i}_profile", default="none")] = selector.SelectSelector(
                    selector.SelectSelectorConfig(options=prof_options, mode=selector.SelectSelectorMode.DROPDOWN, custom_value=False)
                )

        elif self._editor_action == "merge":
            # Get cycles
            cycles_to_merge = [c for c in store.get_past_cycles() if c["id"] in self._editor_selected_ids]
            cycles_to_merge.sort(key=lambda x: x["start_time"])
            
            # Generate SVG
            svg = store.generate_interactive_merge_svg([c["id"] for c in cycles_to_merge])
            b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

            # Profile Selector
            profiles = store.list_profiles()
            prof_options = [
                selector.SelectOptionDict(value="create_new", label="➕ Create New Profile..."),
                selector.SelectOptionDict(value="none", label="(Unlabeled)")
            ]
            for p in profiles:
                prof_options.append(selector.SelectOptionDict(value=p["name"], label=p["name"]))
            
            # Guess best profile?
            default_prof = cycles_to_merge[0].get("profile_name") or "none"

            preview_md = f"""
### Merge Preview
Joining {len(cycles_to_merge)} cycles. Gaps will be filled with 0W readings.
![Preview](data:image/svg+xml;base64,{b64})
"""
            schema = {
                vol.Optional("merged_profile", default=default_prof): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=prof_options, mode=selector.SelectSelectorMode.DROPDOWN)
                ),
                vol.Optional("new_profile_name"): str,
                vol.Required("confirm_commit"): bool
            }
        
        return self.async_show_form(
            step_id="editor_configure",
            data_schema=vol.Schema(schema),
            description_placeholders={"preview_md": preview_md}
        )



    async def async_step_diagnostics(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Diagnostics submenu for maintenance actions."""
        if user_input is not None:
            choice = user_input["action"]

            if choice == "reprocess_history":
                return await self.async_step_reprocess_history()
            if choice == "wipe_history":
                return await self.async_step_wipe_history()
            if choice == "export_import":
                return await self.async_step_export_import()
            if choice == "clear_debug_data":
                return await self.async_step_clear_debug_data()

        # Calculate storage usage stats
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        stats = await manager.profile_store.get_storage_stats()

        # Format stats string
        stats_str = "Storage Usage:\n"
        if "file_size_kb" in stats:
            stats_str += f"- File Size: {stats.get('file_size_kb', 0):.1f} KB\n"
        stats_str += f"- Cycles: {stats.get('total_cycles', 0)}\n"
        stats_str += f"- Profiles: {stats.get('total_profiles', 0)}\n"
        if stats.get("debug_traces_count", 0) > 0:
            stats_str += f"- Debug Traces: {stats.get('debug_traces_count', 0)}\n"

        return self.async_show_form(
            step_id="diagnostics",
            description_placeholders={"storage_stats": stats_str},
            data_schema=vol.Schema(
                {
                    vol.Required("action"): vol.In(
                        {

                            "reprocess_history": "Maintenance: Reprocess & Optimize Data",
                            "clear_debug_data": "Clear Debug Data (Free up space)",
                            "wipe_history": "Wipe ALL data for this device (irreversible)",
                            "export_import": "Export/Import JSON with settings (copy/paste)",
                        }
                    )
                }
            ),
        )

    async def async_step_clear_debug_data(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm clearing debug data."""
        if user_input is not None:
            manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
            count = await manager.profile_store.async_clear_debug_data()
            return self.async_abort(
                reason="debug_data_cleared",
                description_placeholders={"count": str(count)},
            )

        return self.async_show_form(
            step_id="clear_debug_data", description_placeholders={}
        )

    async def async_step_reprocess_history(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle request to reprocess all history."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]

        if user_input is not None:
            # Execute
            count = await manager.profile_store.async_reprocess_all_data()
            return self.async_abort(
                reason="reprocess_success",
                description_placeholders={"count": str(count)},
            )

        return self.async_show_form(
            step_id="reprocess_history",
            data_schema=vol.Schema({}),
            description_placeholders={
                "warning": (
                    "This will recalculate all cycle signatures and rebuild profile "
                    "models (envelopes) using the latest logic.\n\nRaw cycle data is preserved. "
                    "This may take a moment for large histories."
                )
            },
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
            options_to_return = dict(self.config_entry.options)

            if mode == "import":
                try:
                    payload = json.loads(payload_str)
                    config_updates = await manager.profile_store.async_import_data(
                        payload
                    )

                    # Apply imported settings to config entry if present
                    entry_data = config_updates.get("entry_data", {})
                    entry_options = config_updates.get("entry_options", {})

                    if entry_data or entry_options:
                        # Merge imported options with current data/options
                        new_data = {**self.config_entry.data}
                        new_options = {**self.config_entry.options}

                        # Only update settings that exist in the import
                        # (don't overwrite power_sensor/name)
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

                except Exception:  # noqa: BLE001, pylint: disable=broad-exception-caught
                    errors["base"] = "import_failed"
                    # Re-show form with error
                    return self.async_show_form(
                        step_id="export_import",
                        data_schema=vol.Schema(
                            {
                                vol.Required(
                                    "mode", default=mode
                                ): selector.SelectSelector(
                                    selector.SelectSelectorConfig(
                                        options=[
                                            selector.SelectOptionDict(
                                                value="export", label="Export only"
                                            ),
                                            selector.SelectOptionDict(
                                                value="import", label="Import from JSON"
                                            ),
                                        ]
                                    )
                                ),
                                vol.Optional(
                                    "json_payload", default=payload_str
                                ): selector.TextSelector(
                                    selector.TextSelectorConfig(multiline=True)
                                ),
                            }
                        ),
                        errors=errors,
                    )

            return self.async_create_entry(title="", data=options_to_return)

        return self.async_show_form(
            step_id="export_import",
            data_schema=vol.Schema(
                {
                    vol.Required("mode", default="export"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(
                                    value="export", label="Export only"
                                ),
                                selector.SelectOptionDict(
                                    value="import", label="Import from JSON"
                                ),
                            ]
                        )
                    ),
                    vol.Optional(
                        "json_payload", default=export_str
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(multiline=True)
                    ),
                }
            ),
        )

    async def async_step_manage_cycles(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage cycles submenu."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store

        # Build recent cycles list
        recent_cycles = store.get_past_cycles()[-8:]
        recent_lines = []
        for c in reversed(recent_cycles):
            dt = dt_util.parse_datetime(c["start_time"])
            start = dt_util.as_local(dt).strftime("%Y-%m-%d %H:%M") if dt else c["start_time"]
            duration_min = int(c["duration"] / 60)
            prof = c.get("profile_name") or "Unlabeled"
            status = c.get("status", "completed")
            status_icon = (
                "✓"
                if status in ("completed", "force_stopped")
                else "⚠" if status == "resumed" else "✗"
            )
            recent_lines.append(f"{status_icon} {start} - {duration_min}m - {prof}")
        recent_text = (
            "\n".join(recent_lines) if recent_lines else "No cycles recorded yet."
        )

        if user_input is not None:
            action = user_input["action"]
            if action == "auto_label_cycles":
                return await self.async_step_auto_label_cycles()
            elif action == "select_cycle_to_label":
                return await self.async_step_select_cycle_to_label()
            elif action == "select_cycle_to_delete":
                return await self.async_step_select_cycle_to_delete()
            elif action == "interactive_editor":
                # Initialize state
                self._editor_action = None
                self._editor_selected_ids = []
                self._editor_split_gap = 900
                return await self.async_step_interactive_editor()

        return self.async_show_form(
            step_id="manage_cycles",
            data_schema=vol.Schema(
                {
                    vol.Required("action"): vol.In(
                        {
                            "auto_label_cycles": "🤖 Auto-Label Old Cycles",
                            "select_cycle_to_label": "🏷️ Label Specific Cycle",
                            "select_cycle_to_delete": "🗑️ Delete Cycle",
                            "interactive_editor": "✂️ Merge/Split Interactive Editor",
                        }
                    )
                }
            ),
            description_placeholders={"recent_cycles": recent_text},
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
            avg = int(p["avg_duration"] / 60) if p["avg_duration"] else 0
            summary_lines.append(f"- **{p['name']}**: {count} cycles, {avg}m avg")
        summary_text = (
            "\n".join(summary_lines) if summary_lines else "No profiles created yet."
        )

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
            elif action == "cleanup_profile":
                return await self.async_step_cleanup_profile()

        return self.async_show_form(
            step_id="manage_profiles",
            data_schema=vol.Schema(
                {
                    vol.Required("action"): vol.In(
                        {
                            "create_profile": "➕ Create New Profile",
                            "edit_profile": "✏️ Edit/Rename Profile",
                            "delete_profile": "🗑️ Delete Profile",
                            "profile_stats": "📊 Profile Statistics",
                            "cleanup_profile": "🧹 Clean Up History - Graph & Delete",
                        }
                    )
                }
            ),
            description_placeholders={"profile_summary": summary_text},
        )

    async def async_step_cleanup_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select a profile to clean up (via graph)."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        profiles = store.list_profiles()

        if not profiles:
            return self.async_abort(reason="no_profiles_found")

        if user_input is not None:
            self._selected_profile = user_input["profile"]
            return await self.async_step_cleanup_select()

        # Build profile options
        options = []
        for p in profiles:
            count = p["cycle_count"]
            duration_min = int(p["avg_duration"] / 60) if p["avg_duration"] else 0
            label = f"{p['name']} ({count} cycles, ~{duration_min}m avg)"
            options.append(selector.SelectOptionDict(value=p["name"], label=label))

        return self.async_show_form(
            step_id="cleanup_profile",
            data_schema=vol.Schema(
                {
                    vol.Required("profile"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options, mode=selector.SelectSelectorMode.DROPDOWN
                        )
                    )
                }
            ),
        )

    async def async_step_cleanup_select(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show graph and select cycles to delete."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store

        if user_input is not None:
            # Delete selected cycles
            cycles_to_delete = user_input.get("cycles_to_delete", [])
            count = 0
            for cycle_id in cycles_to_delete:
                await store.delete_cycle(cycle_id)
                count += 1

            manager.notify_update()
            # Return to manage profiles
            return self.async_create_entry(
                title="",
                data=dict(self.config_entry.options),
                description_placeholders={
                    "info": f"Deleted {count} cycles from {self._selected_profile}."
                },
            )

        # Generate SVG
        ts = int(time.time())
        stats_dir = self.hass.config.path("www", "ha_washdata", "profiles")
        await self.hass.async_add_executor_job(
            lambda: os.makedirs(stats_dir, exist_ok=True)
        )

        safe_name = slugify(self._selected_profile)
        # Generate SVG with ALL cycles for this profile (outliers included)
        # Returns (svg_string, cycle_metadata_map) where metadata contains colors
        # Run in executor to avoid blocking loop
        svg_content, cycle_colors = await self.hass.async_add_executor_job(
            store.generate_profile_spaghetti_svg, self._selected_profile
        )

        if not svg_content:
            return self.async_abort(
                reason="no_cycles_found",
                description_placeholders={"info": "Not enough data to generate graph."},
            )

        file_path = f"{stats_dir}/cleanup_{safe_name}.svg"

        def write_svg(path=file_path, content=svg_content):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        await self.hass.async_add_executor_job(write_svg)
        graph_url = f"/local/ha_washdata/profiles/cleanup_{safe_name}.svg?v={ts}"

        # Get cycles for selection
        # We need to list ALL cycles for this profile
        all_cycles = store.get_past_cycles()
        profile_cycles = [
            c for c in all_cycles if c.get("profile_name") == self._selected_profile
        ]

        #Sort by start time descending
        profile_cycles.sort(key=lambda x: x["start_time"], reverse=True)

        # Map hex colors to emojis for easier identification
        hex_to_emoji = {
            "#e6194b": "🔴", # Red
            "#3cb44b": "🟢", # Green
            "#ffe119": "🟡", # Yellow
            "#4363d8": "🔵", # Blue
            "#f58231": "🟠", # Orange
            "#911eb4": "🟣", # Purple
            "#42d4f4": "🔵", # Cyan
            "#f032e6": "🟣", # Magenta
            "#bfef45": "🟢", # Lime
            "#fabed4": "🌸", # Pink
            "#469990": "teal", # Teal
            "#dcbeff": "🟣", # Lavender
            "#9A6324": "🟤", # Brown
            "#fffac8": "⚪", # Beige
            "#800000": "🔴", # Maroon
            "#aaffc3": "🟢", # Mint
            "#808000": "🟤", # Olive
            "#ffd8b1": "🟠", # Apricot
            "#000075": "🔵", # Navy
            "#a9a9a9": "⚪", # Grey
        }

        options = []
        for c in profile_cycles:
            dt = dt_util.parse_datetime(c["start_time"])
            start = dt_util.as_local(dt).strftime("%Y-%m-%d %H:%M") if dt else c["start_time"]
            duration_min = int(c["duration"] / 60)
            status = c.get("status", "completed")

            # Status icon
            status_icon = (
                "✓"
                if status in ("completed", "force_stopped")
                else "⚠" if status == "resumed" else "✗"
            )

            # Graph Color
            color_hex = cycle_colors.get(c["id"])
            color_emoji = hex_to_emoji.get(color_hex, "⚫") if color_hex else ""

            # Add energy if available to help identify
            energy = ""
            if "total_energy_kwh" in c:
                energy = f" | {c['total_energy_kwh']:.3f} kWh"

            label = f"{color_emoji} {status_icon} {start} - {duration_min}m{energy}"
            options.append(selector.SelectOptionDict(value=c["id"], label=label))

        return self.async_show_form(
            step_id="cleanup_select",
            data_schema=vol.Schema(
                {
                    vol.Optional("cycles_to_delete"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options,
                            mode=selector.SelectSelectorMode.LIST,
                            multiple=True,
                        )
                    )
                }
            ),
            description_placeholders={
                "graph_url": graph_url,
                "profile_name": self._selected_profile,
            },
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

        profiles = store.list_profiles()
        sections = []
        ts = int(time.time())

        # Get all cycles to find last run
        cycles = store.get_past_cycles()

        for p in profiles:
            name = p["name"]

            # FORCE REFRESH: Rebuild envelope to ensure data is fresh
            # This calculates energy, consistency, etc.
            await store.async_rebuild_envelope(name)

            safe_name = slugify(name)
            count = p["cycle_count"]
            avg = int(p["avg_duration"] / 60) if p["avg_duration"] else 0
            mn = int(p["min_duration"] / 60) if p.get("min_duration") else 0
            mx = int(p["max_duration"] / 60) if p.get("max_duration") else 0

            # Get envelope for advanced stats
            envelope = store.get_envelope(name)
            # Retrieve scalar stats
            kwh = f"{envelope.get('avg_energy', 0):.2f}" if envelope else "-"
            # Calculate Total Energy (Avg * Count)
            total_kwh = "-"
            if envelope and envelope.get('avg_energy') is not None:
                t_kwh = envelope.get('avg_energy', 0) * count
                total_kwh = f"{t_kwh:.2f}"
            
            std_dev = envelope.get("duration_std_dev", 0) if envelope else 0
            consistency = f"±{int(std_dev / 60)}m" if std_dev > 0 else "-"

            # Find last run
            last_run = "-"
            p_cycles = [c for c in cycles if c.get("profile_name") == name]
            if p_cycles:
                last_c = max(p_cycles, key=lambda x: x["start_time"])
                dt = last_c["start_time"].split("T")[0]
                last_run = dt

            # Generate and Write SVG
            # Offload to executor to prevent blocking
            svg_content = await self.hass.async_add_executor_job(
                store.generate_profile_svg, name
            )
            graph_markdown = ""
            if svg_content:
                file_path = f"{stats_dir}/profile_{safe_name}.svg"

                def write_svg(path=file_path, content=svg_content):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)

                await self.hass.async_add_executor_job(write_svg)
                graph_markdown = (
                    f"![{name}](/local/ha_washdata/profiles/profile_{safe_name}.svg?v={ts})"
                )

            # Build Per-Profile Section
            # Headers: Count | Avg | Min | Max | Energy | Consistency | Last Run
            # New: Energy (Avg) | Energy (Total)
            table_header = "| Count | Avg | Min | Max | Energy (Avg) | Energy (Total) | Consist. | Last Run |"
            table_sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
            table_row = (
                f"| {count} | {avg}m | {mn}m | {mx}m | {kwh} kWh | {total_kwh} kWh "
                f"| {consistency} | {last_run} |"
            )

            legend = (
                "> **Graph Legend**: The blue band represents the minimum and maximum power "
                "draw range observed. The line shows the average power curve."
            )

            section = (
                f"## {name}\n{table_header}\n{table_sep}\n{table_row}\n\n"
                f"{graph_markdown}\n\n{legend}"
            )
            sections.append(section)

        content = "\n\n---\n\n".join(sections) if sections else "No profiles found."

        return self.async_show_form(
            step_id="profile_stats",
            data_schema=vol.Schema({}),
            # Key 'stats_table' must match the key in translations/strings (description)
            description_placeholders={"stats_table": content},
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
                        avg_duration=avg_duration,
                    )
                    manager.notify_update()
                    return self.async_create_entry(
                        title="", data=dict(self.config_entry.options)
                    )
                except ValueError:
                    errors["base"] = "profile_exists"

        # Build cycle options for reference
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store
        cycles = store.get_past_cycles()[-20:]

        cycle_options = [
            selector.SelectOptionDict(value="none", label="(No reference cycle)")
        ]
        for c in reversed(cycles):
            dt = dt_util.parse_datetime(c["start_time"])
            start = dt_util.as_local(dt).strftime("%Y-%m-%d %H:%M") if dt else c["start_time"]
            duration_min = int(c["duration"] / 60)
            prof = c.get("profile_name") or "Unlabeled"
            label = f"{start} - {duration_min}m - {prof}"
            cycle_options.append(selector.SelectOptionDict(value=c["id"], label=label))

        return self.async_show_form(
            step_id="create_profile",
            data_schema=vol.Schema(
                {
                    vol.Required("profile_name"): str,
                    vol.Optional(
                        "reference_cycle", default="none"
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=cycle_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("manual_duration"): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            max=480,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement="min",
                        )
                    ),
                }
            ),
            errors=errors,
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
            data_schema=vol.Schema(
                {
                    vol.Required("profile"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options, mode=selector.SelectSelectorMode.DROPDOWN
                        )
                    )
                }
            ),
        )

    async def async_step_rename_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Edit profile settings (Name and Duration)."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        errors = {}

        # Get current profile data
        profiles = manager.profile_store.list_profiles()
        current_data = next(
            (p for p in profiles if p["name"] == self._selected_profile), None
        )
        current_duration_mins = (
            int(current_data["avg_duration"] / 60)
            if current_data and current_data["avg_duration"]
            else 0
        )

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
                        self._selected_profile, new_name, avg_duration=avg_duration
                    )
                    manager.notify_update()
                    return self.async_create_entry(
                        title="", data=dict(self.config_entry.options)
                    )
                except ValueError:
                    errors["base"] = "rename_failed"

        return self.async_show_form(
            step_id="rename_profile",
            data_schema=vol.Schema(
                {
                    vol.Required("new_name", default=self._selected_profile): str,
                    vol.Optional(
                        "manual_duration", default=current_duration_mins
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            max=480,
                            unit_of_measurement="min",
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                }
            ),
            errors=errors,
            description_placeholders={"current_name": self._selected_profile},
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
            data_schema=vol.Schema(
                {
                    vol.Required("profile"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options, mode=selector.SelectSelectorMode.DROPDOWN
                        )
                    )
                }
            ),
        )

    async def async_step_delete_profile_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm profile deletion."""
        if user_input is not None:
            unlabel = user_input["unlabel_cycles"]
            manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
            await manager.profile_store.delete_profile(
                self._selected_profile, unlabel
            )
            manager.notify_update()
            return self.async_create_entry(
                title="", data=dict(self.config_entry.options)
            )

        return self.async_show_form(
            step_id="delete_profile_confirm",
            data_schema=vol.Schema(
                {vol.Required("unlabel_cycles", default=True): bool}
            ),
            description_placeholders={
                "profile_name": self._selected_profile,
                "warning": "⚠️ This will permanently delete the profile.",
            },
        )

    async def async_step_auto_label_cycles(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Auto-label all cycles retroactively."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store

        # Check if there are any profiles to match against
        profiles = store.list_profiles()
        if not profiles:
            return self.async_abort(reason="no_profiles_for_matching")

        total_count = len(store.get_past_cycles())

        if total_count == 0:
            return self.async_abort(reason="no_cycles_found")

        if user_input is not None:
            threshold = user_input["confidence_threshold"]
            # Always pass overwrite=True as per user request to relabel everything
            await store.auto_label_cycles(threshold, overwrite=True)
            manager.notify_update()
            return self.async_create_entry(
                title="",
                data=dict(self.config_entry.options),
            )

        return self.async_show_form(
            step_id="auto_label_cycles",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        "confidence_threshold", default=0.75
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.50,
                            max=0.95,
                            step=0.05,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    )
                }
            ),
            description_placeholders={
                "info": (
                    f"Found {total_count} total cycles. "
                    f"Profiles: {', '.join(p['name'] for p in profiles)}"
                )
            },
        )

    async def async_step_select_cycle_to_label(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select a cycle to label."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store

        # Get last 20 cycles
        cycles = store.get_past_cycles()[-20:]

        # Build readable options with status
        options = []
        for c in reversed(cycles):
            dt = dt_util.parse_datetime(c["start_time"])
            start = dt_util.as_local(dt).strftime("%Y-%m-%d %H:%M") if dt else c["start_time"]
            duration_min = int(c["duration"] / 60)
            prof = c.get("profile_name") or "Unlabeled"
            status = c.get("status", "completed")
            # ✓ = completed/force_stopped (natural end), ⚠ = resumed, ✗ = interrupted (user stopped)
            status_icon = (
                "✓"
                if status in ("completed", "force_stopped")
                else "⚠" if status == "resumed" else "✗"
            )
            label = f"[{status_icon}] {start} - {duration_min}m - {prof}"
            options.append(selector.SelectOptionDict(value=c["id"], label=label))

        if not options:
            return self.async_abort(reason="no_cycles_found")

        if user_input is not None:
            self._selected_cycle_id = user_input["cycle_id"]
            return await self.async_step_label_cycle()

        return self.async_show_form(
            step_id="select_cycle_to_label",
            data_schema=vol.Schema(
                {
                    vol.Required("cycle_id"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options, mode=selector.SelectSelectorMode.DROPDOWN
                        )
                    )
                }
            ),
        )

    async def async_step_select_cycle_to_delete(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select a cycle to delete."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        store = manager.profile_store

        # Get last 20 cycles
        cycles = store.get_past_cycles()[-20:]

        # Build readable options with status
        options = []
        for c in reversed(cycles):
            dt = dt_util.parse_datetime(c["start_time"])
            start = dt_util.as_local(dt).strftime("%Y-%m-%d %H:%M") if dt else c["start_time"]
            duration_min = int(c["duration"] / 60)
            prof = c.get("profile_name") or "Unlabeled"
            status = c.get("status", "completed")
            # ✓ = completed/force_stopped (natural end), ⚠ = resumed, ✗ = interrupted (user stopped)
            status_icon = (
                "✓"
                if status in ("completed", "force_stopped")
                else "⚠" if status == "resumed" else "✗"
            )
            label = f"[{status_icon}] {start} - {duration_min}m - {prof}"
            options.append(selector.SelectOptionDict(value=c["id"], label=label))

        if not options:
            return self.async_abort(reason="no_cycles_found")

        if user_input is not None:
            cycle_id = user_input["cycle_id"]
            await manager.profile_store.delete_cycle(cycle_id)
            # await manager.profile_store.async_save() # Handled inside delete_cycle now
            manager.notify_update()
            return self.async_create_entry(
                title="", data=dict(self.config_entry.options)
            )

        return self.async_show_form(
            step_id="select_cycle_to_delete",
            data_schema=vol.Schema(
                {
                    vol.Required("cycle_id"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options, mode=selector.SelectSelectorMode.DROPDOWN
                        )
                    )
                }
            ),
            description_placeholders={
                "warning": "⚠️ This will permanently delete the selected cycle"
            },
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
                        manager.notify_update()
                        return self.async_create_entry(
                            title="", data=dict(self.config_entry.options)
                        )
                    except ValueError:
                        errors["base"] = "profile_exists"
            elif profile_choice == "__remove_label__":
                # Remove label from cycle
                await store.assign_profile_to_cycle(self._selected_cycle_id, None)
                manager.notify_update()
                return self.async_create_entry(
                    title="", data=dict(self.config_entry.options)
                )
            else:
                # Assign existing profile
                try:
                    await store.assign_profile_to_cycle(
                        self._selected_cycle_id, profile_choice
                    )
                    manager.notify_update()
                    return self.async_create_entry(
                        title="", data=dict(self.config_entry.options)
                    )
                except ValueError:
                    errors["base"] = "assignment_failed"

        # Build profile dropdown options
        profiles = store.list_profiles()
        profile_options = [
            selector.SelectOptionDict(
                value="__create_new__", label="➕ Create New Profile"
            ),
            selector.SelectOptionDict(value="__remove_label__", label="🗑️ Remove Label"),
        ]
        for p in profiles:
            count = p["cycle_count"]
            duration_min = int(p["avg_duration"] / 60) if p["avg_duration"] else 0
            label = f"{p['name']} ({count} cycles, ~{duration_min}m)"
            profile_options.append(
                selector.SelectOptionDict(value=p["name"], label=label)
            )

        # Get cycle info for display
        cycle = next(
            (
                c
                for c in store.get_past_cycles()
                if c["id"] == self._selected_cycle_id
            ),
            None,
        )
        cycle_info = ""
        if cycle:
            dt = dt_util.parse_datetime(cycle["start_time"])
            start = dt_util.as_local(dt).strftime("%Y-%m-%d %H:%M") if dt else cycle["start_time"]
            duration_min = int(cycle["duration"] / 60)
            current_label = cycle.get("profile") or "Unlabeled"
            cycle_info = f"Cycle: {start}, {duration_min}m, Current: {current_label}"

        schema = {
            vol.Required(
                "profile_name",
                default="__create_new__" if not profiles else profiles[0]["name"],
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=profile_options, mode=selector.SelectSelectorMode.DROPDOWN
                )
            )
        }

        # Add new profile name field (shown when "__create_new__" selected)
        schema[vol.Optional("new_profile_name")] = str

        return self.async_show_form(
            step_id="label_cycle",
            data_schema=vol.Schema(schema),
            errors=errors,
            description_placeholders={"cycle_info": cycle_info},
        )

    async def async_step_post_process(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle post-processing options."""
        if user_input is not None:
            choice = user_input["time_range"]
            user_gap = user_input["gap_seconds"]
            manager = self.hass.data[DOMAIN][self.config_entry.entry_id]

            hours = 999999 if choice >= 999999 else int(choice)

            # Use async_run_maintenance to ensure envelopes are rebuilt after merging
            stats = await manager.profile_store.async_run_maintenance(
                lookback_hours=hours, gap_seconds=user_gap
            )
            count_merged = stats.get("merged_cycles", 0)
            count_split = stats.get("split_cycles", 0)
            msg = f"Merged: {count_merged}, Split: {count_split}"

            return self.async_create_entry(
                title="",
                data=dict(self.config_entry.options),
                description_placeholders={"count": msg},
            )



        return self.async_show_form(
            step_id="post_process",
            data_schema=vol.Schema(
                {
                    vol.Required("time_range", default=24): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            max=9999,
                            unit_of_measurement="h",
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),

                }
            ),
            description_placeholders={
                "info": (
                    "Enter number of past hours to process (or use 999999 for all).\n\n"
                )
            },
        )



    async def async_step_wipe_history(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Wipe all stored cycles and profiles for this device (for testing)."""
        if user_input is not None:
            manager = self.hass.data[DOMAIN][self.config_entry.entry_id]

            # Clear all cycles and profiles
            await manager.profile_store.clear_all_data()
            manager.notify_update()

            return self.async_create_entry(
                title="",
                data=dict(self.config_entry.options),
                description_placeholders={"info": "History cleared"},
            )

        return self.async_show_form(
            step_id="wipe_history",
            data_schema=vol.Schema({}),
            description_placeholders={
                "warning": (
                    "⚠️ This will permanently delete ALL stored cycles and profiles for "
                    "this device. This cannot be undone!"
                )
            },
        )

    async def async_step_record_cycle(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle Record Mode menu."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]

        # Determine available actions based on state
        is_recording = manager.recorder.is_recording
        has_last_run = manager.recorder.last_run is not None

        if user_input is not None:
            action = user_input["action"]
            if action == "start_recording":
                return await self.async_step_record_start()
            if action == "stop_recording":
                return await self.async_step_record_stop()
            if action == "process_recording":
                return await self.async_step_record_process()
            if action == "discard_recording":
                await manager.recorder.clear_last_run()
                return await self.async_step_record_cycle()
            if action == "refresh_status":
                return await self.async_step_record_cycle()

        options = {}
        if is_recording:
            options["refresh_status"] = "Refresh Status"
            options["stop_recording"] = "Stop Recording (Save & Process)"
            status = "ACTIVE"
            duration = int(manager.recorder.current_duration)
            samples = len(getattr(manager.recorder, "_buffer", []))
        else:
            options["start_recording"] = "Start New Recording"
            status = "STOPPED"
            duration = 0
            samples = 0

            if has_last_run:
                options["process_recording"] = "Process Last Recording (Trim & Save)"
                options["discard_recording"] = "Discard Last Recording"

                last_run = manager.recorder.last_run
                samples = len(last_run.get("data", []))
                try:
                    start = dt_util.parse_datetime(last_run["start_time"])
                    end = dt_util.parse_datetime(last_run["end_time"])
                    duration = int((end - start).total_seconds())
                    status = "READY TO PROCESS"
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

        return self.async_show_form(
            step_id="record_cycle",
            data_schema=vol.Schema({vol.Required("action"): vol.In(options)}),
            description_placeholders={
                "status": status,
                "duration": str(duration),
                "samples": str(samples)
            },
        )

    async def async_step_record_start(
        self, _user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Start a recording."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        await manager.async_start_recording()
        return await self.async_step_record_cycle()

    async def async_step_record_stop(
        self, _user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Stop a recording."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        await manager.async_stop_recording()
        # Automatically go to process step? Or back to menu?
        # User might want to immediately process.
        return await self.async_step_record_process()

    async def async_step_record_process(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Process (Trim & Save) the recorded cycle."""
        manager = self.hass.data[DOMAIN][self.config_entry.entry_id]
        last_run = manager.recorder.last_run

        if not last_run:
            return self.async_abort(reason="no_recording_found")

        data = last_run.get("data", [])

        if user_input is not None:
            save_mode = user_input["save_mode"]

            if save_mode == "discard":
                await manager.recorder.clear_last_run()
                return self.async_create_entry(title="", data={})

            head_trim = user_input["head_trim"]
            tail_trim = user_input["tail_trim"]
            profile_name = user_input["profile_name"].strip()

            # GET ACTUAL RECORDING BOUNDS
            rec_start_str = manager.recorder.last_run.get("start_time")
            rec_end_str = manager.recorder.last_run.get("end_time")

            # APPLY TRIMS
            # Convert data to timestamps
            parsed = []
            for t_str, p in data:
                t = dt_util.parse_datetime(t_str)
                if t:
                    parsed.append((t.timestamp(), p))

            if not parsed and not rec_start_str:
                return self.async_abort(reason="empty_recording")

            # Use recording bounds if available, fallback to data bounds
            data_start_ts = parsed[0][0] if parsed else 0
            data_end_ts = parsed[-1][0] if parsed else 0
            
            start_ts = dt_util.parse_datetime(rec_start_str).timestamp() if rec_start_str else data_start_ts
            end_ts = dt_util.parse_datetime(rec_end_str).timestamp() if rec_end_str else data_end_ts
            
            # Ensure bounds cover data
            start_ts = min(start_ts, data_start_ts) if parsed else start_ts
            end_ts = max(end_ts, data_end_ts) if parsed else end_ts

            # Calculate cut points
            keep_start = start_ts + head_trim
            keep_end = end_ts - tail_trim

            trimmed_data = []
            for t, p in parsed:
                if t >= keep_start and t <= keep_end:
                    t_iso = dt_util.utc_from_timestamp(t).isoformat()
                    trimmed_data.append((t_iso, p))

            duration = max(0.0, (keep_end - keep_start))

            cycle_data = {
                "id": f"rec_{int(time.time())}",
                "start_time": dt_util.utc_from_timestamp(keep_start).isoformat(),
                "end_time": dt_util.utc_from_timestamp(keep_end).isoformat(),
                "duration": duration,
                "profile_name": profile_name,
                "power_data": trimmed_data,
                "status": "completed",
                "meta": {"source": "recorder", "original_samples": len(data)}
            }

            if save_mode == "new_profile":
                await manager.profile_store.create_profile_standalone(profile_name)
                await manager.profile_store.async_add_cycle(cycle_data)
                await manager.profile_store.async_rebuild_envelope(profile_name)
            else:
                # Add to existing
                await manager.profile_store.async_add_cycle(cycle_data)
                await manager.profile_store.async_rebuild_envelope(profile_name)

            await manager.profile_store.async_save()
            await manager.recorder.clear_last_run()

            return self.async_create_entry(title="", data={})

        # Calculate suggestions
        rec_start_str = manager.recorder.last_run.get("start_time")
        rec_end_str = manager.recorder.last_run.get("end_time")

        rec_start = dt_util.parse_datetime(rec_start_str) if rec_start_str else None
        rec_end = dt_util.parse_datetime(rec_end_str) if rec_end_str else None

        head_suggest, tail_suggest, sampling_rate = manager.recorder.get_trim_suggestions(
            data, recording_start=rec_start, recording_end=rec_end
        )

        # Generate Preview Graph
        ts = int(time.time())
        stats_dir = self.hass.config.path("www", "ha_washdata", "preview")
        await self.hass.async_add_executor_job(
            lambda: os.makedirs(stats_dir, exist_ok=True)
        )

        svg_content = await self.hass.async_add_executor_job(
            manager.profile_store.generate_preview_svg,
            data,
            head_suggest,
            tail_suggest
        )

        graph_url = ""
        if svg_content:
            fname = f"preview_{ts}.svg"
            path = f"{stats_dir}/{fname}"

            def write_svg():
                with open(path, "w", encoding="utf-8") as f:
                    f.write(svg_content)

            await self.hass.async_add_executor_job(write_svg)
            graph_url = f"/local/ha_washdata/preview/{fname}?v={ts}"

        # Profile options
        profiles = list(manager.profile_store.get_profiles().keys())
        profiles.sort(key=profile_sort_key)

        schema = {
            vol.Required("head_trim", default=head_suggest): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=300000, step=0.1, mode=selector.NumberSelectorMode.BOX, unit_of_measurement="s")
            ),
            vol.Required("tail_trim", default=tail_suggest): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=300000, step=0.1, mode=selector.NumberSelectorMode.BOX, unit_of_measurement="s")
            ),
            vol.Required("save_mode", default="existing_profile"): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                         {"value": "new_profile", "label": "Create New Profile"},
                         {"value": "existing_profile", "label": "Add to Existing Profile"},
                         {"value": "discard", "label": "Discard Recording"},
                    ],
                    mode=selector.SelectSelectorMode.LIST
                )
            ),
            # Profile name optional if discarding? No dynamic update, so required logic applies.
            # We can't easily make it conditional. User has to pick something or we allow empty?
            # If user picks "Create New", they need a name. "Existing", they pick one.
            # "Discard", name is ignored.
            # Providing a text input that doubles as selector is tricky.
            # We used SelectSelector with custom_value=True.
            vol.Required("profile_name"): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=profiles,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                    custom_value=True
                )
            )
        }

        # Calculate duration for display
        duration_val = 0.0
        if rec_start and rec_end:
            duration_val = (rec_end - rec_start).total_seconds()

        return self.async_show_form(
            step_id="record_process",
            data_schema=vol.Schema(schema),
            description_placeholders={
                "samples": str(len(data)),
                 "duration": f"{duration_val:.1f}",
                 "graph_url": graph_url,
                 "sampling_rate": str(sampling_rate)
             }
        )

    async def async_step_learning_feedbacks(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step: List pending learning feedbacks."""
        if user_input is not None:
            cycle_id = user_input.get("selected_feedback")
            if cycle_id:
                self._selected_cycle_id = cycle_id
                return await self.async_step_resolve_feedback()

        # Access profile_store from manager
        manager = self.hass.data[DOMAIN][self._config_entry.entry_id]
        profile_store = manager.profile_store
        pending = profile_store.get_pending_feedback()

        if not pending:
            return self.async_show_form(
                step_id="learning_feedbacks_empty",
                data_schema=vol.Schema({}),
                last_step=False,
            )

        options = []
        # Sort by creation time (newest first)
        sorted_pending = sorted(
            pending.values(), 
            key=lambda x: x.get("created_at", ""), 
            reverse=True
        )

        for item in sorted_pending:
            cid = item.get("cycle_id", "unknown")
            prof = item.get("detected_profile", "Unknown")
            conf = item.get("confidence", 0.0)
            created_raw = item.get("created_at", "")
            
            # Formt timestamp: 2023-10-27T10:00... -> 27 Oct 10:00
            t_str = str(created_raw)
            if created_raw:
                try:
                    # Parse using HA util to be safe with timezones
                    dt = dt_util.parse_datetime(str(created_raw))
                    if dt:
                        local_dt = dt_util.as_local(dt)
                        # "27 Oct 10:00" - Short and readable
                        t_str = local_dt.strftime("%d %b %H:%M")
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

            # Format label
            label = f"{prof} ({int(conf*100)}%) - {t_str}"
            options.append(selector.SelectOptionDict(value=cid, label=label))

        return self.async_show_form(
            step_id="learning_feedbacks",
            data_schema=vol.Schema(
                {
                    vol.Required("selected_feedback"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    )
                }
            ),
        )

    async def async_step_learning_feedbacks_empty(
        self, _user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step: Handle empty feedback list (go back)."""
        return await self.async_step_init()

    async def async_step_resolve_feedback(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step: Resolve a specific feedback request."""
        manager = self.hass.data[DOMAIN][self._config_entry.entry_id]
        profile_store = manager.profile_store
        pending = profile_store.get_pending_feedback()
        
        cycle_id = self._selected_cycle_id
        if not cycle_id or cycle_id not in pending:
            return self.async_abort(reason="feedback_not_found")

        item = pending[cycle_id]

        if user_input is not None:
            # Process submission based on action
            action = user_input.get("action", "confirm")
            
            if action == "confirm":
                # User confirms the detection was correct
                if hasattr(manager, "learning_manager"):
                    await manager.learning_manager.async_submit_cycle_feedback(
                       cycle_id=cycle_id,
                       user_confirmed=True,
                       corrected_profile=None,
                       corrected_duration=None,
                       dismiss=False,
                    )
            elif action == "correct":
                # User wants to correct the profile/duration
                new_profile = user_input.get("corrected_profile")
                new_duration = user_input.get("corrected_duration")
                if hasattr(manager, "learning_manager"):
                    await manager.learning_manager.async_submit_cycle_feedback(
                       cycle_id=cycle_id,
                       user_confirmed=False,
                       corrected_profile=new_profile,
                       corrected_duration=int(new_duration * 60) if new_duration else None,
                       dismiss=False,
                    )
            elif action == "dismiss":
                # User wants to dismiss/ignore this feedback request
                if hasattr(manager, "learning_manager"):
                    await manager.learning_manager.async_submit_cycle_feedback(
                       cycle_id=cycle_id,
                       user_confirmed=False,
                       corrected_profile=None,
                       corrected_duration=None,
                       dismiss=True,
                    )
            
            # Return to main menu
            return await self.async_step_init()

        # Prepare form
        detected_profile = item.get("detected_profile", "Unknown")
        confidence = item.get("confidence", 0.0)
        est = item.get("estimated_duration", 0)
        act = item.get("actual_duration", 0)
        
        profiles = list(profile_store.get_profiles().keys())
        profiles.sort(key=profile_sort_key)
        
        # Action options
        action_options = [
            selector.SelectOptionDict(value="confirm", label="✓ Confirm (detection was correct)"),
            selector.SelectOptionDict(value="correct", label="✎ Correct (change profile/duration)"),
            selector.SelectOptionDict(value="dismiss", label="✕ Dismiss (ignore this feedback)"),
        ]
        
        return self.async_show_form(
            step_id="resolve_feedback",
            description_placeholders={
                "profile": detected_profile,
                "confidence": str(int(confidence * 100)),
                "est": str(int(est / 60)),
                "act": str(int(act / 60)),
            },
            data_schema=vol.Schema(
                {
                    vol.Required("action", default="confirm"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=action_options,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                    vol.Optional("corrected_profile", default=detected_profile): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=profiles,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("corrected_duration", default=int(act/60)): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0, max=600, unit_of_measurement="min", mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                }
            ),
        )

