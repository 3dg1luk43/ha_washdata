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

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import (
    HomeAssistantError,
    ServiceValidationError,
    Unauthorized,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
import voluptuous as vol

from .const import (
    FAILED_RESTORE_STORE_SUFFIX,
    NOTIFY_QUEUE_STORE_SUFFIX,
    PRE_IMPORT_STORE_SUFFIX,
    STORAGE_KEY,
)
from .const import (
    DEVICE_COMPLETION_THRESHOLDS,
    DOMAIN,
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
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
    CONF_WATCHDOG_INTERVAL,
    CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_DEVICE_TYPE,
    DEVICE_TYPE_OTHER,
    CONF_START_DURATION_THRESHOLD,
    resolve_watchdog_interval_default,
    resolve_start_duration_default,
    CONF_RUNNING_DEAD_ZONE,
)
from .log_utils import DeviceLoggerAdapter
from .options_utils import strip_null_options

try:  # pragma: no cover - present in every supported HA, guarded on principle
    from homeassistant.setup import SetupPhases, async_pause_setup
except ImportError:  # pragma: no cover
    from contextlib import nullcontext

    class SetupPhases:  # type: ignore[no-redef]
        """Fallback so a missing helper degrades to plain (uncredited) waiting."""

        WAIT_IMPORT_PACKAGES = "wait_import_packages"

    def async_pause_setup(hass, phase):  # type: ignore[misc]
        """No-op stand-in for HA's setup-time credit context manager."""
        return nullcontext()


_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.SELECT,
    Platform.BUTTON,
]

# Shared future for the once-per-HA-instance ML module warm-up (issue #408).
ML_PRELOAD_FUTURE_KEY = "ha_washdata_ml_preload"

# Entry ids whose entity platforms are currently forwarded.  HA leaves forwarded
# platforms in place when async_setup_entry raises, and it refuses to set the same
# platform up twice, so a retry after a part-way failure has to take them down
# first (issue #425).
FORWARDED_ENTRIES_KEY = "ha_washdata_forwarded_entries"


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key=f"{name}_required",
        )
    return value


def _svc(fields: dict[Any, Any]) -> vol.Schema:
    # ALLOW_EXTRA: type the declared fields without rejecting a key an automation
    # already passes (submit_cycle_feedback's `dismiss` is read but undeclared).
    return vol.Schema({vol.Required("device_id"): cv.string, **fields}, extra=vol.ALLOW_EXTRA)


# One schema per service, mirroring services.yaml (audit PLATFORM-10): none had
# one, so `profile_name: 123` raised AttributeError, a non-numeric trim_start_s a
# ValueError traceback, and `unlabel_cycles: "false"` read as true.
def _float(value: Any) -> float:
    """``vol.Coerce(float)`` that also rejects an oversized JSON integer: voluptuous
    catches only ValueError/TypeError, so ``float(10**400)`` escaped as OverflowError."""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as err:
        raise vol.Invalid("expected a number") from err


_OPT_STR = vol.Any(None, cv.string)
_OPT_NUM = vol.Any(None, _float)
_SERVICE_SCHEMAS: dict[str, vol.Schema] = {
    "label_cycle": _svc({vol.Required("cycle_id"): cv.string,
                         vol.Optional("profile_name"): _OPT_STR}),
    "create_profile": _svc({vol.Required("profile_name"): cv.string,
                            vol.Optional("reference_cycle_id"): _OPT_STR}),
    "delete_profile": _svc({vol.Required("profile_name"): cv.string,
                            vol.Optional("unlabel_cycles"): cv.boolean}),
    "auto_label_cycles": _svc({vol.Optional("confidence_threshold"): vol.Any(None, vol.All(
        _float, vol.Range(min=0.0, max=1.0)))}),
    "export_config": _svc({vol.Optional("path"): _OPT_STR}),
    "import_config": _svc({vol.Required("path"): cv.string}),
    "submit_cycle_feedback": vol.Schema({
        vol.Optional("device_id"): _OPT_STR,
        vol.Optional("entry_id"): _OPT_STR,
        vol.Required("cycle_id"): cv.string,
        vol.Optional("user_confirmed"): cv.boolean,
        vol.Optional("corrected_profile"): _OPT_STR,
        # As services.yaml's selector: YAML automations bypass it, and NaN, inf or a
        # negative value reached the stored cycle's duration.
        vol.Optional("corrected_duration"): vol.Any(
            None, vol.All(_float, vol.Range(min=0, max=86400))
        ),
        vol.Optional("notes"): _OPT_STR,
        vol.Optional("dismiss"): cv.boolean,
    }, extra=vol.ALLOW_EXTRA),
    "record_start": _svc({}),
    "record_stop": _svc({}),
    "trim_cycle": _svc({vol.Required("cycle_id"): cv.string,
                        vol.Optional("trim_start_s"): _float,
                        vol.Optional("trim_end_s"): _OPT_NUM}),
    "pause_cycle": _svc({}),
    "resume_cycle": _svc({}),
    "mark_unloaded": _svc({}),
    "trigger_ml_training": _svc({}),
}


def _service_manager(hass: HomeAssistant, device_id: str) -> tuple[str, Any]:
    """``(entry_id, manager)`` for a service call's device (audit PLATFORM-10).

    One resolver for every service: each used to copy-paste it, nine raising a bare
    ValueError (an "unknown error" plus traceback in the UI) and all taking
    ``next(iter(device.config_entries))`` - an arbitrary entry of the device, not
    necessarily ours.
    """
    device = dr.async_get(hass).async_get(device_id)
    if not device:
        raise ServiceValidationError(
            translation_domain=DOMAIN, translation_key="device_not_found"
        )
    loaded = hass.data.get(DOMAIN, {})
    entry_id = next((eid for eid in device.config_entries if eid in loaded), None)
    if entry_id is None:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key=(
                "integration_not_loaded" if device.config_entries else "no_config_entry"
            ),
        )
    return entry_id, loaded[entry_id]


# Options keys that no code reads any more, stripped by the 3.10 -> 3.11 step and
# again by the one-pass legacy migration. Named once so the two cannot drift.
# The value `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO` held before item 311
# widened it. A stored option equal to this is the migration's own seed, not a
# user's choice, and the 3.10 -> 3.11 step heals it.
_OLD_SEEDED_MAX_DURATION_RATIO = 1.5

_DEAD_ABRUPT_KEYS = frozenset(
    {"abrupt_drop_ratio", "abrupt_drop_watts", "abrupt_high_load_factor"}
)


def _heal_seeded_cadence(options: dict[str, Any], device_type: Any) -> list[str]:
    """Replace the seeded 30/5 cadence with the device default (#396), in place.

    The pre-3.9 legacy migration seeded watchdog_interval=30 and
    start_duration_threshold=5. On the coarse (30 s) sampling device types those
    fall below the panel's watchdog>=2*sampling and start_duration>=sampling gates,
    so they are replaced with the device-resolved default, but ONLY where they still
    equal the old scalar default: a value the migration seeded, never a deliberate
    choice. An absent key stays absent, so the runtime default applies.

    One helper for the 3.9 -> 3.10 step and the one-pass legacy path (audit
    PLATFORM-09): the bulk path copied the 3.10 -> 3.11 heal but not this one, so a
    dryer migrating from 3.1 or 3.5 kept 30/5 while the same entry at 3.9 got 61/30.
    A falsy device type means DEFAULT_DEVICE_TYPE, not the coarse scalar fallback.
    """
    dev = device_type or DEFAULT_DEVICE_TYPE
    healed: list[str] = []
    if options.get(CONF_WATCHDOG_INTERVAL) == 30:
        resolved = resolve_watchdog_interval_default(dev)
        if resolved != 30:
            options[CONF_WATCHDOG_INTERVAL] = resolved
            healed.append(CONF_WATCHDOG_INTERVAL)
    if options.get(CONF_START_DURATION_THRESHOLD) == 5:
        resolved_start = resolve_start_duration_default(dev)
        if resolved_start != 5:
            options[CONF_START_DURATION_THRESHOLD] = resolved_start
            healed.append(CONF_START_DURATION_THRESHOLD)
    return healed


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry to the latest version while preserving settings."""
    _log = DeviceLoggerAdapter(_LOGGER, entry.title)
    version = entry.version or 1
    minor_version = entry.minor_version or 1

    if version > CONFIG_ENTRY_VERSION:
        _log.error(
            "Refusing to migrate unsupported future schema %s.%s", version, minor_version
        )
        return False

    if version == CONFIG_ENTRY_VERSION and minor_version >= CONFIG_ENTRY_MINOR_VERSION:
        return True

    # 3.6 → 3.7: remove initial_profile stub key from entry.data.
    if version == 3 and minor_version == 6:
        new_data = {k: v for k, v in entry.data.items() if k != "initial_profile"}
        hass.config_entries.async_update_entry(
            entry, data=new_data, minor_version=7
        )
        minor_version = 7
        _log.debug("Migrated WashData entry from 3.6 to 3.7")

    # 3.7 → 3.8: drop running_dead_zone from options (the setting was never
    # wired to any detection logic and has been retired in 0.5.3).
    if version == 3 and minor_version == 7:
        new_opts = {k: v for k, v in entry.options.items() if k != CONF_RUNNING_DEAD_ZONE}
        hass.config_entries.async_update_entry(
            entry, options=new_opts, minor_version=8
        )
        minor_version = 8
        _log.debug("Migrated WashData entry from 3.7 to 3.8 (removed running_dead_zone)")

    # 3.8 → 3.9: drop options persisted as null. A never-saved setting has no
    # entry in options, so the per-setting Revert used to send the changelog's
    # `old` (null) and ws_set_options stored it verbatim - and a stored None
    # survives options.get(key, DEFAULT), so the numeric casts that build
    # CycleDetectorConfig raised TypeError and the entry could never be set up
    # again without hand-editing .storage (#389). ws_set_options and the import
    # path now strip on write; this heals entries already carrying one. Healing
    # here rather than on a setup failure keeps it deterministic, covers the
    # crash classes a null produces outside a numeric cast (a null power_sensor
    # raises AttributeError inside hass.states.get(None)), and never rewrites the
    # entry mid-setup.
    if version == 3 and minor_version == 8:
        new_opts = strip_null_options(entry.options)
        dropped = sorted(set(entry.options) - set(new_opts))
        hass.config_entries.async_update_entry(
            entry, options=new_opts, minor_version=9
        )
        minor_version = 9
        if dropped:
            _log.warning(
                "Migrated WashData entry from 3.8 to 3.9: dropped option(s) %s "
                "stored as null so the compiled defaults apply again",
                dropped,
            )
        else:
            _log.debug("Migrated WashData entry from 3.8 to 3.9 (no null options)")

    # 3.9 → 3.10: heal cadence defaults that violate the panel's own conflict rules
    # (#396); see _heal_seeded_cadence, which the one-pass legacy path shares.
    if version == 3 and minor_version == 9:
        new_opts = dict(entry.options)
        # `or` (not `.get(..., default)`) so a present-but-null device type also falls
        # through to the data value / DEFAULT_DEVICE_TYPE: a null would otherwise resolve
        # to the coarse scalar defaults and wrongly heal a washing-machine-equivalent
        # entry's 30/5 up to 61/30.
        _healed = _heal_seeded_cadence(
            new_opts,
            new_opts.get(CONF_DEVICE_TYPE) or entry.data.get(CONF_DEVICE_TYPE),
        )
        # LITERAL 10, not CONFIG_ENTRY_MINOR_VERSION, like every other step. These
        # blocks form a chain - each advances minor_version to exactly N+1 so the next
        # block picks it up - so a step that wrote "whatever is current" would, after a
        # future bump to 11, jump a 3.9 entry straight to 11 and skip the new 3.10->3.11
        # step entirely. Only the one-pass legacy write at the end means "land on
        # current" and uses the constant.
        hass.config_entries.async_update_entry(
            entry, options=new_opts, minor_version=10
        )
        minor_version = 10
        _log.debug(
            "Migrated WashData entry from 3.9 to 3.10 (healed seeded cadence "
            "defaults: %s)",
            _healed or "none",
        )

    # 3.10 -> 3.11: drop the abrupt-drop end-detection knobs from options. The bulk
    # strip below reaches only entries still BELOW the current schema, because of the
    # early return right under this block - so without a step of its own an entry
    # created before 558e71e and since migrated to 3.10 keeps the dead keys forever.
    # Same shape as the 3.7 -> 3.8 running_dead_zone retirement, for the same reason.
    if version == 3 and minor_version == 10:
        # Computed BEFORE the update, like the 3.8 -> 3.9 step: async_update_entry
        # replaces entry.options synchronously, so an intersection taken after it
        # is always empty and every migration would log "removed nothing".
        removed = sorted(set(entry.options) & _DEAD_ABRUPT_KEYS)
        new_opts = {k: v for k, v in entry.options.items() if k not in _DEAD_ABRUPT_KEYS}
        # Heal the Stage-1 upper gate that item 311 widened 1.5 -> 1.8. The
        # legacy migration SEEDS this key with `options.setdefault(...,
        # DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO)`, so every entry migrated
        # before that change has a literal 1.5 frozen in its options and the
        # manager reads it in preference to the new default - i.e. the widening
        # reaches nobody who already had WashData installed. It is not rare:
        # 13 of the 33 real exports in `cycle_data/` carry exactly 1.5.
        #
        # Replaced ONLY where it still equals the old seeded default, exactly as
        # the 3.9 -> 3.10 cadence heal is scoped: a user who tuned this produced
        # a value like 1.44 or 1.51, not the old constant on the nose. Item 311
        # measured the wider gate as 4 appliances better and none worse.
        healed: list[str] = []
        if new_opts.get(CONF_PROFILE_MATCH_MAX_DURATION_RATIO) == _OLD_SEEDED_MAX_DURATION_RATIO:
            new_opts[CONF_PROFILE_MATCH_MAX_DURATION_RATIO] = (
                DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
            )
            healed.append(CONF_PROFILE_MATCH_MAX_DURATION_RATIO)
        hass.config_entries.async_update_entry(
            entry, options=new_opts, minor_version=11
        )
        minor_version = 11
        _log.debug(
            "Migrated WashData entry from 3.10 to 3.11 (removed %s; healed %s)",
            removed or "nothing",
            healed or "nothing",
        )

    if version == CONFIG_ENTRY_VERSION and minor_version >= CONFIG_ENTRY_MINOR_VERSION:
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
    # setdefault does not replace a present-but-null value; a null device type would
    # then miss the per-type maps and fall through to the coarse scalar defaults (30/61)
    # instead of the washing-machine defaults DEFAULT_DEVICE_TYPE stands for (5/30).
    # Coerce it here (an explicit type is truthy and preserved) so both resolvers below
    # see a real device type.
    if not options.get(CONF_DEVICE_TYPE):
        options[CONF_DEVICE_TYPE] = DEFAULT_DEVICE_TYPE
    options.setdefault(
        CONF_START_DURATION_THRESHOLD,
        resolve_start_duration_default(options[CONF_DEVICE_TYPE]),
    )

    options.setdefault(CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL)
    options.setdefault(
        CONF_PROFILE_MATCH_MIN_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO
    )
    # CONF_PROFILE_MATCH_MAX_DURATION_RATIO is deliberately NOT seeded here.
    # Seeding it is what produced `_OLD_SEEDED_MAX_DURATION_RATIO` and the two
    # heal sites above: entries took the then-default 1.5 as an explicit value,
    # so item 311's widening to 1.8 reached none of them. Every reader resolves
    # the key with `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO` as its fallback
    # and `ws_get_options` ships `defaults` alongside `options`, so an absent
    # key behaves identically today and the next change to the default actually
    # lands. The min ratio above keeps its seed because that gate is inert for
    # ranking (it has never removed a true candidate on the corpus), so freezing
    # it costs nothing.
    options.setdefault(
        CONF_WATCHDOG_INTERVAL,
        resolve_watchdog_interval_default(options[CONF_DEVICE_TYPE]),
    )
    options.setdefault(
        CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD, DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD
    )
    # The device's own floor, not the scalar 600 s: seeded for a pump it made every
    # run shorter than 10 min `interrupted` (audit DETECT-07).
    options.setdefault(
        CONF_COMPLETION_MIN_SECONDS,
        DEVICE_COMPLETION_THRESHOLDS.get(
            options[CONF_DEVICE_TYPE], DEFAULT_COMPLETION_MIN_SECONDS
        ),
    )
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

    # The abrupt-drop end-detection knobs were removed in 558e71e (the state
    # machine reached the same decision from the energy gates), but nothing ever
    # stripped them, so they still sit in the options of entries created before
    # that. Measured across the real exports in cycle_data/: present in 5 of the
    # 7 full user configurations and read by no Python or panel code. The
    # surviving "abrupt end" in suggestion_engine is the cycle-artifact
    # classifier, which is unrelated hardcoded logic, not these tunables.
    # Entries already on the current schema are handled by the 3.10 -> 3.11 step
    # above; this covers a one-pass legacy migration.
    for k in _DEAD_ABRUPT_KEYS:
        options.pop(k, None)

    # Same heal as the 3.10 -> 3.11 step, for the cohort that never reaches it:
    # an entry below 3.6 (including v1/v2) takes this one-pass path instead of
    # the chain, and the `setdefault` above preserves a stored 1.5 rather than
    # replacing it. Without this those entries keep the pre-item-311 gate.
    if options.get(CONF_PROFILE_MATCH_MAX_DURATION_RATIO) == _OLD_SEEDED_MAX_DURATION_RATIO:
        options[CONF_PROFILE_MATCH_MAX_DURATION_RATIO] = (
            DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
        )

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

    # Same heal as the 3.9 -> 3.10 step (audit PLATFORM-09), after the remap so it
    # resolves against the device type the entry ends up on. The `setdefault` above
    # keeps a stored 30/5, which is exactly what the chain heals.
    _heal_seeded_cadence(options, options.get(CONF_DEVICE_TYPE))

    # Strip the now-retired running_dead_zone from options in the bulk path too
    # (covers entries that migrate straight from v1/v2/early-v3 in one pass).
    options.pop(CONF_RUNNING_DEAD_ZONE, None)

    # Same for an option persisted as null (3.8 -> 3.9), so a one-pass legacy
    # migration lands on the current schema rather than needing a second call.
    options = strip_null_options(options)

    hass.config_entries.async_update_entry(
        entry,
        data=data,
        options=options,
        version=CONFIG_ENTRY_VERSION,
        minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    _log.info(
        "Migrated WashData entry from version %s.%s to %s.%s",
        version, minor_version, CONFIG_ENTRY_VERSION, CONFIG_ENTRY_MINOR_VERSION,
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


async def _async_preload_ml_modules(hass: HomeAssistant) -> None:
    """Import the ML modules off the event loop (issue #328).

    ``ml.engine.resolve_scorer`` / ``resolve_regressor`` are called from the event
    loop (end detection, ETA / energy projection), and Home Assistant flags
    the lazy ``importlib.import_module`` they used to do there as a blocking call.
    Warming the module cache once per setup in the import executor makes every
    later resolution a ``sys.modules`` lookup. Best effort: a failure here only
    means ML stays inert, so it must never block setup.

    Two things matter for startup time here (issue #408), because
    ``hass.import_executor`` is ``max_workers=1`` and is shared with every other
    integration importing during startup, so an awaited job on it costs however
    deep that queue happens to be - measured at 35-95 s in real user
    diagnostics, against ~1 ms of actual work:

    1. The job is coalesced onto ONE shared future for the whole HA instance.
       ``preload_models()`` is idempotent, so a second job per config entry
       would buy nothing but another full trip through that queue.
    2. The wait is wrapped in ``async_pause_setup(WAIT_IMPORT_PACKAGES)``, which
       is how HA core reports its own heavy imports (``workday``, ``holiday``,
       ``stream``, ``mqtt``, ...): the queue wait is credited back instead of
       being billed to us as "Integration startup time".
    """

    def _preload() -> None:
        # pylint: disable=import-outside-toplevel
        from .ml.engine import preload_models

        preload_models()

    future = hass.data.get(ML_PRELOAD_FUTURE_KEY)
    if future is None:
        future = hass.data[ML_PRELOAD_FUTURE_KEY] = hass.async_add_import_executor_job(
            _preload
        )

    try:
        with async_pause_setup(hass, SetupPhases.WAIT_IMPORT_PACKAGES):
            # shield: a cancelled entry setup must not cancel the warm-up that
            # the other entries are waiting on.
            await asyncio.shield(future)
    except Exception:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("ML module preload failed", exc_info=True)
        # Broken install: drop the memo so a later setup/reload retries instead
        # of every entry re-raising the one cached failure forever. A cancelled
        # setup is not caught here (CancelledError is not an Exception), so the
        # memo correctly survives for the entries still awaiting the warm-up.
        if hass.data.get(ML_PRELOAD_FUTURE_KEY) is future:
            hass.data.pop(ML_PRELOAD_FUTURE_KEY, None)


async def _async_setup_shared(
    hass: HomeAssistant, log: DeviceLoggerAdapter
) -> None:
    """Register everything that belongs to the HA instance, not to one appliance.

    The custom card, the sidebar panel, the WebSocket API, the conversation
    intents and the cached integration version are global: the first config
    entry brings them up and every later entry is a guarded no-op.

    Issue #425: this used to sit near the END of ``async_setup_entry``, so any
    earlier per-appliance failure took the whole install's UI down with it - and
    because the panel is where almost every setting is edited (including the
    ``via_device`` link that aborted setup in #418), losing it left the user with
    no way to undo the setting that broke setup. Registering it first makes the
    panel independent of whether any individual appliance sets up.

    Failures inside are already handled per block (the card defers and retries,
    the panel logs and retries on the next setup), so a missing UI never fails
    the entry.
    """
    # Files left behind by appliances deleted before 0.5.8 (no remove hook then):
    # swept once per start, in the background, so setup never waits on disk I/O.
    if not hass.data.get(_ORPHAN_SWEEP_KEY):
        hass.data[_ORPHAN_SWEEP_KEY] = True
        hass.async_create_background_task(
            _async_sweep_orphaned_stores(hass), "ha_washdata orphaned store sweep"
        )

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
            log.warning("Card registration failed, will retry on next setup: %s", err)
        else:
            if register_result == CARD_REGISTERED:
                hass.data["ha_washdata_card_deferred"] = False
                hass.data["ha_washdata_card_registered"] = True
            elif register_result == CARD_DEFERRED:
                hass.data["ha_washdata_card_deferred"] = True
                hass.data["ha_washdata_card_registered"] = False
            else:
                hass.data["ha_washdata_card_deferred"] = False
                hass.data["ha_washdata_card_registered"] = False
                # The reason used to be debug-only inside frontend.py, so this
                # warning told the user something was wrong but not what (#432).
                log.warning(
                    "Card registration failed and was not deferred: %s",
                    getattr(card_reg, "last_failure_reason", "unknown"),
                )
        finally:
            # finally, not one reset per branch: `except Exception` does not catch
            # CancelledError, and HA cancels entry setup on timeout or on a reload
            # racing it. A flag left True is never cleared again, so every later
            # setup skips card registration until HA restarts.
            hass.data["ha_washdata_card_registering"] = False

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

    # Cache the integration version from HA's already-loaded manifest (no file IO).
    # ws_get_constants reads it from hass.data; we can't do a module-level read_text
    # in ws_api.py because the module is imported lazily inside this coroutine and the
    # IO runs on the event loop (#328/#335).
    if "ha_washdata_version" not in hass.data:
        try:
            from homeassistant.loader import async_get_integration as _aget_integration  # pylint: disable=import-outside-toplevel
            _integ = await _aget_integration(hass, DOMAIN)
            hass.data["ha_washdata_version"] = _integ.manifest.get("version", "") or ""
        except Exception as err:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Could not load ha_washdata version from manifest: %s", err)
            # Fall back to reading manifest.json off the event loop.  Only
            # write the key when the read succeeds so that a double failure
            # (loader + file) leaves the key absent rather than storing ""
            # (which would shadow _INTEGRATION_VERSION in ws_get_constants).
            def _read_manifest_version() -> str:
                try:
                    return json.loads(
                        (Path(__file__).parent / "manifest.json").read_text(encoding="utf-8")
                    ).get("version") or ""
                except Exception:  # pylint: disable=broad-exception-caught
                    return ""
            _fallback_version = await hass.async_add_executor_job(_read_manifest_version)
            if _fallback_version:
                hass.data["ha_washdata_version"] = _fallback_version

    async_register_commands(hass)
    hass.data["ha_washdata_ws_registered"] = True

    # Register conversation intents (e.g. "is my washer done?") - once per HA
    # instance. Intents are domain-global, so guard against re-registration when
    # more than one device is configured.
    if not hass.data.get("ha_washdata_intents_registered"):
        from .intents import async_setup_intents  # pylint: disable=import-outside-toplevel

        async_setup_intents(hass)
        hass.data["ha_washdata_intents_registered"] = True


# Panel access levels, lowest first (ws_api._effective_level resolves a user's).
_SERVICE_LEVEL_ORDER = {"none": 0, "read": 1, "edit": 2, "full": 3}

# Services that read or write files / replace a whole device's data: admin-only,
# like their WS twins in ws_api._ADMIN_COMMANDS.
_ADMIN_SERVICES = frozenset({"export_config", "import_config", "trigger_ml_training"})


def _service_entry_ids(hass: HomeAssistant, call: ServiceCall) -> list[str | None]:
    """Every config entry a service call names: its ``entry_id`` and its device's.

    Both are returned, and the caller authorizes each, because handlers differ in
    which one they act on (``submit_cycle_feedback`` prefers ``entry_id``, every
    other handler resolves ``device_id``, and the schemas allow extra keys). Checking
    only ``entry_id`` let a user with edit access to entry B pass B's id beside
    entry A's device and change A. ``[None]`` when the call names no entry.
    """
    ids: list[str | None] = []
    entry_id = call.data.get("entry_id")
    if isinstance(entry_id, str) and entry_id:
        ids.append(entry_id)
    device_id = call.data.get("device_id")
    if isinstance(device_id, str) and device_id:
        device = dr.async_get(hass).async_get(device_id)
        if device is not None:
            loaded = hass.data.get(DOMAIN, {})
            dev_entry = next(
                (e for e in device.config_entries if e in loaded),
                next(iter(device.config_entries), None),
            )
            if dev_entry is not None and dev_entry not in ids:
                ids.append(dev_entry)
    return ids or [None]


async def _async_check_service_access(
    hass: HomeAssistant, call: ServiceCall, level: str
) -> None:
    """Authorize a service call the way the panel authorizes its WS twin.

    Every service used to skip both the WS admin gate and the panel RBAC: a
    read-only user could run ``import_config`` (replace all data) or
    ``export_config`` (write the store under ``/config/www``, served without
    login at ``/local/``), and an RBAC "read" user could delete profiles
    (audit PLATFORM-03). Calls without a user (automations, scripts, the system)
    stay allowed, as for every HA admin service.
    """
    user_id = call.context.user_id
    if user_id is None:
        return
    user = await hass.auth.async_get_user(user_id)
    if user is None:
        raise Unauthorized(context=call.context)
    if user.is_admin:
        return
    if level == "admin":
        raise Unauthorized(context=call.context)
    from .ws_api import _effective_level  # pylint: disable=import-outside-toplevel

    for entry_id in _service_entry_ids(hass, call):
        granted = _effective_level(hass, user, entry_id)
        if _SERVICE_LEVEL_ORDER.get(granted, 0) < _SERVICE_LEVEL_ORDER.get(level, 2):
            raise Unauthorized(context=call.context)


def _guarded_service(hass: HomeAssistant, name: str, handler: Any) -> Any:
    """Wrap a service handler with the access check for its level."""
    level = "admin" if name in _ADMIN_SERVICES else "edit"

    async def _wrapped(call: ServiceCall) -> Any:
        await _async_check_service_access(hass, call, level)
        return await handler(call)

    return _wrapped


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up WashData from a config entry."""
    _log = DeviceLoggerAdapter(_LOGGER, entry.title)
    hass.data.setdefault(DOMAIN, {})

    # Panel / card / WebSocket API first: they belong to the HA instance, not to
    # this appliance, so they must not depend on this entry getting through
    # (#425).  See _async_setup_shared.
    await _async_setup_shared(hass, _log)

    # Debris from an earlier setup that failed part-way.  HA does NOT call
    # async_unload_entry for an entry that never reached LOADED, so the manager
    # this function stores before the failure point is still here - and before
    # #425 we answered that by logging "already set up" and returning True,
    # which told HA the entry was loaded while everything after the failure
    # point (device link, half the services, the update listener) had never
    # run.  Every later reload repeated the lie, so the only way out was to
    # delete and re-add the appliance.  Throw the leftover away and set up for
    # real instead.
    stale = hass.data[DOMAIN].pop(entry.entry_id, None)
    if stale is not None:
        _log.warning(
            "Entry %s still holds a manager from an earlier incomplete setup; "
            "discarding it and setting up again",
            entry.entry_id,
        )
        try:
            await stale.async_shutdown()
        except Exception:  # pylint: disable=broad-exception-caught
            _log.debug("Shutdown of the stale manager failed", exc_info=True)

    # Entity platforms from an earlier attempt are still registered (HA leaves
    # forwarded platforms in place when setup fails); forwarding them again raises
    # "has already been setup", so take them down first.
    #
    # NOT gated on the stale manager: FORWARDED_ENTRIES_KEY is the independent
    # record of a forward, and the two can part company. async_reload_entry's
    # full-reload branch runs precisely when the manager is missing - it calls
    # async_unload_entry, which leaves the marker in place when the platform unload
    # is refused, and then calls this function with nothing to find. Gating the
    # cleanup on `stale` skipped it in exactly the case it exists for.
    if entry.entry_id in hass.data.get(FORWARDED_ENTRIES_KEY, set()):
        unloaded = False
        try:
            unloaded = await hass.config_entries.async_unload_platforms(
                entry, PLATFORMS
            )
        except Exception:  # pylint: disable=broad-exception-caught
            _log.debug("Unloading the stale platforms failed", exc_info=True)
        # Only forget the platforms once they are actually gone. A platform that
        # refuses to unload leaves them forwarded, and dropping the record here
        # would make the next attempt skip the unload and hit "has already been
        # setup" forever - the #425 loop this set exists to break.
        #
        # Setup deliberately CARRIES ON when the unload does not succeed, rather
        # than returning False. `False` does not mean "a live platform refused":
        # HA's ConfigEntry.async_unload catches a never-loaded platform's
        # ValueError("Config entry was never loaded!"), logs "Error unloading entry
        # X for sensor" and returns False (`config_entries.py:997-1015`). Since the
        # marker is written BEFORE the forward on purpose, the common instance of
        # False is "the forward set nothing up", where there is nothing registered
        # to collide with and the retry succeeds. Aborting on False would abort
        # every later attempt identically - a permanent brick, the exact #425 loop
        # this set exists to break, and it is what
        # `test_reload_after_a_failed_setup_really_sets_up` and
        # `test_an_unloadable_stale_platform_does_not_abort_the_retry` pin.
        if unloaded:
            hass.data[FORWARDED_ENTRIES_KEY].discard(entry.entry_id)

    # Warm the ML module cache before anything can score in the event loop.
    await _async_preload_ml_modules(hass)

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

    # Heal a non-numeric numeric setting stored by an older version (audit
    # PLATFORM-13 / register item 279): it raised in the manager's constructor,
    # so the entry could never set up again. Dropped, so its default applies.
    from .const import drop_invalid_numeric_options  # pylint: disable=import-outside-toplevel

    _clean, _dropped = drop_invalid_numeric_options(dict(entry.options))
    if _dropped:
        _log.warning(
            "Dropped non-numeric value(s) for %s; their defaults apply", ", ".join(_dropped)
        )
        hass.config_entries.async_update_entry(entry, options=_clean)

    # pylint: disable=import-outside-toplevel
    from .manager import WashDataManager

    manager = WashDataManager(hass, entry)
    hass.data[DOMAIN][entry.entry_id] = manager

    await manager.async_setup()
    await _migrate_online_to_global(hass, entry, manager)

    # Recorded BEFORE the await on purpose: async_forward_entry_setups gathers the
    # platform setups, so one raising leaves the others set up, and a marker written
    # afterwards would never be reached. The next attempt would then skip the unload
    # and forward an already-registered platform (#425). Marking a forward that in
    # fact set nothing up is harmless: the unload of a never-loaded platform raises,
    # that is caught, and the retry forwards normally.
    hass.data.setdefault(FORWARDED_ENTRIES_KEY, set()).add(entry.entry_id)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    _apply_device_link(hass, entry)

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    # Register service if not already
    if not hass.services.has_service(DOMAIN, "label_cycle"):

        async def handle_label_cycle(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            cycle_id = _require_str(call.data.get("cycle_id"), "cycle_id")
            profile_name = (call.data.get("profile_name") or "").strip()

            # Find the config entry for this device
            entry_id, manager = _service_manager(hass, device_id)

            # Assign existing profile or remove label
            target_profile = profile_name if profile_name else None
            try:
                await manager.profile_store.assign_profile_to_cycle(
                    cycle_id, target_profile
                )
            except ValueError as exc:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="assign_profile_failed",
                    translation_placeholders={"error": str(exc)},
                ) from exc

            # A manual (re)label answers the "was this detected right?" question,
            # so clear any pending feedback and drop it from the review queue (#331).
            if hasattr(manager, "learning_manager"):
                await manager.learning_manager.async_resolve_pending_from_label(
                    cycle_id, target_profile
                )

            manager.notify_update()

        hass.services.async_register(
            DOMAIN, "label_cycle", _guarded_service(hass, "label_cycle", handle_label_cycle),
            schema=_SERVICE_SCHEMAS["label_cycle"]
        )

    # Register create_profile service
    if not hass.services.has_service(DOMAIN, "create_profile"):

        async def handle_create_profile(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            profile_name = _require_str(call.data.get("profile_name"), "profile_name")
            reference_cycle_id = call.data.get("reference_cycle_id")

            entry_id, manager = _service_manager(hass, device_id)
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

        hass.services.async_register(
            DOMAIN, "create_profile", _guarded_service(hass, "create_profile", handle_create_profile),
            schema=_SERVICE_SCHEMAS["create_profile"]
        )

    # Register delete_profile service
    if not hass.services.has_service(DOMAIN, "delete_profile"):

        async def handle_delete_profile(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            profile_name = _require_str(call.data.get("profile_name"), "profile_name")
            unlabel_cycles = call.data.get("unlabel_cycles", True)

            entry_id, manager = _service_manager(hass, device_id)
            await manager.profile_store.delete_profile(profile_name, unlabel_cycles)
            manager.notify_update()

        hass.services.async_register(
            DOMAIN, "delete_profile", _guarded_service(hass, "delete_profile", handle_delete_profile),
            schema=_SERVICE_SCHEMAS["delete_profile"]
        )

    # Register auto_label_cycles service
    if not hass.services.has_service(DOMAIN, "auto_label_cycles"):

        async def handle_auto_label_cycles(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            confidence_threshold = call.data.get("confidence_threshold")

            entry_id, _manager = _service_manager(hass, device_id)

            # The same registry task the panel starts (audit PLATFORM-05): under
            # the write lock, visible as a header pill, cancellable; awaited so an
            # automation step still waits for the result.
            from .ws_api import (  # noqa: PLC0415
                configured_auto_label_threshold,
                start_auto_label_task,
            )

            # Omitted: the device's own Auto-Label Confidence, as the WS command
            # does (audit UI-10), not a hardcoded 0.75 that ignored the setting.
            if confidence_threshold is None:
                confidence_threshold = configured_auto_label_threshold(
                    hass.config_entries.async_get_entry(entry_id)
                )

            _task, raw = start_auto_label_task(hass, entry_id, float(confidence_threshold))
            if raw is not None:
                await raw

        hass.services.async_register(
            DOMAIN,
            "auto_label_cycles",
            _guarded_service(hass, "auto_label_cycles", handle_auto_label_cycles),
            schema=_SERVICE_SCHEMAS["auto_label_cycles"],
        )

    # Register trim_cycle service
    if not hass.services.has_service(DOMAIN, "trim_cycle"):

        async def handle_trim_cycle(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            cycle_id = _require_str(call.data.get("cycle_id"), "cycle_id")
            trim_start_s = max(0.0, float(call.data.get("trim_start_s", 0)))

            entry_id, manager = _service_manager(hass, device_id)
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

        hass.services.async_register(
            DOMAIN, "trim_cycle", _guarded_service(hass, "trim_cycle", handle_trim_cycle),
            schema=_SERVICE_SCHEMAS["trim_cycle"]
        )

    # Belt and braces for the hoist above: the panel's static routes need
    # hass.http. It is a hard manifest dependency (item 487), so HA sets it up
    # before this entry; a caller that invokes async_setup_entry directly skips
    # that, and there the platform forward is what processes the manifest.
    # Every step guards on its own "already done" flag, so this is a no-op
    # whenever the early call did its job.
    await _async_setup_shared(hass, _log)

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
                entry_id, _manager = _service_manager(hass, device_id)

            cycle_id = _require_str(call.data.get("cycle_id"), "cycle_id")
            user_confirmed = call.data.get("user_confirmed", False)
            corrected_profile = call.data.get("corrected_profile")
            corrected_duration = call.data.get("corrected_duration")  # in seconds
            notes = call.data.get("notes") or ""
            dismiss = call.data.get("dismiss", False)

            if entry_id not in hass.data[DOMAIN]:
                raise ServiceValidationError(
                    translation_domain=DOMAIN, translation_key="integration_not_loaded"
                )

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
            _guarded_service(hass, "submit_cycle_feedback", handle_submit_feedback),
            schema=_SERVICE_SCHEMAS["submit_cycle_feedback"],
        )

    # Export store to file (per entry/device)
    if not hass.services.has_service(DOMAIN, "export_config"):

        async def handle_export_config(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            file_path = call.data.get("path")

            entry_id, manager = _service_manager(hass, device_id)
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
            # Serialised like the WS export (audit PERF-11): HA's orjson encoder,
            # compact. indent=2 put every power-trace number on its own line.
            from .ws_api import _export_json  # noqa: PLC0415

            def _dump_and_write():
                text = _export_json(payload)
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

        hass.services.async_register(
            DOMAIN, "export_config", _guarded_service(hass, "export_config", handle_export_config),
            schema=_SERVICE_SCHEMAS["export_config"]
        )

    # Import store from file into the target entry/device
    if not hass.services.has_service(DOMAIN, "import_config"):

        async def handle_import_config(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            file_path = call.data.get("path")

            if not file_path:
                raise ValueError("path is required for import")

            entry_id, manager = _service_manager(hass, device_id)
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

            # Same path as the WS import (audit PLATFORM-02): under the per-entry
            # write lock, local sensor/device bindings dropped, options written
            # under their lock with a changelog entry, entry.data left alone. The
            # service used to copy the exporter's power and door sensors over this
            # device's, which silently disconnects it (register item 317).
            from .ws_api import (  # pylint: disable=import-outside-toplevel
                _entry_write_lock,
                async_apply_imported_entry_options,
            )

            async with _entry_write_lock(hass, entry_id):
                # This device's options ride along so "Undo last import" in the
                # panel can put them back (register item 195).
                config_updates = await manager.profile_store.async_import_data(
                    payload,
                    entry_options=dict(entry.options),
                    source="import_config_service",
                )
                if config_updates:
                    await async_apply_imported_entry_options(
                        hass, entry, config_updates, "import_config_service"
                    )
                    manager._logger.info(
                        "Applied imported settings to config entry %s", entry_id
                    )

            # An old payload re-arms the one-time banked-tail repair. Nothing here
            # reloads the entry unless the payload brought settings with it, so
            # schedule it directly rather than leaving it for the next restart.
            manager.async_schedule_banked_tail_repair()

            manager._logger.info("Imported ha_washdata entry %s from %s", entry_id, source)

        hass.services.async_register(
            DOMAIN, "import_config", _guarded_service(hass, "import_config", handle_import_config),
            schema=_SERVICE_SCHEMAS["import_config"]
        )

    # Register recorder services
    if not hass.services.has_service(DOMAIN, "record_start"):
        async def handle_record_start(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            entry_id, manager = _service_manager(hass, device_id)
            await manager.async_start_recording()

        hass.services.async_register(
            DOMAIN, "record_start", _guarded_service(hass, "record_start", handle_record_start),
            schema=_SERVICE_SCHEMAS["record_start"]
        )

    if not hass.services.has_service(DOMAIN, "record_stop"):
        async def handle_record_stop(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            entry_id, manager = _service_manager(hass, device_id)
            await manager.async_stop_recording()

        hass.services.async_register(
            DOMAIN, "record_stop", _guarded_service(hass, "record_stop", handle_record_stop),
            schema=_SERVICE_SCHEMAS["record_stop"]
        )

    # Register on-device ML training trigger (Stage 4, gated by ENABLE_ML_TRAINING)
    from .const import ENABLE_ML_TRAINING, SERVICE_TRIGGER_ML_TRAINING

    if ENABLE_ML_TRAINING and not hass.services.has_service(
        DOMAIN, SERVICE_TRIGGER_ML_TRAINING
    ):
        async def handle_trigger_ml_training(call: ServiceCall) -> None:
            device_id = _require_str(call.data.get("device_id"), "device_id")
            entry_id, manager = _service_manager(hass, device_id)
            summary = await manager.async_run_ml_training(force=True)
            manager._logger.info("Manual ML training: %s", summary)

        hass.services.async_register(
            DOMAIN,
            SERVICE_TRIGGER_ML_TRAINING,
            _guarded_service(hass, "trigger_ml_training", handle_trigger_ml_training),
            schema=_SERVICE_SCHEMAS["trigger_ml_training"],
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

        hass.services.async_register(
            DOMAIN, "pause_cycle", _guarded_service(hass, "pause_cycle", handle_pause_cycle),
            schema=_SERVICE_SCHEMAS["pause_cycle"]
        )

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

        hass.services.async_register(
            DOMAIN, "resume_cycle", _guarded_service(hass, "resume_cycle", handle_resume_cycle),
            schema=_SERVICE_SCHEMAS["resume_cycle"]
        )

    # Unload confirmation for a device with no door sensor (#451). Deliberately
    # not an error when nothing is waiting: an automation wired to a physical
    # button fires on every press, and "already emptied" is not a failure.
    if not hass.services.has_service(DOMAIN, "mark_unloaded"):
        async def handle_mark_unloaded(call: ServiceCall) -> None:
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

            hass.data[DOMAIN][entry_id].mark_unloaded("mark_unloaded service")

        hass.services.async_register(
            DOMAIN, "mark_unloaded", _guarded_service(hass, "mark_unloaded", handle_mark_unloaded),
            schema=_SERVICE_SCHEMAS["mark_unloaded"]
        )

    return True


def _apply_device_link(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Sync the WashData device's via_device link with the configured option.

    When CONF_LINKED_DEVICE points at an existing device (e.g. the smart plug or
    appliance), the WashData device is shown as "Connected via <device>" in the
    HA device registry. Clearing the option removes the link. Stale targets that
    no longer exist - and a target that is this entry's own WashData device, which
    HA rejects as a self-reference (#418) - are treated as "no link" so the
    registry never references a deleted device or itself.
    """
    _log = DeviceLoggerAdapter(_LOGGER, entry.title)
    registry = dr.async_get(hass)
    identifier = (DOMAIN, entry.entry_id)
    if hasattr(registry, "async_get_device_by_identifier"):
        # HA 2026.9+ deprecated async_get_device because identifiers are no longer
        # unique across config entries (issue #405). Our device's identifier is
        # owned by this entry, so the by-identifier lookup is unambiguous. The
        # attribute guard keeps us working on the older HA the manifest still
        # supports, where the new method does not exist yet.
        washdata_device = registry.async_get_device_by_identifier(
            identifier, entry.entry_id
        )
    else:
        washdata_device = registry.async_get_device(identifiers={identifier})
    if washdata_device is None:
        return

    linked_device_id = entry.options.get(CONF_LINKED_DEVICE) or None
    if linked_device_id and registry.async_get(linked_device_id) is None:
        linked_device_id = None
    if linked_device_id and linked_device_id == washdata_device.id:
        # A device may not be its own via_device: HA 2026.9 raises
        # HomeAssistantError instead of silently accepting it, and this runs inside
        # async_setup_entry - so a self-link aborted setup for the whole entry
        # (#418). The picker used to list this entry's own WashData device, whose
        # name mirrors the entry title (and often the plug's), so it was easy to
        # select by mistake. Treat it as "no link" and fall through to the update
        # below: on an older HA the self-reference may already be stored in the
        # registry, and clearing it is exactly the repair needed.
        _log.warning(
            "Ignoring 'Group Under Device': %s is this appliance's own WashData "
            "device and a device cannot be linked to itself. Pick the smart plug "
            "(or another device) instead, or clear the setting.",
            linked_device_id,
        )
        linked_device_id = None

    if washdata_device.via_device_id != linked_device_id:
        try:
            registry.async_update_device(
                washdata_device.id, via_device_id=linked_device_id
            )
        except (HomeAssistantError, ValueError, OverflowError) as err:
            # The via_device link is cosmetic (it only nests the device in the HA
            # registry UI). A registry rule we do not know about yet must never be
            # able to take the whole entry down with it, as the self-link did in
            # #418 - log it and leave the device standalone.
            _log.warning(
                "Could not link WashData device to %s: %s",
                linked_device_id or "(none)",
                err,
            )


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry - update settings without interrupting running cycles."""
    manager = hass.data[DOMAIN].get(entry.entry_id)
    if manager:
        # Update configuration without interrupting detector
        await manager.async_reload_config(entry)
        # Options changes (e.g. linked device) reload in place without
        # recreating entities, so apply the device link explicitly here, and add or
        # drop the pump-only sensor if the device type changed.
        _apply_device_link(hass, entry)
        from .sensor import async_reconcile_device_type_sensors  # pylint: disable=import-outside-toplevel

        async_reconcile_device_type_sensors(hass, manager, entry)
    else:
        # Full reload if manager not found
        await async_unload_entry(hass, entry)
        await async_setup_entry(hass, entry)


_ORPHAN_SWEEP_KEY = "ha_washdata_orphan_sweep"
# Per-appliance store keys: the profile store, its active-cycle snapshot (0.5.8), its
# pre-import restore point (register item 195), its held-notification queue (audit
# MANAGER-16), its last failed snapshot restore (item 266) and the manual recorder. Global keys
# (``ha_washdata_panel``, ``ha_washdata_online``) use an underscore and never match.
_ENTRY_STORE_RE = re.compile(
    r"^ha_washdata\.(?:recorder\.)?([0-9A-Za-z]{20,40})"
    rf"(?:\.active|\.{PRE_IMPORT_STORE_SUFFIX}|\.{NOTIFY_QUEUE_STORE_SUFFIX}"
    rf"|\.{FAILED_RESTORE_STORE_SUFFIX})?$"
)


def _entry_store_keys(entry_id: str) -> list[str]:
    return [
        f"{STORAGE_KEY}.{entry_id}",
        f"{STORAGE_KEY}.{entry_id}.active",
        f"{STORAGE_KEY}.{entry_id}.{PRE_IMPORT_STORE_SUFFIX}",
        f"{STORAGE_KEY}.{entry_id}.{NOTIFY_QUEUE_STORE_SUFFIX}",
        f"{STORAGE_KEY}.{entry_id}.{FAILED_RESTORE_STORE_SUFFIX}",
        f"{STORAGE_KEY}.recorder.{entry_id}",
    ]


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete an appliance's stored data when the user deletes the appliance.

    Until 0.5.8 there was no remove hook, so every deleted appliance left its
    profiles, cycles and traces in ``.storage`` for good (6.9 MB from 15 deleted
    devices on one install). HA calls this only for a deliberate delete, after the
    entry has unloaded; an unload or a reload never reaches it.
    """
    from homeassistant.helpers.storage import Store  # noqa: PLC0415

    for key in _entry_store_keys(entry.entry_id):
        try:
            await Store(hass, 1, key).async_remove()
        except Exception:  # noqa: BLE001 - a leftover file must not block the delete
            _LOGGER.warning("Could not delete WashData store %s", key, exc_info=True)


async def _async_sweep_orphaned_stores(hass: HomeAssistant) -> None:
    """Delete per-appliance store files whose config entry no longer exists.

    Every entry (loaded, disabled or failed) is listed by the config-entries
    manager before any integration sets up, so a file whose id matches none of them
    belongs to an appliance the user deleted. Never raises.
    """
    try:
        known = {e.entry_id for e in hass.config_entries.async_entries(DOMAIN)}
        if not known:
            return
        storage_dir = hass.config.path(".storage")

        def _orphans() -> list[tuple[str, int]]:
            out = []
            for name in os.listdir(storage_dir):
                m = _ENTRY_STORE_RE.match(name)
                if m and m.group(1) not in known:
                    try:
                        out.append((name, os.path.getsize(os.path.join(storage_dir, name))))
                    except OSError:
                        continue
            return out

        orphans = await hass.async_add_executor_job(_orphans)
        if not orphans:
            return
        from homeassistant.helpers.storage import Store  # noqa: PLC0415

        removed = 0
        for name, _size in orphans:
            try:
                await Store(hass, 1, name).async_remove()
                removed += 1
            except Exception:  # noqa: BLE001
                _LOGGER.debug("Could not delete orphaned store %s", name, exc_info=True)
        _LOGGER.info(
            "Deleted %d WashData store file(s) left by deleted appliances (%.1f MB)",
            removed, sum(size for _n, size in orphans) / 1e6,
        )
    except Exception:  # noqa: BLE001
        _LOGGER.debug("Orphaned store sweep failed", exc_info=True)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data.get(FORWARDED_ENTRIES_KEY, set()).discard(entry.entry_id)
        manager = hass.data[DOMAIN].pop(entry.entry_id)
        await manager.async_shutdown()

        # Settle registry: mark any still-RUNNING tasks for this entry as
        # cancelled so they don't appear as zombies after reload.
        from . import task_registry as _task_registry
        _cancelled_tasks = _task_registry.get_registry(hass).cancel_entry_tasks(entry.entry_id)
        # Drain cancelled WS-spawned tasks so their finally blocks (lock releases)
        # complete before we remove the write lock below.
        if _cancelled_tasks:
            await asyncio.gather(*_cancelled_tasks, return_exceptions=True)

        # Release the per-entry locks so they don't block the next setup. BOTH of
        # them: `_entry_options_lock` is a second per-entry lock created the same
        # way, and dropping only the write lock left one asyncio.Lock per removed
        # entry in hass.data for the lifetime of the process.
        from .ws_api import (
            _WS_OPTIONS_LOCKS_KEY,
            _WS_WRITE_LOCKS_KEY,
            async_clear_history_import,
        )
        hass.data.get(_WS_WRITE_LOCKS_KEY, {}).pop(entry.entry_id, None)
        hass.data.get(_WS_OPTIONS_LOCKS_KEY, {}).pop(entry.entry_id, None)

        # Drop any staged history import (uploaded CSV text or a finished scan's
        # traces). Nothing else owns that memory, so without this an abandoned upload
        # would live for the lifetime of the process.
        async_clear_history_import(hass, entry.entry_id)

        # When the last WashData entry is removed, tear down the shared panel/sidebar
        # so no stale registration flags or sidebar entry linger.
        if not hass.data.get(DOMAIN):
            from .frontend import async_unregister_panel
            await async_unregister_panel(hass)

    return unload_ok
