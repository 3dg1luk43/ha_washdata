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
"""Maintenance-reminder rules (Group E, discussion #461). Pure, no Home Assistant.

The profile store owns the data (the log, the custom tasks, the counting
baselines). This module owns the rules every reader must agree on: which
reminders a device has, which built-in types its editor offers, and when a task
is due. The manager's due list, the state-sensor attribute, the Maintenance-due
binary sensor and the panel payload all go through it.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .const import (
    DEFAULT_MAINTENANCE_REMINDER_CYCLES,
    MAINTENANCE_COUNT_FROM_ENABLE_TYPES,
    MAINTENANCE_EVENT_TYPES,
    MAINTENANCE_PRESETS_BY_DEVICE_TYPE,
    MAINTENANCE_TASK_NAME_MAX,
    MAINTENANCE_TYPES_BY_DEVICE_TYPE,
)

# The editor's built-in rows for a device type without its own list: the original
# five, which is what every device offered before the presets existed.
DEFAULT_EDITOR_TYPES: tuple[str, ...] = (
    "descale",
    "filter_clean",
    "drum_clean",
    "bearing_service",
    "other",
)

_WHITESPACE = re.compile(r"\s+")


def default_reminders(device_type: Any) -> dict[str, int]:
    """The reminder intervals a device starts with, before its config is saved."""
    try:
        preset = MAINTENANCE_PRESETS_BY_DEVICE_TYPE.get(device_type)
    except TypeError:  # an unhashable device type is no device type
        preset = None
    return dict(preset if preset is not None else DEFAULT_MAINTENANCE_REMINDER_CYCLES)


def _positive_int(value: Any) -> int | None:
    """``int(value)`` when it is a usable non-negative count, else None."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def effective_reminders(device_type: Any, saved: Any) -> dict[str, int]:
    """The built-in reminder intervals (cycles) that apply to a device.

    Nothing saved: the device type's preset. A saved config is taken as written
    (an absent type stays off, as before), with one exception: a preset type that
    did not exist when that config was saved (salt, rinse aid, lint filter,
    condenser) is added at its preset value. It counts from a baseline stamped
    when it first applies, so adding it raises no banner (#461). The first panel
    save after that writes every row explicitly, so from then on the saved config
    is the whole answer. Unknown keys are dropped: they cannot be logged, so they
    would read as due forever. Never raises.
    """
    if not isinstance(saved, dict) or not saved:
        return default_reminders(device_type)
    out: dict[str, int] = {}
    for key, value in saved.items():
        if key not in MAINTENANCE_EVENT_TYPES:
            continue
        number = _positive_int(value)
        if number is not None:
            out[key] = number
    for key, value in default_reminders(device_type).items():
        if key in MAINTENANCE_COUNT_FROM_ENABLE_TYPES and key not in saved:
            out[key] = value
    return out


def editor_types(device_type: Any, reminders: Any) -> list[str]:
    """Built-in types the panel offers for a device, in display order.

    The device type's own list, plus any type with a positive interval in
    ``reminders``, so a reminder saved before the device-type lists existed (a
    dishwasher's ``drum_clean``) never vanishes from the editor. Never raises.
    """
    try:
        base = MAINTENANCE_TYPES_BY_DEVICE_TYPE.get(device_type)
    except TypeError:
        base = None
    out = list(base if base is not None else DEFAULT_EDITOR_TYPES)
    if isinstance(reminders, dict):
        for key in MAINTENANCE_EVENT_TYPES:
            if key not in out and (_positive_int(reminders.get(key)) or 0) > 0:
                out.append(key)
    return out


def coerce_interval(value: Any, maximum: int) -> int:
    """A task interval: a whole number in ``[0, maximum]``, None/"" meaning 0 (off).

    Raises ``ValueError`` for anything else, so a typo is refused rather than
    silently turning a reminder off.
    """
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        raise ValueError("An interval must be a whole number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as err:
        raise ValueError("An interval must be a whole number") from err
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise ValueError("An interval must be a whole number of 0 or more")
    if number > maximum:
        raise ValueError(f"An interval cannot exceed {maximum}")
    return int(number)


def clean_task_name(value: Any) -> str:
    """A custom task's display name: trimmed, single-spaced, capped; "" if unusable.

    User text, shown as typed (never translated). Control characters are dropped
    so a pasted name cannot break the panel row or an automation's message.
    """
    if not isinstance(value, str):
        return ""
    text = "".join(ch for ch in value if ch.isprintable() or ch.isspace())
    text = _WHITESPACE.sub(" ", text).strip()
    return text[:MAINTENANCE_TASK_NAME_MAX].strip()


def is_due(
    cycles_since: int,
    cycles_interval: int,
    days_since: float | None,
    days_interval: int,
) -> bool:
    """Due when either enabled interval is reached; an interval of 0 is off."""
    if cycles_interval > 0 and cycles_since >= cycles_interval:
        return True
    return bool(days_interval > 0 and days_since is not None and days_since >= days_interval)
