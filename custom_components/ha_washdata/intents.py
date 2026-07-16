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
"""Home Assistant conversation intents for WashData.

Lets users ask Assist (voice or text) natural-language questions about their
appliances and get a plain-language spoken/text answer derived from the live
manager/sensor state, for example:

    "Is my washer done?"
    "How long until the dryer finishes?"

The single intent registered here is :data:`INTENT_STATUS`
(``"HaWashdataStatus"``), handled by :class:`WashDataStatusIntentHandler`.

Wiring trigger sentences
------------------------
Registering the :class:`intent.IntentHandler` (via :func:`async_setup_intents`)
makes the intent handleable — it can be fired immediately from automations, the
``intent_script`` integration, developer tools, or the Assist pipeline **once a
sentence maps text to it**. Home Assistant has no public runtime API for a
*custom* integration to inject sentences into the built-in conversation agent,
so trigger sentences are wired by the user with a config-directory sentence pack.
A ready-to-use pack ships in the repo at ``docs/custom_sentences/en/ha_washdata.yaml``
- copy it to ``<config>/custom_sentences/en/ha_washdata.yaml`` and restart HA. The
minimal shape is (one file per language)::

    language: en
    intents:
      HaWashdataStatus:
        data:
          - sentences:
              - "is my {name} done"
              - "is the {name} finished"
              - "how long until the {name} finishes"
              - "how long is left on the {name}"
          - sentences:
              - "is the laundry done"
              - "how long until it finishes"
    lists:
      name:
        wildcard: true

The optional ``{name}`` slot disambiguates which appliance when more than one is
configured. Users who prefer templated responses instead of the handler can also
declare the same intent via the ``intent_script`` integration.

This module has no import-time side effects: it only defines constants, helpers
and the handler class. Registration happens when :func:`async_setup_intents` is
called from ``async_setup_entry`` (guarded to run once per HA instance).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, intent, translation
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    STATE_ANTI_WRINKLE,
    STATE_CLEAN,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_PAUSED,
    STATE_RINSE,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_USER_PAUSED,
)

_LOGGER = logging.getLogger(__name__)

# Unique, stable intent identifier. Referenced by sentence packs / automations.
INTENT_STATUS = "HaWashdataStatus"

# States where the appliance is actively working on a cycle.
_ACTIVE_STATES = frozenset(
    {
        STATE_RUNNING,
        STATE_STARTING,
        STATE_ENDING,
        STATE_PAUSED,
        STATE_USER_PAUSED,
        STATE_RINSE,
        STATE_ANTI_WRINKLE,
    }
)

# States that mean a cycle just completed (laundry/dishes ready).
_FINISHED_STATES = frozenset({STATE_FINISHED, STATE_CLEAN})

# How far back a completed cycle still counts as "finished recently" (minutes).
_RECENT_FINISH_WINDOW_MIN = 720  # 12 hours

# English fallback response templates. The canonical translatable copies live in
# strings.json / translations/en.json under the top-level "intent" section and
# are loaded on top of these by _localized_templates(). Kept here so the handler
# always works even if the translation cache is unavailable.
DEFAULT_TEMPLATES: dict[str, str] = {
    "running_with_estimate": "Your {device} is still running. About {minutes} minutes left.",
    "running_no_estimate": "Your {device} is still running.",
    "finished_recently": "Your {device} finished {minutes} minutes ago.",
    "just_finished": "Your {device} just finished.",
    "not_running": "Your {device} is not running.",
    "no_devices": "I couldn't find any WashData appliances.",
    "unknown_device": "I couldn't find a WashData appliance called {device}.",
    "none_running": "None of your WashData appliances are running.",
    "error": "Sorry, I couldn't check your appliances right now.",
}


def _minutes_from_seconds(seconds: Any) -> int | None:
    """Return whole minutes (>=1) from a seconds value, or None when unusable."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return max(1, int(round(value / 60.0)))


def _minutes_since(end: Any, now: datetime) -> int | None:
    """Return whole minutes elapsed since ``end`` (clamped at 0), or None."""
    if not isinstance(end, datetime):
        return None
    try:
        delta = (now - end).total_seconds()
    except (TypeError, ValueError):
        return None
    if delta < 0:
        return 0
    return int(round(delta / 60.0))


def _iter_managers(hass: HomeAssistant) -> list[tuple[str, Any]]:
    """Return (title, manager) pairs for every loaded WashData device."""
    result: list[tuple[str, Any]] = []
    domain_data = hass.data.get(DOMAIN) or {}
    for manager in domain_data.values():
        # hass.data[DOMAIN] maps entry_id -> manager; skip anything unexpected.
        if not hasattr(manager, "check_state"):
            continue
        entry = getattr(manager, "config_entry", None)
        title = getattr(entry, "title", None)
        result.append((title or "appliance", manager))
    return result


def _is_running(manager: Any) -> bool:
    """Return True when the manager reports an active-cycle state."""
    try:
        return manager.check_state() in _ACTIVE_STATES
    except Exception:  # noqa: BLE001 - never let intent handling raise
        return False


def _describe_device(
    title: str, manager: Any, templates: dict[str, str], now: datetime
) -> str:
    """Build a one-sentence plain-language status line for a single device."""
    device = title or "appliance"
    try:
        state = manager.check_state()
    except Exception:  # noqa: BLE001
        state = None

    if state in _ACTIVE_STATES:
        minutes = _minutes_from_seconds(getattr(manager, "time_remaining", None))
        if minutes:
            return templates["running_with_estimate"].format(
                device=device, minutes=minutes
            )
        return templates["running_no_estimate"].format(device=device)

    # Not actively running: mention a recent finish when we can.
    ago = _minutes_since(getattr(manager, "last_cycle_end_time", None), now)

    if state in _FINISHED_STATES:
        if ago and ago > 0:
            return templates["finished_recently"].format(device=device, minutes=ago)
        return templates["just_finished"].format(device=device)

    # Off / idle / interrupted / unknown: only claim a finish if it was recent.
    if ago is not None and 0 < ago <= _RECENT_FINISH_WINDOW_MIN:
        return templates["finished_recently"].format(device=device, minutes=ago)
    if ago == 0:
        return templates["just_finished"].format(device=device)
    return templates["not_running"].format(device=device)


def _build_speech(
    hass: HomeAssistant,
    name_slot: str | None,
    templates: dict[str, str],
    now: datetime,
) -> str:
    """Resolve the target device(s) and build the spoken answer."""
    managers = _iter_managers(hass)
    if not managers:
        return templates["no_devices"]

    needle = str(name_slot).strip().casefold() if name_slot else ""
    if needle:
        # Prefer an exact title, then a whole-word hit, then any substring, so
        # "washer" resolves to "Laundry Washer" rather than "Dishwasher".
        exact = [(t, m) for (t, m) in managers if t.casefold() == needle]
        word = [(t, m) for (t, m) in managers if needle in t.casefold().split()]
        sub = [(t, m) for (t, m) in managers if needle in t.casefold()]
        picked = exact or word or sub
        if not picked:
            return templates["unknown_device"].format(device=name_slot)
        title, manager = picked[0]
        return _describe_device(title, manager, templates, now)

    if len(managers) == 1:
        title, manager = managers[0]
        return _describe_device(title, manager, templates, now)

    # Multiple devices, none specified: summarize the running ones.
    running = [(t, m) for (t, m) in managers if _is_running(m)]
    if not running:
        return templates["none_running"]
    return " ".join(
        _describe_device(t, m, templates, now) for (t, m) in running
    )


async def _localized_templates(
    hass: HomeAssistant, language: str | None
) -> dict[str, str]:
    """Return response templates, overlaying translations onto the English base.

    Falls back to :data:`DEFAULT_TEMPLATES` on any failure so the handler stays
    functional even without a warmed translation cache (e.g. under unit tests).
    """
    templates = dict(DEFAULT_TEMPLATES)
    try:
        flat = await translation.async_get_translations(
            hass, language or "en", "intent", {DOMAIN}
        )
    except Exception:  # noqa: BLE001
        return templates
    prefix = f"component.{DOMAIN}.intent.{INTENT_STATUS}."
    for full_key, value in flat.items():
        if full_key.startswith(prefix) and isinstance(value, str) and value:
            templates[full_key[len(prefix):]] = value
    return templates


class WashDataStatusIntentHandler(intent.IntentHandler):
    """Answer status questions ("is it done?", "how long left?") for a device."""

    intent_type = INTENT_STATUS
    description = (
        "Report whether a WashData appliance is running and how long is left."
    )

    @property
    def slot_schema(self) -> dict:
        """Optional appliance name to disambiguate which device is meant."""
        return {vol.Optional("name"): cv.string}

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Build a plain-language status response; never raises."""
        response = intent_obj.create_response()
        hass = intent_obj.hass
        templates = await _localized_templates(hass, intent_obj.language)
        try:
            slots = intent_obj.slots or {}
            name_value = slots.get("name", {}).get("value")
            speech = _build_speech(hass, name_value, templates, dt_util.now())
        except Exception:  # noqa: BLE001 - graceful degradation, no exceptions escape
            _LOGGER.exception("WashData status intent failed")
            speech = templates.get("error", DEFAULT_TEMPLATES["error"])
        response.async_set_speech(speech)
        return response


@callback
def async_setup_intents(hass: HomeAssistant) -> None:
    """Register WashData conversation intents (domain-global, idempotent).

    Call once per HA instance. ``intent.async_register`` overwrites an existing
    handler of the same ``intent_type`` (with a warning), so callers guard with a
    ``hass.data`` flag to avoid re-registration on multi-entry setups.
    """
    intent.async_register(hass, WashDataStatusIntentHandler())
