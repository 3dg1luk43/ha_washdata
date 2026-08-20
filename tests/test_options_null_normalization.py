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
"""Null option values must never reach the manager's numeric casts.

Reverting a setting that had never been saved sends the changelog's ``old``
value - ``null`` - back through ``ws_set_options``. Persisted verbatim, that
``None`` survives ``options.get(key, DEFAULT)`` (the key exists, so the default
never applies) and ``float(None)`` raises, so the entry fails to set up on every
restart until ``.storage`` is edited by hand.

Fast, pure-unit tests (no HA boot, no file I/O).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_DOOR_SENSOR_ENTITY,
    CONF_MIN_POWER,
    CONF_NOTIFY_QUIET_START_HOUR,
    CONF_PEAK_RATE_THRESHOLD,
    CONF_POWER_OFF_DELAY,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_POWER_SENSOR,
    DEFAULT_POWER_OFF_THRESHOLD_W,
    DOMAIN,
)
from custom_components.ha_washdata.options_utils import (
    has_null_options,
    strip_null_options,
)


# ---------------------------------------------------------------------------
# strip_null_options / has_null_options
# ---------------------------------------------------------------------------

def test_strip_drops_unset_numeric_nulls():
    cleaned = strip_null_options(
        {
            CONF_POWER_OFF_THRESHOLD_W: None,
            CONF_POWER_OFF_DELAY: None,
            CONF_MIN_POWER: 5.0,
        }
    )
    assert cleaned == {CONF_MIN_POWER: 5.0}


def test_strip_keeps_the_one_key_with_a_non_none_fallback():
    """power_sensor falls back to entry.data, so a cleared binding must stay None."""
    options = {CONF_POWER_SENSOR: None}
    assert strip_null_options(options) == options
    assert has_null_options(options) is False


def test_strip_drops_nulls_readers_treat_as_absent_anyway():
    """Every other nullable key reads the same absent or None, so it is dropped."""
    options = {
        CONF_DOOR_SENSOR_ENTITY: None,
        CONF_PEAK_RATE_THRESHOLD: None,
        CONF_NOTIFY_QUIET_START_HOUR: None,
    }
    assert strip_null_options(options) == {}


def test_strip_does_not_mutate_the_input():
    options = {CONF_POWER_OFF_THRESHOLD_W: None}
    strip_null_options(options)
    assert options == {CONF_POWER_OFF_THRESHOLD_W: None}


def test_has_null_options_detects_only_unset_meaning_nulls():
    assert has_null_options({CONF_POWER_OFF_DELAY: None}) is True
    assert has_null_options({CONF_MIN_POWER: 5.0}) is False


def test_stripped_key_restores_the_compiled_default():
    """The read pattern that used to raise TypeError now yields the default."""
    raw = {CONF_POWER_OFF_THRESHOLD_W: None}
    with pytest.raises(TypeError):
        float(raw.get(CONF_POWER_OFF_THRESHOLD_W, DEFAULT_POWER_OFF_THRESHOLD_W))

    cleaned = strip_null_options(raw)
    assert (
        float(cleaned.get(CONF_POWER_OFF_THRESHOLD_W, DEFAULT_POWER_OFF_THRESHOLD_W))
        == DEFAULT_POWER_OFF_THRESHOLD_W
    )


# ---------------------------------------------------------------------------
# ws_set_options
# ---------------------------------------------------------------------------

def _entry(options: dict) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {CONF_POWER_SENSOR: "sensor.power"}
    entry.options = options
    return entry


def _hass() -> tuple[MagicMock, MagicMock]:
    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}
    return hass, manager


async def _set_options(entry: MagicMock, hass: MagicMock, options: dict) -> dict:
    """Run ws_set_options and return the options it persisted."""
    ws_fn = ws_api.ws_set_options.__wrapped__
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_fn(hass, MagicMock(), {"id": 1, "entry_id": "e1", "options": options})
    return hass.config_entries.async_update_entry.call_args.kwargs["options"]


async def test_reverting_to_unset_drops_the_key_instead_of_storing_null():
    entry = _entry({CONF_MIN_POWER: 5.0, CONF_POWER_OFF_THRESHOLD_W: 1.5})
    hass, _manager = _hass()

    saved = await _set_options(entry, hass, {CONF_POWER_OFF_THRESHOLD_W: None})

    assert CONF_POWER_OFF_THRESHOLD_W not in saved
    assert saved[CONF_MIN_POWER] == 5.0


async def test_nulls_already_in_storage_are_cleaned_on_the_next_save():
    entry = _entry({CONF_POWER_OFF_THRESHOLD_W: None, CONF_POWER_OFF_DELAY: None})
    hass, _manager = _hass()

    saved = await _set_options(entry, hass, {CONF_MIN_POWER: 5.0})

    assert CONF_POWER_OFF_THRESHOLD_W not in saved
    assert CONF_POWER_OFF_DELAY not in saved
    assert saved[CONF_MIN_POWER] == 5.0


async def test_clearing_a_selector_unbinds_it():
    """The stored entity id is gone either way; the reader sees None."""
    entry = _entry({CONF_DOOR_SENSOR_ENTITY: "binary_sensor.door"})
    hass, _manager = _hass()

    saved = await _set_options(entry, hass, {CONF_DOOR_SENSOR_ENTITY: ""})

    assert saved.get(CONF_DOOR_SENSOR_ENTITY) is None


async def test_clearing_the_power_sensor_stays_an_explicit_none():
    """Dropping it would resurrect the entry.data binding instead of unbinding."""
    entry = _entry({CONF_POWER_SENSOR: "sensor.other"})
    hass, _manager = _hass()

    saved = await _set_options(entry, hass, {CONF_POWER_SENSOR: None})

    assert CONF_POWER_SENSOR in saved
    assert saved[CONF_POWER_SENSOR] is None


async def test_revert_to_unset_is_still_recorded_in_the_changelog():
    entry = _entry({CONF_POWER_OFF_THRESHOLD_W: 1.5})
    hass, manager = _hass()

    await _set_options(entry, hass, {CONF_POWER_OFF_THRESHOLD_W: None})

    recorded = manager.profile_store.async_record_settings_changes.await_args[0][0]
    assert [(c["key"], c["old"], c["new"]) for c in recorded] == [
        (CONF_POWER_OFF_THRESHOLD_W, 1.5, None)
    ]
