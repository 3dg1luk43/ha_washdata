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
    option_float,
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


def test_strip_drops_a_null_power_sensor_too():
    """The one key with a non-None fallback is stripped as well, on purpose.

    ``power_sensor`` is read as ``options.get(key, entry.data.get(key))``, so
    dropping it falls through to the sensor the config flow required at setup -
    which is the right recovery, because a stored None is not a supported binding:
    ``async_setup`` raises ``AttributeError`` inside ``hass.states.get(None)``
    (``None.lower()``), a brick no numeric-cast guard would catch. Nothing in the
    UI can request an unbound sensor either (the field is not clearable and the
    selector-clearing loop in ``ws_set_options`` omits it).
    """
    options = {CONF_POWER_SENSOR: None}
    assert strip_null_options(options) == {}
    assert has_null_options(options) is True


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


async def test_a_null_power_sensor_falls_back_to_the_data_binding():
    """Reverting the power sensor restores entry.data rather than unbinding it."""
    entry = _entry({CONF_POWER_SENSOR: "sensor.other"})
    hass, _manager = _hass()

    saved = await _set_options(entry, hass, {CONF_POWER_SENSOR: None})

    assert CONF_POWER_SENSOR not in saved
    assert saved.get(CONF_POWER_SENSOR, entry.data[CONF_POWER_SENSOR]) == "sensor.power"


async def test_revert_to_unset_is_still_recorded_in_the_changelog():
    entry = _entry({CONF_POWER_OFF_THRESHOLD_W: 1.5})
    hass, manager = _hass()

    await _set_options(entry, hass, {CONF_POWER_OFF_THRESHOLD_W: None})

    recorded = manager.profile_store.async_record_settings_changes.await_args[0][0]
    assert [(c["key"], c["old"], c["new"]) for c in recorded] == [
        (CONF_POWER_OFF_THRESHOLD_W, 1.5, None)
    ]


# ---------------------------------------------------------------------------
# ws_import_config
# ---------------------------------------------------------------------------

async def test_import_does_not_persist_a_null_option():
    """An export taken from an entry that still held a null must not re-plant it."""
    entry = _entry({CONF_MIN_POWER: 5.0})
    hass, manager = _hass()
    manager.profile_store.async_import_data = AsyncMock(
        return_value={
            "entry_options": {
                CONF_POWER_OFF_DELAY: None,
                CONF_POWER_OFF_THRESHOLD_W: 1.5,
            }
        }
    )
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))

    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_import_config.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "json_data": "{}"}
        )

    saved = hass.config_entries.async_update_entry.call_args.kwargs["options"]
    assert CONF_POWER_OFF_DELAY not in saved
    assert saved[CONF_POWER_OFF_THRESHOLD_W] == 1.5


# ---------------------------------------------------------------------------
# option_float
#
# strip_null_options fixes the wrong *value*; this fixes the wrong *type*. An
# import file is hand-editable and ws_set_options validates the payload as a
# plain dict, so a string can be persisted where a number is expected and
# survive options.get(key, DEFAULT) exactly as a null did.
# ---------------------------------------------------------------------------

def test_option_float_passes_numbers_through():
    assert option_float(0.6, 0.9) == 0.6
    assert option_float(1, 0.9) == 1.0


def test_option_float_accepts_a_numeric_string():
    """The common shape of a hand-edited or form-encoded value."""
    assert option_float("0.6", 0.9) == 0.6


def test_option_float_falls_back_on_garbage():
    for bad in ("high", "", None, [], {}, object()):
        assert option_float(bad, 0.9) == 0.9


def test_option_float_falls_back_to_the_default_not_zero():
    """Zero is a real setting ("accept anything"), so it must not be the fallback."""
    assert option_float("nonsense", 0.9) == 0.9
    assert option_float("nonsense", 0.0) == 0.0


def test_option_float_rejects_non_finite_values():
    """float() accepts these, and a non-finite threshold is worse than a raise.

    Every comparison against nan is False, and every comparison of a real
    confidence against inf is False, so the gated feature goes quietly dead.
    """
    for bad in ("nan", "NaN", "inf", "-inf", "infinity", float("nan"), float("inf")):
        assert option_float(bad, 0.9) == 0.9


def test_option_float_non_finite_would_otherwise_disable_labelling():
    """The concrete consequence, stated as the comparison the manager makes."""
    conf = 0.95
    assert (conf >= float("nan")) is False      # never labels
    assert (conf >= float("inf")) is False      # never labels
    assert (conf >= option_float("nan", 0.9)) is True


def test_option_float_does_not_swallow_a_stored_bool():
    """Not a valid threshold, but float() accepts it, so record the behaviour."""
    assert option_float(True, 0.9) == 1.0


@pytest.mark.asyncio
async def test_a_garbage_confidence_option_does_not_break_cycle_end(hass):
    """The failure this guards: a raised cast inside the spawned cycle-end task.

    Both thresholds are compared against the match confidence in
    ``_async_process_cycle_end``, which runs as a task. A non-numeric option raised
    there - ``float()`` for the learning threshold, ``> 0`` for the auto-label one -
    and killed the task before ``async_add_cycle``, so the cycle was lost outright
    rather than merely mislabelled.
    """
    from custom_components.ha_washdata.const import (
        CONF_AUTO_LABEL_CONFIDENCE,
        CONF_LEARNING_CONFIDENCE,
        DEFAULT_AUTO_LABEL_CONFIDENCE,
        DEFAULT_LEARNING_CONFIDENCE,
    )
    from custom_components.ha_washdata.manager import WashDataManager

    entry = MagicMock()
    entry.entry_id = "e1"
    entry.title = "Washer"
    entry.data = {}
    entry.options = {
        CONF_POWER_SENSOR: "sensor.p",
        CONF_LEARNING_CONFIDENCE: "high",
        CONF_AUTO_LABEL_CONFIDENCE: "very high",
    }
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)

    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(hass, entry)

    assert mgr._learning_confidence == DEFAULT_LEARNING_CONFIDENCE
    assert mgr._auto_label_confidence == DEFAULT_AUTO_LABEL_CONFIDENCE
    # The two comparisons that used to raise.
    assert (0.5 >= float(mgr._learning_confidence or 0.0)) in (True, False)
    assert (mgr._auto_label_confidence > 0) in (True, False)
