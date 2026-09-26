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
"""Issue #422: the panel served the creation-time device name, not the current one.

``entry.data[CONF_NAME]`` is written once by ``async_create_entry`` and is never
updated again: every rename path (reconfigure, options flow, ``ws_set_options``)
deliberately writes only ``entry.title``, and ``_merge_structural_options`` /
``_OPTIONS_IDENTITY_KEYS`` actively strip ``CONF_NAME`` back out of options. So
``entry.title`` is the display name and ``entry.data[CONF_NAME]`` is a fossil.

``ws_get_options`` built its payload as ``{**entry.data, **entry.options}``, and
``name`` is the one key that lives in data only - nothing in options could shadow
it - so the fossil won and the panel's Settings -> Basic "Device Name" field was
painted with the creation-time name while the device list (which serves
``entry.title``) showed the current one.

Blast radius on v0.5.5: ``_saveSettings`` posted every rendered field, so the
fossil was echoed back and written to ``entry.title``, renaming the HA device.
On 0.5.6 #406's ``_changedOptions`` diff stops the echo, leaving a display-only
bug. Both are covered below.

Fast, pure-unit tests.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import config_flow as cf_mod
from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
)
from homeassistant.const import CONF_NAME


def _make_entry(*, title: str, data_name: str, options: dict | None = None):
    """A config entry renamed after creation: title moved on, data[name] did not."""
    entry = MagicMock()
    entry.entry_id = "entry_1"
    entry.title = title
    entry.data = {
        CONF_NAME: data_name,
        CONF_POWER_SENSOR: "sensor.39c0_power",
        CONF_DEVICE_TYPE: "washing_machine",
        CONF_MIN_POWER: 5.0,
    }
    entry.options = {
        CONF_DEVICE_TYPE: "dishwasher",
        CONF_POWER_SENSOR: "sensor.39c0_power",
        CONF_OFF_DELAY: 120,
        **(options or {}),
    }
    return entry


def _run_get_options(entry) -> dict:
    captured: dict = {}

    def _capture(connection, msg_id, cmd, payload):
        captured["payload"] = payload

    with patch.object(ws_api, "_get_entry", return_value=entry), \
         patch.object(ws_api, "_send_result", side_effect=_capture):
        fn = getattr(ws_api.ws_get_options, "__wrapped__", ws_api.ws_get_options)
        fn(MagicMock(), MagicMock(), {"id": 1, "entry_id": entry.entry_id})
    return captured["payload"]


# ---------------------------------------------------------------------------
# The reported symptom
# ---------------------------------------------------------------------------

def test_get_options_serves_the_current_name_not_the_creation_time_one():
    """The reporter's case: created "Washing Machine", later renamed "Dishwasher"."""
    entry = _make_entry(title="Dishwasher", data_name="Washing Machine")

    options = _run_get_options(entry)["options"]

    assert options[CONF_NAME] == "Dishwasher", (
        "#422: the panel was served entry.data['name'], the creation-time fossil"
    )


def test_get_options_name_agrees_with_the_device_list_title():
    """Settings -> Basic and the device list must not disagree about the name.

    ``ws_get_devices`` serves ``entry.title`` directly, so before the fix the two
    panel surfaces showed different names for the same entry.
    """
    entry = _make_entry(title="Washer-Dryer Combo", data_name="Dishwasher")

    options = _run_get_options(entry)["options"]

    assert options[CONF_NAME] == entry.title


def test_get_options_still_merges_data_under_options_for_every_other_key():
    """The fix must override only the name; options-over-data still holds."""
    entry = _make_entry(title="Dishwasher", data_name="Washing Machine")

    options = _run_get_options(entry)["options"]

    # options wins over data
    assert options[CONF_DEVICE_TYPE] == "dishwasher"
    # data-only keys still come through
    assert options[CONF_MIN_POWER] == 5.0
    # options-only keys still come through
    assert options[CONF_OFF_DELAY] == 120


# ---------------------------------------------------------------------------
# Why the fossil exists: no rename path ever updates entry.data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconfigure_rename_updates_the_title_and_leaves_data_stale():
    flow = cf_mod.ConfigFlow()
    flow.hass = MagicMock()
    flow.async_update_reload_and_abort = MagicMock(return_value={"type": "abort"})
    flow.async_show_form = MagicMock(return_value={"type": "form"})
    entry = _make_entry(title="Washing Machine", data_name="Washing Machine")
    flow._get_reconfigure_entry = MagicMock(return_value=entry)

    await flow.async_step_reconfigure({
        CONF_NAME: "Dishwasher",
        CONF_DEVICE_TYPE: "dishwasher",
        CONF_POWER_SENSOR: "sensor.39c0_power",
        CONF_MIN_POWER: 5.0,
    })

    kwargs = flow.async_update_reload_and_abort.call_args[1]
    assert kwargs["title"] == "Dishwasher"
    assert "data" not in kwargs, "reconfigure must not write entry.data"
    assert CONF_NAME not in kwargs["options"], "the name never belongs in options"
    # Hence: data[name] is still the creation-time value.
    assert entry.data[CONF_NAME] == "Washing Machine"


@pytest.mark.asyncio
async def test_options_flow_rename_updates_the_title_and_leaves_data_stale():
    entry = _make_entry(title="Washing Machine", data_name="Washing Machine")
    flow = cf_mod.OptionsFlowHandler(entry)
    flow.hass = MagicMock()
    flow.hass.config_entries.async_update_entry = MagicMock()
    flow.async_create_entry = MagicMock(return_value={"type": "create_entry"})
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    await flow.async_step_init({
        CONF_NAME: "Dishwasher",
        CONF_DEVICE_TYPE: "dishwasher",
        CONF_POWER_SENSOR: "sensor.39c0_power",
        CONF_MIN_POWER: 5.0,
    })

    update_kwargs = flow.hass.config_entries.async_update_entry.call_args[1]
    assert update_kwargs == {"title": "Dishwasher"}
    assert CONF_NAME not in flow.async_create_entry.call_args[1]["data"]
    assert entry.data[CONF_NAME] == "Washing Machine"


# ---------------------------------------------------------------------------
# v0.5.5 blast radius: the served name is echoed back and renames the device
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_full_payload_save_cannot_resurrect_the_creation_time_name():
    """Pre-#406 the panel posted every rendered field, name included.

    With the served name equal to the title, that echo is a no-op rename rather
    than a revert to the fossil - so the v0.5.5 data loss cannot recur even if a
    client (or an old cached panel bundle) posts the full form again.
    """
    entry = _make_entry(title="Dishwasher", data_name="Washing Machine")
    served = _run_get_options(entry)["options"]

    hass = MagicMock()
    hass.config_entries.async_update_entry = MagicMock()
    # A real dict, because ws_set_options now takes the per-entry write lock and
    # `_entry_write_lock` stores it in hass.data - a MagicMock hands back a
    # MagicMock, which cannot be awaited.
    hass.data = {}
    connection = MagicMock()

    with patch.object(ws_api, "_get_entry", return_value=entry), \
         patch.object(ws_api, "_get_manager", return_value=None), \
         patch.object(ws_api, "_send_result", MagicMock()):
        await ws_api.ws_set_options.__wrapped__(
            hass, connection, {"id": 1, "entry_id": entry.entry_id, "options": dict(served)}
        )

    kwargs = hass.config_entries.async_update_entry.call_args[1]
    assert kwargs["title"] == "Dishwasher", (
        "#422: echoing the served payload back renamed the HA device to the fossil"
    )
    assert CONF_NAME not in kwargs["options"]
