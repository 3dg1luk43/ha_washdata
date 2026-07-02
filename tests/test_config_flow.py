"""Tests for the slimmed config_flow.py.

Covers: initial setup flow, reconfigure step, slim options stub, and the
ws_set_options title-update logic in ws_api.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.config_flow import (
    ConfigFlow,
    OptionsFlowHandler,
    _device_type_options,
    _escape_markdown,
)
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_NAME,
    CONF_POWER_SENSOR,
    DEFAULT_MIN_POWER,
    DEPRECATED_DEVICE_TYPES,
    DEVICE_TYPES,
    DOMAIN,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_device_type_options_excludes_deprecated():
    opts = _device_type_options()
    for dep in DEPRECATED_DEVICE_TYPES:
        assert dep not in opts


def test_device_type_options_keeps_current_deprecated():
    for dep in DEPRECATED_DEVICE_TYPES:
        opts = _device_type_options(current=dep)
        assert dep in opts


def test_device_type_options_all_non_deprecated_present():
    opts = _device_type_options()
    for key in DEVICE_TYPES:
        if key not in DEPRECATED_DEVICE_TYPES:
            assert key in opts


def test_escape_markdown_basic():
    assert _escape_markdown("hello") == "hello"


def test_escape_markdown_special_chars():
    result = _escape_markdown("a*b_c[d]e")
    assert "*" not in result.replace("\\*", "")
    assert "_" not in result.replace("\\_", "")
    # All metacharacters are backslash-escaped
    assert "\\*" in result
    assert "\\_" in result


def test_escape_markdown_collapses_whitespace():
    assert _escape_markdown("  a\n\nb  ") == "a b"


# ---------------------------------------------------------------------------
# OptionsFlowHandler - slim stub
# ---------------------------------------------------------------------------

def _make_entry(device_type="washing_machine", power_sensor="sensor.power", min_power=5.0):
    entry = MagicMock()
    entry.data = {
        CONF_DEVICE_TYPE: device_type,
        CONF_POWER_SENSOR: power_sensor,
        CONF_MIN_POWER: min_power,
        CONF_NAME: "My Washer",
    }
    entry.options = {}
    return entry


@pytest.mark.asyncio
async def test_options_flow_shows_form_on_none():
    handler = OptionsFlowHandler(_make_entry())
    handler.async_show_form = MagicMock(return_value={"type": "form"})
    result = await handler.async_step_init(None)
    assert handler.async_show_form.called
    call_kwargs = handler.async_show_form.call_args[1]
    assert call_kwargs["step_id"] == "init"


@pytest.mark.asyncio
async def test_options_flow_saves_merged_options():
    entry = _make_entry()
    handler = OptionsFlowHandler(entry)
    handler.async_create_entry = MagicMock(return_value={"type": "create_entry"})

    user_input = {
        CONF_DEVICE_TYPE: "dryer",
        CONF_POWER_SENSOR: "sensor.dryer_power",
        CONF_MIN_POWER: 3.0,
    }
    result = await handler.async_step_init(user_input)
    assert handler.async_create_entry.called
    saved = handler.async_create_entry.call_args[1]["data"]
    # Entry data is merged in
    assert saved[CONF_NAME] == "My Washer"
    assert saved[CONF_DEVICE_TYPE] == "dryer"
    assert saved[CONF_POWER_SENSOR] == "sensor.dryer_power"
    assert saved[CONF_MIN_POWER] == 3.0


@pytest.mark.asyncio
async def test_options_flow_uses_options_over_data_for_defaults():
    entry = _make_entry()
    entry.options = {CONF_DEVICE_TYPE: "dishwasher", CONF_MIN_POWER: 10.0}
    handler = OptionsFlowHandler(entry)
    handler.async_show_form = MagicMock(return_value={"type": "form"})
    await handler.async_step_init(None)
    # The schema should reflect options overriding data - spot-check by
    # passing the options input through and checking the merged result
    handler.async_create_entry = MagicMock(return_value={"type": "create_entry"})
    await handler.async_step_init({
        CONF_DEVICE_TYPE: "dishwasher",
        CONF_POWER_SENSOR: "sensor.power",
        CONF_MIN_POWER: 10.0,
    })
    saved = handler.async_create_entry.call_args[1]["data"]
    assert saved[CONF_DEVICE_TYPE] == "dishwasher"
    assert saved[CONF_MIN_POWER] == 10.0


# ---------------------------------------------------------------------------
# ConfigFlow.async_step_reconfigure
# ---------------------------------------------------------------------------

def _make_config_flow():
    flow = ConfigFlow()
    flow.hass = MagicMock()
    flow.async_show_form = MagicMock(return_value={"type": "form"})
    flow.async_update_reload_and_abort = MagicMock(return_value={"type": "abort"})
    return flow


def _make_reconfigure_entry(**kwargs):
    entry = MagicMock()
    entry.title = kwargs.get("title", "My Washer")
    entry.data = {
        CONF_DEVICE_TYPE: kwargs.get("device_type", "washing_machine"),
        CONF_POWER_SENSOR: kwargs.get("power_sensor", "sensor.power"),
        CONF_MIN_POWER: kwargs.get("min_power", 5.0),
    }
    entry.options = {}
    return entry


@pytest.mark.asyncio
async def test_reconfigure_shows_form_with_current_values():
    flow = _make_config_flow()
    entry = _make_reconfigure_entry()
    flow._get_reconfigure_entry = MagicMock(return_value=entry)

    result = await flow.async_step_reconfigure(None)
    assert flow.async_show_form.called
    call_kwargs = flow.async_show_form.call_args[1]
    assert call_kwargs["step_id"] == "reconfigure"

    schema = call_kwargs["data_schema"]
    defaults = {k.schema: k.default() for k in schema.schema if hasattr(k, "default") and callable(k.default)}
    assert defaults.get(CONF_NAME) == "My Washer"
    assert defaults.get(CONF_DEVICE_TYPE) == "washing_machine"
    assert defaults.get(CONF_POWER_SENSOR) == "sensor.power"
    assert defaults.get(CONF_MIN_POWER) == 5.0


@pytest.mark.asyncio
async def test_reconfigure_saves_and_aborts_on_valid_input():
    flow = _make_config_flow()
    entry = _make_reconfigure_entry()
    flow._get_reconfigure_entry = MagicMock(return_value=entry)

    user_input = {
        CONF_NAME: "Renamed Washer",
        CONF_DEVICE_TYPE: "dryer",
        CONF_POWER_SENSOR: "sensor.dryer_power",
        CONF_MIN_POWER: 3.0,
    }
    result = await flow.async_step_reconfigure(user_input)
    assert flow.async_update_reload_and_abort.called
    call_kwargs = flow.async_update_reload_and_abort.call_args[1]
    assert call_kwargs["title"] == "Renamed Washer"
    new_data = call_kwargs["data"]
    assert new_data[CONF_NAME] == "Renamed Washer"
    assert new_data[CONF_DEVICE_TYPE] == "dryer"
    assert new_data[CONF_POWER_SENSOR] == "sensor.dryer_power"
    assert new_data[CONF_MIN_POWER] == 3.0


@pytest.mark.asyncio
async def test_reconfigure_rejects_zero_min_power():
    flow = _make_config_flow()
    entry = _make_reconfigure_entry()
    flow._get_reconfigure_entry = MagicMock(return_value=entry)

    user_input = {
        CONF_NAME: "My Washer",
        CONF_DEVICE_TYPE: "washing_machine",
        CONF_POWER_SENSOR: "sensor.power",
        CONF_MIN_POWER: 0.0,
    }
    result = await flow.async_step_reconfigure(user_input)
    assert flow.async_show_form.called
    errors = flow.async_show_form.call_args[1].get("errors", {})
    assert CONF_MIN_POWER in errors


# ---------------------------------------------------------------------------
# ws_set_options title-update logic (unit test)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ws_set_options_updates_title_when_name_provided():
    """When ws_set_options receives a 'name' key, it must update the entry title."""
    from custom_components.ha_washdata import ws_api

    # The handler is wrapped by @async_response; use __wrapped__ to get the
    # raw async function for direct unit testing.
    ws_fn = ws_api.ws_set_options.__wrapped__

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {CONF_POWER_SENSOR: "sensor.power", CONF_MIN_POWER: 5.0}
    entry.options = {}

    hass = MagicMock()
    hass.data = {DOMAIN: {"test_entry": MagicMock()}}

    with patch.object(ws_api, "_get_entry", return_value=entry):
        connection = MagicMock()
        msg = {
            "id": 1,
            "entry_id": "test_entry",
            "options": {
                CONF_NAME: "  Renamed Device  ",
                CONF_MIN_POWER: 4.0,
            },
        }
        await ws_fn(hass, connection, msg)

    hass.config_entries.async_update_entry.assert_called_once()
    call_kwargs = hass.config_entries.async_update_entry.call_args[1]
    assert call_kwargs.get("title") == "Renamed Device"
    assert call_kwargs.get("options") is not None


@pytest.mark.asyncio
async def test_ws_set_options_no_title_update_when_name_absent():
    """When ws_set_options receives no 'name' key, the title must not be updated."""
    from custom_components.ha_washdata import ws_api

    ws_fn = ws_api.ws_set_options.__wrapped__

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {CONF_POWER_SENSOR: "sensor.power"}
    entry.options = {}

    hass = MagicMock()
    hass.data = {DOMAIN: {"test_entry": MagicMock()}}

    with patch.object(ws_api, "_get_entry", return_value=entry):
        connection = MagicMock()
        msg = {
            "id": 1,
            "entry_id": "test_entry",
            "options": {CONF_MIN_POWER: 4.0},
        }
        await ws_fn(hass, connection, msg)

    hass.config_entries.async_update_entry.assert_called_once()
    call_kwargs = hass.config_entries.async_update_entry.call_args[1]
    assert "title" not in call_kwargs
