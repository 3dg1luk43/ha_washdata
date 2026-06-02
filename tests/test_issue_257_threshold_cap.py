"""Issue #257: cannot apply suggested start/stop thresholds for high-power devices.

A user monitoring a domestic well pump received suggested detection thresholds of
~1097 W (start) and ~731 W (stop). The detection-section ``NumberSelector`` ceilings
were hard-capped at 500 W (start) / 100 W (stop) - values tuned for washers/dryers -
so the options form rejected the very values the suggestion engine produced with
"Value 1096.68 is too large for dictionary value @ data['detection_section']
['start_threshold_w']". The ceilings now expand to admit the currently-saved value
and any pending suggestion.
"""
import pytest
import voluptuous as vol
from homeassistant.data_entry_flow import FlowResultType

from custom_components.ha_washdata.config_flow import OptionsFlowHandler
from custom_components.ha_washdata.const import (
    DOMAIN,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
)

# Suggested thresholds from the bug report (a ~914 W active-power floor pump).
SUGGESTED_START = 1096.68
SUGGESTED_STOP = 731.12


def _detection_section(data_schema):
    """Return the inner vol.Schema for the detection section of an options form."""
    for marker, value in data_schema.schema.items():
        if getattr(marker, "schema", None) == "detection_section":
            return value.schema  # Section -> vol.Schema
    raise AssertionError("detection_section not found in advanced settings form")


def _selector_max(inner_schema, conf_key):
    """Return the NumberSelector max bound for a field inside a section schema."""
    for marker, sel in inner_schema.schema.items():
        if getattr(marker, "schema", None) == conf_key:
            return sel.config["max"]
    raise AssertionError(f"{conf_key} not found in detection section")


async def _build_advanced_form(mock_hass, mock_config_entry, *, suggestions, options):
    manager = type("M", (), {})()
    manager.suggestions = suggestions
    mock_config_entry.options = options
    mock_config_entry.data = {}
    mock_hass.data[DOMAIN] = {mock_config_entry.entry_id: manager}
    mock_hass.config_entries.async_get_known_entry.return_value = mock_config_entry

    flow = OptionsFlowHandler(mock_config_entry)
    flow.hass = mock_hass
    flow.handler = mock_config_entry.entry_id

    result = await flow.async_step_advanced_settings()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "advanced_settings"
    return _detection_section(result["data_schema"])


@pytest.mark.asyncio
async def test_suggested_high_thresholds_fit_within_caps(mock_hass, mock_config_entry):
    """A pending high-power suggestion must not exceed the selector ceiling."""
    suggestions = {
        CONF_START_THRESHOLD_W: {"value": SUGGESTED_START, "reason": "pump"},
        CONF_STOP_THRESHOLD_W: {"value": SUGGESTED_STOP, "reason": "pump"},
    }
    detection = await _build_advanced_form(
        mock_hass, mock_config_entry, suggestions=suggestions, options={}
    )

    assert _selector_max(detection, CONF_START_THRESHOLD_W) >= SUGGESTED_START
    assert _selector_max(detection, CONF_STOP_THRESHOLD_W) >= SUGGESTED_STOP

    # The original failure: validating these values raised "value too large".
    detection(
        {
            CONF_START_THRESHOLD_W: SUGGESTED_START,
            CONF_STOP_THRESHOLD_W: SUGGESTED_STOP,
        }
    )


@pytest.mark.asyncio
async def test_saved_high_thresholds_fit_within_caps(mock_hass, mock_config_entry):
    """An already-saved high value must remain editable without rejection."""
    detection = await _build_advanced_form(
        mock_hass,
        mock_config_entry,
        suggestions={},
        options={
            CONF_START_THRESHOLD_W: SUGGESTED_START,
            CONF_STOP_THRESHOLD_W: SUGGESTED_STOP,
        },
    )

    assert _selector_max(detection, CONF_START_THRESHOLD_W) >= SUGGESTED_START
    assert _selector_max(detection, CONF_STOP_THRESHOLD_W) >= SUGGESTED_STOP
    detection(
        {
            CONF_START_THRESHOLD_W: SUGGESTED_START,
            CONF_STOP_THRESHOLD_W: SUGGESTED_STOP,
        }
    )


@pytest.mark.asyncio
async def test_default_caps_preserved_for_typical_appliances(
    mock_hass, mock_config_entry
):
    """Without high values, the friendly washer/dryer ceilings stay in place."""
    detection = await _build_advanced_form(
        mock_hass, mock_config_entry, suggestions={}, options={}
    )

    assert _selector_max(detection, CONF_START_THRESHOLD_W) == 500.0
    assert _selector_max(detection, CONF_STOP_THRESHOLD_W) == 100.0

    # And a clearly-too-large value is still rejected for a typical appliance.
    with pytest.raises(vol.Invalid):
        detection({CONF_START_THRESHOLD_W: 9999.0})
