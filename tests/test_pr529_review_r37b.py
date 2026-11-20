"""Review of PR #529, round 37b: the Minor findings in ws_api.py, each reproduced.

* A Settings save that dropped an invalid Min Power override recorded the
  removal as None, though the detector fell back to the entry data's value.
"""
from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
)


async def _entry_with_data(
    hass: HomeAssistant, data: dict[str, Any], options: dict[str, Any]
) -> MockConfigEntry:
    """``setup_washdata_entry``, with keys of its own in ``entry.data``."""
    hass.states.async_set("sensor.washer_power", "0", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Washer",
        data={
            "name": "Washer", "power_sensor": "sensor.washer_power",
            "device_type": "washing_machine", **data,
        },
        options=dict(options),
        unique_id="washdata_Washer",
        version=CONFIG_ENTRY_VERSION,
        minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_a_dropped_override_is_recorded_as_what_is_left(
    hass: HomeAssistant, enable_custom_integrations: Any, hass_ws_client: Any
) -> None:
    """Min Power 5 W in the entry's data, a 2 W override in its options: an
    invalid Min Power drops the override and the detector runs with 5 W, but
    the history said ``2 -> (none)``. A key neither map holds stays None."""
    entry = await _entry_with_data(
        hass, {CONF_MIN_POWER: 5.0}, {CONF_MIN_POWER: 2.0, CONF_OFF_DELAY: 120},
    )
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_options", "entry_id": entry.entry_id,
        "options": {CONF_MIN_POWER: "abc", CONF_OFF_DELAY: "abc"},
    })
    res = await client.receive_json()
    assert res["success"], res
    await hass.async_block_till_done()
    assert CONF_MIN_POWER not in entry.options and CONF_OFF_DELAY not in entry.options
    rows = {r["key"]: r for r in store.get_settings_changelog()}
    assert rows[CONF_MIN_POWER]["old"] == 2.0 and rows[CONF_MIN_POWER]["new"] == 5.0
    assert rows[CONF_OFF_DELAY]["old"] == 120 and rows[CONF_OFF_DELAY]["new"] is None
