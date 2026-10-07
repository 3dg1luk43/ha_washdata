"""PR #466 round 13: the pump sensor follows an in-place device type change.

Since audit PLATFORM-15 a reconfigure (or a panel save) applies options through
the update listener without re-running platform setup, so `PumpRunsTodaySensor`
was never added when an appliance became a pump, and its entity stayed behind
when it stopped being one, until HA restarted.
"""

from __future__ import annotations

from homeassistant.helpers import entity_registry as er

from custom_components.ha_washdata.const import CONF_DEVICE_TYPE, DOMAIN


def _pump_entity(hass, entry):
    return er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_pump_runs_today")


async def test_the_pump_sensor_follows_the_device_type(hass, setup_washdata_entry) -> None:
    entry = await setup_washdata_entry(device_type="washing_machine")
    assert _pump_entity(hass, entry) is None

    hass.config_entries.async_update_entry(entry, options={**entry.options, CONF_DEVICE_TYPE: "pump"})
    await hass.async_block_till_done()
    entity_id = _pump_entity(hass, entry)
    assert entity_id is not None
    assert hass.states.get(entity_id) is not None

    hass.config_entries.async_update_entry(entry, options={**entry.options, CONF_DEVICE_TYPE: "dishwasher"})
    await hass.async_block_till_done()
    assert _pump_entity(hass, entry) is None
    assert hass.states.get(entity_id) is None

    await hass.config_entries.async_unload(entry.entry_id)
