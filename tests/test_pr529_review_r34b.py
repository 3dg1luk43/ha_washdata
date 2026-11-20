"""Review of PR #529, round 34b: the Minor findings, each reproduced.

* A wholesale import of a payload carrying the cadence rows without their sensor
  (or the sensor without its rows) paired one device's half with the other's.
"""
from __future__ import annotations

from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import DOMAIN

POWER = "sensor.washer_power"
OTHER = "sensor.other_plug"


# ── A wholesale import takes the cadence pair whole or not at all ────────────


async def _store_measured_at_90s(hass: HomeAssistant, setup_washdata_entry: Any) -> Any:
    entry = await setup_washdata_entry(power_sensor=POWER)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.add_cadence_summary(90.0, 90.0, sensor=POWER)
    assert store.get_cadence_summaries(POWER) == [[90.0, 90.0]]
    return store


def _export(**cadence: Any) -> dict[str, Any]:
    return {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}}, "past_cycles": [], **cadence,
    }}


@pytest.mark.parametrize(
    "cadence",
    [{"cadence_summaries": [[2.0, 2.0]]}, {"cadence_sensor": OTHER}, {}],
    ids=["rows-only", "sensor-only", "neither"],
)
async def test_half_a_cadence_pair_never_replaces_this_devices(
    hass: HomeAssistant, setup_washdata_entry: Any, cadence: dict[str, Any]
) -> None:
    """Rows without their sensor were stored under this device's sensor, so the
    operational pass sized from another plug's 2 s; a sensor without its rows
    disowned this plug's 90 s. Either half alone leaves this device's pair."""
    store = await _store_measured_at_90s(hass, setup_washdata_entry)
    await store.async_import_data(_export(**cadence))
    assert store.get_profiles().keys() == {"Cotton"}
    assert store.get_cadence_summaries(POWER) == [[90.0, 90.0]]
    assert store._data.get("cadence_sensor") == POWER  # noqa: SLF001


async def test_a_whole_cadence_pair_is_taken_from_the_file(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    store = await _store_measured_at_90s(hass, setup_washdata_entry)
    await store.async_import_data(
        _export(cadence_summaries=[[2.0, 2.0]], cadence_sensor=OTHER)
    )
    assert store.get_cadence_summaries(OTHER) == [[2.0, 2.0]]
    assert store.get_cadence_summaries(POWER) == []  # another plug's rows
