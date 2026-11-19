"""0.6.1: a 10-15 min washer programme (quick wash, spin, drain) is a completed cycle.

A cycle that ends on its timeout is recorded from its first to its last working
reading, so a nominal 10 min quick wash records 9-10 min. The washer completion
floor was 600 s, which stored it `interrupted`: no finish notification, no
`cycle_finished` trigger, no Clean state. The floor is 5 min now. These run the
default washer setup through a real Home Assistant, as a user's plug would feed it.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import DOMAIN, EVENT_CYCLE_ENDED

POWER = "sensor.quick_washer_power"


def _quick_wash(minutes: float) -> list[tuple[float, float]]:
    """(seconds, W) every 10 s: a 60 s fill at 8 W, a cold wash tumbling 150 W for
    12 s and 10 W for 3 s, a 60 s drain at 35 W, a 60 s spin at 300-450 W."""
    total = minutes * 60
    wash_end = total - 120
    out: list[tuple[float, float]] = []
    t = 0.0
    while t < total:
        if t < 60:
            p = 8.0
        elif t < wash_end:
            p = 150.0 if (t % 15) < 12 else 10.0
        elif t < total - 60:
            p = 35.0
        else:
            p = 300.0 + 150.0 * ((t // 10) % 2)
        out.append((t, p))
        t += 10.0
    return out


async def _run(hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any, minutes: float) -> list[dict]:
    hass.states.async_set(POWER, "0.5", {"unit_of_measurement": "W"})
    await setup_washdata_entry(power_sensor=POWER)
    ended: list[dict] = []
    hass.bus.async_listen(EVENT_CYCLE_ENDED, lambda e: ended.append(dict(e.data)))
    last = 0.0
    for t, p in _quick_wash(minutes):
        freezer.tick(timedelta(seconds=t - last))
        last = t
        hass.states.async_set(POWER, str(p), {"unit_of_measurement": "W"})
        async_fire_time_changed(hass, dt_util.utcnow())
        await hass.async_block_till_done()
    # Standby until the cycle has ended (the washer's off delay and min off gap).
    for _ in range(90):
        freezer.tick(timedelta(seconds=10))
        hass.states.async_set(POWER, "0.5", {"unit_of_measurement": "W"}, force_update=True)
        async_fire_time_changed(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        if ended:
            break
    return ended


@pytest.mark.parametrize("minutes", [10, 12, 15])
async def test_a_short_washer_programme_is_a_completed_cycle(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any, minutes: float
) -> None:
    ended = await _run(hass, setup_washdata_entry, freezer, minutes)
    assert len(ended) == 1, ended
    assert ended[0]["cycle_data"]["status"] == "completed", ended[0]["cycle_data"]
    manager = next(iter(hass.data[DOMAIN].values()))
    stored = manager.profile_store.get_past_cycles()[-1]
    assert stored["status"] == "completed"
    assert 8 * 60 <= float(stored["duration"]) <= minutes * 60 + 60
