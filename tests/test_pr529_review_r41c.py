"""Review of PR #529, round 41c: cadence and suggestion inputs, each reproduced.

* A meter reporting faster than the reading throttle was learned at the
  throttle's cadence: a throttled report never moved the sensor-report clock, so
  a 1 s meter under a 2 s sampling interval was learned as 2 s.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import DOMAIN

POWER = "sensor.washer_power"


async def test_a_meter_faster_than_the_throttle_is_learned_at_its_own_cadence(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """A washer samples at 2 s; its meter reports every 1 s. The detector still
    takes every other report, but the learned interval is the meter's 1 s."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    assert manager._sampling_interval == 2.0  # noqa: SLF001

    async def report(watts: float) -> None:
        freezer.tick(timedelta(seconds=1))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        await hass.async_block_till_done()

    for _ in range(60):
        await report(200.0)
    assert manager.check_state() == "running"
    pending = manager.learning_manager._pending_intervals  # noqa: SLF001
    pending.clear()

    det = manager.detector
    with patch.object(det, "process_reading", wraps=det.process_reading) as spy:
        for _ in range(40):
            await report(200.0)
    # The throttle still drops every other report (the watchdog's resync may take
    # one of them in a report's place, which learns nothing).
    assert spy.call_count <= 21
    deltas = [d for d, _ts in pending]
    assert len(deltas) >= 18
    assert all(abs(d - 1.0) < 0.01 for d in deltas), deltas
