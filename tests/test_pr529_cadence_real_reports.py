"""PR #529 review: the learned cadence is measured between real power reports.

The cycle-end cadence summary is stored per sensor and sizes the watchdog and
timeout suggestions. Its intervals were measured from ``_last_reading_time``,
which the watchdog's keepalives also advance: a plug reporting every 120 s in a
long low-power phase, kept alive every 30 s, learned a 30 s cadence.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import DOMAIN

POWER = "sensor.washer_power"
OPTIONS = {
    "min_power": 2.0,
    "off_delay": 3600,
    "min_off_gap": 3600,
    "watchdog_interval": 30,
    "completion_min_seconds": 300,
}


async def _tick(hass: HomeAssistant, freezer: Any, watts: float | None) -> None:
    """30 s pass; the plug reports ``watts``, or stays silent for None."""
    freezer.tick(timedelta(seconds=30))
    if watts is not None:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        await hass.async_block_till_done()  # the report lands before the watchdog tick
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def test_keepalives_do_not_shorten_the_learned_cadence(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=OPTIONS)
    manager = hass.data[DOMAIN][entry.entry_id]
    for _ in range(10):
        await _tick(hass, freezer, 200)
    assert manager.check_state() == "running"
    pending = manager.learning_manager._pending_intervals  # noqa: SLF001
    assert pending and all(abs(d - 30.0) < 1.0 for d, _ts in pending)
    pending.clear()

    det = manager.detector
    real = det.process_reading
    synthetic: list[Any] = []

    def spy(power: float, ts: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("synthetic"):
            synthetic.append(ts)
        return real(power, ts, *args, **kwargs)

    with patch.object(det, "process_reading", side_effect=spy):
        # A long low-power phase: a report every 120 s, the watchdog every 30 s.
        for _ in range(5):
            for _ in range(3):
                await _tick(hass, freezer, None)
            await _tick(hass, freezer, 1.0)
    assert manager.check_state() in ("running", "paused", "ending")
    assert len(synthetic) >= 8  # the keepalives ran between the reports
    deltas = [d for d, _ts in pending]
    assert len(deltas) == 5
    assert all(abs(d - 120.0) < 1.0 for d in deltas), deltas
