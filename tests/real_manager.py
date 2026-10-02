"""Shared test helpers: boot a REAL WashDataManager (real ProfileStore, Store and detector).

Used by the audit regression tests that need the whole manager to run (cycle-end
tail, restore, reload) rather than a MagicMock hass, which accepts any call.
"""
from __future__ import annotations
from datetime import timedelta
from typing import Any
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed, async_fire_time_changed_exact
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.manager import WashDataManager

POWER = "sensor.plug_power"

def make_entry(hass, options: dict[str, Any] | None = None, title="Washer", entry_id=None):
    opts = {
        "power_sensor": POWER, "device_type": "washing_machine",
        "min_power": 2.0, "off_delay": 60, "min_off_gap": 60,
        "sampling_interval": 1, "watchdog_interval": 30,
        "completion_min_seconds": 300,
        "notify_finish_services": ["notify.mobile_app_phone"],
        "notify_start_services": ["notify.mobile_app_phone"],
    }
    opts.update(options or {})
    kw = {}
    if entry_id:
        kw["entry_id"] = entry_id
    entry = MockConfigEntry(domain=DOMAIN, title=title,
        data={"name": title, "power_sensor": POWER, "device_type": "washing_machine"},
        options=opts, version=3, minor_version=11, **kw)
    entry.add_to_hass(hass)
    return entry

async def boot(hass, entry) -> WashDataManager:
    hass.states.async_set(POWER, "0", {"unit_of_measurement": "W"})
    mgr = WashDataManager(hass, entry)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = mgr
    await mgr.async_setup()
    await hass.async_block_till_done()
    return mgr

async def feed(hass, freezer, watts: float, seconds: int, step: int = 30, block=True, timers=False):
    """Report `watts` every `step` seconds for `seconds` (force_update so equal values still fire)."""
    t = 0
    while t < seconds:
        freezer.tick(timedelta(seconds=step))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        if timers:
            async_fire_time_changed_exact(hass, dt_util.utcnow())
        if block:
            await hass.async_block_till_done()
        t += step

def record_notify(hass):
    calls = []
    async def _svc(call):
        calls.append(dict(call.data))
    hass.services.async_register("notify", "mobile_app_phone", _svc)
    return calls

async def idle_expire(hass, freezer, seconds: int, step: int = 60):
    """Advance time with NO sensor reports, firing timers each step (expiry / watchdog)."""
    t = 0
    while t < seconds:
        freezer.tick(timedelta(seconds=step))
        async_fire_time_changed_exact(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        t += step
