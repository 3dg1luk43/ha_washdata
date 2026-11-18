"""PR #529 final review: three gaps around the door end dwell and the ghost log.

* A door sensor that loses contact while the end dwell runs: the door can close
  unseen, so the dwell is cancelled; a recovery to "on" starts a fresh one through
  the same eligibility check.
* The dwell's expiry checked the cached power, which a reading the sampling
  throttle dropped leaves low while the appliance runs again; it reads the sensor's
  live state now.
* The ghost-cycle log advised a Start Threshold above a peak that had already aged
  out of its 24 h count.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.time_utils import utc_now

from .test_issue_524_door_ends_cycle import DOOR, OPTS, _paused_washer, _tick


async def test_a_door_sensor_that_loses_contact_cancels_the_dwell(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager._remove_door_end_dwell is not None  # noqa: SLF001

    hass.states.async_set(DOOR, "unavailable")
    await hass.async_block_till_done()
    assert manager._remove_door_end_dwell is None  # noqa: SLF001
    await _tick(hass, freezer, 90, None)  # past the original dwell
    assert ends == []

    # Back, still open: a fresh dwell, and the cycle ends when it runs out.
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager._remove_door_end_dwell is not None  # noqa: SLF001
    await _tick(hass, freezer, 70, None)
    assert len(ends) == 1


async def test_back_from_unavailable_and_closed_arms_nothing(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    hass.states.async_set(DOOR, "unavailable")
    await hass.async_block_till_done()
    hass.states.async_set(DOOR, "off")
    await hass.async_block_till_done()
    assert manager._remove_door_end_dwell is None  # noqa: SLF001
    await _tick(hass, freezer, 90, None)
    assert ends == []


async def test_the_dwell_reads_the_live_power_not_the_cache(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """The plug reports 300 W, but the throttle dropped it: the cache still says
    0.5 W. The dwell must not finalise a cycle that is running again."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    manager._current_power = 0.5  # noqa: SLF001
    with patch.object(manager, "_live_power_state", return_value=(300.0, utc_now())):
        manager._door_end_dwell_fired(utc_now())  # noqa: SLF001
    assert ends == []
    assert manager._remove_door_end_dwell is not None  # re-armed  # noqa: SLF001


async def test_the_ghost_log_names_a_peak_inside_its_24_h(
    hass: HomeAssistant, setup_washdata_entry, caplog: Any
) -> None:
    from custom_components.ha_washdata.const import DOMAIN  # noqa: PLC0415

    entry = await setup_washdata_entry(options={"auto_tune_noise_events_threshold": 2})
    manager = hass.data[DOMAIN][entry.entry_id]
    t0 = utc_now()
    caplog.set_level(logging.INFO)
    with patch("custom_components.ha_washdata.manager.utc_now", return_value=t0):
        manager._handle_noise_cycle(40.0)  # noqa: SLF001
    later = t0 + timedelta(hours=25)
    with patch("custom_components.ha_washdata.manager.utc_now", return_value=later):
        manager._handle_noise_cycle(5.0)  # noqa: SLF001
    with patch("custom_components.ha_washdata.manager.utc_now", return_value=later + timedelta(minutes=1)):
        manager._handle_noise_cycle(5.0)  # noqa: SLF001
    logged = [r.getMessage() for r in caplog.records if "ghost cycles in 24 h" in r.getMessage()]
    assert logged and "up to 5.0 W" in logged[-1], logged


async def test_a_new_pick_waits_for_the_detector_before_a_paused_door_end(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Re-picked to a longer program, the detector still holds the old one until its
    next match: the door must not end the run by the old program's length."""
    manager, _ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    assert manager._door_end_may_fire()  # noqa: SLF001 - "Cotton" matched, at its end
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = "Intensive"  # noqa: SLF001 - picked, detector not told yet
    assert manager._door_end_may_fire() is False  # noqa: SLF001
    manager._current_program = "Cotton"  # noqa: SLF001 - the pick and the detector agree
    assert manager._door_end_may_fire()  # noqa: SLF001
