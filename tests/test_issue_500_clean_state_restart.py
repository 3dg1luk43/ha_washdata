# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Issue #500: the Clean state did not survive a Home Assistant restart.

The Clean overlay (cycle finished, load still inside) and its unload reminder
bookkeeping lived in memory only, and the persisted active-cycle snapshot covers
a running cycle, so a restart with the laundry still inside showed Off.

Every restart here is the one a crash leaves: the stores as they were last
written while HA ran (an HA stop does not unload the entry, so nothing written
at unload may be relied on), then ``downtime`` with HA down, then a real setup.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    async_fire_time_changed,
    async_mock_service,
)

from custom_components.ha_washdata.const import (
    CONF_DOOR_SENSOR_ENTITY,
    CONF_NOTIFY_FINISH_SERVICES,
    CONF_NOTIFY_UNLOAD_DELAY_MINUTES,
    CONF_NOTIFY_UNLOAD_MESSAGE,
    CONF_POWER_OFF_DELAY,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_PROGRESS_RESET_DELAY,
    CONF_STOP_THRESHOLD_W,
    CONF_UNLOAD_TRACK_WITHOUT_DOOR,
    DOMAIN,
    STATE_CLEAN,
    STATE_FINISHED,
    STATE_OFF,
    STATE_RUNNING,
    STORAGE_KEY,
)
from custom_components.ha_washdata.profile_store import CLEAN_STATE_KEY

POWER = "sensor.plug_power"
DOOR = "binary_sensor.washer_door"
STATE_ENTITY = "sensor.washer_state"
BUTTON = "button.washer_mark_unloaded"
FAST = {
    "min_power": 2.0,
    "off_delay": 60,
    "min_off_gap": 60,
    "watchdog_interval": 30,
    "completion_min_seconds": 300,
    CONF_PROGRESS_RESET_DELAY: 3600,
    CONF_NOTIFY_UNLOAD_DELAY_MINUTES: 0,
}
NO_DOOR = {**FAST, CONF_UNLOAD_TRACK_WITHOUT_DOOR: True}


def _manager(hass: HomeAssistant, entry: Any) -> Any:
    return hass.data[DOMAIN][entry.entry_id]


def _active_key(entry: Any) -> str:
    return f"{STORAGE_KEY}.{entry.entry_id}.active"


def _record_on_disk(hass_storage: dict[str, Any], entry: Any) -> dict[str, Any] | None:
    raw = hass_storage.get(_active_key(entry))
    data = raw.get("data") if isinstance(raw, dict) else None
    return data.get(CLEAN_STATE_KEY) if isinstance(data, dict) else None


async def _feed(
    hass: HomeAssistant, freezer: Any, watts: float, seconds: int, step: int = 30
) -> None:
    """Report ``watts`` every ``step`` s for ``seconds``, timers included."""
    t = 0
    while t < seconds:
        freezer.tick(timedelta(seconds=step))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        async_fire_time_changed(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        t += step


async def _feed_until(
    hass: HomeAssistant, freezer: Any, watts: float, until: datetime, step: int = 30
) -> None:
    remaining = (until - dt_util.utcnow()).total_seconds()
    if remaining > 0:
        await _feed(hass, freezer, watts, int(remaining), step)


async def _clean_washer(
    hass: HomeAssistant, setup_washdata_entry, freezer, options: dict[str, Any],
    standby_w: float = 0.0,
) -> tuple[Any, Any]:
    entry = await setup_washdata_entry(power_sensor=POWER, options=options)
    await _feed(hass, freezer, 500, 900)
    await _feed(hass, freezer, standby_w, 300)
    manager = _manager(hass, entry)
    assert manager.check_state() == STATE_CLEAN
    return entry, manager


async def _restart(
    hass: HomeAssistant, entry: Any, freezer: Any, hass_storage: dict[str, Any],
    downtime_s: int, before_setup: Any = None,
) -> Any:
    """Restart as a crash would: what was on disk while HA ran, then the downtime."""
    key = _active_key(entry)
    on_disk = copy.deepcopy(hass_storage.get(key))
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    if on_disk is None:
        hass_storage.pop(key, None)
    else:
        hass_storage[key] = on_disk
    freezer.tick(timedelta(seconds=downtime_s))
    if before_setup is not None:
        before_setup()
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return _manager(hass, entry)


async def test_clean_survives_a_restart_with_its_original_start(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    entry, manager = await _clean_washer(hass, setup_washdata_entry, freezer, NO_DOOR)
    start = manager._clean_state_start  # noqa: SLF001
    completed = manager._cycle_completed_time  # noqa: SLF001
    # Written while HA runs: a restart does not unload the entry.
    assert _record_on_disk(hass_storage, entry)["start"] == start.isoformat()

    manager = await _restart(hass, entry, freezer, hass_storage, 600)
    assert manager.check_state() == STATE_CLEAN
    assert hass.states.get(STATE_ENTITY).state == STATE_CLEAN
    assert hass.states.get(BUTTON).state != "unavailable"
    assert manager._clean_state_start == start  # noqa: SLF001
    assert manager.cycle_progress == 100.0
    assert manager.detector.state == STATE_FINISHED

    # The reset delay runs from the original end, not from the restart.
    await _feed_until(hass, freezer, 0, completed + timedelta(seconds=3600 - 90))
    assert manager.check_state() == STATE_CLEAN
    await _feed(hass, freezer, 0, 180)
    assert manager.check_state() == STATE_OFF
    assert _record_on_disk(hass_storage, entry) is None


async def test_reset_delay_that_ran_out_while_down_ends_off(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    entry, _ = await _clean_washer(
        hass, setup_washdata_entry, freezer, {**NO_DOOR, CONF_PROGRESS_RESET_DELAY: 900}
    )
    manager = await _restart(hass, entry, freezer, hass_storage, 1200)
    assert manager.check_state() == STATE_OFF
    assert manager.is_clean_state is False
    assert manager.cycle_progress == 0.0
    assert hass.states.get(BUTTON).state == "unavailable"
    assert _record_on_disk(hass_storage, entry) is None


async def test_a_sent_reminder_is_not_sent_again(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    calls = async_mock_service(hass, "notify", "pager")
    options = {
        **NO_DOOR,
        CONF_NOTIFY_FINISH_SERVICES: ["notify.pager"],
        CONF_NOTIFY_UNLOAD_DELAY_MINUTES: 2,
        CONF_NOTIFY_UNLOAD_MESSAGE: "NAG {duration}",
    }
    entry, _ = await _clean_washer(hass, setup_washdata_entry, freezer, options)
    await _feed(hass, freezer, 0, 240)

    def nags() -> list[Any]:
        return [c for c in calls if str(c.data.get("message", "")).startswith("NAG")]

    assert len(nags()) == 1
    assert _record_on_disk(hass_storage, entry)["notified"] is True

    manager = await _restart(hass, entry, freezer, hass_storage, 300)
    assert manager.check_state() == STATE_CLEAN
    await _feed(hass, freezer, 0, 900)
    assert len(nags()) == 1


async def test_a_pending_reminder_fires_on_its_original_schedule(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    calls = async_mock_service(hass, "notify", "pager")
    options = {
        **NO_DOOR,
        CONF_NOTIFY_FINISH_SERVICES: ["notify.pager"],
        CONF_NOTIFY_UNLOAD_DELAY_MINUTES: 20,
        CONF_NOTIFY_UNLOAD_MESSAGE: "NAG {duration}",
    }
    entry, manager = await _clean_washer(hass, setup_washdata_entry, freezer, options)
    start = manager._clean_state_start  # noqa: SLF001

    def nags() -> list[Any]:
        return [c for c in calls if str(c.data.get("message", "")).startswith("NAG")]

    manager = await _restart(hass, entry, freezer, hass_storage, 300)
    await _feed_until(hass, freezer, 0, start + timedelta(minutes=19, seconds=30))
    assert manager.check_state() == STATE_CLEAN
    assert not nags()
    await _feed(hass, freezer, 0, 90)
    assert [c.data["message"] for c in nags()] == ["NAG 20"]


async def test_a_new_cycle_after_the_restart_clears_it(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    entry, _ = await _clean_washer(hass, setup_washdata_entry, freezer, NO_DOOR)
    manager = await _restart(hass, entry, freezer, hass_storage, 120)
    assert manager.check_state() == STATE_CLEAN
    await _feed(hass, freezer, 500, 300)
    assert manager.check_state() == STATE_RUNNING
    assert manager.is_clean_state is False
    assert _record_on_disk(hass_storage, entry) is None


async def test_mark_unloaded_after_the_restart_clears_it(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    entry, _ = await _clean_washer(hass, setup_washdata_entry, freezer, NO_DOOR)
    manager = await _restart(hass, entry, freezer, hass_storage, 120)
    await hass.services.async_call("button", "press", {"entity_id": BUTTON}, blocking=True)
    await hass.async_block_till_done()
    assert manager.check_state() == STATE_FINISHED
    assert _record_on_disk(hass_storage, entry) is None
    manager = await _restart(hass, entry, freezer, hass_storage, 60)
    assert manager.is_clean_state is False


async def test_with_a_door_sensor(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    hass.states.async_set(DOOR, "off")
    options = {**FAST, CONF_DOOR_SENSOR_ENTITY: DOOR}
    entry, _ = await _clean_washer(hass, setup_washdata_entry, freezer, options)

    # Door still closed: Clean again, and opening it still confirms the unload.
    manager = await _restart(hass, entry, freezer, hass_storage, 120)
    assert manager.check_state() == STATE_CLEAN
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager.check_state() == STATE_FINISHED
    assert _record_on_disk(hass_storage, entry) is None


async def test_a_door_opened_while_down_took_the_load_out(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    hass.states.async_set(DOOR, "off")
    options = {**FAST, CONF_DOOR_SENSOR_ENTITY: DOOR}
    entry, _ = await _clean_washer(hass, setup_washdata_entry, freezer, options)
    manager = await _restart(
        hass, entry, freezer, hass_storage, 120,
        before_setup=lambda: hass.states.async_set(DOOR, "on"),
    )
    assert manager.is_clean_state is False
    assert _record_on_disk(hass_storage, entry) is None


async def test_power_based_off_keeps_it_past_the_reset_delay(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    """Power owns the Off transition (#284), so the delay is no expiry there."""
    options = {
        **NO_DOOR,
        CONF_PROGRESS_RESET_DELAY: 600,
        CONF_STOP_THRESHOLD_W: 2.0,
        CONF_POWER_OFF_THRESHOLD_W: 1.0,
        CONF_POWER_OFF_DELAY: 120,
    }
    entry, _ = await _clean_washer(
        hass, setup_washdata_entry, freezer, options, standby_w=1.5
    )
    manager = await _restart(hass, entry, freezer, hass_storage, 1800)
    assert manager.check_state() == STATE_CLEAN
    assert manager.detector.state == STATE_FINISHED
    await _feed(hass, freezer, 1.5, 300)
    assert manager.check_state() == STATE_CLEAN
    await _feed(hass, freezer, 0.2, 240)
    assert manager.check_state() == STATE_OFF


async def test_a_restored_running_cycle_supersedes_it(
    hass: HomeAssistant, setup_washdata_entry, freezer, hass_storage
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=NO_DOOR)
    await _feed(hass, freezer, 500, 300)
    running = _manager(hass, entry)
    assert running.check_state() == STATE_RUNNING
    # The periodic active-cycle save, taken now.
    await running.profile_store.async_save_active_cycle(
        running._augment_active_snapshot(running.detector.get_state_snapshot())  # noqa: SLF001
    )
    # A record no running cycle could leave behind, which must not win anyway.
    stale = dt_util.utcnow().isoformat()
    hass_storage[_active_key(entry)]["data"][CLEAN_STATE_KEY] = {
        "start": stale, "completed": stale, "detector_state": STATE_FINISHED,
    }
    manager = await _restart(hass, entry, freezer, hass_storage, 30)
    assert manager.check_state() == STATE_RUNNING
    assert manager.is_clean_state is False
    assert _record_on_disk(hass_storage, entry) is None
