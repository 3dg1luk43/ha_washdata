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
"""#510: the 5-minute suggestion pass rewrote the whole store while a cycle ran.

From the second cycle after a start, the operational pass runs every 5 minutes of
a running cycle and re-derives the same watchdog / timeout / off-delay / match
interval values (its cadence model only moves at a cycle end). Each pass saved the
whole store (megabytes) for nothing: a suggestion equal to the current setting
"deleted" a key that was not there, and a pending one was stored again unchanged.
Measured here on a 30 min run: 6 store writes, now 0.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import DOMAIN, STATE_FINISHED, STATE_OFF, STATE_RUNNING

POWER = "sensor.plug_power"
FAST = {
    "min_power": 2.0,
    "off_delay": 60,
    "min_off_gap": 60,
    "watchdog_interval": 30,
    "completion_min_seconds": 300,
}


async def _feed(
    hass: HomeAssistant, freezer: Any, watts: float, seconds: int, step: int = 30
) -> None:
    """Report about ``watts`` every ``step`` s for ``seconds``, timers included.

    The power event is delivered before the timers fire, as on a real plug: fired
    first, the watchdog's resync takes the reading and the event that follows is
    throttled, so the learning pass would never see the cycle's cadence.
    """
    t = 0
    while t < seconds:
        freezer.tick(timedelta(seconds=step))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        await hass.async_block_till_done()
        async_fire_time_changed(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        t += step


def _count_main_store_writes(manager: Any) -> list[Any]:
    """Record every write of the main store file (not the active-cycle file)."""
    store = manager.profile_store._store  # noqa: SLF001
    writes: list[Any] = []
    original = store._async_write_data  # noqa: SLF001

    async def _spy(data: dict[str, Any]) -> None:
        writes.append(dt_util.utcnow())
        await original(data)

    store._async_write_data = _spy  # noqa: SLF001
    return writes


async def _one_cycle_then_idle(hass: HomeAssistant, freezer: Any, manager: Any) -> None:
    """A completed cycle, so the cadence model holds enough intervals to suggest."""
    await _feed(hass, freezer, 500, 900)
    await _feed(hass, freezer, 0, 600)  # ends it and outlasts the cycle-end save burst
    assert manager.detector.state in (STATE_OFF, STATE_FINISHED)
    assert manager.learning_manager._sample_interval_model.count >= 20  # noqa: SLF001


async def test_running_cycle_with_pending_suggestions_does_not_rewrite_the_store(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    await _one_cycle_then_idle(hass, freezer, manager)
    pending = manager.profile_store.get_suggestions()
    assert pending, "the cadence pass should have proposed values for this plug"

    writes = _count_main_store_writes(manager)
    await _feed(hass, freezer, 500, 1800)
    assert manager.detector.state == STATE_RUNNING

    assert writes == []
    # Still pending, same values: nothing was dropped to get there.
    after = manager.profile_store.get_suggestions()
    assert {k: v["value"] for k, v in after.items()} == {
        k: v["value"] for k, v in pending.items()
    }


async def test_running_cycle_with_applied_suggestions_does_not_rewrite_the_store(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    await _one_cycle_then_idle(hass, freezer, manager)
    pending = manager.profile_store.get_suggestions()
    assert pending
    # The user applies every suggestion: each later pass now matches the setting.
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, **{k: v["value"] for k, v in pending.items()}}
    )
    await hass.async_block_till_done()
    for key in pending:
        manager.profile_store.delete_suggestion(key)
    await manager.profile_store.async_save()

    writes = _count_main_store_writes(manager)
    await _feed(hass, freezer, 500, 1800)
    assert manager.detector.state == STATE_RUNNING

    assert writes == []
    assert manager.profile_store.get_suggestions() == {}


async def test_a_changed_suggestion_is_still_saved(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Save only on change, not never: a new value still reaches disk."""
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    await _one_cycle_then_idle(hass, freezer, manager)
    key, item = next(iter(manager.profile_store.get_suggestions().items()))

    writes = _count_main_store_writes(manager)
    engine = manager.learning_manager.suggestion_engine
    engine.apply_suggestions({key: {**item, "value": item["value"] + 7}})
    await hass.async_block_till_done()
    assert len(writes) == 1
    # Re-sending it unchanged writes nothing more.
    engine.apply_suggestions({key: {**item, "value": item["value"] + 7}})
    await hass.async_block_till_done()
    assert len(writes) == 1
