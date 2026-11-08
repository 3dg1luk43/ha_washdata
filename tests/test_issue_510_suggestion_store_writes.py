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

Follow-up: the pass no longer runs during a cycle at all. Everything it reads moves
only at a cycle end, so it runs once there (``close_cycle_cadence``); a running
cycle calls the generator zero times instead of every 5 minutes.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import DOMAIN, STATE_FINISHED, STATE_OFF, STATE_RUNNING
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine

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


def _spy_operational_pass(monkeypatch: Any) -> list[dict[str, Any]]:
    """Every run of the operational generator, with what it returned."""
    runs: list[dict[str, Any]] = []
    original = SuggestionEngine.generate_operational_suggestions

    def _spy(self: SuggestionEngine, p95: float, median: float) -> dict[str, Any]:
        out = original(self, p95, median)
        runs.append({"p95": p95, "median": median, "out": out})
        return out

    monkeypatch.setattr(SuggestionEngine, "generate_operational_suggestions", _spy)
    return runs


async def _one_cycle_then_idle(hass: HomeAssistant, freezer: Any, manager: Any) -> None:
    """A completed cycle, so the cadence model holds enough intervals to suggest."""
    await _feed(hass, freezer, 500, 900)
    await _feed(hass, freezer, 0, 600)  # ends it and outlasts the cycle-end save burst
    assert manager.detector.state in (STATE_OFF, STATE_FINISHED)
    assert manager.learning_manager._sample_interval_model.count >= 20  # noqa: SLF001


async def test_running_cycle_with_pending_suggestions_does_not_rewrite_the_store(
    hass: HomeAssistant, setup_washdata_entry, freezer, monkeypatch
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    await _one_cycle_then_idle(hass, freezer, manager)
    pending = manager.profile_store.get_suggestions()
    assert pending, "the cadence pass should have proposed values for this plug"

    writes = _count_main_store_writes(manager)
    runs = _spy_operational_pass(monkeypatch)
    await _feed(hass, freezer, 500, 1800)
    assert manager.detector.state == STATE_RUNNING

    assert writes == []
    assert runs == []  # no pass at all while the cycle runs
    # Still pending, same values: nothing was dropped to get there.
    after = manager.profile_store.get_suggestions()
    assert {k: v["value"] for k, v in after.items()} == {
        k: v["value"] for k, v in pending.items()
    }


async def test_running_cycle_with_applied_suggestions_does_not_rewrite_the_store(
    hass: HomeAssistant, setup_washdata_entry, freezer, monkeypatch
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
    runs = _spy_operational_pass(monkeypatch)
    await _feed(hass, freezer, 500, 1800)
    assert manager.detector.state == STATE_RUNNING

    assert writes == []
    assert runs == []
    assert manager.profile_store.get_suggestions() == {}


async def test_the_pass_runs_once_per_cycle_end_and_stores_what_it_derived(
    hass: HomeAssistant, setup_washdata_entry, freezer, monkeypatch
) -> None:
    """What the user sees after a cycle is the cycle-end pass's output, as before:
    one run, on the cadence model the cycle just committed."""
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    runs = _spy_operational_pass(monkeypatch)
    await _one_cycle_then_idle(hass, freezer, manager)
    model = manager.learning_manager._sample_interval_model  # noqa: SLF001
    assert len(runs) == 1
    assert runs[0]["p95"] == model.p95 and runs[0]["median"] == model.median

    await _feed(hass, freezer, 500, 900)  # a second cycle: nothing while it runs
    assert manager.detector.state == STATE_RUNNING
    assert len(runs) == 1
    await _feed(hass, freezer, 0, 600)
    assert manager.detector.state in (STATE_OFF, STATE_FINISHED)
    assert len(runs) == 2
    assert runs[1]["p95"] == model.p95 and runs[1]["median"] == model.median
    stored = manager.profile_store.get_suggestions()
    derived = runs[1]["out"]
    shown = {k for k in derived if k in stored}
    assert shown, "the cycle-end pass should still surface its values"
    for key in shown:
        assert stored[key]["value"] == derived[key]["value"]


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


async def test_a_deleted_suggestion_is_saved_when_the_rest_is_unchanged(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """The learning pass drops a locked key and re-sends the rest unchanged:
    apply_suggestions changes nothing itself, but the deletion must reach disk, or
    the locked suggestion comes back at the next restart."""
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    await _one_cycle_then_idle(hass, freezer, manager)
    store = manager.profile_store
    stored = {k: dict(v) for k, v in store.get_suggestions().items()}
    assert len(stored) >= 2, "the cycle should leave at least two suggestions"
    gone = next(iter(stored))
    store._data["locked_suggestions"] = [gone]  # locked, still pending

    writes = _count_main_store_writes(manager)
    manager.learning_manager._apply_suggestions_and_notify(stored)
    await hass.async_block_till_done()
    assert gone not in store.get_suggestions()
    assert len(writes) == 1
