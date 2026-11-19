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
"""Suggestion audit 2026-11-16: the pipeline faults, each reproduced.

* a stored suggestion outlived the condition that produced it (a pre-#527 pair one
  Apply-all away);
* a muted threshold made reconcile store a hidden cascade, so Apply all was
  refused every time;
* reconcile pulled a corrective standby-floor stop back under the standby;
* the end-gate values each pass proposed were judged one at a time, and Apply all
  wrote them together: dishwashers split in their drying phase;
* the suggestions sensor counted what Settings does not show.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_OFF_GAP,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_MIN_POWER,
    DOMAIN,
)
from custom_components.ha_washdata.learning import (
    OWNER_BATCH,
    OWNER_STANDBY_FLOOR,
    LearningManager,
)
from custom_components.ha_washdata.suggestion_engine import (
    OWNER_MODEL,
    OWNER_OPERATIONAL,
    SuggestionEngine,
    end_gate_would_split,
    reconcile_suggestions,
)


class _Store:
    """The suggestion half of ProfileStore, in memory."""

    def __init__(self, cycles: list[dict[str, Any]] | None = None, locked=()) -> None:
        self.suggestions: dict[str, Any] = {}
        self.locked = list(locked)
        self.cycles = list(cycles or [])

    def get_suggestions(self) -> dict[str, Any]:
        return self.suggestions

    def set_suggestion(self, key, value, reason=None, reason_key=None, reason_params=None, flags=None):
        from custom_components.ha_washdata.profile_store import SUGGESTION_FLAG_KEYS  # noqa: PLC0415

        entry: dict[str, Any] = {"value": value, "reason": reason}
        if reason_key is not None:
            entry["reason_key"] = reason_key
        if reason_params is not None:
            entry["reason_params"] = reason_params
        for flag in SUGGESTION_FLAG_KEYS:
            if flags and flags.get(flag):
                entry[flag] = flags[flag]
        self.suggestions[key] = entry

    def delete_suggestion(self, key) -> None:
        self.suggestions.pop(key, None)

    def get_locked_suggestions(self) -> list[str]:
        return self.locked

    def get_past_cycles(self) -> list[dict[str, Any]]:
        return self.cycles

    def get_lifetime_cycle_count(self) -> int:
        return 100

    def get_suggestion_apply_cycle_count(self) -> int:
        return 0

    async def async_save(self) -> None:
        return None


def _learning(
    hass: Any, store: _Store, options: dict[str, Any], device_type: str = "washing_machine"
) -> LearningManager:
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(options)
    hass.config_entries.async_get_entry.return_value = entry
    hass.async_create_task = lambda coro, *_a: coro.close()
    return LearningManager(hass, "e", store, device_type=device_type)


def _sug(value: float, **extra: Any) -> dict[str, Any]:
    return {"value": value, "reason": "r", **extra}


# --------------------------------------------------------------------- withdrawal


def test_a_pass_withdraws_what_it_no_longer_proposes(mock_hass: Any) -> None:
    """The #527 shape: the batch stored a pair, then withholds it."""
    store = _Store()
    lm = _learning(mock_hass, store, {CONF_START_THRESHOLD_W: 81.0, CONF_STOP_THRESHOLD_W: 54.0})
    lm._apply_suggestions_and_notify(
        {CONF_START_THRESHOLD_W: _sug(454.4), CONF_STOP_THRESHOLD_W: _sug(303.0)}, owner=OWNER_BATCH
    )
    assert store.suggestions[CONF_START_THRESHOLD_W]["owner"] == OWNER_BATCH
    lm._apply_suggestions_and_notify({}, owner=OWNER_BATCH)
    assert store.suggestions == {}


def test_another_pass_does_not_withdraw_it(mock_hass: Any) -> None:
    store = _Store()
    lm = _learning(mock_hass, store, {CONF_START_THRESHOLD_W: 5.0, CONF_STOP_THRESHOLD_W: 1.0})
    lm._apply_suggestions_and_notify({CONF_STOP_THRESHOLD_W: _sug(2.0)}, owner=OWNER_BATCH)
    lm._apply_suggestions_and_notify({}, owner=OWNER_STANDBY_FLOOR)
    assert CONF_STOP_THRESHOLD_W in store.suggestions


async def test_the_refresh_drops_entries_no_pass_owns(mock_hass: Any) -> None:
    """Stored before passes owned their entries (0.6.0 and older)."""
    store = _Store()
    store.suggestions = {CONF_START_THRESHOLD_W: _sug(454.44), CONF_STOP_THRESHOLD_W: _sug(302.96)}
    lm = _learning(mock_hass, store, {CONF_START_THRESHOLD_W: 81.0, CONF_STOP_THRESHOLD_W: 54.0})

    async def _executor(fn, *args):
        return fn(*args)

    lm.hass.async_add_executor_job = _executor
    await lm.async_refresh_suggestions()
    assert store.suggestions == {}


# --------------------------------------------------------------------- reconcile


def test_a_muted_side_is_never_stored_and_an_inverting_pair_is_withdrawn(mock_hass: Any) -> None:
    """Stop muted at 3.5 W, the batch proposes start 3.05 W: reconcile used to
    store a hidden stop 2.4 W and Apply all refused the whole set."""
    store = _Store(locked=[CONF_STOP_THRESHOLD_W])
    lm = _learning(mock_hass, store, {CONF_START_THRESHOLD_W: 6.0, CONF_STOP_THRESHOLD_W: 3.5})
    lm._apply_suggestions_and_notify(
        {CONF_START_THRESHOLD_W: _sug(3.05), CONF_OFF_DELAY: _sug(240)}, owner=OWNER_BATCH
    )
    assert CONF_STOP_THRESHOLD_W not in store.suggestions
    assert CONF_START_THRESHOLD_W not in store.suggestions  # would invert the muted stop
    assert store.suggestions[CONF_OFF_DELAY]["value"] == 240  # the rest stays applicable


def test_a_corrective_stop_is_not_pulled_back_under_the_standby() -> None:
    """A stored batch start 3.36 under a corrective floor stop 4.25: the start yields."""
    out, _changed = reconcile_suggestions(
        {
            CONF_START_THRESHOLD_W: _sug(3.36),
            CONF_STOP_THRESHOLD_W: _sug(4.25, corrective=True),
        },
        {CONF_START_THRESHOLD_W: 6.0, CONF_STOP_THRESHOLD_W: 2.0},
    )
    assert out[CONF_STOP_THRESHOLD_W]["value"] == 4.25
    assert out[CONF_START_THRESHOLD_W]["value"] > 4.25


def test_the_corrective_mark_is_stored(mock_hass: Any) -> None:
    store = _Store()
    lm = _learning(mock_hass, store, {CONF_START_THRESHOLD_W: 6.0, CONF_STOP_THRESHOLD_W: 2.0})
    lm._apply_suggestions_and_notify(
        {CONF_STOP_THRESHOLD_W: _sug(4.25, corrective=True)}, owner=OWNER_STANDBY_FLOOR
    )
    assert store.suggestions[CONF_STOP_THRESHOLD_W]["corrective"] is True


def test_reconcile_never_moves_min_power_under_its_floor() -> None:
    """PR #529: a 0.8 W stop cascaded Min Power 1 -> 0.6, and under 1 a kW sensor's
    readings are read raw (``power_unit_scale``), so no cycle would start."""
    current = {CONF_MIN_POWER: 1.0, CONF_START_THRESHOLD_W: 5.0, CONF_STOP_THRESHOLD_W: 2.0}
    out, _ = reconcile_suggestions({CONF_STOP_THRESHOLD_W: _sug(0.8)}, current)
    assert out[CONF_MIN_POWER]["value"] == 0.6  # a W sensor: unchanged
    out, _ = reconcile_suggestions(
        {CONF_STOP_THRESHOLD_W: _sug(0.8)}, current, min_power_floor=1.0
    )
    assert CONF_MIN_POWER not in out
    assert out[CONF_STOP_THRESHOLD_W]["value"] == 0.8


async def test_the_floor_follows_the_sensor_s_unit(hass: Any) -> None:
    engine = SuggestionEngine(hass, "e", _Store())
    hass.states.async_set("sensor.kw", "0.002", {"unit_of_measurement": "kW"})
    hass.states.async_set("sensor.w", "2", {"unit_of_measurement": "W"})
    floor = engine._min_power_floor  # noqa: SLF001
    assert floor({CONF_MIN_POWER: 1.0, "power_sensor": "sensor.kw"}) == 1.0
    assert floor({CONF_MIN_POWER: 1.0, "power_sensor": "sensor.w"}) == 0.0
    assert floor({CONF_MIN_POWER: 0.005, "power_sensor": "sensor.kw"}) == 0.0  # already raw
    assert floor({CONF_MIN_POWER: 1.0, "power_sensor": "sensor.missing"}) == 1.0
    # Unavailable without a unit (PR #529 review): it may come back as kW.
    hass.states.async_set("sensor.gone", "unavailable", {})
    assert floor({CONF_MIN_POWER: 1.0, "power_sensor": "sensor.gone"}) == 1.0
    hass.states.async_set("sensor.gone_w", "unavailable", {"unit_of_measurement": "W"})
    assert floor({CONF_MIN_POWER: 1.0, "power_sensor": "sensor.gone_w"}) == 0.0


# --------------------------------------------------------------------- the end gate


def _drying_cycle(i: int, drying_w: float = 0.8) -> dict[str, Any]:
    """Wash, 90 min drying under a 1.5 W stop, a 20 W terminal pump-out."""
    pts = [[float(t), 2000.0] for t in range(0, 1200, 30)]
    pts += [[1200.0 + t, drying_w] for t in range(0, 5400, 30)]
    pts += [[6600.0, 20.0], [6630.0, 20.0], [6660.0, 0.2]]
    return {
        "id": f"d{i}", "profile_name": "Eco", "status": "completed",
        "termination_reason": "timeout", "duration": 6660.0,
        "start_time": "2026-10-01T10:00:00+00:00", "power_data": pts,
    }


def test_the_drying_pause_is_bridged_only_by_the_settings_in_force() -> None:
    pts = [(float(t), float(p)) for t, p in _drying_cycle(0)["power_data"]]
    assert not end_gate_would_split(pts, 1.5, 1800, 2000, 0.05)
    assert end_gate_would_split(pts, 1.5, 180, 210, 0.75)  # Apply all
    assert end_gate_would_split(pts, 1.5, 1800, 2000, 0.75)  # end_energy alone
    assert end_gate_would_split(pts, 1.5, 180, 2000, 0.05)  # off_delay alone


def test_the_guard_withholds_end_gate_values_that_split_own_cycles() -> None:
    """Drying at 1.1 W: end_energy 0.75 Wh lets an 1800 s window (0.55 Wh) pass.

    Under 0.75 x the stop, so the flat-standby bypass waits the loose 2 h and the
    settings in force hold it (at 1.2 W it would close it after 2000 s)."""
    store = _Store([_drying_cycle(i, drying_w=1.1) for i in range(6)])
    engine = SuggestionEngine(MagicMock(), "e", store, device_type="dishwasher")
    options = {
        CONF_STOP_THRESHOLD_W: 1.5, CONF_START_THRESHOLD_W: 3.0, CONF_OFF_DELAY: 1800,
        CONF_MIN_OFF_GAP: 2000, CONF_END_ENERGY_THRESHOLD: 0.05,
    }
    engine._entry_options = lambda: dict(options)
    drop = (CONF_END_ENERGY_THRESHOLD, CONF_MIN_OFF_GAP)
    out = engine._guard_end_gate(
        {CONF_END_ENERGY_THRESHOLD: _sug(0.75), CONF_MIN_OFF_GAP: _sug(210)}, options,
        drop_order=drop, owner=OWNER_BATCH,
    )
    assert set(out) == {CONF_MIN_OFF_GAP}  # the end_energy raise was the split
    # A pending 180 s off_delay from another pass (harmless alone: 0.055 Wh over
    # its window) is applied with them, and still only end_energy goes.
    store.suggestions[CONF_OFF_DELAY] = _sug(180)
    out = engine._guard_end_gate(
        {CONF_END_ENERGY_THRESHOLD: _sug(0.75), CONF_MIN_OFF_GAP: _sug(210)}, options,
        drop_order=drop, owner=OWNER_BATCH,
    )
    assert set(out) == {CONF_MIN_OFF_GAP}


def _paused_cycle(i: int, pause_w: float = 1.5) -> dict[str, Any]:
    """800 W for an hour with one 1000 s pause (1.5 W by default)."""
    pts = [[float(t), pause_w if 1200 <= t < 2200 else 800.0] for t in range(0, 3600, 10)]
    pts.append([3600.0, 0.0])
    return {
        "id": f"p{i}", "profile_name": "Cotton", "status": "completed",
        "termination_reason": "timeout", "duration": 3600.0,
        "start_time": "2026-10-01T10:00:00+00:00", "power_data": pts,
    }


def test_a_dropped_key_is_judged_with_another_pass_s_pending_value() -> None:
    """PR #529: a 2 W stop makes the 1.5 W pause quiet, which a 1200 s gap bridges
    and a 900 s one does not. The batch proposed the stop and 900 s; dropping its
    own min_off_gap kept the stop, and the model pass's pending 900 s was applied
    with it."""
    pts = [(float(t), float(p)) for t, p in _paused_cycle(0)["power_data"]]
    assert not end_gate_would_split(pts, 1.0, 180, 900, 0.1)  # the gap alone
    assert not end_gate_would_split(pts, 2.0, 180, 1200, 0.1)  # the stop alone
    assert end_gate_would_split(pts, 2.0, 180, 900, 0.1)  # both
    store = _Store([_paused_cycle(i) for i in range(6)])
    engine = SuggestionEngine(MagicMock(), "e", store, device_type="washing_machine")
    options = {
        CONF_STOP_THRESHOLD_W: 1.0, CONF_START_THRESHOLD_W: 5.0, CONF_OFF_DELAY: 180,
        CONF_MIN_OFF_GAP: 1200, CONF_END_ENERGY_THRESHOLD: 0.1,
    }
    engine._entry_options = lambda: dict(options)
    drop = (CONF_END_ENERGY_THRESHOLD, CONF_MIN_OFF_GAP, CONF_STOP_THRESHOLD_W)

    def batch() -> dict[str, Any]:
        return {
            CONF_STOP_THRESHOLD_W: _sug(2.0), CONF_START_THRESHOLD_W: _sug(2.6),
            CONF_MIN_OFF_GAP: _sug(900),
        }

    store.suggestions[CONF_MIN_OFF_GAP] = _sug(900, owner=OWNER_MODEL)
    out = engine._guard_end_gate(batch(), options, drop_order=drop, owner=OWNER_BATCH)
    assert out == {}  # the stop goes too: the 900 s gap is written either way
    # The pass's own pending entry is withdrawn with the key, so it does not count.
    store.suggestions[CONF_MIN_OFF_GAP] = _sug(900, owner=OWNER_BATCH)
    out = engine._guard_end_gate(batch(), options, drop_order=drop, owner=OWNER_BATCH)
    assert set(out) == {CONF_STOP_THRESHOLD_W, CONF_START_THRESHOLD_W}


_JOINT_OPTIONS = {
    CONF_STOP_THRESHOLD_W: 3.0, CONF_START_THRESHOLD_W: 5.0, CONF_OFF_DELAY: 180,
    CONF_MIN_OFF_GAP: 210, CONF_END_ENERGY_THRESHOLD: 0.05,
}


def _joint_setup(mock_hass: Any) -> tuple[_Store, LearningManager]:
    """A 1000 s pause at 2 W under a 3 W stop: 0.1 Wh per 180 s window, so an
    end-energy raise to 0.15 Wh is safe only while an 1800 s off delay waits it out."""
    pts = [(float(t), float(p)) for t, p in _paused_cycle(0, pause_w=2.0)["power_data"]]
    assert not end_gate_would_split(pts, 3.0, 180, 210, 0.05)  # in force
    assert not end_gate_would_split(pts, 3.0, 1800, 1800, 0.15)  # the pending pair
    assert end_gate_would_split(pts, 3.0, 180, 210, 0.15)  # the raise alone
    store = _Store([_paused_cycle(i, pause_w=2.0) for i in range(6)])
    lm = _learning(mock_hass, store, _JOINT_OPTIONS)

    async def _executor(fn, *args):
        return fn(*args)

    lm.hass.async_add_executor_job = _executor
    store.suggestions = {
        CONF_OFF_DELAY: _sug(1800, owner=OWNER_OPERATIONAL),
        CONF_MIN_OFF_GAP: _sug(1800, cascade=True),
        CONF_END_ENERGY_THRESHOLD: _sug(0.15, owner=OWNER_BATCH),
    }
    return store, lm


async def test_the_stored_set_is_judged_after_the_gates_and_reconcile(mock_hass: Any) -> None:
    """PR #529: a pass proposing the 180 s in force passes its guard (the pending
    cascade still waits 1800 s), the equality gate deletes the pending off delay,
    reconcile drops its cascade, and the raise was left to be applied alone."""
    store, lm = _joint_setup(mock_hass)
    await lm._run_pass(  # noqa: SLF001
        lambda: {CONF_OFF_DELAY: _sug(180)}, OWNER_OPERATIONAL, lm._config_generation  # noqa: SLF001
    )
    assert CONF_OFF_DELAY not in store.suggestions
    assert CONF_END_ENERGY_THRESHOLD not in store.suggestions


async def test_a_mute_rejudges_what_is_left(mock_hass: Any) -> None:
    store, lm = _joint_setup(mock_hass)
    await lm.async_check_joint_end_gate()
    assert set(store.suggestions) == {CONF_OFF_DELAY, CONF_MIN_OFF_GAP, CONF_END_ENERGY_THRESHOLD}
    store.locked.append(CONF_OFF_DELAY)
    store.delete_suggestion(CONF_OFF_DELAY)  # what set_suggestion_locked does
    await lm.async_check_joint_end_gate()
    assert CONF_END_ENERGY_THRESHOLD not in store.suggestions


async def test_a_mute_rejudges_the_rest_on_a_real_boot(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """The mute handler's check runs as a tracked task on a real manager."""
    entry = await setup_washdata_entry(options=_JOINT_OPTIONS)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["past_cycles"] = [_paused_cycle(i, pause_w=2.0) for i in range(6)]  # noqa: SLF001
    store.set_suggestion(CONF_OFF_DELAY, 1800, reason="r", flags={"owner": OWNER_OPERATIONAL})
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.15, reason="r", flags={"owner": OWNER_BATCH})
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_suggestion_lock", "entry_id": entry.entry_id,
        "key": CONF_OFF_DELAY, "locked": True,
    })
    assert (await client.receive_json())["success"]
    await hass.async_block_till_done()
    assert CONF_OFF_DELAY not in store.get_suggestions()
    assert CONF_END_ENERGY_THRESHOLD not in store.get_suggestions()


async def test_apply_all_during_a_pass_s_joint_check_writes_no_unsafe_set(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any, monkeypatch: Any
) -> None:
    """The pass has stored its values and its joint check is still running (held
    here) when Apply all arrives: Apply judges what it writes itself."""
    import threading  # noqa: PLC0415

    entry = await setup_washdata_entry(options=_JOINT_OPTIONS)
    manager = hass.data[DOMAIN][entry.entry_id]
    store, lm = manager.profile_store, manager.learning_manager
    store._data["past_cycles"] = [_paused_cycle(i, pause_w=2.0) for i in range(6)]  # noqa: SLF001
    store.set_suggestion(CONF_OFF_DELAY, 1800, reason="r", flags={"owner": OWNER_OPERATIONAL})
    store.set_suggestion(CONF_MIN_OFF_GAP, 1800, reason="r", flags={"cascade": True})
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.15, reason="r", flags={"owner": OWNER_BATCH})
    real = SuggestionEngine.joint_end_gate_drops
    entered, release, calls = threading.Event(), threading.Event(), []

    def held_first(self: Any, stored: dict[str, Any], **kwargs: Any) -> list[str]:
        calls.append(sorted(stored))
        if len(calls) == 1:  # the pass's check
            entered.set()
            release.wait(10)
        return real(self, stored, **kwargs)

    monkeypatch.setattr(SuggestionEngine, "joint_end_gate_drops", held_first)
    task = hass.async_create_task(lm._run_pass(  # noqa: SLF001
        lambda: {CONF_OFF_DELAY: _sug(180)}, OWNER_OPERATIONAL, lm._config_generation  # noqa: SLF001
    ))
    assert await hass.async_add_executor_job(entered.wait, 10)
    assert set(store.get_suggestions()) == {CONF_END_ENERGY_THRESHOLD}  # the window
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
        "keys": [CONF_END_ENERGY_THRESHOLD],
    })
    res = await client.receive_json()
    release.set()
    await task
    assert res["success"] and res["result"]["applied"] == []
    assert entry.options[CONF_END_ENERGY_THRESHOLD] == 0.05
    assert CONF_END_ENERGY_THRESHOLD not in store.get_suggestions()


async def test_apply_judges_a_selected_cascade_as_written(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529: the cascade flag left a selected 1800 s Min Off Gap out of the
    apply-time check, so a raise it makes safe was withdrawn."""
    pts = [(float(t), float(p)) for t, p in _paused_cycle(0, pause_w=1.0)["power_data"]]
    assert not end_gate_would_split(pts, 3.0, 60, 60, 0.01)  # in force
    assert end_gate_would_split(pts, 3.0, 60, 60, 1.0)  # the raise alone
    assert not end_gate_would_split(pts, 3.0, 60, 1800, 1.0)  # with the gap
    entry = await setup_washdata_entry(options={
        CONF_STOP_THRESHOLD_W: 3.0, CONF_START_THRESHOLD_W: 5.0, CONF_OFF_DELAY: 60,
        CONF_MIN_OFF_GAP: 60, CONF_END_ENERGY_THRESHOLD: 0.01,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["past_cycles"] = [_paused_cycle(i, pause_w=1.0) for i in range(6)]  # noqa: SLF001
    store.set_suggestion(CONF_MIN_OFF_GAP, 1800, reason="r", flags={"cascade": True})
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 1.0, reason="r", flags={"owner": OWNER_BATCH})
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
        "keys": [CONF_MIN_OFF_GAP, CONF_END_ENERGY_THRESHOLD],
    })
    res = await client.receive_json()
    assert sorted(res["result"]["applied"]) == [CONF_END_ENERGY_THRESHOLD, CONF_MIN_OFF_GAP]
    await hass.async_block_till_done()


async def test_apply_rechecks_the_pair_against_a_save_it_waited_for(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529: the pair was checked on options read before the lock, so a save of
    start 6 W landing while Apply waited let it write stop 8 W above it."""
    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry(options={
        CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 5.0,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_STOP_THRESHOLD_W, 8.0, reason="r", flags={"owner": OWNER_BATCH})
    client = await hass_ws_client(hass)
    lock = ws_api._entry_options_lock(hass, entry.entry_id)  # noqa: SLF001
    await lock.acquire()
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
        "keys": [CONF_STOP_THRESHOLD_W],
    })
    for _ in range(20):  # let the handler read its options and reach the lock
        await asyncio.sleep(0)
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_START_THRESHOLD_W: 6.0}
    )
    lock.release()
    res = await client.receive_json()
    assert not res["success"] and res["error"]["code"] == "invalid_threshold_pair"
    assert entry.options[CONF_STOP_THRESHOLD_W] == 5.0
    await hass.async_block_till_done()


def test_the_bridge_is_measured_at_the_stop_threshold() -> None:
    """The 20 W pump-out resumes the drying run at a 1.5 W stop; at 2% of the
    2 kW peak (40 W) it never did, and the 90 min pause went unmeasured. With the
    180 s default off_delay the energy gate passes 0.8 W (0.04 Wh <= 0.05 Wh), so
    all of it needs bridging."""
    store = _Store([_drying_cycle(i) for i in range(6)])
    engine = SuggestionEngine(MagicMock(), "e", store, device_type="dishwasher")
    engine._entry_options = lambda: {CONF_OFF_DELAY: 180, CONF_END_ENERGY_THRESHOLD: 0.05}
    longest, _spans, _traced = engine._measured_bridge_requirement(store.cycles, 1.5)
    assert longest >= 5400 - 60


def test_a_pause_the_energy_gate_holds_needs_no_bridge() -> None:
    """Over an 1800 s window the same drying is 0.4 Wh: the gate holds it, so
    min_off_gap is not raised to cover it (it was: 3600 s, a Rinse ended 23 min
    late)."""
    store = _Store([_drying_cycle(i) for i in range(6)])
    engine = SuggestionEngine(MagicMock(), "e", store, device_type="dishwasher")
    engine._entry_options = lambda: {CONF_OFF_DELAY: 1800, CONF_END_ENERGY_THRESHOLD: 0.05}
    assert engine._measured_bridge_requirement(store.cycles, 1.5) is None


# --------------------------------------------------------------------- the sensor


async def test_the_sensor_counts_what_settings_shows(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    entry = await setup_washdata_entry(options={CONF_OFF_DELAY: 300})
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store.set_suggestion(CONF_OFF_DELAY, 300, reason="already applied by hand")
    store.set_suggestion("auto_label_confidence", 0.9, reason="retired key")
    manager.notify_update()
    await hass.async_block_till_done()
    entity_id = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_suggestions")
    state = hass.states.get(entity_id)
    assert state.state == "0"
    assert state.attributes["has_actionable_suggestions"] is False


def test_the_latest_moment_is_found_between_two_grid_points() -> None:
    """PR #529 review: the window energy rises through the limit between two grid
    points, so the latest moment at or under it is that crossing. 0 W, then 5 W
    from 1000 s into the pause: 180 s x 5 W passes 0.05 Wh at 1036 s, not 1000.
    Plus the 100 s since the last high reading, which the quiet timer credits (too
    few intervals to know a cadence: uncapped, as in the detector)."""
    from custom_components.ha_washdata.suggestion_engine import end_gate_pause_needs  # noqa: PLC0415

    pts = [(0.0, 100.0), (100.0, 0.0), (1100.0, 5.0), (2100.0, 50.0)]
    assert end_gate_pause_needs(pts, 6.0, 180, 0.05) == [1136.0]
    assert end_gate_would_split(pts, 6.0, 180, 1120, 0.05)
    assert not end_gate_would_split(pts, 6.0, 180, 1140, 0.05)


def test_a_non_finite_reading_does_not_stall_the_pause_scan() -> None:
    """PR #529 review: a NaN reading is neither at nor below the stop, so the scan
    never advanced past it and the pass (Apply all, inside the options lock) hung.
    Non-finite readings are skipped; the pause reads as it does without them."""
    from custom_components.ha_washdata.suggestion_engine import end_gate_pause_needs  # noqa: PLC0415

    nan = float("nan")
    pts = [(0.0, 100.0), (100.0, 0.0), (1100.0, 5.0), (2100.0, 50.0)]
    with_nan = [(0.0, nan), *pts[:2], (600.0, nan), (700.0, float("inf")), *pts[2:]]
    assert end_gate_pause_needs(with_nan, 6.0, 180, 0.05) == [1136.0]
    assert end_gate_would_split(with_nan, 6.0, 180, 1120, 0.05)
    assert end_gate_pause_needs([(0.0, nan), (10.0, nan)], 6.0, 180, 0.05) == []


async def test_a_pass_from_replaced_options_applies_nothing(mock_hass: Any) -> None:
    """PR #529 review: a pass started before a refresh carries the old options
    snapshot; a refresh in between makes its result stale."""
    store = _Store()
    lm = _learning(mock_hass, store, {CONF_START_THRESHOLD_W: 81.0, CONF_STOP_THRESHOLD_W: 54.0})
    gen = lm._config_generation  # noqa: SLF001
    lm._config_generation += 1  # noqa: SLF001  - what a refresh does first
    lm._apply_if_current(gen, {CONF_START_THRESHOLD_W: _sug(454.4)}, OWNER_BATCH)  # noqa: SLF001
    assert store.suggestions == {}
    lm._apply_if_current(lm._config_generation, {CONF_START_THRESHOLD_W: _sug(454.4)}, OWNER_BATCH)  # noqa: SLF001
    assert CONF_START_THRESHOLD_W in store.suggestions


async def test_passes_run_one_at_a_time(mock_hass: Any) -> None:
    """PR #529 review: a cycle end spawns several passes, and each end-gate guard
    checks against what the others have pending. Run side by side, the second
    read the pending set from before the first stored, so values each safe alone
    were stored together. Now the second scans after the first has stored."""
    import asyncio  # noqa: PLC0415

    async def _executor(fn, *args):
        await asyncio.sleep(0)  # the scan is off the loop: others run meanwhile
        out = fn(*args)
        await asyncio.sleep(0)
        return out

    mock_hass.async_add_executor_job = _executor
    store = _Store()
    lm = _learning(mock_hass, store, {CONF_START_THRESHOLD_W: 81.0, CONF_STOP_THRESHOLD_W: 54.0})
    seen: list[set[str]] = []

    def first() -> dict[str, Any]:
        return {CONF_START_THRESHOLD_W: _sug(454.4)}

    def second() -> dict[str, Any]:
        seen.append(set(store.get_suggestions()))
        return {}

    gen = lm._config_generation  # noqa: SLF001
    await asyncio.gather(
        lm._async_scan_and_apply(first, "first", OWNER_BATCH, gen),  # noqa: SLF001
        lm._async_scan_and_apply(second, "second", OWNER_STANDBY_FLOOR, gen),  # noqa: SLF001
    )
    assert seen == [{CONF_START_THRESHOLD_W}]


async def test_a_failed_joint_check_aborts_apply_all(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: the joint end-gate check swallowed its own failure as "nothing
    to withdraw", so Apply all wrote a set nobody judged. Apply now writes nothing."""
    from unittest.mock import patch  # noqa: PLC0415

    entry = await setup_washdata_entry(options={
        CONF_STOP_THRESHOLD_W: 3.0, CONF_START_THRESHOLD_W: 5.0, CONF_OFF_DELAY: 60,
        CONF_MIN_OFF_GAP: 60, CONF_END_ENERGY_THRESHOLD: 0.01,
    })
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["past_cycles"] = [_paused_cycle(i, pause_w=1.0) for i in range(6)]  # noqa: SLF001
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 1.0, reason="r", flags={"owner": OWNER_BATCH})
    engine_cls = type(manager.learning_manager.suggestion_engine)
    client = await hass_ws_client(hass)
    with patch.object(engine_cls, "_end_gate_splits", side_effect=RuntimeError("boom")):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        res = await client.receive_json()
    assert res["success"] is False and res["error"]["code"] == "end_gate_check_failed"
    await hass.async_block_till_done()
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_END_ENERGY_THRESHOLD] == 0.01


async def test_a_mute_that_lands_while_apply_waits_is_respected(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: Apply all picked its suggestions before waiting for the
    options lock, so a mute completed meanwhile was still written."""
    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.02, reason="r", flags={"owner": OWNER_BATCH})
    client = await hass_ws_client(hass)
    lock = ws_api._entry_options_lock(hass, entry.entry_id)  # noqa: SLF001
    await lock.acquire()  # a settings save in progress
    try:
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        for _ in range(5):
            await asyncio.sleep(0)
        await store.set_suggestion_locked(CONF_END_ENERGY_THRESHOLD, True)  # muted meanwhile
    finally:
        lock.release()
    res = await client.receive_json()
    assert res["success"] and res["result"]["applied"] == []
    await hass.async_block_till_done()
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_END_ENERGY_THRESHOLD] == 0.01


async def test_cadence_rows_belong_to_their_sensor_and_go_with_a_wipe(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: after a sensor swap the old plug's cadence rows outvoted the
    new one's; after a wipe the in-memory interval model still sized the pass."""
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store, lm = manager.profile_store, manager.learning_manager
    sensor = lm._power_sensor()  # noqa: SLF001
    assert sensor
    for _ in range(3):
        store.add_cadence_summary(60.0, 30.0, sensor="sensor.old_plug")
    assert store.get_cadence_summaries(sensor) == []  # another plug's rows
    store.add_cadence_summary(10.0, 5.0, sensor=sensor)
    assert store.get_cadence_summaries(sensor) == [[10.0, 5.0]]  # replaced, not mixed

    from custom_components.ha_washdata.time_utils import utc_now  # noqa: PLC0415

    lm._sample_interval_model.add_samples([30.0] * 40, utc_now())  # noqa: SLF001
    assert lm.cadence() is not None
    gen = lm._config_generation  # noqa: SLF001
    lm._operational_pass_due = True  # noqa: SLF001
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({"type": f"{DOMAIN}/wipe_history", "entry_id": entry.entry_id})
    res = await client.receive_json()
    assert res["success"]
    assert lm.cadence() is None
    # A pass in flight read the old cadence: its result is discarded, none is due.
    assert lm._config_generation > gen and lm._operational_pass_due is False  # noqa: SLF001


async def test_a_joint_verdict_on_a_set_that_changed_withdraws_nothing(mock_hass: Any) -> None:
    """PR #529 review: the verdict was taken with off_delay 180 s pending; once that
    is 1800 s the energy raise may be safe, so it must not be withdrawn."""
    store = _Store()
    lm = _learning(mock_hass, store, {})
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.4, reason="r", flags={"owner": OWNER_BATCH})
    store.set_suggestion(CONF_OFF_DELAY, 1800, reason="r", flags={"owner": OWNER_OPERATIONAL})
    judged = {CONF_END_ENERGY_THRESHOLD: {"value": 0.4}, CONF_OFF_DELAY: {"value": 180}}
    lm._withdraw_joint_drops(judged, [CONF_END_ENERGY_THRESHOLD])  # noqa: SLF001
    assert CONF_END_ENERGY_THRESHOLD in store.get_suggestions()


def test_a_failed_pause_scan_raises_in_strict_mode_only() -> None:
    """PR #529 review: a scan failure read as "no pause", so Apply all's strict
    check could still pass an unverified set. Background passes keep the fallback."""
    from unittest.mock import patch  # noqa: PLC0415

    import pytest  # noqa: PLC0415

    from custom_components.ha_washdata import suggestion_engine as se  # noqa: PLC0415

    pts = [(0.0, 100.0), (100.0, 0.0), (1100.0, 5.0), (2100.0, 50.0)]
    with patch.object(se, "held_wh_until", side_effect=RuntimeError("boom")):
        assert se.end_gate_pause_needs(pts, 6.0, 180, 0.05) == []
        assert se.end_gate_would_split(pts, 6.0, 180, 200, 0.05) is False
        with pytest.raises(RuntimeError):
            se.end_gate_pause_needs(pts, 6.0, 180, 0.05, strict=True)
        with pytest.raises(RuntimeError):
            se.end_gate_would_split(pts, 6.0, 180, 200, 0.05, strict=True)


async def test_min_power_alone_does_not_move_an_unset_stop(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: with Stop unset it follows Min Power (#450), so applying a
    Min Power cascade alone moved Stop past every threshold check."""
    entry = await setup_washdata_entry(options={CONF_MIN_POWER: 5.0, CONF_START_THRESHOLD_W: 10.0})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_MIN_POWER, 1.0, reason="r", flags={"cascade": True})
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id, "keys": [CONF_MIN_POWER],
    })
    res = await client.receive_json()
    assert res["success"] and res["result"]["applied"] == []
    await hass.async_block_till_done()
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_MIN_POWER] == 5.0


async def test_an_explicit_stop_lets_min_power_apply(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    entry = await setup_washdata_entry(options={
        CONF_MIN_POWER: 5.0, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 6.0,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_MIN_POWER, 4.0, reason="r", flags={"owner": OWNER_BATCH})
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id, "keys": [CONF_MIN_POWER],
    })
    res = await client.receive_json()
    assert res["success"] and res["result"]["applied"] == [CONF_MIN_POWER]
    await hass.async_block_till_done()


async def test_apply_writes_nothing_after_a_reload_it_waited_through(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: a reload while Apply awaited its check left it writing to the
    old manager's store."""
    from unittest.mock import MagicMock, patch  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    manager = hass.data[DOMAIN][entry.entry_id]
    manager.profile_store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.02, reason="r", flags={"owner": OWNER_BATCH})
    real = ws_api._get_manager  # noqa: SLF001
    calls = {"n": 0}

    def reloaded(h: Any, eid: str) -> Any:
        calls["n"] += 1
        return real(h, eid) if calls["n"] == 1 else MagicMock()

    client = await hass_ws_client(hass)
    with patch.object(ws_api, "_get_manager", side_effect=reloaded):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        res = await client.receive_json()
    assert res["success"] is False and res["error"]["code"] == "entry_reloaded"
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_END_ENERGY_THRESHOLD] == 0.01


async def test_the_wipe_retires_cadence_work_before_it_awaits_the_store(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: a cadence pass that finished while the wipe awaited its write
    still held the current generation and could refill the cleared suggestions."""
    from unittest.mock import patch  # noqa: PLC0415

    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store, lm = manager.profile_store, manager.learning_manager
    gen = lm._config_generation  # noqa: SLF001
    seen: list[int] = []

    async def _clear() -> None:
        seen.append(lm._config_generation)  # noqa: SLF001
        raise RuntimeError("disk full")

    client = await hass_ws_client(hass)
    with patch.object(store, "clear_all_data", side_effect=_clear):
        await client.send_json_auto_id({"type": f"{DOMAIN}/wipe_history", "entry_id": entry.entry_id})
        res = await client.receive_json()
    assert res["success"] is False
    assert seen and seen[0] > gen


async def test_a_joint_verdict_from_before_a_restore_withdraws_nothing(mock_hass: Any) -> None:
    """PR #529 review: Undo Import replaced the history while the joint check ran.
    The old verdict must not withdraw from the restored set, which is judged again
    on its own history."""
    from types import SimpleNamespace  # noqa: PLC0415
    from unittest.mock import AsyncMock  # noqa: PLC0415

    mock_hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    store = _Store()
    store._data_generation = 0  # noqa: SLF001
    lm = _learning(mock_hass, store, {})
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.4, reason="r", flags={"owner": OWNER_BATCH})
    calls: list[int] = []

    def _drops(stored: dict[str, Any]) -> list[str]:
        calls.append(store._data_generation)  # noqa: SLF001
        if len(calls) == 1:
            store._data_generation += 1  # noqa: SLF001 - restored meanwhile, same pending value
            return [CONF_END_ENERGY_THRESHOLD]
        return []  # the restored history has no pause it splits

    lm.suggestion_engine = SimpleNamespace(
        for_job=lambda: SimpleNamespace(joint_end_gate_drops=_drops),
        apply_suggestions=lambda *a, **k: None,
    )
    await lm._enforce_joint_end_gate()  # noqa: SLF001
    assert calls == [0, 1]
    assert CONF_END_ENERGY_THRESHOLD in store.get_suggestions()


async def test_cadence_is_filed_under_the_sensor_that_reported_it(
    hass: Any, setup_washdata_entry: Any
) -> None:
    """PR #529 review: sensor B selected while A still reports the running cycle
    (the swap waits for the end). A's intervals were stored as B's, and B's cadence
    fell back to A's in-memory model."""
    from datetime import timedelta  # noqa: PLC0415
    from unittest.mock import patch  # noqa: PLC0415

    from custom_components.ha_washdata.time_utils import utc_now  # noqa: PLC0415

    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store, lm = manager.profile_store, manager.learning_manager
    t0 = utc_now()
    with patch.object(lm, "_power_sensor", return_value="sensor.plug_b"):
        for i in range(40):
            lm.process_power_reading(
                300.0, t0 + timedelta(seconds=30 * (i + 1)), t0 + timedelta(seconds=30 * i),
                sensor="sensor.plug_a",
            )
        assert lm.close_cycle_cadence({"status": "completed", "termination_reason": "smart"})
        await hass.async_block_till_done()  # the summary is computed in the executor
        assert store.get_cadence_summaries("sensor.plug_b") == []
        assert store.get_cadence_summaries("sensor.plug_a") != []
        assert lm.cadence() is None  # nothing measured on B yet
    with patch.object(lm, "_power_sensor", return_value="sensor.plug_a"):
        assert lm.cadence() is not None


async def test_apply_stops_at_a_reload_between_its_writes(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: a reload while Apply awaited the suggestion clear left it
    writing on to the old manager's store, then saying nothing about what it had
    already written. The settings history is written after the options, so such
    an Apply records no change it did not make."""
    from unittest.mock import MagicMock, patch  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.02, reason="r", flags={"owner": OWNER_BATCH})
    real_get = ws_api._get_manager  # noqa: SLF001
    state = {"reloaded": False}

    def current(h: Any, eid: str, _real=real_get) -> Any:
        return MagicMock() if state["reloaded"] else _real(h, eid)

    real_clear = store.clear_suggestions

    async def clear(*a: Any) -> None:
        await real_clear(*a)
        state["reloaded"] = True

    cleared = MagicMock(side_effect=clear)
    client = await hass_ws_client(hass)
    with patch.object(ws_api, "_get_manager", side_effect=current), \
            patch.object(store, "clear_suggestions", cleared):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        res = await client.receive_json()
    assert res["success"] is False and res["error"]["code"] == "entry_reloaded"
    assert "cleared" in res["error"]["message"]
    assert cleared.called
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_END_ENERGY_THRESHOLD] == 0.01
    assert store.get_settings_changelog() == []


async def test_a_stop_row_is_stamped_when_the_apply_wrote_it(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any, freezer: Any
) -> None:
    """PR #529 review: the Stop row was stamped before Apply's awaited writes, so a
    cycle that ended meanwhile, under the old Stop, was judged under the new one."""
    from datetime import timedelta  # noqa: PLC0415
    from unittest.mock import patch  # noqa: PLC0415

    from homeassistant.util import dt as dt_util  # noqa: PLC0415

    from custom_components.ha_washdata import suggestion_engine as se  # noqa: PLC0415

    entry = await setup_washdata_entry(options={
        CONF_MIN_POWER: 2.0, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 5.0,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_STOP_THRESHOLD_W, 4.0, reason="r", flags={"owner": OWNER_BATCH})
    real_clear = store.clear_suggestions
    ended: list[Any] = []

    async def clear(*a: Any) -> None:
        freezer.tick(timedelta(seconds=5))  # the write takes a while
        ended.append(dt_util.utcnow())  # a cycle ends meanwhile, under 5 W
        freezer.tick(timedelta(seconds=5))
        await real_clear(*a)

    client = await hass_ws_client(hass)
    with patch.object(store, "clear_suggestions", side_effect=clear):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_STOP_THRESHOLD_W],
        })
        res = await client.receive_json()
    assert res["success"] and res["result"]["applied"] == [CONF_STOP_THRESHOLD_W], res
    changes = se.stop_threshold_changes(store.get_settings_changelog())
    assert [old for _when, old in changes] == [5.0]
    cycle = {"end_time": ended[0].isoformat()}
    assert se._stop_at_end(cycle, changes, 4.0) == 5.0  # noqa: SLF001


async def test_a_pass_from_before_a_restore_applies_nothing(mock_hass: Any) -> None:
    """PR #529 review: Undo Import replaces the store without a settings change, so
    the config generation stays put and a pass in flight stored its result into the
    restored set."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    store = _Store()
    store._data_generation = 0  # noqa: SLF001
    lm = _learning(mock_hass, store, {CONF_START_THRESHOLD_W: 81.0, CONF_STOP_THRESHOLD_W: 54.0})

    def _scan() -> dict[str, Any]:
        store._data_generation += 1  # noqa: SLF001 - the undo lands while it scans
        return {CONF_START_THRESHOLD_W: _sug(454.4)}

    mock_hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    assert await lm._run_pass(_scan, OWNER_BATCH, lm._config_generation) is None  # noqa: SLF001
    assert store.suggestions == {}


def test_user_stopped_runs_do_not_set_a_programme_s_length() -> None:
    """PR #529 review: three runs the user cut short at 2400 s made 2400 s the
    programme's length, so a merged 2400 s trace passed as one run of it."""
    from custom_components.ha_washdata.suggestion_engine import _label_median_durations  # noqa: PLC0415

    def run(d: float, reason: str) -> dict[str, Any]:
        return {"status": "completed", "termination_reason": reason, "profile_name": "Eco", "duration": d}

    cut = [run(2400.0, "user") for _ in range(3)]
    assert _label_median_durations(cut) == {}
    assert _label_median_durations(cut + [run(5400.0, "timeout") for _ in range(3)]) == {"Eco": 5400.0}


def test_standby_before_the_first_working_reading_is_no_pause() -> None:
    """PR #529 review: 500 s of standby ahead of the cycle read as a pause it
    bridged, so the joint check could refuse a Min Off Gap that was safe."""
    from custom_components.ha_washdata.suggestion_engine import end_gate_pause_needs  # noqa: PLC0415

    lead = [(float(t), 0.5) for t in range(0, 500, 10)]
    work = [(500.0 + t, 300.0) for t in range(0, 600, 10)]
    assert end_gate_pause_needs(lead + work, 5.0, 180, 0.05) == []
    # The same quiet after work began is a pause.
    pause = [(1100.0 + t, 0.5) for t in range(0, 500, 10)]
    assert end_gate_pause_needs(work + pause + [(1600.0, 300.0)], 5.0, 180, 0.05) != []


def test_the_settled_tail_is_walked_once_per_stored_trace() -> None:
    """PR #529 review: the device-list callback walked and sorted every hand-stopped
    cycle's tail on each poll (23.6 ms over eight 3 h tails); an edited trace is
    walked again."""
    from unittest.mock import patch  # noqa: PLC0415

    from custom_components.ha_washdata import suggestion_engine as se  # noqa: PLC0415

    pts = [[float(t), 2000.0] for t in range(0, 600, 10)] + [[600.0 + t, 3.4] for t in range(0, 3600, 10)]
    cycle = {"id": "tail-1", "status": "completed", "termination_reason": "user",
             "power_data": pts, "signature": {"max_power": 2000.0}}
    real = se._tail_settled_w_uncached  # noqa: SLF001
    with patch.object(se, "_tail_settled_w_uncached", side_effect=real) as walked:
        for _ in range(5):
            assert se.detect_standby_above_stop([cycle, dict(cycle)], 2.56) is not None
        assert walked.call_count == 1
        cycle["power_data"] = pts[:-60]  # trimmed by the user
        se.detect_standby_above_stop([cycle], 2.56)
        assert walked.call_count == 2


async def test_a_batch_taken_before_a_restore_is_discarded_and_rerun(mock_hass: Any) -> None:
    """PR #529 review: the batch takes its cycles at spawn. A restore before the
    pass first ran passed the store check taken there; and a discarded batch left
    its count, so the restored history waited five more cycles for one."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    store = _Store()
    store._data_generation = 0  # noqa: SLF001
    lm = _learning(mock_hass, store, {})
    mock_hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    spawned: list[Any] = []
    lm._spawn = spawned.append  # noqa: SLF001
    lm._batch_cycles = lambda: [{"status": "completed", "termination_reason": "timeout"}] * 6  # noqa: SLF001
    lm._maybe_run_batch_simulation()  # noqa: SLF001
    assert lm._last_batch_simulation_count == 6  # noqa: SLF001
    store._data_generation += 1  # noqa: SLF001 - Undo Import before the task first runs
    lm.suggestion_engine.for_job = lambda: type("E", (), {
        "run_batch_simulation": staticmethod(lambda cycles: {CONF_START_THRESHOLD_W: _sug(9.0)}),
    })()
    await spawned[0]
    assert store.suggestions == {}
    assert lm._last_batch_simulation_count == 0  # noqa: SLF001


async def test_a_new_sensor_without_cadence_drops_the_old_one_s_entries(mock_hass: Any) -> None:
    """PR #529 review: after a power sensor change the refresh kept the watchdog
    value measured on the old plug, one Apply away, since the operational pass had
    no cadence to run on and so never withdrew it."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    from custom_components.ha_washdata.const import CONF_WATCHDOG_INTERVAL  # noqa: PLC0415

    mock_hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    store = _Store()
    lm = _learning(mock_hass, store, {})
    store.set_suggestion(CONF_WATCHDOG_INTERVAL, 95, reason="r", flags={"owner": OWNER_OPERATIONAL})
    assert lm.cadence() is None
    await lm.async_refresh_suggestions()
    assert CONF_WATCHDOG_INTERVAL not in store.get_suggestions()


async def test_undo_import_forgets_the_cadence_measured_before_it(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: the wipe reset the cadence evidence, Undo Import did not, so
    the in-memory intervals from before it sized the next operational pass. Reset
    once the restore point has replaced the store, before its first write."""
    from unittest.mock import patch  # noqa: PLC0415

    from custom_components.ha_washdata.time_utils import utc_now  # noqa: PLC0415

    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["profiles"]["P"] = {}  # noqa: SLF001  (an import needs a profile)
    await store.async_import_data(store.export_data())  # leaves a restore point
    lm = manager.learning_manager
    lm._sample_interval_model.add_samples([30.0] * 40, utc_now())  # noqa: SLF001
    assert lm.cadence() is not None
    real_save = store.async_save
    at_first_write: list[Any] = []

    async def save(*a: Any, **k: Any) -> Any:
        at_first_write.append(lm.cadence())
        return await real_save(*a, **k)

    client = await hass_ws_client(hass)
    with patch.object(store, "async_save", side_effect=save):
        await client.send_json_auto_id({"type": f"{DOMAIN}/undo_import", "entry_id": entry.entry_id})
        assert (await client.receive_json())["success"]
    assert at_first_write and at_first_write[0] is None
    assert lm.cadence() is None


async def test_a_refused_undo_import_keeps_the_cadence(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: with no restore point Undo Import answers not_found, and the
    running cycle's held intervals and the committed cadence are left as they were."""
    from custom_components.ha_washdata.time_utils import utc_now  # noqa: PLC0415

    entry = await setup_washdata_entry()
    lm = hass.data[DOMAIN][entry.entry_id].learning_manager
    lm._sample_interval_model.add_samples([30.0] * 40, utc_now())  # noqa: SLF001
    lm._pending_intervals.append((30.0, utc_now()))  # noqa: SLF001
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({"type": f"{DOMAIN}/undo_import", "entry_id": entry.entry_id})
    res = await client.receive_json()
    assert res["success"] is False and res["error"]["code"] == "not_found"
    assert lm.cadence() is not None
    assert len(lm._pending_intervals) == 1  # noqa: SLF001


async def test_undo_import_schedules_its_rebuilds_on_the_entry_s_current_manager(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """A reload during the undo's writes retires the manager the handler started
    with; the post-restore rebuilds go to the entry's manager as it is after them,
    and to none when that one is shut down."""
    from types import SimpleNamespace  # noqa: PLC0415
    from unittest.mock import Mock, patch  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry()
    old = hass.data[DOMAIN][entry.entry_id]
    store = old.profile_store
    store._data["profiles"]["P"] = {}  # noqa: SLF001
    names = (
        "async_schedule_banked_tail_repair", "async_schedule_energy_rebuild",
        "async_schedule_cut_short_rebuild", "notify_update",
    )
    new = SimpleNamespace(_is_shutdown=False, **{n: Mock() for n in names})
    real_restore_options = ws_api._async_restore_entry_options  # noqa: SLF001

    async def reloaded_meanwhile(*a: Any) -> None:
        await real_restore_options(*a)
        old._is_shutdown = True  # noqa: SLF001
        if successor is not None:
            hass.data[DOMAIN][entry.entry_id] = successor

    client = await hass_ws_client(hass)
    try:
        for successor in (new, None):
            await store.async_import_data(store.export_data(), entry_options=dict(entry.options))
            with patch.object(ws_api, "_async_restore_entry_options", side_effect=reloaded_meanwhile), \
                    patch.multiple(old, **{n: Mock() for n in names}):
                await client.send_json_auto_id(
                    {"type": f"{DOMAIN}/undo_import", "entry_id": entry.entry_id}
                )
                assert (await client.receive_json())["success"]
                assert not any(getattr(old, n).called for n in names)
            hass.data[DOMAIN][entry.entry_id] = old
            old._is_shutdown = False  # noqa: SLF001
        assert all(getattr(new, n).call_count == 1 for n in names)
    finally:
        hass.data[DOMAIN][entry.entry_id] = old
        old._is_shutdown = False  # noqa: SLF001


async def test_apply_writes_nothing_after_a_restore_it_waited_through(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: Undo Import replaced the store while Apply awaited its
    end-gate check; the manager was the same, so Apply wrote values judged on the
    replaced cycles and cleared the restored suggestions."""
    from unittest.mock import patch  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.02, reason="r", flags={"owner": OWNER_BATCH})
    real = ws_api._drop_unsafe_end_gate_updates  # noqa: SLF001

    async def restored_meanwhile(*a: Any) -> bool:
        ok = await real(*a)
        store._data_generation = getattr(store, "_data_generation", 0) + 1  # noqa: SLF001
        return ok

    client = await hass_ws_client(hass)
    with patch.object(ws_api, "_drop_unsafe_end_gate_updates", side_effect=restored_meanwhile):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        res = await client.receive_json()
    assert res["success"] is False and res["error"]["code"] == "entry_reloaded"
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_END_ENERGY_THRESHOLD] == 0.01
    assert CONF_END_ENERGY_THRESHOLD in store.get_suggestions()


def test_no_min_power_cascade_while_an_unset_start_follows_it() -> None:
    """PR #529 review: Min Power 2 W, Start unset (runs at 3 W), a corrective Stop of
    1.51 W: the Min Power cascade to 1.2 W would move the derived Start to 2.2 W."""
    from custom_components.ha_washdata.const import CONF_MIN_POWER  # noqa: PLC0415
    from custom_components.ha_washdata.suggestion_engine import reconcile_suggestions  # noqa: PLC0415

    current = {CONF_MIN_POWER: 2.0, CONF_START_THRESHOLD_W: 3.0, CONF_STOP_THRESHOLD_W: 1.2}
    stop = {CONF_STOP_THRESHOLD_W: {"value": 1.51, "corrective": True}}
    out, _ = reconcile_suggestions(stop, current, explicit_thresholds=frozenset({CONF_STOP_THRESHOLD_W}))
    assert CONF_MIN_POWER not in out and out[CONF_STOP_THRESHOLD_W]["value"] == 1.51
    # Both set: the cascade still keeps Min Power under the Stop.
    out, _ = reconcile_suggestions(stop, current)
    assert out[CONF_MIN_POWER]["value"] == round(1.51 * 0.8, 1)


async def test_apply_judges_again_when_a_cycle_lands_during_its_check(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: a cycle stored while Apply's end-gate scan ran may or may not
    have been read; the verdict is taken again on the evidence as it then stands."""
    from unittest.mock import patch  # noqa: PLC0415

    entry = await setup_washdata_entry(options={
        CONF_STOP_THRESHOLD_W: 3.0, CONF_START_THRESHOLD_W: 5.0, CONF_OFF_DELAY: 60,
        CONF_MIN_OFF_GAP: 60, CONF_END_ENERGY_THRESHOLD: 0.01,
    })
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 1.0, reason="r", flags={"owner": OWNER_BATCH})
    engine_cls = type(manager.learning_manager.suggestion_engine)
    calls: list[int] = []

    def drops(self: Any, stored: Any, *, strict: bool = False) -> list[str]:
        calls.append(len(store.get_past_cycles()))
        if len(calls) == 1:
            store._data["past_cycles"].append(_paused_cycle(99, pause_w=1.0))  # noqa: SLF001
            return []  # judged without it
        return [CONF_END_ENERGY_THRESHOLD]  # with it, the raise splits that cycle

    client = await hass_ws_client(hass)
    with patch.object(engine_cls, "joint_end_gate_drops", drops):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        res = await client.receive_json()
    assert len(calls) == 2
    assert res["success"] and res["result"]["applied"] == []
    await hass.async_block_till_done()
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_END_ENERGY_THRESHOLD] == 0.01


async def test_apply_writes_nothing_after_a_wipe_it_waited_through(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """PR #529 review: Wipe History cleared the suggestions while Apply awaited its
    check, but left the store generation alone, so Apply wrote the removed value."""
    from unittest.mock import patch  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.02, reason="r", flags={"owner": OWNER_BATCH})
    real = ws_api._drop_unsafe_end_gate_updates  # noqa: SLF001

    async def wiped_meanwhile(*a: Any) -> bool:
        ok = await real(*a)
        await store.clear_all_data()
        return ok

    client = await hass_ws_client(hass)
    with patch.object(ws_api, "_drop_unsafe_end_gate_updates", side_effect=wiped_meanwhile):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        res = await client.receive_json()
    assert res["success"] is False and res["error"]["code"] == "entry_reloaded"
    await hass.async_block_till_done()
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_END_ENERGY_THRESHOLD] == 0.01


def test_standby_at_the_stop_threshold_is_not_below_it() -> None:
    """PR #529 review: two hand stops settled at 3 W against a 3 W Stop counted as
    below it, but the detector's quiet needs readings strictly under it."""
    from custom_components.ha_washdata.suggestion_engine import detect_standby_above_stop  # noqa: PLC0415

    pts = [[float(t), 2000.0] for t in range(0, 600, 10)] + [[600.0 + t, 3.0] for t in range(0, 1800, 10)]
    cycles = [{"id": f"s{i}", "status": "completed", "termination_reason": "user",
               "power_data": [list(p) for p in pts], "signature": {"max_power": 2000.0}} for i in range(2)]
    assert detect_standby_above_stop(cycles, 3.0) is not None
    assert detect_standby_above_stop(cycles, 3.5) is None


def test_the_tail_cache_is_keyed_by_the_trace_itself() -> None:
    """PR #529 review: a restore replaced a 3 W tail with a 1 W one under the same
    id, length, last offset and peak; the cache kept answering 3 W."""
    from custom_components.ha_washdata import suggestion_engine as se  # noqa: PLC0415

    def cyc(level: float) -> dict[str, Any]:
        pts = [[float(t), 2000.0] for t in range(0, 600, 10)] + [[600.0 + t, level] for t in range(0, 1800, 10)]
        return {"id": "same", "status": "completed", "termination_reason": "user",
                "power_data": pts, "signature": {"max_power": 2000.0}}

    first, restored = cyc(3.0), cyc(1.0)
    assert se._tail_settled_w(first, first["power_data"], 2000.0) == 3.0  # noqa: SLF001
    assert se._tail_settled_w(restored, restored["power_data"], 2000.0) == 1.0  # noqa: SLF001


def test_a_resolved_start_is_kept_when_the_stored_option_is_null(mock_hass: Any) -> None:
    """PR #529 review: Start stored as null runs at what it resolves to; the raw null
    overwrote that, so a Stop correction above it passed reconciliation alone."""
    from custom_components.ha_washdata.const import CONF_MIN_POWER  # noqa: PLC0415

    store = _Store()
    lm = _learning(mock_hass, store, {CONF_MIN_POWER: 2.0, CONF_START_THRESHOLD_W: None, CONF_STOP_THRESHOLD_W: None})
    store.set_suggestion(CONF_STOP_THRESHOLD_W, 4.0, reason="r", flags={"owner": OWNER_STANDBY_FLOOR})
    lm.suggestion_engine._reconcile_stored_suggestions()  # noqa: SLF001
    sug = store.get_suggestions()
    stop = sug.get(CONF_STOP_THRESHOLD_W, {}).get("value")
    start = sug.get(CONF_START_THRESHOLD_W, {}).get("value")
    # Either the pair is ordered in the stored set, or the stop is withdrawn.
    assert stop is None or (start is not None and start > stop)


def test_the_tail_cache_keeps_no_trace_alive() -> None:
    """Review of PR #529: the cache held each trace list so an id could not be
    reused; every reload, import or edit left a replaced trace in it (up to 512,
    hundreds of MB on long dishwasher traces). It keeps a fingerprint instead."""
    import gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    from custom_components.ha_washdata import suggestion_engine as se  # noqa: PLC0415

    class _Trace(list):
        pass

    trace = _Trace([[float(t), 2000.0] for t in range(0, 600, 10)] + [[600.0 + t, 3.0] for t in range(0, 1800, 10)])
    ref = weakref.ref(trace)
    assert se._tail_settled_w({"id": "alive"}, trace, 2000.0) == 3.0  # noqa: SLF001
    del trace
    gc.collect()
    assert ref() is None


async def test_a_wipe_lets_the_labelled_passes_run_on_the_new_history(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Review of PR #529: after a wipe the model, detection and batch passes waited
    for the new labelled count to outgrow the count of the history just wiped."""
    entry = await setup_washdata_entry()
    lm = hass.data[DOMAIN][entry.entry_id].learning_manager
    lm._last_suggestions_labeled_count = 120  # noqa: SLF001 - a mature install
    lm._last_batch_simulation_count = 115  # noqa: SLF001
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({"type": f"{DOMAIN}/wipe_history", "entry_id": entry.entry_id})
    assert (await client.receive_json())["success"]
    assert lm._last_suggestions_labeled_count == 0  # noqa: SLF001
    assert lm._last_batch_simulation_count == 0  # noqa: SLF001


async def test_the_suggestions_sensor_filters_once_per_state_write(
    hass: Any, setup_washdata_entry: Any
) -> None:
    """Review of PR #529: the count and the attributes each ran the shared filter."""
    from unittest.mock import patch  # noqa: PLC0415

    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    manager.profile_store.set_suggestion(CONF_START_THRESHOLD_W, 9.0, reason="r", flags={"owner": OWNER_BATCH})
    entity_id = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_suggestions")
    assert entity_id
    real = ws_api._visible_suggestions  # noqa: SLF001
    with patch.object(ws_api, "_visible_suggestions", side_effect=real) as filt:
        manager.notify_update()
        await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert filt.call_count == 1
    assert state.state == "1" and state.attributes["suggestions_count"] == 1


async def test_min_power_is_not_written_once_its_stop_is_withdrawn(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Review of PR #529: Stop unset, a pending Stop and its Min Power cascade; the
    joint check withdrew Stop, and Min Power was written alone, moving the Stop
    that follows it."""
    from unittest.mock import patch  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415
    from custom_components.ha_washdata.const import CONF_MIN_POWER  # noqa: PLC0415

    entry = await setup_washdata_entry(options={CONF_MIN_POWER: 5.0, CONF_START_THRESHOLD_W: 10.0})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_STOP_THRESHOLD_W, 4.0, reason="r", flags={"owner": OWNER_BATCH})
    store.set_suggestion(CONF_MIN_POWER, 3.2, reason="r", flags={"cascade": True})

    async def withdraw_stop(_h: Any, _m: Any, _v: Any, updates: dict[str, Any]) -> bool:
        updates.pop(CONF_STOP_THRESHOLD_W, None)
        return True

    client = await hass_ws_client(hass)
    with patch.object(ws_api, "_drop_unsafe_end_gate_updates", side_effect=withdraw_stop):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_STOP_THRESHOLD_W, CONF_MIN_POWER],
        })
        res = await client.receive_json()
    assert res["success"] and res["result"]["applied"] == []
    await hass.async_block_till_done()
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_MIN_POWER] == 5.0


def test_the_pause_scan_also_reads_energy_as_the_detector_does() -> None:
    """Review of PR #529: a 1000 s pause at 2 W with a reading a minute and a 200 s
    Off Delay. Held from the window's start it is 0.111 Wh, over a 0.105 Wh limit;
    the detector integrates only the readings inside the window (180 s, 0.100 Wh)
    and ends the cycle, so a 210 s Min Off Gap splits it."""
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        end_gate_pause_needs,
        end_gate_would_split,
    )

    work = [(float(t), 300.0) for t in range(0, 600, 60)]
    pause = [(600.0 + t, 2.0) for t in range(0, 1000, 60)]
    pts = work + pause + [(1600.0, 300.0), (1660.0, 300.0)]
    # 960 s from the first quiet reading, plus the 60 s the quiet timer credits.
    assert end_gate_pause_needs(pts, 5.0, 200, 0.105) == [1020.0]
    assert end_gate_would_split(pts, 5.0, 200, 210, 0.105)
    assert not end_gate_would_split(pts, 5.0, 200, 1200, 0.105)


def test_a_pause_is_measured_from_where_the_quiet_timer_starts() -> None:
    """Review of PR #529: the live quiet timer credits the interval since the last
    high reading (capped at the gate cadence), so with a 30 s cadence a pause from
    1000 s to a resume at 1200 s is 210 s long at its last quiet reading: a 210 s
    Min Off Gap splits it, 240 s bridges it."""
    from custom_components.ha_washdata.suggestion_engine import end_gate_would_split  # noqa: PLC0415

    pts = [(float(t), 300.0) for t in range(10, 971, 30)]
    pts += [(float(t), 0.0) for t in range(1000, 1181, 30)]
    pts += [(1200.0, 300.0), (1230.0, 300.0)]
    assert end_gate_would_split(pts, 5.0, 180, 210, 0.05)
    assert not end_gate_would_split(pts, 5.0, 180, 240, 0.05)


async def test_a_mute_schedules_no_check_on_a_manager_retired_by_its_write(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Review of PR #529: an unload during the mute's save retired the manager; the
    joint check spawned on it would have saved its retired store."""
    from unittest.mock import patch  # noqa: PLC0415

    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    real = store.set_suggestion_locked
    scheduled: list[Any] = []

    async def retire_during(*a: Any, **k: Any) -> Any:
        out = await real(*a, **k)
        manager._is_shutdown = True  # noqa: SLF001 - the unload
        hass.data[DOMAIN].pop(entry.entry_id, None)
        return out

    client = await hass_ws_client(hass)
    with patch.object(store, "set_suggestion_locked", side_effect=retire_during), \
            patch.object(type(manager.learning_manager), "schedule_joint_end_gate_check",
                         lambda self: scheduled.append(self)):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/set_suggestion_lock", "entry_id": entry.entry_id,
            "key": CONF_START_THRESHOLD_W, "locked": True,
        })
        await client.receive_json()
    assert scheduled == []
    manager._is_shutdown = False  # noqa: SLF001
    hass.data[DOMAIN][entry.entry_id] = manager


def test_a_window_holding_one_reading_can_close_the_gate() -> None:
    """Review of PR #529: a sparse pause whose window holds a single reading is
    zero energy to the live gate, which then ends the cycle; the scan skipped it."""
    from custom_components.ha_washdata.suggestion_engine import _detector_gate_need  # noqa: SLF001, PLC0415

    # One reading 300 s into a pause, a 200 s window: only it is inside.
    assert _detector_gate_need([300.0], [40.0], 0.0, 200.0, 0.01) == 300.0


async def test_a_set_a_mute_changed_is_judged_again_before_strict_returns(mock_hass: Any) -> None:
    """PR #529 review: a mute during the manual analysis's joint check changed the
    set without moving either generation; the stale verdict was discarded and the
    check returned, so the analysis reported success for a set nothing judged. It is
    judged again, and a set that keeps changing is reported as unchecked."""
    from types import SimpleNamespace  # noqa: PLC0415
    from unittest.mock import AsyncMock  # noqa: PLC0415

    import pytest  # noqa: PLC0415

    from custom_components.ha_washdata.learning import EndGateCheckError  # noqa: PLC0415

    mock_hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a, **k: fn(*a, **k))
    store = _Store()
    lm = _learning(mock_hass, store, {})
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.4, reason="r", flags={"owner": OWNER_BATCH})
    store.set_suggestion(CONF_OFF_DELAY, 1800, reason="r", flags={"owner": OWNER_OPERATIONAL})
    judged: list[set[str]] = []

    def _drops(stored: dict[str, Any], strict: bool = False) -> list[str]:
        judged.append(set(stored))
        if len(judged) == 1:
            store.delete_suggestion(CONF_OFF_DELAY)  # muted meanwhile
            return []
        return [CONF_END_ENERGY_THRESHOLD]  # the raise alone splits a pause

    lm.suggestion_engine = SimpleNamespace(
        for_job=lambda: SimpleNamespace(joint_end_gate_drops=_drops),
        apply_suggestions=lambda *a, **k: None,
    )
    await lm._enforce_joint_end_gate(strict=True)  # noqa: SLF001
    assert judged == [{CONF_END_ENERGY_THRESHOLD, CONF_OFF_DELAY}, {CONF_END_ENERGY_THRESHOLD}]
    assert CONF_END_ENERGY_THRESHOLD not in store.get_suggestions()

    # A set that changes during every check is no checked set.
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.4, reason="r", flags={"owner": OWNER_BATCH})

    def _churn(stored: dict[str, Any], strict: bool = False) -> list[str]:
        value = store.get_suggestions()[CONF_END_ENERGY_THRESHOLD]["value"]
        store.set_suggestion(CONF_END_ENERGY_THRESHOLD, value + 0.1, reason="r")
        return []

    lm.suggestion_engine.for_job = lambda: SimpleNamespace(joint_end_gate_drops=_churn)
    with pytest.raises(EndGateCheckError):
        await lm._enforce_joint_end_gate(strict=True)  # noqa: SLF001
    await lm._enforce_joint_end_gate()  # noqa: SLF001 - background: never raises


def test_the_quiet_credit_uses_the_cadence_at_the_drop() -> None:
    """Review of PR #529: reports every 90 s before a pause and every second inside
    it. The whole trace's median capped the credit at 1-2 s; the detector's rolling
    cadence at the drop is 90 s, so its quiet timer reaches the gap 90 s sooner: a
    500 s gap splits the run, 700 s bridges it."""
    from custom_components.ha_washdata.suggestion_engine import end_gate_would_split  # noqa: PLC0415

    pts = [(float(-90 * k), 300.0) for k in range(12, 0, -1)]
    pts += [(float(t), 0.0) for t in range(0, 451)]
    pts += [(480.0, 300.0), (481.0, 300.0)]
    assert end_gate_would_split(pts, 5.0, 180, 500, 0.05)
    assert not end_gate_would_split(pts, 5.0, 180, 700, 0.05)


def test_the_standby_floor_stays_in_a_raw_kw_sensor_s_units() -> None:
    """Review of PR #529: a kW sensor read raw, running at 0.08 and idling at 0.0034
    with Stop 0.002. The idle level rounded to 0 and the floor added 0.2 in raw units:
    Stop 0.2 and Start 0.25, above the running level, so no run could start."""
    from custom_components.ha_washdata.suggestion_engine import standby_stop_floor  # noqa: PLC0415

    def stopped(i: int) -> dict[str, Any]:
        pts = [[float(t), 0.08] for t in range(0, 1800, 10)] + [[1800.0 + t, 0.0034] for t in range(0, 1200, 10)]
        return {"id": f"k{i}", "status": "completed", "termination_reason": "user",
                "power_data": pts, "signature": {"max_power": 0.08}}

    cycles = [stopped(i) for i in range(6)]
    raw = standby_stop_floor(cycles, 0.002, 180.0, to_w=1000.0)
    assert raw is not None and raw["floor_w"] < 0.08 * 0.1
    assert abs(raw["floor_w"] - 0.0034 * 1.25) < 1e-4
    # The same appliance in watts keeps the watt margin.
    watts = standby_stop_floor(
        [{**c, "power_data": [[t, p * 1000] for t, p in c["power_data"]],
          "signature": {"max_power": 80.0}} for c in cycles], 2.0, 180.0,
    )
    assert watts is not None and watts["floor_w"] == round(max(3.4 * 1.25, 3.4 + 0.2), 2)


def test_a_set_that_moves_a_split_to_another_cycle_is_withdrawn(mock_hass: Any) -> None:
    """Review of PR #529: Stop 3 W / Min Off Gap 210 s splits a 300 s pause at 2 W;
    Stop 5 W / 600 s bridges it but splits a 900 s pause at 4 W. One split either
    way, so the count saw no change and Apply all would have written the set."""
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        CONF_END_ENERGY_THRESHOLD,
        CONF_MIN_OFF_GAP,
        CONF_OFF_DELAY,
        CONF_STOP_THRESHOLD_W,
    )
    from custom_components.ha_washdata.suggestion_engine import SuggestionEngine  # noqa: PLC0415

    def trace(pause_s: int, pause_w: float) -> list[tuple[float, float]]:
        work = [(float(t), 300.0) for t in range(0, 600, 10)]
        pause = [(600.0 + t, pause_w) for t in range(0, pause_s, 10)]
        return work + pause + [(600.0 + pause_s + t, 300.0) for t in range(0, 300, 10)]

    eng = SuggestionEngine(mock_hass, "e", _Store(), device_type="washing_machine")
    eng._end_gate_evidence = lambda: [trace(300, 2.0), trace(900, 4.0)]  # noqa: SLF001
    eng._effective_thresholds = lambda _o: {  # noqa: SLF001
        CONF_STOP_THRESHOLD_W: 3.0, CONF_OFF_DELAY: 180, CONF_MIN_OFF_GAP: 210,
        CONF_END_ENERGY_THRESHOLD: 1.0,
    }
    eng._entry_options = lambda: {}  # noqa: SLF001
    stored = {CONF_STOP_THRESHOLD_W: {"value": 5.0}, CONF_MIN_OFF_GAP: {"value": 600}}
    assert eng.joint_end_gate_drops(stored)  # something is withdrawn


async def test_the_manual_analysis_judges_again_when_a_cycle_lands_during_its_check(
    hass: Any, setup_washdata_entry: Any
) -> None:
    """PR #529 review: a completed cycle stored while the joint check scanned the
    traces changes neither the pending values nor the data generation. The
    verdict taken without it must not stand: the check runs again on the evidence
    as it then stands, and withdraws what splits the new cycle."""
    from unittest.mock import patch  # noqa: PLC0415

    entry = await setup_washdata_entry(options={
        CONF_STOP_THRESHOLD_W: 3.0, CONF_START_THRESHOLD_W: 5.0, CONF_OFF_DELAY: 60,
        CONF_MIN_OFF_GAP: 60, CONF_END_ENERGY_THRESHOLD: 0.01,
    })
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    lm = manager.learning_manager
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 1.0, reason="r", flags={"owner": OWNER_BATCH})
    engine_cls = type(lm.suggestion_engine)
    calls: list[int] = []

    def drops(self: Any, stored: Any, *, strict: bool = False) -> list[str]:
        calls.append(len(store.get_past_cycles()))
        if len(calls) == 1:
            store._data["past_cycles"].append(_paused_cycle(99, pause_w=1.0))  # noqa: SLF001
            return []  # judged without it
        return [CONF_END_ENERGY_THRESHOLD]  # with it, the raise splits that cycle

    with patch.object(engine_cls, "joint_end_gate_drops", drops):
        await lm._enforce_joint_end_gate(strict=True)  # noqa: SLF001
    assert calls == [0, 1]
    assert CONF_END_ENERGY_THRESHOLD not in (store.get_suggestions() or {})


def test_a_sparse_change_only_trace_is_end_gate_evidence(mock_hass: Any) -> None:
    """Review of PR #529 (remeasured): a plug that reports only on change records a
    whole cycle in a handful of readings, its 1800 s pause in one. Under the old
    10-reading minimum the guard never saw it and let a 600 s Min Off Gap split it."""
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        CONF_END_ENERGY_THRESHOLD,
        CONF_MIN_OFF_GAP,
        CONF_OFF_DELAY,
        CONF_STOP_THRESHOLD_W,
    )
    from custom_components.ha_washdata.suggestion_engine import SuggestionEngine  # noqa: PLC0415

    sparse = {
        "id": "s1", "status": "completed", "termination_reason": "timeout",
        "power_data": [[0.0, 2000.0], [600.0, 0.5], [2400.0, 2000.0], [3000.0, 0.4]],
    }
    store = _Store(cycles=[sparse])
    store.get_past_cycles = lambda: store.cycles
    eng = SuggestionEngine(mock_hass, "e", store, device_type="washing_machine")
    assert len(eng._end_gate_evidence()) == 1  # noqa: SLF001
    eng._effective_thresholds = lambda _o: {  # noqa: SLF001
        CONF_STOP_THRESHOLD_W: 2.0, CONF_OFF_DELAY: 180, CONF_MIN_OFF_GAP: 3000,
        CONF_END_ENERGY_THRESHOLD: 0.05,
    }
    eng._entry_options = lambda: {}  # noqa: SLF001
    assert eng.joint_end_gate_drops({CONF_MIN_OFF_GAP: {"value": 600}}) == [CONF_MIN_OFF_GAP]


def test_a_shorter_gap_that_lets_the_flat_standby_bypass_close_a_pause_splits() -> None:
    """A 780 s pause at 3.8 W under a 4 W stop is too much energy for the gate,
    but past max(off_delay, min_off_gap) the detector's flat-standby bypass
    (DETECT-08) closes it whatever its energy: a 360 s gap split this cycle on
    the shipped detector, 900 s kept it whole (review of PR #529)."""
    points = [(float(t), 300.0) for t in range(0, 1200, 10)]
    points += [(float(t), 3.8) for t in range(1200, 1980, 10)]
    points += [(float(t), 300.0) for t in range(1980, 3180, 10)]
    points += [(float(t), 0.5) for t in range(3180, 4000, 10)]
    args = (points, 4.0, 180.0)
    assert end_gate_would_split(*args, 360.0, 0.05, device_type="washing_machine")
    assert not end_gate_would_split(*args, 900.0, 0.05, device_type="washing_machine")


def test_a_later_burst_of_fast_reports_does_not_hide_the_flat_standby_bypass() -> None:
    """The detector's outage ceiling is causal: 90 s reports through the pause
    give 900 s, so a later 1 s burst must not drop it to 60 s and turn the pause's
    own 90 s gaps into outages (review of PR #529)."""
    points = [(float(t), 300.0) for t in range(0, 1890, 90)]
    points += [(float(t), 3.8) for t in range(1890, 2700, 90)]
    points += [(float(t), 300.0) for t in range(2700, 22700)]
    points += [(22700.0 + t, 0.5) for t in range(0, 900, 10)]
    assert end_gate_would_split(points, 4.0, 180.0, 360.0, 0.05, device_type="washing_machine")


async def test_apply_writes_nothing_after_a_native_reconfigure_it_waited_through(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Review of PR #529: a native options save takes no WS lock and reloads in
    place, so Apply merged values judged against the old settings into the new."""
    from unittest.mock import patch  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.02, reason="r", flags={"owner": OWNER_BATCH})
    real = ws_api._drop_unsafe_end_gate_updates  # noqa: SLF001

    async def reconfigured_meanwhile(*a: Any) -> bool:
        ok = await real(*a)
        hass.config_entries.async_update_entry(entry, options={**entry.options, CONF_OFF_DELAY: 90})
        await hass.async_block_till_done()
        return ok

    client = await hass_ws_client(hass)
    with patch.object(ws_api, "_drop_unsafe_end_gate_updates", side_effect=reconfigured_meanwhile):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        res = await client.receive_json()
    assert hass.data[DOMAIN][entry.entry_id] is manager  # reloaded in place
    assert res["success"] is False and res["error"]["code"] == "entry_reloaded"
    await hass.async_block_till_done()
    opts = hass.config_entries.async_get_entry(entry.entry_id).options
    assert opts[CONF_END_ENERGY_THRESHOLD] == 0.01 and opts[CONF_OFF_DELAY] == 90


def test_a_non_finite_reading_does_not_stall_the_flat_standby_scan() -> None:
    """Review of PR #529: NaN compares false against the stop both ways, so the
    flat-standby run scan never moved past it and Analyse now hung."""
    nan = float("nan")
    assert not end_gate_would_split([(0.0, 80.0), (10.0, nan), (400.0, 80.0)], 4.0, 180.0, 360.0, 0.05)
    assert not end_gate_would_split([(0.0, 80.0), (10.0, 3.0), (400.0, 80.0)], nan, 180.0, 360.0, 0.05)


def test_the_running_integral_is_integrate_wh_per_window() -> None:
    """Review of PR #529: the end-gate scan reads window energies from the shared
    running sum, which must equal integrate_wh on every window it reads."""
    import numpy as np  # noqa: PLC0415

    from custom_components.ha_washdata.signal_processing import (  # noqa: PLC0415
        cumulative_wh,
        integrate_wh,
    )

    rng = np.random.default_rng(7)
    ts = np.cumsum(rng.choice([0.0, 2.0, 10.0, 45.0], size=300))
    ps = rng.uniform(0.0, 2000.0, size=300)
    cum = cumulative_wh(ts, ps)
    for lo, k in [(0, 299), (5, 40), (100, 101), (250, 250), (17, 288)]:
        seg = slice(lo, k + 1)
        want = integrate_wh(ts[seg], ps[seg], max_gap_s=60.0)
        assert cum[k] - cum[lo] == pytest.approx(want, rel=1e-9, abs=1e-12)
        assert cum[k] - cum[lo] == pytest.approx(integrate_wh(ts[seg], ps[seg]), rel=1e-9, abs=1e-12)


def test_the_quiet_before_an_outage_is_judged() -> None:
    """Review of PR #529: a pause with an outage-sized hole was skipped whole, so a
    wait the 0 W quiet before the hole already reached passed. The hole itself is
    not credited: 400 s observed splits at 210 s, not at 900 s."""
    points = [(float(t), 300.0) for t in range(0, 1200, 10)]
    points += [(float(t), 0.0) for t in range(1200, 1610, 10)]
    points += [(float(t), 300.0) for t in range(5300, 6500, 10)]
    points += [(6500.0 + t, 0.0) for t in range(0, 600, 10)]
    assert end_gate_would_split(points, 4.0, 180.0, 210.0, 0.05)
    assert not end_gate_would_split(points, 4.0, 180.0, 900.0, 0.05)


def _pulsed_dryer_pause() -> list[tuple[float, float]]:
    """300 W, an 800 s 3.8 W phase with a single 10 W reading every 90 s, 300 W."""
    points = [(float(t), 300.0) for t in range(0, 1200, 10)]
    points += [
        (float(t), 10.0 if t > 1200 and (t - 1200) % 90 == 0 else 3.8)
        for t in range(1200, 2000, 10)
    ]
    points += [(float(t), 300.0) for t in range(2000, 3200, 10)]
    return points + [(3200.0 + t, 0.5) for t in range(0, 600, 10)]


def test_a_dryer_s_ignored_pulses_do_not_hide_a_split() -> None:
    """Review of PR #529: the detector keeps a dryer's quiet across short pulses
    (#474) and leaves them out of its windows, so the flat-standby bypass closes
    this pause at a 360 s Min Off Gap; 900 s stays whole (about 710 s of quiet).
    A washer resets on every pulse, as live."""
    points = _pulsed_dryer_pause()
    args = (points, 4.0, 180.0)
    assert end_gate_would_split(*args, 360.0, 0.05, device_type="dryer", pulse_max_w=400.0)
    assert not end_gate_would_split(*args, 900.0, 0.05, device_type="dryer", pulse_max_w=400.0)
    assert not end_gate_would_split(*args, 360.0, 0.05, device_type="washing_machine", pulse_max_w=400.0)
    # Hotter than the anti-crease ceiling: not a pulse, the pause resets.
    assert not end_gate_would_split(*args, 360.0, 0.05, device_type="dryer", pulse_max_w=5.0)


def test_the_held_running_sum_is_the_stored_cycle_integral() -> None:
    """Review of PR #529: the held model's running sums are shared code and equal
    integrate_cycle_wh wherever no interval is an outage."""
    import numpy as np  # noqa: PLC0415

    from custom_components.ha_washdata.signal_processing import (  # noqa: PLC0415
        cumulative_held_wh,
        held_wh_until,
        integrate_cycle_wh,
    )

    ts = [0.0, 5.0, 65.0, 66.0, 400.0, 401.5]
    ps = [100.0, 0.0, 2000.0, 3.0, 50.0, 0.0]
    cum = cumulative_held_wh(ts, ps)
    assert cum[-1] == pytest.approx(integrate_cycle_wh(np.asarray(ts), np.asarray(ps)))
    assert held_wh_until(ts, ps, cum, 35.0) == pytest.approx(100.0 * 5 / 3600)
    assert held_wh_until(ts, ps, cum, 70.0) == pytest.approx((500 + 2000 * 1 + 3 * 4) / 3600)
