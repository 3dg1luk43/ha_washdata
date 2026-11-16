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

from typing import Any
from unittest.mock import MagicMock

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
    """Drying at 1.2 W: end_energy 0.75 Wh lets an 1800 s window (0.6 Wh) pass."""
    store = _Store([_drying_cycle(i, drying_w=1.2) for i in range(6)])
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
    # A pending 180 s off_delay from another pass (harmless alone: 0.06 Wh over
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

    def held_first(self: Any, stored: dict[str, Any]) -> list[str]:
        calls.append(sorted(stored))
        if len(calls) == 1:  # the pass's check
            entered.set()
            release.wait(10)
        return real(self, stored)

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
    from 1000 s into the pause: 180 s x 5 W passes 0.05 Wh at 1036 s, not 1000."""
    from custom_components.ha_washdata.suggestion_engine import end_gate_pause_needs  # noqa: PLC0415

    pts = [(0.0, 100.0), (100.0, 0.0), (1100.0, 5.0), (2100.0, 50.0)]
    assert end_gate_pause_needs(pts, 6.0, 180, 0.05) == [1036.0]
    assert end_gate_would_split(pts, 6.0, 180, 1020, 0.05)
    assert not end_gate_would_split(pts, 6.0, 180, 1040, 0.05)


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
