"""#530: tell an automation when the hand-picked program is not the one running.

While a program is picked by hand the live match does not run. A background match
on the profile_match_interval cadence ranks the trace against the pick
(``match_rules.program_mismatch_tick``) and, once another program beats it clearly
on several matches in a row, sends the ``program_mismatch`` trigger once and sets
the State sensor's ``program_mismatch`` attribute. It changes nothing else: the
displayed program, the expected duration, the detector's match and the label stay
the pick's.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import (
    DOMAIN,
    PROGRAM_MISMATCH_MIN_FRACTION,
    PROGRAM_MISMATCH_MIN_MARGIN,
    PROGRAM_MISMATCH_PERSISTENCE,
    SIGNAL_DEVICE_TRIGGER,
    TRIGGER_PROGRAM_MISMATCH,
)
from custom_components.ha_washdata.profile_store import MatchResult

POWER = "sensor.washer_power"
FAST = {
    "min_power": 2.0,
    "off_delay": 60,
    "min_off_gap": 60,
    "watchdog_interval": 30,
    "completion_min_seconds": 300,
    "profile_match_interval": 60,
}
DURATION = 1800.0


def _result(best: str, scores: dict[str, float]) -> MatchResult:
    cands = [{"name": n, "score": s} for n, s in sorted(scores.items(), key=lambda kv: -kv[1])]
    margin = cands[0]["score"] - cands[1]["score"] if len(cands) > 1 else 1.0
    return MatchResult(best, cands[0]["score"], DURATION, None, cands, margin < 0.05, margin,
                       ranking=cands)


# ── The rule ─────────────────────────────────────────────────────────────────


def test_the_deficit_reads_the_pick_against_the_winner() -> None:
    res = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.6, "Eco": 0.5})
    assert match_rules.selected_program_deficit(res, "Cotton") == ("Synthetics", 0.9 - 0.6)
    assert match_rules.selected_program_deficit(res, "Synthetics") is None
    # The user's own profile group: near-duplicates are not a mismatch.
    assert match_rules.selected_program_deficit(
        res, "Cotton", {"Cotton": "Wash", "Synthetics": "Wash"}
    ) is None
    # Not scored at all (fewer than five listed): no evidence.
    assert match_rules.selected_program_deficit(res, "Wool") is None
    # Off a full list of five: bounded by the fifth score.
    full = _result("A", {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5})
    assert match_rules.selected_program_deficit(full, "Wool") == ("A", 0.9 - 0.5)
    assert match_rules.selected_program_deficit(None, "Cotton") is None


def test_a_stage5_family_score_stands_for_its_member() -> None:
    res = SimpleNamespace(best_profile="Synthetics", candidates=[
        {"name": "Synthetics", "score": 0.9},
        {"name": "__group__Cottons", "score": 0.7, "group_best_member": "Cotton 60"},
    ])
    fam = {"Cotton 40": "Cottons", "Cotton 60": "Cottons"}
    assert match_rules.selected_program_deficit(res, "Cotton 40", fam) == ("Synthetics", 0.9 - 0.7)
    won = SimpleNamespace(best_profile="Cotton 60", candidates=[
        {"name": "Cotton 60", "score": 0.9, "stage5_group": "__group__Cottons"},
        {"name": "Synthetics", "score": 0.5},
    ])
    assert match_rules.selected_program_deficit(won, "Cotton 40", fam) is None


def test_only_the_stage5_provenance_makes_a_family() -> None:
    """A profile literally named ``__group__Cottons`` is an ordinary programme: only
    the producer's ``group_best_member`` marks a family's own entry."""
    fam = {"Cotton 40": "Cottons", "Cotton 60": "Cottons"}
    res = SimpleNamespace(best_profile="Synthetics", candidates=[
        {"name": "Synthetics", "score": 0.9},
        {"name": "__group__Cottons", "score": 0.85},
        {"name": "Cotton 40", "score": 0.5},
    ])
    assert match_rules.selected_program_deficit(res, "Cotton 40", fam) == ("Synthetics", 0.9 - 0.5)


def test_a_pick_named_like_a_family_is_not_read_off_the_family_row() -> None:
    """PR #529 review: the pick is an ordinary programme named ``__group__Quick``
    and a group ``Quick`` exists. Its family row carries the same name, ranks
    higher, and must not stand for the pick."""
    fam = {"Quick 40": "Quick", "Quick 60": "Quick"}
    res = SimpleNamespace(best_profile="Synthetics", candidates=[
        {"name": "Synthetics", "score": 0.9},
        {"name": "__group__Quick", "score": 0.85, "group_best_member": "Quick 40"},
        {"name": "__group__Quick", "score": 0.3},
    ])
    hit = match_rules.selected_program_deficit(res, "__group__Quick", fam)
    assert hit is not None and hit[0] == "Synthetics"
    assert abs(hit[1] - 0.6) < 1e-9


def test_the_tick_needs_persistence_and_the_minimum_fraction() -> None:
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.9 - PROGRAM_MISMATCH_MIN_MARGIN})
    close = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.85})
    state = match_rules.ProgramMismatchState()
    early = DURATION * PROGRAM_MISMATCH_MIN_FRACTION / 2
    late = DURATION * PROGRAM_MISMATCH_MIN_FRACTION

    def tick(res: Any, elapsed: float) -> bool:
        return match_rules.program_mismatch_tick(
            state, res, "Cotton", elapsed_s=elapsed, expected_s=DURATION
        )

    for _ in range(PROGRAM_MISMATCH_PERSISTENCE - 1):
        assert not tick(wrong, late)
    assert not tick(close, late)  # a match that does not count restarts the run
    assert state.streak == 0
    for _ in range(PROGRAM_MISMATCH_PERSISTENCE + 2):
        assert not tick(wrong, early)  # counted, but too early to establish
    assert tick(wrong, late)
    assert state.fired and state.looks_like == "Synthetics"
    assert state.margin == round(PROGRAM_MISMATCH_MIN_MARGIN, 3)
    assert not tick(wrong, late)  # once
    # No usable expected duration: never.
    other = match_rules.ProgramMismatchState()
    for _ in range(PROGRAM_MISMATCH_PERSISTENCE + 1):
        assert not match_rules.program_mismatch_tick(
            other, wrong, "Cotton", elapsed_s=late, expected_s=None
        )


def test_the_state_round_trips_a_snapshot() -> None:
    state = match_rules.ProgramMismatchState(streak=2, fired=True, looks_like="Eco", margin=0.2)
    back = match_rules.ProgramMismatchState()
    back.restore(state.as_dict())
    assert back.as_dict() == state.as_dict()
    junk = match_rules.ProgramMismatchState()
    junk.restore({"streak": "x", "fired": "yes", "looks_like": 3, "margin": float("nan")})
    assert junk.as_dict() == match_rules.ProgramMismatchState().as_dict()
    # JSON keeps an oversized integer unbounded: float() of it raises, not inf.
    huge = match_rules.ProgramMismatchState()
    huge.restore({"streak": 2, "fired": True, "looks_like": "Eco", "margin": 10**400})
    assert huge.as_dict() == {"fired": True, "looks_like": "Eco", "margin": None}
    # PR #529 review: a pending streak is neither saved nor taken back (an older
    # snapshot's included), nor the data or groups a live one was counted on.
    assert "streak" not in state.as_dict()
    live = match_rules.ProgramMismatchState(streak=4, data_generation=3, families={"A": "G"})
    live.restore({"streak": 5, "fired": False})
    assert (live.streak, live.data_generation, live.families) == (0, None, None)


# ── The manager, on a real boot ──────────────────────────────────────────────


async def _feed(hass: HomeAssistant, freezer: Any, watts: float, seconds: int,
                step: int = 30) -> None:
    t = 0
    while t < seconds:
        freezer.tick(timedelta(seconds=step))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        async_fire_time_changed(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        t += step


def _trace(shape: list[tuple[float, float]], step: float = 30.0) -> list[list[float]]:
    pts, t = [], 0.0
    for minutes, watts in shape:
        end = t + minutes * 60.0
        while t < end:
            pts.append([t, watts])
            t += step
    pts.append([t, 0.0])
    return pts


COTTON = [(10, 2000.0), (20, 200.0)]
SYNTHETICS = [(10, 200.0), (10, 1000.0), (10, 200.0)]


async def _seed(manager: Any) -> None:
    """Two programmes of three cycles each, with envelopes."""
    store = manager.profile_store
    start = dt_util.utcnow() - timedelta(days=10)
    for name, shape in (("Cotton", COTTON), ("Synthetics", SYNTHETICS)):
        pts = _trace(shape)
        for j in range(3):
            begin = start + timedelta(days=j, hours=1 if name == "Cotton" else 5)
            store._data["past_cycles"].append({  # noqa: SLF001
                "id": f"{name}-{j}", "profile_name": name, "label_source": "manual",
                "status": "completed", "start_time": begin.isoformat(),
                "end_time": (begin + timedelta(seconds=pts[-1][0])).isoformat(),
                "duration": pts[-1][0], "power_data": [list(p) for p in pts],
            })
        store._data["profiles"][name] = {  # noqa: SLF001
            "avg_duration": pts[-1][0], "sample_cycle_id": f"{name}-0",
        }
        await store.async_rebuild_envelope(name)


async def _picked_cycle(hass: HomeAssistant, setup_washdata_entry, freezer, program: str):
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    await _seed(manager)
    sent: list[tuple[str, dict]] = []
    async_dispatcher_connect(
        hass, SIGNAL_DEVICE_TRIGGER.format(entry.entry_id),
        lambda kind, variables: sent.append((kind, variables)),
    )
    await _feed(hass, freezer, 200, 90)
    assert manager.check_state() == "running"
    assert manager.set_manual_program(program)
    return entry, manager, sent


def _mismatches(sent: list[tuple[str, dict]]) -> list[dict]:
    return [v for kind, v in sent if kind == TRIGGER_PROGRAM_MISMATCH]


def _attrs(hass: HomeAssistant) -> dict[str, Any]:
    state = hass.states.get("sensor.washer_state")
    assert state is not None
    return dict(state.attributes)


async def test_a_wrong_pick_fires_once_and_changes_nothing(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """The real matcher, a Synthetics-shaped run with Cotton picked."""
    _entry, manager, sent = await _picked_cycle(hass, setup_washdata_entry, freezer, "Cotton")
    detector_profile = manager.detector.matched_profile
    last_live_match = manager._last_match_result  # noqa: SLF001
    assert _attrs(hass)["program_mismatch"] is False

    await _feed(hass, freezer, 200, 8 * 60)
    await _feed(hass, freezer, 1000, 10 * 60)
    await _feed(hass, freezer, 200, 8 * 60)
    fired = _mismatches(sent)
    assert len(fired) == 1, sent
    assert fired[0]["program"] == "Cotton" and fired[0]["looks_like"] == "Synthetics"
    assert fired[0]["margin"] >= PROGRAM_MISMATCH_MIN_MARGIN
    assert fired[0]["elapsed_seconds"] >= PROGRAM_MISMATCH_MIN_FRACTION * DURATION
    attrs = _attrs(hass)
    assert attrs["program_mismatch"] is True
    assert attrs["program_mismatch_looks_like"] == "Synthetics"
    assert manager.program_mismatch == {
        "selected": "Cotton", "looks_like": "Synthetics", "margin": fired[0]["margin"],
    }
    # Nothing the pick drives moved.
    assert manager.current_program == "Cotton" and manager.manual_program_active
    assert manager._matched_profile_duration == DURATION  # noqa: SLF001
    assert manager._last_match_result is last_live_match  # noqa: SLF001
    assert manager.detector.matched_profile in (detector_profile, "Cotton")
    assert manager.detector.expected_duration_seconds == DURATION

    # Once per cycle and pick.
    await _feed(hass, freezer, 200, 3 * 60)
    assert len(_mismatches(sent)) == 1

    # Cleared at the cycle end, and the cycle keeps the pick as its label.
    await _feed(hass, freezer, 0, 300)
    assert manager.check_state() not in ("running", "paused", "ending")
    assert _attrs(hass)["program_mismatch"] is False
    assert "program_mismatch_looks_like" not in _attrs(hass)
    assert manager.program_mismatch is None
    assert manager.profile_store.get_past_cycles()[-1]["profile_name"] == "Cotton"


async def test_the_right_pick_never_fires(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    _entry, manager, sent = await _picked_cycle(
        hass, setup_washdata_entry, freezer, "Synthetics"
    )
    with patch.object(
        match_rules, "program_mismatch_tick", wraps=match_rules.program_mismatch_tick
    ) as checks:
        await _feed(hass, freezer, 200, 8 * 60)
        await _feed(hass, freezer, 1000, 10 * 60)
        await _feed(hass, freezer, 200, 10 * 60)
    # One background match per profile_match_interval (60 s here) once one has passed.
    assert checks.call_count >= 20
    assert _mismatches(sent) == []
    assert _attrs(hass)["program_mismatch"] is False
    assert manager.program_mismatch is None


async def test_a_new_pick_starts_the_check_afresh(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Stubbed matcher: always Synthetics clearly ahead of Cotton."""
    _entry, manager, sent = await _picked_cycle(hass, setup_washdata_entry, freezer, "Cotton")
    res = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    with patch.object(
        manager.profile_store, "async_match_profile", AsyncMock(return_value=res)
    ) as match:
        await _feed(hass, freezer, 200, 20 * 60)
        assert len(_mismatches(sent)) == 1
        assert match.await_args.kwargs["in_progress"] is True
        # Following the hint: Synthetics is now the pick, and it wins.
        assert manager.set_manual_program("Synthetics")
        await hass.async_block_till_done()
        assert manager.program_mismatch is None
        await _feed(hass, freezer, 200, 10 * 60)
        assert len(_mismatches(sent)) == 1
        # Back to auto-detect: no pick, nothing to report.
        manager.clear_manual_program()
        await hass.async_block_till_done()
        assert manager.program_mismatch is None


async def test_the_snapshot_carries_the_check_across_a_restart(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    _entry, manager, sent = await _picked_cycle(hass, setup_washdata_entry, freezer, "Cotton")
    res = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    with patch.object(
        manager.profile_store, "async_match_profile", AsyncMock(return_value=res)
    ):
        await _feed(hass, freezer, 200, 20 * 60)
    assert len(_mismatches(sent)) == 1
    snap = manager._augment_active_snapshot(manager.detector.get_state_snapshot())  # noqa: SLF001
    assert snap["program_mismatch"]["program"] == "Cotton"
    assert snap["program_mismatch"]["fired"] is True
    # A fresh check for the same pick takes it back (what the restore calls).
    manager._program_mismatch = match_rules.ProgramMismatchState()  # noqa: SLF001
    assert manager.program_mismatch is None
    manager._restore_program_mismatch(snap["program_mismatch"])  # noqa: SLF001
    assert manager.program_mismatch["looks_like"] == "Synthetics"
    # Another pick's snapshot is ignored.
    manager._program_mismatch = match_rules.ProgramMismatchState()  # noqa: SLF001
    manager._restore_program_mismatch({**snap["program_mismatch"], "program": "Eco"})  # noqa: SLF001
    assert manager.program_mismatch is None


async def _past_half(hass: HomeAssistant, setup_washdata_entry, freezer):
    """Cotton picked on a run past half its usual duration, no check counted yet."""
    entry, manager, sent = await _picked_cycle(hass, setup_washdata_entry, freezer, "Cotton")
    quiet = _result("Cotton", {"Cotton": 0.9, "Synthetics": 0.5})
    with patch.object(manager.profile_store, "async_match_profile", AsyncMock(return_value=quiet)):
        await _feed(hass, freezer, 200, 16 * 60)
    state = manager._program_mismatch_state()  # noqa: SLF001
    assert state is not None and state.streak == 0 and not state.fired
    return entry, manager, sent, state


async def _check(hass: HomeAssistant, manager: Any, key: Any) -> None:
    await manager._async_check_program_mismatch(  # noqa: SLF001
        list(manager.detector.get_power_trace()), manager._match_identity(),  # noqa: SLF001
        key,
    )
    await hass.async_block_till_done()


async def test_a_group_edited_during_the_match_discards_it(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """The ranking was built with Cotton in the Wash group (so the Wash family's
    win is no mismatch); read against the groups AFTER the match, Cotton is no
    longer in it and its absence from the list read as a deficit."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    won = SimpleNamespace(best_profile="Synthetics", candidates=[
        {"name": "Synthetics", "score": 0.9, "stage5_group": "__group__Wash"},
        *({"name": n, "score": s} for n, s in (("A", 0.8), ("B", 0.7), ("C", 0.6), ("D", 0.5))),
    ])

    async def match_and_ungroup(*_a: Any, **_k: Any) -> Any:
        store._data["profile_groups"] = {}  # noqa: SLF001  (the user ungroups meanwhile)
        return won

    store._data["profile_groups"] = {"Wash": {"members": ["Cotton", "Synthetics"]}}  # noqa: SLF001
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_and_ungroup):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == PROGRAM_MISMATCH_PERSISTENCE - 1  # discarded, not counted
    # The same ranking with the groups unchanged throughout does establish it.
    with patch.object(store, "async_match_profile", AsyncMock(return_value=won)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


async def test_an_undo_import_during_the_match_discards_it(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Undo Import replaces the store while the match runs: the programme names
    and groups are unchanged, but the ranking scored the replaced profiles, so
    it is discarded and the run of counting matches restarts."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    # An import of the device's own export, same names and groups: a restore point.
    await store.async_import_data(store.export_data())
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_and_undo(*_a: Any, **_k: Any) -> Any:
        await store.async_restore_pre_import_snapshot()
        return wrong

    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_and_undo):
        await _check(hass, manager, state.key)
    assert manager._program_mismatch_state() is state  # noqa: SLF001  (same pick)
    assert _mismatches(sent) == []
    assert state.streak == 0 and not state.fired
    # A check on unchanged data still counts.
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


async def test_a_check_from_an_earlier_pick_of_the_same_program_does_not_count(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Cotton -> Synthetics -> Cotton while Cotton's check is still matching: the
    new Cotton pick is a new selection and starts from nothing."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    key = state.key
    gate = asyncio.Event()
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def slow(*_a: Any, **_k: Any) -> Any:
        await gate.wait()
        return wrong

    with patch.object(manager.profile_store, "async_match_profile", new=slow):
        task = hass.async_create_task(_check(hass, manager, key))
        await asyncio.sleep(0)
        assert manager.set_manual_program("Synthetics")
        assert manager.set_manual_program("Cotton")
        gate.set()
        await task
        await hass.async_block_till_done()
    assert _mismatches(sent) == []
    fresh = manager._program_mismatch_state()  # noqa: SLF001
    assert fresh is not None and fresh.key != key and fresh.streak == 0
    # Picking the program already picked is no new selection: the check goes on.
    fresh.streak = 3
    assert manager.set_manual_program("Cotton")
    assert manager._program_mismatch_state() is fresh and fresh.streak == 3  # noqa: SLF001


async def test_the_check_runs_while_the_detector_match_tick_is_frozen(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """The dishwasher ENDING freeze (and the anti-crease one) stop the detector's
    match tick, the pin's only path to the check: it runs from the readings too."""
    _entry, manager, _sent = await _picked_cycle(
        hass, setup_washdata_entry, freezer, "Synthetics"
    )
    with patch.object(manager.detector, "_try_profile_match", lambda *_a, **_k: None), \
            patch.object(match_rules, "program_mismatch_tick",
                         wraps=match_rules.program_mismatch_tick) as checks:
        # Each reading through the power-event handler: `_feed` fires the timer
        # first, so the watchdog's resync takes the report instead.
        for _ in range(20):
            freezer.tick(timedelta(seconds=30))
            hass.states.async_set(POWER, "200", {"unit_of_measurement": "W"}, force_update=True)
            await hass.async_block_till_done()
            async_fire_time_changed(hass, dt_util.utcnow())
            await hass.async_block_till_done()
    # One per profile_match_interval (60 s) once one has passed.
    assert checks.call_count >= 8


def _slow(gate: asyncio.Event, result: Any):
    async def match(*_a: Any, **_k: Any) -> Any:
        await gate.wait()
        return result
    return match


async def test_no_check_starts_or_lands_after_shutdown(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """An unload sets ``_is_shutdown``: a match still in flight must not land, and a
    reading that arrives after it must not start another."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    gate = asyncio.Event()
    try:
        with patch.object(manager.profile_store, "async_match_profile", new=_slow(gate, wrong)):
            task = hass.async_create_task(_check(hass, manager, state.key))
            await asyncio.sleep(0)
            manager._is_shutdown = True  # noqa: SLF001
            gate.set()
            await task
        assert _mismatches(sent) == []
        assert state.streak == PROGRAM_MISMATCH_PERSISTENCE - 1
        manager._program_mismatch_last_s = None  # noqa: SLF001  (a check is due)
        with patch.object(
            manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)
        ) as match:
            manager._schedule_program_mismatch_check()  # noqa: SLF001
            await hass.async_block_till_done()
        assert match.await_count == 0
    finally:
        manager._is_shutdown = False  # noqa: SLF001
    assert _mismatches(sent) == []


async def test_a_new_duration_for_the_same_pick_is_read_when_the_match_lands(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Cotton picked again after its usual length grew is no new selection, so the
    check in flight goes on, but it was dispatched against the old length: past
    half of 30 min is not past half of 2 h, so it counts without firing."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    gate = asyncio.Event()
    with patch.object(manager.profile_store, "async_match_profile", new=_slow(gate, wrong)):
        task = hass.async_create_task(_check(hass, manager, state.key))
        await asyncio.sleep(0)
        manager.profile_store._data["profiles"]["Cotton"]["avg_duration"] = 4 * DURATION  # noqa: SLF001
        assert manager.set_manual_program("Cotton")
        assert manager._program_mismatch_state() is state  # noqa: SLF001
        assert manager._matched_profile_duration == 4 * DURATION  # noqa: SLF001
        gate.set()
        await task
    assert _mismatches(sent) == []
    assert state.streak == PROGRAM_MISMATCH_PERSISTENCE and not state.fired


async def test_the_readings_between_two_checks_never_build_the_cut_trace(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """After a stall the detector's match trace is a new list built from every
    reading (item 511's cut). A reading before the next check is due is turned away
    on the stored trace's span without building it; the check itself matches the
    cut trace, and the fraction gate reads its (programme-time) span."""
    _entry, manager, _sent, _state = await _past_half(hass, setup_washdata_entry, freezer)
    det = manager.detector
    raw = det._power_readings  # noqa: SLF001
    start = det._current_cycle_start  # noqa: SLF001
    det._stall_spans = [((raw[4][0] - start).total_seconds(), 600.0)]  # noqa: SLF001
    cut = det._match_readings()  # noqa: SLF001
    assert cut is not raw and len(cut) < len(raw)
    exact = (cut[-1][0] - cut[0][0]).total_seconds()
    span = (raw[-1][0] - raw[0][0]).total_seconds()
    assert exact < span
    interval = float(manager._profile_match_interval)  # noqa: SLF001
    quiet = _result("Cotton", {"Cotton": 0.9, "Synthetics": 0.5})
    with patch.object(det, "_match_readings", wraps=det._match_readings) as built, \
            patch.object(manager.profile_store, "async_match_profile",
                         AsyncMock(return_value=quiet)) as match:
        manager._program_mismatch_last_s = span + 1.0 - interval  # noqa: SLF001
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert not built.called and match.await_count == 0
        # Due: the trace is built and the check runs on it.
        manager._program_mismatch_last_s = span - interval  # noqa: SLF001
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert built.call_count == 1 and match.await_count == 1
    assert manager._program_mismatch_last_s == span  # noqa: SLF001
    assert match.await_args.args[1] == exact


async def test_a_stall_banked_after_a_check_does_not_push_the_next_one_back(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """PR #529 review: a check runs, then a 600 s stall is banked on resume and the
    cut trace moves back by it. The next check is still due one interval later on
    the stored trace, not once the cut trace has made up the stall: a cycle with
    less than the stall left would otherwise end unchecked."""
    _entry, manager, _sent, _state = await _past_half(hass, setup_washdata_entry, freezer)
    det = manager.detector
    interval = float(manager._profile_match_interval)  # noqa: SLF001
    checked = manager._program_mismatch_last_s  # noqa: SLF001
    assert checked is not None
    raw = det._power_readings  # noqa: SLF001
    start = det._current_cycle_start  # noqa: SLF001
    det._stall_spans = [((raw[4][0] - start).total_seconds(), 600.0)]  # noqa: SLF001
    cut = det._match_readings()  # noqa: SLF001
    assert (cut[-1][0] - cut[0][0]).total_seconds() + interval < checked
    quiet = _result("Cotton", {"Cotton": 0.9, "Synthetics": 0.5})
    with patch.object(manager.profile_store, "async_match_profile",
                      AsyncMock(return_value=quiet)) as match:
        await _feed(hass, freezer, 200, int(2 * interval))
        assert match.await_count >= 1
        # The fraction gate still reads programme time: the cut trace's span.
        cut = det._match_readings()  # noqa: SLF001
        assert match.await_args.args[1] <= (cut[-1][0] - cut[0][0]).total_seconds()
        assert match.await_args.args[1] < manager._program_mismatch_last_s - 500  # noqa: SLF001


async def test_a_pick_in_a_silent_tail_is_checked_on_the_watchdog_keepalives(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """A pick made in a silent low-power tail (a dishwasher's ENDING drying: the
    detector's match tick frozen) refreshes only the ETA, and no reading reaches the
    power-event handler after it. The watchdog's keepalives extend the trace, so
    they schedule the check too."""
    entry = await setup_washdata_entry(
        power_sensor=POWER, options={**FAST, "off_delay": 1800, "min_off_gap": 1800}
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    await _seed(manager)
    await _feed(hass, freezer, 200, 16 * 60)
    await _feed(hass, freezer, 0.5, 180)
    det = manager.detector
    assert det.is_waiting_low_power()
    assert manager.check_state() in ("paused", "ending"), manager.check_state()
    quiet = _result("Cotton", {"Cotton": 0.9, "Synthetics": 0.5})
    with patch.object(det, "_try_profile_match", lambda *_a, **_k: None), \
            patch.object(manager.profile_store, "async_match_profile",
                         AsyncMock(return_value=quiet)) as match:
        assert manager.set_manual_program("Cotton")
        await hass.async_block_till_done()
        assert match.await_count == 0
        held = len(det._power_readings)  # noqa: SLF001
        for _ in range(8):  # the plug stays silent: only the watchdog timer
            freezer.tick(timedelta(seconds=30))
            async_fire_time_changed(hass, dt_util.utcnow())
            await hass.async_block_till_done()
        assert len(det._power_readings) > held  # noqa: SLF001  (keepalives)
        assert manager.check_state() in ("running", "paused", "ending")
        # One per profile_match_interval (60 s) of trace, the keepalives 30 s apart.
        assert match.await_count >= 2
        assert all(c.kwargs["in_progress"] is True for c in match.await_args_list)
        # A report the watchdog's resync catches up on is such a reading too.
        before = match.await_count
        manager._program_mismatch_last_s = None  # noqa: SLF001  (a check is due)
        manager._last_real_reading_time -= timedelta(hours=1)  # noqa: SLF001  (missed)
        held = len(det._power_readings)  # noqa: SLF001
        manager._resync_power_from_state(dt_util.utcnow(), feed_detector=True)  # noqa: SLF001
        await hass.async_block_till_done()
        assert len(det._power_readings) == held + 1  # noqa: SLF001
        assert match.await_count == before + 1


# PR #529 review (CodeRabbit, round 23) ----------------------------------------


async def test_a_pick_that_cannot_be_matched_is_asked_once_per_interval(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """The eligibility scans read the evidence: a pick the matcher cannot consider
    is turned away at most once per profile_match_interval, and asked again once
    an interval has passed, so evidence that arrives still gets checked."""
    _entry, manager, _sent, _state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    quiet = _result("Cotton", {"Cotton": 0.9, "Synthetics": 0.5})
    with patch.object(store, "unmatchable_profiles", return_value={"Cotton": "no_template"}) \
            as unmatchable, \
            patch.object(store, "async_match_profile", AsyncMock(return_value=quiet)) as match:
        manager._program_mismatch_last_s = None  # noqa: SLF001  (a check is due)
        for _ in range(4):  # the readings of one interval
            manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert unmatchable.call_count == 1 and match.await_count == 0
        unmatchable.return_value = {}
        await _feed(hass, freezer, 200, 3 * int(manager._profile_match_interval))  # noqa: SLF001
        assert match.await_count >= 1


async def test_the_snapshot_keeps_the_cadence_not_the_pending_streak(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """A restart one count short of firing, with the next check in flight (PR #529
    review). The trace position of the last check comes back, so the next one
    still waits a full interval. The streak does not: the check in flight may
    have been a miss, and saved with its position it was skipped, so the next
    deficit after the restart completed a run that check would have broken. It
    starts a run instead. An older snapshot without the position keeps ``fired``."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    gate = asyncio.Event()
    with patch.object(store, "async_match_profile", new=_slow(gate, wrong)):
        manager._program_mismatch_last_s = None  # noqa: SLF001  (a check is due)
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await asyncio.sleep(0)
        task = manager._program_mismatch_task  # noqa: SLF001
        assert task is not None and not task.done()
        checked = manager._program_mismatch_last_s  # noqa: SLF001
        assert checked is not None and checked > 0
        snap = manager._augment_active_snapshot(manager.detector.get_state_snapshot())  # noqa: SLF001
        task.cancel()  # the restart: the check in flight never lands
        await hass.async_block_till_done()
    raw = snap["program_mismatch"]
    assert raw["checked_s"] == checked and "streak" not in raw

    manager._program_mismatch = match_rules.ProgramMismatchState()  # noqa: SLF001
    manager._program_mismatch_last_s = None  # noqa: SLF001
    manager._restore_program_mismatch(raw)  # noqa: SLF001
    back = manager._program_mismatch_state()  # noqa: SLF001
    assert back.streak == 0
    assert manager._program_mismatch_last_s == checked  # noqa: SLF001
    # An older snapshot's streak is not taken back either.
    manager._program_mismatch = match_rules.ProgramMismatchState()  # noqa: SLF001
    manager._restore_program_mismatch({**raw, "streak": PROGRAM_MISMATCH_PERSISTENCE - 1})  # noqa: SLF001
    back = manager._program_mismatch_state()  # noqa: SLF001
    assert back.streak == 0
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, back.key)
    assert _mismatches(sent) == []
    assert back.streak == 1 and not back.fired

    old = {k: v for k, v in raw.items() if k != "checked_s"}
    for legacy in (old, {**raw, "checked_s": "x"}, {**raw, "checked_s": 10**400}):
        manager._program_mismatch = match_rules.ProgramMismatchState()  # noqa: SLF001
        manager._program_mismatch_last_s = None  # noqa: SLF001
        manager._restore_program_mismatch({**legacy, "fired": True, "looks_like": "Eco"})  # noqa: SLF001
        back = manager._program_mismatch_state()  # noqa: SLF001
        assert back.streak == 0 and back.fired and back.looks_like == "Eco"
        assert manager._program_mismatch_last_s is None  # noqa: SLF001


async def test_a_streak_counted_before_an_import_restarts_at_the_next_check(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """An import that lands BETWEEN two checks replaces the profiles the pending
    streak was counted on: the first deficit against the new ones starts a run,
    it does not complete the old one."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    await store.async_import_data(store.export_data())  # same names and groups
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
        assert _mismatches(sent) == []
        assert state.streak == 1 and not state.fired
        # Unchanged data from here on: the run goes on and establishes it.
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE - 1):
            await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


async def test_a_streak_counted_before_a_group_edit_restarts_at_the_next_check(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """PR #529 review: a group edit BETWEEN two checks changes the membership the
    deficits are read against. The pending streak was counted under the old
    groups, so the first deficit under the new ones starts a run; a run under
    unchanged groups still establishes it, once."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    assert await store.create_profile_group("Solo", ["Cotton"])
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
        assert _mismatches(sent) == []
        assert state.streak == 1 and not state.fired
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE):
            await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


async def test_a_restart_after_a_group_edit_or_import_starts_the_run_afresh(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """PR #529 review: a group edit or an import after the last snapshot write and
    before a crash. The restore stamped the groups and data it found onto the
    restored streak, which was counted under the old ones, so one deficit after
    the restart completed it and sent a false trigger. No streak is restored."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    for change in ("group", "import"):
        state = manager._program_mismatch_state()  # noqa: SLF001
        state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
        snap = manager._augment_active_snapshot(manager.detector.get_state_snapshot())  # noqa: SLF001
        if change == "group":
            assert await store.create_profile_group("Solo", ["Cotton"])
        else:
            await store.async_import_data(store.export_data())  # same names and groups
        manager._program_mismatch = match_rules.ProgramMismatchState()  # noqa: SLF001
        manager._restore_program_mismatch(snap["program_mismatch"])  # noqa: SLF001
        back = manager._program_mismatch_state()  # noqa: SLF001
        with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
            await _check(hass, manager, back.key)
        assert _mismatches(sent) == [], change
        assert back.streak == 1 and not back.fired, change


async def test_a_pick_named_like_a_placeholder_is_still_checked(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """PR #529 review: a matchable profile named ``off`` can be picked. The check
    used to drop it as "nothing committed", so no mismatch automation, filtered
    or not, could fire for it. The trigger carries the pick's own name."""
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    await _seed(manager)
    store = manager.profile_store
    pts = _trace(COTTON)
    begin = dt_util.utcnow() - timedelta(days=2)
    store._data["past_cycles"].append({  # noqa: SLF001
        "id": "off-0", "profile_name": "off", "label_source": "manual",
        "status": "completed", "start_time": begin.isoformat(),
        "end_time": (begin + timedelta(seconds=pts[-1][0])).isoformat(),
        "duration": pts[-1][0], "power_data": [list(p) for p in pts],
    })
    store._data["profiles"]["off"] = {  # noqa: SLF001
        "avg_duration": pts[-1][0], "sample_cycle_id": "off-0",
    }
    await store.async_rebuild_envelope("off")
    assert "off" not in store.unmatchable_profiles()
    sent: list[tuple[str, dict]] = []
    async_dispatcher_connect(
        hass, SIGNAL_DEVICE_TRIGGER.format(entry.entry_id),
        lambda kind, variables: sent.append((kind, variables)),
    )
    await _feed(hass, freezer, 200, 90)
    assert manager.set_manual_program("off")
    res = _result("Synthetics", {"Synthetics": 0.9, "off": 0.5})
    with patch.object(store, "async_match_profile", AsyncMock(return_value=res)):
        await _feed(hass, freezer, 200, 20 * 60)
    fired = _mismatches(sent)
    assert len(fired) == 1, sent
    assert fired[0]["program"] == "off" and fired[0]["looks_like"] == "Synthetics"
    assert manager.program_mismatch["selected"] == "off"


async def test_a_restart_keeps_fired_whatever_changed_since(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """``fired`` is saved as it is after an Undo Import or a group edit, so a
    restart does not send it again; no pending streak is saved with it."""
    _entry, manager, _sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    state.fired, state.looks_like = True, "Synthetics"
    await store.async_import_data(store.export_data())  # same names and groups
    assert await store.create_profile_group("Solo", ["Cotton"])
    snap = manager._augment_active_snapshot(manager.detector.get_state_snapshot())  # noqa: SLF001
    raw = snap["program_mismatch"]
    assert "streak" not in raw
    assert raw["fired"] is True and raw["looks_like"] == "Synthetics"
    manager._program_mismatch = match_rules.ProgramMismatchState()  # noqa: SLF001
    manager._restore_program_mismatch(raw)  # noqa: SLF001
    assert manager.program_mismatch["looks_like"] == "Synthetics"


def test_a_malformed_group_is_skipped_not_fatal() -> None:
    """PR #529 review: a stored group with ``"members": 1`` raised TypeError here,
    which failed the restore of the whole active cycle."""
    groups = {"Bad": {"members": 1}, "Odd": "x", "Fam": {"members": ["Cotton", 3, "Eco"]}}
    assert match_rules.program_families(groups) == {"Cotton": "Fam", "Eco": "Fam"}


async def test_a_malformed_group_neither_drops_the_check_nor_fails_its_restore(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    _entry, manager, _sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    manager.profile_store._data["profile_groups"]["Bad"] = {"members": 1}  # noqa: SLF001
    state.fired, state.looks_like = True, "Synthetics"
    snap = manager._augment_active_snapshot(manager.detector.get_state_snapshot())  # noqa: SLF001
    assert snap["program_mismatch"]["fired"] is True
    manager._program_mismatch = match_rules.ProgramMismatchState()  # noqa: SLF001
    manager._restore_program_mismatch(snap["program_mismatch"])  # noqa: SLF001
    assert manager.program_mismatch["looks_like"] == "Synthetics"
    # Whatever else fails in it, the cycle's restore goes on without the check.
    with patch.object(
        match_rules.ProgramMismatchState, "restore", side_effect=RuntimeError("boom")
    ):
        manager._restore_program_mismatch(snap["program_mismatch"])  # noqa: SLF001
    assert manager.program_mismatch is None


async def test_a_deleted_pick_is_not_checked(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """PR #529 review: deleting the picked profile left the pin (and the data
    generation) as it was, so the check went on against a name no profile has and
    could fire the trigger for it."""
    _entry, manager, sent, _state = await _past_half(hass, setup_washdata_entry, freezer)
    await manager.profile_store.delete_profile("Cotton")
    assert manager._program_mismatch_key() is None  # noqa: SLF001
    res = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    with patch.object(
        manager.profile_store, "async_match_profile", AsyncMock(return_value=res)
    ):
        await _feed(hass, freezer, 200, 10 * 60)
    assert _mismatches(sent) == []
    assert manager.program_mismatch is None
