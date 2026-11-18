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
