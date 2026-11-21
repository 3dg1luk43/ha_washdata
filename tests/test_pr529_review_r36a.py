"""Review of PR #529, round 36a: the Minor findings, each reproduced.

* A saved #530 check marked fired but naming no programme, or with no finite
  margin, was latched: the State sensor showed a mismatch naming nothing and no
  later check of the pick ran.
* A profile dropped with its last imported or backfilled cycle (relabelled away
  or deleted) left the store without moving the profile set revision, so a #530
  ranking naming it passed the guard da2ea73d added.
* A door opened in PAUSED after a program was picked by hand, before the
  detector's next match tick took it, paused the cycle; opened after the tick,
  the same door ended it.
* A stopped cycle whose last sample is NaN (a wholesale import keeps one) read
  as idling below the Stop Threshold and offset the cycles idling above it.
* A Start withdrawn as inverted against a locked Stop left the Min Power cascade
  its derived Stop had made.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
    STATE_PAUSED,
)
from custom_components.ha_washdata.learning import OWNER_BATCH
from custom_components.ha_washdata.suggestion_engine import (
    SuggestionEngine,
    _idle_evidence_w,
    detect_standby_above_stop,
)

from .test_issue_524_door_ends_cycle import DOOR, OPTS, WASH_S, _paused_washer, _tick
from .test_pr529_review_r35a import _Store

# ── An incomplete fired #530 record starts the check afresh ──────────────────


def test_a_fired_record_naming_nothing_starts_the_check_afresh() -> None:
    incomplete = [
        {"fired": True, "margin": 0.2},
        {"fired": True, "looks_like": "", "margin": 0.2},
        {"fired": True, "looks_like": 3, "margin": 0.2},
        {"fired": True, "looks_like": "Eco"},
        {"fired": True, "looks_like": "Eco", "margin": None},
        {"fired": True, "looks_like": "Eco", "margin": float("nan")},
        {"fired": True, "looks_like": "Eco", "margin": "0.2"},
        {"fired": True, "looks_like": "Eco", "margin": True},
    ]
    for raw in incomplete:
        state = match_rules.ProgramMismatchState(key="k")
        state.restore(raw)
        assert state.as_dict() == {"fired": False, "looks_like": None, "margin": None}, raw
        # The pick is checked again, and a mismatch found is recorded whole.
        assert match_rules.program_mismatch_step(state, ("Quick", 0.3), 1.0, persistence=1)
        assert state.as_dict() == {"fired": True, "looks_like": "Quick", "margin": 0.3}
    # A complete record comes back as saved: no repeated trigger.
    valid = match_rules.ProgramMismatchState(key="k")
    valid.restore({"fired": True, "looks_like": "Eco", "margin": 0.2})
    assert valid.as_dict() == {"fired": True, "looks_like": "Eco", "margin": 0.2}
    assert match_rules.program_mismatch_step(valid, ("Quick", 0.3), 1.0, persistence=1) is False


# ── A profile dropped with its last non-real cycle moves the profile set ─────


async def test_a_profile_dropped_with_its_last_imported_cycle_moves_the_profile_set(
    hass: Any, setup_washdata_entry: Any
) -> None:
    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    profiles = store._data["profiles"]  # noqa: SLF001
    profiles["Imported"] = {"name": "Imported", "sample_cycle_id": "b1"}
    profiles["Store"] = {"name": "Store", "sample_cycle_id": "r1"}
    profiles["Target"] = {"name": "Target"}
    moved = {"id": "b1", "profile_name": "Imported", "status": "completed"}
    first = {"id": "b2", "profile_name": "Imported", "status": "completed"}
    store._data["backfill_cycles"] = [moved, first]  # noqa: SLF001
    store._data["reference_cycles"] = [  # noqa: SLF001
        {"id": "r1", "profile_name": "Store", "status": "completed"}
    ]

    # Relabelled away from a profile that keeps a cycle: no profile leaves.
    before = store.profile_set_revision
    store._relabel_non_real_cycle(first, "Target")  # noqa: SLF001
    assert "Imported" in store.get_profiles()
    assert store.profile_set_revision == before

    # Its last cycle relabelled away: the profile is dropped, the revision moves.
    store._relabel_non_real_cycle(moved, "Target")  # noqa: SLF001
    assert "Imported" not in store.get_profiles()
    assert store.profile_set_revision == before + 1

    # Its last cycle deleted: the same.
    assert await store.delete_cycle("r1")
    assert "Store" not in store.get_profiles()
    assert store.profile_set_revision == before + 2


# ── A door opened before the detector takes a pick is judged by the pick ─────


async def test_a_door_opened_before_the_tick_takes_the_pick_ends_the_cycle(
    hass: Any, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Picked by hand in PAUSED at the programme's end; the door opens before the
    detector's next match tick. The pick is handed over first, so the door is
    judged as it is after that tick: the dwell, then the end. The detector is held
    in PAUSED (no reading reaches it), as in the round-34a test of the reverse
    order, so no ENDING transition arms a dwell on its own."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = "Quick"  # noqa: SLF001 - picked, detector not told yet
    manager._matched_profile_duration = float(WASH_S)  # noqa: SLF001
    assert manager._door_end_may_fire() is False  # noqa: SLF001 - awaits the tick
    with patch.object(manager.detector, "process_reading", lambda *_a, **_k: None):
        hass.states.async_set(DOOR, "on")
        await hass.async_block_till_done()
        assert manager.detector.matched_profile == "Quick"
        assert manager._is_user_paused is False  # noqa: SLF001 - not the sticky pause
        assert manager._remove_door_end_dwell is not None  # noqa: SLF001
        await _tick(hass, freezer, 70, None)
    assert len(ends) == 1, ends


async def test_a_door_opened_before_the_tick_mid_programme_still_pauses(
    hass: Any, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The pick handed over is three times as long: the run is mid-programme by it,
    so the door pauses, as before 0.6.1, and ends nothing."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = "Intensive"  # noqa: SLF001
    manager._matched_profile_duration = float(3 * WASH_S)  # noqa: SLF001
    with patch.object(manager.detector, "process_reading", lambda *_a, **_k: None):
        hass.states.async_set(DOOR, "on")
        await hass.async_block_till_done()
        assert manager._is_user_paused is True  # noqa: SLF001
        assert manager._remove_door_end_dwell is None  # noqa: SLF001
        await _tick(hass, freezer, 120, None)
    assert ends == []
    assert manager.detector.state == STATE_PAUSED


async def test_a_door_opened_after_a_cleared_pick_still_pauses(
    hass: Any, setup_washdata_entry: Any, freezer: Any
) -> None:
    """A cleared pick waits for an automatic match, which no door hands over: the
    door pauses, as before 0.6.1."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = manager.detector.matched_profile  # noqa: SLF001
    manager._matched_profile_duration = float(manager.detector.expected_duration_seconds)  # noqa: SLF001
    manager.clear_manual_program()
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager._is_user_paused is True  # noqa: SLF001
    assert ends == []


# ── A NaN idle level is no evidence ──────────────────────────────────────────


def _hand_stopped(tail: list[float]) -> dict[str, Any]:
    """Ten minutes at 500 W, then ``tail`` at 10 s steps; stopped by hand."""
    pts = [[float(t), 500.0] for t in range(0, 600, 10)]
    pts += [[600.0 + 10.0 * i, p] for i, p in enumerate(tail)]
    return {"status": "completed", "termination_reason": "user", "power_data": pts,
            "signature": {"max_power": 500.0}}


def test_a_nan_last_sample_does_not_hide_a_standby_above_stop() -> None:
    above = _hand_stopped([3.4] * 30)
    nan = _hand_stopped([float("nan")])
    assert _idle_evidence_w(nan, 2.5) is None
    found = detect_standby_above_stop([above, above, nan, nan], 2.5)
    assert found is not None
    assert (found["cycles_above"], found["cycles_checked"]) == (2, 2)
    # A NaN sample under a settled tail: the tail is the reading.
    assert _idle_evidence_w(_hand_stopped([3.4] * 30 + [float("nan")]), 2.5) == 3.4
    # A measured 0 W is still a stop below the threshold.
    zero = _hand_stopped([0.0] * 30)
    assert _idle_evidence_w(zero, 2.5) == 0.0
    assert detect_standby_above_stop([above, above, zero, zero], 2.5) is None


# ── A withdrawn Start takes the cascades derived from it ─────────────────────


class _LockedStore(_Store):
    def __init__(self, locked: list[str]) -> None:
        super().__init__()
        self.locked = locked

    def get_locked_suggestions(self) -> list[str]:
        return list(self.locked)


_WASHER_OPTS = {
    CONF_DEVICE_TYPE: "washing_machine", CONF_START_THRESHOLD_W: 10.0,
    CONF_STOP_THRESHOLD_W: 4.0, CONF_MIN_POWER: 5.0,
    CONF_OFF_DELAY: 300, CONF_MIN_OFF_GAP: 300,
}


def test_a_start_inverted_against_a_locked_stop_leaves_no_cascade_of_its_own() -> None:
    """Start 2 W proposed, Stop locked at 4 W: reconcile derived Stop 1.6 W and from
    it Min Power 1.3 W; the lock dropped the Stop, the pair check the Start, and
    the Min Power stayed without the change that justified it. An Off Delay
    proposed beside it keeps its own Min Off Gap cascade."""
    store = _LockedStore([CONF_STOP_THRESHOLD_W])
    eng = SuggestionEngine(MagicMock(), "e", store, device_type="washing_machine")
    eng._entry_options = lambda: dict(_WASHER_OPTS)  # noqa: SLF001
    store.set_suggestion(CONF_START_THRESHOLD_W, 2.0, reason="r", flags={"owner": OWNER_BATCH})
    store.set_suggestion(CONF_OFF_DELAY, 600, reason="r", flags={"owner": OWNER_BATCH})
    eng._reconcile_stored_suggestions()  # noqa: SLF001
    assert set(store.suggestions) == {CONF_OFF_DELAY, CONF_MIN_OFF_GAP}
    assert store.suggestions[CONF_MIN_OFF_GAP]["cascade"] is True
    assert store.suggestions[CONF_MIN_OFF_GAP]["value"] == 600
