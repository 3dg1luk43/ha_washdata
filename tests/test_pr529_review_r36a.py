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
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import DOMAIN, STATE_PAUSED

from .test_issue_524_door_ends_cycle import DOOR, OPTS, WASH_S, _paused_washer, _tick

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
