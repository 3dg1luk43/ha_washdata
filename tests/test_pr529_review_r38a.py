"""Review of PR #529, round 38a, each reproduced.

* #530: while a user pause was open the program check matched the trace with
  the pause still in its span. Stage 1 read the pause as programme time, so a
  right pick could fall past its upper bound, read at the fifth score, and six
  checks in the pause fired the trigger. The check now waits the pause out.
* #524: restored in PAUSED at the programme's end with the door already open and
  a program picked by hand that the detector had not taken: the restore judged
  the door before handing the pick over, so it armed nothing, and the detector
  took the pick at a later match tick without leaving PAUSED, where nothing arms
  the end dwell. The open contact now hands the pick over first, as an ordinary
  open does; the pick's length still decides.
"""
from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import async_capture_events

from custom_components.ha_washdata.const import (
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DOMAIN,
    EVENT_CYCLE_ENDED,
    STATE_PAUSED,
    STORAGE_KEY,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.profile_store import MatchResult
from custom_components.ha_washdata.time_utils import utc_now

from .test_issue_524_door_ends_cycle import DOOR, OPTS, WASH_S, _paused_washer, _tick
from .test_issue_530_program_mismatch import DURATION, _feed, _mismatches, _picked_cycle

# ── #530: an open user pause defers the check ────────────────────────────────


def _stage1(duration: float) -> MatchResult:
    """The matcher on a right Cotton pick: Cotton wins while the trace is within
    its Stage-1 upper bound; past it Cotton is rejected and missing from a full
    list of five, which the deficit reads at the fifth score."""
    if duration <= DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO * DURATION:
        scores = {"Cotton": 0.9, "Synthetics": 0.6}
    else:
        scores = {"Synthetics": 0.9, "A": 0.85, "B": 0.8, "C": 0.75, "D": 0.6}
    cands = [{"name": n, "score": s} for n, s in sorted(scores.items(), key=lambda kv: -kv[1])]
    margin = cands[0]["score"] - cands[1]["score"]
    return MatchResult(cands[0]["name"], cands[0]["score"], DURATION, None, cands,
                       margin < 0.05, margin, ranking=cands)


async def test_a_right_pick_is_not_checked_against_an_open_user_pause(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Cotton picked and run past half its length, then paused for longer than
    Stage 1 allows the whole trace: no check runs in the pause, nothing fires,
    and after the resume (the pause banked) the pick wins again."""
    _entry, manager, sent = await _picked_cycle(hass, setup_washdata_entry, freezer, "Cotton")
    durations: list[float] = []

    async def match(readings: Any, duration: float, **_k: Any) -> MatchResult:
        durations.append(float(duration))
        return _stage1(float(duration))

    with patch.object(manager.profile_store, "async_match_profile", new=match):
        await _feed(hass, freezer, 200, 18 * 60)
        assert durations and _mismatches(sent) == []
        assert await manager.async_pause_cycle()
        state = manager._program_mismatch_state()  # noqa: SLF001
        streak, checks = state.streak, len(durations)
        await _feed(hass, freezer, 0, 45 * 60)  # the trace now spans > 1.8x Cotton
        assert _mismatches(sent) == []
        assert len(durations) == checks  # no check in the pause
        assert state.streak == streak
        assert await manager.async_resume_cycle()
        await _feed(hass, freezer, 200, 3 * 60)
    assert len(durations) > checks  # checked again after the resume...
    # ...on the trace with the pause banked.
    assert max(durations[checks:]) <= DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO * DURATION
    assert _mismatches(sent) == []
    assert manager.program_mismatch is None


# ── #524: a restore with the door open hands an untaken pick over ───────────


async def _restart_open_with_untaken_pick(
    hass: HomeAssistant, manager: Any, hass_storage: Any, held_s: float, pick_s: float,
) -> Any:
    """A washer PAUSED after its final spin, the detector holding "Eco" at
    ``held_s`` and "Quick" at ``pick_s`` picked by hand after its last match
    tick; saved, then a crash and a restart with the door opened meanwhile. The
    caller holds the detector in PAUSED (no reading reaches it), as round 37a."""
    entry = manager.config_entry
    store = manager.profile_store
    store._data["profiles"]["Eco"] = {"avg_duration": held_s}  # noqa: SLF001
    store._data["profiles"]["Quick"] = {"avg_duration": pick_s}  # noqa: SLF001
    await store.async_save()
    manager.detector._matched_profile = "Eco"  # noqa: SLF001
    manager.detector._expected_duration = float(held_s)  # noqa: SLF001
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = "Quick"  # noqa: SLF001 - the detector not told yet
    manager._matched_profile_duration = float(pick_s)  # noqa: SLF001
    await store.async_save_active_cycle(
        manager._augment_active_snapshot(manager.detector.get_state_snapshot())  # noqa: SLF001
    )
    key = f"{STORAGE_KEY}.{entry.entry_id}.active"
    on_disk = copy.deepcopy(hass_storage[key])
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    hass_storage[key] = on_disk  # as a crash leaves it
    hass.states.async_set(DOOR, "on")  # opened while Home Assistant was down
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    manager = hass.data[DOMAIN][entry.entry_id]
    assert manager.detector.state == STATE_PAUSED
    assert manager.manual_program_active and manager.current_program == "Quick"
    return manager


@pytest.mark.parametrize(("held_s", "pick_s", "ended"), [
    (3 * WASH_S, WASH_S, 1),  # at the pick's end: the door ends it
    (WASH_S, 3 * WASH_S, 0),  # mid-programme by the pick: still nothing
])
async def test_a_restored_open_door_is_judged_by_the_untaken_pick(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any, hass_storage: Any,
    held_s: float, pick_s: float, ended: int,
) -> None:
    manager, _ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    events = async_capture_events(hass, EVENT_CYCLE_ENDED)
    with patch.object(CycleDetector, "process_reading", lambda *_a, **_k: None):
        manager = await _restart_open_with_untaken_pick(
            hass, manager, hass_storage, held_s, pick_s
        )
        assert (manager._remove_door_end_dwell is not None) is bool(ended)  # noqa: SLF001
        # The detector's next match tick, still in PAUSED, arms nothing either way.
        manager.detector.refresh_match(utc_now())
        await _tick(hass, freezer, 70, None)
    assert len(events) == ended, events
    if not ended:
        assert manager.detector.state == STATE_PAUSED
        assert manager.detector.matched_profile == "Quick"  # judged by the pick
