"""Review of PR #529, round 41a, each reproduced.

* #530: a due check turned away unscored (the pick has no template, or no
  profile is backed by evidence) kept the pending streak, so the next scored
  deficit completed a run across the gap. It breaks the run now.
* #530: the trace the check matched kept a shown stall until it was banked, so
  a long one carried the span past the pick's Stage-1 upper bound and checks in
  it could latch a false mismatch. No check is scheduled while a stall is shown.
* A power-cut Pause whose switch call failed after a Resume and a later Pause
  had succeeded rolled back the later Pause and re-armed the open door's dwell,
  which then ended the paused cycle. A Pause rolls back only while the pause
  state is still its own; a failed Resume hands it back.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import (
    PROGRAM_MISMATCH_PERSISTENCE,
    TerminationReason,
)
from custom_components.ha_washdata.profile_store import ProfileStore

from .test_issue_524_door_ends_cycle import DOOR, OPTS, _paused_washer, _tick
from .test_issue_530_program_mismatch import _check, _mismatches, _past_half, _result

WRONG = ("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})


# ── #530: an unscored due check breaks the run ───────────────────────────────


@pytest.mark.parametrize("gap", ["unmatchable", "no_real_profiles"])
async def test_an_unscored_due_check_breaks_the_run(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any, gap: str
) -> None:
    """Five hits, then a due check the matcher cannot score: the next deficit
    starts a run, it does not complete the five."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    wrong = _result(*WRONG)
    if gap == "unmatchable":
        unscored = patch.object(store, "unmatchable_profiles", return_value={"Cotton": "x"})
    else:
        unscored = patch.object(
            ProfileStore, "has_real_profiles", new_callable=PropertyMock, return_value=False
        )
    with unscored, patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)) as match:
        manager._program_mismatch_last_s = None  # noqa: SLF001  (a check is due)
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert match.await_count == 0
    assert state.streak == 0
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == [] and state.streak == 1 and not state.fired


# ── #530: no check while a stall is shown ────────────────────────────────────


async def test_no_check_is_scheduled_while_a_stall_is_shown(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Five hits, then a stall shown: the trace still spans it (a long one carries
    the span past the pick's Stage-1 upper bound), so no check is scheduled and
    the run is kept. Banked and cut out of the trace, the next due check counts."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    det = manager.detector
    store = manager.profile_store
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    start = det._current_cycle_start  # noqa: SLF001
    run = det._power_readings[-1][0] - timedelta(seconds=60)  # noqa: SLF001
    det._stall_active = True  # noqa: SLF001
    det._stall_run_start = run  # noqa: SLF001
    wrong = _result(*WRONG)
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)) as match:
        manager._program_mismatch_last_s = None  # noqa: SLF001  (a check is due)
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert match.await_count == 0
        assert state.streak == PROGRAM_MISMATCH_PERSISTENCE - 1 and _mismatches(sent) == []
        # The stall is banked: cut out of the trace the check matches.
        det._stall_spans.append(((run - start).total_seconds(), 60.0))  # noqa: SLF001
        det._stall_active = False  # noqa: SLF001
        det._stall_run_start = None  # noqa: SLF001
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert match.await_count == 1
    assert len(_mismatches(sent)) == 1


# ── A stale failed power-cut Pause leaves a later Pause alone ────────────────

SWITCH = "switch.washer_plug"
CUT_OPTS = {**OPTS, "pause_cuts_power": True, "switch_entity": SWITCH}


async def _pause_failing_late(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> tuple[Any, list[dict[str, Any]], Any, asyncio.Event]:
    """A washer PAUSED at its programme end, the door open (a 60 s dwell armed),
    Pause A pressed: the dwell expires in its switch call, which then waits for
    ``release`` and fails. Every later switch-off call succeeds."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, CUT_OPTS)
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager._remove_door_end_dwell is not None  # noqa: SLF001
    entered, release = asyncio.Event(), asyncio.Event()
    offs = 0

    async def turn_off(_call: ServiceCall) -> None:
        nonlocal offs
        offs += 1
        if offs > 1:
            return
        freezer.tick(timedelta(seconds=70))
        async_fire_time_changed(hass, dt_util.utcnow())
        await asyncio.sleep(0)
        assert manager._remove_door_end_dwell is None  # noqa: SLF001  (declined in A)
        entered.set()
        await release.wait()
        raise HomeAssistantError("plug offline")

    hass.services.async_register("switch", "turn_off", turn_off)
    pause_a = hass.async_create_task(manager.async_pause_cycle())
    await entered.wait()
    return manager, ends, pause_a, release


async def test_a_stale_failed_pause_leaves_a_later_pause_alone(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Resume and Pause B succeed while A's power cut is pending; A then fails.
    B's pause stands, no dwell is re-armed, and the open door ends nothing."""
    manager, ends, pause_a, release = await _pause_failing_late(
        hass, setup_washdata_entry, freezer
    )

    async def turn_on(_call: ServiceCall) -> None:
        return None

    hass.services.async_register("switch", "turn_on", turn_on)
    assert await manager.async_resume_cycle()
    assert await manager.async_pause_cycle()  # B
    release.set()
    assert await pause_a is False
    await hass.async_block_till_done()
    assert manager._is_user_paused is True  # noqa: SLF001
    assert manager.detector._verified_pause is True  # noqa: SLF001
    assert manager._remove_door_end_dwell is None  # noqa: SLF001
    await _tick(hass, freezer, 120, None)
    assert ends == []


async def test_a_failed_pause_after_a_failed_resume_still_rolls_back(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """A Resume whose power-on fails hands A's pause back, so A's own failure
    still rolls it back and judges the open door again (a fresh dwell)."""
    manager, ends, pause_a, release = await _pause_failing_late(
        hass, setup_washdata_entry, freezer
    )

    async def turn_on(_call: ServiceCall) -> None:
        raise HomeAssistantError("plug offline")

    hass.services.async_register("switch", "turn_on", turn_on)
    assert await manager.async_resume_cycle() is False
    assert manager._is_user_paused is True  # noqa: SLF001
    release.set()
    assert await pause_a is False
    await hass.async_block_till_done()
    assert manager._is_user_paused is False  # noqa: SLF001
    assert manager._remove_door_end_dwell is not None  # noqa: SLF001
    await _tick(hass, freezer, 70, None)
    assert len(ends) == 1 and ends[0]["termination_reason"] == TerminationReason.DOOR
