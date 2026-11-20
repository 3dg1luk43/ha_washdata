"""Review of PR #529, round 37a: the door end in PAUSED (#524) and a pick the
detector has not taken yet, each reproduced.

* A door sensor back from unavailable with the door open: the recovery skipped
  the hand-over the ordinary open makes first, so a pick the detector had not
  taken kept the end dwell unarmed, and nothing armed it once the detector took
  the pick with the door still open.
* The detector held an ambiguous automatic match of the program then picked by
  hand at the same length: nothing waited for the detector to take the pick, so
  the door was judged by the stale ambiguity and paused the cycle, where after
  the next match tick the same door ended it. Inside the anti-crease freeze no
  match lands, the pick's included: there the door still pauses.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from custom_components.ha_washdata.const import STATE_PAUSED

from .test_issue_524_door_ends_cycle import DOOR, OPTS, WASH_S, _paused_washer, _tick

# ── A door back from unavailable, open, hands the pick over ──────────────────


async def test_a_door_open_when_its_sensor_comes_back_hands_the_pick_over(
    hass: Any, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Picked by hand in PAUSED at the programme's end while the door sensor was
    unavailable; it comes back open before the detector's next match tick. The
    detector is held in PAUSED (no reading reaches it), as in round 36a."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    hass.states.async_set(DOOR, "unavailable")
    await hass.async_block_till_done()
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = "Quick"  # noqa: SLF001 - picked, detector not told yet
    manager._matched_profile_duration = float(WASH_S)  # noqa: SLF001
    with patch.object(manager.detector, "process_reading", lambda *_a, **_k: None):
        hass.states.async_set(DOOR, "on")
        await hass.async_block_till_done()
        assert manager.detector.matched_profile == "Quick"
        assert manager._is_user_paused is False  # noqa: SLF001
        assert manager._remove_door_end_dwell is not None  # noqa: SLF001
        await _tick(hass, freezer, 70, None)
    assert len(ends) == 1, ends


async def test_a_door_open_when_its_sensor_comes_back_mid_programme_arms_nothing(
    hass: Any, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The pick handed over is three times as long: mid-programme by it, so the
    door back from unavailable arms no dwell and ends nothing."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    hass.states.async_set(DOOR, "unavailable")
    await hass.async_block_till_done()
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = "Intensive"  # noqa: SLF001
    manager._matched_profile_duration = float(3 * WASH_S)  # noqa: SLF001
    with patch.object(manager.detector, "process_reading", lambda *_a, **_k: None):
        hass.states.async_set(DOOR, "on")
        await hass.async_block_till_done()
        assert manager._remove_door_end_dwell is None  # noqa: SLF001
        await _tick(hass, freezer, 120, None)
    assert ends == []
    assert manager.detector.state == STATE_PAUSED


# ── An ambiguous automatic match under a pick of the same program ────────────


def _pick_the_ambiguous_match(manager: Any, length_s: float) -> None:
    """The detector holds an ambiguous "Cotton" at ``length_s``; the user picks
    Cotton at the same length, so name and length already agree."""
    manager.detector._match_ambiguous = True  # noqa: SLF001
    manager.detector._expected_duration = float(length_s)  # noqa: SLF001
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = manager.detector.matched_profile  # noqa: SLF001
    manager._matched_profile_duration = float(length_s)  # noqa: SLF001


@pytest.mark.parametrize(("length_s", "ended"), [(WASH_S, 1), (3 * WASH_S, 0)])
async def test_a_door_opened_on_an_ambiguous_match_of_the_pick_is_judged_by_the_pick(
    hass: Any, setup_washdata_entry: Any, freezer: Any, length_s: float, ended: int
) -> None:
    """At the programme's end by the pick the door ends the cycle, as it does once
    the detector's next match tick has taken the pick; mid-programme it pauses."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    _pick_the_ambiguous_match(manager, length_s)
    assert manager._door_end_may_fire() is False  # noqa: SLF001 - awaits the pick
    with patch.object(manager.detector, "process_reading", lambda *_a, **_k: None):
        hass.states.async_set(DOOR, "on")
        await hass.async_block_till_done()
        assert manager.detector.match_ambiguous is False  # the pick was handed over
        assert manager._is_user_paused is (not ended)  # noqa: SLF001
        assert (manager._remove_door_end_dwell is not None) is bool(ended)  # noqa: SLF001
        await _tick(hass, freezer, 70, None)
    assert len(ends) == ended, ends


async def test_inside_the_anticrease_freeze_the_door_still_pauses(
    hass: Any, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The freeze drops every match update, the pick's included, so the detector
    keeps its ambiguous match and the door is judged by it: a pause, as before
    0.6.1. Nothing waits for a hand-over that cannot land."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    _pick_the_ambiguous_match(manager, WASH_S)
    with (
        patch.object(manager.detector, "process_reading", lambda *_a, **_k: None),
        patch.object(manager.detector, "_in_anticrease_freeze", return_value=True),
    ):
        assert manager._door_end_awaits_match() is False  # noqa: SLF001
        hass.states.async_set(DOOR, "on")
        await hass.async_block_till_done()
        assert manager.detector.match_ambiguous is True
        assert manager._is_user_paused is True  # noqa: SLF001
        assert manager._remove_door_end_dwell is None  # noqa: SLF001
        await _tick(hass, freezer, 120, None)
    assert ends == []
    assert manager.detector.state == STATE_PAUSED
