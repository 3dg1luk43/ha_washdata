"""Review of PR #529, round 40a, each reproduced.

* #530: the check's fraction gate read the stall shown now as programme time.
  ``_match_readings()`` keeps a shown stall in the trace and ``paused_s`` covered
  only the user pause, so a right pick stalled early reached half its duration
  on the stall's clock. The stall shown now leaves the elapsed time as well
  (``CycleDetector.open_halt_s``); finished stalls are cut from the trace already.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import cycle_detector as cd_mod
from custom_components.ha_washdata.const import PROGRAM_MISMATCH_PERSISTENCE

from .test_audit_detection_514_halt_followups import _det, _feed, _halt, _match, _wash
from .test_issue_530_program_mismatch import _check, _mismatches, _past_half, _result

# ── #530: the stall shown now is no programme time ───────────────────────────


def _span(det: Any) -> float:
    live = det._match_readings()  # noqa: SLF001
    return (live[-1][0] - live[0][0]).total_seconds()


def test_the_stall_shown_now_leaves_the_checks_elapsed_time() -> None:
    """30 min of a 90 min programme, then a halt shown as stalled: the trace the
    check matches still spans the halt, and its span less ``open_halt_s`` is the
    30 min of programme time. Banked on the resume, it is cut from the trace and
    not taken off again."""
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    _feed(det, 1800, 4200, _halt)
    assert det.stalled
    assert _span(det) == pytest.approx(4170.0)
    assert _span(det) - det.open_halt_s() == pytest.approx(1800.0, abs=30.0)
    with patch.object(cd_mod, "STALL_CURRENT_EXCLUDED_FROM_PROGRESS", False):
        assert det.open_halt_s() == 0.0
    _feed(det, 4200, 4800, _wash)
    assert not det.stalled
    assert det.open_halt_s() == 0.0
    assert _span(det) == pytest.approx(4770.0 - 2400.0, abs=30.0)


def test_a_user_pause_inside_the_stall_is_not_taken_off_twice() -> None:
    """The stall shown and a user pause opened in it: their union, once."""
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    _feed(det, 1800, 3000, _halt)
    assert det.stalled
    det.set_user_paused(True, cd_mod.dt_util.as_utc(det._power_readings[-1][0]))  # noqa: SLF001
    _feed(det, 3000, 3600, _halt)
    assert det.open_user_pause_s() > 0
    assert _span(det) - det.open_halt_s() == pytest.approx(1800.0, abs=30.0)


async def test_a_pick_stalled_early_is_not_past_half(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Cotton (30 min) picked, the trace ~17 min long but stalled since minute 5:
    five hits in a row and a sixth do not establish a mismatch, since 5 min of
    programme time is not half of 30. Without the stall the sixth does."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    det = manager.detector
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    det._stall_active = True  # noqa: SLF001  (shown as stalled since minute 5)
    det._stall_run_start = det._current_cycle_start + timedelta(minutes=5)  # noqa: SLF001
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == PROGRAM_MISMATCH_PERSISTENCE and not state.fired
    det._stall_active = False  # noqa: SLF001
    det._stall_run_start = None  # noqa: SLF001
    with patch.object(manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1
