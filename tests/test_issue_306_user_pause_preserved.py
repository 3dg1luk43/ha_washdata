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
"""Regression tests for issue #306.

A user-initiated pause (Pause Cycle button, or the door-open soft pause) must
stay in force until the user explicitly resumes.  The bug: the periodic profile
matcher (`_async_do_perform_matching`) re-judged the low-power reading with the
envelope-alignment heuristic and, on a mismatch, cleared the detector's
``verified_pause`` flag.  With the flag gone the cycle was finalized on the next
ENDING timeout (for a dishwasher, the 1 h ``min_off_gap``), so the State sensor
left "Paused by user" (surfacing as Interrupted / Finished) while the user still
expected it paused.

These tests pin the invariant directly on the matching path: while
``_is_user_paused`` is set, matching must never clear ``verified_pause`` — and it
must not even run the (wasteful) envelope alignment.  The non-paused case is kept
as a control so the auto-detected pause behaviour is proven unchanged.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import STATE_PAUSED
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult

PROFILE = "Cotton 60°C"


def _low_power_readings(count: int = 12, power: float = 1.0) -> list[tuple]:
    """Readings well below stop_threshold (machine paused), spanning a few min."""
    now = dt_util.now()
    return [(now + timedelta(seconds=i * 30), power) for i in range(count)]


def _match_result() -> MatchResult:
    return MatchResult(
        best_profile=PROFILE,
        confidence=0.75,
        expected_duration=3600.0,
        matched_phase=None,
        candidates=[{"name": PROFILE, "score": 0.75}],
        is_ambiguous=False,
        ambiguity_margin=0.0,
    )


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_issue_306"
    entry.title = "Test Dishwasher"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)

    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})

        # A matched, committed cycle now sitting in a low-power pause.
        mgr.detector.matched_profile = PROFILE
        mgr.detector._verified_pause = True
        mgr.detector.state = STATE_PAUSED
        mgr.detector.config.stop_threshold_w = 5.0
        mgr.detector.get_elapsed_seconds = MagicMock(return_value=1800.0)
        mgr.detector.get_power_trace = MagicMock(return_value=[])
        mgr.detector.is_waiting_low_power = MagicMock(return_value=True)
        mgr.detector.set_verified_pause = MagicMock()
        mgr.detector.update_match = MagicMock()

        mgr.profile_store.async_match_profile = AsyncMock(return_value=_match_result())
        # Envelope check reports UNEXPECTED low power (the trigger that, pre-fix,
        # cleared the user's pause).
        mgr.profile_store.async_verify_alignment = AsyncMock(
            return_value=(False, 0.0, None)
        )
        mgr.profile_store.get_profile = MagicMock(
            return_value={"avg_duration": 3600.0}
        )

        # Program already committed -> skip the ML live-match feature block and
        # keep the switching logic in the "same program" branch.
        mgr._current_program = PROFILE
        mgr._matched_profile_duration = 3600.0
        mgr._notified_start = True

        # Keep the post-match tail quiet so the assertion is deterministic.
        mgr._update_remaining_only = MagicMock()
        mgr._check_live_progress_notification = MagicMock()
        mgr._notify_update = MagicMock()

        return mgr


def _last_verified_pause_call(mgr: WashDataManager) -> bool:
    assert mgr.detector.set_verified_pause.called, "set_verified_pause never called"
    return bool(mgr.detector.set_verified_pause.call_args.args[0])


@pytest.mark.asyncio
async def test_user_pause_not_cleared_by_matcher(manager: WashDataManager) -> None:
    """While user-paused, matching must keep verified_pause=True (issue #306)."""
    manager._is_user_paused = True
    manager._user_pause_start = dt_util.now()

    await manager._async_do_perform_matching(_low_power_readings())

    assert _last_verified_pause_call(manager) is True
    # The envelope alignment must be skipped entirely while user-paused.
    manager.profile_store.async_verify_alignment.assert_not_called()


@pytest.mark.asyncio
async def test_user_pause_survives_repeated_matches(manager: WashDataManager) -> None:
    """Every periodic match during a user pause keeps the pause protected."""
    manager._is_user_paused = True
    manager._user_pause_start = dt_util.now()

    for _ in range(4):
        await manager._async_do_perform_matching(_low_power_readings())
        # Simulate the detector honouring the last write on the next tick.
        manager.detector._verified_pause = _last_verified_pause_call(manager)

    assert manager.detector._verified_pause is True
    manager.profile_store.async_verify_alignment.assert_not_called()


@pytest.mark.asyncio
async def test_auto_pause_still_cleared_on_mismatch(manager: WashDataManager) -> None:
    """Control: with no user pause, an envelope mismatch still clears the pause.

    Proves the fix is scoped to user pauses and does not change auto-detected
    pause/mismatch behaviour.
    """
    manager._is_user_paused = False

    await manager._async_do_perform_matching(_low_power_readings())

    assert _last_verified_pause_call(manager) is False
    manager.profile_store.async_verify_alignment.assert_called()
