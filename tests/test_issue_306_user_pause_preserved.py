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
stay in force until the user explicitly resumes.  Two separate bugs were fixed:

1. During a RUNNING cycle the periodic profile matcher re-judged the low-power
   reading with the envelope-alignment heuristic and, on a mismatch, cleared the
   detector's ``verified_pause`` flag.  With the flag gone the cycle was finalized
   on the next ENDING timeout (for a dishwasher, the 1 h ``min_off_gap``), so the
   State sensor left "Paused by user" (surfacing as Interrupted / Finished) while
   the user still expected it paused.

2. Pausing during STARTING state had two sub-issues: (a) without a restart, the
   false-start abort fired unconditionally when power dropped, silently discarding
   the cycle; (b) after an HA restart, the restore block skipped ``STATE_STARTING``
   so ``_is_user_paused`` and ``verified_pause`` were never re-applied, causing the
   cycle to be aborted on the first low-power reading.

These tests cover all three invariants:
- Matcher never clears verified_pause while user-paused (RUNNING path).
- False-start abort is skipped when verified_pause is set (STARTING path).
- Restore promotes a user-paused STARTING snapshot to PAUSED so the existing
  user-pause re-assertion in the restore block fires correctly.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import STATE_PAUSED, STATE_STARTING, STATE_OFF
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
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


# ---------------------------------------------------------------------------
# Fix C: STARTING-state false-start must respect _verified_pause (issue #306)
# ---------------------------------------------------------------------------

def _starting_detector() -> CycleDetector:
    """Real CycleDetector configured for quick STARTING entry."""
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        start_duration_threshold=2.0,
        start_energy_threshold=0.001,
    )
    return CycleDetector(config, MagicMock(), MagicMock())


def test_user_pause_during_starting_not_aborted_as_false_start() -> None:
    """Pausing during STARTING must not trigger the false-start abort (issue #306).

    Before the fix the false-start check fired unconditionally when power dropped
    while in STARTING, even if the user had explicitly paused the cycle.  The
    cycle was silently discarded and the next low-power reading transitioned the
    detector to OFF.
    """
    detector = _starting_detector()
    now = dt_util.now()

    # One reading above threshold is enough to enter STARTING.
    detector.process_reading(200.0, now)
    assert detector.state == STATE_STARTING

    # User pauses the appliance: verified_pause is set to True.
    detector.set_verified_pause(True)

    # Power drops (appliance paused) -- sustained well beyond the 1 s grace period.
    detector.process_reading(0.0, now + timedelta(seconds=2))
    detector.process_reading(0.0, now + timedelta(seconds=5))
    detector.process_reading(0.0, now + timedelta(seconds=10))

    # The cycle must still be in STARTING, not aborted to OFF.
    assert detector.state == STATE_STARTING, (
        f"False-start abort fired despite verified_pause; got state={detector.state!r}"
    )


def test_false_start_still_fires_without_verified_pause() -> None:
    """Control: without verified_pause, a power drop in STARTING still aborts correctly."""
    detector = _starting_detector()
    now = dt_util.now()

    detector.process_reading(200.0, now)
    assert detector.state == STATE_STARTING

    # No user pause -- power drop is a genuine false start.
    detector.process_reading(0.0, now + timedelta(seconds=2))
    detector.process_reading(0.0, now + timedelta(seconds=5))

    assert detector.state == STATE_OFF, (
        f"Expected false-start to abort to OFF; got state={detector.state!r}"
    )


def test_paused_starting_falls_to_off_on_sustained_true_off() -> None:
    """A user-paused STARTING must not stay pinned forever if the machine is
    actually switched off (audit item: STARTING has no timeout).

    verified_pause holds STARTING through a normal pause, but sustained power
    below the stop threshold means the appliance was switched off rather than
    resumed, so after STARTING_PAUSED_TRUE_OFF_TIMEOUT_SECONDS the detector must
    fall back to OFF.
    """
    from custom_components.ha_washdata.const import (
        STARTING_PAUSED_TRUE_OFF_TIMEOUT_SECONDS,
    )

    detector = _starting_detector()
    now = dt_util.now()

    detector.process_reading(200.0, now)
    assert detector.state == STATE_STARTING
    detector.set_verified_pause(True)

    # Short pause with power off - still held (below the timeout).
    detector.process_reading(0.0, now + timedelta(seconds=10))
    assert detector.state == STATE_STARTING

    # Sustained true-off beyond the timeout: the machine was switched off.
    detector.process_reading(
        0.0, now + timedelta(seconds=STARTING_PAUSED_TRUE_OFF_TIMEOUT_SECONDS + 30)
    )
    assert detector.state == STATE_OFF, (
        "Paused STARTING should fall to OFF after sustained true-off; "
        f"got state={detector.state!r}"
    )


def test_paused_starting_standby_power_holds() -> None:
    """Control: a paused STARTING with standby power in the band *above* the stop
    threshold but *below* the start threshold keeps holding (a genuine pause),
    never falling to OFF on the true-off path."""
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        start_duration_threshold=2.0,
        start_energy_threshold=0.001,
        start_threshold_w=10.0,
        stop_threshold_w=2.0,
    )
    detector = CycleDetector(config, MagicMock(), MagicMock())
    now = dt_util.now()

    detector.process_reading(200.0, now)
    assert detector.state == STATE_STARTING
    detector.set_verified_pause(True)

    # Standby power (5 W) sits above stop_threshold_w (2) but below start_threshold_w
    # (10) for a long time - a genuine pause, not true-off. Must hold STARTING.
    for minutes in range(1, 12):
        detector.process_reading(5.0, now + timedelta(minutes=minutes))
    assert detector.state == STATE_STARTING, (
        f"Genuine standby pause must hold STARTING; got state={detector.state!r}"
    )


# ---------------------------------------------------------------------------
# Fix D: manager restore promotes STARTING+user_paused snapshot to PAUSED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_starting_snapshot_promoted_to_paused_on_restore(
    manager: WashDataManager,
) -> None:
    """Restore promotes a user-paused STARTING snapshot to PAUSED (issue #306).

    Before the fix the restore block's condition excluded STATE_STARTING, so
    _is_user_paused was never re-applied and verified_pause was never re-asserted.
    The first low-power reading then fired the false-start abort and the cycle
    became Off after an HA restart.
    """
    now = dt_util.now()
    starting_snap = {
        "state": STATE_STARTING,
        "is_user_paused": True,
        "user_pause_start": now.isoformat(),
        "total_user_paused_seconds": 0.0,
        "manual_program": False,
        "notified_start": False,
        "start_event_fired": False,
        "current_cycle_start": now.isoformat(),
        "power_readings": [],
        "accumulated_energy_wh": 0.0,
        "time_above": 3.0,
        "time_below": 2.5,
        "cycle_max_power": 210.0,
        "last_active_time": None,
        "expected_duration": 0.0,
        "matched_profile": None,
        "state_enter_time": now.isoformat(),
        "end_spike_seen": False,
        "end_spike_duration": 0.0,
        "match_ambiguous": False,
        "match_prefix_ambiguous": False,
        "ml_defer_start_duration": None,
        "sub_state": "Restored",
        "dynamic_min_duration": None,
    }

    # Set up the profile store so the restore path is triggered.
    manager.profile_store.get_active_cycle = MagicMock(return_value=starting_snap)
    manager.profile_store.get_last_active_save = MagicMock(return_value=now)

    # Power sensor returns a low (paused) reading so the viability check passes.
    manager.hass.states.async_set("sensor.test_power", "1.0")

    # The detector mock needs a realistic post-restore state.
    manager.detector.state = STATE_PAUSED  # promoted by the manager
    manager.detector.matched_profile = None

    await manager._attempt_state_restoration()

    # The snapshot passed to restore_state_snapshot must have state=paused, not starting.
    restore_calls = manager.detector.restore_state_snapshot.call_args_list
    assert restore_calls, "restore_state_snapshot was never called"
    restored_snap = restore_calls[0].args[0]
    assert restored_snap.get("state") == STATE_PAUSED, (
        f"Expected promoted state=paused; got {restored_snap.get('state')!r}"
    )

    # And the manager must have re-applied the user-pause state.
    assert manager._is_user_paused is True
    manager.detector.set_verified_pause.assert_called_with(True)
