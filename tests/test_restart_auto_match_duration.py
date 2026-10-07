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
"""An auto-detected program must survive an HA restart with its expected duration.

#404 re-armed only a MANUAL program. An auto-matched one came back as the
detector's match - the last tick's raw winner, not necessarily the program on
display - with ``_matched_profile_duration`` None, so the ETA went blank and the
first tick another candidate led sent ``decide_switch`` down its no-duration
branch to "detecting...". Seen on a washer on 2026-10-06: committed
"30° / 1:07 / utility" at 31 min, restart at 34.5 min, ETA blank, back to
"detecting..." at 37 min until a fresh commit at 45 min.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import STATE_RUNNING
from custom_components.ha_washdata.manager import (
    WashDataManager,
    _read_switch_state,
    _write_switch_state,
)

SHOWN = "30° / 1:07 / utility"
RAW = "60° / 1:17 / 1000"
PROFILES = {SHOWN: {"avg_duration": 9600.0}, RAW: {"avg_duration": 3300.0}}


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "test_restart_auto_match"
    entry.title = "Washer"
    entry.options = {"power_sensor": "sensor.test_power", "device_type": "washing_machine"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    mgr.profile_store.get_past_cycles = MagicMock(return_value=[])
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mgr.profile_store.get_last_active_save = MagicMock(return_value=dt_util.now())
    mgr.profile_store.get_profile = MagicMock(side_effect=PROFILES.get)
    mgr._start_watchdog = MagicMock()
    mgr._maybe_arm_door_end_dwell_if_open = MagicMock()
    mgr._notify_update = MagicMock()
    return mgr


def _snapshot(**manager_fields: Any) -> dict[str, Any]:
    now = dt_util.now()
    return {
        "state": STATE_RUNNING,
        "sub_state": "Restored",
        "current_cycle_start": now.isoformat(),
        "power_readings": [],
        "accumulated_energy_wh": 0.0,
        "time_above": 60.0,
        "time_below": 0.0,
        "cycle_max_power": 1500.0,
        "last_active_time": now.isoformat(),
        "expected_duration": 3300.0,
        "matched_profile": RAW,
        "state_enter_time": now.isoformat(),
        "manual_program": False,
        "manual_program_name": None,
        "notified_start": True,
        "start_event_fired": True,
        "is_user_paused": False,
        "user_pause_start": None,
        "total_user_paused_seconds": 0.0,
        **manager_fields,
    }


async def _restore(mgr: WashDataManager, snap: dict[str, Any]) -> None:
    mgr.profile_store.get_active_cycle = MagicMock(return_value=snap)
    # restore_state_snapshot is mocked: the detector comes back holding the raw
    # winner of the last tick before the restart.
    mgr.detector.state = STATE_RUNNING
    mgr.detector.matched_profile = snap.get("matched_profile")
    mgr.detector.expected_duration_seconds = snap.get("expected_duration")
    await mgr._attempt_state_restoration()


def _tick(mgr: WashDataManager, winner: str, scores: dict[str, float]) -> None:
    """One match tick through the manager's own switching rules."""
    result = SimpleNamespace(
        best_profile=winner,
        confidence=scores[winner],
        candidates=[{"name": n, "score": s} for n, s in scores.items()],
        is_ambiguous=True,
        expected_duration=PROFILES[winner]["avg_duration"],
        matched_phase=None,
        member_confidence=None,
    )
    state = _read_switch_state(mgr)
    tick = match_rules.begin_tick(state, result, 3, 2300.0)
    _write_switch_state(mgr, state, tick.log)
    state = _read_switch_state(mgr)
    log = match_rules.decide_switch(state, tick, result, 3, 0.35)
    _write_switch_state(mgr, state, log)


@pytest.mark.asyncio
async def test_snapshot_records_committed_program_and_duration(
    manager: WashDataManager,
) -> None:
    manager._current_program = SHOWN
    manager._matched_profile_duration = 9600.0
    snap = manager._augment_active_snapshot({})
    assert snap["committed_program"] == SHOWN
    assert snap["committed_program_duration"] == 9600.0

    manager._current_program = "detecting..."
    snap = manager._augment_active_snapshot({})
    assert snap["committed_program"] is None
    assert snap["committed_program_duration"] is None

    # A hand-picked program is carried by manual_program_name (#404), not here.
    manager._current_program = SHOWN
    manager._manual_program_active = True
    snap = manager._augment_active_snapshot({})
    assert snap["committed_program"] is None
    assert snap["manual_program_name"] == SHOWN


@pytest.mark.asyncio
async def test_restore_brings_back_displayed_program_with_duration(
    manager: WashDataManager,
) -> None:
    await _restore(
        manager,
        _snapshot(committed_program=SHOWN, committed_program_duration=9600.0),
    )
    assert manager._current_program == SHOWN  # not the detector's raw winner
    assert manager._matched_profile_duration == 9600.0


@pytest.mark.asyncio
async def test_one_tick_lead_after_restore_keeps_the_program(
    manager: WashDataManager,
) -> None:
    """The observed symptom: the first off-leader tick reverted to detecting."""
    await _restore(
        manager,
        _snapshot(committed_program=SHOWN, committed_program_duration=9600.0),
    )
    _tick(manager, RAW, {RAW: 0.690, SHOWN: 0.685})
    assert manager._current_program == SHOWN
    assert manager._matched_profile_duration == 9600.0


@pytest.mark.asyncio
async def test_restore_without_saved_duration_uses_profile_average(
    manager: WashDataManager,
) -> None:
    await _restore(
        manager,
        _snapshot(committed_program=SHOWN, committed_program_duration=None),
    )
    assert manager._current_program == SHOWN
    assert manager._matched_profile_duration == 9600.0


@pytest.mark.asyncio
async def test_older_snapshot_restores_detector_match_with_duration(
    manager: WashDataManager,
) -> None:
    """No committed_program key: the detector's match and expected duration."""
    await _restore(manager, _snapshot())
    assert manager._current_program == RAW
    assert manager._matched_profile_duration == 3300.0


@pytest.mark.asyncio
async def test_committed_profile_deleted_while_down_restores_detecting(
    manager: WashDataManager,
) -> None:
    await _restore(
        manager,
        _snapshot(committed_program="Gone", committed_program_duration=5000.0),
    )
    assert manager._current_program == "detecting..."
    assert manager._matched_profile_duration is None


@pytest.mark.asyncio
async def test_nothing_committed_restores_detecting(manager: WashDataManager) -> None:
    await _restore(
        manager,
        _snapshot(committed_program=None, committed_program_duration=None),
    )
    assert manager._current_program == "detecting..."
    assert manager._matched_profile_duration is None
