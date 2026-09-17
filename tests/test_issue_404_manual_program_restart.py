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
"""Issue #404 (secondary bug): a manual program override must survive an HA restart.

`_manual_program_active` was already restored, but `_matched_profile_duration`
was not, so on the next post-restart match tick the manual matcher tuple fed
`expected_duration=0`, the detector's `matched_profile` was wiped, and the cycle
dropped to "detecting..." with `expected == 0` - i.e. straight into the unmatched
watchdog guard even though the user had picked a program with a known duration.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import STATE_RUNNING
from custom_components.ha_washdata.manager import WashDataManager

PROGRAM = "Cottons 60 + Dry"


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_issue_404_manual"
    entry.title = "Test Washer-Dryer"
    entry.options = {"power_sensor": "sensor.test_power", "device_type": "washer_dryer"}
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
        mgr.profile_store.get_past_cycles = MagicMock(return_value=[])
        mgr.profile_store.async_clear_active_cycle = AsyncMock()
        mgr.profile_store.get_last_active_save = MagicMock(return_value=dt_util.now())
        # Isolate the restore from the watchdog / entity plumbing.
        mgr._start_watchdog = MagicMock()
        mgr._maybe_arm_door_end_dwell_if_open = MagicMock()
        mgr._notify_update = MagicMock()
        return mgr


def _snapshot(*, manual_program: bool, manual_name: str | None) -> dict[str, Any]:
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
        "expected_duration": 0.0,
        "matched_profile": None,
        "state_enter_time": now.isoformat(),
        "manual_program": manual_program,
        "manual_program_name": manual_name,
        "notified_start": True,
        "start_event_fired": True,
        "is_user_paused": False,
        "user_pause_start": None,
        "total_user_paused_seconds": 0.0,
    }


async def _restore(mgr: WashDataManager, snap: dict[str, Any], matched: str | None) -> None:
    mgr.profile_store.get_active_cycle = MagicMock(return_value=snap)
    # restore_state_snapshot is a mock, so simulate the detector's post-restore state.
    mgr.detector.state = STATE_RUNNING
    mgr.detector.matched_profile = matched
    await mgr._attempt_state_restoration()


@pytest.mark.asyncio
async def test_augment_snapshot_records_manual_program_name(
    manager: WashDataManager,
) -> None:
    manager._manual_program_active = True
    manager._current_program = PROGRAM
    snap = manager._augment_active_snapshot({})
    assert snap["manual_program"] is True
    assert snap["manual_program_name"] == PROGRAM

    manager._manual_program_active = False
    snap2 = manager._augment_active_snapshot({})
    assert snap2["manual_program_name"] is None


@pytest.mark.asyncio
async def test_manual_program_with_duration_survives_restart(
    manager: WashDataManager,
) -> None:
    """The core #404 secondary bug: name AND duration are re-pinned after restart."""
    manager.profile_store.get_profile = MagicMock(return_value={"avg_duration": 23400.0})
    # The detector wiped its matched_profile on the way down (expected==0 tick),
    # so it comes back None - the manual name must repair that.
    await _restore(manager, _snapshot(manual_program=True, manual_name=PROGRAM), matched=None)

    assert manager._manual_program_active is True
    assert manager._current_program == PROGRAM
    assert manager._matched_profile_duration == 23400.0


@pytest.mark.asyncio
async def test_empty_manual_profile_stays_active_with_none_duration(
    manager: WashDataManager,
) -> None:
    """An empty hand-created profile (avg_duration 0) stays selected, duration None."""
    manager.profile_store.get_profile = MagicMock(return_value={"avg_duration": 0.0})
    await _restore(
        manager, _snapshot(manual_program=True, manual_name="New Empty Profile"), matched=None
    )

    assert manager._manual_program_active is True
    assert manager._current_program == "New Empty Profile"
    assert manager._matched_profile_duration is None


@pytest.mark.asyncio
async def test_deleted_manual_profile_reverts_to_auto(
    manager: WashDataManager,
) -> None:
    """If the chosen profile was deleted while HA was down, fall back to auto-detect."""
    manager.profile_store.get_profile = MagicMock(return_value=None)
    await _restore(
        manager, _snapshot(manual_program=True, manual_name="Gone"), matched=None
    )

    assert manager._manual_program_active is False


@pytest.mark.asyncio
async def test_auto_detect_cycle_restore_unaffected(
    manager: WashDataManager,
) -> None:
    """Control: a non-manual cycle restores its detector-matched program as before."""
    manager.profile_store.get_profile = MagicMock(return_value={"avg_duration": 3600.0})
    await _restore(
        manager, _snapshot(manual_program=False, manual_name=None), matched="Eco 50"
    )

    assert manager._manual_program_active is False
    assert manager._current_program == "Eco 50"
