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
"""Discussion #411: picking a program on an idle appliance did nothing, silently.

``set_manual_program`` returned early unless the detector was in exactly
``running``. The panel offers the dropdown at all times, so the common case
(pick a program on an idle machine) set nothing, logged nothing, and still
reported success to the caller because the method returned ``None`` either way.
The panel then refreshed, read ``manual_program: false`` / ``current_program:
null`` and snapped the dropdown back to auto-detect. Two reporters saw exactly
that; one captured the WebSocket exchange showing the success reply.

The chosen behaviour is pre-arming: a program picked while nothing is running is
remembered and applied the moment the next cycle starts.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import (
    STATE_ENDING,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.manager import WashDataManager

PROGRAM = "Normal - Warm - Medium"


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_discussion_411"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power", "device_type": "washing_machine"}
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
        mgr.profile_store.get_profiles = MagicMock(
            return_value={PROGRAM: {"avg_duration": 3600.0}}
        )
        mgr.profile_store.async_set_armed_program = AsyncMock()
        mgr.profile_store.get_armed_program = MagicMock(return_value=None)
        mgr._notify_update = MagicMock()
        mgr._update_estimates = MagicMock()
        mgr.detector.state = STATE_OFF
        return mgr


def _armed_writes(manager: WashDataManager) -> list[Any]:
    """The values handed to the store's armed-program setter, in order."""
    return [c.args[0] for c in manager.profile_store.async_set_armed_program.call_args_list]


# ---------------------------------------------------------------------------
# The reported bug
# ---------------------------------------------------------------------------


def test_picking_a_program_while_off_is_accepted(manager: WashDataManager) -> None:
    """The reporter's case: appliance Off, dropdown used, nothing happened."""
    assert manager.set_manual_program(PROGRAM) is True
    assert manager.armed_program == PROGRAM


def test_picking_a_program_while_off_does_not_claim_a_live_pin(
    manager: WashDataManager,
) -> None:
    """Armed is not the same as running: the status card must not claim a match."""
    manager.set_manual_program(PROGRAM)
    assert manager.manual_program_active is False
    assert manager.current_program == "off"


def test_a_missing_program_is_the_only_rejection_left(manager: WashDataManager) -> None:
    """And it is now reported, rather than returning the same silent None."""
    manager.profile_store.get_profiles = MagicMock(return_value={})
    assert manager.set_manual_program("Ghost") is False
    assert manager.armed_program is None


def test_the_arm_is_persisted_so_it_survives_a_restart(manager: WashDataManager) -> None:
    """Arming happens while idle, so the wait can easily span a restart."""
    manager.set_manual_program(PROGRAM)
    assert _armed_writes(manager) == [PROGRAM]


# ---------------------------------------------------------------------------
# Applying the arm
# ---------------------------------------------------------------------------


def test_the_armed_program_is_applied_when_the_cycle_starts(
    manager: WashDataManager,
) -> None:
    manager.set_manual_program(PROGRAM)
    manager.detector.state = STATE_RUNNING
    assert manager._consume_armed_program() is True
    assert manager.manual_program_active is True
    assert manager.current_program == PROGRAM
    assert manager._matched_profile_duration == 3600.0


def test_the_arm_is_consumed_not_sticky(manager: WashDataManager) -> None:
    """It pins the next cycle, not every future one."""
    manager.set_manual_program(PROGRAM)
    manager.detector.state = STATE_RUNNING
    manager._consume_armed_program()
    assert manager.armed_program is None
    assert _armed_writes(manager) == [PROGRAM, None]


def test_consuming_nothing_is_a_no_op(manager: WashDataManager) -> None:
    assert manager._consume_armed_program() is False
    assert manager.manual_program_active is False


def test_a_program_deleted_before_the_cycle_started_is_dropped(
    manager: WashDataManager,
) -> None:
    manager.set_manual_program(PROGRAM)
    manager.profile_store.get_profiles = MagicMock(return_value={})
    manager.detector.state = STATE_RUNNING
    assert manager._consume_armed_program() is False
    assert manager.manual_program_active is False
    assert manager.armed_program is None


# ---------------------------------------------------------------------------
# States other than "running"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", [STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING])
def test_a_cycle_in_progress_is_pinned_immediately(
    manager: WashDataManager, state: str
) -> None:
    """All four in-progress states, not just `running` as before."""
    manager.detector.state = state
    assert manager.set_manual_program(PROGRAM) is True
    assert manager.manual_program_active is True
    assert manager.current_program == PROGRAM


def test_a_pin_made_during_starting_survives_the_run_transition(
    manager: WashDataManager,
) -> None:
    """The new-cycle reset clears the live pin, so the arm has to restore it.

    Picking a program right after switching the appliance on is the natural
    moment to do it, and STARTING -> RUNNING would otherwise discard the choice.
    """
    manager.detector.state = STATE_STARTING
    manager.set_manual_program(PROGRAM)
    # what the reset at the top of a new cycle does
    manager._current_program = "detecting..."
    manager._manual_program_active = False
    manager._matched_profile_duration = None
    manager.detector.state = STATE_RUNNING

    assert manager._consume_armed_program() is True
    assert manager.current_program == PROGRAM
    assert manager._matched_profile_duration == 3600.0


# ---------------------------------------------------------------------------
# Clearing
# ---------------------------------------------------------------------------


def test_auto_detect_clears_an_arm_made_while_idle(manager: WashDataManager) -> None:
    """Previously impossible: clear bailed out unless a pin was live."""
    manager.set_manual_program(PROGRAM)
    manager.clear_manual_program()
    assert manager.armed_program is None
    assert _armed_writes(manager) == [PROGRAM, None]


def test_auto_detect_still_clears_a_live_pin(manager: WashDataManager) -> None:
    manager.detector.state = STATE_RUNNING
    manager.set_manual_program(PROGRAM)
    manager.clear_manual_program()
    assert manager.manual_program_active is False
    assert manager.current_program == "detecting..."


def test_clearing_nothing_stays_a_no_op(manager: WashDataManager) -> None:
    manager.clear_manual_program()
    assert manager.armed_program is None
    assert _armed_writes(manager) == []


# ---------------------------------------------------------------------------
# The store side: an arm has to outlive a restart
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    from custom_components.ha_washdata.profile_store import ProfileStore

    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


async def test_store_round_trips_the_armed_program(store) -> None:
    assert store.get_armed_program() is None
    await store.async_set_armed_program(PROGRAM)
    assert store.get_armed_program() == PROGRAM
    await store.async_set_armed_program(None)
    assert store.get_armed_program() is None


def test_store_getter_ignores_junk(store) -> None:
    store._data["armed_program"] = 42
    assert store.get_armed_program() is None
    store._data["armed_program"] = ""
    assert store.get_armed_program() is None


async def test_wiping_all_data_clears_the_arm(store) -> None:
    await store.async_set_armed_program(PROGRAM)
    await store.clear_all_data()
    assert store.get_armed_program() is None


# ---------------------------------------------------------------------------
# The WebSocket command must stop reporting success it did not achieve
# ---------------------------------------------------------------------------


def _conn():
    c = MagicMock()
    c.send_result = MagicMock()
    c.send_error = MagicMock()
    return c


def test_ws_set_program_reports_a_rejection() -> None:
    """The reporter captured {"success": true} for a write that never happened."""
    from custom_components.ha_washdata import ws_api

    hass = MagicMock()
    manager = MagicMock()
    manager.set_manual_program = MagicMock(return_value=False)
    conn = _conn()
    with patch.object(ws_api, "_get_manager", return_value=manager):
        ws_api.ws_set_program(
            hass, conn, {"id": 1, "entry_id": "e", "program": "Ghost"}
        )
    conn.send_result.assert_not_called()
    conn.send_error.assert_called_once()
    assert conn.send_error.call_args.args[1] == "not_found"


def test_ws_set_program_still_reports_success_when_accepted() -> None:
    from custom_components.ha_washdata import ws_api

    hass = MagicMock()
    manager = MagicMock()
    manager.set_manual_program = MagicMock(return_value=True)
    conn = _conn()
    with patch.object(ws_api, "_get_manager", return_value=manager):
        ws_api.ws_set_program(
            hass, conn, {"id": 1, "entry_id": "e", "program": PROGRAM}
        )
    conn.send_error.assert_not_called()
    manager.notify_update.assert_called_once()
