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

"""Issue #437: live notification showed 100 % / "less than 1 minute" after a match.

`has_profile_match` only looked at `_matched_profile_duration`. On the tick that
accepts a match, `_update_remaining_only` may have been throttled out (one estimate
per 5 s), so that attribute is set while `_time_remaining` is still None. The
progress branch then read `float(self._time_remaining or 0.0)` as remaining 0,
which makes elapsed == total (a full bar) and minutes_left == 1 - the pre-complete
wording, at the START of a 4 h cycle.
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import STATE_RUNNING
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata import progress as progress_mod


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.components.persistent_notification.async_create = MagicMock()
    hass.config_entries.async_get_entry = MagicMock()
    hass.states.get = MagicMock(return_value=MagicMock(state="home"))
    return hass


@pytest.fixture
def manager(mock_hass: Any) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e437"
    entry.title = "Dishwasher"
    entry.options = {
        "power_sensor": "sensor.p",
        "notify_live_services": ["notify.mobile_app_test"],
    }
    entry.data = {}
    mock_hass.config_entries.async_get_entry.return_value = entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(mock_hass, entry)
    mgr._notify_live_services = ["notify.mobile_app_test"]
    mgr.detector.state = STATE_RUNNING
    mgr.detector.get_elapsed_seconds = MagicMock(return_value=3000.0)
    mgr.profile_store.has_real_profiles = True
    # The reporter's cycle: 4 h 13 min ECO, ~50 min in when the match landed.
    mgr._matched_profile_duration = 15180.0
    mgr._current_program = "ECO + Extra Dry"
    # The waiting notification has already gone out for this cycle.
    mgr._live_waiting_notification_sent = True
    mgr._dispatch_notification = MagicMock(return_value=True)
    return mgr


def test_match_without_estimate_does_not_send_a_100_percent_update(
    manager: WashDataManager,
) -> None:
    """The tick where the match lands but no ETA exists yet must stay silent."""
    manager._time_remaining = None

    manager._check_live_progress_notification()

    manager._dispatch_notification.assert_not_called()
    # The waiting latch is intact, so the waiting message is not re-sent either.
    assert manager._live_waiting_notification_sent is True


def test_estimate_present_sends_the_real_countdown(manager: WashDataManager) -> None:
    """Once an estimate exists the progress branch reports the true remaining time."""
    manager._time_remaining = 12060.0   # 201 min, as the dashboard showed
    manager._total_duration = 15180.0

    manager._check_live_progress_notification()

    manager._dispatch_notification.assert_called_once()
    ev = manager._dispatch_notification.call_args.kwargs["extra_vars"]
    assert ev["time_remaining_seconds"] == 12060
    assert ev["minutes_left"] == 201
    assert ev["progress"] < ev["progress_max"]
    assert ev["progress"] == 15180 - 12060


def test_waiting_latch_still_resets_for_a_restarted_cycle(
    manager: WashDataManager,
) -> None:
    """A real estimate clears the latch so a later match-less phase waits again."""
    manager._time_remaining = 600.0
    manager._total_duration = 15180.0

    manager._check_live_progress_notification()

    assert manager._live_waiting_notification_sent is False


def test_first_estimate_after_a_match_bypasses_the_5s_throttle(
    manager: WashDataManager,
) -> None:
    """Otherwise the live card holds the waiting message for a whole interval."""
    now = datetime.now(timezone.utc)
    manager._time_remaining = None
    manager._last_phase_estimate_time = now - timedelta(seconds=1)  # throttled
    manager.detector.get_power_trace = MagicMock(return_value=[])
    manager._ml_progress_percent = MagicMock(return_value=None)
    result = progress_mod.ProgressResult(20.5, 20.5, 12060.0, 15060.0, None, "linear")

    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=now
    ), patch.object(progress_mod, "compute_progress", return_value=result):
        manager._update_remaining_only()

    assert manager._time_remaining == 12060.0


def test_throttle_still_applies_once_an_estimate_exists(
    manager: WashDataManager,
) -> None:
    """The bypass is one-shot per match, not a removal of the throttle."""
    now = datetime.now(timezone.utc)
    manager._time_remaining = 12060.0
    manager._last_phase_estimate_time = now - timedelta(seconds=1)
    manager.detector.get_power_trace = MagicMock(return_value=[])

    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=now
    ), patch.object(progress_mod, "compute_progress") as cp:
        manager._update_remaining_only()

    cp.assert_not_called()
