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
"""#479: the live update text next to the Android chronometer.

The live text could only say "{minutes} left", so next to the chronometer the phone
showed two different countdowns, and a custom text with any other variable silently
fell back to the default (it looked as if the setting were ignored). Live updates
also went out at normal priority, which Android batches while the phone dozes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import NOTIFY_EVENT_LIVE
from custom_components.ha_washdata.manager import WashDataManager

from .real_manager import boot, make_entry, record_notify

NOW = datetime(2026, 10, 19, 14, 3, 20, tzinfo=timezone.utc)


async def _manager(hass: Any, message: str | None = None) -> WashDataManager:
    options: dict[str, Any] = {
        "notify_live_services": ["notify.mobile_app_phone"],
        "notify_live_chronometer": True,
    }
    if message is not None:
        options["notify_pre_complete_message"] = message
    mgr = await boot(hass, make_entry(hass, options, title="Washing Machine"))
    mgr.detector._state = "running"
    mgr._notify_live_interval_seconds = 30
    mgr._matched_profile_duration = 3360.0
    mgr._total_duration = 3360.0
    mgr._time_remaining = 1560.0
    mgr._last_live_notification_time = None
    return mgr


def _send_live(mgr: WashDataManager) -> tuple[str, dict[str, Any]]:
    mgr._dispatch_notification = MagicMock()
    with patch("custom_components.ha_washdata.manager.utc_now", return_value=NOW), \
            patch.object(mgr.detector, "get_elapsed_seconds", return_value=1800.0):
        mgr._check_live_progress_notification()
    args, kwargs = mgr._dispatch_notification.call_args
    assert kwargs["event_type"] == NOTIFY_EVENT_LIVE
    return args[0], kwargs["extra_vars"]


async def test_finish_time_placeholder_is_the_local_estimated_finish(hass: Any) -> None:
    mgr = await _manager(hass, "{device}: done at {finish_time}")
    msg, extra = _send_live(mgr)
    expected = dt_util.as_local(NOW + timedelta(seconds=1560)).strftime("%H:%M")
    assert msg == f"Washing Machine: done at {expected}"
    # The chronometer counts down to the same moment the text names.
    assert extra["when"] == int(NOW.timestamp()) + 1560


async def test_default_text_is_unchanged(hass: Any) -> None:
    msg, _ = _send_live(await _manager(hass))
    assert msg == "Washing Machine: Less than 26 minutes remaining."


async def test_live_update_is_sent_at_high_priority_with_zero_ttl(hass: Any) -> None:
    _, extra = _send_live(await _manager(hass))
    assert extra["priority"] == "high"
    assert extra["ttl"] == 0


async def test_priority_and_ttl_reach_the_mobile_app_payload(hass: Any) -> None:
    calls = record_notify(hass)
    mgr = await _manager(hass)
    extra: dict[str, Any] = {"tag": "ha_washdata_live", "progress": 1, "progress_max": 2}
    mgr._apply_live_notification_prefs(extra)
    mgr._dispatch_notification("live", event_type=NOTIFY_EVENT_LIVE, extra_vars=extra)
    await hass.async_block_till_done()
    assert len(calls) == 1
    assert calls[0]["data"]["priority"] == "high"
    assert calls[0]["data"]["ttl"] == 0


async def test_unknown_variable_warns_once_and_falls_back(hass: Any) -> None:
    mgr = await _manager(hass, "{device}: ready at {eta}")
    mgr._logger = MagicMock()
    msg, _ = _send_live(mgr)
    assert msg == "Washing Machine: Less than 26 minutes remaining."
    assert mgr._logger.warning.call_count == 1
    warning = mgr._logger.warning.call_args[0]
    assert "{eta}" in warning[1]
    assert "{finish_time}" in warning[-1]

    mgr._last_live_notification_time = None
    _send_live(mgr)
    assert mgr._logger.warning.call_count == 1
