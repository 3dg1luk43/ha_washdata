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
"""Audit MANAGER-10: an interrupted cycle is not a finished wash.

A 3.5 min false start used to send "Washer finished. Duration: 3m" and, with a
door sensor or #451's unload confirmation, enter the Clean state and nag "ready
to unload". Decision (2026-10-04): no finished push and no Clean state for
``interrupted``; ``force_stopped`` keeps both, and the finish template gains
``{status}``. One pure predicate (``notification_rules.cycle_end_is_finish``)
decides it for the manager and the Playground's markers.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import notification_rules as notif_rules
from custom_components.ha_washdata import playground
from custom_components.ha_washdata.const import (
    CONF_NOTIFY_FINISH_MESSAGE,
    CONF_NOTIFY_FINISH_SERVICES,
    CONF_UNLOAD_TRACK_WITHOUT_DOOR,
    NOTIFY_EVENT_FINISH,
)
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.manager import WashDataManager

from .real_manager import boot, feed, make_entry, record_notify

LIVE = "notify.mobile_app_phone"


# ── The predicate ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", True),
        ("force_stopped", True),
        ("interrupted", False),
        (None, True),  # no status: today's behaviour
    ],
)
def test_cycle_end_is_finish(status: Any, expected: bool) -> None:
    assert notif_rules.cycle_end_is_finish(status) is expected


# ── Manager: the finished push and the Clean state ──────────────────────────


def _make_manager(hass: HomeAssistant, options: dict[str, Any]) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "test_manager_10"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power", **options}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    mgr.profile_store.get_profiles = MagicMock(return_value={})
    mgr.profile_store.async_add_cycle = AsyncMock()
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mgr.profile_store.async_rebuild_envelope = AsyncMock()
    mgr.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0, ranking=[])
    )
    mgr._run_post_cycle_processing = AsyncMock()
    return mgr


def _cycle(status: str) -> dict[str, Any]:
    return {
        "id": f"cycle-{status}",
        "start_time": "2026-05-01T08:00:00+00:00",
        "duration": 3600.0 if status != "interrupted" else 210.0,
        "status": status,
        "power_data": [[0.0, 50.0], [60.0, 200.0]],
    }


def _finish_calls(dispatch: MagicMock) -> list[Any]:
    return [
        c for c in dispatch.call_args_list
        if c.kwargs.get("event_type") == NOTIFY_EVENT_FINISH
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "notified", "clean"),
    [
        ("completed", True, True),
        ("force_stopped", True, True),
        ("interrupted", False, False),
    ],
)
async def test_finish_push_and_clean_state_follow_the_status(
    hass: HomeAssistant, status: str, notified: bool, clean: bool
) -> None:
    mgr = _make_manager(hass, {
        CONF_NOTIFY_FINISH_SERVICES: [LIVE],
        CONF_UNLOAD_TRACK_WITHOUT_DOOR: True,
        CONF_NOTIFY_FINISH_MESSAGE: "{device} {status} after {duration}m",
    })
    with patch.object(mgr, "_dispatch_notification") as dispatch:
        await mgr._async_process_cycle_end(_cycle(status))
        await hass.async_block_till_done()

    calls = _finish_calls(dispatch)
    assert bool(calls) is notified
    if notified:
        assert calls[0].args[0] == f"Test Washer {status} after 60m"
        assert calls[0].kwargs["extra_vars"]["status"] == status
    assert mgr.is_clean_state is clean


# ── Real manager: a false start through the whole pipeline ──────────────────


async def test_false_start_sends_no_finished_card_and_clears_the_start(hass, freezer):
    """A real manager, real detector, real notify service: a 2 min burst ends
    ``interrupted``; its start card is answered by a clear, not "finished"."""
    calls = record_notify(hass)
    entry = make_entry(hass, {
        "notify_live_services": [LIVE],
        "completion_min_seconds": 600,
    })
    mgr = await boot(hass, entry)
    lifecycle = mgr._lifecycle_tag

    await feed(hass, freezer, 500, 120)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()

    stored = mgr.profile_store.get_past_cycles()
    assert stored and stored[-1]["status"] == "interrupted"
    assert not any("finished" in str(c.get("message")) for c in calls), calls
    assert any(
        c.get("message") == "clear_notification"
        and (c.get("data") or {}).get("tag") == lifecycle
        for c in calls
    ), "the start card was left on the phone"
    assert mgr.is_clean_state is False
    await mgr.async_shutdown()


async def test_completed_cycle_still_announces_with_status(hass, freezer):
    calls = record_notify(hass)
    entry = make_entry(hass, {"notify_finish_message": "{device}: {status}"})
    mgr = await boot(hass, entry)

    await feed(hass, freezer, 500, 900)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()

    assert mgr.profile_store.get_past_cycles()[-1]["status"] == "completed"
    assert any(c.get("message") == "Washer: completed" for c in calls), calls
    await mgr.async_shutdown()


# ── Playground: the finish marker uses the same predicate ───────────────────


def _synthetic(minutes: int) -> dict[str, Any]:
    pts = [[float(t), 500.0] for t in range(0, minutes * 60, 30)] + [[minutes * 60.0, 0.0]]
    return {
        "id": f"syn-{minutes}",
        "start_time": "2026-05-01T08:00:00+00:00",
        "duration": minutes * 60.0,
        "status": "completed",
        "power_data": pts,
    }


@pytest.mark.parametrize(("minutes", "status", "marker"), [
    (4, "interrupted", False),
    (20, "completed", True),
])
def test_playground_finish_marker_follows_the_status(
    minutes: int, status: str, marker: bool
) -> None:
    cfg = CycleDetectorConfig(
        min_power=2.0, off_delay=60, device_type="washing_machine", min_off_gap=60,
        start_threshold_w=3.0, stop_threshold_w=1.5, completion_min_seconds=600,
    )
    d = playground.simulate_cycle_detail(
        _synthetic(minutes), cfg, None, None, {CONF_NOTIFY_FINISH_SERVICES: ["notify.x"]}
    )
    assert d["outcome"]["status"] == status
    assert any(e["type"] == "notify_finish" for e in d["events"]) is marker
