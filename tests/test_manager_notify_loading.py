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
"""Unit tests for WashDataManager._load_notify_services.

Covers the per-event service lists (notify_start_services,
notify_finish_services, notify_live_services), and that the deprecated
notify_service + notify_events pair no longer feeds them at runtime (#472).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_EVENTS,
    CONF_NOTIFY_FINISH_SERVICES,
    CONF_NOTIFY_LIVE_SERVICES,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_START_SERVICES,
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
    NOTIFY_EVENT_START,
)
from custom_components.ha_washdata.manager import WashDataManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(options: dict) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "test_notify_entry"
    entry.title = "Test Washer"
    entry.options = options
    entry.data = {}
    return entry


def _make_manager(options: dict) -> WashDataManager:
    entry = _make_entry(options)
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
    return mgr


# ---------------------------------------------------------------------------
# New API: per-event service lists
# ---------------------------------------------------------------------------


def test_load_notify_services_reads_finish_services():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_FINISH_SERVICES: ["notify.mobile_app_phone"],
    })
    assert mgr._notify_finish_services == ["notify.mobile_app_phone"]
    assert mgr._notify_start_services == []
    assert mgr._notify_live_services == []


def test_load_notify_services_reads_start_services():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_START_SERVICES: ["notify.alexa_media"],
    })
    assert mgr._notify_start_services == ["notify.alexa_media"]
    assert mgr._notify_finish_services == []


def test_load_notify_services_reads_live_services():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_LIVE_SERVICES: ["notify.mobile_app_pixel"],
    })
    assert mgr._notify_live_services == ["notify.mobile_app_pixel"]


def test_load_notify_services_reads_all_three_lists():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_START_SERVICES: ["notify.a"],
        CONF_NOTIFY_FINISH_SERVICES: ["notify.b", "notify.c"],
        CONF_NOTIFY_LIVE_SERVICES: ["notify.d"],
    })
    assert mgr._notify_start_services == ["notify.a"]
    assert mgr._notify_finish_services == ["notify.b", "notify.c"]
    assert mgr._notify_live_services == ["notify.d"]


def test_load_notify_services_none_value_treated_as_empty():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_FINISH_SERVICES: None,
    })
    assert mgr._notify_finish_services == []


# ---------------------------------------------------------------------------
# No fallback to the deprecated notify_service + notify_events (#472)
# ---------------------------------------------------------------------------
#
# The manager used to fall back to the single notify_service whenever all three
# per-event lists were empty, so a user who cleared every list kept getting
# notified. The 3.11 -> 3.12 migration now converts and drops the key (see
# tests/test_issue_472_legacy_notify_service.py); a stray copy must be inert.


@pytest.mark.parametrize(
    "events",
    [None, [], [NOTIFY_EVENT_START, NOTIFY_EVENT_FINISH, NOTIFY_EVENT_LIVE],
     [NOTIFY_EVENT_LIVE], [NOTIFY_EVENT_FINISH]],
)
def test_stray_legacy_service_is_ignored_when_lists_are_absent(events):
    options = {"power_sensor": "sensor.p", CONF_NOTIFY_SERVICE: "notify.old_service"}
    if events is not None:
        options[CONF_NOTIFY_EVENTS] = events
    mgr = _make_manager(options)
    assert mgr._notify_start_services == []
    assert mgr._notify_finish_services == []
    assert mgr._notify_live_services == []


def test_stray_legacy_service_is_ignored_when_lists_were_cleared():
    """The #472 report: every list saved as [] and still notified."""
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_SERVICE: "notify.old_service",
        CONF_NOTIFY_START_SERVICES: [],
        CONF_NOTIFY_FINISH_SERVICES: [],
        CONF_NOTIFY_LIVE_SERVICES: [],
    })
    assert mgr._notify_start_services == []
    assert mgr._notify_finish_services == []
    assert mgr._notify_live_services == []


def test_stray_legacy_service_is_not_injected_alongside_new_lists():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_SERVICE: "notify.old_service",
        CONF_NOTIFY_FINISH_SERVICES: ["notify.new_service"],
    })
    assert mgr._notify_finish_services == ["notify.new_service"]
    assert mgr._notify_start_services == []
    assert mgr._notify_live_services == []
