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

"""Issue #363 / #329: WashData must also consume unchanged power reports.

HA fires EVENT_STATE_CHANGED only when the state value changes. A plug that
re-reports the same value (Tasmota TelePeriod, Zigbee max reporting interval)
fires EVENT_STATE_REPORTED instead. WashData only subscribed to state_changed,
so those reports were invisible: a flat sub-threshold tail never advanced the
detector's end-of-cycle timer (#363) and a finished cycle lagged by the plug's
reporting interval (#329). The fix subscribes to state_reported as well, routed
into the same handler.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata import manager as mgr_mod
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.config_entries.async_get_entry = MagicMock()
    return hass


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_COMPLETION_MIN_SECONDS: 600,
        "power_sensor": "sensor.test_power",
        "notify_finish_services": [],
    }
    return entry


@pytest.fixture
def manager(mock_hass: Any, mock_entry: Any) -> WashDataManager:
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.learning_manager.process_power_reading = MagicMock()
        return mgr


def _mk_state(value: str) -> Any:
    st = MagicMock()
    st.state = value
    st.last_updated = datetime.now(timezone.utc)
    st.last_reported = datetime.now(timezone.utc)
    return st


def test_subscribe_registers_both_change_and_report_listeners(manager: WashDataManager) -> None:
    """_subscribe_power_sensor wires state_changed AND state_reported to the handler."""
    with patch.object(mgr_mod, "async_track_state_change_event") as chg, \
         patch.object(mgr_mod, "async_track_state_report_event") as rep:
        chg.return_value = lambda: None
        rep.return_value = lambda: None
        manager._subscribe_power_sensor()

    chg.assert_called_once()
    rep.assert_called_once()
    # Both target the configured power sensor and the same handler.
    assert chg.call_args.args[1] == [manager.power_sensor_entity_id]
    assert rep.call_args.args[1] == [manager.power_sensor_entity_id]
    assert chg.call_args.args[2] == manager._async_power_changed
    assert rep.call_args.args[2] == manager._async_power_changed
    # Remove handles are stored so they can be torn down.
    assert manager._remove_listener is not None
    assert manager._remove_report_listener is not None


def test_power_handler_is_hass_callback(manager: WashDataManager) -> None:
    """The power handler MUST run on the event loop, not an executor thread.

    Regression guard: when the #363/#329 refactor extracted registration into
    ``_subscribe_power_sensor``, the ``@callback`` decorator was accidentally
    moved off ``_async_power_changed`` (and doubled onto ``_subscribe_power_sensor``).
    An undecorated sync action is inferred by HA as an *executor* HassJob, so the
    handler ran in a SyncWorker thread and every downstream loop-only call
    (``async_dispatcher_send``, ``hass.async_create_task``, ``hass.bus.async_fire``)
    raised a thread-safety error on HA 2026.8+, silently breaking matching,
    notifications and state saves.
    """
    from homeassistant.core import is_callback

    assert is_callback(manager._async_power_changed), (
        "_async_power_changed must be @callback so HA runs it on the event loop"
    )


def test_resubscribe_removes_previous_listeners(manager: WashDataManager) -> None:
    """Re-subscribing tears down the prior change and report listeners (no leak)."""
    old_chg = MagicMock()
    old_rep = MagicMock()
    manager._remove_listener = old_chg
    manager._remove_report_listener = old_rep
    with patch.object(mgr_mod, "async_track_state_change_event", return_value=lambda: None), \
         patch.object(mgr_mod, "async_track_state_report_event", return_value=lambda: None):
        manager._subscribe_power_sensor()
    old_chg.assert_called_once()
    old_rep.assert_called_once()


def test_report_shaped_event_without_old_state_reaches_detector(manager: WashDataManager) -> None:
    """A state_reported event (new_state, no old_state) still drives the detector."""
    ev = MagicMock()
    ev.data = {"new_state": _mk_state("123.0"), "old_state": None}
    manager._async_power_changed(ev)
    manager.detector.process_reading.assert_called_once()
    assert manager._current_power == 123.0
