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

"""Issue #342: door-auto-open dishwashers (AirDry etc.).

These machines pop the door open at cycle end. Today any door-open during an
active cycle sets a sticky user-pause, so the cycle is stranded in user_paused.
With the "door opens at end" option on, a door-open on a running/ending cycle
instead arms a short dwell timer: if the door stays open past the dwell it
finalizes the cycle (completed, like the External End Trigger); a brief open
(adding an item) closes before the dwell and is a no-op. The sticky pause is not
set for these devices.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata import manager as mgr_mod
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_DOOR_SENSOR_ENTITY, CONF_DOOR_OPENS_AT_END,
    CONF_DOOR_END_DWELL_SECONDS, STATE_RUNNING,
)


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.async_create_task = MagicMock(side_effect=lambda coro: getattr(coro, "close", lambda: None)())
    hass.config_entries.async_get_entry = MagicMock()
    return hass


def _make_manager(mock_hass, options) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.title = "Dishwasher"
    entry.options = {
        CONF_MIN_POWER: 2.0, "power_sensor": "sensor.p",
        CONF_DOOR_SENSOR_ENTITY: "binary_sensor.door", **options,
    }
    mock_hass.config_entries.async_get_entry.return_value = entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(mock_hass, entry)
        mgr._notify_update = MagicMock()
        return mgr


def _door_event(new: str, old: str = "off"):
    ev = MagicMock()
    ns = MagicMock(); ns.state = new
    os_ = MagicMock(); os_.state = old
    ev.data = {"new_state": ns, "old_state": os_, "entity_id": "binary_sensor.door"}
    return ev


def test_door_open_arms_dwell_not_sticky_pause(mock_hass):
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: True, CONF_DOOR_END_DWELL_SECONDS: 60})
    mgr.detector.state = STATE_RUNNING
    with patch.object(mgr_mod, "async_call_later", return_value=lambda: None) as later:
        mgr._handle_door_sensor_change(_door_event("on"))
    later.assert_called_once()
    # Auto-open device must NOT get the sticky user-pause that strands the cycle.
    assert mgr._is_user_paused is False


def _set_door_state(mgr, state: str) -> None:
    """Point the manager's hass at a door sensor reading ``state``."""
    door = MagicMock(); door.state = state
    mgr.hass.states.get = MagicMock(return_value=door)


def test_dwell_fires_finalizes_via_user_stop(mock_hass):
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: True})
    mgr.detector.state = STATE_RUNNING
    _set_door_state(mgr, "on")  # door still open when the dwell fires
    mgr._is_user_paused = False
    mgr._door_end_dwell_fired(None)
    mgr.detector.user_stop.assert_called_once()


def test_dwell_does_not_finalize_while_still_drawing_power(mock_hass):
    """A mid-cycle door open on a still-drawing appliance must not be recorded as a
    completed cycle: the end-of-cycle door pop only happens after power drops, so
    power at/above the stop threshold means the user opened the door mid-cycle.

    The dwell must be RE-ARMED rather than dropped - it is a one-shot timer, so
    abandoning it would permanently lose the door-based finalize for a machine that
    is merely mid-pulse (fan/zeolite drying can draw with the door popped open)."""
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: True})
    mgr.detector.state = STATE_RUNNING
    mgr.detector.config.stop_threshold_w = 2.0
    _set_door_state(mgr, "on")
    mgr._is_user_paused = False
    mgr._current_power = 45.0          # still washing
    with patch.object(mgr_mod, "async_call_later", return_value=lambda: None) as later:
        mgr._door_end_dwell_fired(None)
    mgr.detector.user_stop.assert_not_called()
    later.assert_called_once()          # re-armed, not abandoned


def test_dwell_finalizes_once_power_has_dropped(mock_hass):
    """The legitimate auto-open path is unaffected: the door pop follows the power
    drop, so a quiet appliance with the door held open still finalizes."""
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: True})
    mgr.detector.state = STATE_RUNNING
    mgr.detector.config.stop_threshold_w = 2.0
    _set_door_state(mgr, "on")
    mgr._is_user_paused = False
    mgr._current_power = 0.4           # below the stop threshold
    mgr._door_end_dwell_fired(None)
    mgr.detector.user_stop.assert_called_once()


def test_dwell_does_not_finalize_when_door_closed(mock_hass):
    """A missed close event: if the door reads closed when the dwell fires, do
    not finalize (re-validation guard, #342)."""
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: True})
    mgr.detector.state = STATE_RUNNING
    _set_door_state(mgr, "off")
    mgr._door_end_dwell_fired(None)
    mgr.detector.user_stop.assert_not_called()


def test_dwell_does_not_finalize_when_user_paused(mock_hass):
    """A user pause landing mid-dwell must block auto-open finalization (#342)."""
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: True})
    mgr.detector.state = STATE_RUNNING
    _set_door_state(mgr, "on")
    mgr._is_user_paused = True
    mgr._door_end_dwell_fired(None)
    mgr.detector.user_stop.assert_not_called()


def test_door_close_cancels_pending_dwell(mock_hass):
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: True})
    mgr.detector.state = STATE_RUNNING
    cancel = MagicMock()
    with patch.object(mgr_mod, "async_call_later", return_value=cancel):
        mgr._handle_door_sensor_change(_door_event("on"))
    # Door closes before the dwell -> the pending finalize is cancelled.
    mgr._handle_door_sensor_change(_door_event("off", old="on"))
    cancel.assert_called_once()
    assert mgr._remove_door_end_dwell is None


def test_cycle_end_cancels_stale_dwell(mock_hass):
    """_on_cycle_end must cancel any pending door-end dwell so a stale timer
    cannot fire during a subsequent new cycle (#342 edge case)."""
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: True})
    cancel = MagicMock()
    mgr._remove_door_end_dwell = cancel
    cycle_data = {
        "duration": 3600,
        "max_power": 800,
        "power_data": [[0.0, 800.0], [1800.0, 400.0], [3600.0, 0.0]],
        "start_time": None,
    }
    mgr._on_cycle_end(cycle_data)
    cancel.assert_called_once()
    assert mgr._remove_door_end_dwell is None


def test_door_open_without_flag_keeps_sticky_pause(mock_hass):
    mgr = _make_manager(mock_hass, {CONF_DOOR_OPENS_AT_END: False})
    mgr.detector.state = STATE_RUNNING
    with patch.object(mgr_mod, "async_call_later", return_value=lambda: None) as later:
        mgr._handle_door_sensor_change(_door_event("on"))
    later.assert_not_called()
    # Legacy behaviour unchanged: door-open pauses the cycle.
    assert mgr._is_user_paused is True
    mgr.detector.set_verified_pause.assert_called_with(True)
