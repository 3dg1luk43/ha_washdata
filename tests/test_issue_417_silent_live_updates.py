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

"""Issue #417: recurring live progress updates must not buzz the phone.

iOS alerts on every Live Activity refresh unless the update is marked silent, so a
600 s live interval played a sound and vibrated all cycle long. The refresh now
carries ``silent: true`` plus the companion's generic quiet key
(``push.interruption-level: passive``), while the update that STARTS the activity
stays audible - the same behaviour Uber/Maps-style live activities have.

This is an option on the EXISTING live notification, not a new notification type.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.manager import (
    WashDataManager,
    _MOBILE_ONLY_EXTRA_KEYS,
)
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_NOTIFY_LIVE_SILENT,
    NOTIFY_EVENT_LIVE,
)

_PASSIVE = {"interruption-level": "passive"}


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.config_entries.async_get_entry = MagicMock()
    return hass


def _make_manager(mock_hass, options) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.title = "Test Washer"
    entry.options = {CONF_MIN_POWER: 2.0, "power_sensor": "sensor.p", **options}
    mock_hass.config_entries.async_get_entry.return_value = entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        return WashDataManager(mock_hass, entry)


def test_silent_keys_are_mobile_only():
    """Strict-schema platforms (Signal) reject unknown keys, so these stay mobile."""
    assert "silent" in _MOBILE_ONLY_EXTRA_KEYS
    assert "push" in _MOBILE_ONLY_EXTRA_KEYS


def test_refresh_is_silent_by_default(mock_hass):
    mgr = _make_manager(mock_hass, {})
    mgr._live_activity_started = True
    ev: dict[str, Any] = {}
    mgr._apply_live_notification_prefs(ev)
    assert ev["silent"] is True
    assert ev["push"] == _PASSIVE


def test_activity_start_stays_audible(mock_hass):
    """The first live push of a cycle is the "it's on your Lock Screen now" cue."""
    mgr = _make_manager(mock_hass, {})
    assert mgr._live_activity_started is False
    ev: dict[str, Any] = {}
    mgr._apply_live_notification_prefs(ev)
    assert "silent" not in ev
    assert "push" not in ev


def test_option_off_restores_the_audible_refresh(mock_hass):
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_LIVE_SILENT: False})
    mgr._live_activity_started = True
    ev: dict[str, Any] = {}
    mgr._apply_live_notification_prefs(ev)
    assert "silent" not in ev
    assert "push" not in ev


async def test_reload_picks_up_the_option(mock_hass):
    """A settings save reloads options in place, so the flag must follow (#417).

    The notification options are read in two places - the constructor and
    ``async_reload_config`` - and wiring only the first is the classic miss: the
    toggle would appear to do nothing until HA restarted.
    """
    mgr = _make_manager(mock_hass, {})
    assert mgr._notify_live_silent is True
    # Unrelated to the option wiring, and heavy to mock faithfully.
    mgr.profile_store.get_duration_ratio_limits.return_value = (0.1, 1.5)
    mgr._attempt_state_restoration = AsyncMock()
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.title = "Test Washer"
    entry.data = {"power_sensor": "sensor.p"}
    entry.options = {
        CONF_MIN_POWER: 2.0,
        "power_sensor": "sensor.p",
        CONF_NOTIFY_LIVE_SILENT: False,
    }
    await mgr.async_reload_config(entry)
    assert mgr._notify_live_silent is False


def test_silent_reaches_a_mobile_target(mock_hass):
    mgr = _make_manager(mock_hass, {})
    mgr._send_notification_service(
        "Almost done",
        services=["notify.mobile_app_phone"],
        event_type=NOTIFY_EVENT_LIVE,
        extra_vars={"tag": "wd", "live_update": True, "silent": True, "push": _PASSIVE},
    )
    ((_domain, _service, payload), _kw) = mock_hass.services.async_call.call_args
    assert payload["data"]["silent"] is True
    assert payload["data"]["push"] == _PASSIVE


def test_silent_never_reaches_a_non_mobile_target(mock_hass):
    mgr = _make_manager(mock_hass, {})
    extras = mgr._mobile_service_extras(
        {"silent": True, "push": _PASSIVE}, "notify.signal"
    )
    assert extras == {}
