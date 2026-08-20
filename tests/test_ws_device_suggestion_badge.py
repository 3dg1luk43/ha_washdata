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

"""``get_devices`` reports the suggestion *keys* behind the device-pill badge, and
filters them exactly like ``get_suggestions`` does.

The panel merges these keys with the Calibrated (ML) recommendations it computes
client-side, so a key both engines suggest is only counted once. Filtering has to
match ``ws_get_suggestions`` or the pill badge would disagree with the Settings
tab banner (muted keys, and suggestions equal to the current value, are hidden
there but used to be counted here).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN


def _call_get_devices(suggestions, options, locked=()):
    """Invoke ws_get_devices for one entry and return its device info dict."""
    entry = SimpleNamespace(
        entry_id="e1", title="Washer", data={}, options=dict(options)
    )
    store = SimpleNamespace(
        get_suggestions=lambda: dict(suggestions),
        get_locked_suggestions=lambda: list(locked),
        get_pending_feedback=lambda: {},
        get_past_cycles=lambda: [],
    )
    manager = SimpleNamespace(
        detector=SimpleNamespace(state="off", sub_state=None),
        _current_program=None,
        manual_program_active=False,
        _time_remaining=None,
        _total_duration=None,
        _current_power=None,
        _cycle_progress=None,
        profile_store=store,
        is_user_paused=False,
        recorder=SimpleNamespace(is_recording=False),
    )

    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {DOMAIN: {"e1": manager}}
    connection = MagicMock()

    with patch.object(ws_api, "_effective_level", return_value="admin"):
        ws_api.ws_get_devices(hass, connection, {"id": 1, "type": "x"})

    connection.send_result.assert_called_once()
    return connection.send_result.call_args[0][1]["devices"][0]


def test_pending_suggestion_is_reported_as_a_key():
    dev = _call_get_devices(
        {"off_delay": {"value": 300}}, {"off_delay": 120}
    )
    assert dev["suggestion_keys"] == ["off_delay"]
    assert dev["suggestions_count"] == 1


def test_suggestion_equal_to_current_value_is_not_counted():
    # Numeric-tolerant, like ws_get_suggestions: 120.0 == 120 -> nothing to change.
    dev = _call_get_devices(
        {"off_delay": {"value": 120.0}}, {"off_delay": 120}
    )
    assert dev["suggestion_keys"] == []
    assert dev["suggestions_count"] == 0


def test_muted_suggestion_is_not_counted():
    # #343: a muted key has no card in Settings, so it must not badge the pill.
    dev = _call_get_devices(
        {"off_delay": {"value": 300}, "stop_threshold_w": {"value": 4}},
        {"off_delay": 120, "stop_threshold_w": 3},
        locked=("off_delay",),
    )
    assert dev["suggestion_keys"] == ["stop_threshold_w"]
    assert dev["suggestions_count"] == 1


def test_valueless_suggestion_entry_is_ignored():
    dev = _call_get_devices(
        {"off_delay": {"value": None}, "min_power": "not-a-dict"}, {"off_delay": 120}
    )
    assert dev["suggestion_keys"] == []


def test_keys_are_reported_even_when_no_manager_is_loaded():
    # Contract: the field always exists, so the panel never has to guard on it.
    entry = SimpleNamespace(entry_id="e1", title="Washer", data={}, options={})
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {DOMAIN: {}}
    connection = MagicMock()
    with patch.object(ws_api, "_effective_level", return_value="admin"):
        ws_api.ws_get_devices(hass, connection, {"id": 1, "type": "x"})
    dev = connection.send_result.call_args[0][1]["devices"][0]
    assert dev["suggestion_keys"] == []
    assert dev["suggestions_count"] == 0
