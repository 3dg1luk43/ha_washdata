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


def test_option_defaults_reported_per_device_type():
    """The device-list conflict/suggestion badges score an unset cadence/ratio field
    against the value the integration would actually use (#396/#393), so get_devices
    carries the device-resolved defaults per device - matching the Settings tab."""
    # Unset device_type -> washing_machine (fast-sampling wet appliance).
    od = _call_get_devices({}, {})["option_defaults"]
    assert od["sampling_interval"] == 2.0
    assert od["watchdog_interval"] == 30
    assert od["start_duration_threshold"] == 5.0
    assert od["smart_termination_duration_ratio"] == 0.98
    # A coarse-sampling type resolves to the higher watchdog/start defaults.
    coarse = _call_get_devices({}, {"device_type": "dryer"})["option_defaults"]
    assert coarse["sampling_interval"] == 30.0
    assert coarse["watchdog_interval"] == 61
    assert coarse["start_duration_threshold"] == 30.0


def test_option_defaults_carry_the_effective_end_wait_pair():
    """#445: the detector ends a cycle after max(off_delay, min_off_gap), and
    min_off_gap was absent from this payload entirely.

    With the key missing the panel rendered an empty field for an unset
    min_off_gap and every cross-field rule's ``!= null`` guard short-circuited, so
    nothing showed the number actually governing the wait. The reporter set
    off_delay to 180 s on a washing machine whose real wait was 480 s, waited six
    minutes, and force-stopped three cycles.
    """
    wm = _call_get_devices({}, {})["option_defaults"]
    assert wm["min_off_gap"] == 480
    assert wm["off_delay"] == 180
    # A dishwasher's prior is far longer - this is the one that surprises people.
    dw = _call_get_devices({}, {"device_type": "dishwasher"})["option_defaults"]
    assert dw["min_off_gap"] == 3600
    assert dw["off_delay"] == 1800
    # Both are ints: the panel renders them into number inputs and compares them.
    for key in ("min_off_gap", "off_delay"):
        assert isinstance(wm[key], int) and isinstance(dw[key], int)


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


def test_a_manager_less_device_carries_every_declared_key():
    """The whole ``DeviceInfo`` contract, not one field at a time.

    Everything live is filled in inside ``if manager is not None:``, which is
    skipped for an entry that is mid-setup, stale, or whose setup failed - and
    short-circuited by its own ``except`` even when a manager exists. ``DeviceInfo``
    (``ws_schema.py``) and the generated ``ws-types.d.ts`` both declare these keys
    non-optional, so each one needs a base default. ``envelope_position`` was
    missing it; asserting the full annotation set is what stops the next field
    shipping the same way.
    """
    from custom_components.ha_washdata.ws_schema import DeviceInfo

    entry = SimpleNamespace(entry_id="e1", title="Washer", data={}, options={})
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {DOMAIN: {}}
    connection = MagicMock()
    with patch.object(ws_api, "_effective_level", return_value="admin"):
        ws_api.ws_get_devices(hass, connection, {"id": 1, "type": "x"})
    dev = connection.send_result.call_args[0][1]["devices"][0]

    missing = sorted(set(DeviceInfo.__annotations__) - set(dev))
    assert not missing, f"declared non-optional but absent without a manager: {missing}"
    assert dev["envelope_position"] is None
