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
"""Register item 515 (2): no new write may leave stop_threshold_w >= start_threshold_w.

A contributed washer export (0.4.3) runs with stop 6.0 W above start 2.3 W. The
panel's conflict check holds back such a hand edit, but ``ws_set_options`` itself
took it from every other writer (the Playground publish and sweep, the per-setting
Revert), ``ws_apply_suggestions`` applied one threshold of a reconciled pair alone
(a subset, a muted cascade, a pair reconciled against options that changed since),
the imports copied a foreign pair in, and the reconciler's own 0.1 W rounding put
stop back on start at start <= 0.2 W. An entry that already runs inverted keeps
saving every other setting: only a write that changes the effective pair is held
to it. Fast; the handlers run on the real ``hass`` fixture with a real entry.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_COMPLETION_MIN_SECONDS,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
)
from custom_components.ha_washdata.detector_config import inverted_threshold_pair
from custom_components.ha_washdata.suggestion_engine import reconcile_suggestions

WM = "washing_machine"
INVERTED = {CONF_MIN_POWER: 3.0, CONF_START_THRESHOLD_W: 2.3, CONF_STOP_THRESHOLD_W: 6.0}


# ─── The rule ─────────────────────────────────────────────────────────────────


def test_an_ordered_pair_passes() -> None:
    assert inverted_threshold_pair({}, {CONF_START_THRESHOLD_W: 6.0, CONF_STOP_THRESHOLD_W: 2.8}, WM) is None


def test_a_write_that_inverts_the_pair_is_caught() -> None:
    assert inverted_threshold_pair({}, {CONF_STOP_THRESHOLD_W: 6.0}, WM) == (3.0, 6.0)
    # Equal is inverted too: no hysteresis band, the panel's rule (start <= stop).
    assert inverted_threshold_pair(
        {}, {CONF_START_THRESHOLD_W: 4.0, CONF_STOP_THRESHOLD_W: 4.0}, WM
    ) == (4.0, 4.0)


def test_an_unset_threshold_follows_min_power() -> None:
    """stop unset is 0.6 x min_power: raising min_power under an explicit start can
    invert the pair too (6.0 W is exactly 0.6 x the export's data min_power 10 W)."""
    before = {CONF_MIN_POWER: 3.0, CONF_START_THRESHOLD_W: 2.3}
    after = {**before, CONF_MIN_POWER: 10.0}
    assert inverted_threshold_pair(before, after, WM) == (2.3, 6.0)


def test_an_entry_already_inverted_is_not_held_to_it_until_the_pair_changes() -> None:
    assert inverted_threshold_pair(INVERTED, {**INVERTED, CONF_OFF_DELAY: 300}, WM) is None
    assert inverted_threshold_pair(INVERTED, {**INVERTED, CONF_START_THRESHOLD_W: 2.5}, WM) == (2.5, 6.0)


# ─── ws_set_options ───────────────────────────────────────────────────────────


def _entry(hass: HomeAssistant, options: dict) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN, entry_id="e515", title="Washer",
        data={CONF_POWER_SENSOR: "sensor.power"}, options=dict(options),
    )
    entry.add_to_hass(hass)
    return entry


async def _set_options(hass: HomeAssistant, entry: MockConfigEntry, options: dict) -> MagicMock:
    connection = MagicMock()
    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api.ws_set_options.__wrapped__(
            hass, connection, {"id": 1, "entry_id": entry.entry_id, "options": options}
        )
    return connection


async def test_set_options_refuses_a_stop_above_start(hass: HomeAssistant) -> None:
    entry = _entry(hass, {CONF_START_THRESHOLD_W: 2.3, CONF_STOP_THRESHOLD_W: 1.8})
    # e.g. a Playground "publish" of the stop threshold alone
    connection = await _set_options(hass, entry, {CONF_STOP_THRESHOLD_W: 6.0})
    assert entry.options[CONF_STOP_THRESHOLD_W] == 1.8  # nothing written
    code, message = connection.send_error.call_args.args[1:]
    assert code == "invalid_threshold_pair"
    assert "Stop Threshold (6 W) must be below Start Threshold (2.3 W)" in message


async def test_set_options_still_saves_other_settings_on_an_inverted_entry(
    hass: HomeAssistant,
) -> None:
    entry = _entry(hass, INVERTED)
    connection = await _set_options(hass, entry, {CONF_OFF_DELAY: 300})
    connection.send_error.assert_not_called()
    assert entry.options[CONF_OFF_DELAY] == 300
    assert (entry.options[CONF_START_THRESHOLD_W], entry.options[CONF_STOP_THRESHOLD_W]) == (2.3, 6.0)


async def test_set_options_accepts_the_fix_of_an_inverted_pair(hass: HomeAssistant) -> None:
    entry = _entry(hass, INVERTED)
    connection = await _set_options(hass, entry, {CONF_START_THRESHOLD_W: 7.5})
    connection.send_error.assert_not_called()
    assert entry.options[CONF_START_THRESHOLD_W] == 7.5


# ─── ws_apply_suggestions ─────────────────────────────────────────────────────


async def _apply(
    hass: HomeAssistant, entry: MockConfigEntry, suggestions: dict, keys: list[str]
) -> MagicMock:
    manager = MagicMock()
    manager.device_type = WM
    manager.profile_store.get_suggestions.return_value = suggestions
    manager.profile_store.get_locked_suggestions.return_value = []
    manager.profile_store.get_lifetime_cycle_count.return_value = 10
    manager.profile_store.clear_suggestions = AsyncMock()
    connection = MagicMock()
    with patch.object(ws_api, "_get_manager", return_value=manager), patch.object(
        ws_api, "_record_option_changes", AsyncMock()
    ):
        await ws_api.ws_apply_suggestions.__wrapped__(
            hass, connection, {"id": 1, "entry_id": entry.entry_id, "keys": keys}
        )
    return connection


async def test_applying_one_threshold_of_a_pair_alone_is_refused(hass: HomeAssistant) -> None:
    """A standby-floor stop suggestion (#458) with its cascaded start left out."""
    entry = _entry(hass, {CONF_START_THRESHOLD_W: 2.3, CONF_STOP_THRESHOLD_W: 1.8})
    suggestions = {
        CONF_STOP_THRESHOLD_W: {"value": 6.0},
        CONF_START_THRESHOLD_W: {"value": 7.5, "cascade": True},
    }
    connection = await _apply(hass, entry, suggestions, [CONF_STOP_THRESHOLD_W])
    assert connection.send_error.call_args.args[1] == "invalid_threshold_pair"
    assert entry.options[CONF_STOP_THRESHOLD_W] == 1.8
    connection = await _apply(
        hass, entry, suggestions, [CONF_STOP_THRESHOLD_W, CONF_START_THRESHOLD_W]
    )
    connection.send_error.assert_not_called()
    assert (entry.options[CONF_START_THRESHOLD_W], entry.options[CONF_STOP_THRESHOLD_W]) == (7.5, 6.0)


# ─── Imports ──────────────────────────────────────────────────────────────────


async def test_an_import_keeps_this_devices_pair_when_its_own_is_inverted(
    hass: HomeAssistant,
) -> None:
    """Someone else's tuning: the inverted pair is left out and logged, the rest
    of the import (and its profiles and cycles) still applies."""
    entry = _entry(hass, {CONF_START_THRESHOLD_W: 6.0, CONF_STOP_THRESHOLD_W: 2.8})
    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api.async_apply_imported_entry_options(
            hass, entry, {"entry_options": {**INVERTED, CONF_OFF_DELAY: 240}}, "import_config"
        )
    assert (entry.options[CONF_START_THRESHOLD_W], entry.options[CONF_STOP_THRESHOLD_W]) == (6.0, 2.8)
    assert entry.options[CONF_OFF_DELAY] == 240
    assert entry.options[CONF_MIN_POWER] == 3.0  # it does not invert this device's pair


async def test_a_store_bundle_keeps_this_devices_pair_when_its_own_is_inverted(
    hass: HomeAssistant,
) -> None:
    """The thresholds are shareable settings, so a bundle uploaded from an inverted
    device would hand its pair on."""
    entry = _entry(hass, {CONF_START_THRESHOLD_W: 6.0, CONF_STOP_THRESHOLD_W: 2.8})
    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        applied = await ws_api._apply_store_settings(
            hass, entry.entry_id,
            {"settings": {**INVERTED, CONF_COMPLETION_MIN_SECONDS: 900}}, True,
        )
    assert (entry.options[CONF_START_THRESHOLD_W], entry.options[CONF_STOP_THRESHOLD_W]) == (6.0, 2.8)
    assert entry.options[CONF_COMPLETION_MIN_SECONDS] == 900
    assert applied == 2  # min_power and completion_min_seconds


# ─── reconcile_suggestions ────────────────────────────────────────────────────


def test_reconcile_keeps_a_tiny_start_above_its_stop() -> None:
    """Rule 1a's 0.1 W rounding put stop back on start at start <= 0.2 W."""
    out, _changed = reconcile_suggestions(
        {CONF_START_THRESHOLD_W: {"value": 0.2, "reason": "r"}},
        {CONF_STOP_THRESHOLD_W: 0.5},
    )
    assert out[CONF_STOP_THRESHOLD_W]["value"] < out[CONF_START_THRESHOLD_W]["value"]


def test_reconcile_is_unchanged_for_ordinary_values() -> None:
    out, _changed = reconcile_suggestions(
        {CONF_START_THRESHOLD_W: {"value": 2.76, "reason": "r"}},
        {CONF_STOP_THRESHOLD_W: 6.0},
    )
    assert out[CONF_STOP_THRESHOLD_W]["value"] == 2.2  # round(2.76 x 0.8, 1), as before
