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
"""Issue #407 - diagnostics feature_flags read off the object that owns the flag.

Two of the three flags were read off the manager, which owns neither:
`save_debug_traces` lives on the ProfileStore and `auto_maintenance` has no runtime
attribute at all, so both reported False for every user. The dump is the first thing
consulted when triaging, so a permanently-False flag actively misleads.

Verifying that turned up the functional half: `save_debug_traces` was pushed into the
store only in `WashDataManager.__init__`. An options change reloads in place
(`async_reload_config`) without rebuilding the store, so the panel checkbox did nothing
until HA restarted.

These tests assert concrete flag *values*. A MagicMock manager satisfies every
`getattr`, so an existence check would pass against the bug.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import manager as mgr_mod
from custom_components.ha_washdata.const import (
    CONF_AUTO_MAINTENANCE,
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_POWER_SENSOR,
    CONF_SAVE_DEBUG_TRACES,
    DEFAULT_AUTO_MAINTENANCE,
    DEVICE_TYPE_WASHING_MACHINE,
    DOMAIN,
)
from custom_components.ha_washdata.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import ProfileStore


# ---------------------------------------------------------------------------
# Diagnostics: read the owning object
# ---------------------------------------------------------------------------


def _diag_manager(*, store_debug: bool, scheduler_armed: bool = True):
    manager = MagicMock()
    manager.check_state.return_value = "idle"
    manager.current_program = None
    manager.time_remaining = None
    manager.cycle_progress = 0.0
    manager.sample_interval_stats = {}
    manager.profile_sample_repair_stats = {}
    manager.profile_store.get_suggestions.return_value = {}
    manager.profile_store.export_data.return_value = {}
    manager.diag_buffer.redacted_snapshot.return_value = {}
    manager.profile_store.save_debug_traces = store_debug
    manager._notify_fire_events = True
    manager._remove_maintenance_scheduler = (
        (lambda: None) if scheduler_armed else None
    )
    # Model production faithfully: the manager carries neither of these. A bare
    # MagicMock would answer every getattr with a truthy Mock and hide the bug,
    # which is exactly how the original test passed against it.
    del manager._auto_maintenance
    del manager._save_debug_traces
    return manager


def _diag_entry(options=None):
    entry = MagicMock()
    entry.entry_id = "e_407"
    entry.data = {}
    entry.options = dict(options or {})
    entry.as_dict.return_value = {}
    return entry


async def _flags(manager, entry):
    hass = MagicMock()
    hass.data = {DOMAIN: {entry.entry_id: manager}}
    result = await async_get_config_entry_diagnostics(hass, entry)
    return result["manager_state"]["feature_flags"]


@pytest.mark.asyncio
async def test_save_debug_traces_comes_from_the_store_not_the_manager():
    """The manager has no such attribute; a stale one there must not fake the flag."""
    manager = _diag_manager(store_debug=True)
    # The pre-fix source of truth, deliberately contradicting the store.
    manager._save_debug_traces = False

    flags = await _flags(manager, _diag_entry())

    assert flags["save_debug_traces"] is True


@pytest.mark.asyncio
async def test_save_debug_traces_reports_disabled_when_the_store_has_it_off():
    manager = _diag_manager(store_debug=False)
    manager._save_debug_traces = True

    flags = await _flags(manager, _diag_entry())

    assert flags["save_debug_traces"] is False


@pytest.mark.asyncio
async def test_auto_maintenance_defaults_to_enabled_when_unset():
    """DEFAULT_AUTO_MAINTENANCE is True; the old getattr fallback said False."""
    assert DEFAULT_AUTO_MAINTENANCE is True

    flags = await _flags(_diag_manager(store_debug=False), _diag_entry())

    assert flags["auto_maintenance"] is True


@pytest.mark.asyncio
async def test_auto_maintenance_follows_the_option():
    manager = _diag_manager(store_debug=False)

    on = await _flags(manager, _diag_entry({CONF_AUTO_MAINTENANCE: True}))
    off = await _flags(manager, _diag_entry({CONF_AUTO_MAINTENANCE: False}))

    assert on["auto_maintenance"] is True
    assert off["auto_maintenance"] is False


@pytest.mark.asyncio
async def test_auto_maintenance_falls_back_to_entry_data():
    """Pre-migration entries kept the key in data, not options."""
    entry = _diag_entry()
    entry.data = {CONF_AUTO_MAINTENANCE: False}

    flags = await _flags(_diag_manager(store_debug=False), entry)

    assert flags["auto_maintenance"] is False


@pytest.mark.asyncio
async def test_auto_maintenance_scheduled_tracks_the_timer_handle():
    """Reported separately from the option, so a failed arm is diagnosable."""
    armed = await _flags(
        _diag_manager(store_debug=False, scheduler_armed=True), _diag_entry()
    )
    idle = await _flags(
        _diag_manager(store_debug=False, scheduler_armed=False), _diag_entry()
    )

    assert armed["auto_maintenance_scheduled"] is True
    assert idle["auto_maintenance_scheduled"] is False


# ---------------------------------------------------------------------------
# The functional half: the option must survive an in-place reload
# ---------------------------------------------------------------------------


@pytest.fixture
def store_hass():
    hass = MagicMock()
    hass.data = {}
    return hass


def test_store_exposes_save_debug_traces_over_the_backing_attribute(store_hass):
    """The two internal readers use `_save_debug_traces`; the property must drive it."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        store = ProfileStore(store_hass, "e_407", save_debug_traces=False)

    assert store.save_debug_traces is False
    assert store._save_debug_traces is False

    store.save_debug_traces = True
    assert store.save_debug_traces is True
    assert store._save_debug_traces is True

    # Coerced, so a truthy option value cannot leak a non-bool into the readers.
    store.save_debug_traces = 0
    assert store._save_debug_traces is False


@pytest.fixture
def mgr_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.config_entries.async_get_entry = MagicMock()
    return hass


def _mgr_entry(save_debug_traces=None):
    entry = MagicMock()
    entry.entry_id = "e_407_mgr"
    entry.title = "Test Appliance"
    entry.options = {
        CONF_MIN_POWER: 5.0,
        CONF_POWER_SENSOR: "sensor.test_power",
        CONF_DEVICE_TYPE: DEVICE_TYPE_WASHING_MACHINE,
        "notify_finish_services": [],
    }
    if save_debug_traces is not None:
        entry.options[CONF_SAVE_DEBUG_TRACES] = save_debug_traces
    entry.data = {}
    return entry


async def _reload(mgr, entry):
    mgr.profile_store.get_duration_ratio_limits.return_value = (0.1, 1.5)
    with patch.object(mgr, "_setup_external_end_trigger", AsyncMock()), \
         patch.object(mgr, "_setup_door_sensor_listener", AsyncMock()), \
         patch.object(mgr, "_setup_notify_people_listener", AsyncMock()), \
         patch.object(mgr, "_setup_maintenance_scheduler", AsyncMock()), \
         patch.object(mgr, "_setup_ml_training_scheduler", MagicMock()), \
         patch.object(mgr, "_attempt_state_restoration", AsyncMock()), \
         patch.object(mgr_mod, "async_dispatcher_send", MagicMock()):
        await mgr.async_reload_config(entry)


@pytest.mark.asyncio
async def test_reload_pushes_save_debug_traces_into_the_store(mgr_hass):
    """Ticking the panel checkbox took effect only after an HA restart."""
    entry = _mgr_entry(save_debug_traces=False)
    mgr_hass.config_entries.async_get_entry.return_value = entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(mgr_hass, entry)

    entry.options[CONF_SAVE_DEBUG_TRACES] = True
    await _reload(mgr, entry)

    assert mgr.profile_store.save_debug_traces is True


@pytest.mark.asyncio
async def test_reload_clears_save_debug_traces_when_unticked(mgr_hass):
    entry = _mgr_entry(save_debug_traces=True)
    mgr_hass.config_entries.async_get_entry.return_value = entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(mgr_hass, entry)

    entry.options[CONF_SAVE_DEBUG_TRACES] = False
    await _reload(mgr, entry)

    assert mgr.profile_store.save_debug_traces is False
