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

"""Issue #378: device_type must reach the live CycleDetectorConfig.

CycleDetectorConfig.device_type defaults to washing_machine and the detector
branches on it in ~16 places (dishwasher end-spike wait, keep_tail, passive
drying deferral, etc.). The manager built the detector without passing
device_type and never set it on reload, so every non-washing-machine device
silently ran the washing-machine detection path. Unit tests for the detector
construct the config directly with device_type=..., which is why this wiring
gap went unnoticed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata import manager as mgr_mod
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_POWER_SENSOR,
    DEVICE_TYPE_WASHING_MACHINE,
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_WASHER_DRYER,
)


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


def _make_entry(device_type: str) -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry_378"
    entry.title = "Test Appliance"
    entry.options = {
        CONF_MIN_POWER: 5.0,
        CONF_POWER_SENSOR: "sensor.test_power",
        CONF_DEVICE_TYPE: device_type,
        "notify_finish_services": [],
    }
    entry.data = {}
    return entry


def _build_manager(mock_hass: Any, entry: Any) -> WashDataManager:
    """Construct a manager with a *real* CycleDetector (only ProfileStore mocked)."""
    mock_hass.config_entries.async_get_entry.return_value = entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        return WashDataManager(mock_hass, entry)


@pytest.mark.parametrize(
    "device_type",
    [
        DEVICE_TYPE_WASHING_MACHINE,
        DEVICE_TYPE_DISHWASHER,
        DEVICE_TYPE_DRYER,
        DEVICE_TYPE_WASHER_DRYER,
    ],
)
def test_construction_wires_device_type_into_detector(mock_hass, device_type):
    """The detector's config carries the configured device type, not the default."""
    mgr = _build_manager(mock_hass, _make_entry(device_type))
    assert mgr.detector.config.device_type == device_type


@pytest.mark.asyncio
async def test_reload_updates_detector_device_type(mock_hass):
    """Changing the appliance type in the UI updates the detector config on reload."""
    entry = _make_entry(DEVICE_TYPE_WASHING_MACHINE)
    mgr = _build_manager(mock_hass, entry)
    assert mgr.detector.config.device_type == DEVICE_TYPE_WASHING_MACHINE

    # profile_store is a MagicMock (patched); give reload a real ratio tuple to unpack.
    mgr.profile_store.get_duration_ratio_limits.return_value = (0.1, 1.5)

    # Switch appliance type and reload (no power-sensor change -> no swap branch).
    entry.options[CONF_DEVICE_TYPE] = DEVICE_TYPE_DISHWASHER

    with patch.object(mgr, "_setup_external_end_trigger", AsyncMock()), \
         patch.object(mgr, "_setup_door_sensor_listener", AsyncMock()), \
         patch.object(mgr, "_setup_notify_people_listener", AsyncMock()), \
         patch.object(mgr, "_setup_maintenance_scheduler", AsyncMock()), \
         patch.object(mgr, "_setup_ml_training_scheduler", MagicMock()), \
         patch.object(mgr, "_attempt_state_restoration", AsyncMock()), \
         patch.object(mgr_mod, "async_dispatcher_send", MagicMock()):
        await mgr.async_reload_config(entry)

    assert mgr.device_type == DEVICE_TYPE_DISHWASHER
    assert mgr.detector.config.device_type == DEVICE_TYPE_DISHWASHER
