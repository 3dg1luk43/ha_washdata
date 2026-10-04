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
"""Audit PLATFORM-09: the one-pass legacy migration must agree with the chain.

The same seeded options (a dryer with watchdog_interval=30, start_duration=5)
migrated to 61/30 from 3.9 (the 3.9 -> 3.10 cadence heal) but stayed 30/5 from
3.1 and 3.5, which take the one-pass legacy path: it copied the 3.10 -> 3.11 heal
but not this one. Old-schema fixtures in the test_migration_harness style.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import async_migrate_entry
from custom_components.ha_washdata.const import (
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    CONF_START_DURATION_THRESHOLD,
    CONF_WATCHDOG_INTERVAL,
    DOMAIN,
)
from custom_components.ha_washdata.detector_config import build_detector_config


@dataclass
class _Entry:
    domain: str = DOMAIN
    title: str = "Dryer"
    entry_id: str = "entry-1"
    version: int = 3
    minor_version: int = 1
    data: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


def _apply(e: _Entry, **kw: Any) -> None:
    for key in ("data", "options", "version", "minor_version"):
        if key in kw:
            setattr(e, key, kw[key])


def _seeded(device_type: str, minor: int) -> _Entry:
    return _Entry(
        minor_version=minor,
        data={CONF_POWER_SENSOR: "sensor.dryer_power"},
        options={
            CONF_DEVICE_TYPE: device_type,
            CONF_POWER_SENSOR: "sensor.dryer_power",
            CONF_MIN_POWER: 3.0,
            CONF_OFF_DELAY: 180,
            CONF_WATCHDOG_INTERVAL: 30,
            CONF_START_DURATION_THRESHOLD: 5,
        },
    )


async def _migrate(hass: HomeAssistant, entry: _Entry) -> _Entry:
    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply)
    assert await async_migrate_entry(hass, entry) is True
    assert (entry.version, entry.minor_version) == (
        CONFIG_ENTRY_VERSION, CONFIG_ENTRY_MINOR_VERSION,
    )
    return entry


@pytest.mark.asyncio
@pytest.mark.parametrize("minor", [1, 5])
async def test_one_pass_path_heals_the_seeded_dryer_cadence(
    hass: HomeAssistant, minor: int
) -> None:
    entry = await _migrate(hass, _seeded("dryer", minor))
    assert entry.options[CONF_WATCHDOG_INTERVAL] == 61
    assert entry.options[CONF_START_DURATION_THRESHOLD] == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("device_type", ["dryer", "washing_machine", "dishwasher"])
async def test_3_1_3_5_and_3_9_land_on_the_same_cadence_and_detector(
    hass: HomeAssistant, device_type: str
) -> None:
    """Parity: whichever path an entry takes, it runs the same detector."""
    results = {m: await _migrate(hass, _seeded(device_type, m)) for m in (1, 5, 9)}
    cadence = {
        m: (e.options[CONF_WATCHDOG_INTERVAL], e.options[CONF_START_DURATION_THRESHOLD])
        for m, e in results.items()
    }
    assert len(set(cadence.values())) == 1, cadence
    configs = {
        m: build_detector_config(e.options, e.data, device_type)
        for m, e in results.items()
    }
    assert configs[1] == configs[9]
    assert configs[5] == configs[9]


@pytest.mark.asyncio
async def test_one_pass_path_keeps_a_deliberate_cadence(hass: HomeAssistant) -> None:
    entry = _seeded("dryer", 1)
    entry.options[CONF_WATCHDOG_INTERVAL] = 90
    entry.options[CONF_START_DURATION_THRESHOLD] = 45
    entry = await _migrate(hass, entry)
    assert entry.options[CONF_WATCHDOG_INTERVAL] == 90
    assert entry.options[CONF_START_DURATION_THRESHOLD] == 45
