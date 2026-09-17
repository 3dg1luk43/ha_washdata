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
"""Issue #439 - an energy *meter* configured as the energy *price* entity.

The panel lists every ``sensor.`` in the price picker and a price entity outranks
the static price, so pointing it at the plug's own kWh counter costs every cycle
at ``energy * meter_reading``: the reporter's 1.30 kWh cycle came out at 7.22 EUR
on a 0.37 EUR/kWh tariff, because the counter happened to read 5.55 kWh.

The guard rejects only on *positive* evidence (same entity as this device's power
sensor or energy meter, an energy/power device class, a bare W/kWh unit) and falls
back to the static price. An entity HA has not set up yet has no attributes to
judge and must be left alone, or a tariff sensor that loads after us is silenced.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.manager import WashDataManager


def _entry(**options):
    entry = MagicMock()
    entry.entry_id = "test_entry_439"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.plug_power",
        "energy_sensor": "sensor.plug_energy",
        "energy_price_static": 0.37,
        **options,
    }
    entry.data = {}
    return entry


def _manager(hass, entry):
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        return WashDataManager(hass, entry)


def test_the_plugs_energy_meter_as_price_entity_falls_back_to_the_static_price(hass):
    """The reported configuration: both fields point at the same counter."""
    entry = _entry(energy_price_entity="sensor.plug_energy")
    mgr = _manager(hass, entry)
    hass.states.async_set(
        "sensor.plug_energy",
        "5.554",
        {"device_class": "energy", "unit_of_measurement": "kWh"},
    )

    assert mgr._price_entity_id() is None
    # 0.37, not 5.554: a 1.30 kWh cycle costs 0.48 EUR rather than 7.22 EUR.
    assert mgr._resolve_energy_price() == pytest.approx(0.37)


def test_the_power_sensor_as_price_entity_is_rejected(hass):
    entry = _entry(energy_price_entity="sensor.plug_power")
    mgr = _manager(hass, entry)
    hass.states.async_set(
        "sensor.plug_power", "2100", {"device_class": "power", "unit_of_measurement": "W"}
    )

    assert mgr._price_entity_id() is None
    assert mgr._resolve_energy_price() == pytest.approx(0.37)


@pytest.mark.parametrize(
    "attrs",
    [
        {"device_class": "energy", "unit_of_measurement": "kWh"},
        {"device_class": "power", "unit_of_measurement": "W"},
        {"device_class": "Energy"},  # HA hands these back lower-cased; be tolerant
        {"unit_of_measurement": "Wh"},  # no device class, unit alone is proof
        {"unit_of_measurement": " kW "},
    ],
)
def test_a_third_party_energy_or_power_sensor_is_rejected(hass, attrs):
    """Not this device's own entities, still provably not a price."""
    entry = _entry(energy_price_entity="sensor.other_meter")
    mgr = _manager(hass, entry)
    hass.states.async_set("sensor.other_meter", "12.5", attrs)

    assert mgr._price_entity_id() is None
    assert mgr._resolve_energy_price() == pytest.approx(0.37)


@pytest.mark.parametrize(
    "attrs",
    [
        {"unit_of_measurement": "EUR/kWh"},
        {"device_class": "monetary", "unit_of_measurement": "EUR"},
        {"unit_of_measurement": "ct/kWh"},
        {},  # a bare template sensor with no attributes at all
    ],
)
def test_a_real_tariff_sensor_still_outranks_the_static_price(hass, attrs):
    entry = _entry(energy_price_entity="sensor.tariff")
    mgr = _manager(hass, entry)
    hass.states.async_set("sensor.tariff", "0.42", attrs)

    assert mgr._price_entity_id() == "sensor.tariff"
    assert mgr._resolve_energy_price() == pytest.approx(0.42)


def test_an_entity_ha_has_not_loaded_yet_is_not_rejected(hass):
    """No attributes = no evidence. Rejecting here would kill a tariff integration
    that HA sets up after us, permanently, until a restart in the right order."""
    entry = _entry(energy_price_entity="sensor.tariff_late")
    mgr = _manager(hass, entry)

    assert mgr._price_entity_id() == "sensor.tariff_late"
    assert mgr._resolve_energy_price() == pytest.approx(0.37)  # no reading yet

    hass.states.async_set("sensor.tariff_late", "0.31", {"unit_of_measurement": "EUR/kWh"})
    assert mgr._resolve_energy_price() == pytest.approx(0.31)


async def test_a_rejected_entity_disables_dynamic_pricing_and_its_listener(hass):
    """Otherwise the timeline fills with meter readings and the recorder recost
    reproduces the same wrong cost from history."""
    entry = _entry(energy_price_entity="sensor.plug_energy", energy_price_dynamic=True)
    mgr = _manager(hass, entry)
    hass.states.async_set(
        "sensor.plug_energy", "5.554", {"device_class": "energy", "unit_of_measurement": "kWh"}
    )

    assert mgr._dynamic_pricing_enabled() is False
    await mgr._setup_price_listener()
    assert mgr._remove_price_listener is None
    assert await mgr._async_price_history(dt_util.now(), dt_util.now()) == []


def test_the_rejection_is_logged_once_per_misconfiguration(hass, caplog):
    entry = _entry(energy_price_entity="sensor.plug_energy")
    mgr = _manager(hass, entry)
    hass.states.async_set(
        "sensor.plug_energy", "5.554", {"device_class": "energy", "unit_of_measurement": "kWh"}
    )

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            mgr._price_entity_id()
    assert sum("is not a price per kWh" in r.message for r in caplog.records) == 1

    # A different bad entity is a new misconfiguration and warns again.
    entry.options = {**entry.options, "energy_price_entity": "sensor.plug_power"}
    hass.states.async_set(
        "sensor.plug_power", "2100", {"device_class": "power", "unit_of_measurement": "W"}
    )
    with caplog.at_level(logging.WARNING):
        mgr._price_entity_id()
    assert sum("is not a price per kWh" in r.message for r in caplog.records) == 2
