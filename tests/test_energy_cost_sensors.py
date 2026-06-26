import pytest
from unittest.mock import MagicMock
from custom_components.ha_washdata.sensor import (
    WasherCycleEnergySensor,
    WasherCycleCostSensor,
    WasherLastCycleCostSensor,
    async_setup_entry,
)
from custom_components.ha_washdata.const import DOMAIN, STATE_RUNNING, STATE_OFF


def _make_manager(state=STATE_RUNNING, price=0.30, energy_wh=500.0, accumulated_cost=0.15):
    manager = MagicMock()
    manager.check_state.return_value = state
    manager.get_price.return_value = price
    manager._accumulated_cycle_cost = accumulated_cost
    manager._last_cycle_cost = None
    manager.detector._energy_since_idle_wh = energy_wh
    return manager


# --- Registration ---

@pytest.mark.asyncio
async def test_energy_cost_sensors_registered(hass, mock_config_entry):
    manager = _make_manager()
    hass.data[DOMAIN] = {mock_config_entry.entry_id: manager}
    async_add_entities = MagicMock()
    await async_setup_entry(hass, mock_config_entry, async_add_entities)
    keys = [e.entity_description.key for e in async_add_entities.call_args[0][0]]
    assert "cycle_energy" in keys
    assert "cycle_cost" in keys
    assert "last_cycle_cost" in keys


# --- WasherCycleEnergySensor ---

def test_cycle_energy_returns_value_when_running(mock_config_entry):
    manager = _make_manager(state=STATE_RUNNING, energy_wh=123.4)
    sensor = WasherCycleEnergySensor(manager, mock_config_entry)
    assert sensor.native_value == 123.4


def test_cycle_energy_returns_none_when_off(mock_config_entry):
    manager = _make_manager(state=STATE_OFF)
    sensor = WasherCycleEnergySensor(manager, mock_config_entry)
    assert sensor.native_value is None


def test_cycle_energy_unit(mock_config_entry):
    manager = _make_manager()
    sensor = WasherCycleEnergySensor(manager, mock_config_entry)
    assert sensor.entity_description.native_unit_of_measurement == "Wh"


# --- WasherCycleCostSensor ---

def test_cycle_cost_returns_value_when_running(hass, mock_config_entry):
    manager = _make_manager(state=STATE_RUNNING, accumulated_cost=0.1234)
    sensor = WasherCycleCostSensor(manager, mock_config_entry)
    sensor.hass = hass
    assert sensor.native_value == 0.1234


def test_cycle_cost_returns_none_when_off(hass, mock_config_entry):
    manager = _make_manager(state=STATE_OFF)
    sensor = WasherCycleCostSensor(manager, mock_config_entry)
    sensor.hass = hass
    assert sensor.native_value is None


def test_cycle_cost_returns_none_when_no_price(hass, mock_config_entry):
    manager = _make_manager(price=None)
    sensor = WasherCycleCostSensor(manager, mock_config_entry)
    sensor.hass = hass
    assert sensor.native_value is None


def test_cycle_cost_uses_hass_currency(hass, mock_config_entry):
    manager = _make_manager()
    sensor = WasherCycleCostSensor(manager, mock_config_entry)
    sensor.hass = hass
    assert sensor.native_unit_of_measurement == hass.config.currency


# --- WasherLastCycleCostSensor ---

def test_last_cycle_cost_returns_stored_value(hass, mock_config_entry):
    manager = _make_manager()
    manager._last_cycle_cost = 0.4567
    sensor = WasherLastCycleCostSensor(manager, mock_config_entry)
    sensor.hass = hass
    assert sensor.native_value == 0.4567


def test_last_cycle_cost_none_before_any_cycle(hass, mock_config_entry):
    manager = _make_manager()
    manager._last_cycle_cost = None
    sensor = WasherLastCycleCostSensor(manager, mock_config_entry)
    sensor.hass = hass
    assert sensor.native_value is None


def test_last_cycle_cost_attributes(hass, mock_config_entry):
    manager = _make_manager()
    manager._last_cycle_cost = 0.20
    manager.profile_store.get_past_cycles.return_value = [
        {
            "energy_wh": 800.0,
            "profile_name": "Cotton 40",
            "duration": 3600,
            "end_time": "2026-06-26T10:00:00",
        }
    ]
    sensor = WasherLastCycleCostSensor(manager, mock_config_entry)
    sensor.hass = hass
    attrs = sensor.extra_state_attributes
    assert attrs["energy_kwh"] == 0.8
    assert attrs["profile"] == "Cotton 40"
    assert attrs["duration_min"] == 60.0
    assert attrs["cycle_end"] == "2026-06-26T10:00:00"


def test_last_cycle_cost_attributes_none_when_no_history(hass, mock_config_entry):
    manager = _make_manager()
    manager.profile_store.get_past_cycles.return_value = []
    sensor = WasherLastCycleCostSensor(manager, mock_config_entry)
    sensor.hass = hass
    assert sensor.extra_state_attributes is None
