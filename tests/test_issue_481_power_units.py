"""Issue #481: a power sensor in kW, and an energy counter picked as the power sensor.

WashData read every power reading as watts, so a kW sensor's 2.1 (kW) looked like
2.1 W and no cycle ever started; the wiki's answer was a template sensor. A kWh
energy counter picked as the power sensor detects nothing at all, and nothing
said so.

Now a reading is scaled by its sensor's unit (kW, mW, ...) everywhere WashData
reads the power sensor: live, the recorder charts and the history import. A
device whose Min Power was set in the sensor's own unit (under 1 for kW) keeps
raw readings, so a setup made to work on raw kilowatts does not change. Setup,
reconfigure, the options form and the panel refuse an energy counter.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_capture_events

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.config_flow import ConfigFlow, OptionsFlowHandler
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_POWER_SENSOR,
    DOMAIN,
    SIGNAL_DEVICE_TRIGGER,
    is_energy_counter,
    power_unit_scale,
)
from homeassistant.const import CONF_NAME

POWER = "sensor.washer_power"
ENERGY = "sensor.washer_energy"


@pytest.mark.parametrize(
    ("unit", "min_power", "scale"),
    [
        ("W", 2.0, 1.0),
        (None, 2.0, 1.0),
        ("kW", 2.0, 1000.0),
        ("kw", 2.0, 1000.0),
        ("mW", 2.0, 0.001),
        ("MW", 2.0, 1_000_000.0),
        ("mw", 2.0, 1.0),  # milli or mega: ambiguous, stays watts
        # The Derivative helper over a kWh counter (an energy rate is power).
        ("kWh/h", 2.0, 1000.0),
        ("Wh/s", 2.0, 3600.0),
        ("kWh/min", 2.0, 60000.0),
        ("kWh/h", 0.005, 1.0),
        ("kWh/fortnight", 2.0, 1.0),
        # Thresholds set in the sensor's own unit: raw readings, as before.
        ("kW", 0.005, 1.0),
        ("mW", 2000.0, 1.0),
        # A broken Min Power does not stop the conversion.
        ("kW", "abc", 1000.0),
        ("kW", 10**400, 1000.0),
    ],
)
def test_power_unit_scale(unit, min_power, scale) -> None:
    assert power_unit_scale(unit, min_power) == scale


def test_is_energy_counter() -> None:
    assert is_energy_counter({"unit_of_measurement": "kWh"})
    assert is_energy_counter({"unit_of_measurement": "Wh", "device_class": "power"})
    assert is_energy_counter({"device_class": "energy"})
    assert not is_energy_counter({"unit_of_measurement": "kW", "device_class": "power"})
    assert not is_energy_counter({})
    assert not is_energy_counter(None)
    # A power unit wins over an energy device class set wrong on a template.
    assert not is_energy_counter({"device_class": "energy", "unit_of_measurement": "W"})
    assert not is_energy_counter({"device_class": "energy", "unit_of_measurement": "kWh/h"})
    assert is_energy_counter({"device_class": "energy_storage"})


async def test_an_entry_on_an_energy_counter_ignores_its_readings(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Set up on a kWh counter before setup refused one: its total never reads as
    power (it held a phantom cycle open), and the entry stays loaded."""
    entry = await setup_washdata_entry(power_sensor=ENERGY)
    manager = hass.data[DOMAIN][entry.entry_id]
    for i in range(10):
        freezer.tick(timedelta(seconds=30))
        hass.states.async_set(
            ENERGY, str(412.7 + i * 0.1),
            {"unit_of_measurement": "kWh", "device_class": "energy"}, force_update=True,
        )
        await hass.async_block_till_done()
    assert manager.detector.state == "off"
    assert not manager.diag_buffer.power_samples(0.0)


# ─── Live readings ────────────────────────────────────────────────────────────


async def _feed_kw(hass: HomeAssistant, freezer, kw: float, seconds: int) -> None:
    t = 0
    while t < seconds:
        freezer.tick(timedelta(seconds=10))
        hass.states.async_set(
            POWER, str(kw), {"unit_of_measurement": "kW"}, force_update=True
        )
        await hass.async_block_till_done()
        t += 10


async def test_a_kw_sensor_is_read_as_watts_and_starts_a_cycle(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]

    await _feed_kw(hass, freezer, 2.1, 300)

    assert manager.power_scale() == 1000.0
    assert manager.diag_buffer.power_samples(0.0)[-1][1] == pytest.approx(2100.0)
    assert manager.detector.state == "running"


async def test_a_device_tuned_for_raw_kw_keeps_raw_readings(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(
        power_sensor=POWER,
        options={CONF_MIN_POWER: 0.005, "stop_threshold_w": 0.01, "start_threshold_w": 0.05},
    )
    manager = hass.data[DOMAIN][entry.entry_id]

    await _feed_kw(hass, freezer, 2.1, 60)

    assert manager.power_scale() == 1.0
    assert manager.diag_buffer.power_samples(0.0)[-1][1] == pytest.approx(2.1)


# ─── History: CSV upload (recorder reads are scaled when fetched) ─────────────


def _csv(entity: str, watts: list[float]) -> str:
    rows = ["entity_id,state,last_changed"]
    for i, w in enumerate(watts):
        rows.append(f"{entity},{w},2026-10-20T10:{i:02d}:00+00:00")
    return "\n".join(rows) + "\n"


def test_a_csv_of_the_kw_sensor_is_read_as_watts() -> None:
    slot = {"source": "csv", "chunks": [_csv(POWER, [0.0, 0.4, 2.1, 1.9, 0.2, 0.0])]}
    samples, report = ws_api._history_samples(slot, POWER, None, 1000.0)  # noqa: SLF001
    assert max(p for _, p in samples if p is not None) == pytest.approx(2100.0)
    assert report["peak_w"] == 2100.0
    assert "looks_like_kw" not in report["warnings"]


def test_a_csv_of_another_entity_keeps_its_values_and_the_kw_hint() -> None:
    slot = {"source": "csv", "chunks": [_csv("sensor.other", [0.0, 0.4, 2.1, 1.9, 0.2, 0.0])]}
    samples, report = ws_api._history_samples(slot, POWER, None, 1000.0)  # noqa: SLF001
    assert report["peak_w"] == 2.1
    assert "looks_like_kw" in report["warnings"]


def test_power_scale_falls_back_to_watts_without_a_usable_factor() -> None:
    assert ws_api._power_scale(MagicMock()) == 1.0  # noqa: SLF001
    broken = MagicMock()
    broken.power_scale.side_effect = RuntimeError("x")
    assert ws_api._power_scale(broken) == 1.0  # noqa: SLF001
    ok = MagicMock()
    ok.power_scale.return_value = 1000.0
    assert ws_api._power_scale(ok) == 1000.0  # noqa: SLF001


# ─── Refusing an energy counter ───────────────────────────────────────────────


def _states(hass: HomeAssistant) -> None:
    hass.states.async_set(ENERGY, "412.7", {"unit_of_measurement": "kWh", "device_class": "energy"})
    hass.states.async_set(POWER, "2.1", {"unit_of_measurement": "kW", "device_class": "power"})


def _user_input(sensor: str) -> dict:
    return {CONF_NAME: "Washer", CONF_DEVICE_TYPE: "washing_machine",
            CONF_POWER_SENSOR: sensor, CONF_MIN_POWER: 2.0}


async def test_setup_refuses_an_energy_counter(hass: HomeAssistant) -> None:
    _states(hass)
    flow = ConfigFlow()
    flow.hass = hass
    flow.async_show_form = MagicMock(return_value={"type": "form"})
    flow.async_create_entry = MagicMock(return_value={"type": "create_entry"})

    await flow.async_step_user(_user_input(ENERGY))
    assert flow.async_show_form.call_args[1]["errors"] == {
        CONF_POWER_SENSOR: "power_sensor_energy"
    }
    assert not flow.async_create_entry.called

    await flow.async_step_user(_user_input(POWER))  # a kW power sensor is fine
    assert flow.async_create_entry.called


async def test_reconfigure_and_options_refuse_an_energy_counter(hass: HomeAssistant) -> None:
    _states(hass)
    entry = MockConfigEntry(domain=DOMAIN, title="Washer",
                            data={"name": "Washer", CONF_POWER_SENSOR: POWER})
    entry.add_to_hass(hass)

    flow = ConfigFlow()
    flow.hass = hass
    flow.async_show_form = MagicMock(return_value={"type": "form"})
    with patch.object(flow, "_get_reconfigure_entry", return_value=entry):
        await flow.async_step_reconfigure(_user_input(ENERGY))
    assert flow.async_show_form.call_args[1]["errors"] == {
        CONF_POWER_SENSOR: "power_sensor_energy"
    }

    handler = OptionsFlowHandler(entry)
    handler.hass = hass
    handler.async_show_form = MagicMock(return_value={"type": "form"})
    handler.async_create_entry = MagicMock(return_value={"type": "create_entry"})
    await handler.async_step_init(_user_input(ENERGY))
    assert handler.async_show_form.call_args[1]["errors"] == {
        CONF_POWER_SENSOR: "power_sensor_energy"
    }
    assert not handler.async_create_entry.called


async def _set_options(hass: HomeAssistant, entry: MockConfigEntry, options: dict) -> MagicMock:
    connection = MagicMock()
    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api.ws_set_options.__wrapped__(
            hass, connection, {"id": 1, "entry_id": entry.entry_id, "options": options}
        )
    return connection


async def test_the_panel_refuses_picking_an_energy_counter(hass: HomeAssistant) -> None:
    _states(hass)
    entry = MockConfigEntry(domain=DOMAIN, title="Washer",
                            data={CONF_POWER_SENSOR: POWER}, options={CONF_POWER_SENSOR: POWER})
    entry.add_to_hass(hass)

    connection = await _set_options(hass, entry, {CONF_POWER_SENSOR: ENERGY})
    assert entry.options[CONF_POWER_SENSOR] == POWER
    code, message = connection.send_error.call_args.args[1:]
    assert code == "invalid_power_sensor"
    assert ENERGY in message


async def test_an_entry_already_on_an_energy_counter_still_saves(hass: HomeAssistant) -> None:
    _states(hass)
    entry = MockConfigEntry(domain=DOMAIN, title="Washer",
                            data={CONF_POWER_SENSOR: ENERGY}, options={CONF_POWER_SENSOR: ENERGY})
    entry.add_to_hass(hass)

    connection = await _set_options(
        hass, entry, {CONF_POWER_SENSOR: ENERGY, CONF_MIN_POWER: 3.0}
    )
    assert not connection.send_error.called
    assert entry.options[CONF_MIN_POWER] == 3.0


async def test_a_recorder_read_is_scaled_before_it_is_rounded() -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace

    start = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    states = [
        SimpleNamespace(state="2.134", last_changed=start),
        SimpleNamespace(state="unavailable", last_changed=start + timedelta(seconds=10)),
    ]

    class _Instance:
        async def async_add_executor_job(self, fn, *args):
            return fn(*args)

    with patch("homeassistant.components.recorder.history.state_changes_during_period",
               return_value={POWER: states}), \
         patch("homeassistant.components.recorder.get_instance", return_value=_Instance()):
        rows = await ws_api._recorder_power(  # noqa: SLF001
            SimpleNamespace(), POWER, start, end_dt=start + timedelta(minutes=1),
            keep_unavailable=True, scale=1000.0,
        )
    assert rows == [(start.timestamp(), 2134.0), (start.timestamp() + 10, None)]


async def test_a_cycle_restored_on_an_energy_counter_is_dropped(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """A phantom cycle restored on an energy counter could never end (its readings
    are ignored) and would block a new power sensor until it did: setup drops it."""
    entry = await setup_washdata_entry(power_sensor=ENERGY)
    manager = hass.data[DOMAIN][entry.entry_id]
    hass.states.async_set(ENERGY, "412.7", {"unit_of_measurement": "kWh", "device_class": "energy"})
    manager.detector._state = "running"  # noqa: SLF001  # as the restore left it
    started = dt_util.utcnow() - timedelta(minutes=40)
    manager.detector._current_cycle_start = started  # noqa: SLF001
    manager._cycle_ended_pending = True  # noqa: SLF001  # its start was announced
    manager._current_program = "Cotton 40"  # noqa: SLF001
    manager._cycle_progress = 55.0  # noqa: SLF001
    manager._manual_program_active = True  # noqa: SLF001  # a hand-picked program
    manager._is_user_paused = True  # noqa: SLF001  # restored with a user pause
    ended = async_capture_events(hass, "ha_washdata_cycle_ended")
    triggers: list[str] = []
    async_dispatcher_connect(
        hass, SIGNAL_DEVICE_TRIGGER.format(entry.entry_id),
        callback(lambda t, _v: triggers.append(t)),
    )
    with patch.object(manager.profile_store, "async_clear_active_cycle", AsyncMock()) as clear:
        manager._read_power_state_at_setup()  # noqa: SLF001
        await hass.async_block_till_done()
    assert manager.detector.state == "off"
    clear.assert_awaited()
    # The end owed to its announced start goes out, as an interrupted cycle.
    assert triggers == ["cycle_ended"]
    assert len(ended) == 1 and ended[0].data["cycle_data"]["status"] == "interrupted"
    # The start the restore put on the detector, not "unknown, 0 s".
    assert ended[0].data["start_time"] == started.isoformat()
    assert ended[0].data["duration"] > 0
    # Nothing of the dropped cycle stays on the entities.
    assert manager._current_program == "off" and manager._cycle_progress == 0.0  # noqa: SLF001
    assert not manager._manual_program_active and not manager._is_user_paused  # noqa: SLF001
    assert manager.check_state() != "user_paused"


def test_the_recorder_scaler_asks_once_per_unit() -> None:
    manager = MagicMock()
    manager.power_scale.return_value = 1000.0
    scale = ws_api._power_scaler(manager)  # noqa: SLF001
    assert [scale("kW") for _ in range(5)] == [1000.0] * 5
    assert manager.power_scale.call_count == 1


async def test_a_kw_reading_that_overflows_as_watts_is_rejected(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    hass.states.async_set(POWER, "1e308", {"unit_of_measurement": "kW"})
    assert manager._state_power(hass.states.get(POWER)) is None  # noqa: SLF001
    hass.states.async_set(POWER, "2.1", {"unit_of_measurement": "kW"})
    assert manager._state_power(hass.states.get(POWER)) == pytest.approx(2100.0)  # noqa: SLF001


def test_a_csv_reading_that_overflows_as_watts_is_a_break() -> None:
    slot = {"source": "csv", "chunks": [_csv(POWER, [0.0, 0.4, 1e308, 1.9, 0.2, 0.0])]}
    samples, report = ws_api._history_samples(slot, POWER, None, 1000.0)  # noqa: SLF001
    assert all(p is None or p < 1e6 for _, p in samples)
    assert report["peak_w"] == pytest.approx(1900.0)


async def test_a_recorder_reading_that_is_not_finite_is_a_break() -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace

    start = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    states = [
        SimpleNamespace(state="2.1", last_changed=start),
        SimpleNamespace(state="1e308", last_changed=start + timedelta(seconds=10)),
        SimpleNamespace(state="inf", last_changed=start + timedelta(seconds=20)),
    ]

    class _Instance:
        async def async_add_executor_job(self, fn, *args):
            return fn(*args)

    with patch("homeassistant.components.recorder.history.state_changes_during_period",
               return_value={POWER: states}), \
         patch("homeassistant.components.recorder.get_instance", return_value=_Instance()):
        rows = await ws_api._recorder_power(  # noqa: SLF001
            SimpleNamespace(), POWER, start, end_dt=start + timedelta(minutes=1),
            keep_unavailable=True, scale=1000.0,
        )
    s0 = start.timestamp()
    assert rows == [(s0, 2100.0), (s0 + 10, None), (s0 + 20, None)]


async def test_a_recorder_read_scales_each_state_by_its_own_unit(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """A sensor that switched from W to kW keeps its old readings in watts."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    start = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    states = [
        SimpleNamespace(state="500", last_changed=start, attributes={"unit_of_measurement": "W"}),
        SimpleNamespace(state="2.1", last_changed=start + timedelta(seconds=10),
                        attributes={"unit_of_measurement": "kW"}),
    ]

    class _Instance:
        async def async_add_executor_job(self, fn, *args):
            return fn(*args)

    with patch("homeassistant.components.recorder.history.state_changes_during_period",
               return_value={POWER: states}), \
         patch("homeassistant.components.recorder.get_instance", return_value=_Instance()):
        rows = await ws_api._recorder_power(  # noqa: SLF001
            hass, POWER, start, end_dt=start + timedelta(minutes=1),
            scale=ws_api._power_scaler(manager),  # noqa: SLF001
        )
    assert rows == [(start.timestamp(), 500.0), (start.timestamp() + 10, 2100.0)]
