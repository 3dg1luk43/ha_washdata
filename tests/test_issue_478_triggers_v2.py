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
"""#478 triggers v2: ``trigger: ha_washdata.<type>`` and the full trigger set.

WashData is a Home Assistant trigger platform (``trigger.py``), so it is listed
under "Integrations" in the add-trigger dialog and its triggers under a WashData
device, entity or area. The device triggers stay, with the same types and the
options as extra fields; both listen to the manager's one signal. Every type
fires once per occurrence, never again after a restart, and carries the common
variables plus its own.

Everything runs through Home Assistant's real trigger helper, automation
integration, device registry and the real manager and detector (no cycle_data).
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import voluptuous as vol
from homeassistant.components.device_automation import DeviceAutomationType
from homeassistant.components.device_automation import trigger as da_trigger
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers import trigger as trigger_helper
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from homeassistant.util.yaml import load_yaml_dict
from pytest_homeassistant_custom_component.common import (
    async_fire_time_changed,
    async_get_device_automation_capabilities,
    async_get_device_automations,
)

import custom_components.ha_washdata  # noqa: F401  (before the loader: phcc ships its own custom_components)
from custom_components.ha_washdata import trigger as wd_trigger
from custom_components.ha_washdata import trigger_rules
from custom_components.ha_washdata.const import (
    DEVICE_TRIGGER_ONLY_FOR,
    DEVICE_TRIGGER_TYPES,
    DOMAIN,
)

COMPONENT = Path(trigger_rules.__file__).parent
POWER = "sensor.plug_power"
OTHER_POWER = "sensor.other_plug_power"
FAST = {
    "min_power": 2.0,
    "off_delay": 60,
    "min_off_gap": 60,
    "watchdog_interval": 30,
    "completion_min_seconds": 300,
}
# Every variable a WashData trigger can carry, recorded by the test automations.
KEYS = (
    "id", "platform", "type", "entry_id", "device_id", "device_name", "device_type",
    "program", "power_entity", "switch_entity", "description", "start_time",
    "previous_program", "time_remaining", "expected_end", "phase", "previous_phase",
    "minutes_left", "overrun_ratio", "expected_duration", "reason", "paused_seconds",
    "status", "duration", "energy_kwh", "end_time", "label_applied", "anomaly",
    "minutes_since_finish", "bursts", "burst_seconds", "how", "minutes_waiting",
    "from_state", "to_state", "task", "task_name", "due_reason", "cycle_count",
    "elapsed_seconds", "threshold_seconds",
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _device_id(hass: HomeAssistant, entry: Any) -> str:
    device = dr.async_get(hass).async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    assert device is not None
    return device.id


def _manager(hass: HomeAssistant, entry: Any) -> Any:
    return hass.data[DOMAIN][entry.entry_id]


async def _record(hass: HomeAssistant, triggers: dict[str, dict[str, Any]]) -> list[dict]:
    """One automation per labelled trigger (``trigger.id`` = the label), each
    recording every WashData variable it got."""
    calls: list[dict] = []

    async def _rec(call: ServiceCall) -> None:
        calls.append(dict(call.data))

    hass.services.async_register("test", "record", _rec)
    assert await async_setup_component(hass, "automation", {"automation": [
        {
            "alias": label,
            "mode": "parallel",
            "triggers": [{**conf, "id": label}],
            "actions": [{"action": "test.record", "data": {
                k: f"{{{{ trigger.get('{k}') }}}}" for k in KEYS
            }}],
        }
        for label, conf in triggers.items()
    ]})
    await hass.async_block_till_done()
    return calls


def _wd(trigger_type: str, device_id: str, **options: Any) -> dict[str, Any]:
    conf: dict[str, Any] = {
        "trigger": f"{DOMAIN}.{trigger_type}", "target": {"device_id": device_id},
    }
    if options:
        conf["options"] = options
    return conf


async def _validate(hass: HomeAssistant, conf: dict[str, Any]) -> list[dict[str, Any]]:
    """What the automation editor runs before it saves a trigger."""
    return await trigger_helper.async_validate_trigger_config(
        hass, cv.TRIGGER_SCHEMA([conf])
    )


def _of(calls: list[dict], label: str) -> list[dict]:
    return [c for c in calls if c["id"] == label]


async def _feed(
    hass: HomeAssistant, freezer: Any, watts: float, seconds: int,
    step: int = 30, sensor: str = POWER,
) -> None:
    """Report ``watts`` every ``step`` s for ``seconds``, timers included."""
    t = 0
    while t < seconds:
        freezer.tick(timedelta(seconds=step))
        hass.states.async_set(sensor, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        async_fire_time_changed(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        t += step


async def _send(hass: HomeAssistant, entry: Any, trigger_type: str, **variables: Any) -> None:
    """One occurrence through the manager's own firing path."""
    _manager(hass, entry)._fire_device_trigger(trigger_type, None, **variables)  # noqa: SLF001
    await hass.async_block_till_done()


# One occurrence of each type that passes its default options.
SAMPLES: dict[str, dict[str, Any]] = {
    "cycle_started": {"start_time": "2026-10-19T08:00:00+00:00"},
    "program_recognised": {"program": "Cotton", "previous_program": None,
                           "time_remaining": 3000, "expected_end": "2026-10-19T09:00:00+00:00"},
    "phase_started": {"phase": "Rinse", "previous_phase": "Wash"},
    "finishing_soon": {"minutes_left": 9, "expected_end": "2026-10-19T09:00:00+00:00",
                       "_from_minute": 11, "_to_minute": 9},
    "cycle_overrun": {"overrun_ratio": 1.21, "expected_duration": 3600,
                      "_from_percent": 15, "_to_percent": 21},
    "cycle_paused": {"reason": "user", "_previous_reason": None},
    "cycle_resumed": {"reason": "user", "paused_seconds": 60, "_new_reason": None,
                      "_user_paused_seconds": 60},
    "cycle_stalled": {},
    "cycle_finished": {"status": "completed", "duration": 900.0, "energy_kwh": 0.1,
                       "anomaly": "none", "label_applied": False},
    "cycle_ended": {"status": "completed", "duration": 900.0, "anomaly": "none"},
    "anti_wrinkle_started": {},
    "still_tumbling": {"minutes_since_finish": 20.0, "bursts": 3, "burst_seconds": 40,
                       "_minutes": 20.0, "_prev_minutes": None},
    "unload_pending": {},
    "unloaded": {"how": "door", "minutes_waiting": 12.5},
    "state_changed": {"from_state": "running", "to_state": "ending"},
    "maintenance_due": {"task": "descale", "task_name": "descale", "due_reason": "cycles"},
    "milestone_reached": {"cycle_count": 100},
    "pump_stuck": {"elapsed_seconds": 1900, "threshold_seconds": 1800},
}


def _device_type_for(trigger_type: str) -> str:
    only = DEVICE_TRIGGER_ONLY_FOR.get(trigger_type)
    return "washing_machine" if only is None or "washing_machine" in only else only[0]


# ── The platform and its descriptions ────────────────────────────────────────


def test_samples_cover_every_type() -> None:
    assert set(SAMPLES) == set(DEVICE_TRIGGER_TYPES)


async def test_platform_offers_every_type(hass: HomeAssistant) -> None:
    triggers = await wd_trigger.async_get_triggers(hass)
    assert list(triggers) == list(DEVICE_TRIGGER_TYPES)


def test_triggers_yaml_strings_and_icons_match_the_options() -> None:
    """triggers.yaml passes Home Assistant's own description schema, describes
    exactly the options trigger_rules validates, and every name, field, select
    option and icon it needs is in strings.json / translations/en.json."""
    raw = load_yaml_dict(str(COMPONENT / "triggers.yaml"))
    described = trigger_helper._TRIGGERS_DESCRIPTION_SCHEMA(raw)  # noqa: SLF001
    assert list(described) == list(DEVICE_TRIGGER_TYPES)
    strings = json.loads((COMPONENT / "strings.json").read_text(encoding="utf-8"))
    en = json.loads((COMPONENT / "translations" / "en.json").read_text(encoding="utf-8"))
    icons = json.loads((COMPONENT / "icons.json").read_text(encoding="utf-8"))
    for section in ("triggers", "selector", "device_automation"):
        assert strings[section] == en[section], section
    for key, desc in described.items():
        assert desc["target"] == {
            "entity": [{"integration": DOMAIN}], "device": [{"integration": DOMAIN}],
        }, key
        fields = desc.get("fields", {})
        schema_fields = {str(k): k for k in trigger_rules.OPTION_FIELDS.get(key, {})}
        assert set(fields) == set(schema_fields), key
        text = strings["triggers"][key]
        assert text["name"] and text["description"], key
        assert strings["device_automation"]["trigger_type"][key], key
        assert icons["triggers"][key]["trigger"].startswith("mdi:"), key
        schema = trigger_rules.options_schema(key)
        defaults = schema({})
        for name, field in fields.items():
            assert text["fields"][name]["name"], (key, name)
            assert strings["device_automation"]["extra_fields"][name], name
            if "default" in field:
                assert defaults[name] == field["default"], (key, name)
            select = field["selector"].get("select")
            if select:
                validator = trigger_rules.OPTION_FIELDS[key][schema_fields[name]]
                assert list(select["options"]) == list(validator.container), (key, name)
                labels = strings["selector"][select["translation_key"]]["options"]
                assert set(labels) == set(select["options"]), (key, name)
    assert "—" not in (COMPONENT / "triggers.yaml").read_text(encoding="utf-8")


def test_state_options_are_the_state_sensor_values() -> None:
    from custom_components.ha_washdata import sensor  # noqa: PLC0415

    text = Path(sensor.__file__).read_text(encoding="utf-8")
    block = text.split("options=[", 1)[1].split("]", 1)[0]
    names = [n.strip().rstrip(",") for n in block.split() if n.strip().startswith("STATE_")]
    from custom_components.ha_washdata import const  # noqa: PLC0415

    assert tuple(getattr(const, n) for n in names) == trigger_rules.WASHDATA_STATES


async def test_add_trigger_dialog_lists_washdata(
    hass: HomeAssistant, setup_washdata_entry, hass_ws_client
) -> None:
    """The frontend's trigger_platforms/subscribe (what feeds the add-trigger
    dialog) carries every WashData trigger with its target and fields: HA's own
    loader accepted triggers.yaml. A custom integration is not behind the Labs
    flag, which covers a fixed list of core domains."""
    from homeassistant.components import automation  # noqa: PLC0415

    await setup_washdata_entry()
    assert await async_setup_component(hass, "automation", {})
    assert not automation.is_disabled_experimental_trigger(hass, DOMAIN)
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({"type": "trigger_platforms/subscribe"})
    res = await client.receive_json()
    assert res["success"], res
    seen: dict[str, Any] = {}
    while not all(f"{DOMAIN}.{t}" in seen for t in DEVICE_TRIGGER_TYPES):
        msg = await client.receive_json()
        seen.update(msg["event"])
    for key in DEVICE_TRIGGER_TYPES:
        desc = seen[f"{DOMAIN}.{key}"]
        assert desc["target"]["device"] == [{"integration": DOMAIN}], key
        assert set(desc.get("fields", {})) == {
            str(k) for k in trigger_rules.OPTION_FIELDS.get(key, {})
        }, key


# ── Validation ───────────────────────────────────────────────────────────────


async def test_every_type_validates_and_attaches_with_each_target_kind(
    hass: HomeAssistant, setup_washdata_entry, caplog
) -> None:
    entry = await setup_washdata_entry()
    device_id = _device_id(hass, entry)
    area = ar.async_get(hass).async_create("Laundry")
    dr.async_get(hass).async_update_device(device_id, area_id=area.id)
    targets = {
        "device": {"device_id": device_id},
        "entity": {"entity_id": "sensor.washer_state"},
        "area": {"area_id": area.id},
    }
    triggers: dict[str, dict[str, Any]] = {}
    for key in DEVICE_TRIGGER_TYPES:
        for kind, target in targets.items():
            conf = {"trigger": f"{DOMAIN}.{key}", "target": target}
            validated = await _validate(hass, conf)
            assert validated[0]["options"] == trigger_rules.options_schema(key)({}), key
            triggers[f"{key}/{kind}"] = conf
    await _record(hass, triggers)
    assert len(hass.states.async_entity_ids("automation")) == len(triggers)
    assert all(s.state == "on" for s in hass.states.async_all("automation"))
    assert "Error setting up trigger" not in caplog.text
    assert "Got error" not in caplog.text


@pytest.mark.parametrize(
    "conf",
    [
        {"trigger": f"{DOMAIN}.cycle_exploded", "target": {"device_id": "x"}},
        {"trigger": f"{DOMAIN}.cycle_started"},
        {"trigger": f"{DOMAIN}.cycle_started", "target": {}},
        {"trigger": f"{DOMAIN}.cycle_started", "target": {"device_id": "x"},
         "options": {"minutes": 5}},
        {"trigger": f"{DOMAIN}.finishing_soon", "target": {"device_id": "x"},
         "options": {"minutes": 0}},
        {"trigger": f"{DOMAIN}.cycle_overrun", "target": {"device_id": "x"},
         "options": {"percent": 501}},
        {"trigger": f"{DOMAIN}.cycle_ended", "target": {"device_id": "x"},
         "options": {"status": "exploded"}},
        {"trigger": f"{DOMAIN}.state_changed", "target": {"device_id": "x"},
         "options": {"to": "spinning"}},
    ],
)
async def test_bad_configs_are_rejected(
    hass: HomeAssistant, setup_washdata_entry, conf: dict[str, Any]
) -> None:
    await setup_washdata_entry()
    with pytest.raises(vol.Invalid):
        await _validate(hass, conf)


# ── Firing on a real cycle ───────────────────────────────────────────────────


async def test_a_cycle_fires_each_trigger_once_with_its_variables(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(
        power_sensor=POWER, options={**FAST, "unload_track_without_door": True}
    )
    device_id = _device_id(hass, entry)
    calls = await _record(hass, {
        "started": _wd("cycle_started", device_id),
        "to_running": _wd("state_changed", device_id, to="running"),
        "from_ending": _wd("state_changed", device_id, **{"from": "ending"}),
        "any_state": _wd("state_changed", device_id),
        "finished": _wd("cycle_finished", device_id),
        "finished_cotton": _wd("cycle_finished", device_id, program="Cotton"),
        "ended_completed": _wd("cycle_ended", device_id, status="completed"),
        "ended_interrupted": _wd("cycle_ended", device_id, status="interrupted"),
        "unload_pending": _wd("unload_pending", device_id),
        "unloaded": _wd("unloaded", device_id),
    })

    await _feed(hass, freezer, 500, 900)
    started = _of(calls, "started")
    assert len(started) == 1
    assert started[0]["platform"] == f"{DOMAIN}.cycle_started"
    assert started[0]["type"] == "cycle_started"
    assert started[0]["device_id"] == device_id and started[0]["entry_id"] == entry.entry_id
    assert started[0]["description"] == "Washer cycle started"
    assert started[0]["power_entity"] == POWER
    assert dt_util.parse_datetime(started[0]["start_time"]) is not None
    assert [c["to_state"] for c in _of(calls, "to_running")] == ["running"]

    await _feed(hass, freezer, 0, 300)
    finished = _of(calls, "finished")
    assert len(finished) == 1 and not _of(calls, "finished_cotton")
    assert finished[0]["status"] == "completed" and finished[0]["anomaly"] == "none"
    assert len(_of(calls, "ended_completed")) == 1 and not _of(calls, "ended_interrupted")
    assert len(_of(calls, "unload_pending")) == 1
    assert len(_of(calls, "from_ending")) == 1
    states = [(c["from_state"], c["to_state"]) for c in _of(calls, "any_state")]
    assert states[0] == ("off", "starting") and states[-1][1] == "clean", states
    assert all(a != b for a, b in states)

    await _feed(hass, freezer, 0, 120)
    await hass.services.async_call(
        "button", "press", {"entity_id": "button.washer_mark_unloaded"}, blocking=True
    )
    await hass.async_block_till_done()
    unloaded = _of(calls, "unloaded")
    assert len(unloaded) == 1
    assert unloaded[0]["how"] == "button" and float(unloaded[0]["minutes_waiting"]) >= 2
    # Nothing repeats on more idle readings.
    before = len(calls)
    await _feed(hass, freezer, 0, 300)
    assert len(calls) == before + len(_of(calls[before:], "any_state"))


async def test_only_the_targeted_device_fires(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    washer = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    other = await setup_washdata_entry("Other", power_sensor=OTHER_POWER, options=FAST)
    calls = await _record(hass, {"started": _wd("cycle_started", _device_id(hass, washer))})

    await _feed(hass, freezer, 500, 300, sensor=OTHER_POWER)
    assert _manager(hass, other).check_state() == "running"
    assert calls == []
    await _feed(hass, freezer, 500, 300)
    assert [c["entry_id"] for c in calls] == [washer.entry_id]


async def test_area_floor_label_and_entity_targets_resolve_to_their_device(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    washer = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    other = await setup_washdata_entry("Other", power_sensor=OTHER_POWER, options=FAST)
    dev_reg = dr.async_get(hass)
    floor = (await _floor(hass)).floor_id
    area = ar.async_get(hass).async_create("Laundry", floor_id=floor)
    label = lr.async_get(hass).async_create("Appliances")
    dev_reg.async_update_device(_device_id(hass, washer), area_id=area.id)
    dev_reg.async_update_device(_device_id(hass, other), labels={label.label_id})
    calls = await _record(hass, {
        "area": {"trigger": f"{DOMAIN}.cycle_stalled", "target": {"area_id": area.id}},
        "floor": {"trigger": f"{DOMAIN}.cycle_stalled", "target": {"floor_id": floor}},
        "label": {"trigger": f"{DOMAIN}.cycle_stalled", "target": {"label_id": label.label_id}},
        "entity": {"trigger": f"{DOMAIN}.cycle_stalled",
                   "target": {"entity_id": "sensor.other_state"}},
        "plug": {"trigger": f"{DOMAIN}.cycle_stalled", "target": {"entity_id": POWER}},
    })
    await _send(hass, washer, "cycle_stalled")
    await _send(hass, other, "cycle_stalled")
    got = {(c["id"], c["entry_id"]) for c in calls}
    assert got == {
        ("area", washer.entry_id), ("floor", washer.entry_id),
        ("label", other.entry_id), ("entity", other.entry_id),
    }
    # Moving the device takes effect at once: the target is resolved per occurrence.
    dev_reg.async_update_device(_device_id(hass, other), area_id=area.id)
    calls.clear()
    await _send(hass, other, "cycle_stalled")
    assert {c["id"] for c in calls} == {"area", "floor", "label", "entity"}


async def _floor(hass: HomeAssistant) -> Any:
    from homeassistant.helpers import floor_registry as fr  # noqa: PLC0415

    return fr.async_get(hass).async_create("Ground")


# ── The live edges ───────────────────────────────────────────────────────────

PHASED = {
    "avg_duration": 3600,
    "phases": [
        {"name": "Wash", "start": 0, "end": 1800},
        {"name": "Rinse", "start": 1800, "end": 3600},
    ],
}


async def _running_washer(hass, setup_washdata_entry, freezer, **options: Any):
    entry = await setup_washdata_entry(power_sensor=POWER, options={**FAST, **options})
    manager = _manager(hass, entry)
    manager.profile_store._data["profiles"]["Cotton"] = dict(PHASED)  # noqa: SLF001
    manager.profile_store._data["profiles"]["Eco"] = dict(PHASED)  # noqa: SLF001
    return entry, manager


async def _show(hass, manager, **fields: Any) -> None:
    """Set what the manager shows and refresh, as a match tick does."""
    for name, value in fields.items():
        setattr(manager, f"_{name}", value)
    manager._notify_update()  # noqa: SLF001
    await hass.async_block_till_done()


async def test_program_phase_finishing_and_overrun_fire_once_each(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry, manager = await _running_washer(hass, setup_washdata_entry, freezer)
    device_id = _device_id(hass, entry)
    calls = await _record(hass, {
        "program": _wd("program_recognised", device_id),
        "cotton": _wd("program_recognised", device_id, program="cotton"),
        "phase": _wd("phase_started", device_id),
        "rinse": _wd("phase_started", device_id, phase="Rinse"),
        "soon10": _wd("finishing_soon", device_id),
        "soon5": _wd("finishing_soon", device_id, minutes=5),
        "over10": _wd("cycle_overrun", device_id, percent=10),
        "over20": _wd("cycle_overrun", device_id),
    })
    await _feed(hass, freezer, 500, 120)
    assert manager.check_state() == "running" and calls == []

    await _show(hass, manager, current_program="Cotton", matched_profile_duration=3600.0,
                time_remaining=3000.0, cycle_progress=10.0, overrun_ratio=0.2)
    program = _of(calls, "program")
    assert len(program) == 1 and len(_of(calls, "cotton")) == 1
    assert program[0]["program"] == "Cotton" and program[0]["previous_program"] is None
    assert program[0]["time_remaining"] == 3000 and program[0]["expected_end"]
    assert [c["phase"] for c in _of(calls, "phase")] == ["Wash"]
    assert not _of(calls, "soon10") and not _of(calls, "over10")

    await _show(hass, manager)  # nothing changed: nothing fires
    await _show(hass, manager, cycle_progress=60.0, time_remaining=700.0)
    assert [(c["phase"], c["previous_phase"]) for c in _of(calls, "phase")][-1] == (
        "Rinse", "Wash"
    )
    assert len(_of(calls, "rinse")) == 1
    assert not _of(calls, "soon10")
    await _show(hass, manager, time_remaining=590.0)
    soon = _of(calls, "soon10")
    assert len(soon) == 1 and soon[0]["minutes_left"] == 10 and soon[0]["expected_end"]
    await _show(hass, manager, time_remaining=300.0)
    await _show(hass, manager, time_remaining=120.0)
    assert len(_of(calls, "soon10")) == 1 and len(_of(calls, "soon5")) == 1
    assert _of(calls, "soon5")[0]["minutes_left"] == 5

    await _show(hass, manager, overrun_ratio=1.1)
    assert len(_of(calls, "over10")) == 1 and not _of(calls, "over20")
    await _show(hass, manager, overrun_ratio=1.25)
    await _show(hass, manager, overrun_ratio=1.6)
    over = _of(calls, "over20")
    assert len(over) == 1 and len(_of(calls, "over10")) == 1
    assert over[0]["overrun_ratio"] == 1.25 and over[0]["expected_duration"] == 3600

    # A switch is a new occurrence; the program filter keeps to its program.
    await _show(hass, manager, current_program="Eco")
    assert [(c["program"], c["previous_program"]) for c in _of(calls, "program")] == [
        ("Cotton", None), ("Eco", "Cotton"),
    ]
    assert len(_of(calls, "cotton")) == 1
    # A divergence revert to "detecting..." and back to Eco is no new program.
    await _show(hass, manager, current_program="detecting...")
    await _show(hass, manager, current_program="Eco")
    assert len(_of(calls, "program")) == 2


async def test_a_threshold_device_sends_no_matching_trigger(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(device_type="other", power_sensor=POWER, options=FAST)
    device_id = _device_id(hass, entry)
    calls = await _record(hass, {
        "program": _wd("program_recognised", device_id),
        "pump": _wd("pump_stuck", device_id),
        "started": _wd("cycle_started", device_id),
    })
    for trigger_type in ("program_recognised", "pump_stuck", "cycle_started"):
        await _send(hass, entry, trigger_type, **SAMPLES[trigger_type])
    assert [c["id"] for c in calls] == ["started"]


async def test_pause_and_resume(hass: HomeAssistant, setup_washdata_entry, freezer) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    device_id = _device_id(hass, entry)
    manager = _manager(hass, entry)
    calls = await _record(hass, {
        "paused": _wd("cycle_paused", device_id),
        "paused_any": _wd("cycle_paused", device_id, include_detected=True),
        "resumed": _wd("cycle_resumed", device_id),
        "resumed_any": _wd("cycle_resumed", device_id, include_detected=True),
    })
    await _feed(hass, freezer, 500, 300)
    assert await manager.async_pause_cycle()
    await hass.async_block_till_done()
    assert [c["reason"] for c in _of(calls, "paused")] == ["user"]
    assert [c["reason"] for c in _of(calls, "paused_any")] == ["user"]
    await _feed(hass, freezer, 500, 120)
    assert await manager.async_resume_cycle()
    await hass.async_block_till_done()
    resumed = _of(calls, "resumed")
    assert len(resumed) == 1 and resumed[0]["reason"] == "user"
    assert 110 <= float(resumed[0]["paused_seconds"]) <= 130
    assert len(_of(calls, "resumed_any")) == 1

    # A pause the detector sees in the power: only the include_detected pair.
    await _feed(hass, freezer, 0, 90)
    assert manager.check_state() == "paused", manager.check_state()
    await _feed(hass, freezer, 500, 60)
    assert manager.check_state() == "running"
    assert len(_of(calls, "paused")) == 1 and len(_of(calls, "resumed")) == 1
    assert [c["reason"] for c in _of(calls, "paused_any")] == ["user", "detected"]
    assert [c["reason"] for c in _of(calls, "resumed_any")] == ["user", "detected"]


async def test_a_reload_mid_cycle_replays_nothing(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry, manager = await _running_washer(hass, setup_washdata_entry, freezer)
    await manager.profile_store.async_save()
    device_id = _device_id(hass, entry)
    calls = await _record(hass, {
        t: _wd(t, device_id) for t in (
            "cycle_started", "program_recognised", "phase_started", "state_changed",
            "finishing_soon", "cycle_paused",
        )
    })
    await _feed(hass, freezer, 500, 120)
    assert await manager.async_pause_cycle()
    await _show(hass, manager, current_program="Cotton", matched_profile_duration=3600.0,
                time_remaining=500.0, cycle_progress=60.0)
    await manager.profile_store.async_save_active_cycle(
        manager._augment_active_snapshot(manager.detector.get_state_snapshot())  # noqa: SLF001
    )
    fired = sorted(c["id"] for c in calls)
    assert fired == sorted([
        "cycle_started", "state_changed", "state_changed", "state_changed",
        "cycle_paused", "program_recognised", "phase_started", "finishing_soon",
    ]), fired

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    manager = _manager(hass, entry)
    assert manager.check_state() == "user_paused"
    assert manager.current_program == "Cotton"
    await _feed(hass, freezer, 500, 60)
    await _show(hass, manager, time_remaining=480.0, cycle_progress=62.0)
    assert sorted(c["id"] for c in calls) == fired


async def test_maintenance_due_and_milestone(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options={
        **FAST, "maintenance_reminder_cycles": {"descale": 1}, "notify_milestones": [1, 5],
    })
    device_id = _device_id(hass, entry)
    calls = await _record(hass, {
        "due": _wd("maintenance_due", device_id),
        "due_descale": _wd("maintenance_due", device_id, task="Descale"),
        "due_salt": _wd("maintenance_due", device_id, task="salt"),
        "milestone": _wd("milestone_reached", device_id),
    })
    await _feed(hass, freezer, 500, 900)
    await _feed(hass, freezer, 0, 300)
    due = _of(calls, "due")
    assert len(due) == 1, calls
    assert (due[0]["task"], due[0]["due_reason"]) == ("descale", "cycles")
    assert len(_of(calls, "due_descale")) == 1 and not _of(calls, "due_salt")
    assert [c["cycle_count"] for c in _of(calls, "milestone")] == [1]
    await _feed(hass, freezer, 0, 300)
    assert len(_of(calls, "due")) == 1 and len(_of(calls, "milestone")) == 1


async def test_pump_stuck(hass: HomeAssistant, setup_washdata_entry, freezer) -> None:
    entry = await setup_washdata_entry(
        "Sump", device_type="pump", power_sensor=POWER,
        options={**FAST, "pump_stuck_duration": 240},
    )
    calls = await _record(hass, {"stuck": _wd("pump_stuck", _device_id(hass, entry))})
    await _feed(hass, freezer, 300, 600)
    stuck = _of(calls, "stuck")
    assert len(stuck) == 1, calls
    assert int(stuck[0]["threshold_seconds"]) == 240 and int(stuck[0]["elapsed_seconds"]) >= 240


# ── still_tumbling ───────────────────────────────────────────────────────────

DRYER = {**FAST, "min_power": 5.0}


async def _finished_dryer(hass, setup_washdata_entry, freezer, **options: Any):
    entry = await setup_washdata_entry(
        "Dryer", device_type="dryer", power_sensor=POWER, options={**DRYER, **options}
    )
    device_id = _device_id(hass, entry)
    calls = await _record(hass, {
        "tumbling": _wd("still_tumbling", device_id, minutes=10),
        "tumbling_default": _wd("still_tumbling", device_id),
    })
    await _feed(hass, freezer, 800, 900)
    await _feed(hass, freezer, 0, 300)
    assert _manager(hass, entry).profile_store.get_past_cycles()[-1]["status"] == "completed"
    return entry, calls


async def _tumble(hass, freezer, gap_s: int) -> None:
    """A 40 s tumble, then ``gap_s`` quiet."""
    await _feed(hass, freezer, 200, 40, step=20)
    await _feed(hass, freezer, 0, gap_s)


async def test_still_tumbling_fires_once_past_its_minutes(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    _entry, calls = await _finished_dryer(hass, setup_washdata_entry, freezer)
    await _tumble(hass, freezer, 120)
    await _tumble(hass, freezer, 120)
    assert calls == []  # ~9 min after the finish: before both waits
    await _tumble(hass, freezer, 120)
    await _tumble(hass, freezer, 120)
    tumbling = _of(calls, "tumbling")
    assert len(tumbling) == 1
    assert 10 <= float(tumbling[0]["minutes_since_finish"]) < 15
    assert int(tumbling[0]["bursts"]) >= 3
    assert tumbling[0]["description"] == "Dryer still tumbling"
    for _ in range(3):
        await _tumble(hass, freezer, 120)
    assert len(_of(calls, "tumbling")) == 1
    assert len(_of(calls, "tumbling_default")) == 1


async def test_still_tumbling_needs_tumbles(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    _entry, calls = await _finished_dryer(hass, setup_washdata_entry, freezer)
    await _feed(hass, freezer, 0, 1800, step=60)
    assert calls == []
    # A burst that is too long to be a tumble does not count either.
    await _feed(hass, freezer, 200, 400, step=20)
    await _feed(hass, freezer, 0, 120)
    assert not _of(calls, "tumbling")


async def test_still_tumbling_stops_at_a_new_cycle(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry, calls = await _finished_dryer(hass, setup_washdata_entry, freezer)
    await _feed(hass, freezer, 800, 900)  # the next load, past the 10 min mark
    await _feed(hass, freezer, 0, 300)
    assert len(_manager(hass, entry).profile_store.get_past_cycles()) == 2
    await _tumble(hass, freezer, 120)  # 25 min after the first finish, 6 after this one
    assert calls == []


async def test_a_tumble_detected_as_a_false_start_is_sent_when_it_ends(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Without anti-wrinkle a tumble long enough for the start gates is detected
    as a cycle; it ends as a false start (interrupted) and is a tumble after all."""
    entry, calls = await _finished_dryer(
        hass, setup_washdata_entry, freezer, start_duration_threshold=5.0,
    )
    started: list[str] = []
    hass.bus.async_listen(f"{DOMAIN}_cycle_started", lambda e: started.append("x"))
    await _feed(hass, freezer, 0, 600, step=60)
    await _feed(hass, freezer, 300, 90, step=15)
    assert started, "the tumble was not detected as a cycle start"
    await _feed(hass, freezer, 0, 300)
    manager = _manager(hass, entry)
    assert manager.profile_store.get_past_cycles()[-1]["status"] == "interrupted"
    assert len(_of(calls, "tumbling")) == 1


async def test_only_tumbling_appliances_send_it(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry("Dish", device_type="dishwasher", power_sensor=POWER)
    calls = await _record(hass, {"t": _wd("still_tumbling", _device_id(hass, entry))})
    await _send(hass, entry, "still_tumbling", **SAMPLES["still_tumbling"])
    assert calls == []
    assert not _manager(hass, entry)._tumble_tracker.armed  # noqa: SLF001


# ── Device triggers mirror every type ────────────────────────────────────────


@pytest.mark.parametrize("trigger_type", DEVICE_TRIGGER_TYPES)
async def test_device_trigger_mirrors_the_integration_trigger(
    hass: HomeAssistant, setup_washdata_entry, trigger_type: str
) -> None:
    entry = await setup_washdata_entry(
        device_type=_device_type_for(trigger_type), power_sensor=POWER, options=FAST
    )
    device_id = _device_id(hass, entry)
    offered = await async_get_device_automations(
        hass, DeviceAutomationType.TRIGGER, device_id
    )
    assert trigger_type in {t["type"] for t in offered}
    calls = await _record(hass, {
        "integration": _wd(trigger_type, device_id),
        "device": {"trigger": "device", "domain": DOMAIN, "device_id": device_id,
                   "type": trigger_type},
    })
    await _send(hass, entry, trigger_type, **SAMPLES[trigger_type])
    other = "cycle_stalled" if trigger_type != "cycle_stalled" else "unloaded"
    await _send(hass, entry, other, **SAMPLES[other])  # another type: not these
    integration, device = _of(calls, "integration"), _of(calls, "device")
    assert len(integration) == 1 and len(device) == 1, calls
    skip = {"id", "platform", "description"}
    assert {k: v for k, v in integration[0].items() if k not in skip} == {
        k: v for k, v in device[0].items() if k not in skip
    }
    assert device[0]["platform"] == "device"
    assert integration[0]["platform"] == f"{DOMAIN}.{trigger_type}"
    assert integration[0]["description"] == device[0]["description"]
    for name, value in SAMPLES[trigger_type].items():
        if not name.startswith("_") and name != "program":
            assert integration[0][name] == value, name
    assert not any(k.startswith("_") for k in integration[0])


@pytest.mark.parametrize("trigger_type", DEVICE_TRIGGER_TYPES)
async def test_device_trigger_extra_fields_are_the_options(
    hass: HomeAssistant, setup_washdata_entry, trigger_type: str
) -> None:
    entry = await setup_washdata_entry(device_type=_device_type_for(trigger_type))
    config = {"platform": "device", "domain": DOMAIN, "device_id": _device_id(hass, entry),
              "type": trigger_type}
    caps = await async_get_device_automation_capabilities(
        hass, DeviceAutomationType.TRIGGER, config
    )
    names = {f["name"] for f in caps.get("extra_fields", [])}
    assert names == {str(k) for k in trigger_rules.OPTION_FIELDS.get(trigger_type, {})}
    validated = await da_trigger.async_validate_trigger_config(hass, config)
    assert validated["type"] == trigger_type


async def test_device_trigger_options_filter(hass: HomeAssistant, setup_washdata_entry) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    device_id = _device_id(hass, entry)
    base = {"trigger": "device", "domain": DOMAIN, "device_id": device_id}
    calls = await _record(hass, {
        "soon5": {**base, "type": "finishing_soon", "minutes": 5},
        "soon10": {**base, "type": "finishing_soon", "minutes": 10},
        "interrupted": {**base, "type": "cycle_ended", "status": "interrupted"},
        "to_ending": {**base, "type": "state_changed", "to": "ending"},
    })
    await _send(hass, entry, "finishing_soon", **SAMPLES["finishing_soon"])
    await _send(hass, entry, "cycle_ended", **SAMPLES["cycle_ended"])
    await _send(hass, entry, "state_changed", **SAMPLES["state_changed"])
    assert sorted(c["id"] for c in calls) == ["soon10", "to_ending"]
    with pytest.raises(vol.Invalid):
        await da_trigger.async_validate_trigger_config(
            hass, {**base, "platform": "device", "type": "cycle_started", "minutes": 5}
        )


# ── The pure rules ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("trigger_type", "options", "variables", "fires"),
    [
        ("program_recognised", {"program": " cotton "}, {"program": "Cotton"}, True),
        ("program_recognised", {"program": "Eco"}, {"program": "Cotton"}, False),
        ("cycle_finished", {"program": "Eco"}, {"program": None}, False),
        ("phase_started", {"phase": "rinse"}, {"phase": "Rinse"}, True),
        ("phase_started", {"phase": "Spin"}, {"phase": "Rinse"}, False),
        ("finishing_soon", {"minutes": 10}, {"_from_minute": None, "_to_minute": 3}, True),
        ("finishing_soon", {"minutes": 10}, {"_from_minute": 10, "_to_minute": 9}, False),
        ("finishing_soon", {"minutes": 10}, {"_from_minute": 12, "_to_minute": 11}, False),
        ("cycle_overrun", {"percent": 20}, {"_from_percent": None, "_to_percent": 20}, True),
        ("cycle_overrun", {"percent": 20}, {"_from_percent": 20, "_to_percent": 30}, False),
        ("cycle_overrun", {"percent": 20}, {"_from_percent": 5, "_to_percent": 19}, False),
        ("cycle_paused", {}, {"reason": "detected", "_previous_reason": None}, False),
        ("cycle_paused", {"include_detected": True},
         {"reason": "user", "_previous_reason": "detected"}, False),
        ("cycle_resumed", {}, {"reason": "detected", "_new_reason": None}, False),
        ("cycle_resumed", {"include_detected": True},
         {"reason": "user", "_new_reason": "detected"}, False),
        ("cycle_ended", {"status": "any"}, {"status": "interrupted"}, True),
        ("cycle_ended", {"status": "completed"}, {"status": "force_stopped"}, False),
        ("still_tumbling", {"minutes": 15}, {"_minutes": 14.9, "_prev_minutes": None}, False),
        ("still_tumbling", {"minutes": 15}, {"_minutes": 30.0, "_prev_minutes": 16.0}, False),
        ("still_tumbling", {"minutes": 15}, {"_minutes": 15.0, "_prev_minutes": 3.0}, True),
        ("state_changed", {"from": "running", "to": "ending"},
         {"from_state": "paused", "to_state": "ending"}, False),
        ("maintenance_due", {"task": "Clean drum"},
         {"task": "drum_clean", "task_name": "clean drum"}, True),
        ("maintenance_due", {"task": "salt"}, {"task": "descale", "task_name": "descale"}, False),
    ],
)
def test_run_variables_filters(
    trigger_type: str, options: dict, variables: dict, fires: bool
) -> None:
    result = trigger_rules.run_variables(trigger_type, options, variables)
    assert (result is not None) is fires
    if result is not None:
        assert not any(k.startswith("_") for k in result)


def test_resumed_user_only_reports_the_user_pause() -> None:
    out = trigger_rules.run_variables(
        "cycle_resumed", {},
        {"reason": "user", "paused_seconds": 600, "_new_reason": "detected",
         "_user_paused_seconds": 120},
    )
    assert out == {"reason": "user", "paused_seconds": 120}


def test_marks() -> None:
    assert trigger_rules.remaining_mark(600) == 10
    assert trigger_rules.remaining_mark(600.4) == 11
    assert trigger_rules.remaining_mark(0) == 0
    assert trigger_rules.overrun_mark(1.2) == 20
    assert trigger_rules.overrun_mark(1.199) == 19
    assert trigger_rules.overrun_mark(0.5) == -50


def _inputs(**kw: Any) -> trigger_rules.LiveInputs:
    base: dict[str, Any] = {
        "state": "running", "cycle_active": True, "program": None, "phase": None,
        "phase_start_s": None, "pause_reason": None, "user_pause_since": None, "time_remaining_s": None,
        "remaining_s": None, "overrun_ratio": None, "expected_duration_s": None,
    }
    return trigger_rules.LiveInputs(**{**base, **kw})


def test_a_restore_without_marks_takes_the_first_estimate_as_announced() -> None:
    now = dt_util.utcnow()
    tracker = trigger_rules.LiveTriggerTracker()
    # A restored cycle shows its program and pause at once, the estimates later.
    tracker.baseline(_inputs(program="Cotton", pause_reason="user"), now)
    assert tracker.tick(_inputs(program="Cotton", pause_reason="user"), now) == []
    first = _inputs(program="Cotton", phase="Wash", phase_start_s=0.0, pause_reason="user",
                    time_remaining_s=300.0, remaining_s=300.0, overrun_ratio=1.3)
    assert tracker.tick(first, now) == []
    out = tracker.tick(_inputs(program="Cotton", phase="Wash", phase_start_s=0.0,
                               pause_reason="user", time_remaining_s=200.0,
                               remaining_s=200.0, overrun_ratio=1.5), now)
    assert [t for t, _ in out] == ["finishing_soon", "cycle_overrun"]
    assert out[0][1]["_from_minute"] == 5 and out[0][1]["_to_minute"] == 4


def test_a_restore_keeps_the_announced_marks() -> None:
    """The snapshot's marks: a phase the reset smoothing steps back over, and a
    remaining time that reads longer for a while, announce nothing twice."""
    now = dt_util.utcnow()
    before = trigger_rules.LiveTriggerTracker()
    before.baseline(_inputs(), now)
    before.new_cycle()
    before.tick(_inputs(program="Cotton", phase="Rinse", phase_start_s=1800.0,
                        time_remaining_s=500.0, remaining_s=500.0), now)
    marks = json.loads(json.dumps(before.marks()))  # through the snapshot store
    assert marks == {"program": "Cotton", "phase": "Rinse", "phase_start": 1800.0,
                     "minute": 9, "percent": None}

    after = trigger_rules.LiveTriggerTracker()
    after.baseline(_inputs(program="Cotton"), now, marks)
    for phase, start, remaining in (
        ("Wash", 0.0, 700.0), ("Rinse", 1800.0, 560.0), ("Rinse", 1800.0, 530.0),
    ):
        assert after.tick(_inputs(program="Cotton", phase=phase, phase_start_s=start,
                                  time_remaining_s=remaining, remaining_s=remaining), now) == []
    out = after.tick(_inputs(program="Cotton", phase="Spin", phase_start_s=3000.0,
                             time_remaining_s=100.0, remaining_s=100.0), now)
    assert [(t, v.get("phase"), v.get("_from_minute")) for t, v in out] == [
        ("phase_started", "Spin", None), ("finishing_soon", None, 9),
    ]


def test_a_new_cycle_announces_again() -> None:
    now = dt_util.utcnow()
    tracker = trigger_rules.LiveTriggerTracker()
    tracker.baseline(_inputs(program="Cotton", remaining_s=60.0), now)
    tracker.new_cycle()
    out = tracker.tick(_inputs(program="Cotton", remaining_s=3000.0), now)
    assert [t for t, _ in out] == ["program_recognised", "finishing_soon"]
    assert out[1][1]["_from_minute"] is None and out[1][1]["_to_minute"] == 50
    # Not while no cycle is announced.
    assert tracker.tick(_inputs(cycle_active=False, state="off", program="Eco"), now) == [
        ("state_changed", {"from_state": "running", "to_state": "off"}),
    ]


def test_tumble_tracker() -> None:
    t0 = dt_util.utcnow()
    tracker = trigger_rules.TumbleTracker()
    assert tracker.reading(300, t0, 6, 3) is None  # not armed
    tracker.arm(t0, 300.0, 6.0)  # still high at the finish: the cycle's own tail
    assert tracker.reading(0, t0 + timedelta(seconds=30), 6, 3) is None
    assert tracker.reading(4, t0 + timedelta(minutes=2), 6, 3) is None  # standby band
    assert tracker.reading(200, t0 + timedelta(minutes=3), 6, 3) is None
    out = tracker.reading(1, t0 + timedelta(minutes=3, seconds=40), 6, 3)
    assert out is not None and out["bursts"] == 1 and out["burst_seconds"] == 40
    assert out["_prev_minutes"] is None
    assert tracker.reading(200, t0 + timedelta(hours=5), 6, 3) is None
    assert not tracker.armed  # gave up after 4 h
