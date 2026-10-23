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
"""#478: native Home Assistant device triggers.

Automations keyed on the ha_washdata_* bus events only ran with "Fire HA events"
on, and needed the entry id typed by hand. That option is retired (the events
always fire), and a WashData device offers device triggers (cycle started /
finished / ended, anti-wrinkle started, cycle stalled, unload pending) that fire
once per occurrence and carry the variables a relay or notification automation
needs (``trigger.switch_entity``). ``cycle_finished`` skips an interrupted false
start, like the finished push; ``cycle_ended`` follows every announced start, so
a "while running" automation is always switched back off.

Everything here runs through Home Assistant's real setup, device registry,
device-automation listing and the automation integration.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import voluptuous as vol
from homeassistant.components.device_automation import DeviceAutomationType
from homeassistant.components.device_automation import trigger as da_trigger
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_capture_events,
    async_get_device_automations,
)

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_FIRE_EVENTS,
    CONF_UNLOAD_TRACK_WITHOUT_DOOR,
    DOMAIN,
    EVENT_CYCLE_ENDED,
    EVENT_CYCLE_STARTED,
    SIGNAL_DEVICE_TRIGGER,
)

from .real_manager import POWER, feed

ALL_TYPES = {
    "cycle_started",
    "program_recognised",
    "phase_started",
    "finishing_soon",
    "cycle_overrun",
    "cycle_paused",
    "cycle_resumed",
    "cycle_stalled",
    "cycle_finished",
    "cycle_ended",
    "anti_wrinkle_started",
    "still_tumbling",
    "unload_pending",
    "unloaded",
    "state_changed",
    "maintenance_due",
    "milestone_reached",
    "pump_stuck",
}
# Matching-only types: a Threshold Device (`other`) recognises no program.
MATCHING = {"program_recognised", "phase_started", "finishing_soon", "cycle_overrun"}
TUMBLING = {"anti_wrinkle_started", "still_tumbling"}

FAST = {
    "min_power": 2.0,
    "off_delay": 60,
    "min_off_gap": 60,
    "watchdog_interval": 30,
    "completion_min_seconds": 300,
    # A legacy entry's stale "Fire HA events" off: inert since #478.
    CONF_NOTIFY_FIRE_EVENTS: False,
}


def _device_id(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    devices = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
    ours = [d for d in devices if (DOMAIN, entry.entry_id) in d.identifiers]
    assert len(ours) == 1
    return ours[0].id


async def _offered(hass: HomeAssistant, device_id: str) -> set[str]:
    triggers = await async_get_device_automations(
        hass, DeviceAutomationType.TRIGGER, device_id
    )
    return {t["type"] for t in triggers if t["domain"] == DOMAIN}


def _register_plug(hass: HomeAssistant, *switches: str) -> None:
    """A plug device carrying the power sensor and the given switch entities."""
    plug_entry = MockConfigEntry(domain="test")
    plug_entry.add_to_hass(hass)
    plug = dr.async_get(hass).async_get_or_create(
        config_entry_id=plug_entry.entry_id, identifiers={("test", "plug")}, name="Plug"
    )
    ent_reg = er.async_get(hass)
    ent_reg.async_get_or_create(
        "sensor", "test", "plug_power", suggested_object_id="plug_power",
        device_id=plug.id, config_entry=plug_entry,
    )
    for name in switches:
        ent_reg.async_get_or_create(
            "switch", "test", name, suggested_object_id=name,
            device_id=plug.id, config_entry=plug_entry,
        )


async def _automations(hass: HomeAssistant, device_id: str, *types: str) -> list[dict]:
    """One real automation per trigger type, recording what each run saw."""
    calls: list[dict] = []

    async def _record(call: ServiceCall) -> None:
        calls.append(dict(call.data))

    hass.services.async_register("test", "record", _record)
    keys = ("type", "device_id", "entry_id", "program", "status", "power_entity",
            "switch_entity", "label_applied", "platform", "domain")
    assert await async_setup_component(hass, "automation", {"automation": [
        {
            "alias": f"wd {t}",
            "triggers": [{"trigger": "device", "domain": DOMAIN,
                          "device_id": device_id, "type": t}],
            "actions": [{"action": "test.record", "data": {
                k: f"{{{{ trigger.get('{k}') }}}}"
                for k in (*keys, "duration", "energy_kwh")
            }}],
        }
        for t in types
    ]})
    await hass.async_block_till_done()
    return calls


def _of(calls: list[dict], trigger_type: str) -> list[dict]:
    return [c for c in calls if c["type"] == trigger_type]


# ── Listing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("device_type", "expected"),
    [
        ("washing_machine", ALL_TYPES - {"pump_stuck"}),
        ("dishwasher", ALL_TYPES - TUMBLING - {"pump_stuck"}),
        ("pump", ALL_TYPES - TUMBLING),
        ("other", ALL_TYPES - TUMBLING - MATCHING - {"pump_stuck"}),
        ("generic", ALL_TYPES - TUMBLING - {"pump_stuck"}),
        ("dryer", ALL_TYPES - {"pump_stuck"}),
        ("washer_dryer", ALL_TYPES - {"pump_stuck"}),
    ],
)
async def test_triggers_offered_per_device_type(
    hass: HomeAssistant, setup_washdata_entry, device_type: str, expected: set[str]
) -> None:
    entry = await setup_washdata_entry(device_type=device_type)
    assert await _offered(hass, _device_id(hass, entry)) == expected


async def test_an_unknown_type_is_rejected(hass: HomeAssistant, setup_washdata_entry) -> None:
    entry = await setup_washdata_entry()
    config = {"platform": "device", "domain": DOMAIN,
              "device_id": _device_id(hass, entry), "type": "cycle_exploded"}
    with pytest.raises(vol.Invalid):
        await da_trigger.async_validate_trigger_config(hass, config)
    config["type"] = "cycle_finished"
    assert (await da_trigger.async_validate_trigger_config(hass, config))["type"] == (
        "cycle_finished"
    )


# ── Firing through real automations ──────────────────────────────────────────


async def test_start_and_finish_fire_once_and_the_bus_events_too(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    _register_plug(hass, "plug_relay")
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    device_id = _device_id(hass, entry)
    calls = await _automations(
        hass, device_id, "cycle_started", "cycle_finished", "cycle_ended"
    )
    bus: list[str] = []
    for ev in (EVENT_CYCLE_STARTED, EVENT_CYCLE_ENDED):
        hass.bus.async_listen(ev, lambda e: bus.append(e.event_type))

    # 15 min of load: several match ticks, each of which could re-fire the start.
    await feed(hass, freezer, 500, 900)
    started = _of(calls, "cycle_started")
    assert len(started) == 1 and not _of(calls, "cycle_finished")
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()

    finished = _of(calls, "cycle_finished")
    assert len(_of(calls, "cycle_started")) == 1 and len(finished) == 1, calls
    ended = _of(calls, "cycle_ended")
    assert len(ended) == 1
    assert {k: v for k, v in ended[0].items() if k != "type"} == {
        k: v for k, v in finished[0].items() if k != "type"
    }
    assert bus == [EVENT_CYCLE_STARTED, EVENT_CYCLE_ENDED], "a stale False is inert"
    for c in (started[0], finished[0]):
        assert c["platform"] == "device" and c["domain"] == DOMAIN
        assert c["device_id"] == device_id and c["entry_id"] == entry.entry_id
        assert c["power_entity"] == POWER
        assert c["switch_entity"] == "switch.plug_relay"
        assert c["program"] is None  # no profiles: nothing matched
    assert finished[0]["status"] == "completed"
    assert finished[0]["label_applied"] is False
    assert 840 <= float(finished[0]["duration"]) <= 1200
    assert float(finished[0]["energy_kwh"]) > 0.05


async def test_an_interrupted_false_start_ends_but_does_not_finish(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(
        power_sensor=POWER, options={**FAST, "completion_min_seconds": 600}
    )
    calls = await _automations(
        hass, _device_id(hass, entry), "cycle_started", "cycle_finished", "cycle_ended"
    )

    await feed(hass, freezer, 500, 120)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()

    manager = hass.data[DOMAIN][entry.entry_id]
    assert manager.profile_store.get_past_cycles()[-1]["status"] == "interrupted"
    assert [c["type"] for c in calls] == ["cycle_started", "cycle_ended"]
    assert calls[1]["status"] == "interrupted"


def _listen(hass: HomeAssistant, entry: MockConfigEntry) -> list[tuple[str, dict]]:
    seen: list[tuple[str, dict]] = []
    # A @callback runs in the loop as the signal is sent; a plain lambda is run as
    # an executor job, so an assert straight after the send could find it empty.
    async_dispatcher_connect(
        hass, SIGNAL_DEVICE_TRIGGER.format(entry.entry_id),
        callback(lambda t, v: seen.append((t, v))),
    )
    return seen


async def test_a_discarded_ghost_still_ends_what_it_started(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """A sub-minute blip is never stored or announced as finished, but if its
    start went out, cycle_ended must follow (and only once)."""
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    seen = _listen(hass, entry)
    ghost = {"duration": 30.0, "status": "completed", "max_power": 3.0,
             "start_time": "2026-10-19T08:00:00+00:00", "power_data": []}

    bus = async_capture_events(hass, EVENT_CYCLE_ENDED)

    manager._cycle_ended_pending = True  # noqa: SLF001 - its start fired
    manager._on_cycle_end(dict(ghost))  # noqa: SLF001
    manager._on_cycle_end(dict(ghost))  # noqa: SLF001 - nothing owed any more
    await hass.async_block_till_done()

    assert [t for t, _ in seen] == ["cycle_ended"]
    assert seen[0][1]["status"] == "completed" and seen[0][1]["duration"] == 30.0
    assert not manager.profile_store.get_past_cycles()
    # The HA event owed to ha_washdata_cycle_started is sent too, once.
    assert len(bus) == 1
    assert bus[0].data["duration"] == 30.0 and "power_data" not in bus[0].data["cycle_data"]


async def test_cycle_ended_survives_a_failed_cycle_end(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    manager = hass.data[DOMAIN][entry.entry_id]
    seen = _listen(hass, entry)

    bus = async_capture_events(hass, EVENT_CYCLE_ENDED)

    with patch.object(manager, "_async_cycle_end_steps", side_effect=RuntimeError("x")):
        await manager._async_process_cycle_end(  # noqa: SLF001
            {"duration": 900.0, "status": "completed", "energy_wh": 120.0,
             "start_time": "2026-10-19T08:00:00+00:00"},
        )
    await hass.async_block_till_done()
    assert len(bus) == 1  # the steps never sent it, so the fallback does

    assert [t for t, _ in seen] == ["cycle_ended"]
    assert seen[0][1]["energy_kwh"] == 0.12


async def test_unload_pending_fires_once_on_entering_clean(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(
        power_sensor=POWER, options={**FAST, CONF_UNLOAD_TRACK_WITHOUT_DOOR: True}
    )
    calls = await _automations(hass, _device_id(hass, entry), "unload_pending")
    manager = hass.data[DOMAIN][entry.entry_id]

    await feed(hass, freezer, 500, 900)
    assert calls == []
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    assert manager.check_state() == "clean"
    # More idle readings re-write the state entity; the trigger stays at one.
    await feed(hass, freezer, 0, 120)
    assert len(calls) == 1 and calls[0]["type"] == "unload_pending"


async def test_anti_wrinkle_started_fires_on_entering_anti_wrinkle(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(
        "Dryer", device_type="dryer", power_sensor=POWER,
        options={**FAST, "anti_wrinkle_enabled": True},
    )
    calls = await _automations(hass, _device_id(hass, entry), "anti_wrinkle_started")
    manager = hass.data[DOMAIN][entry.entry_id]

    states: list[str] = []
    hass.bus.async_listen(
        "state_changed",
        lambda e: states.append(e.data["new_state"].state)
        if e.data["entity_id"] == "sensor.dryer_state" and e.data["new_state"] else None,
    )
    await feed(hass, freezer, 500, 900)
    assert calls == []
    # The quiet tail: anti-wrinkle, then off once its idle timeout passes.
    await feed(hass, freezer, 0, 600)
    await hass.async_block_till_done()
    assert "anti_wrinkle" in states and manager.check_state() == "off"
    assert len(calls) == 1 and calls[0]["type"] == "anti_wrinkle_started"


async def test_a_reload_mid_cycle_neither_refires_the_start_nor_loses_the_end(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    calls = await _automations(
        hass, _device_id(hass, entry), "cycle_started", "cycle_ended"
    )

    await feed(hass, freezer, 500, 600)
    assert len(calls) == 1
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    manager = hass.data[DOMAIN][entry.entry_id]
    assert manager.check_state() == "running", "the cycle was not restored"

    await feed(hass, freezer, 500, 600)
    assert len(calls) == 1, calls
    # The restored cycle still owes its cycle_ended.
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    assert [c["type"] for c in calls] == ["cycle_started", "cycle_ended"]


# ── Variables ────────────────────────────────────────────────────────────────


async def test_switch_entity_is_none_when_the_plug_has_two(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    _register_plug(hass, "relay_a", "relay_b")
    entry = await setup_washdata_entry(power_sensor=POWER, options=FAST)
    seen: list[dict[str, Any]] = []

    unsub = await da_trigger.async_attach_trigger(
        hass,
        {"platform": "device", "domain": DOMAIN,
         "device_id": _device_id(hass, entry), "type": "cycle_started"},
        lambda run_variables, context=None: seen.append(run_variables["trigger"]),
        {"domain": "automation", "name": "t", "home_assistant_start": False,
         "variables": {}, "trigger_data": {"id": "0", "idx": "0", "alias": None}},
    )
    await feed(hass, freezer, 500, 300)
    unsub()

    assert len(seen) == 1
    trig = seen[0]
    assert trig["switch_entity"] is None
    assert trig["device_name"] == "Washer" and trig["device_type"] == "washing_machine"
    assert trig["description"] == "Washer cycle started"
    assert {"id", "idx", "alias"} <= set(trig)
