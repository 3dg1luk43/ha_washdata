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
"""#478: the automation blueprints shipped in ``blueprints/automation/``.

Every blueprint goes through Home Assistant's own loader and schemas, is
installed through HA's real ``blueprint/save`` and validated as an automation
against a real WashData device, so a blueprint HA would refuse cannot ship. The
dryer anti-crease blueprint is also run end to end. ``catalog.json`` and the
``ha_washdata/blueprints/list`` WS command are checked against the files.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from homeassistant.components.automation.config import (
    AUTOMATION_BLUEPRINT_SCHEMA,
    async_validate_config_item,
)
from homeassistant.components.blueprint.models import Blueprint
from homeassistant.components.blueprint.schemas import BLUEPRINT_SCHEMA
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from homeassistant.util import yaml as yaml_util
from homeassistant.util.yaml import Input
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
    async_mock_service,
)

import custom_components.ha_washdata  # noqa: F401  (import before the loader: phcc ships its own custom_components)
from custom_components.ha_washdata import blueprint_catalog
from custom_components.ha_washdata.const import DEVICE_TRIGGER_TYPES, DEVICE_TYPES, DOMAIN

COMPONENT = Path(blueprint_catalog.__file__).parent
BP_DIR = COMPONENT / "blueprints" / "automation"
CATALOG = json.loads((COMPONENT / "blueprints" / "catalog.json").read_text(encoding="utf-8"))
BLUEPRINT_IDS = sorted(p.stem for p in BP_DIR.glob("*.yaml"))
SOURCE_URL = (
    "https://github.com/3dg1luk43/ha_washdata/blob/main/"
    "custom_components/ha_washdata/blueprints/automation/{}.yaml"
)
# The device trigger types device_trigger.py offers.
TRIGGER_TYPES = set(DEVICE_TRIGGER_TYPES)
# A value for every input that has no default, so the automation can be saved.
REQUIRED_SAMPLES = {("cycle_running_indicator", "target_entity"): "light.laundry"}

HAS_DEVICE_TRIGGER = (COMPONENT / "device_trigger.py").exists()
needs_device_trigger = pytest.mark.skipif(
    not HAS_DEVICE_TRIGGER,
    reason="device_trigger.py (#478 WashData device triggers) is not in the tree yet",
)


def _load(bp_id: str) -> dict[str, Any]:
    return yaml_util.load_yaml_dict(BP_DIR / f"{bp_id}.yaml")


def _blueprint(bp_id: str) -> Blueprint:
    return Blueprint(
        _load(bp_id), path=f"{bp_id}.yaml", expected_domain="automation",
        schema=AUTOMATION_BLUEPRINT_SCHEMA,
    )


def _walk(node: Any):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _manifest_version() -> tuple[int, ...]:
    raw = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))["version"]
    return tuple(int(x) for x in re.findall(r"\d+", raw)[:3])


# ─── The files ───────────────────────────────────────────────────────────────


def test_there_are_blueprints() -> None:
    assert "dryer_anti_crease_power_cut" in BLUEPRINT_IDS
    assert "cycle_running_indicator" in BLUEPRINT_IDS


@pytest.mark.parametrize("bp_id", BLUEPRINT_IDS)
def test_blueprint_loads_with_home_assistant(bp_id: str) -> None:
    """HA's loader, its blueprint schema and the automation's own schema all accept it."""
    data = _load(bp_id)
    Blueprint(data, path=f"{bp_id}.yaml", expected_domain="automation", schema=BLUEPRINT_SCHEMA)
    bp = _blueprint(bp_id)
    assert bp.validate() is None  # no HA min_version above the one the tests run on
    meta = bp.metadata
    assert meta["author"]
    assert meta["source_url"] == SOURCE_URL.format(bp_id)
    # HA's current syntax: triggers/actions with trigger:/action: keys.
    assert "triggers" in data and "actions" in data, bp_id
    assert "trigger" not in data and "action" not in data, bp_id
    for node in _walk({"triggers": data["triggers"], "actions": data["actions"]}):
        if isinstance(node, dict):
            assert "platform" not in node and "service" not in node, (bp_id, node)


def _washdata_triggers(node: Any) -> list[dict[str, Any]]:
    """Every WashData trigger in a blueprint, the ones in wait_for_trigger too."""
    return [
        n for n in _walk(node)
        if isinstance(n, dict) and str(n.get("trigger", "")).startswith(f"{DOMAIN}.")
    ]


@pytest.mark.parametrize("bp_id", BLUEPRINT_IDS)
def test_device_input_and_triggers_are_washdata(bp_id: str) -> None:
    """Triggers v2: the integration triggers (``trigger: ha_washdata.<type>``),
    targeted at the chosen WashData device, with options from the inputs."""
    bp = _blueprint(bp_id)
    device = bp.inputs["washdata_device"]
    assert "default" not in device
    assert device["selector"]["device"]["filter"] == [{"integration": DOMAIN}]
    data = _load(bp_id)
    triggers = data["triggers"] if isinstance(data["triggers"], list) else [data["triggers"]]
    assert triggers
    found = _washdata_triggers(data)
    assert len(found) >= len(triggers)
    for trig in triggers:
        assert str(trig.get("trigger", "")).startswith(f"{DOMAIN}."), (bp_id, trig)
    assert not [n for n in _walk(data) if isinstance(n, dict) and n.get("trigger") == "device"]
    for trig in found:
        assert trig["trigger"].split(".", 1)[1] in TRIGGER_TYPES, (bp_id, trig)
        assert trig["target"] == {"device_id": Input("washdata_device")}, (bp_id, trig)
        assert set(trig) <= {"trigger", "id", "target", "options"}, (bp_id, trig)


def test_catalog_and_files_match() -> None:
    assert set(CATALOG) == set(BLUEPRINT_IDS)
    known = set(DEVICE_TYPES)
    for bp_id, meta in CATALOG.items():
        assert set(meta) == {"device_types", "min_version", "tags"}, bp_id
        types = meta["device_types"]
        assert types == ["all"] or (types and set(types) <= known), (bp_id, types)
        assert re.fullmatch(r"\d+\.\d+\.\d+", meta["min_version"]), bp_id
        version = tuple(int(x) for x in meta["min_version"].split("."))
        # 0.5.9 is the first release with WashData device triggers, and a blueprint
        # cannot need a release later than the next one.
        assert (0, 5, 9) <= version <= (*_manifest_version()[:2], _manifest_version()[2] + 1)
        assert isinstance(meta["tags"], list) and all(isinstance(t, str) for t in meta["tags"])
        # The description names the WashData version it needs.
        assert f"WashData {meta['min_version']} or newer" in _blueprint(bp_id).metadata["description"]


def test_no_em_dash() -> None:
    files = [*BP_DIR.glob("*.yaml"), COMPONENT / "blueprints" / "catalog.json"]
    for path in files:
        assert "\u2014" not in path.read_text(encoding="utf-8"), path.name


def test_catalog_describes_each_blueprint() -> None:
    data = blueprint_catalog.load_catalog()
    assert "error" not in data
    items = {item["id"]: item for item in data["blueprints"]}
    assert list(items) == list(CATALOG)
    for bp_id, item in items.items():
        bp = _blueprint(bp_id)
        assert item["yaml"] == (BP_DIR / f"{bp_id}.yaml").read_text(encoding="utf-8")
        assert item["path"] == f"ha_washdata/{bp_id}.yaml"
        assert item["source_url"] == SOURCE_URL.format(bp_id)
        assert item["name"] == bp.name
        assert [i["key"] for i in item["inputs"]] == list(bp.inputs)
        expected = all(
            isinstance(spec, dict) and "default" in spec
            for key, spec in bp.inputs.items() if key != "washdata_device"
        )
        assert item["one_click"] is expected, bp_id
    assert items["dryer_anti_crease_power_cut"]["one_click"] is True
    assert items["cycle_running_indicator"]["one_click"] is False


def test_catalog_survives_broken_files(tmp_path: Path) -> None:
    assert blueprint_catalog.load_catalog(tmp_path)["error"]
    (tmp_path / "automation").mkdir()
    (tmp_path / "catalog.json").write_text(json.dumps({
        "good": {"device_types": ["all"], "min_version": "0.5.9", "tags": []},
        "broken": {"device_types": ["all"], "min_version": "0.5.9", "tags": []},
        "missing": {"device_types": ["all"], "min_version": "0.5.9", "tags": []},
        "../escape": {"device_types": ["all"], "min_version": "0.5.9", "tags": []},
    }))
    (tmp_path / "automation" / "good.yaml").write_text(
        "blueprint:\n  name: Good\n  domain: automation\n  input:\n"
        "    section:\n      input:\n        a: {default: 1, selector: {number: {}}}\n"
        "    washdata_device: {selector: {device: {}}}\n"
    )
    (tmp_path / "automation" / "broken.yaml").write_text("blueprint: [\n")
    data = blueprint_catalog.load_catalog(tmp_path)
    assert "error" not in data
    assert [i["id"] for i in data["blueprints"]] == ["good"]
    good = data["blueprints"][0]
    assert [i["key"] for i in good["inputs"]] == ["a", "washdata_device"]
    assert good["one_click"] is True


# ─── The WS command ──────────────────────────────────────────────────────────


async def test_ws_blueprints_list(
    hass, setup_washdata_entry, hass_ws_client, hass_read_only_access_token
) -> None:
    """Any signed-in user gets the documented shape."""
    await setup_washdata_entry()
    client = await hass_ws_client(hass, hass_read_only_access_token)
    await client.send_json_auto_id({"type": f"{DOMAIN}/blueprints/list"})
    res = await client.receive_json()
    assert res["success"], res
    result = res["result"]
    assert set(result) == {"blueprints"}
    assert [b["id"] for b in result["blueprints"]] == list(CATALOG)
    for item in result["blueprints"]:
        assert set(item) == {
            "id", "name", "description", "device_types", "min_version", "tags",
            "source_url", "path", "yaml", "one_click", "inputs",
        }
        assert item["device_types"] == CATALOG[item["id"]]["device_types"]
        for inp in item["inputs"]:
            assert set(inp) == {"key", "name", "description", "selector", "has_default"}
    anti = next(b for b in result["blueprints"] if b["id"] == "dryer_anti_crease_power_cut")
    assert {i["key"]: i["selector"] for i in anti["inputs"]} == {
        "washdata_device": "device", "wait_minutes": "number", "low_power_w": "number",
        "off_seconds": "number", "turn_back_on": "boolean", "presence_entities": "entity",
        "plug_switch": "entity", "extra_actions": "action",
    }


async def test_ws_blueprints_list_reports_an_unreadable_catalog(
    hass, setup_washdata_entry, hass_ws_client, monkeypatch, tmp_path
) -> None:
    await setup_washdata_entry()
    monkeypatch.setattr(blueprint_catalog, "BLUEPRINT_ROOT", tmp_path)
    monkeypatch.setattr(
        blueprint_catalog, "load_catalog",
        lambda root=tmp_path, _real=blueprint_catalog.load_catalog: _real(root),
    )
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({"type": f"{DOMAIN}/blueprints/list"})
    res = await client.receive_json()
    assert res["success"], res
    assert res["result"]["blueprints"] == [] and res["result"]["error"]
    assert str(tmp_path) not in res["result"]["error"]  # no server path to the client


async def test_ws_blueprints_list_does_not_send_the_exception_text(
    hass, setup_washdata_entry, hass_ws_client, monkeypatch, caplog
) -> None:
    """The command is open to every signed-in user; an exception can carry a path."""
    await setup_washdata_entry()

    def _boom(*_a, **_k):
        raise FileNotFoundError("/config/custom_components/secret/catalog.json")

    monkeypatch.setattr(blueprint_catalog, "load_catalog", _boom)
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({"type": f"{DOMAIN}/blueprints/list"})
    res = await client.receive_json()
    assert res["success"], res
    assert res["result"]["error"] and "/config" not in res["result"]["error"]
    assert "secret/catalog.json" in caplog.text  # logged on the server instead


# ─── Installed and used on a real Home Assistant ─────────────────────────────


async def _device_id(hass, entry) -> str:
    device = dr.async_get(hass).async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    assert device is not None
    return device.id


def _instance(bp_id: str, device_id: str, **inputs: Any) -> dict[str, Any]:
    values = {"washdata_device": device_id, **inputs}
    for (owner, key), sample in REQUIRED_SAMPLES.items():
        if owner == bp_id:
            values.setdefault(key, sample)
    return {
        "id": f"t_{bp_id}",
        "alias": bp_id,
        "use_blueprint": {"path": f"ha_washdata/{bp_id}.yaml", "input": values},
    }


@needs_device_trigger
@pytest.mark.parametrize("bp_id", BLUEPRINT_IDS)
async def test_installed_blueprint_validates_as_an_automation(
    hass, setup_washdata_entry, hass_ws_client, tmp_path, bp_id: str
) -> None:
    """The panel's install path: HA's blueprint/save, then an automation with
    only the device (and the required inputs) filled in, through the validation
    HA's automation editor runs before it saves."""
    hass.config.config_dir = str(tmp_path)
    entry = await setup_washdata_entry("Dryer", device_type="dryer")
    assert await async_setup_component(hass, "automation", {})
    client = await hass_ws_client(hass)
    text = (BP_DIR / f"{bp_id}.yaml").read_text(encoding="utf-8")
    await client.send_json_auto_id({
        "type": "blueprint/save", "domain": "automation",
        "path": f"ha_washdata/{bp_id}", "yaml": text, "source_url": SOURCE_URL.format(bp_id),
    })
    res = await client.receive_json()
    assert res["success"], res
    assert (tmp_path / "blueprints" / "automation" / "ha_washdata" / f"{bp_id}.yaml").is_file()

    blueprint = _blueprint(bp_id)
    missing = [
        key for key, spec in blueprint.inputs.items()
        if key != "washdata_device" and not (isinstance(spec, dict) and "default" in spec)
        and (bp_id, key) not in REQUIRED_SAMPLES
    ]
    assert not missing, f"add a sample value for {missing} to REQUIRED_SAMPLES"

    config = _instance(bp_id, await _device_id(hass, entry))
    validated = await async_validate_config_item(hass, config["id"], config)
    assert validated is not None
    assert validated.validation_status == "ok", validated.validation_error


# ─── The dryer anti-crease blueprint, run end to end ─────────────────────────

POWER = "sensor.dryer_power"
PLUG = "switch.dryer_plug"


async def _anti_crease(
    hass, setup_washdata_entry, tmp_path, *, plug_device=True,
    options: dict[str, Any] | None = None, **inputs: Any,
):
    """A dryer, its installed blueprint and one automation from it, every input
    but the waits left at its default. With ``plug_device`` the power sensor and
    the plug's switch share a device, as on a smart plug, so the blueprint finds
    the switch through ``trigger.switch_entity``."""
    hass.config.config_dir = str(tmp_path)
    folder = tmp_path / "blueprints" / "automation" / "ha_washdata"
    folder.mkdir(parents=True)
    shutil.copy(BP_DIR / "dryer_anti_crease_power_cut.yaml", folder)
    if plug_device:
        plug_entry = MockConfigEntry(domain="test")
        plug_entry.add_to_hass(hass)
        plug = dr.async_get(hass).async_get_or_create(
            config_entry_id=plug_entry.entry_id, identifiers={("test", "plug")},
        )
        ent_reg = er.async_get(hass)
        for domain, object_id in (("sensor", "dryer_power"), ("switch", "dryer_plug")):
            ent_reg.async_get_or_create(
                domain, "test", object_id, suggested_object_id=object_id,
                device_id=plug.id, config_entry=plug_entry,
            )
    entry = await setup_washdata_entry(
        "Dryer", device_type="dryer", power_sensor=POWER, options=options
    )
    hass.states.async_set(POWER, "0" if options else "3", {"unit_of_measurement": "W"})
    assert await async_setup_component(hass, "system_log", {})
    calls = {
        "off": async_mock_service(hass, "switch", "turn_off"),
        "on": async_mock_service(hass, "switch", "turn_on"),
    }
    done = []
    hass.bus.async_listen("wd_test_done", callback(lambda e: done.append(e.data)))
    values = {
        "wait_minutes": 1, "off_seconds": 2,
        "extra_actions": [{"event": "wd_test_done",
                           "event_data": {"on": "{{ switched_back_on }}"}}],
        **inputs,
    }
    config = _instance("dryer_anti_crease_power_cut", await _device_id(hass, entry), **values)
    assert await async_setup_component(hass, "automation", {"automation": [config]})
    await hass.async_block_till_done()
    assert hass.states.get("automation.dryer_anti_crease_power_cut").state == "on"
    return entry, calls, done


async def _settle() -> None:
    """Let the automation run to its next wait. Not async_block_till_done: that
    waits for the run itself, which is parked in a delay until time moves."""
    for _ in range(20):
        await asyncio.sleep(0)


async def _advance(hass, seconds: float) -> None:
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=seconds))
    await _settle()


async def _set_power(hass, watts: str) -> None:
    hass.states.async_set(POWER, watts, {"unit_of_measurement": "W"})
    await _settle()


async def _fire(hass, entry, trigger_type: str, **variables: Any) -> None:
    """Send a WashData trigger through the manager's own hook, so the trigger
    variables (power and switch entity) are the real ones."""
    hass.data[DOMAIN][entry.entry_id]._fire_device_trigger(  # noqa: SLF001
        trigger_type, None, **variables
    )
    await _settle()


# still_tumbling as the manager sends it for a tumble 30 min after the finish.
TUMBLE = {"minutes_since_finish": 30.0, "bursts": 4, "burst_seconds": 40,
          "_minutes": 30.0, "_prev_minutes": None}


async def _tumble(hass, entry, **overrides: Any) -> None:
    await _fire(hass, entry, "still_tumbling", **{**TUMBLE, **overrides})


@needs_device_trigger
async def test_anti_crease_cuts_and_restores_the_plug(hass, setup_washdata_entry, tmp_path) -> None:
    entry, calls, done = await _anti_crease(hass, setup_washdata_entry, tmp_path)
    await _set_power(hass, "180")  # the next tumble began before the run
    await _tumble(hass, entry)
    assert not calls["off"]  # waits for a low-power moment
    await _set_power(hass, "2.5")  # between two tumbles
    assert [c.data["entity_id"] for c in calls["off"]] == [[PLUG]]
    assert not calls["on"]
    await _advance(hass, 3)
    assert [c.data["entity_id"] for c in calls["on"]] == [[PLUG]]
    assert done == [{"on": True}]


@needs_device_trigger
async def test_anti_crease_reads_a_kw_sensor_as_watts(hass, setup_washdata_entry, tmp_path) -> None:
    """0.18 kW is a 180 W tumble, not 0.18 W under the 10 W threshold (#481)."""
    entry, calls, _done = await _anti_crease(hass, setup_washdata_entry, tmp_path)
    hass.states.async_set(POWER, "0.18", {"unit_of_measurement": "kW"})
    await _settle()
    await _tumble(hass, entry)
    assert not calls["off"]  # still tumbling at 180 W
    hass.states.async_set(POWER, "0.0025", {"unit_of_measurement": "kW"})
    await _settle()
    assert [c.data["entity_id"] for c in calls["off"]] == [[PLUG]]


@needs_device_trigger
async def test_anti_crease_waits_for_its_minutes(hass, setup_washdata_entry, tmp_path) -> None:
    """The wait is the trigger's option now: a tumble before it does nothing,
    the first one after it does."""
    entry, calls, done = await _anti_crease(
        hass, setup_washdata_entry, tmp_path, wait_minutes=20
    )
    await _tumble(hass, entry, _minutes=8.0, _prev_minutes=None, minutes_since_finish=8.0)
    await _advance(hass, 3)
    assert not calls["off"] and not done
    await _tumble(hass, entry, _minutes=21.0, _prev_minutes=8.0, minutes_since_finish=21.0)
    assert [c.data["entity_id"] for c in calls["off"]] == [[PLUG]]
    await _advance(hass, 3)
    assert done == [{"on": True}]


@needs_device_trigger
async def test_anti_crease_uses_the_chosen_switch(hass, setup_washdata_entry, tmp_path) -> None:
    entry, calls, done = await _anti_crease(
        hass, setup_washdata_entry, tmp_path, plug_device=False, plug_switch="switch.other_plug",
    )
    await _tumble(hass, entry)
    await _advance(hass, 3)
    assert [c.data["entity_id"] for c in calls["off"]] == [["switch.other_plug"]]
    assert [c.data["entity_id"] for c in calls["on"]] == [["switch.other_plug"]]
    assert done == [{"on": True}]


@needs_device_trigger
async def test_anti_crease_leaves_the_plug_off_when_nobody_is_home(
    hass, setup_washdata_entry, tmp_path
) -> None:
    hass.states.async_set("person.alex", "not_home")
    hass.states.async_set("person.sam", "work")
    entry, calls, done = await _anti_crease(
        hass, setup_washdata_entry, tmp_path, presence_entities=["person.alex", "person.sam"],
    )
    await _tumble(hass, entry)
    assert len(calls["off"]) == 1
    await _advance(hass, 3)
    assert not calls["on"]
    assert done == [{"on": False}]


@needs_device_trigger
async def test_anti_crease_is_cancelled_by_a_new_cycle(hass, setup_washdata_entry, tmp_path) -> None:
    entry, calls, done = await _anti_crease(hass, setup_washdata_entry, tmp_path)
    await _set_power(hass, "250")
    await _tumble(hass, entry)
    await _fire(hass, entry, "cycle_started")
    await _set_power(hass, "1")
    await _advance(hass, 16 * 60)
    assert not calls["off"] and not calls["on"] and not done


@needs_device_trigger
async def test_anti_crease_gives_up_when_the_power_never_drops(
    hass, setup_washdata_entry, tmp_path
) -> None:
    entry, calls, done = await _anti_crease(hass, setup_washdata_entry, tmp_path)
    await _set_power(hass, "250")
    await _tumble(hass, entry)
    await _advance(hass, 15 * 60 + 1)
    await _set_power(hass, "1")
    assert not calls["off"] and not done


@needs_device_trigger
async def test_anti_crease_without_a_plug_switch_logs_and_stops(
    hass, setup_washdata_entry, tmp_path, caplog
) -> None:
    entry, calls, done = await _anti_crease(hass, setup_washdata_entry, tmp_path, plug_device=False)
    await _tumble(hass, entry)
    assert not calls["off"] and not done
    assert "no plug switch found" in caplog.text


@needs_device_trigger
async def test_anti_crease_end_to_end_on_a_real_dryer(
    hass, setup_washdata_entry, tmp_path, freezer
) -> None:
    """The whole chain on WashData's real detector: a dryer cycle finishes, the
    drum keeps tumbling, and the first tumble past the wait cuts the plug once."""
    entry, calls, done = await _anti_crease(
        hass, setup_washdata_entry, tmp_path, wait_minutes=10, off_seconds=1,
        options=REAL_DRYER,
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    await _feed(hass, freezer, 800, 900)
    await _feed(hass, freezer, 0, 300)
    assert manager.profile_store.get_past_cycles()[-1]["status"] == "completed"
    assert manager._tumble_tracker.armed  # noqa: SLF001
    # Two tumbles before the wait: nothing.
    for _ in range(2):
        await _feed(hass, freezer, 200, 40, step=20)
        await _feed(hass, freezer, 0, 120)
    assert not calls["off"]
    # Past 10 min: the next tumble ends in a cut, once. From here a run may be
    # parked in the blueprint's power-off delay, which only a moving clock ends.
    await _feed(hass, freezer, 0, 300)
    await _feed(hass, freezer, 200, 40, step=20, clock=True)
    await _feed(hass, freezer, 0, 180, clock=True)
    assert [c.data["entity_id"] for c in calls["off"]] == [[PLUG]]
    assert done == [{"on": True}]
    await _feed(hass, freezer, 200, 40, step=20, clock=True)
    await _feed(hass, freezer, 0, 180, clock=True)
    assert len(calls["off"]) == 1


# A dryer on WashData's real detector, its timing shortened like the testbox's.
REAL_DRYER = {
    "min_power": 5.0, "off_delay": 60, "min_off_gap": 60,
    "watchdog_interval": 30, "completion_min_seconds": 300,
}


async def _feed(
    hass, freezer, watts: float, seconds: int, step: int = 30, clock: bool = False
) -> None:
    """Report ``watts`` on the dryer's power sensor every ``step`` s for ``seconds``.

    ``clock``: settle by moving the clock a second at a time while waiting, so an
    automation run parked in a delay finishes instead of blocking the wait."""
    t = 0
    while t < seconds:
        freezer.tick(timedelta(seconds=step))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        async_fire_time_changed(hass, dt_util.utcnow())
        if clock:
            await _block_with_clock(hass, freezer)
        else:
            await hass.async_block_till_done()
        t += step


async def _block_with_clock(hass, freezer) -> None:
    # The loop clock is frozen too, so only zero sleeps; the short real pause
    # lets an executor job (the cycle-end match) finish between two looks.
    waiter = asyncio.ensure_future(hass.async_block_till_done())
    for attempt in range(200):
        await asyncio.sleep(0)
        if waiter.done():
            break
        time.sleep(0.001)
        if attempt % 10 == 9:
            freezer.tick(timedelta(seconds=1))
            async_fire_time_changed(hass, dt_util.utcnow())
    await waiter
