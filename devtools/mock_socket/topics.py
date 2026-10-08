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
"""The mock plug's MQTT contract: topics, discovery payloads, command parsing.

State lives under ``washdata_mock/<plug id>/``, never inside the discovery prefix. Two
discovery topics keep their pre-rebuild form, because WashData entries are configured
against the entities they created: the power sensor (``<prefix>/sensor/<id>_power``,
unique id ``<id>_power``) and the relay (``<prefix>/switch/<id>``, unique id
``<id>_switch``). Moving either renames the entity and orphans those entries.

Availability is the bridge's last will AND the plug's own topic (``availability_mode:
all``), so killing the mock and unplugging one plug both read as a dead plug. There is
deliberately no ``expire_after``: a silent plug (#424/#427) must stay a valid reading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

BASE = "washdata_mock"
BRIDGE_STATUS = f"{BASE}/status"
ONLINE, OFFLINE = "online", "offline"
#: Command topics, relative to the plug base: (suffix, action).
COMMANDS = {
    "relay/set": "relay",
    "connected/set": "connected",
    "program/set": "program",
    "scenario/set": "scenario",
    "mode/set": "mode",
    "start": "start",
    "stop": "stop",
}
SUBSCRIPTIONS = (f"{BASE}/+/+/set", f"{BASE}/+/start", f"{BASE}/+/stop")


@dataclass(frozen=True)
class PlugTopics:
    plug_id: str

    def _t(self, suffix: str) -> str:
        return f"{BASE}/{self.plug_id}/{suffix}"

    @property
    def power(self) -> str:
        return self._t("power")

    @property
    def energy(self) -> str:
        return self._t("energy")

    @property
    def availability(self) -> str:
        return self._t("availability")

    @property
    def relay(self) -> str:
        return self._t("relay")

    @property
    def connected(self) -> str:
        return self._t("connected")

    @property
    def program(self) -> str:
        return self._t("program")

    @property
    def scenario(self) -> str:
        return self._t("scenario")

    @property
    def mode(self) -> str:
        return self._t("mode")


def _config_topic(prefix: str, component: str, plug_id: str, key: str) -> str:
    # The relay is the plug itself: its topic is the plug id (the pre-rebuild switch).
    object_id = plug_id if component == "switch" and key == "relay" else f"{plug_id}_{key}"
    return f"{prefix}/{component}/{object_id}/config"


def discovery(
    plug_id: str,
    name: str,
    *,
    programs: list[str],
    scenarios: list[str],
    modes: list[str],
    prefix: str = "homeassistant",
) -> list[tuple[str, dict[str, Any]]]:
    """Retained discovery configs for one plug, as ``(topic, payload)``."""
    t = PlugTopics(plug_id)
    device = {
        "identifiers": [plug_id],
        "name": name,
        "manufacturer": "WashData",
        "model": "Mock plug",
    }
    bridge = [{"topic": BRIDGE_STATUS}]
    plug = bridge + [{"topic": t.availability}]

    def entity(component: str, key: str, unique: str, title: str, avail: list, **extra: Any):
        payload = {
            "name": title,
            "unique_id": unique,
            "device": device,
            "availability": avail,
            "availability_mode": "all",
            **extra,
        }
        return _config_topic(prefix, component, plug_id, key), payload

    return [
        entity("sensor", "power", f"{plug_id}_power", "Power", plug,
               state_topic=t.power, device_class="power", unit_of_measurement="W",
               state_class="measurement", suggested_display_precision=1),
        entity("sensor", "energy", f"{plug_id}_energy", "Energy", plug,
               state_topic=t.energy, device_class="energy", unit_of_measurement="kWh",
               state_class="total_increasing", suggested_display_precision=3),
        entity("switch", "relay", f"{plug_id}_switch", "Relay", plug,
               state_topic=t.relay, command_topic=f"{t.relay}/set",
               payload_on="ON", payload_off="OFF"),
        entity("switch", "connected", f"{plug_id}_connected", "Connected", bridge,
               state_topic=t.connected, command_topic=f"{t.connected}/set",
               payload_on="ON", payload_off="OFF", entity_category="config",
               icon="mdi:wifi"),
        entity("select", "program", f"{plug_id}_program", "Program", bridge,
               state_topic=t.program, command_topic=f"{t.program}/set",
               options=["random", *programs]),
        entity("select", "scenario", f"{plug_id}_scenario", "Scenario", bridge,
               state_topic=t.scenario, command_topic=f"{t.scenario}/set",
               options=scenarios),
        entity("select", "mode", f"{plug_id}_mode", "Reporting", bridge,
               state_topic=t.mode, command_topic=f"{t.mode}/set",
               options=modes, entity_category="config"),
        entity("button", "start", f"{plug_id}_start", "Start", bridge,
               command_topic=f"{BASE}/{plug_id}/start", icon="mdi:play"),
        entity("button", "stop", f"{plug_id}_stop", "Stop", bridge,
               command_topic=f"{BASE}/{plug_id}/stop", icon="mdi:stop"),
    ]


def encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def parse_command(topic: str, payload: bytes | str) -> tuple[str, str, str] | None:
    """``(plug id, action, value)`` for a command topic, else ``None``."""
    parts = topic.split("/")
    if len(parts) < 3 or parts[0] != BASE:
        return None
    action = COMMANDS.get("/".join(parts[2:]))
    if action is None:
        return None
    value = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else payload
    return parts[1], action, value.strip()
