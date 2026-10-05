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
"""Runs the model against a wall clock and an MQTT broker, and keeps the ledger.

One asyncio task per plug sleeps until its `PlugSim` has something due, then publishes
what it reports. paho runs its own network thread; its callbacks hop onto the event loop
with ``call_soon_threadsafe``, so every state change happens on one thread.

The ledger (``mock_socket_ledger.jsonl``) gets one line per finished run: what was
played, under which scenario and plug mode, and the UTC span of every cycle the
appliance really ran. That is the ground truth to hold WashData's stored cycles against.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .model import (
    EPS,
    PLUG_MODES,
    SCENARIOS,
    Appliance,
    Event,
    PlugSim,
    Run,
    Variation,
    build_program,
    load_appliance,
)
from .topics import (
    BRIDGE_STATUS,
    OFFLINE,
    ONLINE,
    SUBSCRIPTIONS,
    PlugTopics,
    discovery,
    encode,
    parse_command,
)

_LOGGER = logging.getLogger("washdata_mock")


class LogBuffer(logging.Handler):
    """The last lines of the mock's own log, for the UI."""

    def __init__(self, size: int = 200) -> None:
        super().__init__()
        self.lines: deque[str] = deque(maxlen=size)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


LOG_BUFFER = LogBuffer()
_LOGGER.addHandler(LOG_BUFFER)

DEVTOOLS = Path(__file__).resolve().parents[1]
REPO = DEVTOOLS.parent
STATE_FILE = DEVTOOLS / "mock_socket_config.json"
LEDGER_FILE = DEVTOOLS / "mock_socket_ledger.jsonl"
UPLOAD_DIR = DEVTOOLS / "uploaded_cycles"
STATE_VERSION = 2
#: The pre-rebuild plug. Two WashData entries point at the power sensor it created.
DEFAULT_PLUG_ID = "mock_washer_power"
DEFAULT_PLUG_NAME = "Mock Washer Socket"
HISTORY_DIR = DEVTOOLS / "mock_socket_history"
HISTORY_KEEP_S = 48 * 3600


# --------------------------------------------------------------------------- settings


@dataclass
class MqttSettings:
    host: str = "localhost"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    tls: bool = False
    tls_insecure: bool = False
    prefix: str = "homeassistant"

    @classmethod
    def from_secrets(cls) -> MqttSettings:
        """``devtools/priv_secrets.py`` (or ``mqtt_secrets.py``), as the old mock read it."""
        for name in ("priv_secrets.py", "mqtt_secrets.py"):
            path = DEVTOOLS / name
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location(f"wd_mock_{path.stem}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return cls(
                host=getattr(mod, "MQTT_HOST", cls.host),
                port=int(getattr(mod, "MQTT_PORT", cls.port)),
                username=getattr(mod, "MQTT_USERNAME", None),
                password=getattr(mod, "MQTT_PASSWORD", None),
                tls=bool(getattr(mod, "MQTT_USE_TLS", False)),
                tls_insecure=bool(getattr(mod, "MQTT_TLS_INSECURE", False)),
                prefix=getattr(mod, "MQTT_DISCOVERY_PREFIX", cls.prefix),
            )
        return cls()


@dataclass
class PlugSettings:
    """One plug, as persisted in ``mock_socket_config.json``."""

    id: str
    name: str
    source: str = ""
    program: str = "random"
    scenario: str = "clean"
    mode: str = "recorded"
    idle_w: float = 0.0
    stretch: float = Variation.stretch
    scale: float = Variation.scale
    noise_w: float = Variation.noise_w
    repeat: bool = False
    gap_min: float = 30.0
    energy_kwh: float = 0.0

    @property
    def variation(self) -> Variation:
        return Variation(self.stretch, self.scale, self.noise_w)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PlugSettings:
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        settings = cls(**known)
        if settings.scenario not in SCENARIOS:
            settings.scenario = "clean"
        if settings.mode not in PLUG_MODES:
            settings.mode = "recorded"
        return settings


def load_state(path: Path = STATE_FILE) -> tuple[float, list[PlugSettings]]:
    """(speedup, plugs). A pre-rebuild file is ignored: it held MQTT credentials, which
    now come from ``priv_secrets.py`` only, and a source path the old mock never used."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if raw.get("version") != STATE_VERSION:
        return 1.0, [PlugSettings(DEFAULT_PLUG_ID, DEFAULT_PLUG_NAME)]
    plugs = [PlugSettings.from_dict(p) for p in raw.get("plugs") or [] if p.get("id")]
    return float(raw.get("speedup") or 1.0), plugs or [PlugSettings(DEFAULT_PLUG_ID, DEFAULT_PLUG_NAME)]


def save_state(speedup: float, plugs: list[PlugSettings], path: Path = STATE_FILE) -> None:
    payload = {"version": STATE_VERSION, "speedup": speedup, "plugs": [asdict(p) for p in plugs]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- clock


class Clock:
    """Appliance seconds since the mock started; ``speedup`` of them per wall second."""

    def __init__(self, speedup: float = 1.0) -> None:
        self.speedup = max(1.0, float(speedup))
        self._mono = time.monotonic()
        self._wall = time.time()

    def now(self) -> float:
        return (time.monotonic() - self._mono) * self.speedup

    def wall_delay(self, t: float) -> float:
        return max(0.0, (t - self.now()) / self.speedup)

    def utc(self, t: float) -> datetime:
        return datetime.fromtimestamp(self._wall + t / self.speedup, tz=timezone.utc)

    def ms(self, t: float) -> float:
        """Epoch milliseconds of appliance time ``t``."""
        return (self._wall + t / self.speedup) * 1000.0

    def from_ms(self, ms: float) -> float:
        return (ms / 1000.0 - self._wall) * self.speedup


def run_record(plug: PlugSettings, run: Run, clock: Clock) -> dict[str, Any]:
    """The ledger line for one finished run."""
    program = run.program
    stamp = lambda t: clock.utc(t).isoformat(timespec="seconds")  # noqa: E731
    return {
        "plug": plug.id,
        "status": run.status,
        "scenario": program.scenario,
        "plug_mode": run.mode,
        "speedup": clock.speedup,
        "seed": run.seed,
        "started": stamp(run.started),
        "ended": stamp(run.ended if run.ended is not None else run.started),
        "source": Path(plug.source).name,
        "cycles_played": [{"id": c.cycle_id, "program": c.program} for c in program.sources],
        "variation": asdict(plug.variation),
        "relay_off_s": round(sum(length for _, length in run.frozen), 1),
        "expect": SCENARIOS[program.scenario].expect,
        "truth": [
            {"program": cyc.program, "start": stamp(start), "end": stamp(end),
             "minutes": round((end - start) / 60.0, 1)}
            for cyc, start, end in run.truth()
        ],
    }


# --------------------------------------------------------------------------- plot history


class History:
    """What the plot shows, in epoch ms: the reports Home Assistant got (``None`` = the plug
    went offline), the true draw, and offline / relay-off spans.

    Appended to ``mock_socket_history/<plug>.csv`` and reloaded (last 48 h) on start, so a
    restart keeps the picture; the time the mock itself was down shows as a gap. Clients
    fetch deltas with ``since(cursor)``; pruning bumps ``epoch`` so they reload in full.
    """

    def __init__(self, path: Path | None, keep_s: float = HISTORY_KEEP_S) -> None:
        self.path = path
        self.keep_ms = keep_s * 1000.0
        self.reported: list[tuple[float, float | None]] = []
        self.draw: list[tuple[float, float | None]] = []
        self.spans: list[list[Any]] = []  # [kind, start_ms, end_ms | None]
        self.epoch = 0
        self._fh = None
        if path is not None:
            self._load()

    def _load(self) -> None:
        cutoff = time.time() * 1000.0 - self.keep_ms
        kept: list[str] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                parts = line.split(",", 2)
                if len(parts) != 3:
                    continue
                try:
                    ms = float(parts[0])
                except ValueError:
                    continue
                if ms < cutoff:
                    continue
                kept.append(line)
                self._apply(ms, parts[1], parts[2])
        last = max((p[0] for p in (self.reported[-1:] + self.draw[-1:])), default=None)
        if last is not None:
            # The mock was down from here: no reading, and every open span ends.
            if self.reported and self.reported[-1][1] is not None:
                self.reported.append((last + 1.0, None))
            for span in self.spans:
                if span[2] is None:
                    span[2] = last
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)

    def _apply(self, ms: float, kind: str, value: str) -> None:
        if kind in ("r", "d"):
            target = self.reported if kind == "r" else self.draw
            target.append((ms, float(value) if value else None))
        elif kind == "s":
            self.spans.append([value, ms, None])
        elif kind == "e":
            for span in reversed(self.spans):
                if span[0] == value and span[2] is None:
                    span[2] = ms
                    break

    def add(self, ms: float, kind: str, value: str = "") -> None:
        self._apply(ms, kind, value)
        if self._fh is not None:
            self._fh.write(f"{ms:.0f},{kind},{value}\n")
        if self.reported and self.reported[0][0] < ms - self.keep_ms - 3.6e6:
            self._prune(ms - self.keep_ms)

    def _prune(self, cutoff: float) -> None:
        self.reported = [p for p in self.reported if p[0] >= cutoff]
        self.draw = [p for p in self.draw if p[0] >= cutoff]
        self.spans = [s for s in self.spans if s[2] is None or s[2] >= cutoff]
        self.epoch += 1

    def cursor(self) -> tuple[int, int, int]:
        return self.epoch, len(self.reported), len(self.draw)

    def since(self, cursor: tuple[int, int, int] | None) -> tuple[bool, list, list]:
        """(full, reported, draw): everything if ``cursor`` is stale, else what is new."""
        if cursor is None or cursor[0] != self.epoch:
            return True, list(self.reported), list(self.draw)
        return False, self.reported[cursor[1]:], self.draw[cursor[2]:]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# --------------------------------------------------------------------------- MQTT


class MqttLink:
    """paho, connected in the background, reconnecting on its own."""

    def __init__(self, settings: MqttSettings, loop: asyncio.AbstractEventLoop,
                 on_connected: Callable[[], None],
                 on_command: Callable[[str, str, str], None]) -> None:
        import paho.mqtt.client as mqtt  # noqa: PLC0415 - optional devtools dependency

        self.settings = settings
        self.connected = False
        self.error = ""
        self._loop = loop
        self._on_connected = on_connected
        self._on_command = on_command
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id=f"washdata_mock_{os.getpid()}")
        if settings.username:
            client.username_pw_set(settings.username, settings.password)
        if settings.tls:
            client.tls_set()
            client.tls_insecure_set(settings.tls_insecure)
        client.will_set(BRIDGE_STATUS, OFFLINE, retain=True)
        client.reconnect_delay_set(1, 30)
        client.on_connect = self._connect_cb
        client.on_disconnect = self._disconnect_cb
        client.on_message = self._message_cb
        self.client = client

    def start(self) -> None:
        self.client.connect_async(self.settings.host, self.settings.port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        if self.connected:
            self.client.publish(BRIDGE_STATUS, OFFLINE, retain=True).wait_for_publish(2)
        self.client.disconnect()
        self.client.loop_stop()

    def publish(self, topic: str, payload: str, retain: bool = False) -> None:
        self.client.publish(topic, payload, retain=retain)

    def _connect_cb(self, client, _userdata, _flags, reason, _props) -> None:
        if reason.is_failure:
            self.error = str(reason)
            _LOGGER.error("MQTT refused: %s", reason)
            return
        self.connected, self.error = True, ""
        client.publish(BRIDGE_STATUS, ONLINE, retain=True)
        for pattern in SUBSCRIPTIONS:
            client.subscribe(pattern)
        _LOGGER.info("MQTT connected to %s:%s", self.settings.host, self.settings.port)
        self._loop.call_soon_threadsafe(self._on_connected)

    def _disconnect_cb(self, _client, _userdata, _flags, reason, _props) -> None:
        self.connected = False
        self.error = "" if not reason.is_failure else str(reason)
        _LOGGER.warning("MQTT disconnected (%s)", reason)

    def _message_cb(self, _client, _userdata, msg) -> None:
        parsed = parse_command(msg.topic, msg.payload)
        if parsed is not None:
            self._loop.call_soon_threadsafe(self._on_command, *parsed)


# --------------------------------------------------------------------------- plugs


class PlugRuntime:
    """One plug: its settings, its simulator, and the task that plays it."""

    def __init__(self, settings: PlugSettings, hub: MockHub) -> None:
        self.s = settings
        self.hub = hub
        self.topics = PlugTopics(settings.id)
        self.app: Appliance | None = None
        self.load_error = ""
        self.sim = PlugSim(PLUG_MODES[settings.mode], idle_w=settings.idle_w,
                           energy_kwh=settings.energy_kwh)
        self.history = History(
            hub.history_dir / f"{settings.id}.csv" if hub.history_dir is not None else None
        )
        self.ledger: deque[dict[str, Any]] = deque(maxlen=200)
        self.repeat_at: float | None = None
        self.runs_left: int | None = None  # a CLI --runs budget; None follows s.repeat
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        if settings.source:
            self._load(settings.source)

    # ----------------------------------------------------------------- source

    def _load(self, path: str) -> str | None:
        try:
            self.app = load_appliance(path)
        except ValueError as err:
            self.app, self.load_error = None, str(err)
            return self.load_error
        self.s.source, self.load_error = str(path), ""
        if self.s.program != "random" and self.s.program not in self.app.programs():
            self.s.program = "random"
        return None

    def load_source(self, path: str) -> str | None:
        """Load a corpus file; returns an error message or None."""
        error = self._load(path)
        if error is None:
            self.publish_discovery()
            self.hub.save()
        return error

    # ----------------------------------------------------------------- loop

    def begin(self) -> None:
        self._handle(self.sim.boot(self.hub.clock.now()))
        self._task = asyncio.create_task(self._loop(), name=f"washdata_mock:{self.s.id}")

    async def end(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        clock = self.hub.clock
        while True:
            due = [d for d in (self.sim.next_due(), self.repeat_at) if d is not None]
            timeout = clock.wall_delay(min(due)) if due else None
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                pass
            self._wake.clear()
            now = clock.now()
            self._handle(self.sim.advance(now))
            if self.repeat_at is not None and now >= self.repeat_at - EPS and self.sim.run is None:
                self.repeat_at = None
                error = self.start()
                if error:
                    _LOGGER.warning("[%s] repeat not started: %s", self.s.id, error)

    def _poke(self) -> None:
        self._wake.set()

    def planned_truth_ms(self) -> list[tuple[str, float, float]]:
        """The playing run's real cycles, (programme, start ms, end ms); the future part is
        the plan, shifted by every relay-off pause so far."""
        run = self.sim.run
        if run is None:
            return []
        clock = self.hub.clock
        return [(c.program, clock.ms(run.appliance_time(c.start)), clock.ms(run.appliance_time(c.end)))
                for c in run.program.truth]

    # ----------------------------------------------------------------- controls

    def start(self, program: str | None = None, scenario: str | None = None) -> str | None:
        """Start a run now; returns an error message or None."""
        if self.sim.run is not None:
            return "already playing"
        if self.app is None:
            return self.load_error or "no source file loaded"
        seed = random.randrange(2**31)
        try:
            built = build_program(scenario or self.s.scenario, self.app,
                                  program or self.s.program, self.s.variation, random.Random(seed))
        except ValueError as err:
            return str(err)
        self.repeat_at = None
        self._handle(self.sim.start(built, self.hub.clock.now(), seed))
        _LOGGER.info("[%s] playing %s (%s, %s, %.0f min)", self.s.id, built.name(),
                     built.scenario, self.s.mode, built.end / 60)
        self._poke()
        return None

    def stop(self) -> None:
        self.repeat_at = None
        self._handle(self.sim.stop(self.hub.clock.now()))
        self._poke()

    def set_relay(self, on: bool) -> None:
        if on != self.sim.relay_on:
            now = self.hub.clock.now()
            self.history.add(self.hub.clock.ms(now), "e" if on else "s", "relay")
        self._handle(self.sim.set_relay(on, self.hub.clock.now()))
        self._publish(self.topics.relay, "ON" if on else "OFF", retain=True)
        self._poke()

    def set_connected(self, on: bool) -> None:
        self._handle(self.sim.set_plugged(on, self.hub.clock.now()))
        self._publish(self.topics.connected, "ON" if on else "OFF", retain=True)
        self._poke()

    def set_option(self, key: str, value: Any) -> None:
        """A persisted setting, from the UI or a select entity in Home Assistant."""
        if key == "mode":
            if value not in PLUG_MODES:
                return
            self._handle(self.sim.advance(self.hub.clock.now()))
            self.sim.mode = PLUG_MODES[value]
        elif key == "scenario" and value not in SCENARIOS:
            return
        elif key == "program" and value != "random" and (
            self.app is None or value not in self.app.programs()
        ):
            return
        elif key == "idle_w":
            value = max(0.0, float(value))
            self._handle(self.sim.set_idle_w(value, self.hub.clock.now()))
        elif key == "repeat" and not value:
            self.repeat_at = None
        setattr(self.s, key, value)
        self.publish_selects()
        self.hub.save()
        self._poke()

    # ----------------------------------------------------------------- output

    def _publish(self, topic: str, payload: str, retain: bool = False) -> None:
        if self.hub.link is not None:
            self.hub.link.publish(topic, payload, retain)

    def publish_discovery(self) -> None:
        link = self.hub.link
        if link is None or not link.connected:
            return
        for topic, payload in discovery(
            self.s.id, self.s.name,
            programs=self.app.programs() if self.app else [],
            scenarios=list(SCENARIOS), modes=list(PLUG_MODES),
            prefix=link.settings.prefix,
        ):
            link.publish(topic, encode(payload), retain=True)
        self.publish_selects()
        self._publish(self.topics.availability, ONLINE if self.sim.online else OFFLINE, retain=True)
        self._publish(self.topics.relay, "ON" if self.sim.relay_on else "OFF", retain=True)
        self._publish(self.topics.connected, "ON" if self.sim.plugged else "OFF", retain=True)
        last = self.sim.last_report
        if last is not None:
            self._publish(self.topics.power, f"{last[1]:.1f}")

    def publish_selects(self) -> None:
        self._publish(self.topics.program, self.s.program, retain=True)
        self._publish(self.topics.scenario, self.s.scenario, retain=True)
        self._publish(self.topics.mode, self.s.mode, retain=True)

    def _handle(self, events: list[Event]) -> None:
        for event in events:
            ms = self.hub.clock.ms(event.t)
            if event.kind == "power":
                self._publish(self.topics.power, f"{event.power:.1f}")
                self._publish(self.topics.energy, f"{event.energy_kwh:.4f}")
                self.history.add(ms, "r", f"{event.power:g}")
            elif event.kind == "draw":
                self.history.add(ms, "d", f"{event.power:g}")
            elif event.kind in ("online", "offline"):
                self._publish(self.topics.availability,
                              ONLINE if event.kind == "online" else OFFLINE, retain=True)
                if event.kind == "offline":
                    self.history.add(ms, "r")
                    self.history.add(ms, "s", "offline")
                else:
                    self.history.add(ms, "e", "offline")
                _LOGGER.info("[%s] plug %s", self.s.id, event.kind)
            elif event.kind == "run_end" and event.run is not None:
                self._finish(event.run)

    def _finish(self, run: Run) -> None:
        record = run_record(self.s, run, self.hub.clock)
        self.ledger.appendleft(record)
        self.hub.write_ledger(record)
        self.s.energy_kwh = round(self.sim.energy_kwh, 4)
        self.hub.save()
        _LOGGER.info("[%s] run %s: %s", self.s.id, run.status,
                     ", ".join(f"{t['program']} {t['minutes']} min" for t in record["truth"])
                     or "no cycle")
        if self.runs_left is not None:
            self.runs_left -= 1
            again = self.runs_left > 0
        else:
            again = self.s.repeat
        if again and run.status == "completed":
            self.repeat_at = run.ended + self.s.gap_min * 60.0


class MockHub:
    """All plugs, one broker connection, one clock."""

    def __init__(self, mqtt: MqttSettings | None, *, speedup: float = 1.0,
                 plugs: list[PlugSettings] | None = None,
                 state_path: Path | None = STATE_FILE,
                 ledger_path: Path | None = LEDGER_FILE,
                 history_dir: Path | None = HISTORY_DIR) -> None:
        """``state_path`` / ``ledger_path`` / ``history_dir`` None: keep it in memory."""
        self.mqtt_settings = mqtt
        self.history_dir = history_dir
        self.clock = Clock(speedup)
        self.state_path = state_path
        self.ledger_path = ledger_path
        self.link: MqttLink | None = None
        self.plugs: dict[str, PlugRuntime] = {}
        for settings in plugs or []:
            self.plugs[settings.id] = PlugRuntime(settings, self)

    def _read_ledger_tail(self, lines: int = 400) -> None:
        if self.ledger_path is None or not self.ledger_path.exists():
            return
        with self.ledger_path.open(encoding="utf-8") as fh:
            tail = deque(fh, maxlen=lines)
        for line in tail:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            plug = self.plugs.get(record.get("plug"))
            if plug is not None:
                plug.ledger.appendleft(record)

    async def start(self) -> None:
        self._read_ledger_tail()
        loop = asyncio.get_running_loop()
        if self.mqtt_settings is not None:
            self.link = MqttLink(self.mqtt_settings, loop, self._connected, self._command)
            self.link.start()
        for plug in self.plugs.values():
            plug.begin()

    async def shutdown(self) -> None:
        for plug in self.plugs.values():
            plug.sim.advance(self.clock.now())
            plug.s.energy_kwh = round(plug.sim.energy_kwh, 4)
            await plug.end()
            plug.history.close()
        self.save()
        if self.link is not None:
            self.link.stop()

    def save(self) -> None:
        if self.state_path is not None:
            save_state(self.clock.speedup, [p.s for p in self.plugs.values()], self.state_path)

    def write_ledger(self, record: dict[str, Any]) -> None:
        if self.ledger_path is not None:
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    def add_plug(self, plug_id: str, name: str, source: str = "") -> PlugRuntime:
        if plug_id in self.plugs:
            raise ValueError(f"plug {plug_id!r} exists")
        plug = PlugRuntime(PlugSettings(plug_id, name or plug_id, source=source), self)
        self.plugs[plug_id] = plug
        plug.begin()
        plug.publish_discovery()
        self.save()
        return plug

    async def remove_plug(self, plug_id: str) -> None:
        plug = self.plugs.pop(plug_id)
        await plug.end()
        plug.history.close()
        if self.link is not None and self.link.connected:
            for topic, _payload in discovery(plug_id, plug.s.name, programs=[], scenarios=[],
                                             modes=[], prefix=self.link.settings.prefix):
                self.link.publish(topic, "", retain=True)
        self.save()

    def _connected(self) -> None:
        for plug in self.plugs.values():
            plug.publish_discovery()

    def _command(self, plug_id: str, action: str, value: str) -> None:
        plug = self.plugs.get(plug_id)
        if plug is None:
            return
        _LOGGER.info("[%s] command %s %s", plug_id, action, value)
        if action == "start":
            error = plug.start()
            if error:
                _LOGGER.warning("[%s] cannot start: %s", plug_id, error)
        elif action == "stop":
            plug.stop()
        elif action == "relay":
            plug.set_relay(value.upper() == "ON")
        elif action == "connected":
            plug.set_connected(value.upper() == "ON")
        else:
            plug.set_option(action, value)
