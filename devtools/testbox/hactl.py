#!/usr/bin/env python3
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
"""Driver for the WashData test box: onboarding, REST, WebSocket, cycle replay.

One file so there is one place to look. Everything talks to the container on
http://127.0.0.1:8321 and nothing here can reach the real Home Assistant.

    ./hactl.py onboard                      mint the owner + a long-lived token
    ./hactl.py state sensor.washdata_power  read a state
    ./hactl.py set sensor.washdata_power 1200
    ./hactl.py states [prefix]              list states (optionally filtered)
    ./hactl.py ws ha_washdata/get_options entry_id=<id>
    ./hactl.py setup-device [--name X] [--type dishwasher]
    ./hactl.py entry-id [--name X]
    ./hactl.py set-options <entry_id> key=value ...
    ./hactl.py replay <trace.json|cycle_data/...> [--entry <id>] [--speedup 60]
    ./hactl.py notifications [--since ISO]  captured notification payloads
    ./hactl.py errors [--since ISO]         ERROR/WARNING lines from the HA log

Values in `key=value` arguments are parsed as JSON when possible, so
`notify_live_services='["notify.mobile_app_testbox"]'` and `off_delay=5` do what
they look like; anything unparseable stays a string.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "config"
TOKEN_FILE = CONFIG_DIR / ".testbox_token"
CAPTURE_FILE = CONFIG_DIR / "notify_capture.jsonl"
LOG_FILE = CONFIG_DIR / "home-assistant.log"
BASE_URL = os.environ.get("TESTBOX_URL", "http://127.0.0.1:8321")
CLIENT_ID = f"{BASE_URL}/"
DOMAIN = "ha_washdata"

OWNER = {"name": "Test Box", "username": "testbox", "password": "testbox-pw-0123"}


# ── plumbing ────────────────────────────────────────────────────────────────
def token() -> str:
    if not TOKEN_FILE.exists():
        sys.exit(f"no token at {TOKEN_FILE} - run `./hactl.py onboard` first")
    return TOKEN_FILE.read_text().strip()


def _coerce(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _kv(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"expected key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        out[key] = _coerce(raw)
    return out


async def rest(
    session: aiohttp.ClientSession,
    method: str,
    path: str,
    payload: Any = None,
    auth: bool = True,
) -> Any:
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {token()}"
    async with session.request(
        method, f"{BASE_URL}{path}", headers=headers, json=payload
    ) as resp:
        body = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"{method} {path} -> {resp.status}: {body[:400]}")
        return json.loads(body) if body.strip() else None


class WS:
    """Minimal authenticated WebSocket client for the ha_washdata/* commands."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._id = 0

    async def __aenter__(self) -> "WS":
        self._ws = await self._session.ws_connect(
            f"{BASE_URL}/api/websocket", heartbeat=30
        )
        hello = await self._ws.receive_json()
        assert hello["type"] == "auth_required", hello
        await self._ws.send_json({"type": "auth", "access_token": token()})
        result = await self._ws.receive_json()
        if result["type"] != "auth_ok":
            raise RuntimeError(f"websocket auth failed: {result}")
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def cmd(self, type_: str, **fields: Any) -> Any:
        assert self._ws is not None
        self._id += 1
        msg_id = self._id
        await self._ws.send_json({"id": msg_id, "type": type_, **fields})
        while True:
            msg = await self._ws.receive_json()
            if msg.get("id") != msg_id or msg.get("type") != "result":
                continue  # events and other commands' results
            if not msg.get("success"):
                raise RuntimeError(f"{type_} failed: {msg.get('error')}")
            return msg.get("result")


# ── commands ────────────────────────────────────────────────────────────────
async def cmd_onboard(_args: argparse.Namespace) -> int:
    """Create the owner user and store a long-lived token.

    Idempotent in effect: if onboarding has already run, the stored token is
    reused (Home Assistant only allows the users step once).
    """
    if TOKEN_FILE.exists():
        async with aiohttp.ClientSession() as session:
            try:
                await rest(session, "GET", "/api/")
                print(f"already onboarded, token at {TOKEN_FILE}")
                return 0
            except RuntimeError:
                print("stored token is stale, re-onboarding")

    async with aiohttp.ClientSession() as session:
        step = await rest(
            session,
            "POST",
            "/api/onboarding/users",
            {
                "client_id": CLIENT_ID,
                "name": OWNER["name"],
                "username": OWNER["username"],
                "password": OWNER["password"],
                "language": "en",
            },
            auth=False,
        )
        async with session.post(
            f"{BASE_URL}/auth/token",
            data={
                "grant_type": "authorization_code",
                "code": step["auth_code"],
                "client_id": CLIENT_ID,
            },
        ) as resp:
            tokens = await resp.json()
            if "access_token" not in tokens:
                raise RuntimeError(f"token exchange failed: {tokens}")
        TOKEN_FILE.write_text(tokens["access_token"])

        # Trade the 30-minute access token for a long-lived one, so a session
        # that runs for hours does not lose access halfway through a replay.
        async with WS(session) as ws:
            long_lived = await ws.cmd(
                "auth/long_lived_access_token",
                client_name=f"testbox-{uuid.uuid4().hex[:8]}",
                lifespan=365,
            )
        TOKEN_FILE.write_text(long_lived)
        TOKEN_FILE.chmod(0o600)

        # Finish the remaining onboarding steps so the UI is usable too. Not
        # required for the API, hence best-effort.
        for path, body in (
            ("/api/onboarding/core_config", {}),
            ("/api/onboarding/analytics", {}),
        ):
            try:
                await rest(session, "POST", path, body)
            except RuntimeError:
                pass

    print(f"onboarded. long-lived token written to {TOKEN_FILE}")
    print(f"UI: {BASE_URL}  user: {OWNER['username']}  password: {OWNER['password']}")
    return 0


async def cmd_state(args: argparse.Namespace) -> int:
    async with aiohttp.ClientSession() as session:
        print(json.dumps(await rest(session, "GET", f"/api/states/{args.entity}"), indent=2))
    return 0


async def cmd_states(args: argparse.Namespace) -> int:
    async with aiohttp.ClientSession() as session:
        states = await rest(session, "GET", "/api/states")
    for state in sorted(states, key=lambda s: s["entity_id"]):
        if args.prefix and not state["entity_id"].startswith(args.prefix):
            continue
        print(f"{state['entity_id']:<58} {state['state']}")
    return 0


async def cmd_set(args: argparse.Namespace) -> int:
    """Push a sensor state. This is how the box gets its power readings."""
    async with aiohttp.ClientSession() as session:
        await _push_power(session, args.entity, float(args.value))
    return 0


async def _push_power(
    session: aiohttp.ClientSession, entity: str, watts: float
) -> None:
    await rest(
        session,
        "POST",
        f"/api/states/{entity}",
        {
            "state": f"{watts:.2f}",
            "attributes": {
                "unit_of_measurement": "W",
                "device_class": "power",
                "state_class": "measurement",
                "friendly_name": "WashData test box power",
            },
        },
    )


async def cmd_set_state(args: argparse.Namespace) -> int:
    """Push an arbitrary state, for entities that are not power readings.

    ``set`` shapes its payload as a power sensor (W, device_class power), which
    is wrong for the entities the unload confirmation option takes - an
    ``event.*`` button writes a timestamp, an ``input_button`` the same. Pushing
    the same value twice is a no-op in the state machine, so a second "press"
    must carry a different value.
    """
    async with aiohttp.ClientSession() as session:
        await rest(
            session,
            "POST",
            f"/api/states/{args.entity}",
            {"state": args.value, "attributes": _kv(args.attrs)},
        )
    return 0


async def cmd_call(args: argparse.Namespace) -> int:
    """Call any service on the real service bus, schema validation included."""
    domain, _, service = args.service.partition(".")
    if not service:
        sys.exit("expected domain.service, e.g. ha_washdata.mark_unloaded")
    async with aiohttp.ClientSession() as session:
        result = await rest(
            session, "POST", f"/api/services/{domain}/{service}", _kv(args.fields)
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


async def cmd_device_id(args: argparse.Namespace) -> int:
    """The device-registry id for an entry - what the device_id services take."""
    async with aiohttp.ClientSession() as session:
        entry = args.entry or await _entry_id(session, args.name)
        if entry is None:
            sys.exit("no ha_washdata config entry found")
        async with WS(session) as ws:
            devices = await ws.cmd("config/device_registry/list")
    for device in devices or []:
        if entry in (device.get("config_entries") or []):
            print(device["id"])
            return 0
    sys.exit(f"no device registered for entry {entry}")


async def cmd_ws(args: argparse.Namespace) -> int:
    async with aiohttp.ClientSession() as session, WS(session) as ws:
        result = await ws.cmd(args.type, **_kv(args.fields))
    print(json.dumps(result, indent=2, default=str))
    return 0


async def _entry_id(session: aiohttp.ClientSession, name: str | None) -> str | None:
    entries = await rest(
        session, "GET", f"/api/config/config_entries/entry?domain={DOMAIN}"
    )
    for entry in entries or []:
        if name is None or entry.get("title") == name:
            return entry["entry_id"]
    return None


async def cmd_entry_id(args: argparse.Namespace) -> int:
    async with aiohttp.ClientSession() as session:
        entry = await _entry_id(session, args.name)
    if entry is None:
        return 1
    print(entry)
    return 0


async def cmd_setup_device(args: argparse.Namespace) -> int:
    """Create a WashData config entry through the real config flow."""
    async with aiohttp.ClientSession() as session:
        existing = await _entry_id(session, args.name)
        if existing:
            print(existing)
            return 0
        # The power sensor must exist before the flow validates it.
        await _push_power(session, args.power_sensor, 0.0)
        flow = await rest(
            session,
            "POST",
            "/api/config/config_entries/flow",
            {"handler": DOMAIN, "show_advanced_options": True},
        )
        result = await rest(
            session,
            "POST",
            f"/api/config/config_entries/flow/{flow['flow_id']}",
            {
                "name": args.name,
                "device_type": args.type,
                "power_sensor": args.power_sensor,
                "min_power": args.min_power,
            },
        )
        if result.get("type") != "create_entry":
            raise RuntimeError(f"config flow did not create an entry: {result}")
        entry_id = result["result"]["entry_id"]
    print(entry_id)
    return 0


async def cmd_set_options(args: argparse.Namespace) -> int:
    """Write entry options the way the panel does (ws_set_options)."""
    async with aiohttp.ClientSession() as session, WS(session) as ws:
        result = await ws.cmd(
            "ha_washdata/set_options", entry_id=args.entry_id, options=_kv(args.options)
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


async def cmd_import(args: argparse.Namespace) -> int:
    """Import a WashData export, so the box has real profiles to match against.

    A fresh box has no profiles, and with none WashData deliberately stays quiet
    (`has_real_profiles` gates the live notifications), so a first cycle can
    never exercise the matched paths. Importing an export gives the box a real
    learned history in one step - and puts the storage migration for that
    export's version on the critical path of every smoke run.
    """
    payload = Path(args.export).read_text()
    async with aiohttp.ClientSession() as session, WS(session) as ws:
        result = await ws.cmd(
            "ha_washdata/import_config", entry_id=args.entry_id, json_data=payload
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


_TIME_FIELDS_CYCLE = ("duration", "sampling_interval")
_TIME_FIELDS_PROFILE = (
    "avg_duration", "min_duration", "max_duration", "duration_std_dev"
)
_TIME_FIELDS_ENVELOPE = ("target_duration", "duration_std_dev")


async def cmd_compress_export(args: argparse.Namespace) -> int:
    """Rescale every time axis in an export by 1/speedup.

    Time compression is what makes the box usable, but it cannot be applied to
    the replay alone. The matcher compares the running cycle against LEARNED
    durations, so a 76 s replay matched against a 1720 s profile is a cycle at 7%
    of its expected length - and the detector then refuses to end it
    ("Smart Termination not applied (duration_not_reached)", then deferral until
    expected + max deferral). Every duration the box knows about has to live in
    the same compressed units as the replay, so scale the stored history too:
    cycle durations and sample offsets, the profiles' duration statistics, and
    the envelopes' time grids. Shapes are untouched - only the clock moves.
    """
    payload = json.loads(Path(args.export).read_text())
    factor = float(args.speedup)
    if factor <= 0:
        sys.exit("--speedup must be positive")
    data = payload.get("data") or {}

    def _scale(container: dict[str, Any], fields: tuple[str, ...]) -> None:
        for field in fields:
            value = container.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                container[field] = value / factor

    cycles = data.get("past_cycles") or []
    for cycle in cycles:
        _scale(cycle, _TIME_FIELDS_CYCLE)
        trace = cycle.get("power_data")
        if isinstance(trace, list):
            cycle["power_data"] = [
                [point[0] / factor, point[1]]
                if isinstance(point, list) and len(point) >= 2 else point
                for point in trace
            ]
    for profile in (data.get("profiles") or {}).values():
        if isinstance(profile, dict):
            _scale(profile, _TIME_FIELDS_PROFILE)
    for envelope in (data.get("envelopes") or {}).values():
        if not isinstance(envelope, dict):
            continue
        _scale(envelope, _TIME_FIELDS_ENVELOPE)
        grid = envelope.get("time_grid")
        if isinstance(grid, list):
            envelope["time_grid"] = [
                t / factor if isinstance(t, (int, float)) else t for t in grid
            ]
    # The per-profile phase cache is keyed on absolute seconds and rebuilds
    # itself on the next envelope rebuild, so drop it rather than rescale it.
    data.pop("phase_profiles", None)

    Path(args.out).write_text(json.dumps(payload))
    print(
        f"wrote {args.out}: {len(cycles)} cycles, "
        f"{len(data.get('profiles') or {})} profiles, clock / {factor:g}"
    )
    return 0


def _load_trace(path: Path, cycle: int = 0) -> list[tuple[float, float]]:
    """Return [(offset_s, watts)] from a trace file.

    Accepts the shapes the repo already has: a bare list of [t, w] pairs, a list
    of {"t"/"offset"/"timestamp", "power"/"w"} objects, a dict with a
    "power_data"/"raw_data"/"samples" key, or a full WashData export (in which
    case `cycle` selects one of `data.past_cycles`).
    """
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        stored = raw["data"].get("past_cycles") or []
        if not stored:
            sys.exit(f"{path}: export has no past_cycles")
        if cycle >= len(stored):
            sys.exit(f"{path}: only {len(stored)} cycles, asked for #{cycle}")
        picked = stored[cycle]
        print(
            f"using cycle #{cycle} of {len(stored)}: "
            f"{picked.get('profile_name')}, {float(picked.get('duration') or 0) / 60:.0f} min",
            flush=True,
        )
        raw = picked.get("power_data")
    if isinstance(raw, dict):
        for key in ("power_data", "raw_data", "samples", "data", "trace"):
            if key in raw:
                raw = raw[key]
                break
        else:
            sys.exit(f"{path}: no power_data/raw_data/samples key")
    points: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, dict):
            t = item.get("t", item.get("offset", item.get("timestamp")))
            w = item.get("power", item.get("w", item.get("value")))
        else:
            t, w = item[0], item[1]
        if isinstance(t, str):  # ISO timestamps -> offsets from the first sample
            t = datetime.fromisoformat(t).timestamp()
        points.append((float(t), float(w)))
    if not points:
        sys.exit(f"{path}: trace is empty")
    base = points[0][0]
    return [(t - base, w) for t, w in points]


async def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a power trace into the box, compressed in wall-clock time.

    Compression is the whole reason the box is usable: a 150 min dishwasher is
    replayed in ~2.5 min at 60x. The detector's gates are wall-clock, so the
    entry's timing options must be divided by the same factor - `smoke.sh` does
    that, and `--entry` here does it for an ad-hoc run.
    """
    path = Path(args.trace)
    if not path.exists():
        sys.exit(f"no such trace: {path}")
    points = _load_trace(path, args.cycle)
    span = points[-1][0]
    print(
        f"replaying {len(points)} samples spanning {span / 60:.1f} min "
        f"at {args.speedup}x -> {span / args.speedup:.0f} s of wall clock",
        flush=True,
    )
    started = time.monotonic()
    async with aiohttp.ClientSession() as session:
        for index, (offset, watts) in enumerate(points):
            due = started + offset / args.speedup
            delay = due - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await _push_power(session, args.power_sensor, watts)
            if index % 25 == 0 or index == len(points) - 1:
                print(
                    f"  {offset / 60:7.1f} min  {watts:8.1f} W  "
                    f"({index + 1}/{len(points)})",
                    flush=True,
                )
        # The detector needs quiet time *after* the trace to close the cycle.
        if args.tail > 0:
            print(f"  holding 0 W for {args.tail} s so the cycle can close", flush=True)
            await _push_power(session, args.power_sensor, 0.0)
            await asyncio.sleep(args.tail)
    return 0


def _read_jsonl(path: Path, since: str | None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if since and record.get("ts", "") < since:
            continue
        out.append(record)
    return out


async def cmd_notifications(args: argparse.Namespace) -> int:
    records = _read_jsonl(CAPTURE_FILE, args.since)
    if args.json:
        print(json.dumps(records, indent=2))
        return 0
    for record in records:
        data = record.get("data") or {}
        extras = " ".join(
            f"{k}={data[k]}"
            for k in ("tag", "live_update", "chronometer", "progress", "activity")
            if k in data
        )
        print(f"{record['ts']}  {record['service']:<20} {record['message'][:70]}  {extras}")
    print(f"-- {len(records)} notification(s)")
    return 0


async def cmd_errors(args: argparse.Namespace) -> int:
    """ERROR/WARNING lines from the box's log, including validation failures.

    This is the check that would have caught the `title: None` rejection: it
    surfaces as a MultipleInvalid traceback under "Error doing job", which no
    unit test could see.
    """
    if not LOG_FILE.exists():
        print(f"no log at {LOG_FILE}")
        return 0
    interesting = []
    for line in LOG_FILE.read_text(errors="replace").splitlines():
        if args.since and line[:23] < args.since:
            continue
        if ("ERROR" in line or "WARNING" in line or "MultipleInvalid" in line
                or "Traceback" in line):
            interesting.append(line)
    print("\n".join(interesting) if interesting else "clean: no errors or warnings")
    return 1 if interesting and args.strict else 0


async def cmd_restart(_args: argparse.Namespace) -> int:
    """Restart the container so edited integration code is reloaded."""
    subprocess.run(
        ["docker", "compose", "restart", "ha"], cwd=HERE, check=True
    )
    return await cmd_wait(argparse.Namespace(timeout=180))


async def cmd_wait(args: argparse.Namespace) -> int:
    """Wait for the box to serve HTTP, then for WashData's own commands.

    The second phase matters: `manifest.json` answers well before the config
    entries are set up, and `ha_washdata/*` is registered during setup, so a
    command issued in that window comes back "unknown_command" and looks like a
    broken integration rather than an early one.
    """
    deadline = time.monotonic() + args.timeout
    async with aiohttp.ClientSession() as session:
        up = False
        while time.monotonic() < deadline:
            try:
                async with session.get(f"{BASE_URL}/manifest.json") as resp:
                    if resp.status == 200:
                        up = True
                        break
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(2)
        if not up:
            print("timed out waiting for the box", file=sys.stderr)
            return 1
        if not TOKEN_FILE.exists():
            print("box is up (not onboarded yet)")
            return 0
        # Only wait for the integration if there is something to set up.
        try:
            entries = await rest(
                session, "GET", f"/api/config/config_entries/entry?domain={DOMAIN}"
            )
        except RuntimeError:
            entries = []
        if not entries:
            print("box is up (no WashData entry yet)")
            return 0
        while time.monotonic() < deadline:
            try:
                async with WS(session) as ws:
                    await ws.cmd("ha_washdata/get_devices")
                print("box is up and WashData is set up")
                return 0
            except (RuntimeError, aiohttp.ClientError):
                await asyncio.sleep(2)
    print("box is up but WashData never finished setting up", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("onboard").set_defaults(func=cmd_onboard)
    sub.add_parser("restart").set_defaults(func=cmd_restart)

    p = sub.add_parser("wait")
    p.add_argument("--timeout", type=float, default=180)
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("state")
    p.add_argument("entity")
    p.set_defaults(func=cmd_state)

    p = sub.add_parser("states")
    p.add_argument("prefix", nargs="?")
    p.set_defaults(func=cmd_states)

    p = sub.add_parser("set")
    p.add_argument("entity")
    p.add_argument("value")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("set-state")
    p.add_argument("entity")
    p.add_argument("value")
    p.add_argument("attrs", nargs="*")
    p.set_defaults(func=cmd_set_state)

    p = sub.add_parser("call")
    p.add_argument("service")
    p.add_argument("fields", nargs="*")
    p.set_defaults(func=cmd_call)

    p = sub.add_parser("device-id")
    p.add_argument("--name", default=None)
    p.add_argument("--entry", default=None)
    p.set_defaults(func=cmd_device_id)

    p = sub.add_parser("ws")
    p.add_argument("type")
    p.add_argument("fields", nargs="*")
    p.set_defaults(func=cmd_ws)

    p = sub.add_parser("setup-device")
    p.add_argument("--name", default="Test Dishwasher")
    p.add_argument("--type", default="dishwasher")
    p.add_argument("--power-sensor", default="sensor.washdata_power")
    p.add_argument("--min-power", type=float, default=2.0)
    p.set_defaults(func=cmd_setup_device)

    p = sub.add_parser("entry-id")
    p.add_argument("--name", default="Test Dishwasher")
    p.set_defaults(func=cmd_entry_id)

    p = sub.add_parser("set-options")
    p.add_argument("entry_id")
    p.add_argument("options", nargs="+")
    p.set_defaults(func=cmd_set_options)

    p = sub.add_parser("compress-export")
    p.add_argument("export")
    p.add_argument("out")
    p.add_argument("--speedup", type=float, default=60.0)
    p.set_defaults(func=cmd_compress_export)

    p = sub.add_parser("import")
    p.add_argument("entry_id")
    p.add_argument("export")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("replay")
    p.add_argument("trace")
    p.add_argument("--cycle", type=int, default=0,
                   help="which past_cycles entry to replay when given an export")
    p.add_argument("--speedup", type=float, default=60.0)
    p.add_argument("--power-sensor", default="sensor.washdata_power")
    p.add_argument("--tail", type=float, default=0.0)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("notifications")
    p.add_argument("--since")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_notifications)

    p = sub.add_parser("errors")
    p.add_argument("--since")
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_errors)

    args = parser.parse_args()
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
