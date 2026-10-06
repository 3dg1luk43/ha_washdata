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
"""Mock MQTT smart plugs that play stored WashData cycles into a real Home Assistant.

What it is for: watching the whole integration react to a real cycle - detection,
matching, notifications, the panel - and provoking the plug behaviours the register
found matter (see ``--list``). It is not an accuracy tool: matching and end-detection
accuracy are measured by the ``devtools/*_eval.py`` harnesses, and Home Assistant
boundary behaviour by ``devtools/testbox``.

Usage, from the repo root (needs ``paho-mqtt``; the web UI also ``nicegui``)::

    python3 devtools/mqtt_mock_socket.py                     # web UI on :8081
    python3 devtools/mqtt_mock_socket.py --list --source cycle_data/me/<export>.json
    python3 devtools/mqtt_mock_socket.py --dry-run --source <export> --scenario soak --mode silent
    python3 devtools/mqtt_mock_socket.py --headless --source <export> --play random \\
        --scenario back-to-back --runs 3 --gap-min 20

The broker comes from ``devtools/priv_secrets.py`` (``MQTT_HOST``, ``MQTT_PORT``,
``MQTT_USERNAME``, ``MQTT_PASSWORD``), overridable with ``--mqtt-host``/``--mqtt-port``.
Plug settings persist in ``devtools/mock_socket_config.json``; every finished run is
appended to ``devtools/mock_socket_ledger.jsonl`` with the UTC span of each cycle the
appliance really ran (both gitignored).

Time runs at 1x by default. ``--speedup`` exists, but every WashData gate counts real
seconds, so a compressed run is meaningless until the Home Assistant side is compressed
too; the tool prints exactly what to change (devtools/testbox/README.md, *Time
compression*).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_socket.model import (  # noqa: E402
    PLUG_MODES,
    SCENARIOS,
    PlugSim,
    build_program,
    compression_advice,
    load_appliance,
)
from mock_socket.runner import (  # noqa: E402
    HISTORY_DIR,
    LEDGER_FILE,
    STATE_FILE,
    MockHub,
    MqttSettings,
    PlugSettings,
    load_state,
)

_LOGGER = logging.getLogger("washdata_mock")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Mock MQTT smart plugs playing stored WashData cycles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Scenarios and plug modes: --list.",
    )
    ap.add_argument("--source", "--cycle-source", dest="source",
                    help="export / diagnostics dump to play (sets the plug's source)")
    ap.add_argument("--plug", help="plug id to drive (default: the first configured plug)")
    ap.add_argument("--play", metavar="PROGRAM", help="start a run now: a programme name or 'random'")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), help="scenario for the run")
    ap.add_argument("--mode", choices=sorted(PLUG_MODES), help="plug reporting mode")
    ap.add_argument("--runs", type=int, default=1, help="headless: runs to play (0 = forever)")
    ap.add_argument("--gap-min", type=float, help="minutes between repeated runs")
    ap.add_argument("--exit-after-min", type=float,
                    help="headless: after the last run, keep reporting idle this long, then exit")
    ap.add_argument("--speedup", type=float, help="appliance seconds per wall second (default 1)")
    ap.add_argument("--seed", type=int, help="dry run: random seed")
    ap.add_argument("--mqtt-host", help="broker host (default from priv_secrets.py)")
    ap.add_argument("--mqtt-port", type=int, help="broker port")
    ap.add_argument("--config", type=Path, default=STATE_FILE,
                    help=f"plug settings file (default {STATE_FILE.name})")
    ap.add_argument("--ledger", type=Path, default=LEDGER_FILE,
                    help=f"where finished runs are appended (default {LEDGER_FILE.name})")
    ap.add_argument("--web-host", default="127.0.0.1",
                    help="web UI bind address (default 127.0.0.1; the UI has no login, so 0.0.0.0 opens it to the LAN)")
    ap.add_argument("--web-port", type=int, default=8081, help="web UI port (default 8081)")
    ap.add_argument("--headless", action="store_true", help="no web UI")
    ap.add_argument("--list", action="store_true", help="list scenarios, plug modes and programmes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what a run would publish, without MQTT")
    ap.add_argument("--events", action="store_true", help="dry run: print every event")
    return ap.parse_args(argv)


def _hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def cmd_list(args: argparse.Namespace) -> int:
    print("Scenarios:")
    for s in SCENARIOS.values():
        print(f"  {s.key:20s} {s.title} [{s.ref}]\n  {'':20s} {s.expect}")
    print("\nPlug reporting modes:")
    for m in PLUG_MODES.values():
        print(f"  {m.key:20s} {m.title}\n  {'':20s} {m.why}")
    if args.source:
        app = load_appliance(args.source)
        cfg = app.config
        print(f"\n{Path(app.path).name}: {app.device_type}, {len(app.cycles)} cycles, "
              f"report cadence {app.cadence:.0f} s")
        print(f"  thresholds: start {cfg.start_threshold_w} W, stop {cfg.stop_threshold_w} W, "
              f"off_delay {cfg.off_delay} s, min_off_gap {cfg.min_off_gap} s")
        for name in app.programs():
            runs = [c for c in app.cycles if c.program == name]
            mean = sum(c.duration for c in runs) / len(runs)
            print(f"  {len(runs):3d} x {name}  (~{mean / 60:.0f} min)")
    return 0


def cmd_dry_run(args: argparse.Namespace, plug: PlugSettings) -> int:
    app = load_appliance(plug.source)
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    program = build_program(plug.scenario, app, args.play or plug.program, plug.variation,
                            random.Random(seed))
    sim = PlugSim(PLUG_MODES[plug.mode], idle_w=plug.idle_w)
    lead = 300.0
    events = sim.boot(0.0) + sim.advance(lead)
    events += sim.start(program, lead, seed)
    tail = 2 * 3600.0
    events += sim.advance(lead + program.end + tail)

    spec = SCENARIOS[plug.scenario]
    print(f"{spec.title} [{spec.ref}] · {program.name()} · {plug.mode} · seed {seed}")
    print(f"expect: {spec.expect}")
    for cyc in program.truth:
        print(f"truth:  {cyc.program}  {_hms(cyc.start)} -> {_hms(cyc.end)} "
              f"({(cyc.end - cyc.start) / 60:.1f} min, programme time)")
    for a, b in program.outages:
        print(f"outage: {_hms(a)} -> {_hms(b)}")
    power = [e for e in events if e.kind == "power"]
    in_run = [e for e in power if lead <= e.t <= lead + program.end]
    gaps = sorted(b.t - a.t for a, b in zip(in_run, in_run[1:]))
    print(f"run {_hms(program.end)}: {len(in_run)} reports, median gap "
          f"{gaps[len(gaps) // 2] if gaps else 0:.0f} s, longest silence {gaps[-1] if gaps else 0:.0f} s; "
          f"{sim.energy_kwh:.3f} kWh; {len(power) - len(in_run)} more in the "
          f"{lead / 60:.0f} min before and {tail / 3600:.0f} h after")
    if args.events:
        for e in events:
            value = (f"{e.power:9.1f} W  {e.energy_kwh:.4f} kWh" if e.kind == "power"
                     else f"{e.power:9.1f} W" if e.kind == "draw" else "")
            print(f"  {_hms(e.t - lead):>9s}  {e.kind:8s} {value}")
    return 0


async def run_headless(args: argparse.Namespace, hub: MockHub, plug_id: str) -> int:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await hub.start()
    plug = hub.plugs[plug_id]
    for line in compression_advice(plug.app, hub.clock.speedup):
        _LOGGER.warning(line)
    if args.play:
        plug.runs_left = args.runs if args.runs > 0 else None
        if args.runs <= 0:
            plug.s.repeat = True
        error = plug.start(program=args.play)
        if error:
            _LOGGER.error("cannot start: %s", error)
            await hub.shutdown()
            return 2
    idle_since: float | None = None
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except TimeoutError:
            pass
        busy = plug.sim.run is not None or plug.repeat_at is not None
        if args.exit_after_min is None or not args.play or busy:
            idle_since = None
            continue
        now = hub.clock.now()
        idle_since = now if idle_since is None else idle_since
        if now - idle_since >= args.exit_after_min * 60.0:
            break
    await hub.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.list:
        return cmd_list(args)

    speedup, plugs = load_state(args.config)
    plug = next((p for p in plugs if p.id == args.plug), None) if args.plug else plugs[0]
    if plug is None:
        sys.exit(f"no plug {args.plug!r}; configured: {', '.join(p.id for p in plugs)}")
    if args.source:
        plug.source = str(Path(args.source).resolve())
    if args.scenario:
        plug.scenario = args.scenario
    if args.mode:
        plug.mode = args.mode
    if args.play:
        plug.program = args.play
    if args.gap_min is not None:
        plug.gap_min = args.gap_min
    if args.speedup is not None:
        speedup = args.speedup

    if args.dry_run:
        if not plug.source:
            sys.exit("--dry-run needs --source (or a plug with a source)")
        return cmd_dry_run(args, plug)

    mqtt = MqttSettings.from_secrets()
    if args.mqtt_host:
        mqtt.host = args.mqtt_host
    if args.mqtt_port:
        mqtt.port = args.mqtt_port

    if args.headless:
        # A scripted session must not rewrite the UI's saved settings; the ledger still counts.
        hub = MockHub(mqtt, speedup=speedup, plugs=plugs, state_path=None, ledger_path=args.ledger,
                      history_dir=args.config.parent / HISTORY_DIR.name)
        return asyncio.run(run_headless(args, hub, plug.id))

    from nicegui import app, ui  # noqa: PLC0415 - only the web UI needs it

    from mock_socket.ui import mount  # noqa: PLC0415

    hub = MockHub(mqtt, speedup=speedup, plugs=plugs, state_path=args.config,
                  ledger_path=args.ledger, history_dir=args.config.parent / HISTORY_DIR.name)
    hub.save()
    mount(hub)

    async def startup() -> None:
        await hub.start()
        if args.play:
            error = hub.plugs[plug.id].start()
            if error:
                _LOGGER.error("cannot start: %s", error)

    app.on_startup(startup)
    app.on_shutdown(hub.shutdown)
    ui.run(title="WashData mock plugs", host=args.web_host, port=args.web_port, show=False,
           reload=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
