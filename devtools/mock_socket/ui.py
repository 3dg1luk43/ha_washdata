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
"""The web page: one card per plug. Holds no state; everything lives in the hub."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from nicegui import events, ui

from .model import PLUG_MODES, SCENARIOS, compression_advice
from .runner import LOG_BUFFER, REPO, UPLOAD_DIR, MockHub, PlugRuntime

CHART_WINDOW_S = 3 * 3600


def _hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _source_options(current: str) -> dict[str, str]:
    files = sorted(REPO.glob("cycle_data/**/*.json")) + sorted(UPLOAD_DIR.glob("*.json"))
    options = {str(p): str(p.relative_to(REPO)) for p in files}
    if current and current not in options:
        options[current] = current
    return options


def _program_options(plug: PlugRuntime) -> dict[str, str]:
    options = {"random": "random"}
    if plug.app is not None:
        counts: dict[str, int] = {}
        for cycle in plug.app.cycles:
            counts[cycle.program] = counts.get(cycle.program, 0) + 1
        options.update({name: f"{name} ({n})" for name, n in sorted(counts.items())})
    return options


def _chart_data(plug: PlugRuntime) -> list[list[float | None]]:
    cutoff = time.time() - CHART_WINDOW_S
    return [[ts * 1000, w] for ts, w in plug.history if ts >= cutoff]


def _chip(chip: ui.chip, text: str, color: str) -> None:
    chip.set_text(text)
    chip.props(f"color={color} text-color=white")


def _plug_card(hub: MockHub, plug: PlugRuntime) -> Callable[[], None]:
    s = plug.s
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            with ui.column().classes("gap-0"):
                ui.label(s.name).classes("text-lg font-bold")
                ui.label(s.id).classes("text-xs text-gray-500 font-mono")
            with ui.row().classes("items-center gap-1"):
                online_chip = ui.chip(icon="wifi")
                relay_chip = ui.chip(icon="power")
                run_chip = ui.chip(icon="local_laundry_service")
            with ui.column().classes("items-end gap-0"):
                power_lbl = ui.label().classes("text-4xl font-mono font-bold text-blue-600")
                energy_lbl = ui.label().classes("text-xs text-gray-500 font-mono")
        playing_lbl = ui.label().classes("text-sm font-medium")
        expect_lbl = ui.label().classes("text-xs text-gray-600")

        def start() -> None:
            error = plug.start()
            if error:
                ui.notify(error, type="warning")

        with ui.row().classes("gap-2"):
            start_btn = ui.button("Start", icon="play_arrow", on_click=start).props("color=positive")
            stop_btn = ui.button("Stop", icon="stop", on_click=plug.stop).props("color=negative outline")
            relay_btn = ui.button(icon="power_settings_new",
                                  on_click=lambda: plug.set_relay(not plug.sim.relay_on)).props("outline")
            net_btn = ui.button(icon="wifi_off",
                                on_click=lambda: plug.set_connected(not plug.sim.plugged)).props("outline")
        relay_btn.tooltip("Relay off cuts the appliance's power and freezes its programme "
                          "(what WashData's pause-via-switch option does)")
        net_btn.tooltip("Takes the plug off the network: Home Assistant sees it unavailable "
                        "while the appliance keeps running")

        with ui.expansion("Next run", icon="tune", value=plug.app is None).classes("w-full"):
            with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-1"):
                program_sel = ui.select(_program_options(plug), value=s.program, label="Program",
                                        with_input=True,
                                        on_change=lambda e: plug.set_option("program", e.value))

                def load(path: str | None) -> None:
                    if not path or path == plug.s.source:
                        return
                    error = plug.load_source(path)
                    if error:
                        ui.notify(error, type="negative")
                        return
                    program_sel.set_options(_program_options(plug), value=plug.s.program)
                    ui.notify(f"{len(plug.app.cycles)} cycles, {len(plug.app.programs())} programmes "
                              f"({plug.app.device_type})")

                source_sel = ui.select(_source_options(s.source), value=s.source or None,
                                       label="Source: export or diagnostics dump", with_input=True,
                                       on_change=lambda e: load(e.value))

                async def upload(e: events.UploadEventArguments) -> None:
                    UPLOAD_DIR.mkdir(exist_ok=True)
                    dest = UPLOAD_DIR / Path(e.file.name).name
                    dest.write_bytes(await e.file.read())
                    source_sel.set_options(_source_options(str(dest)), value=str(dest))

                ui.select({k: f"{v.title} ({v.ref})" for k, v in SCENARIOS.items()},
                          value=s.scenario, label="Scenario",
                          on_change=lambda e: plug.set_option("scenario", e.value))
                ui.upload(label="Upload a file", auto_upload=True, on_upload=upload) \
                    .props("accept=.json flat bordered").classes("w-full")
                ui.select({k: m.title for k, m in PLUG_MODES.items()}, value=s.mode,
                          label="Plug reporting", on_change=lambda e: plug.set_option("mode", e.value))
                ui.number("Idle draw", value=s.idle_w, min=0, step=0.1, suffix="W",
                          on_change=lambda e: plug.set_option("idle_w", float(e.value or 0)))
                ui.number("Duration stretch", value=round(s.stretch * 100), min=0, max=50, step=1,
                          prefix="±", suffix="%",
                          on_change=lambda e: plug.set_option("stretch", float(e.value or 0) / 100))
                ui.number("Amplitude scale", value=round(s.scale * 100), min=0, max=50, step=1,
                          prefix="±", suffix="%",
                          on_change=lambda e: plug.set_option("scale", float(e.value or 0) / 100))
                ui.number("Noise σ", value=s.noise_w, min=0, step=0.5, suffix="W",
                          on_change=lambda e: plug.set_option("noise_w", float(e.value or 0)))
                with ui.row().classes("items-center"):
                    ui.switch("Repeat", value=s.repeat,
                              on_change=lambda e: plug.set_option("repeat", bool(e.value)))
                    ui.number("gap", value=s.gap_min, min=0, step=1, suffix="min",
                              on_change=lambda e: plug.set_option("gap_min", float(e.value or 0))) \
                        .classes("w-24")
            scenario_doc = ui.label().classes("text-xs text-gray-600")
            mode_doc = ui.label().classes("text-xs text-gray-600")

            async def remove() -> None:
                with ui.dialog() as dialog, ui.card():
                    ui.label(f"Remove {s.name}? Its entities disappear from Home Assistant, "
                             "and any WashData device using its power sensor stops getting readings.")
                    with ui.row():
                        ui.button("Remove", on_click=lambda: dialog.submit(True)).props("color=negative")
                        ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
                if await dialog:
                    await hub.remove_plug(s.id)
                    ui.navigate.reload()

            ui.button("Remove plug", icon="delete", on_click=remove).props("flat dense color=negative")

        chart = ui.echart({
            "animation": False,
            "grid": {"left": 48, "right": 16, "top": 16, "bottom": 28},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "time"},
            "yAxis": {"type": "value", "name": "W"},
            "series": [{"type": "line", "step": "end", "showSymbol": True, "symbolSize": 3,
                        "connectNulls": False, "areaStyle": {"opacity": 0.12},
                        "data": _chart_data(plug)}],
        }).classes("w-full h-56")
        ui.label("Each dot is one report Home Assistant received; gaps are the plug offline.") \
            .classes("text-xs text-gray-500")
        ledger_box = ui.column().classes("w-full gap-0 text-xs font-mono")

    seen = {"reports": plug.reports, "ledger": None}

    def refresh() -> None:
        sim, clock = plug.sim, hub.clock
        now = clock.now()
        power_lbl.set_text(f"{sim.power:.1f} W")
        last = sim.last_report
        ago = f"last report {_hms(now - last[0])} ago" if last else "no report yet"
        energy_lbl.set_text(f"{sim.energy_kwh:.3f} kWh · {ago}")
        _chip(online_chip, "online" if sim.online else "offline", "green" if sim.online else "grey")
        _chip(relay_chip, "relay on" if sim.relay_on else "relay off",
              "green" if sim.relay_on else "orange")
        run = sim.run
        scenario = SCENARIOS[run.program.scenario if run else s.scenario]
        if run is not None:
            _chip(run_chip, "paused" if run.frozen_since is not None else "playing", "blue")
            playing_lbl.set_text(
                f"{run.program.name()} · {scenario.title} · {run.mode} · "
                f"{_hms(run.program_time(now))} / {_hms(run.program.end)}")
        elif plug.repeat_at is not None:
            _chip(run_chip, "waiting", "purple")
            playing_lbl.set_text(f"Next run in {_hms(plug.repeat_at - now)}")
        else:
            _chip(run_chip, "idle", "grey")
            playing_lbl.set_text(plug.load_error or ("Pick a source file to start"
                                                     if plug.app is None else "Idle"))
        expect_lbl.set_text(f"Expect: {scenario.expect}")
        start_btn.set_enabled(run is None and plug.app is not None)
        stop_btn.set_enabled(run is not None or plug.repeat_at is not None)
        relay_btn.set_text("Relay off" if sim.relay_on else "Relay on")
        net_btn.set_text("Disconnect" if sim.plugged else "Reconnect")
        scenario_doc.set_text(f"{SCENARIOS[s.scenario].title}: {SCENARIOS[s.scenario].expect}")
        mode_doc.set_text(f"{PLUG_MODES[s.mode].title}: {PLUG_MODES[s.mode].why}")

        if plug.reports != seen["reports"]:
            seen["reports"] = plug.reports
            chart.run_chart_method("setOption", {"series": [{"data": _chart_data(plug)}]})
        head = plug.ledger[0] if plug.ledger else None
        if head is not seen["ledger"]:
            seen["ledger"] = head
            ledger_box.clear()
            with ledger_box:
                for rec in list(plug.ledger)[:6]:
                    truth = "; ".join(
                        f"{t['program']} {t['start'][11:19]}-{t['end'][11:19]} ({t['minutes']} min)"
                        for t in rec["truth"]) or "no cycle"
                    ui.label(f"{rec['started'][:19]}Z  {rec['scenario']} · {rec['plug_mode']} · "
                             f"{rec['status']}  ->  {truth}")

    return refresh


def mount(hub: MockHub) -> None:
    """Register the page."""

    @ui.page("/")
    def index() -> None:
        ui.page_title("WashData mock plugs")
        with ui.header().classes("items-center justify-between"):
            ui.label("WashData mock plugs").classes("text-lg font-bold")
            mqtt_lbl = ui.label().classes("text-sm")
        if hub.clock.speedup > 1.0:
            with ui.card().classes("w-full bg-amber-50"):
                for plug in hub.plugs.values():
                    for line in compression_advice(plug.app, hub.clock.speedup):
                        ui.label(line).classes("text-sm")
        refreshers = [_plug_card(hub, plug) for plug in hub.plugs.values()]

        with ui.expansion("Add a plug", icon="add").classes("w-full"):
            with ui.row().classes("items-end"):
                new_id = ui.input("id (entity ids derive from it)").props("dense")
                new_name = ui.input("Name").props("dense")

                def add() -> None:
                    plug_id = (new_id.value or "").strip().lower().replace(" ", "_")
                    if not plug_id.replace("_", "").isalnum():
                        ui.notify("id: letters, digits and _ only", type="warning")
                        return
                    try:
                        hub.add_plug(plug_id, (new_name.value or "").strip())
                    except ValueError as err:
                        ui.notify(str(err), type="warning")
                        return
                    ui.navigate.reload()

                ui.button("Add", on_click=add)

        with ui.expansion("Log", icon="list").classes("w-full"):
            log_box = ui.label().classes("whitespace-pre font-mono text-xs")

        def tick() -> None:
            link = hub.link
            if link is None:
                mqtt_lbl.set_text("MQTT off")
            else:
                where = f"{link.settings.host}:{link.settings.port}"
                state = "connected to" if link.connected else "connecting to"
                mqtt_lbl.set_text(f"MQTT {state} {where}{' (' + link.error + ')' if link.error else ''}"
                                  f" · {hub.clock.speedup:g}x")
            for refresh in refreshers:
                refresh()
            log_box.set_text("\n".join(list(LOG_BUFFER.lines)[-40:]))

        ui.timer(0.1, tick, once=True)
        ui.timer(1.0, tick)
