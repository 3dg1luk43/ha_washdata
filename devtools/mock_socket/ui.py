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
"""The web page: a tab per plug, a typeset power plot to measure on, and the runs.

The plot is the instrument. It shows two step traces - what Home Assistant received and
what the appliance really drew - over bands for the cycles the appliance really ran, the
plug's offline time and relay-off pauses. Wheel zooms, drag pans, Select mode measures a
stretch (and the selection can be dragged and resized), a run in the table zooms to it.

The plot's data lives in the browser (``window.wd`` below): a page receives the history
once and then only what is new each second.
"""
from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import events, ui

from .measure import Measurement, hms, measure
from .model import PLUG_MODES, SCENARIOS, compression_advice
from .runner import LOG_BUFFER, REPO, UPLOAD_DIR, MockHub, PlugRuntime

INK, INK_SOFT, RULE, GRID = "#1B1F24", "#5B6470", "#DCE1E7", "#EEF1F4"
BLUE, ORANGE, GREEN, RED, PURPLE = "#5E81B5", "#E19C24", "#8FB032", "#EB6235", "#8778B3"
FONT_UI = "'Source Sans 3', system-ui, sans-serif"
FONT_MATH = "'Source Serif 4', Georgia, serif"
#: Plot windows, in seconds; None = everything kept (48 h).
WINDOWS = {"15 min": 900, "1 h": 3600, "3 h": 10800, "12 h": 43200, "All": None}
STALE_S = 300

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600&family=Source+Serif+4:ital@1&display=swap">
<style>
:root { --paper:#F3F5F7; --canvas:#FFFFFF; --ink:%(ink)s; --ink-soft:%(soft)s; --rule:%(rule)s;
        --blue:%(blue)s; --orange:%(orange)s; --green:%(green)s; --red:%(red)s; --purple:%(purple)s; }
body { background: var(--paper); color: var(--ink); font-family: %(ui)s; font-size: 14px; }
.q-field, .q-btn, .q-tab, .q-item, .q-table { font-family: %(ui)s; }
.wd-num { font-variant-numeric: tabular-nums; }
.wd-q { font-family: %(math)s; font-style: italic; font-weight: 400; }
.wd-header { background: var(--canvas); color: var(--ink); border-bottom: 1px solid var(--rule); }
.wd-brand { font-size: 17px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; }
.wd-brand span { font-weight: 300; color: var(--ink-soft); }
.wd-tabs .q-tab { min-height: 52px; padding: 0 14px; }
.wd-tabs .q-tab__indicator { height: 2px; background: var(--ink); }
.wd-tab-label { display: flex; align-items: baseline; gap: 8px; color: var(--ink); }
.wd-tab-w { color: var(--ink-soft); font-variant-numeric: tabular-nums; min-width: 5.5em; text-align: right; }
.wd-dot { width: 8px; height: 8px; border-radius: 50%%; display: inline-block; flex-shrink: 0; align-self: center; }
.wd-page { padding: 18px 20px 28px; gap: 18px; max-width: 1800px; margin: 0 auto; }
.wd-rail { width: 340px; flex-shrink: 0; gap: 14px; }
.wd-main { flex: 1 1 auto; min-width: 0; gap: 12px; }
.wd-panel { background: var(--canvas); border: 1px solid var(--rule); border-radius: 3px; }
.wd-section { padding: 14px 16px; }
.wd-section + .wd-section { border-top: 1px solid var(--rule); }
.wd-title { font-size: 20px; font-weight: 600; line-height: 1.2; }
.wd-muted { color: var(--ink-soft); }
.wd-small { font-size: 12.5px; line-height: 1.45; }
.wd-readout { font-size: 60px; font-weight: 300; line-height: 1; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.wd-unit { font-size: 22px; font-weight: 300; color: var(--ink-soft); margin-left: 6px; }
.wd-status { display: flex; gap: 14px; flex-wrap: wrap; }
.wd-status > div { display: flex; align-items: center; gap: 6px; }
.wd-stale { color: #A15C00; }
.wd-kv { display: grid; grid-template-columns: auto auto; justify-content: space-between; gap: 3px 20px; }
.wd-kv > .k { color: var(--ink-soft); }
.wd-kv > .v { text-align: right; font-variant-numeric: tabular-nums; }
.wd-kv[style*="start"] > .v { text-align: left; }
.wd-measure { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px 28px; }
.wd-measure .k { color: var(--ink-soft); font-size: 12.5px; }
.wd-measure .v { font-size: 19px; font-variant-numeric: tabular-nums; }
.wd-warn { color: #A15C00; } .wd-bad { color: #B3261E; }
.wd-toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 8px 10px; }
.wd-toolbar .q-btn-group, .wd-toolbar .q-btn-toggle { box-shadow: none; border: 1px solid var(--rule); }
.wd-plot { height: max(420px, calc(100vh - 430px)); width: 100%%; }
.wd-hint { color: var(--ink-soft); font-size: 12.5px; }
.wd-strip { background: #FFF6E5; border: 1px solid #F1D7A6; border-radius: 3px; color: #6B4A00; font-size: 13px; }
.wd-log { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 11.5px; white-space: pre-wrap; line-height: 1.5; }
.wd-table .q-table th { font-weight: 600; color: var(--ink-soft); }
.wd-table tbody tr { cursor: pointer; }
kbd { font-family: %(ui)s; border: 1px solid var(--rule); border-bottom-width: 2px; border-radius: 3px; padding: 0 5px; font-size: 12px; }
:focus-visible { outline: 2px solid var(--blue) !important; outline-offset: 2px; }
.wd-key { display: flex; gap: 16px; flex-wrap: wrap; padding: 0 12px 10px 70px; color: var(--ink-soft); font-size: 12.5px; }
.wd-key > span { display: inline-flex; align-items: center; gap: 6px; }
.wd-key i { width: 14px; height: 10px; display: inline-block; border-radius: 1px; }
@media (max-width: 960px) {
  .wd-layout { flex-direction: column; align-items: stretch !important; }
  .wd-rail, .wd-main { width: 100%%; }
  .wd-key { padding-left: 12px; }
}
@media (max-width: 720px) { .wd-hide-sm { display: none !important; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
<script>
window.wd = {
  s: {},
  chart(id) { const el = getElement(id); return el ? el.chart : null; },
  push(id, p) {
    let st = this.s[id];
    if (!st || p.reset) st = this.s[id] = {rep: [], draw: [], bands: [], now: 0, view: null};
    for (const q of p.rep) st.rep.push(q);
    for (const q of p.draw) st.draw.push(q);
    let cut = 0;
    while (cut < st.rep.length && st.rep[cut][0] < p.cutoff) cut++;
    if (cut) st.rep.splice(0, cut);
    cut = 0;
    while (cut < st.draw.length && st.draw[cut][0] < p.cutoff) cut++;
    if (cut) st.draw.splice(0, cut);
    if (p.bands) st.bands = p.bands;
    if (p.view) st.view = p.view;
    if (p.axis_max) st.axisMax = p.axis_max;
    st.now = p.now;
    this.apply(id);
  },
  apply(id) {
    const st = this.s[id], c = this.chart(id);
    if (!st || !c || !c.getDom().offsetParent) return;  // hidden tab: drawn when shown
    if (!c.__wdBound) {
      c.__wdBound = true;
      // ECharts' own dblclick only fires on drawn data, not on the empty plot.
      c.getZr().on('dblclick', () => getElement(id).$emit('wd-live', {}));
      c.on('datazoom', () => {
        // Hold a hand-set window in absolute time: ECharts keeps it as percentages,
        // which would slide it as new readings extend the axis. (rangeMode 'value'
        // would do this natively but stops the wheel from zooming at all.)
        const z = c.getOption().dataZoom[0], cur = this.s[id];
        if (cur) cur.hold = [z.startValue, z.endValue];
        this.relabel(id);
      });
    }
    const opt = {series: [
      {id: 'draw', data: st.draw, markLine: {data: [{xAxis: st.now}]}},
      {id: 'rep', data: st.rep, markArea: {data: st.bands}},
    ], xAxis: {max: st.axisMax ? (v => Math.max(v.max, st.axisMax)) : null}};
    if (st.view) { st.hold = st.view; st.view = null; }
    if (st.hold) {
      opt.dataZoom = [{id: 'dzIn', startValue: st.hold[0], endValue: st.hold[1]},
                      {id: 'dzSl', startValue: st.hold[0], endValue: st.hold[1]}];
    }
    c.setOption(opt);
    this.relabel(id);
  },
  relabel(id) {
    // A band names itself only where its name fits inside it, so zoomed-out runs
    // do not pile their labels on top of each other.
    const st = this.s[id], c = this.chart(id);
    if (!st || !c) return;
    const px = v => c.convertToPixel({xAxisIndex: 0}, v);
    const data = st.bands.map(([a, b]) => {
      const fits = !!a.name && (px(b.xAxis) - px(a.xAxis)) > 6.5 * a.name.length + 14;
      return [Object.assign({}, a, {label: Object.assign({}, a.label, {show: fits})}), b];
    });
    c.setOption({series: [{id: 'rep', markArea: {data: data}}]});
  },
  shown(id) { const c = this.chart(id); if (c) { c.resize(); this.apply(id); } },
  mode(id, select) {
    const c = this.chart(id); if (!c) return;
    c.dispatchAction({type: 'takeGlobalCursor', key: 'brush',
      brushOption: select ? {brushType: 'lineX', brushMode: 'single'} : {brushType: false}});
  },
  select(id, a, b) {
    const c = this.chart(id); if (!c) return;
    c.dispatchAction({type: 'brush', areas: a == null ? [] :
      [{brushType: 'lineX', xAxisIndex: 0, coordRange: [a, b]}]});
  },
  visible(id) {
    const c = this.chart(id); if (!c) return null;
    const z = c.getOption().dataZoom[0];
    return [z.startValue, z.endValue];
  },
  png(id, name) {
    const c = this.chart(id); if (!c) return;
    const a = document.createElement('a');
    a.href = c.getDataURL({type: 'png', pixelRatio: 2, backgroundColor: '#FFFFFF',
                           excludeComponents: ['dataZoom']});
    a.download = name; a.click();
  },
};
</script>
""" % {"ink": INK, "soft": INK_SOFT, "rule": RULE, "blue": BLUE, "orange": ORANGE,
       "green": GREEN, "red": RED, "purple": PURPLE, "ui": FONT_UI, "math": FONT_MATH}


def _local(ms_or_iso: float | str, fmt: str = "%H:%M:%S") -> str:
    if isinstance(ms_or_iso, str):
        return datetime.fromisoformat(ms_or_iso).astimezone().strftime(fmt)
    return datetime.fromtimestamp(ms_or_iso / 1000.0).strftime(fmt)


def _iso_ms(stamp: str) -> float:
    return datetime.fromisoformat(stamp).timestamp() * 1000.0


def _source_options(current: str) -> dict[str, str]:
    files = sorted(REPO.glob("cycle_data/**/*.json")) + sorted(UPLOAD_DIR.glob("*.json"))
    options = {str(p): str(p.relative_to(REPO)) for p in files}
    if current and current not in options:
        options[current] = current
    return options


def _program_options(plug: PlugRuntime) -> dict[str, str]:
    options = {"random": "Any programme, at random"}
    if plug.app is not None:
        counts: dict[str, int] = {}
        for cycle in plug.app.cycles:
            counts[cycle.program] = counts.get(cycle.program, 0) + 1
        options.update({name: f"{name} ({n} stored)" for name, n in sorted(counts.items())})
    return options


def _axis(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "nameTextStyle": {"color": INK, "fontSize": 13,
                          "rich": {"q": {"fontFamily": FONT_MATH, "fontStyle": "italic",
                                         "fontSize": 15, "color": INK}}},
        "axisLine": {"show": True, "onZero": False, "lineStyle": {"color": INK}},
        "axisTick": {"show": True, "inside": True, "length": 6, "lineStyle": {"color": INK}},
        "minorTick": {"show": True, "splitNumber": 5, "length": 3, "lineStyle": {"color": INK_SOFT}},
        "splitLine": {"show": True, "lineStyle": {"color": GRID}},
        "axisLabel": {"color": INK_SOFT, "fontFamily": FONT_UI, "fontSize": 12, "hideOverlap": True},
        **extra,
    }


def _chart_options() -> dict[str, Any]:
    return {
        "animation": False,
        "textStyle": {"fontFamily": FONT_UI},
        "grid": {"left": 70, "right": 28, "top": 40, "bottom": 78, "show": True,
                 "borderColor": INK, "borderWidth": 1, "backgroundColor": "#FFFFFF"},
        "legend": {"top": 8, "left": "center", "itemWidth": 22, "itemHeight": 3, "icon": "rect",
                   "textStyle": {"color": INK_SOFT, "fontSize": 12.5}, "data": ["Reported", "True draw"]},
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross", "snap": False,
                            "lineStyle": {"color": "#9AA4AF", "type": "dashed"},
                            "crossStyle": {"color": "#9AA4AF"},
                            "label": {"backgroundColor": INK, "fontFamily": FONT_UI}},
            "backgroundColor": "#FFFFFF", "borderColor": RULE, "borderWidth": 1,
            "textStyle": {"color": INK, "fontSize": 12.5, "fontFamily": FONT_UI},
            ":valueFormatter": "v => v == null ? 'no reading' : Number(v).toFixed(1) + ' W'",
        },
        "xAxis": _axis("{q|t}", type="time", nameGap=10,
                       axisPointer={"label": {":formatter": "p => new Date(p.value).toLocaleString()"}}),
        "yAxis": _axis("{q|P} / W", type="value", min=0, nameGap=14,
                       axisPointer={"label": {":formatter": "p => Number(p.value).toFixed(1) + ' W'"}}),
        "dataZoom": [
            {"id": "dzIn", "type": "inside", "xAxisIndex": 0, "filterMode": "none",
             "zoomOnMouseWheel": True, "moveOnMouseMove": True,
             "moveOnMouseWheel": False, "preventDefaultMouseMove": True},
            {"id": "dzSl", "type": "slider", "xAxisIndex": 0, "filterMode": "none",
             "height": 28, "bottom": 12, "borderColor": RULE,
             "backgroundColor": "#FAFBFC", "fillerColor": "rgba(94,129,181,0.12)",
             "handleStyle": {"color": "#FFFFFF", "borderColor": INK_SOFT},
             "moveHandleStyle": {"color": "#C9D2DC"},
             "dataBackground": {"lineStyle": {"color": BLUE, "width": 0.7, "opacity": 0.8},
                                "areaStyle": {"color": BLUE, "opacity": 0.12}},
             "selectedDataBackground": {"lineStyle": {"color": BLUE, "width": 0.7},
                                        "areaStyle": {"color": BLUE, "opacity": 0.25}},
             "textStyle": {"color": INK_SOFT, "fontFamily": FONT_UI},
             ":labelFormatter": "v => new Date(v).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})"},
            {"id": "dzY", "type": "inside", "yAxisIndex": 0, "filterMode": "none",
             "zoomOnMouseWheel": "shift", "moveOnMouseMove": "shift", "moveOnMouseWheel": False},
        ],
        "brush": {"xAxisIndex": 0, "brushType": "lineX", "brushMode": "single",
                  "transformable": True, "removeOnClick": False,
                  "brushStyle": {"color": "rgba(27,31,36,0.06)", "borderColor": "rgba(27,31,36,0.6)",
                                 "borderWidth": 1},
                  "throttleType": "debounce", "throttleDelay": 150},
        "toolbox": {"show": False, "feature": {"brush": {"type": ["lineX", "clear"]}}},
        "series": [
            {"id": "draw", "name": "True draw", "type": "line", "step": "end", "showSymbol": False,
             "sampling": "lttb", "z": 2, "lineStyle": {"color": ORANGE, "width": 1.3},
             "itemStyle": {"color": ORANGE}, "emphasis": {"disabled": True}, "data": [],
             "markLine": {"silent": True, "symbol": "none", "animation": False,
                          "lineStyle": {"color": "#9AA4AF", "type": "dashed", "width": 1},
                          "label": {"show": True, "formatter": "now", "color": INK_SOFT,
                                    "fontFamily": FONT_UI, "position": "end"},
                          "data": []}},
            {"id": "rep", "name": "Reported", "type": "line", "step": "end", "showSymbol": True,
             "symbol": "circle", "symbolSize": 4, "connectNulls": False, "z": 3,
             "lineStyle": {"color": BLUE, "width": 1.6}, "itemStyle": {"color": BLUE},
             "emphasis": {"disabled": True}, "data": [],
             "markArea": {"silent": True, "animation": False, "data": []}},
        ],
    }


def _band(a: float, b: float, label: str, color: str, alpha: float, text: str,
          dashed: bool = False) -> list[dict[str, Any]]:
    r, g, bl = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    style = {"color": f"rgba({r},{g},{bl},{alpha})"}
    if dashed:
        style.update(borderColor=color, borderType="dashed", borderWidth=1)
    return [{"xAxis": a, "name": label, "itemStyle": style,
             "label": {"show": bool(label), "position": "insideTopLeft", "color": text,
                       "fontSize": 11.5,
                       "fontFamily": FONT_UI, "distance": 6}},
            {"xAxis": b}]


class PlugView:
    """One plug's tab, in one open page."""

    def __init__(self, hub: MockHub, plug: PlugRuntime) -> None:
        self.hub, self.plug = hub, plug
        self.follow = True
        self.window = "3 h"
        self.selecting = False
        self.selection: tuple[float, float] | None = None
        self._cursor = None
        self._bands_json = ""
        self._ledger_head: Any = object()

    # ----------------------------------------------------------------- build

    def tab_label(self) -> None:
        with ui.element("div").classes("wd-tab-label"):
            self.tab_dot = ui.element("span").classes("wd-dot")
            ui.label(self.plug.s.name)
            self.tab_w = ui.label().classes("wd-tab-w")

    def build(self) -> None:
        with ui.row().classes("wd-layout w-full no-wrap items-start").style("gap: 18px"):
            with ui.column().classes("wd-rail"):
                self._build_rail()
            with ui.column().classes("wd-main"):
                advice = compression_advice(self.plug.app, self.hub.clock.speedup)
                if advice:
                    with ui.expansion(advice[0], icon="schedule").classes("wd-strip w-full") \
                            .props("dense"):
                        for line in advice[1:]:
                            ui.label(line).classes("wd-small")
                self._build_plot()
                self._build_measure()
                self._build_runs()

    def _build_rail(self) -> None:
        plug, s = self.plug, self.plug.s
        with ui.element("div").classes("wd-panel w-full"):
            with ui.element("div").classes("wd-section"):
                ui.label(s.name).classes("wd-title")
                ui.label(f"Plug id {s.id}").classes("wd-small wd-muted")
                with ui.element("div").classes("wd-status wd-small").style("margin-top: 10px"):
                    self.st_net = self._status_item()
                    self.st_relay = self._status_item()
                    self.st_run = self._status_item()
            with ui.element("div").classes("wd-section"):
                with ui.row().classes("items-baseline no-wrap").style("gap: 0"):
                    self.readout = ui.label().classes("wd-readout")
                    ui.label("W").classes("wd-unit")
                self.last_report = ui.label().classes("wd-small wd-muted").style("margin-top: 6px")
                with ui.element("div").classes("wd-kv wd-small").style("margin-top: 10px"):
                    ui.html('<span class="wd-q">P</span> drawn now', sanitize=False).classes("k")
                    self.kv_draw = ui.label().classes("v")
                    ui.html('<span class="wd-q">E</span> since the plug was added', sanitize=False) \
                        .classes("k")
                    self.kv_energy = ui.label().classes("v")
            with ui.element("div").classes("wd-section"):
                self.run_title = ui.label().classes("text-base font-semibold")
                self.run_meta = ui.label().classes("wd-small wd-muted")
                self.run_bar = ui.linear_progress(value=0, show_value=False, size="6px") \
                    .props("color=grey-9 track-color=grey-3").style("margin: 10px 0 4px")
                with ui.row().classes("w-full justify-between wd-small wd-num"):
                    self.run_elapsed = ui.label()
                    self.run_end = ui.label().classes("wd-muted")
                self.run_expect = ui.label().classes("wd-small wd-muted").style("margin-top: 8px")
                with ui.row().classes("w-full").style("gap: 8px; margin-top: 12px"):
                    self.btn_start = ui.button("Start", icon="play_arrow", on_click=self._start) \
                        .props("color=grey-10 unelevated no-caps").classes("flex-grow")
                    self.btn_stop = ui.button("Stop", icon="stop", on_click=self._stop) \
                        .props("outline color=red-9 no-caps").classes("flex-grow")
                with ui.row().classes("w-full").style("gap: 18px; margin-top: 6px"):
                    self.sw_relay = ui.switch("Relay", value=plug.sim.relay_on,
                                              on_change=lambda e: self._relay(e.value)).props("color=grey-9")
                    self.sw_net = ui.switch("On the network", value=plug.sim.plugged,
                                            on_change=lambda e: self._net(e.value)).props("color=grey-9")
                self.sw_relay.tooltip("Off cuts the appliance's power and freezes its programme: "
                                      "what WashData's pause-via-switch option does")
                self.sw_net.tooltip("Off makes the plug unavailable in Home Assistant while the "
                                    "appliance keeps running")
            with ui.element("div").classes("wd-section"):
                self._build_settings()

    def _status_item(self) -> tuple[ui.element, ui.label]:
        with ui.element("div"):
            dot = ui.element("span").classes("wd-dot")
            text = ui.label()
        return dot, text

    def _build_settings(self) -> None:
        plug, s = self.plug, self.plug.s
        with ui.expansion("Next run", value=plug.app is None).classes("w-full") \
                .props("dense header-class=text-weight-medium"):
            with ui.column().classes("w-full").style("gap: 6px; padding-top: 6px"):
                with ui.row().classes("w-full no-wrap items-end").style("gap: 6px"):
                    self.sel_source = ui.select(_source_options(s.source), value=s.source or None,
                                                label="Cycles from", with_input=True) \
                        .props("dense options-dense").classes("flex-grow")
                    uploader = ui.upload(auto_upload=True, on_upload=self._upload) \
                        .props("accept=.json").classes("hidden")
                    ui.button(icon="upload", on_click=lambda: uploader.run_method("pickFiles")) \
                        .props("flat dense round").tooltip("Upload an export or diagnostics dump")
                self._uploader = uploader
                self.sel_source.on_value_change(lambda e: self._load(e.value))
                self.sel_program = ui.select(_program_options(plug), value=s.program, label="Programme",
                                             with_input=True,
                                             on_change=lambda e: plug.set_option("program", e.value)) \
                    .props("dense options-dense").classes("w-full")
                ui.select({k: v.title for k, v in SCENARIOS.items()}, value=s.scenario,
                          label="Scenario", on_change=lambda e: plug.set_option("scenario", e.value)) \
                    .props("dense options-dense").classes("w-full")
                self.doc_scenario = ui.label().classes("wd-small wd-muted")
                ui.select({k: m.title for k, m in PLUG_MODES.items()}, value=s.mode,
                          label="Reporting", on_change=lambda e: plug.set_option("mode", e.value)) \
                    .props("dense options-dense").classes("w-full")
                self.doc_mode = ui.label().classes("wd-small wd-muted")
                with ui.grid(columns=2).classes("w-full").style("gap: 2px 12px"):
                    ui.number("Duration ±", value=round(s.stretch * 100), min=0, max=50, step=1,
                              suffix="%", on_change=lambda e: plug.set_option(
                                  "stretch", float(e.value or 0) / 100)).props("dense")
                    ui.number("Power ±", value=round(s.scale * 100), min=0, max=50, step=1, suffix="%",
                              on_change=lambda e: plug.set_option(
                                  "scale", float(e.value or 0) / 100)).props("dense")
                    ui.number("Noise σ", value=s.noise_w, min=0, step=0.5, suffix="W",
                              on_change=lambda e: plug.set_option(
                                  "noise_w", float(e.value or 0))).props("dense")
                    ui.number("Idle draw", value=s.idle_w, min=0, step=0.1, suffix="W",
                              on_change=lambda e: plug.set_option(
                                  "idle_w", float(e.value or 0))).props("dense")
                with ui.row().classes("w-full items-center no-wrap").style("gap: 10px"):
                    ui.switch("Repeat", value=s.repeat,
                              on_change=lambda e: plug.set_option("repeat", bool(e.value))) \
                        .props("color=grey-9")
                    ui.number("every", value=s.gap_min, min=0, step=1, suffix="min after the last",
                              on_change=lambda e: plug.set_option(
                                  "gap_min", float(e.value or 0))).props("dense").classes("flex-grow")
                ui.button("Remove this plug", icon="delete_outline", on_click=self._remove) \
                    .props("flat dense no-caps color=red-9").classes("self-start")

    def _build_plot(self) -> None:
        with ui.element("div").classes("wd-panel w-full"):
            with ui.element("div").classes("wd-toolbar"):
                self.tg_window = ui.toggle(list(WINDOWS), value=self.window,
                                           on_change=lambda e: self.set_window(e.value)) \
                    .props("no-caps unelevated toggle-color=grey-9 padding='4px 12px'")
                self.btn_live = ui.button("Live", icon="sensors",
                                          on_click=lambda: self.set_window(self.window)) \
                    .props("no-caps unelevated padding='4px 12px'")
                self.btn_live.tooltip("Follow the newest reading (L)")
                ui.element("div").style("width: 1px; height: 22px; background: var(--rule)")
                self.tg_mode = ui.toggle({False: "Pan", True: "Select"}, value=False,
                                         on_change=lambda e: self.set_selecting(bool(e.value))) \
                    .props("no-caps unelevated toggle-color=grey-9 padding='4px 12px'")
                self.tg_mode.tooltip("Pan drags the plot; Select drags out a stretch to measure (M)")
                ui.element("div").classes("flex-grow")
                self.plot_hint = ui.label().classes("wd-hint")
                ui.button(icon="image", on_click=self._png).props("flat dense round") \
                    .tooltip("Save the plot as PNG")
                ui.button(icon="download", on_click=self._csv).props("flat dense round") \
                    .tooltip("Download the readings in view (or in the selection) as CSV")
            self.chart = ui.echart(_chart_options()).classes("wd-plot")
            ui.html(
                '<div class="wd-key">'
                f'<span><i style="background: rgba(143,176,50,.35)"></i>cycle the appliance really ran</span>'
                f'<span><i style="border: 1px dashed {GREEN}; background: rgba(143,176,50,.1)"></i>'
                'planned, still to come</span>'
                '<span><i style="background: rgba(235,98,53,.35)"></i>plug offline</span>'
                '<span><i style="background: rgba(135,120,179,.35)"></i>relay off</span>'
                '</div>', sanitize=False)
            self.chart.on("chart:brushEnd", self._on_brush, ["areas"])
            self.chart.on("chart:datazoom", self._on_user_zoom, ["type"])
            self.chart.on("wd-live", lambda _e: self.set_window(self.window))
        self._update_hint()

    def _build_measure(self) -> None:
        with ui.element("div").classes("wd-panel w-full"):
            with ui.element("div").classes("wd-section"):
                with ui.row().classes("w-full items-center justify-between"):
                    self.measure_title = ui.label("Measure").classes("text-base font-semibold")
                    with ui.row().style("gap: 4px"):
                        self.btn_copy = ui.button("Copy", icon="content_copy", on_click=self._copy) \
                            .props("flat dense no-caps padding='2px 10px'")
                        self.btn_fit = ui.button("Zoom to it", icon="zoom_in",
                                                 on_click=self.zoom_to_selection) \
                            .props("flat dense no-caps padding='2px 10px'")
                        self.btn_clear = ui.button("Clear", icon="close",
                                                   on_click=lambda: self.select(None)) \
                            .props("flat dense no-caps padding='2px 10px'")
                self.measure_hint = ui.html(
                    "Switch to <b>Select</b> (or press <kbd>M</kbd>) and drag across the plot to "
                    "measure that stretch. Click a run below to measure the whole run.",
                    sanitize=False).classes("wd-hint")
                self.measure_grid = ui.element("div").classes("wd-measure").style("margin-top: 10px")
            self._show_measure(None)

    def _build_runs(self) -> None:
        columns = [
            {"name": "started", "label": "Started", "field": "started", "align": "left"},
            {"name": "programs", "label": "Programme", "field": "programs", "align": "left"},
            {"name": "scenario", "label": "Scenario", "field": "scenario", "align": "left"},
            {"name": "mode", "label": "Reporting", "field": "mode", "align": "left"},
            {"name": "truth", "label": "Real cycles", "field": "truth", "align": "left"},
            {"name": "status", "label": "Status", "field": "status", "align": "left"},
        ]
        with ui.element("div").classes("wd-panel w-full wd-table"):
            with ui.element("div").classes("wd-section").style("padding-bottom: 4px"):
                ui.label("Runs").classes("text-base font-semibold")
                ui.label("Every run the plug played, with the cycles the appliance really ran. "
                         "Click one to zoom to it and measure it.").classes("wd-hint")
            self.runs = ui.table(rows=[], columns=columns, row_key="key", pagination=8) \
                .props("flat dense wrap-cells").classes("w-full")
            self.runs.on("rowClick", self._on_run_click)

    # ----------------------------------------------------------------- actions

    def _start(self) -> None:
        error = self.plug.start()
        if error:
            ui.notify(error, type="warning")
            return
        run = self.plug.sim.run
        ui.notify(f"Started {run.program.name()}: {SCENARIOS[run.program.scenario].title}, "
                  f"{PLUG_MODES[run.mode].title.lower()}")
        self.set_window(self.window)

    def _stop(self) -> None:
        self.plug.stop()
        ui.notify("Stopped")

    def _relay(self, on: bool) -> None:
        if on != self.plug.sim.relay_on:
            self.plug.set_relay(on)

    def _net(self, on: bool) -> None:
        if on != self.plug.sim.plugged:
            self.plug.set_connected(on)

    def _load(self, path: str | None) -> None:
        if not path or path == self.plug.s.source:
            return
        error = self.plug.load_source(path)
        if error:
            ui.notify(error, type="negative")
            return
        app = self.plug.app
        self.sel_program.set_options(_program_options(self.plug), value=self.plug.s.program)
        ui.notify(f"Loaded {len(app.cycles)} cycles of {len(app.programs())} programmes "
                  f"({app.device_type.replace('_', ' ')})")

    async def _upload(self, e: events.UploadEventArguments) -> None:
        UPLOAD_DIR.mkdir(exist_ok=True)
        dest = UPLOAD_DIR / Path(e.file.name).name
        dest.write_bytes(await e.file.read())
        self._uploader.reset()
        self.sel_source.set_options(_source_options(str(dest)), value=str(dest))

    async def _remove(self) -> None:
        s = self.plug.s
        with ui.dialog() as dialog, ui.card().style("max-width: 440px"):
            ui.label(f"Remove {s.name}?").classes("text-lg font-semibold")
            ui.label("Its entities disappear from Home Assistant, and any WashData device "
                     "reading its power sensor stops getting readings. The runs it played "
                     "stay in the ledger.").classes("wd-muted")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat no-caps")
                ui.button("Remove plug", on_click=lambda: dialog.submit(True)) \
                    .props("unelevated no-caps color=red-9")
        if await dialog:
            await self.hub.remove_plug(s.id)
            ui.navigate.reload()

    def set_window(self, name: str) -> None:
        self.window, self.follow = name, True
        self.tg_window.value = name
        self._push(force_view=True)

    def _update_hint(self) -> None:
        self.plot_hint.set_text(
            "Drag to select a stretch; drag the selection or its edges to adjust it"
            if self.selecting else
            "Wheel zooms time, Shift+wheel zooms power, drag pans, double-click goes live")

    def set_selecting(self, on: bool) -> None:
        self.selecting = on
        self.tg_mode.value = on
        self._update_hint()
        ui.run_javascript(f"wd.mode({self.chart.id}, {json.dumps(on)})")

    def select(self, span: tuple[float, float] | None, draw: bool = True) -> None:
        self.selection = (min(span), max(span)) if span else None
        if draw:
            a, b = self.selection or (None, None)
            ui.run_javascript(f"wd.select({self.chart.id}, {json.dumps(a)}, {json.dumps(b)})")
        self._show_measure(self._measure())

    def zoom_to_selection(self) -> None:
        if self.selection is None:
            return
        a, b = self.selection
        pad = max(30_000.0, 0.06 * (b - a))
        self._view(a - pad, b + pad)

    def _view(self, a: float, b: float) -> None:
        self.follow = False
        self._send({"reset": False, "rep": [], "draw": [], "view": [a, b]})

    def _on_brush(self, e: events.GenericEventArguments) -> None:
        areas = (e.args or {}).get("areas") or []
        rng = areas[0].get("coordRange") if areas else None
        self.select((float(rng[0]), float(rng[1])) if rng else None, draw=False)

    def _on_user_zoom(self, _e: events.GenericEventArguments) -> None:
        self.follow = False

    def _on_run_click(self, e: events.GenericEventArguments) -> None:
        row = e.args[1] if isinstance(e.args, list) and len(e.args) > 1 else None
        if not row or row.get("a") is None:
            return
        a, b = float(row["a"]), float(row["b"])
        self.select((a, b))
        self.zoom_to_selection()

    async def _copy(self) -> None:
        m = self._measure()
        if m is not None:
            ui.clipboard.write(m.as_text())
            ui.notify("Measurement copied")

    def _png(self) -> None:
        name = f"{self.plug.s.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ui.run_javascript(f"wd.png({self.chart.id}, {json.dumps(name)})")

    async def _csv(self) -> None:
        span = self.selection
        if span is None:
            span = await ui.run_javascript(f"return wd.visible({self.chart.id})", timeout=3)
        if not span or span[0] is None:
            ui.notify("Nothing in view to export", type="warning")
            return
        a, b = float(span[0]), float(span[1])
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["time", "series", "watts"])
        h = self.plug.history
        rows = [(ts, "reported", w) for ts, w in h.reported if a <= ts <= b]
        rows += [(ts, "drawn", w) for ts, w in h.draw if a <= ts <= b]
        for ts, series, w in sorted(rows, key=lambda r: r[0]):
            writer.writerow([datetime.fromtimestamp(ts / 1000).isoformat(timespec="milliseconds"),
                             series, "" if w is None else w])
        ui.download.content(out.getvalue(), f"{self.plug.s.id}_{_local(a, '%Y%m%d_%H%M%S')}.csv",
                            "text/csv")

    # ----------------------------------------------------------------- measurement

    def _measure(self) -> Measurement | None:
        if self.selection is None:
            return None
        h = self.plug.history
        return measure(h.reported, h.draw, *self.selection)

    def _show_measure(self, m: Measurement | None) -> None:
        for button in (self.btn_copy, self.btn_fit, self.btn_clear):
            button.set_visibility(m is not None)
        self.measure_hint.set_visibility(m is None)
        self.measure_grid.clear()
        if m is None:
            self.measure_title.set_text("Measure")
            return
        self.measure_title.set_text(
            f"{_local(m.start_ms)} to {_local(m.end_ms)}, {hms(m.duration_s)}")
        err = m.error_pct
        err_cls = "" if err is None or abs(err) < 2 else ("wd-warn" if abs(err) < 10 else "wd-bad")
        cells = [
            ("Energy from the reports", f"{m.energy_reported_wh:.2f} Wh", ""),
            ("Energy drawn", "-" if m.energy_true_wh is None else f"{m.energy_true_wh:.2f} Wh", ""),
            ("Reporting error", "-" if err is None else f"{err:+.2f} %", err_cls),
            ("Mean power", "-" if m.mean_w is None else f"{m.mean_w:.1f} W", ""),
            ("Min to max", "-" if m.min_w is None else f"{m.min_w:.1f} to {m.max_w:.1f} W", ""),
            ("Standard deviation", "-" if m.std_w is None else f"{m.std_w:.1f} W", ""),
            ("Peak drawn", "-" if m.true_peak_w is None else f"{m.true_peak_w:.1f} W", ""),
            ("Reports", str(m.reports), ""),
            ("Median interval", "-" if m.median_interval_s is None else f"{m.median_interval_s:.1f} s", ""),
            ("Longest silence", hms(m.longest_silence_s),
             "wd-warn" if m.longest_silence_s >= STALE_S else ""),
            ("Offline", hms(m.offline_s), "wd-warn" if m.offline_s else ""),
        ]
        with self.measure_grid:
            for key, value, cls in cells:
                with ui.element("div"):
                    ui.label(key).classes("k")
                    ui.label(value).classes(f"v {cls}")

    # ----------------------------------------------------------------- refresh

    def _bands(self, now_ms: float) -> list[list[dict[str, Any]]]:
        bands: list[list[dict[str, Any]]] = []
        cutoff = now_ms - self.plug.history.keep_ms
        for rec in self.plug.ledger:
            for t in rec.get("truth") or []:
                a, b = _iso_ms(t["start"]), _iso_ms(t["end"])
                if b >= cutoff:
                    bands.append(_band(a, b, t["program"], GREEN, 0.15, "#4E6B10"))
        for program, a, b in self.plug.planned_truth_ms():
            if a < now_ms:
                bands.append(_band(a, min(b, now_ms), program, GREEN, 0.15, "#4E6B10"))
            if b > now_ms:
                bands.append(_band(max(a, now_ms), b, "" if a < now_ms else program,
                                   GREEN, 0.06, "#4E6B10", dashed=True))
        for kind, a, b in self.plug.history.spans:
            end = b if b is not None else now_ms
            if end >= cutoff:
                color, text = (RED, "#9E3A17") if kind == "offline" else (PURPLE, "#4F4380")
                bands.append(_band(a, end, "", color, 0.16, text))
        return bands

    def _push(self, force_view: bool = False) -> None:
        now_ms = time.time() * 1000.0
        full, rep, draw = self.plug.history.since(self._cursor)
        self._cursor = self.plug.history.cursor()
        bands = self._bands(now_ms)
        bands_json = json.dumps(bands)
        payload: dict[str, Any] = {"reset": full, "rep": rep, "draw": draw,
                                   "cutoff": now_ms - self.plug.history.keep_ms}
        if bands_json != self._bands_json or full:
            payload["bands"] = bands
            self._bands_json = bands_json
        seconds = WINDOWS[self.window]
        # A little room right of "now", so the newest reading never sits on the frame.
        pad = max(60_000.0, (seconds or 3 * 3600) * 20.0)
        payload["axis_max"] = now_ms + pad
        if self.follow or force_view:
            if seconds is None:
                first = self.plug.history.reported[0][0] if self.plug.history.reported else now_ms - 3.6e6
                payload["view"] = [first, now_ms + pad]
            else:
                payload["view"] = [now_ms - seconds * 1000.0, now_ms + pad]
        self._send(payload)

    def _send(self, payload: dict[str, Any]) -> None:
        payload.setdefault("now", time.time() * 1000.0)
        payload.setdefault("cutoff", payload["now"] - self.plug.history.keep_ms)
        ui.run_javascript(f"wd.push({self.chart.id}, {json.dumps(payload)})")

    def shown(self) -> None:
        ui.run_javascript(f"wd.shown({self.chart.id})")

    def refresh(self) -> None:
        plug, sim, clock = self.plug, self.plug.sim, self.hub.clock
        now = clock.now()
        run = sim.run

        playing = run is not None
        state = ("paused" if playing and run.frozen_since is not None else
                 "playing" if playing else "waiting" if plug.repeat_at is not None else "idle")
        tab_color = {"playing": BLUE, "paused": PURPLE, "waiting": ORANGE}.get(state, "#B8C0C8")
        if not sim.online:
            tab_color = RED
        self.tab_dot.style(f"background: {tab_color}")
        self.tab_w.set_text(f"{sim.power:.1f} W")

        for (dot, text), ok, yes, no, bad in (
            (self.st_net, sim.online, "Online", "Offline", RED),
            (self.st_relay, sim.relay_on, "Relay on", "Relay off", PURPLE),
        ):
            dot.style(f"background: {GREEN if ok else bad}")
            text.set_text(yes if ok else no)
        self.st_run[0].style(f"background: {tab_color if state != 'idle' else '#B8C0C8'}")
        self.st_run[1].set_text(state.capitalize())

        last = sim.last_report
        self.readout.set_text(f"{last[1]:.1f}" if last and sim.online else "--")
        if last is None or not sim.online:
            self.last_report.set_text("Home Assistant gets no reading: the plug is offline"
                                      if not sim.online else "No report yet")
            self.last_report.classes(replace="wd-small wd-stale")
        else:
            ago = (now - last[0]) / clock.speedup
            stale = ago >= STALE_S
            self.last_report.set_text(
                f"Home Assistant has heard nothing for {hms(ago)}" if stale
                else f"Reported {hms(ago)} ago")
            self.last_report.classes(replace=f"wd-small {'wd-stale' if stale else 'wd-muted'}")
        self.kv_draw.set_text(f"{sim.power:.1f} W")
        self.kv_energy.set_text(f"{sim.energy_kwh:.3f} kWh")

        spec = SCENARIOS[run.program.scenario if playing else plug.s.scenario]
        if playing:
            elapsed, total = run.program_time(now), run.program.end
            self.run_title.set_text(run.program.name())
            self.run_meta.set_text(f"{spec.title}, {PLUG_MODES[run.mode].title.lower()}")
            self.run_bar.set_value(min(1.0, elapsed / total) if total else 0)
            self.run_elapsed.set_text(f"{hms(elapsed)} of {hms(total)}")
            ends = [b for _p, _a, b in plug.planned_truth_ms()]
            now_ms = time.time() * 1000.0
            future = [b for b in ends if b > now_ms]
            self.run_end.set_text(
                f"real cycle ends in {hms((min(future) - now_ms) / 1000)}" if future
                else "real cycle has ended" if ends else "no cycle in this run")
        else:
            self.run_title.set_text(
                "Next run in " + hms((plug.repeat_at - now) / clock.speedup)
                if plug.repeat_at is not None else
                (plug.load_error or "Choose where the cycles come from") if plug.app is None
                else "Ready")
            self.run_meta.set_text(f"{spec.title}, {PLUG_MODES[plug.s.mode].title.lower()}")
            self.run_bar.set_value(0)
            self.run_elapsed.set_text("")
            self.run_end.set_text("")
        self.run_expect.set_text(spec.expect)
        self.btn_start.set_enabled(not playing and plug.app is not None)
        self.btn_stop.set_enabled(playing or plug.repeat_at is not None)
        if self.sw_relay.value != sim.relay_on:
            self.sw_relay.value = sim.relay_on
        if self.sw_net.value != sim.plugged:
            self.sw_net.value = sim.plugged
        self.doc_scenario.set_text(SCENARIOS[plug.s.scenario].expect)
        self.doc_mode.set_text(PLUG_MODES[plug.s.mode].why)
        self.btn_live.props(f"color={'grey-10' if self.follow else 'grey-4'} "
                            f"text-color={'white' if self.follow else 'grey-9'}")

        self._push()
        if self.selection is not None and playing:
            self._show_measure(self._measure())
        head = plug.ledger[0] if plug.ledger else None
        if head is not self._ledger_head or playing:
            self._ledger_head = head
            self.runs.rows = self._run_rows()
            self.runs.update()

    def _run_rows(self) -> list[dict[str, Any]]:
        rows = []
        run = self.plug.sim.run
        if run is not None:
            planned = self.plug.planned_truth_ms()
            rows.append({
                "key": "playing", "started": _local(self.hub.clock.ms(run.started), "%d %b %H:%M"),
                "programs": run.program.name(), "scenario": SCENARIOS[run.program.scenario].title,
                "mode": PLUG_MODES[run.mode].title, "status": "Playing",
                "truth": ", ".join(f"{(c.end - c.start) / 60:.1f} min planned"
                                   for c in run.program.truth) or "none",
                "a": self.hub.clock.ms(run.started),
                "b": max([b for _p, _a, b in planned] + [self.hub.clock.ms(run.started + run.program.end)]),
            })
        for i, rec in enumerate(self.plug.ledger):
            truth = rec.get("truth") or []
            a = _iso_ms(rec["started"])
            b = _iso_ms(rec["ended"])
            rows.append({
                "key": f"{rec['started']}-{i}", "started": _local(rec["started"], "%d %b %H:%M"),
                "programs": ", ".join(c["program"] for c in rec.get("cycles_played") or []) or "(idle)",
                "scenario": SCENARIOS[rec["scenario"]].title if rec.get("scenario") in SCENARIOS
                else rec.get("scenario"),
                "mode": PLUG_MODES[rec["plug_mode"]].title if rec.get("plug_mode") in PLUG_MODES
                else rec.get("plug_mode"),
                "status": str(rec.get("status", "")).capitalize(),
                "truth": ", ".join(f"{t['minutes']} min" for t in truth) or "none",
                "a": a, "b": max(b, a + 60_000.0),
            })
        return rows


def mount(hub: MockHub) -> None:
    """Register the page."""
    ui.button.default_props("no-caps")

    @ui.page("/")
    def index() -> None:
        ui.page_title("WashData mock plugs")
        ui.add_head_html(HEAD)
        ui.colors(primary=INK)
        views: dict[str, PlugView] = {}

        with ui.header().classes("wd-header items-center no-wrap").style("padding: 0 20px; gap: 22px"):
            ui.html('WashData <span>mock plugs</span>', sanitize=False).classes("wd-brand")
            with ui.tabs().classes("wd-tabs").props("no-caps dense inline-label align=left "
                                                    "active-color=grey-10 indicator-color=grey-10") \
                    as tabs:
                for plug in hub.plugs.values():
                    view = views[plug.s.id] = PlugView(hub, plug)
                    with ui.tab(plug.s.id, label=""):
                        view.tab_label()
            ui.button(icon="add", on_click=lambda: add_dialog.open()) \
                .props("flat dense round color=grey-9").tooltip("Add a plug")
            ui.element("div").classes("flex-grow")
            with ui.row().classes("items-center no-wrap wd-small wd-hide-sm").style("gap: 8px"):
                mqtt_dot = ui.element("span").classes("wd-dot")
                mqtt_lbl = ui.label().classes("wd-muted")
                speed_lbl = ui.label().classes("wd-num")
            ui.button(icon="keyboard", on_click=lambda: keys_dialog.open()) \
                .props("flat dense round color=grey-9").tooltip("Keyboard shortcuts")
            ui.button(icon="article", on_click=lambda: log_drawer.toggle()) \
                .props("flat dense round color=grey-9").tooltip("Log")

        with ui.right_drawer(value=False).props("width=520 bordered") as log_drawer:
            ui.label("Log").classes("text-base font-semibold")
            log_box = ui.label().classes("wd-log")

        first = next(iter(views), None)
        with ui.tab_panels(tabs, value=first, animated=False).classes("w-full") \
                .style("background: transparent"):
            for plug_id, view in views.items():
                with ui.tab_panel(plug_id).classes("wd-page").style("padding: 18px 20px 28px"):
                    view.build()
        if not views:
            with ui.column().classes("wd-page items-start"):
                ui.label("No plugs yet").classes("wd-title")
                ui.label("Add one to start playing cycles into Home Assistant.").classes("wd-muted")
                ui.button("Add a plug", icon="add", on_click=lambda: add_dialog.open()) \
                    .props("unelevated color=grey-10")

        def on_tab(e: events.ValueChangeEventArguments) -> None:
            if e.value in views:
                views[e.value].shown()

        tabs.on_value_change(on_tab)

        with ui.dialog() as add_dialog, ui.card().style("min-width: 420px"):
            ui.label("Add a plug").classes("text-lg font-semibold")
            ui.label("Each plug is its own device in Home Assistant, with its own power sensor. "
                     "Entity ids derive from the id.").classes("wd-muted wd-small")
            new_name = ui.input("Name", placeholder="Mock Dishwasher Socket").classes("w-full")
            new_id = ui.input("Id", placeholder="mock_dishwasher").classes("w-full")
            new_source = ui.select(_source_options(""), label="Cycles from", with_input=True) \
                .classes("w-full")

            def add() -> None:
                plug_id = (new_id.value or "").strip().lower().replace(" ", "_").replace("-", "_")
                if not plug_id or not plug_id.replace("_", "").isalnum():
                    ui.notify("The id takes letters, digits and underscores", type="warning")
                    return
                try:
                    hub.add_plug(plug_id, (new_name.value or "").strip() or plug_id,
                                 new_source.value or "")
                except ValueError as err:
                    ui.notify(str(err), type="warning")
                    return
                ui.navigate.reload()

            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=add_dialog.close).props("flat")
                ui.button("Add plug", on_click=add).props("unelevated color=grey-10")

        with ui.dialog() as keys_dialog, ui.card().style("min-width: 360px"):
            ui.label("Keyboard shortcuts").classes("text-lg font-semibold")
            ui.html(
                '<div class="wd-kv" style="justify-content: start; gap: 6px 28px">'
                '<span class="k"><kbd>1</kbd> to <kbd>9</kbd></span><span class="v">switch plug</span>'
                '<span class="k"><kbd>M</kbd></span><span class="v">pan or select</span>'
                '<span class="k"><kbd>Esc</kbd></span><span class="v">clear the selection</span>'
                '<span class="k"><kbd>F</kbd></span><span class="v">zoom to the selection</span>'
                '<span class="k"><kbd>L</kbd></span><span class="v">follow live</span>'
                '<span class="k">Wheel</span><span class="v">zoom time</span>'
                '<span class="k">Shift + wheel</span><span class="v">zoom power</span>'
                '<span class="k">Drag</span><span class="v">pan, or select in Select mode</span>'
                '</div>', sanitize=False)

        def on_key(e: events.KeyEventArguments) -> None:
            if not e.action.keydown or e.modifiers.ctrl or e.modifiers.meta or e.modifiers.alt:
                return
            view = views.get(tabs.value)
            number = e.key.number
            if number is not None and 1 <= number <= len(views):
                tabs.set_value(list(views)[number - 1])
            elif view is None:
                return
            elif e.key.escape:
                view.select(None)
            elif e.key.name in ("m", "M"):
                view.set_selecting(not view.selecting)
            elif e.key.name in ("f", "F"):
                view.zoom_to_selection()
            elif e.key.name in ("l", "L"):
                view.set_window(view.window)
            elif e.key.name == "?":
                keys_dialog.open()

        ui.keyboard(on_key=on_key, repeating=False)

        def tick() -> None:
            link = hub.link
            if link is None:
                mqtt_dot.style(f"background: {RED}")
                mqtt_lbl.set_text("No broker")
            else:
                where = f"{link.settings.host}:{link.settings.port}"
                mqtt_dot.style(f"background: {GREEN if link.connected else ORANGE}")
                mqtt_lbl.set_text(f"Broker {where}" if link.connected else
                                  f"Connecting to {where}{': ' + link.error if link.error else ''}")
            speed_lbl.set_text(f"{hub.clock.speedup:g}x time")
            speed_lbl.classes(replace="wd-num " + ("wd-warn" if hub.clock.speedup > 1 else "wd-muted"))
            for view in views.values():
                view.refresh()
            if log_drawer.value:
                log_box.set_text("\n".join(list(LOG_BUFFER.lines)[-120:]))

        ui.timer(0.2, tick, once=True)
        ui.timer(1.0, tick)
