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
"""The mock plug (devtools/mock_socket): its model, its MQTT contract, and its claims.

The model and topic modules import neither paho nor nicegui, so this runs in the fast
suite (the old synthesis test was skipped wherever nicegui was missing, i.e. in CI).
The scenario tests replay the plug's output through the real CycleDetector: a scenario
whose "expect" line the shipped detector contradicts is a broken scenario.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.const import STANDBY_BAND_WINDOW_S
from custom_components.ha_washdata.cycle_detector import CycleDetector
from devtools.mock_socket.model import (
    PLUG_MODES,
    SCENARIOS,
    PlugSim,
    Program,
    SourceCycle,
    TruthCycle,
    Variation,
    build_program,
    load_appliance,
    standby_band_w,
)
from devtools.mock_socket.topics import BRIDGE_STATUS, discovery, parse_command

NO_VARIATION = Variation(0.0, 0.0, 0.0)
T0 = datetime(2026, 10, 4, 8, 0, tzinfo=timezone.utc)


def _wash() -> list[list[float]]:
    """A 46 min wash at a 10-20 s cadence: heat, tumble, spin, pump-out."""
    trace, t, k = [], 0, 0
    while t < 600:
        trace.append([t, 2000.0])
        t += 10
    while t < 2400:
        trace.append([t, 150.0 if k % 3 < 2 else 20.0])
        t, k = t + 20, k + 1
    while t < 2700:
        trace.append([t, 500.0])
        t += 10
    return trace + [[2700, 5.0], [2760, 0.0], [2820, 0.0]]


def _cycle(cid: str, name: str, status: str = "completed", trace=None) -> dict:
    return {"id": cid, "profile_name": name, "status": status,
            "power_data": _wash() if trace is None else trace}


@pytest.fixture
def export_path(tmp_path):
    doc = {
        "version": 17,
        "data": {"past_cycles": [
            _cycle("a", "Cotton 40"),
            _cycle("b", "Cotton 40"),
            _cycle("cut", "Cotton 40", status="interrupted"),
            _cycle("tiny", "Quick", trace=[[0, 100.0], [10, 0.0]]),
        ]},
        "entry_data": {"device_type": "washing_machine"},
        "entry_options": {},
    }
    path = tmp_path / "export.json"
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def app(export_path):
    return load_appliance(export_path)


def _play(sim: PlugSim, program: Program, lead: float = 600.0, tail: float = 3 * 3600.0):
    events = sim.boot(0.0) + sim.advance(lead) + sim.start(program, lead)
    return events + sim.advance(lead + program.end + tail)


def _detect(app, events) -> list[dict]:
    """What the shipped detector records from the plug's reports (unmatched)."""
    ended: list[dict] = []
    det = CycleDetector(app.config, lambda _o, _n: None, ended.append)
    for e in events:
        ts = T0 + timedelta(seconds=e.t)
        if e.kind == "power":
            det.process_reading(e.power, ts)
        elif e.kind == "offline":
            det.mark_sensor_unavailable(ts)
    return ended


# --------------------------------------------------------------------------- corpus


def test_load_appliance_keeps_whole_programmes_only(app):
    assert app.device_type == "washing_machine"
    assert [c.cycle_id for c in app.cycles] == ["a", "b"]  # interrupted + 2-point dropped
    assert app.programs() == ["Cotton 40"]
    assert app.config.min_off_gap == 480  # the shipped washing-machine default
    assert app.cadence == 20.0  # median spacing: the tumble phase reports every 20 s


def test_load_appliance_reads_diagnostics_dumps(tmp_path):
    doc = {"data": {"store_export": {
        "data": {"past_cycles": [_cycle("d", "Eco 50")]},
        "entry_data": {"device_type": "dishwasher"}, "entry_options": {"off_delay": 900},
    }}}
    path = tmp_path / "diag.json"
    path.write_text(json.dumps(doc))
    app = load_appliance(path)
    assert app.device_type == "dishwasher"
    assert app.config.off_delay == 900
    assert app.programs() == ["Eco 50"]


@pytest.mark.parametrize("payload", [[1, 2, 3], {"data": {"nothing": 1}}])
def test_load_appliance_rejects_other_files(tmp_path, payload):
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not a WashData export"):
        load_appliance(path)


def test_pick_unknown_programme_is_an_error(app):
    with pytest.raises(ValueError, match="no stored cycle"):
        app.pick("Wool", random.Random(0))


# --------------------------------------------------------------------------- variation


def test_variation_scales_the_whole_run_within_bounds(app):
    cycle = app.cycles[0]
    assert NO_VARIATION.apply(cycle, random.Random(0)).trace == cycle.trace
    rng = random.Random(1)
    for _ in range(50):
        varied = Variation(stretch=0.1, scale=0.2).apply(cycle, rng)
        k_t = varied.duration / cycle.duration
        assert 0.9 <= k_t <= 1.1
        # one factor for the whole run: every reading moves by the same ratio (to 0.1 W)
        ratios = [v[1] / c[1] for v, c in zip(varied.trace, cycle.trace) if c[1] >= 100]
        assert max(ratios) - min(ratios) < 0.002 and 0.8 <= ratios[0] <= 1.2
        assert [round(v[0] / k_t, 1) for v in varied.trace] == [round(c[0], 1) for c in cycle.trace]


def test_noise_leaves_zero_readings_at_zero(app):
    varied = Variation(0, 0, noise_w=20.0).apply(app.cycles[0], random.Random(2))
    for (_, w), (_, orig) in zip(varied.trace, app.cycles[0].trace):
        assert w >= 0.0
        if orig == 0.0:
            assert w == 0.0


# --------------------------------------------------------------------------- scenarios


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_every_scenario_builds_a_well_formed_programme(app, key):
    program = build_program(key, app, None, NO_VARIATION, random.Random(4))
    times = [t for t, _ in program.knots]
    assert times == sorted(times) and times[0] == 0.0
    assert all(w >= 0.0 for _, w in program.knots)
    assert len(program.truth) == {"back-to-back": 2, "idle-blips": 0}.get(key, 1)
    assert len(program.sources) == SCENARIOS[key].cycles
    for cyc in program.truth:
        assert 0.0 <= cyc.start < cyc.end <= program.end


def test_scenarios_are_sized_from_the_device_config(app):
    cfg = app.config
    rng = random.Random(5)

    soak = build_program("soak", app, None, NO_VARIATION, rng)
    quiet = longest_run(soak, lambda w: w < cfg.stop_threshold_w)
    assert quiet >= cfg.min_off_gap + 120.0

    standby = build_program("standby-above-stop", app, None, NO_VARIATION, rng)
    band = standby_band_w(cfg)
    assert cfg.stop_threshold_w < band < cfg.start_threshold_w
    assert longest_run(standby, lambda w: w == band) >= 3 * STANDBY_BAND_WINDOW_S

    dropout = build_program("dropout", app, None, NO_VARIATION, rng)
    (start, end), = dropout.outages
    assert end - start >= 2 * cfg.off_delay

    b2b = build_program("back-to-back", app, None, NO_VARIATION, rng)
    first, second = b2b.truth
    assert second.start - first.end == pytest.approx(1.5 * cfg.min_off_gap)

    pulled = build_program("plug-pull", app, None, NO_VARIATION, rng)
    clean_end = build_program("clean", app, None, NO_VARIATION, rng).truth[0].end
    assert pulled.truth[0].end == pytest.approx(0.9 * clean_end, abs=1.0)


def test_idle_blips_stay_under_both_start_gates(app):
    cfg = app.config
    for seed in range(20):
        program = build_program("idle-blips", app, None, NO_VARIATION, random.Random(seed))
        for (t, w), (t_next, _) in zip(program.knots, program.knots[1:]):
            if w > 0:
                assert w > cfg.start_threshold_w  # it does cross the power threshold...
                assert t_next - t < cfg.start_duration_threshold  # ...but not for long
                assert w * (t_next - t) / 3600.0 < cfg.start_energy_threshold


def longest_run(program: Program, pred) -> float:
    best, since = 0.0, None
    for (t, w), (t_next, _) in zip(program.knots, program.knots[1:]):
        if pred(w):
            since = t if since is None else since
            best = max(best, t_next - since)
        else:
            since = None
    return best


@pytest.mark.parametrize("key, cycles", [
    ("clean", 1), ("delay-start", 1), ("dropout", 1), ("plug-pull", 1),
    ("anti-crease", 1), ("back-to-back", 2), ("idle-blips", 0),
])
def test_the_shipped_detector_agrees_with_the_scenario(app, key, cycles):
    """Under the recorded plug, each scenario produces what its "expect" line says.

    Soak and standby-above-stop are not here: both claims are about a MATCHED cycle
    (item 390, the standby-band finalize), which an unmatched replay cannot make.
    """
    program = build_program(key, app, None, NO_VARIATION, random.Random(6))
    ended = _detect(app, _play(PlugSim(PLUG_MODES["recorded"]), program))
    assert [d["status"] for d in ended] == ["completed"] * cycles


# --------------------------------------------------------------------------- the plug


def _steady(level: float, minutes: float) -> Program:
    knots = ((0.0, level), (minutes * 60.0, level))
    return Program("clean", knots, (TruthCycle("x", 0.0, minutes * 60.0),),
                   (SourceCycle("x", knots, "x", "past"),))


def _power(events):
    return [e for e in events if e.kind == "power"]


def test_recorded_mode_reports_at_the_trace_times(app):
    program = build_program("clean", app, None, NO_VARIATION, random.Random(0))
    sim = PlugSim(PLUG_MODES["recorded"])
    events = sim.boot(0.0) + sim.start(program, 100.0) + sim.advance(100.0 + program.end)
    in_run = [e.t - 100.0 for e in _power(events) if e.t >= 100.0]
    assert in_run == pytest.approx([t for t, _ in program.knots])


def test_silent_mode_says_nothing_at_a_steady_draw():
    sim = PlugSim(PLUG_MODES["silent"])
    events = sim.boot(0.0) + sim.start(_steady(300.0, 60), 100.0) + sim.advance(3 * 3600.0)
    # boot at 0 W, the jump to 300 W, the run end back to 0 W - nothing else in 3 h
    assert [(e.t, e.power) for e in _power(events)] == [(0.0, 0.0), (100.0, 300.0), (3700.0, 0.0)]


def test_on_change_heartbeat_and_rate_limit():
    mode = PLUG_MODES["on-change"]
    sim = PlugSim(mode)
    events = sim.boot(0.0) + sim.advance(1000.0)
    assert [e.t for e in _power(events)] == [0.0, 300.0, 600.0, 900.0]

    knots = ((0.0, 100.0), (1.0, 200.0), (2.0, 300.0), (60.0, 300.0))
    burst = Program("clean", knots, ())
    sim = PlugSim(mode)
    events = sim.boot(0.0) + sim.advance(10.0) + sim.start(burst, 10.0) + sim.advance(100.0)
    reported = [(e.t, e.power) for e in _power(events)]
    # 100 W at 10 s; 200 W and 300 W arrive inside the 5 s limit, so only the
    # latest value goes out when it lifts; the run end (0 W) is reported at 70 s.
    assert reported == [(0.0, 0.0), (10.0, 100.0), (15.0, 300.0), (70.0, 0.0)]


def test_poll_mode_resends_unchanged_values():
    sim = PlugSim(PLUG_MODES["poll-30s"])
    events = sim.boot(0.0) + sim.start(_steady(50.0, 10), 0.0) + sim.advance(300.0)
    power = _power(events)
    assert [e.t for e in power] == [30.0 * i for i in range(11)]
    assert {e.power for e in power[1:10]} == {50.0}


def test_outage_takes_the_plug_offline_and_back(app):
    program = build_program("dropout", app, None, NO_VARIATION, random.Random(0))
    (start, end), = program.outages
    sim = PlugSim(PLUG_MODES["poll-30s"])
    events = sim.boot(0.0) + sim.start(program, 0.0) + sim.advance(program.end + 600.0)
    kinds = [(e.kind, e.t) for e in events if e.kind in ("online", "offline")]
    assert kinds == [("online", 0.0), ("offline", start), ("online", end)]
    assert not [e for e in _power(events) if start < e.t < end]
    assert any(e.t == end for e in _power(events))  # a reconnecting plug reports at once


def test_relay_off_freezes_the_programme():
    program = _steady(400.0, 30)
    sim = PlugSim(PLUG_MODES["recorded"])
    events = sim.boot(0.0) + sim.start(program, 0.0)
    events += sim.set_relay(False, 600.0) + sim.advance(900.0) + sim.set_relay(True, 900.0)
    events += sim.advance(10_000.0)
    assert (600.0, 0.0) in [(e.t, e.power) for e in _power(events)]
    (end,) = [e for e in events if e.kind == "run_end"]
    assert end.run.ended == pytest.approx(1800.0 + 300.0)  # 30 min + 5 min off
    ((_cyc, t0, t1),) = end.run.truth()
    assert (t0, t1) == (0.0, 2100.0)
    assert sim.energy_kwh == pytest.approx(400.0 * 1800.0 / 3.6e6)


def test_stop_cuts_the_truth_short(app):
    program = build_program("back-to-back", app, None, NO_VARIATION, random.Random(0))
    sim = PlugSim(PLUG_MODES["recorded"])
    sim.boot(0.0)
    sim.start(program, 0.0)
    events = sim.stop(1000.0)
    (end,) = [e for e in events if e.kind == "run_end"]
    assert end.run.status == "stopped"
    ((cyc, t0, t1),) = end.run.truth()  # the second load never started
    assert (t0, t1) == (program.truth[0].start, 1000.0)


@pytest.mark.parametrize("mode", list(PLUG_MODES))
def test_energy_is_exact_whatever_the_reporting(app, mode):
    program = build_program("anti-crease", app, None, NO_VARIATION, random.Random(3))
    expected_wh = sum(w * (t1 - t0) for (t0, w), (t1, _) in zip(program.knots, program.knots[1:])) / 3600
    sim = PlugSim(PLUG_MODES[mode])
    _play(sim, program, lead=100.0, tail=600.0)
    assert sim.energy_kwh * 1000 == pytest.approx(expected_wh, rel=1e-9)


# --------------------------------------------------------------------------- MQTT contract


def test_discovery_keeps_the_entities_existing_entries_point_at():
    """Two WashData entries read sensor.mock_washer_socket_mock_washer_power: its
    discovery topic and unique id must not move, or HA renames the entity."""
    configs = dict(discovery("mock_washer_power", "Mock Washer Socket",
                             programs=["Eco"], scenarios=["clean"], modes=["recorded"]))
    power = configs["homeassistant/sensor/mock_washer_power_power/config"]
    assert power["unique_id"] == "mock_washer_power_power"
    relay = configs["homeassistant/switch/mock_washer_power/config"]
    assert relay["unique_id"] == "mock_washer_power_switch"
    assert len({c["unique_id"] for c in configs.values()}) == len(configs)
    for payload in configs.values():
        assert payload["availability"][0] == {"topic": BRIDGE_STATUS}
        assert "expire_after" not in payload  # a silent plug is still a valid reading
    assert power["availability"][1] == {"topic": "washdata_mock/mock_washer_power/availability"}
    program = configs["homeassistant/select/mock_washer_power_program/config"]
    assert program["options"] == ["random", "Eco"]


@pytest.mark.parametrize("topic, payload, expected", [
    ("washdata_mock/p1/start", b"PRESS", ("p1", "start", "PRESS")),
    ("washdata_mock/p1/relay/set", b"OFF", ("p1", "relay", "OFF")),
    ("washdata_mock/p1/program/set", "Eco 50 ", ("p1", "program", "Eco 50")),
    ("washdata_mock/p1/power", b"12", None),
    ("homeassistant/switch/p1/set", b"ON", None),
])
def test_parse_command(topic, payload, expected):
    assert parse_command(topic, payload) == expected
