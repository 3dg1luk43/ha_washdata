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
"""What the mock appliance draws, and when the mock plug tells Home Assistant.

A real installation has three separate things, and the old mock blurred them into one
set of knobs:

* the **appliance**: a recorded cycle, played as the step function the detector itself
  assumes (a reading holds until the next one). Variation re-times and re-scales the
  whole run (`Variation`); it never warps it piecewise, which moved phases around.
* the **scenario**: what happens around that cycle. Each one is a failure mode the
  register measured, sized from the device's own detector config (`SCENARIOS`).
* the **plug**: when a reading reaches Home Assistant (`PLUG_MODES`). At the trace's own
  report times, on change with a heartbeat, on change with none (the silent plug of
  #424/#427), or polled (unchanged values re-sent, which WashData sees as
  ``state_reported``, #363).

Everything here runs on an abstract appliance clock in seconds and touches no MQTT, UI
or wall clock, so the tests drive exactly the code the live mock runs.
"""
from __future__ import annotations

import bisect
import importlib.util
import math
import random
import statistics
import sys
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom_components.ha_washdata.const import (  # noqa: E402
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DISHWASHER_MIN_CYCLE_DURATION_S,
    STANDBY_BAND_WINDOW_S,
    resolve_sampling_interval_default,
    resolve_watchdog_interval_default,
)
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig  # noqa: E402
from custom_components.ha_washdata.detector_config import build_detector_config  # noqa: E402

EPS = 1e-6
#: Statuses whose trace is not a whole programme (the detector filed it as cut short).
UNPLAYABLE_STATUSES = frozenset({"interrupted", "force_stopped"})
UNLABELLED = "(unlabelled)"


# --------------------------------------------------------------------------- corpus


def _corpus_module() -> Any:
    """``devtools/eval.py``, for its corpus unwrap (export / diagnostics / legacy)."""
    name = "wd_mock_socket_corpus"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "devtools" / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _offset(raw: Any) -> float | None:
    if isinstance(raw, str):  # pre-0.4 traces carried ISO stamps
        from datetime import datetime  # noqa: PLC0415

        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def normalise_trace(raw: Any) -> tuple[tuple[float, float], ...]:
    """``[[t, W], ...]`` as sorted ``(seconds from the first reading, W)`` pairs."""
    points: list[tuple[float, float]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t, w = _offset(item[0]), _offset(item[1])
        if t is None or w is None:
            continue
        points.append((t, max(0.0, w)))
    points.sort(key=lambda p: p[0])
    if not points:
        return ()
    base = points[0][0]
    return tuple((round(t - base, 3), w) for t, w in points)


@dataclass(frozen=True)
class SourceCycle:
    """One stored cycle: its programme and the readings WashData kept for it."""

    program: str
    trace: tuple[tuple[float, float], ...]
    cycle_id: str
    origin: str  # past / reference / backfill

    @property
    def duration(self) -> float:
        return self.trace[-1][0]

    @property
    def peak(self) -> float:
        return max(w for _, w in self.trace)


@dataclass
class Appliance:
    """One WashData config entry, read from an export or a diagnostics dump."""

    path: str
    device_type: str
    entry_data: dict[str, Any]
    entry_options: dict[str, Any]
    cycles: list[SourceCycle]

    @cached_property
    def config(self) -> CycleDetectorConfig:
        """The detector config this entry runs with - the shipped resolution, not a copy."""
        return build_detector_config(self.entry_options, self.entry_data, self.device_type)

    @cached_property
    def cadence(self) -> float:
        """Median spacing of the recorded readings: the source plug's report interval.

        Scenario-made stretches (standby, soak, gaps) are written at this spacing, so
        under the ``recorded`` plug mode they look like the same plug.
        """
        gaps = [
            b[0] - a[0]
            for c in self.cycles
            for a, b in zip(c.trace, c.trace[1:])
            if b[0] > a[0]
        ]
        return min(60.0, max(1.0, statistics.median(gaps))) if gaps else 10.0

    def programs(self) -> list[str]:
        return sorted({c.program for c in self.cycles})

    def pick(self, program: str | None, rng: random.Random) -> SourceCycle:
        """A cycle of ``program``; any cycle for ``None`` or ``"random"``."""
        pool = self.cycles
        if program and program != "random":
            pool = [c for c in self.cycles if c.program == program]
            if not pool:
                raise ValueError(f"no stored cycle of {program!r} in {self.path}")
        return rng.choice(pool)


def load_appliance(path: str | Path) -> Appliance:
    """Read an export, diagnostics dump or legacy dump into playable cycles.

    Raises ``ValueError`` with a message fit for the UI when the file is not one of
    those shapes or holds no playable cycle.
    """
    import json  # noqa: PLC0415

    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"{path.name}: cannot read ({err})") from err
    unwrapped = _corpus_module()._unwrap(doc) if isinstance(doc, dict) else None  # noqa: SLF001
    if unwrapped is None:
        raise ValueError(f"{path.name}: not a WashData export or diagnostics dump")
    data, entry_data, entry_options, _fmt = unwrapped
    device_type = str(
        entry_options.get("device_type") or entry_data.get("device_type") or "washing_machine"
    )
    cycles: list[SourceCycle] = []
    for key, origin in (("past_cycles", "past"), ("reference_cycles", "reference"),
                        ("backfill_cycles", "backfill")):
        for raw in data.get(key) or []:
            if not isinstance(raw, dict) or raw.get("status") in UNPLAYABLE_STATUSES:
                continue
            trace = normalise_trace(raw.get("power_data"))
            if len(trace) < 4 or trace[-1][0] <= 0:
                continue
            cycles.append(SourceCycle(
                program=str(raw.get("profile_name") or UNLABELLED),
                trace=trace,
                cycle_id=str(raw.get("id") or f"{origin}-{len(cycles)}"),
                origin=origin,
            ))
    if not cycles:
        raise ValueError(f"{path.name}: no completed cycle with a power trace")
    return Appliance(str(path), device_type, dict(entry_data), dict(entry_options), cycles)


# --------------------------------------------------------------------------- variation


@dataclass(frozen=True)
class Variation:
    """Run-to-run difference, drawn once per run.

    The defaults are small on purpose: a verbatim replay of a stored cycle matches a
    profile built from that same cycle, which flatters the matcher.
    """

    stretch: float = 0.05  # duration scaled by U(1 - s, 1 + s)
    scale: float = 0.05  # every reading scaled by U(1 - s, 1 + s)
    noise_w: float = 0.0  # Gaussian sigma on readings >= 1 W; 0 W stays 0 W

    def apply(self, cycle: SourceCycle, rng: random.Random) -> SourceCycle:
        k_t = rng.uniform(1 - self.stretch, 1 + self.stretch) if self.stretch > 0 else 1.0
        k_w = rng.uniform(1 - self.scale, 1 + self.scale) if self.scale > 0 else 1.0
        trace = []
        for t, w in cycle.trace:
            w *= k_w
            if self.noise_w > 0 and w >= 1.0:
                w = max(0.0, w + rng.gauss(0.0, self.noise_w))
            trace.append((round(t * k_t, 3), round(w, 1)))
        return replace(cycle, trace=tuple(trace))


# --------------------------------------------------------------------------- programmes


@dataclass(frozen=True)
class TruthCycle:
    """A cycle the appliance really ran, in programme seconds."""

    program: str
    start: float  # first reading at or above start_threshold_w
    end: float  # the moment power last fell below stop_threshold_w


@dataclass(frozen=True)
class Program:
    """Everything the appliance does in one run, in seconds from its start.

    ``knots`` is a step function: each reading holds until the next one, and the last
    knot ends the run (the appliance then sits at the plug's idle draw). ``outages`` are
    stretches where the plug itself is offline while the appliance carries on.
    """

    scenario: str
    knots: tuple[tuple[float, float], ...]
    truth: tuple[TruthCycle, ...]
    sources: tuple[SourceCycle, ...] = ()
    outages: tuple[tuple[float, float], ...] = ()

    @property
    def end(self) -> float:
        return self.knots[-1][0]

    @cached_property
    def _times(self) -> list[float]:
        return [t for t, _ in self.knots]

    def value_at(self, pt: float) -> float:
        i = bisect.bisect_right(self._times, pt + EPS) - 1
        return self.knots[max(0, i)][1]

    def name(self) -> str:
        return " + ".join(c.program for c in self.sources) or "(idle)"


def _active_span(trace: tuple[tuple[float, float], ...], cfg: CycleDetectorConfig) -> tuple[float, float]:
    """(start, end) of real activity: first reading >= start, last fall below stop."""
    start = next((t for t, w in trace if w >= cfg.start_threshold_w), trace[0][0])
    end = trace[-1][0]
    for i in range(len(trace) - 1, -1, -1):
        if trace[i][1] >= cfg.stop_threshold_w:
            end = trace[i + 1][0] if i + 1 < len(trace) else trace[i][0]
            break
    return start, end


def _truth(cycle: SourceCycle, cfg: CycleDetectorConfig, offset: float = 0.0) -> TruthCycle:
    start, end = _active_span(cycle.trace, cfg)
    return TruthCycle(cycle.program, round(offset + start, 3), round(offset + end, 3))


def _hold(t0: float, t1: float, w: float, cadence: float) -> list[tuple[float, float]]:
    """Readings of ``w`` from ``t0`` (inclusive) to ``t1`` (exclusive) every ``cadence``."""
    n = max(1, math.ceil((t1 - t0) / cadence - EPS))
    return [(round(t0 + i * cadence, 3), w) for i in range(n)]


def _shift(trace: tuple[tuple[float, float], ...] | list, dt: float) -> list[tuple[float, float]]:
    return [(round(t + dt, 3), w) for t, w in trace]


def standby_band_w(cfg: CycleDetectorConfig) -> float:
    """A draw between the stop and start thresholds: on, but neither off nor running.

    Where a machine with its display on sits (#445's Miele at 3.4 W against a 2.56 W
    stop), and the band the delayed-start detector watches.
    """
    lo, hi = cfg.stop_threshold_w, cfg.start_threshold_w
    return round((lo + hi) / 2 if hi > lo else lo + 0.5, 1)


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    ref: str  # register item / issue it reproduces
    expect: str  # what a correct WashData does
    cycles: int  # source cycles it plays
    build: Callable[[Appliance, list[SourceCycle], random.Random], Program]


def _clean(app: Appliance, cycles: list[SourceCycle], _rng: random.Random) -> Program:
    (c,) = cycles
    return Program("clean", c.trace, (_truth(c, app.config),), (c,))


def _delay_start(app: Appliance, cycles: list[SourceCycle], _rng: random.Random) -> Program:
    (c,) = cycles
    cfg = app.config
    lead = max(1800.0, 2 * cfg.delay_confirm_seconds)
    knots = _hold(0.0, lead, standby_band_w(cfg), app.cadence) + _shift(c.trace, lead)
    return Program("delay-start", tuple(knots), (_truth(c, cfg, lead),), (c,))


def _soak(app: Appliance, cycles: list[SourceCycle], _rng: random.Random) -> Program:
    """A silent stretch just over min_off_gap, inside the wash."""
    (c,) = cycles
    cfg = app.config
    quiet = cfg.min_off_gap + 120.0
    _start, end = _active_span(c.trace, cfg)
    inside = [i for i, (t, _) in enumerate(c.trace) if 0.3 * end <= t <= 0.7 * end]
    if not inside:
        inside = [len(c.trace) // 2]
    # Extend the longest quiet stretch already there (a soak is where the drum rests);
    # failing that, cut in at the middle.
    quiet_runs: list[tuple[float, int]] = []  # (length, first index)
    k = 0
    while k < len(inside):
        if c.trace[inside[k]][1] >= cfg.stop_threshold_w:
            k += 1
            continue
        last = k
        while last + 1 < len(inside) and c.trace[inside[last + 1]][1] < cfg.stop_threshold_w:
            last += 1
        after = c.trace[min(inside[last] + 1, len(c.trace) - 1)][0]
        quiet_runs.append((after - c.trace[inside[k]][0], inside[k]))
        k = last + 1
    best = max(quiet_runs)[1] if quiet_runs else inside[len(inside) // 2]
    at = c.trace[best][0]
    knots = list(c.trace[:best]) + _hold(at, at + quiet, 0.0, app.cadence) + _shift(c.trace[best:], quiet)
    t0, t1 = _active_span(c.trace, cfg)
    truth = TruthCycle(c.program, t0 if t0 < at else t0 + quiet, t1 + quiet if t1 > at else t1)
    return Program("soak", tuple(knots), (truth,), (c,))


def _dropout(app: Appliance, cycles: list[SourceCycle], _rng: random.Random) -> Program:
    (c,) = cycles
    cfg = app.config
    start, end = _active_span(c.trace, cfg)
    at = start + 0.5 * (end - start)
    length = max(300.0, 2.0 * cfg.off_delay)
    return Program("dropout", c.trace, (_truth(c, cfg),), (c,), outages=((at, at + length),))


def _plug_pull(app: Appliance, cycles: list[SourceCycle], _rng: random.Random) -> Program:
    (c,) = cycles
    cfg = app.config
    _start, end = _active_span(c.trace, cfg)
    cut = round(0.9 * end, 3)
    knots = tuple([(t, w) for t, w in c.trace if t < cut] + [(cut, 0.0)])
    pulled = replace(c, trace=knots)
    return Program("plug-pull", knots, (_truth(pulled, cfg),), (c,))


def _standby_above_stop(app: Appliance, cycles: list[SourceCycle], _rng: random.Random) -> Program:
    (c,) = cycles
    cfg = app.config
    _start, end = _active_span(c.trace, cfg)
    hold = max(3600.0, 3 * STANDBY_BAND_WINDOW_S)
    knots = [(t, w) for t, w in c.trace if t < end]
    knots += _hold(end, end + hold, standby_band_w(cfg), app.cadence) + [(end + hold, 0.0)]
    return Program("standby-above-stop", tuple(knots), (_truth(c, cfg),), (c,))


def _anti_crease(app: Appliance, cycles: list[SourceCycle], rng: random.Random) -> Program:
    """Standby plus a short tumble every ~5 min for 30 min, then the door opens."""
    (c,) = cycles
    cfg = app.config
    _start, end = _active_span(c.trace, cfg)
    base = standby_band_w(cfg)
    tumble = round(min(0.6 * cfg.anti_wrinkle_max_power, max(80.0, 0.08 * c.peak)), 1)
    knots = [(t, w) for t, w in c.trace if t < end]
    t, stop = end, end + 1800.0
    while t < stop:
        quiet = 300.0 + rng.uniform(-60.0, 60.0)
        knots += _hold(t, min(t + quiet, stop), base, app.cadence)
        t += quiet
        if t < stop:
            spin = rng.uniform(20.0, 40.0)
            knots += [(round(t, 3), tumble)]
            t += spin
    knots.append((round(max(t, stop), 3), 0.0))
    return Program("anti-crease", tuple(knots), (_truth(c, cfg),), (c,))


def _back_to_back(app: Appliance, cycles: list[SourceCycle], _rng: random.Random) -> Program:
    a, b = cycles
    cfg = app.config
    _s, a_end = _active_span(a.trace, cfg)
    gap = max(120.0, 1.5 * cfg.min_off_gap)
    knots = [(t, w) for t, w in a.trace if t < a_end]
    knots += _hold(a_end, a_end + gap, 0.0, app.cadence) + _shift(b.trace, a_end + gap)
    truth = (_truth(a, cfg), _truth(b, cfg, a_end + gap))
    return Program("back-to-back", tuple(knots), truth, (a, b))


def _idle_blips(app: Appliance, _cycles: list[SourceCycle], rng: random.Random) -> Program:
    """Short spikes an idle machine makes (display wake, a drain pump), sized below the
    start gates: shorter than start_duration_threshold, under half the start energy."""
    cfg = app.config
    span, knots = 1800.0, []
    length_cap = max(1.0, 0.8 * cfg.start_duration_threshold)
    for at in sorted(rng.uniform(60.0, span - 60.0) for _ in range(6)):
        power = round(min(200.0, max(cfg.start_threshold_w + 5.0, rng.uniform(10.0, 200.0))), 1)
        # Size the length from the chosen power: the power floor above can exceed what
        # a fixed length allows under half the start energy.
        length = rng.uniform(1.0, length_cap)
        if cfg.start_energy_threshold > 0:
            length = min(length, 0.45 * cfg.start_energy_threshold * 3600.0 / power)
        if knots and at <= knots[-1][0]:
            continue
        knots += [(round(at, 3), power), (round(at + length, 3), 0.0)]
    knots = [(0.0, 0.0)] + knots + [(span, 0.0)]
    return Program("idle-blips", tuple(knots), ())


SCENARIOS: dict[str, Scenario] = {s.key: s for s in (
    Scenario("clean", "Clean cycle", "baseline",
             "One cycle: started, matched, ended once, stored once.", 1, _clean),
    Scenario("delay-start", "Delayed start", "item 389, DELAY_WAIT",
             "30 min in the standby band first. With delayed-start detection on the device "
             "shows Delay wait; either way the cycle starts with the programme, not before.",
             1, _delay_start),
    Scenario("soak", "Mid-cycle soak", "item 390",
             "A 0 W pause min_off_gap + 2 min long inside the programme. One stored "
             "cycle; two is the item-390 split (worst with the silent plug mode).", 1, _soak),
    Scenario("dropout", "Plug drops off mid-cycle", "item 266",
             "The plug is unavailable for 2 x off_delay halfway through. The cycle must "
             "survive: ending during the outage is item 266.", 1, _dropout),
    Scenario("plug-pull", "Cut short at 90%", "ML-08",
             "Power cut at 90% of the programme (door opened, plug pulled). Dishwashers "
             "with a committed match finalize fast; others wait their normal off_delay.",
             1, _plug_pull),
    Scenario("standby-above-stop", "Standby above stop", "#445, item 383",
             "After the programme the machine idles between the stop and start "
             "thresholds for an hour. off_delay can never elapse; only the standby-band "
             "finalize (10 min flat past the expected duration) ends it.",
             1, _standby_above_stop),
    Scenario("anti-crease", "Anti-crease tail", "#296",
             "30 min of standby with a short tumble every ~5 min, then the door opens. "
             "One cycle ending with the programme; the tumbles must not start a new one.",
             1, _anti_crease),
    Scenario("back-to-back", "Two loads back to back", "min_off_gap",
             "Two cycles 1.5 x min_off_gap apart. Two stored cycles; one means they merged.",
             2, _back_to_back),
    Scenario("idle-blips", "Idle blips", "start gates",
             "30 min idle with six spikes below the start gates. No cycle may start.",
             0, _idle_blips),
)}


def build_program(
    scenario: str,
    app: Appliance,
    program: str | None,
    variation: Variation,
    rng: random.Random,
) -> Program:
    """Pick the source cycle(s), vary them, and wrap them in the scenario."""
    spec = SCENARIOS[scenario]
    picked = [variation.apply(app.pick(program, rng), rng) for _ in range(spec.cycles)]
    return spec.build(app, picked, rng)


# --------------------------------------------------------------------------- the plug


@dataclass(frozen=True)
class PlugMode:
    """When the plug publishes a reading."""

    key: str
    title: str
    kind: str  # "replay" | "on_change" | "poll"
    why: str
    heartbeat_s: float = 0.0  # re-report after this much silence; 0 = never
    interval_s: float = 0.0  # poll
    delta_w: float = 0.0  # on_change deadband, absolute
    delta_frac: float = 0.0  # on_change deadband, relative to the last report
    min_interval_s: float = 0.0  # on_change rate limit


PLUG_MODES: dict[str, PlugMode] = {m.key: m for m in (
    PlugMode("recorded", "Recorded timing", "replay",
             "Publishes at the trace's own report times: exactly what Home Assistant saw "
             "from the real plug. While idle, 0 W every 60 s.", heartbeat_s=60.0),
    PlugMode("on-change", "On change, 5 min heartbeat", "on_change",
             "Tasmota/Zigbee style: a change past 1 W or 5% is reported (at most every "
             "5 s), and the value is re-sent after 5 min of silence.",
             heartbeat_s=300.0, delta_w=1.0, delta_frac=0.05, min_interval_s=5.0),
    PlugMode("silent", "On change, no heartbeat", "on_change",
             "#424/#427: at a steady draw the plug says nothing at all, so WashData's "
             "watchdog is the only thing driving the detector.",
             heartbeat_s=0.0, delta_w=1.0, delta_frac=0.05, min_interval_s=5.0),
    PlugMode("poll-30s", "Polled every 30 s", "poll",
             "A polled integration: the value is re-sent every 30 s even when unchanged; "
             "WashData takes those as state_reported events (#363).", interval_s=30.0),
)}


@dataclass(frozen=True)
class Event:
    """Something the plug tells Home Assistant (or the ledger), at appliance time ``t``."""

    t: float
    kind: str  # "power" | "draw" | "online" | "offline" | "run_end"
    power: float | None = None  # "power": what was reported; "draw": what the appliance drew
    energy_kwh: float | None = None
    run: Run | None = None


@dataclass
class Run:
    """One programme being played, and the relay-off pauses that stretched it."""

    program: Program
    started: float  # appliance s
    seed: int = 0
    mode: str = ""  # PlugMode key it was reported under
    frozen: list[tuple[float, float]] = field(default_factory=list)  # (programme s, length)
    frozen_since: float | None = None  # appliance s
    status: str = "running"  # running | completed | stopped
    ended: float | None = None

    def _held(self) -> float:
        return sum(length for _, length in self.frozen)

    def program_time(self, t: float) -> float:
        if self.frozen_since is not None:
            t = self.frozen_since
        return t - self.started - self._held()

    def appliance_time(self, pt: float) -> float:
        return self.started + pt + sum(length for at, length in self.frozen if at < pt - EPS)

    def truth(self) -> list[tuple[TruthCycle, float, float]]:
        """``(cycle, start, end)`` in appliance seconds; a stopped run cuts them short."""
        out = []
        stop_pt = self.program_time(self.ended) if self.status == "stopped" else math.inf
        for cyc in self.program.truth:
            if cyc.start >= stop_pt:
                continue
            out.append((cyc, self.appliance_time(cyc.start),
                        self.appliance_time(min(cyc.end, stop_pt))))
        return out


class PlugSim:
    """A plug and the appliance behind it, advanced along the appliance clock.

    Every mutator takes the current appliance time and returns the events it caused;
    ``advance(t)`` processes everything due up to ``t``. Between two processed instants
    the draw is constant, so the energy counter is exact whatever the reporting mode.
    """

    def __init__(self, mode: PlugMode, *, idle_w: float = 0.0, energy_kwh: float = 0.0,
                 resolution_w: float = 0.1) -> None:
        self.mode = mode
        self.idle_w = idle_w
        self.energy_kwh = energy_kwh
        self.resolution_w = resolution_w
        self.run: Run | None = None
        self.relay_on = True
        self.plugged = True
        self.now: float | None = None
        self._value = idle_w
        self._online = False
        self._last: tuple[float, float] | None = None  # last power report (t, W)
        self._pending = False
        self._knot_i = 0
        self._out_i = 0  # next outage boundary (start, end, start, ...)

    # ----------------------------------------------------------------- state

    @property
    def online(self) -> bool:
        return self._online

    @property
    def power(self) -> float:
        return self._value

    @property
    def last_report(self) -> tuple[float, float] | None:
        return self._last

    def _boundaries(self) -> list[float]:
        return [b for span in (self.run.program.outages if self.run else ()) for b in span]

    def _in_outage(self) -> bool:
        return self._out_i % 2 == 1

    def _draw(self, t: float) -> float:
        if not self.relay_on:
            return 0.0
        if self.run is None:
            return self.idle_w
        return self.run.program.value_at(self.run.program_time(t))

    # ----------------------------------------------------------------- clock

    def next_due(self) -> float | None:
        """The next appliance time at which something can change or be reported."""
        due: list[float] = []
        run = self.run
        if run is not None and run.frozen_since is None:
            if self._knot_i < len(run.program.knots):
                due.append(run.appliance_time(run.program.knots[self._knot_i][0]))
            bounds = self._boundaries()
            if self._out_i < len(bounds):
                due.append(run.appliance_time(bounds[self._out_i]))
        if self._online and self._last is not None:
            last_t = self._last[0]
            m = self.mode
            if m.kind == "poll":
                due.append(last_t + m.interval_s)
            elif m.kind == "on_change":
                if self._pending:
                    due.append(last_t + m.min_interval_s)
                if m.heartbeat_s > 0:
                    due.append(last_t + m.heartbeat_s)
            elif m.heartbeat_s > 0 and run is None:
                due.append(last_t + m.heartbeat_s)
        return min(due) if due else None

    def advance(self, t: float) -> list[Event]:
        events: list[Event] = []
        last: float | None = None
        while True:
            due = self.next_due()
            if due is None or due > t + EPS:
                break
            if last is not None and due <= last + EPS:
                # A step resolves everything due by its time, so the same instant coming
                # back means next_due and _report disagree; in the live mock that would
                # spin the event loop forever.
                raise RuntimeError(f"plug scheduler stalled at t={due}")
            last = due if self.now is None else max(due, self.now)
            self._step(last, events)
        self._accrue(t)
        return events

    def _accrue(self, t: float) -> None:
        if self.now is not None and t > self.now:
            self.energy_kwh += self._value * (t - self.now) / 3.6e6
        self.now = t if self.now is None else max(self.now, t)

    def _step(self, t: float, events: list[Event], changed: bool = False) -> None:
        self._accrue(t)
        run = self.run
        if run is not None and run.frozen_since is None:
            knots = run.program.knots
            while self._knot_i < len(knots) and run.appliance_time(knots[self._knot_i][0]) <= t + EPS:
                self._knot_i += 1
                changed = True
            bounds = self._boundaries()
            while self._out_i < len(bounds) and run.appliance_time(bounds[self._out_i]) <= t + EPS:
                self._out_i += 1
            if self._knot_i >= len(knots):
                run.status, run.ended = "completed", t
                self.run = None
                self._out_i = 0
                events.append(Event(t, "run_end", run=run))
                changed = True
        value = self._draw(t)
        if value != self._value:
            events.append(Event(t, "draw", power=value))
        self._value = value
        online = self.plugged and not self._in_outage()
        if online != self._online:
            self._online = online
            events.append(Event(t, "online" if online else "offline"))
            self._last = None  # a reconnecting plug reports its state straight away
            self._pending = False
        if self._online:
            self._report(t, changed, events)

    def _report(self, t: float, changed: bool, events: list[Event]) -> None:
        res = self.resolution_w
        w = round(round(self._value / res) * res, 3)
        m, last = self.mode, self._last
        publish = last is None
        if not publish and m.kind == "replay":
            publish = changed or (self.run is None and m.heartbeat_s > 0
                                  and t >= last[0] + m.heartbeat_s - EPS)
        elif not publish and m.kind == "poll":
            publish = t >= last[0] + m.interval_s - EPS
        elif not publish:
            moved = abs(w - last[1]) >= max(m.delta_w, m.delta_frac * abs(last[1]))
            moved = moved or ((w == 0.0) != (last[1] == 0.0))  # off is always reported
            if moved and t >= last[0] + m.min_interval_s - EPS:
                publish = True
            else:
                self._pending = moved
            if not publish and m.heartbeat_s > 0 and t >= last[0] + m.heartbeat_s - EPS:
                publish = True
        if publish:
            self._pending = False
            self._last = (t, w)
            events.append(Event(t, "power", power=w, energy_kwh=round(self.energy_kwh, 4)))

    # ----------------------------------------------------------------- controls

    def boot(self, t: float) -> list[Event]:
        """The plug powers up and reports."""
        events: list[Event] = [Event(t, "draw", power=self._value)]
        self.now = t
        self._step(t, events, changed=True)
        return events

    def start(self, program: Program, t: float, seed: int = 0) -> list[Event]:
        if self.run is not None:
            raise RuntimeError("a run is already playing")
        events = self.advance(t)
        self.run = Run(program, started=t, seed=seed, mode=self.mode.key)
        self._knot_i = self._out_i = 0
        self._step(t, events, changed=True)
        return events

    def stop(self, t: float) -> list[Event]:
        events = self.advance(t)
        run = self.run
        if run is not None:
            run.status, run.ended = "stopped", t
            self.run = None
            self._out_i = 0
            events.append(Event(t, "run_end", run=run))
            self._step(t, events, changed=True)
        return events

    def set_relay(self, on: bool, t: float) -> list[Event]:
        """Relay off cuts the appliance's power: it reads 0 W and its programme freezes
        until the relay is back on (how WashData's pause-via-switch option drives it)."""
        events = self.advance(t)
        if on == self.relay_on:
            return events
        self.relay_on = on
        run = self.run
        if run is not None and not on and run.frozen_since is None:
            run.frozen_since = t
        elif run is not None and on and run.frozen_since is not None:
            at = run.program_time(t)
            run.frozen.append((at, t - run.frozen_since))
            run.frozen_since = None
        self._step(t, events, changed=True)
        return events

    def set_idle_w(self, watts: float, t: float) -> list[Event]:
        """The draw between runs (a display, a standby board)."""
        events = self.advance(t)
        self.idle_w = max(0.0, float(watts))
        self._step(t, events, changed=True)
        return events

    def set_plugged(self, plugged: bool, t: float) -> list[Event]:
        """Pull the plug off the network (not the appliance off the mains)."""
        events = self.advance(t)
        self.plugged = plugged
        self._step(t, events, changed=True)
        return events


# --------------------------------------------------------------------------- time compression


#: Options `devtools/testbox/smoke.sh` divides for a compressed run.
COMPRESSED_OPTIONS = (
    "off_delay", "min_off_gap", "sampling_interval", "watchdog_interval",
    "profile_match_interval", "start_duration_threshold", "completion_min_seconds",
    "interrupted_min_seconds",
)


def compression_advice(app: Appliance | None, speedup: float) -> list[str]:
    """What has to change on the Home Assistant side before a run at ``speedup`` means
    anything, and what never scales (devtools/testbox/README.md, *Time compression*)."""
    if speedup <= 1.0:
        return []
    lines = [
        f"Time runs {speedup:g}x fast, but every WashData gate counts real seconds. "
        "Without the steps below a run tests nothing.",
    ]
    if app is not None:
        cfg, opts = app.config, app.entry_options
        current = {
            "off_delay": cfg.off_delay,
            "min_off_gap": cfg.min_off_gap,
            "sampling_interval": opts.get("sampling_interval")
            or resolve_sampling_interval_default(app.device_type),
            "watchdog_interval": opts.get("watchdog_interval")
            or resolve_watchdog_interval_default(app.device_type),
            "profile_match_interval": cfg.match_interval or DEFAULT_PROFILE_MATCH_INTERVAL,
            "start_duration_threshold": cfg.start_duration_threshold,
            "completion_min_seconds": cfg.completion_min_seconds,
            "interrupted_min_seconds": cfg.interrupted_min_seconds,
        }
        pairs = " ".join(
            f"{k}={max(1, round(float(current[k]) / speedup))}" for k in COMPRESSED_OPTIONS
        )
        lines.append(f"1. Divide the timing options: {pairs}")
        lines.append(
            "2. Compress the history it matches against: devtools/testbox/hactl.py "
            f"compress-export {Path(app.path).name} <out.json> --speedup {speedup:g}, then import it."
        )
        lines.append(
            "3. The start-energy gate is in Wh: start_energy_threshold="
            f"{cfg.start_energy_threshold / speedup:.4f}"
        )
    lines.append(
        f"Never scales: the dishwasher {DISHWASHER_MIN_CYCLE_DURATION_S / 60:.0f} min floor "
        f"(use <= 4x), the {STANDBY_BAND_WINDOW_S / 60:.0f} min standby-band window (use 2x), "
        "and short gaps between loads (they merge). Durations in the ledger are appliance time."
    )
    return lines
