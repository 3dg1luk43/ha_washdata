#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the START gates on a raw power history: missed, late and false starts.

Register item 488 (audit MATCH-EVAL-16). Every other replay in ``devtools/`` starts
from a STORED cycle, and a stored cycle begins where the live gates committed: it
carries no idle before it, so no false start can happen in it, and none of the
probes the gates aborted (the #430 curve pre-roll is off by default). A stored
cycle can therefore only ever show a gate made stricter, never one made looser,
and never a phantom. Judging ``start_threshold_w``, ``start_duration_threshold``,
``start_energy_threshold`` or the delayed-start band needs the idle stream too.

**Replay.** One fresh real ``CycleDetector`` (config from ``build_detector_config``,
the builder the manager uses) is fed the whole history, idle included, the way the
manager feeds it:

* the sampling throttle of ``manager._async_power_changed``: a reading within
  ``sampling_interval`` of the last processed one is dropped unless it is a low
  reading during a cycle or a genuine drop from at/above ``min_power``;
* ``unavailable``/``unknown`` rows -> ``mark_sensor_unavailable``;
* the watchdog's low-power keepalives, the same rule as the Playground replay
  (``playground._DetailSim._watchdog_keepalives``, item 390);
* the timer-based Finished -> Off expiry after ``progress_reset_delay``.

Unmatched, like ``history_import`` and ``min_off_gap_eval``: the start path never
consults the matcher, but Smart Termination cannot end a cycle here, so the cycle
ENDS (and with them any merge of back-to-back loads) are the unmatched ones. Not
emulated: the anti-wrinkle keepalives, the opt-in power-based Off, and the
watchdog's STARTING resync (it only feeds a reading the event stream missed).

**Sources** (``--history``):

* a diagnostics dump: ``live_diagnostics.power_trace`` holds every raw reading of
  the last ``window_hours`` (24 h) as the manager received it, re-reports included.
  The dump's own options and stored cycles come with it.
* Home Assistant's history CSV download (``entity_id,state,last_changed``),
  read by ``history_import.parse_history_csv``. Change-only: a re-report of an
  unchanged value is not in it, so a commit can land one change later than live.
* ``sqlite:<path>`` with ``--entity``: the recorder's ``states`` rows, likewise
  change-only, opened ``mode=ro&immutable=1`` (no locks; it never blocks a running
  Home Assistant, and may read a stale page while one writes - use a copy for a
  number you quote).
* the hand-made recording CSV attached to #43 (``minutes_from_start,watts,timestamp_utc``).

Any of them may be gzipped (``*.gz``). ``--history`` repeats to merge several
downloads of one sensor, ``--config`` to union the stored cycles of several dumps of
one entry, and ``--since``/``--until`` clip the window: a History CSV older than the
recorder's retention carries HA's hourly long-term statistics, which are not readings.

**Corpus** (``--manifest``). ``cycle_data/github_issues/start_gate_sources.jsonl``
lists every raw history found (one JSON object per line: ``label``, ``history``,
``config``, ``entity``, ``device_type``, ``truth``, ``since``/``until``, ``extra``,
``exclude``, ``aggregate``, ``note``; paths relative to ``cycle_data/``, a glob names
a ``user-Contributed`` file by entry id). The issue attachments sit under
``cycle_data/github_issues/<issue>/`` and the maintainer's recorder snapshot under
``cycle_data/me/recorder_<date>/``, all gzipped, so no ``*.json``/``*.csv`` corpus
glob of the other harnesses or the slow tests reads them. ``--manifest`` prints each
source, then one table per variant and device type (counts and idle days summed,
late starts pooled per reference). ``--shipped-defaults`` replays on the shipped
start gates (``START_GATE_KEYS``) instead of each user's tuned ones.

**Truth.** ``--truth stored`` (the default when the window holds any): the stored
cycles of ``--config`` (the dump itself by default) that start inside the window -
what the live detector recorded and the user kept. ``--truth blocks``: the history
import's usable activity blocks (``find_activity_blocks`` + ``classify_blocks``)
active for at least ``completion_min_seconds``, for a history with no stored cycles;
their cut rules use the baseline thresholds, never a variant's. ``--truth union``:
stored, plus every block that overlaps none of them (a user who wiped the history
between downloads, #101). A manifest's ``exclude`` drops a stored record that is not
a cycle and ``extra`` adds a hand-labelled one (each says why in its ``note``). Each reference
cycle's ONSET is the earliest reading at or above the
baseline ``start_threshold_w`` chained into its start (consecutive high readings no
more than ``PREROLL_CHAIN_BREAK_SECONDS`` apart, at most ``CURVE_PREROLL_MAX_SECONDS``
back): the #430 pre-roll's own notion of "the same start, probed twice", fixed once
so every variant is measured against the same point.

**Metrics** per variant (baseline = the device's own options; ``--set k=v`` layers
one variant on top, ``--sweep k=v1,v2`` one variant per value):

* ``missed``: reference cycles no committed (STARTING -> RUNNING) run overlaps;
* ``late_start_s``: the run's recorded start minus the onset; ``commit_lag_s``: the
  RUNNING transition minus the onset (how long the state still read off/starting);
* ``merged``: reference cycles whose run already covered an earlier one;
* ``phantoms``: committed runs overlapping no reference cycle, and per idle day;
* ``idle_probes``: the detector's STARTING -> OFF aborts (and STARTING ->
  DELAY_WAIT, item 504) outside every reference cycle, per idle day;
  ``cycle_probes`` are the aborts inside one, which is what makes a start late;
* ``flickers``: what the entities show of those probes (``exposed_state``, which
  ``manager.check_state`` reads): off -> starting -> off outside every reference
  cycle, per idle day. Each is two state rows in the recorder and a trigger for
  any automation keyed on ``starting``. Since item 501 a standby RE-probe (no
  reading below ``stop_threshold_w`` since the last false start) is not shown
  until it fills ``STANDBY_REPROBE_SHOW_ENERGY_FRACTION`` of the energy gate, so
  a standby that straddles the start threshold no longer flickers;
  ``starting_unshown`` counts the judged starts that went off -> running with no
  ``starting`` in between (a hidden re-probe that committed);
* ``wait_drops``: shown delay_wait -> off outside every reference cycle, per idle
  day (a cancel, the timeout, or before item 504 any false start out of
  DELAY_WAIT): the delayed-start band failing to hold;
* ``fidelity`` (stored truth, baseline only): recorded start minus the stored
  ``start_time``. Near 0 means the replay reproduces what live did; it is the check
  that the harness, not the gate, is not the thing being measured.

Usage::

    python3 devtools/start_gate_eval.py --history cycle_data/<dir>/<diagnostics>.json
    python3 devtools/start_gate_eval.py --history history.csv --config export.json \\
        --sweep start_energy_threshold=0.05,0.2,0.5
    python3 devtools/start_gate_eval.py --history sqlite:/path/home-assistant_v2.db \\
        --entity sensor.washer_power --config export.json --json out.json
    python3 devtools/start_gate_eval.py --manifest cycle_data/github_issues/start_gate_sources.jsonl \\
        --shipped-defaults --sweep start_energy_threshold=0.05,0.1,0.2,0.5

**Last run** (2026-10-05, 13 aggregated sources on 10 appliances, 26 judged starts,
32.5 idle days; shipped gates): 0 missed, 0 phantoms; late start median 0 s, p90 45 s,
max 85 s, the four over 30 s all on one dishwasher (#43, aborted first probe);
commit lag median 61 s, p90 128 s. Sweeping start_threshold_w 2-50 W,
start_energy_threshold 0.01-2 Wh, start_duration_threshold 0-60 s,
curve_preroll_seconds 0-600 s and sampling_interval 1-60 s produced no miss and no
phantom anywhere. The threshold only costs (5 W: 6 starts over 30 s late, 10 W: 12);
the duration gate is inert below 30 s (the energy gate binds first); pre-roll >= 120 s
removes every late start; sampling >= 20 s makes dishwasher starts late (median 66 s).
Lowering the energy gate cuts the commit lag (0.1 Wh: median 30 s, p90 40 s) but on
#35, the one standby that straddles the start threshold (not aggregated: 1127
STARTING probes in 2.2 idle days at the shipped gates), 0.1 Wh and below commit a
phantom and 0.15 Wh does not. Every aggregated idle floor is quiet (<= 2 W), so the
phantom side is barely stressed: no shipped default is contradicted.

**Item 501** (2026-10-05, the hidden standby re-probe): #35's 1127 idle probes show as
6 flickers (2.8 per idle day, was 522), 1098 -> 6 with delayed start on; no start
went off -> running unshown. Every other source and every detection metric above is
identical to the run before it (internal probes, rows and aggregates compared field
by field, shipped gates and users' own). Show fraction 0.25 leaves 56 flickers on
#35, 0.75 leaves 4, 1.0 hides ``starting`` on 2 of its real starts.

**Item 504** (2026-10-05, a false start out of DELAY_WAIT returns there): with delayed
start on, #35's wait drops 34 -> 1 (15.75 -> 0.46 per idle day), idle probes
1098 -> 749, flickers 6 -> 6. Every detection metric of every source identical
(field by field, shipped gates and users' own, delayed start on and off); #35 is the
only source that enters DELAY_WAIT. The largest aborted probe on #35 stays at 0.53 of
the energy gate; holding the wait with the old DELAY_WAIT seed (first reading's
power for the whole window) took it to 0.78 and showed 17 flickers.

**Discussion #452** (2026-10-05, the idle display and the hidden terminal-state probe):
``idle_flips`` counts off <-> idle changes of the shown state outside every reference
cycle; ``--standby-level W`` replays as if that standby level had been learned. No
manifest source learns one (every idle floor is ~0 W, and #35's standby has no off
level), so the shipped run shows 0 changes in 32.5 idle days; at a forced 2 W level
still 0 (none is ever seen switched off below it, which a learned level requires);
with ``power_off_threshold_w=0.5`` one, on #35 (into idle, never back). A standby-band
probe out of Finished is hidden like a re-probe: flickers 8 -> 3 (#35 6 -> 3, #214
and the maintainer's washer 1 -> 0). Every detection metric and every per-reference
row of every source identical (shipped gates and users' own, delayed start on and off).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import io
import json
import math
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
from custom_components.ha_washdata import history_import  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
    CONF_CURVE_PREROLL_SECONDS,
    CONF_PROGRESS_RESET_DELAY,
    CONF_SAMPLING_INTERVAL,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_ENERGY_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_WATCHDOG_INTERVAL,
    CURVE_PREROLL_MAX_SECONDS,
    DEFAULT_PROGRESS_RESET_DELAY,
    PREROLL_CHAIN_BREAK_SECONDS,
    STATE_DELAY_WAIT,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_FORCE_STOPPED,
    STATE_IDLE,
    STATE_INTERRUPTED,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    resolve_sampling_interval_default,
    resolve_watchdog_interval_default,
)
from custom_components.ha_washdata.cycle_detector import (  # noqa: E402
    CycleDetector,
    CycleDetectorConfig,
    learned_standby_level_w,
)
from custom_components.ha_washdata.detector_config import (  # noqa: E402
    build_detector_config,
)

Reading = tuple[datetime, "float | None"]

#: Same bound as the Playground's keepalive emulation (8 h at a 30 s watchdog).
MAX_KEEPALIVES_PER_GAP = 960
#: |recorded start - stored start| above which a reference is listed as an outlier.
FIDELITY_OUTLIER_S = 60.0
_TERMINAL = (STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED)
_DAY = 86400.0


# ─── Inputs ───────────────────────────────────────────────────────────────────


def _utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _power(value: Any) -> float | None:
    try:
        power = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, power) if math.isfinite(power) else None


def _corpus() -> Any:
    """``devtools/eval.py``, for its corpus unwrap (export / diagnostics / legacy)."""
    import importlib.util  # noqa: PLC0415

    name = "wd_start_gate_eval_corpus"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "devtools" / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def readings_from_dump(doc: dict[str, Any]) -> list[Reading]:
    """``live_diagnostics.power_trace`` of a diagnostics dump, oldest first."""
    trace = ((doc.get("data") or {}).get("live_diagnostics") or {}).get("power_trace")
    out: list[Reading] = []
    for row in trace or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        ts = _utc(row[0])
        if ts is not None:
            out.append((ts, _power(row[1])))
    out.sort(key=lambda r: r[0])
    return out


#: Header of the hand-made recording CSVs attached to issue #43 (one row per reading,
#: naive UTC stamps); not a Home Assistant format, so ``parse_history_csv`` rejects it.
RECORDING_CSV_COLUMNS = ("watts", "timestamp_utc")


def readings_from_recording_csv(text: str) -> list[Reading]:
    """``minutes_from_start,watts,timestamp_utc`` rows, oldest first."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    out: list[Reading] = []
    for row in reader:
        ts = _utc(row.get("timestamp_utc"))
        if ts is not None:
            out.append((ts, _power(row.get("watts"))))
    out.sort(key=lambda r: r[0])
    return out


def readings_from_csv(text: str, entity_id: str | None) -> list[Reading]:
    header = [h.strip().lower() for h in text.lstrip("﻿").split("\n", 1)[0].split(",")]
    if all(col in header for col in RECORDING_CSV_COLUMNS):
        return readings_from_recording_csv(text)
    parsed = history_import.parse_history_csv(text, entity_id=entity_id)
    if isinstance(parsed, dict):
        raise SystemExit(f"CSV: {parsed.get('error')}")
    return [(t.astimezone(timezone.utc), p) for t, p in parsed.samples]


def read_text(path: Path) -> str:
    """A file's text; ``*.gz`` is decompressed (the issue downloads are stored so)."""
    raw = path.read_bytes()
    if path.suffix.lower() == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8-sig")


def _inner_suffix(path: Path) -> str:
    """``.json`` for ``x.json`` and ``x.json.gz``; ``.txt`` exports count as JSON."""
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    suffix = Path(name).suffix
    return ".json" if suffix == ".txt" else suffix


def clip(readings: list[Reading], since: datetime | None, until: datetime | None) -> list[Reading]:
    """Readings inside ``[since, until]``: a History CSV older than the recorder's
    retention holds HA's hourly long-term statistics, which are not readings."""
    return [
        r for r in readings
        if (since is None or r[0] >= since) and (until is None or r[0] <= until)
    ]


def merge(histories: list[list[Reading]]) -> list[Reading]:
    """Several downloads of one sensor as one history (exact repeats dropped)."""
    seen: set[tuple[datetime, float | None]] = set()
    out: list[Reading] = []
    for row in sorted((r for h in histories for r in h), key=lambda r: r[0]):
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def readings_from_recorder(path: str, entity_id: str) -> list[Reading]:
    """The recorder's state rows for one entity, read without taking a lock."""
    uri = f"file:{Path(path).resolve()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT s.state, s.last_updated_ts FROM states s "
            "JOIN states_meta m ON s.metadata_id = m.metadata_id "
            "WHERE m.entity_id = ? ORDER BY s.last_updated_ts",
            (entity_id,),
        ).fetchall()
    finally:
        con.close()
    return [
        (datetime.fromtimestamp(float(ts), tz=timezone.utc), _power(state))
        for state, ts in rows
        if ts is not None
    ]


def load_history(spec: str, entity_id: str | None) -> tuple[list[Reading], dict | None]:
    """(readings, the document itself when it is a dump that can serve as config)."""
    if spec.startswith("sqlite:"):
        if not entity_id:
            raise SystemExit("--entity is required with a sqlite: history")
        return readings_from_recorder(spec[len("sqlite:"):], entity_id), None
    path = Path(spec)
    text = read_text(path)
    if _inner_suffix(path) == ".json":
        doc = json.loads(text)
        return readings_from_dump(doc), doc
    return readings_from_csv(text, entity_id), None


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


@dataclass
class Device:
    """What one history replays against: options, entry data and stored cycles."""

    device_type: str
    entry_data: dict[str, Any]
    options: dict[str, Any]
    stored: list[dict[str, Any]] = field(default_factory=list)


def load_device(doc: dict[str, Any] | None, device_type: str | None) -> Device:
    if doc is None:
        if not device_type:
            raise SystemExit("no --config and no --device-type: nothing to build a config from")
        return Device(device_type, {}, {}, [])
    unwrapped = _corpus()._unwrap(doc)  # noqa: SLF001
    if unwrapped is None:
        raise SystemExit("--config: not an export, diagnostics dump or legacy dump")
    data, entry_data, options, _fmt = unwrapped
    dtype = device_type or options.get("device_type") or entry_data.get("device_type")
    if not dtype:
        raise SystemExit("the config names no device type; pass --device-type")
    # past_cycles only: a backfill cycle was cut from history by this same detector,
    # so as truth it would only agree with itself.
    stored = [
        c for c in data.get("past_cycles") or [] if isinstance(c, dict) and c.get("start_time")
    ]
    opts = {k: v for k, v in options.items() if v is not None}
    return Device(str(dtype), dict(entry_data), opts, stored)


def load_devices(docs: list[dict[str, Any] | None], device_type: str | None) -> Device:
    """Several dumps of ONE entry: options from the first, stored cycles from all.

    A user who wipes the history between two downloads (#101) leaves each dump with
    only the cycles of its own window; the union, by cycle id, is the truth for both.
    """
    first = load_device(docs[0] if docs else None, device_type)
    seen = {str(c.get("id")) for c in first.stored}
    for doc in docs[1:]:
        for cycle in load_device(doc, first.device_type).stored:
            if str(cycle.get("id")) not in seen:
                seen.add(str(cycle.get("id")))
                first.stored.append(cycle)
    return first


#: The gates ``--shipped-defaults`` resets to what a new entry runs, and the ones
#: the sweep in the module doc moves. ``min_power``/``stop_threshold_w`` stay the
#: user's: they calibrate the plug's idle floor, and the start threshold's default
#: is derived from ``min_power``.
START_GATE_KEYS = (
    CONF_START_THRESHOLD_W,
    CONF_START_ENERGY_THRESHOLD,
    CONF_START_DURATION_THRESHOLD,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_SAMPLING_INTERVAL,
)


def shipped_defaults(device: Device) -> Device:
    """The device with its own start-gate values dropped (see ``START_GATE_KEYS``)."""
    opts = {k: v for k, v in device.options.items() if k not in START_GATE_KEYS}
    data = {k: v for k, v in device.entry_data.items() if k not in START_GATE_KEYS}
    return Device(device.device_type, data, opts, list(device.stored))


#: ``--standby-level``: replay every source as if this standby level had been
#: learned (#452 idle display what-if); None keeps each device's own.
STANDBY_LEVEL_OVERRIDE: float | None = None


def detector_setup(device: Device, overrides: dict[str, Any]) -> tuple[CycleDetectorConfig, dict[str, Any]]:
    """The detector config and the manager-side timings, as the manager builds them."""
    opts = {**device.options, **overrides}
    data = {k: v for k, v in device.entry_data.items() if k not in overrides}
    config = build_detector_config(opts, data, device.device_type)

    def _num(key: str, default: float) -> float:
        try:
            value = float(opts.get(key, default))
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    manager = {
        "sampling_interval": _num(
            CONF_SAMPLING_INTERVAL, resolve_sampling_interval_default(device.device_type)
        ),
        "watchdog_s": max(
            0.0,
            _num(CONF_WATCHDOG_INTERVAL, resolve_watchdog_interval_default(device.device_type)),
        ),
        "progress_reset_delay": _num(CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY),
        # The idle display's standby level, as the manager learns it (#452), or
        # the --standby-level what-if.
        "standby_level_w": (
            STANDBY_LEVEL_OVERRIDE if STANDBY_LEVEL_OVERRIDE is not None
            else learned_standby_level_w(
                device.stored, config.stop_threshold_w, config.start_threshold_w
            )
        ),
    }
    return config, manager


# ─── Replay ───────────────────────────────────────────────────────────────────


@dataclass
class Run:
    """One committed start: STARTING -> RUNNING."""

    start: datetime
    commit: datetime
    end: datetime | None = None
    status: str | None = None
    shown_starting: bool = False  # the entities showed `starting` before `running`


@dataclass
class Replay:
    runs: list[Run]
    probes: list[tuple[datetime, datetime]]  # (entered STARTING, aborted to OFF)
    readings_in: int
    readings_processed: int
    # (shown as starting, back to off) on the entities: ``exposed_state`` (item 501)
    flickers: list[tuple[datetime, datetime]] = field(default_factory=list)
    # (shown as delay_wait, back to off) on the entities (item 504)
    wait_drops: list[tuple[datetime, datetime]] = field(default_factory=list)
    # (moment, shown state) of every off <-> idle change on the entities (#452)
    idle_flips: list[tuple[datetime, str]] = field(default_factory=list)


def replay(readings: list[Reading], config: CycleDetectorConfig, manager: dict[str, Any]) -> Replay:
    """Feed a raw history through one fresh detector the way the manager does."""
    now: dict[str, datetime | None] = {"t": None}
    runs: list[Run] = []
    probes: list[tuple[datetime, datetime]] = []
    flickers: list[tuple[datetime, datetime]] = []
    wait_drops: list[tuple[datetime, datetime]] = []
    idle_flips: list[tuple[datetime, str]] = []
    starting_since: dict[str, datetime | None] = {"t": None}
    # What the entities show (manager.check_state reads ``exposed_state``), sampled
    # where the manager writes them: on every transition and after every reading.
    shown: dict[str, Any] = {"state": STATE_OFF, "since": None, "wait_since": None}
    completed_at: dict[str, datetime | None] = {"t": None}

    det: CycleDetector

    def _expose() -> None:
        ts, state, prev = now["t"], det.exposed_state, shown["state"]
        if ts is None or state == prev:
            return
        if state == STATE_STARTING:
            shown["since"] = ts
        elif state == STATE_DELAY_WAIT and prev != STATE_STARTING:
            shown["wait_since"] = ts
        if prev == STATE_STARTING and state in (STATE_OFF, STATE_IDLE, STATE_DELAY_WAIT):
            # Item 504: a false start out of DELAY_WAIT returns there, not to OFF.
            # #452: off shows as idle on a two-level appliance at its standby level.
            flickers.append((shown["since"] or ts, ts))
        elif prev == STATE_DELAY_WAIT and state in (STATE_OFF, STATE_IDLE):
            wait_drops.append((shown["wait_since"] or ts, ts))
        elif state == STATE_RUNNING and runs and runs[-1].commit == ts:
            runs[-1].shown_starting = prev == STATE_STARTING
        if {prev, state} == {STATE_OFF, STATE_IDLE}:
            idle_flips.append((ts, state))
        shown["state"] = state

    def _on_state(old: str, new: str) -> None:
        ts = now["t"]
        if ts is None:
            return
        if new == STATE_STARTING:
            starting_since["t"] = ts
        elif old == STATE_STARTING and new == STATE_RUNNING:
            start = det.current_cycle_start or starting_since["t"] or ts
            runs.append(Run(start=start, commit=ts))
        elif old == STATE_STARTING and new in (STATE_OFF, STATE_DELAY_WAIT):
            probes.append((starting_since["t"] or ts, ts))
        _expose()

    def _on_end(cycle: dict[str, Any]) -> None:
        completed_at["t"] = now["t"]
        if runs and runs[-1].end is None:
            runs[-1].end = _utc(cycle.get("end_time")) or now["t"]
            runs[-1].status = cycle.get("status")

    det = CycleDetector(config, _on_state, _on_end, None, device_name="start_gate_eval")
    det.set_standby_level(manager.get("standby_level_w"))

    sampling = float(manager["sampling_interval"])
    watchdog = float(manager["watchdog_s"])
    reset_after = float(manager["progress_reset_delay"])
    min_p = float(config.min_power)
    last_processed: datetime | None = None
    current_power = 0.0  # the manager's _current_power: last PROCESSED value
    prev_raw: float | None = None  # the previous raw row (a change event's old_state)
    last_real: tuple[datetime, float] | None = None
    processed = 0

    def _feed(ts: datetime, power: float, *, synthetic: bool = False) -> None:
        now["t"] = ts
        det.process_reading(power, ts, synthetic=synthetic, observed=True)
        _expose()

    for ts, power in readings:
        # Finished -> Off after progress_reset_delay (manager._handle_state_expiry,
        # timer-based branch), stamped at the moment the timer would have fired.
        if (
            completed_at["t"] is not None
            and det.state in _TERMINAL
            and (ts - completed_at["t"]).total_seconds() > reset_after
        ):
            now["t"] = completed_at["t"] + timedelta(seconds=reset_after)
            det.reset(STATE_OFF, now["t"])
            completed_at["t"] = None
        if power is None:
            now["t"] = ts
            det.mark_sensor_unavailable(ts)
            prev_raw = None
            continue
        # Watchdog keepalives inside a silent low-power stretch (item 390), placed
        # half an interval into each period like the Playground's emulation.
        if last_real is not None and watchdog > 0:
            prev_ts, prev_w = last_real
            k = 1
            while k <= MAX_KEEPALIVES_PER_GAP:
                tick = prev_ts + timedelta(seconds=watchdog * (k + 0.5))
                if tick >= ts or not det.is_waiting_low_power():
                    break
                _feed(tick, prev_w, synthetic=True)
                k += 1
        # manager._async_power_changed's throttle. A re-report (same value) carries
        # no old_state there, so it falls back to the last processed power.
        old = prev_raw if prev_raw is not None and prev_raw != power else current_power
        prev_raw = power
        is_low = power < min_p and (
            det.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING) or old >= min_p
        )
        if (
            not is_low
            and last_processed is not None
            and (ts - last_processed).total_seconds() < sampling
        ):
            continue
        last_processed = ts
        current_power = power
        processed += 1
        _feed(ts, power)
        last_real = (ts, power)
    return Replay(runs, probes, len(readings), processed, flickers, wait_drops, idle_flips)


# ─── Truth ────────────────────────────────────────────────────────────────────


@dataclass
class Reference:
    start: datetime  # the stored start_time, or the block's first high reading
    end: datetime
    onset: datetime
    ident: str
    partial: bool = False  # onset too close to the window start to judge: not scored


def onset_before(
    readings: list[Reading],
    start: datetime,
    level_w: float,
    *,
    chain_s: float = PREROLL_CHAIN_BREAK_SECONDS,
    lookback_s: float = CURVE_PREROLL_MAX_SECONDS,
) -> datetime:
    """Earliest reading >= ``level_w`` chained into ``start`` (see the module doc).

    The chain is anchored on the first high reading at or after ``start`` (within
    ``chain_s``): a stored start can sit a few seconds off the trace's own readings
    when it was recorded by older code or stamped at receipt.
    """
    highs = [t for t, p in readings if p is not None and p >= level_w]
    j = bisect.bisect_left(highs, start)
    anchor = start
    if j < len(highs) and (highs[j] - start).total_seconds() <= chain_s:
        anchor = highs[j]
    onset = anchor
    for ts in reversed(highs[:j]):
        if (onset - ts).total_seconds() > chain_s:
            break
        if (anchor - ts).total_seconds() > lookback_s:
            break
        onset = ts
    return onset


def _stored_end(cycle: dict[str, Any], start: datetime) -> datetime:
    end = _utc(cycle.get("end_time"))
    if end is not None:
        return end
    try:
        return start + timedelta(seconds=float(cycle.get("duration") or 0.0))
    except (TypeError, ValueError):
        return start


def references_stored(
    readings: list[Reading], stored: list[dict[str, Any]], level_w: float
) -> list[Reference]:
    if not readings:
        return []
    lo, hi = readings[0][0], readings[-1][0]
    refs: list[Reference] = []
    for cycle in stored:
        start = _utc(cycle.get("start_time"))
        if start is None:
            continue
        end = _stored_end(cycle, start)
        if end < lo or start > hi:
            continue
        onset = onset_before(readings, start, level_w) if start >= lo else start
        refs.append(
            Reference(
                start=start,
                end=end,
                onset=onset,
                ident=str(cycle.get("id") or start.isoformat()),
                # The history must show the quiet before the onset, or the chain
                # may run on past the window and the onset is only a bound.
                partial=(onset - lo).total_seconds() < PREROLL_CHAIN_BREAK_SECONDS,
            )
        )
    refs.sort(key=lambda r: r.start)
    return refs


def references_blocks(readings: list[Reading], config: CycleDetectorConfig) -> list[Reference]:
    blocks, _skipped = history_import.find_activity_blocks(readings, config)
    usable, _rejected = history_import.classify_blocks(blocks, config)
    level = float(config.start_threshold_w)
    refs: list[Reference] = []
    floor = float(config.completion_min_seconds)
    for block in usable:
        highs = [t for t, p in block.samples if p is not None and p >= level]
        # A block is cut on quiet, so an isolated blip arrives wrapped in idle
        # samples; only a block ACTIVE for as long as the detector requires of a
        # completed cycle counts as one.
        if not highs or (highs[-1] - highs[0]).total_seconds() < floor:
            continue
        first = highs[0]
        onset = onset_before(readings, first, level)
        refs.append(
            Reference(
                start=first,
                end=block.end,
                onset=onset,
                ident=f"block@{first.isoformat()}",
                partial=(onset - readings[0][0]).total_seconds() < PREROLL_CHAIN_BREAK_SECONDS,
            )
        )
    return refs


def references_manual(
    readings: list[Reading], windows: list[tuple[datetime, datetime]], level_w: float
) -> list[Reference]:
    """Hand-labelled cycles (a manifest's ``extra``): the first high reading in each
    window is its start, the onset chained back from it like any other reference."""
    refs: list[Reference] = []
    for lo, hi in windows:
        first = next((t for t, p in readings if lo <= t <= hi and p is not None and p >= level_w), None)
        if first is None:
            continue
        onset = onset_before(readings, first, level_w)
        refs.append(Reference(
            start=first, end=hi, onset=onset, ident=f"manual@{first.isoformat()}",
            partial=(onset - readings[0][0]).total_seconds() < PREROLL_CHAIN_BREAK_SECONDS,
        ))
    return refs


def _disjoint(base: list[Reference], more: list[Reference]) -> list[Reference]:
    """``base`` plus each of ``more`` that overlaps none of it."""
    out = list(base)
    for ref in more:
        if not any(_overlaps(ref.onset, ref.end, b.onset, b.end) for b in base):
            out.append(ref)
    return sorted(out, key=lambda r: r.start)


# ─── Scoring ──────────────────────────────────────────────────────────────────


def _overlaps(a0: datetime, a1: datetime | None, b0: datetime, b1: datetime) -> bool:
    return a0 <= b1 and (a1 is None or a1 >= b0)


def score(
    refs: list[Reference],
    result: Replay,
    window: tuple[datetime, datetime],
) -> dict[str, Any]:
    """Per-reference rows and the variant's summary."""
    rows: list[dict[str, Any]] = []
    used: dict[int, str] = {}
    for ref in refs:
        hit = next(
            (
                (i, run)
                for i, run in enumerate(result.runs)
                if _overlaps(run.start, run.end, ref.onset, ref.end)
            ),
            None,
        )
        row: dict[str, Any] = {"ref": ref.ident, "onset": ref.onset.isoformat(),
                               "stored_start": ref.start.isoformat(), "partial": ref.partial}
        if hit is None:
            row["missed"] = True
        else:
            i, run = hit
            row.update(
                missed=False,
                merged_into=used.get(i),
                run_start=run.start.isoformat(),
                late_start_s=round((run.start - ref.onset).total_seconds(), 1),
                commit_lag_s=round((run.commit - ref.onset).total_seconds(), 1),
                vs_stored_s=round((run.start - ref.start).total_seconds(), 1),
                status=run.status,
                shown_starting=run.shown_starting,
            )
            used.setdefault(i, ref.ident)
        rows.append(row)

    def _inside_any(t0: datetime, t1: datetime | None) -> bool:
        return any(_overlaps(t0, t1, r.onset, r.end) for r in refs)

    phantoms = [run for run in result.runs if not _inside_any(run.start, run.end)]
    split = sum(
        1
        for ref in refs
        if not ref.partial
        and sum(1 for run in result.runs if _overlaps(run.start, run.end, ref.onset, ref.end)) > 1
    )
    idle_probes = [p for p in result.probes if not _inside_any(p[0], p[1])]
    cycle_probes = len(result.probes) - len(idle_probes)
    flickers = [f for f in result.flickers if not _inside_any(f[0], f[1])]
    wait_drops = [w for w in result.wait_drops if not _inside_any(w[0], w[1])]
    idle_flips = [f for f in result.idle_flips if not _inside_any(f[0], f[0])]

    lo, hi = window
    busy = 0.0
    for ref in refs:
        a, b = max(ref.onset, lo), min(ref.end, hi)
        busy += max(0.0, (b - a).total_seconds())
    idle_days = max(0.0, (hi - lo).total_seconds() - busy) / _DAY

    judged = [r for r in rows if not r["partial"]]
    hits = [r for r in judged if not r["missed"]]
    late = sorted(r["late_start_s"] for r in hits)
    lag = sorted(r["commit_lag_s"] for r in hits)
    fid = sorted(abs(r["vs_stored_s"]) for r in hits)

    def _q(xs: list[float], q: float) -> float | None:
        if not xs:
            return None
        return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1)

    def _rate(n: int) -> float | None:
        return round(n / idle_days, 2) if idle_days > 0 else None

    summary = {
        "references": len(judged),
        "missed": sum(1 for r in judged if r["missed"]),
        "merged": sum(1 for r in hits if r.get("merged_into")),
        "split": split,
        "late_start_s": {"median": _q(late, 0.5), "p90": _q(late, 0.9), "max": _q(late, 1.0)},
        "commit_lag_s": {"median": _q(lag, 0.5), "p90": _q(lag, 0.9), "max": _q(lag, 1.0)},
        "phantoms": len(phantoms),
        "phantoms_per_idle_day": _rate(len(phantoms)),
        "idle_probes": len(idle_probes),
        "idle_probes_per_idle_day": _rate(len(idle_probes)),
        "cycle_probes": cycle_probes,
        "flickers": len(flickers),
        "flickers_per_idle_day": _rate(len(flickers)),
        "wait_drops": len(wait_drops),
        "wait_drops_per_idle_day": _rate(len(wait_drops)),
        # #452: off <-> idle changes of the shown state outside every reference
        # cycle (each is a recorder row), and the learned level that drives them.
        "idle_flips": len(idle_flips),
        "idle_flips_per_idle_day": _rate(len(idle_flips)),
        "idle_shown": sum(1 for _t, st in idle_flips if st == STATE_IDLE),
        "starting_unshown": sum(
            1 for r in hits if not r.get("merged_into") and not r["shown_starting"]
        ),
        "idle_days": round(idle_days, 3),
        "fidelity_abs_s": {"median": _q(fid, 0.5), "max": _q(fid, 1.0)},
        # A replay that disagrees with live by more than a minute is not measuring the
        # gate: a restart that lost the cycle's head, older code, a deleted record.
        "fidelity_outliers": {
            r["ref"]: r["vs_stored_s"] for r in hits if abs(r["vs_stored_s"]) > FIDELITY_OUTLIER_S
        },
        "readings": {"in": result.readings_in, "processed": result.readings_processed},
        "phantom_starts": [p.start.isoformat() for p in phantoms],
    }
    return {"summary": summary, "rows": rows}


# ─── Driver ───────────────────────────────────────────────────────────────────


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def variants(sets: list[str], sweeps: list[str]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = [("baseline", {})]
    common: dict[str, Any] = {}
    for item in sets:
        key, _, raw = item.partition("=")
        common[key.strip()] = _parse_value(raw)
    if common and not sweeps:
        out.append((",".join(f"{k}={v}" for k, v in common.items()), dict(common)))
    for item in sweeps:
        key, _, raws = item.partition("=")
        for raw in raws.split(","):
            value = _parse_value(raw)
            out.append(
                (",".join([*(f"{k}={v}" for k, v in common.items()), f"{key}={value}"]),
                 {**common, key.strip(): value})
            )
    return out


def evaluate(
    readings: list[Reading],
    device: Device,
    *,
    sets: list[str] | None = None,
    sweeps: list[str] | None = None,
    truth: str = "auto",
    extra: list[tuple[datetime, datetime]] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Replay every variant and score it against one fixed truth.

    ``truth``: ``stored``, ``blocks``, ``auto`` (stored when the window holds any,
    else blocks) or ``union`` (stored, plus every block that overlaps none of them:
    for a user who wiped the history between runs, #101). ``exclude`` drops stored
    ids or block idents that are not cycles (a force-stopped standby record);
    ``extra`` windows are hand-labelled cycles and replace any reference they overlap.
    """
    real = [r for r in readings if r[1] is not None]
    if len(real) < 2:
        return {"error": "the history holds fewer than two readings"}
    window = (real[0][0], real[-1][0])
    base_config, _ = detector_setup(device, {})
    level = float(base_config.start_threshold_w)
    dropped = set(exclude or ())
    stored: list[Reference] = []
    if truth in ("auto", "stored", "union"):
        stored = [r for r in references_stored(readings, device.stored, level) if r.ident not in dropped]
    refs, used_truth = stored, "stored"
    if truth == "blocks" or (truth == "auto" and not stored) or truth == "union":
        blocks = [r for r in references_blocks(real, base_config) if r.ident not in dropped]
        refs, used_truth = (_disjoint(stored, blocks), "union") if truth == "union" else (blocks, "blocks")
    if extra:
        refs = _disjoint(references_manual(readings, extra, level), refs)
        used_truth += "+manual"
    out: dict[str, Any] = {
        "device_type": device.device_type,
        "window": [window[0].isoformat(), window[1].isoformat()],
        "truth": used_truth,
        "onset_level_w": level,
        "variants": {},
    }
    for name, overrides in variants(sets or [], sweeps or []):
        config, manager = detector_setup(device, overrides)
        scored = score(refs, replay(readings, config, manager), window)
        out["variants"][name] = {
            "overrides": overrides,
            "gates": {
                "start_threshold_w": config.start_threshold_w,
                "start_duration_threshold": config.start_duration_threshold,
                "start_energy_threshold": config.start_energy_threshold,
                "curve_preroll_seconds": config.curve_preroll_seconds,
                "sampling_interval": manager["sampling_interval"],
                "standby_level_w": manager.get("standby_level_w"),
            },
            **scored,
        }
    return out


# ─── Manifest: many histories, one table per device type ──────────────────────


@dataclass
class Source:
    """One manifest line: a raw history, what to replay it against, and its truth."""

    label: str
    history: list[str]
    config: list[str] = field(default_factory=list)
    entity: str | None = None
    device_type: str | None = None
    truth: str = "auto"
    since: str | None = None
    until: str | None = None
    extra: list[list[str]] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    aggregate: bool = True
    note: str = ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def read_manifest(path: Path) -> list[Source]:
    """JSON Lines, one source per line; blank lines and ``#`` lines are skipped."""
    out: list[Source] = []
    for n, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n}: {exc}") from exc
        out.append(Source(
            label=str(raw.get("label") or f"line{n}"),
            history=[str(h) for h in _as_list(raw.get("history"))],
            config=[str(c) for c in _as_list(raw.get("config"))],
            entity=raw.get("entity"),
            device_type=raw.get("device_type"),
            truth=str(raw.get("truth") or "auto"),
            since=raw.get("since"),
            until=raw.get("until"),
            extra=[list(w) for w in _as_list(raw.get("extra"))],
            exclude=[str(i) for i in _as_list(raw.get("exclude"))],
            aggregate=bool(raw.get("aggregate", True)),
            note=str(raw.get("note") or ""),
        ))
    return out


def _resolve(spec: str, root: Path) -> str:
    """A manifest path: absolute, ``sqlite:<path>``, or relative to ``cycle_data/``."""
    if spec.startswith("sqlite:"):
        return "sqlite:" + _resolve(spec[len("sqlite:"):], root)
    if any(ch in spec for ch in "*?["):
        # A glob, so a manifest can name a file whose name carries a person's name
        # (user-Contributed/) by its entry id alone.
        hits = sorted(root.glob(spec))
        if len(hits) != 1:
            raise SystemExit(f"{spec}: {len(hits)} matches under {root}, need exactly one")
        return str(hits[0])
    path = Path(spec)
    return str(path if path.is_absolute() else root / path)


def evaluate_source(
    src: Source,
    *,
    root: Path,
    sets: list[str] | None = None,
    sweeps: list[str] | None = None,
    shipped: bool = False,
) -> dict[str, Any]:
    """Load one manifest source and :func:`evaluate` it."""
    histories: list[list[Reading]] = []
    dump: dict | None = None
    for spec in src.history:
        readings, doc = load_history(_resolve(spec, root), src.entity)
        histories.append(readings)
        dump = dump or doc
    readings = clip(merge(histories), _utc(src.since) if src.since else None,
                    _utc(src.until) if src.until else None)
    docs: list[dict | None] = [load_config(Path(_resolve(c, root))) for c in src.config] or [dump]
    device = load_devices(docs, src.device_type)
    if shipped:
        device = shipped_defaults(device)
    extra = [(a, b) for a, b in ((_utc(w[0]), _utc(w[1])) for w in src.extra if len(w) == 2)
             if a is not None and b is not None]
    result = evaluate(readings, device, sets=sets, sweeps=sweeps, truth=src.truth,
                      extra=extra, exclude=src.exclude)
    result["label"] = src.label
    result["aggregate"] = src.aggregate
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{variant: {device_type | "ALL": summary}}`` over the aggregated sources.

    Counts and idle days add up; late starts and commit lags are pooled per judged
    reference, so a source with many cycles weighs as many cycles, not as one.
    """
    acc: dict[str, dict[str, dict[str, Any]]] = {}
    keys = ("references", "missed", "merged", "split", "phantoms", "idle_probes", "cycle_probes",
            "flickers", "wait_drops", "starting_unshown", "idle_flips")
    for res in results:
        if "error" in res or not res.get("aggregate", True):
            continue
        for name, var in res["variants"].items():
            for group in (res["device_type"], "ALL"):
                slot = acc.setdefault(name, {}).setdefault(group, {
                    **{k: 0 for k in keys}, "idle_days": 0.0, "sources": 0, "late": [], "lag": []})
                slot["sources"] += 1
                for k in keys:
                    slot[k] += var["summary"][k]
                slot["idle_days"] += var["summary"]["idle_days"]
                for row in var["rows"]:
                    if not row["partial"] and not row["missed"] and not row.get("merged_into"):
                        slot["late"].append(row["late_start_s"])
                        slot["lag"].append(row["commit_lag_s"])

    def _q(xs: list[float], q: float) -> float | None:
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1) if xs else None

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, groups in acc.items():
        for group, s in groups.items():
            days = s["idle_days"]
            out.setdefault(name, {})[group] = {
                **{k: s[k] for k in keys}, "sources": s["sources"], "idle_days": round(days, 2),
                "phantoms_per_idle_day": round(s["phantoms"] / days, 3) if days > 0 else None,
                "idle_probes_per_idle_day": round(s["idle_probes"] / days, 2) if days > 0 else None,
                "flickers_per_idle_day": round(s["flickers"] / days, 2) if days > 0 else None,
                "wait_drops_per_idle_day": round(s["wait_drops"] / days, 2) if days > 0 else None,
                "idle_flips_per_idle_day": round(s["idle_flips"] / days, 2) if days > 0 else None,
                "late_start_s": {"median": _q(s["late"], 0.5), "p90": _q(s["late"], 0.9),
                                 "max": _q(s["late"], 1.0)},
                "late_over_30s": sum(1 for x in s["late"] if x > 30.0),
                "commit_lag_s": {"median": _q(s["lag"], 0.5), "p90": _q(s["lag"], 0.9)},
            }
    return out


def _print_aggregate(agg: dict[str, dict[str, dict[str, Any]]]) -> None:
    print(f"\n{'variant':<40} {'group':<16} {'src':>3} {'refs':>4} {'miss':>4} {'merge':>5} "
          f"{'split':>5} {'late med/p90/max s':>20} {'>30s':>4} {'commit med/p90':>15} "
          f"{'phantom (/idle d)':>18} {'probes':>6} {'flick':>5} {'waits':>5} {'unshown':>7} "
          f"{'idle d':>7} {'idle<>off':>9}")
    for name, groups in agg.items():
        for group in sorted(groups, key=lambda g: (g == "ALL", g)):
            s = groups[group]
            late, lag = s["late_start_s"], s["commit_lag_s"]
            print(f"{name[:40]:<40} {group:<16} {s['sources']:>3} {s['references']:>4} "
                  f"{s['missed']:>4} {s['merged']:>5} {s['split']:>5} "
                  f"{str(late['median']) + '/' + str(late['p90']) + '/' + str(late['max']):>20} "
                  f"{s['late_over_30s']:>4} {str(lag['median']) + '/' + str(lag['p90']):>15} "
                  f"{str(s['phantoms']) + ' (' + str(s['phantoms_per_idle_day']) + ')':>18} "
                  f"{s['idle_probes']:>6} {s['flickers']:>5} {s['wait_drops']:>5} "
                  f"{s['starting_unshown']:>7} "
                  f"{s['idle_days']:>7} {s['idle_flips']:>9}")


def _print(result: dict[str, Any], label: str) -> None:
    if "error" in result:
        print(f"{label}: {result['error']}")
        return
    print(f"\n{label}  [{result['device_type']}]  {result['window'][0]} -> {result['window'][1]}")
    print(f"truth: {result['truth']} (onset level {result['onset_level_w']} W)")
    head = (f"{'variant':<44} {'refs':>4} {'miss':>4} {'merge':>5} {'split':>5} {'late med/p90/max s':>20} "
            f"{'commit med/p90 s':>17} {'phantom (/idle d)':>18} {'probes (/idle d)':>17} "
            f"{'flickers (/idle d)':>18} {'waits (/idle d)':>16} {'unshown':>7} {'fid med/max s':>14}")
    print(head)
    for name, v in result["variants"].items():
        s = v["summary"]
        late, lag, fid = s["late_start_s"], s["commit_lag_s"], s["fidelity_abs_s"]
        late_txt = f"{late['median']}/{late['p90']}/{late['max']}"
        lag_txt = f"{lag['median']}/{lag['p90']}"
        ph_txt = f"{s['phantoms']} ({s['phantoms_per_idle_day']})"
        pr_txt = f"{s['idle_probes']} ({s['idle_probes_per_idle_day']})"
        fl_txt = f"{s['flickers']} ({s['flickers_per_idle_day']})"
        wt_txt = f"{s['wait_drops']} ({s['wait_drops_per_idle_day']})"
        fid_txt = f"{fid['median']}/{fid['max']}"
        print(
            f"{name[:44]:<44} {s['references']:>4} {s['missed']:>4} {s['merged']:>5} "
            f"{s['split']:>5} {late_txt:>20} {lag_txt:>17} {ph_txt:>18} {pr_txt:>17} "
            f"{fl_txt:>18} {wt_txt:>16} {s['starting_unshown']:>7} {fid_txt:>14}"
        )
        if s["fidelity_outliers"] and name == "baseline":
            print(f"{'':<4}fidelity outliers (s, run start - stored start): {s['fidelity_outliers']}")
    first = next(iter(result["variants"].values()))
    print(f"idle days: {first['summary']['idle_days']}, readings in/processed (baseline): "
          f"{first['summary']['readings']['in']}/{first['summary']['readings']['processed']}")
    print(f"idle display (#452): standby level {first['gates'].get('standby_level_w')} W, "
          f"off<->idle changes {first['summary']['idle_flips']} "
          f"({first['summary']['idle_flips_per_idle_day']} per idle day)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--history", action="append", default=[],
                    help="diagnostics dump (.json), history CSV, or sqlite:<recorder db>; "
                         "*.gz is read too; repeat to merge several downloads of one sensor")
    ap.add_argument("--manifest", help="JSON Lines of sources (see the module doc); replaces "
                                       "--history and adds a per-device-type table")
    ap.add_argument("--source", action="append", default=[],
                    help="with --manifest: only sources whose label contains this")
    ap.add_argument("--root", default=str(REPO / "cycle_data"),
                    help="with --manifest: what its relative paths are relative to")
    ap.add_argument("--entity", help="power sensor entity id (CSV with several, or sqlite:)")
    ap.add_argument("--config", action="append", default=[],
                    help="export / diagnostics dump for options and stored cycles (default: "
                         "the --history dump); repeat for several dumps of one entry")
    ap.add_argument("--device-type", help="override or supply the device type")
    ap.add_argument("--since", help="drop readings before this ISO time")
    ap.add_argument("--until", help="drop readings after this ISO time")
    ap.add_argument("--shipped-defaults", action="store_true",
                    help="baseline on the shipped start gates, not the device's own "
                         f"({', '.join(START_GATE_KEYS)})")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="option override for one variant (repeatable; shared by --sweep)")
    ap.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2",
                    help="one variant per value")
    ap.add_argument("--truth", choices=("auto", "stored", "blocks", "union"), default="auto")
    ap.add_argument("--exclude", action="append", default=[],
                    help="stored cycle id or block ident that is not a cycle")
    ap.add_argument("--standby-level", type=float, default=None, metavar="W",
                    help="#452 what-if: replay as if this standby level had been learned "
                         "(the idle display; detection never reads it)")
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args(argv)
    if not args.history and not args.manifest:
        ap.error("--history or --manifest is required")
    global STANDBY_LEVEL_OVERRIDE  # noqa: PLW0603
    STANDBY_LEVEL_OVERRIDE = args.standby_level

    root = Path(args.root)
    if args.manifest:
        sources = read_manifest(Path(args.manifest))
        if args.source:
            sources = [s for s in sources if any(sub in s.label for sub in args.source)]
        results = []
        for src in sources:
            result = evaluate_source(src, root=root, sets=args.set, sweeps=args.sweep,
                                     shipped=args.shipped_defaults)
            _print(result, src.label + ("" if src.aggregate else "  (not aggregated)"))
            results.append(result)
        agg = aggregate(results)
        _print_aggregate(agg)
        if args.json:
            Path(args.json).write_text(json.dumps({"sources": results, "aggregate": agg}, indent=2),
                                       encoding="utf-8")
            print(f"wrote {args.json}")
        return 0 if all("error" not in r for r in results) else 1

    src = Source(label=args.history[0], history=args.history, config=args.config,
                 entity=args.entity, device_type=args.device_type, truth=args.truth,
                 since=args.since, until=args.until, exclude=args.exclude)
    result = evaluate_source(src, root=Path.cwd(), sets=args.set, sweeps=args.sweep,
                             shipped=args.shipped_defaults)
    label = args.history[0]
    if label.startswith(str(REPO / "cycle_data")) or label.startswith("cycle_data/"):
        rel = label.split("cycle_data/", 1)[1]
        label = "cycle_data/" + _corpus().public_key(rel)
    _print(result, label)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
