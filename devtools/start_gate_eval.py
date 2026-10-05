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

**Truth.** ``--truth stored`` (the default when the window holds any): the stored
cycles of ``--config`` (the dump itself by default) that start inside the window -
what the live detector recorded and the user kept. ``--truth blocks``: the history
import's usable activity blocks (``find_activity_blocks`` + ``classify_blocks``)
active for at least ``completion_min_seconds``, for a history with no stored cycles;
their cut rules use the baseline thresholds, never a variant's. Each reference
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
* ``idle_probes``: STARTING -> OFF aborts outside every reference cycle, per idle day
  (invisible unless something keys on the ``starting`` state); ``cycle_probes``
  are the aborts inside one, which is what makes a start late;
* ``fidelity`` (stored truth, baseline only): recorded start minus the stored
  ``start_time``. Near 0 means the replay reproduces what live did; it is the check
  that the harness, not the gate, is not the thing being measured.

Usage::

    python3 devtools/start_gate_eval.py --history cycle_data/<dir>/<diagnostics>.json
    python3 devtools/start_gate_eval.py --history history.csv --config export.json \\
        --sweep start_energy_threshold=0.05,0.2,0.5
    python3 devtools/start_gate_eval.py --history sqlite:/path/home-assistant_v2.db \\
        --entity sensor.washer_power --config export.json --json out.json
"""
from __future__ import annotations

import argparse
import bisect
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
    CONF_PROGRESS_RESET_DELAY,
    CONF_SAMPLING_INTERVAL,
    CONF_WATCHDOG_INTERVAL,
    CURVE_PREROLL_MAX_SECONDS,
    DEFAULT_PROGRESS_RESET_DELAY,
    PREROLL_CHAIN_BREAK_SECONDS,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_FORCE_STOPPED,
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


def readings_from_csv(text: str, entity_id: str | None) -> list[Reading]:
    parsed = history_import.parse_history_csv(text, entity_id=entity_id)
    if isinstance(parsed, dict):
        raise SystemExit(f"CSV: {parsed.get('error')}")
    return [(t.astimezone(timezone.utc), p) for t, p in parsed.samples]


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
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        doc = json.loads(text)
        return readings_from_dump(doc), doc
    return readings_from_csv(text, entity_id), None


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


@dataclass
class Replay:
    runs: list[Run]
    probes: list[tuple[datetime, datetime]]  # (entered STARTING, aborted to OFF)
    readings_in: int
    readings_processed: int


def replay(readings: list[Reading], config: CycleDetectorConfig, manager: dict[str, Any]) -> Replay:
    """Feed a raw history through one fresh detector the way the manager does."""
    now: dict[str, datetime | None] = {"t": None}
    runs: list[Run] = []
    probes: list[tuple[datetime, datetime]] = []
    starting_since: dict[str, datetime | None] = {"t": None}
    completed_at: dict[str, datetime | None] = {"t": None}

    det: CycleDetector

    def _on_state(old: str, new: str) -> None:
        ts = now["t"]
        if ts is None:
            return
        if new == STATE_STARTING:
            starting_since["t"] = ts
        elif old == STATE_STARTING and new == STATE_RUNNING:
            start = det.current_cycle_start or starting_since["t"] or ts
            runs.append(Run(start=start, commit=ts))
        elif old == STATE_STARTING and new == STATE_OFF:
            probes.append((starting_since["t"] or ts, ts))

    def _on_end(cycle: dict[str, Any]) -> None:
        completed_at["t"] = now["t"]
        if runs and runs[-1].end is None:
            runs[-1].end = _utc(cycle.get("end_time")) or now["t"]
            runs[-1].status = cycle.get("status")

    det = CycleDetector(config, _on_state, _on_end, None, device_name="start_gate_eval")

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
    return Replay(runs, probes, len(readings), processed)


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
) -> dict[str, Any]:
    """Replay every variant and score it against one fixed truth."""
    real = [r for r in readings if r[1] is not None]
    if len(real) < 2:
        return {"error": "the history holds fewer than two readings"}
    window = (real[0][0], real[-1][0])
    base_config, _ = detector_setup(device, {})
    level = float(base_config.start_threshold_w)
    refs = references_stored(readings, device.stored, level) if truth in ("auto", "stored") else []
    used_truth = "stored"
    if truth == "blocks" or (truth == "auto" and not refs):
        refs = references_blocks(real, base_config)
        used_truth = "blocks"
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
                "sampling_interval": manager["sampling_interval"],
            },
            **scored,
        }
    return out


def _print(result: dict[str, Any], label: str) -> None:
    if "error" in result:
        print(f"{label}: {result['error']}")
        return
    print(f"\n{label}  [{result['device_type']}]  {result['window'][0]} -> {result['window'][1]}")
    print(f"truth: {result['truth']} (onset level {result['onset_level_w']} W)")
    head = (f"{'variant':<44} {'refs':>4} {'miss':>4} {'merge':>5} {'split':>5} {'late med/p90/max s':>20} "
            f"{'commit med/p90 s':>17} {'phantom (/idle d)':>18} {'probes (/idle d)':>17} {'fid med/max s':>14}")
    print(head)
    for name, v in result["variants"].items():
        s = v["summary"]
        late, lag, fid = s["late_start_s"], s["commit_lag_s"], s["fidelity_abs_s"]
        late_txt = f"{late['median']}/{late['p90']}/{late['max']}"
        lag_txt = f"{lag['median']}/{lag['p90']}"
        ph_txt = f"{s['phantoms']} ({s['phantoms_per_idle_day']})"
        pr_txt = f"{s['idle_probes']} ({s['idle_probes_per_idle_day']})"
        fid_txt = f"{fid['median']}/{fid['max']}"
        print(
            f"{name[:44]:<44} {s['references']:>4} {s['missed']:>4} {s['merged']:>5} "
            f"{s['split']:>5} {late_txt:>20} {lag_txt:>17} {ph_txt:>18} {pr_txt:>17} {fid_txt:>14}"
        )
        if s["fidelity_outliers"] and name == "baseline":
            print(f"{'':<4}fidelity outliers (s, run start - stored start): {s['fidelity_outliers']}")
    first = next(iter(result["variants"].values()))
    print(f"idle days: {first['summary']['idle_days']}, readings in/processed (baseline): "
          f"{first['summary']['readings']['in']}/{first['summary']['readings']['processed']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--history", required=True,
                    help="diagnostics dump (.json), history CSV, or sqlite:<recorder db>")
    ap.add_argument("--entity", help="power sensor entity id (CSV with several, or sqlite:)")
    ap.add_argument("--config", help="export / diagnostics dump for options and stored cycles "
                                     "(default: the --history dump)")
    ap.add_argument("--device-type", help="override or supply the device type")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="option override for one variant (repeatable; shared by --sweep)")
    ap.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2",
                    help="one variant per value")
    ap.add_argument("--truth", choices=("auto", "stored", "blocks"), default="auto")
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args(argv)

    readings, dump = load_history(args.history, args.entity)
    config_doc = dump
    if args.config:
        config_doc = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = load_device(config_doc, args.device_type)
    result = evaluate(readings, device, sets=args.set, sweeps=args.sweep, truth=args.truth)
    label = args.history
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
