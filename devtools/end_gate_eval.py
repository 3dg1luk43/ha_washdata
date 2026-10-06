#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure what the fallback end gate costs and saves, on real traces.

Register item 306 shortened the ENDING fallback timeout from
``max(off_delay, min_off_gap)`` to ``max(off_delay, min(min_off_gap, 300))``
once a matched cycle passes ``1.05 x`` its own expected duration, and reported
median end lag 10.00 -> 7.73 min over 427 real cycles with early ends and splits
unchanged.  **That harness was never checked in**, so when round 6 of the PR #448
review added the missing ``match_confidence_threshold`` guard to the same gate
there was no way to re-cut the number.  This is that harness, checked in.

What it measures, per replayed cycle:

* **end lag** - the replay offset at which the detector left ENDING, minus the
  trace's own active span (first to last above-``stop_threshold_w`` reading).
  That is the wall-clock time between the appliance finishing and WashData
  saying so.  Deliberately NOT ``final_duration_s``: since item 297 the stored
  duration is trimmed back to the last activity, so it is ~0 lag by
  construction and measures nothing about the gate.
* **early end** - the detector closed the cycle *before* the appliance stopped,
  counted at the >1 min and >5 min marks.  This is the axis that must never move
  in the wrong direction: an early end truncates the cycle and corrupts the
  profile it is recorded against.
* **split** - the trace did not survive as one cycle covering its active span.

Replay is the real thing, not a model of it: every cycle goes through
``playground.simulate_cycle_detail``, which drives the real ``CycleDetector``
and the real Stage 1-5 matcher over the cycle's own trace.  So
``_last_match_confidence`` is whatever the shipped matcher actually produces -
which is the whole point, since the guard under test reads exactly that.
Since audit F7 each match is also applied with the manager's own post-match
rules (``match_rules``: the envelope verified pause and its releases, the
confident-mismatch revoke, the switching that names the program), and every
candidate template is re-gridded to the query's step as live does. Figures
taken before that never saw a verified pause, so they under-report the end lag
of devices that engage one (the #427 AEG washer: 1.5-15.5 min per cycle).

**Configuration (audit F2 / DETECT-12).**  Each export is replayed with the
configuration its own options produce in production: the detector config and the
ProfileStore come from a real ``WashDataManager`` built on the export's entry
data and options (the same ``build_detector_config`` the manager uses), and every
envelope is rebuilt with the current code (exports carry stale ones). The old
hand-rolled config defaulted ``min_off_gap`` to 480 s for every device and read a
key no option is stored under, which understated washer end lag 16.2 -> 12.2 min.

``--loo`` matches each cycle against profiles rebuilt WITHOUT it (leave-one-out),
like ``devtools/eval.py``. Without it each cycle is matched against an envelope it
helped build, which flatters confidence and ambiguity - use ``--loo`` for any
figure you quote.

**Running the A/B.**  The two arms are two states of the code. Do not ``git
stash`` in a shared working tree; check the other arm out in a worktree:

    git worktree add --detach /tmp/before <ref> && ln -s "$PWD/cycle_data" /tmp/before/
    (cd /tmp/before && python3 devtools/end_gate_eval.py --loo --json /tmp/before.json)
    python3 devtools/end_gate_eval.py --loo --json /tmp/after.json
    python3 devtools/end_gate_eval.py --compare /tmp/before.json /tmp/after.json

``--no-shortening`` patches ``const.END_GATE_LATE_RATIO`` (and the per-device
map) out of reach to give the
pre-306 arm, which needs no checkout.

**Corpus and watchdog (register item 465).** By default only the export format
(``device_fingerprint`` + ``data.past_cycles``) is read, which skips every
diagnostics dump: none of ``cycle_data/user-Contributed/`` is replayed.
``--all-formats`` reads the corpus the way ``devtools/eval.py`` does (all three
export shapes, clone files dropped; private paths keyed by hash). The replay emits
the live watchdog's keepalives inside silent stretches at each export's own
``watchdog_interval``; ``--shipped-watchdog`` drops that option so every device
runs at its type's shipped default instead. That is not cosmetic: a keepalive is
the only reading inside a silence, so the interval decides whether an end gate is
evaluated between the expected end and a late pump-out at all (01KGM619: 599 s in
the export, 30 s the dishwasher default, 61 s what Apply-all suggests for it).

**Anti-crease (register item 207).** ``--anti-wrinkle force`` turns anti-wrinkle
on for every washer, dryer and washer-dryer export (``export``, the default, keeps
each export's own setting). ``--tumble-tail`` replaces the synthetic 0 W tail of
those devices with the #296 shape: the trace's trailing quiet is trimmed and the
reporter's Knitterschutz tail follows (a ~3 W baseline with a sub-400 W drum burst
every 37 s), so the ordinary end gates cannot close the cycle and the anti-crease
finalise is the only closer. Every row records whether the #399 spin guard held
that finalise, whether it released on the spin (``event``) or at the
``ANTI_CREASE_SPIN_WAIT_MAX_RATIO`` cap, and whether it fired before the trace's
own last reading above ``anti_wrinkle_max_power`` (an early release: the spin then
opens a second cycle). ``--device-types`` restricts the corpus, ``--export SUBSTR``
to the exports whose path contains it.

**Synthetic halt (discussion #452).** ``--halt-at F --halt-min M`` inserts a flat
standby plateau into every replayed cycle: at fraction ``F`` of its active span
(``F >= 1``: right after its last above-stop reading, the display left on after
the programme ended) for ``M`` minutes (``Mx``: M times the active span), every
later reading shifted by that much. The level alternates +-0.4 W around
``--halt-level`` (default ``auto``: 4.5 W clamped into ``[stop + 0.5, near-stop
ceiling - 0.5]``, so it is the #452 shape on every device). Rows record whether
and when the stall display flagged (``stall_*``), whether it flagged outside the
plateau (a false flag; on an unmodified run every flag is one), when the
standby-band finalize fired, and whether the cycle closed inside the plateau.
``--no-stall-guard`` patches ``STALL_HOLDS_STANDBY_BAND`` off for the before arm.
For a mid-cycle halt the yardstick span includes the plateau; for ``F >= 1`` it
does not (the programme had ended). ``auto`` can sit at or above a device's start
threshold, so with ``F >= 1`` the plateau may also open phantom cycles there: read
the split column of such a run as the harness's, and compare arms row by row.

Measured 2026-10-05, ``--loo --all-formats`` over the 258 washer, washer-dryer and
dryer cycles: unmodified, no stall is ever flagged (0 of 472 rows, every device
type) and the standby-band finalize never fires. A 45 min halt at 50% of the
active span shows as stalled in 196 (76%), median 10.6 min in; the hold keeps 28 of
the 106 the finalize used to close inside the halt (splits 117 -> 89). At 90% (the
final spin): 118 (46%), 39 of 180 kept. 20 min at 30%: 109 (42%). With the display
left on for 45 min after a real end, 46 (18%) show as stalled for a while (the
match still says the programme owes work: it ends early on it) and the hold delays
6 (2.3%) of the 204 closes the finalize makes inside that time. Early ends unchanged
in every run.

**User pause (register item 514).** ``--user-pause`` (with ``--halt-at``) makes the
plateau a user pause instead of a halt: the detector is told what the manager's
``async_pause_cycle`` tells it (user-paused, verified pause) at the plateau's first
reading and what ``async_resume_cycle`` tells it at the first reading after it.
``--halt-level 0.4`` is a pause that cuts the plug's power (0.0 / 0.8 W). Every row
also records the cycle-end label verdict (``label_profile`` / ``label_reason``, the
Playground's ``would_label``) for the label check of a halted cycle. Measured
2026-10-06 over the 258 cycles, 45 min pauses: before item 514 the gates read the
raw clock after a resume, and a power-cutting pause at 50% / 30% / 90% split 48 /
50 / 40 cycles (early ends > 1 min 6 / 5 / 18); a resumed pause now leaves the gate
clock: 12 / 22 / 33 split, early ends 5 / 7 / 8, each new one either the unpaused
cycle's own Smart Termination end (one washer; unpaused it ends from RUNNING, which
``end_offset_s`` does not see) or a former split. Paused on the display level (shown
as stalled, so item 511 already banked most of it): 9 -> 6 split at 50%, 14 -> 9 at
90%, early ends 1 -> 0.

**Option overrides (register item 469).** ``--set KEY=VALUE`` layers an option onto
every replayed export as Apply all would save it (the active-span yardstick keeps
the export's own stop threshold), e.g. ``--set profile_match_interval=45`` for the
shorter match interval Apply all suggests.

**Committed baseline (audit TESTING-09).** ``devtools/end_gate_baseline.json``
holds the per-device-type end-gate figures (n, median / mean / p90 lag, early
ends > 1 / > 5 min, splits) of one ``--loo --all-formats`` run, with the replay
flags and the tree state (HEAD, dirty files, a hash of the integration code) it
was taken on. Regenerate it after an intended end-gate change::

    python3 devtools/end_gate_eval.py --loo --all-formats \\
        --json /tmp/rows.json --write-baseline devtools/end_gate_baseline.json

and gate a change against it::

    python3 devtools/end_gate_eval.py --check devtools/end_gate_baseline.json
    python3 devtools/end_gate_eval.py --check devtools/end_gate_baseline.json --rows /tmp/rows.json

``--check`` replays with the baseline's own flags (or summarises ``--rows``, the
``--json`` output of such a run, without replaying) and exits **1** when any
device type regresses beyond ``BASELINE_TOLERANCE``: median lag + 0.25 min, mean
lag + 0.5 min, p90 lag + 1.0 min, and **no** new early end (> 1 or > 5 min) or
split. Early ends and splits are counts with zero tolerance because they are the
axes that must never move the wrong way; lag is a cost and gets a small band. It
exits **2** when the corpus no longer matches (a device type's cycle count
differs: ``cycle_data/`` is maintainer-local, so re-baseline rather than compare),
and 0 otherwise. The replay is deterministic, so an unchanged tree reproduces the
baseline exactly.

**Parallel replay.** ``--jobs N`` (default ``cpu_count - 1``, at most 8) replays
in N worker processes, one cycle per unit, longest export first and its longest
cycles first (the previous run's per-cycle times in ``--timings``, else trace
length), handed out one at a time as workers free up; a worker keeps its last two
exports' setup. ``--jobs-file F`` re-reads the worker count from F after every
cycle, so a scheduler (``devtools/verify.sh``) can add cores mid-run. Rows are put
back in the serial order, so the ``--json`` rows, the summary and ``--check`` are
identical to ``--jobs 1``, the serial loop. Progress goes to stderr. Measured
2026-10-05 on the 472-row ``--loo --all-formats`` corpus: 278 s serial, 82 s at
``--jobs 4`` (4 cores), rows and ``--check`` byte-identical.

Run from the repo root.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if TYPE_CHECKING:
    from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
    from custom_components.ha_washdata.profile_store import ProfileStore

# The integration is imported by `_integration()` (from `main` and `_production`),
# not here: importing any of it loads Home Assistant (~3 s), and `--help` must not.
playground: Any = None
CONF_ANTI_WRINKLE_ENABLED = "anti_wrinkle_enabled"
CONF_WATCHDOG_INTERVAL = "watchdog_interval"
resolve_watchdog_interval_default: Any = None
_cycle_readings: Any = None


def _integration() -> None:
    """Bind the integration names this module uses (idempotent)."""
    global playground, CONF_ANTI_WRINKLE_ENABLED, CONF_WATCHDOG_INTERVAL  # noqa: PLW0603
    global resolve_watchdog_interval_default, _cycle_readings  # noqa: PLW0603
    if playground is not None:
        return
    from custom_components.ha_washdata import const  # noqa: PLC0415
    from custom_components.ha_washdata import playground as _pg  # noqa: PLC0415
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        _cycle_readings as _readings,
    )

    CONF_ANTI_WRINKLE_ENABLED = const.CONF_ANTI_WRINKLE_ENABLED
    CONF_WATCHDOG_INTERVAL = const.CONF_WATCHDOG_INTERVAL
    resolve_watchdog_interval_default = const.resolve_watchdog_interval_default
    _cycle_readings = _readings
    playground = _pg

#: A cycle must carry at least this many readings to be worth replaying.
MIN_READINGS = 10
#: Exports with fewer stored cycles than this cannot build usable profiles.
MIN_CYCLES = 5
#: Device types the anti-crease finalise can arm for (`_anticrease_gate_open`).
AC_DEVICE_TYPES = ("washing_machine", "dryer", "washer_dryer")

#: Anti-crease probe state for the replay in progress (one at a time).
_AC: dict[str, Any] = {}


def _install_anticrease_probe() -> None:
    """Record what the #399 spin guard and the anti-crease finalise did.

    Wraps three detector methods; the replay itself is unchanged. A hold is the
    guard returning True on the finalise path (``_is_anticrease_tail``); the
    standby-band path calls the same predicate and is recorded separately.
    """
    from custom_components.ha_washdata import cycle_detector as cd  # noqa: PLC0415
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        ANTI_CREASE_SPIN_WAIT_MAX_RATIO,
        ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC,
    )

    det_cls = cd.CycleDetector
    if getattr(det_cls, "_eval_probe", False):
        return
    orig_tail = det_cls._is_anticrease_tail  # noqa: SLF001
    orig_pending = det_cls._anticrease_spin_pending  # noqa: SLF001
    orig_final = det_cls._maybe_finalize_anticrease_tail  # noqa: SLF001

    def is_tail(self: Any, ts: Any) -> bool:
        self._eval_ac_ctx = True
        try:
            return orig_tail(self, ts)
        finally:
            self._eval_ac_ctx = False

    def pending(self: Any, ts: Any) -> bool:
        out = orig_pending(self, ts)
        if "ac_final_ts" in _AC:
            return out
        if not getattr(self, "_eval_ac_ctx", False):
            _AC["sb_held"] = _AC.get("sb_held", False) or bool(out)
            return out
        if out:
            _AC["ac_held"] = True
            return out
        block = self._matched_terminal_high  # noqa: SLF001
        start = self._current_cycle_start  # noqa: SLF001
        expected = self._expected_duration  # noqa: SLF001
        elapsed = (ts - start).total_seconds() if start is not None else 0.0
        if block is None or block[0] < ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC:
            _AC["ac_reason"] = "unarmed"
        elif expected > 0 and elapsed >= expected * ANTI_CREASE_SPIN_WAIT_MAX_RATIO:
            _AC["ac_reason"] = "cap"
        else:
            _AC["ac_reason"] = "event"
        return out

    def finalize(self: Any, ts: Any) -> bool:
        fired = orig_final(self, ts)
        if fired:
            _AC.setdefault("ac_finals", []).append(ts)
        if fired and "ac_final_ts" not in _AC:
            _AC["ac_final_ts"] = ts
            if _AC.get("ac_held"):
                _AC["ac_release"] = _AC.get("ac_reason")
        return fired

    det_cls._is_anticrease_tail = is_tail  # noqa: SLF001
    det_cls._anticrease_spin_pending = pending  # noqa: SLF001
    det_cls._maybe_finalize_anticrease_tail = finalize  # noqa: SLF001
    det_cls._eval_probe = True


def _install_stall_probe() -> None:
    """Record the stall display's flips and the standby-band finalize (#452)."""
    from custom_components.ha_washdata import cycle_detector as cd  # noqa: PLC0415

    det_cls = cd.CycleDetector
    if getattr(det_cls, "_eval_stall_probe", False):
        return
    orig_set = det_cls._set_stalled  # noqa: SLF001
    orig_sb = det_cls._maybe_finalize_standby_band  # noqa: SLF001

    def set_stalled(self: Any, stalled: bool, ts: Any) -> None:
        before = self._stall_active  # noqa: SLF001
        orig_set(self, stalled, ts)
        if self._stall_active != before:  # noqa: SLF001
            _AC.setdefault("stall", []).append((bool(stalled), ts))
            if stalled and "stall_why" not in _AC:
                # The evidence the first flag stood on: the run's position on the
                # match it began under, and whether that match ends on a block.
                name, expected, block, _cat = self._stall_match or (None, 0.0, None, None)  # noqa: SLF001
                start, run = self._current_cycle_start, self._stall_run_start  # noqa: SLF001
                _AC["stall_why"] = {
                    "position": round((run - start).total_seconds() / expected, 3)
                    if name and expected > 0 and start and run else None,
                    "armed": bool(block is not None and block[0] >= 0.9),
                    "eval": self._stall_eval,  # noqa: SLF001
                }

    def finalize_sb(self: Any, ts: Any, power: float) -> bool:
        start = self._current_cycle_start  # noqa: SLF001
        expected = float(self._expected_duration)  # noqa: SLF001
        why = {
            "ratio": round((ts - start).total_seconds() / expected, 3)
            if start is not None and expected > 0 else None,
            "stalled": bool(self._stall_active),  # noqa: SLF001
            "eval": self._stall_eval,  # noqa: SLF001
            "run_s": round((ts - self._stall_run_start).total_seconds())  # noqa: SLF001
            if self._stall_run_start is not None else None,  # noqa: SLF001
        }
        fired = orig_sb(self, ts, power)
        if fired:
            _AC.setdefault("sb_finals", []).append(ts)
            _AC.setdefault("sb_why", []).append(why)
        return fired

    det_cls._set_stalled = set_stalled  # noqa: SLF001
    det_cls._maybe_finalize_standby_band = finalize_sb  # noqa: SLF001
    det_cls._eval_stall_probe = True


def _install_user_pause_probe() -> None:
    """``--user-pause``: pause and resume the detector around the plateau (item 514).

    Inert unless the replay in progress set ``_AC["user_pause"]`` to its window.
    """
    from custom_components.ha_washdata import cycle_detector as cd  # noqa: PLC0415
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        STATE_ENDING, STATE_PAUSED, STATE_RUNNING, STATE_STARTING,
    )

    det_cls = cd.CycleDetector
    if getattr(det_cls, "_eval_user_pause_probe", False):
        return
    orig = det_cls.process_reading
    open_states = (STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING)

    def process_reading(self: Any, power: float, timestamp: Any, *a: Any, **k: Any) -> Any:
        window = _AC.get("user_pause")
        if window is not None:
            paused = bool(self._user_paused)  # noqa: SLF001
            if not paused and window[0] <= timestamp < window[1] and self.state in open_states:
                self.set_user_paused(True, timestamp)  # async_pause_cycle
                self.set_verified_pause(True)
                _AC["user_paused_at"] = timestamp
            elif paused and timestamp >= window[1]:
                self.set_user_paused(False, timestamp)  # async_resume_cycle
                self.set_verified_pause(False)
        return orig(self, power, timestamp, *a, **k)

    det_cls.process_reading = process_reading
    det_cls._eval_user_pause_probe = True  # noqa: SLF001


def _halt_level(stop: float, level: str) -> float:
    from custom_components.ha_washdata.cycle_detector import (  # noqa: PLC0415
        standby_near_stop_ceiling,
    )

    if level != "auto":
        return float(level)
    return min(max(4.5, stop + 0.5), standby_near_stop_ceiling(stop) - 0.5)


def _with_halt(
    cycle: dict[str, Any], pts: list[tuple[float, float]], stop: float,
    at: float, length: str, level: float,
) -> tuple[dict[str, Any], float, float, float]:
    """The cycle with a flat standby plateau inserted (#452).

    Returns ``(cycle, halt_start_s, halt_end_s, yardstick_span_s)``.
    """
    active = [t for t, p in pts if p > stop]
    span = active[-1] - active[0]
    length_s = (
        float(length[:-1]) * span if length.endswith("x") else float(length) * 60.0
    )
    gaps = [b - a for (a, _p), (b, _q) in zip(pts, pts[1:]) if b > a]
    step = min(60.0, max(10.0, float(np.median(gaps)) if gaps else 30.0))
    if at >= 1.0:
        t0 = active[-1] + step
        head = [[float(t), float(p)] for t, p in pts if t <= active[-1]]
        tail = [[t0 + length_s, 0.0], [t0 + length_s + 600.0, 0.0]]
        yard = span
    else:
        t0 = active[0] + at * span
        head = [[float(t), float(p)] for t, p in pts if t < t0]
        tail = [[float(t) + length_s, float(p)] for t, p in pts if t >= t0]
        yard = span + length_s
    plateau = []
    t, k = t0, 0
    while t < t0 + length_s:
        plateau.append([t, round(level + (0.4 if k % 2 else -0.4), 2)])
        t += step
        k += 1
    out = dict(cycle)
    out["power_data"] = head + plateau + tail
    return out, t0, t0 + length_s, yard


def _with_tumble_tail(
    cycle: dict[str, Any], pts: list[tuple[float, float]], stop: float, level: float
) -> dict[str, Any]:
    """The cycle with its trailing quiet replaced by the #296 tumble tail.

    The tail is the reporter's own (tron4r export, the merged back-to-back
    cycle): a ~3.3 W baseline and a ~60 W drum burst every ~37 s, reported on
    change. It runs for an hour or 0.6x the trace, whichever is longer, which
    reaches past the 1.25x spin-wait cap of the trace's own programme.
    """
    peak = max(p for _t, p in pts)
    floor = max(1.0, peak * 0.02)
    end = len(pts) - 1
    while end > 0 and pts[end][1] <= floor:
        end -= 1
    out = [[float(t), float(p)] for t, p in pts[: end + 1]]
    burst = min(max(60.0, 2.0 * stop), 0.5 * level)
    base_w = 3.3
    t0 = out[-1][0]
    span = max(3600.0, 0.6 * t0)
    t = t0 + 30.0
    while t < t0 + span:
        out += [[t, burst], [t + 2.0, burst * 0.75], [t + 4.0, base_w], [t + 20.0, base_w]]
        t += 37.0
    tailed = dict(cycle)
    tailed["power_data"] = out
    return tailed


class _Entry:
    def __init__(self, data: dict, options: dict, title: str) -> None:
        self.data, self.options, self.title = data, options, title
        self.entry_id, self.domain = "end-gate-eval", "ha_washdata"

    def async_on_unload(self, *_a: Any, **_k: Any) -> None:
        return None

    def add_update_listener(self, *_a: Any, **_k: Any) -> Any:
        return lambda: None


class _InlineHass:
    """Executor jobs run inline: deterministic, and nothing else is reachable."""

    async def async_add_executor_job(self, fn: Any, *args: Any) -> Any:
        return fn(*args)


class _NullStore:
    """Replaces the WashDataStore: every save is dropped."""

    async def async_save(self, _data: Any) -> None:
        return None

    async def async_load(self) -> None:
        return None


def _run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _production(
    doc: dict[str, Any],
    data: dict[str, Any],
    *,
    shipped_watchdog: bool = False,
    force_anti_wrinkle: bool = False,
    overrides: dict[str, Any] | None = None,
) -> tuple[CycleDetectorConfig, ProfileStore, dict[str, Any]]:
    """(detector config, ProfileStore, options) exactly as the manager builds them.

    ``shipped_watchdog`` drops the export's ``watchdog_interval`` (entry data and
    options) so the Playground resolves the device type's shipped default.
    ``force_anti_wrinkle`` sets ``anti_wrinkle_enabled`` before the manager reads it.
    ``overrides`` (``--set``) are layered onto the options last, as Apply all would
    save them, and dropped from the entry data so the option wins.
    """
    _integration()
    from custom_components.ha_washdata.manager import WashDataManager  # noqa: PLC0415

    entry_data = {"power_sensor": "sensor.end_gate_eval", "name": "eval",
                  **{k: v for k, v in (doc.get("entry_data") or {}).items() if v is not None}}
    opts = {k: v for k, v in (doc.get("entry_options") or {}).items() if v is not None}
    if shipped_watchdog:
        entry_data.pop(CONF_WATCHDOG_INTERVAL, None)
        opts.pop(CONF_WATCHDOG_INTERVAL, None)
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    if device_type:
        opts.setdefault("device_type", device_type)
    if force_anti_wrinkle:
        entry_data.pop(CONF_ANTI_WRINKLE_ENABLED, None)
        opts[CONF_ANTI_WRINKLE_ENABLED] = True
    for key, value in (overrides or {}).items():
        entry_data.pop(key, None)
        opts[key] = value
    mgr = WashDataManager(MagicMock(), _Entry(entry_data, opts, "eval"))
    store = mgr.profile_store
    store.hass = _InlineHass()
    store._store = _NullStore()  # noqa: SLF001
    store._data = data  # noqa: SLF001
    return mgr.detector.config, store, {**entry_data, **opts}


def _rebuild_envelopes(store: ProfileStore, names: Any) -> None:
    async def _go() -> None:
        for name in names:
            await store.async_rebuild_envelope(name)

    _run(_go())


def _fold_data(base: dict[str, Any], cycle: dict[str, Any]) -> dict[str, Any]:
    """The store without ``cycle`` (by identity), sharing what it does not touch."""
    d = dict(base)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        d[key] = [c for c in (base.get(key) or []) if c is not cycle]
    d["profiles"] = {k: dict(v) if isinstance(v, dict) else v
                     for k, v in (base.get("profiles") or {}).items()}
    d["envelopes"] = dict(base.get("envelopes") or {})
    return d


def _end_offset(events: list[dict[str, Any]]) -> float | None:
    """Replay offset of the transition OUT of ENDING - when the cycle closed.

    This is the number the user experiences as "the wash is done" arriving late,
    and the only one the fallback end gate moves.
    """
    for ev in reversed(events):
        if ev.get("type") != "state":
            continue
        detail = str(ev.get("detail") or "")
        if detail.startswith("ending->"):
            return float(ev.get("t") or 0.0)
    return None


def _active_span(points: list[tuple[float, float]], stop: float) -> float:
    """Seconds from the first to the last above-stop reading."""
    active = [t for t, p in points if p > stop]
    return (active[-1] - active[0]) if len(active) >= 2 else 0.0


_EVAL_MOD: Any = None


def _corpus_module() -> Any:
    """``devtools/eval.py``, for its corpus loader (all export shapes, clones dropped)."""
    global _EVAL_MOD  # noqa: PLW0603
    if _EVAL_MOD is None:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location(
            "wd_end_gate_eval_corpus", Path(__file__).resolve().parent / "eval.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclasses resolve their module through it
        spec.loader.exec_module(mod)
        _EVAL_MOD = mod
    return _EVAL_MOD


def _load_doc(path: Path, all_formats: bool) -> dict[str, Any] | None:
    """The export, or with ``all_formats`` any corpus shape normalised to it."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not all_formats or not isinstance(doc, dict):
        return doc if isinstance(doc, dict) else None
    unwrapped = _corpus_module()._unwrap(doc)  # noqa: SLF001
    if unwrapped is None:
        return None
    data, entry_data, entry_options, _fmt = unwrapped
    device_type = entry_options.get("device_type") or entry_data.get("device_type")
    return {
        "device_fingerprint": {"device_type": device_type},
        "entry_data": entry_data,
        "entry_options": entry_options,
        "data": data,
    }


def _export_key(path: Path) -> str:
    rel = str(path.relative_to(REPO))
    if rel.startswith("cycle_data/"):
        # Contributors' real names are in some file names (eval.py PRIVATE_DIRS).
        return "cycle_data/" + _corpus_module().public_key(rel[len("cycle_data/"):])
    return rel


def _measure_export(
    path: Path, no_shortening: bool, loo: bool = False, **kw: Any,
) -> list[dict[str, Any]]:
    """Replay every usable cycle in one export; one row per cycle."""
    return [row for _ci, row in _replay_export(path, no_shortening, loo, **kw)]


def _export_doc(
    path: Path, all_formats: bool, device_types: tuple[str, ...] | None,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]] | None:
    """``(doc, device type, past cycles)``, or None for an export that is not replayed."""
    doc = _load_doc(path, all_formats)
    if doc is None:
        return None
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    data = doc.get("data") or {}
    cycles = data.get("past_cycles") or []
    if not device_type or len(cycles) < MIN_CYCLES:
        return None
    if device_types and device_type not in device_types:
        return None
    return doc, device_type, cycles


def _export_setup(
    path: Path,
    *,
    all_formats: bool = False,
    shipped_watchdog: bool = False,
    anti_wrinkle: str = "export",
    device_types: tuple[str, ...] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Everything the per-cycle loop shares: the production store, rebuilt envelopes."""
    found = _export_doc(path, all_formats, device_types)
    if found is None:
        return None
    doc, device_type, _cycles = found
    data = doc.get("data") or {}
    ac_device = device_type in AC_DEVICE_TYPES
    force_aw = anti_wrinkle == "force" and ac_device
    base = dict(data)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        base[key] = list(base.get(key) or [])
    base["profiles"] = {k: dict(v) if isinstance(v, dict) else v
                        for k, v in (base.get("profiles") or {}).items()}
    base["envelopes"] = dict(base.get("envelopes") or {})
    cfg, store, opts = _production(
        doc, base, shipped_watchdog=shipped_watchdog, force_anti_wrinkle=force_aw,
        overrides=overrides,
    )
    # The yardstick (active span) stays at the export's own stop threshold, so an
    # overridden one cannot move what it is measured against.
    stop = float(
        _production(doc, base, shipped_watchdog=shipped_watchdog)[0].stop_threshold_w
        if overrides else cfg.stop_threshold_w
    )
    level = float(cfg.anti_wrinkle_max_power)
    # Setup runs the sample repair before any match (e.g. it drops a profile's
    # pointer at another programme's run), and mutates base["profiles"] in place,
    # so the LOO fold stores inherit it.
    _run(store.async_repair_profile_samples())
    # Exports carry the envelopes the exporting version built; rebuild them with
    # the code under test, as the live store would after an upgrade.
    _rebuild_envelopes(store, list(base["profiles"]))
    try:
        prebuilt = playground._build_match_snapshots(store)  # noqa: SLF001
    except Exception:
        prebuilt = None
    return {
        "doc": doc, "device_type": device_type, "ac_device": ac_device,
        "force_aw": force_aw, "base": base, "cfg": cfg, "store": store,
        "opts": opts, "stop": stop, "level": level, "prebuilt": prebuilt,
    }


#: Per worker process: the setups of the exports it replayed last (``--jobs``).
_SETUP_MEMO: dict[str, dict[str, Any] | None] = {}
SETUP_MEMO_SIZE = 2


def _replay_export(
    path: Path,
    no_shortening: bool,
    loo: bool = False,
    *,
    all_formats: bool = False,
    shipped_watchdog: bool = False,
    anti_wrinkle: str = "export",
    tumble_tail: bool = False,
    device_types: tuple[str, ...] | None = None,
    overrides: dict[str, Any] | None = None,
    halt: tuple[float, str, str] | None = None,
    user_pause: bool = False,
    only: frozenset[int] | None = None,
    memo: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    """``(cycle index, row)`` per replayed cycle of one export.

    ``only`` replays just the cycles at those indexes of ``past_cycles`` (one
    ``--jobs`` unit). The setup does not depend on it, and with ``memo`` a worker
    keeps the setups of its last ``SETUP_MEMO_SIZE`` exports for the next unit.
    """
    setup_kw = {
        "all_formats": all_formats, "shipped_watchdog": shipped_watchdog,
        "anti_wrinkle": anti_wrinkle, "device_types": device_types,
        "overrides": overrides,
    }
    if memo:
        key = json.dumps([str(path), setup_kw], sort_keys=True, default=str)
        if key not in _SETUP_MEMO:
            while len(_SETUP_MEMO) >= SETUP_MEMO_SIZE:
                _SETUP_MEMO.pop(next(iter(_SETUP_MEMO)))
            _SETUP_MEMO[key] = _export_setup(path, **setup_kw)
        ctx = _SETUP_MEMO[key]
    else:
        ctx = _export_setup(path, **setup_kw)
    if ctx is None:
        return []
    device_type, ac_device, force_aw = ctx["device_type"], ctx["ac_device"], ctx["force_aw"]
    doc, base, cfg, store, opts = ctx["doc"], ctx["base"], ctx["cfg"], ctx["store"], ctx["opts"]
    stop, level, prebuilt = ctx["stop"], ctx["level"], ctx["prebuilt"]
    cycles = base["past_cycles"]
    rows: list[tuple[int, dict[str, Any]]] = []
    for ci, cyc in enumerate(cycles):
        if only is not None and ci not in only:
            continue
        pts = _cycle_readings(cyc)
        if len(pts) < MIN_READINGS:
            continue
        span = _active_span(pts, stop)
        if span <= 0:
            continue
        fold_store, fold_prebuilt = store, prebuilt
        name = cyc.get("profile_name")
        if loo and name and name in base["profiles"]:
            _cfg_f, fold_store, _o = _production(
                doc, _fold_data(base, cyc), shipped_watchdog=shipped_watchdog,
                force_anti_wrinkle=force_aw, overrides=overrides,
            )
            _rebuild_envelopes(fold_store, [name])
            try:
                fold_prebuilt = playground._build_match_snapshots(fold_store)  # noqa: SLF001
            except Exception:
                fold_prebuilt = None
        replayed = (
            _with_tumble_tail(cyc, pts, stop, level) if tumble_tail and ac_device else cyc
        )
        halt_s: tuple[float, float] | None = None
        if halt is not None:
            replayed, h0, h1, span = _with_halt(
                replayed, _cycle_readings(replayed), stop, halt[0], halt[1],
                _halt_level(stop, halt[2]),
            )
            halt_s = (h0, h1)
        _AC.clear()
        if user_pause and halt_s is not None:
            _b = playground._cycle_base_time(replayed)  # noqa: SLF001
            _AC["user_pause"] = (_b + timedelta(seconds=halt_s[0]), _b + timedelta(seconds=halt_s[1]))
        try:
            sim = playground.simulate_cycle_detail(
                replayed, cfg, None, fold_store, opts, price=None,
                compute_series=False, prebuilt=fold_prebuilt,
            )
        except Exception:
            continue
        if "error" in sim:
            continue
        out = sim.get("outcome") or {}
        final = out.get("final_duration_s")
        events = sim.get("events") or []
        end_t = _end_offset(events)
        first_end = next(
            (float(ev.get("t") or 0.0) for ev in events if ev.get("type") == "finished"),
            None,
        )
        highs = [t for t, p in pts if p > level]
        active = [t for t, p in pts if p > stop]
        ac_final = _AC.get("ac_final_ts")
        base_t = playground._cycle_base_time(replayed)  # noqa: SLF001
        ac_final_s = (ac_final - base_t).total_seconds() if ac_final is not None else None
        # Every anti-crease finalise, not only the first: a run that already split
        # can release early again on a later piece.
        ac_early_n = sum(
            1 for ts in _AC.get("ac_finals", ())
            if highs and (ts - base_t).total_seconds() < highs[-1]
        )
        # #452 stall display: flips as (on, offset) and the standby-band finalizes.
        flips = [(on, (ts - base_t).total_seconds()) for on, ts in _AC.get("stall", ())]
        ons = [t for on, t in flips if on]
        sb_finals = [(ts - base_t).total_seconds() for ts in _AC.get("sb_finals", ())]
        finishes = [
            float(ev.get("t") or 0.0) for ev in events if ev.get("type") == "finished"
        ]
        in_halt = (lambda t: halt_s is not None and halt_s[0] <= t <= halt_s[1] + 60.0)
        stall_fields = {
            "halt_start_s": round(halt_s[0], 1) if halt_s else None,
            "halt_end_s": round(halt_s[1], 1) if halt_s else None,
            "stall_on_s": round(ons[0], 1) if ons else None,
            "stall_n": len(ons),
            "stall_in_halt": any(in_halt(t) for t in ons),
            "stall_outside_halt": any(not in_halt(t) for t in ons),
            "stall_latency_s": (
                round(min(t for t in ons if in_halt(t)) - halt_s[0], 1)
                if halt_s and any(in_halt(t) for t in ons) else None
            ),
            "sb_final_s": round(sb_finals[0], 1) if sb_finals else None,
            "sb_why": (_AC.get("sb_why") or [None])[0],
            "stall_why": _AC.get("stall_why"),
            "closed_in_halt": any(
                halt_s is not None and halt_s[0] <= t < halt_s[1] for t in finishes
            ),
        }
        rows.append((ci, {
            "export": _export_key(path),
            "device_type": device_type,
            "id": str(cyc.get("id"))[:12],
            "label": cyc.get("profile_name"),
            "active_span_s": round(span, 1),
            "detected": bool(out.get("detected")),
            "detected_count": int(out.get("detected_count") or 0),
            "final_duration_s": round(float(final), 1) if final else None,
            "end_offset_s": round(end_t, 1) if end_t is not None else None,
            "matched_profile": out.get("matched_profile"),
            "confidence": out.get("confidence"),
            "termination_reason": out.get("termination_reason"),
            "off_delay": cfg.off_delay,
            "min_off_gap": cfg.min_off_gap,
            "no_shortening": no_shortening,
            "loo": loo,
            "watchdog_s": opts.get(
                CONF_WATCHDOG_INTERVAL, resolve_watchdog_interval_default(device_type)
            ),
            # Anti-crease (register item 207). Offsets are seconds into the trace.
            "anti_wrinkle": bool(cfg.anti_wrinkle_enabled),
            "tumble_tail": bool(tumble_tail and ac_device),
            "first_end_s": round(first_end, 1) if first_end is not None else None,
            "active_end_s": round(active[-1], 1) if active else None,
            "last_high_s": round(highs[-1], 1) if highs else None,
            "ac_final_s": round(ac_final_s, 1) if ac_final_s is not None else None,
            "ac_held": bool(_AC.get("ac_held")),
            "ac_release": _AC.get("ac_release"),
            "ac_early": bool(
                ac_final_s is not None and highs and ac_final_s < highs[-1]
            ),
            "ac_early_n": ac_early_n,
            "sb_held": bool(_AC.get("sb_held")),
            "user_pause": bool(user_pause and halt_s is not None),
            # The cycle-end label verdict of the longest detected piece (item 514).
            "label_profile": out.get("label_profile"),
            "label_reason": out.get("label_reason"),
            **stall_fields,
        }))
    return rows


def _summarise(rows: list[dict[str, Any]], device_type: str | None = None) -> dict[str, Any]:
    """Lag / early-end / split figures over a set of replayed cycles."""
    sel = [r for r in rows if device_type is None or r["device_type"] == device_type]
    finished = [
        r for r in sel
        if r["detected"] and r.get("end_offset_s") is not None
    ]
    if not finished:
        return {"n": 0}
    lags = [r["end_offset_s"] - r["active_span_s"] for r in finished]
    early = [lag for lag in lags if lag < 0]
    splits = sum(
        1 for r in finished
        if r["detected_count"] > 1 or r["final_duration_s"] < 0.9 * r["active_span_s"]
    )
    matched = [r for r in finished if r["matched_profile"]]
    weak = [
        r for r in matched
        if r["confidence"] is not None and float(r["confidence"]) < 0.4
    ]
    return {
        "n": len(finished),
        "median_lag_min": round(float(np.median(lags)) / 60.0, 2),
        "mean_lag_min": round(float(np.mean(lags)) / 60.0, 2),
        "p90_lag_min": round(float(np.percentile(lags, 90)) / 60.0, 2),
        "early_1min_pct": round(100.0 * sum(1 for x in early if x < -60) / len(finished), 2),
        "early_5min_pct": round(100.0 * sum(1 for x in early if x < -300) / len(finished), 2),
        "split_pct": round(100.0 * splits / len(finished), 2),
        "matched_pct": round(100.0 * len(matched) / len(finished), 2),
        "weak_match_pct": round(100.0 * len(weak) / max(1, len(matched)), 2),
        "weak_match_n": len(weak),
    }


def _summarise_ac(
    rows: list[dict[str, Any]], device_type: str | None = None
) -> dict[str, Any]:
    """Anti-crease figures over the anti-wrinkle-enabled washer/dryer rows (item 207).

    ``early`` counts anti-crease finalises (every one, on any piece of the run)
    BEFORE the trace's last reading above ``anti_wrinkle_max_power``: the spin was
    still ahead, so it opens another cycle. ``cut`` is the same test for the first end by ANY path. ``lag`` is the
    first end minus the trace's last above-stop reading (with ``--tumble-tail``
    the harness's own split/lag columns also count the tail's own detections, so
    read these instead); ``held lag`` is the same over the finalises the #399
    spin guard held.
    """
    sel = [
        r for r in rows
        if r.get("anti_wrinkle") and r["device_type"] in AC_DEVICE_TYPES
        and (device_type is None or r["device_type"] == device_type)
    ]
    if not sel:
        return {"n": 0}
    fin = [r for r in sel if r.get("ac_final_s") is not None]
    held = [r for r in fin if r.get("ac_held")]
    lag = [
        (r["first_end_s"] - r["active_end_s"]) / 60.0
        for r in sel
        if r.get("first_end_s") is not None and r.get("active_end_s") is not None
    ]
    held_lag = [
        (r["ac_final_s"] - r["active_end_s"]) / 60.0
        for r in held if r.get("active_end_s") is not None
    ]
    cut = sum(
        1 for r in sel
        if r.get("first_end_s") is not None and r.get("last_high_s") is not None
        and r["first_end_s"] < r["last_high_s"]
    )

    def _r(vals: list[float], fn: Any) -> float | None:
        return round(float(fn(vals)), 2) if vals else None

    return {
        "n": len(sel),
        "finalized": len(fin),
        "held": len(held),
        "event": sum(1 for r in held if r.get("ac_release") == "event"),
        "cap": sum(1 for r in held if r.get("ac_release") == "cap"),
        "unarmed": sum(1 for r in held if r.get("ac_release") == "unarmed"),
        "early": sum(int(r.get("ac_early_n", int(bool(r.get("ac_early"))))) for r in sel),
        "cut": cut,
        "med_lag_min": _r(lag, np.median),
        "p90_lag_min": _r(lag, lambda v: np.percentile(v, 90)),
        "max_lag_min": _r(lag, np.max),
        "held_med_lag_min": _r(held_lag, np.median),
        "held_max_lag_min": _r(held_lag, np.max),
    }


_AC_KEYS = (
    "n", "finalized", "held", "event", "cap", "unarmed", "early", "cut",
    "med_lag_min", "p90_lag_min", "max_lag_min", "held_med_lag_min", "held_max_lag_min",
)


def _print_ac_summary(rows: list[dict[str, Any]]) -> None:
    scopes = [None, *sorted({r["device_type"] for r in rows if r["device_type"] in AC_DEVICE_TYPES})]
    lines = [(scope or "ALL", _summarise_ac(rows, scope)) for scope in scopes]
    lines = [(name, s) for name, s in lines if s["n"]]
    if not lines:
        return
    print("\nanti-crease (anti-wrinkle on; item 207)")
    hdr = (
        f"{'scope':<18}{'n':>5}{'final':>7}{'held':>6}{'event':>7}{'cap':>5}"
        f"{'unarm':>7}{'EARLY':>7}{'cut':>5}{'med lag':>9}{'p90':>8}{'max':>8}"
        f"{'held med':>10}{'max':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    def _f(v: Any) -> str:
        return "-" if v is None else f"{v:.2f}"

    for name, s in lines:
        print(
            f"{name:<18}{s['n']:>5}{s['finalized']:>7}{s['held']:>6}{s['event']:>7}"
            f"{s['cap']:>5}{s['unarmed']:>7}{s['early']:>7}{s['cut']:>5}"
            f"{_f(s['med_lag_min']):>9}{_f(s['p90_lag_min']):>8}{_f(s['max_lag_min']):>8}"
            f"{_f(s['held_med_lag_min']):>10}{_f(s['held_max_lag_min']):>8}"
        )
    print(
        "final = anti-crease finalises; held = the #399 spin guard held one; "
        "event/cap = how it released;\nEARLY = finalised before the trace's last "
        "reading above anti_wrinkle_max_power; cut = first end (any path) before it;"
        "\nlag = first end minus last above-stop reading, minutes."
    )


def _summarise_stall(
    rows: list[dict[str, Any]], device_type: str | None = None
) -> dict[str, Any]:
    """Stall display figures (#452) over the rows of one scope.

    ``detected``: a stall flagged inside the inserted plateau; ``false``: a row
    flagged outside it (on an unmodified run, any flag); ``closed in halt``: the
    cycle was closed while the plateau was still running (it then splits on resume).
    """
    sel = [r for r in rows if device_type is None or r["device_type"] == device_type]
    halted = [r for r in sel if r.get("halt_start_s") is not None]
    det = [r for r in halted if r.get("stall_in_halt")]
    lat = [r["stall_latency_s"] / 60.0 for r in det if r.get("stall_latency_s") is not None]
    return {
        "n": len(sel),
        "halted": len(halted),
        "detected": len(det),
        "detected_pct": round(100.0 * len(det) / len(halted), 1) if halted else None,
        "med_latency_min": round(float(np.median(lat)), 1) if lat else None,
        "false": sum(1 for r in sel if r.get("stall_outside_halt")),
        "closed_in_halt": sum(1 for r in halted if r.get("closed_in_halt")),
        "sb_fired": sum(1 for r in sel if r.get("sb_final_s") is not None),
    }


_STALL_KEYS = (
    "n", "halted", "detected", "detected_pct", "med_latency_min", "false",
    "closed_in_halt", "sb_fired",
)


def _print_stall_summary(rows: list[dict[str, Any]]) -> None:
    if not any(r.get("halt_start_s") is not None or r.get("stall_n") for r in rows):
        return
    print("\nstall display (#452)")
    hdr = (
        f"{'scope':<18}{'n':>5}{'halted':>8}{'detect':>8}{'%':>7}{'lat min':>9}"
        f"{'false':>7}{'closed':>8}{'sb fin':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for scope in [None, *sorted({r["device_type"] for r in rows})]:
        st = _summarise_stall(rows, scope)
        if not st["n"]:
            continue
        print(
            f"{scope or 'ALL':<18}{st['n']:>5}{st['halted']:>8}{st['detected']:>8}"
            f"{st['detected_pct']!s:>7}{st['med_latency_min']!s:>9}{st['false']:>7}"
            f"{st['closed_in_halt']:>8}{st['sb_fired']:>8}"
        )
    print(
        "detect = flagged inside the inserted plateau; false = flagged outside it; "
        "closed = the cycle\nwas closed while the plateau ran; sb fin = standby-band "
        "finalizes."
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    devices = sorted({r["device_type"] for r in rows})
    hdr = (
        f"{'scope':<18}{'n':>5}{'med lag':>9}{'mean':>8}{'p90':>8}"
        f"{'early>1m':>10}{'early>5m':>10}{'splits':>8}{'matched':>9}{'weak':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for scope in [None, *devices]:
        s = _summarise(rows, scope)
        if not s["n"]:
            continue
        name = scope or "ALL"
        print(
            f"{name:<18}{s['n']:>5}{s['median_lag_min']:>9.2f}{s['mean_lag_min']:>8.2f}"
            f"{s['p90_lag_min']:>8.2f}{s['early_1min_pct']:>9.2f}%{s['early_5min_pct']:>9.2f}%"
            f"{s['split_pct']:>7.2f}%{s['matched_pct']:>8.1f}%{s['weak_match_n']:>7}"
        )
    print(
        "\nlag = when the detector left ENDING, minus the trace's own active span,"
        " in minutes."
    )
    print("weak = matched cycles whose final confidence is below 0.4 (the gate's bar).")
    _print_ac_summary(rows)
    _print_stall_summary(rows)


def _compare(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    b_by_id = {(r["export"], r["id"]): r for r in before}
    a_by_id = {(r["export"], r["id"]): r for r in after}
    common = sorted(set(b_by_id) & set(a_by_id))
    print(f"paired cycles: {len(common)}  (before {len(before)}, after {len(after)})\n")

    devices = sorted({b_by_id[k]["device_type"] for k in common})
    for scope in [None, *devices]:
        bs = _summarise([b_by_id[k] for k in common], scope)
        as_ = _summarise([a_by_id[k] for k in common], scope)
        if not bs["n"] or not as_["n"]:
            continue
        name = scope or "ALL"
        print(f"=== {name} (n={as_['n']})")
        for key, unit in (
            ("median_lag_min", " min"), ("mean_lag_min", " min"), ("p90_lag_min", " min"),
            ("early_1min_pct", "%"), ("early_5min_pct", "%"), ("split_pct", "%"),
        ):
            print(f"    {key:<18} {bs[key]:>8}{unit} -> {as_[key]:>8}{unit}")
        print()

    ac_scopes = [None, *sorted({
        b_by_id[k]["device_type"] for k in common
        if b_by_id[k]["device_type"] in AC_DEVICE_TYPES
    })]
    for scope in ac_scopes:
        bs = _summarise_ac([b_by_id[k] for k in common], scope)
        as_ = _summarise_ac([a_by_id[k] for k in common], scope)
        if not bs["n"] or not as_["n"]:
            continue
        print(f"=== anti-crease {scope or 'ALL'} (n={as_['n']})")
        for key in _AC_KEYS[1:]:
            print(f"    {key:<18} {bs[key]!s:>8} -> {as_[key]!s:>8}")
        print()
    ac_moved = [
        k for k in common
        if b_by_id[k].get("ac_final_s") != a_by_id[k].get("ac_final_s")
    ]
    if ac_moved:
        print(f"cycles whose anti-crease finalise moved: {len(ac_moved)} / {len(common)}")
        for k in ac_moved[:40]:
            b, a = b_by_id[k], a_by_id[k]

            def _at(r: dict[str, Any]) -> str:
                v = r.get("ac_final_s")
                tag = r.get("ac_release") or ("free" if v is not None else "none")
                early = " EARLY" if r.get("ac_early") else ""
                return "-" if v is None else f"{v / 60:.1f}m {tag}{early}"

            print(
                f"    {b['device_type']:<16} {b['id']:<14} {str(b['label'])[:24]:<24} "
                f"{_at(b):>18} -> {_at(a):<18}"
            )
        print()

    for scope in [None, *devices]:
        bs = _summarise_stall([b_by_id[k] for k in common], scope)
        as_ = _summarise_stall([a_by_id[k] for k in common], scope)
        if not (bs["halted"] or as_["halted"] or bs["false"] or as_["false"]):
            continue
        print(f"=== stall {scope or 'ALL'} (n={as_['n']})")
        for key in _STALL_KEYS[1:]:
            print(f"    {key:<18} {bs[key]!s:>8} -> {as_[key]!s:>8}")
        print()

    moved = [
        k for k in common
        if (b_by_id[k].get("end_offset_s") or 0) != (a_by_id[k].get("end_offset_s") or 0)
    ]
    print(f"cycles whose end moved at all: {len(moved)} / {len(common)}")
    for k in moved[:25]:
        b, a = b_by_id[k], a_by_id[k]
        db = (b.get("end_offset_s") or 0) - b["active_span_s"]
        da = (a.get("end_offset_s") or 0) - a["active_span_s"]
        print(
            f"    {b['device_type']:<16} {b['id']:<14} conf="
            f"{b['confidence']!s:<6} lag {db / 60:>7.2f} -> {da / 60:>7.2f} min"
        )
    if len(moved) > 25:
        print(f"    ... and {len(moved) - 25} more")


#: ``--check`` regression tolerances (audit TESTING-09): how far a metric may move
#: the wrong way, per device type, before the check fails. Lags in minutes;
#: early ends and splits are cycle counts.
BASELINE_TOLERANCE: dict[str, float] = {
    "median_lag_min": 0.25,
    "mean_lag_min": 0.5,
    "p90_lag_min": 1.0,
    "early_1min_n": 0,
    "early_5min_n": 0,
    "split_n": 0,
}
#: The replay flags a baseline records and ``--check`` replays with.
BASELINE_FLAGS = (
    "loo", "all_formats", "shipped_watchdog", "anti_wrinkle", "tumble_tail",
    "device_types", "no_shortening",
)


def _baseline_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per device type (and ``ALL``): the ``_summarise`` figures plus counts."""
    out: dict[str, dict[str, Any]] = {}
    for scope in [None, *sorted({r["device_type"] for r in rows})]:
        s = _summarise(rows, scope)
        if not s["n"]:
            continue
        n = s["n"]
        out[scope or "ALL"] = {
            "n": n,
            "median_lag_min": s["median_lag_min"],
            "mean_lag_min": s["mean_lag_min"],
            "p90_lag_min": s["p90_lag_min"],
            "early_1min_n": int(round(s["early_1min_pct"] * n / 100.0)),
            "early_5min_n": int(round(s["early_5min_pct"] * n / 100.0)),
            "split_n": int(round(s["split_pct"] * n / 100.0)),
            "early_1min_pct": s["early_1min_pct"],
            "early_5min_pct": s["early_5min_pct"],
            "split_pct": s["split_pct"],
        }
    return out


def _git(*argv: str) -> str:
    try:
        return subprocess.run(
            ["git", *argv], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.rstrip()
    except OSError:
        return ""


def _tree_state() -> dict[str, Any]:
    """HEAD, the uncommitted files, and a hash of every integration source file."""
    digest = hashlib.sha256()
    pkg = REPO / "custom_components" / "ha_washdata"
    for path in sorted(pkg.rglob("*.py")):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return {
        "head": _git("rev-parse", "--short", "HEAD"),
        "dirty": [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()],
        "integration_py_sha256": digest.hexdigest()[:16],
    }


def _write_baseline(
    path: str, rows: list[dict[str, Any]], flags: dict[str, Any], tree: dict[str, Any]
) -> None:
    doc = {
        "about": (
            "end_gate_eval baseline (audit TESTING-09). Regenerate with "
            "--write-baseline after an intended end-gate change; gate with --check."
        ),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flags": flags,
        "tree": tree,
        "tolerance": BASELINE_TOLERANCE,
        "summary": _baseline_summary(rows),
    }
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote baseline ({len(doc['summary'])} scopes) to {path}")


def _check_baseline(baseline: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    """0 within tolerance, 1 on a regression, 2 when the corpus no longer matches."""
    tol = {**BASELINE_TOLERANCE, **(baseline.get("tolerance") or {})}
    before = baseline.get("summary") or {}
    after = _baseline_summary(rows)
    regressions: list[str] = []
    drift: list[str] = []
    print(f"{'scope':<18}{'metric':<16}{'baseline':>10}{'now':>10}{'tol':>7}")
    for scope in sorted(set(before) | set(after)):
        b, a = before.get(scope), after.get(scope)
        if b is None or a is None or b["n"] != a["n"]:
            drift.append(
                f"{scope}: n {b['n'] if b else 0} -> {a['n'] if a else 0}"
            )
            continue
        for key, limit in tol.items():
            if key not in b or key not in a:
                continue
            delta = float(a[key]) - float(b[key])
            flag = ""
            if delta > float(limit) + 1e-9:
                flag = "  REGRESSION"
                regressions.append(f"{scope} {key} {b[key]} -> {a[key]} (tol +{limit})")
            elif abs(delta) > 1e-9:
                flag = "  improved" if delta < 0 else "  within tol"
            print(f"{scope:<18}{key:<16}{b[key]!s:>10}{a[key]!s:>10}{limit!s:>7}{flag}")
    if drift:
        print("\ncorpus differs from the baseline (re-baseline, do not compare):")
        for line in drift:
            print(f"    {line}")
    if regressions:
        print("\nREGRESSION beyond tolerance:")
        for line in regressions:
            print(f"    {line}")
        return 1
    if drift:
        return 2
    print("\nend gates within the baseline's tolerance")
    return 0


#: ``--jobs`` unit timings of the previous run (longest-first ordering only).
TIMINGS_FILE = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "ha_washdata_end_gate" / "timings.json"
)


#: ``--jobs`` / ``--jobs-file`` ceiling.
MAX_JOBS = 8


def default_jobs() -> int:
    """One core left for the rest of the machine, at most ``MAX_JOBS``."""
    return max(1, min((os.cpu_count() or 2) - 1, MAX_JOBS))


def _setup_replay(no_shortening: bool, no_stall_guard: bool, repo: str | None = None) -> None:
    """Patches and probes every replay needs; idempotent (also the pool initializer)."""
    global REPO  # noqa: PLW0603
    if repo is not None:
        REPO = Path(repo)  # a worker re-imports this module: keep the caller's root
    _integration()
    if no_shortening:
        # Patch `const`, not `cycle_detector`. Since register item 355 the gate
        # calls `resolve_end_gate_late_ratio(device_type)`, which reads these two
        # names out of `const` at call time - rebinding the detector module's
        # imported copy no longer reaches it, and this arm would silently stop
        # disabling the shortening while still reporting itself as the pre-306
        # baseline. Both names, because the per-device map wins for washers.
        from custom_components.ha_washdata import const as _const  # noqa: PLC0415

        _const.END_GATE_LATE_RATIO = 1e9
        _const.END_GATE_LATE_RATIO_BY_DEVICE = {}

    if no_stall_guard:
        from custom_components.ha_washdata import cycle_detector as _cd  # noqa: PLC0415

        _cd.STALL_HOLDS_STANDBY_BAND = False

    logging.getLogger("custom_components.ha_washdata").setLevel(logging.ERROR)
    _install_anticrease_probe()
    _install_stall_probe()
    _install_user_pause_probe()


def _unit_key(path: Path, ci: int, cyc: dict[str, Any]) -> str:
    return f"{_export_key(path)}#{ci}#{str(cyc.get('id'))[:12]}"


def _plan_units(
    paths: list[Path], all_formats: bool, device_types: tuple[str, ...] | None,
    timings: dict[str, float],
) -> list[tuple[float, int, int, str]]:
    """``(estimated s, export index, cycle index, key)`` per cycle, in submit order.

    One unit per stored cycle, so the slowest export spreads over every worker.
    The estimate is the unit's time in the previous run (``timings``), else its
    trace length scaled by the seconds per reading those timings show.
    """
    raw: list[tuple[int, int, str, int]] = []
    for i, path in enumerate(paths):
        found = _export_doc(path, all_formats, device_types)
        if found is None:
            continue
        for ci, cyc in enumerate(found[2]):
            cyc = cyc if isinstance(cyc, dict) else {}
            raw.append((i, ci, _unit_key(path, ci, cyc), len(cyc.get("power_data") or [])))
    known = [(timings[k], n) for _i, _c, k, n in raw if k in timings]
    per_reading = (sum(t for t, _n in known) / max(1, sum(n for _t, n in known))) if known else 1e-3
    units = [(timings.get(k, n * per_reading), i, ci, k) for i, ci, k, n in raw]
    # Longest export first, and its longest cycles first. Export-major because a
    # worker that moves to another export repeats that export's setup: cycle-major
    # order cost 271 setups (79 core-s) on the corpus at --jobs 4, this ~100.
    total: dict[int, float] = {}
    for est, i, _ci, _k in units:
        total[i] = total.get(i, 0.0) + est
    return sorted(units, key=lambda u: (-total[u[1]], u[1], -u[0], u[2]))


def _replay_unit(
    unit: tuple[int, int, Path, bool, bool, dict[str, Any]],
) -> tuple[int, int, list[tuple[int, dict[str, Any]]], float]:
    idx, ci, path, no_shortening, loo, kw = unit
    t0 = time.monotonic()
    rows = _replay_export(path, no_shortening, loo, only=frozenset((ci,)), memo=True, **kw)
    return idx, ci, rows, time.monotonic() - t0


def _load_timings(path: Path | None) -> dict[str, float]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    return {k: float(v) for k, v in doc.items() if isinstance(v, (int, float))}


def _save_timings(path: Path | None, timings: dict[str, float]) -> None:
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(timings, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # an ordering hint only


def _granted(jobs: int, jobs_file: Path | None) -> int:
    """Workers allowed right now: ``--jobs``, or the count in ``--jobs-file``."""
    if jobs_file is None:
        return jobs
    try:
        return max(1, min(MAX_JOBS, int(jobs_file.read_text(encoding="utf-8").strip())))
    except (OSError, ValueError):
        return jobs


def _replay_parallel(
    paths: list[Path], jobs: int, no_shortening: bool, no_stall_guard: bool,
    loo: bool, kw: dict[str, Any], timings_path: Path | None = None,
    jobs_file: Path | None = None,
) -> list[dict[str, Any]]:
    """The serial loop's rows, in its order, from ``jobs`` worker processes.

    Units are single cycles (``_plan_units`` order), handed out one at a time
    as workers free up, so no worker is left holding a long tail. With
    ``jobs_file`` the number of units in flight is re-read after every unit, so
    a scheduler (``devtools/verify.py``) can hand the run more cores mid-way;
    spawned workers start on demand. Rows are put back in (export, cycle index)
    order, so the output is identical to ``--jobs 1``; progress goes to stderr.
    """
    timings = _load_timings(timings_path)
    units = _plan_units(paths, kw["all_formats"], kw["device_types"], timings)
    todo = iter(units)
    got: list[tuple[int, int, dict[str, Any]]] = []
    t0 = time.monotonic()
    step = max(1, len(units) // 20)
    finished = 0
    # spawn, not fork: the caller may have threads (pytest does), and a fork copies
    # their held locks. Each worker imports the integration once (~3 s).
    with ProcessPoolExecutor(
        max(jobs, MAX_JOBS) if jobs_file else jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_setup_replay, initargs=(no_shortening, no_stall_guard, str(REPO)),
    ) as ex:
        inflight: dict[Any, str] = {}

        def top_up() -> None:
            limit = _granted(jobs, jobs_file)
            while len(inflight) < limit:
                unit = next(todo, None)
                if unit is None:
                    return
                _est, i, ci, key = unit
                inflight[ex.submit(_replay_unit, (i, ci, paths[i], no_shortening, loo, kw))] = key

        top_up()
        while inflight:
            done, _pending = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                idx, ci, rows, took = fut.result()
                timings[inflight.pop(fut)] = round(took, 3)
                got.extend((idx, ci, row) for _c, row in rows)
                finished += 1
                if finished % step == 0 or finished == len(units):
                    print(
                        f"end_gate_eval: {finished}/{len(units)} cycles, "
                        f"{time.monotonic() - t0:.0f}s, {_granted(jobs, jobs_file)} workers",
                        file=sys.stderr, flush=True,
                    )
            top_up()
    _save_timings(timings_path, timings)
    return [row for _i, _c, row in sorted(got, key=lambda r: (r[0], r[1]))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write per-cycle rows here for --compare")
    ap.add_argument(
        "--jobs", type=int, default=default_jobs(), metavar="N",
        help="worker processes (default: cpu_count - 1, at most 8; 1 = serial). "
        "Rows and summary are identical for every N",
    )
    ap.add_argument(
        "--jobs-file", metavar="FILE",
        help="re-read the worker count from FILE after every cycle (a scheduler such as "
        "devtools/verify.sh grows it mid-run); --jobs is the start value",
    )
    ap.add_argument(
        "--timings", default=str(TIMINGS_FILE), metavar="FILE",
        help="--jobs: per-cycle timings of the previous run, read to hand out the "
        "longest cycles first and rewritten after the run ('' = none)",
    )
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument(
        "--no-shortening", action="store_true",
        help="pre-306 arm: patch END_GATE_LATE_RATIO out of reach",
    )
    ap.add_argument(
        "--loo", action="store_true",
        help="leave-one-out: match each cycle against profiles rebuilt without it",
    )
    ap.add_argument(
        "--all-formats", action="store_true",
        help="read every corpus shape (diagnostics dumps too) via eval.py, clones dropped",
    )
    ap.add_argument(
        "--shipped-watchdog", action="store_true",
        help="ignore each export's watchdog_interval: replay at the device type's default",
    )
    ap.add_argument(
        "--anti-wrinkle", choices=("export", "force"), default="export",
        help="export: each export's own anti_wrinkle_enabled; force: on for every "
        "washer/dryer/washer-dryer (item 207)",
    )
    ap.add_argument(
        "--tumble-tail", action="store_true",
        help="washers/dryers: replace the trailing quiet with the #296 anti-crease "
        "tumble tail, so only the anti-crease finalise can close the cycle",
    )
    ap.add_argument(
        "--device-types", default="",
        help="comma-separated device types to replay (default: all)",
    )
    ap.add_argument(
        "--write-baseline", metavar="FILE",
        help="write the per-device-type summary, flags and tree state here (TESTING-09)",
    )
    ap.add_argument(
        "--check", metavar="BASELINE",
        help="replay with BASELINE's flags and exit 1 on a regression beyond its "
        "tolerance, 2 when the corpus differs",
    )
    ap.add_argument(
        "--rows", metavar="FILE",
        help="with --check: summarise these --json rows instead of replaying",
    )
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override one option on every replayed export, as Apply all saves it "
        "(repeatable; VALUE is parsed as JSON, else kept as a string). Not with --check",
    )
    ap.add_argument(
        "--export", action="append", default=[], metavar="SUBSTR",
        help="replay only exports whose corpus path contains SUBSTR (repeatable). "
        "Not with --check",
    )
    ap.add_argument(
        "--halt-at", type=float, default=None, metavar="F",
        help="insert a flat standby plateau at fraction F of each cycle's active span "
        "(F >= 1: after its last activity) (#452)",
    )
    ap.add_argument(
        "--halt-min", default="30", metavar="M",
        help="with --halt-at: plateau length in minutes, or 'Nx' for N x the active span",
    )
    ap.add_argument(
        "--halt-level", default="auto", metavar="W",
        help="with --halt-at: plateau level in W (+-0.4), or 'auto' (see the docstring)",
    )
    ap.add_argument(
        "--user-pause", action="store_true",
        help="with --halt-at: the plateau is a user pause (paused at its first reading, "
        "resumed after it), not a halt",
    )
    ap.add_argument(
        "--no-stall-guard", action="store_true",
        help="before arm: the stall display does not hold the standby-band finalize",
    )

    args = ap.parse_args(argv)
    overrides: dict[str, Any] = {}
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            ap.error(f"--set needs KEY=VALUE, got {item!r}")
        try:
            overrides[key.strip()] = json.loads(raw)
        except ValueError:
            overrides[key.strip()] = raw
    if args.check and (overrides or args.export):
        ap.error("--check replays the baseline's corpus and options: drop --set/--export")

    if args.compare:
        _compare(*args.compare)
        return 0

    baseline: dict[str, Any] | None = None
    if args.check:
        baseline = json.loads(Path(args.check).read_text(encoding="utf-8"))
        if args.rows:
            return _check_baseline(baseline, json.loads(Path(args.rows).read_text()))
        # Replay exactly what the baseline measured, whatever else was passed.
        for key, value in (baseline.get("flags") or {}).items():
            if key in BASELINE_FLAGS:
                setattr(args, key, ",".join(value) if key == "device_types" else value)
    elif args.rows:
        ap.error("--rows needs --check")
    _integration()
    device_types = tuple(t.strip() for t in args.device_types.split(",") if t.strip())
    # Taken before the replay: other work may change the tree while it runs.
    tree = _tree_state() if args.write_baseline else {}

    if args.user_pause and args.halt_at is None:
        ap.error("--user-pause needs --halt-at (the plateau it pauses over)")
    _setup_replay(args.no_shortening, args.no_stall_guard)
    halt = (
        (float(args.halt_at), str(args.halt_min), str(args.halt_level))
        if args.halt_at is not None else None
    )
    if args.check and halt is not None:
        ap.error("--check replays the baseline's corpus unmodified: drop --halt-at")
    rows: list[dict[str, Any]] = []
    corpus = REPO / "cycle_data"
    if args.all_formats:
        devices, _clones = _corpus_module().load_corpus(corpus)
        paths = [corpus / dev.path for dev in devices]
    else:
        paths = sorted(corpus.rglob("*.json"))
    if args.export:
        paths = [p for p in paths if any(sub in str(p) for sub in args.export)]
    kw = {
        "all_formats": args.all_formats, "shipped_watchdog": args.shipped_watchdog,
        "anti_wrinkle": args.anti_wrinkle, "tumble_tail": args.tumble_tail,
        "device_types": device_types or None, "overrides": overrides or None,
        "halt": halt,
        "user_pause": bool(args.user_pause),
    }
    if (args.jobs > 1 or args.jobs_file) and paths:
        rows = _replay_parallel(
            paths, args.jobs, args.no_shortening, args.no_stall_guard, args.loo, kw,
            Path(args.timings) if args.timings else None,
            Path(args.jobs_file) if args.jobs_file else None,
        )
    else:
        for path in paths:
            rows.extend(_measure_export(path, args.no_shortening, args.loo, **kw))

    if not rows:
        print("no replayable cycles found - is cycle_data/ present?")
        return 1

    _print_summary(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {len(rows)} rows to {args.json}")
    if args.write_baseline:
        flags = {key: getattr(args, key) for key in BASELINE_FLAGS}
        flags["device_types"] = list(device_types)
        _write_baseline(args.write_baseline, rows, flags, tree)
    if baseline is not None:
        print()
        return _check_baseline(baseline, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
