#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How often a suggestion is offered over a device's history, replayed in order (item 637).

    python3 devtools/suggestion_churn_eval.py [--arm shipped|legacy|both] [--device SUBSTR]
                                              [--jobs N] [--corpus DIR] [--json FILE]

``suggestion_loop_eval.py`` asks whether Apply all reaches a fixed point on one
snapshot of history. This asks how often the same suggestions come back while the
history grows: a user who applies every suggestion as soon as it appears, one
cycle end at a time.

**Walk** (per corpus device, stored cycles in start order, all production code):

* every cycle that ended on its own commits its update intervals (``0.1 < dt < 1800``
  s from each real reading to the real reading before it, keepalives skipped, as
  the live clock runs: ``suggestion_loop_eval.stored_cycle_intervals``;
  the newest ``learning._CYCLE_MAX_INTERVALS``), as ``LearningManager.close_cycle_cadence`` does; then the operational pass runs on the
  cadence model (``suggestion_loop_eval.cadence``: one ``learning.cycle_cadence``
  summary per cycle, the median over the last ``CADENCE_RECENT_CYCLES``);
* at every stored cycle end, whatever its status, the threshold correctives run
  (``LearningManager.run_threshold_correctives``: the standby floor, then the
  steady-load start threshold);
* when the labelled cycles that ended on their own grew, the model and then the
  detection pass run (``LearningManager.process_cycle_end``'s evidence rule), before
  the batch, so what they leave pending is part of the set the batch is judged with;
* each time 5 more labelled cycles have ended on their own, the batch pass runs
  (``_maybe_run_batch_simulation``'s own rule) on ``LearningManager._batch_cycles``;
* each proposal goes through ``LearningManager._apply_suggestions_and_notify`` (noise
  gates, cooldown, reconcile) and the stored set is judged as a whole, as
  ``LearningManager._run_pass`` does, on a manager built from the device's options as
  applied so far, with the history up to that cycle;
* whatever is then stored for a counted key is applied at once and the post-apply
  cooldown (``MIN_SUGGESTION_COOLDOWN_CYCLES`` on the odometer) runs from there;
* the profiles are the history prefix's (``prefix_profile``): a programme exists
  from its first stored run (``ProfileStore.create_profile``) and each later run
  rebuilds it (``async_rebuild_envelope``), so the shortest programme that caps
  Profile Match Interval is one already run. Reference cycles count from the start
  only when imported before the walk; backfill cycles not at all.

Counted keys: the cadence keys (``learning.CADENCE_KEYS``) and Stop / Start Threshold.

**Legacy arm** (0.6.0, before item 637), the rest of the code as shipped: the cadence
from the newest 200 committed intervals (0.6.0's ``StatisticalModel(max_samples=200)``),
the noise gate's ``MIN_SUGGESTION_REL_DELTA`` relative bar on every key without the
seconds rule (``learning.CADENCE_HOLD_REL`` and ``learning._SECONDS_SUFFIXES`` patched),
and no batch stop hold band (``suggestion_engine.STOP_HOLD_BAND`` patched to infinity).
The cadence-key formulas and ``_suggestion_min_abs_delta`` are unchanged since 0.6.0.

**Held fixed**: the stored traces (not re-detected under the applied values; that is
``suggestion_loop_eval.py``), so a threshold or cadence change does not alter the
evidence the next pass reads. ``store/`` corpus files (community downloads, not one
appliance's history) are skipped.

Exit codes: 0, 2 no corpus.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterator  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom_components.ha_washdata import learning, suggestion_engine  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    MIN_SUGGESTION_REL_DELTA,
)
from custom_components.ha_washdata.profile_store import _parse_start_dt  # noqa: E402
from custom_components.ha_washdata.suggestion_engine import (  # noqa: E402
    OWNER_BATCH,
    OWNER_DETECTION,
    OWNER_MODEL,
    OWNER_OPERATIONAL,
    OWNER_STANDBY_FLOOR,
    OWNER_START_THRESHOLD,
)

_SLE_PATH = REPO / "devtools" / "suggestion_loop_eval.py"
_spec = importlib.util.spec_from_file_location("wd_suggestion_churn_sle", _SLE_PATH)
sle = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sle
_spec.loader.exec_module(sle)
ev = sle.ev

KEYS = (*learning.CADENCE_KEYS, CONF_STOP_THRESHOLD_W, CONF_START_THRESHOLD_W)
ARMS = ("shipped", "legacy")
#: ``LearningManager._maybe_run_batch_simulation``: first run and re-run step.
BATCH_MIN = 5
BATCH_RERUN_DELTA = 5
#: 0.6.0's cadence model: the newest 200 committed intervals.
LEGACY_WINDOW = 200


@contextlib.contextmanager
def legacy_patch(on: bool) -> Iterator[None]:
    """The 0.6.0 noise gate and batch stop rule on the shipped code (see module doc)."""
    if not on:
        yield
        return
    saved = (learning.CADENCE_HOLD_REL, learning._SECONDS_SUFFIXES, suggestion_engine.STOP_HOLD_BAND)  # noqa: SLF001
    learning.CADENCE_HOLD_REL = MIN_SUGGESTION_REL_DELTA
    learning._SECONDS_SUFFIXES = ()  # noqa: SLF001 - str.endswith(()) is False
    suggestion_engine.STOP_HOLD_BAND = math.inf
    try:
        yield
    finally:
        learning.CADENCE_HOLD_REL, learning._SECONDS_SUFFIXES, suggestion_engine.STOP_HOLD_BAND = saved  # noqa: SLF001


def legacy_cadence(per_cycle: list[list[float]]) -> tuple[float, float] | None:
    flat = [v for ivs in per_cycle for v in ivs][-LEGACY_WINDOW:]
    if len(flat) < 20:
        return None
    arr = np.asarray(flat)
    return float(np.percentile(arr, 95)), float(np.median(arr))


def _intervals(cycle: dict[str, Any]) -> list[float]:
    """The intervals a cycle commits to the live cadence model (between real reports)."""
    return sle.stored_cycle_intervals(cycle)


def _start_key(cycle: dict[str, Any]) -> datetime:
    parsed = _parse_start_dt(cycle.get("start_time"))
    if parsed is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _new_evidence(cycle: dict[str, Any]) -> bool:
    """What ``LearningManager.process_cycle_end`` counts before the model and
    detection passes: labelled (not noise) and ended on its own, traced or not."""
    return (
        isinstance(cycle, dict) and bool(cycle.get("profile_name"))
        and cycle.get("profile_name") != "noise" and learning._ended_on_its_own(cycle)  # noqa: SLF001
    )


def _labelled_own(cycle: dict[str, Any]) -> bool:
    return (
        bool(cycle.get("profile_name")) and cycle.get("profile_name") != "noise"
        and bool(cycle.get("power_data")) and learning._ended_on_its_own(cycle)  # noqa: SLF001
    )


def prefix_profile(store: Any, name: str, cycle: dict[str, Any]) -> None:
    """``cycle`` (now stored) reaches programme ``name`` the way the store puts it there.

    The first run of a programme creates it (``ProfileStore.create_profile``); every
    later one rebuilds it (``async_rebuild_envelope``), which re-derives its duration
    stats, sample and envelope from the evidence stored so far. Only that programme's
    evidence changed, so the other profiles are what a full rebuild would give.
    """
    if name in (store.get_profiles() or {}):
        ev.run_coro(store.async_rebuild_envelope(name))
        return
    cid = cycle.get("id")
    if cid and isinstance(cycle.get("duration"), (int, float)) \
            and store.find_stored_cycle(cid)[0] is cycle:
        ev.run_coro(store.create_profile(name, cid))
        return
    # No id to create it from (or a duplicate one): create_profile's own record.
    store.get_profiles()[name] = {"avg_duration": cycle.get("duration")}
    ev.run_coro(store.async_rebuild_envelope(name))


def walk(corpus: str, path: str, arm: str) -> dict[str, Any] | None:
    """One device's history under one arm: applies per counted key."""
    sle._quiet()  # noqa: SLF001
    dev = ev.load_device(Path(corpus), path)
    if dev is None:
        return None
    entry_data, options = ev.entry_dicts(dev, {})
    base = ev.base_data(dev)
    past = sorted(
        (c for c in base.get("past_cycles") or [] if isinstance(c, dict)), key=_start_key
    )
    # Profile state from the history prefix (PR #529 review): the exported profiles
    # carry every later run, so a programme first run late was known (and capped
    # Profile Match Interval) from the first pass. A programme exists from its first
    # stored run; the exported set only says which labels are programmes. Evidence
    # outside the walk is kept only when known to exist before it: a reference
    # cycle is stamped when it was imported, a backfill cycle only with its run's
    # own time (its import could be later), so backfill is left out.
    known = {n for n, p in (base.get("profiles") or {}).items() if isinstance(p, dict)}
    walk_start = _start_key(past[0]) if past else None
    references = [
        c for c in base.get("reference_cycles") or []
        if isinstance(c, dict) and walk_start is not None
        and _parse_start_dt(c.get("start_time")) is not None and _start_key(c) < walk_start
    ]
    profiles: dict[str, Any] = {}
    envelopes: dict[str, Any] = {}
    applies: dict[str, list[list[float]]] = {k: [] for k in KEYS}
    per_cycle: list[list[float]] = []
    ends = labelled = last_batch = last_apply = last_evidence = 0
    cadence_of = legacy_cadence if arm == "legacy" else sle.cadence
    with legacy_patch(arm == "legacy"):
        for i, cycle in enumerate(past):
            cad = None
            if learning._ended_on_its_own(cycle):  # noqa: SLF001
                # The newest _CYCLE_MAX_INTERVALS, as LearningManager._pending_intervals keeps.
                per_cycle.append(_intervals(cycle)[-learning._CYCLE_MAX_INTERVALS:])  # noqa: SLF001
                cad = cadence_of(per_cycle)
            if _labelled_own(cycle):
                labelled += 1
            run_batch = labelled >= BATCH_MIN and labelled - last_batch >= BATCH_RERUN_DELTA
            # LearningManager.process_cycle_end: the model and detection passes run
            # when the labelled cycles that ended on their own grew (new evidence).
            evidence = sum(1 for c in past[: i + 1] if _new_evidence(c))
            run_model = evidence > last_evidence
            # No early exit: the threshold correctives run at every cycle end.
            data = dict(base)
            data.update(
                past_cycles=past[: i + 1], reference_cycles=references, backfill_cycles=[],
                profiles=profiles, envelopes=envelopes,
                suggestions={}, locked_suggestions=[],
                suggestion_apply_cycle_count=last_apply, lifetime_cycle_count=0,
            )
            mgr = sle._manager(entry_data, options, dev.path, data)  # noqa: SLF001
            for stored in (references if i == 0 else []) + [cycle]:
                name = stored.get("profile_name")
                if isinstance(name, str) and name in known:
                    prefix_profile(mgr.profile_store, name, stored)
            lm = mgr.learning_manager
            eng = lm.suggestion_engine

            def gate(sug: dict[str, Any], owner: str) -> None:
                lm._apply_suggestions_and_notify(sug, owner=owner)  # noqa: SLF001
                stored = {k: dict(v) for k, v in (mgr.profile_store.get_suggestions() or {}).items()
                          if isinstance(v, dict)}
                lm._withdraw_joint_drops(stored, eng.for_job().joint_end_gate_drops(stored))  # noqa: SLF001

            if cad is not None:
                ends += 1
                gate(eng.for_job().generate_operational_suggestions(*cad), OWNER_OPERATIONAL)
            # LearningManager.run_threshold_correctives: every cycle end, whatever
            # its status, after the operational pass (PR #529 review: the start
            # pass proposes Start Threshold with no cadence and no batch).
            gate(eng.for_job().generate_standby_floor_suggestions(), OWNER_STANDBY_FLOOR)
            gate(eng.for_job().generate_start_threshold_suggestions(), OWNER_START_THRESHOLD)
            if run_model:
                # Before the batch, as production spawns them: a value they leave
                # pending is part of the set the batch's proposals are judged with.
                last_evidence = evidence
                gate(eng.for_job().generate_model_suggestions(), OWNER_MODEL)
                gate(eng.for_job().generate_detection_suggestions(), OWNER_DETECTION)
            if run_batch:
                last_batch = labelled
                gate(eng.for_job().run_batch_simulation(lm._batch_cycles()), OWNER_BATCH)  # noqa: SLF001
            stored = mgr.profile_store.get_suggestions() or {}
            changed = False
            for key in KEYS:
                entry = stored.get(key)
                value = entry.get("value") if isinstance(entry, dict) else None
                if value is None:
                    continue
                old = options.get(key)
                applies[key].append([old if old is None else float(old), float(value)])
                options[key] = value
                changed = True
            if changed:
                last_apply = i + 1  # the odometer: one stored cycle per run here
    return {
        "device": ev.public_key(dev.path), "device_type": options.get("device_type")
        or entry_data.get("device_type"), "arm": arm, "cycles": len(past), "cycle_ends": ends,
        "applies": {k: len(v) for k, v in applies.items()}, "history": applies,
    }


def _job(args: tuple[str, str, str]) -> dict[str, Any] | None:
    return walk(*args)


def run(corpus: Path, arms: tuple[str, ...], devices: list[str] | None, jobs: int) -> list[dict]:
    # store/ left out before clone removal, so a package cannot drop a history.
    kept, _dropped = ev.load_corpus(corpus, exclude=("store/",))
    paths = [d.path for d in kept if not devices or any(s in d.path for s in devices)]
    tasks = [(str(corpus), p, arm) for arm in arms for p in paths]
    if jobs <= 1:
        out = [walk(*t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            out = list(pool.map(_job, tasks))
    return [r for r in out if r is not None]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    ap.add_argument("--device", action="append", help="substring of a corpus path (repeatable)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--corpus", default=str(REPO / "cycle_data"))
    ap.add_argument("--json", help="write the per-device results here")
    a = ap.parse_args(argv)
    corpus = Path(a.corpus)
    if not corpus.is_dir():
        print(f"no corpus at {corpus}", file=sys.stderr)
        return 2
    arms = ARMS if a.arm == "both" else (a.arm,)
    rows = run(corpus, arms, a.device, a.jobs)
    if not rows:
        print("no devices", file=sys.stderr)
        return 2
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        print(f"== {arm}: {len(mine)} devices, {sum(r['cycle_ends'] for r in mine)} cycle ends")
        for r in sorted(mine, key=lambda r: r["device"]):
            if any(r["applies"].values()):
                print(f"  {r['device'][:56]:56s} ends={r['cycle_ends']:4d} "
                      + " ".join(f"{k}={n}" for k, n in r["applies"].items() if n))
        print("  TOTAL " + " ".join(f"{k}={sum(r['applies'][k] for r in mine)}" for k in KEYS))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
