#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the live ETA on real traces: when it first appears and how wrong it is.

Audit PROGRESS-12: the only ETA harness in the repo (``eta_phase_eval.py``, since
removed) did not run the shipped estimator, and the audit's own replay was never
committed, so PROGRESS-19's "first ETA a median 22 min in, 17 % of cycles with one
at 10 % elapsed" could not be re-cut once register item 404 changed the first-match
cadence. This is that harness, checked in.

**Replay.** Every cycle goes through ``playground.simulate_cycle_detail``: the real
``CycleDetector`` (incl. item 404's half-interval matching until the first commit),
the real Stage 1-5 matcher re-gridded per query, the manager's ``match_rules``
(commit, switching, verified pause, revoke), the live watchdog's keepalives, and the
same ``progress`` functions with the EMA carried and dt-scaled as live. Nothing
here computes a progress figure. The only change made to the Playground is lifting
its display caps (``MAX_SERIES_PER_CYCLE`` / ``MAX_EVENTS_PER_CYCLE``) for the run,
because thinning a long cycle's series to 600 points would move its first ETA.

**Leave-one-out by default.** Each cycle is matched against a store rebuilt WITHOUT
it (``end_gate_eval --loo``'s fold, reused from there). ``--in-sample`` turns that
off; in-sample a cycle matches an envelope it helped build, which flatters both
the match and the duration, so never quote it.

**Store state.** Each export gets the detector config, options and ProfileStore a
real ``WashDataManager`` builds from it (``end_gate_eval._production``), the
one-time banked-tail repair an upgraded store has run (``--no-repair`` skips it),
and every envelope rebuilt with the code under test.

**Population.** Labelled ``past_cycles`` whose programme is a stored profile, status
``completed``/``force_stopped`` (what an envelope accepts), at least 10 readings
and a truth duration of at least 10 min. ``n_other`` in the JSON says how many other
cycles of the programme were left to learn from (0 = the programme is new to it).

**Truth.** ``D`` = the offset of the trace's last reading above ``stop_threshold_w``:
when the appliance actually stopped working. At ``f`` elapsed the ETA shown is the
latest replayed estimate at or before ``f x D``; its error is
``remaining_s - (D - t)`` (+ = predicted more time left than there was), and the
relative error divides that by the true remaining ``D - t``. Dishwasher drying
tails sit past ``D``, so a dishwasher's positive bias is partly by design.
``stored_s`` (the repaired stored duration) is in the JSON for that re-cut.

**First ETA** = the replay offset of the first estimate with a remaining time,
in minutes from the trace's first reading. The replay steps the estimator at most
every 30 s of replay time (``_SIM_SERIES_THROTTLE_S``) and only at readings, so it
can trail the commit by up to 30 s, or more on a plug that is silent at steady
power; ``first_commit_s`` (the commit event itself) is in the JSON beside it.

**Synthetic halt (register item 514).** ``--halt-at F --halt-min M`` inserts the
#452 standby plateau of ``end_gate_eval --halt-at`` (same helper, same level) at
fraction ``F`` (0 < F < 1) of each cycle's active span, for ``M`` minutes, every
later reading shifted by it. The truth is then the halted trace's own last
reading above stop, so the plateau counts as time the cycle took. Each row adds
``halt_*``: the remaining time shown when the plateau began (``rem_h0_s``), when
the stall display first flagged (``rem_stall_s``) and when the plateau ended
(``rem_h1_s``), so ``rem_h0_s - rem_h1_s`` is how far the ETA counted down
through a halt (the plateau's own length when it ignores it), and the error of the
ETA shown ``HALT_AFTER_S`` after the wash resumed. ``--device-types`` restricts
the corpus (the stall display runs on washers, washer-dryers and dryers only).
Measured 2026-10-06, ``--all-formats``, 45 min at 50% over the 246 washer and
washer-dryer targets (189 shown as stalled): counting stalled time, the ETA fell a
median 40.9 min through the halt and read 41.8 / 36.4 / 21.6 min wrong (median
|error|) 1 / 10 / 30 min after the resume; leaving finished stalls out, 40.6 min
and 25.0 / 16.7 / 13.7; leaving the stall shown now out too (shipped), 0.5 min and
13.2 / 16.7 / 13.7. The real corpus is unchanged (0 of 450 rows).

Usage, from the repo root (``--all-formats`` is ~450 cycles, ~13 min at ``--jobs 3``;
a worker holds ~200 MB)::

    python3 devtools/eta_eval.py --all-formats --jobs 4 --json /tmp/eta.json
    python3 devtools/eta_eval.py --compare /tmp/before.json /tmp/after.json

The A/B arms are two states of the code; check the other one out in a worktree
(``git worktree add --detach /tmp/before <ref> && ln -s "$PWD/cycle_data"
/tmp/before/``), never by stashing a shared working tree. ``--pre-404`` is the
one arm that needs no checkout: it pins the detector's commit flag so matching
runs at the full ``match_interval`` before the first commit too (item 404 off).
``--all-formats`` / ``--shipped-watchdog`` mean what they mean in
``devtools/end_gate_eval.py``.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import math
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from custom_components.ha_washdata import playground  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
    CONF_MATCH_PERSISTENCE,
    CONF_WATCHDOG_INTERVAL,
    resolve_watchdog_interval_default,
)
from custom_components.ha_washdata.profile_store import (  # noqa: E402
    decompress_power_data,
)


def _load_end_gate() -> Any:
    """``devtools/end_gate_eval.py``: its production store, LOO fold and corpus reader."""
    name = "wd_eta_eval_end_gate"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, REPO / "devtools" / "end_gate_eval.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


EG = _load_end_gate()

#: Elapsed fractions of the truth duration at which the shown ETA is scored.
FRACS = (0.10, 0.25, 0.50, 0.75, 0.90)
#: Shorter "cycles" are fragments: their fractions are seconds apart.
MIN_TRUTH_S = 600.0
STATUSES = ("completed", "force_stopped")
#: Cycles per parallel job (an export's cycles are spread over its jobs).
CHUNK = 12
#: ``--halt-at``: seconds after the plateau ended at which the ETA error is read.
HALT_AFTER_S = (60.0, 600.0, 1800.0)


def _fkey(f: float) -> str:
    return f"{f:.2f}"


@contextlib.contextmanager
def _pre_404_cadence(enabled: bool) -> Iterator[None]:
    """``--pre-404``: the detector always reads itself as committed.

    Register item 404 halves ``match_interval`` until the first commit, gated on
    ``CycleDetector._match_committed`` (read only by that rate limit). Pinning it True
    gives the pre-404 cadence on the code under test, like ``end_gate_eval
    --no-shortening`` gives the pre-306 gate.
    """
    if not enabled:
        yield
        return
    from custom_components.ha_washdata.cycle_detector import CycleDetector  # noqa: PLC0415

    CycleDetector._match_committed = property(  # type: ignore[assignment]  # noqa: SLF001
        lambda _self: True, lambda _self, _value: None
    )
    try:
        yield
    finally:
        del CycleDetector._match_committed  # noqa: SLF001


@contextlib.contextmanager
def _full_series() -> Iterator[None]:
    """Lift the Playground's display caps: the first ETA must not be thinned away."""
    saved = (playground.MAX_SERIES_PER_CYCLE, playground.MAX_EVENTS_PER_CYCLE)
    playground.MAX_SERIES_PER_CYCLE = 10**9
    playground.MAX_EVENTS_PER_CYCLE = 10**9
    try:
        yield
    finally:
        playground.MAX_SERIES_PER_CYCLE, playground.MAX_EVENTS_PER_CYCLE = saved


def truth_end(points: list[tuple[float, float]], stop: float) -> float:
    """Offset of the last reading above ``stop``: when the appliance stopped working."""
    active = [t for t, p in points if p > stop]
    return float(active[-1]) if active else 0.0


def cycle_metrics(series: list[dict[str, Any]], truth_s: float, label: str | None) -> dict[str, Any]:
    """First ETA and the ETA shown at each fraction of ``truth_s``, from a replay series."""
    est = [p for p in series if p.get("remaining_s") is not None]
    first = est[0] if est else None
    out: dict[str, Any] = {
        "first_eta_s": float(first["t"]) if first else None,
        "first_eta_program": first.get("matched_profile") if first else None,
        "first_eta_right": (first.get("matched_profile") == label) if first else None,
        "first_eta_err_s": (
            round(float(first["remaining_s"]) - (truth_s - float(first["t"])), 1) if first else None
        ),
        "at": {},
    }
    for f in FRACS:
        t_f = f * truth_s
        shown = None
        for p in series:
            if float(p["t"]) > t_f:
                break
            shown = p
        if shown is None or shown.get("remaining_s") is None:
            out["at"][_fkey(f)] = None
            continue
        true_rem = truth_s - float(shown["t"])
        err = float(shown["remaining_s"]) - true_rem
        out["at"][_fkey(f)] = {
            "t": float(shown["t"]),
            "err_s": round(err, 1),
            "rel": round(err / true_rem, 4) if true_rem > 0 else None,
            "right": shown.get("matched_profile") == label,
        }
    return out


def halt_metrics(
    series: list[dict[str, Any]], truth_s: float, h0: float, h1: float
) -> dict[str, Any]:
    """What the ETA did through an inserted halt (``--halt-at``), from a replay series.

    ``rem_*_s`` is the remaining time of the latest estimate at or before that
    moment (None without one); ``after`` maps each ``HALT_AFTER_S`` offset past the
    plateau's end to the error of the ETA shown then, as in :func:`cycle_metrics`.
    """
    def shown_at(t: float) -> dict[str, Any] | None:
        shown = None
        for p in series:
            if float(p["t"]) > t:
                break
            shown = p
        return shown

    def rem(t: float | None) -> float | None:
        p = shown_at(t) if t is not None else None
        return float(p["remaining_s"]) if p and p.get("remaining_s") is not None else None

    stall_on = next(
        (float(p["t"]) for p in series if p.get("stalled") and h0 <= float(p["t"]) <= h1), None
    )
    after: dict[str, Any] = {}
    for d in HALT_AFTER_S:
        p = shown_at(h1 + d)
        if p is None or p.get("remaining_s") is None or float(p["t"]) < h1 or truth_s <= h1 + d:
            after[str(int(d))] = None
            continue
        true_rem = truth_s - float(p["t"])
        after[str(int(d))] = round(float(p["remaining_s"]) - true_rem, 1)
    return {
        "halt_start_s": round(h0, 1), "halt_end_s": round(h1, 1),
        "halt_stall_on_s": round(stall_on, 1) if stall_on is not None else None,
        "rem_h0_s": rem(h0), "rem_stall_s": rem(stall_on), "rem_h1_s": rem(h1),
        "halt_after_err_s": after,
    }


def _first_event(events: list[dict[str, Any]], etype: str) -> float | None:
    for ev in events:
        if ev.get("type") == etype:
            return float(ev.get("t") or 0.0)
    return None


# ---------------------------------------------------------------------- replay

def _prepare(path: Path, flags: dict[str, bool]) -> dict[str, Any] | None:
    """The export as an upgraded live store holds it: config, store, snapshots, targets."""
    doc = EG._load_doc(path, flags["all_formats"])  # noqa: SLF001
    if doc is None:
        return None
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    data = doc.get("data") or {}
    if not device_type or len(data.get("past_cycles") or []) < EG.MIN_CYCLES:
        return None
    if flags.get("device_types") and device_type not in flags["device_types"]:
        return None
    base = dict(data)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        base[key] = list(base.get(key) or [])
    base["profiles"] = {k: dict(v) if isinstance(v, dict) else v
                        for k, v in (base.get("profiles") or {}).items()}
    base["envelopes"] = dict(base.get("envelopes") or {})
    sw = flags["shipped_watchdog"]
    cfg, store, opts = EG._production(doc, base, shipped_watchdog=sw)  # noqa: SLF001
    stop = float(cfg.stop_threshold_w)
    if flags["repair"]:
        EG._run(store.async_repair_banked_tails(stop, device_type))  # noqa: SLF001
    # Setup's sample repair before any match, as eval.py / end_gate_eval.py; it
    # mutates base["profiles"] in place, so the LOO fold stores inherit it.
    EG._run(store.async_repair_profile_samples())  # noqa: SLF001
    EG._rebuild_envelopes(store, list(base["profiles"]))  # noqa: SLF001
    try:
        prebuilt = playground._build_match_snapshots(store)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        prebuilt = None
    targets = []
    for cyc in base["past_cycles"]:
        if cyc.get("profile_name") not in base["profiles"] or cyc.get("status") not in STATUSES:
            continue
        pts = decompress_power_data(cyc)
        if len(pts) < EG.MIN_READINGS or truth_end(pts, stop) < MIN_TRUTH_S:
            continue
        targets.append(cyc)
    return {"doc": doc, "base": base, "cfg": cfg, "store": store, "opts": opts,
            "prebuilt": prebuilt, "targets": targets, "device_type": device_type}


def _replay_one(ctx: dict[str, Any], cyc: dict[str, Any], key: str,
                flags: dict[str, bool]) -> dict[str, Any] | None:
    base, cfg, opts = ctx["base"], ctx["cfg"], ctx["opts"]
    name = cyc.get("profile_name")
    stop = float(cfg.stop_threshold_w)
    store, prebuilt = ctx["store"], ctx["prebuilt"]
    if flags["loo"]:
        _c, store, _o = EG._production(  # noqa: SLF001
            ctx["doc"], EG._fold_data(base, cyc),  # noqa: SLF001
            shipped_watchdog=flags["shipped_watchdog"],
        )
        EG._rebuild_envelopes(store, [name])  # noqa: SLF001
        try:
            prebuilt = playground._build_match_snapshots(store)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            prebuilt = None
    replayed, halt_s = cyc, None
    if flags.get("halt"):
        at, length, level = flags["halt"]
        replayed, h0, h1, _span = EG._with_halt(  # noqa: SLF001
            cyc, decompress_power_data(cyc), stop, float(at), str(length),
            EG._halt_level(stop, str(level)),  # noqa: SLF001
        )
        halt_s = (h0, h1)
    with _full_series():
        sim = playground.simulate_cycle_detail(
            replayed, cfg, None, store, opts, price=None, compute_series=True, prebuilt=prebuilt,
        )
    if "error" in sim:
        print(f"replay failed: {key} {str(cyc.get('id'))[:12]}: {sim['error']}", file=sys.stderr)
        return None
    pts = decompress_power_data(replayed)
    truth_s = truth_end(pts, stop)
    out = sim.get("outcome") or {}
    n_other = sum(
        1 for k in ("past_cycles", "reference_cycles", "backfill_cycles")
        for c in base.get(k) or [] if c is not cyc and c.get("profile_name") == name
    )
    try:
        stored = float(cyc.get("duration") or 0.0)
    except (TypeError, ValueError):
        stored = 0.0
    return {
        "export": key,
        "device_type": ctx["device_type"],
        "id": str(cyc.get("id"))[:12],
        "label": name,
        "n_other": n_other,
        "readings": len(pts),
        "truth_s": round(truth_s, 1),
        "stored_s": round(stored, 1),
        "detected_count": int(out.get("detected_count") or 0),
        "final_program": out.get("matched_profile"),
        "expected_s": out.get("expected_s"),
        "first_commit_s": _first_event(sim.get("events") or [], "match_commit"),
        **cycle_metrics(sim.get("series") or [], truth_s, name),
        **(halt_metrics(sim.get("series") or [], truth_s, *halt_s) if halt_s else {}),
        "loo": flags["loo"],
        "repair": flags["repair"],
        "pre_404": flags.get("pre_404", False),
        "match_interval_s": getattr(cfg, "match_interval", None),
        "match_persistence": opts.get(CONF_MATCH_PERSISTENCE),
        "watchdog_s": opts.get(
            CONF_WATCHDOG_INTERVAL, resolve_watchdog_interval_default(ctx["device_type"])
        ),
    }


_CTX_MEMO: dict[str, dict[str, Any] | None] = {}


def _job(args: tuple[str, str, str, int, int, dict[str, bool]]) -> list[dict[str, Any]]:
    """Replay every ``k``-th of ``of`` target cycles of one export."""
    path, corpus, key, k, of, flags = args
    memo_key = f"{path}|{json.dumps(flags, sort_keys=True)}"
    if memo_key not in _CTX_MEMO:
        while len(_CTX_MEMO) >= 2:
            _CTX_MEMO.pop(next(iter(_CTX_MEMO)))
        _CTX_MEMO[memo_key] = _prepare(Path(path), flags)
    ctx = _CTX_MEMO[memo_key]
    if ctx is None:
        return []
    rows = []
    for cyc in ctx["targets"][k::of]:
        try:
            row = _replay_one(ctx, cyc, key, flags)
        except Exception as exc:  # noqa: BLE001 - one bad cycle must not cost the run
            print(f"replay failed: {key} {str(cyc.get('id'))[:12]}: {exc!r}", file=sys.stderr)
            row = None
        if row is not None:
            rows.append(row)
    return rows


def _export_key(path: Path, corpus: Path) -> str:
    rel = str(path.relative_to(corpus))
    # Contributors' real names are in some file names (eval.py PRIVATE_DIRS).
    return "cycle_data/" + EG._corpus_module().public_key(rel)  # noqa: SLF001


def _corpus_paths(corpus: Path, all_formats: bool) -> list[Path]:
    if all_formats:
        devices, _clones = EG._corpus_module().load_corpus(corpus)  # noqa: SLF001
        return [corpus / dev.path for dev in devices]
    return sorted(corpus.rglob("*.json"))


def _n_past(path: Path, all_formats: bool) -> int:
    doc = EG._load_doc(path, all_formats)  # noqa: SLF001
    if not isinstance(doc, dict):
        return 0
    data = doc.get("data")
    return len(data.get("past_cycles") or []) if isinstance(data, dict) else 0


def run(
    corpus: Path,
    *,
    loo: bool = True,
    all_formats: bool = False,
    shipped_watchdog: bool = False,
    repair: bool = True,
    pre_404: bool = False,
    jobs: int = 1,
    only: str | None = None,
    log: Any = None,
    halt: tuple[float, str, str] | None = None,
    device_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Replay the corpus; one row per target cycle, sorted by (export, id)."""
    flags: dict[str, Any] = {
        "loo": loo, "all_formats": all_formats,
        "shipped_watchdog": shipped_watchdog, "repair": repair, "pre_404": pre_404,
    }
    if halt is not None:
        flags["halt"] = list(halt)
    if device_types:
        flags["device_types"] = sorted(device_types)
    pkg_log = logging.getLogger("custom_components.ha_washdata")
    saved_level = pkg_log.level
    pkg_log.setLevel(logging.ERROR)
    # Workers are forked inside this block, so they inherit the patch.
    with _pre_404_cadence(pre_404):
        try:
            return _run(corpus, flags, all_formats, only, jobs, log)
        finally:
            _CTX_MEMO.clear()
            pkg_log.setLevel(saved_level)


def _run(corpus: Path, flags: dict[str, bool], all_formats: bool, only: str | None,
         jobs: int, log: Any) -> list[dict[str, Any]]:
    work = []
    for path in _corpus_paths(corpus, all_formats):
        if only and only not in str(path):
            continue
        n = _n_past(path, all_formats)
        if n < EG.MIN_CYCLES:
            continue
        of = max(1, math.ceil(n / CHUNK)) if jobs > 1 else 1
        key = _export_key(path, corpus)
        work += [(str(path), str(corpus), key, k, of, flags) for k in range(of)]
    rows: list[dict[str, Any]] = []
    if jobs <= 1:
        for job in work:
            rows += _job(job)
            if log:
                log(f"  {job[2]}: {len(rows)} rows so far")
    else:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
            for done, part in enumerate(ex.map(_job, work), 1):
                rows += part
                if log and done % 10 == 0:
                    log(f"  {done}/{len(work)} jobs, {len(rows)} rows")
    return sorted(rows, key=lambda r: (r["export"], r["id"]))


# ---------------------------------------------------------------------- report

def _pct(num: float, den: float) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def summarise(rows: list[dict[str, Any]], device_type: str | None = None) -> dict[str, Any]:
    """First-ETA timing, coverage and error figures over a set of replayed cycles."""
    sel = [r for r in rows if device_type is None or r["device_type"] == device_type]
    if not sel:
        return {"n": 0}
    first = [r["first_eta_s"] for r in sel if r["first_eta_s"] is not None]
    rights = [r["first_eta_right"] for r in sel if r["first_eta_s"] is not None]
    out: dict[str, Any] = {
        "n": len(sel),
        "never": len(sel) - len(first),
        "first_median_min": round(float(np.median(first)) / 60.0, 1) if first else None,
        "first_p75_min": round(float(np.percentile(first, 75)) / 60.0, 1) if first else None,
        "first_right_pct": _pct(sum(1 for x in rights if x), len(rights)),
        "at": {},
    }
    for f in FRACS:
        shown = [r["at"][_fkey(f)] for r in sel if r["at"].get(_fkey(f)) is not None]
        errs = np.array([s["err_s"] for s in shown], dtype=float) / 60.0
        rels = np.array([abs(s["rel"]) for s in shown if s["rel"] is not None], dtype=float)
        out["at"][_fkey(f)] = {
            "cov_pct": _pct(len(shown), len(sel)),
            "mae_min": round(float(np.mean(np.abs(errs))), 1) if errs.size else None,
            "medae_min": round(float(np.median(np.abs(errs))), 1) if errs.size else None,
            "bias_min": round(float(np.mean(errs)), 1) if errs.size else None,
            "rel_med_pct": round(100.0 * float(np.median(rels)), 1) if rels.size else None,
            "right_pct": _pct(sum(1 for s in shown if s["right"]), len(shown)),
        }
    return out


def _f(v: Any, width: int, unit: str = "") -> str:
    return f"{'-' if v is None else f'{v}{unit}':>{width}}"


def print_summary(rows: list[dict[str, Any]]) -> None:
    scopes = [None, *sorted({r["device_type"] for r in rows})]
    hdr = (f"{'scope':<16}{'n':>5}{'never':>7}{'1st ETA med':>13}{'p75':>7}"
           f"{'ETA@10%':>9}{'ETA@25%':>9}{'1st right':>11}")
    print(hdr)
    print("-" * len(hdr))
    for scope in scopes:
        s = summarise(rows, scope)
        if not s["n"]:
            continue
        print(f"{scope or 'ALL':<16}{s['n']:>5}{s['never']:>7}{_f(s['first_median_min'], 13)}"
              f"{_f(s['first_p75_min'], 7)}{_f(s['at']['0.10']['cov_pct'], 9, '%')}"
              f"{_f(s['at']['0.25']['cov_pct'], 9, '%')}{_f(s['first_right_pct'], 11, '%')}")
    print("\n1st ETA in minutes from the trace's first reading (cycles that got one);"
          " never = no ETA at all.\nETA@x% = an ETA is shown at x% of the true duration;"
          " 1st right = the first ETA's programme is the cycle's label.\n")
    hdr = (f"{'scope':<16}{'at':>5}{'shown':>8}{'MAE':>7}{'med|e|':>8}{'bias':>7}"
           f"{'med rel':>9}{'right':>8}")
    print(hdr)
    print("-" * len(hdr))
    for scope in scopes:
        s = summarise(rows, scope)
        if not s["n"]:
            continue
        for f in FRACS:
            a = s["at"][_fkey(f)]
            print(f"{(scope or 'ALL') if f == FRACS[0] else '':<16}{int(f * 100):>4}%"
                  f"{_f(a['cov_pct'], 8, '%')}{_f(a['mae_min'], 7)}{_f(a['medae_min'], 8)}"
                  f"{_f(a['bias_min'], 7)}{_f(a['rel_med_pct'], 9, '%')}{_f(a['right_pct'], 8, '%')}")
    print("\nminutes; bias + = predicted more time left than there was; med rel = median"
          " |error| / true remaining;\nright = the shown ETA's programme is the label."
          " Truth = last reading above stop_threshold_w.")


def summarise_halt(rows: list[dict[str, Any]], device_type: str | None = None) -> dict[str, Any]:
    """``--halt-at`` figures: how far the ETA counted down through the plateau, and
    how wrong it was after the wash resumed (minutes; medians of absolute values)."""
    sel = [r for r in rows if r.get("halt_start_s") is not None
           and (device_type is None or r["device_type"] == device_type)]
    if not sel:
        return {"n": 0}

    def med(vals: list[float]) -> float | None:
        return round(float(np.median(vals)) / 60.0, 1) if vals else None

    down = [r["rem_h0_s"] - r["rem_h1_s"] for r in sel
            if r.get("rem_h0_s") is not None and r.get("rem_h1_s") is not None]
    stalled = [r for r in sel if r.get("halt_stall_on_s") is not None]
    down_st = [r["rem_stall_s"] - r["rem_h1_s"] for r in stalled
               if r.get("rem_stall_s") is not None and r.get("rem_h1_s") is not None]
    out: dict[str, Any] = {
        "n": len(sel), "stalled": len(stalled),
        "halt_min": med([r["halt_end_s"] - r["halt_start_s"] for r in sel]),
        "countdown_med_min": med(down),
        "countdown_stalled_med_min": med(down_st),
        "after": {},
    }
    for d in HALT_AFTER_S:
        errs = [r["halt_after_err_s"][str(int(d))] for r in sel
                if (r.get("halt_after_err_s") or {}).get(str(int(d))) is not None]
        out["after"][str(int(d))] = {
            "n": len(errs),
            "medae_min": med([abs(e) for e in errs]),
            "bias_min": round(float(np.mean(errs)) / 60.0, 1) if errs else None,
        }
    return out


def print_halt_summary(rows: list[dict[str, Any]]) -> None:
    if not any(r.get("halt_start_s") is not None for r in rows):
        return
    print("\nsynthetic halt (--halt-at)")
    hdr = (f"{'scope':<16}{'n':>5}{'stalled':>9}{'halt':>6}{'down':>7}{'down st':>9}"
           + "".join(f"{f'+{int(d) // 60}m |e|':>10}{'bias':>6}" for d in HALT_AFTER_S))
    print(hdr)
    print("-" * len(hdr))
    for scope in [None, *sorted({r["device_type"] for r in rows})]:
        h = summarise_halt(rows, scope)
        if not h["n"]:
            continue
        line = (f"{scope or 'ALL':<16}{h['n']:>5}{h['stalled']:>9}{_f(h['halt_min'], 6)}"
                f"{_f(h['countdown_med_min'], 7)}{_f(h['countdown_stalled_med_min'], 9)}")
        for d in HALT_AFTER_S:
            a = h["after"][str(int(d))]
            line += f"{_f(a['medae_min'], 10)}{_f(a['bias_min'], 6)}"
        print(line)
    print("down = median minutes the remaining time fell from the plateau's start to its end"
          " (its length when\nthe ETA ignores the halt); down st = from the stall flag to the"
          " end; +Nm |e| / bias = ETA error N min\nafter the wash resumed (median |error|, mean"
          " signed error; + = more time left than there was).")


def compare(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    b_by = {(r["export"], r["id"]): r for r in before}
    a_by = {(r["export"], r["id"]): r for r in after}
    common = sorted(set(b_by) & set(a_by))
    print(f"paired cycles: {len(common)}  (before {len(before)}, after {len(after)})\n")
    bs_rows = [b_by[k] for k in common]
    as_rows = [a_by[k] for k in common]
    for scope in [None, *sorted({r["device_type"] for r in bs_rows})]:
        bs, as_ = summarise(bs_rows, scope), summarise(as_rows, scope)
        if not bs["n"]:
            continue
        print(f"=== {scope or 'ALL'} (n={as_['n']})")
        bh, ah = summarise_halt(bs_rows, scope), summarise_halt(as_rows, scope)
        if bh["n"] or ah["n"]:
            for key in ("stalled", "countdown_med_min", "countdown_stalled_med_min"):
                print(f"    halt {key:<27}{_f(bh.get(key), 8)} -> {_f(ah.get(key), 8)}")
            for d in HALT_AFTER_S:
                b, a = bh["after"][str(int(d))], ah["after"][str(int(d))]
                print(f"    halt +{int(d) // 60:>2}m  |e| {_f(b['medae_min'], 5)} -> "
                      f"{_f(a['medae_min'], 5)}   bias {_f(b['bias_min'], 5)} -> "
                      f"{_f(a['bias_min'], 5)}")
        for key in ("never", "first_median_min", "first_p75_min", "first_right_pct"):
            print(f"    {key:<22}{_f(bs[key], 8)} -> {_f(as_[key], 8)}")
        for f in FRACS:
            b, a = bs["at"][_fkey(f)], as_["at"][_fkey(f)]
            print(f"    @{int(f * 100):>2}%  shown {_f(b['cov_pct'], 6)} -> {_f(a['cov_pct'], 6)}"
                  f"   MAE {_f(b['mae_min'], 5)} -> {_f(a['mae_min'], 5)}"
                  f"   bias {_f(b['bias_min'], 5)} -> {_f(a['bias_min'], 5)}"
                  f"   right {_f(b['right_pct'], 5)} -> {_f(a['right_pct'], 5)}")
        print()

    def _first(r: dict[str, Any]) -> float:
        return r["first_eta_s"] if r["first_eta_s"] is not None else math.inf

    moved = sorted(
        (k for k in common if abs(_first(a_by[k]) - _first(b_by[k])) >= 60
         or (_first(a_by[k]) == math.inf) != (_first(b_by[k]) == math.inf)),
        key=lambda k: -abs(np.nan_to_num(_first(a_by[k]) - _first(b_by[k]), posinf=1e9, neginf=-1e9)),
    )
    print(f"cycles whose first ETA moved by >= 1 min: {len(moved)} / {len(common)}")
    for k in moved[:25]:
        b, a = b_by[k], a_by[k]
        fb = "never" if b["first_eta_s"] is None else f"{b['first_eta_s'] / 60:.1f}"
        fa = "never" if a["first_eta_s"] is None else f"{a['first_eta_s'] / 60:.1f}"
        print(f"    {b['device_type']:<16} {b['id']:<14} {fb:>7} -> {fa:>7} min")
    if len(moved) > 25:
        print(f"    ... and {len(moved) - 25} more")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write per-cycle rows here (input for --compare)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--in-sample", action="store_true",
                    help="match each cycle against profiles that include it (flattering)")
    ap.add_argument("--all-formats", action="store_true",
                    help="read every corpus shape (diagnostics dumps too) via eval.py, clones dropped")
    ap.add_argument("--shipped-watchdog", action="store_true",
                    help="ignore each export's watchdog_interval: replay at the device type's default")
    ap.add_argument("--no-repair", action="store_true",
                    help="skip the one-time banked-tail repair an upgraded store has run")
    ap.add_argument("--pre-404", action="store_true",
                    help="item 404 off: match at the full interval before the first commit too")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--only", help="replay only exports whose path contains this")
    ap.add_argument("--halt-at", type=float, default=None, metavar="F",
                    help="insert a #452 standby plateau at fraction F (0-1) of each cycle's "
                    "active span (see the docstring)")
    ap.add_argument("--halt-min", default="45", metavar="M",
                    help="with --halt-at: plateau length in minutes, or 'Nx' for N x the active span")
    ap.add_argument("--halt-level", default="auto", metavar="W",
                    help="with --halt-at: plateau level in W (+-0.4), or 'auto' (end_gate_eval's)")
    ap.add_argument("--device-types", default="",
                    help="comma-separated device types to replay (default: all)")
    ap.add_argument("--corpus", default=str(REPO / "cycle_data"))
    args = ap.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return 0
    if args.halt_at is not None and not 0.0 < args.halt_at < 1.0:
        ap.error("--halt-at must be a fraction of the active span in (0, 1)")
    halt = (
        (float(args.halt_at), str(args.halt_min), str(args.halt_level))
        if args.halt_at is not None else None
    )
    t0 = time.monotonic()
    rows = run(
        Path(args.corpus), loo=not args.in_sample, all_formats=args.all_formats,
        shipped_watchdog=args.shipped_watchdog, repair=not args.no_repair,
        pre_404=args.pre_404, jobs=args.jobs, only=args.only, log=lambda m: print(m, file=sys.stderr, flush=True),
        halt=halt,
        device_types=tuple(t for t in args.device_types.split(",") if t) or None,
    )
    if not rows:
        print("no replayable cycles found - is cycle_data/ present?")
        return 1
    print_summary(rows)
    print_halt_summary(rows)
    print(f"\n{'in-sample' if args.in_sample else 'leave-one-out'}, "
          f"{'no repair' if args.no_repair else 'banked tails repaired'}, "
          f"{'shipped' if args.shipped_watchdog else 'export'} watchdog"
          f"{', pre-404 cadence' if args.pre_404 else ''}; "
          f"{len({r['export'] for r in rows})} exports, {time.monotonic() - t0:.0f} s")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"wrote {len(rows)} rows to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
