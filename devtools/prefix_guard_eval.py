#!/usr/bin/env python3
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
"""Threshold sweep for the Smart-Termination split guards (issue #364).

    python3 devtools/prefix_guard_eval.py [--quiet-cuts] [--sweep] [--jobs N]
                                          [--json FILE] [--dump FILE]

``devtools/eval.py`` answers "did matching accuracy regress?" - it scores the
ranking, so it cannot say whether these guards discriminate. This harness answers
that, by rebuilding the two populations that matter:

  POSITIVE ("would split")   a long cycle's trace TRUNCATED to the point where a
                             SHORTER profile is winning the match. This is the
                             #364 failure: Smart Termination fires at 0.98x that
                             shorter profile's duration and the remainder becomes
                             a second cycle. The guard SHOULD fire here.
  NEGATIVE ("genuine end")   a cycle at its own true end. The guard MUST NOT
                             fire: every false fire is a legitimate cycle pushed
                             onto the slower power-based fallback timeout.

Two independent guards are swept:

  prefix     the #364 prefix-fit term, REMOVED from the integration in 0.5.8 and
             kept here as a harness copy (``_prefix_rule`` / ``_prefix_score``,
             the shipped Stage-6 scoring and rule byte for byte, checked against
             it on the last tree that shipped it: 0 disagreeing folds) - does a
             LONGER candidate's curve, truncated to the elapsed duration, explain
             the trace better than the winner does, by a margin (0.15 shipped)?
  power      is the trailing mean power more than SMART_TERM_TAIL_MAX_RATIO x what
             the matched profile draws at its own end
             (``ProfileStore.profile_tail_power``)? This is the one that ships.

Both are shorten-only: firing can only ever BLOCK an early finish, never end a
cycle sooner. That asymmetry is why the false-block column is a cost (a later
finish) and the missed column is the bug (a split cycle), and why the operating
point sits well clear of the false-block knee.

**The shipped matcher (audit MATCH-CORE-07 / MATCH-EVAL-04 / MATCH-EVAL-09).**
Until 0.5.8 this harness scored a pipeline that does not ship: a representative
training cycle as the template instead of the envelope, ``max_duration_ratio``
1.5 (shipped 1.8), no ``energy_mode`` (washers integrate energy), no Stage 5, a
stop threshold read from the file's top-level ``entry_options`` (absent from every
diagnostics dump and from any export running the default, so those devices got no
pause evidence), and a ``--quiet-cuts`` flag that still OR-ed in the #288
full-shape term the ENDING flag dropped in audit LIVE-18. Every fold now runs
``ProfileStore.async_match_profile(..., in_progress=True, stop_threshold_w=...)``
on a store a real ``WashDataManager`` builds from the device's own entry data and
options (``end_gate_eval._production``), with every envelope rebuilt by the code
under test and the held-out cycle's programme rebuilt WITHOUT it (leave-one-out,
so the cycle is neither template nor pause evidence for itself). The prefix rule
runs on the full post-collapse candidate list that call hands
``_match_prefix_ambiguity`` (recorded by a pass-through wrapper), with prefix
scores the harness adds to every longer candidate after the worker returns (no
score or ranking key is touched). The stop threshold is the production
detector config's. The corpus is every shape ``devtools/eval.py`` reads (exports
and diagnostics dumps, clone files dropped), and every labelled cycle in
``past_cycles``, ``reference_cycles`` and ``backfill_cycles`` is a fold, as there
(the old loader also folded community reference cycles). The old ``--in-progress`` /
``--no-prefix-shape`` switches are gone: the live matcher always scores a running
cycle in progress, with the prefix shape. Figures taken before this rewrite
(the const.py #364 / #424 blocks) measured the old pipeline.

Not production, still: the trailing mean is a time-weighted mean over the raw
trace (the detector's ``_trailing_mean_power`` also cuts outage gaps), and a
fold is judged at one instant, not through the 5-minute match cadence and the
detector's own ENDING timeline (``end_gate_eval.py`` replays that).

**``--quiet-cuts`` (#424): the populations Smart Termination actually meets.** It
only runs in ENDING, after the power has sat below ``stop_threshold_w``, so:

  NEGATIVE      the whole cycle plus ``QUIET_CUT_S`` of trailing 0 W quiet.
  POS_RANDOM    the fixed-fraction cuts above - kept for comparison, but a cut in
                the middle of activity can never reach ENDING, so it is not a
                split the guards could prevent.
  POS_QUIET     ``QUIET_CUT_S`` into a mid-cycle pause below the threshold that
                power later RESUMED from, with a shorter programme winning and the
                run already at >= 0.9x that programme's duration: the moment a
                split can really happen.

and reports the prefix rule both without and with the pause evidence
(``SMART_TERM_PREFIX_MIN_PAUSE_S``; ``profile_pauses_below`` on the fold store).
Quote these populations. Without ``--quiet-cuts`` a negative is judged at its
last stored reading, and a trace stored up to its last activity still shows
running power there, which the detector never meets in ENDING: on 2026-10-04 the
power term false-blocked 116/574 (20%) such ends at 3.5x and 0/675 with the
quiet. Measured that day (71 devices, LOO, ``--quiet-cuts``): the prefix term
fired on 0 of 713 genuine ends, on 3 of 528 random-cut positives without the
pause evidence and 0 with it, and on 0 of 7 quiet positives; the power term
caught 156/508 (31%) positives at 3.5x with no false block.

**``--sweep``: why the prefix term was removed.** The margin x floor x ratio grid
(the shipped cap of 3 scorings re-applied in rank order), the largest prefix
margin per fold, and how far the prefix score is from the candidate's own shape
score. Since #400 Stages 2/3 score a running cycle against each candidate's
template truncated to the elapsed time (while that is <= 0.7 x its span), so for
those candidates the prefix score IS the shape score (median difference 0.0000)
and a longer programme whose start explains the trace better mostly wins the
match itself. 2026-10-04 (71 devices, LOO): the 7 quiet positives' best prefix
margin was -0.024, so no grid point (margin 0-0.15, floor 0-0.60, ratio
1.0-1.5) caught one; the random-cut catches it bought cost genuine-end blocks
(margin 0.05: 43 caught, 5 of 713 blocked) and never reach ENDING anyway. And
Stage 6 was half the matcher worker's CPU: removing it took the worker from 35.1
to 17.4 ms of process time per live match (1876 in-progress matches at 30-98% of
every labelled cycle of the first 40 corpus devices; it scored on 41% of them).

Exit codes: 0 ok, 2 no corpus or not enough folds. cycle_data/ is gitignored,
so this is maintainer-local.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "devtools"))

# The corpus / production-store plumbing is decisive_margin_eval's (itself
# end_gate_eval's `_production` / `_fold_data`), so the three agree on what a fold is.
from decisive_margin_eval import LISTS, _device, _fold_store, _key, _map, _paths  # noqa: E402

# Where along a long cycle the shorter profile can win. 0.98 is the ratio both
# Smart-Termination paths fire at, so the cut points bracket it.
CUT_FRACTIONS = (0.55, 0.65, 0.75, 0.85)
MARGIN_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.30)
RATIO_SWEEP = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
# --quiet-cuts: how long the trace has been quiet when it is judged. Roughly what
# ENDING entry plus the Smart-Termination debounce take on the shipped defaults.
QUIET_CUT_S = 300.0
#: Spacing of the synthetic 0 W readings that stand in for ENDING quiet.
QUIET_STEP_S = 30.0
MIN_READINGS = 20
#: Candidate keys the prefix rule reads (the rest is dropped to keep rows small).
#: ``ps_wide`` is the harness's own UNCAPPED prefix score (every candidate longer
#: than the winner, not just the first three past 1.10x), ``qfrac`` the elapsed
#: share of that candidate's template, ``paused`` its pause-evidence verdict.
_CAND_KEYS = ("name", "profile_duration", "shape_score", "score", "ps_wide", "qfrac",
              "paused")
#: --sweep grid for the prefix term (margin x floor x ratio).
GRID_MARGIN = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15)
GRID_MIN_SHAPE = (0.0, 0.40, 0.60)
GRID_MIN_RATIO = (1.0, 1.10, 1.25, 1.50)
#: The candidate cap Stage 6 shipped with (SMART_TERM_PREFIX_MAX_CANDIDATES).
PREFIX_MAX_CANDIDATES = 3
#: The template-coverage floor Stage 6 shipped with (SMART_TERM_PREFIX_MIN_COVERAGE).
PREFIX_MIN_COVERAGE = 0.90

#: The last ``_match_prefix_ambiguity`` call's inputs (one fold at a time per process).
_LAST: dict[str, Any] = {}


def _prefix_score(
    curr_arr: Any, sample: Any, elapsed: float, span_s: float, peak: float, config: dict
) -> float | None:
    """The #364 Stage-6 prefix score: the live trace against ``sample`` truncated
    to ``elapsed``, Stage-2 alignment blended with Stage-3 DTW exactly as
    ``analysis.prefix_shape_score`` computed it until 0.5.8. A harness-local copy
    so the measurement survives the rule's removal from the integration."""
    from custom_components.ha_washdata import analysis as an  # noqa: PLC0415
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        DEFAULT_DTW_BANDWIDTH,
        DEFAULT_DTW_MODE,
        MATCH_CORR_WEIGHT,
        MATCH_DDTW_DIST_SCALE,
        MATCH_DTW_BLEND,
        MATCH_DTW_DIST_SCALE,
        MATCH_DTW_ENSEMBLE_W,
    )

    pair = an.prefix_shape_arrays(curr_arr, sample, elapsed, span_s)
    if pair is None:
        return None
    a, b = pair
    score, _m, _o = an.find_best_alignment(
        a, b, 1.0, corr_weight=float(config.get("corr_weight", MATCH_CORR_WEIGHT))
    )
    band = float(config.get("dtw_bandwidth", DEFAULT_DTW_BANDWIDTH))
    if band <= 0.0:
        return float(score)
    dtw = an._stage3_dtw_score(  # noqa: SLF001
        a, b, peak,
        dtw_mode=str(config.get("dtw_mode", DEFAULT_DTW_MODE)),
        dtw_bandwidth=band,
        l1_scale=float(config.get("dtw_l1_scale", MATCH_DTW_DIST_SCALE)),
        ddtw_scale=float(config.get("dtw_ddtw_scale", MATCH_DDTW_DIST_SCALE)),
        ensemble_w=float(config.get("dtw_ensemble_w", MATCH_DTW_ENSEMBLE_W)),
    )
    blend = float(config.get("dtw_blend", MATCH_DTW_BLEND))
    return float(blend * score + (1.0 - blend) * dtw)


def _annotate_wide(cands: list[dict], current_power: Any, elapsed: float, config: dict) -> None:
    """``ps_wide`` / ``qfrac`` on EVERY candidate longer than the winner that the
    trace has not outlasted and whose template covers its duration - the shipped
    Stage-6 eligibility with the ratio floor at 1.0 and no cap, so the sweep can
    apply any floor and the shipped cap afterwards. Never touches a score."""
    import numpy as np  # noqa: PLC0415

    curr = np.asarray(current_power, dtype=float)
    if elapsed <= 0 or len(cands) < 2 or curr.size == 0:
        return
    best_dur = float(cands[0].get("profile_duration") or 0.0)
    if best_dur <= 0:
        return
    peak = float(np.max(curr))
    for cand in cands[1:]:
        prof_dur = float(cand.get("profile_duration") or 0.0)
        if prof_dur <= best_dur or prof_dur <= elapsed:
            continue
        span = float(cand.get("sample_span_s") or prof_dur)
        if span < prof_dur * PREFIX_MIN_COVERAGE:
            continue
        score = _prefix_score(curr, cand.get("sample") or [], elapsed, span, peak, config)
        if score is not None:
            cand["ps_wide"] = score
            cand["qfrac"] = elapsed / span


@contextlib.contextmanager
def _record_prefix_inputs() -> Iterator[None]:
    """Capture what ``async_match_profile`` hands ``_match_prefix_ambiguity``.

    A pass-through wrapper: the candidates are the full post-collapse list (not
    the ``[:5]`` the result carries) and ``best_duration`` the one the #288 flag
    used. The worker is wrapped too, to add the ``ps_wide`` prefix scores after it
    returns (purely additive: no score or ranking key is touched).
    """
    from custom_components.ha_washdata import analysis as an  # noqa: PLC0415
    from custom_components.ha_washdata import profile_store as ps  # noqa: PLC0415

    orig = ps._match_prefix_ambiguity  # noqa: SLF001
    orig_worker = an.compute_matches_worker

    def wrapped(candidates: list[dict], best_duration: float, pauses_below: Any = None):
        _LAST.update(cands=list(candidates), best_dur=float(best_duration or 0.0),
                     pauses=pauses_below)
        return orig(candidates, best_duration, pauses_below)

    def worker(current_power: Any, elapsed: float, snapshots: Any, config: dict) -> Any:
        out = orig_worker(current_power, elapsed, snapshots, config)
        _annotate_wide(out, current_power, float(elapsed or 0.0), config)
        return out

    ps._match_prefix_ambiguity = wrapped  # noqa: SLF001
    an.compute_matches_worker = worker
    try:
        yield
    finally:
        ps._match_prefix_ambiguity = orig  # noqa: SLF001
        an.compute_matches_worker = orig_worker


#: The constants the #364 prefix term shipped with (SMART_TERM_PREFIX_MARGIN /
#: _MIN_SHAPE / _MIN_RATIO), the operating point the sweep marks.
SHIPPED_PREFIX = (0.15, 0.40, 1.10)


def _prefix_rule(
    cands: list[dict],
    best_dur: float,
    margin: float = SHIPPED_PREFIX[0],
    min_shape: float = SHIPPED_PREFIX[1],
    min_ratio: float = SHIPPED_PREFIX[2],
    *,
    pauses: bool = False,
    key: str = "ps_wide",
) -> bool:
    """The #364 prefix-fit term at any operating point.

    A longer candidate (``> min_ratio`` x the winner) whose prefix score reaches
    ``min_shape`` and beats the winner's SHAPE score by ``margin``. ``key`` picks
    the score: ``ps_wide`` (uncapped; the shipped cap of PREFIX_MAX_CANDIDATES is
    re-applied here in rank order, as Stage 6 did) or ``prefix_score`` (what Stage
    6 annotated, on a tree that still has it). ``pauses`` applies the #424 pause
    evidence: a candidate whose programme never paused below the stop threshold
    does not count.
    """
    if best_dur <= 0 or len(cands) < 2:
        return False
    shape = cands[0].get("shape_score")
    best = float((shape if shape is not None else cands[0].get("score")) or 0.0)
    scored = 0
    for cand in cands[1:]:
        score = cand.get(key)
        if score is None or float(cand.get("profile_duration") or 0) <= best_dur * min_ratio:
            continue
        scored += 1
        if scored > PREFIX_MAX_CANDIDATES:
            break
        if pauses and cand.get("paused") is False:
            continue
        if float(score) >= min_shape and float(score) >= best + margin:
            return True
    return False


def _prefix_fires(cands: list[dict], best_dur: float, margin: float) -> bool:
    """The prefix term (no pause evidence) at ``margin``, shipped floor and ratio."""
    return _prefix_rule(cands, best_dur, margin)


def _tail_window_s(expected: float) -> float:
    """``CycleDetector._tail_window_s`` for a given expected duration."""
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        SMART_TERM_TAIL_WINDOW_FRAC,
        SMART_TERM_TAIL_WINDOW_MIN_S,
        SMART_TERM_TAIL_WINDOW_S,
    )

    if expected <= 0:
        return SMART_TERM_TAIL_WINDOW_S
    return min(
        SMART_TERM_TAIL_WINDOW_S,
        max(SMART_TERM_TAIL_WINDOW_MIN_S, expected * SMART_TERM_TAIL_WINDOW_FRAC),
    )


def _trailing_mean(points: list[tuple[float, float]], window_s: float) -> float | None:
    """Time-weighted mean power over the trailing ``window_s`` (trapezoid)."""
    if len(points) < 3:
        return None
    end = points[-1][0]
    win = [(t, p) for t, p in points if t >= end - window_s]
    if len(win) < 3:
        return None
    span = win[-1][0] - win[0][0]
    if span <= 0:
        return None
    energy = sum((p0 + p1) / 2.0 * (t1 - t0) for (t0, p0), (t1, p1) in zip(win, win[1:]))
    return energy / span


def _with_quiet(points: list[tuple[float, float]], until: float) -> list[tuple[float, float]]:
    """``points`` plus 0 W readings every ``QUIET_STEP_S`` up to offset ``until``."""
    out = list(points)
    t = out[-1][0] + 1.0
    while t < until:
        out.append((t, 0.0))
        t += QUIET_STEP_S
    if until > out[-1][0]:  # never a reading back in time, nor a duplicate
        out.append((until, 0.0))
    return out


def _fold(store: Any, loop: Any, points: list[tuple[float, float]], stop: float) -> dict | None:
    """One production match over ``points``; the row the sweeps read, or None."""
    _LAST.clear()
    duration = points[-1][0] - points[0][0]
    try:
        result = loop.run_until_complete(store.async_match_profile(
            points, duration, in_progress=True, stop_threshold_w=stop
        ))
    except Exception:  # noqa: BLE001
        return None
    if not result.best_profile or "cands" not in _LAST:
        return None
    pauses = _LAST.get("pauses")
    cands = []
    for c in _LAST["cands"]:
        row = {k: c.get(k) for k in _CAND_KEYS}
        if pauses is not None and row.get("ps_wide") is not None:
            try:
                row["paused"] = pauses(str(c.get("name") or ""))
            except Exception:  # noqa: BLE001 - no opinion, as in production
                row["paused"] = None
        cands.append(row)
    best_dur = _LAST["best_dur"]
    expected = float(result.expected_duration or 0.0)
    try:
        tail = store.profile_tail_power(result.best_profile)
    except Exception:  # noqa: BLE001
        tail = None
    trailing = _trailing_mean(points, _tail_window_s(expected))
    return {
        "top1": result.best_profile,
        "best_dur": best_dur,
        "cands": cands,
        # The removed #364 flag at its shipped constants, without and with the
        # pause evidence (the second is what the detector got).
        "landscape": _prefix_rule(cands, best_dur),
        "landscape_paused": _prefix_rule(cands, best_dur, pauses=True),
        "power_ratio": (trailing / tail) if (tail and trailing is not None and tail > 0) else None,
    }


def _collect_device(job: tuple[str, bool]) -> dict[str, list[dict]]:
    """The positive / negative / quiet-positive folds of one device."""
    path, quiet_cuts = job
    out: dict[str, list] = {"pos": [], "neg": [], "qpos": [], "cuts": [0, 0, 0]}
    dev = _device(Path(path), True, min_cycles=0)
    if dev is None:
        return out
    doc, base, store, cfg, _opts = dev
    # Every labelled cycle in every list is a fold, as in devtools/eval.py.
    folds = [c for key in LISTS for c in base.get(key) or [] if c.get("profile_name")]
    labels = {c.get("profile_name") for c in folds}
    if len(labels & set(base["profiles"])) < 2:
        return out
    from custom_components.ha_washdata.signal_processing import (  # noqa: PLC0415
        resumed_pauses,
    )
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        _cycle_readings,
    )

    # Contributors' names are in some corpus paths: key them by hash (eval.py).
    src = _key(Path(path))
    stop = float(cfg.stop_threshold_w)
    loop = asyncio.new_event_loop()
    try:
        with _record_prefix_inputs():
            for cyc in folds:
                name = cyc.get("profile_name")
                if not name or name not in base["profiles"]:
                    continue
                pts = [(float(t), float(p)) for t, p in _cycle_readings(cyc)]
                if len(pts) < MIN_READINGS:
                    continue
                t0 = pts[0][0]
                dur = pts[-1][0] - t0
                if dur <= 0:
                    continue
                fold = _fold_store(doc, base, cyc, store, True)
                tag = {"src": src, "true": name}

                # NEGATIVE: the whole cycle, at its own end (+ the ENDING quiet).
                neg_pts = _with_quiet(pts, pts[-1][0] + QUIET_CUT_S) if quiet_cuts else pts
                neg = _fold(fold, loop, neg_pts, stop)
                if neg is not None:
                    out["neg"].append({**tag, **neg})

                # POSITIVE: truncated to where a SHORTER profile is winning.
                for f in CUT_FRACTIONS:
                    prefix = [(t, p) for t, p in pts if t - t0 <= dur * f]
                    if len(prefix) < MIN_READINGS:
                        continue
                    pos = _fold(fold, loop, prefix, stop)
                    if pos is None:
                        continue
                    out["cuts"][0] += 1
                    out["cuts"][1] += pos["top1"] == name
                    if pos["top1"] == name or pos["best_dur"] >= dur * 0.95:
                        continue
                    out["cuts"][2] += 1
                    out["pos"].append({**tag, **pos})

                if not quiet_cuts:
                    continue
                # POS_QUIET: QUIET_CUT_S into a resumed mid-cycle pause.
                for start_frac, seconds in resumed_pauses(pts, stop):
                    if seconds < QUIET_CUT_S or start_frac < 0.2:
                        continue
                    start = t0 + start_frac * dur
                    cut = start + QUIET_CUT_S
                    prefix = [(t, p) for t, p in pts if t <= start]
                    prefix = _with_quiet(prefix, cut)
                    pos = _fold(fold, loop, prefix, stop)
                    if (
                        pos is not None
                        and pos["top1"] != name
                        and pos["best_dur"] < dur * 0.95
                        and (cut - t0) >= 0.9 * pos["best_dur"]
                    ):
                        out["qpos"].append({**tag, **pos})
    finally:
        loop.close()
    return out


def _report(pos: list[dict], neg: list[dict], qpos: list[dict], quiet_cuts: bool) -> None:
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        SMART_TERM_PREFIX_MIN_PAUSE_S,
        SMART_TERM_TAIL_MAX_RATIO,
    )

    print("\n=== removed #364 prefix term: margin over the winner (no pause evidence) ===")
    print(f"{'margin':>7} {'caught':>16} {'false blocks':>16}")
    for m in MARGIN_SWEEP:
        c = sum(1 for r in pos if _prefix_fires(r["cands"], r["best_dur"], m))
        f = sum(1 for r in neg if _prefix_fires(r["cands"], r["best_dur"], m))
        star = "  <- was shipped" if abs(m - SHIPPED_PREFIX[0]) < 1e-9 else ""
        print(f"{m:7.2f} {c:6d}/{len(pos)} ({c/len(pos)*100:3.0f}%) "
              f"{f:6d}/{len(neg)} ({f/len(neg)*100:3.0f}%){star}")

    print("\n=== power term: trailing mean vs the matched profile's own tail ===")
    pr_pos = [r for r in pos if r["power_ratio"] is not None]
    pr_neg = [r for r in neg if r["power_ratio"] is not None]
    print(f"{'ratio':>7} {'caught':>16} {'false blocks':>16}")
    for x in RATIO_SWEEP:
        c = sum(1 for r in pr_pos if r["power_ratio"] > x)
        f = sum(1 for r in pr_neg if r["power_ratio"] > x)
        star = "  <- shipped" if abs(x - SMART_TERM_TAIL_MAX_RATIO) < 1e-9 else ""
        print(f"{x:7.1f} {c:6d}/{max(1, len(pr_pos))} ({c/max(1, len(pr_pos))*100:3.0f}%) "
              f"{f:6d}/{max(1, len(pr_neg))} ({f/max(1, len(pr_neg))*100:3.0f}%){star}")

    print("\n=== power term alone (ships) vs power + the removed prefix flag ===")

    def _power(r: dict) -> bool:
        return r["power_ratio"] is not None and r["power_ratio"] > SMART_TERM_TAIL_MAX_RATIO

    for label, fn in (("power", _power),
                      ("power + prefix", lambda r: r["landscape_paused"] or _power(r))):
        c = sum(1 for r in pos if fn(r))
        f = sum(1 for r in neg if fn(r))
        print(f"{label:<15} caught {c}/{len(pos)} ({c/len(pos)*100:.0f}%) | "
              f"false blocks {f}/{len(neg)} ({f/len(neg)*100:.0f}%)")
    print("\nA false block costs a later finish (the power-based fallback still ends")
    print("the cycle). A miss costs a split cycle. Prefer the conservative side.")

    if not quiet_cuts:
        return
    print("\n=== --quiet-cuts: removed prefix flag, without / with pause evidence ===")
    print(f"pause evidence: SMART_TERM_PREFIX_MIN_PAUSE_S = {SMART_TERM_PREFIX_MIN_PAUSE_S:.0f} s")
    for label, rows in (("negatives", neg), ("pos_random", pos), ("pos_quiet", qpos)):
        if not rows:
            print(f"{label:<11} n=0")
            continue

        def _pct(key: str, _rows: list[dict] = rows) -> str:
            k = sum(1 for r in _rows if r[key])
            return f"{k}/{len(_rows)} ({k / len(_rows) * 100:.1f}%)"

        print(f"{label:<11} n={len(rows):<4} fires: without {_pct('landscape')}  "
              f"with {_pct('landscape_paused')}")


def _quantiles(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    v = sorted(vals)

    def q(f: float) -> float:
        return v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]

    return (f"n={len(v):<4} p50 {q(0.5):+.3f}  p90 {q(0.9):+.3f}  p99 {q(0.99):+.3f}  "
            f"max {v[-1]:+.3f}  >=0.05: {sum(x >= 0.05 for x in v)}  "
            f">=0.15: {sum(x >= 0.15 for x in v)}")


def _best_margin(row: dict, min_ratio: float) -> float | None:
    """The fold's largest prefix margin: max over every longer candidate (uncapped)
    of its prefix score minus the winner's shape score."""
    cands = row["cands"]
    if len(cands) < 2:
        return None
    shape = cands[0].get("shape_score")
    best = float((shape if shape is not None else cands[0].get("score")) or 0.0)
    m = [float(c["ps_wide"]) - best for c in cands[1:]
         if c.get("ps_wide") is not None
         and float(c.get("profile_duration") or 0) > row["best_dur"] * min_ratio]
    return max(m) if m else None


def _report_sweep(pos: list[dict], neg: list[dict], qpos: list[dict], cuts: list[int]) -> None:
    """--sweep: why the prefix term is inert on the shipped matcher, and the grid."""
    print("\n=== --sweep: the #364 prefix term on the shipped matcher ===")
    n, top_true, shorter = cuts
    print(f"random cuts matched: {n}; true programme on top {top_true} "
          f"({top_true / max(1, n) * 100:.1f}%); a shorter programme winning {shorter}")

    # Since #400 Stages 2/3 already score a running cycle against each candidate's
    # template truncated to the elapsed time (while elapsed <= 0.7 x its span), so
    # the Stage-6 prefix score is mostly the candidate's own shape score again.
    near, far = [], []
    for r in (*pos, *neg, *qpos):
        for c in r["cands"][1:]:
            if c.get("ps_wide") is None or c.get("shape_score") is None:
                continue
            d = abs(float(c["ps_wide"]) - float(c["shape_score"]))
            (near if float(c.get("qfrac") or 1.0) <= 0.7 else far).append(d)
    for label, ds in (("elapsed <= 0.7 x span (Stage 2/3 prefix)", near),
                      ("elapsed >  0.7 x span (Stage 2/3 full)  ", far)):
        if ds:
            ds.sort()
            print(f"|prefix - shape| {label}: n={len(ds)} median {ds[len(ds) // 2]:.4f} "
                  f"share < 0.005 {sum(d < 0.005 for d in ds) / len(ds) * 100:.0f}%")

    print("\nlargest prefix margin per fold (uncapped, every longer candidate):")
    for ratio in (1.0, 1.10):
        print(f"  ratio > {ratio:.2f}")
        for label, rows in (("negatives", neg), ("pos_random", pos), ("pos_quiet", qpos)):
            vals = [m for m in (_best_margin(r, ratio) for r in rows) if m is not None]
            print(f"    {label:<11} {_quantiles(vals)}  (of {len(rows)})")

    print("\ngrid (shipped cap of 3 re-applied; 'paused' = with the #424 pause evidence):")
    print(f"{'margin':>6} {'floor':>5} {'ratio':>5}  {'pos_random':>14} {'pos_quiet':>10} "
          f"{'false blocks':>13}  {'paused: pos':>11} {'pos_q':>5} {'false':>5}")
    for ratio in GRID_MIN_RATIO:
        for floor in GRID_MIN_SHAPE:
            for margin in GRID_MARGIN:
                cells = []
                for pz in (False, True):
                    for rows in (pos, qpos, neg):
                        cells.append(sum(_prefix_rule(r["cands"], r["best_dur"], margin, floor,
                                                      ratio, pauses=pz) for r in rows))
                star = "  <- was shipped" if (margin, floor, ratio) == SHIPPED_PREFIX else ""
                print(f"{margin:6.2f} {floor:5.2f} {ratio:5.2f}  "
                      f"{cells[0]:5d}/{len(pos):<4} ({cells[0] / max(1, len(pos)) * 100:3.0f}%) "
                      f"{cells[1]:4d}/{len(qpos):<4} {cells[2]:5d}/{len(neg):<6}  "
                      f"{cells[3]:11d} {cells[4]:5d} {cells[5]:5d}{star}")
    # Of the folds the prefix term catches, how many does the power term catch anyway?
    from custom_components.ha_washdata.const import SMART_TERM_TAIL_MAX_RATIO  # noqa: PLC0415

    for margin in (0.05, 0.10):
        hit = [r for r in pos if _prefix_rule(r["cands"], r["best_dur"], margin, 0.40, 1.10)]
        both = sum(1 for r in hit if (r["power_ratio"] or 0.0) > SMART_TERM_TAIL_MAX_RATIO)
        print(f"margin {margin:.2f} pos_random catches {len(hit)}, of which the power term "
              f"at {SMART_TERM_TAIL_MAX_RATIO}x already catches {both}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quiet-cuts", action="store_true",
                    help="judge every fold after QUIET_CUT_S of ENDING quiet (#424)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--json", help="write the folds (without candidate lists) here")
    ap.add_argument("--sweep", action="store_true",
                    help="prefix term: margin/floor/ratio grid + score distributions")
    ap.add_argument("--dump", help="write every fold WITH its candidate list here")
    args = ap.parse_args(argv)

    paths = _paths(True)
    if not paths:
        print("no corpus: cycle_data/ is missing or empty", file=sys.stderr)
        return 2
    pos: list[dict] = []
    neg: list[dict] = []
    qpos: list[dict] = []
    cuts = [0, 0, 0]
    devices = 0
    for part in _map(_collect_device, [(p, args.quiet_cuts) for p in paths], args.jobs):
        devices += bool(part["neg"])
        pos += part["pos"]
        neg += part["neg"]
        qpos += part["qpos"]
        cuts = [a + b for a, b in zip(cuts, part["cuts"])]

    print("Smart-Termination split-guard threshold sweep (#364), shipped matcher, LOO")
    print(f"devices: {devices} | genuine ends: {len(neg)}")
    print(f"positives (a shorter profile is winning mid-cycle): {len(pos)}")
    print(f"negatives (genuine cycle end): {len(neg)}")
    if not pos or not neg:
        print("not enough folds to sweep - is cycle_data/ present?", file=sys.stderr)
        return 2
    _report(pos, neg, qpos, args.quiet_cuts)
    if args.sweep:
        _report_sweep(pos, neg, qpos, cuts)
    if args.dump:
        Path(args.dump).write_text(json.dumps(
            {"pos": pos, "neg": neg, "qpos": qpos, "cuts": cuts}), encoding="utf-8")
    if args.json:
        slim = {k: [{kk: vv for kk, vv in r.items() if kk != "cands"} for r in v]
                for k, v in (("pos", pos), ("neg", neg), ("qpos", qpos))}
        Path(args.json).write_text(json.dumps(slim, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    # Only as a script: a library call (the tests) must not leave logging disabled.
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
