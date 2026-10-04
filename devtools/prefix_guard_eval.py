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

    python3 devtools/prefix_guard_eval.py [--quiet-cuts] [--jobs N] [--json FILE]

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

  prefix     the #364 prefix term (``profile_store._match_prefix_ambiguity``,
             the flag ``MatchResult.is_prefix_ambiguous`` carries) - does a LONGER
             candidate's curve, truncated to the elapsed duration, explain the
             trace better than the winner does, by SMART_TERM_PREFIX_MARGIN?
  power      is the trailing mean power more than SMART_TERM_TAIL_MAX_RATIO x what
             the matched profile draws at its own end
             (``ProfileStore.profile_tail_power``)?

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
so the cycle is neither template nor pause evidence for itself). The flag read is
the one the detector gets; the margin sweep re-runs the same production rule on
the full post-collapse candidate list that call handed ``match_prefix_flags``,
with ``SMART_TERM_PREFIX_MARGIN`` rebound. The stop threshold is the production
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

and reports the production prefix flag both without and with the pause evidence
(``SMART_TERM_PREFIX_MIN_PAUSE_S``; ``profile_pauses_below`` on the fold store).
Quote these populations. Without ``--quiet-cuts`` a negative is judged at its
last stored reading, and a trace stored up to its last activity still shows
running power there, which the detector never meets in ENDING: on 2026-10-04 the
power term false-blocked 116/574 (20%) such ends at 3.5x and 0/675 with the
quiet. Measured that day (71 devices, LOO, ``--quiet-cuts``): the prefix term
fired on 0 of 713 genuine ends, on 3 of 528 random-cut positives without the
pause evidence and 0 with it, and on 0 of 7 quiet positives; the power term
caught 156/508 (31%) positives at 3.5x with no false block.

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
_CAND_KEYS = ("name", "profile_duration", "shape_score", "score", "prefix_score")

#: The last ``match_prefix_flags`` call's inputs (one fold at a time per process).
_LAST: dict[str, Any] = {}


@contextlib.contextmanager
def _record_prefix_inputs() -> Iterator[None]:
    """Capture what ``async_match_profile`` hands ``match_prefix_flags``.

    A pass-through wrapper: the candidates are the full post-collapse list (not
    the ``[:5]`` the result carries) and ``best_duration`` the one the flag used.
    """
    from custom_components.ha_washdata import profile_store as ps  # noqa: PLC0415

    orig = ps.match_prefix_flags

    def wrapped(candidates: list[dict], best_duration: float, pauses_below: Any = None):
        _LAST.update(cands=list(candidates), best_dur=float(best_duration or 0.0),
                     pauses=pauses_below)
        return orig(candidates, best_duration, pauses_below)

    ps.match_prefix_flags = wrapped
    try:
        yield
    finally:
        ps.match_prefix_flags = orig


def _prefix_fires(cands: list[dict], best_dur: float, margin: float) -> bool:
    """The production prefix term (no pause evidence) at an arbitrary margin."""
    from custom_components.ha_washdata import profile_store as ps  # noqa: PLC0415

    saved = ps.SMART_TERM_PREFIX_MARGIN
    ps.SMART_TERM_PREFIX_MARGIN = margin
    try:
        return bool(ps._match_prefix_ambiguity(cands, best_dur)[1])  # noqa: SLF001
    finally:
        ps.SMART_TERM_PREFIX_MARGIN = saved


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
    from custom_components.ha_washdata import profile_store as ps  # noqa: PLC0415

    cands = [{k: c.get(k) for k in _CAND_KEYS} for c in _LAST["cands"]]
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
        "landscape": bool(ps._match_prefix_ambiguity(cands, best_dur)[1]),  # noqa: SLF001
        "landscape_paused": bool(result.is_prefix_ambiguous),
        "power_ratio": (trailing / tail) if (tail and trailing is not None and tail > 0) else None,
    }


def _collect_device(job: tuple[str, bool]) -> dict[str, list[dict]]:
    """The positive / negative / quiet-positive folds of one device."""
    path, quiet_cuts = job
    out: dict[str, list[dict]] = {"pos": [], "neg": [], "qpos": []}
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
                    if pos is None or pos["top1"] == name or pos["best_dur"] >= dur * 0.95:
                        continue
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
        SMART_TERM_PREFIX_MARGIN,
        SMART_TERM_PREFIX_MIN_PAUSE_S,
        SMART_TERM_TAIL_MAX_RATIO,
    )

    print("\n=== prefix term: margin over the winner (production rule, no pause evidence) ===")
    print(f"{'margin':>7} {'caught':>16} {'false blocks':>16}")
    for m in MARGIN_SWEEP:
        c = sum(1 for r in pos if _prefix_fires(r["cands"], r["best_dur"], m))
        f = sum(1 for r in neg if _prefix_fires(r["cands"], r["best_dur"], m))
        star = "  <- shipped" if abs(m - SMART_TERM_PREFIX_MARGIN) < 1e-9 else ""
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

    print("\n=== both guards, at the shipped constants (prefix flag as the detector gets it) ===")

    def _either(r: dict) -> bool:
        return r["landscape_paused"] or (
            r["power_ratio"] is not None and r["power_ratio"] > SMART_TERM_TAIL_MAX_RATIO
        )

    c = sum(1 for r in pos if _either(r))
    f = sum(1 for r in neg if _either(r))
    print(f"caught {c}/{len(pos)} ({c/len(pos)*100:.0f}%) | "
          f"false blocks {f}/{len(neg)} ({f/len(neg)*100:.0f}%)")
    print("\nA false block costs a later finish (the power-based fallback still ends")
    print("the cycle). A miss costs a split cycle. Prefer the conservative side.")

    if not quiet_cuts:
        return
    print("\n=== --quiet-cuts: production prefix flag, without / with pause evidence ===")
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quiet-cuts", action="store_true",
                    help="judge every fold after QUIET_CUT_S of ENDING quiet (#424)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--json", help="write the folds (without candidate lists) here")
    args = ap.parse_args(argv)

    paths = _paths(True)
    if not paths:
        print("no corpus: cycle_data/ is missing or empty", file=sys.stderr)
        return 2
    pos: list[dict] = []
    neg: list[dict] = []
    qpos: list[dict] = []
    devices = 0
    for part in _map(_collect_device, [(p, args.quiet_cuts) for p in paths], args.jobs):
        devices += bool(part["neg"])
        pos += part["pos"]
        neg += part["neg"]
        qpos += part["qpos"]

    print("Smart-Termination split-guard threshold sweep (#364), shipped matcher, LOO")
    print(f"devices: {devices} | genuine ends: {len(neg)}")
    print(f"positives (a shorter profile is winning mid-cycle): {len(pos)}")
    print(f"negatives (genuine cycle end): {len(neg)}")
    if not pos or not neg:
        print("not enough folds to sweep - is cycle_data/ present?", file=sys.stderr)
        return 2
    _report(pos, neg, qpos, args.quiet_cuts)
    if args.json:
        slim = {k: [{kk: vv for kk, vv in r.items() if kk != "cands"} for r in v]
                for k, v in (("pos", pos), ("neg", neg), ("qpos", qpos))}
        Path(args.json).write_text(json.dumps(slim, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    # Only as a script: a library call (the tests) must not leave logging disabled.
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
