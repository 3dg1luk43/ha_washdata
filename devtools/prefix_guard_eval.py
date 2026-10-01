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

``dtw_ab_eval.py`` answers "did matching accuracy regress?" - it only ever scores
COMPLETE cycles, so it cannot say whether these guards discriminate. This harness
answers that, by rebuilding the two populations that matter:

  POSITIVE ("would split")   a long cycle's trace TRUNCATED to the point where a
                             SHORTER profile is winning the match. This is the
                             #364 failure: Smart Termination fires at 0.98x that
                             shorter profile's duration and the remainder becomes
                             a second cycle. The guard SHOULD fire here.
  NEGATIVE ("genuine end")   a cycle at its own true end, matched to its own
                             profile. The guard MUST NOT fire: every false fire
                             is a legitimate cycle pushed onto the slower
                             power-based fallback timeout.

Two independent guards are swept:

  prefix     _match_prefix_ambiguity's prefix term - does a LONGER candidate's
             curve, truncated to the elapsed duration, explain the trace better
             than the winner does, by SMART_TERM_PREFIX_MARGIN?
  power      is the trailing mean power more than SMART_TERM_TAIL_MAX_RATIO x what
             the matched profile draws at its own end?

Both are shorten-only: firing can only ever BLOCK an early finish, never end a
cycle sooner. That asymmetry is why the false-block column is a cost (a later
finish) and the missed column is the bug (a split cycle), and why the operating
point sits well clear of the false-block knee.

**Grid (register item 303, fixed here for #424).** This harness used to hand the
matcher the RAW trace as the query against templates resampled to 5 s, the same
index-by-index mismatch item 303 found in ``dtw_ab_eval.py``. Every fold now puts
the query on ``resample_adaptive``'s grid and re-grids each template to that same
``dt``, exactly as production does, and carries ``sample_span_s`` so the prefix
truncation sees a real span. Figures taken before this fix are void.

**``--quiet-cuts`` (#424): the populations Smart Termination actually meets.** It
only runs in ENDING, after the power has sat below ``stop_threshold_w``, so:

  NEGATIVE      the whole cycle plus ``QUIET_CUT_S`` of trailing quiet.
  POS_RANDOM    the fixed-fraction cuts above - kept for comparison, but a cut in
                the middle of activity can never reach ENDING, so it is not a
                split the guards could prevent.
  POS_QUIET     ``QUIET_CUT_S`` into a mid-cycle pause below the threshold that
                power later RESUMED from, with a shorter programme winning and the
                run already at >= 0.9x that programme's duration: the moment a
                split can really happen.

and reports the production ``_match_prefix_ambiguity`` both without and with the
pause evidence (``SMART_TERM_PREFIX_MIN_PAUSE_S``, leave-one-out: the held-out
cycle never counts as evidence for its own profile). The stop threshold is each
export's own ``entry_options.stop_threshold_w``; a source without one gets no
pause evidence, so the guard is kept there.

Run from the repo root:  python3 devtools/prefix_guard_eval.py [--quiet-cuts]
cycle_data/ is gitignored, so this is maintainer-local.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

from custom_components.ha_washdata import analysis  # noqa: E402
from custom_components.ha_washdata.profile_store import (  # noqa: E402
    _match_prefix_ambiguity,
)
from custom_components.ha_washdata.signal_processing import (  # noqa: E402
    has_resumed_pause,
    resample_uniform,
)
from custom_components.ha_washdata.const import (  # noqa: E402
    SMART_TERM_PREFIX_MIN_PAUSE_S,
    SMART_TERM_PREFIX_MARGIN,
    SMART_TERM_PREFIX_MIN_RATIO,
    SMART_TERM_PREFIX_MIN_SHAPE,
    SMART_TERM_TAIL_MAX_RATIO,
    SMART_TERM_TAIL_WINDOW_FRAC,
    SMART_TERM_TAIL_WINDOW_MIN_S,
    SMART_TERM_TAIL_WINDOW_S,
)
from dtw_ab_eval import (  # noqa: E402
    _BASE_CFG,
    _group_by_source,
    _prep_cycles,
    _query_grid,
)

# Where along a long cycle the shorter profile can win. 0.98 is the ratio both
# Smart-Termination paths fire at, so the cut points bracket it.
CUT_FRACTIONS = (0.55, 0.65, 0.75, 0.85)
MARGIN_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.30)
RATIO_SWEEP = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
# --quiet-cuts: how long the trace has been quiet when it is judged. Roughly what
# ENDING entry plus the Smart-Termination debounce take on the shipped defaults.
QUIET_CUT_S = 300.0


def _match_config() -> dict:
    from custom_components.ha_washdata.const import (
        DEFAULT_DTW_BANDWIDTH,
        DEFAULT_DTW_MODE,
    )

    cfg = {**_BASE_CFG, "dtw_bandwidth": DEFAULT_DTW_BANDWIDTH, "dtw_mode": DEFAULT_DTW_MODE}
    # --in-progress: score the mid-cycle populations the way the LIVE matcher now
    # does (#400), so the guard operating point can be read under the ranking it
    # will actually see in production rather than under the pre-#400 one.
    if "--in-progress" in sys.argv or "--quiet-cuts" in sys.argv:
        cfg["in_progress"] = True
        # Shipped default also scores SHAPE on the truncated template (#400);
        # --no-prefix-shape reads the operating point without it.
        if "--no-prefix-shape" in sys.argv:
            cfg["prefix_shape"] = False
    return cfg


def _tail_level(sample: list[float], frac: float = SMART_TERM_TAIL_WINDOW_FRAC) -> float | None:
    """Mean of the last ``frac`` of a profile curve - the hass-free equivalent of
    ProfileStore.profile_tail_power (the curve is uniform over its own span)."""
    arr = np.asarray(sample, dtype=float)
    if arr.size < 4:
        return None
    k = max(1, int(round(arr.size * frac)))
    return float(arr[-k:].mean())


def _tail_window_s(expected: float) -> float:
    """Mirror of CycleDetector._tail_window_s: the trailing window must cover the
    same fraction of the run as the profile tail it is compared against."""
    if expected <= 0:
        return SMART_TERM_TAIL_WINDOW_S
    return min(
        SMART_TERM_TAIL_WINDOW_S,
        max(SMART_TERM_TAIL_WINDOW_MIN_S, expected * SMART_TERM_TAIL_WINDOW_FRAC),
    )


def _trailing_mean(powers: list[float], duration: float, window_s: float) -> float | None:
    """Mean power over the trailing ``window_s`` of a uniformly-sampled trace."""
    arr = np.asarray(powers, dtype=float)
    if arr.size < 3 or duration <= 0:
        return None
    k = max(3, int(round(arr.size * min(1.0, window_s / duration))))
    return float(arr[-k:].mean())


def _prefix_fires(cands: list[dict], best_dur: float, margin: float) -> bool:
    """The prefix term at an arbitrary margin (production uses the constant)."""
    if best_dur <= 0 or len(cands) < 2:
        return False
    # Winner's shape score (same scale as prefix_score); blended score only as fallback.
    _best_shape = cands[0].get("shape_score")
    best_score = float(
        (_best_shape if _best_shape is not None else cands[0].get("score")) or 0.0
    )
    for c in cands[1:]:
        ps = c.get("prefix_score")
        if ps is None:
            continue
        if (
            float(c.get("profile_duration") or 0) > best_dur * SMART_TERM_PREFIX_MIN_RATIO
            and float(ps) >= SMART_TERM_PREFIX_MIN_SHAPE
            and float(ps) >= best_score + margin
        ):
            return True
    return False


def _snapshots(by_profile: dict, exclude_key: tuple, dt: float) -> list[dict]:
    """One snapshot per profile on the QUERY's grid (item 303), with its span.

    The template is the training cycle closest to the profile mean, as in
    ``dtw_ab_eval._build_snapshots``; ``sample_span_s`` is added because the prefix
    truncation reads it and production snapshots carry it.
    """
    snaps = []
    for name, cycles in by_profile.items():
        pool = [
            c for idx, c in enumerate(cycles)
            if (name, idx) != exclude_key and c.get("_ts") is not None and c.get("_pw")
        ]
        if not pool:
            continue
        avg = float(np.mean([c["_dur"] for c in pool]))
        rep = min(pool, key=lambda c: abs(c["_dur"] - avg))
        segs = resample_uniform(
            rep["_ts"], np.asarray(rep["_pw"], dtype=float), dt_s=dt, gap_s=21600.0
        )
        if not segs:
            continue
        seg = max(segs, key=lambda s: len(s.power))
        if len(seg.power) < 2:
            continue
        snaps.append({
            "name": name, "avg_duration": avg, "sample_power": seg.power.tolist(),
            "sample_span_s": len(seg.power) * dt,
        })
    return snaps


def _stop_threshold(src: str, _cache: dict = {}) -> float | None:  # noqa: B006
    """The export's own stop threshold, or None when it does not record one."""
    if src not in _cache:
        try:
            with open(src, encoding="utf-8") as fh:
                raw = (json.load(fh).get("entry_options") or {}).get("stop_threshold_w")
            _cache[src] = float(raw) if raw is not None else None
        except Exception:  # noqa: BLE001
            _cache[src] = None
    return _cache[src]


def _pause_table(by_profile: dict, stop: float | None) -> dict[str, list[bool]]:
    """Per profile, per cycle: has this cycle paused below ``stop`` and resumed?"""
    if stop is None:
        return {}
    out: dict[str, list[bool]] = {}
    for name, cycles in by_profile.items():
        out[name] = [
            bool(c.get("_ts") is not None and has_resumed_pause(
                list(zip((float(t) for t in c["_ts"]), (float(p) for p in c["_pw"]))),
                stop, SMART_TERM_PREFIX_MIN_PAUSE_S,
            ))
            for c in cycles
        ]
    return out


def _collect(by_source: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Build the positive (would-split) and negative (genuine-end) folds."""
    cfg = _match_config()
    quiet_cuts = "--quiet-cuts" in sys.argv
    positives: list[dict] = []
    negatives: list[dict] = []
    quiet_pos: list[dict] = []

    for src, by_profile in by_source.items():
        if len(by_profile) < 2:
            continue
        stop = _stop_threshold(src)
        pauses = _pause_table(by_profile, stop)
        for name, cycles in by_profile.items():
            for idx, cyc in enumerate(cycles):
                q = _query_grid(cyc)
                if q is None:
                    continue
                powers, dt = q
                dur = cyc.get("_dur") or 0.0
                if len(powers) < 40 or dur <= 0 or dt <= 0:
                    continue
                # Leave-one-out: the cycle under test never trains its own profile,
                # and never counts as pause evidence for it either.
                snaps = _snapshots(by_profile, (name, idx), dt)
                if len(snaps) < 2:
                    continue

                def _paused(cand: str, _name=name, _idx=idx) -> bool | None:
                    vals = list(pauses.get(cand) or [])
                    if cand == _name and _idx < len(vals):
                        vals.pop(_idx)
                    return any(vals) if vals else None

                def _fold(pw: list[float], d: float) -> dict | None:
                    cands = analysis.compute_matches_worker(pw, d, snaps, cfg)
                    if not cands:
                        return None
                    best = cands[0]
                    tail = _tail_level(best.get("sample") or [])
                    expected = float(best.get("profile_duration") or 0.0)
                    trailing = _trailing_mean(pw, d, _tail_window_s(expected))
                    return {
                        "src": src,
                        "true": name,
                        "top1": best.get("name"),
                        "best_dur": float(best.get("profile_duration") or 0.0),
                        "cands": cands,
                        "landscape": any(_match_prefix_ambiguity(cands, expected)),
                        "landscape_paused": any(
                            _match_prefix_ambiguity(cands, expected, _paused)
                        ),
                        "power_ratio": (trailing / tail) if (tail and trailing is not None and tail > 0) else None,
                    }

                # NEGATIVE: the whole cycle, at its own end (+ the ENDING quiet).
                n_quiet = int(round(QUIET_CUT_S / dt)) if quiet_cuts else 0
                neg = _fold(powers + [0.0] * n_quiet, dur + n_quiet * dt)
                if neg is not None:
                    negatives.append(neg)

                # POSITIVE: truncated to where a SHORTER profile is winning.
                for f in CUT_FRACTIONS:
                    cut = int(len(powers) * f)
                    if cut < 30:
                        continue
                    pos = _fold(powers[:cut], dur * f)
                    if pos is None:
                        continue
                    # The split only happens when a shorter profile actually wins.
                    if pos["top1"] == name or pos["best_dur"] >= dur * 0.95:
                        continue
                    positives.append(pos)

                if not quiet_cuts or stop is None:
                    continue
                # POS_QUIET: QUIET_CUT_S into a resumed mid-cycle pause.
                arr = np.asarray(powers, dtype=float)
                i = 0
                while i < arr.size:
                    if arr[i] >= stop:
                        i += 1
                        continue
                    j = i
                    while j < arr.size and arr[j] < stop:
                        j += 1
                    resumed = j < arr.size
                    if resumed and (j - i) >= n_quiet and i * dt >= 0.2 * dur:
                        cut = i + n_quiet
                        pos = _fold(powers[:cut], cut * dt)
                        if (
                            pos is not None
                            and pos["top1"] != name
                            and pos["best_dur"] < dur * 0.95
                            and cut * dt >= 0.9 * pos["best_dur"]
                        ):
                            quiet_pos.append(pos)
                    i = j
    return positives, negatives, quiet_pos


def main() -> None:
    from tests.benchmarks.parameter_optimizer import DataLoader

    root = Path(__file__).resolve().parent.parent
    loader = DataLoader([str(root / "cycle_data")])
    loader.load_data()
    real = [c for c in loader.cycles if c.get("profile_name") and c.get("power_data")]
    if not real:
        print("(no labelled real cycles found in cycle_data/)")
        return
    by_source = _group_by_source(real)
    _prep_cycles(by_source)

    print("Smart-Termination split-guard threshold sweep (#364)")
    print(f"labelled real cycles: {len(real)} | devices: {len(by_source)}")
    pos, neg, qpos = _collect(by_source)
    print(f"positives (a shorter profile is winning mid-cycle): {len(pos)}")
    print(f"negatives (genuine cycle end, own profile winning): {len(neg)}")
    if not pos or not neg:
        print("(not enough folds to sweep)")
        return

    print("\n=== prefix term: margin over the winner ===")
    print(f"{'margin':>7} {'caught':>16} {'false blocks':>16}")
    for m in MARGIN_SWEEP:
        c = sum(1 for r in pos if _prefix_fires(r["cands"], r["best_dur"], m))
        f = sum(1 for r in neg if _prefix_fires(r["cands"], r["best_dur"], m))
        star = "  <- shipped" if abs(m - SMART_TERM_PREFIX_MARGIN) < 1e-9 else ""
        print(f"{m:7.2f} {c:6d}/{len(pos)} ({c/len(pos)*100:3.0f}%) {f:6d}/{len(neg)} ({f/len(neg)*100:3.0f}%){star}")

    print("\n=== power term: trailing mean vs the matched profile's own tail ===")
    pr_pos = [r for r in pos if r["power_ratio"] is not None]
    pr_neg = [r for r in neg if r["power_ratio"] is not None]
    print(f"{'ratio':>7} {'caught':>16} {'false blocks':>16}")
    for x in RATIO_SWEEP:
        c = sum(1 for r in pr_pos if r["power_ratio"] > x)
        f = sum(1 for r in pr_neg if r["power_ratio"] > x)
        star = "  <- shipped" if abs(x - SMART_TERM_TAIL_MAX_RATIO) < 1e-9 else ""
        print(f"{x:7.1f} {c:6d}/{len(pr_pos)} ({c/len(pr_pos)*100:3.0f}%) {f:6d}/{len(pr_neg)} ({f/len(pr_neg)*100:3.0f}%){star}")

    print("\n=== both guards, at the shipped constants ===")
    def _either(r: dict) -> bool:
        return _prefix_fires(r["cands"], r["best_dur"], SMART_TERM_PREFIX_MARGIN) or (
            r["power_ratio"] is not None and r["power_ratio"] > SMART_TERM_TAIL_MAX_RATIO
        )
    c = sum(1 for r in pos if _either(r))
    f = sum(1 for r in neg if _either(r))
    print(f"caught {c}/{len(pos)} ({c/len(pos)*100:.0f}%) | false blocks {f}/{len(neg)} ({f/len(neg)*100:.0f}%)")
    print("\nA false block costs a later finish (the power-based fallback still ends")
    print("the cycle). A miss costs a split cycle. Prefer the conservative side.")

    if "--quiet-cuts" not in sys.argv:
        return
    print("\n=== --quiet-cuts: production prefix guard, without / with pause evidence ===")
    print(f"pause evidence: SMART_TERM_PREFIX_MIN_PAUSE_S = {SMART_TERM_PREFIX_MIN_PAUSE_S:.0f} s")
    for label, rows in (("negatives", neg), ("pos_random", pos), ("pos_quiet", qpos)):
        if not rows:
            print(f"{label:<11} n=0")
            continue
        def _pct(key: str, _rows=rows) -> str:
            k = sum(1 for r in _rows if r[key])
            return f"{k}/{len(_rows)} ({k / len(_rows) * 100:.1f}%)"
        print(f"{label:<11} n={len(rows):<4} fires: without {_pct('landscape')}  with {_pct('landscape_paused')}")


if __name__ == "__main__":
    main()
