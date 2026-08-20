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

Run from the repo root:  python3 devtools/prefix_guard_eval.py
cycle_data/ is gitignored, so this is maintainer-local.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_components.ha_washdata import analysis  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
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
    _build_snapshots,
    _group_by_source,
    _prep_cycles,
)

# Where along a long cycle the shorter profile can win. 0.98 is the ratio both
# Smart-Termination paths fire at, so the cut points bracket it.
CUT_FRACTIONS = (0.55, 0.65, 0.75, 0.85)
MARGIN_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.30)
RATIO_SWEEP = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0)


def _match_config() -> dict:
    from custom_components.ha_washdata.const import (
        DEFAULT_DTW_BANDWIDTH,
        DEFAULT_DTW_MODE,
    )

    return {**_BASE_CFG, "dtw_bandwidth": DEFAULT_DTW_BANDWIDTH, "dtw_mode": DEFAULT_DTW_MODE}


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
    best_score = float(cands[0].get("score") or 0.0)
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


def _collect(by_source: dict) -> tuple[list[dict], list[dict]]:
    """Build the positive (would-split) and negative (genuine-end) folds."""
    cfg = _match_config()
    positives: list[dict] = []
    negatives: list[dict] = []

    for src, by_profile in by_source.items():
        if len(by_profile) < 2:
            continue
        for name, cycles in by_profile.items():
            for idx, cyc in enumerate(cycles):
                powers, dur = cyc.get("_pw") or [], cyc.get("_dur") or 0.0
                if len(powers) < 40 or dur <= 0:
                    continue
                # Leave-one-out: the cycle under test never trains its own profile.
                snaps = _build_snapshots(by_profile, (name, idx))
                if len(snaps) < 2:
                    continue

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
                        "power_ratio": (trailing / tail) if (tail and trailing is not None and tail > 0) else None,
                    }

                # NEGATIVE: the whole cycle, at its own end.
                neg = _fold(powers, dur)
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
    return positives, negatives


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
    pos, neg = _collect(by_source)
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


if __name__ == "__main__":
    main()
