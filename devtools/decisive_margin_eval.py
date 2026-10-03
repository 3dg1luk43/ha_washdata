#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How often does the mid-cycle decisive-margin bypass fire with no runner-up?

Register item 305 replaced a `confidence > 0.8` mid-cycle switch override with
one keyed on the top1-vs-top2 margin, measured at end-of-cycle correctness
70.4% -> 72.6% for 0.14 displayed switches per cycle.  Round 6 of the PR #448
review pointed out that `match_margin` keeps a **1.0 sentinel** when no other
candidate scored, so the bypass could fire on a margin that was never computed.

The full switching replay behind item 305 is not checked in, but the question
the fix raises does not need it: the change is inert unless the bypass actually
fires with `_runner_up is None`.  This counts that, on the real corpus, at the
same mid-cycle checkpoints production matches at.

For each labelled cycle it runs the REAL matcher over a prefix at each
checkpoint and reproduces the manager's own runner-up arithmetic
(`match_rules.begin_tick`, which `manager._async_do_perform_matching` calls),
reporting:

  * checkpoints with a single scored candidate (where the sentinel applies)
  * of those, how many would have taken the bypass before the fix
  * how many would still take it after

The match is ``ProfileStore.async_match_profile`` itself (audit PLAYGROUND-04),
on a store built by a real ``WashDataManager`` from the export's options
(``end_gate_eval._production``): ``energy_mode`` is what the manager sets
(integrated energy for washers), candidate templates are re-gridded to each
query's step, and the runner-up is read from the same post-collapse,
Stage-5-relabelled ``candidates`` the manager reads. Until 0.5.8 it called the
worker directly with mean-power energy, 5 s templates whatever the plug's
cadence, and a group win named ``__group__...``, so its figures predate that.

Run from the repo root:  python3 devtools/decisive_margin_eval.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "devtools"))

from end_gate_eval import _production  # noqa: E402

from custom_components.ha_washdata.const import MATCH_DECISIVE_MARGIN  # noqa: E402
from custom_components.ha_washdata.suggestion_engine import _cycle_readings  # noqa: E402

#: Elapsed fractions to probe, mirroring a 5 min match interval over a wash.
CHECKPOINTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MIN_READINGS = 10
MIN_CYCLES = 5


def _runner_up_of(candidates: list[dict[str, Any]], best: str | None) -> float | None:
    """The manager's own arithmetic: best OTHER candidate's score, or None.

    Measured against the best other candidate rather than by list index, because
    Stage-5 collapsing rebuilds the result and `best_profile` is not guaranteed
    to be `candidates[0]`.
    """
    runner_up: float | None = None
    for c in candidates:
        if c.get("name") == best:
            continue
        try:
            score = float(c.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if runner_up is None or score > runner_up:
            runner_up = score
    return runner_up


def _scan_export(path: Path) -> Counter:
    tally: Counter = Counter()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return tally
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    data = doc.get("data") or {}
    cycles = data.get("past_cycles") or []
    if not device_type or len(cycles) < MIN_CYCLES:
        return tally
    try:
        _cfg, store, _opts = _production(doc, data)
    except Exception:
        return tally
    if not store.has_real_profiles:
        return tally
    loop = asyncio.new_event_loop()

    for cyc in cycles:
        label = cyc.get("profile_name")
        pts = _cycle_readings(cyc)
        if len(pts) < MIN_READINGS:
            continue
        total = pts[-1][0] - pts[0][0]
        if total <= 0:
            continue
        for frac in CHECKPOINTS:
            cut = pts[0][0] + frac * total
            prefix = [(t, p) for t, p in pts if t <= cut]
            if len(prefix) < MIN_READINGS:
                continue
            duration = prefix[-1][0] - prefix[0][0]
            try:
                result = loop.run_until_complete(
                    store.async_match_profile(prefix, duration, in_progress=frac < 1.0)
                )
            except Exception:
                tally["error"] += 1
                continue
            cands = list(result.candidates or [])
            if not result.best_profile or not cands:
                tally["unmatched"] += 1
                continue
            best_name = result.best_profile
            confidence = float(result.confidence or 0.0)
            tally["checkpoints"] += 1
            runner_up = _runner_up_of(list(cands), best_name)
            correct = bool(label) and str(best_name).strip() == str(label).strip()
            if runner_up is None:
                tally["single_candidate"] += 1
                if label:
                    tally["single_labelled"] += 1
                    tally["single_correct"] += int(correct)
                # The sentinel path: margin 1.0 clears the threshold outright, and
                # `current_program_score` is 0.0 whenever the displayed program is
                # not among the candidates - which is the case the finding is about.
                if confidence > 0.0:
                    tally["sentinel_bypass_before_fix"] += 1
            else:
                if confidence - runner_up > MATCH_DECISIVE_MARGIN:
                    tally["real_margin_bypass"] += 1
                    if label:
                        tally["real_margin_labelled"] += 1
                        tally["real_margin_correct"] += int(correct)
    loop.close()
    return tally


def main() -> int:
    logging.disable(logging.CRITICAL)
    total: Counter = Counter()
    for path in sorted((REPO / "cycle_data").rglob("*.json")):
        total.update(_scan_export(path))

    cps = total["checkpoints"]
    if not cps:
        print("no matched checkpoints - is cycle_data/ present?")
        return 1

    print(f"matched checkpoints                 : {cps}")
    print(f"unmatched checkpoints (skipped)     : {total['unmatched']}")
    print(f"single scored candidate (sentinel)  : {total['single_candidate']}"
          f"  ({100.0 * total['single_candidate'] / cps:.2f}%)")
    print(f"  ...would bypass before the fix    : {total['sentinel_bypass_before_fix']}")
    print(f"  ...bypass after the fix           : 0  (by construction)")
    print(f"bypass on a REAL margin (unchanged) : {total['real_margin_bypass']}"
          f"  ({100.0 * total['real_margin_bypass'] / cps:.2f}%)")

    print("\nWas the switch the bypass would have made the RIGHT one?")
    for kind in ("single", "real_margin"):
        n = total[f"{kind}_labelled"]
        if not n:
            continue
        ok = total[f"{kind}_correct"]
        name = (
            "sentinel (no runner-up)" if kind == "single"
            else "real margin > threshold"
        )
        print(f"  {name:<28}: {ok}/{n} correct  ({100.0 * ok / n:.1f}%)")
    if total["error"]:
        print(f"matcher errors                      : {total['error']}")
    print(
        "\nThe fix is inert unless 'would bypass before the fix' is non-zero:"
        "\nevery other checkpoint takes the identical branch either way."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
