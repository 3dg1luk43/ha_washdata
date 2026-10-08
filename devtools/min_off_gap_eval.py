#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pick the `min_off_gap` statistic by replay, not by taste.

`min_off_gap` is bounded from two sides and the bounds are measurable:

* **must not split** - it has to outlast the longest quiet span *inside* a cycle
  that is followed by more of that same cycle (a dishwasher's passive-drying gap
  before the terminal pump-out; a washer's soak).
* **must not merge** - it has to stay below the shortest gap the user leaves
  between two separate loads, or the next load is absorbed into the previous
  cycle record (a high reading in ENDING revives the same cycle).

The lower bound is a percentile over "bridged spans", and which percentile is
correct is *not* obvious: on dishwashers every candidate agrees (~2078 s), but on
washers they spread 4x because thousands of sampling-jitter dips dilute the
percentile.  This harness replays every trace in ``cycle_data/`` through the real
``CycleDetector`` at each candidate and counts actual splits and merges, so the
statistic is chosen on evidence.

Replays are deliberately run **unmatched** (no profile matcher): a confident
match closes the cycle via Smart Termination long before `min_off_gap` is
consulted, so the matched path would not exercise the bound at all.

**Production config and keepalives (audit TESTING-08 / MATCH-EVAL-09).** Each
candidate value is replayed with the detector config ``build_detector_config``
builds from the export's own entry data and options with ``min_off_gap``
overridden (what a settings change does), so every other gate runs at the value
the device really runs, device-resolved defaults included. Until 0.5.8 a
hand-rolled ``_cfg`` passed 14 options through and filled the missing ones with
its own defaults: start/stop thresholds 2.0 / 2.0 W on every device (shipped 3.0 /
1.2 W), a dishwasher's completion minimum 600 s (900 s) and Smart Termination
ratio 0.98 (0.99), a dryer's start gates 5 s / 0.2 Wh (30 s / 0.5 Wh), and every
option it did not list at the dataclass default. Every replay goes through the
Playground replay (``playground._DetailSim``, unmatched), which emulates the live
watchdog's keepalives inside silent stretches at the device's
``watchdog_interval``: the old bare-detector loop saw a silence as one interval at
the next real reading, so no end gate was evaluated inside it. The stop threshold
is the production config's. ``--all-formats`` reads diagnostics dumps too (the
``devtools/eval.py`` corpus loader, clones dropped); by default only the export
format is read, as before. Figures taken before this rewrite (the suggestion
engine's "152 clean cycles" note) measured the old loop.

Usage:  python3 devtools/min_off_gap_eval.py [--all-formats] [--jobs N]

Exit codes: 0 ok, 2 no usable export in cycle_data/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "devtools"))

#: Candidate lower-bound statistics over the measured bridged spans.
CANDIDATES: dict[str, Callable[[list[float], list[float]], float]] = {
    "p95_all": lambda spans, _sig: float(np.percentile(spans, 95)),
    "p95_significant": lambda _spans, sig: float(np.percentile(sig, 95)) if sig else 0.0,
    "p99_significant": lambda _spans, sig: float(np.percentile(sig, 99)) if sig else 0.0,
    "max": lambda spans, _sig: float(max(spans)),
}

BUFFER_S = 60.0
#: A bridged span shorter than this is sampling jitter, not a phase gap.
SIGNIFICANT_S = 60.0


def _cfg(entry_data: dict, options: dict, device_type: str, min_off_gap: int | None) -> Any:
    """The production detector config for these options, ``min_off_gap`` overridden."""
    from custom_components.ha_washdata.const import CONF_MIN_OFF_GAP  # noqa: PLC0415
    from custom_components.ha_washdata.detector_config import (  # noqa: PLC0415
        build_detector_config,
    )

    opts = dict(options)
    if min_off_gap is not None:
        opts[CONF_MIN_OFF_GAP] = int(min_off_gap)
    return build_detector_config(opts, entry_data, device_type)


def _replay(cfg: Any, options: dict, points: list[tuple[float, float]]) -> list[dict]:
    """Feed (offset, power) points through the unmatched Playground replay; return
    the cycles it ended (the real detector, live keepalives, the synthetic tail)."""
    from custom_components.ha_washdata import playground  # noqa: PLC0415

    cycle = {
        "id": "min-off-gap-eval",
        "start_time": "2026-01-01T12:00:00+00:00",
        "power_data": [[float(t), float(p)] for t, p in points],
    }
    sim = playground._DetailSim(  # noqa: SLF001
        cycle, cfg, None, None, options, None, compute_series=False
    )
    if not sim.ready:
        return []
    sim.step(0, sim.n_readings)
    sim.run_tail()
    return list(sim.captured)


def _active_span(points: list[tuple[float, float]], stop: float) -> float:
    """Seconds from first to last above-stop reading (what a whole cycle covers)."""
    active = [t for t, p in points if p > stop]
    return (active[-1] - active[0]) if len(active) >= 2 else 0.0


def _is_split(ended: list[dict], points, stop: float) -> bool:
    """True when the trace did not survive as ONE cycle covering its active span.

    Counting ``len(ended) > 1`` alone is not enough: ``completion_min_seconds``
    silently discards a split fragment that is too short, so a truncated cycle
    would otherwise score as clean. Require a single emitted cycle that still
    covers ~all of the source trace's active span.
    """
    if len(ended) != 1:
        return True
    span = _active_span(points, stop)
    if span <= 0:
        return False
    return float(ended[0].get("duration") or 0.0) < 0.9 * span


def _bridged_spans(clean: list[dict], stop: float) -> tuple[list[float], int]:
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        _CLEAN_ACTIVE_FLOOR_RATIO,
        _cycle_readings,
        _resumed_low_runs,
    )

    spans: list[float] = []
    traced = 0
    for c in clean:
        pts = _cycle_readings(c)
        if len(pts) < 10:
            continue
        peak = max((p for _, p in pts), default=0.0)
        if peak <= 0:
            continue
        traced += 1
        thr = max(stop, _CLEAN_ACTIVE_FLOOR_RATIO * peak)
        for low_start, resume_idx in _resumed_low_runs(
            pts, thr, 3600.0, min_resume_active_s=0.0
        ):
            spans.append(pts[resume_idx][0] - low_start)
    return spans, traced


def _real_gaps(cycles: list[dict]) -> list[float]:
    from custom_components.ha_washdata.suggestion_engine import _parse_ts  # noqa: PLC0415

    timed = []
    for c in cycles:
        if c.get("status") not in ("completed", "force_stopped"):
            continue
        s, e = _parse_ts(c.get("start_time")), _parse_ts(c.get("end_time"))
        if s and e and e > s:
            timed.append((s, e))
    timed.sort()
    return [
        timed[i][0] - timed[i - 1][1]
        for i in range(1, len(timed))
        if 30 <= timed[i][0] - timed[i - 1][1] <= 86400
    ]


def _shipped_suggestion(device_type, clean, raw, stop) -> int | None:
    """Whatever `SuggestionEngine._suggest_min_off_gap` proposes for this export."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        SuggestionEngine,
    )

    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = None
    store = MagicMock()
    store.get_past_cycles.return_value = raw
    store.get_profiles.return_value = {}
    store.get_suggestions.return_value = {}
    eng = SuggestionEngine(hass, "eval", store, device_type=device_type)
    out = eng._suggest_min_off_gap(  # pylint: disable=protected-access
        clean, stop_threshold_w=stop, gap_cycles=raw
    )
    return int(out["value"]) if out else None


def _scan(job: tuple[str, bool]) -> tuple[list[str], dict[str, dict[str, int]]] | None:
    """One export: (report lines, per-candidate totals), or None when unusable."""
    path_s, all_formats = job
    import end_gate_eval  # noqa: PLC0415
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        DEFAULT_MIN_OFF_GAP,
        resolve_min_off_gap_default,
    )
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        _cycle_readings,
        select_clean_cycles,
    )

    end_gate_eval._integration()  # noqa: SLF001
    path = Path(path_s)
    doc = end_gate_eval._load_doc(path, all_formats)  # noqa: SLF001
    if doc is None:
        return None
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    cycles = ((doc.get("data") or {}).get("past_cycles")) or []
    if not device_type or len(cycles) < 5:
        return None
    entry_data = {k: v for k, v in (doc.get("entry_data") or {}).items() if v is not None}
    options = {k: v for k, v in (doc.get("entry_options") or {}).items() if v is not None}
    options.setdefault("device_type", device_type)
    replay_opts = {**entry_data, **options}
    stop = float(_cfg(entry_data, options, device_type, None).stop_threshold_w)
    clean, _ = select_clean_cycles(cycles, stop_threshold_w=stop)
    traces = [c for c in clean if len(_cycle_readings(c)) >= 10]
    spans, traced = _bridged_spans(clean, stop)
    if traced < 5 or len(spans) < 3 or not traces:
        return None
    significant = [s for s in spans if s > SIGNIFICANT_S]
    gaps = _real_gaps(cycles)
    floor = resolve_min_off_gap_default(device_type)

    from decisive_margin_eval import _key  # noqa: PLC0415

    lines = [f"\n=== {_key(path)}"]
    lines.append(
        f"    {device_type}  traced={traced}  spans={len(spans)} "
        f"(>{SIGNIFICANT_S:.0f}s: {len(significant)})  stop={stop:g} W  "
        f"device default={floor}s  shortest real gap="
        + (f"{min(gaps):.0f}s" if gaps else "n/a")
    )

    # The value the shipped heuristic actually proposes, evaluated alongside
    # the raw candidates so the regression is validated, not just the choice.
    shipped = _shipped_suggestion(device_type, clean, cycles, stop)
    candidates = dict(CANDIDATES)
    if shipped is not None:
        candidates["SHIPPED"] = lambda _s, _g, v=shipped: float(v) - BUFFER_S
    else:
        lines.append("      SHIPPED          -> (suppressed)")

    totals: dict[str, dict[str, int]] = {}
    for name, fn in candidates.items():
        value = int(max(DEFAULT_MIN_OFF_GAP, round(fn(spans, significant) + BUFFER_S)))
        cfg_split = _cfg(entry_data, options, device_type, value)
        splits = 0
        split_ids: list[str] = []
        for c in traces:
            pts = _cycle_readings(c)
            if _is_split(_replay(cfg_split, replay_opts, pts), pts, stop):
                splits += 1
                split_ids.append(str(c.get("id"))[:12])
        # Merge probe: replay a cycle, hold quiet for the user's shortest real
        # inter-load gap, then replay it again. Two cycles must come out.
        merges = 0
        if gaps:
            probe_gap = min(gaps)
            a = _cycle_readings(traces[0])
            end = a[-1][0]
            joined = list(a)
            joined += [(end + probe_gap + t, p) for t, p in a]
            if len(_replay(cfg_split, replay_opts, joined)) < 2:
                merges = 1
        totals[name] = {"split": splits, "merge": merges, "cycles": len(traces)}
        ids = (" " + ",".join(split_ids)) if split_ids else ""
        merge_txt = ("MERGED" if merges else "ok") if gaps else "n/a"
        lines.append(
            f"      {name:16} -> {value:>6}s   splits {splits}/{len(traces)}{ids}"
            f"   merge probe: {merge_txt}"
        )
    return lines, totals


def _paths(all_formats: bool) -> list[str]:
    corpus = REPO / "cycle_data"
    if not corpus.is_dir():
        return []
    if all_formats:
        import end_gate_eval  # noqa: PLC0415

        devices, _clones = end_gate_eval._corpus_module().load_corpus(corpus)  # noqa: SLF001
        return [str(corpus / dev.path) for dev in devices]
    return [str(p) for p in sorted(corpus.rglob("*.json"))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--all-formats", action="store_true",
                    help="read every corpus shape (diagnostics dumps too), clones dropped")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    args = ap.parse_args(argv)

    jobs = [(p, args.all_formats) for p in _paths(args.all_formats)]
    if args.jobs > 1:
        from decisive_margin_eval import _silence_logging  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_silence_logging) as pool:
            results = list(pool.map(_scan, jobs))
    else:
        results = [_scan(j) for j in jobs]
    results = [r for r in results if r is not None]
    if not results:
        print("no usable export in cycle_data/ (>= 5 cycles, >= 5 clean traces)",
              file=sys.stderr)
        return 2

    totals = {
        name: {"split": 0, "merge": 0, "cycles": 0}
        for name in list(CANDIDATES) + ["SHIPPED"]
    }
    for lines, part in results:
        print("\n".join(lines))
        for name, t in part.items():
            for key, val in t.items():
                totals[name][key] += val

    print("\n=== totals (lower is better; splits are disqualifying) ===")
    for name, t in totals.items():
        print(
            f"  {name:16} splits {t['split']:>3}/{t['cycles']:<4} "
            f"merges {t['merge']}"
        )
    return 0


if __name__ == "__main__":
    # Only as a script: a library call (the tests) must not leave logging disabled.
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
