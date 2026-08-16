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

Usage:  python3 devtools/min_off_gap_eval.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from custom_components.ha_washdata.cycle_detector import (  # noqa: E402
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import (  # noqa: E402
    DEFAULT_MIN_OFF_GAP,
    DEFAULT_MIN_OFF_GAP_BY_DEVICE,
)
from custom_components.ha_washdata.suggestion_engine import (  # noqa: E402
    _CLEAN_ACTIVE_FLOOR_RATIO,
    _cycle_readings,
    _parse_ts,
    _resumed_low_runs,
    select_clean_cycles,
)

#: Candidate lower-bound statistics over the measured bridged spans.
CANDIDATES: dict[str, callable] = {
    "p95_all": lambda spans, _sig: float(np.percentile(spans, 95)),
    "p95_significant": lambda _spans, sig: float(np.percentile(sig, 95)) if sig else 0.0,
    "p99_significant": lambda _spans, sig: float(np.percentile(sig, 99)) if sig else 0.0,
    "max": lambda spans, _sig: float(max(spans)),
}

BUFFER_S = 60.0
#: A bridged span shorter than this is sampling jitter, not a phase gap.
SIGNIFICANT_S = 60.0


def _cfg(opts: dict, device_type: str, min_off_gap: int) -> CycleDetectorConfig:
    """Detector config from an export's own options, overriding min_off_gap."""
    g = opts.get
    return CycleDetectorConfig(
        min_power=float(g("min_power", 2.0)),
        off_delay=int(g("off_delay", 180)),
        device_type=device_type,
        completion_min_seconds=int(g("completion_min_seconds", 600)),
        start_duration_threshold=float(g("start_duration_threshold", 5.0)),
        start_energy_threshold=float(g("start_energy_threshold", 0.2)),
        end_energy_threshold=float(g("end_energy_threshold", 0.05)),
        end_repeat_count=int(g("end_repeat_count", 1)),
        min_off_gap=min_off_gap,
        start_threshold_w=float(g("start_threshold_w", 2.0)),
        stop_threshold_w=float(g("stop_threshold_w", 2.0)),
        power_off_threshold_w=float(g("power_off_threshold_w", 0.0)),
        power_off_delay=float(g("power_off_delay", 30.0)),
        anti_wrinkle_enabled=bool(g("anti_wrinkle_enabled", False)),
        delay_detect_enabled=bool(g("delay_start_detect_enabled", False)),
    )


def _replay(cfg: CycleDetectorConfig, segments, tail_s: float = 7200.0) -> list[dict]:
    """Feed (offset, power) segments through a real detector; return ended cycles.

    ``segments`` is already a flat, monotonically increasing trace. A long quiet
    tail is appended so any cycle still open at the end is allowed to finalize.
    """
    ended: list[dict] = []
    det = CycleDetector(
        config=cfg,
        on_state_change=lambda _o, _n: None,
        on_cycle_end=lambda d: ended.append(d),
        profile_matcher=None,
    )
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for off, pw in segments:
        det.process_reading(float(pw), base + timedelta(seconds=float(off)))
    last = segments[-1][0]
    step = 30.0
    for i in range(1, int(tail_s / step) + 1):
        det.process_reading(0.0, base + timedelta(seconds=last + i * step))
    return ended


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


def _shipped_suggestion(doc, device_type, clean, raw, stop) -> int | None:
    """Whatever `SuggestionEngine._suggest_min_off_gap` proposes for this export."""
    from unittest.mock import MagicMock

    from custom_components.ha_washdata.suggestion_engine import SuggestionEngine

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


def main() -> int:
    exports = sorted((REPO / "cycle_data").rglob("*.json"))
    totals = {
        name: {"split": 0, "merge": 0, "cycles": 0}
        for name in list(CANDIDATES) + ["SHIPPED"]
    }

    for path in exports:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        device_type = (doc.get("device_fingerprint") or {}).get("device_type")
        cycles = ((doc.get("data") or {}).get("past_cycles")) or []
        if not device_type or len(cycles) < 5:
            continue
        opts = doc.get("entry_options") or {}
        stop = float(opts.get("stop_threshold_w") or 1.0)
        clean, _ = select_clean_cycles(cycles, stop_threshold_w=stop)
        traces = [c for c in clean if len(_cycle_readings(c)) >= 10]
        spans, traced = _bridged_spans(clean, stop)
        if traced < 5 or len(spans) < 3 or not traces:
            continue
        significant = [s for s in spans if s > SIGNIFICANT_S]
        gaps = _real_gaps(cycles)
        floor = DEFAULT_MIN_OFF_GAP_BY_DEVICE.get(device_type, DEFAULT_MIN_OFF_GAP)

        print(f"\n=== {path.relative_to(REPO)}")
        print(
            f"    {device_type}  traced={traced}  spans={len(spans)} "
            f"(>{SIGNIFICANT_S:.0f}s: {len(significant)})  "
            f"blind floor={floor}s  shortest real gap="
            f"{min(gaps):.0f}s" if gaps else f"    {device_type}  traced={traced}"
        )

        # The value the shipped heuristic actually proposes, evaluated alongside
        # the raw candidates so the regression is validated, not just the choice.
        shipped = _shipped_suggestion(doc, device_type, clean, cycles, stop)
        candidates = dict(CANDIDATES)
        if shipped is not None:
            candidates["SHIPPED"] = lambda _s, _g, v=shipped: float(v) - BUFFER_S
        else:
            print("      SHIPPED          -> (suppressed)")

        for name, fn in candidates.items():
            value = int(max(DEFAULT_MIN_OFF_GAP, round(fn(spans, significant) + BUFFER_S)))
            cfg_split = _cfg(opts, device_type, value)
            splits = 0
            split_ids: list[str] = []
            for c in traces:
                pts = _cycle_readings(c)
                if _is_split(_replay(cfg_split, pts), pts, stop):
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
                if len(_replay(cfg_split, joined)) < 2:
                    merges = 1
            totals[name]["split"] += splits
            totals[name]["merge"] += merges
            totals[name]["cycles"] += len(traces)
            ids = (" " + ",".join(split_ids)) if split_ids else ""
            merge_txt = ("MERGED" if merges else "ok") if gaps else "n/a"
            print(
                f"      {name:16} -> {value:>6}s   splits {splits}/{len(traces)}{ids}"
                f"   merge probe: {merge_txt}"
            )

    print("\n=== totals (lower is better; splits are disqualifying) ===")
    for name, t in totals.items():
        print(
            f"  {name:16} splits {t['split']:>3}/{t['cycles']:<4} "
            f"merges {t['merge']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
