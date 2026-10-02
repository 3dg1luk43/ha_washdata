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

Run from the repo root.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from custom_components.ha_washdata import playground  # noqa: E402
from custom_components.ha_washdata.cycle_detector import (  # noqa: E402
    CycleDetectorConfig,
)
from custom_components.ha_washdata.profile_store import ProfileStore  # noqa: E402
from custom_components.ha_washdata.suggestion_engine import (  # noqa: E402
    _cycle_readings,
)

#: A cycle must carry at least this many readings to be worth replaying.
MIN_READINGS = 10
#: Exports with fewer stored cycles than this cannot build usable profiles.
MIN_CYCLES = 5


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


def _production(doc: dict[str, Any], data: dict[str, Any]) -> tuple[CycleDetectorConfig, ProfileStore, dict[str, Any]]:
    """(detector config, ProfileStore, options) exactly as the manager builds them."""
    from custom_components.ha_washdata.manager import WashDataManager  # noqa: PLC0415

    entry_data = {"power_sensor": "sensor.end_gate_eval", "name": "eval",
                  **{k: v for k, v in (doc.get("entry_data") or {}).items() if v is not None}}
    opts = {k: v for k, v in (doc.get("entry_options") or {}).items() if v is not None}
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    if device_type:
        opts.setdefault("device_type", device_type)
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


def _measure_export(path: Path, no_shortening: bool, loo: bool = False) -> list[dict[str, Any]]:
    """Replay every usable cycle in one export; one row per cycle."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    data = doc.get("data") or {}
    cycles = data.get("past_cycles") or []
    if not device_type or len(cycles) < MIN_CYCLES:
        return []
    base = dict(data)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        base[key] = list(base.get(key) or [])
    base["profiles"] = {k: dict(v) if isinstance(v, dict) else v
                        for k, v in (base.get("profiles") or {}).items()}
    base["envelopes"] = dict(base.get("envelopes") or {})
    cfg, store, opts = _production(doc, base)
    stop = float(cfg.stop_threshold_w)
    # Exports carry the envelopes the exporting version built; rebuild them with
    # the code under test, as the live store would after an upgrade.
    _rebuild_envelopes(store, list(base["profiles"]))
    try:
        prebuilt = playground._build_match_snapshots(store)  # noqa: SLF001
    except Exception:
        prebuilt = None
    cycles = base["past_cycles"]
    rows: list[dict[str, Any]] = []
    for cyc in cycles:
        pts = _cycle_readings(cyc)
        if len(pts) < MIN_READINGS:
            continue
        span = _active_span(pts, stop)
        if span <= 0:
            continue
        fold_store, fold_prebuilt = store, prebuilt
        name = cyc.get("profile_name")
        if loo and name and name in base["profiles"]:
            _cfg_f, fold_store, _o = _production(doc, _fold_data(base, cyc))
            _rebuild_envelopes(fold_store, [name])
            try:
                fold_prebuilt = playground._build_match_snapshots(fold_store)  # noqa: SLF001
            except Exception:
                fold_prebuilt = None
        try:
            sim = playground.simulate_cycle_detail(
                cyc, cfg, None, fold_store, opts, price=None,
                compute_series=False, prebuilt=fold_prebuilt,
            )
        except Exception:
            continue
        if "error" in sim:
            continue
        out = sim.get("outcome") or {}
        final = out.get("final_duration_s")
        end_t = _end_offset(sim.get("events") or [])
        rows.append({
            "export": str(path.relative_to(REPO)),
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
        })
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write per-cycle rows here for --compare")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument(
        "--no-shortening", action="store_true",
        help="pre-306 arm: patch END_GATE_LATE_RATIO out of reach",
    )
    ap.add_argument(
        "--loo", action="store_true",
        help="leave-one-out: match each cycle against profiles rebuilt without it",
    )
    args = ap.parse_args()

    if args.compare:
        _compare(*args.compare)
        return 0

    if args.no_shortening:
        # Patch `const`, not `cycle_detector`. Since register item 355 the gate
        # calls `resolve_end_gate_late_ratio(device_type)`, which reads these two
        # names out of `const` at call time - rebinding the detector module's
        # imported copy no longer reaches it, and this arm would silently stop
        # disabling the shortening while still reporting itself as the pre-306
        # baseline. Both names, because the per-device map wins for washers.
        from custom_components.ha_washdata import const as _const

        _const.END_GATE_LATE_RATIO = 1e9
        _const.END_GATE_LATE_RATIO_BY_DEVICE = {}

    logging.getLogger("custom_components.ha_washdata").setLevel(logging.ERROR)
    rows: list[dict[str, Any]] = []
    for path in sorted((REPO / "cycle_data").rglob("*.json")):
        rows.extend(_measure_export(path, args.no_shortening, args.loo))

    if not rows:
        print("no replayable cycles found - is cycle_data/ present?")
        return 1

    _print_summary(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {len(rows)} rows to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
