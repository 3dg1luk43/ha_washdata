#!/usr/bin/env python3
"""analyze_diag.py - WashData diagnostic export analyser.

Reads a WashData diagnostic export (JSON) and prints a comparison of the
*current* settings stored in the file against *suggested* settings derived
from the recorded cycle data.

Usage:
    python3 devtools/analyze_diag.py <path/to/export.json>
    python3 devtools/analyze_diag.py  # interactive file prompt

The script must be run from the repository root with the venv activated:
    source .venv/bin/activate
    python3 devtools/analyze_diag.py cycle_data/.../<export>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repository root on sys.path so we can import the integration's const.py
# and the benchmark ParameterOptimizer.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import numpy as np
    from custom_components.ha_washdata.const import (
        CONF_STOP_THRESHOLD_W,
        CONF_START_THRESHOLD_W,
        CONF_END_ENERGY_THRESHOLD,
        CONF_START_ENERGY_THRESHOLD,
        CONF_RUNNING_DEAD_ZONE,
        CONF_MIN_OFF_GAP,
        CONF_OFF_DELAY,
        CONF_WATCHDOG_INTERVAL,
        CONF_NO_UPDATE_ACTIVE_TIMEOUT,
        CONF_PROFILE_MATCH_INTERVAL,
        CONF_DURATION_TOLERANCE,
        CONF_PROFILE_DURATION_TOLERANCE,
        CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
        CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
        DEFAULT_OFF_DELAY_BY_DEVICE,
        DEFAULT_OFF_DELAY,
        DEFAULT_MIN_OFF_GAP_BY_DEVICE,
        DEFAULT_MIN_OFF_GAP,
    )
    from tests.benchmarks.parameter_optimizer import DataLoader, ParameterOptimizer
except ModuleNotFoundError as exc:
    print(
        f"\n[ERROR] Could not import required modules: {exc}\n"
        "Make sure you run this script from the repository root with the venv activated:\n"
        "  source .venv/bin/activate\n"
        "  python3 devtools/analyze_diag.py <export.json>\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str: return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def cyan(t: str) -> str:   return _c(t, "36")
def bold(t: str) -> str:   return _c(t, "1")
def dim(t: str) -> str:    return _c(t, "2")
def red(t: str) -> str:    return _c(t, "31")


# ---------------------------------------------------------------------------
# Export loading
# ---------------------------------------------------------------------------

def load_export(path: str) -> dict[str, Any]:
    """Load and parse the diagnostic export JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_store(data: dict[str, Any]) -> dict[str, Any]:
    """Return the store data dict regardless of export format."""
    # New format: data -> store_export -> data
    store_export = data.get("data", {}).get("store_export", {})
    if isinstance(store_export, dict) and "data" in store_export:
        return store_export["data"]
    # Legacy format: data -> store_data
    return data.get("data", {}).get("store_data", {})


def extract_current_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Merge entry.data + entry.options into a flat settings dict."""
    entry = data.get("data", {}).get("entry", {})
    merged: dict[str, Any] = {}
    entry_present = isinstance(entry, dict) and (entry.get("data") or entry.get("options"))
    if isinstance(entry, dict):
        merged.update(entry.get("data", {}))
        merged.update(entry.get("options", {}))
    # Fallback: older store_export fields - only when entry block is absent or empty
    if not entry_present:
        store_export = data.get("data", {}).get("store_export", {})
        if isinstance(store_export, dict):
            merged.update(store_export.get("entry_data", {}))
            merged.update(store_export.get("entry_options", {}))
    return merged


def extract_existing_suggestions(data: dict[str, Any]) -> dict[str, Any]:
    """Return suggestions already stored in the export (from manager_state)."""
    return data.get("data", {}).get("manager_state", {}).get("suggestions", {})


def extract_sample_stats(data: dict[str, Any]) -> dict[str, Any]:
    """Return the sample_interval_stats dict if present."""
    return data.get("data", {}).get("manager_state", {}).get("sample_interval_stats", {})


# ---------------------------------------------------------------------------
# Suggestion computation (standalone, no HA mock needed)
# ---------------------------------------------------------------------------

def _parse_ts(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def compute_suggestions(
    cycles: list[dict[str, Any]],
    profiles: dict[str, Any],
    device_type: str | None,
    sample_stats: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run all suggestion heuristics and return {key: {value, reason}}."""

    suggestions: dict[str, dict[str, Any]] = {}
    labeled = [
        c for c in cycles
        if isinstance(c, dict) and c.get("profile_name") and c.get("power_data")
        and c.get("status") != "interrupted"
    ]

    # ------------------------------------------------------------------ #
    # 1. Power thresholds (batch multi-cycle)                              #
    # ------------------------------------------------------------------ #
    optimizer = ParameterOptimizer(labeled) if labeled else None

    if optimizer:
        stop_w = 2.0
        pt = optimizer.analyze_power_thresholds()
        if pt:
            stop_w = pt.get("suggested_stop_threshold_w", 2.0)
            suggestions[CONF_STOP_THRESHOLD_W] = {
                "value": round(stop_w, 2),
                "reason": f"5th-percentile of minimum-active-power across {len(labeled)} labeled cycles.",
            }
            suggestions[CONF_START_THRESHOLD_W] = {
                "value": round(pt.get("suggested_start_threshold_w", stop_w * 1.2), 2),
                "reason": "Hysteresis band above the stop threshold (x1.2).",
            }

        et = optimizer.analyze_energy_thresholds(stop_threshold=stop_w)
        if et:
            if "suggested_end_energy_threshold" in et:
                suggestions[CONF_END_ENERGY_THRESHOLD] = {
                    "value": round(et["suggested_end_energy_threshold"], 4),
                    "reason": "Maximum false-end pause energy (Wh) with 20% safety buffer.",
                }
            if "suggested_start_energy_threshold" in et:
                suggestions[CONF_START_ENERGY_THRESHOLD] = {
                    "value": round(et["suggested_start_energy_threshold"], 4),
                    "reason": "5th-percentile of first-60s energy (Wh) x 0.5 safety factor.",
                }

        tp = optimizer.analyze_timing_parameters()
        if tp:
            suggestions[CONF_RUNNING_DEAD_ZONE] = {
                "value": tp.get("suggested_running_dead_zone", 60),
                "reason": "75th-percentile of early (<5 min) power-dip timestamps across cycles.",
            }

    # ------------------------------------------------------------------ #
    # 2. Duration tolerance & match ratio bounds                          #
    # ------------------------------------------------------------------ #
    ratios: list[float] = []
    for c in labeled[-100:]:
        name = c.get("profile_name")
        prof = profiles.get(name) if isinstance(name, str) else None
        if not isinstance(prof, dict):
            continue
        try:
            avg = float(prof.get("avg_duration") or 0.0)
            dur = float(c.get("duration") or 0.0)
        except (TypeError, ValueError):
            continue
        if avg > 60 and dur > 60:
            ratios.append(dur / avg)

    if len(ratios) >= 10:
        arr = np.array(ratios)
        p95_dev = float(np.percentile(np.abs(arr - 1.0), 95))
        tol = round(min(0.50, max(0.10, p95_dev + 0.05)), 2)
        n = len(ratios)
        suggestions[CONF_DURATION_TOLERANCE] = {
            "value": tol,
            "reason": f"p95 duration deviation across {n} labeled cycles = {p95_dev:.2f} -> +0.05 buffer.",
        }
        suggestions[CONF_PROFILE_DURATION_TOLERANCE] = {
            "value": tol,
            "reason": f"Same as duration_tolerance (shared source: {n} cycles).",
        }
        p05 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        min_r = max(0.1, round(p05 - 0.1, 2))
        max_r = min(3.0, round(p95 + 0.1, 2))
        if min_r < max_r - 0.2:
            suggestions[CONF_PROFILE_MATCH_MIN_DURATION_RATIO] = {
                "value": min_r,
                "reason": f"p05 of actual/expected duration ratio ({p05:.2f}) - 0.10 buffer.",
            }
            suggestions[CONF_PROFILE_MATCH_MAX_DURATION_RATIO] = {
                "value": max_r,
                "reason": f"p95 of actual/expected duration ratio ({p95:.2f}) + 0.10 buffer.",
            }

    # ------------------------------------------------------------------ #
    # 3. Min-off-gap from inter-cycle gaps                                #
    # ------------------------------------------------------------------ #
    timed: list[tuple[float, float]] = []
    for c in labeled:
        s = _parse_ts(c.get("start_time"))
        e = _parse_ts(c.get("end_time"))
        if s and e and e > s:
            timed.append((s, e))
    timed.sort(key=lambda x: x[0])
    gaps = [
        timed[i][0] - timed[i - 1][1]
        for i in range(1, len(timed))
        if 30 <= timed[i][0] - timed[i - 1][1] <= 86400
    ]
    if len(gaps) >= 3:
        device_floor = (
            DEFAULT_MIN_OFF_GAP_BY_DEVICE.get(device_type, DEFAULT_MIN_OFF_GAP)
            if device_type else DEFAULT_MIN_OFF_GAP
        )
        p05_gap = float(np.percentile(gaps, 5))
        suggested_gap = int(max(device_floor, min(p05_gap * 0.8, 3600)))
        suggestions[CONF_MIN_OFF_GAP] = {
            "value": suggested_gap,
            "reason": (
                f"p05 of {len(gaps)} inter-cycle gaps ({p05_gap:.0f}s) x 0.8, "
                f"floored by device default ({device_floor}s)."
            ),
        }

    # ------------------------------------------------------------------ #
    # 4. Operational timings from sample_interval_stats or power_data     #
    # ------------------------------------------------------------------ #
    p95_dt: float | None = None
    median_dt: float | None = None

    # Try manager_state.sample_interval_stats first
    if isinstance(sample_stats, dict) and sample_stats:
        p95_dt_raw = sample_stats.get("p95") or sample_stats.get("p95_dt")
        median_raw = sample_stats.get("median") or sample_stats.get("median_dt")
        if p95_dt_raw:
            try:
                p95_dt = float(p95_dt_raw)
                median_dt = float(median_raw) if median_raw else p95_dt * 0.6
            except (TypeError, ValueError):
                pass

    # Fallback: derive from power_data timestamps across cycles
    if p95_dt is None and labeled:
        deltas: list[float] = []
        for c in labeled[:20]:  # cap at 20 cycles to keep it fast
            pd = c.get("power_data", [])
            for i in range(1, min(len(pd), 50)):
                try:
                    dt = float(pd[i][0]) - float(pd[i - 1][0])
                    if 0.5 < dt < 300:
                        deltas.append(dt)
                except (IndexError, TypeError, ValueError):
                    pass
        if len(deltas) >= 10:
            p95_dt = float(np.percentile(deltas, 95))
            median_dt = float(np.median(deltas))

    if p95_dt and median_dt:
        device_off_floor = (
            DEFAULT_OFF_DELAY_BY_DEVICE.get(device_type, DEFAULT_OFF_DELAY)
            if device_type else DEFAULT_OFF_DELAY
        )
        suggestions[CONF_WATCHDOG_INTERVAL] = {
            "value": int(max(30, p95_dt * 10)),
            "reason": f"p95 sampling interval ({p95_dt:.1f}s) x 10, minimum 30s.",
        }
        suggestions[CONF_NO_UPDATE_ACTIVE_TIMEOUT] = {
            "value": int(max(60, p95_dt * 20)),
            "reason": f"p95 sampling interval ({p95_dt:.1f}s) x 20, minimum 60s.",
        }
        suggestions[CONF_OFF_DELAY] = {
            "value": int(max(device_off_floor, p95_dt * 5)),
            "reason": (
                f"p95 sampling interval ({p95_dt:.1f}s) x 5, "
                f"floored by device default ({device_off_floor}s)."
            ),
        }
        suggestions[CONF_PROFILE_MATCH_INTERVAL] = {
            "value": int(max(10, median_dt * 10)),
            "reason": f"Median sampling interval ({median_dt:.1f}s) x 10.",
        }

    return suggestions


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

# Human-readable labels for each parameter key, grouped by section
_SECTIONS: list[tuple[str, list[str]]] = [
    ("Power Thresholds", [
        CONF_STOP_THRESHOLD_W,
        CONF_START_THRESHOLD_W,
        CONF_RUNNING_DEAD_ZONE,
    ]),
    ("Energy Gates", [
        CONF_END_ENERGY_THRESHOLD,
        CONF_START_ENERGY_THRESHOLD,
    ]),
    ("Timing & Operational", [
        CONF_WATCHDOG_INTERVAL,
        CONF_NO_UPDATE_ACTIVE_TIMEOUT,
        CONF_OFF_DELAY,
        CONF_MIN_OFF_GAP,
        CONF_PROFILE_MATCH_INTERVAL,
    ]),
    ("Matching & Learning", [
        CONF_DURATION_TOLERANCE,
        CONF_PROFILE_DURATION_TOLERANCE,
        CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
        CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    ]),
]

_LABELS: dict[str, str] = {
    CONF_STOP_THRESHOLD_W: "Stop threshold (W)",
    CONF_START_THRESHOLD_W: "Start threshold (W)",
    CONF_RUNNING_DEAD_ZONE: "Running dead zone (s)",
    CONF_END_ENERGY_THRESHOLD: "End energy gate (Wh)",
    CONF_START_ENERGY_THRESHOLD: "Start energy gate (Wh)",
    CONF_WATCHDOG_INTERVAL: "Watchdog interval (s)",
    CONF_NO_UPDATE_ACTIVE_TIMEOUT: "No-update timeout (s)",
    CONF_OFF_DELAY: "Off delay (s)",
    CONF_MIN_OFF_GAP: "Min off gap (s)",
    CONF_PROFILE_MATCH_INTERVAL: "Profile match interval (s)",
    CONF_DURATION_TOLERANCE: "Duration tolerance",
    CONF_PROFILE_DURATION_TOLERANCE: "Profile duration tolerance",
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO: "Match min duration ratio",
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO: "Match max duration ratio",
}

_UNITS: dict[str, str] = {}  # units already in labels above


def _fmt_val(v: Any, key: str) -> str:
    """Format a value for display."""
    if v is None:
        return dim("-")
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _delta_arrow(current: Any, suggested: Any) -> str:
    """Return a coloured delta indicator."""
    if current is None:
        return cyan("NEW")
    try:
        c_f, s_f = float(current), float(suggested)
        if abs(c_f - s_f) < 1e-6:
            return green("✓ OK")
        pct = abs(s_f - c_f) / max(abs(c_f), 1e-9) * 100
        arrow = "↑" if s_f > c_f else "↓"
        color = yellow if pct < 30 else red
        return color(f"{arrow} {pct:.0f}%")
    except (TypeError, ValueError):
        return yellow("≠")


def _print_section_header(title: str) -> None:
    width = 92
    print()
    print(bold(f"  {'━' * 4}  {title}  {'━' * (width - len(title) - 9)}"))


def _col(text: str, width: int, align: str = "<") -> str:
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    pad = max(0, width - len(plain))
    if align == ">":
        return " " * pad + text
    return text + " " * pad


def print_report(
    export_path: str,
    current: dict[str, Any],
    suggestions: dict[str, Any],
    stored_suggestions: dict[str, Any],
    cycles: list[dict],
    profiles: dict,
    device_type: str | None,
) -> None:
    """Render the full comparison report to stdout."""

    labeled = [c for c in cycles if c.get("profile_name")]
    with_power = [c for c in labeled if c.get("power_data")]

    print()
    print(bold("=" * 92))
    print(bold("  WashData Diagnostic Analyser"))
    print(bold("=" * 92))
    print(f"  File        : {dim(export_path)}")
    print(f"  Device type : {cyan(device_type or 'unknown')}")
    print(f"  Cycles      : {len(cycles)} total, {len(labeled)} labeled, {len(with_power)} with power data")
    print(f"  Profiles    : {len(profiles)}")
    if profiles:
        names = ", ".join(list(profiles.keys())[:6])
        if len(profiles) > 6:
            names += f"  … +{len(profiles) - 6} more"
        print(f"  Programs    : {dim(names)}")
    print()

    # Column widths: parameter | current | suggested | delta | rationale
    W_PARAM = 32
    W_VAL   = 12
    W_DELTA = 8

    header = (
        bold(_col("  Parameter", W_PARAM))
        + bold(_col("Current", W_VAL, ">"))
        + bold(_col("Suggested", W_VAL, ">"))
        + bold(_col("Change", W_DELTA + 2, ">"))
        + bold("  Rationale")
    )
    divider = dim("  " + "─" * (W_PARAM + W_VAL + W_VAL + W_DELTA + 40))

    for section_name, keys in _SECTIONS:
        section_suggestions = {k: suggestions[k] for k in keys if k in suggestions}
        if not section_suggestions:
            continue

        _print_section_header(section_name)
        print(header)
        print(divider)

        for key in keys:
            if key not in suggestions:
                continue
            label = _LABELS.get(key, key)
            current_val = current.get(key)
            sugg = suggestions[key]
            suggested_val = sugg["value"]
            reason = sugg["reason"]

            # Wrap rationale to fit on one line (truncate)
            max_reason = 65
            if len(reason) > max_reason:
                reason = reason[:max_reason - 1] + "…"

            c_str = _col(_fmt_val(current_val, key), W_VAL, ">")
            s_str = _col(green(_fmt_val(suggested_val, key)), W_VAL, ">")
            d_str = _col(_delta_arrow(current_val, suggested_val), W_DELTA + 2, ">")

            print(
                f"  {_col(label, W_PARAM - 2)}"
                f"{c_str}"
                f"{s_str}"
                f"{d_str}"
                f"  {dim(reason)}"
            )

    # ------------------------------------------------------------------ #
    # Already-stored suggestions (from previous online learning)          #
    # ------------------------------------------------------------------ #
    if stored_suggestions:
        _print_section_header("Previously Stored Suggestions (from live operation)")
        print(header)
        print(divider)
        for key, data in sorted(stored_suggestions.items()):
            if not isinstance(data, dict):
                continue
            label = _LABELS.get(key, key)
            current_val = current.get(key)
            stored_val = data.get("value")
            reason = data.get("reason", "")
            updated = data.get("updated", "")
            if updated:
                try:
                    ts = datetime.fromisoformat(updated)
                    updated = ts.strftime("%Y-%m-%d")
                except ValueError:
                    pass
            max_reason = 50
            if len(reason) > max_reason:
                reason = reason[:max_reason - 1] + "…"

            c_str = _col(_fmt_val(current_val, key), W_VAL, ">")
            s_str = _col(yellow(_fmt_val(stored_val, key)), W_VAL, ">")
            d_str = _col(_delta_arrow(current_val, stored_val), W_DELTA + 2, ">")
            tag = dim(f"[{updated}]") if updated else ""

            print(
                f"  {_col(label, W_PARAM - 2)}"
                f"{c_str}"
                f"{s_str}"
                f"{d_str}"
                f"  {dim(reason)} {tag}"
            )

    # ------------------------------------------------------------------ #
    # Summary / action prompt                                              #
    # ------------------------------------------------------------------ #
    def _values_equal(a: Any, b: Any) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return a == b

    actionable_keys = [
        k for k in suggestions
        if not _values_equal(suggestions[k].get("value"), current.get(k))
    ]
    print()
    print(bold("=" * 92))
    if actionable_keys:
        n = len(actionable_keys)
        print(
            f"  {green('✔')}  {bold(str(n))} parameter{'s' if n != 1 else ''} can be improved.\n"
            f"     To apply: Settings → Devices & Services → WashData → Configure\n"
            f"               → Advanced Settings → Apply Suggested Values"
        )
    else:
        print(f"  {green('✔')}  All analysed parameters look well-tuned for this device.")
    print(bold("=" * 92))
    print()


# ---------------------------------------------------------------------------
# Profile summary
# ---------------------------------------------------------------------------

def print_profile_summary(profiles: dict[str, Any], cycles: list[dict[str, Any]]) -> None:
    """Print a brief profile / cycle history table."""
    from collections import defaultdict
    import statistics

    if not profiles:
        return

    profile_cycles: dict[str, list[float]] = defaultdict(list)
    for c in cycles:
        name = c.get("profile_name")
        dur = c.get("duration")
        if name and dur:
            try:
                profile_cycles[name].append(float(dur))
            except (TypeError, ValueError):
                pass

    _print_section_header("Cycle History - Profiles")
    header = (
        bold(f"  {'Program':<35}")
        + bold(f"{'Cycles':>7}")
        + bold(f"{'Avg (min)':>10}")
        + bold(f"{'SD (min)':>9}")
        + bold(f"{'CV (%)':>9}")
    )
    print(header)
    print(dim("  " + "─" * 75))

    for name in sorted(profile_cycles, key=lambda n: -len(profile_cycles[n])):
        durs = profile_cycles[name]
        avg = statistics.mean(durs)
        std = statistics.stdev(durs) if len(durs) > 1 else 0.0
        cv = std / avg if avg else 0.0
        cv_str = f"{cv:.1%}"
        cv_col = green(cv_str) if cv < 0.10 else (yellow(cv_str) if cv < 0.25 else red(cv_str))
        print(
            f"  {name:<35}{len(durs):>7}"
            f"{avg/60:>10.1f}"
            f"{std/60:>9.1f}"
            f"    {cv_col}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse a WashData diagnostic export and suggest optimal settings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "export",
        nargs="?",
        metavar="EXPORT_JSON",
        help="Path to the diagnostic export JSON file.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output.",
    )
    args = parser.parse_args()

    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False

    export_path: str | None = args.export

    if not export_path:
        # Interactive prompt
        print("Enter the path to the diagnostic export JSON file:")
        export_path = input("  > ").strip().strip("'\"")

    if not export_path or not os.path.isfile(export_path):
        print(f"[ERROR] File not found: {export_path!r}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading {export_path} …")
    raw = load_export(export_path)
    store = extract_store(raw)
    current = extract_current_settings(raw)
    stored_suggestions = extract_existing_suggestions(raw)
    sample_stats = extract_sample_stats(raw)

    cycles: list[dict] = store.get("past_cycles", [])
    profiles: dict = store.get("profiles", {})
    device_type: str | None = current.get("device_type")

    # Compute fresh suggestions from cycle data
    suggestions = compute_suggestions(cycles, profiles, device_type, sample_stats)

    print_report(
        export_path=export_path,
        current=current,
        suggestions=suggestions,
        stored_suggestions=stored_suggestions,
        cycles=cycles,
        profiles=profiles,
        device_type=device_type,
    )

    if cycles:
        print_profile_summary(profiles, cycles)
        print()


if __name__ == "__main__":
    main()
