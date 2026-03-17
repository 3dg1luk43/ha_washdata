"""Analyse real cycle data to quantify the impact of trimming trailing zero-power
readings.

For every past_cycle in every JSON file under cycle_data/ the test:

  1. Counts how many trailing samples are at or below the trim threshold.
  2. Computes the duration and energy *before* and *after* a hypothetical trim.
  3. Summarises the results per user/device and in aggregate.

Assertions are enforced for every cycle to verify the following invariants:
  - energy can only decrease or stay the same after trimming
  - duration can only decrease or stay the same after trimming
  - the trailing-zero count equals the number of samples removed
  - when trailing zeros are present, tail duration and energy delta are non-negative

Run with:
    pytest tests/test_trailing_zero_impact.py -v -s
"""

import json
import os
import glob
from dataclasses import dataclass, field

import pytest

CYCLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../cycle_data")
TRIM_THRESHOLD_W = 1.0  # matches add_cycle / trim_zero_power_data usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(power_data: list) -> list:
    """Return a list of valid [ts, power] pairs, skipping malformed rows."""
    cleaned = []
    for row in power_data:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            cleaned.append([float(row[0]), float(row[1])])
        except (TypeError, ValueError):
            continue
    return cleaned


def _trailing_zero_count(power_data: list, threshold: float) -> int:
    """Return how many samples at the *end* have power <= threshold."""
    data = _clean(power_data)
    count = 0
    for point in reversed(data):
        if point[1] <= threshold:
            count += 1
        else:
            break
    return count


def _trim_trailing(power_data: list, threshold: float) -> list:
    """Return power_data with trailing zero/near-zero samples removed."""
    data = _clean(power_data)
    for i in range(len(data) - 1, -1, -1):
        if data[i][1] > threshold:
            return data[: i + 1]
    # all zero — keep one point
    return data[:1] if data else []


def _energy_wh(power_data: list) -> float:
    """Trapezoid energy integration, returns Wh."""
    data = _clean(power_data)
    if len(data) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(data)):
        dt = (data[i][0] - data[i - 1][0]) / 3600.0  # seconds → hours
        avg_w = (data[i][1] + data[i - 1][1]) / 2.0
        total += avg_w * dt
    return total


def _load_all_cycles() -> list[dict]:
    """Yield dicts with metadata + power_data for every cycle in cycle_data/."""
    results = []
    pattern = os.path.join(CYCLE_DATA_DIR, "**", "*.json")
    for filepath in sorted(glob.glob(pattern, recursive=True)):
        rel = os.path.relpath(filepath, CYCLE_DATA_DIR)
        try:
            with open(filepath) as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        store = raw.get("data", {}).get("store_data", raw.get("store_data", raw))
        cycles = store.get("past_cycles", [])
        for idx, cycle in enumerate(cycles):
            pd = cycle.get("power_data")
            if not pd:
                continue
            results.append(
                {
                    "file": rel,
                    "index": idx,
                    "status": cycle.get("status", "unknown"),
                    "profile_name": cycle.get("profile_name"),
                    "power_data": pd,
                }
            )
    return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_trailing_zero_impact():
    """Print a per-cycle and aggregate report of trailing-zero trim impact."""
    cycles = _load_all_cycles()
    if not cycles:
        pytest.skip("No cycle data found under cycle_data/")

    @dataclass
    class Stats:
        total: int = 0
        affected: int = 0           # cycles that have ≥1 trailing zero
        samples_removed: int = 0    # total samples that would be removed
        duration_saved_s: float = 0.0
        energy_saved_wh: float = 0.0
        max_tail_s: float = 0.0     # longest single tail
        max_tail_file: str = ""
        by_status: dict = field(default_factory=dict)

    agg = Stats()

    print()
    print("=" * 78)
    print(f"{'FILE / CYCLE IDX':<42} {'STATUS':<12} {'TAIL_S':>7} {'TAIL_N':>7} {'ΔE_Wh':>8}")
    print("-" * 78)

    for c in cycles:
        pd = c["power_data"]
        status = c["status"]
        cleaned = _clean(pd)
        n_trailing = _trailing_zero_count(pd, TRIM_THRESHOLD_W)

        duration_before = cleaned[-1][0] - cleaned[0][0] if len(cleaned) > 1 else 0.0
        energy_before = _energy_wh(cleaned)

        trimmed = _trim_trailing(pd, TRIM_THRESHOLD_W)
        duration_after = trimmed[-1][0] - trimmed[0][0] if len(trimmed) > 1 else 0.0
        energy_after = _energy_wh(trimmed)

        tail_s = duration_before - duration_after
        delta_e = energy_before - energy_after

        # Invariants that must hold for every cycle
        assert energy_after <= energy_before + 1e-9, (
            f"energy increased after trim in {c['file']}[{c['index']}]"
        )
        assert duration_after <= duration_before + 1e-9, (
            f"duration increased after trim in {c['file']}[{c['index']}]"
        )
        assert n_trailing == len(cleaned) - len(trimmed), (
            f"trailing count mismatch in {c['file']}[{c['index']}]"
        )
        if n_trailing > 0:
            assert tail_s >= 0, (
                f"negative tail_s in {c['file']}[{c['index']}]"
            )
            assert delta_e >= 0, (
                f"negative energy delta in {c['file']}[{c['index']}]"
            )

        agg.total += 1
        st = agg.by_status.setdefault(status, Stats())
        st.total += 1

        if n_trailing > 0:
            agg.affected += 1
            agg.samples_removed += n_trailing
            agg.duration_saved_s += tail_s
            agg.energy_saved_wh += delta_e
            if tail_s > agg.max_tail_s:
                agg.max_tail_s = tail_s
                agg.max_tail_file = f"{c['file']}[{c['index']}]"
            st.affected += 1
            st.samples_removed += n_trailing
            st.duration_saved_s += tail_s
            st.energy_saved_wh += delta_e

        label = f"{c['file']}[{c['index']}]"
        if len(label) > 42:
            label = "…" + label[-41:]

        marker = " <<<" if n_trailing > 0 else ""
        print(
            f"{label:<42} {status:<12} {tail_s:>7.1f} {n_trailing:>7d} {delta_e:>8.4f}{marker}"
        )

    print("=" * 78)
    print()
    print("AGGREGATE SUMMARY")
    print(f"  Total cycles analysed : {agg.total}")
    print(f"  Cycles with tail zeros : {agg.affected}  ({100*agg.affected/max(agg.total,1):.1f}%)")
    print(f"  Total samples removed  : {agg.samples_removed}")
    if agg.affected:
        avg_tail = agg.duration_saved_s / agg.affected
        print(f"  Avg tail duration      : {avg_tail:.1f}s  ({avg_tail/60:.2f} min)")
        print(f"  Max tail duration      : {agg.max_tail_s:.1f}s  ({agg.max_tail_s/60:.2f} min)  ← {agg.max_tail_file}")
        print(f"  Total duration trimmed : {agg.duration_saved_s:.1f}s  ({agg.duration_saved_s/60:.1f} min)")
        print(f"  Total energy trimmed   : {agg.energy_saved_wh:.4f} Wh  (negligible by design)")
    print()
    print("BREAKDOWN BY STATUS")
    for status, st in sorted(agg.by_status.items()):
        pct = 100 * st.affected / max(st.total, 1)
        avg = st.duration_saved_s / max(st.affected, 1)
        print(
            f"  {status:<15} total={st.total:3d}  affected={st.affected:3d} ({pct:5.1f}%)"
            f"  avg_tail={avg:6.1f}s  removed_samples={st.samples_removed}"
        )
    print()
