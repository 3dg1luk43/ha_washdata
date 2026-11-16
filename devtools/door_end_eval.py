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
"""door_end_eval.py - where a door held open in PAUSED may end the cycle (#524).

    python3 devtools/door_end_eval.py [--root cycle_data] [--json OUT]

With "Opening the Door Ends the Cycle" on, a door held open in PAUSED ends the
cycle only once ``CycleDetector.door_end_due`` says the run has reached its
matched programme's end (``DOOR_END_PAUSED_MIN_RATIO``, per device type); earlier
it is a pause, as before 0.6.1. This drives that shipped gate on a detector set
up as the matched run, over the recorded corpus, per device type:

* **ends reached**: completed cycles that ended on their own (not by hand, not
  force-stopped) where the gate is open at their stored end, with the expected
  duration the mean of the OTHER cycles of their programme (leave-one-out, at
  least 2 others). An end it misses is one where a door opened at the end still
  pauses.
* **spells exposed**: quiet spells inside those cycles (under the Stop
  Threshold, activity after them) where a door opened while the detector is in
  PAUSED, and held for the dwell (``--dwell``, the default 60 s) inside the
  spell, finds the gate open: that door would end the run mid-programme. Only
  the PAUSED part of a spell counts: from ``_dynamic_pause_threshold`` to
  ``_dynamic_end_threshold`` into it, sized from the cadence of the 20 intervals
  before it as ``_gate_cadence`` sizes them; past it the detector is in ENDING,
  where a held door ended the cycle before 0.6.1 too. The share of spells, and
  of cycles with any.

Rows per ratio patch the constants to one ratio for every device type; the
``shipped`` row runs them as they are. What this does not replay: the matcher
(the expected duration is the leave-one-out mean, not the live gate's, and the
match is taken as unambiguous) and the door timing itself (opened at the latest
moment the spell allows).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "devtools"))

RATIOS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NATURAL = ("timeout", "smart", "terminal_drop", "door", None, "")


def _stop_threshold(dev: Any) -> tuple[str, float]:
    from custom_components.ha_washdata.const import CONF_STOP_THRESHOLD_W  # noqa: PLC0415
    from custom_components.ha_washdata.detector_config import (  # noqa: PLC0415
        effective_option_values,
    )

    merged = {**(dev.entry_data or {}), **(dev.entry_options or {})}
    device_type = str(merged.get("device_type") or "washing_machine")
    eff = effective_option_values(merged, device_type)
    return device_type, float(eff.get(CONF_STOP_THRESHOLD_W) or 0.0)


def _cadence(dts: list[float]) -> float:
    """``CycleDetector._gate_cadence`` over these intervals (the last 20)."""
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        GATE_CADENCE_MEDIAN_FACTOR,
    )
    from custom_components.ha_washdata.signal_processing import (  # noqa: PLC0415
        median_fast,
        percentile_linear,
    )

    recent = [d for d in dts if d > 0.1][-20:]
    if len(recent) < 5:
        return max(recent[-1], 1.0) if recent else 1.0
    return min(percentile_linear(recent, 95), GATE_CADENCE_MEDIAN_FACTOR * median_fast(recent))


def _paused_door_moments(pts: list[list[float]], stop: float, dwell: float) -> list[float]:
    """Per quiet spell followed by activity, the latest moment a door can open in
    its PAUSED part and stay open for ``dwell`` inside the spell (the gate is
    monotonic in time, so the latest moment decides). Spells with no such moment
    are left out."""
    out: list[float] = []
    start: float | None = None
    dts: list[float] = []
    for i, (t, p) in enumerate(pts):
        if i:
            dts.append(t - pts[i - 1][0])
        if p < stop:
            if start is None:
                start = t
                cad = _cadence(dts[:-1])
                pause_thr = max(15.0, 3.0 * cad)
                end_thr = max(3.0 * cad, pause_thr + 15.0)
        else:
            if start is not None:
                latest = min(start + end_thr, t - dwell)
                if latest >= start + pause_thr:
                    out.append(latest)
            start = None
    return out


def _gate(device_type: str) -> Any:
    """A detector to ask ``door_end_due``, matched and unambiguous."""
    from custom_components.ha_washdata.cycle_detector import (  # noqa: PLC0415
        CycleDetector,
        CycleDetectorConfig,
    )

    det = CycleDetector(
        CycleDetectorConfig(min_power=1.0, off_delay=120, device_type=device_type),
        lambda _old, _new: None,
        lambda _c: None,
    )
    det._matched_profile = "eval"  # noqa: SLF001
    det._match_ambiguous = False  # noqa: SLF001
    return det


def _due(det: Any, expected: float, at_s: float) -> bool:
    det._expected_duration = float(expected)  # noqa: SLF001
    det._current_cycle_start = _T0  # noqa: SLF001
    return bool(det.door_end_due(_T0 + timedelta(seconds=at_s)))


@contextmanager
def _ratio(r: float | None) -> Any:
    """Every device type at ratio ``r``; None keeps the shipped constants."""
    from custom_components.ha_washdata import cycle_detector as cd  # noqa: PLC0415

    saved = (cd.DOOR_END_PAUSED_MIN_RATIO, cd.DOOR_END_PAUSED_MIN_RATIO_BY_TYPE)
    if r is not None:
        cd.DOOR_END_PAUSED_MIN_RATIO, cd.DOOR_END_PAUSED_MIN_RATIO_BY_TYPE = r, {}
    try:
        yield
    finally:
        cd.DOOR_END_PAUSED_MIN_RATIO, cd.DOOR_END_PAUSED_MIN_RATIO_BY_TYPE = saved


def measure(root: Path, dwell: float) -> dict[str, Any]:
    import eval as ev  # noqa: PLC0415

    devices, _dropped = ev.load_corpus(root)
    # (expected, duration, spell ends) per cycle
    rows: dict[str, list[tuple[float, float, list[float]]]] = defaultdict(list)
    for dev in devices:
        device_type, stop = _stop_threshold(dev)
        if stop <= 0:
            continue
        by_profile: dict[str, list[tuple[dict, float]]] = defaultdict(list)
        for c, evidence in dev.cycles:
            if evidence != "past" or c.get("status") != "completed":
                continue
            if c.get("termination_reason") not in _NATURAL:
                continue
            pts = ev.query_points(c)
            dur = ev.cycle_duration(c, pts)
            if dur > 0:
                by_profile[str(c["profile_name"])].append((c, dur))
        for members in by_profile.values():
            if len(members) < 3:
                continue
            total = sum(d for _c, d in members)
            for c, dur in members:
                expected = (total - dur) / (len(members) - 1)
                if expected <= 0:
                    continue
                pts = ev.query_points(c)
                t0 = pts[0][0] if pts else 0.0
                rel = [[t - t0, p] for t, p in pts]
                rows[device_type].append((expected, dur, _paused_door_moments(rel, stop, dwell)))
    report: dict[str, Any] = {}
    for device_type, items in sorted(rows.items()):
        det = _gate(device_type)
        n_spells = sum(len(s) for _e, _d, s in items)
        per_ratio = {}
        for r in (None, *RATIOS):
            with _ratio(r):
                reached = sum(1 for e, d, _s in items if _due(det, e, d))
                exposed = [sum(1 for x in s if _due(det, e, x)) for e, _d, s in items]
            per_ratio["shipped" if r is None else str(r)] = {
                "ends_reached": round(100.0 * reached / len(items), 1),
                "spells_exposed": round(100.0 * sum(exposed) / n_spells, 1) if n_spells else 0.0,
                "cycles_with_spell_exposed": round(
                    100.0 * sum(1 for n in exposed if n) / len(items), 1
                ),
            }
        report[device_type] = {"cycles": len(items), "spells": n_spells, "ratios": per_ratio}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=str(_REPO_ROOT / "cycle_data"))
    parser.add_argument("--dwell", type=float, default=60.0,
                        help="door-open dwell, s (default 60, the option's default)")
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args(argv)
    report = measure(Path(args.root), args.dwell)
    for device_type, rep in report.items():
        print(f"\n{device_type}: {rep['cycles']} cycles, {rep['spells']} mid-cycle quiet spells "
              "with a PAUSED door moment")
        print(f"  {'ratio':>8}{'ends reached %':>16}{'spells exposed %':>18}"
              f"{'cycles w/ exposed spell %':>27}")
        for r, m in rep["ratios"].items():
            print(f"  {r:>8}{m['ends_reached']:>16}{m['spells_exposed']:>18}"
                  f"{m['cycles_with_spell_exposed']:>27}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
