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
cycle only once the run has reached ``DOOR_END_PAUSED_MIN_RATIO`` of its matched
programme's expected duration; earlier it is a pause, as before 0.6.1. This
measures that bar on the recorded corpus, per device type and ratio:

* **ends reached**: completed cycles that ended on their own (not by hand, not
  force-stopped) whose duration is at least ``ratio`` x the mean of the OTHER
  cycles of their programme (leave-one-out, at least 2 others). An end below
  the bar is one where a door opened at the end still pauses.
* **pauses past the bar**: quiet spells inside those cycles (at least
  ``--min-quiet`` s under the Stop Threshold, activity after them) that begin at
  or past ``ratio`` x that mean: where a door held open would end the run
  mid-programme. The share of spells, and of cycles with any.

The duration is the stored one; live, the bar reads programme time against the
gate's expected duration, which is close but not identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "devtools"))

RATIOS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
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


def _quiet_spells(pts: list[list[float]], stop: float, min_quiet: float) -> list[float]:
    """Start offsets of quiet spells followed by activity."""
    out: list[float] = []
    start: float | None = None
    for t, p in pts:
        if p < stop:
            if start is None:
                start = t
        else:
            if start is not None and t - start >= min_quiet:
                out.append(start)
            start = None
    return out


def measure(root: Path, min_quiet: float) -> dict[str, Any]:
    import eval as ev  # noqa: PLC0415

    devices, _dropped = ev.load_corpus(root)
    rows: dict[str, list[tuple[float, list[float]]]] = defaultdict(list)
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
                spells = [s / expected for s in _quiet_spells(rel, stop, min_quiet)]
                rows[device_type].append((dur / expected, spells))
    report: dict[str, Any] = {}
    for device_type, items in sorted(rows.items()):
        n_spells = sum(len(s) for _r, s in items)
        per_ratio = {}
        for r in RATIOS:
            reached = sum(1 for e, _s in items if e >= r)
            spells_past = sum(1 for _e, s in items for x in s if x >= r)
            cycles_past = sum(1 for _e, s in items if any(x >= r for x in s))
            per_ratio[str(r)] = {
                "ends_reached": round(100.0 * reached / len(items), 1),
                "spells_past": round(100.0 * spells_past / n_spells, 1) if n_spells else 0.0,
                "cycles_with_spell_past": round(100.0 * cycles_past / len(items), 1),
            }
        report[device_type] = {"cycles": len(items), "spells": n_spells, "ratios": per_ratio}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=str(_REPO_ROOT / "cycle_data"))
    parser.add_argument("--min-quiet", type=float, default=60.0,
                        help="shortest quiet spell counted, s (default 60)")
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args(argv)
    report = measure(Path(args.root), args.min_quiet)
    for device_type, rep in report.items():
        print(f"\n{device_type}: {rep['cycles']} cycles, {rep['spells']} mid-cycle quiet spells")
        print(f"  {'ratio':>6}{'ends reached %':>16}{'spells past %':>15}{'cycles w/ spell past %':>24}")
        for r, m in rep["ratios"].items():
            print(f"  {r:>6}{m['ends_reached']:>16}{m['spells_past']:>15}"
                  f"{m['cycles_with_spell_past']:>24}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
