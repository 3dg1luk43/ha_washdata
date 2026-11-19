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
  PAUSED, and held for the dwell (``--dwell``, the default 60 s) while still
  quiet, finds the gate open: that door would end the run mid-programme. The
  PAUSED intervals are the detector's own: each cycle is replayed through a
  ``CycleDetector`` with the export's config (off delay and min off gap out of
  reach so it stays open), and its state changes are read. Past PAUSED the
  detector is in ENDING, where a held door ended the cycle before 0.6.1 too. The
  share of PAUSED intervals, and of cycles with any.

Both are judged in programme time, as ``_gate_elapsed_s`` reads it: the replay's
finished stalls up to that moment are left out. It runs from the stored cycle
start (the trace's first reading), where the live detector that recorded the run
started and from which the expected durations are measured; the replay's own
start can come later (a false start it aborts and makes again, or a sparse trace
it never holds), which the live run did not.

Rows per ratio patch the constants to one ratio for every device type; the
``shipped`` row runs them as they are. What this does not replay: the matcher
(the expected duration is the leave-one-out mean, not the live gate's, and the
match is taken as unambiguous) and the door timing itself (opened at the latest
moment the spell allows).
"""
from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "devtools"))

RATIOS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
# Ends the appliance reached: every stored reason but a hand stop and a force
# stop (the External End Trigger is one, `TerminationReason.EXTERNAL`).
_NATURAL = ("timeout", "smart", "terminal_drop", "door", "external", None, "")


def _device(dev: Any) -> tuple[str, Any]:
    """The device type and the detector config the export's entry runs."""
    from custom_components.ha_washdata.detector_config import (  # noqa: PLC0415
        build_detector_config,
    )

    merged = {**(dev.entry_data or {}), **(dev.entry_options or {})}
    device_type = str(merged.get("device_type") or "washing_machine")
    return device_type, build_detector_config(
        dev.entry_options or {}, dev.entry_data or {}, device_type
    )


def _paused_door_moments(
    pts: list[list[float]], cfg: Any, stop: float, dwell: float, duration: float
) -> tuple[list[float], float]:
    """The cycle replayed through a ``CycleDetector`` (off delay and min off gap
    out of reach, so it stays open across every spell). Per PAUSED interval that
    activity follows, the latest moment a door can open in it and stay open for
    ``dwell`` while the power is still under the Stop Threshold (the gate is
    monotonic in time, so the latest moment decides); intervals with no such
    moment are left out. Returned in programme time, with the stored
    ``duration`` too: less every stall the replay had finished by then, as
    ``_gate_elapsed_s`` reads it live."""
    from custom_components.ha_washdata.const import STATE_PAUSED  # noqa: PLC0415
    from custom_components.ha_washdata.cycle_detector import CycleDetector  # noqa: PLC0415

    now = [0.0]
    changes: list[tuple[float, str]] = []
    det = CycleDetector(
        replace(cfg, off_delay=10**7, min_off_gap=10**7),
        lambda _old, new: changes.append((now[0], new)),
        lambda _c: None,
    )
    marks: list[float] = []
    left_out: list[float] = []  # finished stall seconds after each reading
    for t, p in pts:
        now[0] = t
        det.process_reading(float(p), _T0 + timedelta(seconds=t))
        marks.append(t)
        left_out.append(sum(length for _at, length in det._gate_spans()))  # noqa: SLF001

    def programme(at: float) -> float:
        i = bisect_right(marks, at) - 1
        return at - (left_out[i] if i >= 0 else 0.0)

    out: list[float] = []
    for i, (at, state) in enumerate(changes):
        if state != STATE_PAUSED:
            continue
        left = changes[i + 1][0] if i + 1 < len(changes) else None
        resumed = next((t for t, p in pts if t > at and p >= stop), None)
        if resumed is None:
            continue  # the final quiet: the end, not a mid-cycle pause
        # Strictly before both boundaries: a dwell expiring at the resumption
        # meets a reading at the threshold and re-arms instead of ending.
        latest = min(resumed - dwell, left if left is not None else resumed) - 1e-3
        if latest >= at:
            out.append(programme(latest))
    return out, programme(duration)


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
        device_type, cfg = _device(dev)
        stop = float(cfg.stop_threshold_w)
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
                moments, programme_dur = _paused_door_moments(rel, cfg, stop, dwell, dur)
                rows[device_type].append((expected, programme_dur, moments))
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


def _dwell(text: str) -> float:
    """``--dwell``: a finite number of seconds above 0. NaN would drop every door
    moment (no comparison holds) and report zero exposure as if the ratio were safe."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number of seconds above 0: {text!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=str(_REPO_ROOT / "cycle_data"))
    parser.add_argument("--dwell", type=_dwell, default=60.0,
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
