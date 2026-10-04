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
"""analyze_diag.py - what the shipped suggestion engine proposes for one file.

    python3 devtools/analyze_diag.py <export-or-diagnostics.json> [--json FILE]
    python3 devtools/analyze_diag.py            # interactive file prompt

Reads a WashData export or diagnostics dump (every shape ``devtools/eval.py``
reads) and prints, per suggestible setting, the value the device runs with
(the option, else its device-resolved default), what the suggestion engine
would propose for it now, the suggestion the file itself carried, and the
engine's reason; then a per-programme cycle summary.

**Nothing here is a heuristic of its own (audit MATCH-EVAL-13).** Until 0.5.8 this
tool carried a fork of every suggestion formula, which had drifted: it still
proposed ``running_dead_zone`` (removed in 0.5.3) and sized ``min_off_gap`` as
``min(p05 inter-cycle gap x 0.8, 3600)``, the bound item 104 replaced. It now
builds the device's ``WashDataManager`` from the file's own entry data and
options (storage stubbed, the ``devtools/suggestion_loop_eval.py`` construction),
rebuilds every envelope with the current code, runs every suggestion pass in
cycle-end order through the ``LearningManager`` gates (``run_passes``: the
operational, standby-floor, model, detection and batch passes, then
``reconcile_suggestions``), and shows what ``ws_api._visible_suggestions`` lets
through: the list the Settings tab would show, muted keys and values the device
already runs with dropped.

One input is approximated: the operational pass reads the cadence model the
manager fills from live reading intervals of cycles that ended on their own
(``LearningManager.close_cycle_cadence``). A file does not carry it, so it is
rebuilt from the stored traces of completed cycles (intervals in (0.1, 1800) s,
the newest 200 per cycle and overall). A trace stored with a throttle or a trim
can differ from what the manager saw.

``--no-color`` is accepted for compatibility (the output has no colour).

Run from the repository root with the venv:
    .venv/bin/python devtools/analyze_diag.py cycle_data/.../<export>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "devtools"))

#: Reading intervals the cadence model accepts (``LearningManager.process_power_reading``).
_MIN_INTERVAL_S, _MAX_INTERVAL_S = 0.1, 1800.0
_MAX_PER_CYCLE = 200


def _cadence_intervals(cycles: list[dict[str, Any]]) -> list[float]:
    """Reading intervals of the cycles that ended on their own, oldest first."""
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        _cycle_readings,
    )

    out: list[float] = []
    for c in cycles:
        if not isinstance(c, dict) or c.get("status") != "completed":
            continue
        if str(c.get("termination_reason") or "") in ("user", "force_stopped"):
            continue
        pts = _cycle_readings(c)
        deltas = [
            b[0] - a[0] for a, b in zip(pts, pts[1:])
            if _MIN_INTERVAL_S < b[0] - a[0] < _MAX_INTERVAL_S
        ]
        out.extend(deltas[-_MAX_PER_CYCLE:])
    return out


def analyse(path: Path) -> dict[str, Any]:
    """The report for one file: settings, suggestions and a cycle summary."""
    import suggestion_loop_eval as slo  # noqa: PLC0415

    from custom_components.ha_washdata import ws_api  # noqa: PLC0415
    from custom_components.ha_washdata.detector_config import (  # noqa: PLC0415
        effective_option_values,
    )

    ev = slo.ev
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    unwrapped = ev._unwrap(doc) if isinstance(doc, dict) else None  # noqa: SLF001
    if unwrapped is None:
        raise SystemExit(f"{path}: not a WashData export or diagnostics dump")
    data, entry_data, entry_options, fmt = unwrapped
    entry_data = {"power_sensor": "sensor.analyze_diag", "name": "analyze",
                  **{k: v for k, v in entry_data.items() if v is not None}}
    options = {k: v for k, v in entry_options.items() if v is not None}
    stored_suggestions = dict(data.get("suggestions") or {})

    base = dict(data)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        base[key] = list(base.get(key) or [])
    base["profiles"] = {k: dict(v) if isinstance(v, dict) else v
                        for k, v in (base.get("profiles") or {}).items()}
    base["envelopes"] = dict(base.get("envelopes") or {})
    base["suggestions"] = {}

    mgr = slo._manager(entry_data, options, path.name, base)  # noqa: SLF001
    device_type = mgr.device_type

    async def _rebuild() -> None:
        for name in list(base["profiles"]):
            await mgr.profile_store.async_rebuild_envelope(name)

    ev.run_coro(_rebuild())
    cad = slo.cadence(_cadence_intervals(base["past_cycles"]))
    slo.run_passes(mgr, cad)

    merged = {**entry_data, **options}
    effective = effective_option_values(merged, device_type)
    visible = {
        key: (suggested, item)
        for key, item, suggested, _current in ws_api._visible_suggestions(  # noqa: SLF001
            mgr.profile_store, merged, device_type
        )
    }
    rows = []
    for key in ws_api._SUGGESTION_KEYS:  # noqa: SLF001
        current = merged.get(key)
        if current is None:
            current = effective.get(key)
        suggested, item = visible.get(key, (None, {}))
        in_file = stored_suggestions.get(key)
        rows.append({
            "key": key,
            "current": current,
            "suggested": suggested,
            "in_file": in_file.get("value") if isinstance(in_file, dict) else None,
            "reason": item.get("reason") if isinstance(item, dict) else None,
        })

    per_profile: dict[str, list[float]] = defaultdict(list)
    for c in base["past_cycles"]:
        name, dur = c.get("profile_name"), c.get("duration")
        try:
            if name and dur:
                per_profile[str(name)].append(float(dur))
        except (TypeError, ValueError):
            continue
    return {
        "file": str(path),
        "format": fmt,
        "device_type": device_type,
        "cycles": len(base["past_cycles"]),
        "profiles": len(base["profiles"]),
        "cadence": None if cad is None else {"p95_s": cad[0], "median_s": cad[1]},
        "settings": rows,
        "programmes": {
            name: {
                "cycles": len(durs),
                "avg_min": round(statistics.mean(durs) / 60.0, 1),
                "sd_min": round((statistics.stdev(durs) if len(durs) > 1 else 0.0) / 60.0, 1),
            }
            for name, durs in sorted(per_profile.items(), key=lambda kv: -len(kv[1]))
        },
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def print_report(rep: dict[str, Any]) -> None:
    print(f"\n{rep['file']}")
    cad = rep["cadence"]
    cad_txt = (
        f"p95 {cad['p95_s']:.1f} s, median {cad['median_s']:.1f} s" if cad
        else "fewer than 20 intervals: operational pass skipped"
    )
    print(f"  format {rep['format']} | device {rep['device_type']} | "
          f"{rep['cycles']} cycles | {rep['profiles']} profiles | cadence {cad_txt}")
    print(f"\n  {'setting':<40}{'runs with':>12}{'suggested':>12}{'in file':>12}  reason")
    print("  " + "-" * 100)
    for r in rep["settings"]:
        if r["suggested"] is None and r["in_file"] is None:
            continue
        print(f"  {r['key']:<40}{_fmt(r['current']):>12}{_fmt(r['suggested']):>12}"
              f"{_fmt(r['in_file']):>12}  {r['reason'] or ''}")
    quiet = [r["key"] for r in rep["settings"] if r["suggested"] is None and r["in_file"] is None]
    if quiet:
        print(f"\n  no suggestion now or in the file: {', '.join(quiet)}")
    if rep["programmes"]:
        print(f"\n  {'programme':<40}{'cycles':>7}{'avg min':>9}{'sd min':>8}")
        for name, s in rep["programmes"].items():
            print(f"  {name[:40]:<40}{s['cycles']:>7}{s['avg_min']:>9.1f}{s['sd_min']:>8.1f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("export", nargs="?", metavar="EXPORT_JSON",
                        help="a WashData export or diagnostics dump")
    parser.add_argument("--json", metavar="FILE", help="also write the report here")
    parser.add_argument("--no-color", action="store_true", help="accepted; no effect")
    args = parser.parse_args(argv)

    export_path = args.export
    if not export_path:
        print("Enter the path to the export / diagnostics JSON file:")
        export_path = input("  > ").strip().strip("'\"")
    path = Path(export_path or "")
    if not path.is_file():
        print(f"file not found: {export_path!r}", file=sys.stderr)
        return 2
    rep = analyse(path)
    print_report(rep)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    # Only as a script: a library call (the tests) must not leave logging disabled.
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
