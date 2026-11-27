#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does a cycle's end moved past its trace from the recorder rebuild what was there? (#536)

    python3 devtools/extend_end_eval.py --store STORE.json --db home-assistant_v2.db \\
        --entity sensor.x_power [--cut-min 5,15,30,60] [--keepalive S] [--ha-tz ZONE] [--json OUT]

Trim can move a cycle's end past its stored trace: the readings come from the power
sensor's recorder history (``ws_api._trim_extension`` ->
``profile_store.cycle_extension_points``). This harness checks that on real data
without a running Home Assistant: for each stored live cycle the recorder still
covers, it cuts the trace ``--cut-min`` minutes before its end (as a cycle ended
too early would be), rebuilds the cut tail from the recorder with the shipped
builder, and compares it with the stored tail:

* ``end_diff_s``: the rebuilt end against the stored one (the store snaps to the
  last recorder reading at or before the requested end);
* ``energy_rel``: the cycle's energy (``integrate_cycle_wh``) rebuilt / stored - 1;
* ``matched``: the share of the stored tail's sensor reports (not keepalives) the
  recorder holds at the same time (within 1 s) and value (within 0.15 W or 1 %).

The recorder is read like ``ws_api._recorder_power`` reads it: state changes only
(``last_changed == last_updated``), the state in force at the window's start
included, ``unavailable``/``unknown`` as gaps. Open the DB copy read-only; copy a
live one with its -wal/-shm first.

Exit codes: 0, 2 bad input.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom_components.ha_washdata.profile_store import (  # noqa: E402
    cycle_extension_points,
    decompress_power_data,
)
from custom_components.ha_washdata.signal_processing import integrate_cycle_wh  # noqa: E402
from custom_components.ha_washdata.time_utils import synthetic_offset_keys  # noqa: E402
from custom_components.ha_washdata.trace_codec import decode_store_payload  # noqa: E402

_UNIT_SCALE = {"W": 1.0, "kW": 1000.0, "mW": 0.001}


# Home Assistant's configured zone (`--ha-tz`): an offset-less start_time (an
# imported or hand-edited cycle) is read in it, as the WS trim path reads it
# (`dt_util.as_utc` assumes DEFAULT_TIME_ZONE), whatever this host's zone.
_HA_TZ: Any = timezone.utc


def _ts(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_HA_TZ)
    return parsed.timestamp()


class Recorder:
    """The entity's state changes from a recorder DB copy, as (unix_ts, watts|None)."""

    def __init__(self, db: str, entity_id: str) -> None:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = con.execute(
            "SELECT metadata_id FROM states_meta WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            raise SystemExit(f"no recorder rows for {entity_id}")
        rows = con.execute(
            "SELECT s.state, s.last_updated_ts, s.last_changed_ts, a.shared_attrs "
            "FROM states s LEFT JOIN state_attributes a ON s.attributes_id = a.attributes_id "
            "WHERE s.metadata_id = ? ORDER BY s.last_updated_ts",
            (row[0],),
        ).fetchall()
        con.close()
        self.rows: list[tuple[float, float | None]] = []
        for state, updated, changed, attrs in rows:
            if changed is not None and abs(changed - updated) > 1e-6:
                continue  # an attribute-only write: not a state change
            unit = "W"
            try:
                unit = str(json.loads(attrs or "{}").get("unit_of_measurement") or "W")
            except ValueError:
                pass
            try:
                watts = float(state) * _UNIT_SCALE.get(unit, 1.0)
                if not math.isfinite(watts):
                    raise ValueError
                self.rows.append((float(updated), round(watts, 1)))
            except (TypeError, ValueError):
                self.rows.append((float(updated), None))
        self.first = self.rows[0][0] if self.rows else math.inf
        self.last = self.rows[-1][0] if self.rows else -math.inf

    def window(self, lo: float, hi: float) -> list[tuple[float, float | None]]:
        """Rows in [lo, hi], with the state in force at lo clamped to it."""
        before = [r for r in self.rows if r[0] <= lo]
        inside = [r for r in self.rows if lo < r[0] <= hi]
        return ([(lo, before[-1][1])] if before else []) + inside


def evaluate(cycle: dict[str, Any], rec: Recorder, cut_s: float, keepalive_s: float) -> dict[str, Any] | None:
    trace = [(float(o), float(p)) for o, p in decompress_power_data(cycle)]
    start = _ts(cycle.get("start_time"))
    if start is None or len(trace) < 3:
        return None
    end = trace[-1][0]
    cut_at = end - cut_s
    head = [pt for pt in trace if pt[0] <= cut_at]
    if len(head) < 2 or start + head[-1][0] < rec.first or start + end > rec.last:
        return None
    t0 = head[-1][0]
    rows = [(ts - start, w) for ts, w in rec.window(start + t0, start + end + 1.0)]
    ext = cycle_extension_points(t0, head[-1][1], rows, end, keepalive_s)
    syn = synthetic_offset_keys(cycle)
    tail_reports = [pt for pt in trace if pt[0] > t0 and (syn is None or round(pt[0] * 10) not in syn)]
    if ext is None:
        return {"cut_min": cut_s / 60, "rebuilt": False, "tail_reports": len(tail_reports)}
    pts, synthetic = ext
    rebuilt = [*head, *[(o, w) for o, w in pts]]
    stored_wh = integrate_cycle_wh([p[0] for p in trace], [p[1] for p in trace])
    rebuilt_wh = integrate_cycle_wh([p[0] for p in rebuilt], [p[1] for p in rebuilt])
    real_rows = [(o, w) for o, w in rows if w is not None and o > t0]

    def _held(o: float, w: float) -> bool:
        return any(abs(ro - o) <= 1.0 and abs(rw - w) <= max(0.15, 0.01 * abs(w)) for ro, rw in real_rows)

    matched = sum(1 for o, w in tail_reports if _held(o, w))
    return {
        "cut_min": cut_s / 60,
        "rebuilt": True,
        "end_diff_s": round(rebuilt[-1][0] - end, 1),
        "energy_rel": round(rebuilt_wh / stored_wh - 1.0, 4) if stored_wh > 0 else None,
        "tail_wh_stored": round(stored_wh - integrate_cycle_wh([p[0] for p in head], [p[1] for p in head]), 2),
        "tail_wh_rebuilt": round(rebuilt_wh - integrate_cycle_wh([p[0] for p in head], [p[1] for p in head]), 2),
        "tail_reports": len(tail_reports),
        "recorder_rows": len(real_rows),
        "keepalives": len(synthetic),
        "matched": round(matched / len(tail_reports), 3) if tail_reports else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", required=True, help="a ha_washdata.<entry_id> store file")
    ap.add_argument("--db", required=True, help="a recorder DB copy")
    ap.add_argument("--entity", required=True, help="the device's power sensor")
    ap.add_argument("--cut-min", default="5,15,30,60")
    ap.add_argument("--keepalive", type=float, default=30.0, help="the device's watchdog interval, s")
    ap.add_argument("--json")
    ap.add_argument("--ha-tz", default="UTC",
                    help="Home Assistant's time zone (IANA name), for offset-less start times")
    args = ap.parse_args(argv)
    global _HA_TZ  # noqa: PLW0603 - read by _ts
    try:
        _HA_TZ = ZoneInfo(args.ha_tz)
    except (ZoneInfoNotFoundError, ValueError):
        ap.error(f"unknown time zone {args.ha_tz!r}")
    try:
        cuts = [float(c) * 60 for c in args.cut_min.split(",") if c.strip()]
        payload = json.loads(Path(args.store).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"bad input: {exc}", file=sys.stderr)
        return 2
    data = payload.get("data", payload)
    decode_store_payload(data)
    rec = Recorder(args.db, args.entity)
    results = []
    for cycle in data.get("past_cycles") or []:
        for cut in cuts:
            r = evaluate(cycle, rec, cut, args.keepalive)
            if r is not None:
                results.append({"id": cycle.get("id"), "start": cycle.get("start_time"),
                                "status": cycle.get("status"), **r})
    print(f"{args.entity}: recorder {len(rec.rows)} rows, "
          f"{datetime.fromtimestamp(rec.first).isoformat(timespec='minutes') if rec.rows else '-'} .. "
          f"{datetime.fromtimestamp(rec.last).isoformat(timespec='minutes') if rec.rows else '-'}")
    print(f"{'cycle':14} {'cut':>4} {'end_diff':>8} {'energy':>8} {'tail Wh':>15} {'reports':>7} {'rows':>5} {'kalv':>5} {'matched':>7}")
    for r in results:
        if not r["rebuilt"]:
            print(f"{str(r['id'])[:14]:14} {r['cut_min']:>4.0f}  not rebuilt (no recorder reading after the cut; {r['tail_reports']} stored reports)")
            continue
        print(f"{str(r['id'])[:14]:14} {r['cut_min']:>4.0f} {r['end_diff_s']!s:>8} {r['energy_rel']!s:>8} "
              f"{r['tail_wh_stored']:>7}/{r['tail_wh_rebuilt']:<7} {r['tail_reports']:>7} {r['recorder_rows']:>5} "
              f"{r['keepalives']:>5} {r['matched']!s:>7}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
