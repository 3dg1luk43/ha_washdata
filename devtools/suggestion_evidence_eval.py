#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the suggestion engine reads as evidence, per corpus device (PR #529 audit).

    python3 devtools/suggestion_evidence_eval.py [--device SUBSTR] [--corpus DIR] [--json FILE]

Two rules of the 0.6.1 suggestion audit decide which stored cycles count, before any
pass proposes anything. Each is measured with its constant swept and with the rule
as 0.6.0 had it, on the shipped code otherwise. Pure statistics, no replay.

**Programme length** (``suggestion_engine._PROGRAMME_LENGTH_RATIO``): a completed
cycle whose trace shows a long low phase is dropped as ``mid_restart`` (two loads
merged) unless it lasted ``[lo, hi]`` x its label's median duration. Arms: ``off``
(no exemption, 0.6.0), ``0.9-1.1``, ``0.8-1.25`` (shipped), ``0.7-1.4``. Per arm: clean
cycles and ``mid_restart`` drops over every device's stored cycles
(``select_clean_cycles`` at the device's Stop Threshold), and for the cycles the
shipped band admits, how long each ran against its label's median (a merged double
load runs about 2x).

**Standby advisory** (``detect_standby_above_stop``, "Still drawing X W", #445/#525,
and the #458 floor it feeds): the device's last 8 stored cycles at its Stop
Threshold, with its recorded Stop Threshold changes (item 646). Arms: ``legacy``
(0.6.0: every cycle's last stored sample, all of them above the threshold) and the
shipped rule with ``_TAIL_SETTLED_FRACTION`` at 0.1, 0.25 (shipped) and 0.5. Per arm:
the devices flagged and the level reported.

Stop Threshold: the device's options as ``effective_option_values`` resolves them.
``store/`` corpus files are skipped. Exit codes: 0, 2 no corpus.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom_components.ha_washdata import suggestion_engine as se  # noqa: E402
from custom_components.ha_washdata.const import CONF_STOP_THRESHOLD_W  # noqa: E402
from custom_components.ha_washdata.detector_config import effective_option_values  # noqa: E402

_EVAL_PATH = REPO / "devtools" / "eval.py"
_spec = importlib.util.spec_from_file_location("wd_suggestion_evidence_corpus", _EVAL_PATH)
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev
_spec.loader.exec_module(ev)

LENGTH_ARMS: dict[str, tuple[float, float]] = {
    "off": (math.inf, 0.0),
    "0.9-1.1": (0.9, 1.1),
    "0.8-1.25": (0.8, 1.25),
    "0.7-1.4": (0.7, 1.4),
}
TAIL_ARMS: dict[str, float | None] = {"legacy": None, "0.1": 0.1, "0.25": 0.25, "0.5": 0.5}
RECENT = 8


@contextlib.contextmanager
def _patched(name: str, value: Any) -> Iterator[None]:
    saved = getattr(se, name)
    setattr(se, name, value)
    # The settled-tail cache is keyed by trace, not by _TAIL_SETTLED_FRACTION: an
    # arm would read the level the previous arm computed (PR #529 review).
    se._TAIL_SETTLED_CACHE.clear()  # noqa: SLF001
    try:
        yield
    finally:
        setattr(se, name, saved)
        se._TAIL_SETTLED_CACHE.clear()  # noqa: SLF001


def legacy_standby(cycles: list[dict[str, Any]], stop: float, recent: int = RECENT) -> float | None:
    """0.6.0's rule: each cycle's last stored sample (skipped above 10% of its peak);
    flagged only when every one is above the threshold, at least 2, consistent."""
    finals: list[float] = []
    for cycle in list(cycles)[-recent:]:
        raw = cycle.get("power_data") if isinstance(cycle, dict) else None
        if not isinstance(raw, list) or not raw or not isinstance(raw[-1], (list, tuple)):
            continue
        try:
            final_w = float(raw[-1][1])
        except (TypeError, ValueError, OverflowError, IndexError):
            continue
        if not math.isfinite(final_w):  # "NaN" parses, and its median is NaN, not None
            continue
        peak = se._cycle_peak_w(cycle, raw)  # noqa: SLF001 - unchanged since 0.6.0
        if peak > 0 and final_w > peak * se._STANDBY_PEAK_FRACTION:  # noqa: SLF001
            continue
        finals.append(final_w)
    if len(finals) < 2 or any(f <= stop for f in finals):
        return None
    median = float(np.median(finals))
    if median <= 0 or float(np.median([abs(f - median) for f in finals])) / median > se._STANDBY_MAX_MAD:  # noqa: SLF001
        return None
    return round(median, 2)


def _label(cycle: dict[str, Any]) -> Any:
    """The label ``select_clean_cycles`` reads the programme median for."""
    return cycle.get("profile_name") or cycle.get("label")


def _selection_duration(cycle: dict[str, Any]) -> float:
    """The length ``select_clean_cycles`` weighs against that median: the stored
    duration, or the trace span when it is missing or not positive (PR #529 review:
    a cycle stored without one is admitted by its span, and reported 0.0)."""
    try:
        duration = float(cycle.get("duration") or 0.0)
    except (TypeError, ValueError, OverflowError):
        duration = 0.0
    if duration <= 0:
        readings = se._cycle_readings(cycle)  # noqa: SLF001
        if readings:
            duration = readings[-1][0] - readings[0][0]
    return duration


def device_row(dev: Any) -> dict[str, Any]:
    entry_data, options = ev.entry_dicts(dev, {})
    device_type = options.get("device_type") or entry_data.get("device_type") or "washing_machine"
    stop = float(effective_option_values({**entry_data, **options}, device_type)[CONF_STOP_THRESHOLD_W])
    past = [c for c in dev.data.get("past_cycles") or [] if isinstance(c, dict)]
    row: dict[str, Any] = {"device": ev.public_key(dev.path), "device_type": device_type,
                           "stop_w": stop, "cycles": len(past), "length": {}, "standby": {}}
    # By the record itself (select_clean_cycles returns the objects it was given),
    # not its stored id: two cycles without one, or a repeated id, are one key, so
    # a cycle the band admits could vanish from the ratios (PR #529 review).
    clean_ids: dict[str, set[int]] = {}
    for arm, ratio in LENGTH_ARMS.items():
        with _patched("_PROGRAMME_LENGTH_RATIO", ratio):
            clean, excluded = se.select_clean_cycles(past, stop_threshold_w=stop)
        clean_ids[arm] = {id(c) for c in clean}
        row["length"][arm] = {"clean": len(clean), "mid_restart": excluded.get("mid_restart", 0)}
    medians = se._label_median_durations(past)  # noqa: SLF001
    admitted = clean_ids["0.8-1.25"] - clean_ids["off"]
    row["admitted_ratios"] = sorted(
        round(_selection_duration(c) / medians[_label(c)], 2)
        for c in past
        if id(c) in admitted and medians.get(_label(c))
    )
    changes = se.stop_threshold_changes(dev.data.get("settings_changelog"))
    for arm, frac in TAIL_ARMS.items():
        if frac is None:
            row["standby"][arm] = legacy_standby(past, stop)
            continue
        with _patched("_TAIL_SETTLED_FRACTION", frac):
            adv = se.detect_standby_above_stop(past, stop, stop_changes=changes)
        row["standby"][arm] = adv["idle_w"] if adv else None
    return row


def run(corpus: Path, devices: list[str] | None = None) -> list[dict[str, Any]]:
    # store/ left out before clone removal: a package holding a history's traces
    # would otherwise drop that history as its clone (PR #529 review).
    kept, _dropped = ev.load_corpus(corpus, exclude=("store/",))
    return [device_row(d) for d in kept if not devices or any(s in d.path for s in devices)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--device", action="append", help="substring of a corpus path (repeatable)")
    ap.add_argument("--corpus", default=str(REPO / "cycle_data"))
    ap.add_argument("--json", help="write the per-device results here")
    a = ap.parse_args(argv)
    corpus = Path(a.corpus)
    if not corpus.is_dir():
        print(f"no corpus at {corpus}", file=sys.stderr)
        return 2
    rows = run(corpus, a.device)
    if not rows:
        print("no devices", file=sys.stderr)
        return 2
    print(f"== programme length: {len(rows)} devices, {sum(r['cycles'] for r in rows)} stored cycles")
    print(f"  {'arm':10s} {'clean':>6s} {'mid_restart':>12s}")
    for arm in LENGTH_ARMS:
        print(f"  {arm:10s} {sum(r['length'][arm]['clean'] for r in rows):6d} "
              f"{sum(r['length'][arm]['mid_restart'] for r in rows):12d}")
    for r in rows:
        if r["length"]["off"]["mid_restart"]:
            print(f"  {r['device'][:52]:52s} mid_restart "
                  + " / ".join(str(r["length"][a]["mid_restart"]) for a in LENGTH_ARMS)
                  + f"  admitted x median: {r['admitted_ratios']}")
    print(f"== standby advisory: last {RECENT} cycles, level flagged per arm "
          f"({' / '.join(TAIL_ARMS)})")
    for r in rows:
        levels = [r["standby"][a] for a in TAIL_ARMS]
        if any(v is not None for v in levels):
            print(f"  {r['device'][:52]:52s} stop {r['stop_w']:g} W: "
                  + " / ".join("-" if v is None else f"{v:g}" for v in levels))
    print("  flagged: " + "  ".join(
        f"{a}={sum(1 for r in rows if r['standby'][a] is not None)}" for a in TAIL_ARMS))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
