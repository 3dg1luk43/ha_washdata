#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chronological closed-loop label replay of the shipped matcher and label gate.

    python3 devtools/closed_loop_eval.py [--jobs N] [--arms oracle,gate,review]
        [--device-types washing_machine,dryer] [--export SUBSTR] [--json OUT]
        [--seeds N] [--config-override JSON|@FILE] [--corpus DIR]
    python3 devtools/closed_loop_eval.py --compare BEFORE.json AFTER.json

``eval.py`` scores every fold against a store in which every other cycle carries
its TRUE label. No install has that store: a cycle is labelled by the cycle-end
gate (``match_rules.cycle_end_label_verdict`` -> ``profile_store.label_verdict``
at ``learning_confidence``), by the user answering a review request, or not at
all, and what it was labelled shapes every later match. This replays that loop.

Per device, each programme's FIRST cycle by start time (``--seeds N``: first N)
is a seed, labelled by hand: all seeds are stored and their envelopes built
before the replay, so every programme exists from the start. Every
later cycle, in start-time order, is matched as a complete cycle
(``in_progress=False``, the cycle-end match) and at 50% of its duration
(``in_progress=True``) against the store built so far, scored against its stored
label, then stored with the label its arm gives it (the labelled profile's
envelope is rebuilt, as the manager does at cycle end):

  oracle  its true label: the world eval.py measures, for reference.
  gate    the label gate's winner, else unlabelled: the user never answers a
          review request.
  review  gate, plus a refused cycle whose ``label_confidence`` reaches
          ``learning_confidence`` is labelled truthfully: the user answers every
          review request ``learning.py`` raises for a refused gate (register
          item 433).

Each row: ``ok`` (complete-cycle top-1 == truth), ``ok50`` (top-1 at 50%),
``lab`` (the label stored, None = unlabelled), ``truth``, ``reason`` (the gate's
reason), and the complete match's ``best`` / ``lc`` (label_confidence) / ``mg``
(margin). The summary reports per device, per device type and in total: top-1 at
the end and at 50%, the share of cycles labelled, and how many labels are wrong.

Devices: observed cycles (``past_cycles``) only, so a store package (a download,
not a chronology) is never replayed; at least 10 labelled traced cycles and 2
programmes; clone files dropped as in eval.py. **Twin entries are excluded**
(``eval.twin_entries``): two config entries on one plug record the same runs
under two label sets (the maintainer's washer ``01KBWSV8``/``01KXGA3C``, the
dishwasher ``01KX67PQ``/``01KDMTAA``), so replaying both counts each physical run
twice and weights one household double; only the entry with the most labelled
traced cycles is kept.

Limits: the profile records (``avg_duration`` etc.) come from the export, i.e.
from the whole history, and only envelopes are rebuilt as cycles arrive; every
other field of a stored cycle (a golden pin, ``manual_duration``) travels with it
whatever label the arm gives it; the live 5-minute cadence, switching rules and
end detection are not replayed (use ``decisive_margin_eval.py`` /
``end_gate_eval.py``). A closed loop amplifies small changes: one refused cycle
labelled differently can move every later match of its device, so judge a change
on the paired flips and on more than one device, and re-check it with
``--seeds 2``. Runs the code of the tree it lives in. ``--config-override`` takes
eval.py's format (options / options_by_device / const).

``--compare`` pairs devices by path and rows by position (the same corpus and
seeds give the same sequence) and prints, per device, device type and arm, the
deltas of top-1 / top-1@50 / labelled / wrong labels with the paired flips
(fixed / broken) and McNemar's exact p. File names under ``user-Contributed/``
carry contributors' names, so output keys those devices by a hash of the path
(``eval.public_key``); the audit's legacy list format is read too.

Exit codes: 0 ok, 2 nothing to replay or to pair.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterable  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as E  # noqa: E402

ARMS = ("oracle", "gate", "review")
MIN_CYCLES = 10          # labelled traced observed cycles a device needs
MIN_PROGRAMMES = 2
MIN_POINTS = 8           # readings a replayed cycle needs (eval.MIN_TRACE_POINTS)
HALF = 0.5               # the mid-cycle checkpoint


# ------------------------------------------------------------------- pure parts

def device_type_of(dev: E.Device) -> str:
    """The device type the manager is built with (eval.entry_dicts' rule)."""
    data, opts = E.entry_dicts(dev, {})
    return str(opts.get("device_type", data.get("device_type")) or "washing_machine")


def replay_cycles(dev: E.Device) -> list[dict]:
    """Labelled observed cycles with a usable trace."""
    return [c for c in (dev.data.get("past_cycles") or [])
            if isinstance(c, dict) and c.get("profile_name")
            and isinstance(c.get("power_data"), list) and len(c["power_data"]) >= MIN_POINTS]


def select_devices(devs: Iterable[E.Device], twins: dict[str, Any],
                   device_types: set[str] | None = None,
                   export: str | None = None) -> list[E.Device]:
    """Devices to replay: observed, enough cycles and programmes, no twin entry."""
    out = []
    for d in devs:
        if d.path.startswith("store/") or d.path in twins:
            continue
        if export and export not in d.path:
            continue
        if device_types and device_type_of(d) not in device_types:
            continue
        pcs = [c for c in (d.data.get("past_cycles") or [])
               if isinstance(c, dict) and c.get("profile_name") and c.get("power_data")]
        if len(pcs) >= MIN_CYCLES and len({c["profile_name"] for c in pcs}) >= MIN_PROGRAMMES:
            out.append(d)
    return out


def start_ts(c: dict) -> float:
    """Start time (epoch s) for ordering; 0 when unparseable."""
    from custom_components.ha_washdata.profile_store import _parse_start_dt  # noqa: PLC0415
    p = _parse_start_dt(c.get("start_time"))
    return p.timestamp() if p else 0.0


def split_seeds(cycles: list[dict], key: Callable[[dict], float] = start_ts,
                per_programme: int = 1) -> tuple[list[dict], list[dict]]:
    """(seeds, sequence): each programme's first ``per_programme`` cycles by ``key``,
    then the rest in order.

    Stable, so cycles with the same start keep their stored order.
    """
    seen: dict[str, int] = {}
    seeds: list[dict] = []
    seq: list[dict] = []
    for c in sorted(cycles, key=key):
        name = c["profile_name"]
        (seq if seen.get(name, 0) >= per_programme else seeds).append(c)
        seen[name] = seen.get(name, 0) + 1
    return seeds, seq


def arm_label(arm: str, verdict: str | None, best: str | None, conf: float,
              floor: float, truth: str) -> str | None:
    """The label the arm stores for a cycle (None = unlabelled)."""
    if arm == "oracle":
        return truth
    if verdict:
        return verdict
    if arm == "review" and best and conf >= floor:
        return truth
    return None


def tally(rows: list[dict]) -> dict[str, int]:
    """Counts behind every summary figure."""
    return {
        "n": len(rows),
        "ok": sum(1 for r in rows if r["ok"]),
        "ok50": sum(1 for r in rows if r["ok50"]),
        "lab": sum(1 for r in rows if r["lab"]),
        "wrong": sum(1 for r in rows if r["lab"] and r["lab"] != r["truth"]),
    }


def _add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def fmt_tally(t: dict[str, int]) -> str:
    n = t.get("n", 0)
    return (f"n {n} top1 {_pct(t.get('ok', 0), n):.1f}% top1@50 {_pct(t.get('ok50', 0), n):.1f}% "
            f"labelled {_pct(t.get('lab', 0), n):.1f}% wrong-labels {t.get('wrong', 0)} "
            f"({_pct(t.get('wrong', 0), t.get('lab', 0)):.1f}% of labels)")


def load_doc(path: str | Path) -> dict:
    """A result file, also the audit's legacy ``[[path, seeds, {arm: rows}], ...]``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"meta": {}, "devices": [
            {"path": E.public_key(p), "dev": "?", "seeds": s, "arms": arms} for p, s, arms in raw]}
    return raw


def summarize(doc: dict, out: Callable[[str], Any] = print) -> None:
    by_type: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    tot: dict[str, dict[str, int]] = defaultdict(dict)
    for d in doc["devices"]:
        line = [d["path"][:50], d.get("dev", "?"), f"seeds {d.get('seeds')}"]
        for arm, rows in d["arms"].items():
            t = tally(rows)
            by_type[(d.get("dev", "?"), arm)] = _add(by_type[(d.get("dev", "?"), arm)], t)
            tot[arm] = _add(tot[arm], t)
            line.append(f"{arm} {t['ok']}/{t['n']} @50 {t['ok50']} lab {t['lab']} wrong {t['wrong']}")
        out(" | ".join(line))
    for (dtype, arm), t in sorted(by_type.items()):
        out(f"{dtype} {arm}: {fmt_tally(t)}")
    for arm, t in tot.items():
        out(f"TOTAL {arm}: {fmt_tally(t)}")


def compare_docs(before: dict, after: dict, out: Callable[[str], Any] = print) -> int:
    """Paired per-device deltas; 2 when no device/arm pairs."""
    bdev = {d["path"]: d for d in before["devices"]}
    pairs = 0
    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for d in after["devices"]:
        b = bdev.get(d["path"])
        if b is None:
            continue
        for arm, rows in d["arms"].items():
            brows = b["arms"].get(arm)
            if brows is None:
                continue
            if [r["truth"] for r in brows] != [r["truth"] for r in rows]:
                out(f"{d['path'][:50]} {arm}: sequences differ, not paired")
                continue
            pairs += 1
            tb, ta = tally(brows), tally(rows)
            delta = {k: ta[k] - tb[k] for k in ta}
            delta.update(
                n=ta["n"],
                fixed=sum(1 for x, y in zip(brows, rows) if not x["ok"] and y["ok"]),
                broken=sum(1 for x, y in zip(brows, rows) if x["ok"] and not y["ok"]),
                fixed50=sum(1 for x, y in zip(brows, rows) if not x["ok50"] and y["ok50"]),
                broken50=sum(1 for x, y in zip(brows, rows) if x["ok50"] and not y["ok50"]),
            )
            for key in ((d.get("dev", "?"), arm), ("TOTAL", arm)):
                agg[key] = _add(agg[key], delta)
            out(f"{d['path'][:50]} {d.get('dev', '?')} {arm}: " + _fmt_delta(delta))
    for (dtype, arm), dl in sorted(agg.items(), key=lambda kv: (kv[0][0] == "TOTAL", kv[0])):
        out(f"{dtype} {arm}: " + _fmt_delta(dl))
    return 0 if pairs else 2


def _fmt_delta(d: dict[str, int]) -> str:
    """Deltas with the paired flips (fixed / broken) and McNemar's exact p."""
    n = d.get("n", 0)
    fx, br, fx50, br50 = (d.get(k, 0) for k in ("fixed", "broken", "fixed50", "broken50"))
    return (f"n {n} top1 {_pct(d.get('ok', 0), n):+.1f}pp (+{fx} -{br} p={E.mcnemar_p(fx, br):.2f}) "
            f"@50 {_pct(d.get('ok50', 0), n):+.1f}pp (+{fx50} -{br50} p={E.mcnemar_p(fx50, br50):.2f}) "
            f"labelled {d.get('lab', 0):+d} wrong {d.get('wrong', 0):+d}")


# --------------------------------------------------------------------- replay

async def run_arm(dev: E.Device, arm: str, seq: list[dict], seeds: list[dict],
                  override: dict) -> tuple[str, list[dict]]:
    from custom_components.ha_washdata import match_rules  # noqa: PLC0415
    data = E.base_data(dev)
    data["past_cycles"] = []
    data["envelopes"] = {}
    st, mgr = E.fresh_store(dev, data, override)
    floor = float(mgr._learning_confidence or 0.0)  # noqa: SLF001
    data["past_cycles"].extend(seeds)
    for name in list(data["profiles"]):
        await st.async_rebuild_envelope(name)
    rows = []
    for c in seq:
        pts = E.query_points(c)
        dur = E.cycle_duration(c, pts)
        res = await st.async_match_profile([list(p) for p in pts], dur, in_progress=False)
        half = [p for p in pts if p[0] <= pts[0][0] + HALF * dur]
        res50 = (await st.async_match_profile([list(p) for p in half], half[-1][0] - half[0][0],
                                              in_progress=True)
                 if len(half) >= E.MIN_PREFIX_POINTS else None)
        truth = c["profile_name"]
        verdict, reason = match_rules.cycle_end_label_verdict(res, floor, st.get_profiles())
        conf = float(getattr(res, "label_confidence", 0.0) or 0.0)
        lab = arm_label(arm, verdict, res.best_profile, conf, floor, truth)
        rows.append({"ok": res.best_profile == truth,
                     "ok50": bool(res50 and res50.best_profile == truth),
                     "lab": lab, "truth": truth, "reason": reason,
                     "best": res.best_profile, "lc": round(conf, 4),
                     "mg": round(float(res.ambiguity_margin or 0.0), 4)})
        c2 = dict(c)
        if lab:
            c2["profile_name"] = lab
        else:
            c2.pop("profile_name", None)
        data["past_cycles"].append(c2)
        if lab:
            await st.async_rebuild_envelope(lab)
    return mgr.device_type, rows


_W: dict[str, Any] = {}


def _init(root: str, override: dict, seeds: int = 1) -> None:
    _W.update(root=Path(root), override=override, seeds=seeds)
    E._quiet_logs()  # noqa: SLF001
    if override.get("const") and not _W.get("cm"):
        cm = E.const_overrides(override["const"])
        cm.__enter__()  # held for the worker's lifetime
        _W["cm"] = cm


def _job(args: tuple[str, str]) -> tuple[str, str, int, str, list[dict]]:
    path, arm = args
    dev = E.load_device(_W["root"], path)
    assert dev is not None, path
    seeds, seq = split_seeds(replay_cycles(dev), per_programme=_W.get("seeds", 1))
    dtype, rows = E.run_coro(run_arm(dev, arm, seq, seeds, _W["override"]))
    return path, arm, len(seeds), dtype, rows


def run(corpus: Path, arms: list[str], jobs: int, override: dict,
        device_types: set[str] | None, export: str | None, seeds: int = 1) -> dict:
    devs, _ = E.load_corpus(corpus)
    twins = E.twin_entries([d for d in devs if not d.path.startswith("store/")])
    picked = select_devices(devs, twins, device_types, export)
    work = [(d.path, a) for d in picked for a in arms]
    if not work:
        return {"meta": {}, "devices": []}
    if jobs <= 1:
        _init(str(corpus), override, seeds)
        try:
            results = [_job(w) for w in work]
        finally:
            cm = _W.pop("cm", None)
            if cm is not None:
                cm.__exit__(None, None, None)
    else:
        with ProcessPoolExecutor(jobs, initializer=_init, initargs=(str(corpus), override, seeds)) as ex:
            results = list(ex.map(_job, work))
    by_path: dict[str, dict] = {}
    for path, arm, nseed, dtype, rows in results:
        d = by_path.setdefault(path, {"path": E.public_key(path), "dev": dtype,
                                      "seeds": nseed, "arms": {}})
        d["arms"][arm] = rows
    return {
        "meta": {"rev": E._git_rev(), "code_sha": E.code_sha(), "arms": arms,  # noqa: SLF001
                 "override": override, "seeds_per_programme": seeds,
                 "twins_excluded": sorted(E.public_key(p) for p in twins)},
        "devices": [by_path[d.path] for d in picked if d.path in by_path],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=E.REPO / "cycle_data")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--device-types", default="", help="comma-separated, e.g. washing_machine,dryer")
    ap.add_argument("--export", default="", help="only exports whose path contains this")
    ap.add_argument("--json", type=Path, help="write the rows here")
    ap.add_argument("--seeds", type=int, default=1, help="hand-labelled first runs per programme")
    ap.add_argument("--config-override", default=None, help="eval.py's override (JSON or @file)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args(argv)
    if a.compare:
        return compare_docs(load_doc(a.compare[0]), load_doc(a.compare[1]))
    arms = [x for x in a.arms.split(",") if x]
    bad = set(arms) - set(ARMS)
    if bad:
        ap.error(f"unknown arm(s): {', '.join(sorted(bad))}")
    types = {x for x in a.device_types.split(",") if x} or None
    doc = run(a.corpus, arms, a.jobs, E.parse_override(a.config_override), types, a.export or None,
              max(1, a.seeds))
    if not doc["devices"]:
        print("nothing to replay", file=sys.stderr)
        return 2
    if a.json:
        a.json.write_text(json.dumps(doc), encoding="utf-8")
    summarize(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
