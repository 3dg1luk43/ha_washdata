#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hand-picked program mismatch (#530): false alarms on the right pick, catches on a wrong one.

    python3 devtools/program_mismatch_eval.py [--corpus DIR] [--device SUBSTR]
                                              [--max-per-device N] [--jobs N]
                                              [--json FILE] [--from-rows FILE] [--sweep]

While a program is picked by hand the live match does not run; the manager runs a
background match on the ``profile_match_interval`` cadence and establishes a
mismatch with ``match_rules.program_mismatch_tick`` (another program beats the
picked one by ``PROGRAM_MISMATCH_MIN_MARGIN`` on ``PROGRAM_MISMATCH_PERSISTENCE``
matches in a row, with ``PROGRAM_MISMATCH_MIN_FRACTION`` of the picked program's
usual duration elapsed). This replays that on the corpus.

**Replay** (leave-one-out, production code): each scorable labelled cycle of a
device with >= 2 programmes whose type the manager runs the check on
(``trigger_rules.trigger_offered``: never a Threshold Device), by
``devtools/eval.py``'s corpus rules (clone files and twin entries dropped, the
label keeps another traced cycle) is removed from every list with every other
stored copy of its trace (same content at stored precision, any name, any stored
format: ``run_hash``), and the envelope and usual duration of each programme they
were stored under rebuilt from the others (a fold whose rebuild learns no duration
for the label is left out), on a store a real ``WashDataManager`` builds from the
export's options (``eval.fresh_store``). A record whose stored duration is no
finite number is left out of every list. A check runs where the manager schedules
one (``check_prefixes``): on the first recorded reading at least one
``profile_match_interval`` (the device's own, 300 s by default) after the last
check, strictly before the recorded end. A stretch without a reading runs none, so
a silent plug never feeds the persistence rule one unchanged prefix several times.
The prefix through that reading is matched with
``ProfileStore.async_match_profile(..., in_progress=True)``, what the background
check calls. One match per tick serves every pick: the rule reads the ranking
against the picked program, so each tick is judged for

* **(a) the right pick**: the cycle's own label is selected. A fire is a false
  alarm (``false_alarm``, per cycle: the trigger fires at most once);
* **(b) a wrong pick**: each other programme of the device, alias twins of the
  label excluded (the same trace stored under two names, ``eval.alias_classes``).
  ``detect`` is the share of (cycle, wrong pick) pairs that fire strictly before
  the cycle ends, ``elapsed_frac`` the median fraction of the cycle's own duration
  at the fire. ``nearest`` repeats it for the wrong pick whose usual duration is
  closest to the label's **in proportion** (smallest ``|ln(pick / label)|``, then
  name), the confusion a user is most likely to make.

Figures are pooled, then split into ``own history`` (one appliance's recorded
cycles) and ``store packages`` (``store/``: community downloads, several users'
runs per file, where one programme can sit under two names), then by device type.
A pick the manager would not check (no template, ``unmatchable_profiles``, or no
learned duration) never fires and stays in the denominator as a miss.

``--sweep`` replays the rule over a grid of margin x persistence x minimum fraction
on the recorded ticks (no new matching) and prints every row with its false-alarm
and detection rates; the shipped constants are marked. ``--json`` writes the ticks
and the summary; ``--from-rows`` reads such a file instead of matching again.

**Last run** (2026-11-19, 946 folds of 86 devices, ~8 min on 7 workers), the
shipped 0.18 / 6 / 0.5: false alarms 12 (1.3%), 5 of them on the 437 own-history
cycles (1.1%); wrong picks caught 38.4% (1904 of 4957) at a median 0.62 of the
cycle, the nearest-duration wrong pick 42.6% at 0.52. By type: washers 1.4% /
40.7%, dishwashers 0.9% / 33.2%, washer-dryers 3.0% / 40.3%, dryers 1.0% / 14.4%
(short programmes rarely hold six matches past half their length). Dropping every
stored copy of the held-out run moved only store packages (one trace stored under
two names): the run before kept them, 16 alarms (1.7%), 39.5% (1957) caught,
43.3% nearest, the same folds. The held-out
programme's usual duration cleared before its rebuild (the base learned it from
the held-out cycle too) left out 12 own-history folds whose rebuild learns none;
the run before, with them: 958 folds, 39.3% of 4981 caught, 42.8% nearest, the
same 16 alarms. Reading the fire fraction from the cycle start moved no median.
Each fold on its own copy of the cycles, the end check on the stored offset and
full-precision rows left every figure as it was (one pair of 4981 moved at
0.12 / 3 / 0.25).
The replay before it matched at fixed interval marks (an unchanged prefix again on a silent
plug) and left out picks with no trace: 15 (1.6%), 3 own-history, 41.6% of 4855
at 0.62. Per tick (that replay) the deficit separates a wrong pick from the right
one at AUC 0.89, the pick's own score at 0.79.

**Not modelled**: the keepalives the watchdog injects into a silent low-power
stretch (each a new reading live, so a check) where the recorded trace has none;
the stall cut the detector applies to its match trace (``_match_readings``); a
reading that arrives while a check is still matching; a restart.

Exit codes: 0, 2 no corpus, 3 a ``--from-rows`` file whose ticks lack their stored
offset (written before it was kept: match again).
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Iterator  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom_components.ha_washdata import match_rules, trigger_rules  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
    PROGRAM_MISMATCH_MIN_FRACTION,
    PROGRAM_MISMATCH_MIN_MARGIN,
    PROGRAM_MISMATCH_PERSISTENCE,
    TRIGGER_PROGRAM_MISMATCH,
)

_EVAL_PATH = REPO / "devtools" / "eval.py"
_spec = importlib.util.spec_from_file_location("wd_program_mismatch_eval_corpus", _EVAL_PATH)
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev
_spec.loader.exec_module(ev)

DEFAULT_CORPUS = REPO / "cycle_data"
#: What a tick keeps of each listed candidate (all the rule reads).
_CAND_KEYS = ("name", "score", "stage5_group", "group_best_member")
SHIPPED = (PROGRAM_MISMATCH_MIN_MARGIN, PROGRAM_MISMATCH_PERSISTENCE, PROGRAM_MISMATCH_MIN_FRACTION)
SWEEP_MARGINS = (0.08, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30)
SWEEP_PERSISTENCE = (2, 3, 4, 5, 6)
SWEEP_FRACTIONS = (0.15, 0.25, 0.35, 0.5)


# ----------------------------------------------------------------------- replay

def check_prefixes(pts: list[list[float]], interval: float,
                   end: float) -> Iterator[tuple[list[list[float]], float]]:
    """``(prefix, elapsed s)`` for each background check the manager would run on
    this trace, one at a time: on the first reading at least ``interval`` after the
    last check (``manager._schedule_program_mismatch_check``), strictly before
    ``end``. A stretch without a reading runs no check, so a silent plug never feeds
    the persistence rule one unchanged prefix several times.

    ``end`` is the recorded duration, measured from the cycle start, so the end
    check reads the reading's stored offset (``prefix[-1][0]``); the cadence and
    the rule read seconds since the first reading, what the manager's trace spans."""
    if not pts:
        return
    t0 = pts[0][0]
    last: float | None = None
    for i, p in enumerate(pts):
        if not p[0] < end:
            break
        elapsed = p[0] - t0
        if i < 1 or elapsed < (interval if last is None else last + interval):
            continue
        last = elapsed
        yield [list(x) for x in pts[:i + 1]], elapsed


def recorded_duration(cyc: dict, pts: list) -> float | None:
    """The cycle's end in seconds from its start, the clock the end check reads
    (``check_prefixes``), or None when there is none to use.

    The recorded duration when it is a positive number. Without one (absent, null,
    zero or negative) the trace's final offset, on that same clock: the span from
    the first reading ended a trace that starts late that much early. A duration
    that is not a finite number (an imported ``1e400`` reads as ``inf``: no check
    would reach its end, and the rows would no longer be valid JSON) is None."""
    try:
        dur = float(cyc.get("duration") or 0.0)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(dur):
        return None
    if dur <= 0:
        dur = float(pts[-1][0]) if len(pts) > 1 else 0.0
    return dur if math.isfinite(dur) and dur > 0 else None


def _usable_duration(cyc: Any) -> bool:
    """False for a record whose stored ``duration`` is there but no finite number:
    an int ``float`` cannot hold (``10**400`` raises OverflowError when the folds
    are selected), ``inf``/``nan``, a string or a null (the rebuild's
    ``duration > 60`` raises). An absent one is the store's "unknown"."""
    if not isinstance(cyc, dict) or "duration" not in cyc:
        return True
    raw = cyc["duration"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return False
    try:
        return math.isfinite(float(raw))
    except OverflowError:
        return False


def usable_device(dev: Any) -> Any:
    """``dev`` without the records whose duration is unusable (``_usable_duration``),
    in every list a rebuild reads and in its cycles, so one malformed record neither
    aborts the run nor trains a fold. Applied where the folds are selected and where
    a worker loads the device, so both see the same cycles in the same order."""
    if dev is None:
        return None
    if all(_usable_duration(c) for key, _ in ev.LISTS for c in dev.data.get(key) or []):
        return dev
    data = dict(dev.data)
    for key, _ in ev.LISTS:
        if isinstance(data.get(key), list):
            data[key] = [c for c in data[key] if _usable_duration(c)]
    keep = [k for k, (c, _e) in enumerate(dev.cycles) if _usable_duration(c)]
    return dataclasses.replace(
        dev, data=data, cycles=[dev.cycles[k] for k in keep],
        hashes=[dev.hashes[k] for k in keep])


def private_device(dev: Any) -> Any:
    """``dev`` with its own copy of the store data. A rebuild cuts imported
    dishwasher cycles in place (``meta.tail_cut_s``, #518): on the loaded objects
    the cut would reach every later fold, and a worker's folds would differ from
    another's by which rebuilds it happened to run."""
    return dataclasses.replace(dev, data=pickle.loads(
        pickle.dumps(dev.data, protocol=pickle.HIGHEST_PROTOCOL)))


def fold_store_data(dev: Any, base: dict, blob: bytes, pos: tuple[str, int],
                    copies: Any = ()) -> dict:
    """The store without the held-out cycle (``pos``: its list and index) or any
    other stored copy of it (``copies``: more positions, ``stored_copies``), on this
    fold's own copy of every other cycle (``blob``: the lists as loaded, pickled),
    of the profiles and of the envelope map, so nothing a fold's rebuild changes
    reaches the next fold (``private_device``).

    The base learned the usual duration of every programme holding a dropped copy
    from it, so those are cleared: only what the fold's rebuild of them
    (``held_programmes``) learns from the others may size the gate
    (``device_folds`` skips a fold whose rebuild learns none for the label)."""
    d = ev.base_data(dev)
    lists = pickle.loads(blob)
    drop = {pos, *copies}
    held = held_programmes(lists, drop)
    lists = {k: [c for m, c in enumerate(cs) if (k, m) not in drop] for k, cs in lists.items()}
    d.update(lists)
    d["profiles"] = pickle.loads(pickle.dumps(base["profiles"], protocol=pickle.HIGHEST_PROTOCOL))
    for name in held:
        prof = d["profiles"].get(name)
        if isinstance(prof, dict):
            prof.pop("avg_duration", None)
    d["envelopes"] = dict(base["envelopes"])
    return d


def mismatch_offered(device_type: Any) -> bool:
    """Whether the manager runs the check on this device type at all
    (``manager._schedule_program_mismatch_check`` returns first otherwise)."""
    return trigger_rules.trigger_offered(TRIGGER_PROGRAM_MISMATCH, device_type)


def held_programmes(lists: dict[str, list], drop: Any) -> list[str]:
    """The programmes the records at ``drop`` (list, index) are labelled with."""
    names = {lists[k][m].get("profile_name") for k, m in drop if isinstance(lists[k][m], dict)}
    return sorted(n for n in names if isinstance(n, str) and n)


def run_hash(pts: list) -> str | None:
    """A run's trace content from its ``eval.query_points``, at the precision a
    stored cycle keeps: each offset from the first reading, to 0.1 s (what the
    detector stores and every legacy format normalises to), each reading as
    stored. From the first reading because the reference-cycle writer rebases
    an imported trace to offset 0 (``ProfileStore._add_reference_cycle_nosave``):
    a run stored from 30 s and its reference copy are one run (PR #529 review).
    So every stored format of one run agrees (``eval.trace_hash`` reads the
    stored rows and has none for legacy ``(iso_str, power)`` ones), and two
    distinct runs on one sample grid never do: ``trace_hash``'s 1 W rounding made
    a 0.08 kW and a 0.15 kW run one trace, and holding out either dropped both
    (PR #529 review)."""
    if not pts:
        return None
    try:
        t0 = float(pts[0][0])
        rows = [[round(float(t) - t0, 1) + 0.0, float(p) + 0.0] for t, p in pts]
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()


def stored_copies(lists: dict[str, list]) -> dict[str, list[tuple[str, int]]]:
    """Every list position per trace content (``run_hash``), so a fold drops each
    stored copy of the held-out run, under any name, in any list and any stored
    format (PR #529 review: a second copy trained the rebuild on the trace it then
    scored)."""
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, cs in lists.items():
        for m, c in enumerate(cs):
            h = run_hash(ev.query_points(c)) if isinstance(c, dict) else None
            if h:
                out[h].append((key, m))
    return out


async def device_folds(dev: Any, base: dict, idxs: list[int]) -> list[dict]:
    """One row per held-out cycle: its label, durations and the tick rankings.

    Each fold reads the held-out cycle as loaded and trains on its own copy of the
    others (``fold_store_data``), so a row depends neither on the folds run before
    it nor on ``--jobs``."""
    lists = {key: list(dev.data.get(key) or []) for key, _ in ev.LISTS}
    where = {id(c): (key, j) for key, cs in lists.items() for j, c in enumerate(cs)}
    blob = pickle.dumps(lists, protocol=pickle.HIGHEST_PROTOCOL)
    copies = stored_copies(lists)
    ids = ev.fold_ids(dev)
    aliases = ev.alias_classes(dev)
    rows: list[dict] = []
    for i in idxs:
        cyc, _evidence = dev.cycles[i]
        label = cyc["profile_name"]
        pts = ev.query_points(cyc)
        dur = recorded_duration(cyc, pts)
        if len(pts) < ev.MIN_TRACE_POINTS or dur is None or id(cyc) not in where:
            continue
        drop = {where[id(cyc)], *copies.get(run_hash(pts) or "", ())}
        fd = fold_store_data(dev, base, blob, where[id(cyc)], drop)
        st, mgr = ev.fresh_store(dev, fd, {})
        if not mismatch_offered(mgr.device_type):
            return []  # never checked on this type (``replay`` leaves such a device out)
        # The label, and any other programme a copy of the run was stored under.
        for name in held_programmes(lists, drop):
            await st.async_rebuild_envelope(name)
        interval = float(mgr._profile_match_interval or 300)  # noqa: SLF001
        # Every programme a user can pick, and the learned durations: a pick
        # without one, or without a template, is never checked (a miss). The
        # label's is the rebuild's, from the other cycles (``fold_store_data``):
        # none learned, no fold.
        programs = sorted(str(n) for n in (st.get_profiles() or {}))
        durations = {}
        for name, prof in (st.get_profiles() or {}).items():
            avg = match_rules.profile_duration(prof.get("avg_duration")) if isinstance(prof, dict) else None
            if avg:
                durations[str(name)] = avg
        if label not in durations:
            continue
        ticks = []
        for q, qdur in check_prefixes(pts, interval, dur):
            res = await st.async_match_profile(q, qdur, in_progress=True)
            # Full precision (``t`` since the first reading, ``at`` the stored
            # offset the end check reads): a rounded tick could reach the rounded end.
            ticks.append({
                "t": qdur,
                "at": q[-1][0],
                "best": res.best_profile,
                "c": [{k2: c.get(k2) for k2 in _CAND_KEYS if c.get(k2) is not None}
                      for c in (res.candidates or [])],
            })
        rows.append({
            "path": dev.key, "id": ids[i], "dev": mgr.device_type, "label": label,
            "duration": dur, "interval": interval, "durations": durations,
            "programs": programs,
            "families": match_rules.program_families(st.get_profile_groups()),
            # The manager never checks a pick the matcher cannot score.
            "unmatchable": sorted(st.unmatchable_profiles()),
            "twins": sorted(aliases.get(label, ())),
            "ticks": ticks,
        })
    return rows


_W: dict[str, Any] = {}


def _worker_init(root: str) -> None:
    _W["root"] = Path(root)
    _W["devs"] = {}  # loaded under this root only: the cache is keyed by relative path
    ev._quiet_logs()  # noqa: SLF001


def _device(path: str) -> Any:
    memo = _W.setdefault("devs", {})
    if path not in memo:
        memo[path] = usable_device(ev.load_device(_W["root"], path))
    return memo[path]


def _job_base(path: str) -> tuple[str, dict]:
    return path, ev.run_coro(ev.rebuild_base(private_device(_device(path)), {}))


def _job_folds(args: tuple[str, dict, list[int]]) -> list[dict]:
    path, base, idxs = args
    return ev.run_coro(device_folds(_device(path), base, idxs))


def replay(corpus: Path, device: str | None = None, max_per_device: int | None = None,
           jobs: int = 1, log=print) -> list[dict]:
    devices, _dropped = ev.load_corpus(corpus)
    # Durations validated before anything reads them (``ev.scorable`` converts
    # them), the same way the workers load each device (``_device``).
    devices = [usable_device(d) for d in devices]
    devices = [d for d in devices if len(d.labels) >= 2 and (not device or device in d.path)]
    # Only a device type the manager runs the check on (a Threshold Device never
    # does), before any fold: its picks are neither scored nor missed.
    devices = [d for d in devices if mismatch_offered(ev.fresh_store(d, {}, {})[1].device_type)]
    twins = ev.twin_entries(devices)
    devices = [d for d in devices if d.path not in twins]
    targets: dict[str, list[int]] = {}
    for d in devices:
        idx = ev.scorable(d)
        if max_per_device:
            idx = ev.fast_pick(d, idx, max_per_device)
        if idx:
            targets[d.path] = idx
    paths = sorted(targets, key=lambda p: -len(targets[p]))
    log(f"[mismatch] {len(paths)} devices, {sum(len(v) for v in targets.values())} cycles", file=sys.stderr)
    rows: list[dict] = []
    if jobs <= 1:
        _worker_init(str(corpus))
        for p in paths:
            _, base = _job_base(p)
            rows += _job_folds((p, base, targets[p]))
        return sorted(rows, key=lambda r: (r["path"], r["id"]))
    with ProcessPoolExecutor(jobs, initializer=_worker_init, initargs=(str(corpus),)) as ex:
        bases = dict(ex.map(_job_base, paths))
        chunks = [(p, bases[p], targets[p][k:k + 4]) for p in paths for k in range(0, len(targets[p]), 4)]
        for part in ex.map(_job_folds, chunks):
            rows += part
    return sorted(rows, key=lambda r: (r["path"], r["id"]))


# ---------------------------------------------------------------------- scoring

#: Corpus folders of community store downloads (several users' runs per file),
#: reported apart from the folders that hold one appliance's own history.
STORE_DIRS = ("store",)


def _wrong_picks(row: dict) -> list[str]:
    """Every other programme of the device (alias twins of the label excluded),
    with or without a learned duration or a template: a user can pick any of them,
    and one the manager cannot check is a miss, not a pair left out."""
    twins = set(row.get("twins") or ())
    names = set(row.get("programs") or ()) | set(row["durations"])
    return sorted(s for s in names if s != row["label"] and s not in twins)


def _nearest_wrong(row: dict) -> str | None:
    """The wrong pick closest to the label in **proportion** of usual duration
    (``|ln(pick / label)|``, then name): 30 min reads as close to 40 as 60 to 80."""
    wrong = [s for s in _wrong_picks(row) if row["durations"].get(s)]
    if not wrong:
        return None
    own = row["durations"][row["label"]]
    return min(wrong, key=lambda s: (abs(math.log(row["durations"][s] / own)), s))


def pick_sequence(row: dict, selected: str) -> list[tuple[float, Any, float, float]]:
    """``(fraction of the pick's usual duration, selected_program_deficit, elapsed s,
    stored offset s)`` per tick: what ``match_rules.program_mismatch_tick`` reads,
    computed once, and where the end check reads the tick. Empty for a pick the
    manager does not check (no template, no learned duration). A tick without its
    stored offset raises KeyError (``rows_have_offsets``)."""
    if selected in (row.get("unmatchable") or ()) or not row["durations"].get(selected):
        return []
    expected = float(row["durations"][selected])
    fam = row.get("families") or {}
    return [
        (tick["t"] / expected,
         match_rules.selected_program_deficit(
             SimpleNamespace(best_profile=tick["best"], candidates=tick["c"]), selected, fam),
         tick["t"], tick["at"])
        for tick in row["ticks"]
    ]


def rows_have_offsets(rows: list[dict]) -> bool:
    """Whether every tick keeps its stored offset (``at``). A rows file written
    before it was kept holds only ``t``, the seconds since the first reading: on a
    trace that starts late, read as the offset it puts a tick past the cycle end
    inside it. Such a file is matched again, never rescored."""
    return all(isinstance(t, dict) and "at" in t for r in rows for t in r.get("ticks") or ())


def fire_at(seq: list[tuple[float, Any, float, float]], margin: float, persistence: int,
            fraction: float, end: float | None = None) -> float | None:
    """The stored offset (seconds from the cycle start) of the tick at which the
    rule fires on ``seq`` strictly before ``end`` (the cycle's recorded end, on the
    same clock; None: no bound), or None (``match_rules.program_mismatch_step``,
    the tick's own rule, which reads the seconds since the first reading)."""
    state = match_rules.ProgramMismatchState()
    for frac, hit, _t, at in seq:
        if end is not None and not at < end:
            break
        if match_rules.program_mismatch_step(
            state, hit, frac, min_margin=margin, persistence=persistence, min_fraction=fraction
        ):
            return at
    return None


def prepare(rows: list[dict]) -> list[dict]:
    """Each row's pick sequences: its own label, every wrong pick, the nearest."""
    out = []
    for r in rows:
        nearest = _nearest_wrong(r)
        out.append({
            "dev": r["dev"], "duration": r["duration"],
            "store": r["path"].split("/")[0] in STORE_DIRS,
            "own": pick_sequence(r, r["label"]),
            "wrong": [pick_sequence(r, s) for s in _wrong_picks(r)],
            "nearest": pick_sequence(r, nearest) if nearest else None,
        })
    return out


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 3) if xs else None


def _slices(prep: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in prep:
        groups["all"].append(p)
        groups["store packages" if p["store"] else "own history"].append(p)
    for p in prep:
        groups[p["dev"]].append(p)
    return groups


def score(prep: list[dict], margin: float, persistence: int, fraction: float,
          only: str | None = None) -> dict[str, Any]:
    """The rule's figures: pooled (``all``), by corpus kind, by device type."""
    out: dict[str, Any] = {}
    for name, grp in _slices(prep).items():
        if only is not None and name != only:
            continue
        fa = det = pairs = n_det = n_pairs = 0
        fracs: list[float] = []
        n_fracs: list[float] = []
        for p in grp:
            end = p["duration"]
            if fire_at(p["own"], margin, persistence, fraction, end) is not None:
                fa += 1
            # The fraction of the cycle at the fire: its offset from the cycle
            # start, the duration's clock.
            for seq in p["wrong"]:
                pairs += 1
                at = fire_at(seq, margin, persistence, fraction, end)
                if at is not None:
                    det += 1
                    fracs.append(at / p["duration"])
            if p["nearest"] is not None:
                n_pairs += 1
                at = fire_at(p["nearest"], margin, persistence, fraction, end)
                if at is not None:
                    n_det += 1
                    n_fracs.append(at / p["duration"])
        out[name] = {
            "cycles": len(grp),
            "false_alarm": fa, "false_alarm_rate": round(fa / len(grp), 4) if grp else None,
            "pairs": pairs, "detect": det, "detect_rate": round(det / pairs, 4) if pairs else None,
            "elapsed_frac": _median(fracs),
            "nearest_pairs": n_pairs, "nearest_detect": n_det,
            "nearest_rate": round(n_det / n_pairs, 4) if n_pairs else None,
            "nearest_elapsed_frac": _median(n_fracs),
        }
    return out


def sweep(prep: list[dict]) -> list[dict]:
    out = []
    for m in SWEEP_MARGINS:
        for k in SWEEP_PERSISTENCE:
            for f in SWEEP_FRACTIONS:
                s = score(prep, m, k, f, only="all")["all"]
                own = score(prep, m, k, f, only="own history").get("own history") or {}
                out.append({"margin": m, "persistence": k, "min_fraction": f, **s,
                            "own_history_false_alarm_rate": own.get("false_alarm_rate"),
                            "own_history_detect_rate": own.get("detect_rate")})
    return out


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"{'device':16s} {'cycles':>6s} {'false':>11s} {'wrong picks':>18s} {'frac':>6s}"
          f" {'nearest by ratio':>16s} {'frac':>6s}")
    for name, s in summary.items():
        print(f"{name:16s} {s['cycles']:6d} {s['false_alarm']:4d} {100 * (s['false_alarm_rate'] or 0):5.1f}%"
              f" {s['detect']:5d}/{s['pairs']:<5d} {100 * (s['detect_rate'] or 0):5.1f}%"
              f" {s['elapsed_frac'] if s['elapsed_frac'] is not None else '-':>6}"
              f" {s['nearest_detect']:4d}/{s['nearest_pairs']:<4d} {100 * (s['nearest_rate'] or 0):5.1f}%"
              f" {s['nearest_elapsed_frac'] if s['nearest_elapsed_frac'] is not None else '-':>6}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--device", default=None, help="only corpus files whose path contains this")
    ap.add_argument("--max-per-device", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--from-rows", type=Path, default=None)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args(argv)
    if a.from_rows:
        rows = json.loads(a.from_rows.read_text(encoding="utf-8"))["rows"]
        if not rows_have_offsets(rows):
            print(f"{a.from_rows}: ticks without their stored offset (written before it "
                  "was kept): run the replay again", file=sys.stderr)
            return 3
    else:
        if not a.corpus.is_dir():
            print(f"no corpus at {a.corpus}", file=sys.stderr)
            return 2
        rows = replay(a.corpus, a.device, a.max_per_device, a.jobs)
    if not rows:
        print("no scorable cycle", file=sys.stderr)
        return 2
    prep = prepare(rows)
    summary = score(prep, *SHIPPED)
    print(f"shipped: margin {SHIPPED[0]}, persistence {SHIPPED[1]}, min fraction {SHIPPED[2]}")
    _print_summary(summary)
    grid = sweep(prep) if a.sweep else None
    if grid is not None:
        print("\nmargin persist minfrac  false%  detect%  frac  nearest%  frac | own history false% detect%"
              "  (nearest: the closest usual duration by ratio)")
        for g in grid:
            mark = " <- shipped" if (g["margin"], g["persistence"], g["min_fraction"]) == SHIPPED else ""
            print(f"{g['margin']:6.2f} {g['persistence']:7d} {g['min_fraction']:7.2f}"
                  f" {100 * (g['false_alarm_rate'] or 0):7.2f} {100 * (g['detect_rate'] or 0):8.1f}"
                  f" {g['elapsed_frac'] if g['elapsed_frac'] is not None else '-':>5}"
                  f" {100 * (g['nearest_rate'] or 0):9.1f}"
                  f" {g['nearest_elapsed_frac'] if g['nearest_elapsed_frac'] is not None else '-':>5}"
                  f" | {100 * (g['own_history_false_alarm_rate'] or 0):17.2f}"
                  f" {100 * (g['own_history_detect_rate'] or 0):7.1f}{mark}")
    if a.json:
        a.json.write_text(json.dumps({"summary": summary, "sweep": grid, "rows": rows}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
