#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Are the two computed-but-unrendered profile advisories worth showing?

    python3 devtools/advisory_usefulness_eval.py [--jobs N] [--only SUBSTR]
                                                  [--corpus DIR] [--out FILE]

Measures, on the ``cycle_data/`` corpus (eval.py's loader, de-cloning and its
real-``WashDataManager`` store builder, so options resolve as in production):

A. ``ProfileStore.suggest_coverage_gaps`` ("these N unmatched cycles look like a
   programme you have not created"). Its clusters are ``profile_suggestions``.

   * Held-out recall: per device with >= 2 programmes, each programme P with >= 3
     traced ``past_cycles`` is deleted (profile, envelope, its reference/backfill
     cycles) and its past cycles are unlabelled, as if the user never created P.
     A hit is a cluster >= 70% made of P's cycles holding >= 2 of them. Four
     scenarios: ``oracle`` (every P cycle unmatched) or ``realistic`` (each traced
     P cycle is matched against the store without P and keeps the WRONG label when
     the live cycle-end gate - ``label_verdict`` at the device's learning
     confidence - accepts it) or ``live`` (realistic, AND every other scorable past
     cycle carries its simulated live label, below), crossed with ``recent`` (P's
     cycles moved to the newest end of the list, inside the 30-cycle window) or
     ``natural`` (stored order). The primary number is oracle/recent; ``live`` is
     the honest precision test, since only there can known programmes' unlabelled
     cycles compete. Each oracle/recent miss is diagnosed on P's largest 15-min
     duration bucket, including its shape correlation resampled by sample index
     (as shipped) versus by timestamp.
   * Precision: every cluster from those runs, classed held-out (>= 70% P),
     unlabelled (>= 70% cycles the export itself left unlabelled), other
     programme (a false alarm), or mixture.
   * False alarms on intact stores: ``as-is`` (the export's own unlabelled cycles)
     and ``simulated`` (each scorable past cycle is relabelled by its own
     leave-one-out complete-cycle match through the same learning gate; a refused
     cycle is unmatched, exactly what the live cycle-end path stores). Every
     simulated cluster belongs to a programme that already exists.

B. ``ProfileStore.suggest_profile_groups`` (near-duplicate profile groups for the
   Stage-5 hierarchical match).

   * What it proposes on the intact store (envelopes rebuilt with current code),
     with the user's own groups (``as-is``, what a user would see) and without
     them (``scratch``); each suggestion's ``group_cohesion`` (below
     GROUP_MIN_COHESION the matcher ignores it) and whether accepting it would be
     rejected because a member already sits in another group.
   * Ground truth: the shipped matcher's confusion, leave-one-cycle-out on every
     scorable COMPLETE cycle with no groups at all. A pair is confused when >= 2
     of the pair's folds were matched as the other AND that is >= 10% of the
     pair's folds. A suggestion is useful when, counted the same way, its members
     are matched as each other in >= 2 folds and >= 10% of its members' folds;
     otherwise pointless. Confused pairs no suggestion covers are misses.
   * Accuracy effect, same folds: top-1 and group-level top-1 with ``none`` (no
     groups), ``user`` (the export's groups), ``user+sug`` (plus every suggestion
     computed on that fold's store and accepted the way the panel would save it:
     union into ``existing_group``, refused on a cross-group conflict) and ``sug``
     (scratch suggestions only). Suggestions are recomputed per fold after the
     held-out cycle's profile is rebuilt, so no fold sees its own cycle.

Does NOT measure: whether a user would act on a suggestion, the setup-card
wording, mid-cycle (prefix) matching, or live persistence/switching; group
effects are top-1 on complete cycles only. The coverage-gap ground truth is the
exported labels, which are partly the matcher's own guesses (provenance is
reported); untraced past cycles of P stay unmatched in every scenario. Nothing
is cached and nothing outside ``--out`` is written; the full corpus runs in
about 3 minutes on 8 cores. The ``user`` variant reproduces ``eval.py run
--mode full --cuts 1.0`` top-1 / group top-1 exactly (same folds, same path).

Exit codes: 0 ok, 2 no corpus.
"""
from __future__ import annotations

import os

# Before NumPy is imported: one BLAS thread per worker (determinism + throughput).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as E  # noqa: E402

PURITY = 0.70              # a cluster "is" a programme when >= this share of it
MIN_HELD_CYCLES = 3        # traced past cycles a held-out programme needs
CONFUSED_MIN_COUNT = 2     # cross-matches for a pair / group to count as confused
CONFUSED_MIN_RATE = 0.10   # ... and their share of the pair's / group's folds
UNLABELLED = "?unlabelled"
VARIANTS = ("none", "user", "user+sug", "sug")
SCENARIOS = (("oracle", "recent"), ("oracle", "natural"), ("realistic", "recent"), ("realistic", "natural"),
             ("live", "recent"), ("live", "natural"))
OUT_DEFAULT = E.DEFAULT_CACHE / "advisory_usefulness.json"


# ------------------------------------------------------------------- coverage gaps

def _scenario_past(past: list[dict], held: str | None, order: str,
                   relabel: dict[int, str | None] | None = None,
                   confs: dict[int, float] | None = None) -> list[dict]:
    """Copies of ``past`` with synthetic ids ``x<i>``; ``held``'s cycles unlabelled
    (or relabelled from ``relabel``), moved to the newest end for ``recent``.
    ``confs`` replaces ``match_confidence`` with the simulated match's, which is
    what the live cycle end stores (``label_confidence``)."""
    out, tail = [], []
    for i, c in enumerate(past):
        c2 = dict(c)
        c2["id"] = f"x{i}"
        if confs is not None and i in confs:
            c2["match_confidence"] = confs[i]
        if relabel is not None and i in relabel:
            c2["profile_name"] = relabel[i]
        elif held is not None and c.get("profile_name") == held:
            c2["profile_name"] = None
        (tail if order == "recent" and held is not None and c.get("profile_name") == held else out).append(c2)
    return out + tail


# Keyword arguments for suggest_coverage_gaps (``--gap-kwargs``), set per worker.
_GAP_KW: dict = {}


def _gaps(dev: E.Device, past: list[dict]) -> dict:
    st, _ = E.fresh_store(dev, {"past_cycles": past}, {})
    return st.suggest_coverage_gaps(**_GAP_KW) or {}


def _clusters(res: dict, truth: dict[str, str]) -> list[dict]:
    out = []
    for s in res.get("profile_suggestions") or []:
        comp = Counter(truth.get(str(cid), UNLABELLED) for cid in s.get("cycle_ids") or [])
        out.append({"n": sum(comp.values()), "comp": dict(comp),
                    "avg_min": round(float(s.get("avg_duration_s") or 0) / 60, 1),
                    "sim": s.get("similarity")})
    return out


def _cluster_class(cl: dict, held: str | None) -> str:
    n = max(1, cl["n"])
    top, k = Counter(cl["comp"]).most_common(1)[0] if cl["comp"] else (None, 0)
    if held is not None and cl["comp"].get(held, 0) / n >= PURITY:
        return "heldout"
    if k / n >= PURITY:
        return "unlabelled" if top == UNLABELLED else "other"
    return "mixture"


# ------------------------------------------------------------------------ groups

def _sig(groups: dict) -> str:
    return json.dumps(sorted((k, sorted(v.get("members") or [])) for k, v in groups.items()))


def _accept(groups: dict, suggestions: list[dict]) -> tuple[dict, list[dict]]:
    """Apply suggestions the way the panel saves them: union into ``existing_group``
    (else a new group); a member owned by ANOTHER group makes the save fail
    (``_members_in_other_groups``), so that suggestion is refused."""
    new = copy.deepcopy(groups)
    owner = {m: g for g, v in new.items() for m in v.get("members") or []}
    log = []
    for k, s in enumerate(suggestions):
        target = s.get("existing_group") or f"__suggested_{k + 1}"
        members = sorted(set(s["members"]) | set((new.get(target) or {}).get("members") or []))
        conflicts = sorted(m for m in members if owner.get(m) not in (None, target))
        log.append({"members": sorted(s["members"]), "into": s.get("existing_group"),
                    "refused": conflicts or None})
        if conflicts:
            continue
        new[target] = {"members": members}
        for m in members:
            owner[m] = target
    return new, log


def _describe(st, suggestions: list[dict], groups: dict) -> list[dict]:
    from custom_components.ha_washdata.const import GROUP_MIN_COHESION  # noqa: PLC0415
    _, log = _accept(groups, suggestions)
    out = []
    for s, a in zip(suggestions, log):
        coh = float(st.group_cohesion(list(s["members"])))
        out.append({"members": sorted(s["members"]), "existing_group": s.get("existing_group"),
                    "cohesion": round(coh, 3), "cohesive": coh >= GROUP_MIN_COHESION,
                    "refused": a["refused"]})
    return out


# ------------------------------------------------------------------------ device job

async def _match(st, pts: list, dur: float, learn: float) -> dict:
    from custom_components.ha_washdata.profile_store import label_verdict  # noqa: PLC0415
    E._CAPTURE.clear()  # noqa: SLF001
    res = await st.async_match_profile([list(p) for p in pts], dur, in_progress=False)
    pre = list(E._CAPTURE.get("pre") or [])  # noqa: SLF001
    mapped = {k: list(v) for k, v in (E._CAPTURE.get("groups") or {}).items()}  # noqa: SLF001
    return {"top1": res.best_profile, "pre3": pre[:3], "mapped": mapped,
            "conf": round(float(res.label_confidence or 0.0), 4),
            "mg": round(float(res.ambiguity_margin or 0.0), 4), "amb": bool(res.is_ambiguous),
            "verdict": label_verdict(res, learn)[0]}


def _has_group_suggestions() -> bool:
    """``suggest_profile_groups`` was deleted in 0.5.8 on this harness's own numbers
    (register item 432). Part B then runs without it; check out ``c006c92`` to re-run
    the group measurement."""
    from custom_components.ha_washdata.profile_store import ProfileStore  # noqa: PLC0415
    return hasattr(ProfileStore, "suggest_profile_groups")


async def _device(dev: E.Device) -> dict:
    from custom_components.ha_washdata.profile_store import _AUTO_LABEL_SOURCES  # noqa: PLC0415
    base = await E.rebuild_base(dev, {})
    full = E.base_data(dev)
    full["profiles"], full["envelopes"] = base["profiles"], base["envelopes"]
    user_groups = {k: {"members": list(v.get("members") or [])}
                   for k, v in (dev.data.get("profile_groups") or {}).items() if isinstance(v, dict)}
    full["profile_groups"] = {}
    st0, mgr0 = E.fresh_store(dev, {**full, "profile_groups": copy.deepcopy(user_groups)}, {})
    learn = float(mgr0._learning_confidence or 0.0)  # noqa: SLF001
    out: dict[str, Any] = {"key": dev.key, "dev": mgr0.device_type, "user": dev.user,
                           "programs": len(dev.labels), "user_groups": user_groups,
                           "learn_floor": learn}

    has_sug = _has_group_suggestions()
    # B1: what the intact store would suggest.
    if len(dev.labels) >= 2 and has_sug:
        out["sug_asis"] = _describe(st0, st0.suggest_profile_groups(), user_groups)
        st0._data["profile_groups"] = {}  # noqa: SLF001
        st0._cohesion_cache_generation += 1  # noqa: SLF001
        out["sug_scratch"] = _describe(st0, st0.suggest_profile_groups(), {})
        st0._data["profile_groups"] = copy.deepcopy(user_groups)  # noqa: SLF001
        st0._cohesion_cache_generation += 1  # noqa: SLF001

    # B2: leave-one-cycle-out folds, every scorable complete cycle, four group configs.
    past = [c for c in dev.data.get("past_cycles") or [] if isinstance(c, dict)]
    past_idx = {id(c): i for i, c in enumerate(past)}
    folds = []
    for i in E.scorable(dev):
        cyc, ev = dev.cycles[i]
        label = cyc["profile_name"]
        pts = E.query_points(cyc)
        dur = E.cycle_duration(cyc, pts)
        fd = E.fold_data(full, cyc)
        own = sum(1 for key, _ in E.LISTS for x in fd[key]
                  if isinstance(x, dict) and x.get("profile_name") == label and x.get("power_data"))
        if len(pts) < E.MIN_TRACE_POINTS or dur <= 0 or own == 0:
            continue
        st, _ = E.fresh_store(dev, fd, {})
        await st.async_rebuild_envelope(label)
        cfg: dict[str, dict] = {"none": {}, "user": copy.deepcopy(user_groups)}
        sugs: dict[str, list] = {}
        if len(dev.labels) >= 2 and has_sug:
            for name, start in (("user+sug", user_groups), ("sug", {})):
                st._data["profile_groups"] = copy.deepcopy(start)  # noqa: SLF001
                st._cohesion_cache_generation += 1  # noqa: SLF001
                sugs[name] = st.suggest_profile_groups()
                cfg[name], _ = _accept(start, sugs[name])
        else:
            cfg["user+sug"], cfg["sug"] = cfg["user"], cfg["none"]
        by_sig: dict[str, dict] = {}
        row: dict[str, Any] = {"label": label, "ev": ev, "past_i": past_idx.get(id(cyc)),
                               "prov": E.provenance(cyc, ev, _AUTO_LABEL_SOURCES),
                               "sug_sig": {k: _sig({str(j): {"members": s["members"]} for j, s in enumerate(v)})
                                           for k, v in sugs.items()}}
        for v in VARIANTS:
            s = _sig(cfg[v])
            if s not in by_sig:
                st._data["profile_groups"] = copy.deepcopy(cfg[v])  # noqa: SLF001
                st._cohesion_cache_generation += 1  # noqa: SLF001
                by_sig[s] = await _match(st, pts, dur, learn)
            m = by_sig[s]
            member_group = {x: g for g, ms in m["mapped"].items() for x in ms}
            ok = m["top1"] == label
            row[v] = {"top1": m["top1"], "ok": ok, "pre3": m["pre3"], "mg": m["mg"], "amb": m["amb"],
                      "verdict": m["verdict"], "conf": m["conf"],
                      "gok": ok or (label in member_group and member_group.get(m["top1"]) == member_group[label])}
        folds.append(row)
    out["folds"] = folds

    # A: coverage gaps (past_cycles only - the function reads nothing else).
    if past:
        truth = {f"x{i}": (c.get("profile_name") or UNLABELLED) for i, c in enumerate(past)}
        out["gaps_asis"] = _clusters(_gaps(dev, _scenario_past(past, None, "natural")), truth)
        # The export's own unlabelled traced cycles, matched against the intact store.
        unl = []
        for i, c in enumerate(past):
            if not c.get("profile_name") and c.get("power_data"):
                pts = E.query_points(c)
                dur = E.cycle_duration(c, pts)
                if len(pts) >= E.MIN_TRACE_POINTS and dur > 0:
                    m = await _match(st0, pts, dur, learn)
                    unl.append({"i": i, "top1": m["top1"], "verdict": m["verdict"]})
        out["unlabelled_verdicts"] = unl
        # Simulated live labels: each scorable past cycle relabelled by its own LOO match.
        relabel = {r["past_i"]: r["user"]["verdict"] for r in folds if r["past_i"] is not None}
        sim_conf = {r["past_i"]: r["user"]["conf"] for r in folds if r["past_i"] is not None}
        sim = _gaps(dev, _scenario_past(past, None, "natural", relabel, sim_conf))
        out["gaps_sim"] = _clusters(sim, truth)
        out["sim_unmatched"] = sum(1 for v in relabel.values() if v is None)
        out["sim_mislabelled"] = sum(1 for i, v in relabel.items() if v is not None and v != past[i]["profile_name"])
        out["sim_scored"] = len(relabel)
        held_rows = []
        traced = Counter(c["profile_name"] for c in past if c.get("profile_name") and c.get("power_data"))
        if len(dev.labels) >= 2:
            for P in sorted(n for n, k in traced.items() if k >= MIN_HELD_CYCLES):
                held_rows.append(await _held_out(dev, full, user_groups, past, truth, P, learn, relabel, sim_conf))
        out["held"] = held_rows
    return out


async def _held_out(dev: E.Device, full: dict, user_groups: dict, past: list[dict],
                    truth: dict[str, str], P: str, learn: float, sim: dict[int, str | None],
                    sim_conf: dict[int, float] | None = None) -> dict:
    """The store as if the user never created P, and what the gap finder makes of it."""
    data = {k: v for k, v in full.items()}
    data["profiles"] = {k: v for k, v in full["profiles"].items() if k != P}
    data["envelopes"] = {k: v for k, v in full["envelopes"].items() if k != P}
    for key, _ in E.LISTS[1:]:
        data[key] = [c for c in full[key] if not (isinstance(c, dict) and c.get("profile_name") == P)]
    data["past_cycles"] = [dict(c, profile_name=None) if c.get("profile_name") == P else c for c in past]
    data["profile_groups"] = copy.deepcopy(user_groups)
    st, _ = E.fresh_store(dev, data, {})
    relabel: dict[int, str | None] = {}
    p_conf: dict[int, float] = {}
    for i, c in enumerate(past):
        if c.get("profile_name") != P:
            continue
        relabel[i] = None
        if c.get("power_data"):
            pts = E.query_points(c)
            dur = E.cycle_duration(c, pts)
            if len(pts) >= E.MIN_TRACE_POINTS and dur > 0:
                m = await _match(st, pts, dur, learn)
                relabel[i] = m["verdict"]
                p_conf[i] = m["conf"]
    row: dict[str, Any] = {"P": P, "n": sum(1 for c in past if c.get("profile_name") == P),
                           "n_traced": sum(1 for c in past if c.get("profile_name") == P and c.get("power_data")),
                           "absorbed": dict(Counter(v for v in relabel.values() if v)),
                           "manual_share": round(sum(1 for c in past if c.get("profile_name") == P
                                                     and (c.get("label_source") == "manual"
                                                          or (c.get("ml_review") or {}).get("golden")))
                                                 / max(1, len(relabel)), 2)}
    live = {**{i: v for i, v in sim.items() if past[i].get("profile_name") != P}, **relabel}
    live_conf = {**{i: v for i, v in (sim_conf or {}).items() if past[i].get("profile_name") != P}, **p_conf}
    for kind, order in SCENARIOS:
        res = _gaps(dev, _scenario_past(past, P, order, {"oracle": None, "realistic": relabel, "live": live}[kind],
                                        {"oracle": p_conf, "realistic": p_conf, "live": live_conf}[kind]))
        cls = _clusters(res, truth)
        for cl in cls:
            cl["class"] = _cluster_class(cl, P)
        row[f"{kind}/{order}"] = {
            "clusters": cls, "unmatched": res.get("unmatched_count", 0),
            "suggest_create": bool(res.get("suggest_create")),
            "hit": any(cl["comp"].get(P, 0) >= 2 and cl["comp"].get(P, 0) / max(1, cl["n"]) >= PURITY
                       for cl in cls),
        }
    if not row["oracle/recent"]["hit"]:
        row["miss_diag"] = _shape_diag(_scenario_past(past, P, "recent")[-30:], P, truth)
    return row


def _shape_diag(window: list[dict], P: str, truth: dict[str, str]) -> dict:
    """Why P was missed: its largest 15-min duration bucket in the window, and the
    bucket's mean pairwise correlation as shipped (resampled by SAMPLE INDEX) and
    resampled on the timestamps instead."""
    import numpy as np  # noqa: PLC0415
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        CLUSTER_RESAMPLE_N,
        CLUSTER_SHAPE_SIMILARITY_THRESHOLD,
    )
    from custom_components.ha_washdata.profile_store import decompress_power_data  # noqa: PLC0415
    from custom_components.ha_washdata.signal_processing import resample_to_n  # noqa: PLC0415
    mine = [c for c in window if truth.get(c["id"]) == P and isinstance(c.get("duration"), (int, float))
            and c["duration"] > 0]
    buckets = Counter(int(float(c["duration"]) // 900.0) for c in mine)
    if not buckets:
        return {"reason": "no P cycle in window"}
    b, _ = buckets.most_common(1)[0]
    traces_i, traces_t = [], []
    for c in [c for c in mine if c.get("power_data") and int(float(c["duration"]) // 900.0) == b][:5]:
        raw = sorted(decompress_power_data(c) or [], key=lambda x: x[0])
        if len(raw) < 5:
            continue
        t = np.asarray([float(x[0]) for x in raw])
        w = np.asarray([float(x[1]) for x in raw])
        for arr, out in ((np.asarray(resample_to_n(list(w), CLUSTER_RESAMPLE_N)), traces_i),
                         (np.interp(np.linspace(t[0], t[-1], CLUSTER_RESAMPLE_N), t, w), traces_t)):
            out.append(arr / (arr.max() or 1.0))
    if len(traces_i) < 2:
        return {"reason": "P cycles split across duration buckets", "buckets": len(buckets)}

    def mean_corr(tr: list) -> float:
        cs = [float(np.corrcoef(tr[i], tr[j])[0, 1]) for i in range(len(tr)) for j in range(i + 1, len(tr))]
        cs = [x for x in cs if np.isfinite(x)]
        return round(float(np.mean(cs)), 3) if cs else float("nan")
    ci, ct = mean_corr(traces_i), mean_corr(traces_t)
    thr = CLUSTER_SHAPE_SIMILARITY_THRESHOLD
    reason = ("shape: index-resampled corr below threshold, time-resampled above" if ci < thr <= ct else
              "shape: below threshold either way" if ci < thr else "passes shape (other cause)")
    return {"reason": reason, "corr_index": ci, "corr_time": ct, "n": len(traces_i)}


_ROOT: list[str] = []


def _init(root: str, gap_kw: str = "{}") -> None:
    _ROOT[:] = [root]
    _GAP_KW.clear()
    _GAP_KW.update(json.loads(gap_kw))
    E._quiet_logs()  # noqa: SLF001
    E.install_instrumentation()


def _job(path: str) -> dict:
    dev = E.load_device(Path(_ROOT[0]), path)
    assert dev is not None, path
    return E.run_coro(_device(dev))


# ------------------------------------------------------------------------ reporting

def _p(a: float, b: float) -> str:
    return f"{100.0 * a / b:5.1f}%" if b else "    -"


def _report_gaps(devs: list[dict]) -> dict:
    out: dict[str, Any] = {}
    print("\n== A. suggest_coverage_gaps ==")
    held = [(d, h) for d in devs for h in d.get("held") or []]
    print(f"held-out programmes: {len(held)} on {len({d['key'] for d, _ in held})} devices "
          f"(>= {MIN_HELD_CYCLES} traced past cycles, device >= 2 programmes); hit = cluster >= "
          f"{PURITY:.0%} P with >= 2 P cycles")
    hdr = "".join(f"{k + '/' + o:>20}" for k, o in SCENARIOS)
    print(f"{'recall':<24}{hdr}")
    slices: dict[str, list] = defaultdict(list)
    for d, h in held:
        slices["ALL"].append(h)
        slices[f"dev={d['dev']}"].append(h)
        n = h["n_traced"]
        slices["P size " + ("3-4" if n < 5 else "5-9" if n < 10 else ">=10")].append(h)
        slices["P manual>=50%" if h["manual_share"] >= 0.5 else "P manual<50%"].append(h)
    for s in sorted(slices, key=lambda x: (x != "ALL", x)):
        hs = slices[s]
        cells = "".join(f"{f'{sum(h[f'{k}/{o}']['hit'] for h in hs)}/{len(hs)} {_p(sum(h[f'{k}/{o}']['hit'] for h in hs), len(hs))}':>20}"
                        for k, o in SCENARIOS)
        print(f"{s:<24}{cells}")
    ndev = len({d["key"] for d, _ in held})
    dev_hits = [len({d["key"] for d, h in held if h[f"{k}/{o}"]["hit"]}) for k, o in SCENARIOS]
    print(f"{'devices with >= 1 hit':<24}" + "".join(f"{f'{n}/{ndev}':>20}" for n in dev_hits))
    print(f"{'precision (clusters)':<24}" + hdr)
    prec = {}
    for k, o in SCENARIOS:
        c = Counter(cl["class"] for _, h in held for cl in h[f"{k}/{o}"]["clusters"])
        prec[f"{k}/{o}"] = dict(c)
    for cls in ("heldout", "unlabelled", "other", "mixture"):
        cells = "".join(f"{f'{prec[s].get(cls, 0)}/{sum(prec[s].values())} {_p(prec[s].get(cls, 0), sum(prec[s].values()))}':>20}"
                        for s in prec)
        print(f"  {cls:<22}{cells}")
    hits = [h for _, h in held if h["oracle/recent"]["hit"]]
    frag = Counter(sum(1 for cl in h["oracle/recent"]["clusters"] if cl["class"] == "heldout") for h in hits)
    print(f"oracle/recent: clusters per hit programme {dict(sorted(frag.items()))}")
    tot_tr = sum(h["n_traced"] for _, h in held)
    absorbed = sum(sum(h["absorbed"].values()) for _, h in held)
    print(f"realistic: {absorbed}/{tot_tr} ({_p(absorbed, tot_tr).strip()}) of held-out traced cycles pass the "
          f"live label gate as ANOTHER programme (invisible to the gap finder)")
    misses = [(d, h) for d, h in held if not h["oracle/recent"]["hit"]]
    why = Counter()
    for d, h in misses:
        r = h["oracle/recent"]
        # unmatched_count is only reported when the function returns a result, so 0
        # means it returned {} (below min_unmatched, whatever --gap-kwargs set).
        why["too few unmatched (function returns {})" if not r["unmatched"] else
            "clusters but none P-pure" if r["clusters"] else "no shape-similar duration bucket"] += 1
    print(f"oracle/recent misses by reason: {dict(why)}")
    diag = Counter(h["miss_diag"]["reason"] for _, h in misses if "miss_diag" in h)
    print(f"oracle/recent misses, P's largest duration bucket: {dict(diag)}")
    # Intact stores.
    with_past = [d for d in devs if "gaps_asis" in d]
    for name, key in (("intact as-is", "gaps_asis"), ("intact simulated-live", "gaps_sim")):
        fired = [d for d in with_past if d[key]]
        comp = Counter({"other": "existing programme"}.get(c, c)
                       for d in fired for c in (_cluster_class(cl, None) for cl in d[key]))
        print(f"{name}: fires on {len(fired)}/{len(with_past)} devices with past cycles, "
              f"{sum(len(d[key]) for d in fired)} clusters, dominated by {dict(comp)}")
        for d in fired:
            print(f"    {d['dev']:<16} {d['key'][:44]:<44} " + "; ".join(
                f"{cl['n']} cyc ~{cl['avg_min']}m {cl['comp']}" for cl in d[key]))
    sim_u = sum(d.get("sim_unmatched", 0) for d in with_past)
    sim_m = sum(d.get("sim_mislabelled", 0) for d in with_past)
    sim_n = sum(d.get("sim_scored", 0) for d in with_past)
    print(f"simulated-live labels: {sim_u}/{sim_n} scorable past cycles refused by the learning gate "
          f"(unmatched), {sim_m} mislabelled")
    uv = [u for d in with_past for u in d.get("unlabelled_verdicts") or []]
    print(f"export-unlabelled traced past cycles: {len(uv)}; the intact store's gate would label "
          f"{sum(1 for u in uv if u['verdict'])} of them")
    out.update(precision=prec, held=len(held))
    return out


def _confusion(folds: list[dict]) -> tuple[Counter, Counter]:
    n_by = Counter(r["label"] for r in folds)
    cross = Counter()
    for r in folds:
        t = r["none"]["top1"]
        if t and t != r["label"]:
            cross[frozenset((r["label"], t))] += 1
    return n_by, cross


def _useful(members: list[str], folds: list[dict]) -> tuple[int, int]:
    ms = set(members)
    mine = [r for r in folds if r["label"] in ms]
    return sum(1 for r in mine if r["none"]["top1"] in ms - {r["label"]}), len(mine)


def _report_groups(devs: list[dict]) -> dict:
    print("\n== B. suggest_profile_groups ==")
    multi = [d for d in devs if d["programs"] >= 2 and "sug_asis" in d]
    print(f"thresholds: confused = >= {CONFUSED_MIN_COUNT} cross-matches AND >= {CONFUSED_MIN_RATE:.0%} of the "
          f"folds (pair or group); LOO on every scorable complete cycle, no groups")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in multi:
        by_type[d["dev"]].append(d)
    for t in sorted(by_type):
        ds = by_type[t]
        print(f"  {t:<16} devices {len(ds):3d}  fires as-is {sum(1 for d in ds if d['sug_asis']):3d}  "
              f"fires scratch {sum(1 for d in ds if d['sug_scratch']):3d}")
    print(f"  {'ALL':<16} devices {len(multi):3d}  fires as-is {sum(1 for d in multi if d['sug_asis']):3d}  "
          f"fires scratch {sum(1 for d in multi if d['sug_scratch']):3d}")
    verdicts = Counter()
    rows_out = []
    print("suggestions on intact stores (as-is unless marked scratch):")
    for d in multi:
        _, cross = _confusion(d["folds"])
        seen = set()
        for kind in ("sug_asis", "sug_scratch"):
            for s in d[kind]:
                key = tuple(s["members"])
                if kind == "sug_scratch" and key in seen:
                    continue
                seen.add(key)
                k, n = _useful(s["members"], d["folds"])
                useful = k >= CONFUSED_MIN_COUNT and n and k / n >= CONFUSED_MIN_RATE
                # Judgeable only when >= 2 members have folds (else no cross-match can exist).
                judge = sum(1 for m in s["members"] if any(r["label"] == m for r in d["folds"])) >= 2
                pairs = [(a, b) for i, a in enumerate(s["members"]) for b in s["members"][i + 1:]]
                cp = sum(1 for a, b in pairs if cross[frozenset((a, b))])
                if kind == "sug_asis":
                    verdicts["useful" if useful else "pointless" if judge else "unjudgeable"] += 1
                rows_out.append({"key": d["key"], "kind": kind, **s, "within": k, "member_folds": n,
                                 "useful": bool(useful)})
                print(f"  {d['dev']:<16} {d['key'][:40]:<40} {'scratch ' if kind == 'sug_scratch' else ''}"
                      f"{s['members']} into={s['existing_group']} coh={s['cohesion']}"
                      f"{'' if s['cohesive'] else ' (INERT <0.80)'}{' REFUSED ' + str(s['refused']) if s['refused'] else ''}"
                      f" | within-confusion {k}/{n}, pairs ever confused {cp}/{len(pairs)} -> "
                      f"{'USEFUL' if useful else 'pointless' if judge else 'unjudgeable (< 2 members scored)'}")
    tot = sum(verdicts.values())
    jt = tot - verdicts["unjudgeable"]
    print(f"as-is precision (useful/suggested): {verdicts['useful']}/{tot} {_p(verdicts['useful'], tot).strip()}; "
          f"judgeable only {verdicts['useful']}/{jt} {_p(verdicts['useful'], jt).strip()}")
    # Confused pairs and coverage by suggestions / user groups.
    print("confused pairs (no groups) and who covers them:")
    missed = covered = 0
    for d in multi:
        n_by, cross = _confusion(d["folds"])
        sug_sets = [set(s["members"]) for s in d["sug_asis"] + d["sug_scratch"]]
        ug_sets = [set(v["members"]) for v in d["user_groups"].values()]
        for pair, k in sorted(cross.items(), key=lambda kv: -kv[1]):
            a, b = sorted(pair)
            n = n_by[a] + n_by[b]
            if k < CONFUSED_MIN_COUNT or k / n < CONFUSED_MIN_RATE:
                continue
            in_s = any(pair <= s for s in sug_sets)
            in_u = any(pair <= s for s in ug_sets)
            covered += in_s
            missed += not in_s
            print(f"  {d['dev']:<16} {d['key'][:40]:<40} {a!r} <-> {b!r}: {k}/{n} "
                  f"suggested={'Y' if in_s else 'N'} user_group={'Y' if in_u else 'N'}")
    print(f"confused pairs covered by a suggestion: {covered}/{covered + missed}")
    # Accuracy effect.
    print("top-1 / group top-1 on complete cycles, LOO (per-fold suggestions):")
    print(f"  {'slice':<44}{'n':>5}" + "".join(f"{v:>16}" for v in VARIANTS))
    groups_rows = [r for d in multi for r in d["folds"]]

    def line(name: str, rows: list[dict]) -> None:
        cells = "".join(f"{f'{_p(sum(r[v]['ok'] for r in rows), len(rows)).strip()}/{_p(sum(r[v]['gok'] for r in rows), len(rows)).strip()}':>16}"
                        for v in VARIANTS)
        print(f"  {name:<44}{len(rows):>5}{cells}")

    line("ALL", groups_rows)
    for t in sorted(by_type):
        line(f"dev={t}", [r for d in by_type[t] for r in d["folds"]])
    changed = [d for d in multi if any(r["user+sug"] != r["user"] or r["sug"] != r["none"] for r in d["folds"])]
    for d in changed:
        line(f"{d['dev'][:10]} {d['key'][:32]}", d["folds"])
    for a, b in (("user", "user+sug"), ("none", "sug"), ("none", "user")):
        better = sum(1 for r in groups_rows if r[b]["ok"] and not r[a]["ok"])
        worse = sum(1 for r in groups_rows if r[a]["ok"] and not r[b]["ok"])
        print(f"  {a} -> {b}: top-1 +{better}/-{worse} folds, McNemar p {E.mcnemar_p(better, worse):.3f}")
    vary = sum(1 for d in multi for r in d["folds"]
               if r["sug_sig"].get("user+sug") != _sig({str(j): {"members": s["members"]}
                                                       for j, s in enumerate(d["sug_asis"])}))
    print(f"folds whose own suggestions differ from the intact-store suggestions: {vary}/{len(groups_rows)}")
    return {"suggestions": rows_out, "precision": dict(verdicts)}


# ------------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=E.REPO / "cycle_data")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--only", default=None, help="restrict to devices whose path contains this")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help=f"result JSON (default {OUT_DEFAULT})")
    ap.add_argument("--gap-kwargs", default="{}",
                    help='JSON kwargs for suggest_coverage_gaps, e.g. \'{"min_unmatched": 3}\'')
    a = ap.parse_args(argv)
    if not a.corpus.is_dir():
        print(f"no corpus at {a.corpus} (cycle_data/ is gitignored: symlink or pass --corpus)", file=sys.stderr)
        return 2
    t0 = time.time()
    devices, dropped = E.load_corpus(a.corpus)
    devices = [d for d in devices if d.labels and (not a.only or a.only in d.path)]
    if not devices:
        print("no labelled device in the corpus", file=sys.stderr)
        return 2
    paths = sorted((d.path for d in devices), key=lambda p: -len(next(x for x in devices if x.path == p).cycles))
    with ProcessPoolExecutor(max(1, a.jobs), initializer=_init, initargs=(str(a.corpus), a.gap_kwargs)) as ex:
        res = sorted(ex.map(_job, paths), key=lambda r: r["key"])
    print(f"[advisory] {len(res)} devices ({len(dropped)} clone files dropped), "
          f"{sum(len(r['folds']) for r in res)} LOO folds, {time.time() - t0:.1f}s, rev {E._git_rev()}")  # noqa: SLF001
    gaps = _report_gaps(res)
    if _has_group_suggestions():
        groups = _report_groups(res)
    else:
        groups = {}
        print("\n== B. suggest_profile_groups == deleted (register item 432); skipped")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"meta": {"rev": E._git_rev(), "code_sha": E.code_sha(),  # noqa: SLF001
                                          "purity": PURITY, "confused": [CONFUSED_MIN_COUNT, CONFUSED_MIN_RATE]},
                                 "gaps": gaps, "groups": groups, "devices": res},
                                default=str, sort_keys=True), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    # String hashing is per-process random; pin it so set order cannot make runs differ.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    raise SystemExit(main())
