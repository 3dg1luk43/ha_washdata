#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Leave-one-cycle-out evaluation of the SHIPPED matcher (audit item F1).

    python3 devtools/eval.py run [--mode fast|full] [--cuts 1.0,0.75,0.5,0.25]
                                 [--jobs N] [--out FILE] [--config-override JSON|@FILE]
    python3 devtools/eval.py compare BASE.json NEW.json [--tol TOL.json]

Every fold is the path users run. The device's store is built by a real
``WashDataManager`` from the export's own entry data/options (so ratios, DTW band,
``energy_mode``, evidence sources and label thresholds come out exactly as
``manager.py`` resolves them - never a partial config). All envelopes are rebuilt
once with the current code; then, per held-out cycle, the cycle is removed from
every list, ITS profile is rebuilt with ``ProfileStore.async_rebuild_envelope`` and
the cycle (or a prefix of it, ``in_progress=True``) is matched with
``ProfileStore.async_match_profile``: snapshot builder, Stage 1-5, the 12-point
floor, ambiguity and ``label_confidence`` included. The only instrumentation is a
pass-through wrapper on ``collapse_group_candidates`` that records the full
pre-collapse ranking (for top-3 / rank). Storage is stubbed; nothing is written
except the result file and the cache.

Measures, per cut of the cycle's own duration: top-1, top-3, group-level top-1,
no-candidate rate, the post-cycle auto-label gate (margin >= MATCH_LABEL_MIN_MARGIN,
not ambiguous, label_confidence >= the device's auto_label_confidence) as coverage
and precision (``learn_*``: the same gate at learning_confidence, the live
cycle-end bar), margin and confidence AUC as correctness predictors, and Stage-5
group wins (label inside the winning group, picked member right). Sliced by device
type, user, evidence list and label provenance.

Does NOT measure: end detection (end lag, early ends, splits - use
``end_gate_eval.py``), the live switching/persistence state machine
(``decisive_margin_eval.py``), the prefix guard (``prefix_guard_eval.py``), ETA,
or the ML providers. Prefix cuts are matched once at that elapsed time, not
through the 5-minute live cadence. Store data is not run through the storage
migration (old exports are used as stored).

``--mode fast``: every device with >= 2 programmes, 8 scorable cycles each spread
over its programmes by a stable trace hash - a regression detector, not the number
of record. ``--mode full``: every scorable fold (the held-out cycle's programme
keeps at least one other traced cycle). Clone files (>= 80% of their traces
already in a larger file) are dropped.

``--config-override`` takes a JSON object (or ``@file``):

    {"options": {"profile_match_max_duration_ratio": 1.8}}      # every device
    {"options_by_device": {"dishwasher": {"dtw_bandwidth": 0.1}}}
    {"const": {"MATCH_DURATION_WEIGHT": 0.15}}

``options`` are merged over each export's entry options before the manager builds
the store (what a settings change does). ``const`` rebinds the name in ``const``
and in every loaded integration module that imported it by name; a value captured
at import time (a default argument, a dict built from other constants) is NOT
affected - check the code you are tuning before trusting a null result.

Results are cached under ``~/.cache/ha_washdata_eval`` (``--cache-dir``,
``--no-cache``) keyed by a hash of the matcher-relevant sources (the transitive
local imports of ``profile_store.py``, plus ``manager.py``, ``options_utils.py``
and this file), the override, NumPy's version and each corpus file's bytes, so any
code change invalidates it. ``compare`` pairs folds by (source path, cycle id,
cut), prints deltas with McNemar exact p and a device-cluster paired bootstrap 95%
CI, and exits 1 when a guarded metric drops beyond its tolerance (``--tol`` file,
else the tolerances recorded in BASE). File names under ``user-Contributed/`` carry
contributors' names, so result files key those devices by a hash of the path.

Exit codes: 0 ok, 1 guarded regression (compare), 2 no corpus / nothing to pair.
"""
from __future__ import annotations

import os

# Before NumPy is imported: one BLAS thread per worker (determinism + throughput).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterator  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PKG_NAME = "custom_components.ha_washdata"
PKG_DIR = REPO / "custom_components" / "ha_washdata"

HARNESS_VERSION = 1
DEFAULT_CUTS = (1.0, 0.75, 0.5, 0.25)
FAST_PER_DEVICE = 8
CLONE_FRACTION = 0.8
MIN_TRACE_POINTS = 8      # a fold needs this many readings
MIN_PREFIX_POINTS = 5     # and a prefix this many (the 12-point floor is the store's)
BOOTSTRAP_ROUNDS = 2000
#: metric -> max allowed drop in percentage points (rise, for no_candidate). Sized
#: for the fast baseline (~310 folds per cut, one fold ~0.32pp; the auto-label
#: gate passes ~45 of them, one fold ~2pp). Recorded in every result file; compare
#: uses BASE's unless --tol is given.
DEFAULT_TOLERANCES = {
    "top1@1.0": 0.5,
    "top1@0.75": 1.0,
    "top1@0.5": 1.0,
    "top1@0.25": 1.5,
    "learn_precision@1.0": 1.0,
    "label_precision@1.0": 3.0,
    "no_candidate@1.0": 0.5,
}
LOWER_IS_BETTER = ("no_candidate",)
LISTS = (("past_cycles", "past"), ("reference_cycles", "reference"), ("backfill_cycles", "backfill"))
DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ha_washdata_eval"


# --------------------------------------------------------------------------- corpus

@dataclass
class Device:
    """One exported config entry: its store data, entry data/options and cycles."""

    path: str
    user: str
    fmt: str
    fsha: str
    entry_data: dict
    entry_options: dict
    data: dict
    cycles: list = field(default_factory=list)   # [(cycle, evidence)]
    hashes: list = field(default_factory=list)

    @property
    def labels(self) -> set[str]:
        return {c["profile_name"] for c, _ in self.cycles}

    @property
    def key(self) -> str:
        return public_key(self.path)


#: Corpus folders whose file names carry contributors' real names. Result files (the
#: committed baseline included) key these by a hash of the path instead.
PRIVATE_DIRS = ("user-Contributed",)


def public_key(rel: str) -> str:
    top = rel.split("/")[0]
    if top in PRIVATE_DIRS:
        return f"{top}/{hashlib.sha1(rel.encode()).hexdigest()[:12]}"
    return rel


def _unwrap(doc: dict) -> tuple[dict, dict, dict, str] | None:
    """(store data, entry data, entry options, format) for the three export shapes."""
    data = doc.get("data")
    if not isinstance(data, dict):
        return None
    entry = data.get("entry") if isinstance(data.get("entry"), dict) else {}
    se = data.get("store_export")
    if isinstance(se, dict) and isinstance(se.get("data"), dict):   # diagnostics dump
        ed = {**(entry.get("data") or {}), **(se.get("entry_data") or {})}
        eo = {**(entry.get("options") or {}), **(se.get("entry_options") or {})}
        return se["data"], ed, eo, "diagnostics"
    sd = data.get("store_data")
    if isinstance(sd, dict) and "past_cycles" in sd:                  # legacy diagnostics
        return sd, dict(entry.get("data") or {}), dict(entry.get("options") or {}), "legacy"
    if isinstance(data.get("past_cycles"), list) or isinstance(data.get("reference_cycles"), list):
        ed = dict(doc.get("entry_data") or {})
        eo = dict(doc.get("entry_options") or {})
        fp = doc.get("device_fingerprint")
        if not (eo.get("device_type") or ed.get("device_type")) and isinstance(fp, dict) and fp.get("device_type"):
            ed["device_type"] = fp["device_type"]
        return data, ed, eo, "export"
    return None


def trace_hash(c: dict) -> str | None:
    """Stable hash of a trace rounded to 1 s / 1 W (clone detection, fast pick)."""
    try:
        s = json.dumps([[round(float(p[0]), 0), round(float(p[1]), 0)] for p in c.get("power_data") or []])
    except (TypeError, ValueError, IndexError):
        return None
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def load_device(root: Path, rel: str) -> Device | None:
    path = root / rel
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    un = _unwrap(doc)
    if un is None:
        return None
    data, ed, eo, fmt = un
    dev = Device(rel, rel.split("/")[0], fmt, hashlib.sha256(raw).hexdigest()[:16], ed, eo, data)
    for key, ev in LISTS:
        for c in data.get(key) or []:
            if isinstance(c, dict) and c.get("profile_name") and isinstance(c.get("power_data"), list) \
                    and len(c["power_data"]) >= 4:
                dev.cycles.append((c, ev))
    dev.hashes = [trace_hash(c) for c, _ in dev.cycles]
    return dev


def load_corpus(root: Path) -> tuple[list[Device], dict[str, str]]:
    """All devices, minus clone files. Returns (kept sorted by path, {dropped: kept-twin})."""
    devices = [d for d in (load_device(root, str(p.relative_to(root)))
                           for p in sorted(root.rglob("*.json"))) if d is not None]
    seen: dict[str, str] = {}
    kept: list[Device] = []
    dropped: dict[str, str] = {}
    for dev in sorted(devices, key=lambda d: -len(d.cycles)):   # stable: ties keep path order
        hs = {h for h in dev.hashes if h}
        dup = [h for h in sorted(hs) if h in seen]
        if hs and len(dup) / len(hs) >= CLONE_FRACTION:
            dropped[dev.path] = seen[dup[0]]
            continue
        for h in hs:
            seen.setdefault(h, dev.path)
        kept.append(dev)
    return sorted(kept, key=lambda d: d.path), dropped


def provenance(c: dict, evidence: str, auto_sources: tuple[str, ...]) -> str:
    """reference (store) / manual (user-set or golden) / auto (matcher's guess) / unknown."""
    if evidence == "reference":
        return "reference"
    src = c.get("label_source")
    golden = isinstance(c.get("ml_review"), dict) and bool(c["ml_review"].get("golden"))
    if src == "manual" or golden:
        return "manual"
    if src in auto_sources or c.get("auto_labeled") is True:
        return "auto"
    return "unknown"


# --------------------------------------------------------------------- code identity

def matcher_sources() -> list[Path]:
    """Transitive local imports of profile_store.py, + manager/options_utils + this file."""
    todo = [PKG_DIR / "profile_store.py"]
    seen: set[Path] = set()
    while todo:
        f = todo.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level < 1:
                continue
            base = f.parent
            for _ in range(node.level - 1):
                base = base.parent
            parts = node.module.split(".") if node.module else []
            target = base.joinpath(*parts)
            cands = [target.with_suffix(".py"), target / "__init__.py"]
            if not node.module:
                cands = [base / f"{a.name}.py" for a in node.names]
            else:
                cands += [target / f"{a.name}.py" for a in node.names]
            todo.extend(c for c in cands if c.exists() and PKG_DIR in c.parents)
    seen.update({PKG_DIR / "manager.py", PKG_DIR / "options_utils.py", Path(__file__).resolve()})
    return sorted(seen)


def code_sha() -> str:
    h = hashlib.sha256()
    for f in matcher_sources():
        h.update(str(f.relative_to(REPO)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip() or "?"
    except OSError:
        return "?"


# ----------------------------------------------------------------- overrides / patches

_MISSING = object()


def parse_override(text: str | None) -> dict:
    if not text:
        return {}
    if text.startswith("@"):
        text = Path(text[1:]).read_text(encoding="utf-8")
    ov = json.loads(text)
    if not isinstance(ov, dict) or set(ov) - {"options", "options_by_device", "const"}:
        raise SystemExit("--config-override: expected an object with options / options_by_device / const")
    return ov


@contextlib.contextmanager
def const_overrides(consts: dict | None) -> Iterator[None]:
    """Rebind constants in const and every loaded integration module that imported them."""
    if not consts:
        yield
        return
    import custom_components.ha_washdata.const as C  # noqa: PLC0415
    for mod in ("manager", "profile_store", "analysis"):   # load the consumers before patching
        importlib.import_module(f"{PKG_NAME}.{mod}")
    undo: list[tuple[Any, str, Any]] = []
    try:
        for name, val in consts.items():
            orig = getattr(C, name, _MISSING)
            if orig is _MISSING:
                raise SystemExit(f"--config-override: const.{name} does not exist")
            if isinstance(orig, tuple) and isinstance(val, list):
                val = tuple(val)
            for mname, mod in list(sys.modules.items()):
                if mod is not None and (mname == PKG_NAME or mname.startswith(PKG_NAME + ".")) \
                        and getattr(mod, name, _MISSING) is orig:
                    undo.append((mod, name, orig))
                    setattr(mod, name, val)
        yield
    finally:
        for mod, name, orig in reversed(undo):
            setattr(mod, name, orig)


_CAPTURE: dict[str, Any] = {}


def install_instrumentation() -> None:
    """Pass-through wrapper on collapse_group_candidates recording the full ranking."""
    from custom_components.ha_washdata import profile_store as ps  # noqa: PLC0415
    if getattr(ps.collapse_group_candidates, "_eval_wrapped", False):
        return
    orig = ps.collapse_group_candidates

    def wrapped(candidates, group_members):
        _CAPTURE["pre"] = [str(c.get("name")) for c in candidates]
        _CAPTURE["groups"] = {k: list(v) for k, v in (group_members or {}).items()}
        return orig(candidates, group_members)

    wrapped._eval_wrapped = True  # type: ignore[attr-defined]
    wrapped._eval_orig = orig  # type: ignore[attr-defined]
    ps.collapse_group_candidates = wrapped


def uninstall_instrumentation() -> None:
    from custom_components.ha_washdata import profile_store as ps  # noqa: PLC0415
    orig = getattr(ps.collapse_group_candidates, "_eval_orig", None)
    if orig is not None:
        ps.collapse_group_candidates = orig


# ------------------------------------------------------------------ store construction

class _Entry:
    def __init__(self, data: dict, options: dict, title: str) -> None:
        self.data, self.options, self.title = data, options, title
        self.entry_id, self.domain = "eval", "ha_washdata"

    def async_on_unload(self, *_a, **_k) -> None:
        return None

    def add_update_listener(self, *_a, **_k):
        return lambda: None


class _InlineHass:
    """Executor jobs run inline: deterministic, and nothing else is reachable."""

    async def async_add_executor_job(self, fn, *args):
        return fn(*args)


class _NullStore:
    """Replaces the WashDataStore: every save is dropped (no ./MagicMock/ files)."""

    async def async_save(self, _data) -> None:
        return None

    async def async_load(self):
        return None

    def async_delay_save(self, *_a, **_k) -> None:
        return None


def entry_dicts(dev: Device, override: dict) -> tuple[dict, dict]:
    data = {"power_sensor": "sensor.eval_power", "name": "eval",
            **{k: v for k, v in dev.entry_data.items() if v is not None}}
    opts = {k: v for k, v in dev.entry_options.items() if v is not None}
    opts.update(override.get("options") or {})
    dtype = opts.get("device_type", data.get("device_type"))
    opts.update((override.get("options_by_device") or {}).get(dtype or "washing_machine", {}))
    return data, opts


def fresh_store(dev: Device, data: dict, override: dict):
    """(ProfileStore exactly as WashDataManager builds it, the manager)."""
    from custom_components.ha_washdata.manager import WashDataManager  # noqa: PLC0415
    ed, eo = entry_dicts(dev, override)
    mgr = WashDataManager(MagicMock(), _Entry(ed, eo, dev.path))
    st = mgr.profile_store
    st.hass = _InlineHass()
    st._store = _NullStore()  # noqa: SLF001
    st._data = data  # noqa: SLF001
    return st, mgr


def device_config(dev: Device, override: dict) -> dict:
    st, mgr = fresh_store(dev, {}, override)
    return {
        "user": dev.user, "dev": mgr.device_type, "fmt": dev.fmt,
        "cycles": len(dev.cycles), "profiles": len(dev.labels),
        "groups": len(dev.data.get("profile_groups") or {}),
        "min_ratio": st._min_duration_ratio, "max_ratio": st._max_duration_ratio,  # noqa: SLF001
        "dtw_bandwidth": st.dtw_bandwidth, "energy_mode": st.energy_mode,
        "evidence": list(st.evidence_sources),
        "auto_label_conf": float(mgr._auto_label_confidence),  # noqa: SLF001
        "learning_conf": float(mgr._learning_confidence),  # noqa: SLF001
    }


def base_data(dev: Device) -> dict:
    d = dict(dev.data)
    for key, _ in LISTS:
        d[key] = list(d.get(key) or [])
    d["profiles"] = {k: dict(v) if isinstance(v, dict) else v for k, v in (d.get("profiles") or {}).items()}
    d["envelopes"] = dict(d.get("envelopes") or {})
    d["profile_groups"] = d.get("profile_groups") or {}
    return d


async def rebuild_base(dev: Device, override: dict) -> dict:
    """Every envelope rebuilt with the current code (exports carry stale ones)."""
    data = base_data(dev)
    st, _ = fresh_store(dev, data, override)
    for name in list(data["profiles"]):
        await st.async_rebuild_envelope(name)
    return {"profiles": data["profiles"], "envelopes": data["envelopes"]}


def fold_data(base: dict, cycle: dict) -> dict:
    """The store without ``cycle`` (by identity), sharing everything it does not touch."""
    d = dict(base)
    for key, _ in LISTS:
        d[key] = [c for c in base[key] if c is not cycle]
    d["profiles"] = dict(base["profiles"])
    name = cycle.get("profile_name")
    if isinstance(d["profiles"].get(name), dict):
        d["profiles"][name] = dict(d["profiles"][name])
    d["envelopes"] = dict(base["envelopes"])
    return d


def query_points(c: dict) -> list[list[float]]:
    from custom_components.ha_washdata.profile_store import decompress_power_data  # noqa: PLC0415
    return [[t, p] for t, p in sorted(decompress_power_data(c), key=lambda x: x[0])]


def cycle_duration(c: dict, pts: list) -> float:
    try:
        d = float(c.get("duration") or 0.0)
    except (TypeError, ValueError):
        d = 0.0
    if d <= 0 and len(pts) > 1:
        d = pts[-1][0] - pts[0][0]
    return d


def scorable(dev: Device) -> list[int]:
    """Cycles with a usable trace whose programme keeps another traced cycle."""
    counts: Counter = Counter()
    for key, _ in LISTS:
        for c in dev.data.get(key) or []:
            if isinstance(c, dict) and c.get("profile_name") and c.get("power_data"):
                counts[c["profile_name"]] += 1
    out = []
    for i, (c, _) in enumerate(dev.cycles):
        pts = query_points(c)
        if len(pts) >= MIN_TRACE_POINTS and cycle_duration(c, pts) > 0 and counts[c["profile_name"]] >= 2:
            out.append(i)
    return out


def fast_pick(dev: Device, idxs: list[int], k: int = FAST_PER_DEVICE) -> list[int]:
    by: dict[str, list[int]] = defaultdict(list)
    for i in idxs:
        by[dev.cycles[i][0]["profile_name"]].append(i)
    for v in by.values():
        v.sort(key=lambda i: (dev.hashes[i] or "", str(dev.cycles[i][0].get("id"))))
    out: list[int] = []
    while len(out) < k and any(by.values()):
        for name in sorted(by):
            if by[name] and len(out) < k:
                out.append(by[name].pop(0))
    return sorted(out)


def fold_ids(dev: Device) -> list[str]:
    ids, seen = [], Counter()
    for i, (c, _) in enumerate(dev.cycles):
        fid = str(c.get("id") or f"idx{i}")
        seen[fid] += 1
        ids.append(fid if seen[fid] == 1 else f"{fid}#{i}")
    return ids


async def run_folds(dev: Device, base: dict, idxs: list[int], cuts: tuple[float, ...],
                    override: dict) -> dict[int, dict[str, dict | None]]:
    """{cycle index: {str(cut): row or None (unscorable at that cut)}}."""
    from custom_components.ha_washdata import const as C  # noqa: PLC0415
    from custom_components.ha_washdata.profile_store import _AUTO_LABEL_SOURCES  # noqa: PLC0415
    full = base_data(dev)   # same cycle objects as dev.cycles, so removal is by identity
    full["profiles"], full["envelopes"] = base["profiles"], base["envelopes"]
    ids = fold_ids(dev)
    out: dict[int, dict[str, dict | None]] = {}
    for i in idxs:
        cyc, ev = dev.cycles[i]
        label = cyc["profile_name"]
        pts = query_points(cyc)
        dur = cycle_duration(cyc, pts)
        res_i: dict[str, dict | None] = {str(c): None for c in cuts}
        out[i] = res_i
        fd = fold_data(full, cyc)
        own_n = sum(1 for key, _ in LISTS for x in fd[key]
                    if isinstance(x, dict) and x.get("profile_name") == label and x.get("power_data"))
        if len(pts) < MIN_TRACE_POINTS or dur <= 0 or own_n == 0:
            continue
        st, mgr = fresh_store(dev, fd, override)
        await st.async_rebuild_envelope(label)
        thr = float(mgr._auto_label_confidence)  # noqa: SLF001
        thr_learn = float(mgr._learning_confidence or 0.0)  # noqa: SLF001
        for cut in cuts:
            # A cut above 1.0 is an ABSOLUTE elapsed time in seconds (``5m`` -> 300):
            # the early-cycle window the fraction cuts never reach. A cycle shorter
            # than the cut is not scored there.
            if cut > 1.0:
                if cut >= dur:
                    continue
                q = [p for p in pts if p[0] <= pts[0][0] + cut]
            else:
                q = pts if cut >= 1.0 else [p for p in pts if p[0] <= pts[0][0] + cut * dur]
            if len(q) < MIN_PREFIX_POINTS or q[-1][0] - q[0][0] <= 0:
                continue
            partial = cut != 1.0
            qdur = q[-1][0] - q[0][0] if partial else dur
            _CAPTURE.clear()
            res = await st.async_match_profile([list(p) for p in q], qdur, in_progress=partial)
            pre = _CAPTURE.get("pre") or []
            groups = _CAPTURE.get("groups") or {}
            member_group = {m: g for g, ms in groups.items() for m in ms}
            top1 = res.best_profile
            ok = top1 is not None and top1 == label
            g = (res.ranking[0].get("stage5_group") if top1 and res.ranking else None) or None
            margin = float(res.ambiguity_margin or 0.0)
            lc = float(res.label_confidence or 0.0)
            gate = bool(top1 and margin >= C.MATCH_LABEL_MIN_MARGIN and not res.is_ambiguous)
            res_i[str(cut)] = {
                "path": dev.key, "id": ids[i], "cut": cut, "dev": mgr.device_type,
                "prov": provenance(cyc, ev, _AUTO_LABEL_SOURCES), "ev": ev,
                "label": label, "top1": top1, "ok": ok,
                "gok": ok or (label in member_group and member_group.get(top1) == member_group[label]),
                "rank": pre.index(label) + 1 if label in pre else None, "nc": len(pre),
                "conf": round(float(res.confidence or 0.0), 4), "mg": round(margin, 4),
                "amb": bool(res.is_ambiguous), "lc": round(lc, 4),
                # The post-cycle auto-label gate (manager.py, "Post-Cycle Auto-Labeling"),
                # and the same margin/ambiguity gate at learning_confidence - the bar
                # the live cycle-end label uses (applied here to this cut's match).
                "lok": gate and thr > 0 and lc >= thr,
                "lokl": gate and lc >= thr_learn,
                "g": g.removeprefix("__group__") if g else None,
                "gin": (label in groups.get(g, [])) if g else None,
            }
    return out


# ------------------------------------------------------------------------ worker pool

_W: dict[str, Any] = {}
_DEV_MEMO: dict[str, Device] = {}


def run_coro(coro):
    """Run on a private loop. Unlike asyncio.run this leaves the thread's current
    loop alone, so an in-process run cannot break a host that relies on it (pytest)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _quiet_logs() -> None:
    logging.getLogger(PKG_NAME).setLevel(logging.ERROR)
    logging.getLogger("homeassistant").setLevel(logging.ERROR)


def _worker_init(root: str, cache: str, key: str, override: dict) -> None:
    _W.update(root=Path(root), cache=Path(cache), key=key, override=override)
    _quiet_logs()
    install_instrumentation()
    if override.get("const") and not _W.get("const_cm"):
        cm = const_overrides(override["const"])
        cm.__enter__()  # held for the worker's lifetime
        _W["const_cm"] = cm


def _device(path: str) -> Device:
    """The device (inherited from the parent under fork, else re-read)."""
    mk = str(_W["root"] / path)
    dev = _DEV_MEMO.get(mk)
    if dev is None:
        dev = load_device(_W["root"], path)
        assert dev is not None, path
        _DEV_MEMO[mk] = dev
    return dev


def _base_file(dev: Device) -> Path:
    return _W["cache"] / f"base-{dev.fsha}-{_W['key']}.pkl"


def _job_base(path: str) -> str:
    dev = _device(path)
    f = _base_file(dev)
    if not f.exists():
        base = run_coro(rebuild_base(dev, _W["override"]))
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(base, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(f)
    return path


_BASE_MEMO: dict[str, dict] = {}


def _job_folds(args: tuple[str, list[int], tuple[float, ...]]):
    path, idxs, cuts = args
    dev = _device(path)
    f = str(_base_file(dev))
    base = _BASE_MEMO.get(f)
    if base is None:
        base = pickle.loads(Path(f).read_bytes())
        while len(_BASE_MEMO) >= 3:
            _BASE_MEMO.pop(next(iter(_BASE_MEMO)))
        _BASE_MEMO[f] = base
    return path, run_coro(run_folds(dev, base, idxs, cuts, _W["override"]))


def _execute(jobs: int, init: tuple, bases: list[str], chunks: list[tuple]) -> list[tuple[str, dict]]:
    """Run base rebuilds, then each device's fold chunks as soon as its base exists."""
    if jobs <= 1:
        saved = dict(_W)
        _worker_init(*init)
        try:
            for p in bases:
                _job_base(p)
            return [_job_folds(c) for c in chunks]
        finally:
            cm = _W.pop("const_cm", None)
            if cm is not None:
                cm.__exit__(None, None, None)
            _W.clear()
            _W.update(saved)
            _BASE_MEMO.clear()
            uninstall_instrumentation()
    by_path: dict[str, list[tuple]] = defaultdict(list)
    for c in chunks:
        by_path[c[0]].append(c)
    results: list[tuple[str, dict]] = []
    with ProcessPoolExecutor(jobs, initializer=_worker_init, initargs=init) as ex:
        pending = {ex.submit(_job_base, p) for p in bases}
        waiting = set(bases)
        pending |= {ex.submit(_job_folds, c) for p, cs in by_path.items() if p not in waiting for c in cs}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                res = fut.result()
                if isinstance(res, str):   # a base finished: release its folds
                    pending |= {ex.submit(_job_folds, c) for c in by_path.get(res, [])}
                else:
                    results.append(res)
    return results


# --------------------------------------------------------------------------- metrics

def auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUC with average ranks for ties."""
    if not pos or not neg:
        return None
    vals = np.asarray(pos + neg, dtype=float)
    order = np.argsort(vals, kind="mergesort")
    sv = vals[order]
    ranks = np.empty(len(vals))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _pct(num: float, den: float) -> float | None:
    return round(100.0 * num / den, 2) if den else None


def cut_metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    lab = [r for r in rows if r["lok"]]
    labl = [r for r in rows if r["lokl"]]
    won = [r for r in rows if r["top1"] is not None]
    wins = [r for r in rows if r["g"]]
    hit = [r for r in wins if r["gin"]]
    a_m = auc([r["mg"] for r in won if r["ok"]], [r["mg"] for r in won if not r["ok"]])
    a_c = auc([r["conf"] for r in won if r["ok"]], [r["conf"] for r in won if not r["ok"]])
    return {
        "n": n,
        "top1": _pct(sum(r["ok"] for r in rows), n),
        "top3": _pct(sum(1 for r in rows if r["rank"] and r["rank"] <= 3), n),
        "group_top1": _pct(sum(r["gok"] for r in rows), n),
        "no_candidate": _pct(n - len(won), n),
        "label_coverage": _pct(len(lab), n),
        "label_precision": _pct(sum(r["ok"] for r in lab), len(lab)),
        "learn_coverage": _pct(len(labl), n),
        "learn_precision": _pct(sum(r["ok"] for r in labl), len(labl)),
        "margin_auc": None if a_m is None else round(a_m, 3),
        "conf_auc": None if a_c is None else round(a_c, 3),
        "s5_wins": len(wins),
        "s5_group_hit": _pct(len(hit), len(wins)),
        "s5_member_ok": _pct(sum(r["ok"] for r in hit), len(hit)),
    }


def compute_metrics(rows: list[dict]) -> dict[str, dict[str, Any]]:
    slices: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for s in ("ALL", f"dev={r['dev']}", f"user={r['path'].split('/')[0]}",
                  f"prov={r['prov']}", f"ev={r['ev']}"):
            slices[s].append(r)
    out: dict[str, dict[str, Any]] = {}
    for s in sorted(slices):
        m: dict[str, Any] = {}
        by_cut: dict[float, list[dict]] = defaultdict(list)
        for r in slices[s]:
            by_cut[r["cut"]].append(r)
        for cut in sorted(by_cut, reverse=True):
            for k, v in cut_metrics(by_cut[cut]).items():
                m[f"{k}@{cut}"] = v
        out[s] = m
    return out


# ------------------------------------------------------------------------------- run

def _parse_cuts(text: str) -> tuple[float, ...]:
    """``1.0,0.5,10m,600s`` -> fractions as given, ``Nm`` / ``Ns`` as seconds (> 1)."""
    out: list[float] = []
    for raw in str(text).split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        if tok.endswith("m"):
            out.append(float(tok[:-1]) * 60.0)
        elif tok.endswith("s"):
            out.append(float(tok[:-1]))
        else:
            out.append(float(tok))
    return tuple(out)


def run_eval(corpus: Path, mode: str = "fast", cuts: tuple[float, ...] = DEFAULT_CUTS,
             jobs: int = 1, override: dict | None = None, cache_dir: Path | None = None,
             use_cache: bool = True, only: str | None = None, log=print) -> dict:
    """Run the LOO evaluation and return the result document (see module docstring)."""
    levels = {n: logging.getLogger(n).level for n in (PKG_NAME, "homeassistant")}
    _quiet_logs()
    try:
        return _run_eval(Path(corpus), mode, cuts, jobs, override or {}, cache_dir, use_cache, only, log)
    finally:
        for n, lv in levels.items():
            logging.getLogger(n).setLevel(lv)


def _run_eval(corpus: Path, mode: str, cuts: tuple[float, ...], jobs: int, override: dict,
              cache_dir: Path | None, use_cache: bool, only: str | None, log) -> dict:
    cuts = tuple(sorted({float(c) for c in cuts}, reverse=True))
    from custom_components.ha_washdata import profile_store as ps  # noqa: PLC0415
    if PKG_DIR.resolve() not in Path(ps.__file__).resolve().parents:
        raise SystemExit(f"imported {ps.__file__}, not this checkout's {PKG_DIR}")
    t0 = time.time()
    devices, dropped = load_corpus(corpus)
    devices = [d for d in devices if len(d.labels) >= 2 and (not only or only in d.path)]
    if not devices:
        raise SystemExit(f"no device with >= 2 labelled programmes under {corpus}")
    csha = code_sha()
    ov_json = json.dumps(override, sort_keys=True)
    key = hashlib.sha256(f"{HARNESS_VERSION}|{csha}|{ov_json}|{np.__version__}".encode()).hexdigest()[:16]
    tmp_cache = None
    if cache_dir is None or not use_cache:
        tmp_cache = tempfile.TemporaryDirectory(prefix="wd_eval_")
        cache = Path(tmp_cache.name)
    else:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
    _DEV_MEMO.update({str(corpus / d.path): d for d in devices})
    with const_overrides(override.get("const")):
        devinfo = {d.key: device_config(d, override) for d in devices}
    targets: dict[str, list[int]] = {}
    for d in devices:
        idx = scorable(d)
        targets[d.path] = fast_pick(d, idx) if mode == "fast" else idx
    # Row cache: per device file, {idx: {cut: row|None}}.
    cached: dict[str, dict[str, dict]] = {}
    todo: list[tuple[str, list[int]]] = []
    for d in devices:
        f = cache / f"rows-{d.fsha}-{key}.json"
        c = json.loads(f.read_text()) if use_cache and f.exists() else {}
        cached[d.path] = c
        miss = [i for i in targets[d.path] if any(str(cut) not in c.get(str(i), {}) for cut in cuts)]
        if miss:
            todo.append((d.path, miss))
    # Envelope DTW dominates: cost ~ cycles x mean trace length.
    cost = {d.path: len(d.cycles) * max(1, sum(len(c.get("power_data") or []) for c, _ in d.cycles)
                                        // max(1, len(d.cycles))) for d in devices}
    bases = sorted((p for p, _ in todo), key=lambda p: (-cost[p], p))
    chunks: list[tuple[str, list[int], tuple[float, ...]]] = []
    for p, miss in todo:
        size = 1 if mode == "fast" else 2
        chunks += [(p, miss[k:k + size], cuts) for k in range(0, len(miss), size)]
    chunks.sort(key=lambda c: (-cost[c[0]], c[0], c[1]))
    t1 = time.time()
    results = _execute(jobs, (str(corpus), str(cache), key, override), bases, chunks)
    t2 = time.time()
    for path, res in results:
        for i, per_cut in res.items():
            cached[path].setdefault(str(i), {}).update(per_cut)
    if use_cache and tmp_cache is None:
        for d in devices:
            if any(p == d.path for p, _ in todo):
                f = cache / f"rows-{d.fsha}-{key}.json"
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(cached[d.path], sort_keys=True))
                tmp.replace(f)
    rows = [cached[d.path][str(i)][str(cut)] for d in devices for i in targets[d.path] for cut in cuts
            if cached[d.path].get(str(i), {}).get(str(cut))]
    rows.sort(key=lambda r: (r["path"], r["id"], -r["cut"]))
    if len({d.key for d in devices}) != len(devices):
        raise SystemExit("two corpus files map to the same public key")
    if tmp_cache is not None:
        tmp_cache.cleanup()
    manifest = hashlib.sha256("\n".join(f"{d.path}:{d.fsha}" for d in devices).encode()).hexdigest()[:16]
    log(f"[eval] {len(devices)} devices, {sum(len(c[1]) for c in chunks)} folds computed "
        f"({sum(len(v) for v in targets.values()) - sum(len(c[1]) for c in chunks)} cached), jobs={jobs}: "
        f"setup {t1 - t0:.1f}s, folds {t2 - t1:.1f}s, total {time.time() - t0:.1f}s", file=sys.stderr)
    return {
        "meta": {
            "harness_version": HARNESS_VERSION, "rev": _git_rev(), "code_sha": csha,
            "mode": mode, "cuts": list(cuts), "override": override or None,
            "numpy": np.__version__, "corpus_manifest": manifest, "devices": len(devices),
            "folds": len(rows), "dropped_clones": {public_key(k): public_key(v) for k, v in dropped.items()},
        },
        "tolerances": DEFAULT_TOLERANCES,
        "devices": devinfo,
        "metrics": compute_metrics(rows),
        "folds": rows,
    }


def write_result(doc: dict, path: Path) -> None:
    """Deterministic JSON: sorted keys, one fold row per line."""
    parts = [f'"{k}": {json.dumps(doc[k], sort_keys=True, indent=1, ensure_ascii=False)}'
             for k in ("meta", "tolerances", "devices", "metrics")]
    folds = ",\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                       for r in doc["folds"])
    path.write_text("{\n" + ",\n".join(parts) + ',\n"folds": [\n' + folds + "\n]}\n", encoding="utf-8")


def print_summary(doc: dict) -> None:
    m = doc["metrics"]
    cuts = doc["meta"]["cuts"]
    print(f"rev {doc['meta']['rev']} code {doc['meta']['code_sha']} mode {doc['meta']['mode']} "
          f"devices {doc['meta']['devices']} folds {doc['meta']['folds']}")
    hdr = "".join(f"{'@' + str(c):>16}" for c in cuts)
    print(f"{'slice':<22}{hdr}   (top1% / margin AUC, n)")
    for s in ["ALL"] + sorted(k for k in m if k.startswith("dev=")):
        cells = []
        for c in cuts:
            t, a, n = m[s].get(f"top1@{c}"), m[s].get(f"margin_auc@{c}"), m[s].get(f"n@{c}")
            cells.append(f"{'-' if t is None else f'{t:.1f}'}/{'-' if a is None else f'{a:.2f}'} n{n or 0}")
        print(f"{s:<22}" + "".join(f"{x:>16}" for x in cells))
    a = m["ALL"]
    c0 = cuts[0]
    print(f"ALL @{c0}: top3 {a.get(f'top3@{c0}')} group_top1 {a.get(f'group_top1@{c0}')} "
          f"no_candidate {a.get(f'no_candidate@{c0}')} auto-label cov/prec {a.get(f'label_coverage@{c0}')}/"
          f"{a.get(f'label_precision@{c0}')} learn-gate cov/prec {a.get(f'learn_coverage@{c0}')}/"
          f"{a.get(f'learn_precision@{c0}')} conf AUC {a.get(f'conf_auc@{c0}')} "
          f"s5 wins {a.get(f's5_wins@{c0}')} member ok {a.get(f's5_member_ok@{c0}')}")


# --------------------------------------------------------------------------- compare

def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cluster_bootstrap(clusters: list[str], num_b, den_b, num_n, den_n,
                      rounds: int = BOOTSTRAP_ROUNDS, seed: int = 0) -> tuple[float, float] | None:
    """95% CI of (num_n/den_n - num_b/den_b)*100, resampling device clusters."""
    keys = sorted(set(clusters))
    idx = {k: i for i, k in enumerate(keys)}
    agg = np.zeros((4, len(keys)))
    for j, cl in enumerate(clusters):
        agg[:, idx[cl]] += (num_b[j], den_b[j], num_n[j], den_n[j])
    w = np.random.default_rng(seed).multinomial(len(keys), [1.0 / len(keys)] * len(keys), size=rounds)
    s = w @ agg.T
    ok = (s[:, 1] > 0) & (s[:, 3] > 0)
    if not ok.any():
        return None
    d = 100.0 * (s[ok, 2] / s[ok, 3] - s[ok, 0] / s[ok, 1])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def _fmt(v, nd=2, sign=False) -> str:
    return "-" if v is None else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def _fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def compare(base: dict, new: dict, tol: dict | None = None, out=print) -> int:
    """Paired comparison; returns 0 ok, 1 guarded regression, 2 nothing to pair."""
    tol = tol if tol is not None else (base.get("tolerances") or DEFAULT_TOLERANCES)
    for k in ("corpus_manifest", "mode", "harness_version"):
        if base["meta"].get(k) != new["meta"].get(k):
            out(f"WARNING: {k} differs ({base['meta'].get(k)} vs {new['meta'].get(k)}); unpaired folds dropped")
    key = lambda r: (r["path"], r["id"], float(r["cut"]))  # noqa: E731
    bf = {key(r): r for r in base["folds"]}
    nf = {key(r): r for r in new["folds"]}
    common = sorted(set(bf) & set(nf))
    out(f"paired folds {len(common)} (base-only {len(bf) - len(common)}, new-only {len(nf) - len(common)}); "
        f"base {base['meta'].get('rev')}/{base['meta'].get('code_sha')} "
        f"new {new['meta'].get('rev')}/{new['meta'].get('code_sha')} override {new['meta'].get('override')}")
    if not common:
        return 2
    out(f"{'cut':<5}{'metric':<17}{'base':>8}{'new':>8}{'delta':>8}  {'CI95 (device cluster)':<22}{'+/-':>8}{'p':>9}")
    paired: dict[str, float | None] = {}
    for cut in sorted({k[2] for k in common}, reverse=True):
        ks = [k for k in common if k[2] == cut]
        B, N = [bf[k] for k in ks], [nf[k] for k in ks]
        mb, mn = cut_metrics(B), cut_metrics(N)
        cl = [k[0] for k in ks]
        ones = [1] * len(ks)
        for name, fn in (("top1", lambda r: r["ok"]), ("group_top1", lambda r: r["gok"]),
                         ("top3", lambda r: bool(r["rank"] and r["rank"] <= 3)),
                         ("no_candidate", lambda r: r["top1"] is None),
                         ("label_coverage", lambda r: r["lok"]),
                         ("learn_coverage", lambda r: r["lokl"])):
            vb, vn = [int(fn(r)) for r in B], [int(fn(r)) for r in N]
            better = sum(1 for x, y in zip(vb, vn) if y > x)
            worse = sum(1 for x, y in zip(vb, vn) if y < x)
            if name in LOWER_IS_BETTER:
                better, worse = worse, better
            ci = cluster_bootstrap(cl, vb, ones, vn, ones)
            paired[f"{name}@{cut}"] = mn[name] - mb[name] if None not in (mn[name], mb[name]) else None
            out(f"{cut:<5}{name:<17}{_fmt(mb[name]):>8}{_fmt(mn[name]):>8}{_fmt(paired[f'{name}@{cut}'], 2, True):>8}  "
                f"{'-' if ci is None else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]':<22}{f'{better}/{worse}':>8}"
                f"{_fmt_p(mcnemar_p(better, worse)):>9}")
        for name, flag in (("label_precision", "lok"), ("learn_precision", "lokl")):
            ci = cluster_bootstrap(cl, [int(r["ok"] and r[flag]) for r in B], [int(r[flag]) for r in B],
                                   [int(r["ok"] and r[flag]) for r in N], [int(r[flag]) for r in N])
            d = None if None in (mb[name], mn[name]) else mn[name] - mb[name]
            paired[f"{name}@{cut}"] = d
            out(f"{cut:<5}{name:<17}{_fmt(mb[name]):>8}{_fmt(mn[name]):>8}{_fmt(d, 2, True):>8}  "
                f"{'-' if ci is None else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]':<22}")
        for name in ("margin_auc", "conf_auc"):
            d = None if None in (mb[name], mn[name]) else mn[name] - mb[name]
            out(f"{cut:<5}{name:<17}{_fmt(mb[name], 3):>8}{_fmt(mn[name], 3):>8}{_fmt(d, 3, True):>8}")
        by: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for b, n in zip(B, N):
            by[str(n["dev"])][0] += 1
            by[str(n["dev"])][1] += b["ok"]
            by[str(n["dev"])][2] += n["ok"]
        out("      top1 by device: " + "; ".join(
            f"{g} {100 * b / t:.1f}->{100 * a / t:.1f} (n={t})" for g, (t, b, a) in sorted(by.items())))
    failed = False
    out("guarded (max drop, pp):")
    for metric, limit in tol.items():
        d = paired.get(metric)
        if d is None:
            out(f"  {metric:<22} not measured")
            continue
        worse_by = d if metric.split("@")[0] in LOWER_IS_BETTER else -d
        verdict = "FAIL" if worse_by > float(limit) + 1e-9 else "ok"
        failed |= verdict == "FAIL"
        out(f"  {metric:<22} {d:+.2f}  (tol {float(limit):.2f})  {verdict}")
    return 1 if failed else 0


# ------------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the leave-one-cycle-out evaluation")
    r.add_argument("--mode", choices=("fast", "full"), default="fast")
    r.add_argument("--cuts", default=",".join(str(c) for c in DEFAULT_CUTS))
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    r.add_argument("--out", type=Path, default=None, help="result JSON (default: <cache>/last_<mode>.json)")
    r.add_argument("--config-override", default=None, help="JSON object or @file (see module docstring)")
    r.add_argument("--corpus", type=Path, default=REPO / "cycle_data")
    r.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    r.add_argument("--no-cache", action="store_true", help="ignore and do not write the row cache")
    r.add_argument("--only", default=None, help="restrict to devices whose path contains this")
    c = sub.add_parser("compare", help="paired comparison of two run results")
    c.add_argument("base", type=Path)
    c.add_argument("new", type=Path)
    c.add_argument("--tol", type=Path, default=None, help="JSON {metric@cut: max drop pp}")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        base = json.loads(a.base.read_text(encoding="utf-8"))
        new = json.loads(a.new.read_text(encoding="utf-8"))
        tol = json.loads(a.tol.read_text(encoding="utf-8")) if a.tol else None
        return compare(base, new, tol)
    if not a.corpus.is_dir():
        print(f"no corpus at {a.corpus} (cycle_data/ is gitignored: symlink or pass --corpus)", file=sys.stderr)
        return 2
    try:
        doc = run_eval(a.corpus, a.mode, _parse_cuts(a.cuts), max(1, a.jobs),
                       parse_override(a.config_override), a.cache_dir, not a.no_cache, a.only)
    except SystemExit as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 2
    out = a.out or (a.cache_dir / f"last_{a.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_result(doc, out)
    print_summary(doc)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    # String hashing is per-process random; pin it so any set-order dependence in the
    # matcher cannot make two runs differ.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    raise SystemExit(main())
