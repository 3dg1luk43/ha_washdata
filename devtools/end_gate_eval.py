#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure what the fallback end gate costs and saves, on real traces.

Register item 306 shortened the ENDING fallback timeout from
``max(off_delay, min_off_gap)`` to ``max(off_delay, min(min_off_gap, 300))``
once a matched cycle passes ``1.05 x`` its own expected duration, and reported
median end lag 10.00 -> 7.73 min over 427 real cycles with early ends and splits
unchanged.  **That harness was never checked in**, so when round 6 of the PR #448
review added the missing ``match_confidence_threshold`` guard to the same gate
there was no way to re-cut the number.  This is that harness, checked in.

What it measures, per replayed cycle:

* **end lag** - the replay offset at which the detector left ENDING, minus the
  trace's own active span (first to last above-``stop_threshold_w`` reading).
  That is the wall-clock time between the appliance finishing and WashData
  saying so.  Deliberately NOT ``final_duration_s``: since item 297 the stored
  duration is trimmed back to the last activity, so it is ~0 lag by
  construction and measures nothing about the gate.
* **early end** - the detector closed the cycle *before* the appliance stopped,
  counted at the >1 min and >5 min marks.  This is the axis that must never move
  in the wrong direction: an early end truncates the cycle and corrupts the
  profile it is recorded against.
* **split** - the trace did not survive as one cycle covering its active span.

Replay is the real thing, not a model of it: every cycle goes through
``playground.simulate_cycle_detail``, which drives the real ``CycleDetector``
and the real Stage 1-5 matcher over the cycle's own trace.  So
``_last_match_confidence`` is whatever the shipped matcher actually produces -
which is the whole point, since the guard under test reads exactly that.
Since audit F7 each match is also applied with the manager's own post-match
rules (``match_rules``: the envelope verified pause and its releases, the
confident-mismatch revoke, the switching that names the program), and every
candidate template is re-gridded to the query's step as live does. Figures
taken before that never saw a verified pause, so they under-report the end lag
of devices that engage one (the #427 AEG washer: 1.5-15.5 min per cycle).

**Configuration (audit F2 / DETECT-12).**  Each export is replayed with the
configuration its own options produce in production: the detector config and the
ProfileStore come from a real ``WashDataManager`` built on the export's entry
data and options (the same ``build_detector_config`` the manager uses), and every
envelope is rebuilt with the current code (exports carry stale ones). The old
hand-rolled config defaulted ``min_off_gap`` to 480 s for every device and read a
key no option is stored under, which understated washer end lag 16.2 -> 12.2 min.

``--loo`` matches each cycle against profiles rebuilt WITHOUT it (leave-one-out),
like ``devtools/eval.py``. Without it each cycle is matched against an envelope it
helped build, which flatters confidence and ambiguity - use ``--loo`` for any
figure you quote.

**Running the A/B.**  The two arms are two states of the code. Do not ``git
stash`` in a shared working tree; check the other arm out in a worktree:

    git worktree add --detach /tmp/before <ref> && ln -s "$PWD/cycle_data" /tmp/before/
    (cd /tmp/before && python3 devtools/end_gate_eval.py --loo --json /tmp/before.json)
    python3 devtools/end_gate_eval.py --loo --json /tmp/after.json
    python3 devtools/end_gate_eval.py --compare /tmp/before.json /tmp/after.json

``--no-shortening`` patches ``const.END_GATE_LATE_RATIO`` (and the per-device
map) out of reach to give the
pre-306 arm, which needs no checkout.

**Corpus and watchdog (register item 465).** By default only the export format
(``device_fingerprint`` + ``data.past_cycles``) is read, which skips every
diagnostics dump: none of ``cycle_data/user-Contributed/`` is replayed.
``--all-formats`` reads the corpus the way ``devtools/eval.py`` does (all three
export shapes, clone files dropped; private paths keyed by hash). The replay emits
the live watchdog's keepalives inside silent stretches at each export's own
``watchdog_interval``; ``--shipped-watchdog`` drops that option so every device
runs at its type's shipped default instead. That is not cosmetic: a keepalive is
the only reading inside a silence, so the interval decides whether an end gate is
evaluated between the expected end and a late pump-out at all (01KGM619: 599 s in
the export, 30 s the dishwasher default, 61 s what Apply-all suggests for it).

**Anti-crease (register item 207).** ``--anti-wrinkle force`` turns anti-wrinkle
on for every washer, dryer and washer-dryer export (``export``, the default, keeps
each export's own setting). ``--tumble-tail`` replaces the synthetic 0 W tail of
those devices with the #296 shape: the trace's trailing quiet is trimmed and the
reporter's Knitterschutz tail follows (a ~3 W baseline with a sub-400 W drum burst
every 37 s), so the ordinary end gates cannot close the cycle and the anti-crease
finalise is the only closer. Every row records whether the #399 spin guard held
that finalise, whether it released on the spin (``event``) or at the
``ANTI_CREASE_SPIN_WAIT_MAX_RATIO`` cap, and whether it fired before the trace's
own last reading above ``anti_wrinkle_max_power`` (an early release: the spin then
opens a second cycle). ``--device-types`` restricts the corpus.

**Committed baseline (audit TESTING-09).** ``devtools/end_gate_baseline.json``
holds the per-device-type end-gate figures (n, median / mean / p90 lag, early
ends > 1 / > 5 min, splits) of one ``--loo --all-formats`` run, with the replay
flags and the tree state (HEAD, dirty files, a hash of the integration code) it
was taken on. Regenerate it after an intended end-gate change::

    python3 devtools/end_gate_eval.py --loo --all-formats \\
        --json /tmp/rows.json --write-baseline devtools/end_gate_baseline.json

and gate a change against it::

    python3 devtools/end_gate_eval.py --check devtools/end_gate_baseline.json
    python3 devtools/end_gate_eval.py --check devtools/end_gate_baseline.json --rows /tmp/rows.json

``--check`` replays with the baseline's own flags (or summarises ``--rows``, the
``--json`` output of such a run, without replaying) and exits **1** when any
device type regresses beyond ``BASELINE_TOLERANCE``: median lag + 0.25 min, mean
lag + 0.5 min, p90 lag + 1.0 min, and **no** new early end (> 1 or > 5 min) or
split. Early ends and splits are counts with zero tolerance because they are the
axes that must never move the wrong way; lag is a cost and gets a small band. It
exits **2** when the corpus no longer matches (a device type's cycle count
differs: ``cycle_data/`` is maintainer-local, so re-baseline rather than compare),
and 0 otherwise. The replay is deterministic, so an unchanged tree reproduces the
baseline exactly.

Run from the repo root.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if TYPE_CHECKING:
    from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
    from custom_components.ha_washdata.profile_store import ProfileStore

# The integration is imported by `_integration()` (from `main` and `_production`),
# not here: importing any of it loads Home Assistant (~3 s), and `--help` must not.
playground: Any = None
CONF_ANTI_WRINKLE_ENABLED = "anti_wrinkle_enabled"
CONF_WATCHDOG_INTERVAL = "watchdog_interval"
resolve_watchdog_interval_default: Any = None
_cycle_readings: Any = None


def _integration() -> None:
    """Bind the integration names this module uses (idempotent)."""
    global playground, CONF_ANTI_WRINKLE_ENABLED, CONF_WATCHDOG_INTERVAL  # noqa: PLW0603
    global resolve_watchdog_interval_default, _cycle_readings  # noqa: PLW0603
    if playground is not None:
        return
    from custom_components.ha_washdata import const  # noqa: PLC0415
    from custom_components.ha_washdata import playground as _pg  # noqa: PLC0415
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        _cycle_readings as _readings,
    )

    CONF_ANTI_WRINKLE_ENABLED = const.CONF_ANTI_WRINKLE_ENABLED
    CONF_WATCHDOG_INTERVAL = const.CONF_WATCHDOG_INTERVAL
    resolve_watchdog_interval_default = const.resolve_watchdog_interval_default
    _cycle_readings = _readings
    playground = _pg

#: A cycle must carry at least this many readings to be worth replaying.
MIN_READINGS = 10
#: Exports with fewer stored cycles than this cannot build usable profiles.
MIN_CYCLES = 5
#: Device types the anti-crease finalise can arm for (`_anticrease_gate_open`).
AC_DEVICE_TYPES = ("washing_machine", "dryer", "washer_dryer")

#: Anti-crease probe state for the replay in progress (one at a time).
_AC: dict[str, Any] = {}


def _install_anticrease_probe() -> None:
    """Record what the #399 spin guard and the anti-crease finalise did.

    Wraps three detector methods; the replay itself is unchanged. A hold is the
    guard returning True on the finalise path (``_is_anticrease_tail``); the
    standby-band path calls the same predicate and is recorded separately.
    """
    from custom_components.ha_washdata import cycle_detector as cd  # noqa: PLC0415
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        ANTI_CREASE_SPIN_WAIT_MAX_RATIO,
        ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC,
    )

    det_cls = cd.CycleDetector
    if getattr(det_cls, "_eval_probe", False):
        return
    orig_tail = det_cls._is_anticrease_tail  # noqa: SLF001
    orig_pending = det_cls._anticrease_spin_pending  # noqa: SLF001
    orig_final = det_cls._maybe_finalize_anticrease_tail  # noqa: SLF001

    def is_tail(self: Any, ts: Any) -> bool:
        self._eval_ac_ctx = True
        try:
            return orig_tail(self, ts)
        finally:
            self._eval_ac_ctx = False

    def pending(self: Any, ts: Any) -> bool:
        out = orig_pending(self, ts)
        if "ac_final_ts" in _AC:
            return out
        if not getattr(self, "_eval_ac_ctx", False):
            _AC["sb_held"] = _AC.get("sb_held", False) or bool(out)
            return out
        if out:
            _AC["ac_held"] = True
            return out
        block = self._matched_terminal_high  # noqa: SLF001
        start = self._current_cycle_start  # noqa: SLF001
        expected = self._expected_duration  # noqa: SLF001
        elapsed = (ts - start).total_seconds() if start is not None else 0.0
        if block is None or block[0] < ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC:
            _AC["ac_reason"] = "unarmed"
        elif expected > 0 and elapsed >= expected * ANTI_CREASE_SPIN_WAIT_MAX_RATIO:
            _AC["ac_reason"] = "cap"
        else:
            _AC["ac_reason"] = "event"
        return out

    def finalize(self: Any, ts: Any) -> bool:
        fired = orig_final(self, ts)
        if fired:
            _AC.setdefault("ac_finals", []).append(ts)
        if fired and "ac_final_ts" not in _AC:
            _AC["ac_final_ts"] = ts
            if _AC.get("ac_held"):
                _AC["ac_release"] = _AC.get("ac_reason")
        return fired

    det_cls._is_anticrease_tail = is_tail  # noqa: SLF001
    det_cls._anticrease_spin_pending = pending  # noqa: SLF001
    det_cls._maybe_finalize_anticrease_tail = finalize  # noqa: SLF001
    det_cls._eval_probe = True


def _with_tumble_tail(
    cycle: dict[str, Any], pts: list[tuple[float, float]], stop: float, level: float
) -> dict[str, Any]:
    """The cycle with its trailing quiet replaced by the #296 tumble tail.

    The tail is the reporter's own (tron4r export, the merged back-to-back
    cycle): a ~3.3 W baseline and a ~60 W drum burst every ~37 s, reported on
    change. It runs for an hour or 0.6x the trace, whichever is longer, which
    reaches past the 1.25x spin-wait cap of the trace's own programme.
    """
    peak = max(p for _t, p in pts)
    floor = max(1.0, peak * 0.02)
    end = len(pts) - 1
    while end > 0 and pts[end][1] <= floor:
        end -= 1
    out = [[float(t), float(p)] for t, p in pts[: end + 1]]
    burst = min(max(60.0, 2.0 * stop), 0.5 * level)
    base_w = 3.3
    t0 = out[-1][0]
    span = max(3600.0, 0.6 * t0)
    t = t0 + 30.0
    while t < t0 + span:
        out += [[t, burst], [t + 2.0, burst * 0.75], [t + 4.0, base_w], [t + 20.0, base_w]]
        t += 37.0
    tailed = dict(cycle)
    tailed["power_data"] = out
    return tailed


class _Entry:
    def __init__(self, data: dict, options: dict, title: str) -> None:
        self.data, self.options, self.title = data, options, title
        self.entry_id, self.domain = "end-gate-eval", "ha_washdata"

    def async_on_unload(self, *_a: Any, **_k: Any) -> None:
        return None

    def add_update_listener(self, *_a: Any, **_k: Any) -> Any:
        return lambda: None


class _InlineHass:
    """Executor jobs run inline: deterministic, and nothing else is reachable."""

    async def async_add_executor_job(self, fn: Any, *args: Any) -> Any:
        return fn(*args)


class _NullStore:
    """Replaces the WashDataStore: every save is dropped."""

    async def async_save(self, _data: Any) -> None:
        return None

    async def async_load(self) -> None:
        return None


def _run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _production(
    doc: dict[str, Any],
    data: dict[str, Any],
    *,
    shipped_watchdog: bool = False,
    force_anti_wrinkle: bool = False,
) -> tuple[CycleDetectorConfig, ProfileStore, dict[str, Any]]:
    """(detector config, ProfileStore, options) exactly as the manager builds them.

    ``shipped_watchdog`` drops the export's ``watchdog_interval`` (entry data and
    options) so the Playground resolves the device type's shipped default.
    ``force_anti_wrinkle`` sets ``anti_wrinkle_enabled`` before the manager reads it.
    """
    _integration()
    from custom_components.ha_washdata.manager import WashDataManager  # noqa: PLC0415

    entry_data = {"power_sensor": "sensor.end_gate_eval", "name": "eval",
                  **{k: v for k, v in (doc.get("entry_data") or {}).items() if v is not None}}
    opts = {k: v for k, v in (doc.get("entry_options") or {}).items() if v is not None}
    if shipped_watchdog:
        entry_data.pop(CONF_WATCHDOG_INTERVAL, None)
        opts.pop(CONF_WATCHDOG_INTERVAL, None)
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    if device_type:
        opts.setdefault("device_type", device_type)
    if force_anti_wrinkle:
        entry_data.pop(CONF_ANTI_WRINKLE_ENABLED, None)
        opts[CONF_ANTI_WRINKLE_ENABLED] = True
    mgr = WashDataManager(MagicMock(), _Entry(entry_data, opts, "eval"))
    store = mgr.profile_store
    store.hass = _InlineHass()
    store._store = _NullStore()  # noqa: SLF001
    store._data = data  # noqa: SLF001
    return mgr.detector.config, store, {**entry_data, **opts}


def _rebuild_envelopes(store: ProfileStore, names: Any) -> None:
    async def _go() -> None:
        for name in names:
            await store.async_rebuild_envelope(name)

    _run(_go())


def _fold_data(base: dict[str, Any], cycle: dict[str, Any]) -> dict[str, Any]:
    """The store without ``cycle`` (by identity), sharing what it does not touch."""
    d = dict(base)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        d[key] = [c for c in (base.get(key) or []) if c is not cycle]
    d["profiles"] = {k: dict(v) if isinstance(v, dict) else v
                     for k, v in (base.get("profiles") or {}).items()}
    d["envelopes"] = dict(base.get("envelopes") or {})
    return d


def _end_offset(events: list[dict[str, Any]]) -> float | None:
    """Replay offset of the transition OUT of ENDING - when the cycle closed.

    This is the number the user experiences as "the wash is done" arriving late,
    and the only one the fallback end gate moves.
    """
    for ev in reversed(events):
        if ev.get("type") != "state":
            continue
        detail = str(ev.get("detail") or "")
        if detail.startswith("ending->"):
            return float(ev.get("t") or 0.0)
    return None


def _active_span(points: list[tuple[float, float]], stop: float) -> float:
    """Seconds from the first to the last above-stop reading."""
    active = [t for t, p in points if p > stop]
    return (active[-1] - active[0]) if len(active) >= 2 else 0.0


_EVAL_MOD: Any = None


def _corpus_module() -> Any:
    """``devtools/eval.py``, for its corpus loader (all export shapes, clones dropped)."""
    global _EVAL_MOD  # noqa: PLW0603
    if _EVAL_MOD is None:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location(
            "wd_end_gate_eval_corpus", REPO / "devtools" / "eval.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclasses resolve their module through it
        spec.loader.exec_module(mod)
        _EVAL_MOD = mod
    return _EVAL_MOD


def _load_doc(path: Path, all_formats: bool) -> dict[str, Any] | None:
    """The export, or with ``all_formats`` any corpus shape normalised to it."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not all_formats or not isinstance(doc, dict):
        return doc if isinstance(doc, dict) else None
    unwrapped = _corpus_module()._unwrap(doc)  # noqa: SLF001
    if unwrapped is None:
        return None
    data, entry_data, entry_options, _fmt = unwrapped
    device_type = entry_options.get("device_type") or entry_data.get("device_type")
    return {
        "device_fingerprint": {"device_type": device_type},
        "entry_data": entry_data,
        "entry_options": entry_options,
        "data": data,
    }


def _export_key(path: Path) -> str:
    rel = str(path.relative_to(REPO))
    if rel.startswith("cycle_data/"):
        # Contributors' real names are in some file names (eval.py PRIVATE_DIRS).
        return "cycle_data/" + _corpus_module().public_key(rel[len("cycle_data/"):])
    return rel


def _measure_export(
    path: Path,
    no_shortening: bool,
    loo: bool = False,
    *,
    all_formats: bool = False,
    shipped_watchdog: bool = False,
    anti_wrinkle: str = "export",
    tumble_tail: bool = False,
    device_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Replay every usable cycle in one export; one row per cycle."""
    doc = _load_doc(path, all_formats)
    if doc is None:
        return []
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    data = doc.get("data") or {}
    cycles = data.get("past_cycles") or []
    if not device_type or len(cycles) < MIN_CYCLES:
        return []
    if device_types and device_type not in device_types:
        return []
    ac_device = device_type in AC_DEVICE_TYPES
    force_aw = anti_wrinkle == "force" and ac_device
    base = dict(data)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        base[key] = list(base.get(key) or [])
    base["profiles"] = {k: dict(v) if isinstance(v, dict) else v
                        for k, v in (base.get("profiles") or {}).items()}
    base["envelopes"] = dict(base.get("envelopes") or {})
    cfg, store, opts = _production(
        doc, base, shipped_watchdog=shipped_watchdog, force_anti_wrinkle=force_aw
    )
    stop = float(cfg.stop_threshold_w)
    level = float(cfg.anti_wrinkle_max_power)
    # Exports carry the envelopes the exporting version built; rebuild them with
    # the code under test, as the live store would after an upgrade.
    _rebuild_envelopes(store, list(base["profiles"]))
    try:
        prebuilt = playground._build_match_snapshots(store)  # noqa: SLF001
    except Exception:
        prebuilt = None
    cycles = base["past_cycles"]
    rows: list[dict[str, Any]] = []
    for cyc in cycles:
        pts = _cycle_readings(cyc)
        if len(pts) < MIN_READINGS:
            continue
        span = _active_span(pts, stop)
        if span <= 0:
            continue
        fold_store, fold_prebuilt = store, prebuilt
        name = cyc.get("profile_name")
        if loo and name and name in base["profiles"]:
            _cfg_f, fold_store, _o = _production(
                doc, _fold_data(base, cyc), shipped_watchdog=shipped_watchdog,
                force_anti_wrinkle=force_aw,
            )
            _rebuild_envelopes(fold_store, [name])
            try:
                fold_prebuilt = playground._build_match_snapshots(fold_store)  # noqa: SLF001
            except Exception:
                fold_prebuilt = None
        replayed = (
            _with_tumble_tail(cyc, pts, stop, level) if tumble_tail and ac_device else cyc
        )
        _AC.clear()
        try:
            sim = playground.simulate_cycle_detail(
                replayed, cfg, None, fold_store, opts, price=None,
                compute_series=False, prebuilt=fold_prebuilt,
            )
        except Exception:
            continue
        if "error" in sim:
            continue
        out = sim.get("outcome") or {}
        final = out.get("final_duration_s")
        events = sim.get("events") or []
        end_t = _end_offset(events)
        first_end = next(
            (float(ev.get("t") or 0.0) for ev in events if ev.get("type") == "finished"),
            None,
        )
        highs = [t for t, p in pts if p > level]
        active = [t for t, p in pts if p > stop]
        ac_final = _AC.get("ac_final_ts")
        base_t = playground._cycle_base_time(replayed)  # noqa: SLF001
        ac_final_s = (ac_final - base_t).total_seconds() if ac_final is not None else None
        # Every anti-crease finalise, not only the first: a run that already split
        # can release early again on a later piece.
        ac_early_n = sum(
            1 for ts in _AC.get("ac_finals", ())
            if highs and (ts - base_t).total_seconds() < highs[-1]
        )
        rows.append({
            "export": _export_key(path),
            "device_type": device_type,
            "id": str(cyc.get("id"))[:12],
            "label": cyc.get("profile_name"),
            "active_span_s": round(span, 1),
            "detected": bool(out.get("detected")),
            "detected_count": int(out.get("detected_count") or 0),
            "final_duration_s": round(float(final), 1) if final else None,
            "end_offset_s": round(end_t, 1) if end_t is not None else None,
            "matched_profile": out.get("matched_profile"),
            "confidence": out.get("confidence"),
            "termination_reason": out.get("termination_reason"),
            "off_delay": cfg.off_delay,
            "min_off_gap": cfg.min_off_gap,
            "no_shortening": no_shortening,
            "loo": loo,
            "watchdog_s": opts.get(
                CONF_WATCHDOG_INTERVAL, resolve_watchdog_interval_default(device_type)
            ),
            # Anti-crease (register item 207). Offsets are seconds into the trace.
            "anti_wrinkle": bool(cfg.anti_wrinkle_enabled),
            "tumble_tail": bool(tumble_tail and ac_device),
            "first_end_s": round(first_end, 1) if first_end is not None else None,
            "active_end_s": round(active[-1], 1) if active else None,
            "last_high_s": round(highs[-1], 1) if highs else None,
            "ac_final_s": round(ac_final_s, 1) if ac_final_s is not None else None,
            "ac_held": bool(_AC.get("ac_held")),
            "ac_release": _AC.get("ac_release"),
            "ac_early": bool(
                ac_final_s is not None and highs and ac_final_s < highs[-1]
            ),
            "ac_early_n": ac_early_n,
            "sb_held": bool(_AC.get("sb_held")),
        })
    return rows


def _summarise(rows: list[dict[str, Any]], device_type: str | None = None) -> dict[str, Any]:
    """Lag / early-end / split figures over a set of replayed cycles."""
    sel = [r for r in rows if device_type is None or r["device_type"] == device_type]
    finished = [
        r for r in sel
        if r["detected"] and r.get("end_offset_s") is not None
    ]
    if not finished:
        return {"n": 0}
    lags = [r["end_offset_s"] - r["active_span_s"] for r in finished]
    early = [lag for lag in lags if lag < 0]
    splits = sum(
        1 for r in finished
        if r["detected_count"] > 1 or r["final_duration_s"] < 0.9 * r["active_span_s"]
    )
    matched = [r for r in finished if r["matched_profile"]]
    weak = [
        r for r in matched
        if r["confidence"] is not None and float(r["confidence"]) < 0.4
    ]
    return {
        "n": len(finished),
        "median_lag_min": round(float(np.median(lags)) / 60.0, 2),
        "mean_lag_min": round(float(np.mean(lags)) / 60.0, 2),
        "p90_lag_min": round(float(np.percentile(lags, 90)) / 60.0, 2),
        "early_1min_pct": round(100.0 * sum(1 for x in early if x < -60) / len(finished), 2),
        "early_5min_pct": round(100.0 * sum(1 for x in early if x < -300) / len(finished), 2),
        "split_pct": round(100.0 * splits / len(finished), 2),
        "matched_pct": round(100.0 * len(matched) / len(finished), 2),
        "weak_match_pct": round(100.0 * len(weak) / max(1, len(matched)), 2),
        "weak_match_n": len(weak),
    }


def _summarise_ac(
    rows: list[dict[str, Any]], device_type: str | None = None
) -> dict[str, Any]:
    """Anti-crease figures over the anti-wrinkle-enabled washer/dryer rows (item 207).

    ``early`` counts anti-crease finalises (every one, on any piece of the run)
    BEFORE the trace's last reading above ``anti_wrinkle_max_power``: the spin was
    still ahead, so it opens another cycle. ``cut`` is the same test for the first end by ANY path. ``lag`` is the
    first end minus the trace's last above-stop reading (with ``--tumble-tail``
    the harness's own split/lag columns also count the tail's own detections, so
    read these instead); ``held lag`` is the same over the finalises the #399
    spin guard held.
    """
    sel = [
        r for r in rows
        if r.get("anti_wrinkle") and r["device_type"] in AC_DEVICE_TYPES
        and (device_type is None or r["device_type"] == device_type)
    ]
    if not sel:
        return {"n": 0}
    fin = [r for r in sel if r.get("ac_final_s") is not None]
    held = [r for r in fin if r.get("ac_held")]
    lag = [
        (r["first_end_s"] - r["active_end_s"]) / 60.0
        for r in sel
        if r.get("first_end_s") is not None and r.get("active_end_s") is not None
    ]
    held_lag = [
        (r["ac_final_s"] - r["active_end_s"]) / 60.0
        for r in held if r.get("active_end_s") is not None
    ]
    cut = sum(
        1 for r in sel
        if r.get("first_end_s") is not None and r.get("last_high_s") is not None
        and r["first_end_s"] < r["last_high_s"]
    )

    def _r(vals: list[float], fn: Any) -> float | None:
        return round(float(fn(vals)), 2) if vals else None

    return {
        "n": len(sel),
        "finalized": len(fin),
        "held": len(held),
        "event": sum(1 for r in held if r.get("ac_release") == "event"),
        "cap": sum(1 for r in held if r.get("ac_release") == "cap"),
        "unarmed": sum(1 for r in held if r.get("ac_release") == "unarmed"),
        "early": sum(int(r.get("ac_early_n", int(bool(r.get("ac_early"))))) for r in sel),
        "cut": cut,
        "med_lag_min": _r(lag, np.median),
        "p90_lag_min": _r(lag, lambda v: np.percentile(v, 90)),
        "max_lag_min": _r(lag, np.max),
        "held_med_lag_min": _r(held_lag, np.median),
        "held_max_lag_min": _r(held_lag, np.max),
    }


_AC_KEYS = (
    "n", "finalized", "held", "event", "cap", "unarmed", "early", "cut",
    "med_lag_min", "p90_lag_min", "max_lag_min", "held_med_lag_min", "held_max_lag_min",
)


def _print_ac_summary(rows: list[dict[str, Any]]) -> None:
    scopes = [None, *sorted({r["device_type"] for r in rows if r["device_type"] in AC_DEVICE_TYPES})]
    lines = [(scope or "ALL", _summarise_ac(rows, scope)) for scope in scopes]
    lines = [(name, s) for name, s in lines if s["n"]]
    if not lines:
        return
    print("\nanti-crease (anti-wrinkle on; item 207)")
    hdr = (
        f"{'scope':<18}{'n':>5}{'final':>7}{'held':>6}{'event':>7}{'cap':>5}"
        f"{'unarm':>7}{'EARLY':>7}{'cut':>5}{'med lag':>9}{'p90':>8}{'max':>8}"
        f"{'held med':>10}{'max':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    def _f(v: Any) -> str:
        return "-" if v is None else f"{v:.2f}"

    for name, s in lines:
        print(
            f"{name:<18}{s['n']:>5}{s['finalized']:>7}{s['held']:>6}{s['event']:>7}"
            f"{s['cap']:>5}{s['unarmed']:>7}{s['early']:>7}{s['cut']:>5}"
            f"{_f(s['med_lag_min']):>9}{_f(s['p90_lag_min']):>8}{_f(s['max_lag_min']):>8}"
            f"{_f(s['held_med_lag_min']):>10}{_f(s['held_max_lag_min']):>8}"
        )
    print(
        "final = anti-crease finalises; held = the #399 spin guard held one; "
        "event/cap = how it released;\nEARLY = finalised before the trace's last "
        "reading above anti_wrinkle_max_power; cut = first end (any path) before it;"
        "\nlag = first end minus last above-stop reading, minutes."
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    devices = sorted({r["device_type"] for r in rows})
    hdr = (
        f"{'scope':<18}{'n':>5}{'med lag':>9}{'mean':>8}{'p90':>8}"
        f"{'early>1m':>10}{'early>5m':>10}{'splits':>8}{'matched':>9}{'weak':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for scope in [None, *devices]:
        s = _summarise(rows, scope)
        if not s["n"]:
            continue
        name = scope or "ALL"
        print(
            f"{name:<18}{s['n']:>5}{s['median_lag_min']:>9.2f}{s['mean_lag_min']:>8.2f}"
            f"{s['p90_lag_min']:>8.2f}{s['early_1min_pct']:>9.2f}%{s['early_5min_pct']:>9.2f}%"
            f"{s['split_pct']:>7.2f}%{s['matched_pct']:>8.1f}%{s['weak_match_n']:>7}"
        )
    print(
        "\nlag = when the detector left ENDING, minus the trace's own active span,"
        " in minutes."
    )
    print("weak = matched cycles whose final confidence is below 0.4 (the gate's bar).")
    _print_ac_summary(rows)


def _compare(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    b_by_id = {(r["export"], r["id"]): r for r in before}
    a_by_id = {(r["export"], r["id"]): r for r in after}
    common = sorted(set(b_by_id) & set(a_by_id))
    print(f"paired cycles: {len(common)}  (before {len(before)}, after {len(after)})\n")

    devices = sorted({b_by_id[k]["device_type"] for k in common})
    for scope in [None, *devices]:
        bs = _summarise([b_by_id[k] for k in common], scope)
        as_ = _summarise([a_by_id[k] for k in common], scope)
        if not bs["n"] or not as_["n"]:
            continue
        name = scope or "ALL"
        print(f"=== {name} (n={as_['n']})")
        for key, unit in (
            ("median_lag_min", " min"), ("mean_lag_min", " min"), ("p90_lag_min", " min"),
            ("early_1min_pct", "%"), ("early_5min_pct", "%"), ("split_pct", "%"),
        ):
            print(f"    {key:<18} {bs[key]:>8}{unit} -> {as_[key]:>8}{unit}")
        print()

    ac_scopes = [None, *sorted({
        b_by_id[k]["device_type"] for k in common
        if b_by_id[k]["device_type"] in AC_DEVICE_TYPES
    })]
    for scope in ac_scopes:
        bs = _summarise_ac([b_by_id[k] for k in common], scope)
        as_ = _summarise_ac([a_by_id[k] for k in common], scope)
        if not bs["n"] or not as_["n"]:
            continue
        print(f"=== anti-crease {scope or 'ALL'} (n={as_['n']})")
        for key in _AC_KEYS[1:]:
            print(f"    {key:<18} {bs[key]!s:>8} -> {as_[key]!s:>8}")
        print()
    ac_moved = [
        k for k in common
        if b_by_id[k].get("ac_final_s") != a_by_id[k].get("ac_final_s")
    ]
    if ac_moved:
        print(f"cycles whose anti-crease finalise moved: {len(ac_moved)} / {len(common)}")
        for k in ac_moved[:40]:
            b, a = b_by_id[k], a_by_id[k]

            def _at(r: dict[str, Any]) -> str:
                v = r.get("ac_final_s")
                tag = r.get("ac_release") or ("free" if v is not None else "none")
                early = " EARLY" if r.get("ac_early") else ""
                return "-" if v is None else f"{v / 60:.1f}m {tag}{early}"

            print(
                f"    {b['device_type']:<16} {b['id']:<14} {str(b['label'])[:24]:<24} "
                f"{_at(b):>18} -> {_at(a):<18}"
            )
        print()

    moved = [
        k for k in common
        if (b_by_id[k].get("end_offset_s") or 0) != (a_by_id[k].get("end_offset_s") or 0)
    ]
    print(f"cycles whose end moved at all: {len(moved)} / {len(common)}")
    for k in moved[:25]:
        b, a = b_by_id[k], a_by_id[k]
        db = (b.get("end_offset_s") or 0) - b["active_span_s"]
        da = (a.get("end_offset_s") or 0) - a["active_span_s"]
        print(
            f"    {b['device_type']:<16} {b['id']:<14} conf="
            f"{b['confidence']!s:<6} lag {db / 60:>7.2f} -> {da / 60:>7.2f} min"
        )
    if len(moved) > 25:
        print(f"    ... and {len(moved) - 25} more")


#: ``--check`` regression tolerances (audit TESTING-09): how far a metric may move
#: the wrong way, per device type, before the check fails. Lags in minutes;
#: early ends and splits are cycle counts.
BASELINE_TOLERANCE: dict[str, float] = {
    "median_lag_min": 0.25,
    "mean_lag_min": 0.5,
    "p90_lag_min": 1.0,
    "early_1min_n": 0,
    "early_5min_n": 0,
    "split_n": 0,
}
#: The replay flags a baseline records and ``--check`` replays with.
BASELINE_FLAGS = (
    "loo", "all_formats", "shipped_watchdog", "anti_wrinkle", "tumble_tail",
    "device_types", "no_shortening",
)


def _baseline_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per device type (and ``ALL``): the ``_summarise`` figures plus counts."""
    out: dict[str, dict[str, Any]] = {}
    for scope in [None, *sorted({r["device_type"] for r in rows})]:
        s = _summarise(rows, scope)
        if not s["n"]:
            continue
        n = s["n"]
        out[scope or "ALL"] = {
            "n": n,
            "median_lag_min": s["median_lag_min"],
            "mean_lag_min": s["mean_lag_min"],
            "p90_lag_min": s["p90_lag_min"],
            "early_1min_n": int(round(s["early_1min_pct"] * n / 100.0)),
            "early_5min_n": int(round(s["early_5min_pct"] * n / 100.0)),
            "split_n": int(round(s["split_pct"] * n / 100.0)),
            "early_1min_pct": s["early_1min_pct"],
            "early_5min_pct": s["early_5min_pct"],
            "split_pct": s["split_pct"],
        }
    return out


def _git(*argv: str) -> str:
    try:
        return subprocess.run(
            ["git", *argv], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.rstrip()
    except OSError:
        return ""


def _tree_state() -> dict[str, Any]:
    """HEAD, the uncommitted files, and a hash of every integration source file."""
    digest = hashlib.sha256()
    pkg = REPO / "custom_components" / "ha_washdata"
    for path in sorted(pkg.rglob("*.py")):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return {
        "head": _git("rev-parse", "--short", "HEAD"),
        "dirty": [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()],
        "integration_py_sha256": digest.hexdigest()[:16],
    }


def _write_baseline(
    path: str, rows: list[dict[str, Any]], flags: dict[str, Any], tree: dict[str, Any]
) -> None:
    doc = {
        "about": (
            "end_gate_eval baseline (audit TESTING-09). Regenerate with "
            "--write-baseline after an intended end-gate change; gate with --check."
        ),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flags": flags,
        "tree": tree,
        "tolerance": BASELINE_TOLERANCE,
        "summary": _baseline_summary(rows),
    }
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote baseline ({len(doc['summary'])} scopes) to {path}")


def _check_baseline(baseline: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    """0 within tolerance, 1 on a regression, 2 when the corpus no longer matches."""
    tol = {**BASELINE_TOLERANCE, **(baseline.get("tolerance") or {})}
    before = baseline.get("summary") or {}
    after = _baseline_summary(rows)
    regressions: list[str] = []
    drift: list[str] = []
    print(f"{'scope':<18}{'metric':<16}{'baseline':>10}{'now':>10}{'tol':>7}")
    for scope in sorted(set(before) | set(after)):
        b, a = before.get(scope), after.get(scope)
        if b is None or a is None or b["n"] != a["n"]:
            drift.append(
                f"{scope}: n {b['n'] if b else 0} -> {a['n'] if a else 0}"
            )
            continue
        for key, limit in tol.items():
            if key not in b or key not in a:
                continue
            delta = float(a[key]) - float(b[key])
            flag = ""
            if delta > float(limit) + 1e-9:
                flag = "  REGRESSION"
                regressions.append(f"{scope} {key} {b[key]} -> {a[key]} (tol +{limit})")
            elif abs(delta) > 1e-9:
                flag = "  improved" if delta < 0 else "  within tol"
            print(f"{scope:<18}{key:<16}{b[key]!s:>10}{a[key]!s:>10}{limit!s:>7}{flag}")
    if drift:
        print("\ncorpus differs from the baseline (re-baseline, do not compare):")
        for line in drift:
            print(f"    {line}")
    if regressions:
        print("\nREGRESSION beyond tolerance:")
        for line in regressions:
            print(f"    {line}")
        return 1
    if drift:
        return 2
    print("\nend gates within the baseline's tolerance")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write per-cycle rows here for --compare")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument(
        "--no-shortening", action="store_true",
        help="pre-306 arm: patch END_GATE_LATE_RATIO out of reach",
    )
    ap.add_argument(
        "--loo", action="store_true",
        help="leave-one-out: match each cycle against profiles rebuilt without it",
    )
    ap.add_argument(
        "--all-formats", action="store_true",
        help="read every corpus shape (diagnostics dumps too) via eval.py, clones dropped",
    )
    ap.add_argument(
        "--shipped-watchdog", action="store_true",
        help="ignore each export's watchdog_interval: replay at the device type's default",
    )
    ap.add_argument(
        "--anti-wrinkle", choices=("export", "force"), default="export",
        help="export: each export's own anti_wrinkle_enabled; force: on for every "
        "washer/dryer/washer-dryer (item 207)",
    )
    ap.add_argument(
        "--tumble-tail", action="store_true",
        help="washers/dryers: replace the trailing quiet with the #296 anti-crease "
        "tumble tail, so only the anti-crease finalise can close the cycle",
    )
    ap.add_argument(
        "--device-types", default="",
        help="comma-separated device types to replay (default: all)",
    )
    ap.add_argument(
        "--write-baseline", metavar="FILE",
        help="write the per-device-type summary, flags and tree state here (TESTING-09)",
    )
    ap.add_argument(
        "--check", metavar="BASELINE",
        help="replay with BASELINE's flags and exit 1 on a regression beyond its "
        "tolerance, 2 when the corpus differs",
    )
    ap.add_argument(
        "--rows", metavar="FILE",
        help="with --check: summarise these --json rows instead of replaying",
    )
    args = ap.parse_args()

    if args.compare:
        _compare(*args.compare)
        return 0

    baseline: dict[str, Any] | None = None
    if args.check:
        baseline = json.loads(Path(args.check).read_text(encoding="utf-8"))
        if args.rows:
            return _check_baseline(baseline, json.loads(Path(args.rows).read_text()))
        # Replay exactly what the baseline measured, whatever else was passed.
        for key, value in (baseline.get("flags") or {}).items():
            if key in BASELINE_FLAGS:
                setattr(args, key, ",".join(value) if key == "device_types" else value)
    elif args.rows:
        ap.error("--rows needs --check")
    _integration()
    device_types = tuple(t.strip() for t in args.device_types.split(",") if t.strip())
    # Taken before the replay: other work may change the tree while it runs.
    tree = _tree_state() if args.write_baseline else {}

    if args.no_shortening:
        # Patch `const`, not `cycle_detector`. Since register item 355 the gate
        # calls `resolve_end_gate_late_ratio(device_type)`, which reads these two
        # names out of `const` at call time - rebinding the detector module's
        # imported copy no longer reaches it, and this arm would silently stop
        # disabling the shortening while still reporting itself as the pre-306
        # baseline. Both names, because the per-device map wins for washers.
        from custom_components.ha_washdata import const as _const

        _const.END_GATE_LATE_RATIO = 1e9
        _const.END_GATE_LATE_RATIO_BY_DEVICE = {}

    logging.getLogger("custom_components.ha_washdata").setLevel(logging.ERROR)
    _install_anticrease_probe()
    rows: list[dict[str, Any]] = []
    corpus = REPO / "cycle_data"
    if args.all_formats:
        devices, _clones = _corpus_module().load_corpus(corpus)
        paths = [corpus / dev.path for dev in devices]
    else:
        paths = sorted(corpus.rglob("*.json"))
    for path in paths:
        rows.extend(_measure_export(
            path, args.no_shortening, args.loo,
            all_formats=args.all_formats, shipped_watchdog=args.shipped_watchdog,
            anti_wrinkle=args.anti_wrinkle, tumble_tail=args.tumble_tail,
            device_types=device_types or None,
        ))

    if not rows:
        print("no replayable cycles found - is cycle_data/ present?")
        return 1

    _print_summary(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {len(rows)} rows to {args.json}")
    if args.write_baseline:
        flags = {key: getattr(args, key) for key in BASELINE_FLAGS}
        flags["device_types"] = list(device_types)
        _write_baseline(args.write_baseline, rows, flags, tree)
    if baseline is not None:
        print()
        return _check_baseline(baseline, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
