#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Iterate "Apply all" on every corpus device: fixed point, oscillation or ladder? (audit F10)

    python3 devtools/suggestion_loop_eval.py [--rounds 5] [--jobs N] [--device SUBSTR]
                                             [--legacy sampling_interval|completion_min|confidence]
                                             [--json FILE]

A suggestion is a feedback loop: the setting it proposes changes how the next cycles
are detected, and those cycles are the evidence for the next proposal. A formula that
looks right on one snapshot of history can still walk a setting without bound once its
own output is fed back (audit SUGGEST-16: three such ratchets shipped until 0.5.8, none
visible to a unit test because no test re-recorded history under the applied values).
This harness closes that loop on the real corpus.

**One round** (per device, all production code, nothing re-implemented):

1. ``WashDataManager`` is built from the device's entry data and CURRENT options (the
   same construction ``devtools/eval.py`` uses, so ``build_detector_config``, the
   matcher's ratios/DTW/energy mode and the LearningManager come out exactly as the
   manager resolves them - never a partial config).
2. Every stored cycle that ended on its own is **re-detected** under that config: its
   trace goes through the Playground replay (real ``CycleDetector`` + real Stage 1-5
   matcher + emulated watchdog keepalives, ``playground._DetailSim``, with a quiet
   tail long enough for a dishwasher's end waits: :func:`run_tail`). What the detector
   emits passes the manager's ghost / dishwasher pump-out suppression and
   ``ProfileStore._add_cycle_data`` (the production normaliser: trim, sampling interval,
   signature, energy) and replaces the stored cycle.
3. The update-cadence model behind the operational pass is rebuilt from the
   re-detected cycles that ended on their own (last 200 in-cycle intervals, >= 20 to
   run, as ``LearningManager.close_cycle_cadence`` commits them).
4. Every suggestion pass runs in cycle-end order (operational, standby floor, model,
   detection, batch) through ``LearningManager._apply_suggestions_and_notify`` - the
   noise gates, the post-apply cooldown, locks, ``reconcile_suggestions``.
   First with the cooldown still active (anything non-corrective stored then is a
   cooldown leak), then once it has expired.
5. "Apply all" exactly as ``ws_apply_suggestions``: ``ws_api._visible_suggestions``
   (the one filter behind the Settings list, muted and unchanged keys dropped), int
   coercion, cooldown stamped on the lifetime odometer, suggestions cleared, values
   layered onto the options.

Repeat until Apply all changes nothing (a fixed point) or ``--rounds`` applies. Round 0
starts from the export's own options with no pending suggestions, no cooldown and no
locks (a muted key would hide its own ladder; ``--keep-locks`` keeps the export's).

**Held fixed** (cannot be re-derived from an export): the raw power traces. A stored
trace stops where its cycle ended (trimmed at the stop threshold it was recorded under
unless the end kept its tail) and the replay appends a synthetic 0 W tail, so a stop
threshold moving below the level the appliance really idles at is not judged by the
replay itself: ``idle_above_stop`` counts the cycles whose kept tail sits at or above
the new threshold, and ``--idle-hold`` replays those as never ending (force-ended, the
#458 outcome) instead. Also fixed: profiles, envelopes and the match pool
(rebuilt once with the current code; matching is in-sample, which flatters confidence);
labels (``profile_name`` / ``label_source`` / ``match_confidence`` carried from the
source to its longest re-detected fragment, other fragments unlabelled); cycles that
did not end on their own (force- or user-stopped: the replay cannot reproduce the
human or the 6 h watchdog, and they are the standby-floor evidence) and untraced
cycles; the wall-clock position of every cycle. Each stored cycle is replayed on its
own, so a back-to-back merge across two stored cycles is not observable here (use
``min_off_gap_eval.py``). The manager's reading throttle (``sampling_interval``) is
emulated on the trace only when the option rises above the value the trace was
recorded at (re-throttling at the recorded value is idempotent; a lower value cannot
restore dropped readings).

**Classification** per (device, setting), from the value after each apply:
``stable`` never changed; ``converged`` stopped changing before the last evaluated
round (reversals and returns to an earlier value are counted on the way); ``ladder``
still changing at the end after >= 3 applies in one direction; ``oscillates`` still
changing after a direction reversal or a return to an earlier value; ``unsettled``
still changing otherwise (too few rounds to tell). A loop can also stop because it
has eaten its own evidence: ``erased`` lists the labelled cycles stored completed
under the export's options that the fixed point's options turn interrupted,
force-stopped or unstored, which every pass then ignores (the SUGGEST-03 ratchet:
Completion Minimum turned the shortest programme interrupted, so the next value rose
again). That is a failure too. ``fragmented`` (still completed, plus an extra record)
is only reported.

**Revert check** (``--legacy``): re-enables a suggestion removed in 0.5.8 with its old
logic copied verbatim from ``c006c92~1:custom_components/ha_washdata/suggestion_engine.py``
(the keys are re-admitted to the visible set; the old reconcile Rules 3a-6 are not):
``sampling_interval`` (SUGGEST-04, the median stored interval, read back through the
throttle it sets), ``completion_min`` (SUGGEST-03, half the p05 clean duration with
interrupted cycles excluded and no cap) and ``confidence`` (SUGGEST-01, the learning /
auto-label / match thresholds from percentiles of auto-labels; history is relabelled
each round by the thresholds in force, as ``ladder_sim.py`` did). Measured: the old
Sampling Interval is a ladder (the #427 AEG washer: 2 -> 7 -> 11.1 -> 15 -> 18.2 -> 22
-> 25.3 s, dragging the watchdog and match interval with it); the old Completion
Minimum climbs 900 -> 2294 -> 4071 -> 4322 s on a contributed dishwasher and stops
only once it has erased its shortest programme (two "Quick wash" and a "Rinse" stored
interrupted); the confidence thresholds drop once (auto-label 0.9 -> 0.69, match 0.4 ->
0.68 from the shipped defaults) and hold - a wrong target, not a loop, whose cost is in
label routing, which a detection replay does not score.

Runtime: ~1-10 s of CPU per replayed cycle per round, most of it the matcher. A
device's rounds are sequential, the replays inside a round are not: they are spread
over ``--jobs`` processes, and a matcher call whose inputs are byte-identical to an
earlier one (a pure function) is answered from a per-process memo. Serial and
parallel runs give identical results.

Exit codes: 0 every setting stable/converged, 1 a ladder / oscillation / unsettled
setting, erased evidence, a cooldown leak or a muted key applied, 2 no corpus.
"""
from __future__ import annotations

import os

# Before NumPy is imported: one BLAS thread per worker (determinism + throughput).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterator  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import numpy as np  # noqa: E402
from homeassistant.util import dt as dt_util  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom_components.ha_washdata import analysis as analysis_mod  # noqa: E402
from custom_components.ha_washdata import playground, ws_api  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_LEARNING_CONFIDENCE,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_SAMPLING_INTERVAL,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_LEARNING_CONFIDENCE,
    MIN_SUGGESTION_COOLDOWN_CYCLES,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_UNKNOWN,
    TerminationReason,
    resolve_sampling_interval_default,
)
from custom_components.ha_washdata.signal_processing import (  # noqa: E402
    energy_gap_threshold_s,
    integrate_wh,
)
from custom_components.ha_washdata.suggestion_engine import (  # noqa: E402
    SuggestionEngine,
    _cycle_readings,
    select_clean_cycles,
)

_EVAL_PATH = REPO / "devtools" / "eval.py"
_spec = importlib.util.spec_from_file_location("wd_suggest_loop_eval_corpus", _EVAL_PATH)
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(ev)

DEFAULT_ROUNDS = 5
#: A device needs this many re-detectable cycles: the batch and detection passes
#: both require 5 clean cycles before they propose anything.
MIN_REPLAYABLE = 5
MIN_READINGS = 10
#: The engine reads at most the last 200 stored cycles (the retention cap).
MAX_CYCLES = 200
#: Applies in one direction, still moving at the end, before a key is a ladder.
LADDER_MIN_APPLIES = 3
LEGACY_RULES = ("sampling_interval", "completion_min", "confidence")
_LEGACY_KEYS = {
    "sampling_interval": (CONF_SAMPLING_INTERVAL,),
    "completion_min": (),  # a key the shipped engine still suggests
    "confidence": (CONF_LEARNING_CONFIDENCE, CONF_AUTO_LABEL_CONFIDENCE, CONF_PROFILE_MATCH_THRESHOLD),
}
#: Carried from a stored cycle to its longest re-detected fragment.
_LABEL_FIELDS = (
    "profile_name", "label", "label_source", "match_confidence", "original_auto_label",
    "auto_labeled", "ml_review",
)
_AUTO_SOURCES = ("auto_match", "auto_label_post", "auto_label_service")
#: The replay's quiet tail is at least this long (see run_tail).
_TAIL_MIN_S = 7200.0
#: A kept tail must be at least this long to say where the appliance idles.
_POST_LEVEL_MIN_S = 60.0
_POST_LEVEL_MIN_SAMPLES = 3


# --------------------------------------------------------------------- matcher memo

_MEMO: dict[tuple, Any] = {}
_MEMO_MAX = 20000
_MEMO_PATCHED: list[tuple[Any, Any]] = []
_MEMO_STATS = {"hit": 0, "miss": 0}


def _install_matcher_memo() -> None:
    """Share Stage 1-4 results between rounds: same inputs, same output.

    ``compute_matches_worker`` is pure NumPy, so a call whose power series, duration,
    candidate snapshots and matcher config are byte-identical returns the identical
    ranking. A setting that only moves the END of a cycle leaves every earlier match
    call identical, which is most of the replay cost after round 0. Patched on
    ``analysis`` and on every loaded integration module that bound it by name, so it
    holds whichever module the replay calls it through.
    """
    real = analysis_mod.compute_matches_worker
    if getattr(real, "_loop_eval_memo", False):
        return

    def memo(powers: Any, duration: Any, snapshots: Any, cfg: Any) -> Any:
        key = (
            hashlib.blake2b(np.asarray(powers, dtype=np.float64).tobytes(), digest_size=16).digest(),
            round(float(duration), 3),
            id(snapshots),
            hashlib.blake2b(repr(sorted((cfg or {}).items())).encode(), digest_size=12).digest(),
        )
        hit = _MEMO.get(key)
        if hit is None:
            if len(_MEMO) >= _MEMO_MAX:
                _MEMO.clear()  # bounded: a worker sees every device's matches
            _MEMO_STATS["miss"] += 1
            hit = (snapshots, real(powers, duration, snapshots, cfg))  # pin snapshots: id() stays valid
            _MEMO[key] = hit
        else:
            _MEMO_STATS["hit"] += 1
        return copy.deepcopy(hit[1])

    memo._loop_eval_memo = True  # type: ignore[attr-defined]
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith(ev.PKG_NAME) and getattr(
            mod, "compute_matches_worker", None
        ) is real:
            mod.compute_matches_worker = memo  # type: ignore[attr-defined]
            _MEMO_PATCHED.append((mod, real))


@contextlib.contextmanager
def _isolated() -> Iterator[None]:
    """Quiet logs and the matcher memo for one run, then leave the process as found.

    A test calls :func:`run` in-process: a logger left at CRITICAL or a patched
    ``compute_matches_worker`` would leak into every test after it.
    """
    loggers = [logging.getLogger(ev.PKG_NAME), logging.getLogger("homeassistant")]
    levels = [lg.level for lg in loggers]
    _quiet()
    try:
        yield
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)
        while _MEMO_PATCHED:
            mod, real = _MEMO_PATCHED.pop()
            if getattr(getattr(mod, "compute_matches_worker", None), "_loop_eval_memo", False):
                mod.compute_matches_worker = real  # type: ignore[attr-defined]
        _MEMO.clear()


def _memo_snapshot_builder(store: Any) -> None:
    """One candidate list per query grid for the whole run, not one per replay.

    ``_DetailSim`` re-grids the templates to each match's ``used_dt`` through
    ``store.build_match_snapshots`` and caches the result per sim only; with the store
    fixed the result is too, and sharing the list object is what lets the matcher
    memo key on it. The list is the one ``_grouped_snapshots`` returns unchanged.
    """
    real = store.build_match_snapshots
    cache: dict[float, Any] = {}

    def build_match_snapshots(used_dt: float) -> Any:
        key = round(float(used_dt), 2)
        if key not in cache:
            cache[key] = real(used_dt)
        return cache[key]

    store.build_match_snapshots = build_match_snapshots


# ------------------------------------------------------------------ legacy rules

def _legacy_suggestions(engine: SuggestionEngine, legacy: tuple[str, ...]) -> dict[str, Any]:
    """The removed suggestions, verbatim from c006c92~1 (suggestion_engine.py).

    Same inputs as ``generate_detection_suggestions``: the last 200 stored cycles,
    filtered by ``select_clean_cycles`` at the current stop threshold.
    """
    options = engine._entry_options()  # noqa: SLF001
    stop_thr = engine._current_stop_threshold(options)  # noqa: SLF001
    all_cycles = engine.profile_store.get_past_cycles()[-200:]
    clean, _excluded = select_clean_cycles(all_cycles, stop_threshold_w=stop_thr)
    out: dict[str, Any] = {}
    if len(clean) < 5:
        return out
    if "sampling_interval" in legacy:  # c006c92~1:1560-1586 (audit SUGGEST-04)
        vals = []
        for c in clean:
            try:
                si = float(c.get("sampling_interval") or 0.0)
            except (TypeError, ValueError):
                continue
            if si > 0:
                vals.append(si)
        if len(vals) >= 5:
            out[CONF_SAMPLING_INTERVAL] = {
                "value": round(float(np.median(vals)), 1), "reason": "legacy sampling_interval",
            }
    if "completion_min" in legacy:  # c006c92~1:1640-1667 (audit SUGGEST-03)
        durations = [
            float(c["duration"]) for c in clean
            if isinstance(c.get("duration"), (int, float))
            and not isinstance(c.get("duration"), bool) and float(c["duration"]) > 0
        ]
        if len(durations) >= 10:
            p05d = float(np.percentile(durations, 5))
            out[CONF_COMPLETION_MIN_SECONDS] = {
                "value": int(max(120, round(p05d * 0.5))), "reason": "legacy completion_min",
            }
    if "confidence" in legacy:  # c006c92~1:1683-1743 (audit SUGGEST-01)
        manual, auto_ok = [], []
        for c in clean:
            raw = c.get("match_confidence")
            if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
                continue
            src = c.get("label_source")
            if src == "manual":
                manual.append(float(raw))
            elif src in _AUTO_SOURCES and not c.get("original_auto_label"):
                auto_ok.append(float(raw))
        if len(manual) >= 10:
            out[CONF_LEARNING_CONFIDENCE] = {
                "value": round(min(max(float(np.percentile(manual, 5)), 0.3), 0.9), 2),
                "reason": "legacy learning_confidence",
            }
        if len(auto_ok) >= 15:
            out[CONF_AUTO_LABEL_CONFIDENCE] = {
                "value": round(min(max(float(np.percentile(auto_ok, 15)), 0.5), 0.98), 2),
                "reason": "legacy auto_label_confidence",
            }
            out[CONF_PROFILE_MATCH_THRESHOLD] = {
                "value": round(min(max(float(np.percentile(auto_ok, 10)), 0.3), 0.9), 2),
                "reason": "legacy profile_match_threshold",
            }
    return out


@contextlib.contextmanager
def legacy_patch(legacy: tuple[str, ...]) -> Iterator[None]:
    """Re-enable removed suggestions for the duration of a run (revert check)."""
    if not legacy:
        yield
        return
    orig_gen = SuggestionEngine.generate_detection_suggestions
    orig_keys = ws_api._SUGGESTION_KEYS  # noqa: SLF001

    def generate_detection_suggestions(self: SuggestionEngine) -> dict[str, Any]:
        out = orig_gen(self)
        if "completion_min" in legacy:
            out.pop(CONF_COMPLETION_MIN_SECONDS, None)
        out.update(_legacy_suggestions(self, legacy))
        return out

    extra = tuple(k for rule in legacy for k in _LEGACY_KEYS[rule] if k not in orig_keys)
    SuggestionEngine.generate_detection_suggestions = generate_detection_suggestions  # type: ignore[method-assign]
    ws_api._SUGGESTION_KEYS = orig_keys + extra  # noqa: SLF001
    try:
        yield
    finally:
        SuggestionEngine.generate_detection_suggestions = orig_gen  # type: ignore[method-assign]
        ws_api._SUGGESTION_KEYS = orig_keys  # noqa: SLF001


def _relabel(cycle: dict[str, Any], learning: float, auto: float) -> None:
    """Label provenance under the thresholds in force (legacy ``confidence`` arm only).

    The confidence ladder fed on ``label_source``, which the thresholds it suggested
    decide: >= auto_label is labelled silently, >= learning asks, below is unlabelled
    (``ladder_sim.py``). Manual labels are the user's and never move.
    """
    conf = cycle.get("match_confidence")
    if cycle.get("label_source") == "manual" or not isinstance(conf, (int, float)) or isinstance(conf, bool):
        return
    cycle["label_source"] = (
        "auto_label_post" if conf >= auto else "auto_match" if conf >= learning else None
    )
    cycle.pop("original_auto_label", None)


# ------------------------------------------------------------------ production pieces

def _make_hass(entry: Any) -> Any:
    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = entry
    hass.config_entries.async_update_entry = MagicMock()

    def _create_task(coro: Any, *_a: Any, **_k: Any) -> None:
        close = getattr(coro, "close", None)
        if close is not None:
            close()

    hass.async_create_task = _create_task
    return hass


def _manager(entry_data: dict, options: dict, title: str, data: dict) -> Any:
    """``WashDataManager`` exactly as production builds it, storage stubbed."""
    from custom_components.ha_washdata.manager import WashDataManager  # noqa: PLC0415

    entry = ev._Entry(entry_data, dict(options), title)  # noqa: SLF001
    hass = _make_hass(entry)
    mgr = WashDataManager(hass, entry)
    store = mgr.profile_store
    store.hass = ev._InlineHass()  # noqa: SLF001
    store._store = ev._NullStore()  # noqa: SLF001
    store._data = data  # noqa: SLF001
    # The LearningManager / engine keep the MagicMock hass: their config-entry read
    # is what `for_job()` snapshots, exactly as on the loop.
    return mgr


def _throttle(points: list[tuple[float, float]], s: float, min_p: float) -> list[tuple[float, float]]:
    """The manager's reading throttle (``async_handle_power_change``) on a stored trace.

    A reading at or above ``min_power`` that arrives less than ``sampling_interval``
    after the last processed one is dropped; inside a cycle every low reading bypasses
    the throttle (the detector is active). Same model as ``sampling_ratchet.py``.
    """
    kept: list[tuple[float, float]] = []
    last_t: float | None = None
    for t, p in points:
        if p >= min_p and last_t is not None and (t - last_t) < s:
            continue
        kept.append((t, p))
        last_t = t
    return kept


def _energy_wh(points: list) -> float:
    try:
        arr = sorted((float(p[0]), float(p[1])) for p in points)
    except (TypeError, ValueError, IndexError):
        return 0.0
    if len(arr) < 2:
        return 0.0
    ts = np.array([a for a, _ in arr])
    ps = np.array([b for _, b in arr])
    return float(integrate_wh(ts, ps, max_gap_s=energy_gap_threshold_s(ts)))


def _active_span(points: list[tuple[float, float]], stop: float) -> float:
    act = [t for t, p in points if p > stop]
    return (act[-1] - act[0]) if len(act) >= 2 else 0.0


def _replayable(c: Any) -> bool:
    return (
        isinstance(c, dict)
        and c.get("status") in ("completed", "interrupted")
        and c.get("termination_reason") not in (TerminationReason.USER, TerminationReason.FORCE_STOPPED)
        and len(_cycle_readings(c)) >= MIN_READINGS
    )


class _ManagerTap:
    """What the manager does around the detector that the Playground replay does not.

    * The cadence model: every reading the manager processes while the detector is
      active is timed against the previous processed reading (a watchdog keepalive
      counts as processed but is never itself timed), held per cycle, dropped on a
      start from idle, and committed at the cycle's end only when it ended on its
      own (``process_power_reading`` / ``discard_cycle_cadence`` /
      ``close_cycle_cadence``). Readings past the end of the recorded trace (the
      replay's synthetic quiet tail) are not timed: their spacing is invented.
    * The end clock: ``_on_cycle_end`` stamps the moment the end was detected,
      which the dishwasher pump-out rule measures the next start against.
    """

    _ACTIVE = (STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING)

    def __init__(self, sim: Any) -> None:
        det = sim.detector
        self.pending: list[float] = []
        self.committed: list[float] = []
        self.ends: list[float] = []
        self._last: Any = None
        self._t_recorded_end = sim.readings[-1][0]
        orig_pr, orig_end, orig_state = det.process_reading, det._on_cycle_end, det._on_state_change  # noqa: SLF001

        def process_reading(power: float, timestamp: Any, synthetic: bool = False,
                            observed: bool = True) -> None:
            if (
                not synthetic and self._last is not None and det.state in self._ACTIVE
                and timestamp <= self._t_recorded_end
            ):
                delta = (timestamp - self._last).total_seconds()
                if 0.1 < delta < 1800:
                    self.pending.append(delta)
            self._last = timestamp
            orig_pr(power, timestamp, synthetic, observed)

        def on_state_change(old: str, new: str) -> None:
            if new == STATE_STARTING and old in (STATE_OFF, STATE_UNKNOWN):
                self.pending.clear()
            orig_state(old, new)

        def on_cycle_end(data: dict) -> None:
            self.ends.append(float(sim.cursor["t"]))
            if data.get("status") == "completed" and data.get("termination_reason") not in (
                TerminationReason.USER, TerminationReason.FORCE_STOPPED,
            ):
                self.committed.extend(self.pending)
            self.pending.clear()
            orig_end(data)

        det.process_reading = process_reading
        det._on_state_change = on_state_change  # noqa: SLF001
        det._on_cycle_end = on_cycle_end  # noqa: SLF001


def post_cycle_level(points: list[tuple[float, float]], recorded_stop: float) -> float | None:
    """What the appliance drew once its cycle was over, if the trace recorded it.

    Only a trace that kept its tail (Smart Termination, the dishwasher end spike)
    holds readings past the end, and they are the trailing run below the stop
    threshold it was recorded under. Its last three samples are the level the
    appliance sat at; None when there is no such run (a trimmed trace) or it is 0 W.
    """
    run: list[tuple[float, float]] = []
    for t, p in reversed(points):
        if p >= recorded_stop:
            break
        run.append((t, p))
    if len(run) < _POST_LEVEL_MIN_SAMPLES or run[0][0] - run[-1][0] < _POST_LEVEL_MIN_S:
        return None
    level = float(np.median([p for _, p in run[:_POST_LEVEL_MIN_SAMPLES]]))
    return level if level > 0 else None


def run_tail(sim: Any, cfg: Any) -> None:
    """``_DetailSim.run_tail`` with a quiet tail long enough for every end path.

    The Playground sizes its 0 W tail from ``off_delay`` / ``min_off_gap`` alone
    (x1.5 + 300 s). A dishwasher's own end waits outlast that once those two are
    lowered: on a corpus Eco cycle at off_delay = min_off_gap = 200 s Smart
    Termination needed 1530 s of 0 W against a 600 s tail, so the replay force-ended
    a cycle live would have finished, and the loop read it as lost evidence. Same
    readings, same break and flush; at least ``_TAIL_MIN_S`` long.
    """
    last_ts = sim.readings[-1][0]
    span = max(
        max(float(cfg.off_delay or 0.0), float(cfg.min_off_gap or 0.0)) * 1.5 + 300.0,
        _TAIL_MIN_S,
    )
    step = 30.0
    n_steps = int(span / step) + 1
    for i in range(1, n_steps + 1):
        ts = last_ts + timedelta(seconds=step * i)
        sim.cursor["t"] = (ts - sim.base).total_seconds()
        sim.detector.process_reading(0.0, ts)
        sim._sample(ts)  # noqa: SLF001
        if sim.detector.state in (STATE_OFF, STATE_FINISHED) and sim.captured:
            break
    if not sim.captured and sim.detector.state != STATE_OFF:
        flush_ts = last_ts + timedelta(seconds=step * (n_steps + 2))
        sim.cursor["t"] = (flush_ts - sim.base).total_seconds()
        sim.detector.force_end(flush_ts)


def replay_one(src: dict, cfg: Any, match_store: Any, prebuilt: Any, options: dict,
               throttle_s: float | None, recorded_stop: float,
               idle_hold: bool = False) -> dict[str, Any] | None:
    """One stored cycle through the Playground replay under ``cfg``, manager tap attached.

    After the recorded trace the Playground appends 0 W, so every cycle can end. That
    is optimistic exactly where the #458 class lives: a trace that kept its tail can
    show the appliance still drawing a level at or above the NEW stop threshold
    (``idle_above_stop``), and if that level is standby no end gate can ever fire and
    live the cycle runs until a watchdog force-ends it. Whether it is standby or a
    drying phase that later drops to 0 W the trace cannot say, so by default the
    replay keeps the 0 W tail and only counts the case; ``idle_hold`` takes the
    pessimistic reading and records the force-end instead.

    Returns what the detector emitted (``captured``), when each end was detected
    (``ends``, seconds from ``base``), the cadence intervals the manager would have
    committed and ``idle_above_stop``; None when the trace is too short to replay.
    """
    cyc = src
    if throttle_s:
        pts = _cycle_readings(src)
        thinned = _throttle(pts, throttle_s, float(cfg.min_power))
        if len(thinned) < len(pts):
            cyc = dict(src, power_data=[[round(t, 1), p] for t, p in thinned])
    sim = playground._DetailSim(  # noqa: SLF001
        cyc, cfg, None, match_store, options, None, compute_series=False, prebuilt=prebuilt,
    )
    if not sim.ready:
        return None
    tap = _ManagerTap(sim)
    sim.step(0, sim.n_readings)
    level = post_cycle_level(_cycle_readings(src), recorded_stop)
    idle_above = level is not None and level >= float(cfg.stop_threshold_w)
    if idle_hold and idle_above:
        sim.detector.force_end(sim.readings[-1][0] + timedelta(seconds=1))
    else:
        run_tail(sim, cfg)
    return {
        "captured": copy.deepcopy(sim.captured), "ends": list(tap.ends),
        "committed": list(tap.committed), "base": sim.base, "idle_above_stop": idle_above,
    }


def assemble(
    sources: list[dict], replays: dict[int, dict[str, Any] | None], store: Any, cfg: Any,
    options: dict, device_type: str, legacy: tuple[str, ...],
) -> tuple[list[dict], list[float], dict[str, Any]]:
    """Store what the replays emitted, in history order, as the manager would.

    Returns the new ``past_cycles``, the committed cadence intervals in cycle order,
    and counts. ``replays`` maps a source index to :func:`replay_one`'s result; a
    source without one is kept as stored.
    """
    stop = float(cfg.stop_threshold_w)
    out: list[dict] = []
    intervals: list[float] = []
    pool: set[Any] = set()
    m: dict[str, Any] = {"replayed": 0, "stored": 0, "splits": 0, "lost": 0, "truncated": 0,
                         "interrupted": 0, "smart": 0, "ghosts": 0, "idle_above_stop": 0}
    # Per stored cycle: [cycles it became, status of the main one or None if lost].
    outcome: dict[int, list[Any]] = {}
    prev_end = None  # wall clock of the previous end detection (pump-out rule)
    learning = float(options.get(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE))
    auto = float(options.get(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE))
    m["outcome"] = outcome
    for idx, src in enumerate(sources):
        rep = replays.get(idx)
        if rep is None:
            out.append(src)
            continue
        m["replayed"] += 1
        m["idle_above_stop"] += bool(rep.get("idle_above_stop"))
        intervals.extend(rep["committed"])
        kept: list[dict] = []
        for i, cd in enumerate(rep["captured"]):
            duration = float(cd.get("duration") or 0.0)
            energy = _energy_wh(cd.get("power_data") or [])
            ends = rep["ends"]
            end_detect = rep["base"] + timedelta(seconds=ends[i] if i < len(ends) else 0.0)
            start_dt = dt_util.parse_datetime(str(cd.get("start_time") or ""))
            # manager._on_cycle_end: a ghost (and a dishwasher pump-out right after
            # the previous end) is never stored.
            ghost = duration < 60 and energy < 0.05
            if (
                not ghost and device_type == "dishwasher" and prev_end is not None
                and start_dt is not None
                and 0 < (start_dt - prev_end).total_seconds() < 600
                and duration < 300 and energy < 1.0
            ):
                ghost = True
            prev_end = end_detect
            if ghost:
                m["ghosts"] += 1
                continue
            cd = copy.deepcopy(cd)
            cd["energy_wh"] = round(energy, 3)
            kept.append(cd)
        if not kept:
            m["lost"] += 1
            outcome[idx] = [0, None]
            continue
        if len(kept) > 1:
            m["splits"] += 1
        primary = max(
            kept, key=lambda d: (d.get("status") == "completed", float(d.get("duration") or 0.0))
        )
        outcome[idx] = [len(kept), primary.get("status")]
        span = _active_span(_cycle_readings(src), stop)
        if span > 0 and float(primary.get("duration") or 0.0) < 0.9 * span:
            m["truncated"] += 1
        for cd in kept:
            if cd is primary:
                for f in _LABEL_FIELDS:
                    if f in src:
                        cd[f] = copy.deepcopy(src[f])
                if "confidence" in legacy:
                    _relabel(cd, learning, auto)
            else:
                cd["profile_name"] = None
            store._add_cycle_data(cd, target=out, id_pool=pool)  # noqa: SLF001
            if cd is primary and src.get("id") is not None:
                cd["id"] = src["id"]  # keeps profile sample_cycle_id resolvable
            m["stored"] += 1
            m["interrupted"] += cd.get("status") == "interrupted"
            m["smart"] += str(cd.get("termination_reason")) in ("smart", str(TerminationReason.SMART))
    return out, intervals, m


def cadence(intervals: list[float]) -> tuple[float, float] | None:
    """(p95, median) of the cadence model, or None while it holds fewer than 20.

    ``StatisticalModel(max_samples=200)``: the newest 200 committed intervals.
    """
    ivs = intervals[-200:]
    if len(ivs) < 20:
        return None
    arr = np.array(ivs)
    return float(np.percentile(arr, 95)), float(np.median(arr))


def run_passes(mgr: Any, cad: tuple[float, float] | None) -> dict[str, Any]:
    """Every suggestion pass, cycle-end order, through the LearningManager gates.

    Returns what the gates let through to ``SuggestionEngine.apply_suggestions``
    (before reconcile), so a caller can tell a corrective entry from one the
    post-apply cooldown should have held back.
    """
    lm = mgr.learning_manager
    eng = lm.suggestion_engine
    passed: dict[str, Any] = {}
    orig_apply = eng.apply_suggestions

    def _capture(sug: dict[str, Any]) -> None:
        passed.update(copy.deepcopy(sug))
        orig_apply(sug)

    eng.apply_suggestions = _capture
    try:
        gate = lm._apply_suggestions_and_notify  # noqa: SLF001
        if cad is not None:
            gate(eng.for_job().generate_operational_suggestions(*cad))
        gate(eng.for_job().generate_standby_floor_suggestions())
        gate(eng.for_job().generate_model_suggestions())
        gate(eng.for_job().generate_detection_suggestions())
        labeled = [
            c for c in mgr.profile_store.get_past_cycles()
            if isinstance(c, dict) and c.get("profile_name") and c.get("profile_name") != "noise"
            and c.get("power_data") and c.get("status") in ("completed", "force_stopped")
        ]
        gate(eng.for_job().run_batch_simulation(labeled))
    finally:
        eng.apply_suggestions = orig_apply
    return passed


def apply_all(mgr: Any, merged: dict, device_type: str) -> dict[str, Any]:
    """``ws_apply_suggestions`` for every visible key: the option updates."""
    updates: dict[str, Any] = {}
    for key, item, _s, _c in ws_api._visible_suggestions(  # noqa: SLF001
        mgr.profile_store, merged, device_type
    ):
        val = item["value"]
        updates[key] = int(float(val)) if key in ws_api._SUGGESTION_INT_KEYS else float(val)  # noqa: SLF001
    return updates


# ----------------------------------------------------------------- classification

def _same(a: Any, b: Any) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))


def classify(values: list[Any], changed_rounds: list[int], last_round: int) -> dict[str, Any]:
    """Verdict for one setting from its value after each apply.

    ``values[0]`` is the value round 0 ran with, ``values[i]`` the value after the
    i-th apply; ``changed_rounds`` the evaluation rounds whose Apply all moved it;
    ``last_round`` the final evaluation round (it applied nothing for this key iff the
    key settled).
    """
    seq = [values[0]]
    for v in values[1:]:
        if not _same(v, seq[-1]):
            seq.append(v)
    n = len(seq) - 1
    if n == 0:
        return {"verdict": "stable", "applies": 0, "values": seq}
    dirs = []
    for a, b in zip(seq, seq[1:]):
        try:
            dirs.append(1 if float(b) > float(a) else -1)
        except (TypeError, ValueError):
            dirs.append(0)
    reversals = sum(1 for a, b in zip(dirs, dirs[1:]) if a and b and a != b)
    revisit = any(_same(seq[j], seq[i]) for j in range(len(seq)) for i in range(j - 1))
    still = bool(changed_rounds) and changed_rounds[-1] == last_round
    if not still:
        verdict = "converged"
    elif revisit or reversals:
        verdict = "oscillates"
    elif n >= LADDER_MIN_APPLIES and len(set(dirs)) == 1:
        verdict = "ladder"
    else:
        verdict = "unsettled"
    return {"verdict": verdict, "applies": n, "values": seq, "reversals": reversals,
            "revisit": revisit}


# ---------------------------------------------------------------------- per device

def _initial_value(key: str, options: dict, device_type: str) -> Any:
    """The value a key ran with before its first apply: set, else its effective default."""
    from custom_components.ha_washdata.detector_config import (  # noqa: PLC0415
        build_detector_config,
        effective_option_values,
    )

    if options.get(key) is not None:
        return options[key]
    legacy_defaults = {
        CONF_SAMPLING_INTERVAL: resolve_sampling_interval_default(device_type),
        CONF_LEARNING_CONFIDENCE: DEFAULT_LEARNING_CONFIDENCE,
        CONF_AUTO_LABEL_CONFIDENCE: DEFAULT_AUTO_LABEL_CONFIDENCE,
    }
    if key in legacy_defaults:
        return legacy_defaults[key]
    effective = effective_option_values(options, device_type)
    if key in effective:
        return effective[key]
    return getattr(build_detector_config(options, options, device_type), key, None)


def _lock_probe(mgr: Any, cad: Any, merged: dict, device_type: str, locked: list[str]) -> dict[str, Any]:
    """Mute every key Apply all just changed, re-run on the same evidence (SUGGEST-11).

    A muted key may still be stored by a reconcile cascade, but Apply all must never
    change it.
    """
    st = mgr.profile_store
    saved = (dict(st._data.get("suggestions") or {}), list(st._data.get("locked_suggestions") or []))  # noqa: SLF001
    st._data["suggestions"] = {}  # noqa: SLF001
    st._data["locked_suggestions"] = sorted(locked)  # noqa: SLF001
    try:
        run_passes(mgr, cad)
        stored = sorted(k for k in st.get_suggestions() if k in locked)
        applied = sorted(k for k in apply_all(mgr, merged, device_type) if k in locked)
    finally:
        st._data["suggestions"], st._data["locked_suggestions"] = saved  # noqa: SLF001
    return {"locked": sorted(locked), "stored_by_cascade": stored, "applied": applied}


# --------------------------------------------------------------------------- driver

def device_paths(corpus: Path, only: list[str] | None = None) -> list[str]:
    """Corpus files with enough re-detectable history, clones dropped (eval.py rules)."""
    devices, _dropped = ev.load_corpus(corpus)
    out = []
    for dev in devices:
        if only and not any(s in dev.path for s in only):
            continue
        past = list(dev.data.get("past_cycles") or [])[-MAX_CYCLES:]
        if sum(1 for c in past if _replayable(c)) >= MIN_REPLAYABLE:
            out.append(dev.path)
    return out


#: This process's devices, by corpus path. Filled before the worker pool is forked,
#: so every worker inherits each device's rebuilt envelopes instead of rebuilding.
_DEVICES: dict[str, "DeviceLoop"] = {}
_MATCH_SIDES: dict[tuple[str, str], tuple[Any, Any]] = {}
_CFGS: dict[tuple[str, str], Any] = {}


def _opts_key(options: dict) -> str:
    return json.dumps(options, sort_keys=True, default=str)


class DeviceLoop:
    """One corpus device's Apply-all loop; the parent drives it a round at a time."""

    def __init__(self, corpus: Path, path: str, rounds: int, legacy: tuple[str, ...],
                 keep_locks: bool, lock_probe: bool, idle_hold: bool = False) -> None:
        dev = ev.load_device(corpus, path)
        assert dev is not None, path
        self.path, self.rounds, self.legacy, self.lock_probe = path, rounds, legacy, lock_probe
        self.idle_hold = idle_hold
        self.title = ev.public_key(dev.path)
        self.base = ev.base_data(dev)
        self.past = list(self.base.get("past_cycles") or [])[-MAX_CYCLES:]
        self.entry_data, self.options = ev.entry_dicts(dev, {})
        # Every envelope rebuilt once with the current code (exports carry the ones the
        # exporting version built): the fixed match evidence for every round.
        mgr0 = _manager(self.entry_data, self.options, self.title, self.base)
        self.device_type = mgr0.device_type
        # The stop threshold the stored traces were recorded under (as near as an
        # export can say: the one it was exported with).
        self.recorded_stop = float(mgr0.detector.config.stop_threshold_w)

        async def _rebuild() -> None:
            for name in list(self.base["profiles"]):
                await mgr0.profile_store.async_rebuild_envelope(name)

        ev.run_coro(_rebuild())
        self.locks = list(self.base.get("locked_suggestions") or []) if keep_locks else []
        self.s0 = float(self.options.get(
            CONF_SAMPLING_INTERVAL, resolve_sampling_interval_default(self.device_type)
        ))
        self.round = 0
        self.done = False
        self.history: list[dict[str, Any]] = []
        self.trajectory: dict[str, list[tuple[int, Any]]] = {}
        self.initial: dict[str, Any] = {}
        self.stamp = 0
        self.lifetime = max(
            int(self.base.get("lifetime_cycle_count") or 0), len(self.base.get("past_cycles") or [])
        )
        self.leaks: list[str] = []
        self.lock_result: dict[str, Any] | None = None
        self.cpu = 0.0

    # --- worker side ----------------------------------------------------------------

    def throttle_s(self) -> float | None:
        s_now = float(self.options.get(CONF_SAMPLING_INTERVAL, self.s0))
        return s_now if s_now > self.s0 else None

    def jobs(self) -> list[tuple[str, dict, int, float | None]]:
        """This round's replays: one per stored cycle that ended on its own."""
        return [(self.path, self.options, i, self.throttle_s())
                for i, c in enumerate(self.past) if _replayable(c)]

    def match_side(self, options: dict) -> tuple[Any, Any]:
        """(manager over the fixed evidence, prebuilt snapshots), one per matcher config.

        The duration-ratio bounds are suggested keys, so the matcher config can move
        between rounds; the evidence it matches against does not.
        """
        mm = _manager(self.entry_data, options, self.title, self.base)
        sig = repr(sorted(playground._matching_config(mm.profile_store, in_progress=True).items()))  # noqa: SLF001
        hit = _MATCH_SIDES.get((self.path, sig))
        if hit is None:
            _memo_snapshot_builder(mm.profile_store)
            hit = (mm, playground._build_match_snapshots(mm.profile_store))  # noqa: SLF001
            _MATCH_SIDES[(self.path, sig)] = hit
        return hit

    def detector_config(self, options: dict) -> Any:
        key = (self.path, _opts_key(options))
        cfg = _CFGS.get(key)
        if cfg is None:
            cfg = _manager(self.entry_data, options, self.title, {}).detector.config
            _CFGS[key] = cfg
        return cfg

    # --- parent side ----------------------------------------------------------------

    def _data(self, past_cycles: list[dict]) -> dict:
        d = dict(self.base)
        d["past_cycles"] = past_cycles
        d["suggestions"] = {}
        d["locked_suggestions"] = list(self.locks)
        return d

    def finish_round(self, replays: dict[int, dict[str, Any] | None], memo_hit: int,
                     memo_miss: int, cpu: float) -> None:
        """Store the round's re-detected history, run every pass, Apply all."""
        t0 = time.process_time()
        rnd, options = self.round, self.options
        mgr = _manager(self.entry_data, options, self.title, self._data([]))
        cfg = mgr.detector.config
        merged = {**self.entry_data, **options}
        st = mgr.profile_store
        new_past, intervals, m = assemble(
            self.past, replays, st, cfg, merged, self.device_type, self.legacy
        )
        st._data["past_cycles"] = new_past  # noqa: SLF001
        cad = cadence(intervals)
        n_new = m["stored"]
        # --- cooldown phase: the first cycle end after the apply --------------------
        if self.stamp:
            self.lifetime += 1
            st._data["lifetime_cycle_count"] = self.lifetime  # noqa: SLF001
            st._data["suggestion_apply_cycle_count"] = self.stamp  # noqa: SLF001
            passed = run_passes(mgr, cad)
            self.leaks += [f"r{rnd}:{k}" for k, v in passed.items() if not v.get("corrective")]
        # --- cooldown expired ----------------------------------------------------------
        self.lifetime += max(MIN_SUGGESTION_COOLDOWN_CYCLES, n_new)
        st._data["lifetime_cycle_count"] = self.lifetime  # noqa: SLF001
        st._data["suggestion_apply_cycle_count"] = self.stamp  # noqa: SLF001
        cooldown_active = (
            st.get_suggestion_apply_cycle_count() > 0
            and st.get_lifetime_cycle_count() - st.get_suggestion_apply_cycle_count()
            < MIN_SUGGESTION_COOLDOWN_CYCLES
        )
        run_passes(mgr, cad)
        updates = apply_all(mgr, merged, self.device_type)
        if self.lock_probe and rnd == 0 and updates:
            self.lock_result = _lock_probe(mgr, cad, merged, self.device_type, list(updates))
        cpu += time.process_time() - t0
        self.cpu += cpu
        self.history.append({
            "round": rnd, "applied": {k: [options.get(k), v] for k, v in updates.items()},
            "cooldown_active_after_expiry": cooldown_active, "cadence": cad,
            "replay": m, "cpu_s": round(cpu, 1), "memo_hit": memo_hit, "memo_miss": memo_miss,
        })
        if (
            rnd == 0 and m["replayed"] and not (memo_hit + memo_miss)
            and getattr(self.match_side(options)[0].profile_store, "has_real_profiles", False)
        ):
            # The replay swallows a matcher exception as "no candidate", so a broken
            # call path would silently turn every replay unmatched.
            raise RuntimeError(f"{self.title}: the replay never reached the matcher")
        if not updates or rnd == self.rounds:
            self.done = True
            return
        for key, val in updates.items():
            if key not in self.initial:
                self.initial[key] = _initial_value(key, options, self.device_type)
            self.trajectory.setdefault(key, []).append((rnd, val))
        self.options = {**options, **updates}
        self.stamp = self.lifetime  # ws_apply_suggestions stamps the odometer
        self.round += 1

    def result(self) -> dict[str, Any]:
        """Verdict per setting; the final round's would-be applies count as still moving."""
        last_round = self.round
        final_updates = self.history[-1]["applied"] if self.history else {}
        keys: dict[str, Any] = {}
        for key in sorted(set(self.trajectory) | set(final_updates)):
            if key not in self.initial:
                self.initial[key] = _initial_value(key, self.options, self.device_type)
            pts = list(self.trajectory.get(key, []))
            if key in final_updates:
                pts.append((last_round, final_updates[key][1]))
            values = [self.initial[key]] + [v for _, v in pts]
            keys[key] = classify(values, [r for r, _ in pts], last_round)
        # Labelled cycles detected whole and completed under the export's options. Under
        # the fixed point's, `erased` ones are no longer stored as completed
        # (interrupted, force-stopped or not stored at all) - evidence every pass then
        # ignores, which is how a suggestion ratchets on its own output (SUGGEST-03);
        # `fragmented` ones are still completed but shed an extra record.
        first = self.history[0]["replay"]["outcome"] if self.history else {}
        last = self.history[-1]["replay"]["outcome"] if self.history else {}
        erased, fragmented = [], []
        for idx, before in first.items():
            src = self.past[idx]
            label = src.get("profile_name")
            after = last.get(idx)
            if not label or label == "noise" or before != [1, "completed"] or after is None:
                continue
            row = {"id": src.get("id"), "label": label, "after": after}
            if after[0] == 0 or after[1] != "completed":
                erased.append(row)
            elif after[0] > 1:
                fragmented.append(row)
        return {
            "device": self.title, "device_type": self.device_type, "cycles": len(self.past),
            "replayable": sum(1 for c in self.past if _replayable(c)),
            "rounds": self.history, "keys": keys, "fixed_point": not final_updates,
            "erased": erased, "fragmented": fragmented,
            "cooldown_leaks": self.leaks, "lock_probe": self.lock_result,
            "legacy": list(self.legacy), "cpu_s": round(self.cpu, 1),
        }


def _replay_task(path: str, options: dict, idx: int, throttle_s: float | None) -> tuple:
    """Worker: replay one stored cycle of one device under ``options``."""
    t0 = time.process_time()
    _install_matcher_memo()
    hit0, miss0 = _MEMO_STATS["hit"], _MEMO_STATS["miss"]
    dev = _DEVICES[path]
    match_mgr, prebuilt = dev.match_side(options)
    rep = replay_one(
        dev.past[idx], dev.detector_config(options), match_mgr.profile_store, prebuilt,
        {**dev.entry_data, **options}, throttle_s, dev.recorded_stop, dev.idle_hold,
    )
    return (idx, rep, _MEMO_STATS["hit"] - hit0, _MEMO_STATS["miss"] - miss0,
            time.process_time() - t0)


def _quiet() -> None:
    logging.getLogger(ev.PKG_NAME).setLevel(logging.CRITICAL)
    logging.getLogger("homeassistant").setLevel(logging.CRITICAL)


def _drive(loops: list[DeviceLoop], jobs: int) -> None:
    """Run every device's rounds; replays are spread over ``jobs`` processes.

    A device's rounds are sequential (each needs the previous Apply all), but the
    cycles inside a round are independent, so they are the unit of work: the
    slowest device no longer sets the wall time on its own.
    """
    acc: dict[str, dict[str, Any]] = {}

    def _start(loop: DeviceLoop) -> list[tuple]:
        work = loop.jobs()
        # Longest traces first, so a round's tail is short.
        work.sort(key=lambda j: -len(loop.past[j[2]].get("power_data") or []))
        acc[loop.path] = {"left": len(work), "replays": {}, "hit": 0, "miss": 0, "cpu": 0.0}
        return work

    def _collect(loop: DeviceLoop, out: tuple) -> bool:
        idx, rep, hit, miss, cpu = out
        a = acc[loop.path]
        a["replays"][idx] = rep
        a["hit"] += hit
        a["miss"] += miss
        a["cpu"] += cpu
        a["left"] -= 1
        if a["left"]:
            return False
        loop.finish_round(a["replays"], a["hit"], a["miss"], a["cpu"])
        return True

    if jobs <= 1:
        _install_matcher_memo()
        for loop in loops:
            while not loop.done:
                for job in _start(loop):
                    _collect(loop, _replay_task(*job))
        return
    import multiprocessing  # noqa: PLC0415
    from concurrent.futures import FIRST_COMPLETED, wait  # noqa: PLC0415

    by_path = {loop.path: loop for loop in loops}
    # Fork, so the workers inherit every device's rebuilt envelopes. Under pytest the
    # parent also runs pycares' idle shutdown thread (blocked on a queue, holding no
    # lock a worker takes), which is what Python's multi-threaded-fork warning is
    # about; the pool forks lazily on submit, so the whole block is covered.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*use of fork\(\) may lead to deadlocks.*",
            category=DeprecationWarning,
        )
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("fork"), initializer=_quiet,
        ) as pool:
            pending: dict[Any, str] = {}
            for loop in loops:
                for job in _start(loop):
                    pending[pool.submit(_replay_task, *job)] = loop.path
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    loop = by_path[pending.pop(fut)]
                    if _collect(loop, fut.result()) and not loop.done:
                        for job in _start(loop):
                            pending[pool.submit(_replay_task, *job)] = loop.path


def run(corpus: Path, rounds: int = DEFAULT_ROUNDS, jobs: int = 1, only: list[str] | None = None,
        legacy: tuple[str, ...] = (), keep_locks: bool = False, lock_probe: bool = True,
        idle_hold: bool = False) -> list[dict]:
    """Run the loop on every selected device; one result per device."""
    paths = device_paths(corpus, only)
    # Biggest first: their round-0 replays start before anything else is queued.
    paths.sort(key=lambda p: -(corpus / p).stat().st_size)
    with _isolated(), legacy_patch(tuple(legacy)):
        loops = [DeviceLoop(corpus, p, rounds, tuple(legacy), keep_locks, lock_probe, idle_hold)
                 for p in paths]
        _DEVICES.clear()
        _DEVICES.update({loop.path: loop for loop in loops})
        try:
            _drive(loops, jobs)
        finally:
            _DEVICES.clear()
            _MATCH_SIDES.clear()
            _CFGS.clear()
    return sorted((loop.result() for loop in loops), key=lambda r: r["device"])


BAD = ("ladder", "oscillates", "unsettled")


def failures(results: list[dict]) -> list[str]:
    """Everything the fixed-point contract forbids, one line each."""
    out = []
    for r in results:
        for key, k in r["keys"].items():
            if k["verdict"] in BAD:
                out.append(f"{r['device']}: {key} {k['verdict']} {k['values']}")
        for leak in r["cooldown_leaks"]:
            out.append(f"{r['device']}: non-corrective {leak} stored during the cooldown")
        if any(row["cooldown_active_after_expiry"] for row in r["rounds"]):
            out.append(f"{r['device']}: cooldown never expired")
        lp = r.get("lock_probe") or {}
        if lp.get("applied"):
            out.append(f"{r['device']}: Apply all changed muted {lp['applied']}")
        if r.get("erased"):
            out.append(
                f"{r['device']}: the loop erased {len(r['erased'])} labelled cycle(s) "
                f"{[(d['label'], d['after']) for d in r['erased']]}"
            )
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def print_report(results: list[dict], rounds: int) -> None:
    print(f"rounds <= {rounds} applies; per device: replayable cycles, fixed point reached,"
          " CPU seconds")
    for r in results:
        first, last = r["rounds"][0]["replay"], r["rounds"][-1]["replay"]
        moved = " ".join(
            f"{k}={first[k]}->{last[k]}" if first[k] != last[k] else f"{k}={first[k]}"
            for k in ("splits", "lost", "truncated", "interrupted", "smart", "idle_above_stop")
        )
        print(f"\n{r['device']} [{r['device_type']}] cycles={r['cycles']} replayable={r['replayable']}"
              f" applies={len(r['rounds']) - 1} fixed_point={r['fixed_point']} cpu={r['cpu_s']}s"
              f"\n    replay first->last round: {moved}")
        for key, k in r["keys"].items():
            seq = " -> ".join(_fmt(v) for v in k["values"])
            print(f"    {key:34s} {k['verdict']:10s} applies={k['applies']}  {seq}")
        for kind in ("erased", "fragmented"):
            if r.get(kind):
                print(f"    labelled cycles {kind}: {len(r[kind])}"
                      f" {[(d['label'], d['after']) for d in r[kind]]}")
        lp = r.get("lock_probe")
        if lp:
            print(f"    lock probe: muted={len(lp['locked'])} stored_by_cascade={lp['stored_by_cascade']}"
                  f" applied={lp['applied']}")
        if r["cooldown_leaks"]:
            print(f"    cooldown leaks: {r['cooldown_leaks']}")
    tally: dict[str, dict[str, int]] = {}
    for r in results:
        for key, k in r["keys"].items():
            tally.setdefault(key, {}).setdefault(k["verdict"], 0)
            tally[key][k["verdict"]] += 1
    print("\nper setting (devices):")
    for key in sorted(tally):
        print(f"    {key:34s} " + ", ".join(f"{v} {n}" for v, n in sorted(tally[key].items())))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="max applies per device")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--device", action="append", help="substring of a corpus path (repeatable)")
    ap.add_argument("--legacy", action="append", choices=LEGACY_RULES, default=[],
                    help="re-enable a removed suggestion with its old logic (revert check)")
    ap.add_argument("--keep-locks", action="store_true", help="keep the export's muted keys")
    ap.add_argument("--no-lock-probe", action="store_true")
    ap.add_argument("--idle-hold", action="store_true",
                    help="a kept tail at/above the new stop threshold is standby: force-end, no 0 W tail")
    ap.add_argument("--corpus", default=str(REPO / "cycle_data"))
    ap.add_argument("--json", help="write the per-device results here")
    a = ap.parse_args(argv)
    corpus = Path(a.corpus)
    if not corpus.is_dir():
        print(f"no corpus at {corpus}")
        return 2
    t0 = time.time()
    results = run(corpus, a.rounds, a.jobs, a.device, tuple(a.legacy), a.keep_locks,
                  not a.no_lock_probe, a.idle_hold)
    if not results:
        print("no device with enough re-detectable history")
        return 2
    print_report(results, a.rounds)
    bad = failures(results)
    print(f"\n{len(results)} devices, wall {time.time() - t0:.0f}s")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1, default=str))
    if bad:
        print("\nNOT A FIXED POINT:")
        for line in bad:
            print("    " + line)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
