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
"""Headless cycle-replay 'Playground' backend (Group F3).

Pure, executor-safe logic behind the panel's Playground tab. Nothing here
touches Home Assistant, fires events, or does I/O; the WebSocket handlers in
``ws_api.py`` call these helpers inside ``hass.async_add_executor_job``.

Two entry points:

- :func:`run_playground_batch` - replays stored cycles through a *fresh*
  headless :class:`CycleDetector` (with the device's live settings, optionally
  overridden) and returns a structured per-cycle event log + outcome plus an
  aggregate summary. The detector is driven exactly as in production
  (``process_reading`` fed the cycle's own trace), and a synchronous profile
  matcher (the real Stage 1-4 pipeline via ``analysis.compute_matches_worker``)
  is wired in so match/ambiguous/unmatched events are captured. No live HA
  events are fired - transitions are collected into an in-memory buffer.

- :func:`dtw_debug_payload` - the score breakdown (Stage 2 / DTW / Stage 4),
  the two resampled traces on a shared grid, and the DTW warping path for one
  cycle vs one profile (the DTW visualizer).

Both top-level entry points are defensive: they never raise, returning an
``{"error": ...}`` marker instead so the WS handlers can relay it.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np

from homeassistant.util import dt as dt_util

from . import analysis
from . import notification_rules as notif_rules
from . import progress as progress_mod
from .const import (
    CONF_ABRUPT_DROP_WATTS,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_END_REPEAT_COUNT,
    CONF_INTERRUPTED_MIN_SECONDS,
    CONF_MATCH_PERSISTENCE,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_NOTIFY_ACTIONS,
    CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_NOTIFY_FINISH_SERVICES,
    CONF_NOTIFY_MILESTONES,
    CONF_NOTIFY_START_SERVICES,
    CONF_OFF_DELAY,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_RUNNING_DEAD_ZONE,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CYCLE_OVERRUN_ANOMALY_RATIO,
    CYCLE_UNDERRUN_ANOMALY_RATIO,
    DEFAULT_MATCH_PERSISTENCE,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_NOTIFY_MILESTONES,
    MATCH_CORR_WEIGHT,
    MATCH_DDTW_DIST_SCALE,
    MATCH_DTW_BLEND,
    MATCH_DTW_DIST_SCALE,
    MATCH_DTW_ENSEMBLE_W,
    MATCH_DTW_RESAMPLE_N,
    MATCH_DURATION_SCALE,
    MATCH_DURATION_WEIGHT,
    MATCH_ENERGY_SCALE,
    MATCH_ENERGY_WEIGHT,
    MATCH_MAE_PEAK_FLOOR,
    MATCH_MAE_REF_PEAK,
    MATCH_MAE_SCALE,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_IDLE,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_UNKNOWN,
    TerminationReason,
)
from .cycle_detector import CycleDetector, CycleDetectorConfig
from .profile_store import _ambiguity_from_candidates, decompress_power_data

_LOGGER = logging.getLogger(__name__)

# The most recent N cycles to replay when the caller does not name any.
DEFAULT_RECENT_CYCLES = 20
# Hard upper bound on cycles simulated in one batch call (defence in depth on
# top of the caller-supplied ``concurrency`` cap).
MAX_BATCH_CYCLES = 50
# Cap the per-cycle event log so a pathological trace cannot bloat the payload.
MAX_EVENTS_PER_CYCLE = 300

# Override keys the Playground honours, mapped to CycleDetectorConfig fields.
# Only detection-relevant knobs matter; everything else in settings_override is
# ignored safely.
_OVERRIDE_FIELD_MAP: dict[str, tuple[str, Callable[[Any], Any]]] = {
    CONF_MIN_POWER: ("min_power", float),
    CONF_OFF_DELAY: ("off_delay", int),
    CONF_MIN_OFF_GAP: ("min_off_gap", int),
    CONF_COMPLETION_MIN_SECONDS: ("completion_min_seconds", int),
    CONF_END_REPEAT_COUNT: ("end_repeat_count", int),
    CONF_START_THRESHOLD_W: ("start_threshold_w", float),
    CONF_STOP_THRESHOLD_W: ("stop_threshold_w", float),
    CONF_RUNNING_DEAD_ZONE: ("running_dead_zone", int),
    CONF_START_DURATION_THRESHOLD: ("start_duration_threshold", float),
    CONF_ABRUPT_DROP_WATTS: ("abrupt_drop_watts", float),
    CONF_INTERRUPTED_MIN_SECONDS: ("interrupted_min_seconds", int),
}

# Matching options the Playground honours, mapped to the ``match_config`` key
# (Stage-1 duration gate) they drive. Only options the user can ACTUALLY set in
# Settings are here, so a value found in the Playground can be applied for real -
# the scoring weights (corr/duration/energy) and dtw_bandwidth are ML-tuned, not
# user-settable, so exposing them would be pointless. Anything else in
# ``settings_override`` is ignored.
_MATCH_OVERRIDE_KEYS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO: ("min_duration_ratio", float),
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO: ("max_duration_ratio", float),
}


def apply_match_overrides(
    match_config: dict[str, Any], settings_override: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a copy of ``match_config`` with the recognised matching options from
    ``settings_override`` overlaid onto the matcher-config keys they drive.
    Unknown/None/malformed values are ignored, so a detection-only override leaves
    matching byte-identical to the live config."""
    if not settings_override:
        return match_config
    out = dict(match_config)
    for opt_key, (cfg_key, coerce) in _MATCH_OVERRIDE_KEYS.items():
        val = settings_override.get(opt_key)
        if val is None:
            continue
        try:
            out[cfg_key] = coerce(val)
        except (TypeError, ValueError):
            pass
    return out


def build_sim_config(
    base: CycleDetectorConfig, settings_override: dict[str, Any] | None
) -> CycleDetectorConfig:
    """Return a copy of ``base`` with the recognised override keys applied.

    Unknown keys and un-coercible values are ignored so a malformed override can
    never break a simulation. ``base`` is left untouched.
    """
    if not isinstance(settings_override, dict) or not settings_override:
        return base
    changes: dict[str, Any] = {}
    for key, value in settings_override.items():
        mapping = _OVERRIDE_FIELD_MAP.get(key)
        if mapping is None or value is None:
            continue
        field, coerce = mapping
        try:
            changes[field] = coerce(value)
        except (TypeError, ValueError):
            continue
    if not changes:
        return base
    try:
        return replace(base, **changes)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return base


def _cycle_base_time(cycle: dict[str, Any]) -> datetime:
    """Timezone-aware anchor for a cycle's offset-0 reading.

    Prefers the stored ISO ``start_time``; falls back to a fixed UTC epoch so
    offsets remain well-defined even for malformed cycles.
    """
    raw = cycle.get("start_time")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str) and raw:
        parsed = dt_util.parse_datetime(raw)
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


def _cycle_label(cycle: dict[str, Any]) -> str | None:
    """The cycle's confirmed profile label (profile_name, else label)."""
    for key in ("profile_name", "label"):
        val = cycle.get(key)
        if isinstance(val, str) and val and val.lower() != "noise":
            return val
    return None


def _build_match_snapshots(
    store: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    """Prepare the matcher snapshots + config once from the store.

    Mirrors :meth:`ProfileStore.match_profile`: one snapshot per profile using
    its sample cycle's decompressed trace, plus the store's live matching config
    (with any on-device tuned weight overrides merged in).

    Also applies Stage-5 group collapsing via
    :meth:`ProfileStore._grouped_snapshots`: returns the collapsed snapshot list
    (where each cohesive group is represented by a single ``__group__*``
    aggregate candidate), plus ``group_members`` and ``member_snaps`` for the
    Stage-5 member-resolution step in :func:`_simulate_one`.

    Returns ``(grouped_snapshots, match_config, group_members, member_snaps)``.
    When no cohesive groups exist ``group_members`` and ``member_snaps`` are both
    empty dicts and behaviour is identical to before.
    """
    snapshots: list[dict[str, Any]] = []
    try:
        data = getattr(store, "_data", {}) or {}
        profiles = data.get("profiles", {}) or {}
        # Include imported reference cycles: an import-only profile samples from
        # reference_cycles, so without them it would be dropped as a candidate and
        # the Playground auto-detect would never match a downloaded profile.
        past = data.get("past_cycles", []) or []
        refs = data.get("reference_cycles", []) or []
        by_id = {c.get("id"): c for c in (list(past) + list(refs)) if isinstance(c, dict)}
        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            sample_cycle = by_id.get(profile.get("sample_cycle_id"))
            if not sample_cycle:
                continue
            sample_p = decompress_power_data(sample_cycle)
            if not sample_p:
                continue
            avg_dur = (
                profile.get("avg_duration")
                or sample_cycle.get("duration")
                or 0.0
            )
            snapshots.append(
                {
                    "name": name,
                    "avg_duration": float(avg_dur),
                    "sample_power": [p for _, p in sample_p],
                }
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground: snapshot build failed: %s", exc)

    # Stage-5: collapse cohesive profile groups into aggregate candidates.
    group_members: dict[str, list[str]] = {}
    member_snaps: dict[str, Any] = {}
    try:
        grouped_snaps, group_members, member_snaps = store._grouped_snapshots(  # pylint: disable=protected-access
            snapshots
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground: _grouped_snapshots failed: %s", exc)
        grouped_snaps = snapshots

    config = _matching_config(store)
    return grouped_snaps, config, group_members, member_snaps


def _matching_config(store: Any) -> dict[str, Any]:
    """Live matcher config from the store (defaults + tuned overrides)."""
    config: dict[str, Any] = {
        "min_duration_ratio": float(getattr(store, "_min_duration_ratio", 0.07)),
        "max_duration_ratio": float(getattr(store, "_max_duration_ratio", 1.5)),
        "dtw_bandwidth": float(getattr(store, "dtw_bandwidth", 0.2)),
    }
    try:
        overrides = store._matching_overrides()  # pylint: disable=protected-access
        if isinstance(overrides, dict):
            config.update(overrides)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return config


def decide_commit(
    raw_name: str | None,
    is_ambiguous: bool,
    commit_state: dict[str, Any],
    persistence: int,
) -> str | None:
    """Advance the persistence-gated match-commit state (mirror of the manager's
    core rule) and report the event to emit.

    Mutates ``commit_state`` (``candidate``/``count``/``name``): a candidate must
    be the non-ambiguous top-1 for ``persistence`` consecutive calls before it is
    committed, and the committed ``name`` is held (a one-off wobble resets the
    streak but never switches the commit). Returns ``"match_commit"`` on the first
    commit, ``"match_changed"`` on a later switch, or ``None`` otherwise. Pure +
    unit-testable; event emission / reporting stay in the caller.
    """
    if not raw_name or is_ambiguous:
        return None
    if raw_name == commit_state["candidate"]:
        commit_state["count"] += 1
    else:
        commit_state["candidate"] = raw_name
        commit_state["count"] = 1
    if commit_state["count"] >= persistence and commit_state["name"] != raw_name:
        prev = commit_state["name"]
        commit_state["name"] = raw_name
        return "match_changed" if prev else "match_commit"
    return None


def _readings_from_cycle(
    cycle: dict[str, Any],
) -> tuple[list[tuple[datetime, float]], list[tuple[float, float]], datetime]:
    """Reconstruct (datetime, power) readings + (offset, power) points + base time."""
    points = decompress_power_data(cycle)
    base = _cycle_base_time(cycle)
    readings = [(base + timedelta(seconds=float(o)), float(p)) for o, p in points]
    return readings, points, base


def _simulate_one(
    cycle: dict[str, Any],
    sim_config: CycleDetectorConfig,
    snapshots: list[dict[str, Any]],
    match_config: dict[str, Any],
    store: Any = None,
    group_members: dict[str, list[str]] | None = None,
    member_snaps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one stored cycle through a fresh headless detector.

    Returns ``{cycle_id, profile_name, events, outcome}``. Never raises.

    When ``group_members`` is non-empty the matcher applies Stage-5 group
    resolution: a winning ``__group__*`` aggregate candidate is resolved to its
    best-fitting member profile so ``outcome["match_profile"]`` always contains
    a real profile name, never a group key.
    """
    cycle_id = cycle.get("id")
    label = _cycle_label(cycle)
    events: list[dict[str, Any]] = []
    outcome: dict[str, Any] = {
        "detected": False,
        "detected_duration_s": None,
        "stored_duration_s": _safe_float(cycle.get("duration")),
        "match_profile": None,
        "match_correct": None,
        "ambiguous": False,
        "termination_reason": None,
        "status": None,
        "detected_count": 0,
    }

    try:
        readings, _points, base = _readings_from_cycle(cycle)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground: bad cycle %s: %s", cycle_id, exc)
        return {
            "cycle_id": cycle_id,
            "profile_name": label,
            "events": events,
            "outcome": outcome,
        }

    if len(readings) < 5:
        return {
            "cycle_id": cycle_id,
            "profile_name": label,
            "events": events,
            "outcome": outcome,
        }

    # Shared mutable state for the callbacks (offset seconds from cycle start).
    cursor = {"t": 0.0}
    captured: list[dict[str, Any]] = []
    last_match: dict[str, Any] = {"name": None, "conf": 0.0, "ambiguous": False}
    # Dedupe consecutive identical match outcomes so the log stays readable.
    last_logged_match: dict[str, Any] = {"kind": None, "name": None}

    def _emit(etype: str, detail: str) -> None:
        if len(events) < MAX_EVENTS_PER_CYCLE:
            events.append({"t": round(cursor["t"], 1), "type": etype, "detail": detail})

    def _on_state_change(old_state: str, new_state: str) -> None:
        _emit("state", f"{old_state}->{new_state}")

    def _on_cycle_end(cycle_data: dict[str, Any]) -> None:
        captured.append(cycle_data)
        _emit(
            "end",
            "reason={reason} status={status} dur={dur:.0f}s".format(
                reason=cycle_data.get("termination_reason"),
                status=cycle_data.get("status"),
                dur=float(cycle_data.get("duration") or 0.0),
            ),
        )

    def _matcher(
        det_readings: list[tuple[datetime, float]],
    ) -> tuple[str | None, float, float, str | None, bool, bool]:
        if len(det_readings) < 5 or not snapshots:
            return (None, 0.0, 0.0, None, False, False)
        powers = [p for _, p in det_readings]
        duration = (det_readings[-1][0] - det_readings[0][0]).total_seconds()
        try:
            candidates = analysis.compute_matches_worker(
                powers, duration, snapshots, match_config
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Playground match failed: %s", exc)
            candidates = []

        if not candidates:
            if last_logged_match["kind"] != "unmatched":
                _emit("unmatched", "no candidate")
                last_logged_match["kind"] = "unmatched"
                last_logged_match["name"] = None
            last_match["name"] = None
            last_match["conf"] = 0.0
            last_match["ambiguous"] = False
            return (None, 0.0, 0.0, None, False, False)

        # Stage-5: resolve a winning group aggregate to its best-fitting member.
        # This mirrors profile_store._stage5_pick_member used in production.
        # Note: dtw_debug_payload always works on an individual profile, so
        # Stage-5 resolution is only needed here in the batch matcher.
        if group_members and candidates:
            top = candidates[0]
            gkey = top.get("name", "")
            if gkey.startswith("__group__") and store is not None:
                members = group_members.get(gkey, [])
                if members:
                    try:
                        powers_list = list(powers)
                        member_name, _, _ = store._stage5_pick_member(  # pylint: disable=protected-access
                            powers_list, duration, members, member_snaps or {}
                        )
                        candidates[0] = dict(top, name=member_name)
                        _emit("group_resolved", f"{gkey} -> {member_name}")
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        _LOGGER.debug(
                            "Playground stage5 pick_member failed: %s", exc
                        )

        best = candidates[0]
        margin, is_ambiguous = _ambiguity_from_candidates(candidates)
        name = best.get("name")
        conf = float(best.get("score") or 0.0)
        expected = float(best.get("profile_duration") or 0.0)
        last_match["name"] = name
        last_match["conf"] = conf
        last_match["ambiguous"] = bool(is_ambiguous)

        if is_ambiguous:
            runner = candidates[1].get("name") if len(candidates) > 1 else None
            if (
                last_logged_match["kind"] != "ambiguous"
                or last_logged_match["name"] != name
            ):
                _emit("ambiguous", f"{name} vs {runner} (margin={margin:.3f})")
                last_logged_match["kind"] = "ambiguous"
                last_logged_match["name"] = name
        elif (
            last_logged_match["kind"] != "matched"
            or last_logged_match["name"] != name
        ):
            _emit("matched", f"{name} (conf={conf:.2f})")
            last_logged_match["kind"] = "matched"
            last_logged_match["name"] = name

        return (name, conf, expected, None, False, bool(is_ambiguous))

    detector = CycleDetector(
        sim_config,
        _on_state_change,
        _on_cycle_end,
        profile_matcher=_matcher,
        device_name="playground",
    )

    try:
        for ts, power in readings:
            cursor["t"] = (ts - base).total_seconds()
            detector.process_reading(power, ts)

        # Feed a synthetic quiet tail so a natural end (timeout / min-off-gap)
        # can fire, exactly as it would in production once the appliance goes
        # idle. Sized to comfortably exceed both the off-delay and the
        # soak-bridging min_off_gap.
        last_ts = readings[-1][0]
        tail_span = max(
            float(sim_config.off_delay or 0.0),
            float(sim_config.min_off_gap or 0.0),
        ) * 1.5 + 300.0
        step = 30.0
        n_steps = min(int(tail_span / step) + 1, 400)
        for i in range(1, n_steps + 1):
            ts = last_ts + timedelta(seconds=step * i)
            cursor["t"] = (ts - base).total_seconds()
            detector.process_reading(0.0, ts)
            if detector.state in (STATE_OFF, STATE_FINISHED) and captured:
                break

        # If a cycle started but never finalized (unusual), flush it so the
        # outcome is well-defined; it lands in the log as force-stopped.
        if not captured and detector.state != STATE_OFF:
            flush_ts = last_ts + timedelta(seconds=step * (n_steps + 2))
            cursor["t"] = (flush_ts - base).total_seconds()
            detector.force_end(flush_ts)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground simulate failed for %s: %s", cycle_id, exc)

    outcome["detected_count"] = len(captured)
    if captured:
        primary = max(captured, key=lambda c: float(c.get("duration") or 0.0))
        outcome["detected"] = True
        outcome["detected_duration_s"] = _safe_float(primary.get("duration"))
        outcome["termination_reason"] = primary.get("termination_reason")
        outcome["status"] = primary.get("status")

    outcome["match_profile"] = last_match["name"]
    outcome["ambiguous"] = bool(last_match["ambiguous"])
    if outcome["detected"] and last_match["name"] and label:
        outcome["match_correct"] = last_match["name"].strip() == label.strip()
    else:
        outcome["match_correct"] = None

    return {
        "cycle_id": cycle_id,
        "profile_name": label,
        "events": events,
        "outcome": outcome,
    }


def run_playground_batch(
    store: Any,
    cycle_ids: list[str] | None,
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    concurrency: int,
) -> dict[str, Any]:
    """Replay a set of cycles headlessly; return {results, summary}.

    ``concurrency`` caps how many of the selected cycles are simulated in this
    batch (batch size), clamped 1..MAX_BATCH_CYCLES. Executor-safe; never raises.
    """
    try:
        concurrency = max(1, min(MAX_BATCH_CYCLES, int(concurrency)))
    except (TypeError, ValueError):
        concurrency = 1

    summary: dict[str, Any] = {
        "cycles": 0,
        "requested": 0,
        "concurrency": concurrency,
        "detected": 0,
        "missed": 0,
        "false_end": 0,
        "match_correct": 0,
        "match_wrong": 0,
        "unmatched": 0,
        "skipped_ids": [],
    }

    try:
        past = list(store.get_past_cycles() or [])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground: get_past_cycles failed: %s", exc)
        return {"results": [], "summary": summary}

    by_id = {c.get("id"): c for c in past if isinstance(c, dict)}

    selected: list[dict[str, Any]] = []
    skipped: list[str] = []
    if cycle_ids:
        for cid in cycle_ids:
            cycle = by_id.get(cid)
            if cycle is None:
                skipped.append(cid)
            else:
                selected.append(cycle)
    else:
        selected = past[-DEFAULT_RECENT_CYCLES:]

    summary["requested"] = len(cycle_ids) if cycle_ids else len(selected)

    # Batch-size cap: simulate up to ``concurrency`` of the selected cycles
    # (the runner is sequential). Any selected cycles beyond the cap are reported
    # as skipped rather than silently dropped, so ``requested`` always reconciles
    # with ``len(results) + len(skipped_ids)``.
    to_run = selected[:concurrency]
    if len(selected) > concurrency:
        # Account for every capped cycle (even one lacking an id) so that
        # requested == len(results) + len(skipped_ids) always reconciles.
        skipped.extend(str(c.get("id") or "") for c in selected[concurrency:])
    summary["skipped_ids"] = skipped

    config = build_sim_config(base_config, settings_override)
    snapshots, match_config, group_members, member_snaps = _build_match_snapshots(store)
    match_config = apply_match_overrides(match_config, settings_override)

    results: list[dict[str, Any]] = []
    for cycle in to_run:
        res = _simulate_one(
            cycle, config, snapshots, match_config,
            store=store, group_members=group_members, member_snaps=member_snaps,
        )
        results.append(res)
        oc = res["outcome"]
        summary["cycles"] += 1
        if oc.get("detected"):
            summary["detected"] += 1
            if int(oc.get("detected_count") or 0) > 1:
                summary["false_end"] += 1
            correct = oc.get("match_correct")
            if oc.get("match_profile") is None:
                summary["unmatched"] += 1
            elif correct is True:
                summary["match_correct"] += 1
            elif correct is False:
                summary["match_wrong"] += 1
        else:
            summary["missed"] += 1

    return {"results": results, "summary": summary}


# ─── Single-cycle faithful simulation (Simulate mode) ───────────────────────────


_PROGRESS_STATES = (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
# States in which no progress estimate is shown (mirrors _update_remaining_only).
_DEAD_STATES = (STATE_OFF, STATE_UNKNOWN, STATE_IDLE)
_SIM_SERIES_THROTTLE_S = 5.0  # production recomputes progress every 5s


def simulate_cycle_detail(
    cycle: dict[str, Any],
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    store: Any,
    options: dict[str, Any] | None,
    price: float | None = None,
    compute_series: bool = True,
    prebuilt: tuple[Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Faithful single-cycle replay for the Playground "Simulate" view.

    Drives the REAL :class:`CycleDetector` + real Stage 1-4 matcher over the
    cycle's own trace and, at production's 5s cadence, calls the SAME
    :mod:`progress` and :mod:`notification_rules` functions the live integration
    uses - so the returned timeline is byte-for-byte what would happen live. No
    detection/progress/notification math is implemented here; this only
    orchestrates the shared code. Never raises; returns ``{"error": ...}`` on
    failure. Read-only: nothing is persisted and no notifications are sent.

    Returns ``{cycle_id, label, duration_s, config_summary, series, events,
    alerts, outcome}`` (see the design doc for the field contract).
    """
    options = options or {}
    try:
        return _simulate_cycle_detail_inner(
            cycle, base_config, settings_override, store, options, price,
            compute_series, prebuilt,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground detail sim failed for %s: %s", cycle.get("id"), exc)
        return {"error": str(exc), "cycle_id": cycle.get("id")}


def _device_type_of(config: CycleDetectorConfig) -> str:
    return getattr(config, "device_type", "washing_machine")


def simulate_cycle_detail_by_id(
    store: Any,
    cycle_id: str,
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    options: dict[str, Any] | None,
    price: float | None = None,
) -> dict[str, Any]:
    """Find a stored cycle by id and simulate it. Runs the (in-memory) store
    lookup and the replay together so the WS handler can offload the whole thing
    to an executor thread. Returns ``{"error": "not_found", ...}`` when the id is
    unknown. Never raises."""
    try:
        cycle = next(
            (c for c in store.get_past_cycles() if c.get("id") == cycle_id), None
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground detail lookup failed for %s: %s", cycle_id, exc)
        return {"error": str(exc), "cycle_id": cycle_id}
    if cycle is None:
        return {"error": "not_found", "cycle_id": cycle_id}
    return simulate_cycle_detail(
        cycle, base_config, settings_override, store, options, price
    )


def _simulate_cycle_detail_inner(
    cycle: dict[str, Any],
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    store: Any,
    options: dict[str, Any],
    price: float | None,
    compute_series: bool = True,
    prebuilt: tuple[Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    config = build_sim_config(base_config, settings_override)
    device_type = _device_type_of(config)
    label = _cycle_label(cycle)
    readings, _points, base = _readings_from_cycle(cycle)
    stored_duration = _safe_float(cycle.get("duration"))

    outcome: dict[str, Any] = {
        "detected": False,
        "detected_count": 0,
        "termination_reason": None,
        "status": None,
        "final_duration_s": None,
        "matched_profile": None,
        "match_correct": None,
        "confidence": None,
        "expected_s": None,
        "overrun_ratio": None,
        "projected_energy_wh": None,
        "projected_cost": None,
    }
    if prebuilt is not None:
        snapshots, match_config, group_members, member_snaps = prebuilt
    else:
        snapshots, match_config, group_members, member_snaps = _build_match_snapshots(store)
    # Overlay any matcher-knob overrides. Because history/sweep run through this
    # same function, a swept matching value flows in via settings_override too;
    # applying to a copy keeps the shared prebuilt match_config untouched.
    match_config = apply_match_overrides(match_config, settings_override)

    empty = {
        "cycle_id": cycle.get("id"),
        "label": label,
        "duration_s": stored_duration,
        "config_summary": _sim_config_summary(config),
        "series": [],
        "events": [],
        "alerts": [],
        "outcome": outcome,
    }
    if len(readings) < 5:
        return empty

    # Per-sim end-expectation cache, threaded through the shared progress helpers
    # exactly like the manager threads self._ml_end_expectation_cache.
    endexp_cache: list[Any] = [None]

    def _end_exp_fn(name: str, dur: float) -> Any:
        exp, endexp_cache[0] = progress_mod.profile_end_expectation(
            store, name, dur, endexp_cache[0]
        )
        return exp

    events: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    captured: list[dict[str, Any]] = []
    cursor = {"t": 0.0}
    last_match: dict[str, Any] = {
        "name": None, "conf": 0.0, "ambiguous": False, "expected": 0.0,
    }
    last_logged = {"kind": None, "name": None}
    # Persistence-gated commit mirroring the manager: a candidate must be top-1 for
    # `match_persistence` consecutive matches (and not ambiguous) before it is
    # committed, and the committed match is HELD (a one-off wobble doesn't switch
    # it). The detector still receives the raw top-1 (detection unchanged); only the
    # reported series/events use the committed match, so the Playground shows what
    # the live integration would show - not raw per-interval churn.
    match_persistence = max(1, int(
        (options or {}).get(CONF_MATCH_PERSISTENCE, DEFAULT_MATCH_PERSISTENCE)
    ))
    commit_state: dict[str, Any] = {"candidate": None, "count": 0, "name": None}
    smoothed = {"v": 0.0}
    flags = {"detected": False, "pre_complete": False, "start": False}

    def _emit(etype: str, detail: str, severity: str = "info") -> None:
        if len(events) < MAX_EVENTS_PER_CYCLE:
            events.append(
                {"t": round(cursor["t"], 1), "type": etype, "detail": detail,
                 "severity": severity}
            )

    # --- notification config (decisions reuse notification_rules) ---
    start_configured = bool(
        options.get(CONF_NOTIFY_START_SERVICES) or options.get(CONF_NOTIFY_ACTIONS)
    )
    finish_configured = bool(
        options.get(CONF_NOTIFY_FINISH_SERVICES) or options.get(CONF_NOTIFY_ACTIONS)
    )
    before_end = float(
        options.get(CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES)
        or 0.0
    )
    quiet_bounds = notif_rules.quiet_hours_bounds(options)

    def _held(offset: float) -> bool:
        return notif_rules.in_quiet_hours(quiet_bounds, base + timedelta(seconds=offset))

    def _on_state_change(old_state: str, new_state: str) -> None:
        _emit("state", f"{old_state}->{new_state}")
        # A new cycle is starting after a previous one ended: clear the inherited
        # match-persistence streak so this cycle matches fresh (see _on_cycle_end).
        # PAUSED->RUNNING resumes don't arm pending_reset, so they are unaffected.
        if new_state == STATE_RUNNING and flags.get("pending_reset"):
            flags["pending_reset"] = False
            commit_state.update(candidate=None, count=0, name=None)
            last_match.update(name=None, conf=0.0, expected=0.0, ambiguous=False)
            last_logged.update(kind=None, name=None)
        if (
            not flags["detected"]
            and new_state == STATE_RUNNING
            and old_state in (STATE_OFF, STATE_UNKNOWN, STATE_STARTING, STATE_IDLE)
        ):
            flags["detected"] = True
            _emit("detected", "cycle detected (running)")
            if start_configured and not flags["start"]:
                flags["start"] = True
                # Start notifications are never delayed by quiet hours (live contract),
                # so the sim always emits them immediately.
                _emit("notify_start", "start notification")

    def _on_cycle_end(cycle_data: dict[str, Any]) -> None:
        captured.append(cycle_data)
        reason = cycle_data.get("termination_reason")
        _emit("finished", f"reason={reason} status={cycle_data.get('status')}", "info")
        # Arm a match-state reset for the NEXT cycle. We reset at the next cycle's
        # start (not here) so the final cycle's committed match survives to be read
        # into `outcome` after the loop; a second sub-cycle then starts a fresh
        # match-persistence streak, mirroring the live manager (per-cycle reset).
        flags["pending_reset"] = True

    def _matcher(det_readings: list[tuple[datetime, float]]):
        if len(det_readings) < 5 or not snapshots:
            return (None, 0.0, 0.0, None, False, False)
        powers = [p for _, p in det_readings]
        duration = (det_readings[-1][0] - det_readings[0][0]).total_seconds()
        try:
            candidates = analysis.compute_matches_worker(
                powers, duration, snapshots, match_config
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Playground detail match failed: %s", exc)
            candidates = []
        if not candidates:
            if last_logged["kind"] != "unmatched":
                _emit("unmatched", "no candidate")
                last_logged["kind"] = "unmatched"
            # Hold any committed match on a transient miss (as the manager does).
            last_match.update(ambiguous=False)
            return (None, 0.0, 0.0, None, False, False)
        if group_members and candidates[0].get("name", "").startswith("__group__"):
            gkey = candidates[0]["name"]
            members = group_members.get(gkey, [])
            if members and store is not None:
                try:
                    member_name, _, _ = store._stage5_pick_member(  # noqa: SLF001
                        list(powers), duration, members, member_snaps or {}
                    )
                    candidates[0] = dict(candidates[0], name=member_name)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
        best = candidates[0]
        margin, is_ambiguous = _ambiguity_from_candidates(candidates)
        raw_name = best.get("name")
        raw_conf = float(best.get("score") or 0.0)
        raw_expected = float(best.get("profile_duration") or 0.0)

        # Persistence-gated commit (mirror of the manager's core rule), extracted
        # into decide_commit() for unit-testability.
        commit_event = decide_commit(raw_name, is_ambiguous, commit_state, match_persistence)
        if commit_event:
            _emit(commit_event, f"{raw_name} (conf={raw_conf:.2f})")
            last_logged.update(kind="matched", name=raw_name)
        elif is_ambiguous and not commit_state["name"]:
            # Ambiguous before any commit: stay 'detecting', surface it once.
            if last_logged["kind"] != "ambiguous" or last_logged["name"] != raw_name:
                runner = candidates[1].get("name") if len(candidates) > 1 else None
                _emit("match_ambiguous", f"{raw_name} vs {runner} (margin={margin:.3f})", "warn")
                last_logged.update(kind="ambiguous", name=raw_name)

        # Reported state = the COMMITTED match (held); its confidence/expected are
        # that profile's own values this interval (looked up among the candidates).
        cname = commit_state["name"]
        if cname:
            cc = next((c for c in candidates if c.get("name") == cname), None)
            last_match.update(
                name=cname,
                conf=float(cc.get("score") or 0.0) if cc else (last_match.get("conf") or 0.0),
                expected=float(cc.get("profile_duration") or 0.0) if cc else (last_match.get("expected") or 0.0),
                ambiguous=False,
            )
        else:
            last_match.update(name=None, conf=0.0, expected=0.0, ambiguous=bool(is_ambiguous))

        # The DETECTOR still receives the RAW top-1, so detection / smart-termination
        # behaviour is byte-identical to before this reporting change.
        return (raw_name, raw_conf, raw_expected, None, False, bool(is_ambiguous))

    detector = CycleDetector(
        config, _on_state_change, _on_cycle_end,
        profile_matcher=_matcher, device_name="playground-detail",
    )

    last_sample_t = -1e9

    def _sample(ts: datetime) -> None:
        nonlocal last_sample_t
        if not compute_series:
            return  # batch/sweep rows only need the outcome, not the per-step series
        offset = (ts - base).total_seconds()
        if offset - last_sample_t < _SIM_SERIES_THROTTLE_S:
            return
        last_sample_t = offset
        state = detector.state
        power = 0.0
        trace = detector.get_power_trace()
        if trace:
            power = float(trace[-1][1])
        energy_wh = float(getattr(detector, "_energy_since_idle_wh", 0.0) or 0.0)
        pt: dict[str, Any] = {
            "t": round(offset, 1),
            "power": round(power, 1),
            "energy_wh": round(energy_wh, 2),
            "state": state,
            "progress": None,
            "remaining_s": None,
            "phase": None,
            "confidence": round(last_match["conf"], 3) if last_match["name"] else None,
            "matched_profile": last_match["name"],
        }
        matched_dur = float(last_match["expected"] or 0.0)
        program = last_match["name"]
        if state not in _DEAD_STATES and program and matched_dur > 0:
            phase_result = None
            if len(trace) >= 10 and program != "detecting...":
                phase_result = progress_mod.estimate_phase_progress(
                    store, trace, offset, program
                )
            ml_pct = progress_mod.ml_progress_percent(
                store, options, matched_dur, trace, program, _end_exp_fn
            )
            result = progress_mod.compute_progress(
                device_type, matched_dur, offset, smoothed["v"], phase_result, ml_pct
            )
            if result is not None:
                smoothed["v"] = result.smoothed
                pt["progress"] = round(result.progress, 1)
                pt["remaining_s"] = round(result.remaining, 0)
                pt["phase"] = progress_mod.current_phase(
                    store, state, program, result.progress
                )
                wh, cost = progress_mod.projected_energy(
                    store, options, matched_dur, trace, program, result.progress,
                    energy_wh, price, _end_exp_fn,
                )
                pt["projected_energy_wh"] = round(wh, 1) if wh is not None else None
                pt["projected_cost"] = round(cost, 4) if cost is not None else None
                # One-time pre-completion marker (reuses the production predicate).
                if not flags["pre_complete"] and notif_rules.should_notify_pre_completion(
                    before_end, flags["pre_complete"], result.remaining,
                    result.progress, last_match["ambiguous"],
                ):
                    flags["pre_complete"] = True
                    held = _held(offset)
                    _emit(
                        "notify_held" if held else "notify_pre_complete",
                        "pre-completion notification"
                        + (" (held: quiet hours)" if held else ""),
                    )
        series.append(pt)

    try:
        for ts, power in readings:
            cursor["t"] = (ts - base).total_seconds()
            detector.process_reading(power, ts)
            _sample(ts)
        # Synthetic quiet tail so a natural end can fire, as in _simulate_one.
        last_ts = readings[-1][0]
        tail_span = max(
            float(config.off_delay or 0.0), float(config.min_off_gap or 0.0)
        ) * 1.5 + 300.0
        step = 30.0
        n_steps = min(int(tail_span / step) + 1, 400)
        for i in range(1, n_steps + 1):
            ts = last_ts + timedelta(seconds=step * i)
            cursor["t"] = (ts - base).total_seconds()
            detector.process_reading(0.0, ts)
            _sample(ts)
            if detector.state in (STATE_OFF, STATE_FINISHED) and captured:
                break
        if not captured and detector.state != STATE_OFF:
            flush_ts = last_ts + timedelta(seconds=step * (n_steps + 2))
            cursor["t"] = (flush_ts - base).total_seconds()
            detector.force_end(flush_ts)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground detail replay failed for %s: %s", cycle.get("id"), exc)

    # --- outcome ---
    primary = None
    if captured:
        primary = max(captured, key=lambda c: float(c.get("duration") or 0.0))
        outcome["detected"] = True
        outcome["detected_count"] = len(captured)
        outcome["termination_reason"] = primary.get("termination_reason")
        outcome["status"] = primary.get("status")
        outcome["final_duration_s"] = _safe_float(primary.get("duration"))
    outcome["matched_profile"] = last_match["name"]
    outcome["confidence"] = (
        round(float(last_match["conf"]), 3) if last_match["name"] else None
    )
    outcome["expected_s"] = (
        round(float(last_match["expected"] or 0.0), 1) or None
    )
    if outcome["detected"] and last_match["name"] and label:
        outcome["match_correct"] = last_match["name"].strip() == label.strip()
    # Projected energy/cost: the last LIVE estimate (the post-finish tail resets
    # the detector's accumulated energy, so series[-1] would read None).
    for pt in reversed(series):
        if pt.get("projected_energy_wh") is not None:
            outcome["projected_energy_wh"] = pt.get("projected_energy_wh")
            outcome["projected_cost"] = pt.get("projected_cost")
            break

    # --- finish + milestone markers (reuse production predicates) ---
    if captured and finish_configured:
        held = _held(cursor["t"])
        _emit(
            "notify_held" if held else "notify_finish",
            "finish notification" + (" (held: quiet hours)" if held else ""),
        )
        try:
            prev_life = int(store.get_lifetime_cycle_count())
        except Exception:  # pylint: disable=broad-exception-caught
            prev_life = 0
        crossed = notif_rules.milestone_crossed(
            prev_life, prev_life + 1,
            options.get(CONF_NOTIFY_MILESTONES, DEFAULT_NOTIFY_MILESTONES),
        )
        if crossed is not None:
            # Milestone notifications are held during quiet hours (live contract).
            m_held = _held(cursor["t"])
            _emit(
                "notify_held" if m_held else "notify_milestone",
                f"milestone {crossed} cycles" + (" (held: quiet hours)" if m_held else ""),
            )

    # --- alerts ---
    alerts: list[dict[str, Any]] = []
    expected_dur = float(last_match["expected"] or 0.0)
    final_dur = outcome["final_duration_s"] or 0.0
    if not outcome["detected"]:
        alerts.append({"code": "did_not_finish", "severity": "error",
                       "detail": "Cycle never reached a terminal state in the replay."})
    if outcome["detected"] and (outcome["detected_count"] or 0) > 1:
        alerts.append({"code": "false_end", "severity": "error",
                       "detail": f"Split into {outcome['detected_count']} cycles."})
    if outcome["matched_profile"] is None:
        alerts.append({"code": "unmatched", "severity": "warn",
                       "detail": "No profile matched this cycle."})
    if last_match["ambiguous"]:
        alerts.append({"code": "ambiguous", "severity": "warn",
                       "detail": "Match was ambiguous (two programs scored close)."})
    # How the cycle ended: predictive (smart / terminal-drop) vs the static
    # low-power fallback. Under auto-detect an unmatched cycle cannot use smart
    # end-prediction, so it only stops once power stays low for the off-delay -
    # or, if it never goes quiet, not at all. Surface which happened.
    term = str(outcome.get("termination_reason") or "")
    if outcome["detected"] and term == str(TerminationReason.FORCE_STOPPED):
        alerts.append({"code": "would_run_indefinitely", "severity": "error",
                       "detail": ("The cycle never ended on its own - only the safety "
                                  "force-stop finalized it in simulation. In real use it "
                                  "would keep counting as running until power stays low.")})
    elif outcome["detected"] and term == str(TerminationReason.TIMEOUT):
        off_min = max(1, round(float(getattr(config, "off_delay", 0) or 0) / 60))
        if outcome["matched_profile"] is None:
            alerts.append({"code": "timeout_end", "severity": "warn",
                           "detail": (f"Ended only by the low-power timeout: no profile matched, "
                                      f"so smart end-prediction could not run and it waited out "
                                      f"the {off_min} min off-delay after power dropped.")})
        else:
            alerts.append({"code": "timeout_end", "severity": "info",
                           "detail": (f"Ended by the low-power timeout, not smart prediction: it "
                                      f"waited out the {off_min} min off-delay after power dropped.")})
    if expected_dur > 0 and final_dur > 0:
        ratio = final_dur / expected_dur
        outcome["overrun_ratio"] = round(ratio, 3)
        if ratio >= CYCLE_OVERRUN_ANOMALY_RATIO:
            alerts.append({"code": "overrun", "severity": "warn",
                           "detail": f"Ran {ratio:.0%} of the profile's typical duration."})
        elif ratio <= CYCLE_UNDERRUN_ANOMALY_RATIO:
            alerts.append({"code": "underrun", "severity": "warn",
                           "detail": f"Finished at {ratio:.0%} of typical duration."})

    return {
        "cycle_id": cycle.get("id"),
        "label": label,
        "duration_s": stored_duration,
        "config_summary": _sim_config_summary(config),
        "series": series,
        "events": events,
        "alerts": alerts,
        "outcome": outcome,
    }


def _sim_config_summary(config: CycleDetectorConfig) -> dict[str, Any]:
    """Compact view of the effective detector config used for the sim."""
    return {
        "device_type": getattr(config, "device_type", None),
        "min_power": getattr(config, "min_power", None),
        "off_delay": getattr(config, "off_delay", None),
        "min_off_gap": getattr(config, "min_off_gap", None),
        "start_threshold_w": getattr(config, "start_threshold_w", None),
        "stop_threshold_w": getattr(config, "stop_threshold_w", None),
    }


# ─── Test-on-history rows + before/after diff ───────────────────────────────────


def _detail_to_row(detail: dict[str, Any]) -> dict[str, Any]:
    """Compact per-cycle row for the Test-on-history table from a detail sim."""
    o = detail.get("outcome", {})
    return {
        "cycle_id": detail.get("cycle_id"),
        "label": detail.get("label"),
        "detected": bool(o.get("detected")),
        "detected_count": int(o.get("detected_count") or 0),
        "matched_profile": o.get("matched_profile"),
        "match_correct": o.get("match_correct"),
        "confidence": o.get("confidence"),
        "termination_reason": (
            str(o.get("termination_reason")) if o.get("termination_reason") else None
        ),
        "status": o.get("status"),
        "duration_s": o.get("final_duration_s"),
        "stored_duration_s": detail.get("duration_s"),
        "expected_s": o.get("expected_s"),
        "overrun_ratio": o.get("overrun_ratio"),
        "alerts": [a.get("code") for a in detail.get("alerts", [])],
    }


def _rows_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    detected = sum(1 for r in rows if r["detected"])
    labelled = [r for r in rows if r["label"]]
    correct = sum(1 for r in labelled if r["match_correct"] is True)
    wrong = sum(1 for r in labelled if r["match_correct"] is False)
    unmatched = sum(1 for r in rows if r["detected"] and r["matched_profile"] is None)
    false_end = sum(1 for r in rows if (r["detected_count"] or 0) > 1)
    return {
        "cycles": total,
        "detected": detected,
        "labelled": len(labelled),
        "match_correct": correct,
        "match_wrong": wrong,
        "unmatched": unmatched,
        "false_end": false_end,
    }


def _run_rows(
    store: Any,
    cycles: list[dict[str, Any]],
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    options: dict[str, Any],
    price: float | None,
) -> list[dict[str, Any]]:
    # Snapshots are store-derived (independent of the cycle and the detector-level
    # settings_override), so build them ONCE and reuse across all cycles/values.
    prebuilt = _build_match_snapshots(store)
    rows: list[dict[str, Any]] = []
    for cycle in cycles:
        detail = simulate_cycle_detail(
            cycle, base_config, settings_override, store, options, price,
            compute_series=False, prebuilt=prebuilt,
        )
        if "error" in detail:
            continue
        rows.append(_detail_to_row(detail))
    return rows


def run_playground_history(
    store: Any,
    cycle_ids: list[str] | None,
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    options: dict[str, Any] | None,
    price: float | None,
    concurrency: int,
) -> dict[str, Any]:
    """Per-cycle rows for the Test-on-history table, plus a before/after diff when
    ``settings_override`` is set. Executor-safe; never raises."""
    options = options or {}
    try:
        concurrency = max(1, min(MAX_BATCH_CYCLES, int(concurrency)))
    except (TypeError, ValueError):
        concurrency = MAX_BATCH_CYCLES
    try:
        past = list(store.get_past_cycles() or [])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground history: get_past_cycles failed: %s", exc)
        return {"rows": [], "summary": _rows_summary([])}

    by_id = {c.get("id"): c for c in past if isinstance(c, dict)}
    if cycle_ids:
        selected = [by_id[c] for c in cycle_ids if c in by_id]
    else:
        selected = past[-DEFAULT_RECENT_CYCLES:]
    selected = selected[:concurrency]

    override = settings_override or None
    rows = _run_rows(store, selected, base_config, override, options, price)
    payload: dict[str, Any] = {"rows": rows, "summary": _rows_summary(rows)}

    if override:
        base_rows = _run_rows(store, selected, base_config, None, options, price)
        payload["baseline_rows"] = base_rows
        payload["baseline_summary"] = _rows_summary(base_rows)
        payload["diff"] = _diff_rows(base_rows, rows)
    return payload


def finalize_history(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    has_override: bool,
) -> dict[str, Any]:
    """Assemble the Test-on-history payload from rows collected across chunks
    (used by the server-side task runner). Reuses the same summary/diff helpers as
    the one-shot :func:`run_playground_history` so there is one aggregation path."""
    payload: dict[str, Any] = {"rows": rows, "summary": _rows_summary(rows)}
    if has_override and baseline_rows:
        payload["baseline_rows"] = baseline_rows
        payload["baseline_summary"] = _rows_summary(baseline_rows)
        payload["diff"] = _diff_rows(baseline_rows, rows)
    return payload


def _diff_rows(
    baseline: list[dict[str, Any]], override: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Which cycles changed between baseline and override runs (keyed by id)."""
    base_by = {r["cycle_id"]: r for r in baseline}
    newly_correct: list[str] = []
    regressed: list[str] = []
    end_timing_changed: list[str] = []
    for r in override:
        b = base_by.get(r["cycle_id"])
        if b is None:
            continue
        if b["match_correct"] is not True and r["match_correct"] is True:
            newly_correct.append(r["cycle_id"])
        elif b["match_correct"] is True and r["match_correct"] is not True:
            regressed.append(r["cycle_id"])
        bd, od = b.get("duration_s") or 0.0, r.get("duration_s") or 0.0
        if b.get("termination_reason") != r.get("termination_reason") or abs(bd - od) > 60.0:
            end_timing_changed.append(r["cycle_id"])
    return {
        "newly_correct": newly_correct,
        "regressed": regressed,
        "end_timing_changed": end_timing_changed,
    }


# ─── Parameter sweep (1D curve + 2D heatmap) ────────────────────────────────────


_SWEEP_OBJECTIVES = (
    "match_accuracy",
    "end_timing_accuracy",
    "false_end_rate",
    "median_overrun",
    "ambiguity_rate",
)
# Objectives where a LOWER metric is better (best = minimum), so the sweep picks
# the right winner and the panel colours the heatmap consistently.
_SWEEP_LOWER_IS_BETTER = frozenset({
    "false_end_rate",
    "median_overrun",
    "ambiguity_rate",
})


def _sweep_is_better(candidate: float, best: float, objective: str) -> bool:
    if objective in _SWEEP_LOWER_IS_BETTER:
        return candidate < best
    return candidate > best


def finalize_sweep_1d(
    param: str, objective: str, points: list[dict[str, Any]], current_value: Any
) -> dict[str, Any]:
    """Assemble a 1D sweep payload from per-value points collected across chunks,
    picking the best via the same direction rule as :func:`run_playground_sweep`."""
    best: dict[str, Any] | None = None
    for p in points:
        m = p.get("metric")
        if m is None:
            continue
        if best is None or _sweep_is_better(m, best["metric"], objective):
            best = {"value": p["value"], "metric": m}
    return {
        "param": param, "objective": objective, "points": points,
        "current_value": current_value,
        "best_value": best["value"] if best else None,
        "best_metric": best["metric"] if best else None,
    }


def finalize_sweep_2d(
    param_x: str, param_y: str, objective: str,
    x_values: list[float], y_values: list[float],
    grid: list[list[float | None]], current: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a 2D sweep payload from grid cells collected across chunks."""
    best: dict[str, Any] | None = None
    for j, row in enumerate(grid):
        for i, v in enumerate(row or []):
            if v is None:
                continue
            if best is None or _sweep_is_better(v, best["metric"], objective):
                best = {"x": x_values[i], "y": y_values[j], "metric": v}
    return {
        "param_x": param_x, "param_y": param_y, "objective": objective,
        "x_values": x_values, "y_values": y_values, "grid": grid,
        "best": best, "current": current,
    }


def objective_metric(rows: list[dict[str, Any]], objective: str) -> float | None:
    """Reduce a set of per-cycle rows to a single objective metric (0-1, or a
    ratio for median_overrun). Higher is better EXCEPT false_end_rate /
    median_overrun deviation (the caller/panel knows the direction)."""
    if not rows:
        return None
    detected = [r for r in rows if r["detected"]]
    labelled = [r for r in detected if r["label"]]
    if objective == "match_accuracy":
        if not labelled:
            return None
        return sum(1 for r in labelled if r["match_correct"] is True) / len(labelled)
    if objective == "false_end_rate":
        if not detected:
            return None
        return sum(1 for r in detected if (r["detected_count"] or 0) > 1) / len(detected)
    if objective == "ambiguity_rate":
        if not detected:
            return None
        return sum(1 for r in detected if "ambiguous" in (r["alerts"] or [])) / len(detected)
    if objective == "end_timing_accuracy":
        # Fraction of cycles whose *detected* end lands within 10% of that cycle's own
        # recorded duration (its true end) - NOT the profile median. Scoring against
        # the median would reward ending at the typical length even for a cycle that
        # legitimately ran long or short, so the sweep must compare to stored_duration_s.
        ok = 0
        n = 0
        for r in detected:
            ref = float(r.get("stored_duration_s") or 0.0)
            dur = float(r.get("duration_s") or 0.0)
            if ref <= 0 or dur <= 0:
                continue
            n += 1
            if abs(dur - ref) <= 0.10 * ref:
                ok += 1
        return (ok / n) if n else None
    if objective == "median_overrun":
        # Score by the median duration's DEVIATION from the profile's expected
        # duration (|ratio - 1|), so "best" is the value that makes cycles land
        # closest to their typical length - not the smallest raw ratio (which
        # would reward a severe *underrun*, e.g. 0.5x, as if it were ideal).
        ratios = sorted(
            float(r["overrun_ratio"]) for r in detected if r.get("overrun_ratio")
        )
        if not ratios:
            return None
        mid = len(ratios) // 2
        median = ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2.0
        return abs(median - 1.0)
    return None


def _coerce_param(base_config: CycleDetectorConfig, param: str, value: float) -> Any:
    """Coerce a sweep value to the override map's expected type."""
    mapping = _OVERRIDE_FIELD_MAP.get(param)
    if mapping is None:
        return value
    _field, coerce = mapping
    try:
        return coerce(value)
    except (TypeError, ValueError):
        return value


def run_playground_sweep(
    store: Any,
    cycle_ids: list[str] | None,
    base_config: CycleDetectorConfig,
    param: str,
    values: list[float],
    objective: str,
    options: dict[str, Any] | None,
    price: float | None,
    concurrency: int,
    param_y: str | None = None,
    values_y: list[float] | None = None,
) -> dict[str, Any]:
    """Sweep one param (1D curve) or two params (2D heatmap) and score each point
    by ``objective`` computed from the per-cycle rows. Executor-safe; never raises.
    """
    options = options or {}
    if objective not in _SWEEP_OBJECTIVES:
        objective = "match_accuracy"
    try:
        concurrency = max(1, min(MAX_BATCH_CYCLES, int(concurrency)))
    except (TypeError, ValueError):
        concurrency = MAX_BATCH_CYCLES
    try:
        past = list(store.get_past_cycles() or [])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground sweep: get_past_cycles failed: %s", exc)
        return {"error": "no cycles"}
    by_id = {c.get("id"): c for c in past if isinstance(c, dict)}
    if cycle_ids:
        selected = [by_id[c] for c in cycle_ids if c in by_id]
    else:
        selected = past[-DEFAULT_RECENT_CYCLES:]
    selected = selected[:concurrency]

    def _metric_for(override: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
        rows = _run_rows(store, selected, base_config, override, options, price)
        return objective_metric(rows, objective), _rows_summary(rows)

    current_x = _sim_config_summary(base_config).get(
        _OVERRIDE_FIELD_MAP.get(param, (param,))[0]
    )

    if param_y and values_y:
        grid: list[list[float | None]] = []
        best: dict[str, Any] | None = None
        for vy in values_y:
            row_metrics: list[float | None] = []
            for vx in values:
                override = {
                    param: _coerce_param(base_config, param, vx),
                    param_y: _coerce_param(base_config, param_y, vy),
                }
                metric, _ = _metric_for(override)
                row_metrics.append(round(metric, 4) if metric is not None else None)
                if metric is not None and (
                    best is None or _sweep_is_better(metric, best["metric"], objective)
                ):
                    best = {"x": vx, "y": vy, "metric": round(metric, 4)}
            grid.append(row_metrics)
        current_y = _sim_config_summary(base_config).get(
            _OVERRIDE_FIELD_MAP.get(param_y, (param_y,))[0]
        )
        return {
            "param_x": param, "param_y": param_y, "objective": objective,
            "x_values": values, "y_values": values_y, "grid": grid, "best": best,
            "current": {"x": current_x, "y": current_y},
        }

    points: list[dict[str, Any]] = []
    best_1d: dict[str, Any] | None = None
    for vx in values:
        override = {param: _coerce_param(base_config, param, vx)}
        metric, summary = _metric_for(override)
        points.append(
            {"value": vx, "metric": round(metric, 4) if metric is not None else None,
             "summary": summary}
        )
        if metric is not None and (
            best_1d is None or _sweep_is_better(metric, best_1d["metric"], objective)
        ):
            best_1d = {"value": vx, "metric": round(metric, 4)}
    return {
        "param": param, "objective": objective, "points": points,
        "current_value": current_x,
        "best_value": best_1d["value"] if best_1d else None,
        "best_metric": best_1d["metric"] if best_1d else None,
    }


# ─── DTW debug ────────────────────────────────────────────────────────────────


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _profile_trace(store: Any, profile_name: str) -> tuple[list[float], float] | None:
    """Return (power_values, duration_s) for a profile's average envelope.

    Prefers the cached envelope ``avg`` curve; falls back to the profile's
    sample cycle trace. Returns None when nothing usable exists.
    """
    try:
        env = store.get_envelope(profile_name)
    except Exception:  # pylint: disable=broad-exception-caught
        env = None
    if env and env.get("avg"):
        avg = env["avg"]
        powers: list[float] = []
        times: list[float] = []
        for pt in avg:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                times.append(float(pt[0]))
                powers.append(float(pt[1]))
            else:
                powers.append(float(pt))
        if powers:
            duration = float(env.get("target_duration") or 0.0)
            if not duration and len(times) > 1:
                duration = times[-1] - times[0]
            return powers, duration

    # Fallback: the profile's sample cycle.
    try:
        data = getattr(store, "_data", {}) or {}
        profile = (data.get("profiles", {}) or {}).get(profile_name)
        if not isinstance(profile, dict):
            return None
        # past + imported reference cycles: an import-only profile's sample lives in
        # reference_cycles, so resolve against both (mirrors _build_match_snapshots).
        pool = (data.get("past_cycles", []) or []) + (data.get("reference_cycles", []) or [])
        sample = next(
            (
                c
                for c in pool
                if isinstance(c, dict) and c.get("id") == profile.get("sample_cycle_id")
            ),
            None,
        )
        if not sample:
            return None
        pts = decompress_power_data(sample)
        if not pts:
            return None
        duration = float(
            profile.get("avg_duration")
            or sample.get("duration")
            or (pts[-1][0] if pts else 0.0)
        )
        return [p for _, p in pts], duration
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def dtw_debug_payload(
    store: Any, cycle_id: str, profile_name: str | None
) -> dict[str, Any]:
    """Score breakdown + resampled traces + DTW warp path for a cycle vs profile.

    Returns ``{"error": <code>}`` when the cycle or profile is unavailable.
    Executor-safe; never raises.
    """
    try:
        cycle = next(
            (c for c in store.get_past_cycles() if c.get("id") == cycle_id), None
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"error": "store_error", "detail": str(exc)}
    if cycle is None:
        return {"error": "cycle_not_found"}

    target_profile = profile_name or _cycle_label(cycle)
    if not target_profile:
        return {"error": "no_profile"}

    cycle_pts = decompress_power_data(cycle)
    if not cycle_pts or len(cycle_pts) < 2:
        return {"error": "cycle_no_data", "profile_name": target_profile}

    prof = _profile_trace(store, target_profile)
    if prof is None:
        return {"error": "profile_not_found", "profile_name": target_profile}
    prof_powers, prof_duration = prof
    if len(prof_powers) < 2:
        return {"error": "profile_no_data", "profile_name": target_profile}

    try:
        return _compute_dtw_debug(
            store, cycle, cycle_pts, target_profile, prof_powers, prof_duration
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground dtw_debug failed for %s: %s", cycle_id, exc)
        return {
            "error": "compute_error",
            "detail": str(exc),
            "profile_name": target_profile,
        }


def _compute_dtw_debug(
    store: Any,
    cycle: dict[str, Any],
    cycle_pts: list[tuple[float, float]],
    profile_name: str,
    prof_powers: list[float],
    prof_duration: float,
) -> dict[str, Any]:
    cfg = _matching_config(store)
    corr_weight = float(cfg.get("corr_weight", MATCH_CORR_WEIGHT))
    dur_weight = float(cfg.get("duration_weight", MATCH_DURATION_WEIGHT))
    en_weight = float(cfg.get("energy_weight", MATCH_ENERGY_WEIGHT))
    dur_scale = float(cfg.get("duration_scale", MATCH_DURATION_SCALE))
    en_scale = float(cfg.get("energy_scale", MATCH_ENERGY_SCALE))
    band = float(cfg.get("dtw_bandwidth", 0.2))
    blend = float(cfg.get("dtw_blend", MATCH_DTW_BLEND))
    l1_scale = float(cfg.get("dtw_l1_scale", MATCH_DTW_DIST_SCALE))
    ddtw_scale = float(cfg.get("dtw_ddtw_scale", MATCH_DDTW_DIST_SCALE))
    ensemble_w = float(cfg.get("dtw_ensemble_w", MATCH_DTW_ENSEMBLE_W))

    cycle_powers = [p for _, p in cycle_pts]
    cycle_duration = float(cycle_pts[-1][0] - cycle_pts[0][0])
    current_peak = float(max(cycle_powers)) if cycle_powers else 0.0

    # --- Stage 2: core similarity on the raw traces (matcher-faithful) ---
    score, metrics, _offset = analysis.find_best_alignment(
        cycle_powers, prof_powers, corr_weight=corr_weight
    )
    corr = float(metrics.get("corr", 0.0))
    mae = float(metrics.get("mae", 0.0))
    scaled_mae = mae * MATCH_MAE_REF_PEAK / max(current_peak, MATCH_MAE_PEAK_FLOOR)
    mae_score = MATCH_MAE_SCALE / (MATCH_MAE_SCALE + scaled_mae)
    stage2_score = float(score)

    # --- DTW components on a common resampled grid ---
    curr_arr = np.asarray(cycle_powers, dtype=float)
    sample_arr = np.asarray(prof_powers, dtype=float)
    l1_score = analysis._dtw_component_score(
        curr_arr, sample_arr, current_peak, band, False, l1_scale
    )
    ddtw_score = analysis._dtw_component_score(
        curr_arr, sample_arr, current_peak, band, True, ddtw_scale
    )
    ensemble_score = ensemble_w * l1_score + (1.0 - ensemble_w) * ddtw_score
    blended_score = blend * stage2_score + (1.0 - blend) * ensemble_score

    # --- Stage 4: duration + energy agreement over the DTW-blended score ---
    cur_mean = float(np.mean(curr_arr)) if curr_arr.size else 0.0
    prof_mean = float(np.mean(sample_arr)) if sample_arr.size else 0.0
    dur_ag = analysis._agreement(cycle_duration, prof_duration, dur_scale)
    en_ag = analysis._agreement(cur_mean, prof_mean, en_scale)
    shape_w = 1.0 - dur_weight - en_weight
    final_score = shape_w * blended_score + dur_weight * dur_ag + en_weight * en_ag

    # --- Resampled traces on one shared grid (progress fraction 0..1) ---
    n = MATCH_DTW_RESAMPLE_N
    a = analysis._resample_to(curr_arr, n)
    b = analysis._resample_to(sample_arr, n)
    grid = np.linspace(0.0, 1.0, n)
    cycle_trace = [[round(float(g), 4), round(float(p), 1)] for g, p in zip(grid, a)]
    profile_trace = [[round(float(g), 4), round(float(p), 1)] for g, p in zip(grid, b)]

    # --- DTW warping path on the same resampled arrays ---
    try:
        raw_path = analysis.compute_dtw_path(a, b, band_width_ratio=band)
        warp_path = [[int(i), int(j)] for i, j in raw_path]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground warp path failed: %s", exc)
        warp_path = []

    return {
        "cycle_id": cycle.get("id"),
        "profile_name": profile_name,
        "grid_n": n,
        "cycle_duration_s": round(cycle_duration, 1),
        "profile_duration_s": round(float(prof_duration), 1),
        "cycle_trace": cycle_trace,
        "profile_trace": profile_trace,
        "stage2": {
            "correlation": round(corr, 4),
            "mae_score": round(float(mae_score), 4),
            "score": round(stage2_score, 4),
        },
        "dtw": {
            "l1_score": round(float(l1_score), 4),
            "ddtw_score": round(float(ddtw_score), 4),
            "ensemble_score": round(float(ensemble_score), 4),
            "blend_weight": round(blend, 4),
            "blended_score": round(float(blended_score), 4),
        },
        "stage4": {
            "duration_agreement": round(float(dur_ag), 4),
            "energy_agreement": round(float(en_ag), 4),
            "final_score": round(float(final_score), 4),
        },
        "warp_path": warp_path,
    }
