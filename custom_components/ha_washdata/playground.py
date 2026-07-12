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
from .const import (
    CONF_COMPLETION_MIN_SECONDS,
    CONF_END_REPEAT_COUNT,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_RUNNING_DEAD_ZONE,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
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
    STATE_FINISHED,
    STATE_OFF,
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
}


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


def _build_match_snapshots(store: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prepare the matcher snapshots + config once from the store.

    Mirrors :meth:`ProfileStore.match_profile`: one snapshot per profile using
    its sample cycle's decompressed trace, plus the store's live matching config
    (with any on-device tuned weight overrides merged in).
    """
    snapshots: list[dict[str, Any]] = []
    try:
        data = getattr(store, "_data", {}) or {}
        profiles = data.get("profiles", {}) or {}
        past = data.get("past_cycles", []) or []
        by_id = {c.get("id"): c for c in past if isinstance(c, dict)}
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

    config = _matching_config(store)
    return snapshots, config


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
) -> dict[str, Any]:
    """Replay one stored cycle through a fresh headless detector.

    Returns ``{cycle_id, profile_name, events, outcome}``. Never raises.
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
    snapshots, match_config = _build_match_snapshots(store)

    results: list[dict[str, Any]] = []
    for cycle in to_run:
        res = _simulate_one(cycle, config, snapshots, match_config)
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
        past = data.get("past_cycles", []) or []
        sample = next(
            (
                c
                for c in past
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
