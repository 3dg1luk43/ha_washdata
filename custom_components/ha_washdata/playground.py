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

Main entry points:

- :func:`simulate_cycle_detail` - faithful single-cycle replay with per-step
  progress/remaining-time/phase/energy series and typed event log.
- :func:`run_playground_history` - per-cycle rows + optional before/after diff.
- :func:`run_playground_sweep` - objective 1D grid sweep.

All top-level entry points are defensive: they never raise, returning an
``{"error": ...}`` marker instead so the WS handlers can relay it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np

from homeassistant.util import dt as dt_util

from . import match_rules
from . import notification_rules as notif_rules
from . import progress as progress_mod
from .options_utils import option_float, option_int
from .signal_processing import (
    compact_price_timeline,
    cycle_cost,
    energy_gap_threshold_s,
    integrate_wh,
)
from .const import (
    CONF_ANTI_WRINKLE_ENABLED,
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_ANTI_WRINKLE_IDLE_TIMEOUT,
    CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE,
    CONF_SMART_TERMINATION_DURATION_RATIO,
    CONF_ANTI_CREASE_FINALIZE_RATIO,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_ANTI_WRINKLE_MAX_DURATION,
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_END_ENERGY_THRESHOLD,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_INTERRUPTED_MIN_SECONDS,
    CONF_LEARNING_CONFIDENCE,
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
    CONF_PROFILE_UNMATCH_THRESHOLD,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_WATCHDOG_INTERVAL,
    CYCLE_OVERRUN_ANOMALY_RATIO,
    CYCLE_UNDERRUN_ANOMALY_RATIO,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_MATCH_PERSISTENCE,
    DEFAULT_MAX_DEFERRAL_SECONDS,
    DISHWASHER_END_SPIKE_WAIT_SECONDS,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_NOTIFY_MILESTONES,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_IDLE,
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_UNKNOWN,
    TerminationReason,
    resolve_watchdog_interval_default,
)
from .cycle_detector import (
    MatchContext,
    CycleDetector,
    CycleDetectorConfig,
    effective_anticrease_finalize_ratio,
    effective_curve_preroll_seconds,
    standby_near_stop_ceiling,
    terminal_high_for_guards,
)
from .profile_store import (
    MatchResult,
    ProfileStore,
    decompress_power_data,
)
from .detector_config import (
    terminal_drop_baseline_for,
    terminal_drop_enabled,
    terminal_drop_fires,
    terminal_drop_may_fire,
)
from .time_utils import power_data_to_offsets

_LOGGER = logging.getLogger(__name__)

# The most recent N cycles to replay when the caller does not name any.
DEFAULT_RECENT_CYCLES = 20
# Hard upper bound on cycles simulated in one batch call (defence in depth on
# top of the caller-supplied ``concurrency`` cap).
MAX_BATCH_CYCLES = 50
# Cap the per-cycle event log so a pathological trace cannot bloat the payload.
MAX_EVENTS_PER_CYCLE = 300
# Cap the per-cycle timeline series so a very long cycle (4h dishwasher = ~2800 pts)
# does not bloat the task result. Points are thinned at finalize time — evenly-spaced,
# so the shape is preserved rather than truncated.
MAX_SERIES_PER_CYCLE = 600

def _coerce_bool(value: Any) -> bool:
    """Strict bool coercion for override values.

    Plain ``bool()`` would read the string ``"false"`` as True, so a toggle sent
    as a string could switch a mode *on* when the user asked for it off. Unknown
    values raise, which ``build_sim_config`` turns into "ignore this override".
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # Only the two values that actually mean a toggle. Anything else (2, -1,
        # NaN, inf) is a malformed override, not an intent to switch a mode on.
        if value == 0:
            return False
        if value == 1:
            return True
        raise ValueError(f"not a boolean: {value!r}")
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
    raise ValueError(f"not a boolean: {value!r}")


# Override keys the Playground honours, mapped to CycleDetectorConfig fields.
# Only detection-relevant knobs matter; everything else in settings_override is
# ignored safely.
_OVERRIDE_FIELD_MAP: dict[str, tuple[str, Callable[[Any], Any]]] = {
    CONF_MIN_POWER: ("min_power", float),
    CONF_ANTI_WRINKLE_ENABLED: ("anti_wrinkle_enabled", _coerce_bool),
    CONF_ANTI_WRINKLE_MAX_POWER: ("anti_wrinkle_max_power", float),
    CONF_ANTI_WRINKLE_MAX_DURATION: ("anti_wrinkle_max_duration", float),
    CONF_ANTI_WRINKLE_EXIT_POWER: ("anti_wrinkle_exit_power", float),
    CONF_ANTI_WRINKLE_IDLE_TIMEOUT: ("anti_wrinkle_idle_timeout", float),
    CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE: ("dishwasher_end_spike_quiet_release", float),
    CONF_SMART_TERMINATION_DURATION_RATIO: ("smart_termination_duration_ratio", float),
    CONF_ANTI_CREASE_FINALIZE_RATIO: ("anti_crease_finalize_ratio", float),
    CONF_CURVE_PREROLL_SECONDS: ("curve_preroll_seconds", float),
    CONF_OFF_DELAY: ("off_delay", int),
    CONF_MIN_OFF_GAP: ("min_off_gap", int),
    CONF_COMPLETION_MIN_SECONDS: ("completion_min_seconds", int),
    CONF_START_THRESHOLD_W: ("start_threshold_w", float),
    CONF_STOP_THRESHOLD_W: ("stop_threshold_w", float),
    CONF_START_DURATION_THRESHOLD: ("start_duration_threshold", float),
    CONF_INTERRUPTED_MIN_SECONDS: ("interrupted_min_seconds", int),
    # Suggested settings the Playground could not what-if (audit SUGGEST-19).
    CONF_END_ENERGY_THRESHOLD: ("end_energy_threshold", float),
    CONF_PROFILE_MATCH_THRESHOLD: ("match_confidence_threshold", float),
    CONF_PROFILE_MATCH_INTERVAL: ("match_interval", int),
}

# Matching options the Playground honours, mapped to the ``match_config`` key
# they drive: the two Stage-1 duration ratios, both real user settings. The Stage
# 2-4 scoring weights and DTW knobs were sandbox-only overrides until 0.5.8; they
# could not persist, matching is saturated on them, and tuning them on 20
# in-sample cycles only overfit. Anything else in ``settings_override`` is ignored.
_MATCH_OVERRIDE_KEYS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO: ("min_duration_ratio", float),
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO: ("max_duration_ratio", float),
}


# Canonical default for every matching override key, keyed by the OPTION key the
# Playground uses. ``ws_get_constants`` ships it as ``pg_match_defaults`` and
# ``effective_settings`` falls back to it for any key the live matcher config does
# not carry.
MATCH_DEFAULTS_BY_OPTION: dict[str, Any] = {
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO: DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO: DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
}

# Every option key the Playground control panel may carry (detection + matching).
# Anything else submitted by a client is dropped.
SETTING_KEYS: frozenset[str] = frozenset(_OVERRIDE_FIELD_MAP) | frozenset(_MATCH_OVERRIDE_KEYS)

# The keys a user may publish from the Playground back into the live config. Every
# Playground key is now a real config-entry option, so this is all of them; the
# panel still gates its publish buttons on this list (shipped by
# ``get_playground_settings``).
PUBLISHABLE_SETTING_KEYS: frozenset[str] = SETTING_KEYS


def effective_settings(
    base_config: CycleDetectorConfig, match_config: dict[str, Any] | None
) -> dict[str, Any]:
    """Option-keyed view of the values a simulation runs with when NO override is
    staged - i.e. the device's live, fully-resolved settings.

    The exact inverse of ``build_sim_config`` / ``apply_match_overrides``: it reads
    back the same fields those two write, so the Playground control panel shows the
    values the integration actually uses (device-type defaults included) instead of
    a static schema default that may have drifted. Never raises.
    """
    out: dict[str, Any] = {}
    for opt_key, (field, coerce) in _OVERRIDE_FIELD_MAP.items():
        value = getattr(base_config, field, None)
        if value is None:
            continue
        try:
            out[opt_key] = coerce(value)
        except (TypeError, ValueError, OverflowError):  # pragma: no cover - defensive
            continue
    cfg = match_config or {}
    for opt_key, (cfg_key, coerce) in _MATCH_OVERRIDE_KEYS.items():
        value = cfg.get(cfg_key, MATCH_DEFAULTS_BY_OPTION.get(opt_key))
        if value is None:
            continue
        try:
            out[opt_key] = coerce(value)
        except (TypeError, ValueError, OverflowError):  # pragma: no cover - defensive
            continue
    return out


def sanitize_setting_values(values: Any) -> dict[str, Any]:
    """Filter a client-supplied settings map down to storable Playground values.

    Keeps only keys in :data:`SETTING_KEYS`, coerced with the same coercers the
    simulation uses, so an override can never carry an unknown key or a value that
    would be silently ignored at replay time. Never raises.
    """
    if not isinstance(values, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        mapping = _OVERRIDE_FIELD_MAP.get(key) or _MATCH_OVERRIDE_KEYS.get(key)
        if mapping is None:
            continue
        _target, coerce = mapping
        try:
            coerced = coerce(value)
        # OverflowError: the override payload is JSON-decoded, so an oversized
        # integer literal arrives as an unbounded int and float() on one raises
        # rather than returning inf. Dropping the value is this function's
        # documented behaviour; escaping would fail the whole save.
        except (TypeError, ValueError, OverflowError):
            continue
        if isinstance(coerced, float) and not math.isfinite(coerced):
            continue
        # Every Playground setting is a physical quantity - watts, seconds, a
        # count, or a ratio - so a negative value is structurally meaningless and
        # would make the replayed detector behave in ways the live one never can
        # (e.g. an off_delay that expires before it starts). Rejected rather than
        # clamped: silently rewriting a value the user typed would make the sim
        # disagree with the control panel showing it back.
        if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
            if coerced < 0:
                continue
        out[key] = coerced
    return out


def apply_match_overrides(
    match_config: dict[str, Any], settings_override: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a copy of ``match_config`` with the recognised matching options from
    ``settings_override`` overlaid onto the matcher-config keys they drive.
    Unknown/None/malformed values are ignored, so a detection-only override leaves
    matching byte-identical to the live config."""
    settings_override = sanitize_setting_values(settings_override)
    if not isinstance(settings_override, dict) or not settings_override:
        return match_config
    out = dict(match_config)
    for opt_key, (cfg_key, coerce) in _MATCH_OVERRIDE_KEYS.items():
        val = settings_override.get(opt_key)
        if val is None:
            continue
        try:
            out[cfg_key] = coerce(val)
        except (TypeError, ValueError, OverflowError):
            pass
    return out


def build_sim_config(
    base: CycleDetectorConfig, settings_override: dict[str, Any] | None
) -> CycleDetectorConfig:
    """Return a copy of ``base`` with the recognised override keys applied.

    Unknown keys and un-coercible values are ignored so a malformed override can
    never break a simulation. ``base`` is left untouched.
    """
    settings_override = sanitize_setting_values(settings_override)
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
        except (TypeError, ValueError, OverflowError):
            continue
    if not changes:
        return base
    try:
        return replace(base, **changes)
    except (TypeError, ValueError, OverflowError):  # pragma: no cover - defensive
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


# The grid a replay's snapshots are first built on: `resample_adaptive`'s floor,
# which is what most cycles resolve to (it is max(5 s, the trace's median step)).
_PLAYGROUND_START_DT = 5.0

# Bound on the keepalives emulated inside one silent stretch (8 h at a 30 s
# watchdog): the detector's own 8 h cap ends any cycle long before this.
_MAX_KEEPALIVES_PER_GAP = 960


def _build_match_snapshots(
    store: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    """Prepare the matcher snapshots + config once from the store.

    Mirrors the store's async matching path: one snapshot per profile using
    its sample cycle's decompressed trace, plus the store's live matching config
    (with any on-device tuned weight overrides merged in).

    Also resolves Stage-5 groups via :meth:`ProfileStore._grouped_snapshots`, the
    same call the live matcher makes. Note what that returns since #400: the
    **individual member** snapshots, unchanged, plus ``group_members`` and
    ``member_snaps``. It no longer averages a family into one ``__group__*``
    aggregate - that averaged curve belonged to no member and cost the family its
    program-level match, so members are scored individually and each cohesive
    family is collapsed to its best member afterwards by
    :func:`collapse_group_candidates`. This docstring described the old aggregate
    behaviour long after the code stopped doing it.

    Returns ``(snapshots, match_config, group_members, member_snaps)``. When no
    cohesive groups exist ``group_members`` and ``member_snaps`` are both empty
    dicts and behaviour is identical to before.
    """
    # in_progress: the sim replays a cycle step by step, so every match it runs is
    # a live one - the same footing as manager._async_do_perform_matching (#400).
    config = _matching_config(store, in_progress=True)
    # The live builder (item 387a), on the grid a replayed cycle starts on;
    # `_SimStore` re-grids per match, as live does. A store without it (the sim
    # run with no store at all) has nothing to match against. The hand-rolled
    # builder that used to serve a MagicMock store here had no production caller.
    if not callable(getattr(type(store), "build_match_snapshots", None)):
        return [], config, {}, {}
    try:
        snapshots = store.build_match_snapshots(_PLAYGROUND_START_DT)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground: live snapshot builder failed: %s", exc)
        snapshots = []
    # Stage-5: map cohesive profile groups to their members; every member is
    # scored on its own curve and collapse_group_candidates forms the family.
    try:
        grouped_snaps, group_members, member_snaps = store._grouped_snapshots(  # pylint: disable=protected-access
            snapshots
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground: _grouped_snapshots failed: %s", exc)
        grouped_snaps, group_members, member_snaps = snapshots, {}, {}
    return grouped_snaps, config, group_members, member_snaps


def _matching_config(store: Any, in_progress: bool = False) -> dict[str, Any]:
    """Live matcher config from the store (the live matcher runs no overrides)."""
    return {
        "min_duration_ratio": float(getattr(store, "_min_duration_ratio", DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO)),
        "max_duration_ratio": float(getattr(store, "_max_duration_ratio", DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO)),
        "dtw_bandwidth": float(getattr(store, "dtw_bandwidth", 0.2)),
        # Mirror the live Stage-4 energy discriminator so the sim is byte-identical.
        "energy_mode": str(getattr(store, "energy_mode", "mean")),
        "in_progress": bool(in_progress),
    }


class _InlineExecutor:
    """``hass`` for :class:`_SimStore`: an executor job runs inline, in this thread.

    The replay is already in an executor thread (or a harness with no loop), and
    the store's real ``hass`` belongs to the event loop, so it must not be used.
    """

    async def async_add_executor_job(self, fn: Callable[..., Any], *args: Any) -> Any:
        return fn(*args)


def _run_inline(coro: Any) -> Any:
    """Drive a store coroutine whose only awaits are :class:`_InlineExecutor` jobs.

    Those complete without suspending, so the coroutine finishes on its first step.
    If a future change gives it a real suspension point this raises instead of
    returning a wrong answer, and the replay reports the failure.
    """
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    coro.close()
    raise RuntimeError("store coroutine suspended; the Playground cannot await it")


#: Distinct query grids a replay keeps candidate templates for.
_MAX_SNAPSHOT_GRIDS = 64
#: How long the synthetic tail may wait out a held verified pause: the watchdog's
#: own limit for silence under one (``manager._watchdog_check_stuck_cycle``).
_TAIL_VERIFIED_PAUSE_CAP_S = DEFAULT_MAX_DEFERRAL_SECONDS + 1800.0


def _tail_span_s(config: Any) -> float:
    """How long the synthetic 0 W tail runs so a natural end can fire.

    Past the longest ordinary end gate (off delay / min off gap), plus margin. A
    dishwasher also waits up to ``DISHWASHER_END_SPIKE_WAIT_SECONDS`` for a late
    pump-out, so its tail covers that: sized on the two settings alone, a what-if
    that lowered them (as Apply all does) force-stopped a cycle the detector would
    have ended normally (one Eco cycle needed 1530 s against a 600 s tail; found by
    devtools/suggestion_loop_eval.py, register item 455).
    """
    gate = max(float(config.off_delay or 0.0), float(config.min_off_gap or 0.0))
    if getattr(config, "device_type", None) == "dishwasher":
        gate = max(gate, DISHWASHER_END_SPIKE_WAIT_SECONDS)
    return gate * 1.5 + 300.0


class _SimStore:
    """The device's store as the live matcher sees it, callable from a replay.

    The replay runs the REAL ``ProfileStore.async_match_profile`` and
    ``ProfileStore.async_verify_alignment`` with this object as ``self`` (item 387a,
    audit PLAYGROUND-01): every attribute not defined here is the store's own, so
    the candidate pool, Stage 1-5 (incl. the in-progress member preference), the
    12-point floor, ambiguity, the prefix flags, the member confidence, the phase
    lookup and the alignment thresholds are the live code rather than a copy that
    can drift. Three things differ, all deliberate:

    * ``hass`` runs executor jobs inline (:class:`_InlineExecutor`);
    * the matcher config is the sim's - the live config plus any what-if override
      of the Stage-1 ratios - returned from ``_matching_overrides``, which
      ``async_match_profile`` merges last;
    * candidate templates are cached per query grid, seeded with the prebuilt 5 s
      set a batch shares. A store without the live builder (a MagicMock in tests)
      gets the prebuilt set whatever the grid, as before.
    """

    def __init__(
        self,
        store: Any,
        match_config: dict[str, Any],
        prebuilt: tuple[Any, Any, Any, Any],
    ) -> None:
        self._store = store
        self.hass = _InlineExecutor()
        self._config = {k: v for k, v in (match_config or {}).items() if k != "in_progress"}
        self.dtw_bandwidth = self._config.get(
            "dtw_bandwidth", getattr(store, "dtw_bandwidth", 0.2)
        )
        snaps, _cfg, group_members, member_snaps = prebuilt
        self._prebuilt = (snaps, (snaps, group_members or {}, member_snaps or {}))
        self._live_builder = callable(getattr(type(store), "build_match_snapshots", None))
        self._grids: dict[float, tuple[Any, Any]] = (
            {float(_PLAYGROUND_START_DT): self._prebuilt} if self._live_builder else {}
        )
        self._pending: tuple[Any, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def _matching_overrides(self) -> dict[str, Any]:
        return dict(self._config)

    def build_match_snapshots(self, used_dt: float) -> list[dict[str, Any]]:
        if not self._live_builder:
            self._pending = self._prebuilt
            return self._prebuilt[0]
        key = float(used_dt)
        hit = self._grids.get(key)
        if hit is None:
            snaps = self._store.build_match_snapshots(used_dt)
            hit = (snaps, self._store._grouped_snapshots(snaps))  # noqa: SLF001
            if len(self._grids) >= _MAX_SNAPSHOT_GRIDS:
                self._grids.clear()
            self._grids[key] = hit
        self._pending = hit
        return hit[0]

    def _grouped_snapshots(self, snapshots: list[dict[str, Any]]) -> Any:
        pending = self._pending
        if pending is not None and pending[0] is snapshots:
            return pending[1]
        return self._store._grouped_snapshots(snapshots)  # noqa: SLF001

    def match(
        self,
        readings: Any,
        duration: float,
        in_progress: bool = False,
        stop_threshold_w: float | None = None,
    ) -> MatchResult:
        """``ProfileStore.async_match_profile`` on this view, run to completion."""
        return _run_inline(
            ProfileStore.async_match_profile(
                self, readings, duration,  # type: ignore[arg-type]
                in_progress=in_progress, stop_threshold_w=stop_threshold_w,
            )
        )

    def verify_alignment(self, profile_name: str, trace: Any) -> tuple[bool, float, float]:
        """``ProfileStore.async_verify_alignment`` on this view, run to completion."""
        return _run_inline(
            ProfileStore.async_verify_alignment(self, profile_name, trace)  # type: ignore[arg-type]
        )


def _readings_from_cycle(
    cycle: dict[str, Any],
) -> tuple[list[tuple[datetime, float]], list[tuple[float, float]], datetime]:
    """Reconstruct (datetime, power) readings + (offset, power) points + base time."""
    points = decompress_power_data(cycle)
    base = _cycle_base_time(cycle)
    readings = [(base + timedelta(seconds=float(o)), float(p)) for o, p in points]
    return readings, points, base


# ─── Single-cycle faithful simulation (Simulate mode) ───────────────────────────


# States in which no progress estimate is shown (mirrors _update_remaining_only).
_DEAD_STATES = (STATE_OFF, STATE_UNKNOWN, STATE_IDLE)
_SIM_SERIES_THROTTLE_S = 30.0  # cap estimator calls; 5s matched cadence made this a no-op

# Terminal-drop baselines per stored-cycle list (audit ML-08). A History/Optimize
# batch replays many cycles against one store and the baseline decompresses every
# completed trace, so it is built once per list. The list is held, so its id
# cannot be recycled while cached; an append changes the length in the key.
_TERMINAL_DROP_BASELINES: dict[tuple[int, int, float], tuple[Any, Any]] = {}
_MAX_TERMINAL_DROP_BASELINES = 16


def _sim_terminal_drop_baseline(
    store: Any, stop_threshold_w: float
) -> tuple[float | None, tuple[float, float] | None]:
    """The live terminal-drop baseline over ``store``'s stored cycles. Never raises.

    Built from every stored cycle, the replayed one included, like the rest of the
    Playground (in-sample). That can only make a completed cycle LESS likely to
    fire: its own first quiet span is in the baseline.
    """
    try:
        cycles = store.get_past_cycles()
        if not isinstance(cycles, list):
            return None, None
        key = (id(cycles), len(cycles), float(stop_threshold_w))
        hit = _TERMINAL_DROP_BASELINES.get(key)
        if hit is not None and hit[0] is cycles:
            return hit[1]
        baseline = terminal_drop_baseline_for(list(cycles), stop_threshold_w)
        if len(_TERMINAL_DROP_BASELINES) >= _MAX_TERMINAL_DROP_BASELINES:
            _TERMINAL_DROP_BASELINES.clear()
        _TERMINAL_DROP_BASELINES[key] = (cycles, baseline)
        return baseline
    except Exception:  # pylint: disable=broad-exception-caught
        return None, None


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

    Drives the REAL :class:`CycleDetector`, the real matcher and the manager's
    :mod:`match_rules` over the cycle's own trace, and calls the SAME
    :mod:`progress` and :mod:`notification_rules` functions the live integration
    uses (the estimator every ``_SIM_SERIES_THROTTLE_S`` = 30 s of replay time, with
    the live per-second EMA scaling) - so the timeline is what would happen live. No
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


def build_cycle_detail_sim_by_id(
    store: Any,
    cycle_id: str,
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    options: dict[str, Any] | None,
    price: float | None = None,
) -> "_DetailSim | dict[str, Any]":
    """Look up a stored cycle by id and build a resumable :class:`_DetailSim`.

    Used by the chunked background-task driver in ``ws_api`` so the heavy replay
    can be stepped across many small executor jobs (issue #311). Returns a
    ``{"error": ...}`` marker (not a sim) when the id is unknown or setup fails,
    so the caller can surface it. The store lookup + build run together so the WS
    handler can offload the whole thing to an executor thread. Never raises."""
    options = options or {}
    try:
        cycle = next(
            (c for c in store.get_past_cycles() if c.get("id") == cycle_id), None
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground detail lookup failed for %s: %s", cycle_id, exc)
        return {"error": str(exc), "cycle_id": cycle_id}
    if cycle is None:
        return {"error": "not_found", "cycle_id": cycle_id}
    try:
        return _DetailSim(cycle, base_config, settings_override, store, options, price)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground detail sim build failed for %s: %s", cycle_id, exc)
        return {"error": str(exc), "cycle_id": cycle_id}


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
    """One-shot faithful replay: build the resumable sim and run it to completion.

    The chunked (background-task) driver in ``ws_api`` builds the same
    :class:`_DetailSim` and calls ``step``/``run_tail``/``finalize`` across many
    small executor jobs so the event loop breathes on very long cycles (issue
    #311). Because both paths drive the identical object in the identical order,
    the timeline is byte-for-byte the same (tests/test_playground_chunked_parity.py).
    """
    sim = _DetailSim(
        cycle, base_config, settings_override, store, options, price,
        compute_series, prebuilt,
    )
    if not sim.ready:
        return sim.empty_payload()
    sim.step(0, sim.n_readings)
    sim.run_tail()
    return sim.finalize()


class _DetailSim:
    """Resumable single-cycle Playground "Simulate" replay.

    Drives the REAL :class:`CycleDetector` + the real matcher
    (``ProfileStore.async_match_profile``, via :class:`_SimStore`) over the
    cycle's own trace, applies each match with the manager's own rules
    (:mod:`match_rules`: switching, verified pause, confident-mismatch revoke)
    and calls the SAME :mod:`progress` and :mod:`notification_rules` functions
    the live integration uses. No detection/matching/progress/notification math
    is implemented here; this only orchestrates the shared code. Read-only:
    nothing is persisted and no notifications are sent.

    The replay is split into :meth:`step` (a slice of the real readings),
    :meth:`run_tail` (the synthetic quiet tail + flush) and :meth:`finalize`
    (outcome + alerts) so a long cycle can be replayed chunk-by-chunk across
    executor jobs without holding the GIL for the whole run.
    """

    def __init__(
        self,
        cycle: dict[str, Any],
        base_config: CycleDetectorConfig,
        settings_override: dict[str, Any] | None,
        store: Any,
        options: dict[str, Any],
        price: float | None,
        compute_series: bool = True,
        prebuilt: tuple[Any, Any, Any, Any] | None = None,
    ) -> None:
        self.cycle = cycle
        self.store = store
        self.options = options or {}
        self.price = price
        # Dynamic tariff timeline frozen onto the stored cycle (#426). Replaying it
        # is what keeps the sim's projected cost identical to the one the live
        # estimator produced for that cycle; without it a dynamically-priced cycle
        # would replay at a single flat price and silently diverge.
        self.price_points = compact_price_timeline(
            [
                (entry[0], entry[1])
                for entry in (cycle.get("price_timeline") or [])
                if isinstance(entry, (list, tuple)) and len(entry) >= 2
            ]
        )
        self.compute_series = compute_series
        self.config = build_sim_config(base_config, settings_override)
        self.device_type = _device_type_of(self.config)
        self.label = _cycle_label(cycle)
        self.readings, _points, self.base = _readings_from_cycle(cycle)
        self.stored_duration = _safe_float(cycle.get("duration"))
        # When the appliance actually ran, for the Optimize end-timing objectives
        # (audit PLAYGROUND-08): first/last reading above the LIVE stop threshold,
        # never the override's, so a swept threshold cannot move its own yardstick.
        # Same definition as devtools/end_gate_eval.py (`_active_span`).
        _truth_stop = float(getattr(base_config, "stop_threshold_w", 0.0) or 0.0)
        _active = [
            (ts - self.base).total_seconds() for ts, p in self.readings if p > _truth_stop
        ]
        self.active_end_s: float | None = _active[-1] if _active else None
        self.active_span_s: float | None = (
            _active[-1] - _active[0] if len(_active) >= 2 else None
        )

        self.outcome: dict[str, Any] = {
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
            # Would the manager auto-label the (primary) finished cycle, as which
            # profile, and why not: match_rules.cycle_end_label_verdict reasons
            # (ok / no_winner / below_floor / ambiguous / margin / unknown_profile),
            # plus no_cycle / no_store / error from the replay itself.
            "would_label": False,
            "label_profile": None,
            "label_reason": "no_cycle",
            "end_offset_s": None,
        }
        if prebuilt is None:
            prebuilt = _build_match_snapshots(store)
        # Overlay any matcher-knob overrides. Because history/sweep run through this
        # same class, a swept matching value flows in via settings_override too;
        # applying to a copy keeps the shared prebuilt match_config untouched.
        self.match_config = apply_match_overrides(prebuilt[1], settings_override)
        # The live matcher and alignment check, run inline (see _SimStore).
        self.view: _SimStore | None = (
            _SimStore(store, self.match_config, prebuilt) if store is not None else None
        )

        self.ready = len(self.readings) >= 5
        # Per-sim end-expectation cache, threaded through the shared progress helpers
        # exactly like the manager threads self._ml_end_expectation_cache.
        self.endexp_cache: list[Any] = [None]

        self.events: list[dict[str, Any]] = []
        self.series: list[dict[str, Any]] = []
        self.captured: list[dict[str, Any]] = []
        self.cursor = {"t": 0.0}
        self.last_match: dict[str, Any] = {
            "name": None, "conf": 0.0, "ambiguous": False, "expected": 0.0,
        }
        self.last_logged = {"kind": None, "name": None}
        # The manager's switching state and the options its rules read, resolved
        # exactly as `manager._load_runtime_options` resolves them (audit
        # PLAYGROUND-03): the reported program is the one live would display.
        self.switch = match_rules.SwitchState()
        self.match_persistence = option_int(
            self.options.get(CONF_MATCH_PERSISTENCE, DEFAULT_MATCH_PERSISTENCE),
            DEFAULT_MATCH_PERSISTENCE,
            minimum=1,
        )
        self.unmatch_threshold = self.options.get(
            CONF_PROFILE_UNMATCH_THRESHOLD, DEFAULT_PROFILE_UNMATCH_THRESHOLD
        )
        self.learning_floor = float(
            option_float(
                self.options.get(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE),
                DEFAULT_LEARNING_CONFIDENCE,
            )
            or 0.0
        )
        # The last live MatchResult (manager._last_match_result) and, per finished
        # cycle, what the manager held when it ended (its cycle-end inputs).
        self._last_result: Any = None
        self.cycle_ends: list[dict[str, Any]] = []
        self.smoothed: dict[str, Any] = {"v": 0.0, "program": None}
        self.flags = {"detected": False, "pre_complete": False, "start": False}

        # --- notification config (decisions reuse notification_rules) ---
        self.start_configured = bool(
            self.options.get(CONF_NOTIFY_START_SERVICES) or self.options.get(CONF_NOTIFY_ACTIONS)
        )
        self.finish_configured = bool(
            self.options.get(CONF_NOTIFY_FINISH_SERVICES) or self.options.get(CONF_NOTIFY_ACTIONS)
        )
        self.before_end = float(
            self.options.get(CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES)
            or 0.0
        )
        self.quiet_bounds = notif_rules.quiet_hours_bounds(self.options)

        self.last_sample_t = -1e9
        self._aborted = False
        # Watchdog cadence (item 390). Live injects a keepalive on this cadence
        # while a cycle sits below the stop threshold and the plug is silent; the
        # sim does the same inside a silent stretch of the trace (see step()).
        try:
            _wd = float(
                {**self.options, **(settings_override or {})}.get(
                    CONF_WATCHDOG_INTERVAL,
                    resolve_watchdog_interval_default(self.device_type),
                )
            )
        except (TypeError, ValueError, OverflowError):
            _wd = float(resolve_watchdog_interval_default(self.device_type))
        self.watchdog_s = _wd if math.isfinite(_wd) and _wd > 0 else 0.0
        self._last_real: tuple[datetime, float] | None = None

        if self.ready:
            self.detector = CycleDetector(
                self.config, self._on_state_change, self._on_cycle_end,
                profile_matcher=self._matcher, device_name="playground-detail",
                terminal_drop_provider=self._terminal_drop_provider(
                    {**self.options, **(settings_override or {})}
                ),
            )

    def _terminal_drop_provider(
        self, options: dict[str, Any]
    ) -> Callable[[list[tuple[float, float]], float], bool] | None:
        """``manager._terminal_drop_provider`` for this replay; None where live
        would not run it (``detector_config.terminal_drop_enabled``, audit ML-08)."""
        if not terminal_drop_enabled(self.device_type, options):
            return None
        stop = float(getattr(self.config, "stop_threshold_w", 0.0) or 0.0)
        baseline = _sim_terminal_drop_baseline(self.store, stop)
        device_type = self.device_type

        def _provider(points: list[tuple[float, float]], _expected: float) -> bool:
            # The live gate on the sim's own detector (built after this closure).
            if not terminal_drop_may_fire(device_type, options, self.detector):
                return False
            return terminal_drop_fires(points, baseline, stop)

        return _provider

    @property
    def n_readings(self) -> int:
        return len(self.readings)

    def empty_payload(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle.get("id"),
            "label": self.label,
            "duration_s": self.stored_duration,
            "start_time": self.cycle.get("start_time"),
            "active_end_s": _safe_float(self.active_end_s),
            "active_span_s": _safe_float(self.active_span_s),
            "config_summary": _sim_config_summary(self.config),
            "series": [],
            "events": [],
            "alerts": [],
            "outcome": self.outcome,
        }

    def _end_exp_fn(self, name: str, dur: float) -> Any:
        exp, self.endexp_cache[0] = progress_mod.profile_end_expectation(
            self.store, name, dur, self.endexp_cache[0]
        )
        return exp

    def _emit(self, etype: str, detail: str, severity: str = "info") -> None:
        if len(self.events) < MAX_EVENTS_PER_CYCLE:
            self.events.append(
                {"t": round(self.cursor["t"], 1), "type": etype, "detail": detail,
                 "severity": severity}
            )

    def _held(self, offset: float) -> bool:
        # Quiet hours are local clock hours and replay timestamps are UTC, so
        # `.hour` on the raw stamp held the wrong hours (audit PROGRESS-13).
        return notif_rules.in_quiet_hours(
            self.quiet_bounds, dt_util.as_local(self.base + timedelta(seconds=offset))
        )

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        self._emit("state", f"{old_state}->{new_state}")
        # A new cycle: the switching state starts fresh, exactly when
        # `manager._on_state_change` resets it (RUNNING from OFF / STARTING /
        # UNKNOWN). PAUSED/ENDING -> RUNNING is a resume and keeps it.
        if new_state == STATE_RUNNING and old_state in (STATE_OFF, STATE_STARTING, STATE_UNKNOWN):
            self.switch.start_cycle()
            self.last_match.update(name=None, conf=0.0, expected=0.0, ambiguous=False)
            self.last_logged.update(kind=None, name=None)
        if (
            not self.flags["detected"]
            and new_state == STATE_RUNNING
            and old_state in (STATE_OFF, STATE_UNKNOWN, STATE_STARTING, STATE_IDLE)
        ):
            self.flags["detected"] = True
            self._emit("detected", "cycle detected (running)")
            if self.start_configured and not self.flags["start"]:
                self.flags["start"] = True
                # Start notifications are never delayed by quiet hours (live
                # contract), so the sim always emits them immediately.
                self._emit("notify_start", "start notification")

    def _on_cycle_end(self, cycle_data: dict[str, Any]) -> None:
        self.captured.append(cycle_data)
        reason = cycle_data.get("termination_reason")
        self._emit("finished", f"reason={reason} status={cycle_data.get('status')}", "info")
        # What `manager._async_process_cycle_end` freezes before its first await:
        # the displayed program and the last live match (its label inputs).
        st = self.switch
        program = st.current_program if match_rules.program_is_committed(st.current_program) else None
        self.cycle_ends.append({
            "program": program,
            "confidence": st.last_confidence if program else None,
            "expected": st.matched_duration if program else None,
            "ambiguous": bool(self.last_match.get("ambiguous")),
            "live_result": self._last_result,
            # Replay offset at which WashData said "done" (end lag, PLAYGROUND-08).
            "t": self.cursor["t"],
        })
        # ...and the terminal reset at its tail: the next cycle starts from "off"
        # with no live result, as live does once the cycle has been processed.
        st.current_program = "off"
        st.matched_duration = None
        self._last_result = None

    def _has_real_profiles(self) -> bool:
        """The gate `manager._async_perform_combined_matching` checks first."""
        if self.store is None:
            return False
        try:
            return bool(self.store.has_real_profiles)
        except Exception:  # pylint: disable=broad-exception-caught
            return False

    def _verify_alignment(
        self, profile_name: str, det_readings: list[tuple[datetime, float]]
    ) -> tuple[bool, float]:
        """The live alignment check; a failure counts as unconfirmed, as live."""
        try:
            assert self.view is not None
            formatted = power_data_to_offsets(det_readings)  # type: ignore[arg-type]
            is_confirmed, mapped_time, _ = self.view.verify_alignment(profile_name, formatted)
            return bool(is_confirmed), mapped_time
        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Playground alignment check failed", exc_info=True)
            return False, 0.0

    def _get_profile(self, name: str) -> Any:
        return self.store.get_profile(name)

    def _matcher(self, det_readings: list[tuple[datetime, float]]) -> Any:
        """One live match tick: ``manager._async_do_perform_matching`` in sequence.

        The real matcher (``ProfileStore.async_match_profile`` via
        :class:`_SimStore`), then the manager's own rules from
        :mod:`match_rules`: switching, the envelope verified pause and its
        releases, the consistency override (which is also how a confident mismatch
        drops the program). Like the manager it pushes the
        verified pause and the commit flag to the detector, and hands it the
        tick's name - which after a divergence revert is "detecting...", as live -
        not the displayed program. Returns the match context the detector applies,
        or None where live would not have matched at all (no real profiles).

        The alignment check runs synchronously here. Live awaits it, and the
        matcher, while readings keep arriving; the replay applies the tick at the
        reading that triggered it.
        """
        if not det_readings or not self._has_real_profiles():
            return None
        assert self.view is not None
        det = self.detector
        current_duration = (det_readings[-1][0] - det_readings[0][0]).total_seconds()
        stop_w = float(det.config.stop_threshold_w)
        result = self.view.match(
            det_readings, current_duration, in_progress=True, stop_threshold_w=stop_w
        )
        self._last_result = result

        st = self.switch
        prev_program = st.current_program
        tick = match_rules.begin_tick(st, result, self.match_persistence, current_duration)
        match_rules.decide_switch(
            st, tick, result, self.match_persistence, self.unmatch_threshold
        )
        match_rules.record_scores(st, result.candidates)

        current_matched = det.matched_profile
        prev_verified = getattr(det, "_verified_pause", False)
        current_power = det_readings[-1][1]
        # The detector mirrors the manager's user pause (`set_user_paused`); a replay
        # never pauses, but a harness can (`end_gate_eval --user-pause`, item 514).
        user_paused = getattr(det, "_user_paused", False) is True
        alignment: tuple[bool, float] | None = None
        if match_rules.needs_alignment_check(current_matched, current_power, stop_w, user_paused):
            alignment = self._verify_alignment(current_matched, det_readings)
        pause = match_rules.decide_alignment_pause(
            verified_pause=prev_verified,
            current_matched=current_matched,
            alignment=alignment,
            envelope_span=self.view.envelope_time_span,
        )
        pause = match_rules.decide_pause_release(
            verified_pause=pause.verified_pause,
            current_matched=current_matched,
            current_power=current_power,
            stop_threshold_w=getattr(det.config, "stop_threshold_w", 5.0),
            user_paused=user_paused,
            expected_duration=det.expected_duration_seconds,
            current_duration=current_duration,
            time_below=getattr(
                det, "_time_below_threshold_gapfree", getattr(det, "_time_below_threshold", 0.0)
            ),
            program=st.current_program,
        )
        verified = pause.verified_pause
        match_rules.consistency_override(st, tick, result, verified, self._get_profile)
        # The manager's ENDING pause hold (register item 469b).
        verified = match_rules.hold_in_ending(
            ending=det.state == STATE_ENDING,
            is_ambiguous=bool(result.is_ambiguous),
            current_matched=current_matched,
            prev_verified=prev_verified,
            verified_pause=verified,
            user_paused=user_paused,
        ).verified_pause
        det.set_verified_pause(verified)
        det.set_match_committed(match_rules.program_is_committed(st.current_program))
        self._report_tick(prev_program, bool(prev_verified), bool(verified), result)
        return self._match_context(tick, tick.phase_name, result)

    def _report_tick(
        self, prev_program: Any, prev_verified: bool, verified: bool, result: Any
    ) -> None:
        """Events + the reported match after a tick: what live would display."""
        st = self.switch
        program = st.current_program if match_rules.program_is_committed(st.current_program) else None
        before = prev_program if match_rules.program_is_committed(prev_program) else None
        if program != before:
            if program and not before:
                self._emit("match_commit", f"{program} (conf={float(st.last_confidence):.2f})")
            elif program:
                self._emit(
                    "match_changed",
                    f"{before} -> {program} (conf={float(st.last_confidence):.2f})",
                )
            else:
                self._emit("match_reverted", f"{before} -> detecting")
            self.last_logged.update(kind="matched" if program else "reverted", name=program)
        if result.is_confident_mismatch:
            if self.last_logged["kind"] != "unmatched":
                self._emit("unmatched", "no candidate")
                self.last_logged["kind"] = "unmatched"
        elif result.is_ambiguous and not program and result.best_profile:
            # Ambiguous before any commit: stay 'detecting', surface it once per name.
            raw = result.best_profile
            if self.last_logged["kind"] != "ambiguous" or self.last_logged["name"] != raw:
                cands = result.candidates or []
                runner = cands[1].get("name") if len(cands) > 1 else None
                self._emit(
                    "match_ambiguous",
                    f"{raw} vs {runner} (margin={float(result.ambiguity_margin):.3f})",
                    "warn",
                )
                self.last_logged.update(kind="ambiguous", name=raw)
        if verified != prev_verified:
            self._emit("verified_pause", "engaged" if verified else "released")
        self.last_match.update(
            name=program,
            conf=float(st.last_confidence or 0.0) if program else 0.0,
            expected=float(st.matched_duration or 0.0) if program else 0.0,
            # The raw tick's flag, as `manager._last_match_ambiguous` holds it.
            ambiguous=bool(result.is_ambiguous),
        )

    def _match_context(self, tick: Any, phase_name: str | None, result: Any) -> MatchContext:
        """The named context the manager builds for ``update_match`` (DETECT-15)."""
        store = self.store
        det = self.detector
        # The manager's own rule: a divergence revert revokes the detector's match.
        name, revoke = match_rules.detector_match(tick, result)

        def _ask(fn: Callable[[], Any]) -> Any:
            # Guarded like the rest of the sim: a partial test double without one
            # of these methods must not turn every match into "unmatched".
            try:
                return fn()
            except Exception:  # pylint: disable=broad-exception-caught
                return None

        stop_w = float(det.config.stop_threshold_w)
        return MatchContext(
            profile_name=name,
            confidence=tick.confidence,
            expected_duration=tick.matched_duration,
            phase_name=phase_name,
            is_confident_mismatch=revoke,
            is_ambiguous=result.is_ambiguous,
            is_prefix_ambiguous_full_shape=result.is_prefix_ambiguous_full_shape,
            tail_power=_ask(lambda: store.profile_tail_power(name)) if name else None,
            # One implementation with the manager's `_terminal_high_for_guards`.
            terminal_high=_ask(lambda: terminal_high_for_guards(
                store, det.config, getattr(det, "_cycle_max_power", 0.0), name
            )),
            terminal_quiet_s=(
                _ask(lambda: store.profile_terminal_quiet_seconds(name)) if name else None
            ),
            longest_candidate_s=float(getattr(result, "longest_candidate_duration_s", 0.0) or 0.0),
            trusted_min_s=(
                _ask(lambda: store.profile_trusted_min_duration(name)) if name else None
            ),
            pause_catalogue=(
                _ask(lambda: store.profile_pause_catalogue(name, stop_w)) if name else None
            ),
            # #452, as the manager supplies it (the stall display and its
            # standby-band hold).
            stall_catalogue=(
                (lambda: _ask(lambda: store.profile_pause_catalogue(
                    name, standby_near_stop_ceiling(stop_w)
                ))) if name else None
            ),
        )

    def _label_decision(self, cycle_data: dict[str, Any], live_result: Any) -> tuple[str | None, str]:
        """Would the manager auto-label this finished cycle? ``(profile, reason)``.

        The manager's cycle-end path: ONE complete-cycle match on the stored trace
        (``match_rules.final_match_input`` + ``async_match_profile``, not in
        progress), falling back to the last live match when the trace is too short,
        then ``match_rules.cycle_end_label_verdict`` at the learning floor.
        """
        if self.view is None:
            return None, "no_store"
        try:
            final = None
            final_input = match_rules.final_match_input(cycle_data)
            if final_input is not None:
                final = self.view.match(final_input[0], final_input[1])
            match_result = final if final is not None else live_result
            return match_rules.cycle_end_label_verdict(
                match_result, self.learning_floor, self.store.get_profiles()
            )
        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Playground label decision failed", exc_info=True)
            return None, "error"

    def _price_at(self, offset_s: float) -> float | None:
        """The tariff in force at a replay offset, or the flat price with no timeline.

        Live charges the *remaining* energy at whatever ``_resolve_energy_price()``
        returns at that instant, so a replay has to move through its own stored
        timeline instead of pinning the whole cycle to one price. Otherwise a
        dynamically-priced cycle diverges from the projection the live estimator
        actually produced, which is the one thing this sim exists to reproduce.
        """
        price = self.price
        for point_offset, point_price in self.price_points:
            if point_offset > offset_s:
                break
            price = point_price
        return price

    def _cost_so_far(
        self, trace: list[tuple[datetime, float]]
    ) -> tuple[float, float] | None:
        """``(cost, charged_wh)`` incurred up to this point of the replay, or None.

        Mirrors ``manager._live_cost_so_far``: the integrated trace charged at the
        prices the cycle actually ran through. None (no stored timeline) puts the
        projection back on the flat-price formula, which is the right answer for a
        cycle recorded before dynamic pricing existed.
        """
        if not self.price_points or len(trace) < 2:
            return None
        try:
            base_ts = self.base.timestamp()
            timestamps = np.asarray([t.timestamp() - base_ts for t, _ in trace], dtype=float)
            power = np.asarray([p for _, p in trace], dtype=float)
            max_gap_s = energy_gap_threshold_s(timestamps)
            result = cycle_cost(timestamps, power, self.price_points, max_gap_s=max_gap_s)
            if result is None:
                return None
            return result[0], float(integrate_wh(timestamps, power, max_gap_s=max_gap_s))
        except Exception:  # noqa: BLE001 - the sim never raises
            return None

    def _sample(self, ts: datetime) -> None:
        if not self.compute_series:
            return  # batch/sweep rows only need the outcome, not the per-step series
        offset = (ts - self.base).total_seconds()
        if offset - self.last_sample_t < _SIM_SERIES_THROTTLE_S:
            return
        prev_sample_t = self.last_sample_t
        self.last_sample_t = offset
        state = self.detector.state
        power = 0.0
        trace = self.detector.get_power_trace()
        if trace:
            power = float(trace[-1][1])
        energy_wh = float(getattr(self.detector, "_energy_since_idle_wh", 0.0) or 0.0)
        pt: dict[str, Any] = {
            "t": round(offset, 1),
            "power": round(power, 1),
            "energy_wh": round(energy_wh, 2),
            "state": state,
            "progress": None,
            "remaining_s": None,
            "phase": None,
            "confidence": round(self.last_match["conf"], 3) if self.last_match["name"] else None,
            "matched_profile": self.last_match["name"],
        }
        if getattr(self.detector, "stalled", False) is True:
            pt["stalled"] = True  # #452: shown as paused / Stalled live
        matched_dur = float(self.last_match["expected"] or 0.0)
        program = self.last_match["name"]
        if state not in _DEAD_STATES and program and matched_dur > 0:
            # Item 514, as live: progress reads programme time (a halt is not
            # progress); the energy and the cost below read the whole trace.
            prog_trace = self.detector.progress_trace(ts)
            prog_t = self.detector.progress_elapsed_s(offset, ts)
            phase_result = None
            if len(prog_trace) >= 10 and program != "detecting...":
                phase_result = progress_mod.estimate_phase_progress(
                    self.store, prog_trace, prog_t, program,
                    quiet_threshold_w=float(
                        getattr(self.detector.config, "stop_threshold_w", 0.0) or 0.0
                    ),
                )
            result = progress_mod.compute_progress(
                self.device_type, matched_dur, prog_t,
                progress_mod.ema_seed(self.smoothed["v"], self.smoothed["program"], program),
                phase_result,
                # Same time-scaled smoothing as live: the sim steps the estimator
                # at its own throttle, so without this the replay would smooth
                # over 30 s steps as if they were the manager's 5 s ones.
                dt_seconds=(
                    offset - prev_sample_t if prev_sample_t >= 0.0 else None
                ),
            )
            if result is not None:
                self.smoothed["v"] = result.smoothed
                self.smoothed["program"] = program
                pt["progress"] = round(result.progress, 1)
                pt["remaining_s"] = round(result.remaining, 0)
                pt["phase"] = progress_mod.current_phase(
                    self.store, state, program, result.progress, matched_dur
                )
                sim_cost = self._cost_so_far(trace)
                wh, cost = progress_mod.projected_energy(
                    self.store, self.options, matched_dur, trace, program, result.progress,
                    energy_wh, self._price_at(offset), self._end_exp_fn,
                    cost_so_far=sim_cost[0] if sim_cost else None,
                    cost_so_far_wh=sim_cost[1] if sim_cost else None,
                )
                pt["projected_energy_wh"] = round(wh, 1) if wh is not None else None
                pt["projected_cost"] = round(cost, 4) if cost is not None else None
                # One-time pre-completion marker (reuses the production predicate).
                if not self.flags["pre_complete"] and notif_rules.should_notify_pre_completion(
                    self.before_end, self.flags["pre_complete"], result.remaining,
                    result.progress, self.last_match["ambiguous"],
                ):
                    self.flags["pre_complete"] = True
                    held = self._held(offset)
                    self._emit(
                        "notify_held" if held else "notify_pre_complete",
                        "pre-completion notification"
                        + (" (held: quiet hours)" if held else ""),
                    )
        self.series.append(pt)

    def step(self, i0: int, i1: int) -> None:
        """Replay readings[i0:i1] through the detector (a chunk of the cycle)."""
        if self._aborted or not self.ready:
            return
        try:
            for ts, power in self.readings[i0:i1]:
                self._watchdog_keepalives(ts)
                self.cursor["t"] = (ts - self.base).total_seconds()
                self.detector.process_reading(power, ts)
                self._note_stall()
                self._last_real = (ts, power)
                self._sample(ts)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._aborted = True
            _LOGGER.debug(
                "Playground detail replay failed for %s: %s", self.cycle.get("id"), exc
            )

    def _note_stall(self) -> None:
        """A ``stalled`` event when the detector starts showing a stall (#452).

        The same display rule live shows (``CycleDetector.stalled``) and the
        moment the manager fires ``ha_washdata_cycle_stalled``: real readings only.
        """
        stalled = getattr(self.detector, "stalled", False) is True
        if stalled == getattr(self, "_stall_shown", False):
            return
        self._stall_shown = stalled
        if stalled:
            info = self.detector.stall_info() or {}
            self._emit("stalled", f"stalled at {info.get('plateau_w')} W (display only)", "warn")

    def _watchdog_keepalives(self, until: datetime) -> None:
        """Inject the keepalives live would have injected before ``until`` (item 390).

        Mirrors the low-power branch of ``manager._watchdog_check_stuck_cycle``:
        while the detector waits below the stop threshold and the plug has been
        silent for more than ``watchdog_interval``, each tick feeds the sensor's
        last value as an observed synthetic reading. Without it a silent stretch
        reached the detector as one interval at the next real reading - after the
        power had already come back - so the end gates were never evaluated
        inside it, and a soak that live 0.5.7 ends on (item 290 credits the
        silence) replayed as one cycle. Ticks sit half an interval into each
        period, the mean phase of a free-running timer. Traces recorded on 0.5.7+
        already hold these readings, so for them this is a no-op. The watchdog's
        staleness force-end and ghost/zombie branches are not emulated.
        """
        last = self._last_real
        step = self.watchdog_s
        if last is None or step <= 0:
            return
        prev_ts, prev_w = last
        k = 1
        while k <= _MAX_KEEPALIVES_PER_GAP:
            ts = prev_ts + timedelta(seconds=step * (k + 0.5))
            if ts >= until or not self.detector.is_waiting_low_power():
                return
            self.cursor["t"] = (ts - self.base).total_seconds()
            self.detector.process_reading(prev_w, ts, synthetic=True, observed=True)
            self._sample(ts)
            k += 1

    def run_tail(self) -> None:
        """Synthetic quiet tail so a natural end can fire.

        While the detector holds an envelope-verified pause every ENDING finalize is
        deferred, and live keeps waiting - its watchdog extends its own silence
        limit to ``DEFAULT_MAX_DEFERRAL_SECONDS`` + 30 min for the same reason. So
        the tail runs on for as long as one is held (bounded by that same limit),
        then the usual span, instead of force-ending a cycle the release rules
        (95% of the envelope, #375) would have ended a few minutes later.
        """
        if self._aborted or not self.ready:
            return
        try:
            last_ts = self.readings[-1][0]
            tail_span = _tail_span_s(self.config)
            step = 30.0
            n_steps = min(int(tail_span / step) + 1, 400)
            max_steps = n_steps + int(_TAIL_VERIFIED_PAUSE_CAP_S / step)
            budget = n_steps
            i = 0
            while i < budget:
                i += 1
                ts = last_ts + timedelta(seconds=step * i)
                self.cursor["t"] = (ts - self.base).total_seconds()
                self.detector.process_reading(0.0, ts)
                self._sample(ts)
                if self.detector.state in (STATE_OFF, STATE_FINISHED) and self.captured:
                    break
                if getattr(self.detector, "_verified_pause", False):
                    budget = min(max_steps, max(budget, i + n_steps))
            if not self.captured and self.detector.state != STATE_OFF:
                flush_ts = last_ts + timedelta(seconds=step * (i + 2))
                self.cursor["t"] = (flush_ts - self.base).total_seconds()
                self.detector.force_end(flush_ts)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug(
                "Playground detail replay failed for %s: %s", self.cycle.get("id"), exc
            )

    def finalize(self) -> dict[str, Any]:
        outcome = self.outcome
        last_match = dict(self.last_match)
        # --- outcome ---
        if self.captured:
            idx = max(
                range(len(self.captured)),
                key=lambda i: float(self.captured[i].get("duration") or 0.0),
            )
            primary = self.captured[idx]
            outcome["detected"] = True
            outcome["detected_count"] = len(self.captured)
            outcome["termination_reason"] = primary.get("termination_reason")
            outcome["status"] = primary.get("status")
            outcome["final_duration_s"] = _safe_float(primary.get("duration"))
            # The program the manager displayed when THAT cycle ended, not whatever
            # a later sub-cycle left behind.
            end = self.cycle_ends[idx] if idx < len(self.cycle_ends) else None
            if end is not None:
                outcome["end_offset_s"] = _safe_float(end.get("t"))
                last_match.update(
                    name=end["program"],
                    conf=float(end["confidence"] or 0.0),
                    expected=float(end["expected"] or 0.0),
                    ambiguous=end["ambiguous"],
                )
                label, reason = self._label_decision(primary, end["live_result"])
                outcome["would_label"] = label is not None
                outcome["label_profile"] = label
                outcome["label_reason"] = reason
        outcome["matched_profile"] = last_match["name"]
        outcome["confidence"] = (
            round(float(last_match["conf"]), 3) if last_match["name"] else None
        )
        outcome["expected_s"] = (
            round(float(last_match["expected"] or 0.0), 1) or None
        )
        if outcome["detected"] and last_match["name"] and self.label:
            outcome["match_correct"] = last_match["name"].strip() == self.label.strip()
        # Projected energy/cost: the last LIVE estimate (the post-finish tail resets
        # the detector's accumulated energy, so series[-1] would read None).
        for pt in reversed(self.series):
            if pt.get("projected_energy_wh") is not None:
                outcome["projected_energy_wh"] = pt.get("projected_energy_wh")
                outcome["projected_cost"] = pt.get("projected_cost")
                break

        # --- finish + milestone markers (reuse production predicates) ---
        if self.captured and self.finish_configured:
            # No finished push for an interrupted cycle (audit MANAGER-10); the
            # milestone below does not depend on it, as live.
            if notif_rules.cycle_end_is_finish(outcome["status"]):
                held = self._held(self.cursor["t"])
                self._emit(
                    "notify_held" if held else "notify_finish",
                    "finish notification" + (" (held: quiet hours)" if held else ""),
                )
            try:
                prev_life = int(self.store.get_lifetime_cycle_count())
            except Exception:  # pylint: disable=broad-exception-caught
                prev_life = 0
            crossed = notif_rules.milestone_crossed(
                prev_life, prev_life + 1,
                self.options.get(CONF_NOTIFY_MILESTONES, DEFAULT_NOTIFY_MILESTONES),
            )
            if crossed is not None:
                # Milestone notifications are held during quiet hours (live contract).
                m_held = self._held(self.cursor["t"])
                self._emit(
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
            off_min = max(1, round(float(getattr(self.config, "off_delay", 0) or 0) / 60))
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

        series = self.series
        if len(series) > MAX_SERIES_PER_CYCLE:
            # Thin evenly so the shape is preserved (first + last always kept).
            # Span (len-1)/(N-1) so the final index lands on the true last point
            # (a plain len/N tops out below it and drops the terminal sample).
            step = (len(series) - 1) / (MAX_SERIES_PER_CYCLE - 1)
            series = [series[round(i * step)] for i in range(MAX_SERIES_PER_CYCLE)]

        return {
            "cycle_id": self.cycle.get("id"),
            "label": self.label,
            "duration_s": self.stored_duration,
            "start_time": self.cycle.get("start_time"),
            "active_end_s": _safe_float(self.active_end_s),
            "active_span_s": _safe_float(self.active_span_s),
            "config_summary": _sim_config_summary(self.config),
            "series": series,
            "events": self.events,
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
        "anti_wrinkle_enabled": getattr(config, "anti_wrinkle_enabled", None),
        "anti_wrinkle_max_power": getattr(config, "anti_wrinkle_max_power", None),
        "anti_wrinkle_max_duration": getattr(config, "anti_wrinkle_max_duration", None),
        "anti_wrinkle_exit_power": getattr(config, "anti_wrinkle_exit_power", None),
        "anti_wrinkle_idle_timeout": getattr(config, "anti_wrinkle_idle_timeout", None),
        "dishwasher_end_spike_quiet_release": getattr(config, "dishwasher_end_spike_quiet_release", None),
        "smart_termination_duration_ratio": getattr(config, "smart_termination_duration_ratio", None),
        # Both are clamped by the detector wherever it reads them, so report the
        # effective figure: a summary carrying an out-of-range override would
        # describe a sim that did not run.
        "anti_crease_finalize_ratio": effective_anticrease_finalize_ratio(
            getattr(config, "anti_crease_finalize_ratio", None)
        ),
        "curve_preroll_seconds": effective_curve_preroll_seconds(
            getattr(config, "curve_preroll_seconds", None)
        ),
    }


# ─── Test-on-history rows + before/after diff ───────────────────────────────────


# A detected end more than this before the appliance's last activity is an early
# end (devtools/end_gate_eval.py's `early_1min`): the cycle was cut short.
_EARLY_END_S = 60.0
# A run whose longest piece covers less than this share of its active span did not
# survive as one cycle (end_gate_eval's split rule).
_SPLIT_SPAN_FRAC = 0.9


def _end_timing(detail: dict[str, Any]) -> tuple[float | None, bool, bool]:
    """``(end_lag_s, early_end, split)`` of one replay (audit PLAYGROUND-08).

    The lag is when WashData said "done" minus when the appliance last drew power,
    the number ``devtools/end_gate_eval.py`` measures; the stored duration is
    trimmed back to the last activity (item 297) and cannot show it. A run that
    never finished counts as split: it did not survive as one cycle.
    """
    o = detail.get("outcome", {}) or {}
    if not o.get("detected"):
        return None, False, True
    end_t = o.get("end_offset_s")
    active_end = detail.get("active_end_s")
    lag = (
        round(float(end_t) - float(active_end), 1)
        if end_t is not None and active_end is not None
        else None
    )
    span = detail.get("active_span_s")
    final = o.get("final_duration_s")
    split = int(o.get("detected_count") or 0) > 1 or bool(
        span and final is not None and float(final) < _SPLIT_SPAN_FRAC * float(span)
    )
    return lag, lag is not None and lag < -_EARLY_END_S, split


def _detail_to_row(detail: dict[str, Any]) -> dict[str, Any]:
    """Compact per-cycle row for the Test-on-history table from a detail sim."""
    o = detail.get("outcome", {})
    end_lag, early_end, split = _end_timing(detail)
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
        "would_label": bool(o.get("would_label")),
        "label_profile": o.get("label_profile"),
        "label_reason": o.get("label_reason"),
        "alerts": [a.get("code") for a in detail.get("alerts", [])],
        "end_lag_s": end_lag,
        "early_end": early_end,
        "split": split,
        # "Last N" can reach past the panel's loaded page of cycles, so the row
        # carries its own date (PLAYGROUND-18).
        "start_time": (detail.get("start_time") if isinstance(detail.get("start_time"), str) else None),
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
        "early_end": sum(1 for r in rows if r.get("early_end")),
        "split": sum(1 for r in rows if r.get("split")),
    }


def _run_rows(
    store: Any,
    cycles: list[dict[str, Any]],
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    options: dict[str, Any],
    price: float | None,
    prebuilt: tuple[Any, Any, Any, Any] | None = None,
) -> list[dict[str, Any]]:
    # Snapshots are store-derived (independent of the cycle and the detector-level
    # settings_override), so build them ONCE and reuse across all cycles/values.
    # Callers that drive many chunks should pass prebuilt= to avoid rebuilding per chunk.
    if prebuilt is None:
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


def _select_cycles(
    store: Any, cycle_ids: list[str] | None, count: int | None = None
) -> list[dict[str, Any]]:
    """The stored cycles a History / Optimize run replays, at most
    ``MAX_BATCH_CYCLES``: ``cycle_ids`` in order (unknown ids dropped); else the
    ``count`` most recent, newest first - the panel's "Last N", which its loaded
    page of 25 cycles could not supply as ids (audit PLAYGROUND-18); else the most
    recent ``DEFAULT_RECENT_CYCLES`` in stored order."""
    past = [c for c in (store.get_past_cycles() or []) if isinstance(c, dict)]
    if cycle_ids:
        by_id = {c.get("id"): c for c in past}
        selected = [by_id[c] for c in cycle_ids if c in by_id]
    elif count:
        n = max(1, min(MAX_BATCH_CYCLES, int(count)))
        selected = list(reversed(past[-n:]))
    else:
        selected = past[-DEFAULT_RECENT_CYCLES:]
    return selected[:MAX_BATCH_CYCLES]


def run_playground_history(
    store: Any,
    cycle_ids: list[str] | None,
    base_config: CycleDetectorConfig,
    settings_override: dict[str, Any] | None,
    options: dict[str, Any] | None,
    price: float | None,
    concurrency: int,
    prebuilt: tuple[Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Per-cycle rows for the Test-on-history table, plus a before/after diff when
    ``settings_override`` is set. Executor-safe; never raises."""
    options = options or {}
    try:
        concurrency = max(1, min(MAX_BATCH_CYCLES, int(concurrency)))
    except (TypeError, ValueError, OverflowError):
        concurrency = MAX_BATCH_CYCLES
    try:
        selected = _select_cycles(store, cycle_ids)[:concurrency]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground history: get_past_cycles failed: %s", exc)
        return {"rows": [], "summary": _rows_summary([])}

    override = settings_override or None
    rows = _run_rows(store, selected, base_config, override, options, price, prebuilt)
    payload: dict[str, Any] = {"rows": rows, "summary": _rows_summary(rows)}

    if override:
        base_rows = _run_rows(store, selected, base_config, None, options, price, prebuilt)
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


# ─── Parameter sweep (1D curve) ────────────────────────────────────────────────────


_SWEEP_OBJECTIVES = (
    "match_accuracy",
    "end_timing_accuracy",
    "false_end_rate",
    "median_overrun",
    "ambiguity_rate",
    # What the end-gate settings actually move (audit PLAYGROUND-08): none of the
    # five above changed while off_delay 60 -> 1800 s moved the mean end lag
    # 19.8 -> 25.4 min, because the stored duration is trimmed to the activity.
    "end_lag",
    "early_end_rate",
    "split_rate",
)
# Objectives where a LOWER metric is better (best = minimum), so the sweep picks
# the right winner and the panel colours the heatmap consistently.
_SWEEP_LOWER_IS_BETTER = frozenset({
    "false_end_rate",
    "median_overrun",
    "ambiguity_rate",
    "end_lag",
    "early_end_rate",
    "split_rate",
})
# Smallest change worth recommending over the current value. A rate moves in
# steps of one cycle (1/N, passed by the caller), so anything less is the
# denominator moving, not a cycle getting better; the end lag is measured on a
# replay whose tail steps are 30 s; the overrun deviation is a share of the
# profile's length.
_SWEEP_MIN_GAIN = {"end_lag": 60.0, "median_overrun": 0.01}


def _sweep_is_better(candidate: float, best: float, objective: str) -> bool:
    if objective in _SWEEP_LOWER_IS_BETTER:
        return candidate < best
    return candidate > best


def _sweep_min_gain(objective: str, n_cycles: int) -> float:
    if objective in _SWEEP_MIN_GAIN:
        return _SWEEP_MIN_GAIN[objective]
    # A tiny epsilon below one cycle so 1/N itself, after rounding, still counts.
    return (1.0 / n_cycles - 1e-6) if n_cycles > 0 else 0.0


def _sweep_regresses(summary: Any, baseline: Any) -> bool:
    """Does ``summary`` lose what the current setting has? The hard guard: a value
    that ends more cycles early, splits more of them, or detects fewer can never be
    recommended, whatever its objective says (audit PLAYGROUND-08)."""
    if not isinstance(summary, dict) or not isinstance(baseline, dict):
        return False
    return (
        int(summary.get("early_end") or 0) > int(baseline.get("early_end") or 0)
        or int(summary.get("split") or 0) > int(baseline.get("split") or 0)
        or int(summary.get("detected") or 0) < int(baseline.get("detected") or 0)
    )


def sweep_baseline(
    store: Any,
    cycle_ids: list[str] | None,
    base_config: CycleDetectorConfig,
    objective: str,
    options: dict[str, Any] | None,
    price: float | None,
    prebuilt: tuple[Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """The current settings' metric and summary over the sweep's cycles: what every
    swept value has to beat. Executor-safe; never raises."""
    try:
        selected = _select_cycles(store, cycle_ids)
        rows = _run_rows(store, selected, base_config, None, options or {}, price, prebuilt)
        metric = objective_metric(rows, objective)
        return {
            "metric": round(metric, 4) if metric is not None else None,
            "summary": _rows_summary(rows),
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground sweep baseline failed: %s", exc)
        return {"metric": None, "summary": None}


def finalize_sweep_1d(
    param: str,
    objective: str,
    points: list[dict[str, Any]],
    current_value: Any,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a 1D sweep payload from per-value points collected across chunks.

    With a ``baseline`` (:func:`sweep_baseline`, the current settings on the same
    cycles) the recommendation is conservative (audit PLAYGROUND-08): a value that
    regresses early ends, splits or detection is never best; and unless the best
    value beats the current one by at least one cycle (:func:`_sweep_min_gain`) the
    answer is to keep the current value - a tie used to pick the first swept value,
    so "Apply best" offered ``off_delay=60``. Without one, ties still prefer the
    current value over the first."""
    base_summary = (baseline or {}).get("summary")
    base_metric = (baseline or {}).get("metric")
    n_cycles = int((base_summary or {}).get("cycles") or 0) or max(
        (int((p.get("summary") or {}).get("cycles") or 0) for p in points), default=0
    )

    def _is_current(value: Any) -> bool:
        try:
            return current_value is not None and abs(float(value) - float(current_value)) < 1e-6
        except (TypeError, ValueError, OverflowError):
            return False

    best: dict[str, Any] | None = None
    for p in points:
        m = p.get("metric")
        guarded = base_summary is not None and _sweep_regresses(p.get("summary"), base_summary)
        p["guarded"] = guarded
        if m is None or guarded:
            continue
        if (
            best is None
            or _sweep_is_better(m, best["metric"], objective)
            or (m == best["metric"] and _is_current(p["value"]))
        ):
            best = {"value": p["value"], "metric": m}
    keep_current = False
    if base_metric is not None:
        gain = _sweep_min_gain(objective, n_cycles)
        if best is None or not (
            (base_metric - best["metric"] if objective in _SWEEP_LOWER_IS_BETTER
             else best["metric"] - base_metric) >= gain
        ):
            best = {"value": current_value, "metric": base_metric}
            keep_current = True
    elif best is not None and _is_current(best["value"]):
        keep_current = True
    return {
        "param": param, "objective": objective, "points": points,
        "current_value": current_value,
        "current_metric": base_metric,
        "current_summary": base_summary,
        "best_value": best["value"] if best else None,
        "best_metric": best["metric"] if best else None,
        "keep_current": keep_current,
        "cycles": n_cycles,
        "lower_is_better": objective in _SWEEP_LOWER_IS_BETTER,
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
    if objective == "end_lag":
        # Median seconds from the appliance's last activity to WashData's "done".
        lags = [float(r["end_lag_s"]) for r in detected if r.get("end_lag_s") is not None]
        if not lags:
            return None
        lags.sort()
        mid = len(lags) // 2
        return lags[mid] if len(lags) % 2 else (lags[mid - 1] + lags[mid]) / 2.0
    if objective in ("early_end_rate", "split_rate"):
        # Over EVERY replayed cycle, not just the detected ones: a value that stops
        # detecting a hard cycle must not shrink its own denominator and win.
        key = "early_end" if objective == "early_end_rate" else "split"
        return sum(1 for r in rows if r.get(key)) / len(rows)
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
    except (TypeError, ValueError, OverflowError):
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
    prebuilt: tuple[Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Sweep one param and score each value by ``objective`` computed from the
    per-cycle rows. Executor-safe; never raises. (The 2D heatmap was removed:
    the panel never sent a second parameter, audit PLAYGROUND.)
    """
    options = options or {}
    if objective not in _SWEEP_OBJECTIVES:
        objective = "match_accuracy"
    try:
        concurrency = max(1, min(MAX_BATCH_CYCLES, int(concurrency)))
    except (TypeError, ValueError, OverflowError):
        concurrency = MAX_BATCH_CYCLES
    try:
        selected = _select_cycles(store, cycle_ids)[:concurrency]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground sweep: get_past_cycles failed: %s", exc)
        return {"error": "no cycles"}

    def _metric_for(override: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
        rows = _run_rows(store, selected, base_config, override, options, price, prebuilt)
        return objective_metric(rows, objective), _rows_summary(rows)

    current_x = _sim_config_summary(base_config).get(
        _OVERRIDE_FIELD_MAP.get(param, (param,))[0]
    )
    if current_x is None:
        # The summary carries only part of the config, so 8 sweepable keys (the
        # completion/interrupted/start-duration thresholds, the two ratios, ...)
        # reported no current value and the panel drew no marker for it.
        try:
            current_x = effective_settings(
                base_config, prebuilt[1] if prebuilt else None
            ).get(param)
        except Exception:  # pylint: disable=broad-exception-caught
            current_x = None

    points: list[dict[str, Any]] = []
    for vx in values:
        override = {param: _coerce_param(base_config, param, vx)}
        metric, summary = _metric_for(override)
        points.append(
            {"value": vx, "metric": round(metric, 4) if metric is not None else None,
             "summary": summary}
        )
    # One selection rule for the one-shot and the chunked task (which adds the
    # current-settings baseline the guard and the keep-current rule need).
    return finalize_sweep_1d(param, objective, points, current_x)


# ─── DTW debug ────────────────────────────────────────────────────────────────


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 2)
    except (TypeError, ValueError, OverflowError):
        return None
