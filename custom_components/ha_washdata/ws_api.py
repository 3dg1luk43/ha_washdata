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
"""WebSocket API commands for the WashData full-screen panel."""
from __future__ import annotations

import asyncio
import collections
import copy
import functools
import json
import logging
import math
import os
import re
import time
import uuid
from datetime import timedelta
from typing import Any

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    coerce_numeric_option,
    drop_invalid_numeric_options,
    numeric_option_keys,
    CONF_NAME,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_DEVICE_TYPE,
    CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE,
    CONF_SMART_TERMINATION_DURATION_RATIO,
    CONF_ANTI_CREASE_FINALIZE_RATIO,
    ANTI_CREASE_FINALIZE_RATIO_MIN,
    ANTI_CREASE_FINALIZE_RATIO_MAX,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_DOOR_SENSOR_ENTITY,
    CONF_UNLOAD_CONFIRM_ENTITY,
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_END_ENERGY_THRESHOLD,
    CONF_ENERGY_SENSOR,
    CONF_EXTERNAL_END_TRIGGER,
    CONF_LINKED_DEVICE,
    CONF_MAINTENANCE_REMINDER_CYCLES,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MIN_WARMUP_CYCLES,
    CONF_PUMP_STUCK_DURATION,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_POWER_SENSOR,
    CONF_ENERGY_PRICE_ENTITY,
    CONF_NOTIFY_PEOPLE,
    CONF_SAMPLING_INTERVAL,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_SWITCH_ENTITY,
    CONF_WATCHDOG_INTERVAL,
    DEFAULT_DEVICE_TYPE,
    DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS,
    DEFAULT_ANTI_CREASE_FINALIZE_RATIO,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    CURVE_PREROLL_MAX_SECONDS,
    resolve_sampling_interval_default,
    resolve_watchdog_interval_default,
    resolve_start_duration_default,
    resolve_smart_termination_duration_ratio_default,
    MAINTENANCE_CUSTOM_TASK_MAX,
    MAINTENANCE_INTERVAL_CYCLES_MAX,
    MAINTENANCE_INTERVAL_DAYS_MAX,
    MAINTENANCE_TASK_NAME_MAX,
    resolve_min_off_gap_default,
    resolve_off_delay_default,
    DEVICE_TYPE_PUMP,
    MAINTENANCE_EVENT_TYPES,
    PLAYGROUND_PRESET_MAX,
    ML_HEALTH_MIN_TRACE_POINTS,
    DEVICE_TYPES,
    DOMAIN,
    ENABLE_ML_TRAINING,
    HISTORY_IMPORT_CHUNK_BYTES,
    HISTORY_IMPORT_CHUNK_SAMPLES,
    HISTORY_IMPORT_MAX_BYTES,
    HISTORY_IMPORT_MAX_ROWS,
    HISTORY_IMPORT_MAX_SEGMENTS,
    HISTORY_IMPORT_MAX_TOTAL_CYCLES,
    HISTORY_IMPORT_RECORDER_EMPTY_DAY_STOP,
    HISTORY_IMPORT_RECORDER_MAX_DAYS,
    SHOW_ML_LAB,
    STATE_COLORS,
)
from . import history_import
from . import playground
from . import task_registry
from .cycle_detector import (
    CycleDetectorConfig,
)
from .detector_config import (
    build_detector_config,
    effective_option_values,
    inverted_threshold_pair,
)
from .maintenance import editor_types, effective_reminders
from .options_utils import strip_null_options
from .setup_advisor import compute_setup_phase
from .ws_schema import WS_OPEN_RESPONSES, WS_RESPONSE_TYPES

_LOGGER = logging.getLogger(__name__)

# Populated once during async_setup_entry (stored in hass.data["ha_washdata_version"])
# so ws_get_constants never does blocking I/O on the event loop.  A module-level
# read_text() here would run on the event loop the first time ws_api is imported
# inside async_setup_entry (#328/#335).
_INTEGRATION_VERSION: str = ""

# ─── WS response contract (Group H1) ────────────────────────────────────────────
# Debug-only validation of every send_result payload against the TypedDict
# registered for its command in ws_schema.py. OFF by default so production has
# ZERO overhead: _send_result forwards straight to connection.send_result unless
# the flag is on. Enable by exporting HA_WASHDATA_WS_CONTRACT=1 before starting
# HA, or by flipping ws_api._WS_CONTRACT_CHECK = True (tests do the latter).
_WS_CONTRACT_CHECK: bool = bool(os.environ.get("HA_WASHDATA_WS_CONTRACT"))


def _validate_ws_contract(command: str, data: Any) -> list[str]:
    """Return contract-violation messages for ``data`` vs its response TypedDict.

    Pure and defensive: an unknown command or a non-dict payload for a typed
    command is reported, missing required top-level keys are reported, and
    unexpected top-level keys are reported unless the command is registered as
    open-ended in ``WS_OPEN_RESPONSES``. Returns an empty list when everything
    checks out. Never raises.
    """
    td = WS_RESPONSE_TYPES.get(command)
    if td is None:
        return [f"{command}: no response type registered"]
    if not isinstance(data, dict):
        return [f"{command}: response is {type(data).__name__}, expected dict"]
    required = set(getattr(td, "__required_keys__", ()) or ())
    optional = set(getattr(td, "__optional_keys__", ()) or ())
    keys = set(data.keys())
    problems: list[str] = []
    missing = required - keys
    if missing:
        problems.append(f"{command}: missing required keys {sorted(missing)}")
    if command not in WS_OPEN_RESPONSES:
        unexpected = keys - (required | optional)
        if unexpected:
            problems.append(f"{command}: unexpected keys {sorted(unexpected)}")
    return problems


def _send_result(
    connection: websocket_api.ActiveConnection,
    msg_id: int,
    command: str,
    data: Any,
) -> None:
    """Send a WS result, validating its shape against the contract in debug mode.

    Behaviourally identical to ``connection.send_result(msg_id, data)``; the
    contract check is a no-op (never touched) unless ``_WS_CONTRACT_CHECK`` is on,
    and even then it only logs — it never mutates ``data`` nor raises, so it can
    never change what the client receives.
    """
    if __debug__ and _WS_CONTRACT_CHECK:
        try:
            problems = _validate_ws_contract(command, data)
            if problems:
                _LOGGER.warning("WS contract mismatch: %s", "; ".join(problems))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("WS contract check failed for %s: %s", command, exc)
    connection.send_result(msg_id, data)


# Fields too large or not serialisable to send over WebSocket.
_CYCLE_STRIP_KEYS = frozenset({"power_data", "power_trace", "debug_data", "samples"})

# Settings keys that can be staged from suggestions. Mirrors the OptionsFlow's
# _suggestion_keys_to_apply so the panel and the flow agree on what is tunable.
# The keys the suggestion engine can still produce. A stored suggestion for any
# other key - one written by an older version for a setting no longer suggested
# (sampling_interval, smoothing_window, start debounce, the confidence ladder, the
# tolerances, end_repeat_count; audit SUGGEST-01/04/12) - is never shown or applied.
_SUGGESTION_KEYS: tuple[str, ...] = (
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    # Held back from 2026-10-04 (audit SUGGEST-21) until register item 469(b): a
    # shorter interval let an ambiguous tick land in the end wait and delay the end
    # (6.7 -> 23.2 min on one cycle). `match_rules.hold_in_ending` closes that.
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_MIN_OFF_GAP,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_END_ENERGY_THRESHOLD,
    CONF_COMPLETION_MIN_SECONDS,
    # Device-class-specific suggestions from reconcile_suggestions
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_PUMP_STUCK_DURATION,
)

# Suggestion keys coerced to int when applied (mirrors the OptionsFlow).
_SUGGESTION_INT_KEYS: frozenset[str] = frozenset({
    CONF_OFF_DELAY,
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_MIN_OFF_GAP,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_PUMP_STUCK_DURATION,
})


def _coerce_suggested(key: str, val: Any) -> Any:
    """Round a raw suggestion value the way the UI would apply it.

    Int keys are floored to int, everything else rounded to 4 dp - the same
    coercion ``ws_get_suggestions`` does before the equivalence test. The device
    pill's badge filter must use it too, or a value like 30.4 for an int key with
    a current 30 is hidden in the Settings list (30 == 30) but still counted on
    the pill, leaving a badge the user cannot clear. Returns the raw value
    unchanged if it is not numeric.
    """
    try:
        return int(float(val)) if key in _SUGGESTION_INT_KEYS else round(float(val), 4)
    except (TypeError, ValueError, OverflowError):
        return val


def _suggestion_equivalent(suggested: Any, current: Any) -> bool:
    """True when a suggested value is effectively the same as the current one.

    Numeric-tolerant so an int option (30) matches a float suggestion (30.0);
    a missing current value never counts as equivalent (so it still surfaces).
    Used to hide suggestions that would not change anything.
    """
    if current is None:
        return False
    try:
        return abs(float(suggested) - float(current)) < 1e-6
    except (TypeError, ValueError, OverflowError):
        return str(suggested) == str(current)


def _visible_suggestions(
    store: Any, merged: dict[str, Any], device_type: str
) -> list[tuple[str, dict[str, Any], Any, Any]]:
    """``(key, stored item, suggested, current)`` for every suggestion worth showing.

    The ONE filter behind the Settings list, the Overview card, Apply-all and
    the setup advisor, which had drifted (audit SUGGEST-11/17): a muted key could
    be re-created by the reconcile cascade and applied by Apply-all, and the setup
    card counted stale keys nothing would show. Drops keys the engine no longer
    produces, muted keys, and values equal to what the key already RUNS with - the
    effective default when it is unset, not ``None`` (SUGGEST-10).
    """
    raw = store.get_suggestions() or {}
    try:
        muted = set(store.get_locked_suggestions() or [])
    except Exception:  # pylint: disable=broad-exception-caught
        muted = set()
    effective = effective_option_values(merged, device_type)
    out: list[tuple[str, dict[str, Any], Any, Any]] = []
    for key in _SUGGESTION_KEYS:
        item = raw.get(key)
        if key in muted or not isinstance(item, dict) or item.get("value") is None:
            continue
        suggested = _coerce_suggested(key, item["value"])
        current = merged.get(key)
        if current is None:
            current = effective.get(key)
        if _suggestion_equivalent(suggested, current):
            continue
        out.append((key, item, suggested, current))
    return out


def _threshold_pair_message(start: float, stop: float) -> str:
    """The error a write that would invert the threshold pair is refused with (item 515)."""
    return (
        f"Stop Threshold ({stop:g} W) must be below Start Threshold ({start:g} W). "
        "Nothing was saved; change both together."
    )


def _without_inverted_threshold_pair(
    entry: ConfigEntry, updates: dict[str, Any], source: str
) -> dict[str, Any]:
    """An import's settings, less its threshold pair when that would invert this
    device's (item 515): the device keeps its own pair, the rest still applies.

    An import is someone else's tuning (an export, a store bundle), so a refused
    pair is logged rather than failing the import that carries profiles and cycles.
    """
    merged = {**entry.data, **entry.options}
    device_type = merged.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)
    pair = inverted_threshold_pair(merged, {**merged, **updates}, device_type)
    if pair is None:
        return updates
    kept = {
        k: v for k, v in updates.items()
        if k not in (CONF_START_THRESHOLD_W, CONF_STOP_THRESHOLD_W)
    }
    if inverted_threshold_pair(merged, {**merged, **kept}, device_type) is not None:
        kept.pop(CONF_MIN_POWER, None)  # it sets an unset threshold's default
    _LOGGER.warning(
        "%s for %s: not applying its thresholds (start %g W, stop %g W): the stop "
        "threshold must be below the start threshold; the device keeps its own",
        source, entry.title, pair[0], pair[1],
    )
    return kept


def _downsample(samples: Any, max_points: int = 240) -> list[list[float]]:
    """Reduce a [(offset_s, watts), ...] series to ~max_points, preserving extrema.

    Splits the series into contiguous buckets and keeps each bucket's MIN and MAX
    power sample (in time order), plus the global first and last. Two points per
    bucket, so ~max_points/2 buckets keeps the payload at the same budget striding
    used. Unlike striding this never drops a single-sample load peak (it is its
    bucket's max) or a lone 0 W self-shutdown sample between two non-zero readings
    (a zero is always its bucket's min) - the signals that carry the meaning.

    Power curves can hold thousands of points; the panel only needs enough to draw
    a faithful line, and WebSocket payloads should stay lean.
    """
    try:
        pairs = list(samples or [])
    except TypeError:
        return []
    n = len(pairs)
    if n == 0:
        return []

    def _pt(item: Any) -> list[float]:
        return [round(float(item[0]), 2), round(float(item[1]), 1)]

    if n <= max_points:
        return [_pt(it) for it in pairs]

    nbuckets = max(1, max_points // 2)
    keep: set[int] = {0, n - 1}
    for b in range(nbuckets):
        lo = (b * n) // nbuckets
        hi = ((b + 1) * n) // nbuckets
        if hi <= lo:
            continue
        min_i = max_i = lo
        min_v = max_v = float(pairs[lo][1])
        for i in range(lo + 1, hi):
            v = float(pairs[i][1])
            if v < min_v:
                min_v, min_i = v, i
            elif v > max_v:
                max_v, max_i = v, i
        keep.add(min_i)
        keep.add(max_i)
    return [_pt(pairs[i]) for i in sorted(keep)]


async def _recorder_power(
    hass: HomeAssistant,
    entity_id: str,
    start_dt: Any,
    *,
    end_dt: Any = None,
    keep_unavailable: bool = False,
) -> list[tuple[float, float | None]]:
    """Raw (unix_ts, watts) readings for entity_id over a window, via the recorder.

    ``end_dt`` defaults to now (the live chart overlay's use). Passing it lets a caller
    read the history in bounded windows instead of one unbounded query - a month of
    5-second data is millions of rows in a single recorder-executor job.

    A row that is not a number (``unavailable``/``unknown``) is skipped, unless
    ``keep_unavailable`` asks for it as ``(ts, None)``: the cycle-context view
    (register item 513) breaks its line there instead of holding the last reading
    across an outage. Watts are never None without it.
    """
    try:
        from homeassistant.components.recorder import (  # pylint: disable=import-outside-toplevel
            get_instance,
            history,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return []
    # tz-aware; use dt_util.now() per the datetime convention
    window_end = end_dt if end_dt is not None else dt_util.now()

    def _query() -> list[tuple[float, float | None]]:
        res = history.state_changes_during_period(
            hass, start_dt, window_end, entity_id, include_start_time_state=True
        )
        rows: list[tuple[float, float | None]] = []
        start_ts = start_dt.timestamp() if hasattr(start_dt, "timestamp") else None
        for s in res.get(entity_id, []) or []:
            try:
                ts = s.last_changed.timestamp()
                # include_start_time_state yields the state in force at the window
                # start, whose last_changed can predate it by days. Clamp it to the
                # window so a caller reading day-by-day windows does not see a huge
                # synthetic leading gap (or the same reading twice).
                if start_ts is not None and ts < start_ts:
                    ts = start_ts
                rows.append((ts, round(float(s.state), 1)))
            except (ValueError, TypeError, OverflowError):
                if keep_unavailable:
                    try:
                        gap_ts = s.last_changed.timestamp()
                    except (AttributeError, TypeError, ValueError, OverflowError):
                        continue
                    if start_ts is not None and gap_ts < start_ts:
                        gap_ts = start_ts
                    rows.append((gap_ts, None))
                continue
        return rows

    try:
        return await get_instance(hass).async_add_executor_job(_query)
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def _cycle_kwh(c: dict[str, Any]) -> float | None:
    """Cycle energy in kWh for display.

    Prefers the external meter's reading (``energy_meter_wh``, issue #316) when a
    cycle recorded one, otherwise the integrated ``energy_wh``. Cycles store energy
    in Wh; convert to kWh.
    """
    wh = c.get("energy_meter_wh")
    if wh is None:
        wh = c.get("energy_wh")
    if wh is not None:
        try:
            return round(float(wh) / 1000.0, 4)
        except (TypeError, ValueError, OverflowError):
            pass
    return c.get("energy_kwh")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_manager(hass: HomeAssistant, entry_id: str) -> Any | None:
    domain_data: dict[str, Any] = hass.data.get(DOMAIN, {})
    return domain_data.get(entry_id) if isinstance(domain_data, dict) else None


# Per-entry serialization lock for the heavy, multi-await store-mutating WS
# handlers (process_recording / reprocess_history / import_config). Holding one
# lock for the whole operation prevents two of them from interleaving and
# clobbering each other's persisted state or colliding on cycle IDs. Stored on
# hass.data (not module-global) so each lock is created inside — and bound to —
# the running event loop, which keeps it correct across test event loops.
_WS_WRITE_LOCKS_KEY = f"{DOMAIN}_ws_write_locks"
_WS_OPTIONS_LOCKS_KEY = f"{DOMAIN}_ws_options_locks"


def _entry_write_lock(hass: HomeAssistant, entry_id: str) -> asyncio.Lock:
    """Return the shared per-entry write lock, creating it on first use."""
    locks: dict[str, asyncio.Lock] = hass.data.setdefault(_WS_WRITE_LOCKS_KEY, {})
    lock = locks.get(entry_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[entry_id] = lock
    return lock


def _entry_options_lock(hass: HomeAssistant, entry_id: str) -> asyncio.Lock:
    """Per-entry lock for ``entry.options`` read-record-update sections only.

    Deliberately NOT ``_entry_write_lock``. That one is held for the whole run
    of the long detached tasks - ``_reprocess_task`` across rematching,
    suggestions, ML training, recosting and health recompute;
    ``_ml_training_task`` across a full training run; ``_rebuild_envelopes_task``
    across every profile - so putting a Settings save behind it means the save
    blocks for as long as the task takes, which on a slow host is minutes with
    no feedback to the user. Serialising the option writers against each other
    is all the race needs.

    **Lock order where both are held: write lock first, then this one.** The
    import handlers are the only place that happens, and they follow it, so the
    pair cannot deadlock.
    """
    locks: dict[str, asyncio.Lock] = hass.data.setdefault(_WS_OPTIONS_LOCKS_KEY, {})
    lock = locks.get(entry_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[entry_id] = lock
    return lock


def _get_entry(hass: HomeAssistant, entry_id: str) -> Any | None:
    return next(
        (e for e in hass.config_entries.async_entries(DOMAIN) if e.entry_id == entry_id),
        None,
    )


def _err_not_found(connection: websocket_api.ActiveConnection, msg_id: int, entry_id: str) -> None:
    connection.send_error(msg_id, "not_found", f"No active WashData manager for entry {entry_id!r}")


def _strip_cycle(c: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in c.items() if k not in _CYCLE_STRIP_KEYS}


def _cycle_capabilities(cycle: dict[str, Any], origin: str) -> dict[str, Any]:
    """What the panel may offer for one cycle, given which list it lives in.

    Three separate answers, because "not a real cycle" is not one capability:

    * ``is_reference`` - it lives outside ``past_cycles``, so it feeds envelopes and the
      matcher but never usage statistics. Always derived from list membership, never from
      a ``meta.source`` string: the cycle list and the inspector both answer through here,
      and they must not be able to disagree.
    * ``labelable`` - a profile can be assigned to it. True everywhere: naming the
      programs found in imported history (#344) is the entire point of that feature, and
      the store labels a non-real cycle in place.
    * ``editable`` - trim / split / review apply. Real cycles only; those store functions
      operate on ``past_cycles`` and would silently no-op elsewhere.

    ``origin`` is the list name from :meth:`ProfileStore.find_stored_cycle`
    (``past`` / ``reference`` / ``backfill``) and is passed through so the UI can say
    where a cycle came from - a curated community-store template and a segment the
    importer detected in the user's own history warrant different wording.
    """
    if origin == "past":
        return {"is_reference": False, "labelable": True, "editable": True}
    return {
        "is_reference": True,
        "labelable": True,
        "editable": False,
        "cycle_origin": origin or "reference",
    }


# Option keys that are identity/transient churn and are never recorded in the
# settings changelog (D7): name/title edits flow through the separate `title`
# kwarg, and suggestion application uses its own apply_suggestions command.
_CHANGELOG_SKIP_KEYS = frozenset({CONF_NAME})

# Identity keys that must NEVER be persisted into entry.options; they are
# partitioned out of any submitted/imported option payload before it is saved.
# In this integration only the display name is a pure options-forbidden identity
# key -- it is carried by the config entry title. device_type / power_sensor /
# min_power deliberately live in entry.options post-3.6 (config_flow writes them
# there and the manager resolves them options-first), so they are NOT listed
# here; relocating them to entry.data would shadow the option-first reads.
_OPTIONS_IDENTITY_KEYS = frozenset({CONF_NAME})

# Option keys whose value names an entity or device on THIS Home Assistant.
# They are ordinary tunables when the panel writes them (the user picks from a
# selector listing their own entities), but an *imported* export carries the
# SOURCE system's ids, and applying those re-points this device at entities that
# do not exist here. ``power_sensor`` is the severe one: the integration goes
# silently dead, state stuck at "off" and current_power at 0, with nothing in the
# log but one INFO line. ``ws_import_config`` already refuses to write the
# exporter's ``entry.data`` for exactly this reason - "blindly applying it would
# hijack this device's sensor binding" - but the same keys live in entry.options
# post-3.6, so they arrived through the other door. Found by the test box
# (devtools/testbox) on its first run; see register item 317.
#
# Deliberately NOT here: the notify_*_services lists. They also name the source
# user's targets, but they are shown plainly in the panel's Notifications
# section, and carrying them is usually the point when migrating your own setup.
_IMPORT_LOCAL_BINDING_KEYS = _OPTIONS_IDENTITY_KEYS | frozenset({
    CONF_POWER_SENSOR,
    CONF_ENERGY_SENSOR,
    CONF_ENERGY_PRICE_ENTITY,
    CONF_DOOR_SENSOR_ENTITY,
    CONF_UNLOAD_CONFIRM_ENTITY,
    CONF_SWITCH_ENTITY,
    CONF_EXTERNAL_END_TRIGGER,
    CONF_LINKED_DEVICE,
    CONF_NOTIFY_PEOPLE,
})


def _json_safe(value: Any) -> Any:
    """Best-effort coercion of an option value to a JSON-serializable form."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _diff_option_changes(
    old_effective: dict[str, Any], submitted: dict[str, Any]
) -> list[dict[str, Any]]:
    """Diff submitted option values against the pre-update effective options.

    Returns one changelog entry ``{"key", "old", "new", "timestamp"}`` per key
    whose value genuinely changed, skipping identity/transient keys. Only keys
    present in ``submitted`` are considered so unrelated merged options never
    produce spurious entries. ``old=None`` with a real ``new`` value is
    recorded; an unchanged ``None -> None`` is not.
    """
    ts = dt_util.now().isoformat()
    changes: list[dict[str, Any]] = []
    for key, new_val in submitted.items():
        if key in _CHANGELOG_SKIP_KEYS:
            continue
        old_val = old_effective.get(key)
        if old_val == new_val:
            continue
        changes.append(
            {
                "key": str(key),
                "old": _json_safe(old_val),
                "new": _json_safe(new_val),
                "timestamp": ts,
            }
        )
    return changes


async def _record_option_changes(
    hass: HomeAssistant, entry: Any, updates: dict[str, Any], source: str
) -> None:
    """Record an option write in the settings changelog (#442).

    ``ws_set_options`` did this inline and was the only writer that did, so every
    other path that writes ``entry.options`` was invisible in the history and its
    values could not be reverted per setting. The one the reporter hit is "Apply
    all": nine tunables changed at once with nothing recorded, and the previous
    values were only recoverable because they happened to have an older
    diagnostics dump.

    Must be awaited BEFORE ``async_update_entry``, which schedules a reload that
    rebuilds the store. Never raises: a changelog failure must not cost the write
    it is describing (same contract as the inline version).
    """
    if not updates:
        return
    try:
        old_effective = {**getattr(entry, "data", {}), **getattr(entry, "options", {})}
        changes = _diff_option_changes(old_effective, updates)
        if not changes:
            return
        manager = _get_manager(hass, entry.entry_id)
        store = getattr(manager, "profile_store", None) if manager else None
        if store is not None:
            await store.async_record_settings_changes(changes)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug(
            "Settings changelog recording failed for %s (%s): %s",
            getattr(entry, "entry_id", "?"),
            source,
            exc,
        )


# ─── Panel config + RBAC ────────────────────────────────────────────────────────

_PANEL_STORE_VERSION = 1
_PANEL_STORE_FILE = "ha_washdata_panel"
_PANEL_DATA_KEY = "ha_washdata_panel_cfg"

_LEVEL_RANK = {"none": 0, "read": 1, "edit": 2, "full": 3}
_PANEL_TABS = ("status", "history", "profiles", "settings", "tools", "panel", "ml_lab", "playground")

# Valid values for the two per-user string prefs (validated in ws_set_user_prefs).
_PREF_DATE_FORMATS = ("relative", "absolute")
# lang_override: empty string clears it (fall back to system language); otherwise a
# BCP-47-ish tag (e.g. "en", "pt-BR", "sr-Latn"). Kept as a bounded pattern rather
# than coupling the WS handler to the translations/panel/ language file list —
# the panel already falls back to system language for any tag it can't load.
_PREF_LANG_TAG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
# cycle_context_min: the cycle chart's recorder-history length per device, in
# minutes (register item 513). The panel offers exactly these; 0 = off.
_PREF_CYCLE_CONTEXT_MINUTES = (0, 5, 10, 30, 60)

# Commands that require 'full' (destructive or full-data export/import).
_FULL_COMMANDS = frozenset({
    "wipe_history", "import_config", "export_config", "clear_debug_data", "reprocess_history",
    "trigger_ml_training",
    # Selective export/import wizard (same full-data reach as the wholesale variants).
    "get_export_inventory", "analyze_import", "export_config_selective", "import_config_selective",
    # Undo of a replace import: a whole-store replace like the import itself.
    "undo_import",
    # Reverting on-device models discards learned state -> full access.
    "revert_ml_models",
    # Rewrites the appliance's lifetime odometer, which drives maintenance schedules.
    "set_lifetime_cycle_count",
    # Historical power-data import: ingests a whole power history and writes cycles.
    "history_import_begin", "history_import_chunk", "history_import_recorder",
    "start_history_import_scan", "apply_history_import",
})
# Commands allowed for any authenticated user regardless of device permissions.
_OPEN_COMMANDS = frozenset({
    "get_constants", "get_panel_config", "set_user_prefs",
})
# Admin-only commands.
# Commands that require administrator access ALWAYS — even when RBAC is disabled
# (default), where every authenticated user otherwise resolves to "full". These are
# destructive/global: they can wipe stored data, overwrite config, read/write files
# on disk, or reprocess the whole history. A non-admin HA user must not reach them.
_ADMIN_COMMANDS = frozenset({
    "set_panel_config",
    "get_logs",
    "wipe_history",
    "import_config",
    "export_config",
    "get_export_inventory",
    "analyze_import",
    "export_config_selective",
    "import_config_selective",
    "undo_import",
    "reprocess_history",
    "clear_debug_data",
    # Global community-store mutations: these change the ONE integration-wide GitHub
    # connection / online flag shared by every entry, so an editor of a single device
    # must not be able to connect, disconnect, or toggle online for the whole install.
    "store_connect",
    "store_disconnect",
    "store_set_online",
    "store_set_prefs",
    # They publish under that same install-wide account (audit PLATFORM-11): with
    # RBAC off, any non-admin could otherwise post under the admin's identity.
    "store_upload_cycle",
    "store_upload_device",
    "store_rate_device",
    "store_confirm_device",
    # Historical power-data import: reads a whole recorder history (or an uploaded
    # export of one) and writes cycles into the store, so it is admin-only even with
    # RBAC disabled, exactly like the selective import wizard.
    "history_import_begin",
    "history_import_chunk",
    "history_import_recorder",
    "start_history_import_scan",
    "apply_history_import",
})
# Mutating commands intentionally allowed at the 'read' level. Picking the live
# program is a benign runtime action (it changes detection, not stored data), so
# read users may use the Status program selector. The Playground simulation is a
# read-only what-if replay (it never persists anything) whose name does not start
# with get_, so it is whitelisted here to gate at the 'read' level.
_READ_WRITE_COMMANDS = frozenset({
    # Picking the live program stays a read-level exception by design, although
    # the pick is stored as the cycle's label at cycle end (audit PLATFORM-11
    # proposed edit; the maintainer kept it open to every user of the device).
    "set_program",
    # Background-task registry: read-level runtime actions (watch progress, fetch
    # a what-if/maintenance result, or stop a task). None mutate stored data.
    "subscribe_tasks",
    "cancel_task",
    "get_task_result",
    "start_playground_history",
    "start_playground_sweep",
    "start_playground_cycle_detail",
    # Community store: read-only browse is read-level (writes below default to 'edit').
    "store_status",
    "store_search_devices",
    "store_list_brands",
    "store_get_profiles",
    "store_get_cycles",
    "store_get_device_profiles",
    "store_get_catalog_entry",
    # NB: store_refresh_catalog is deliberately NOT here. It drops the install-wide
    # catalog cache in the shared StoreClient, so a read-level user could bust the
    # 1-hour TTL that protects the free-tier read budget on repeat. It defaults to
    # 'edit', like the other install-wide store actions above read level.
})

_LOG_BUFFER_KEY = "ha_washdata_log_buffer"
_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


class _RingLogHandler(logging.Handler):
    """In-memory ring buffer of recent ha_washdata log records for the Logs page."""

    def __init__(self, maxlen: int = 500) -> None:
        super().__init__()
        self.records: collections.deque = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append({
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name.split(".")[-1],
                "device": getattr(record, "wd_device", None),
                "msg": record.getMessage(),
            })
        except Exception:  # pylint: disable=broad-exception-caught
            pass


def _default_panel_cfg() -> dict[str, Any]:
    return {
        "panel": {"poll_interval_s": 5, "default_tab": "status", "hidden_tabs": []},
        "rbac": {"enabled": False, "default_level": "none", "users": {}},
        "prefs": {},
    }


async def async_load_panel_config(hass: HomeAssistant) -> None:
    """Load (once) the panel-global config + RBAC store into hass.data."""
    if _PANEL_DATA_KEY in hass.data:
        return
    store = Store(hass, _PANEL_STORE_VERSION, _PANEL_STORE_FILE)
    cfg = _default_panel_cfg()
    try:
        loaded = await store.async_load()
        if isinstance(loaded, dict):
            if isinstance(loaded.get("panel"), dict):
                cfg["panel"].update(loaded["panel"])
            if isinstance(loaded.get("rbac"), dict):
                for k in ("enabled", "default_level", "users"):
                    if k in loaded["rbac"]:
                        cfg["rbac"][k] = loaded["rbac"][k]
                if not isinstance(cfg["rbac"].get("users"), dict):
                    cfg["rbac"]["users"] = {}
            if isinstance(loaded.get("prefs"), dict):
                cfg["prefs"] = loaded["prefs"]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Failed to load panel config, using defaults: %s", exc)
    hass.data[_PANEL_DATA_KEY] = {"store": store, "data": cfg}

    if _LOG_BUFFER_KEY not in hass.data:
        handler = _RingLogHandler()
        handler.setLevel(logging.DEBUG)
        wd_logger = logging.getLogger("custom_components.ha_washdata")
        wd_logger.addHandler(handler)
        # The ring handler captures at the logger's configured effective level
        # (HA's default is INFO, so lifecycle activity shows out of the box). We do
        # NOT raise the logger level here: doing so would override a user who set
        # this integration to WARNING and leak INFO records into home-assistant.log.
        # To see more in the panel Logs view, set the integration's log level in HA.
        hass.data[_LOG_BUFFER_KEY] = handler


def _panel_data(hass: HomeAssistant) -> dict[str, Any]:
    holder = hass.data.get(_PANEL_DATA_KEY)
    return holder["data"] if holder else _default_panel_cfg()


async def _save_panel_data(hass: HomeAssistant) -> None:
    holder = hass.data.get(_PANEL_DATA_KEY)
    if holder:
        await holder["store"].async_save(holder["data"])


def _effective_level(hass: HomeAssistant, user: Any, entry_id: str | None) -> str:
    """Resolve a user's access level for a device (none/read/edit/full)."""
    if user is None:
        return "none"
    if getattr(user, "is_admin", False):
        return "full"
    rbac = _panel_data(hass).get("rbac", {})
    if not rbac.get("enabled"):
        return "full"  # RBAC disabled -> unrestricted (original behavior)
    u = (rbac.get("users") or {}).get(user.id)
    if isinstance(u, dict):
        if entry_id and entry_id in (u.get("devices") or {}):
            return u["devices"][entry_id]
        return u.get("default", "none")
    return rbac.get("default_level", "none")


# Background-task commands that may reference a task by id with no device context.
_TASK_COMMANDS = frozenset({"subscribe_tasks", "cancel_task", "get_task_result"})
# Cancelling a task needs the level its START needed (audit PLATFORM-11): a read
# user could stop an admin's history import or reprocess. Read-level kinds are the
# what-if Playground runs; admin kinds are started by admin-only commands.
_READ_TASK_KINDS = frozenset({"pg_history", "pg_sweep", "pg_detail"})
_ADMIN_TASK_KINDS = frozenset({"reprocess", "history_import", "history_import_apply"})


def _rbac_ok(hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]) -> bool:
    """Authorize a command for the calling user; sends an error and returns False if denied."""
    user = getattr(connection, "user", None)
    if user is None:
        connection.send_error(msg["id"], "unauthorized", "No authenticated user")
        return False
    if getattr(user, "is_admin", False):
        return True
    cmd = str(msg.get("type", "")).split("/", 1)[-1]
    if cmd in _ADMIN_COMMANDS:
        connection.send_error(msg["id"], "forbidden", "Administrator access required")
        return False
    if cmd in _OPEN_COMMANDS:
        return True
    entry_id = msg.get("entry_id")
    # Background-task commands can address a task by id without an entry_id. Under RBAC,
    # a non-admin must not read/cancel a task on a device they aren't authorized for by
    # simply omitting entry_id (which would otherwise fall through the allow-all below).
    # When RBAC is disabled this block is skipped entirely, so default behavior is intact.
    if not entry_id and cmd in _TASK_COMMANDS and _panel_data(hass).get("rbac", {}).get("enabled"):
        task_id = msg.get("task_id")
        if task_id:
            from . import task_registry  # pylint: disable=import-outside-toplevel

            task = task_registry.get_registry(hass).get(task_id)
            entry_id = getattr(task, "entry_id", None) if task is not None else None
            if not entry_id:
                connection.send_error(msg["id"], "forbidden", "Task not found or not authorized")
                return False
        else:
            # subscribe_tasks with no device context: require a device so
            # the result set can be authorized (the panel passes entry_id per device).
            connection.send_error(msg["id"], "forbidden", "You need to specify a device")
            return False
    if cmd == "cancel_task" and msg.get("task_id"):
        from . import task_registry  # pylint: disable=import-outside-toplevel

        task = task_registry.get_registry(hass).get(msg["task_id"])
        kind = getattr(task, "kind", None)
        if kind in _ADMIN_TASK_KINDS:
            connection.send_error(msg["id"], "forbidden", "Administrator access required")
            return False
        if kind not in _READ_TASK_KINDS:
            entry_id = entry_id or getattr(task, "entry_id", None)
            if entry_id and _LEVEL_RANK.get(
                _effective_level(hass, user, entry_id), 0
            ) < _LEVEL_RANK["edit"]:
                connection.send_error(
                    msg["id"], "forbidden", "You need edit access to this device"
                )
                return False
    if not entry_id:
        return True  # no device context and not admin/open: harmless read-style command
    if cmd in _READ_WRITE_COMMANDS:
        required = "read"
    elif cmd in _FULL_COMMANDS:
        required = "full"
    elif cmd.startswith("get_"):
        required = "read"
    else:
        required = "edit"
    have = _effective_level(hass, user, entry_id)
    if _LEVEL_RANK.get(have, 0) >= _LEVEL_RANK[required]:
        return True
    connection.send_error(msg["id"], "forbidden", f"You need {required} access to this device")
    return False


def _guard(handler: Any) -> Any:
    """Wrap a websocket handler with an RBAC check.

    Uses functools.wraps so the websocket_command/async_response markers and
    schema attributes carry over verbatim, keeping sync (@callback) and async
    (@async_response) handlers working unchanged.
    """
    @functools.wraps(handler)
    def wrapper(hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]) -> Any:
        if not _rbac_ok(hass, connection, msg):
            return None
        return handler(hass, connection, msg)
    return wrapper


def _sanitize_panel(p: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    out = dict(current)
    if "poll_interval_s" in p:
        try:
            out["poll_interval_s"] = max(2, min(60, int(p["poll_interval_s"])))
        except (TypeError, ValueError, OverflowError):
            pass
    if p.get("default_tab") in _PANEL_TABS:
        out["default_tab"] = p["default_tab"]
    if isinstance(p.get("hidden_tabs"), list):
        out["hidden_tabs"] = [t for t in p["hidden_tabs"] if t in _PANEL_TABS and t not in ("status", "panel")]
    return out


def _sanitize_rbac(r: dict[str, Any]) -> dict[str, Any]:
    levels = set(_LEVEL_RANK)
    dlevel = r.get("default_level", "none")
    out: dict[str, Any] = {
        "enabled": bool(r.get("enabled", False)),
        "default_level": dlevel if dlevel in levels else "none",
        "users": {},
    }
    for uid, u in (r.get("users") or {}).items():
        if not isinstance(u, dict):
            continue
        d = u.get("default", "none")
        devices = {str(eid): lvl for eid, lvl in (u.get("devices") or {}).items() if lvl in levels}
        out["users"][str(uid)] = {"default": d if d in levels else "none", "devices": devices}
    return out


# ─── Registration ─────────────────────────────────────────────────────────────

# ─── Community store (online features) ─────────────────────────────────────────

def _store_ctx(hass: HomeAssistant, entry_id: str) -> tuple[Any, dict[str, Any]] | None:
    """Return (manager, options) when online features are enabled (global), else None."""
    from .store import online_features_enabled
    entry = _get_entry(hass, entry_id)
    manager = _get_manager(hass, entry_id)
    if entry is None or manager is None:
        return None
    if not online_features_enabled(hass):
        return None
    return manager, dict(entry.options)


@websocket_api.websocket_command({vol.Required("type"): "ha_washdata/store_status", vol.Required("entry_id"): str})
@websocket_api.async_response
async def ws_store_status(hass, connection, msg):
    from .store import online_features_enabled
    manager = _get_manager(hass, msg["entry_id"])
    if manager is None:
        _err_not_found(connection, msg["id"], msg["entry_id"])
        return
    if not online_features_enabled(hass):
        _send_result(connection, msg["id"], "store_status", {"enabled": False})
        return
    _send_result(connection, msg["id"], "store_status", manager.store_bridge.status())


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_connect", vol.Required("entry_id"): str,
    vol.Required("refresh_token"): str, vol.Required("uid"): str, vol.Optional("name"): vol.Any(str, None),
})
@websocket_api.async_response
async def ws_store_connect(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_connect", {"disabled": True})
        return
    manager, _ = ctx
    res = await manager.store_bridge.connect(msg["refresh_token"], msg["uid"], msg.get("name"))
    _send_result(connection, msg["id"], "store_connect", res)


@websocket_api.websocket_command({vol.Required("type"): "ha_washdata/store_disconnect", vol.Required("entry_id"): str})
@websocket_api.async_response
async def ws_store_disconnect(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_disconnect", {"disabled": True})
        return
    manager, _ = ctx
    _send_result(connection, msg["id"], "store_disconnect", await manager.store_bridge.disconnect())


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_search_devices", vol.Required("entry_id"): str,
    vol.Optional("query"): vol.Any(str, None), vol.Optional("appliance_type"): vol.Any(str, None),
    vol.Optional("model_query"): vol.Any(str, None), vol.Optional("include_pending"): bool,
})
@websocket_api.async_response
async def ws_store_search_devices(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_search_devices", {"disabled": True})
        return
    manager, _ = ctx
    items = await manager.store_bridge.search_devices(
        msg.get("query"), msg.get("appliance_type"),
        model_query=msg.get("model_query"), include_pending=bool(msg.get("include_pending", False)),
    )
    _send_result(connection, msg["id"], "store_search_devices", {"items": items})


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_list_brands", vol.Required("entry_id"): str,
    vol.Optional("query"): vol.Any(str, None), vol.Optional("include_pending"): bool,
})
@websocket_api.async_response
async def ws_store_list_brands(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_list_brands", {"disabled": True})
        return
    manager, _ = ctx
    items = await manager.store_bridge.list_brands(msg.get("query"), include_pending=bool(msg.get("include_pending", True)))
    _send_result(connection, msg["id"], "store_list_brands", {"items": items})


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_get_catalog_entry", vol.Required("entry_id"): str,
    vol.Required("brand"): str, vol.Required("model"): str, vol.Required("appliance_type"): str,
})
@websocket_api.async_response
async def ws_store_get_catalog_entry(hass, connection, msg):
    """Resolve just this appliance's catalog brand + device documents, by id.

    Two point reads, replacing the brand list + device list the settings form used to
    download purely to locate these two rows (measured: 128 documents, 119 KB). The
    pickers still fetch the full lists, but only when the user opens one.
    """
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_get_catalog_entry", {"disabled": True})
        return
    manager, _ = ctx
    res = await manager.store_bridge.catalog_entry(msg["brand"], msg["model"], msg["appliance_type"])
    _send_result(connection, msg["id"], "store_get_catalog_entry", res)


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_refresh_catalog", vol.Required("entry_id"): str,
})
@websocket_api.async_response
async def ws_store_refresh_catalog(hass, connection, msg):
    """Drop the cached catalog so the next browse re-reads the store.

    The cache is deliberately long-lived (the catalog is near-static and every read is
    charged against a shared free-tier budget), so this is the escape hatch for "someone
    told me my brand was just approved".
    """
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_refresh_catalog", {"disabled": True})
        return
    manager, _ = ctx
    _send_result(connection, msg["id"], "store_refresh_catalog", manager.store_bridge.refresh_catalog())


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_get_device_profiles", vol.Required("entry_id"): str,
    vol.Required("brand"): str, vol.Required("model"): str, vol.Required("appliance_type"): str,
})
@websocket_api.async_response
async def ws_store_get_device_profiles(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_get_device_profiles", {"disabled": True})
        return
    manager, _ = ctx
    res = await manager.store_bridge.device_profiles(msg["brand"], msg["model"], msg["appliance_type"])
    _send_result(connection, msg["id"], "store_get_device_profiles", res)


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_confirm_device", vol.Required("entry_id"): str,
    vol.Required("device_id"): str,
})
@websocket_api.async_response
async def ws_store_confirm_device(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_confirm_device", {"disabled": True})
        return
    manager, _ = ctx
    _send_result(connection, msg["id"], "store_confirm_device", await manager.store_bridge.confirm_device(msg["device_id"]))


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_rate_device", vol.Required("entry_id"): str,
    vol.Required("device_id"): str,
    vol.Required("rating"): vol.All(int, vol.Range(min=1, max=5)),
})
@websocket_api.async_response
async def ws_store_rate_device(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_rate_device", {"disabled": True})
        return
    manager, _ = ctx
    _send_result(connection, msg["id"], "store_rate_device", await manager.store_bridge.rate_device(msg["device_id"], int(msg["rating"])))


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_set_online", vol.Required("entry_id"): str,
    vol.Required("enabled"): bool,
})
@websocket_api.async_response
async def ws_store_set_online(hass, connection, msg):
    """Enable/disable online features integration-wide (device-agnostic)."""
    from . import store_account
    await store_account.async_set_online(hass, bool(msg["enabled"]))
    manager = _get_manager(hass, msg["entry_id"])
    if manager is not None:
        manager.notify_update()
    _send_result(connection, msg["id"], "store_set_online", {"enabled": store_account.online_enabled(hass)})


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_set_prefs", vol.Required("entry_id"): str,
    vol.Required("prefs"): dict,
})
@websocket_api.async_response
async def ws_store_set_prefs(hass, connection, msg):
    """Merge integration-wide community-store preferences (only known keys)."""
    from . import store_account
    prefs = await store_account.async_set_prefs(hass, msg.get("prefs") or {})
    manager = _get_manager(hass, msg["entry_id"])
    if manager is not None:
        manager.notify_update()
    _send_result(connection, msg["id"], "store_set_prefs", {"prefs": prefs})


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_get_profiles", vol.Required("entry_id"): str,
    vol.Required("device_id"): str,
})
@websocket_api.async_response
async def ws_store_get_profiles(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_get_profiles", {"disabled": True})
        return
    manager, _ = ctx
    items = await manager.store_bridge.get_profiles(msg["device_id"])
    # None = the store could not be reached, not "no shared programs" (audit STORE-09).
    _send_result(connection, msg["id"], "store_get_profiles",
                 {"items": items} if items is not None else {"items": [], "error": "store_unreachable"})


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_get_cycles", vol.Required("entry_id"): str,
    vol.Required("profile_id"): str,
})
@websocket_api.async_response
async def ws_store_get_cycles(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_get_cycles", {"disabled": True})
        return
    manager, _ = ctx
    items = await manager.store_bridge.get_cycles(msg["profile_id"])
    _send_result(connection, msg["id"], "store_get_cycles",
                 {"items": items} if items is not None else {"items": [], "error": "store_unreachable"})


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_import_cycle", vol.Required("entry_id"): str,
    vol.Required("cycle_id"): str,
    vol.Optional("target_profile"): vol.Any(str, None), vol.Optional("new_profile_name"): vol.Any(str, None),
})
@websocket_api.async_response
async def ws_store_import_cycle(hass, connection, msg):
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_import_cycle", {"disabled": True})
        return
    manager, _ = ctx
    # Under the write lock imports, reprocess and merges hold (audit STORE-11): a
    # replace-mode selective import resets reference_cycles mid-download otherwise.
    async with _entry_write_lock(hass, msg["entry_id"]):
        res = await manager.store_bridge.import_cycle(
            msg["cycle_id"], msg.get("target_profile"), msg.get("new_profile_name")
        )
    manager.notify_update()
    _send_result(connection, msg["id"], "store_import_cycle", res)


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_upload_cycle", vol.Required("entry_id"): str,
    vol.Required("local_cycle_id"): str, vol.Required("program"): str,
    vol.Optional("description"): vol.Any(str, None),
})
@websocket_api.async_response
async def ws_store_upload_cycle(hass, connection, msg):
    from .const import CONF_STORE_BRAND, CONF_STORE_MODEL, CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_upload_cycle", {"disabled": True})
        return
    manager, opts = ctx
    brand = str(opts.get(CONF_STORE_BRAND) or "").strip()
    model = str(opts.get(CONF_STORE_MODEL) or "").strip()
    if not brand or not model:
        _send_result(connection, msg["id"], "store_upload_cycle", {"error": "no_appliance_declared"})
        return
    appliance = opts.get(CONF_DEVICE_TYPE, manager.config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE))
    res = await manager.store_bridge.share_cycle(
        msg["local_cycle_id"], msg["program"], brand, model, appliance,
        description=msg.get("description") or "",
    )
    _send_result(connection, msg["id"], "store_upload_cycle", res)


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_upload_device", vol.Required("entry_id"): str,
    vol.Required("items"): [dict], vol.Optional("include_phases"): [str],
    vol.Optional("include_settings"): bool,
})
@websocket_api.async_response
async def ws_store_upload_device(hass, connection, msg):
    """Share a whole-device bundle. Brand/model/type come from this device's options;
    ``items`` = the panel's tree selection ``[{local_cycle_id, program}]``;
    ``include_phases`` = programs whose phase map should ride along;
    ``include_settings`` bundles the allow-listed recognition/matching settings."""
    from .const import (
        CONF_STORE_BRAND, CONF_STORE_MODEL, CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE,
        sanitize_shared_settings,
    )
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_upload_device", {"disabled": True})
        return
    manager, opts = ctx
    brand = str(opts.get(CONF_STORE_BRAND) or "").strip()
    model = str(opts.get(CONF_STORE_MODEL) or "").strip()
    if not brand or not model:
        _send_result(connection, msg["id"], "store_upload_device", {"error": "no_appliance_declared"})
        return
    appliance = opts.get(CONF_DEVICE_TYPE, manager.config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE))
    settings = None
    if msg.get("include_settings"):
        # Only the allow-listed numeric thresholds; the WS layer owns entry.options.
        settings = sanitize_shared_settings(dict(opts))
    res = await manager.store_bridge.share_device(
        brand, model, appliance, msg["items"],
        include_phases=msg.get("include_phases"), settings=settings,
    )
    _send_result(connection, msg["id"], "store_upload_device", res)


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/store_download_device", vol.Required("entry_id"): str,
    vol.Required("device_id"): str, vol.Optional("include_settings"): bool,
})
@websocket_api.async_response
async def ws_store_download_device(hass, connection, msg):
    """Adopt a whole-device bundle into this device's reference cycles (merge/upsert).
    When ``include_settings`` is set, also apply the bundle's allow-listed
    recognition/matching settings onto this device's options (overwrites live tuning)."""
    from .const import CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE
    ctx = _store_ctx(hass, msg["entry_id"])
    if ctx is None:
        _send_result(connection, msg["id"], "store_download_device", {"disabled": True})
        return
    manager, opts = ctx
    device_type = opts.get(CONF_DEVICE_TYPE, manager.config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE))
    # A registry task under the write lock (audit STORE-10/11): a bundle import ran
    # 7-61 s inside this request, unlocked against reprocess and imports.
    reg = task_registry.get_registry(hass)
    task = reg.create(
        msg["entry_id"], "store_download", "Downloading from the store",
        label_key="task.store_download.running",
    )
    raw = hass.async_create_task(_store_download_task(
        hass, task, msg["entry_id"], manager, msg["device_id"], device_type,
        bool(msg.get("include_settings")),
    ))
    if raw is not None:
        reg.link_asyncio_task(task.id, raw)
    _send_result(connection, msg["id"], "store_download_device", {"task_id": task.id})


async def _store_download_task(
    hass: HomeAssistant, task: Any, entry_id: str, manager: Any, device_id: str,
    device_type: str, include_settings: bool,
) -> None:
    reg = task_registry.get_registry(hass)
    try:
        async with _entry_write_lock(hass, entry_id):
            if _get_manager(hass, entry_id) is not manager:
                reg.finish(task, state=task_registry.STATE_ERROR, error="device reloaded")
                return
            res = await manager.store_bridge.download_device(
                device_id, device_type,
                progress=lambda done, total: reg.update(task, done=done, total=total),
                should_cancel=lambda: task.cancel_requested,
            )
            res = {**res, "settings_applied": await _apply_store_settings(
                hass, entry_id, res, include_settings,
            )}
        manager.notify_update()
        reg.finish(
            task,
            state=task_registry.STATE_CANCELLED if res.get("cancelled")
            else task_registry.STATE_DONE,
            result=res,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))


async def _apply_store_settings(
    hass: HomeAssistant, entry_id: str, res: dict[str, Any], include_settings: bool,
) -> int:
    """Write a bundle's opted-in settings; returns how many were applied."""
    from .const import sanitize_shared_settings  # noqa: PLC0415

    settings_applied = 0
    if include_settings:
        bundle_settings = res.get("settings") if isinstance(res.get("settings"), dict) else {}
        # Accept only allow-listed, numeric (non-bool) values - matching what the
        # upload side ever writes - so a malformed/hostile bundle can't inject a
        # string/list/bool into this device's live options.
        filtered = sanitize_shared_settings(bundle_settings)
        entry = _get_entry(hass, entry_id)
        if filtered and entry is not None:
            # Same critical section as ws_set_options: the changelog snapshot and
            # the options write have to be atomic per entry, or a concurrent
            # writer records the same "old" value and one of the two updates is
            # silently lost (#442 follow-up).
            async with _entry_options_lock(hass, entry_id):
                filtered = _without_inverted_threshold_pair(
                    entry, filtered, "store_download"
                )  # item 515
                await _record_option_changes(hass, entry, filtered, "store_download")
                hass.config_entries.async_update_entry(
                    entry, options={**entry.options, **filtered}
                )
            settings_applied = len(filtered)
    return settings_applied


@websocket_api.websocket_command({
    vol.Required("type"): "ha_washdata/get_shareable_cycles", vol.Required("entry_id"): str,
})
@websocket_api.async_response
async def ws_get_shareable_cycles(hass, connection, msg):
    """All recorded/golden reference cycles eligible to share (the share-device tree),
    plus the subset of programs that carry a local phase map (for the phase toggle)."""
    manager = _get_manager(hass, msg["entry_id"])
    if manager is None:
        _err_not_found(connection, msg["id"], msg["entry_id"])
        return
    store = manager.profile_store
    items = store.get_shareable_cycles()
    # All known profiles (not just those with shareable cycles) so the panel can show
    # profiles that exist but have no golden/recorded cycles yet, with guidance.
    all_programs = sorted(store.get_profiles().keys())
    # Phase toggle covers ALL profiles that have a phase map, not just those with
    # shareable cycles — so users with phases but no reference cycles yet still see
    # the toggle (shown as a dimmed no-cycle row in the share tree).
    phase_programs = sorted(p for p in all_programs if store.get_profile_phase_ranges(p))
    _send_result(connection, msg["id"], "get_shareable_cycles",
                 {"items": items, "phase_programs": phase_programs, "all_programs": all_programs})


def async_register_commands(hass: HomeAssistant) -> None:
    """Register all WebSocket commands for the WashData panel.

    Every handler is wrapped in _guard so RBAC is enforced centrally and no
    command can accidentally ship unprotected.
    """
    handlers = [
        ws_get_devices, ws_get_device_cycles,
        # Settings
        ws_get_options, ws_set_options, ws_get_settings_changelog,
        ws_get_setup_status,
        # Profiles
        ws_get_profiles, ws_create_profile, ws_rename_profile, ws_delete_profile,
        ws_rebuild_envelopes, ws_get_profile_phases, ws_set_profile_phases,
        # Profile groups (Stage 5)
        ws_get_profile_groups, ws_save_profile_group, ws_rename_profile_group, ws_delete_profile_group,
        # Maintenance log (Group E)
        ws_get_maintenance_log, ws_add_maintenance_event, ws_delete_maintenance_event,
        ws_set_lifetime_cycle_count,
        ws_add_maintenance_task, ws_update_maintenance_task, ws_delete_maintenance_task,
        # Cycles
        ws_label_cycle, ws_delete_cycle, ws_auto_label_cycles,
        # Phase catalog
        ws_get_phase_catalog, ws_create_phase, ws_update_phase, ws_delete_phase,
        # Recording
        ws_get_recording_state, ws_start_recording, ws_stop_recording,
        ws_process_recording, ws_discard_recording,
        # Feedbacks
        ws_get_feedbacks, ws_resolve_feedback, ws_dismiss_all_feedbacks,
        # Diagnostics
        ws_get_diagnostics, ws_reprocess_history, ws_clear_debug_data,
        ws_wipe_history, ws_export_config, ws_import_config,
        # Selective export/import wizard (inventory + analyze + selective export/import)
        ws_get_export_inventory, ws_analyze_import,
        ws_export_config_selective, ws_import_config_selective,
        # "Undo last import" (register item 195)
        ws_undo_import,
        # Shared constants
        ws_get_constants,
        # Suggestions
        ws_get_suggestions, ws_apply_suggestions, ws_clear_suggestions, ws_run_suggestion_analysis,
        ws_set_suggestion_lock,
        # Cycle curve / interactive editing
        ws_get_cycle_power_data, ws_trim_cycle, ws_analyze_split, ws_apply_split, ws_apply_merge,
        # Recorder history around a stored cycle, display only (item 513)
        ws_get_cycle_context,
        # Profile envelope / member cycles
        ws_get_profile_envelope, ws_get_profile_cycles,
        # Panel config + RBAC
        ws_get_panel_config, ws_set_panel_config, ws_set_user_prefs,
        # Logs
        ws_get_logs,
        # Live power history
        ws_get_power_history,
        # Manual program selection
        ws_set_program,
        # Live match debug
        ws_get_match_debug,
        # ML Lab shadow-mode comparison
        ws_get_ml_comparison,
        # ML Lab review write-back (Stage 4b)
        ws_set_ml_review,
        # On-device ML training (status + manual trigger + models revert)
        ws_get_ml_training_status, ws_trigger_ml_training, ws_revert_ml_models,
        # Cycle controls (pause / resume / force-stop)
        ws_pause_cycle, ws_resume_cycle, ws_terminate_cycle,
        # Playground settings control panel: live values + named presets
        ws_get_playground_settings, ws_save_playground_preset, ws_delete_playground_preset,
        # Background-task registry (progress / cancel / reconnect-safe results)
        ws_subscribe_tasks, ws_cancel_task, ws_get_task_result,
        # Playground batch/sweep as detached registry-tracked tasks
        ws_start_playground_history, ws_start_playground_sweep,
        ws_start_playground_cycle_detail,
        # Community store (online features): status/connect/disconnect/browse/import/upload
        ws_store_status, ws_store_connect, ws_store_disconnect,
        ws_store_search_devices, ws_store_get_profiles, ws_store_get_cycles,
        ws_store_import_cycle, ws_store_upload_cycle,
        # Device-bundle sharing (Stage 1): upload a whole device + adopt one
        ws_store_upload_device, ws_store_download_device,
        # Local reference cycles eligible to share (share-device tree source)
        ws_get_shareable_cycles,
        # Community catalog: brand list, device quality, confirm/rate, global online toggle
        ws_store_list_brands, ws_store_get_device_profiles,
        ws_store_confirm_device, ws_store_rate_device, ws_store_set_online,
        ws_store_set_prefs,
        # Catalog identity badges (point reads) + manual cache refresh
        ws_store_get_catalog_entry, ws_store_refresh_catalog,
        # Historical power-data import: staged ingest, background scan, apply
        ws_history_import_begin, ws_history_import_chunk, ws_history_import_recorder,
        ws_start_history_import_scan, ws_apply_history_import,
    ]
    for handler in handlers:
        websocket_api.async_register_command(hass, _guard(handler))


# ─── Devices ──────────────────────────────────────────────────────────────────

@websocket_api.websocket_command({vol.Required("type"): "ha_washdata/get_devices"})
@callback
def ws_get_devices(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return all WashData config entries with their live state (RBAC-filtered)."""
    entries = hass.config_entries.async_entries(DOMAIN)
    domain_data: dict[str, Any] = hass.data.get(DOMAIN, {})
    user = getattr(connection, "user", None)
    devices: list[dict[str, Any]] = []

    for entry in entries:
        level = _effective_level(hass, user, entry.entry_id)
        if level == "none":
            continue  # device hidden from this user by RBAC
        manager = domain_data.get(entry.entry_id) if isinstance(domain_data, dict) else None

        info: dict[str, Any] = {
            "entry_id": entry.entry_id,
            "perm": level,
            "title": entry.title,
            "detector_state": "unknown",
            "sub_state": None,
            # Declared non-optional in DeviceInfo, so it is defaulted HERE rather
            # than at each exit: a device with no loaded manager never reaches
            # the probe at all, which is the case the contract test caught.
            "standby_above_stop": None,
            "current_program": None,
            "time_remaining_s": None,
            "total_duration_s": None,
            "expected_duration_s": None,
            "current_power_w": None,
            "cycle_progress_pct": None,
            "suggestions_count": 0,
            # Which keys those are, so the panel can merge them with the
            # Calibrated (ML) recommendations it computes client-side without
            # double-counting a key both engines suggest.
            "suggestion_keys": [],
            "feedback_count": 0,
            "recording": False,
            "is_user_paused": False,
            "manual_program": False,
            "armed_program": None,
            # Defaulted here rather than only in the manager branch below: both are
            # declared non-optional by DeviceInfo (ws_schema.py) and ws-types.d.ts,
            # and the branch is skipped for a manager-less entry (mid-setup, stale,
            # or a setup that failed) as well as short-circuited by its own except.
            "envelope_position": None,
            # Top two of an undecided live match (MATCH-DECIDE-15): display only.
            "match_uncertainty": None,
            # Merged data+options, matching ``ws_get_options`` and the
            # options-first resolution in WashDataManager (#450). An entry added
            # after its last schema migration carries the structural keys in
            # ``entry.data`` only, so serving bare options handed the panel a
            # device_type fallback that did not match the running manager.
            "options": {**entry.data, **entry.options, CONF_NAME: entry.title},
            # Device-resolved defaults for the cadence/ratio fields (#396/#393) so the
            # device-list conflict/suggestion badges score an unset field against the
            # value the integration would actually use, matching the Settings tab.
            "option_defaults": _resolved_option_defaults(
                {**entry.data, **entry.options}.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)
            ),
        }

        if manager is not None:
            try:
                detector = getattr(manager, "detector", None)
                if detector is not None:
                    # What the entities show (item 501): a hidden standby re-probe
                    # stays "off" here too instead of flickering the Status card.
                    info["detector_state"] = getattr(detector, "exposed_state", detector.state)
                    # Matching it (#452: "Stalled" while a stalled cycle shows paused).
                    info["sub_state"] = getattr(detector, "exposed_sub_state", detector.sub_state)

                program: str | None = getattr(manager, "_current_program", None)
                if program in (None, "off", "unknown", "detecting...", "restored..."):
                    program = None
                info["current_program"] = program
                info["manual_program"] = bool(getattr(manager, "manual_program_active", False))
                # A program pinned for the NEXT cycle (#411). Kept separate from
                # current_program so an idle device does not claim to be running one.
                info["armed_program"] = getattr(manager, "armed_program", None)

                info["time_remaining_s"] = getattr(manager, "_time_remaining", None)
                info["total_duration_s"] = getattr(manager, "_total_duration", None)
                # The matched program's expected length: the span the phase sensor
                # maps progress onto, so the Status timeline names the same phase
                # (audit PROGRESS-10).
                info["expected_duration_s"] = getattr(
                    manager, "_matched_profile_duration", None
                )

                # Read through the property, not the raw cache (#409): it falls back
                # to the sensor's live state so the panel can never show a power the
                # sensor never reported.
                power = getattr(manager, "current_power", None)
                info["current_power_w"] = round(float(power), 2) if power is not None else None

                progress = getattr(manager, "_cycle_progress", None)
                if progress is not None:
                    info["cycle_progress_pct"] = round(float(progress), 1)

                # Where the run maps onto the matched profile's own curve (item 269).
                # Independent of elapsed time, so it stays meaningful when a cycle
                # over- or under-runs; refreshed only during low-power phases, which
                # the panel's tooltip says.
                info["envelope_position"] = getattr(manager, "envelope_position", None)

                # "Uncertain: X or Y, ~N% sure" on the Status card while the live
                # match is undecided (MATCH-DECIDE-15). The sensor state keeps its
                # raw `detecting...` value, which automations compare against.
                _unc = getattr(manager, "match_uncertainty", None)
                info["match_uncertainty"] = _unc if isinstance(_unc, dict) else None

                store = getattr(manager, "profile_store", None)
                if store is not None:
                    # Assigned per entry and OUTSIDE both probes. It used to be
                    # set inside the suggestion-badge `try`, after
                    # `store.get_suggestions()`: if that raised, `merged` kept
                    # the PREVIOUS entry's options and the standby probe below
                    # read another device's stop threshold - or, on the first
                    # entry, raised NameError into a debug-level log.
                    merged = {**entry.data, **entry.options}
                    try:
                        # The same filter as the Settings list, so the Overview
                        # card can never disagree with the Settings tab banner.
                        keys = [
                            k for k, _i, _s, _c in _visible_suggestions(
                                store, merged,
                                getattr(manager, "device_type", None) or DEFAULT_DEVICE_TYPE,
                            )
                        ]
                        info["suggestion_keys"] = keys
                        info["suggestions_count"] = len(keys)
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                    try:
                        # Imported here rather than at module scope to keep the
                        # ws_api import graph free of suggestion_engine, as every
                        # other use in this file does.
                        from .suggestion_engine import (  # pylint: disable=import-outside-toplevel
                            detect_standby_above_stop,
                        )

                        # #445 cause 1: an appliance whose standby draw sits ABOVE
                        # stop_threshold_w can never finish a cycle on its own,
                        # because the off delay only starts once power is below it.
                        # Surfaced as its own attention card rather than folded into
                        # a suggestion: no threshold value can fix the case where the
                        # appliance's idle and working power are the same level, so
                        # this explains rather than proposes.
                        # Resolved in steps, not as an inline `.get(key, default)`:
                        # Python evaluates that default eagerly, so a detector
                        # without a bound `config` raised AttributeError before the
                        # option was even consulted - and the handler below turned
                        # that into a silently absent advisory.
                        _stop_w = merged.get(CONF_STOP_THRESHOLD_W)
                        if _stop_w is None:
                            _cfg = getattr(getattr(manager, "detector", None), "config", None)
                            _stop_w = getattr(_cfg, "stop_threshold_w", 0.0)
                        info["standby_above_stop"] = detect_standby_above_stop(
                            store.get_past_cycles() or [],
                            float(_stop_w or 0.0),
                        )
                    except Exception:  # pylint: disable=broad-exception-caught
                        # Logged, not swallowed silently: this advisory shipped dead
                        # for a round because a NameError here was indistinguishable
                        # from "no pattern found".
                        _LOGGER.debug("standby_above_stop probe failed", exc_info=True)
                        # Key already defaulted to None where `info` is built.
                    try:
                        # Count only pending feedback whose cycle still exists, so the
                        # badge cannot outrun the review list after a cycle is deleted,
                        # merged or split (#362). Defense-in-depth on top of the prune in
                        # those mutation paths; also self-heals pre-existing orphans.
                        _pend = store.get_pending_feedback() or {}
                        _live = {c.get("id") for c in store.get_past_cycles()}
                        info["feedback_count"] = sum(1 for cid in _pend if cid in _live)
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                info["is_user_paused"] = bool(getattr(manager, "is_user_paused", False))
                recorder = getattr(manager, "recorder", None)
                if recorder is not None:
                    info["recording"] = bool(getattr(recorder, "is_recording", False))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("Error reading manager state for entry %s: %s", entry.entry_id, exc)

        devices.append(info)

    _send_result(connection, msg["id"], "get_devices", {"devices": devices})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_device_cycles",
        vol.Required("entry_id"): str,
        vol.Optional("limit", default=50): vol.All(int, vol.Range(min=1, max=200)),
        vol.Optional("offset", default=0): vol.All(int, vol.Range(min=0)),
        vol.Optional("imported_offset"): vol.All(int, vol.Range(min=0)),
    }
)
@callback
def ws_get_device_cycles(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return a page of recent cycles for a device, stripping large binary fields.

    Real cycles are returned most-recent-first and sliced ``[offset : offset+limit]``
    so the panel can page. ``total`` is the device's real cycle count and
    ``has_more`` is True when real cycles remain beyond the returned window.

    Imported cycles (``reference_cycles`` + ``backfill_cycles``) page on their own
    cursor, ``imported_offset``, over one newest-start-first list of both, with the
    same ``limit``; ``imported_total`` / ``imported_has_more`` describe it. They keep
    their own cursor because they stay out of ``total`` (they never enter usage
    stats), so folding them into ``offset`` would change what ``total`` counts.
    A client that omits ``imported_offset`` gets the pre-paging behaviour: every
    imported cycle on ``offset == 0`` and none after (register item 129a).
    """
    entry_id: str = msg["entry_id"]
    limit: int = msg.get("limit", 50)
    offset: int = msg.get("offset", 0)
    imported_offset: int | None = msg.get("imported_offset")

    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    cycles: list[dict[str, Any]] = []
    reference_cycles: list[dict[str, Any]] = []
    backfill_cycles: list[dict[str, Any]] = []
    total = 0
    imported_total = 0
    imported_end = 0
    try:
        store = getattr(manager, "profile_store", None)
        if store is not None:
            raw: list[Any] = store.get_past_cycles()
            total = len(raw)
            # History order is oldest-first in storage; present most-recent-first
            # and slice the requested page. offset=0 is identical to the legacy
            # reversed(raw[-limit:]) behaviour.
            ordered = list(reversed(raw))
            window = ordered[offset:offset + limit]
            for c in window:
                cycles.append(_strip_cycle(c))
            # Imported store recordings and cycles recovered from raw history are kept
            # out of `cycles`/`total` (they never enter usage stats), tagged so the
            # panel can badge them and route edits/deletes correctly. They travel in
            # separate arrays because they are separate categories: a curated community
            # template and an auto-detected segment from the user's own past are not
            # the same claim about a cycle.
            imported = _imported_cycles_newest_first(store)
            imported_total = len(imported)
            if imported_offset is not None:
                imported_window = imported[imported_offset:imported_offset + limit]
                imported_end = imported_offset + len(imported_window)
            elif offset == 0:
                imported_window = imported
                imported_end = imported_total
            else:
                imported_window = []
                imported_end = imported_total
            for c, origin in imported_window:
                item = _strip_cycle(c)
                item.update(_cycle_capabilities(c, origin))
                (reference_cycles if origin == "reference" else backfill_cycles).append(item)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error fetching cycles for entry %s: %s", entry_id, exc)

    has_more = (offset + len(cycles)) < total
    _send_result(connection, msg["id"], "get_device_cycles", {
            "entry_id": entry_id,
            "cycles": cycles,
            "reference_cycles": reference_cycles,
            "backfill_cycles": backfill_cycles,
            "total": total,
            "has_more": has_more,
            "imported_total": imported_total,
            "imported_has_more": imported_end < imported_total,
        },
    )


def _imported_cycles_newest_first(store: Any) -> list[tuple[dict[str, Any], str]]:
    """Reference + backfill cycles as ``(cycle, origin)``, newest start first.

    One ordering for both lists so a page holds the most recent imports whatever their
    category. Storage order (reversed, newest-added first) breaks ties and places
    cycles whose ``start_time`` does not parse last, so the order is total and stable
    across requests, which paging needs.
    """
    rows: list[tuple[dict[str, Any], str]] = []
    for origin, getter in (
        ("reference", "get_reference_cycles"),
        ("backfill", "get_backfill_cycles"),
    ):
        seq = getattr(store, getter)()
        if not isinstance(seq, list):
            continue
        rows.extend((c, origin) for c in reversed(seq) if isinstance(c, dict))

    def _start_ts(row: tuple[dict[str, Any], str]) -> float:
        value = row[0].get("start_time")
        try:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)  # legacy numeric unix timestamp
            parsed = dt_util.parse_datetime(str(value or ""))
            # Aware stamps only: a naive one would compare in local time.
            if parsed is not None and parsed.tzinfo is not None:
                return parsed.timestamp()
        except (TypeError, ValueError, OverflowError, OSError):
            pass
        return float("-inf")

    # sorted() is stable, so equal starts keep the newest-added-first order above.
    return sorted(rows, key=_start_ts, reverse=True)


# ─── Settings ─────────────────────────────────────────────────────────────────


def _resolved_option_defaults(device_type: str) -> dict[str, Any]:
    """Device-resolved defaults for the cadence/ratio settings whose default varies
    by device type (#396/#393).

    The panel uses these as the render/conflict-check/suggestion-comparison fallback
    for an unset field, so it shows - and validates against - the value the
    integration would actually use, not a static schema literal that would spuriously
    trip (or silently miss) the watchdog>=2*sampling / start_duration>=sampling rules
    on a coarse-sampling device type. Shared by ws_get_options (current device) and
    ws_get_devices (per device) so the two never diverge.
    """
    return {
        CONF_SAMPLING_INTERVAL: resolve_sampling_interval_default(device_type),
        CONF_WATCHDOG_INTERVAL: resolve_watchdog_interval_default(device_type),
        CONF_START_DURATION_THRESHOLD: resolve_start_duration_default(device_type),
        CONF_SMART_TERMINATION_DURATION_RATIO: (
            resolve_smart_termination_duration_ratio_default(device_type)
        ),
        # #429: deliberately NOT device-resolved (the safe value is a property of
        # the individual machine), but the panel pre-populates it from the same
        # payload, so it is published here rather than left to the JS default.
        CONF_ANTI_CREASE_FINALIZE_RATIO: DEFAULT_ANTI_CREASE_FINALIZE_RATIO,
        # Item 311 raised this to 1.8 in Python and the panel's schema literal
        # stayed at 1.5, so an entry without the key rendered 1.5 while the
        # matcher used 1.8. Not device-resolved either; published for the same
        # reason as the line above, so the constant is the only source and the
        # two cannot drift again. Round 17 removed the migration seed that had
        # been hiding this for legacy entries.
        CONF_PROFILE_MATCH_MAX_DURATION_RATIO: (
            DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
        ),
        # #445: the pair that decides when a cycle is allowed to end. The detector
        # waits max(off_delay, min_off_gap), so an unset min_off_gap silently
        # raises a hand-lowered off_delay to the per-device prior - and with the
        # key absent here the panel rendered an empty field and the cross-field
        # rule's `!= null` guard short-circuited, so nothing anywhere showed the
        # number that was actually governing the wait.
        CONF_MIN_OFF_GAP: resolve_min_off_gap_default(device_type),
        CONF_OFF_DELAY: resolve_off_delay_default(device_type),
    }


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_options", vol.Required("entry_id"): str}
)
@callback
def ws_get_options(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return merged data+options for a config entry."""
    entry = _get_entry(hass, msg["entry_id"])
    if not entry:
        connection.send_error(msg["id"], "not_found", f"Entry {msg['entry_id']!r} not found")
        return
    # #422: the display name is carried by ``entry.title``, never by options -
    # ``_merge_structural_options`` and ``_OPTIONS_IDENTITY_KEYS`` strip CONF_NAME
    # back out on every save. ``entry.data[CONF_NAME]`` is therefore a fossil from
    # ``async_create_entry`` that no rename path updates, and being data-only it is
    # the one key nothing in options can shadow. Pin it to the title so Settings ->
    # Basic shows the same name as the device list (which serves entry.title) and a
    # save can never echo the creation-time name back over the current one.
    options = {**entry.data, **entry.options, CONF_NAME: entry.title}
    # Device-resolved defaults for the cadence settings whose defaults vary by
    # device type (#396). The panel uses these as the render/conflict-check
    # fallback for an unset field so it shows (and validates against) the value the
    # integration would actually use - not a static schema literal that would
    # spuriously trip the panel's own watchdog>=2*sampling / start_duration>=sampling
    # rules on a coarse-sampling device type.
    device_type = options.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)
    _send_result(
        connection,
        msg["id"],
        "get_options",
        {"options": options, "defaults": _resolved_option_defaults(device_type)},
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/set_options",
        vol.Required("entry_id"): str,
        vol.Required("options"): dict,
    }
)
@websocket_api.async_response
async def ws_set_options(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Persist updated options and trigger an entry reload."""
    entry = _get_entry(hass, msg["entry_id"])
    if not entry:
        connection.send_error(msg["id"], "not_found", f"Entry {msg['entry_id']!r} not found")
        return
    # Serialised per entry (#442 follow-up): the snapshot below, the changelog
    # write and `async_update_entry` must be one critical section. Without it
    # two concurrent writers both read the same `entry.options`, both record the
    # same "old" value in the settings history, and the second update silently
    # overwrites the first - so a later per-setting revert restores a value that
    # was never current. The import handlers already hold this lock; the option
    # writers did not. acquire/release rather than `async with` so the handler
    # keeps its early-return validation paths. The OPTIONS lock, not the write
    # lock: see `_entry_options_lock` for why a save must not queue behind a
    # reprocess or an ML training run.
    lock = _entry_options_lock(hass, msg["entry_id"])
    await lock.acquire()
    try:
        # Build the new options from the *existing* options plus the submitted
        # values only. Never spread entry.data in: that would copy identity and
        # data-only keys (name, initial_profile, stale creation-time identity) into
        # options where they don't belong. Tunables (including device_type /
        # power_sensor / min_power, which live in options post-3.6) are preserved
        # from entry.options and overridden by the submission.
        new_options = {**entry.options, **msg["options"]}

        # A numeric setting must be a finite number (audit PLATFORM-13): stored,
        # "abc" raised in the manager's constructor and the entry never set up
        # again. A bad value drops the key so its default applies - the contract
        # the per-key blocks below already follow; None / "" reach their reset path.
        _numeric = numeric_option_keys()
        for _key, _val in msg["options"].items():
            if _key not in _numeric or _val is None or _val == "":
                continue
            _num = coerce_numeric_option(_val, _numeric[_key])
            if _num is None:
                new_options.pop(_key, None)
            else:
                new_options[_key] = _num

        # Capture the submitted display name for the entry title before it is
        # stripped out of options below.
        submitted_name = new_options.get(CONF_NAME)

        # Mirror the OptionsFlow save-time normalization so the panel can never
        # persist stale or invalid values:
        #  - a cleared selector (entity / linked device / trigger) becomes None so
        #    the link or subscription is removed rather than left dangling;
        #  - pump-only keys are dropped for non-pump device types;
        #  - the transient "apply suggestions" flag is never stored.
        for key in (
            CONF_EXTERNAL_END_TRIGGER,
            CONF_DOOR_SENSOR_ENTITY,
            CONF_UNLOAD_CONFIRM_ENTITY,
            CONF_LINKED_DEVICE,
            CONF_SWITCH_ENTITY,
            CONF_ENERGY_SENSOR,
        ):
            if key in new_options and not new_options[key]:
                new_options[key] = None

        # Resolve the effective device type option-first (submission -> existing
        # options -> data -> default) so the pump-only key is dropped correctly even
        # when the submission omits device_type.
        effective_device_type = new_options.get(
            CONF_DEVICE_TYPE, entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)
        )
        if effective_device_type != DEVICE_TYPE_PUMP:
            new_options.pop(CONF_PUMP_STUCK_DURATION, None)

        # Numeric-finite validation for fields that the cycle-detector float()-casts at
        # build time; coerce bad submissions to the compiled default so storage stays clean.
        if CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE in new_options:
            try:
                _qr = float(new_options[CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE])
                if not math.isfinite(_qr):
                    raise ValueError("non-finite")
                new_options[CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE] = _qr
            # OverflowError too, same reason as the anti-crease block below
            # (register item 194): an unbounded int from JSON raises on float()
            # rather than returning inf, and uncaught it fails the WHOLE save
            # with unknown_error instead of dropping this one key.
            except (TypeError, ValueError, OverflowError):
                new_options[CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE] = (
                    DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS
                )

        # Smart-Termination duration ratio (#393): fraction of expected duration, so it
        # is meaningless outside [0.50, 1.00] - clamp valid submissions to the range.
        # An empty or non-numeric value drops the key so the device-type default
        # (resolved in the config builder, 0.99 dishwasher / 0.98 other) applies again;
        # coercing to a single scalar default here would be wrong for dishwashers.
        if CONF_SMART_TERMINATION_DURATION_RATIO in new_options:
            _raw_str = new_options[CONF_SMART_TERMINATION_DURATION_RATIO]
            if _raw_str in (None, ""):
                new_options.pop(CONF_SMART_TERMINATION_DURATION_RATIO, None)
            else:
                try:
                    _str = float(_raw_str)
                    if not math.isfinite(_str):
                        raise ValueError("non-finite")
                    new_options[CONF_SMART_TERMINATION_DURATION_RATIO] = min(
                        1.0, max(0.5, _str)
                    )
                # OverflowError too, see the anti-crease block below.
                except (TypeError, ValueError, OverflowError):
                    new_options.pop(CONF_SMART_TERMINATION_DURATION_RATIO, None)

        # #429: the anti-crease finalise ratio is the same kind of value against the
        # same kind of mean, and equally meaningless outside [0.50, 1.00]. An empty or
        # non-numeric submission drops the key so DEFAULT_ANTI_CREASE_FINALIZE_RATIO
        # applies again.
        if CONF_ANTI_CREASE_FINALIZE_RATIO in new_options:
            _raw_ac = new_options[CONF_ANTI_CREASE_FINALIZE_RATIO]
            if _raw_ac in (None, ""):
                new_options.pop(CONF_ANTI_CREASE_FINALIZE_RATIO, None)
            else:
                try:
                    _ac = float(_raw_ac)
                    if not math.isfinite(_ac):
                        raise ValueError("non-finite")
                    new_options[CONF_ANTI_CREASE_FINALIZE_RATIO] = min(
                        ANTI_CREASE_FINALIZE_RATIO_MAX,
                        max(ANTI_CREASE_FINALIZE_RATIO_MIN, _ac),
                    )
                # OverflowError too (register item 194): json parses an integer literal
                # of any length into an unbounded int, and float() on one of those raises
                # rather than returning inf. Uncaught it becomes ERR_UNKNOWN_ERROR and
                # the whole save fails, instead of this key falling back to its default.
                except (TypeError, ValueError, OverflowError):
                    new_options.pop(CONF_ANTI_CREASE_FINALIZE_RATIO, None)

        # #430: seconds, 0 = off. Clamped to [0, CURVE_PREROLL_MAX_SECONDS] so a
        # mistyped value cannot drag minutes of unrelated standby into a curve; empty
        # or non-numeric drops the key and restores the default (off).
        if CONF_CURVE_PREROLL_SECONDS in new_options:
            _raw_pr = new_options[CONF_CURVE_PREROLL_SECONDS]
            if _raw_pr in (None, ""):
                new_options.pop(CONF_CURVE_PREROLL_SECONDS, None)
            else:
                try:
                    _pr = float(_raw_pr)
                    if not math.isfinite(_pr):
                        raise ValueError("non-finite")
                    new_options[CONF_CURVE_PREROLL_SECONDS] = min(
                        CURVE_PREROLL_MAX_SECONDS, max(0.0, _pr)
                    )
                except (TypeError, ValueError, OverflowError):
                    new_options.pop(CONF_CURVE_PREROLL_SECONDS, None)

        # A None outside the clearable selectors means "not set", not a value: the
        # per-setting Revert sends the changelog's `old`, which is null for a setting
        # never saved before. Stored, it would survive options.get(key, DEFAULT) and
        # break the float()/int() casts at setup, so drop the key and let the default
        # apply again; this also cleans nulls persisted by earlier builds.
        _pre_strip_keys = set(new_options)
        new_options = strip_null_options(new_options)
        dropped_null_keys = _pre_strip_keys - set(new_options)

        # Partition identity out of options: the display name is carried by the
        # entry title, never persisted in options (matches the config-flow invariant
        # that CONF_NAME is absent from options).
        for key in _OPTIONS_IDENTITY_KEYS:
            new_options.pop(key, None)

        # Register item 515: a save must not leave the stop threshold at or above
        # the start threshold. The panel's own conflict check holds such an edit
        # back, but the Playground publish/sweep, the per-setting Revert and any
        # other client wrote it straight through. An entry already running an
        # inverted pair keeps saving every other setting (see the helper).
        _inverted = inverted_threshold_pair(
            {**entry.data, **entry.options},
            {**entry.data, **new_options},
            effective_device_type,
        )
        if _inverted is not None:
            connection.send_error(
                msg["id"], "invalid_threshold_pair", _threshold_pair_message(*_inverted)
            )
            return

        update_kwargs: dict[str, Any] = {"options": new_options}
        if isinstance(submitted_name, str) and submitted_name.strip():
            update_kwargs["title"] = submitted_name.strip()

        # Settings change history (D7): diff the pre-update effective options against
        # the post-normalization values, but only for keys the user actually
        # submitted, and persist BEFORE async_update_entry (which schedules a reload
        # that rebuilds the store). A changelog failure must never block the save.
        # Through the shared helper, not a second copy of it. The per-setting
        # Revert trusts this history, so the two writers have to follow one
        # contract - and `_record_option_changes`'s own docstring already says it
        # replaces the inline version that used to live here. Keys dropped by the
        # null-strip above are still recorded as a change to None ("reverted to
        # unset"); a None -> None no-op is skipped inside `_diff_option_changes`.
        submitted_post = {
            k: new_options.get(k)
            for k in msg["options"]
            if k in new_options or k in dropped_null_keys
        }
        await _record_option_changes(hass, entry, submitted_post, "set_options")

        # When online features are disabled, clear the persisted store account so the
        # user's identity isn't silently retained after they opt out.
        from .const import CONF_ENABLE_ONLINE_FEATURES  # pylint: disable=import-outside-toplevel
        was_online = bool(entry.options.get(CONF_ENABLE_ONLINE_FEATURES, False))
        now_online = bool(new_options.get(CONF_ENABLE_ONLINE_FEATURES, False))
        if was_online and not now_online:
            try:
                manager = _get_manager(hass, msg["entry_id"])
                store = getattr(manager, "profile_store", None) if manager else None
                if store is not None:
                    await store.clear_store_account()
                    await store.async_save()
            except Exception:  # pylint: disable=broad-exception-caught
                # Warning, not a silent pass: the user asked for this identity to
                # be removed, and a bare `pass` here leaves it in storage with no
                # trace. A bare `pass` is also what hid the #445 standby advisory
                # being dead for a whole review round (register item 325).
                _LOGGER.warning(
                    "Could not clear the store account for %s after online "
                    "features were disabled; the stored identity may remain",
                    msg["entry_id"],
                    exc_info=True,
                )

        hass.config_entries.async_update_entry(entry, **update_kwargs)
        _send_result(connection, msg["id"], "set_options", {"success": True})
    finally:
        lock.release()


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_settings_changelog",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_get_settings_changelog(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the settings-change history for a device (most-recent-first)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    changelog: list[dict[str, Any]] = []
    try:
        store = getattr(manager, "profile_store", None)
        if store is not None:
            changelog = store.get_settings_changelog()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug(
            "Error fetching settings changelog for entry %s: %s", entry_id, exc
        )

    _send_result(connection, msg["id"], "get_settings_changelog", {"changelog": changelog})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_setup_status",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_get_setup_status(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the current setup phase for the adoption guidance card."""
    manager = _get_manager(hass, msg["entry_id"])
    if not manager:
        connection.send_error(msg["id"], "not_found", "Device not found")
        return

    # Gather user's skipped steps from user prefs
    skipped_steps: dict[str, str | None] = {}
    user = getattr(connection, "user", None)
    if user:
        holder = hass.data.get(_PANEL_DATA_KEY)
        if holder:
            prefs = holder["data"].get("prefs", {}).get(user.id, {})
            for k, v in prefs.items():
                if k.startswith("setup_skip_"):
                    skipped_steps[k] = v

    # Gather store data (executor-safe reads)
    store = manager.profile_store
    profile_names = list(store._data.get("profiles", {}).keys())
    # The evidence view, split back into its three lists: every advisor check asks
    # "can this profile be matched", which is what has_real_profiles answers with the
    # same view, so the card's phase0 coincides with the manager skipping matching.
    # Backfilled history is this machine's own (item 129d); before, only past_cycles
    # were read and an import-only device was told to start recording.
    evidence = {id(c) for c in store.iter_evidence_cycles()}
    past_cycles = [c for c in store.get_past_cycles() if id(c) in evidence]
    backfill_cycles = [c for c in store.get_backfill_cycles() if id(c) in evidence]
    known_profiles = set(profile_names)
    ref_names: set[str] = {
        rc["profile_name"]
        for rc in store.get_reference_cycles()
        if id(rc) in evidence and rc.get("profile_name") in known_profiles
    }

    # Read from a loop-side snapshot: the executor must not iterate live store
    # dicts (audit PERF-10).
    coverage_gap = await hass.async_add_executor_job(
        _store_read_snapshot(store).suggest_coverage_gaps
    )
    # suggestions: read from the store (cheap dict lookup; heavy computation happens
    # in the SuggestionEngine background task, not here).
    _entry = _get_entry(hass, msg["entry_id"])
    _merged = {**_entry.data, **_entry.options} if _entry is not None else {}
    suggestions = [
        item for _k, item, _s, _c in _visible_suggestions(
            store, _merged, getattr(manager, "device_type", None) or DEFAULT_DEVICE_TYPE
        )
    ]

    device_type = manager.device_type

    result = compute_setup_phase(
        device_type=device_type,
        profile_names=profile_names,
        past_cycles=past_cycles,
        ref_profile_names=ref_names,
        coverage_gap=coverage_gap,
        suggestions=suggestions,
        skipped_steps=skipped_steps,
        now=dt_util.now(),
        backfill_cycles=backfill_cycles,
    )

    _send_result(connection, msg["id"], "get_setup_status", {
        "phase": result.phase,
        "message_key": result.message_key,
        "message_params": result.message_params,
        "cta_label_key": result.cta_label_key,
        "cta_action": result.cta_action,
        "secondary_label_key": result.secondary_label_key,
        "secondary_action": result.secondary_action,
        "skippable": result.skippable,
        "dismissible": result.dismissible,
        "step_key": result.step_key,
    })


def _store_read_snapshot(store: Any) -> Any:
    """A copy of ``store`` an executor job can read while the loop keeps writing.

    Loop-only. Shallow-copies the store object and gives it a ``_data`` whose
    containers are copies two levels deep: every top-level list and dict, and every
    dict inside them (each cycle, profile, envelope). The executor can then iterate
    what it reads while the loop appends a cycle, adds a profile or sets a key on a
    cycle (audit PERF-10, same class as register items 82 and 264). Large values
    (power traces, envelope arrays) are shared, not copied: the loop replaces them
    rather than mutating them in place. Cache attributes the store's methods set on
    ``self`` land on the copy and are dropped with it. Anything without a real
    ``_data`` dict (a test double) is returned as is.
    """
    data = getattr(store, "_data", None)
    if not isinstance(data, dict):
        return store

    def _copy(value: Any) -> Any:
        if isinstance(value, list):
            return [dict(v) if isinstance(v, dict) else v for v in value]
        if isinstance(value, dict):
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in value.items()}
        return value

    snapshot = copy.copy(store)
    snapshot._data = {key: _copy(value) for key, value in data.items()}  # pylint: disable=protected-access
    return snapshot


# ─── Profiles ─────────────────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_profiles", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_get_profiles(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return all profiles for a device."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    # Everything below reads the store from an executor thread while the loop keeps
    # appending cycles and rebuilding envelopes, and each block swallows its own
    # error - so "dictionary changed size during iteration" used to show up as a
    # silently empty health/trends/advisories panel (audit PERF-10). The executor
    # reads a snapshot taken here, on the loop; list_profiles is cached and stays
    # on the loop.
    profiles: list[dict[str, Any]] = []
    try:
        profiles = manager.profile_store.list_profiles()
        # #158: the panel shows a computed duration read-only (rows are copies).
        learned = manager.profile_store.learned_duration_profiles()
        for row in profiles:
            row["duration_learned"] = row.get("name") in learned
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error listing profiles for %s: %s", entry_id, exc)
    store = _store_read_snapshot(manager.profile_store)

    def _compute_stats() -> dict[str, Any]:
        health: dict[str, dict] = {}
        try:
            health = store.compute_profile_health()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        trends: dict[str, dict] = {}
        try:
            trends = store.compute_profile_trends()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # Coverage gaps: unlabelled recent cycles that look like one program the user
        # has not created. Measured by devtools/advisory_usefulness_eval.py: 92-95%
        # of its clusters are the missing program, and it stays silent on intact
        # stores, so the Profiles tab shows them (audit UI-13, register item 432).
        coverage_gaps: dict[str, Any] = {}
        try:
            coverage_gaps = store.suggest_coverage_gaps()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        advisories: list[dict] = []
        try:
            # Reuse the health/trends computed above instead of recomputing both.
            advisories = store.compute_profile_advisories(health, trends)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # How each programme ENDS, measured from its own cycles. Read-only: no
        # detection path consults it, and `consistency` is there to be read first,
        # because the event it describes is present in only some runs of the same
        # programme (register item 238).
        terminal: dict[str, Any] = {}
        try:
            for _profile in profiles:
                # `profiles` is list_profiles(): a list of profile DICTS, not names.
                # Passing the dict straight in compared it against each cycle's
                # `profile_name` string, which can never match, so the feature
                # returned {} for every device while raising nothing.
                _name = _profile.get("name") if isinstance(_profile, dict) else None
                if not _name:
                    continue
                _sig = store.compute_profile_terminal_signature(_name)
                if _sig is not None:
                    terminal[_name] = _sig
        except Exception:  # pylint: disable=broad-exception-caught
            terminal = {}

        # How often the matcher labelled one of this appliance's own cycles with
        # each program (STORE-21): the panel shows it on imported program cards so
        # an import that never fits can be pruned. Local only, no store writes.
        matcher_counts: dict[str, int] = {}
        try:
            matcher_counts = store.matcher_label_counts()
        except Exception:  # pylint: disable=broad-exception-caught
            matcher_counts = {}

        return {
            "profiles": profiles,
            "profile_health": health,
            "profile_trends": trends,
            "coverage_gaps": coverage_gaps,
            "profile_advisories": advisories,
            "profile_terminal": terminal,
            "profile_matcher_counts": matcher_counts,
        }

    stats = await hass.async_add_executor_job(_compute_stats)
    _send_result(connection, msg["id"], "get_profiles", stats)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/create_profile",
        vol.Required("entry_id"): str,
        vol.Required("name"): str,
        vol.Optional("reference_cycle"): vol.Any(str, None),
        vol.Optional("manual_duration_min"): vol.Any(vol.Coerce(float), None),
    }
)
@websocket_api.async_response
async def ws_create_profile(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Create a new profile, optionally seeded from a cycle."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    name = str(msg["name"]).strip()
    if not name:
        connection.send_error(msg["id"], "invalid_format", "Profile name must not be empty")
        return

    ref_cycle = msg.get("reference_cycle")
    manual_mins = msg.get("manual_duration_min")
    avg_duration = float(manual_mins) * 60.0 if manual_mins and float(manual_mins) > 0 else None

    try:
        await manager.profile_store.create_profile_standalone(
            name,
            ref_cycle if ref_cycle not in (None, "none", "") else None,
            avg_duration=avg_duration,
        )
        manager.notify_update()
        _send_result(connection, msg["id"], "create_profile", {"success": True, "name": name})
    except ValueError as exc:
        connection.send_error(msg["id"], "profile_exists", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/rename_profile",
        vol.Required("entry_id"): str,
        vol.Required("profile_name"): str,
        vol.Required("new_name"): str,
        vol.Optional("manual_duration_min"): vol.Any(vol.Coerce(float), None),
    }
)
@websocket_api.async_response
async def ws_rename_profile(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Rename a profile and optionally update its manual duration."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    new_name = str(msg["new_name"]).strip()
    if not new_name:
        connection.send_error(msg["id"], "invalid_format", "New name must not be empty")
        return

    manual_mins = msg.get("manual_duration_min")
    avg_duration = float(manual_mins) * 60.0 if manual_mins and float(manual_mins) > 0 else None

    try:
        await manager.profile_store.update_profile(
            msg["profile_name"], new_name, avg_duration=avg_duration
        )
        manager.notify_update()
        _send_result(connection, msg["id"], "rename_profile", {"success": True})
    except ValueError as exc:
        connection.send_error(msg["id"], "rename_failed", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/delete_profile",
        vol.Required("entry_id"): str,
        vol.Required("profile_name"): str,
        vol.Optional("unlabel_cycles", default=True): bool,
    }
)
@websocket_api.async_response
async def ws_delete_profile(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Delete a profile, optionally removing cycle labels."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.profile_store.delete_profile(
            msg["profile_name"], msg.get("unlabel_cycles", True)
        )
        manager.notify_update()
        _send_result(connection, msg["id"], "delete_profile", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# ─── Profile groups (Stage 5) ──────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_profile_groups", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_get_profile_groups(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return profile groups with member lists and cohesion. (Near-duplicate group
    suggestions were deleted in 0.5.8, register item 432.)"""
    from .const import GROUP_MIN_COHESION  # pylint: disable=import-outside-toplevel
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    store = manager.profile_store
    groups = []
    for name, g in store.get_profile_groups().items():
        members = list(g.get("members") or [])
        if len(members) >= 2:
            coh = await hass.async_add_executor_job(store.group_cohesion, members)
        else:
            coh = 1.0
        groups.append({
            "name": name,
            "members": members,
            "cohesion": round(coh, 3),
            "cohesive": coh >= GROUP_MIN_COHESION,  # False => not aggregated by matcher; UI warns
        })
    # Group suggestions were computed on every Profiles visit and rendered nowhere
    # (audit UI-13), so they are no longer sent.
    _send_result(connection, msg["id"], "get_profile_groups", {
        "groups": groups,
        "min_cohesion": GROUP_MIN_COHESION,
    })


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/save_profile_group",
        vol.Required("entry_id"): str,
        vol.Required("name"): str,
        vol.Required("members"): [str],
    }
)
@websocket_api.async_response
async def ws_save_profile_group(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Create a group or replace an existing group's members."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        store = manager.profile_store
        if msg["name"] in store.get_profile_groups():
            await store.set_profile_group_members(msg["name"], msg["members"])
        else:
            await store.create_profile_group(msg["name"], msg["members"])
        manager.notify_update()
        _send_result(connection, msg["id"], "save_profile_group", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/rename_profile_group",
        vol.Required("entry_id"): str,
        vol.Required("name"): str,
        vol.Required("new_name"): str,
    }
)
@websocket_api.async_response
async def ws_rename_profile_group(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        await manager.profile_store.rename_profile_group(msg["name"], msg["new_name"])
        manager.notify_update()
        _send_result(connection, msg["id"], "rename_profile_group", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/delete_profile_group",
        vol.Required("entry_id"): str,
        vol.Required("name"): str,
    }
)
@websocket_api.async_response
async def ws_delete_profile_group(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        await manager.profile_store.delete_profile_group(msg["name"])
        manager.notify_update()
        _send_result(connection, msg["id"], "delete_profile_group", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/rebuild_envelopes", vol.Required("entry_id"): str}
)
@callback
def ws_rebuild_envelopes(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Rebuild power-profile envelopes for all profiles.

    Runs as a detached, registry-tracked task that rebuilds one profile per step
    (each DTW pass already offloads to the executor): rebuilding every profile
    serially inside the WS request stalled low-power hosts for the whole run
    (issue #311). Progress/result come via the task registry."""
    entry_id: str = msg["entry_id"]
    if _get_manager(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    reg = task_registry.get_registry(hass)
    task = reg.create(
        entry_id, "rebuild", "Rebuilding envelopes",
        label_key="task.rebuild.envelopes", label_params={},
    )
    _raw = hass.async_create_task(_rebuild_envelopes_task(hass, task, entry_id))
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "rebuild_envelopes", {"task_id": task.id})


async def _rebuild_envelopes_task(hass: HomeAssistant, task: Any, entry_id: str) -> None:
    """Detached runner: rebuild every profile's envelope one at a time, reporting
    progress and honouring cancel, so the loop breathes between profiles."""
    reg = task_registry.get_registry(hass)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    store = manager.profile_store
    lock = _entry_write_lock(hass, entry_id)
    acquired = False
    try:
        await lock.acquire()
        acquired = True
        names = list(store.get_profiles().keys())
        reg.update(task, total=len(names), done=0)
        rebuilt = 0
        for i, name in enumerate(names):
            if task.cancel_requested:
                break
            try:
                if await store.async_rebuild_envelope(name):
                    rebuilt += 1
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("Envelope rebuild failed for %s/%s: %s", entry_id, name, exc)
            reg.update(task, done=i + 1)
        if _get_manager(hass, entry_id) is manager:
            manager.notify_update()
        reg.finish(
            task,
            state=task_registry.STATE_CANCELLED if task.cancel_requested else task_registry.STATE_DONE,
            result={"success": True, "rebuilt": rebuilt},
        )
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Rebuild-envelopes task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))
    finally:
        if acquired:
            lock.release()


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_profile_phases",
        vol.Required("entry_id"): str,
        vol.Required("profile_name"): str,
    }
)
@callback
def ws_get_profile_phases(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return phase ranges assigned to a profile."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    phases: list[dict[str, Any]] = []
    try:
        phases = manager.profile_store.get_profile_phase_ranges(msg["profile_name"])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error getting profile phases: %s", exc)

    _send_result(connection, msg["id"], "get_profile_phases", {"phases": phases or []})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/set_profile_phases",
        vol.Required("entry_id"): str,
        vol.Required("profile_name"): str,
        vol.Required("phases"): list,
    }
)
@websocket_api.async_response
async def ws_set_profile_phases(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Save phase ranges for a profile."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.profile_store.async_set_profile_phase_ranges(
            msg["profile_name"], msg["phases"]
        )
        _send_result(connection, msg["id"], "set_profile_phases", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# ─── Maintenance log (Group E) ──────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_maintenance_log", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_get_maintenance_log(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the maintenance log, due reminders, event types, and reminder config."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    device_type = manager.device_type
    # The reminders that actually apply (#461): the device type's preset until the
    # config is saved, the saved config after. Until #461 this merged the washer
    # defaults over the saved config, so a partial config showed rows the manager
    # never evaluated.
    reminders = effective_reminders(
        device_type, manager.config_entry.options.get(CONF_MAINTENANCE_REMINDER_CYCLES)
    )
    store = manager.profile_store
    status = store.get_maintenance_status(reminders)
    _send_result(connection, msg["id"], "get_maintenance_log", {
        "log": store.get_maintenance_log(),
        "due": [row["id"] for row in status if row["due"]],
        # The built-in types this device's editor offers (device-type aware).
        "event_types": editor_types(device_type, reminders),
        "reminders": reminders,
        # Cycles run since each task was last done, and the odometer they are
        # measured against (#414). Computed backend-side before this and never
        # sent, so the panel could show "due" but never "how close".
        "cycles_since": {
            evt: store.cycles_since_maintenance(evt) for evt in MAINTENANCE_EVENT_TYPES
        },
        "custom_tasks": store.get_maintenance_tasks(),
        # Every active reminder (built-in and custom) with its progress.
        "status": status,
        "lifetime_cycle_count": manager.lifetime_cycle_count,
        "limits": {
            "tasks_max": MAINTENANCE_CUSTOM_TASK_MAX,
            "name_max": MAINTENANCE_TASK_NAME_MAX,
            "cycles_max": MAINTENANCE_INTERVAL_CYCLES_MAX,
            "days_max": MAINTENANCE_INTERVAL_DAYS_MAX,
        },
    })


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/set_lifetime_cycle_count",
        vol.Required("entry_id"): str,
        vol.Required("count"): vol.All(vol.Coerce(int), vol.Range(min=0, max=1000000)),
    }
)
@websocket_api.async_response
async def ws_set_lifetime_cycle_count(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Correct the appliance's lifetime cycle odometer (requires full access).

    The one sanctioned way the count moves other than a cycle completing (#414).
    It exists because WashData cannot always tell a real short run from an
    artefact, and because an appliance may have run for years before WashData was
    installed - rather than asking "was that a real cycle?" on every deletion, the
    user corrects the total once. Recorded in the settings changelog.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        store = manager.profile_store
        count = int(msg["count"])
        # Serialize the whole read-check-mutate-save-rollback under the per-entry
        # write lock. The rollback below restores the value read at the start, so two
        # corrections that interleave across the save could see the earlier one's
        # failure undo the later one's success (A reads 10, B reads and writes 4000
        # successfully, A's save fails and A restores 10). Holding the lock across the
        # await is the point; taking it only around the mutation would not help.
        async with _entry_write_lock(hass, msg["entry_id"]):
            previous = store.get_lifetime_cycle_count()
            # A stored record is evidence of a run, so the odometer can never read below
            # the number of records on hand - that floor is what get_lifetime_cycle_count
            # applies. Without this check the setter accepted a lower value, the getter
            # masked it while the history was long, and it then surfaced later once
            # records were deleted: a correction that appeared to do nothing and took
            # effect retroactively, i.e. the odometer regression #414 exists to prevent.
            # Refused with the floor named rather than silently clamped, so the user is
            # told why their number was not taken.
            floor = len(store.get_past_cycles())
            if count < floor:
                connection.send_error(
                    msg["id"],
                    "invalid_format",
                    f"Cannot set the lifetime count below the {floor} cycle records "
                    f"currently stored; delete records first or choose {floor} or more",
                )
                return
            store.set_lifetime_cycle_count(count, force=True)
            try:
                # ONE save for both mutations: async_record_settings_changes persists the
                # store, and the new count is already staged in memory. Saving separately
                # beforehand meant a changelog failure reported unknown_error on a
                # correction that had in fact been written, and left `previous` reading
                # the new value on a retry - recording old == new.
                await store.async_record_settings_changes(
                    [{"key": "lifetime_cycle_count", "old": previous, "new": count}]
                )
            except Exception:
                # Undo only OUR write. A cycle completing during the await bumps the
                # same counter (manager._async_process_cycle_end -> lifetime energy
                # save), and that path deliberately does not take this lock: it lives
                # in the manager, on the hot cycle-end path, and reaching into the WS
                # layer's lock from there would invert the layering for a window this
                # narrow. Compare-and-swap keeps the rollback honest without it - if
                # the value is no longer what we wrote, someone else owns it now and
                # restoring `previous` would discard their increment.
                if store.get_lifetime_cycle_count() == count:
                    store.set_lifetime_cycle_count(previous, force=True)
                raise
        # Re-validate the manager is still live after the awaited save: a reload
        # during it detaches this manager (and its per-entry lock, which
        # async_unload_entry pops), so notifying it would target stale state and
        # reporting its store's count would show the panel a number from a store
        # nothing reads any more. Mirrors the guard the recording-persist and import
        # handlers use. Reported as success because the save itself did happen; the
        # count comes from whatever store is live now.
        current_manager = _get_manager(hass, entry_id)
        if current_manager is not manager:
            _LOGGER.warning(
                "Manager replaced during lifetime-count correction for %s; "
                "skipping notify", entry_id,
            )
        else:
            manager.notify_update()
        live_store = (
            current_manager.profile_store if current_manager is not None else store
        )
        _send_result(
            connection,
            msg["id"],
            "set_lifetime_cycle_count",
            {"success": True, "lifetime_cycle_count": live_store.get_lifetime_cycle_count()},
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/add_maintenance_event",
        vol.Required("entry_id"): str,
        vol.Required("event_type"): str,
        vol.Optional("date"): vol.Any(str, None),
        vol.Optional("notes"): str,
    }
)
@websocket_api.async_response
async def ws_add_maintenance_event(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Log a maintenance event (requires edit access via central RBAC guard)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        event = await manager.profile_store.async_add_maintenance_event(
            msg["event_type"], msg.get("date"), msg.get("notes", "")
        )
        manager.notify_update()
        _send_result(connection, msg["id"], "add_maintenance_event", {"success": True, "event": event})
    except ValueError as exc:
        connection.send_error(msg["id"], "invalid_format", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/delete_maintenance_event",
        vol.Required("entry_id"): str,
        vol.Required("event_id"): str,
    }
)
@websocket_api.async_response
async def ws_delete_maintenance_event(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Delete a maintenance event by id (requires edit access via central RBAC guard)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        removed = await manager.profile_store.async_delete_maintenance_event(msg["event_id"])
        if removed:
            manager.notify_update()
        _send_result(connection, msg["id"], "delete_maintenance_event", {"success": removed})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# Custom maintenance tasks (#461). Edit level, like the maintenance log: a task is
# a reminder the user defines, never a change to detection or stored cycles.
# Intervals are validated by the store (whole numbers, bounded) so the WS layer,
# the panel and any future service share one rule; a refusal is invalid_format.
_TASK_INTERVAL = vol.Any(None, int, float, str)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/add_maintenance_task",
        vol.Required("entry_id"): str,
        vol.Required("name"): str,
        vol.Optional("cycles"): _TASK_INTERVAL,
        vol.Optional("days"): _TASK_INTERVAL,
    }
)
@websocket_api.async_response
async def ws_add_maintenance_task(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Create a custom maintenance task (free-text name, cycles and/or days)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        task = await manager.profile_store.async_add_maintenance_task(
            msg["name"], msg.get("cycles"), msg.get("days")
        )
        manager.notify_update()
        _send_result(connection, msg["id"], "add_maintenance_task", {"success": True, "task": task})
    except ValueError as exc:
        connection.send_error(msg["id"], "invalid_format", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/update_maintenance_task",
        vol.Required("entry_id"): str,
        vol.Required("task_id"): str,
        vol.Optional("name"): str,
        vol.Optional("cycles"): _TASK_INTERVAL,
        vol.Optional("days"): _TASK_INTERVAL,
    }
)
@websocket_api.async_response
async def ws_update_maintenance_task(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Rename a custom maintenance task or change its intervals.

    An omitted field is left as it is; an explicit ``null`` interval switches
    that axis off, the same as 0.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    def _interval(key: str) -> Any:
        # The store reads None as "leave it"; on the wire null means "off".
        if key not in msg:
            return None
        return 0 if msg[key] is None else msg[key]

    try:
        task = await manager.profile_store.async_update_maintenance_task(
            msg["task_id"],
            name=msg.get("name"),
            cycles=_interval("cycles"),
            days=_interval("days"),
        )
        manager.notify_update()
        _send_result(connection, msg["id"], "update_maintenance_task", {"success": True, "task": task})
    except ValueError as exc:
        connection.send_error(msg["id"], "invalid_format", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/delete_maintenance_task",
        vol.Required("entry_id"): str,
        vol.Required("task_id"): str,
    }
)
@websocket_api.async_response
async def ws_delete_maintenance_task(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Remove a custom maintenance task. Its log entries stay, with their name."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        removed = await manager.profile_store.async_delete_maintenance_task(msg["task_id"])
        if removed:
            manager.notify_update()
        _send_result(connection, msg["id"], "delete_maintenance_task", {"success": removed})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# ─── Cycles ───────────────────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/label_cycle",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Optional("profile_name"): vol.Any(str, None),
        vol.Optional("new_profile_name"): vol.Any(str, None),
    }
)
@websocket_api.async_response
async def ws_label_cycle(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Assign (or remove) a profile label from a cycle.

    profile_name=None removes the label.
    profile_name='__create_new__' + new_profile_name creates and assigns.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    cycle_id: str = msg["cycle_id"]
    profile_name: str | None = msg.get("profile_name")
    new_profile_name: str | None = msg.get("new_profile_name")

    try:
        if profile_name == "__create_new__":
            if not new_profile_name or not new_profile_name.strip():
                connection.send_error(msg["id"], "invalid_format", "New profile name required")
                return
            applied_profile: str | None = new_profile_name.strip()
            await manager.profile_store.create_profile(applied_profile, cycle_id)
        else:
            applied_profile = profile_name
            await manager.profile_store.assign_profile_to_cycle(cycle_id, profile_name)
        # Manually (re)labelling a cycle awaiting verification IS the user's answer,
        # so resolve any pending feedback and drop it from the review queue (#331).
        if hasattr(manager, "learning_manager"):
            await manager.learning_manager.async_resolve_pending_from_label(
                cycle_id, applied_profile
            )
        manager.notify_update()
        _send_result(connection, msg["id"], "label_cycle", {"success": True})
    except ValueError as exc:
        connection.send_error(msg["id"], "label_failed", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/delete_cycle",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
    }
)
@websocket_api.async_response
async def ws_delete_cycle(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Delete a single cycle."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.profile_store.delete_cycle(msg["cycle_id"])
        manager.notify_update()
        _send_result(connection, msg["id"], "delete_cycle", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# The panel's bulk auto-label threshold range (the modal input's min/max).
AUTO_LABEL_MIN_THRESHOLD = 0.5
AUTO_LABEL_MAX_THRESHOLD = 0.95


def configured_auto_label_threshold(entry: ConfigEntry | None) -> float:
    """The device's Auto-Label Confidence, clamped to the bulk pass's range.

    The bulk pass and the cycle-end label both put a guess on a cycle with no
    confirmation, so they share the user's one bar for that. Never raises.
    """
    from .const import (  # pylint: disable=import-outside-toplevel
        CONF_AUTO_LABEL_CONFIDENCE,
        DEFAULT_AUTO_LABEL_CONFIDENCE,
    )

    merged = {**(entry.data if entry else {}), **(entry.options if entry else {})}
    try:
        value = float(merged.get(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE))
    except (TypeError, ValueError, OverflowError):
        value = DEFAULT_AUTO_LABEL_CONFIDENCE
    if not math.isfinite(value):
        value = DEFAULT_AUTO_LABEL_CONFIDENCE
    return min(max(value, AUTO_LABEL_MIN_THRESHOLD), AUTO_LABEL_MAX_THRESHOLD)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/auto_label_cycles",
        vol.Required("entry_id"): str,
        vol.Optional("confidence_threshold"): vol.All(
            vol.Coerce(float),
            vol.Range(min=AUTO_LABEL_MIN_THRESHOLD, max=AUTO_LABEL_MAX_THRESHOLD),
        ),
    }
)
@websocket_api.async_response
async def ws_auto_label_cycles(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Auto-label all cycles with matched profiles above the confidence threshold.

    Without a threshold the device's own Auto-Label Confidence applies (audit
    UI-10: the panel used to send a fixed 0.75, below the 0.9 default).
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    threshold = msg.get("confidence_threshold")
    if threshold is None:
        threshold = configured_auto_label_threshold(_get_entry(hass, entry_id))
    task, _raw = start_auto_label_task(hass, entry_id, float(threshold))
    _send_result(connection, msg["id"], "auto_label_cycles", {"task_id": task.id})


def start_auto_label_task(
    hass: HomeAssistant, entry_id: str, threshold: float
) -> tuple[Any, asyncio.Task[Any] | None]:
    """Start the one auto-label runner (panel and service) as a registry task.

    It re-matches every unlabelled stored cycle - 56 s for 201 cycles - so it runs
    detached with progress and cancel, under the per-entry write lock that
    reprocess, merge and imports hold (audit PLATFORM-05). Never overwrites (audit
    MANAGER-01 / UI-10).
    """
    reg = task_registry.get_registry(hass)
    task = reg.create(
        entry_id, "auto_label", "Auto-labelling cycles", label_key="task.auto_label.running",
    )
    raw = hass.async_create_task(_auto_label_task(hass, task, entry_id, threshold))
    if raw is not None:
        reg.link_asyncio_task(task.id, raw)
    return task, raw


async def _auto_label_task(
    hass: HomeAssistant, task: Any, entry_id: str, threshold: float
) -> None:
    reg = task_registry.get_registry(hass)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    try:
        async with _entry_write_lock(hass, entry_id):
            if _get_manager(hass, entry_id) is not manager:
                reg.finish(task, state=task_registry.STATE_ERROR, error="device reloaded")
                return
            stats = await manager.profile_store.auto_label_cycles(
                threshold,
                overwrite=False,
                progress=lambda done, total: reg.update(task, done=done, total=total),
                should_cancel=lambda: task.cancel_requested,
            )
        manager.notify_update()
        reg.finish(
            task,
            state=task_registry.STATE_CANCELLED if stats.get("cancelled")
            else task_registry.STATE_DONE,
            result=stats,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))


# ─── Phase catalog ────────────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_phase_catalog",
        vol.Required("entry_id"): str,
        vol.Optional("device_type"): vol.Any(str, None),
    }
)
@callback
def ws_get_phase_catalog(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return phase catalog for a device type (or all types)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    device_type: str | None = msg.get("device_type") or getattr(manager, "device_type", None)
    phases: list[dict[str, Any]] = []
    try:
        phases = manager.profile_store.list_phase_catalog(device_type or "")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error listing phase catalog: %s", exc)

    _send_result(connection, msg["id"], "get_phase_catalog", {"phases": phases, "device_type": device_type})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/create_phase",
        vol.Required("entry_id"): str,
        # Optional since #450: an omitted or empty device_type resolves to
        # ``manager.device_type``, the exact value ``ws_get_phase_catalog``
        # lists against, so create and list can never disagree about scope.
        vol.Optional("device_type", default=""): str,
        vol.Required("name"): str,
        vol.Optional("description", default=""): str,
    }
)
@websocket_api.async_response
async def ws_create_phase(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Create a custom phase in the catalog."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    # #450: a panel served by an entry whose device_type lives only in
    # ``entry.data`` used to send the ``'washing_machine'`` fallback here, so the
    # phase was stored under a scope the catalog never lists - invisible, yet
    # still tripping the duplicate check on the next attempt.
    device_type = str(msg.get("device_type") or "").strip() or str(
        getattr(manager, "device_type", "") or ""
    ).strip()
    # ...and refuse rather than store under an empty scope. `manager.device_type`
    # reads `options.get(CONF_DEVICE_TYPE, data.get(..., DEFAULT))`, and `.get`
    # hands back a persisted empty string or null verbatim instead of the default
    # - the #389 class that `strip_null_options` exists for. An empty scope is the
    # worst outcome available here, not a harmless one: `list_phase_catalog` never
    # lists it, so the phase is invisible, yet it still trips the duplicate check
    # on the next attempt, so the user cannot create it again either. That is
    # exactly the #450 symptom the comment above describes, reached from the
    # entry's own options rather than from the panel's payload.
    if not device_type:
        connection.send_error(
            msg["id"],
            "invalid_device_type",
            "This entry has no usable device type, so a phase created now could "
            "not be listed again. Re-save the device settings and retry.",
        )
        return

    try:
        await manager.profile_store.async_create_custom_phase(
            device_type, msg["name"], msg.get("description", "")
        )
        _send_result(connection, msg["id"], "create_phase", {"success": True})
    except ValueError as exc:
        connection.send_error(msg["id"], "duplicate_phase", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/update_phase",
        vol.Required("entry_id"): str,
        vol.Required("phase_id"): str,
        vol.Required("new_name"): str,
        vol.Optional("description", default=""): str,
    }
)
@websocket_api.async_response
async def ws_update_phase(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Rename/update a phase in the catalog."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.profile_store.async_update_custom_phase(
            msg["phase_id"], msg["new_name"], msg.get("description", "")
        )
        _send_result(connection, msg["id"], "update_phase", {"success": True})
    except ValueError as exc:
        connection.send_error(msg["id"], "phase_not_found", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/delete_phase",
        vol.Required("entry_id"): str,
        vol.Required("phase_id"): str,
    }
)
@websocket_api.async_response
async def ws_delete_phase(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Delete a custom phase from the catalog."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.profile_store.async_delete_custom_phase(msg["phase_id"])
        _send_result(connection, msg["id"], "delete_phase", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# ─── Recording ────────────────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_recording_state", vol.Required("entry_id"): str}
)
@callback
def ws_get_recording_state(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return current recording state for a device."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    recorder = getattr(manager, "recorder", None)
    if recorder is None:
        _send_result(connection, msg["id"], "get_recording_state", {"state": "unavailable"})
        return

    is_recording: bool = getattr(recorder, "is_recording", False)
    last_run: dict[str, Any] | None = getattr(recorder, "last_run", None)

    info: dict[str, Any] = {"state": "recording" if is_recording else "idle"}

    if is_recording:
        info["duration_s"] = int(getattr(recorder, "current_duration", 0))
        buf = getattr(recorder, "_buffer", [])
        info["sample_count"] = len(buf)
    elif last_run:
        info["state"] = "stopped"
        info["sample_count"] = len(last_run.get("data", []))
        info["start_time"] = last_run.get("start_time")
        info["end_time"] = last_run.get("end_time")
        try:
            start_str = last_run.get("start_time")
            end_str = last_run.get("end_time")
            if start_str and end_str:
                start = dt_util.parse_datetime(start_str)
                end = dt_util.parse_datetime(end_str)
                if start and end:
                    info["duration_s"] = int((end - start).total_seconds())
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    _send_result(connection, msg["id"], "get_recording_state", info)


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/start_recording", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_start_recording(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Start manual recording mode."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.async_start_recording()
        _send_result(connection, msg["id"], "start_recording", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/stop_recording", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_stop_recording(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Stop recording mode."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.async_stop_recording()
        _send_result(connection, msg["id"], "stop_recording", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/process_recording",
        vol.Required("entry_id"): str,
        vol.Required("profile_name"): str,
        vol.Required("save_mode"): vol.In(["new_profile", "existing_profile"]),
        vol.Optional("head_trim", default=0.0): vol.Coerce(float),
        vol.Optional("tail_trim", default=0.0): vol.Coerce(float),
    }
)
@websocket_api.async_response
async def ws_process_recording(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Trim and save a completed recording to a profile."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    recorder = getattr(manager, "recorder", None)
    if not recorder:
        connection.send_error(msg["id"], "no_recording", "No completed recording to process")
        return

    head_trim: float = msg.get("head_trim", 0.0)
    tail_trim: float = msg.get("tail_trim", 0.0)
    profile_name: str = msg["profile_name"].strip()
    save_mode: str = msg["save_mode"]

    if not profile_name:
        connection.send_error(msg["id"], "invalid_format", "Profile name must not be empty")
        return

    # Serialize the whole claim+persist under the per-entry write lock so two
    # concurrent process_recording calls cannot both consume the same recording
    # (which would double-persist and collide on cycle IDs). The claim is the
    # read of ``last_run`` *inside* the lock; it is cleared only after a
    # successful persist, so a failure leaves the recording intact for retry.
    async with _entry_write_lock(hass, entry_id):
        last_run = getattr(recorder, "last_run", None)
        if not last_run:
            connection.send_error(msg["id"], "no_recording", "No completed recording to process")
            return
        data = last_run.get("data", [])

        try:
            rec_start_str = last_run.get("start_time")
            rec_end_str = last_run.get("end_time")

            parsed: list[tuple[float, float]] = []
            for item in data:
                t_str, p = (item[0], item[1]) if isinstance(item, (list, tuple)) else (None, None)
                if t_str:
                    t = dt_util.parse_datetime(str(t_str))
                    if t:
                        parsed.append((t.timestamp(), float(p or 0)))

            data_start_ts = parsed[0][0] if parsed else 0.0
            data_end_ts = parsed[-1][0] if parsed else 0.0

            if rec_start_str:
                parsed_dt = dt_util.parse_datetime(rec_start_str)
                if parsed_dt is None:
                    connection.send_error(msg["id"], "invalid_format", "Invalid recording start_time format")
                    return
                start_ts = parsed_dt.timestamp()
            else:
                start_ts = data_start_ts

            if rec_end_str:
                parsed_dt = dt_util.parse_datetime(rec_end_str)
                if parsed_dt is None:
                    connection.send_error(msg["id"], "invalid_format", "Invalid recording end_time format")
                    return
                end_ts = parsed_dt.timestamp()
            else:
                end_ts = data_end_ts

            if parsed:
                start_ts = min(start_ts, data_start_ts)
                end_ts = max(end_ts, data_end_ts)

            keep_start = start_ts + head_trim
            keep_end = end_ts - tail_trim
            duration = keep_end - keep_start

            trimmed_data = [
                (dt_util.utc_from_timestamp(t).isoformat(), p)
                for t, p in parsed
                if keep_start <= t <= keep_end
            ]

            # Reject a trim that removes everything before it can persist a
            # corrupt cycle: an inverted/empty window (end <= start) or a window
            # that keeps no samples.
            if keep_end <= keep_start:
                connection.send_error(
                    msg["id"], "invalid_format",
                    "Trim removes the entire recording (end is at or before start)",
                )
                return
            if not trimmed_data:
                connection.send_error(
                    msg["id"], "invalid_format",
                    "Trim leaves no power data",
                )
                return

            cycle_data: dict[str, Any] = {
                # High-entropy ID so two recordings processed in the same second
                # (or a retry) never collide on an existing cycle ID.
                "id": f"rec_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                "start_time": dt_util.utc_from_timestamp(keep_start).isoformat(),
                "end_time": dt_util.utc_from_timestamp(keep_end).isoformat(),
                "duration": duration,
                "profile_name": profile_name,
                "power_data": trimmed_data,
                "status": "completed",
                "meta": {"source": "recorder", "original_samples": len(data)},
                # A recorded cycle is a hand-picked clean example -> it IS the golden
                # reference. Prefill the single ml_review flag (recorded == golden;
                # no separate "recorded" field) so it seeds matching immediately.
                "ml_review": {
                    "golden": True,
                    "quality": "good",
                    "reviewed_at": dt_util.now().isoformat(),
                },
            }

            if save_mode == "new_profile":
                await manager.profile_store.create_profile_standalone(profile_name)

            await manager.profile_store.async_add_cycle(cycle_data)
            await manager.profile_store.async_rebuild_envelope(profile_name)
            await manager.profile_store.async_save()
            # Clear only after a successful persist so the recording is consumed
            # exactly once; a failure above keeps it available for another try.
            await recorder.clear_last_run()

            # Re-validate the manager is still live after the awaited persist chain
            # (create/add/rebuild/save/clear): if the entry was reloaded meanwhile
            # the original manager is detached and its notify would target stale
            # state. Mirror the ws_reprocess_history guard.
            current_manager = _get_manager(hass, entry_id)
            if current_manager is not manager:
                _LOGGER.warning(
                    "Manager replaced during recording persist for %s; skipping notify",
                    entry_id,
                )
                _send_result(connection, msg["id"], "process_recording", {"success": True})
                return
            manager.notify_update()

            _send_result(connection, msg["id"], "process_recording", {"success": True})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/discard_recording", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_discard_recording(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Discard the last completed recording."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        recorder = getattr(manager, "recorder", None)
        if recorder:
            await recorder.clear_last_run()
        _send_result(connection, msg["id"], "discard_recording", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# ─── Learning feedbacks ───────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_feedbacks", vol.Required("entry_id"): str}
)
@callback
def ws_get_feedbacks(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return all pending learning feedbacks."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    feedbacks: list[dict[str, Any]] = []
    try:
        pending: dict[str, Any] = manager.profile_store.get_pending_feedback()
        feedbacks = sorted(
            [{"cycle_id": cid, **item} for cid, item in pending.items()],
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error fetching feedbacks for %s: %s", entry_id, exc)

    _send_result(connection, msg["id"], "get_feedbacks", {"feedbacks": feedbacks})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/resolve_feedback",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Required("action"): vol.In(["confirm", "correct", "ignore", "delete"]),
        vol.Optional("corrected_profile"): vol.Any(str, None),
        vol.Optional("corrected_duration_min"): vol.Any(vol.Coerce(float), None),
    }
)
@websocket_api.async_response
async def ws_resolve_feedback(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Resolve a pending learning feedback."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    cycle_id: str = msg["cycle_id"]
    action: str = msg["action"]

    try:
        if action == "delete":
            await manager.profile_store.delete_cycle(cycle_id)
        elif hasattr(manager, "learning_manager"):
            corrected_duration_min = msg.get("corrected_duration_min")
            corrected_duration_s = (
                int(float(corrected_duration_min) * 60)
                if corrected_duration_min is not None
                else None
            )
            await manager.learning_manager.async_submit_cycle_feedback(
                cycle_id=cycle_id,
                user_confirmed=(action == "confirm"),
                corrected_profile=msg.get("corrected_profile") if action == "correct" else None,
                corrected_duration=corrected_duration_s,
                dismiss=(action == "ignore"),
            )
        else:
            connection.send_error(msg["id"], "not_available", "Learning manager not available")
            return
        manager.notify_update()
        _send_result(connection, msg["id"], "resolve_feedback", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/dismiss_all_feedbacks", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_dismiss_all_feedbacks(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Dismiss all pending learning feedbacks."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        pending: dict[str, Any] = manager.profile_store.get_pending_feedback()
        if pending and hasattr(manager, "learning_manager"):
            cycle_ids = list(pending.keys())
            for cid in cycle_ids:
                await manager.learning_manager.async_submit_cycle_feedback(
                    cycle_id=cid,
                    user_confirmed=False,
                    corrected_profile=None,
                    corrected_duration=None,
                    dismiss=True,
                )
        manager.notify_update()
        _send_result(connection, msg["id"], "dismiss_all_feedbacks", {"success": True, "dismissed": len(pending)})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# ─── Diagnostics ──────────────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_diagnostics", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_get_diagnostics(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return storage statistics for a device."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        stats = await manager.profile_store.get_storage_stats()
        # The restore point of the last replace import rides along on the fetch that
        # already fills the Export / Import card, so the panel needs no second call.
        undo = await manager.profile_store.async_get_pre_import_snapshot()
        _send_result(
            connection, msg["id"], "get_diagnostics", {"stats": stats, "import_undo": undo}
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


async def _reprocess_task(hass: HomeAssistant, task: Any, entry_id: str) -> None:
    """Detached runner for the full "Process history" pass, reporting phase-level
    progress to the task registry and storing the summary as the result. Survives
    a dropped socket; the panel re-attaches via the registry."""
    reg = task_registry.get_registry(hass)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    store = manager.profile_store
    summary: dict[str, Any] = {"success": True}
    # Serialize under the per-entry write lock: this multi-await pass rematches,
    # retrains and rewrites the store, and must not interleave with a concurrent
    # import / recording persist for the same entry.
    lock = _entry_write_lock(hass, entry_id)
    acquired = False
    try:
        await lock.acquire()
        acquired = True
        if task.cancel_requested:
            reg.finish(task, state=task_registry.STATE_CANCELLED)
            return

        reg.update(task, total=6, done=0, label="Reprocessing: matching cycles",
                   label_key="task.reprocess.matching")
        summary["count"] = await store.async_reprocess_all_data()

        if task.cancel_requested:
            reg.finish(task, state=task_registry.STATE_CANCELLED, result=summary)
            return

        reg.update(task, done=1, label="Reprocessing: backfilling golden",
                   label_key="task.reprocess.golden")
        try:
            summary["golden_backfilled"] = await store.async_backfill_recorded_golden()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("golden backfill failed for %s: %s", entry_id, exc)

        if task.cancel_requested:
            reg.finish(task, state=task_registry.STATE_CANCELLED, result=summary)
            return

        reg.update(task, done=2, label="Reprocessing: suggestions",
                   label_key="task.reprocess.suggestions")
        learning = getattr(manager, "learning_manager", None)
        if learning is not None and hasattr(learning, "async_run_full_analysis"):
            try:
                res = await learning.async_run_full_analysis()
                summary["suggestions"] = (res or {}).get("count", 0)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("suggestion analysis failed for %s: %s", entry_id, exc)

        reg.update(task, done=3, label="Reprocessing: ML training",
                   label_key="task.reprocess.ml_training")
        if ENABLE_ML_TRAINING and not task.cancel_requested:
            try:
                tr = await manager.async_run_ml_training(force=True)
                summary["ml_training"] = {
                    "ok": bool(tr.get("ok")),
                    "promoted": tr.get("promoted", []),
                    "reason": tr.get("reason"),
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("ML training failed for %s: %s", entry_id, exc)

        if task.cancel_requested:
            reg.finish(task, state=task_registry.STATE_CANCELLED, result=summary)
            return

        reg.update(task, done=4, label="Reprocessing: energy costs",
                   label_key="task.reprocess.costs")
        # Recost stored cycles against the recorder's price history (#426). Cheap
        # and a no-op unless a dynamic price entity is configured, so it runs
        # unconditionally rather than behind yet another switch.
        try:
            summary["costs_recomputed"] = await manager.async_recompute_cycle_costs()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("cost recompute failed for %s: %s", entry_id, exc)

        if task.cancel_requested:
            reg.finish(task, state=task_registry.STATE_CANCELLED, result=summary)
            return

        reg.update(task, done=5, label="Reprocessing: cycle health",
                   label_key="task.reprocess.health")
        # Recompute per-cycle health against the (possibly retrained) model.
        # Skip when training already recomputed it (a promotion refreshes health).
        if not (summary.get("ml_training", {}) or {}).get("promoted"):
            try:
                summary["health_recomputed"] = await manager.async_recompute_cycle_health()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("health recompute failed for %s: %s", entry_id, exc)

        reg.update(task, done=6)
        # Re-validate the manager is still live after a long chain of awaits; if the
        # entry was reloaded, the original manager is detached and must not notify.
        if _get_manager(hass, entry_id) is manager:
            manager.notify_update()
        reg.finish(task, state=task_registry.STATE_DONE, result=summary)
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Task-level failure: log at WARNING so it is visible in the default HA log
        # and the panel Logs view (sub-step failures above stay at debug on purpose).
        _LOGGER.warning("Reprocess task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))
    finally:
        if acquired:
            lock.release()


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/reprocess_history", vol.Required("entry_id"): str}
)
@callback
def ws_reprocess_history(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Kick off the full "Process history" pass as a detached, registry-tracked
    task; returns its id immediately. Runs, in order: reprocess (rematch + rebuild
    envelopes) -> backfill golden -> refresh suggestions -> on-device ML training
    (when enabled) -> recost cycles from recorder price history -> recompute cycle
    health. Progress + result via the registry."""
    entry_id: str = msg["entry_id"]
    if _get_manager(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    reg = task_registry.get_registry(hass)
    task = reg.create(entry_id, "reprocess", "Reprocessing")
    _raw = hass.async_create_task(_reprocess_task(hass, task, entry_id))
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "reprocess_history", {"task_id": task.id})


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/clear_debug_data", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_clear_debug_data(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Clear stored debug traces."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        count = await manager.profile_store.async_clear_debug_data()
        _send_result(connection, msg["id"], "clear_debug_data", {"success": True, "count": count})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/wipe_history", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_wipe_history(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Wipe all cycles and profiles (destructive)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.profile_store.clear_all_data()
        # clear_all_data pops the persisted arm, but the manager's in-memory field is
        # the authoritative one, so the pin has to be retired there too or the next
        # cycle re-applies a program from before the wipe.
        manager.clear_armed_program()
        manager.notify_update()
        _send_result(connection, msg["id"], "wipe_history", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


def _export_json(payload: dict[str, Any]) -> str:
    """Serialise an export payload compactly, the way the store is written to disk.

    ``json.dumps(indent=2)`` put every number of every power trace on its own
    indented line, and now that no cycle is dropped (item 463) that is one
    multi-megabyte WebSocket frame per click (audit PERF-11). Home Assistant's
    orjson encoder is the one ``Store`` saves with, so anything persisted
    serialises the same way, and ``import_config``'s ``json.loads`` reads it back
    unchanged. It also runs as one C call, so the event loop cannot mutate the
    shallow-copied store under it mid-encode the way it could between the
    pure-Python encoder's steps.
    """
    from homeassistant.helpers.json import json_dumps  # pylint: disable=import-outside-toplevel

    return json_dumps(payload)


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/export_config", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_export_config(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Export profiles and cycles as a JSON string."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    entry = _get_entry(hass, entry_id)
    try:
        payload = manager.profile_store.export_data(
            entry_data=dict(entry.data) if entry else {},
            entry_options=dict(entry.options) if entry else {},
        )
        # Offload serialization to executor - power traces can be megabytes
        json_str = await hass.async_add_executor_job(_export_json, payload)
        _send_result(connection, msg["id"], "export_config", {"json_data": json_str})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


async def async_apply_imported_entry_options(
    hass: HomeAssistant,
    entry: ConfigEntry,
    config_updates: dict[str, Any],
    source: str,
) -> None:
    """Apply an import payload's tunables to ``entry.options``, the one safe way.

    Shared by the ``import_config`` WS command and the ``import_config`` service,
    which had drifted: the service still rebound the device to the EXPORTER's
    power and door sensors (the item-317 "integration silently goes dead"
    failure), skipped the options lock and wrote no settings changelog
    (audit PLATFORM-02).

    Identity never lands in options and local entity/device bindings are dropped:
    an import carries the exporter's ids. ``config_updates["entry_data"]`` is
    deliberately NOT written to ``entry.data`` - it is the source device's raw,
    un-redacted identity; identity changes go through the reconfigure flow.
    The caller holds the per-entry write lock; order is always write -> options.
    """
    entry_options_updates = dict(config_updates.get("entry_options", {}) or {})
    for key in _IMPORT_LOCAL_BINDING_KEYS:
        entry_options_updates.pop(key, None)
    if not entry_options_updates:
        return
    async with _entry_options_lock(hass, entry.entry_id):
        # Read INSIDE the lock: a snapshot taken before the `async with` suspended
        # would silently revert a ws_set_options that committed while we waited.
        # A persisted null survives options.get(key, DEFAULT) and breaks setup
        # (#389), so the same write-boundary strip as ws_set_options applies.
        entry_options_updates = _without_inverted_threshold_pair(
            entry, entry_options_updates, source
        )
        new_options = strip_null_options({**entry.options, **entry_options_updates})
        # ...and a non-numeric numeric setting is dropped, same reason (PLATFORM-13).
        new_options, _ = drop_invalid_numeric_options(new_options)
        await _record_option_changes(hass, entry, entry_options_updates, source)
        hass.config_entries.async_update_entry(entry, options=new_options)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/import_config",
        vol.Required("entry_id"): str,
        vol.Required("json_data"): str,
    }
)
@websocket_api.async_response
async def ws_import_config(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Import profiles and cycles from a JSON string."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    # Serialize the whole import under the per-entry write lock so it cannot
    # interleave with a concurrent reprocess / recording persist (which would
    # corrupt the store the import is rewriting).
    async with _entry_write_lock(hass, entry_id):
        try:
            payload = await hass.async_add_executor_job(json.loads, msg["json_data"])
            pre_entry = _get_entry(hass, entry_id)
            config_updates = await manager.profile_store.async_import_data(
                payload,
                # Kept with the restore point so an undo puts them back (item 195).
                entry_options=dict(pre_entry.options) if pre_entry else None,
            )
            result = {
                "success": True,
                "restore_point_saved": bool(config_updates.get("restore_point_saved")),
            }

            # Re-validate after the awaits — the entry may have been reloaded
            # (a new manager + store) during the import. Persisting through the
            # detached manager/entry would clobber the live one, so re-fetch both
            # and bail out if the manager is no longer the live one.
            current_manager = _get_manager(hass, entry_id)
            if current_manager is not manager:
                _LOGGER.warning(
                    "Manager replaced during import for %s; aborting notify", entry_id
                )
                _send_result(connection, msg["id"], "import_config", result)
                return

            entry = _get_entry(hass, entry_id)
            if entry and config_updates:
                await async_apply_imported_entry_options(
                    hass, entry, config_updates, "import_config"
                )

            # An old payload re-arms the one-time banked-tail repair, and the
            # options write above is the only thing here that could reload the
            # entry - so an import that carries no options would otherwise leave
            # the imported tails inflating avg_duration until the next restart.
            manager.async_schedule_banked_tail_repair()
            manager.notify_update()
            _send_result(connection, msg["id"], "import_config", result)
        except json.JSONDecodeError as exc:
            connection.send_error(msg["id"], "invalid_json", str(exc))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            connection.send_error(msg["id"], "unknown_error", str(exc))


# Soft cap on a pasted/uploaded import blob so a huge string can't wedge the
# executor / event loop. Whole-store exports with full traces are a few MB; 64 MB
# is comfortably above any legitimate payload.
_MAX_IMPORT_JSON_BYTES = 64 * 1024 * 1024


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_export_inventory", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_get_export_inventory(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return this device's per-category data inventory for the export wizard tree."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    entry = _get_entry(hass, entry_id)
    try:
        opts = dict(entry.options) if entry else {}
        # Run the inventory scan synchronously on the event loop. It's a fast walk
        # (counts + small per-cycle {id,date,duration} dicts) and staying on the loop
        # is inherently safe from concurrent mutation - offloading it to an executor
        # would read the live, mutable store lists from another thread and could raise
        # "list changed size during iteration" without a full (expensive) snapshot.
        manifest = manager.profile_store.get_export_inventory(opts)
        _send_result(connection, msg["id"], "get_export_inventory", {"manifest": manifest})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/analyze_import",
        vol.Required("entry_id"): str,
        vol.Required("json_data"): str,
    }
)
@websocket_api.async_response
async def ws_analyze_import(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Parse a pasted/uploaded export and describe what it contains (no mutation).

    Synchronous (not a detached task): parsing + walking a few-MB file is tens of ms.
    Returns ``{"manifest": {...}}``; a structural problem is reported inside the
    manifest as ``manifest.error`` so the panel can render it inline.
    """
    from .const import CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE

    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    json_data: str = msg["json_data"]
    if len(json_data.encode("utf-8", "ignore")) > _MAX_IMPORT_JSON_BYTES:
        connection.send_error(msg["id"], "invalid_json", "Import payload is too large")
        return
    entry = _get_entry(hass, entry_id)
    device_type = ""
    if entry is not None:
        device_type = entry.options.get(
            CONF_DEVICE_TYPE, entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)
        )
    try:
        from .profile_store import build_import_manifest

        local_names = list(manager.profile_store.get_profiles().keys())

        def _analyze() -> dict[str, Any]:
            payload = json.loads(json_data)
            return build_import_manifest(
                payload, local_device_type=device_type, local_profile_names=local_names
            )

        manifest = await hass.async_add_executor_job(_analyze)
        _send_result(connection, msg["id"], "analyze_import", {"manifest": manifest})
    except json.JSONDecodeError as exc:
        connection.send_error(msg["id"], "invalid_json", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/export_config_selective",
        vol.Required("entry_id"): str,
        vol.Required("selection"): dict,
    }
)
@websocket_api.async_response
async def ws_export_config_selective(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Export only the selected categories/items as a JSON string."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    entry = _get_entry(hass, entry_id)
    selection = msg["selection"]
    try:
        payload = manager.profile_store.export_data(
            entry_data=dict(entry.data) if entry else {},
            entry_options=dict(entry.options) if entry else {},
            selection=selection,
        )
        json_str = await hass.async_add_executor_job(_export_json, payload)
        _send_result(
            connection, msg["id"], "export_config_selective", {"json_data": json_str}
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/import_config_selective",
        vol.Required("entry_id"): str,
        vol.Required("json_data"): str,
        vol.Required("selection"): dict,
        vol.Optional("mode", default="merge"): vol.In(["merge", "replace"]),
        vol.Optional("conflict_resolutions", default=dict): dict,
        vol.Optional("cycle_destination", default="reference"): vol.In(
            ["reference", "real_history"]
        ),
        vol.Optional("apply_settings", default=True): bool,
    }
)
@websocket_api.async_response
async def ws_import_config_selective(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Selectively import chosen categories/items, merging into existing data."""
    from .const import CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE, sanitize_shared_settings

    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    json_data: str = msg["json_data"]
    if len(json_data.encode("utf-8", "ignore")) > _MAX_IMPORT_JSON_BYTES:
        connection.send_error(msg["id"], "invalid_json", "Import payload is too large")
        return

    # Serialize under the per-entry write lock so the import can't interleave with a
    # concurrent reprocess / recording persist (which would corrupt the store).
    async with _entry_write_lock(hass, entry_id):
        try:
            entry = _get_entry(hass, entry_id)
            device_type = ""
            if entry is not None:
                device_type = entry.options.get(
                    CONF_DEVICE_TYPE, entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)
                )
            payload = await hass.async_add_executor_job(json.loads, json_data)
            summary = await manager.profile_store.async_import_data_selective(
                payload,
                selection=msg["selection"],
                mode=msg["mode"],
                conflict_resolutions=msg["conflict_resolutions"],
                cycle_destination=msg["cycle_destination"],
                apply_settings=msg["apply_settings"],
                local_device_type=device_type,
                # Kept with the restore point of a destructive import so an undo puts
                # them back (item 195); never applied to the entry from here.
                entry_options=dict(entry.options) if entry is not None else None,
            )

            # Re-validate after the awaits — the entry may have been reloaded (a new
            # manager + store) during the import. Persisting through the detached
            # manager/entry would clobber the live one, so bail out.
            current_manager = _get_manager(hass, entry_id)
            if current_manager is not manager:
                _LOGGER.warning(
                    "Manager replaced during selective import for %s; aborting notify",
                    entry_id,
                )
                _send_result(
                    connection, msg["id"], "import_config_selective",
                    {"success": True, "summary": summary},
                )
                return

            # Apply the imported settings subset onto this device's options. Only the
            # SHAREABLE numeric allow-list ever reaches here (the store already filters),
            # and identity keys are stripped for defense-in-depth. entry.data is never
            # written (no power_sensor/device_type hijack).
            entry = _get_entry(hass, entry_id)
            settings = summary.get("settings") if isinstance(summary.get("settings"), dict) else {}
            settings_applied = 0
            if entry is not None and settings:
                filtered = sanitize_shared_settings(settings)
                for key in _OPTIONS_IDENTITY_KEYS:
                    filtered.pop(key, None)
                if filtered:
                    # Nested inside the write lock this handler already holds;
                    # order is always write -> options, so no deadlock.
                    async with _entry_options_lock(hass, entry_id):
                        filtered = _without_inverted_threshold_pair(
                            entry, filtered, "store_device_package"
                        )  # item 515
                        await _record_option_changes(
                            hass, entry, filtered, "store_device_package"
                        )
                        hass.config_entries.async_update_entry(
                            entry, options={**entry.options, **filtered}
                        )
                    settings_applied = len(filtered)
            summary = {**summary, "settings_applied": settings_applied}

            # Same as the wholesale path: `apply_settings=False` (and any import
            # whose settings subset comes back empty) skips the options write
            # above, so nothing reloads the entry and the re-armed repair would
            # wait for a restart.
            manager.async_schedule_banked_tail_repair()
            manager.notify_update()
            _send_result(
                connection, msg["id"], "import_config_selective",
                {"success": True, "summary": summary},
            )
        except json.JSONDecodeError as exc:
            connection.send_error(msg["id"], "invalid_json", str(exc))
        except ValueError as exc:
            connection.send_error(msg["id"], "invalid_format", str(exc))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            connection.send_error(msg["id"], "unknown_error", str(exc))


async def _async_restore_entry_options(
    hass: HomeAssistant, entry: ConfigEntry, snapshot_options: dict[str, Any]
) -> None:
    """Put ``entry.options`` back to what they were before the undone import.

    Exactly the snapshot's options, so a key the import added goes away again, with
    one exception: this device's own entity/device bindings keep their CURRENT
    values. The import never wrote them (``_IMPORT_LOCAL_BINDING_KEYS``), so any
    change to them since was the user's own, and reverting it could re-point the
    device at a sensor it no longer uses. Same write-boundary hygiene as the import.
    """
    async with _entry_options_lock(hass, entry.entry_id):
        current = dict(entry.options)
        restored = {
            k: v for k, v in snapshot_options.items() if k not in _IMPORT_LOCAL_BINDING_KEYS
        }
        restored.update({k: current[k] for k in _IMPORT_LOCAL_BINDING_KEYS if k in current})
        restored = strip_null_options(restored)
        restored, _ = drop_invalid_numeric_options(restored)
        if restored == current:
            return
        await _record_option_changes(
            hass, entry, {k: v for k, v in restored.items() if current.get(k) != v},
            "undo_import",
        )
        hass.config_entries.async_update_entry(entry, options=restored)


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/undo_import", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_undo_import(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Undo the last replace import: restore the store and options saved before it.

    The restore point is consumed, so the panel's undo affordance disappears once
    this succeeds. No restore point is a clean ``not_found`` error (register item 195).
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    # A restore rewrites the whole store, so it takes the same per-entry write lock
    # as the imports and cannot interleave with a reprocess / recording persist.
    async with _entry_write_lock(hass, entry_id):
        try:
            restored = await manager.profile_store.async_restore_pre_import_snapshot()
            result = {
                "success": True,
                "summary": {
                    "restored_from": restored.get("restored_from"),
                    "counts": restored.get("counts") or {},
                },
            }
            # Re-validate after the awaits, as the imports do: the entry may have
            # been reloaded and the options belong to the live one.
            if _get_manager(hass, entry_id) is not manager:
                _LOGGER.warning(
                    "Manager replaced during import undo for %s; aborting notify", entry_id
                )
                _send_result(connection, msg["id"], "undo_import", result)
                return
            entry = _get_entry(hass, entry_id)
            snapshot_options = restored.get("entry_options")
            if entry is not None and isinstance(snapshot_options, dict):
                await _async_restore_entry_options(hass, entry, snapshot_options)
            manager.async_schedule_banked_tail_repair()
            manager.notify_update()
            _send_result(connection, msg["id"], "undo_import", result)
        except ValueError as exc:
            connection.send_error(msg["id"], "not_found", str(exc))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            connection.send_error(msg["id"], "unknown_error", str(exc))


# ─── Shared constants ─────────────────────────────────────────────────────────

@websocket_api.websocket_command({vol.Required("type"): "ha_washdata/get_constants"})
@callback
def ws_get_constants(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return shared display constants so the panel hardcodes nothing.

    Device-type and state labels are localised client-side via hass.localize
    against the integration translations; the values/colors here are the
    canonical fallback and the single source for state colors.
    """
    device_types = [
        {"id": key, "label": label}
        for key, label in DEVICE_TYPES.items()
    ]
    from .frontend import BRAND_ICON_URL as _BRAND_ICON_URL, BRAND_ICON_REGISTERED_KEY as _BRAND_ICON_KEY  # pylint: disable=import-outside-toplevel
    from .const import STORE_WEB_ORIGIN
    from . import store_account
    _send_result(connection, msg["id"], "get_constants", {
            "version": hass.data.get("ha_washdata_version", _INTEGRATION_VERSION),
            "icon_url": _BRAND_ICON_URL if hass.data.get(_BRAND_ICON_KEY) else None,
            "device_types": device_types,
            "state_colors": dict(STATE_COLORS),
            "ml_lab_enabled": SHOW_ML_LAB,
            "ml_training_available": ENABLE_ML_TRAINING,
            "PROFILE_MIN_WARMUP_CYCLES": CONF_PROFILE_MIN_WARMUP_CYCLES,
            # Canonical matcher defaults for the Playground's matcher-param fields, so
            # the panel's _PG_MATCH_DEFAULTS table cannot silently drift from const.py.
            # The panel keeps that table only as an offline fallback. Single source:
            # playground.MATCH_DEFAULTS_BY_OPTION (also used by effective_settings).
            "pg_match_defaults": dict(playground.MATCH_DEFAULTS_BY_OPTION),
            # Community store: the panel opens <origin>/connect.html for the GitHub
            # handoff and validates postMessage against new URL(origin).origin.
            "store_online_available": True,
            # Online features are integration-wide (device-agnostic), set in the gear menu.
            "store_online_enabled": store_account.online_enabled(hass),
            "store_web_origin": STORE_WEB_ORIGIN,
            # Community-store display/behaviour preferences (declarative; see
            # store_account._DEFAULT_PREFS + the panel's _STORE_PREFS list).
            "store_prefs": store_account.get_prefs(hass),
        },
    )


# ─── Suggestions ──────────────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_suggestions", vol.Required("entry_id"): str}
)
@callback
def ws_get_suggestions(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return applicable tuning suggestions with current vs suggested values."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    entry = _get_entry(hass, entry_id)
    merged: dict[str, Any] = {**entry.data, **entry.options} if entry else {}

    out: list[dict[str, Any]] = []
    try:
        for key, item, suggested, current in _visible_suggestions(
            manager.profile_store, merged,
            getattr(manager, "device_type", None) or DEFAULT_DEVICE_TYPE,
        ):
            out.append(
                {
                    "key": key,
                    "suggested": suggested,
                    "reason": item.get("reason", ""),
                    # Localization sidecars: panel renders _t(reason_key,
                    # reason_params, reason). Absent on old/reconciled entries.
                    "reason_key": item.get("reason_key"),
                    "reason_params": item.get("reason_params"),
                    # Structured excluded-cycle summary; the panel renders it as a
                    # localized note appended to the reason. Absent/empty on most.
                    "exclusions": item.get("exclusions"),
                    "current": current,
                    "updated": item.get("updated"),
                }
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error reading suggestions for %s: %s", entry_id, exc)

    try:
        locked = manager.profile_store.get_locked_suggestions()
    except Exception:  # pylint: disable=broad-exception-caught
        locked = []
    _send_result(
        connection, msg["id"], "get_suggestions",
        {"suggestions": out, "locked_suggestions": locked},
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/apply_suggestions",
        vol.Required("entry_id"): str,
        vol.Required("keys"): [str],
    }
)
@websocket_api.async_response
async def ws_apply_suggestions(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Stage selected suggested values into options, then clear suggestions."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    entry = _get_entry(hass, entry_id)
    if not entry:
        connection.send_error(msg["id"], "not_found", f"Entry {entry_id!r} not found")
        return

    try:
        merged = {**entry.data, **entry.options}
        # Only what the Settings list would show: a muted key re-created by the
        # reconcile cascade used to be applied here (audit SUGGEST-11).
        visible = {
            key: item
            for key, item, _s, _c in _visible_suggestions(
                manager.profile_store, merged,
                getattr(manager, "device_type", None) or DEFAULT_DEVICE_TYPE,
            )
        }
        updates: dict[str, Any] = {}
        for key in msg["keys"]:
            item = visible.get(key)
            if not isinstance(item, dict) or item.get("value") is None:
                continue
            val = item["value"]
            updates[key] = (
                int(float(val)) if key in _SUGGESTION_INT_KEYS else float(val)
            )

        # Item 515: the reconciler keeps a stored start/stop pair ordered, but
        # applying one of the two alone (a subset, a muted cascade, or a pair
        # reconciled against options that changed since) could still invert it.
        _inverted = inverted_threshold_pair(
            merged,
            {**merged, **updates},
            getattr(manager, "device_type", None) or DEFAULT_DEVICE_TYPE,
        )
        if _inverted is not None:
            connection.send_error(
                msg["id"], "invalid_threshold_pair", _threshold_pair_message(*_inverted)
            )
            return

        if updates:
            # Clear before updating the entry: async_update_entry schedules a
            # reload that rebuilds the store, so persist the cleared state first.
            # The odometer, like the cooldown check (audit SUGGEST-05): at the
            # retention cap len(past_cycles) never grows past the stamp again.
            cycle_count = manager.profile_store.get_lifetime_cycle_count()
            manager.profile_store.set_suggestion_apply_cycle_count(cycle_count)
            # Same critical section as ws_set_options (#442 follow-up): the
            # changelog snapshot and the options write must be atomic per entry.
            async with _entry_options_lock(hass, entry_id):
                # Record BEFORE clear_suggestions/async_update_entry: both persist,
                # and the reload the latter schedules rebuilds the store (#442).
                await _record_option_changes(hass, entry, updates, "apply_suggestions")
                await manager.profile_store.clear_suggestions()
                # Suggested values are all tunables -> layer them onto the existing
                # options; never spread entry.data into options.
                new_options = {**entry.options, **updates}
                hass.config_entries.async_update_entry(entry, options=new_options)
            manager.notify_update()

        _send_result(connection, msg["id"], "apply_suggestions", {"success": True, "applied": list(updates.keys())}
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/clear_suggestions", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_clear_suggestions(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Discard all pending tuning suggestions for a device."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.profile_store.clear_suggestions()
        manager.notify_update()
        _send_result(connection, msg["id"], "clear_suggestions", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/set_suggestion_lock",
        vol.Required("entry_id"): str,
        vol.Required("key"): str,
        vol.Required("locked"): bool,
    }
)
@websocket_api.async_response
async def ws_set_suggestion_lock(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Lock or unlock a tuning suggestion key so the auto-tuner stops (or resumes)
    proposing it (#343)."""
    entry_id: str = msg["entry_id"]
    key: str = msg["key"]
    # Only LOCKING needs a key the engine can suggest. Unlocking must accept any
    # stored key: a mute on a setting that 0.5.8 stopped suggesting (the confidence
    # thresholds, sampling interval, ...) could otherwise never be cleared, and
    # "Reset muted" failed on it forever.
    if msg["locked"] and key not in _SUGGESTION_KEYS:
        connection.send_error(msg["id"], "invalid_key", f"Unknown suggestion key: {key!r}")
        return
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        await manager.profile_store.set_suggestion_locked(key, bool(msg["locked"]))
        manager.notify_update()
        _send_result(
            connection, msg["id"], "set_suggestion_lock",
            {"success": True, "locked_suggestions": manager.profile_store.get_locked_suggestions()},
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/run_suggestion_analysis", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_run_suggestion_analysis(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Run all suggestion passes on demand (manual 'analyze now' trigger)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    learning = getattr(manager, "learning_manager", None)
    if learning is None or not hasattr(learning, "async_run_full_analysis"):
        connection.send_error(msg["id"], "unavailable", "Suggestion analysis unavailable")
        return
    try:
        result = await learning.async_run_full_analysis()
        manager.notify_update()
        _send_result(connection, msg["id"], "run_suggestion_analysis", {"success": True, **(result or {})})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# ─── Cycle curve / interactive editing ─────────────────────────────────────────

@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_cycle_power_data",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
    }
)
@websocket_api.async_response
async def ws_get_cycle_power_data(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return a single cycle's downsampled power curve plus its metadata."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    cycle_id: str = msg["cycle_id"]
    samples: list[Any] = []
    meta: dict[str, Any] = {}
    try:
        store = manager.profile_store
        samples = store.get_cycle_power_data(cycle_id)
        cycle, origin = store.find_stored_cycle(cycle_id)
        if cycle:
            meta = {
                "start_time": cycle.get("start_time"),
                "end_time": cycle.get("end_time"),
                "duration": cycle.get("duration"),
                "profile_name": cycle.get("profile_name"),
                "status": cycle.get("status"),
                "energy_kwh": _cycle_kwh(cycle),
                **_cycle_capabilities(cycle, origin),
            }
            # Transient artifacts (door-open pauses, out-of-band dips/spikes) for
            # graph markers. Prefer the value frozen at cycle end; compute on the
            # fly for older cycles that predate artifact storage.
            artifacts = cycle.get("artifacts")
            if artifacts is None and cycle.get("profile_name") and samples:
                # Offload CPU-intensive NumPy work to executor thread
                artifacts = await hass.async_add_executor_job(
                    store.detect_cycle_artifacts, cycle["profile_name"], samples
                )
            # Only with a trace to draw them on (#459).
            meta["artifacts"] = (artifacts or []) if samples else []
            # HA restart gaps recorded during this cycle (for panel shading).
            meta["restart_gaps"] = cycle.get("restart_gaps") or []
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error getting cycle power data %s: %s", cycle_id, exc)

    _ds = _downsample(samples)
    # The matched profile's expected curve, projected onto THIS cycle's time axis
    # through the same alignment the artifact/conformance comparison used, so the
    # overlay, the trace and the artifact shading all share one axis. Computed on
    # the full trace but emitted at the thinned x values the panel plots.
    if meta.get("profile_name") and len(_ds) >= 4 and len(samples) >= 4:
        try:
            meta["expected"] = await hass.async_add_executor_job(
                manager.profile_store.expected_curve_for_cycle,
                meta["profile_name"],
                samples,
                [float(p[0]) for p in _ds],
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Expected curve for cycle %s failed: %s", cycle_id, exc)
    _send_result(connection, msg["id"], "get_cycle_power_data", {
            "cycle_id": cycle_id,
            "samples": _ds,
            # Declare thinning (#395) so a gap from decimation is not mistaken for
            # a gap from a sensor that stopped reporting: the panel can show
            # "N of M samples" and disambiguate the two.
            "sample_count": len(samples),
            "decimated": len(_ds) < len(samples),
            "full_duration_s": round(float(samples[-1][0]), 1) if samples else 0.0,
            **meta,
        },
    )


# Cycle context (register item 513, discussion #463): the power sensor's recorder
# history just before and just after a stored cycle, for the cycle chart to draw
# greyed either side of the stored trace. The cap bounds the two recorder queries.
_CYCLE_CONTEXT_MAX_S = 3600.0
_CYCLE_CONTEXT_MAX_POINTS = 300


def _cycle_context_points(
    rows: list[tuple[float, float | None]],
    origin_ts: float,
    lo_ts: float,
    hi_ts: float,
    *,
    hi_inclusive: bool,
    max_points: int = _CYCLE_CONTEXT_MAX_POINTS,
) -> list[list[float | None]]:
    """Recorder rows inside a window as ``[offset_s, watts]`` from ``origin_ts``.

    ``watts`` is None at an unavailable (or non-finite) row, so the panel breaks the
    line there rather than holding the last reading across an outage. Each run
    between two such rows is thinned on its own (``_downsample`` keeps a run's
    extrema), so thinning never bridges a gap.
    """
    inside = [
        (ts, w) for ts, w in rows
        if lo_ts <= ts and (ts <= hi_ts if hi_inclusive else ts < hi_ts)
    ]
    total = sum(1 for _ts, w in inside if w is not None) or 1
    out: list[list[float | None]] = []
    run: list[tuple[float, float]] = []

    def _flush() -> None:
        if run:
            budget = max(4, round(max_points * len(run) / total))
            out.extend(_downsample([(ts - origin_ts, w) for ts, w in run], budget))
            run.clear()

    for ts, w in inside:
        if w is None or not math.isfinite(float(w)):
            _flush()
            if not out or out[-1][1] is not None:
                out.append([round(ts - origin_ts, 2), None])
            continue
        run.append((ts, float(w)))
    _flush()
    return out


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_cycle_context",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Optional("before_s", default=600.0): vol.Coerce(float),
        vol.Optional("after_s", default=600.0): vol.Coerce(float),
    }
)
@websocket_api.async_response
async def ws_get_cycle_context(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """The power sensor's recorder history around one stored cycle (item 513).

    ``before`` covers ``[start - before_s, start)`` and ``after`` covers
    ``[trace end, trace end + after_s]`` (clamped to now), both as
    ``[offset_s, watts]`` from the cycle's stored start: the axis the cycle chart
    already plots, so the panel draws them either side of the stored trace.
    Display only: it reads the recorder and the cycle's start and trace end, writes
    nothing, and nothing it returns reaches duration, energy, matching or envelopes.

    Only for a cycle this device observed live (``past``): a community-store
    reference was recorded on someone else's machine and a backfill came from an
    imported history, so this plug's recorder at that time says nothing about
    either. ``available`` is False while the recorder holds no reading for either
    window (it keeps 10 days by default; an excluded entity has none), and
    ``reason`` says why.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    def _window(key: str) -> float:
        value = float(msg.get(key) or 0.0)
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(_CYCLE_CONTEXT_MAX_S, value))

    cycle_id: str = msg["cycle_id"]
    before_s, after_s = _window("before_s"), _window("after_s")
    out: dict[str, Any] = {
        "cycle_id": cycle_id,
        "available": False,
        "reason": None,
        "entity_id": None,
        "before_s": before_s,
        "after_s": after_s,
        "trace_end_s": 0.0,
        "after_end_s": 0.0,
        "before": [],
        "after": [],
    }
    try:
        store = manager.profile_store
        cycle, origin = store.find_stored_cycle(cycle_id)
        entity_id = getattr(manager, "power_sensor_entity_id", None)
        start_raw = cycle.get("start_time") if cycle else None
        start_dt = dt_util.parse_datetime(str(start_raw)) if start_raw else None
        if cycle is None:
            out["reason"] = "not_found"
        elif origin != "past":
            out["reason"] = "not_live"
        elif not entity_id:
            out["reason"] = "no_sensor"
        elif start_dt is None:
            out["reason"] = "no_start"
        else:
            out["entity_id"] = entity_id
            start_dt = dt_util.as_utc(start_dt)
            start_ts = start_dt.timestamp()
            samples = store.get_cycle_power_data(cycle_id)
            try:
                trace_end = float(samples[-1][0]) if samples else float(cycle.get("duration") or 0.0)
            except (TypeError, ValueError, OverflowError):
                trace_end = 0.0
            trace_end = max(0.0, trace_end) if math.isfinite(trace_end) else 0.0
            out["trace_end_s"] = round(trace_end, 2)
            if before_s > 0:
                lo = start_ts - before_s
                rows = await _recorder_power(
                    hass, entity_id, start_dt - timedelta(seconds=before_s),
                    end_dt=start_dt, keep_unavailable=True,
                )
                out["before"] = _cycle_context_points(
                    rows, start_ts, lo, start_ts, hi_inclusive=False
                )
            end_dt = start_dt + timedelta(seconds=trace_end)
            after_end_dt = min(end_dt + timedelta(seconds=after_s), dt_util.utcnow())
            if after_s > 0 and after_end_dt > end_dt:
                rows = await _recorder_power(
                    hass, entity_id, end_dt, end_dt=after_end_dt, keep_unavailable=True,
                )
                out["after"] = _cycle_context_points(
                    rows, start_ts, end_dt.timestamp(), after_end_dt.timestamp(),
                    hi_inclusive=True,
                )
                out["after_end_s"] = round(after_end_dt.timestamp() - start_ts, 2)
            out["available"] = any(p[1] is not None for p in out["before"] + out["after"])
            if not out["available"]:
                out["reason"] = "no_history"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error building cycle context for %s: %s", cycle_id, exc)
        out.update(available=False, reason="no_history", before=[], after=[])

    _send_result(connection, msg["id"], "get_cycle_context", out)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/trim_cycle",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Required("start_s"): vol.Coerce(float),
        vol.Required("end_s"): vol.Coerce(float),
    }
)
@callback
def ws_trim_cycle(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Trim a cycle's power data to the [start_s, end_s] offset window.

    Runs as a detached, registry-tracked task (returns a task_id immediately): the
    trim recomputes the signature and rebuilds the profile envelope, which stalls
    low-power hosts if held on the event loop for the whole WS request (issue #311).
    Progress/result come via subscribe_tasks/get_task_result."""
    entry_id: str = msg["entry_id"]
    if _get_manager(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    reg = task_registry.get_registry(hass)
    task = reg.create(
        entry_id, "trim", "Trimming cycle", label_key="task.trim.apply", label_params={},
    )
    _raw = hass.async_create_task(_trim_task(
        hass, task, entry_id, msg["cycle_id"], float(msg["start_s"]), float(msg["end_s"]),
    ))
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "trim_cycle", {"task_id": task.id})


async def _trim_task(
    hass: HomeAssistant, task: Any, entry_id: str,
    cycle_id: str, start_s: float, end_s: float,
) -> None:
    """Detached runner for a cycle trim (recompute + single-profile envelope
    rebuild). Serialized under the per-entry write lock like reprocess."""
    reg = task_registry.get_registry(hass)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    store = manager.profile_store
    lock = _entry_write_lock(hass, entry_id)
    acquired = False
    try:
        await lock.acquire()
        acquired = True
        if task.cancel_requested:
            reg.finish(task, state=task_registry.STATE_CANCELLED)
            return
        reg.update(task, total=1, done=0)
        ok = await store.trim_cycle_power_data(cycle_id, start_s, end_s)
        reg.update(task, done=1)
        if not ok:
            reg.finish(task, state=task_registry.STATE_ERROR, error="trim_failed")
            return
        if _get_manager(hass, entry_id) is manager:
            manager.notify_update()
        reg.finish(task, state=task_registry.STATE_DONE, result={"success": True})
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Trim task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))
    finally:
        if acquired:
            lock.release()


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/analyze_split",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Optional("gap_seconds", default=900): vol.All(
            int, vol.Range(min=30, max=21600)
        ),
    }
)
@websocket_api.async_response
async def ws_analyze_split(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Auto-detect split boundaries for a cycle; return the curve and offsets."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    store = manager.profile_store
    cycle_id: str = msg["cycle_id"]
    gap: int = msg.get("gap_seconds", 900)
    try:
        cycle = next(
            (c for c in store.get_past_cycles() if c.get("id") == cycle_id), None
        )
        if not cycle:
            connection.send_error(msg["id"], "not_found", f"Cycle {cycle_id!r} not found")
            return

        segs = await hass.async_add_executor_job(
            store.analyze_split_sync, cycle, gap, 2.0
        )
        samples = store.get_cycle_power_data(cycle_id)
        split_offsets = (
            [round(float(s[1]), 1) for s in segs[:-1]] if segs and len(segs) > 1 else []
        )
        _ds = _downsample(samples)
        _send_result(connection, msg["id"], "analyze_split", {
                "segments": [
                    [round(float(a), 1), round(float(b), 1)] for a, b in (segs or [])
                ],
                "split_offsets": split_offsets,
                "samples": _ds,
                # Declare thinning (#395); see ws_get_cycle_power_data.
                "sample_count": len(samples),
                "decimated": len(_ds) < len(samples),
                "full_duration_s": round(float(samples[-1][0]), 1) if samples else 0.0,
            },
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/apply_split",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Required("split_offsets"): [vol.Coerce(float)],
        vol.Optional("segment_profiles"): list,
    }
)
@callback
def ws_apply_split(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Split a cycle at the given offsets, optionally labeling each segment.

    Cheap validation (cycle exists, at least two segments) runs synchronously so a
    bad request fails fast; the heavy apply (per-segment extraction + affected
    envelope rebuilds + save) then runs as a detached, registry-tracked task so it
    never holds the event loop for the whole request on low-power hosts (issue
    #311). Progress/result come via subscribe_tasks/get_task_result."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    store = manager.profile_store
    cycle_id: str = msg["cycle_id"]
    cycle = next(
        (c for c in store.get_past_cycles() if c.get("id") == cycle_id), None
    )
    if not cycle:
        connection.send_error(msg["id"], "not_found", f"Cycle {cycle_id!r} not found")
        return

    offsets = [float(o) for o in msg["split_offsets"]]
    seg_bounds = store.build_split_segments_from_offsets(cycle, offsets)
    if len(seg_bounds) < 2:
        connection.send_error(
            msg["id"],
            "split_failed",
            "Split points did not produce at least two segments",
        )
        return

    profiles = msg.get("segment_profiles") or []
    segments: list[dict[str, Any]] = []
    for i, (seg_start, seg_end) in enumerate(seg_bounds):
        prof = profiles[i] if i < len(profiles) else None
        if prof in ("", "none", "__none__"):
            prof = None
        segments.append(
            {"start": float(seg_start), "end": float(seg_end), "profile": prof}
        )

    reg = task_registry.get_registry(hass)
    task = reg.create(
        entry_id, "split", "Splitting cycle", label_key="task.split.apply", label_params={},
    )
    _raw = hass.async_create_task(_apply_split_task(hass, task, entry_id, cycle_id, segments))
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "apply_split", {"task_id": task.id})


async def _apply_split_task(
    hass: HomeAssistant, task: Any, entry_id: str,
    cycle_id: str, segments: list[dict[str, Any]],
) -> None:
    """Detached runner for a cycle split. Rebuilds only the affected profiles'
    envelopes (done inside apply_split_interactive). Serialized under the per-entry
    write lock like reprocess."""
    reg = task_registry.get_registry(hass)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    store = manager.profile_store
    lock = _entry_write_lock(hass, entry_id)
    acquired = False
    try:
        await lock.acquire()
        acquired = True
        if task.cancel_requested:
            reg.finish(task, state=task_registry.STATE_CANCELLED)
            return
        reg.update(task, total=1, done=0)
        new_ids = await store.apply_split_interactive(cycle_id, segments)
        reg.update(task, done=1)
        if _get_manager(hass, entry_id) is manager:
            manager.notify_update()
        reg.finish(
            task, state=task_registry.STATE_DONE,
            result={"success": True, "new_ids": new_ids},
        )
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Apply-split task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))
    finally:
        if acquired:
            lock.release()


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/apply_merge",
        vol.Required("entry_id"): str,
        vol.Required("cycle_ids"): [str],
        vol.Optional("target_profile"): vol.Any(str, None),
        vol.Optional("new_profile_name"): vol.Any(str, None),
    }
)
@callback
def ws_apply_merge(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Merge two or more cycles into one, optionally labeling the result.

    Cheap validation runs synchronously; the merge (gap-fill + signature recompute
    + affected-profile envelope rebuilds + save) then runs as a detached,
    registry-tracked task so it never holds the event loop for the whole request
    on low-power hosts (issue #311). Progress/result via the task registry."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    ids: list[str] = msg["cycle_ids"]
    if len(ids) < 2:
        connection.send_error(
            msg["id"], "merge_failed", "Select at least two cycles to merge"
        )
        return

    target = msg.get("target_profile")
    new_name: str | None = None
    if target == "__create_new__":
        new_name = (msg.get("new_profile_name") or "").strip()
        if not new_name:
            connection.send_error(msg["id"], "invalid_format", "New profile name required")
            return
    elif target in ("", "none", "__none__"):
        target = None

    reg = task_registry.get_registry(hass)
    task = reg.create(
        entry_id, "merge", "Merging cycles", label_key="task.merge.apply", label_params={},
    )
    _raw = hass.async_create_task(_apply_merge_task(hass, task, entry_id, ids, target, new_name))
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "apply_merge", {"task_id": task.id})


async def _apply_merge_task(
    hass: HomeAssistant, task: Any, entry_id: str,
    ids: list[str], target: str | None, new_name: str | None,
) -> None:
    """Detached runner for a cycle merge. Rebuilds only the affected profiles'
    envelopes (done inside apply_merge_interactive). Serialized under the per-entry
    write lock like reprocess."""
    reg = task_registry.get_registry(hass)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    store = manager.profile_store
    lock = _entry_write_lock(hass, entry_id)
    acquired = False
    try:
        await lock.acquire()
        acquired = True
        if task.cancel_requested:
            reg.finish(task, state=task_registry.STATE_CANCELLED)
            return
        reg.update(task, total=1, done=0)
        created_new = False
        if new_name:
            # Raises ValueError if the name already exists, so reaching the next
            # line guarantees we created a brand-new (empty) profile that must be
            # rolled back if the merge below fails.
            await store.create_profile_standalone(new_name)
            created_new = True
            target = new_name
        new_id = await store.apply_merge_interactive(ids, target)
        reg.update(task, done=1)
        if not new_id:
            if created_new and new_name:
                # Merge rejected the cycle set (stale/invalid id, unparseable start
                # time, ...). Delete the empty profile we just created so a failed
                # merge doesn't orphan a cycle-less profile in the store.
                await store.delete_profile(new_name, unlabel_cycles=False)
            reg.finish(task, state=task_registry.STATE_ERROR, error="merge_failed")
            return
        if _get_manager(hass, entry_id) is manager:
            manager.notify_update()
        reg.finish(
            task, state=task_registry.STATE_DONE,
            result={"success": True, "new_id": new_id},
        )
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Apply-merge task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))
    finally:
        if acquired:
            lock.release()


# ─── Profile envelope / member cycles ──────────────────────────────────────────

@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_profile_envelope",
        vol.Required("entry_id"): str,
        vol.Required("profile_name"): str,
    }
)
@callback
def ws_get_profile_envelope(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return a profile's averaged power envelope (downsampled) and stats."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    env_out: dict[str, Any] | None = None
    try:
        env = manager.profile_store.get_envelope(msg["profile_name"])
        if env:
            env_out = {
                "avg": _downsample(env.get("avg") or []),
                "min": _downsample(env.get("min") or []),
                "max": _downsample(env.get("max") or []),
                "target_duration": env.get("target_duration"),
                "avg_energy": env.get("avg_energy"),
                "duration_std_dev": env.get("duration_std_dev"),
                "cycle_count": env.get("cycle_count"),
            }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error getting envelope for %s: %s", msg.get("profile_name"), exc)

    _send_result(connection, msg["id"], "get_profile_envelope", {"envelope": env_out})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_profile_cycles",
        vol.Required("entry_id"): str,
        vol.Required("profile_name"): str,
        vol.Optional("limit", default=150): vol.All(int, vol.Range(min=1, max=400)),
    }
)
@websocket_api.async_response
async def ws_get_profile_cycles(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return cycles labeled with a profile, each with a downsampled curve.

    Powers the history-cleanup spaghetti view: the panel overlays every curve
    and lets the user delete outliers. Colors are assigned client-side.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    profile_name: str = msg["profile_name"]
    limit: int = msg.get("limit", 150)

    def _collect() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            store = manager.profile_store
            matched = [
                c for c in store.get_past_cycles() if c.get("profile_name") == profile_name
            ]
            for c in matched[-limit:]:
                cid = c.get("id")
                samples = store.get_cycle_power_data(cid) if cid else []
                out.append(
                    {
                        "cycle_id": cid,
                        "start_time": c.get("start_time"),
                        "duration": c.get("duration"),
                        "status": c.get("status"),
                        "energy_kwh": _cycle_kwh(c),
                        "samples": _downsample(samples, 160),
                    }
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Error getting profile cycles for %s: %s", profile_name, exc)
        return out

    result = await hass.async_add_executor_job(_collect)
    _send_result(connection, msg["id"], "get_profile_cycles", {"cycles": result})


# ─── Panel config + RBAC commands ──────────────────────────────────────────────

@websocket_api.websocket_command({vol.Required("type"): "ha_washdata/get_panel_config"})
@websocket_api.async_response
async def ws_get_panel_config(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return panel settings + the caller's prefs; admins also get RBAC + user list."""
    user = getattr(connection, "user", None)
    cfg = _panel_data(hass)
    is_admin = bool(getattr(user, "is_admin", False))
    uid = getattr(user, "id", "") or ""
    out: dict[str, Any] = {
        "panel": dict(cfg.get("panel", {})),
        "is_admin": is_admin,
        "user": {"id": uid, "name": getattr(user, "name", None)},
        "prefs": dict((cfg.get("prefs") or {}).get(uid, {})),
    }
    if is_admin:
        rbac = cfg.get("rbac", {})
        out["rbac"] = {
            "enabled": bool(rbac.get("enabled", False)),
            "default_level": rbac.get("default_level", "none"),
            "users": {
                k: {"default": v.get("default", "none"), "devices": dict(v.get("devices") or {})}
                for k, v in (rbac.get("users") or {}).items()
            },
        }
        users: list[dict[str, Any]] = []
        try:
            for u in await hass.auth.async_get_users():
                if u.system_generated or not u.is_active:
                    continue
                users.append({"id": u.id, "name": u.name or "Unnamed user", "is_admin": bool(u.is_admin)})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Could not list users: %s", exc)
        out["users"] = users
    _send_result(connection, msg["id"], "get_panel_config", out)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/set_panel_config",
        vol.Optional("panel"): dict,
        vol.Optional("rbac"): dict,
    }
)
@websocket_api.async_response
async def ws_set_panel_config(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Persist panel settings and/or RBAC config (admin only; enforced by _guard)."""
    holder = hass.data.get(_PANEL_DATA_KEY)
    if not holder:
        await async_load_panel_config(hass)
        holder = hass.data.get(_PANEL_DATA_KEY)
    cfg = holder["data"]
    try:
        if isinstance(msg.get("panel"), dict):
            cfg["panel"] = _sanitize_panel(msg["panel"], cfg.get("panel", {}))
        if isinstance(msg.get("rbac"), dict):
            cfg["rbac"] = _sanitize_rbac(msg["rbac"])
        await _save_panel_data(hass)
        _send_result(connection, msg["id"], "set_panel_config", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/set_user_prefs",
        vol.Required("prefs"): dict,
    }
)
@websocket_api.async_response
async def ws_set_user_prefs(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Persist the calling user's own view preferences (any authenticated user)."""
    user = getattr(connection, "user", None)
    if user is None:
        connection.send_error(msg["id"], "unauthorized", "No authenticated user")
        return
    holder = hass.data.get(_PANEL_DATA_KEY)
    if not holder:
        await async_load_panel_config(hass)
        holder = hass.data.get(_PANEL_DATA_KEY)
    cfg = holder["data"]
    prefs = cfg.setdefault("prefs", {})
    cur = dict(prefs.get(user.id, {}))
    p = msg["prefs"]
    if p.get("default_tab") in _PANEL_TABS:
        cur["default_tab"] = p["default_tab"]
    for k in ("show_expected", "show_raw", "show_raw_active", "show_debug", "onboarding_dismissed",
              # Settings: reveal the internal tuning fields (audit UI-02).
              "show_internal"):
        if k in p:
            cur[k] = bool(p[k])
    # F2: per-user Basic/Advanced settings disclosure level.
    if p.get("settings_level") in ("basic", "advanced"):
        cur["settings_level"] = p["settings_level"]
    # Panel font-size multiplier (accessibility). Coerce + clamp to safe bounds so a
    # bad value can never break rendering; without this it was silently dropped and
    # the setting vanished on refresh.
    if "font_scale" in p:
        try:
            cur["font_scale"] = max(0.7, min(2.0, float(p["font_scale"])))
        except (TypeError, ValueError, OverflowError):
            cur.pop("font_scale", None)
    # Display prefs: cycle date format + panel language override (paired with the
    # panel's save-prefs payload; without these they would be silently dropped).
    if p.get("date_format") in _PREF_DATE_FORMATS:
        cur["date_format"] = p["date_format"]
    if "lang_override" in p:
        lang = p["lang_override"]
        if lang == "":
            cur.pop("lang_override", None)  # empty clears -> system default
        elif isinstance(lang, str) and _PREF_LANG_TAG_RE.match(lang):
            cur["lang_override"] = lang
    # Cycle chart context length per device (item 513): {entry_id: minutes}. Merged
    # into the stored map, so a panel only has to send the device it changed.
    ctx = p.get("cycle_context_min")
    if isinstance(ctx, dict):
        ctx_map = dict(cur.get("cycle_context_min") or {})
        for eid, minutes in ctx.items():
            if not isinstance(eid, str) or not eid or len(eid) > 64:
                continue
            if isinstance(minutes, bool) or minutes not in _PREF_CYCLE_CONTEXT_MINUTES:
                continue
            ctx_map[eid] = int(minutes)
        # Bounded: one key per config entry, never more than any real install has.
        cur["cycle_context_min"] = dict(list(ctx_map.items())[-64:])
    # Allow setup guidance skip keys: setup_skip_<step_key> -> "never" | ISO timestamp
    for k, v in p.items():
        if isinstance(k, str) and k.startswith("setup_skip_"):
            if v is None:
                cur.pop(k, None)
            elif v == "never" or (isinstance(v, str) and len(v) <= 40):
                cur[k] = v
    prefs[user.id] = cur
    await _save_panel_data(hass)
    _send_result(connection, msg["id"], "set_user_prefs", {"success": True})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_match_debug",
        vol.Required("entry_id"): str,
    }
)
@callback
def ws_get_match_debug(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the latest live match result for the Status debug panel.

    Confidence, ambiguity flag, and the ranked candidate list (from the last
    in-cycle match attempt). Empty until the first match runs.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    out: dict[str, Any] = {"confidence": None, "ambiguous": False, "candidates": []}
    try:
        # All three from the SAME result (audit MATCH-DECIDE-14). The confidence
        # used to be `_last_match_confidence`, the committed program's (moved only
        # when the switching rules commit or confirm it), shown beside the newest
        # result's ambiguity flag and candidates.
        mr = getattr(manager, "_last_match_result", None)
        if mr is not None:
            conf = getattr(mr, "confidence", None)
            out["confidence"] = round(float(conf), 4) if conf is not None else None
            out["ambiguous"] = bool(getattr(mr, "is_ambiguous", False))
            out["candidates"] = manager.profile_store.get_match_candidates_summary(mr, 5)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error building match debug for %s: %s", entry_id, exc)

    _send_result(connection, msg["id"], "get_match_debug", out)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/set_program",
        vol.Required("entry_id"): str,
        vol.Required("program"): vol.Any(str, None),
    }
)
@callback
def ws_set_program(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Manually set the active program, or clear back to auto-detect.

    Drives the same manager methods as the program-select entity.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        prog = msg.get("program")
        if not prog or prog in ("auto_detect", "__auto__", "none"):
            manager.clear_manual_program()
        elif not manager.set_manual_program(prog):
            # Only a program that does not exist can fail now (#411). It used to
            # fail for the far more common reason of no cycle being under way, and
            # this line reported success anyway because the manager returned None
            # either way, so the panel showed a confirmation and then quietly
            # reverted the dropdown.
            connection.send_error(
                msg["id"], "not_found", f"No such program: {prog}"
            )
            return
        manager.notify_update()
        _send_result(connection, msg["id"], "set_program", {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_power_history",
        vol.Required("entry_id"): str,
        vol.Optional("with_raw", default=False): bool,
    }
)
@websocket_api.async_response
async def ws_get_power_history(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the live power trace for the status chart, held server-side.

    While a cycle runs ``live`` is the in-progress cycle trace (offsets from
    cycle start, so it lines up with the matched profile envelope); otherwise it
    is the recent readings. When ``with_raw`` is set and a cycle is running,
    ``raw`` is the configured power-sensor entity's recorder history over the
    cycle window, so the real socket data can be compared side by side with the
    integration's processed/sampled trace. Server-held so it survives a browser
    refresh, and the cycle trace survives an HA restart via state restore.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    with_raw = bool(msg.get("with_raw"))
    out: dict[str, Any] = {"cycle_active": False, "cycle_elapsed_s": 0.0, "live": [], "raw": [], "restart_gaps": []}
    try:
        detector = getattr(manager, "detector", None)
        diag = getattr(manager, "diag_buffer", None)
        trace = detector.get_power_trace() if detector else []
        cycle_start = getattr(detector, "current_cycle_start", None) if detector else None
        if cycle_start and trace:
            start_dt = trace[0][0]
            start_ts = start_dt.timestamp()
            live = [[round((t - start_dt).total_seconds(), 1), round(float(p), 1)] for t, p in trace]
            # Overlay any diag_buffer readings strictly newer than the last trace
            # point. During STATE_ENDING the detector trace can lag the raw sensor
            # by up to one sampling interval (readings are throttled; the diag_buffer
            # records every reading before the throttle), so the chart would appear
            # frozen at the last active reading until the next throttle-pass.
            if diag is not None and live:
                last_trace_ts = trace[-1][0].timestamp()
                for raw_ts, raw_w in diag.power_samples(last_trace_ts + 0.1):
                    offset_s = round(raw_ts - start_ts, 1)
                    if offset_s > live[-1][0]:
                        live.append([offset_s, round(float(raw_w), 1)])
            out["cycle_active"] = True
            out["live"] = _downsample(live)
            out["cycle_elapsed_s"] = live[-1][0] if live else 0.0
            out["cycle_start_iso"] = start_dt.isoformat()
            out["restart_gaps"] = list(getattr(manager, "_restart_gaps", []))
            if with_raw:
                ent = getattr(manager, "power_sensor_entity_id", None)
                if ent:
                    samples = await _recorder_power(hass, ent, start_dt)
                    start_ts = start_dt.timestamp()
                    out["raw"] = _downsample(
                        [[max(0.0, round(ts - start_ts, 1)), w] for ts, w in samples], 400
                    )
        elif diag is not None:
            recent = diag.power_samples(time.time() - 900.0)
            if recent:
                base = recent[0][0]
                out["live"] = _downsample([[round(ts - base, 1), round(float(w), 1)] for ts, w in recent])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error building power history for %s: %s", entry_id, exc)

    _send_result(connection, msg["id"], "get_power_history", out)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_logs",
        vol.Optional("level"): vol.Any(str, None),
        vol.Optional("limit", default=200): vol.All(int, vol.Range(min=1, max=500)),
    }
)
@callback
def ws_get_logs(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return recent ha_washdata log records (admin only; enforced by _guard)."""
    handler = hass.data.get(_LOG_BUFFER_KEY)
    recs = list(handler.records) if handler else []
    level = msg.get("level")
    if level and level in _LOG_LEVELS:
        minl = _LOG_LEVELS[level]
        recs = [r for r in recs if _LOG_LEVELS.get(r["level"], 0) >= minl]
    limit = msg.get("limit", 200)
    _send_result(connection, msg["id"], "get_logs", {"logs": recs[-limit:]})


# ─── ML Lab (shadow-mode comparison) ──────────────────────────────────────────

def _ml_median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _compute_cycle_events(
    points: list[tuple[float, float]],
    expectation: dict[str, float],
    end_feat_fn: Any,
    end_score_fn: Any,
    max_events: int = 20,
) -> list[dict[str, Any]]:
    """Find all low-power events in a trace and score each with the ML end detector.

    Each event represents a contiguous below-threshold power segment.  Events that
    were followed by power resumption are classified as pauses (``is_end=False``);
    the last event is marked as the end trigger (``is_end=True``).

    Motor-cycling appliances (washing machines) produce tens of micro-dips per cycle
    from drum motor switching.  A 30 s minimum filters these while still capturing
    genuine pause events which are typically > 1 min.
    """
    _MIN_EVENT_S = 30.0
    if not points or len(points) < 4:
        return []
    powers = [p for _, p in points]
    peak = max(powers) if powers else 0.0
    low_thresh = max(1.0, 0.05 * peak)

    events: list[dict[str, Any]] = []
    in_low = False
    seg_start_s = 0.0

    for i, (offset_s, pwr) in enumerate(points):
        if not in_low and pwr < low_thresh:
            in_low = True
            seg_start_s = offset_s
        elif in_low and pwr >= low_thresh:
            seg_end_s = points[i - 1][0] if i > 0 else offset_s
            low_run_s = seg_end_s - seg_start_s
            if low_run_s >= _MIN_EVENT_S:
                ml_conf: float | None = None
                try:
                    feat = end_feat_fn(points[:i], expectation)
                    if feat is not None:
                        ml_conf = round(float(end_score_fn(feat)), 3)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                events.append({"offset_s": float(round(seg_start_s, 0)), "low_run_s": float(round(low_run_s, 0)), "ml_end_confidence": ml_conf, "is_end": False})
            in_low = False
            if len(events) >= max_events:
                break

    if in_low:
        low_run_s = points[-1][0] - seg_start_s
        if low_run_s >= _MIN_EVENT_S:
            ml_conf = None
            try:
                feat = end_feat_fn(points, expectation)
                if feat is not None:
                    ml_conf = round(float(end_score_fn(feat)), 3)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            events.append({"offset_s": float(round(seg_start_s, 0)), "low_run_s": float(round(low_run_s, 0)), "ml_end_confidence": ml_conf, "is_end": True})

    if events and not events[-1]["is_end"]:
        events[-1]["is_end"] = True
    return events[:max_events]


# Signature persisted with each cycle's ``ml_health``: the shipped quality/end
# baselines. It used to carry an on-device model's ``trained_at`` so a retrain
# invalidated the cache; no on-device classifier exists since 0.5.8, and the
# value is unchanged so health already persisted stays valid.
_HEALTH_MODEL_SIG = "quality:base|end:base"


def _compute_ml_comparison(
    store: Any, *, force_recompute: bool = False
) -> dict[str, Any]:
    """Build the ML shadow-mode comparison report (CPU-intensive; runs in executor).

    Iterates all stored cycles, extracts features, and scores them against the
    embedded ML models.  Original detection logic is completely untouched; this
    is read-only analysis for the ML Lab panel page.

    Per-cycle health (quality + end scores, labels and the events timeline) is
    **persisted** on each cycle under ``ml_health`` keyed by the active model
    signature. Subsequent calls reuse the cached value instead of re-scoring on
    every panel load; it is only recomputed when the model changes (signature
    mismatch), when ``force_recompute`` is set (reprocess / maintenance /
    processing trigger), or for cycles that have never been assessed. The caller
    persists the store when ``result['_health_dirty']`` is true.
    """
    # Lazy imports so this module loads instantly even when ML deps are absent.
    try:
        from .ml.engine import resolve_scorer
        from .ml.feature_extraction import latest_end_event_features, quality_features
        from .profile_store import decompress_power_data
    except Exception:  # pylint: disable=broad-exception-caught
        return {"enabled": False, "error": "ML models not available", "cycles": []}

    # The shipped embedded baselines (no on-device classifier exists since 0.5.8).
    quality_score_fn, quality_source = resolve_scorer("quality")
    end_score_fn, end_source = resolve_scorer("end")
    if quality_score_fn is None and end_score_fn is None:
        return {"enabled": False, "error": "ML models not available", "cycles": []}

    model_sig = _HEALTH_MODEL_SIG
    health_dirty = False
    health_updates: dict[str, Any] = {}
    cycles: list[Any] = store.get_past_cycles()

    # --- Build per-profile statistics from cycle history ---
    raw_stats: dict[str, dict[str, list[float]]] = {}
    for cycle in cycles:
        name = cycle.get("profile_name")
        if not name:
            continue
        s = raw_stats.setdefault(name, {"durations": [], "energies": [], "peaks": []})
        dur = cycle.get("duration")
        energy = cycle.get("energy_wh")
        peak = cycle.get("max_power")
        if dur is not None:
            s["durations"].append(float(dur))
        if energy is not None:
            s["energies"].append(float(energy))
        if peak is not None:
            s["peaks"].append(float(peak))

    profile_medians: dict[str, dict[str, float]] = {}
    for name, s in raw_stats.items():
        profile_medians[name] = {
            "duration_s": _ml_median(s["durations"]) or 1800.0,
            "energy_wh": _ml_median(s["energies"]) or 500.0,
            "peak_w": _ml_median(s["peaks"]) or 500.0,
            "count": len(s["durations"]),
        }

    # --- Evaluate the newest 200 cycles ---
    # `cycles` is oldest-first (new cycles are appended at the end), so the newest
    # 200 are the tail; older ones are skipped before their trace is decompressed.
    evaluated: list[dict[str, Any]] = []
    recent_start_idx = max(0, len(cycles) - 200)

    for idx, cycle in enumerate(cycles):
        profile_name: str | None = cycle.get("profile_name")
        duration: float = float(cycle.get("duration") or 0)
        energy_wh: float = float(cycle.get("energy_wh") or 0)
        status: str = cycle.get("status", "completed")

        # A stored confidence of 0 most likely means the field wasn't recorded
        # (manually-labeled or pre-confidence cycles), not that the match was bad.
        # Treat 0 as "unknown" so we don't poison every quality feature with
        # worst-case proxies.
        raw_conf = cycle.get("match_confidence")
        conf_known: bool = isinstance(raw_conf, (int, float)) and raw_conf > 0
        match_conf: float = float(raw_conf) if conf_known else 0.0

        # Proxy feature values for quality scoring.  When confidence is known,
        # derive distance/margin/fit from it.  When unknown, use neutral values
        # so the model scores on trace shape alone.
        if conf_known:
            proxy_dist = max(0.0, 1.0 - match_conf)
            proxy_margin = match_conf
            proxy_fit = match_conf
        else:
            proxy_dist, proxy_margin, proxy_fit = 0.25, 0.30, 0.75

        if idx < recent_start_idx:
            continue

        points = decompress_power_data(cycle)

        # Reuse persisted per-cycle health when it was computed against the
        # current model (unless a recompute is forced). This is what keeps the
        # panel from re-scoring every cycle on every load.
        cached = cycle.get("ml_health")
        has_trace = len(points) >= ML_HEALTH_MIN_TRACE_POINTS
        if not has_trace:
            # #459: a cycle whose trace was pruned by retention cannot be scored.
            # `quality_features` falls back to a `has_trace = 0` row that no model
            # was trained on (the baseline sits ~38 sigma away from it), so every
            # pruned cycle came back as ~0.99 / "review" at the next forced
            # recompute and joined the review queue. Keep the last assessment that
            # was made WITH a trace; anything else (never scored, or scored after
            # the trace was gone, which is what the nightly run did until now) is
            # "no data". Checked before the cache so polluted entries heal on the
            # next panel load rather than at the next maintenance run.
            if isinstance(cached, dict) and cached.get("has_trace"):
                ml_quality = cached.get("score")
                quality_label = cached.get("label", "no_data")
                ml_end_conf = cached.get("end_score")
                end_label = cached.get("end_label", "no_event")
                events = cached.get("events") or []
            else:
                ml_quality = None
                quality_label = "no_data"
                ml_end_conf = None
                end_label = "no_event"
                events = []
                cycle_id_key = cycle.get("id", "")
                if cycle_id_key and not (
                    isinstance(cached, dict)
                    and cached.get("label") == "no_data"
                    and cached.get("score") is None
                ):
                    health_updates[cycle_id_key] = {
                        "score": None,
                        "label": "no_data",
                        "end_score": None,
                        "end_label": "no_event",
                        "events": [],
                        "model_sig": model_sig,
                        "has_trace": False,
                        "at": dt_util.now().isoformat(),
                    }
                    health_dirty = True
        elif (
            not force_recompute
            and isinstance(cached, dict)
            and cached.get("model_sig") == model_sig
        ):
            ml_quality = cached.get("score")
            quality_label = cached.get("label", "no_data")
            ml_end_conf = cached.get("end_score")
            end_label = cached.get("end_label", "no_event")
            events = cached.get("events") or []
        else:
            ml_quality = None
            if profile_name and profile_name in profile_medians:
                pm = profile_medians[profile_name]
                try:
                    feat = quality_features(
                        points=points,
                        profile_median_duration_s=pm["duration_s"],
                        profile_median_energy_wh=pm["energy_wh"],
                        profile_median_peak_w=pm["peak_w"],
                        profile_distance=proxy_dist,
                        label_margin=proxy_margin,
                        profile_fit_score=proxy_fit,
                        flag_count=0,
                    )
                    if quality_score_fn is not None:
                        ml_quality = float(quality_score_fn(feat))
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

            expectation: dict[str, float] = {}
            if profile_name and profile_name in profile_medians:
                pm = profile_medians[profile_name]
                expectation = {"duration": pm["duration_s"], "energy": pm["energy_wh"], "peak": pm["peak_w"]}

            ml_end_conf = None
            if expectation and points:
                try:
                    end_feat = latest_end_event_features(points, expectation)
                    if end_feat is not None and end_score_fn is not None:
                        ml_end_conf = float(end_score_fn(end_feat))
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

            # Per-cycle events timeline for the modal
            events = []
            if expectation and points and end_score_fn is not None:
                events = _compute_cycle_events(points, expectation, latest_end_event_features, end_score_fn)

            if ml_quality is None:
                quality_label = "no_data"
            elif ml_quality < 0.3:
                quality_label = "ok"
            elif ml_quality < 0.6:
                quality_label = "uncertain"
            else:
                quality_label = "review"

            if ml_end_conf is None:
                end_label = "no_event"
            elif ml_end_conf >= 0.6:
                end_label = "likely_end"
            elif ml_end_conf >= 0.35:
                end_label = "uncertain"
            else:
                end_label = "likely_pause"

            # Collect freshly-computed health for the event-loop to apply
            # back to the live store dicts (avoids mutating from executor thread).
            cycle_id_key = cycle.get("id", "")
            if cycle_id_key:
                health_updates[cycle_id_key] = {
                    "score": round(ml_quality, 3) if ml_quality is not None else None,
                    "label": quality_label,
                    "end_score": round(ml_end_conf, 3) if ml_end_conf is not None else None,
                    "end_label": end_label,
                    "events": events,
                    "model_sig": model_sig,
                    # #459: marks a score as trace-backed, so it survives the
                    # trace being pruned later instead of being re-scored blind.
                    "has_trace": True,
                    "at": dt_util.now().isoformat(),
                }
            health_dirty = True

        start_raw = cycle.get("start_time", "")
        evaluated.append({
            "id": cycle.get("id", ""),
            "start_time": start_raw if isinstance(start_raw, str) else "",
            "duration_s": round(duration, 0),
            "status": status,
            "profile_name": profile_name,
            "match_confidence": round(match_conf, 3),
            "confidence_known": conf_known,
            "energy_wh": round(energy_wh, 1),
            "ml_quality_score": round(ml_quality, 3) if ml_quality is not None else None,
            "ml_quality_label": quality_label,
            "ml_end_confidence": round(ml_end_conf, 3) if ml_end_conf is not None else None,
            "ml_end_label": end_label,
            "has_power_data": has_trace,
            "events": events,
            "ml_review": cycle.get("ml_review") or {},
        })

    # Panel expects most-recent-first ordering; the loop appended oldest-first.
    evaluated.reverse()

    return {
        "enabled": True,
        "cycle_count": len(cycles),
        "evaluated_count": sum(1 for e in evaluated if e["ml_quality_score"] is not None),
        "cycles": evaluated,
        "model_source": {"quality": quality_source, "end": end_source},
        "_health_dirty": health_dirty,
        "_health_updates": health_updates,
        "profile_stats": {
            name: {"count": int(m["count"]), "median_duration_s": int(m["duration_s"]), "median_energy_wh": round(m["energy_wh"], 1)}
            for name, m in profile_medians.items()
        },
    }


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_ml_comparison",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_get_ml_comparison(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the ML shadow-mode comparison report for the ML Lab panel page.

    Runs the embedded ML models against stored cycle history and compares with
    the existing detection/suggestion logic.  Read-only: no side effects on the
    proven algorithms.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        store = manager.profile_store
        result = await hass.async_add_executor_job(_compute_ml_comparison, store)
        # Apply any freshly-computed per-cycle health updates on the event loop
        # (the executor must not mutate live store dicts directly), then persist.
        health_updates = result.pop("_health_updates", {})
        result.pop("_health_dirty", False)
        if health_updates:
            for cycle in store.get_past_cycles():
                cid = cycle.get("id", "")
                if cid and cid in health_updates:
                    cycle["ml_health"] = health_updates[cid]
            await store.async_save()
        _send_result(connection, msg["id"], "get_ml_comparison", result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("ML comparison failed for %s: %s", entry_id, exc)
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_ml_training_status",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_get_ml_training_status(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return on-device ML training status for the Tuning > ML Training panel."""
    from .const import (  # pylint: disable=import-outside-toplevel
        CONF_ML_TRAINING_ENABLED,
        CONF_ML_TRAINING_HOUR,
        CONF_ML_TRAINING_INTERVAL_DAYS,
        CONF_ML_TRAINING_MIN_CYCLES,
        DEFAULT_ML_TRAINING_ENABLED,
        DEFAULT_ML_TRAINING_HOUR,
        DEFAULT_ML_TRAINING_INTERVAL_DAYS,
        DEFAULT_ML_TRAINING_MIN_CYCLES,
    )

    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    entry = _get_entry(hass, entry_id)
    merged: dict[str, Any] = {**(entry.data if entry else {}), **(entry.options if entry else {})}
    store = manager.profile_store
    versions = store.get_ml_model_versions() or {}
    last = manager._last_ml_training_at()  # pylint: disable=protected-access
    # Plain-language names + a one-line "what it does" for each capability, so the
    # panel never has to show raw internal keys to non-ML users.
    _cap_labels = {
        "total_energy": ("Energy estimate", "Predicting total energy and cost"),
    }
    # Only what is trained on-device AND consumed is listed, which since 0.5.8 is
    # `total_energy` alone. `end` is no longer trained (its consumer is frozen
    # off, audit ML-05/11) and `remaining_time` / `live_match` / `quality` lost
    # consumer and training; storage v17 drops all four records, so an older
    # import is the only way one can still be here.
    live_caps = set(_cap_labels)
    models: dict[str, Any] = {}
    for cap, v in versions.items():
        if not isinstance(v, dict) or cap not in live_caps:
            continue
        spec = v.get("spec") if isinstance(v.get("spec"), dict) else {}
        label, blurb = _cap_labels.get(cap, (cap, ""))
        info: dict[str, Any] = {
            "trained_at": v.get("trained_at"),
            "cycle_count": v.get("cycle_count"),
            "kind": spec.get("kind"),
            # ``label``/``blurb`` are the English fallbacks; the panel renders
            # _t(label_key/blurb_key, {}, fallback). Keyed by capability so a
            # missing key falls back cleanly to the English text.
            "label": label,
            "label_key": f"ml.cap_label.{cap}" if cap in _cap_labels else None,
            "blurb": blurb,
            "blurb_key": f"ml.cap_blurb.{cap}" if cap in _cap_labels else None,
        }
        # Raw metric numbers so the panel can render a humanized quality indicator
        # (a bar + word) with the exact figure on hover: MAE vs the naive estimate
        # (every listed capability is a regressor since 0.5.8).
        if v.get("model_mae") is not None and v.get("naive_mae") is not None:
            info["model_mae"] = round(float(v["model_mae"]), 5)
            info["naive_mae"] = round(float(v["naive_mae"]), 5)
            info["metric"] = f"error {float(v['model_mae']):.3f} vs {float(v['naive_mae']):.3f} baseline"
            info["metric_key"] = "ml.metric_mae"
            info["metric_params"] = {
                "model": f"{float(v['model_mae']):.3f}",
                "baseline": f"{float(v['naive_mae']):.3f}",
            }
        # How many cycles the metric was measured on (audit ML-20); absent on a
        # record promoted before the count was stored.
        if isinstance(v.get("held_out_cycles"), int):
            info["held_out_cycles"] = int(v["held_out_cycles"])
        models[cap] = info

    # Fit trend across the PROMOTED runs (drift): compare the mean held-out score
    # of the most-recent third of the models that were put in use to the oldest
    # third, respecting each metric's direction. A candidate that was rejected
    # never served, so its score says nothing about the model in use (audit
    # ML-20); entries from before that rule carry no `promoted` and are skipped.
    # Only meaningful with a few promotions of history.
    history = store.get_ml_training_history()
    for cap, info in models.items():
        series = history.get(cap) if isinstance(history, dict) else None
        if not isinstance(series, list):
            continue
        promoted_runs = [
            e for e in series
            if isinstance(e, dict) and e.get("promoted") is True and "score" in e
        ]
        scores = [float(e["score"]) for e in promoted_runs]
        if len(scores) < 4:
            continue
        higher_better = bool(promoted_runs[-1].get("higher_better", True))
        third = max(1, len(scores) // 3)
        old_mean = sum(scores[:third]) / third
        new_mean = sum(scores[-third:]) / third
        if abs(old_mean) < 1e-9:
            continue
        rel = (new_mean - old_mean) / abs(old_mean)
        if not higher_better:
            rel = -rel  # for error metrics, a decrease is an improvement
        info["trend"] = "improving" if rel > 0.03 else "declining" if rel < -0.03 else "steady"

    # The last run per capability, so the panel can say why nothing was (re)learnt
    # (audit ML-20). Only runs recorded with `promoted` qualify.
    last_run: dict[str, Any] = {}
    for cap in live_caps:
        series = history.get(cap) if isinstance(history, dict) else None
        entry = series[-1] if isinstance(series, list) and series else None
        if not isinstance(entry, dict) or "promoted" not in entry:
            continue
        run: dict[str, Any] = {"ts": entry.get("ts"), "promoted": bool(entry["promoted"])}
        if not run["promoted"]:
            params = entry.get("reason_params")
            run["reason_code"] = str(entry.get("reason_code") or "unknown")
            run["reason_params"] = dict(params) if isinstance(params, dict) else {}
            run["reason"] = str(entry.get("reason") or "")
        last_run[cap] = run

    _send_result(connection, msg["id"], "get_ml_training_status", {
            "available": ENABLE_ML_TRAINING,
            "enabled": bool(merged.get(CONF_ML_TRAINING_ENABLED, DEFAULT_ML_TRAINING_ENABLED)),
            "running": bool(getattr(manager, "_ml_training_running", False)),
            "last_trained": last.isoformat() if last else None,
            "cycle_count": len(store.get_past_cycles()),
            "min_cycles": int(merged.get(CONF_ML_TRAINING_MIN_CYCLES, DEFAULT_ML_TRAINING_MIN_CYCLES)),
            "interval_days": int(merged.get(CONF_ML_TRAINING_INTERVAL_DAYS, DEFAULT_ML_TRAINING_INTERVAL_DAYS)),
            "hour": int(merged.get(CONF_ML_TRAINING_HOUR, DEFAULT_ML_TRAINING_HOUR)),
            "on_device_models": models,
            "last_run": last_run,
        },
    )


async def _ml_training_task(hass: HomeAssistant, task: Any, entry_id: str) -> None:
    """Detached runner for on-device ML training; stores the summary as the result.

    NOT a WS handler: it is a plain coroutine kicked off via ``hass.async_create_task``
    by ``ws_trigger_ml_training``. It must carry no ``@websocket_command`` /
    ``@async_response`` decorators (those would rewrite it into a sync handler that
    returns ``None``, so the direct call would pass ``None`` to ``async_create_task``)."""
    reg = task_registry.get_registry(hass)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    # Serialize under the per-entry write lock (same lock _reprocess_task uses, and
    # reprocess itself runs ML training): training rewrites the store, so two runs
    # for the same entry must not interleave.
    lock = _entry_write_lock(hass, entry_id)
    acquired = False
    try:
        await lock.acquire()
        acquired = True
        summary = await manager.async_run_ml_training(force=True)
        reg.finish(task, state=task_registry.STATE_DONE, result=summary)
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Task-level failure: log at WARNING so it surfaces in the default HA log and
        # the panel Logs view (not swallowed at debug like a routine sub-step miss).
        _LOGGER.warning("ML training task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))
    finally:
        if acquired:
            lock.release()


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/trigger_ml_training",
        vol.Required("entry_id"): str,
    }
)
@callback
def ws_trigger_ml_training(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Kick off on-device ML training as a detached, registry-tracked task (manual,
    bypasses the schedule guards); returns its id. Result via the registry."""
    entry_id: str = msg["entry_id"]
    if _get_manager(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    if not ENABLE_ML_TRAINING:
        connection.send_error(msg["id"], "not_available", "ML training is not enabled in this build")
        return
    reg = task_registry.get_registry(hass)
    task = reg.create(entry_id, "ml_training", "Learning")
    _raw = hass.async_create_task(_ml_training_task(hass, task, entry_id))
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "trigger_ml_training", {"task_id": task.id})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/revert_ml_models",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_revert_ml_models(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Revert all on-device trained models to the shipped embedded baselines.

    Drops every promoted spec in ``ml_model_versions`` so ``resolve_regressor``
    becomes inert for the baseline-less ``total_energy`` regressor. The next training pass may
    re-promote models if they beat the baseline again.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    await manager.profile_store.clear_ml_model_versions()
    manager.notify_update()
    _send_result(connection, msg["id"], "revert_ml_models", {"success": True})


_ML_REVIEW_QUALITIES = {"", "good", "bad", "unusable"}


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/set_ml_review",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Optional("quality"): vol.In(sorted(_ML_REVIEW_QUALITIES)),
        vol.Optional("golden"): bool,
        vol.Optional("tags"): [str],
        vol.Optional("notes"): str,
    }
)
@websocket_api.async_response
async def ws_set_ml_review(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Attach an ML-Lab review (quality / golden / tags / notes) to a cycle.

    The write-back from the read-only shadow view. (Until 0.5.8 a review also
    labelled the on-device quality model, whose training was removed.)
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        await manager.profile_store.set_cycle_review(
            msg["cycle_id"],
            quality=msg.get("quality"),
            golden=msg.get("golden"),
            tags=msg.get("tags"),
            notes=msg.get("notes"),
        )
        manager.notify_update()
        _send_result(connection, msg["id"], "set_ml_review", {"success": True})
    except ValueError as exc:
        connection.send_error(msg["id"], "not_found", str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


# ── Cycle Controls ────────────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/pause_cycle", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_pause_cycle(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """User-pause the active cycle."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    ok = await manager.async_pause_cycle()
    _send_result(connection, msg["id"], "pause_cycle", {"ok": bool(ok)})


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/resume_cycle", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_resume_cycle(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Resume a user-paused cycle."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    ok = await manager.async_resume_cycle()
    _send_result(connection, msg["id"], "resume_cycle", {"ok": bool(ok)})


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/terminate_cycle", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_terminate_cycle(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Force-terminate the active cycle."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    await manager.async_terminate_cycle()
    _send_result(connection, msg["id"], "terminate_cycle", {"ok": True})


# ─── Playground (F3): headless what-if replay + DTW visualizer ──────────────────


def _playground_base_config(manager: Any, entry: Any) -> CycleDetectorConfig:
    """Resolve the device's live CycleDetectorConfig as the simulation base.

    Prefers the running detector's config (already merged with every default);
    falls back to a minimal config derived from the entry's effective options so
    the Playground still works if the detector is not yet initialised.
    """
    detector = getattr(manager, "detector", None)
    cfg = getattr(detector, "config", None)
    if isinstance(cfg, CycleDetectorConfig):
        return cfg
    # The same builder the manager uses (audit F2): this fallback had its own
    # defaults (min_off_gap 60 s for every device, scalar completion floor) and
    # gave a not-yet-initialised device a configuration no detector runs.
    options = dict(getattr(entry, "options", {}) or {}) if entry is not None else {}
    data = dict(getattr(entry, "data", {}) or {}) if entry is not None else {}
    device_type = str(
        options.get(CONF_DEVICE_TYPE, data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE))
    )
    return build_detector_config(options, data, device_type)


def _playground_context(hass: HomeAssistant, entry_id: str):
    """Return (manager, store, base_config, options, price) for a Playground call,
    or None (after sending the appropriate error) when unavailable."""
    entry = _get_entry(hass, entry_id)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        return None
    store = getattr(manager, "profile_store", None)
    if store is None:
        return None
    base_config = _playground_base_config(manager, entry)
    options = {}
    if entry is not None:
        options = {**getattr(entry, "data", {}), **getattr(entry, "options", {})}
    try:
        price = manager._resolve_energy_price()  # noqa: SLF001
    except Exception:  # pylint: disable=broad-exception-caught
        price = None
    return manager, store, base_config, options, price


# ─── Playground settings control panel (live values + presets) ─────────────────


def _playground_preset_list(store: Any) -> list[dict[str, Any]]:
    """Presets as a name-sorted list for the panel dropdown.

    Each stored record's values are passed through the same allow-list a save
    uses, so a preset written before 0.5.8 still loads: keys the Playground no
    longer offers (the Stage 2-4 matcher weights) are dropped from the view while
    the stored record stays untouched.
    """
    presets = store.get_playground_presets()
    out: list[dict[str, Any]] = []
    for name, record in presets.items():
        if not isinstance(record, dict):
            continue
        out.append({
            "name": name,
            "values": playground.sanitize_setting_values(record.get("values")),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        })
    out.sort(key=lambda p: str(p["name"]).lower())
    return out


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_playground_settings",
        vol.Required("entry_id"): str,
        vol.Optional("include_suggestions", default=True): bool,
    }
)
@websocket_api.async_response
async def ws_get_playground_settings(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the device's LIVE effective Playground settings plus saved presets.

    ``effective`` is read back off the same live detector/matcher config the
    simulation uses, so the control panel always opens on what the integration is
    really running - never on a stale schema default. ``publishable`` lists the
    keys the panel may write back to the config entry.

    ``include_suggestions=False`` skips the auto-tuner suggestions. They exist only to
    label the "Load suggested" button, so the panel opens without them and fetches them
    in the background. Defaults to True so every other caller is unaffected.
    """
    entry_id: str = msg["entry_id"]
    ctx = _playground_context(hass, entry_id)
    if ctx is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    _manager, store, base_config, options, _price = ctx
    try:
        match_config = playground._matching_config(store)  # noqa: SLF001
    except Exception:  # pylint: disable=broad-exception-caught
        match_config = {}

    include_suggestions = bool(msg.get("include_suggestions", True))

    # Classic suggestions from the store (periodic analysis results), filtered to
    # keys the Playground actually exposes — cheap dict read, no executor needed.
    raw_sugg: dict[str, Any] = {}
    try:
        raw_sugg = store.get_suggestions() if include_suggestions else {}
        raw_sugg = raw_sugg or {}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug(
            "Could not read Playground classic suggestions for %s: %s", entry_id, exc
        )
    classic_sugg: dict[str, Any] = {}
    for key in playground.SETTING_KEYS:
        item = raw_sugg.get(key)
        if isinstance(item, dict) and item.get("value") is not None:
            val = item["value"]
            try:
                classic_sugg[key] = int(float(val)) if key in _SUGGESTION_INT_KEYS else round(float(val), 4)
            except (TypeError, ValueError, OverflowError):
                pass

    _send_result(connection, msg["id"], "get_playground_settings", {
        "effective": playground.effective_settings(base_config, match_config),
        "presets": _playground_preset_list(store),
        "publishable": sorted(playground.PUBLISHABLE_SETTING_KEYS),
        "preset_limit": PLAYGROUND_PRESET_MAX,
        "classic_suggestions": classic_sugg,
    })


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/save_playground_preset",
        vol.Required("entry_id"): str,
        vol.Required("name"): str,
        vol.Required("values"): dict,
    }
)
@websocket_api.async_response
async def ws_save_playground_preset(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Save (or overwrite) a named snapshot of the Playground's settings."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    store = getattr(manager, "profile_store", None) if manager else None
    if store is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    values = playground.sanitize_setting_values(msg["values"])
    try:
        await store.async_save_playground_preset(msg["name"], values)
    except ValueError as exc:
        connection.send_error(msg["id"], "invalid_format", str(exc))
        return
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # The store write can fail (disk full, permissions). Without this the
        # exception escapes the handler and the client never gets a reply, so the
        # panel's save button spins forever instead of reporting the failure.
        _LOGGER.warning("Saving playground preset failed for %s: %s", entry_id, exc)
        connection.send_error(msg["id"], "unknown_error", str(exc))
        return
    _send_result(connection, msg["id"], "save_playground_preset", {
        "success": True,
        "presets": _playground_preset_list(store),
    })


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/delete_playground_preset",
        vol.Required("entry_id"): str,
        vol.Required("name"): str,
    }
)
@websocket_api.async_response
async def ws_delete_playground_preset(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Delete a saved Playground settings preset."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    store = getattr(manager, "profile_store", None) if manager else None
    if store is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    try:
        removed = await store.async_delete_playground_preset(msg["name"])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Deleting playground preset failed for %s: %s", entry_id, exc)
        connection.send_error(msg["id"], "unknown_error", str(exc))
        return
    _send_result(connection, msg["id"], "delete_playground_preset", {
        "success": removed,
        "presets": _playground_preset_list(store),
    })


# ─── Background-task registry (progress / cancel / reconnect-safe results) ──────


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/subscribe_tasks",
        vol.Optional("entry_id"): vol.Any(str, None),
    }
)
@callback
def ws_subscribe_tasks(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Live push of task progress. Sends the current snapshot as `task` events on
    subscribe, then one event per change until the client unsubscribes / the
    socket closes. The client dedupes by id and keeps the latest `updated_at`."""
    reg = task_registry.get_registry(hass)
    entry_id = msg.get("entry_id")
    iden = msg["id"]

    @callback
    def _forward(snap: dict[str, Any]) -> None:
        if entry_id and snap.get("entry_id") != entry_id:
            return
        connection.send_message(
            websocket_api.event_message(iden, {"type": "task", "task": snap})
        )

    connection.subscriptions[iden] = reg.add_listener(_forward)
    connection.send_result(iden)
    for snap in reg.snapshot(entry_id):
        connection.send_message(
            websocket_api.event_message(iden, {"type": "task", "task": snap})
        )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/cancel_task",
        vol.Required("task_id"): str,
    }
)
@callback
def ws_cancel_task(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Request cancellation of a running task (consumers stop at the next chunk)."""
    reg = task_registry.get_registry(hass)
    _send_result(connection, msg["id"], "cancel_task", {"cancelled": reg.cancel(msg["task_id"])})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_task_result",
        vol.Required("task_id"): str,
    }
)
@callback
def ws_get_task_result(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Fetch a finished task's stored result (reloadable after a tab switch /
    reconnect until the task is evicted)."""
    reg = task_registry.get_registry(hass)
    task = reg.get(msg["task_id"])
    if task is None:
        connection.send_error(msg["id"], "not_found", "Task not found")
        return
    _send_result(connection, msg["id"], "get_task_result", task.snapshot(include_result=True))


# -- Playground batch/sweep as detached, registry-tracked tasks -----------------
# The heavy replay runs in the executor CHUNK-BY-CHUNK (awaiting between chunks so
# the event loop breathes and the executor thread is freed), updating the task's
# progress and checking its cancel flag. Because the task is detached
# (async_create_task), it survives a dropped socket; the panel re-attaches via
# subscribe_tasks and reads the result with get_task_result.

_PG_HISTORY_CHUNK = 2

# One batch replay (Test on history / Optimize) per device at a time (audit
# PLATFORM-04): each is minutes of executor CPU on a Pi, and nothing stopped a
# reconnecting panel or a script from stacking them. A second is refused with
# `task_busy`; a new Simulate supersedes the previous one instead (PLAYGROUND-12).
_PG_BATCH_KINDS = frozenset({"pg_history", "pg_sweep"})


def _pg_running(reg: Any, entry_id: str, kinds: frozenset[str]) -> list[str]:
    """Ids of this entry's running tasks of ``kinds``."""
    return [
        t["id"] for t in reg.snapshot(entry_id)
        if t.get("state") == task_registry.STATE_RUNNING and t.get("kind") in kinds
    ]


def _pg_refuse_busy(hass: HomeAssistant, connection: Any, msg_id: int, entry_id: str) -> bool:
    """Send ``task_busy`` and return True while a batch replay runs for the entry."""
    if not _pg_running(task_registry.get_registry(hass), entry_id, _PG_BATCH_KINDS):
        return False
    connection.send_error(
        msg_id, "task_busy",
        "A Test-on-history or Optimize run is already in progress for this device",
    )
    return True


async def _pg_history_task(
    hass: HomeAssistant, task: Any, entry_id: str,
    cycle_ids: list[str], override: dict[str, Any] | None,
    count: int | None = None,
) -> None:
    reg = task_registry.get_registry(hass)
    ctx = _playground_context(hass, entry_id)
    if ctx is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    _manager, store, base_config, options, price = ctx
    try:
        selected = await hass.async_add_executor_job(
            playground._select_cycles, store, cycle_ids or None, count  # noqa: SLF001
        )
        ids = [c.get("id") for c in selected if c.get("id")]
        # Build match snapshots once — they are store-derived and identical for
        # every chunk; rebuilding per chunk is O(n_profiles) wasted work.
        prebuilt = await hass.async_add_executor_job(playground._build_match_snapshots, store)
        reg.update(task, total=len(ids))
        rows: list[dict[str, Any]] = []
        base_rows: list[dict[str, Any]] = []
        for i in range(0, len(ids), _PG_HISTORY_CHUNK):
            if task.cancel_requested:
                break
            chunk = ids[i:i + _PG_HISTORY_CHUNK]
            r = await hass.async_add_executor_job(
                playground.run_playground_history,
                store, chunk, base_config, override, options, price, len(chunk), prebuilt,
            )
            rows.extend(r.get("rows") or [])
            base_rows.extend(r.get("baseline_rows") or [])
            reg.update(task, done=min(len(ids), i + len(chunk)))
        payload = playground.finalize_history(rows, base_rows, bool(override))
        payload["partial"] = task.cancel_requested
        reg.finish(
            task,
            state=task_registry.STATE_CANCELLED if task.cancel_requested else task_registry.STATE_DONE,
            result=payload,
        )
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground history task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))


async def _pg_sweep_task(
    hass: HomeAssistant, task: Any, entry_id: str,
    param: str, values: list[float], objective: str,
    cycle_ids: list[str] | None = None,
    count: int | None = None,
) -> None:
    reg = task_registry.get_registry(hass)
    ctx = _playground_context(hass, entry_id)
    if ctx is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    _manager, store, base_config, options, price = ctx
    try:
        # The cycles the panel's "Last N" picked (audit PLAYGROUND-18), else the
        # most recent ones; the selection rule is the playground's own.
        selected = await hass.async_add_executor_job(
            playground._select_cycles, store, cycle_ids or None, count  # noqa: SLF001
        )
        ids = [c.get("id") for c in selected if c.get("id")]
        n = max(1, len(ids))
        # Build match snapshots once — identical for every sweep value/cell.
        prebuilt = await hass.async_add_executor_job(playground._build_match_snapshots, store)
        reg.update(task, total=len(values) + 1)
        # The current settings on the same cycles: what a value has to beat, and
        # the early-end / split / detection floor none may lose (PLAYGROUND-08).
        baseline = await hass.async_add_executor_job(
            playground.sweep_baseline,
            store, ids, base_config, objective, options, price, prebuilt,
        )
        reg.update(task, done=1)
        points: list[dict[str, Any]] = []
        current_value: Any = None
        for i, vx in enumerate(values):
            if task.cancel_requested:
                break
            r = await hass.async_add_executor_job(
                playground.run_playground_sweep,
                store, ids, base_config, param, [vx], objective, options, price, n,
                prebuilt,
            )
            points.extend(r.get("points") or [])
            if r.get("current_value") is not None:
                current_value = r["current_value"]
            reg.update(task, done=i + 2)
        payload = playground.finalize_sweep_1d(
            param, objective, points, current_value, baseline
        )
        payload["partial"] = task.cancel_requested
        reg.finish(
            task,
            state=task_registry.STATE_CANCELLED if task.cancel_requested else task_registry.STATE_DONE,
            result=payload,
        )
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground sweep task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/start_playground_history",
        vol.Required("entry_id"): str,
        vol.Optional("cycle_ids", default=list): [str],
        vol.Optional("settings_override", default=dict): dict,
        # "Last N" (audit PLAYGROUND-18): the most recent N, newest first.
        vol.Optional("count"): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=playground.MAX_BATCH_CYCLES)
        ),
    }
)
@callback
def ws_start_playground_history(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Kick off a detached, registry-tracked Test-on-history replay; returns the
    task id immediately. Progress/result come via subscribe_tasks/get_task_result."""
    entry_id = msg["entry_id"]
    if _playground_context(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    if _pg_refuse_busy(hass, connection, msg["id"], entry_id):
        return
    reg = task_registry.get_registry(hass)
    task = reg.create(entry_id, "pg_history", "Test on history")
    override = dict(msg.get("settings_override") or {}) or None
    cycle_ids = list(msg.get("cycle_ids") or [])
    _raw = hass.async_create_task(
        _pg_history_task(hass, task, entry_id, cycle_ids, override, msg.get("count"))
    )
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "start_playground_history", {"task_id": task.id})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/start_playground_sweep",
        vol.Required("entry_id"): str,
        vol.Required("param"): str,
        # Capped like the old one-shot command's 20x20 (audit PLAYGROUND-16).
        vol.Required("values"): vol.All([vol.Coerce(float)], vol.Length(max=20)),
        vol.Required("objective"): str,
        # Which cycles (audit PLAYGROUND-18): the panel's "Last N" as `count`, the
        # most recent N; explicit ids win; neither = the most recent 20.
        vol.Optional("cycle_ids", default=list): [str],
        vol.Optional("count"): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=playground.MAX_BATCH_CYCLES)
        ),
    }
)
@callback
def ws_start_playground_sweep(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Kick off a detached, registry-tracked Optimize sweep; returns the task id."""
    entry_id = msg["entry_id"]
    if _playground_context(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    if _pg_refuse_busy(hass, connection, msg["id"], entry_id):
        return
    reg = task_registry.get_registry(hass)
    task = reg.create(
        entry_id, "pg_sweep", f"Optimize: {msg['param']}",
        label_key="task.pg_sweep.optimize", label_params={"param": msg["param"]},
    )
    _raw = hass.async_create_task(_pg_sweep_task(
        hass, task, entry_id, msg["param"], list(msg.get("values") or []),
        msg["objective"], list(msg.get("cycle_ids") or []), msg.get("count"),
    ))
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "start_playground_sweep", {"task_id": task.id})


# Readings replayed per executor job for the single-cycle detail sim. The event
# loop breathes between chunks; a ~233min/5s dishwasher cycle (~2800 readings)
# becomes ~11 short jobs instead of one multi-minute GIL-holding call (issue #311).
_PG_DETAIL_CHUNK = 250


async def _pg_detail_task(
    hass: HomeAssistant, task: Any, entry_id: str,
    cycle_id: str, override: dict[str, Any] | None,
) -> None:
    reg = task_registry.get_registry(hass)
    ctx = _playground_context(hass, entry_id)
    if ctx is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    _manager, store, base_config, options, price = ctx
    try:
        sim = await hass.async_add_executor_job(
            playground.build_cycle_detail_sim_by_id,
            store, cycle_id, base_config, override, options, price,
        )
        if isinstance(sim, dict):  # {"error": ...} marker (not_found / build failure)
            if sim.get("error") == "not_found":
                reg.finish(task, state=task_registry.STATE_ERROR, error="not_found")
            else:
                reg.finish(task, state=task_registry.STATE_ERROR, error=str(sim.get("error")))
            return
        if not sim.ready:
            reg.finish(task, state=task_registry.STATE_DONE, result=sim.empty_payload())
            return
        total = sim.n_readings
        reg.update(task, total=total)
        for i in range(0, total, _PG_DETAIL_CHUNK):
            if task.cancel_requested:
                break
            await hass.async_add_executor_job(sim.step, i, i + _PG_DETAIL_CHUNK)
            reg.update(task, done=min(total, i + _PG_DETAIL_CHUNK))
        if not task.cancel_requested:
            await hass.async_add_executor_job(sim.run_tail)
        payload = await hass.async_add_executor_job(sim.finalize)
        payload["partial"] = task.cancel_requested
        reg.finish(
            task,
            state=task_registry.STATE_CANCELLED if task.cancel_requested else task_registry.STATE_DONE,
            result=payload,
        )
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground detail task failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/start_playground_cycle_detail",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Optional("settings_override", default=dict): dict,
    }
)
@callback
def ws_start_playground_cycle_detail(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Kick off a detached, registry-tracked single-cycle "Simulate" replay;
    returns the task id immediately. The heavy per-5s replay runs chunk-by-chunk
    in the executor so a long cycle no longer stalls Home Assistant (issue #311).
    Progress/result come via subscribe_tasks/get_task_result."""
    entry_id = msg["entry_id"]
    if _playground_context(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    reg = task_registry.get_registry(hass)
    # A new Simulate replaces the previous one: its result would be dropped by the
    # panel anyway, and it kept replaying to the end (PLAYGROUND-12 / PLATFORM-04).
    for stale in _pg_running(reg, entry_id, frozenset({"pg_detail"})):
        reg.cancel(stale)
    task = reg.create(
        entry_id, "pg_detail", "Simulate cycle",
        label_key="task.pg_detail.simulate", label_params={},
    )
    override = dict(msg.get("settings_override") or {})
    _raw = hass.async_create_task(
        _pg_detail_task(hass, task, entry_id, msg["cycle_id"], override)
    )
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "start_playground_cycle_detail", {"task_id": task.id})


# ─── Historical power-data import (issue #344) ─────────────────────────────────
#
# Four steps, because the data is large and the answer is the user's to approve:
#
#   1. ingest   - `history_import_begin` + `history_import_chunk` stage CSV text, or
#                 `history_import_recorder` fills the same buffer from the recorder.
#                 Chunked because Home Assistant builds its WebSocket with aiohttp's
#                 default 4 MiB frame cap, and ten days of 5-second data is 5-8 MB of
#                 text; an over-cap frame is not rejected, it closes the connection.
#   2. scan     - `start_history_import_scan` replays the stream through fresh detectors
#                 as a detached registry task, chunk by chunk (see history_import.py for
#                 why a raw stream cannot be fed to one detector).
#   3. review   - the panel shows one row per candidate; the whole traces never cross
#                 the wire (they would blow the same frame cap).
#   4. apply    - `apply_history_import` persists the accepted rows into
#                 `backfill_cycles`.
#
# Parsing lives in Python, not in the panel, so one implementation is under test.

_HISTORY_IMPORT_KEY = f"{DOMAIN}_history_import"


def _history_staging(hass: HomeAssistant) -> dict[str, dict[str, Any]]:
    """Per-entry staging area for an in-flight import, keyed by entry_id.

    One slot per entry: a new upload replaces the previous one, so an abandoned 8 MB
    paste cannot accumulate. Cleared by :func:`async_clear_history_import` on unload.
    """
    return hass.data.setdefault(_HISTORY_IMPORT_KEY, {})


def async_clear_history_import(hass: HomeAssistant, entry_id: str) -> None:
    """Drop any staged upload and scan result for an entry (called on unload)."""
    _history_staging(hass).pop(entry_id, None)


def _history_slot(hass: HomeAssistant, entry_id: str, token: str) -> dict[str, Any] | None:
    slot = _history_staging(hass).get(entry_id)
    if not isinstance(slot, dict) or slot.get("token") != token:
        return None
    return slot


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/history_import_begin",
        vol.Required("entry_id"): str,
    }
)
@callback
def ws_history_import_begin(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Open a staging slot for a CSV upload and return the token chunks must carry."""
    entry_id: str = msg["entry_id"]
    if _get_manager(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    token = uuid.uuid4().hex
    _history_staging(hass)[entry_id] = {
        "token": token,
        "chunks": [],
        "bytes": 0,
        "next_seq": 0,
        "source": "csv",
    }
    _send_result(connection, msg["id"], "history_import_begin", {
        "token": token,
        "max_bytes": HISTORY_IMPORT_MAX_BYTES,
        "chunk_bytes": HISTORY_IMPORT_CHUNK_BYTES,
    })


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/history_import_chunk",
        vol.Required("entry_id"): str,
        vol.Required("token"): str,
        vol.Required("seq"): int,
        vol.Required("text"): str,
    }
)
@callback
def ws_history_import_chunk(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Append one chunk of CSV text to the staging slot.

    Sequence-checked: a dropped or re-ordered chunk would splice the file silently, so a
    mismatch is an error the panel can restart from rather than a corrupt import.
    """
    entry_id: str = msg["entry_id"]
    slot = _history_slot(hass, entry_id, msg["token"])
    if slot is None:
        connection.send_error(msg["id"], "not_found", "No upload in progress; start again")
        return
    if int(msg["seq"]) != slot["next_seq"]:
        connection.send_error(
            msg["id"], "invalid_format",
            f"Out-of-order chunk {msg['seq']}, expected {slot['next_seq']}",
        )
        return
    text: str = msg["text"]
    size = len(text.encode("utf-8", "ignore"))
    if slot["bytes"] + size > HISTORY_IMPORT_MAX_BYTES:
        _history_staging(hass).pop(entry_id, None)
        connection.send_error(msg["id"], "invalid_format", "Upload is too large")
        return
    slot["chunks"].append(text)
    slot["bytes"] += size
    slot["next_seq"] += 1
    _send_result(connection, msg["id"], "history_import_chunk", {
        "received_bytes": slot["bytes"],
        "next_seq": slot["next_seq"],
    })


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/history_import_recorder",
        vol.Required("entry_id"): str,
        # Either a start date (what the panel sends: "import since <date>") or a plain
        # day count. `start_date` wins when both are present.
        vol.Optional("start_date"): vol.Any(None, str),
        vol.Optional("days", default=10): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=HISTORY_IMPORT_RECORDER_MAX_DAYS)
        ),
    }
)
@websocket_api.async_response
async def ws_history_import_recorder(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Fill the staging slot from Home Assistant's own recorder.

    The entity is always the device's configured power sensor, never client-supplied:
    this command reads arbitrary history, and letting the caller name the entity would
    make it an information-disclosure hole.

    Read one day at a time. `state_changes_during_period` has no row cap, and a single
    query over a long window materialises the whole result in one recorder-executor job.
    Home Assistant purges states after `purge_keep_days` (10 by default), so a request
    reaching further back simply returns fewer rows - that is reported, not an error.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    entity_id = getattr(manager, "power_sensor_entity_id", None)
    if not entity_id:
        connection.send_error(msg["id"], "not_found", "This device has no power sensor configured")
        return

    now = dt_util.now()
    days = int(msg.get("days") or 10)
    start_date = msg.get("start_date")
    if start_date:
        # "Import since <date>" - resolve to a day count so the windowed read below is
        # unchanged. The date is read as a LOCAL calendar day (it comes from a date
        # picker, where the user means their own midnight), and clamped to the supported
        # range: a future date reads today, an older one is capped at the maximum.
        try:
            parsed_date = dt_util.parse_date(str(start_date))
        except Exception:  # pylint: disable=broad-exception-caught
            parsed_date = None
        if parsed_date is None:
            connection.send_error(msg["id"], "invalid_format", "Not a valid start date")
            return
        span = (now.date() - parsed_date).days + 1
        days = max(1, min(HISTORY_IMPORT_RECORDER_MAX_DAYS, span))
    rows: list[tuple[float, float]] = []
    # Walk BACKWARDS from today, not forwards from the oldest requested day: the window can
    # now be years (a recorder configured to keep full-resolution states that long), and
    # starting at the far end would issue thousands of empty queries before reaching any
    # data - and, on truncation, would keep the OLDEST rows rather than the most recent.
    # `samples_from_readings` sorts, so the accumulation order does not matter downstream.
    empty_run = 0
    # The loop can stop early (empty-day run, or the row cap), so the requested `days` is
    # not what was read. Track the oldest window actually queried and report THAT, or the
    # panel would name a range it never looked at.
    oldest_queried = now
    queried_days = 0
    for day in range(1, days + 1):
        window_start = now - timedelta(days=day)
        window_end = now - timedelta(days=day - 1)
        day_rows = await _recorder_power(hass, entity_id, window_start, end_dt=window_end)
        oldest_queried = window_start
        queried_days += 1
        if day_rows:
            empty_run = 0
            rows.extend(day_rows)
        else:
            # Inside the retention window a day always yields at least the carried
            # start-time state, so a run of empty days means the recorder is purged past
            # here and every further query would be wasted.
            empty_run += 1
            if empty_run >= HISTORY_IMPORT_RECORDER_EMPTY_DAY_STOP:
                break
        if len(rows) > HISTORY_IMPORT_MAX_ROWS:
            break

    token = uuid.uuid4().hex
    _history_staging(hass)[entry_id] = {
        "token": token,
        "chunks": [],
        "bytes": 0,
        "next_seq": 0,
        "source": "recorder",
        "rows": rows[:HISTORY_IMPORT_MAX_ROWS],
        "entity_id": entity_id,
        # Carried into the parse report, so the review step says the read was cut.
        "truncated": len(rows) > HISTORY_IMPORT_MAX_ROWS,
    }
    _send_result(connection, msg["id"], "history_import_recorder", {
        "token": token,
        "rows": len(rows[:HISTORY_IMPORT_MAX_ROWS]),
        "entity_id": entity_id,
        # Both describe the window actually read, not the one asked for: counted from
        # the loop itself, so an early stop cannot report days that were never queried.
        "days": queried_days,
        "start_date": oldest_queried.date().isoformat(),
        "truncated": len(rows) > HISTORY_IMPORT_MAX_ROWS,
    })


def _history_samples(
    slot: dict[str, Any], entity_id: str | None, parser: Any = None
) -> Any:
    """Turn a staging slot into samples, whichever way it was filled.

    Executor-side: CSV parsing is pure Python over megabytes of text, so the scan task
    steps a ``HistoryCsvParser`` across jobs first and hands it in (``parser``).
    Returns either a ``(samples, report)`` pair or an ``{"error": ...}`` marker.
    """
    if slot.get("source") == "recorder":
        # `parser` is then a stepped RecorderReadings (audit PLAYGROUND-11).
        samples = (
            parser.result()
            if parser is not None
            else history_import.samples_from_readings(slot.get("rows") or [])
        )
        if len(samples) < 2:
            return {"error": "no_readings"}
        # The same report a CSV gets, so the review step shows the span, the peak,
        # the kW hint and the row-limit cut for a recorder read too (PLAYGROUND-11).
        report = history_import.ParsedHistory(
            samples=samples,
            entity_id=slot.get("entity_id"),
            rows_total=len(samples),
            rows_parsed=len(samples),
            truncated=bool(slot.get("truncated")),
        ).report()
        report["source"] = "recorder"
        return samples, report
    parsed = (
        parser.result()
        if parser is not None
        else history_import.parse_history_csv(
            "".join(slot.get("chunks") or []), entity_id=entity_id
        )
    )
    if isinstance(parsed, dict):
        return parsed
    report = parsed.report()
    report["source"] = "csv"
    return parsed.samples, report


async def _history_import_scan_task(
    hass: HomeAssistant, task: Any, entry_id: str, token: str
) -> None:
    """Replay a staged history stream, chunk by chunk, into candidate cycles.

    The full traces are held in the staging slot rather than in the task result: the
    result is served verbatim by `get_task_result`, and a few hundred traces would exceed
    the WebSocket frame cap and take the connection down. The result carries only preview
    rows, which is also all the review UI needs.
    """
    reg = task_registry.get_registry(hass)
    ctx = _playground_context(hass, entry_id)
    if ctx is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    manager, _store, base_config, options, _price = ctx
    slot = _history_slot(hass, entry_id, token)
    if slot is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="upload expired")
        return
    # Segmentation is only as good as the thresholds it runs with, and
    # `_playground_base_config`'s fallback uses the *scalar* defaults (min_off_gap 60,
    # off_delay 180) rather than the per-device ones - which on a dishwasher would cut
    # the stream at every drying pause and produce nothing but fragments. Refuse rather
    # than scan against the wrong thresholds.
    if not isinstance(getattr(getattr(manager, "detector", None), "config", None), CycleDetectorConfig):
        reg.finish(task, state=task_registry.STATE_ERROR, error="detector_unavailable")
        return
    try:
        entity_id = getattr(manager, "power_sensor_entity_id", None)
        parser = None
        if slot.get("source") != "recorder":
            # Parsed a slice per executor job, not in one (audit PLAYGROUND-11).
            parser = await hass.async_add_executor_job(functools.partial(
                history_import.HistoryCsvParser,
                "".join(slot.get("chunks") or []), entity_id=entity_id,
            ))
            reg.update(task, total=parser.rows_estimate)
            while not parser.finished:
                if task.cancel_requested:
                    reg.finish(task, state=task_registry.STATE_CANCELLED)
                    return
                await hass.async_add_executor_job(parser.step, history_import.PARSE_STEP_ROWS)
                reg.update(task, done=min(parser.rows_estimate, parser.out.rows_total))
        else:
            # The recorder rows are converted a slice per job too.
            parser = history_import.RecorderReadings(slot.get("rows") or [])
            reg.update(task, total=parser.rows_estimate)
            while not parser.finished:
                if task.cancel_requested:
                    reg.finish(task, state=task_registry.STATE_CANCELLED)
                    return
                await hass.async_add_executor_job(parser.step, history_import.PARSE_STEP_ROWS)
                reg.update(task, done=parser.done)
        parsed = await hass.async_add_executor_job(_history_samples, slot, entity_id, parser)
        if isinstance(parsed, dict):
            reg.finish(task, state=task_registry.STATE_ERROR, error=str(parsed.get("error")))
            return
        samples, report = parsed
        sampling_interval = options.get(CONF_SAMPLING_INTERVAL)
        # Blocks, gates and densification, a slice per job (audit PLAYGROUND-11).
        builder = history_import.ScanBuilder(
            samples, base_config,
            sampling_interval_s=sampling_interval, parse_report=report,
        )
        while not builder.finished:
            if task.cancel_requested:
                reg.finish(task, state=task_registry.STATE_CANCELLED)
                return
            await hass.async_add_executor_job(
                builder.step, history_import.SCAN_BUILD_STEP_SAMPLES
            )
        runner = builder.result()
        if isinstance(runner, dict):
            # A stream with nothing usable in it is a *result*, not a failure: the panel
            # explains which spans were skipped and why (six months of hourly averages
            # is the common case), so the user is not left staring at "0 cycles".
            reg.finish(task, state=task_registry.STATE_DONE, result={
                "segments": [],
                "skipped": runner.get("skipped") or [],
                "parse": runner.get("parse") or report,
                "found": 0,
                "error": runner.get("error"),
            })
            return
        reg.update(task, total=runner.total)
        while not runner.finished:
            if task.cancel_requested:
                break
            await hass.async_add_executor_job(runner.step, HISTORY_IMPORT_CHUNK_SAMPLES)
            reg.update(task, done=min(runner.total, runner.done))
        payload = await hass.async_add_executor_job(
            functools.partial(runner.finalize, partial=task.cancel_requested)
        )
        # The replay is unmatched, so a dishwasher's timeout finish banked its whole
        # end wait (audit PLAYGROUND-07): cut it by the programme it matches.
        if not task.cancel_requested:
            await history_import.async_import_tail_cuts(
                manager.profile_store, base_config, payload.get("cycles") or [],
                options, payload.get("segments") or [],
            )
        # A candidate that overlaps a cycle WashData already holds is that cycle
        # replayed from raw history: show it, but never pre-tick it (audit
        # PLAYGROUND-05 - 81% of them slipped past the exact start/duration key, and
        # a labelled copy double-weights the real cycle in its envelope).
        history_import.mark_already_recorded(
            payload.get("segments") or [],
            history_import.stored_intervals(manager.profile_store.iter_stored_cycles()),
        )
        # Split the payload: traces stay server-side, keyed by this task so a reconnect
        # can still apply them; only the preview rows travel.
        cycles = payload.pop("cycles", [])
        current = _history_slot(hass, entry_id, token)
        if current is not None:
            current["scan_task_id"] = task.id
            current["cycles"] = cycles
            current.pop("chunks", None)  # the raw text is no longer needed
            current.pop("rows", None)
        payload["token"] = token
        payload["settings"] = {
            "min_power": base_config.min_power,
            "off_delay": base_config.off_delay,
            "min_off_gap": base_config.min_off_gap,
            "device_type": base_config.device_type,
        }
        reg.finish(
            task,
            state=task_registry.STATE_CANCELLED if task.cancel_requested else task_registry.STATE_DONE,
            result=payload,
        )
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Task-level failure at WARNING like the other task runners, so it lands in the
        # default HA log and the panel Logs view (sub-step failures stay at debug).
        _LOGGER.warning("History-import scan failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/start_history_import_scan",
        vol.Required("entry_id"): str,
        vol.Required("token"): str,
    }
)
@callback
def ws_start_history_import_scan(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Kick off the replay as a detached, registry-tracked task."""
    entry_id: str = msg["entry_id"]
    if _get_manager(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    if _history_slot(hass, entry_id, msg["token"]) is None:
        connection.send_error(msg["id"], "not_found", "No upload in progress; start again")
        return
    reg = task_registry.get_registry(hass)
    task = reg.create(
        entry_id, "history_import", "Scanning power history",
        label_key="task.history_import.scanning",
    )
    _raw = hass.async_create_task(
        _history_import_scan_task(hass, task, entry_id, msg["token"])
    )
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "start_history_import_scan", {"task_id": task.id})


async def _history_import_apply_task(
    hass: HomeAssistant, task: Any, entry_id: str, scan_task_id: str, accept: list[int]
) -> None:
    """Persist the accepted candidates into ``backfill_cycles``."""
    reg = task_registry.get_registry(hass)
    manager = _get_manager(hass, entry_id)
    if manager is None:
        reg.finish(task, state=task_registry.STATE_ERROR, error="device unavailable")
        return
    scan = reg.get(scan_task_id)
    # Ownership check: get_task_result is not entry-scoped, so without this a stale or
    # foreign task id could write another device's cycles into this store.
    if scan is None or scan.entry_id != entry_id or scan.kind != "history_import":
        reg.finish(task, state=task_registry.STATE_ERROR, error="scan_expired")
        return
    try:
        async with _entry_write_lock(hass, entry_id):
            if _get_manager(hass, entry_id) is not manager:
                reg.finish(task, state=task_registry.STATE_ERROR, error="device reloaded")
                return
            # Validate AND read the staging slot under the lock, not before it: two
            # concurrent applies must not both pass a pre-lock check and each persist the
            # same candidate (one whose dedup_key is None bypasses existing_dedup_keys),
            # and the slot must not be swapped between the check and the read.
            slot = _history_staging(hass).get(entry_id)
            if not isinstance(slot, dict) or slot.get("scan_task_id") != scan_task_id:
                # The registry keeps only the last 30 finished tasks per entry, and an
                # entry reload clears the staging area, so a scan can legitimately be
                # gone by now.
                reg.finish(task, state=task_registry.STATE_ERROR, error="scan_expired")
                return
            cycles: list[dict[str, Any]] = list(slot.get("cycles") or [])
            # Dedupe the client-supplied indices (order-preserving): a repeated index
            # would visit the same candidate twice, and one whose dedup_key is None is
            # not caught by the in-loop dedup set, so it would store a second copy.
            wanted = list(dict.fromkeys(i for i in accept if 0 <= i < len(cycles)))
            if not wanted:
                reg.finish(
                    task, state=task_registry.STATE_DONE,
                    result={"imported": 0, "duplicates": 0},
                )
                return
            store = manager.profile_store
            target = store.get_backfill_cycles()
            room = max(0, HISTORY_IMPORT_MAX_TOTAL_CYCLES - len(target))
            existing = history_import.existing_dedup_keys(store.iter_stored_cycles())
            # Time overlap, not just the exact key: the same run recorded live and
            # replayed from history rarely agrees to the second (PLAYGROUND-05).
            intervals = history_import.stored_intervals(store.iter_stored_cycles())
            id_pool = {c.get("id") for c in target if isinstance(c, dict)}
            imported = 0
            duplicates = 0
            reg.update(task, total=len(wanted))
            for done, index in enumerate(wanted, start=1):
                if task.cancel_requested:
                    break
                raw = cycles[index]
                duration = history_import.effective_duration(raw)
                key = history_import.dedup_key(raw.get("start_time"), duration)
                if (key is not None and key in existing) or history_import.overlaps_stored(
                    raw.get("start_time"), duration, intervals
                ):
                    duplicates += 1
                    reg.update(task, done=done)
                    continue
                if imported >= room:
                    break
                stored = history_import.build_backfill_cycle(raw)
                store._add_cycle_data(  # noqa: SLF001 - the bulk insert primitive
                    stored,
                    target=target,
                    id_pool=id_pool,
                )
                if raw.get("tail_cut_s"):
                    # The banked-tail repair's own trim (PLAYGROUND-07).
                    store._apply_repaired_duration(stored, float(raw["tail_cut_s"]))  # noqa: SLF001
                if key is not None:
                    existing.add(key)
                interval = history_import.stored_intervals([{**raw, "duration": duration}])
                if interval:
                    intervals = sorted([*intervals, *interval])
                imported += 1
                reg.update(task, done=done)
            if imported:
                await store.async_save()
            # "capped" means the cap decided where we stopped - not the user cancelling,
            # and not the duplicates that were legitimately skipped.
            capped = not task.cancel_requested and imported < (len(wanted) - duplicates)
            # Read the total inside the lock: another task could append to the same
            # backfill list after the lock releases, inflating a count read outside it.
            total_backfill = len(target)
            # Consume the slot we applied - but only if it is still that same object.
            # A fresh upload (which does not take this lock) can replace the slot during
            # the async_save() await above; clearing unconditionally would discard it.
            if _history_staging(hass).get(entry_id) is slot:
                async_clear_history_import(hass, entry_id)
        manager.notify_update()
        reg.finish(
            task,
            state=task_registry.STATE_CANCELLED if task.cancel_requested else task_registry.STATE_DONE,
            result={
                "imported": imported,
                "duplicates": duplicates,
                "capped": capped,
                "total_backfill": total_backfill,
            },
        )
    except asyncio.CancelledError:
        reg.finish(task, state=task_registry.STATE_CANCELLED)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # A failed apply is a failed data write; log at WARNING like the other task
        # runners so it is visible in the default HA log and the panel Logs view.
        _LOGGER.warning("History-import apply failed for %s: %s", entry_id, exc)
        reg.finish(task, state=task_registry.STATE_ERROR, error=str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/apply_history_import",
        vol.Required("entry_id"): str,
        vol.Required("scan_task_id"): str,
        vol.Required("accept"): list,
    }
)
@callback
def ws_apply_history_import(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Persist the candidates the user kept, as a detached, registry-tracked task."""
    entry_id: str = msg["entry_id"]
    if _get_manager(hass, entry_id) is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    accept: list[int] = []
    for item in msg["accept"][:HISTORY_IMPORT_MAX_SEGMENTS]:
        try:
            accept.append(int(item))
        except (TypeError, ValueError, OverflowError):
            continue
    reg = task_registry.get_registry(hass)
    task = reg.create(
        entry_id, "history_import_apply", "Importing cycles",
        label_key="task.history_import.importing",
    )
    _raw = hass.async_create_task(
        _history_import_apply_task(hass, task, entry_id, msg["scan_task_id"], accept)
    )
    if _raw is not None:
        reg.link_asyncio_task(task.id, _raw)
    _send_result(connection, msg["id"], "apply_history_import", {"task_id": task.id})
