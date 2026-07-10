"""WebSocket API commands for the WashData full-screen panel."""
from __future__ import annotations

import collections
import functools
import json
import logging
import time
from typing import Any

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_NAME,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_DEVICE_TYPE,
    CONF_DOOR_SENSOR_ENTITY,
    CONF_DURATION_TOLERANCE,
    CONF_END_ENERGY_THRESHOLD,
    CONF_END_REPEAT_COUNT,
    CONF_EXTERNAL_END_TRIGGER,
    CONF_LEARNING_CONFIDENCE,
    CONF_LINKED_DEVICE,
    CONF_MAINTENANCE_REMINDER_CYCLES,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_PROFILE_MIN_WARMUP_CYCLES,
    CONF_PUMP_STUCK_DURATION,
    CONF_RUNNING_DEAD_ZONE,
    CONF_SAMPLING_INTERVAL,
    CONF_SMOOTHING_WINDOW,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_SWITCH_ENTITY,
    CONF_WATCHDOG_INTERVAL,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_MAINTENANCE_REMINDER_CYCLES,
    DEFAULT_OFF_DELAY,
    DEFAULT_OFF_DELAY_BY_DEVICE,
    DEVICE_TYPE_PUMP,
    MAINTENANCE_EVENT_TYPES,
    DEVICE_TYPES,
    DOMAIN,
    ENABLE_ML_SUGGESTIONS,
    ENABLE_ML_TRAINING,
    SHOW_ML_LAB,
    STATE_COLORS,
)
from . import playground
from .cycle_detector import CycleDetectorConfig

_LOGGER = logging.getLogger(__name__)

# Fields too large or not serialisable to send over WebSocket.
_CYCLE_STRIP_KEYS = frozenset({"power_data", "power_trace", "debug_data", "samples"})

# Settings keys that can be staged from suggestions. Mirrors the OptionsFlow's
# _suggestion_keys_to_apply so the panel and the flow agree on what is tunable.
_SUGGESTION_KEYS: tuple[str, ...] = (
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_SAMPLING_INTERVAL,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_MIN_OFF_GAP,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_END_ENERGY_THRESHOLD,
    CONF_RUNNING_DEAD_ZONE,
    # Stage 1 detection suggestions
    CONF_SMOOTHING_WINDOW,
    CONF_START_DURATION_THRESHOLD,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_LEARNING_CONFIDENCE,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_END_REPEAT_COUNT,
)

# Suggestion keys coerced to int when applied (mirrors the OptionsFlow).
_SUGGESTION_INT_KEYS: frozenset[str] = frozenset({
    CONF_OFF_DELAY,
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_MIN_OFF_GAP,
    CONF_RUNNING_DEAD_ZONE,
    CONF_SMOOTHING_WINDOW,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_END_REPEAT_COUNT,
})


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
    except (TypeError, ValueError):
        return str(suggested) == str(current)

# Settings surfaced in the ML Lab side-by-side comparison: key -> (label, unit).
# Order defines display order. Only keys either engine produces are shown.
_ML_COMPARE_SETTINGS: tuple[tuple[str, str, str], ...] = (
    (CONF_OFF_DELAY, "Off Delay", "s"),
    (CONF_END_REPEAT_COUNT, "End Repeat Count", "x"),
    (CONF_AUTO_LABEL_CONFIDENCE, "Auto-label Confidence", ""),
    (CONF_STOP_THRESHOLD_W, "Stop Threshold", "W"),
    (CONF_START_THRESHOLD_W, "Start Threshold", "W"),
    (CONF_END_ENERGY_THRESHOLD, "End Energy Threshold", "Wh"),
    (CONF_RUNNING_DEAD_ZONE, "Running Dead Zone", "s"),
    (CONF_MIN_POWER, "Min Power", "W"),
    (CONF_MIN_OFF_GAP, "Min Off Gap", "s"),
    (CONF_DURATION_TOLERANCE, "Duration Tolerance", ""),
)


def _downsample(samples: Any, max_points: int = 240) -> list[list[float]]:
    """Reduce a [(offset_s, watts), ...] series to <= max_points via striding.

    Keeps the first and last samples so the time axis is preserved. Power curves
    can hold thousands of points; the panel only needs enough to draw a faithful
    line, and WebSocket payloads should stay lean.
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

    step = n / float(max_points)
    out: list[list[float]] = []
    last_i = -1
    idx = 0.0
    while int(idx) < n:
        i = int(idx)
        if i != last_i:
            out.append(_pt(pairs[i]))
            last_i = i
        idx += step
    last_pt = _pt(pairs[-1])
    if not out or out[-1][0] != last_pt[0]:
        out.append(last_pt)
    return out


async def _recorder_power(hass: HomeAssistant, entity_id: str, start_dt: Any) -> list[tuple[float, float]]:
    """Raw (unix_ts, watts) readings for entity_id from start_dt to now, via the recorder."""
    try:
        from homeassistant.components.recorder import (  # pylint: disable=import-outside-toplevel
            get_instance,
            history,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return []
    end_dt = dt_util.utcnow()

    def _query() -> list[tuple[float, float]]:
        res = history.state_changes_during_period(
            hass, start_dt, end_dt, entity_id, include_start_time_state=True
        )
        rows: list[tuple[float, float]] = []
        for s in res.get(entity_id, []) or []:
            try:
                rows.append((s.last_changed.timestamp(), round(float(s.state), 1)))
            except (ValueError, TypeError):
                continue
        return rows

    try:
        return await get_instance(hass).async_add_executor_job(_query)
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def _cycle_kwh(c: dict[str, Any]) -> float | None:
    """Cycle energy in kWh. Cycles store energy as ``energy_wh``; convert."""
    wh = c.get("energy_wh")
    if wh is not None:
        try:
            return round(float(wh) / 1000.0, 4)
        except (TypeError, ValueError):
            pass
    return c.get("energy_kwh")


class _RankingMatchResult:
    """Minimal MatchResult-like wrapper around a stored feedback ranking.

    get_match_candidates_summary only reads .ranking and .expected_duration, so
    this lets us rebuild the candidates table from a pending feedback record
    without re-running a live match (mirrors the OptionsFlow's _MatchResultLike).
    """

    def __init__(self, ranking: list[Any], expected_duration: float) -> None:
        self.ranking = ranking or []
        self.expected_duration = expected_duration or 0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_manager(hass: HomeAssistant, entry_id: str) -> Any | None:
    domain_data: dict[str, Any] = hass.data.get(DOMAIN, {})
    return domain_data.get(entry_id) if isinstance(domain_data, dict) else None


def _get_entry(hass: HomeAssistant, entry_id: str) -> Any | None:
    return next(
        (e for e in hass.config_entries.async_entries(DOMAIN) if e.entry_id == entry_id),
        None,
    )


def _err_not_found(connection: websocket_api.ActiveConnection, msg_id: int, entry_id: str) -> None:
    connection.send_error(msg_id, "not_found", f"No active WashData manager for entry {entry_id!r}")


def _strip_cycle(c: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in c.items() if k not in _CYCLE_STRIP_KEYS}


# Option keys that are identity/transient churn and are never recorded in the
# settings changelog (D7): name/title edits flow through the separate `title`
# kwarg, and suggestion application uses its own apply_suggestions command.
_CHANGELOG_SKIP_KEYS = frozenset({CONF_NAME})


def _json_safe(value: Any) -> Any:
    """Best-effort coercion of an option value to a JSON-serializable form."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
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


# ─── Panel config + RBAC ────────────────────────────────────────────────────────

_PANEL_STORE_VERSION = 1
_PANEL_STORE_FILE = "ha_washdata_panel"
_PANEL_DATA_KEY = "ha_washdata_panel_cfg"

_LEVEL_RANK = {"none": 0, "read": 1, "edit": 2, "full": 3}
_PANEL_TABS = ("status", "history", "profiles", "settings", "tools", "panel", "ml_lab", "playground")

# Commands that require 'full' (destructive or full-data export/import).
_FULL_COMMANDS = frozenset({
    "wipe_history", "import_config", "export_config", "clear_debug_data", "reprocess_history",
    "trigger_ml_training",
})
# Commands allowed for any authenticated user regardless of device permissions.
_OPEN_COMMANDS = frozenset({
    "get_constants", "get_panel_config", "set_user_prefs",
})
# Admin-only commands.
_ADMIN_COMMANDS = frozenset({"set_panel_config", "get_logs"})
# Mutating commands intentionally allowed at the 'read' level. Picking the live
# program is a benign runtime action (it changes detection, not stored data), so
# read users may use the Status program selector. The Playground simulation is a
# read-only what-if replay (it never persists anything) whose name does not start
# with get_, so it is whitelisted here to gate at the 'read' level.
_READ_WRITE_COMMANDS = frozenset({"set_program", "run_playground_simulation"})

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
        logging.getLogger("custom_components.ha_washdata").addHandler(handler)
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
        except (TypeError, ValueError):
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

@callback
def async_register_commands(hass: HomeAssistant) -> None:
    """Register all WebSocket commands for the WashData panel.

    Every handler is wrapped in _guard so RBAC is enforced centrally and no
    command can accidentally ship unprotected.
    """
    handlers = [
        ws_get_devices, ws_get_device_cycles,
        # Settings
        ws_get_options, ws_set_options, ws_get_settings_changelog,
        # Profiles
        ws_get_profiles, ws_create_profile, ws_rename_profile, ws_delete_profile,
        ws_rebuild_envelopes, ws_get_profile_phases, ws_set_profile_phases,
        # Profile groups (Stage 5)
        ws_get_profile_groups, ws_save_profile_group, ws_rename_profile_group, ws_delete_profile_group,
        # Maintenance log (Group E)
        ws_get_maintenance_log, ws_add_maintenance_event, ws_delete_maintenance_event,
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
        # Shared constants
        ws_get_constants,
        # Suggestions
        ws_get_suggestions, ws_apply_suggestions, ws_clear_suggestions, ws_run_suggestion_analysis,
        # Cycle curve / interactive editing
        ws_get_cycle_power_data, ws_trim_cycle, ws_analyze_split, ws_apply_split, ws_apply_merge,
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
        # On-device ML training (status + manual trigger + matcher-tuning revert + models revert)
        ws_get_ml_training_status, ws_trigger_ml_training, ws_revert_matching_config,
        ws_revert_ml_models,
        # Cycle controls (pause / resume / force-stop)
        ws_pause_cycle, ws_resume_cycle, ws_terminate_cycle,
        # Playground (F3): headless what-if replay + DTW visualizer
        ws_run_playground_simulation, ws_get_dtw_debug,
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
            "current_program": None,
            "time_remaining_s": None,
            "total_duration_s": None,
            "current_power_w": None,
            "cycle_progress_pct": None,
            "suggestions_count": 0,
            "feedback_count": 0,
            "recording": False,
            "is_user_paused": False,
            "manual_program": False,
            "options": dict(entry.options),
        }

        if manager is not None:
            try:
                detector = getattr(manager, "detector", None)
                if detector is not None:
                    info["detector_state"] = detector.state
                    info["sub_state"] = detector.sub_state

                program: str | None = getattr(manager, "_current_program", None)
                if program in (None, "off", "unknown", "detecting...", "restored..."):
                    program = None
                info["current_program"] = program
                info["manual_program"] = bool(getattr(manager, "manual_program_active", False))

                info["time_remaining_s"] = getattr(manager, "_time_remaining", None)
                info["total_duration_s"] = getattr(manager, "_total_duration", None)

                power = getattr(manager, "_current_power", None)
                info["current_power_w"] = round(float(power), 2) if power is not None else None

                progress = getattr(manager, "_cycle_progress", None)
                if progress is not None:
                    info["cycle_progress_pct"] = round(float(progress), 1)

                store = getattr(manager, "profile_store", None)
                if store is not None:
                    try:
                        raw = store.get_suggestions() or {}
                        info["suggestions_count"] = sum(
                            1 for k in _SUGGESTION_KEYS
                            if isinstance(raw.get(k), dict) and raw[k].get("value") is not None
                        )
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                    try:
                        info["feedback_count"] = len(store.get_pending_feedback() or {})
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                info["is_user_paused"] = bool(getattr(manager, "is_user_paused", False))
                recorder = getattr(manager, "recorder", None)
                if recorder is not None:
                    info["recording"] = bool(getattr(recorder, "is_recording", False))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("Error reading manager state for entry %s: %s", entry.entry_id, exc)

        devices.append(info)

    connection.send_result(msg["id"], {"devices": devices})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_device_cycles",
        vol.Required("entry_id"): str,
        vol.Optional("limit", default=50): vol.All(int, vol.Range(min=1, max=200)),
        vol.Optional("offset", default=0): vol.All(int, vol.Range(min=0)),
    }
)
@callback
def ws_get_device_cycles(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return a page of recent cycles for a device, stripping large binary fields.

    Cycles are returned most-recent-first and sliced ``[offset : offset+limit]``
    so the panel can page. ``total`` is the device's full cycle count and
    ``has_more`` is True when cycles remain beyond the returned window.
    """
    entry_id: str = msg["entry_id"]
    limit: int = msg.get("limit", 50)
    offset: int = msg.get("offset", 0)

    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    cycles: list[dict[str, Any]] = []
    total = 0
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
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error fetching cycles for entry %s: %s", entry_id, exc)

    has_more = (offset + len(cycles)) < total
    connection.send_result(
        msg["id"],
        {
            "entry_id": entry_id,
            "cycles": cycles,
            "total": total,
            "has_more": has_more,
        },
    )


# ─── Settings ─────────────────────────────────────────────────────────────────

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
    options = {**entry.data, **entry.options}
    connection.send_result(msg["id"], {"options": options})


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
    new_options = {**entry.data, **entry.options, **msg["options"]}

    # Mirror the OptionsFlow save-time normalization so the panel can never
    # persist stale or invalid values:
    #  - a cleared selector (entity / linked device / trigger) becomes None so
    #    the link or subscription is removed rather than left dangling;
    #  - pump-only keys are dropped for non-pump device types;
    #  - the transient "apply suggestions" flag is never stored.
    for key in (
        CONF_EXTERNAL_END_TRIGGER,
        CONF_DOOR_SENSOR_ENTITY,
        CONF_LINKED_DEVICE,
        CONF_SWITCH_ENTITY,
    ):
        if key in new_options and not new_options[key]:
            new_options[key] = None

    if new_options.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE) != DEVICE_TYPE_PUMP:
        new_options.pop(CONF_PUMP_STUCK_DURATION, None)

    update_kwargs: dict[str, Any] = {"options": new_options}
    if CONF_NAME in new_options and new_options[CONF_NAME]:
        update_kwargs["title"] = new_options[CONF_NAME].strip()

    # Settings change history (D7): diff the pre-update effective options against
    # the post-normalization values, but only for keys the user actually
    # submitted, and persist BEFORE async_update_entry (which schedules a reload
    # that rebuilds the store). A changelog failure must never block the save.
    try:
        old_effective = {**entry.data, **entry.options}
        submitted_post = {
            k: new_options[k] for k in msg["options"] if k in new_options
        }
        changes = _diff_option_changes(old_effective, submitted_post)
        if changes:
            manager = _get_manager(hass, msg["entry_id"])
            store = getattr(manager, "profile_store", None) if manager else None
            if store is not None:
                await store.async_record_settings_changes(changes)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug(
            "Settings changelog recording failed for %s: %s", msg["entry_id"], exc
        )

    hass.config_entries.async_update_entry(entry, **update_kwargs)
    connection.send_result(msg["id"], {"success": True})


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

    connection.send_result(msg["id"], {"changelog": changelog})


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

    def _compute_stats() -> dict[str, Any]:
        profiles: list[dict[str, Any]] = []
        try:
            profiles = manager.profile_store.list_profiles()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Error listing profiles for %s: %s", entry_id, exc)

        health: dict[str, dict] = {}
        try:
            health = manager.profile_store.compute_profile_health()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        trends: dict[str, dict] = {}
        try:
            trends = manager.profile_store.compute_profile_trends()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        coverage_gaps: dict[str, Any] = {}
        try:
            coverage_gaps = manager.profile_store.suggest_coverage_gaps()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        advisories: list[dict] = []
        try:
            advisories = manager.profile_store.compute_profile_advisories()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        return {
            "profiles": profiles,
            "profile_health": health,
            "profile_trends": trends,
            "coverage_gaps": coverage_gaps,
            "profile_advisories": advisories,
        }

    stats = await hass.async_add_executor_job(_compute_stats)
    connection.send_result(msg["id"], stats)


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
        connection.send_result(msg["id"], {"success": True, "name": name})
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
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True})
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
    """Return profile groups with member lists, cohesion, and near-duplicate
    suggestions the user can act on."""
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
    suggestions = await hass.async_add_executor_job(store.suggest_profile_groups)
    connection.send_result(msg["id"], {
        "groups": groups,
        "min_cohesion": GROUP_MIN_COHESION,
        "suggestions": suggestions,
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
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/rebuild_envelopes", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_rebuild_envelopes(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Rebuild power-profile envelopes for all profiles."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        await manager.profile_store.async_rebuild_all_envelopes()
        connection.send_result(msg["id"], {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


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

    connection.send_result(msg["id"], {"phases": phases or []})


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
        connection.send_result(msg["id"], {"success": True})
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
    reminders = dict(DEFAULT_MAINTENANCE_REMINDER_CYCLES)
    cfg = manager.config_entry.options.get(CONF_MAINTENANCE_REMINDER_CYCLES)
    if isinstance(cfg, dict):
        reminders.update(cfg)
    connection.send_result(msg["id"], {
        "log": manager.profile_store.get_maintenance_log(),
        "due": manager.maintenance_due,
        "event_types": list(MAINTENANCE_EVENT_TYPES),
        "reminders": reminders,
    })


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
        connection.send_result(msg["id"], {"success": True, "event": event})
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
        connection.send_result(msg["id"], {"success": removed})
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
            await manager.profile_store.create_profile(new_profile_name.strip(), cycle_id)
        else:
            await manager.profile_store.assign_profile_to_cycle(cycle_id, profile_name)
        manager.notify_update()
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/auto_label_cycles",
        vol.Required("entry_id"): str,
        vol.Optional("confidence_threshold", default=0.75): vol.All(
            vol.Coerce(float), vol.Range(min=0.5, max=0.95)
        ),
    }
)
@websocket_api.async_response
async def ws_auto_label_cycles(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Auto-label all cycles with matched profiles above the confidence threshold."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    threshold: float = msg.get("confidence_threshold", 0.75)
    try:
        await manager.profile_store.auto_label_cycles(threshold, overwrite=True)
        manager.notify_update()
        connection.send_result(msg["id"], {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


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

    connection.send_result(msg["id"], {"phases": phases, "device_type": device_type})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/create_phase",
        vol.Required("entry_id"): str,
        vol.Required("device_type"): str,
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

    try:
        await manager.profile_store.async_create_custom_phase(
            msg["device_type"], msg["name"], msg.get("description", "")
        )
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"state": "unavailable"})
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

    connection.send_result(msg["id"], info)


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
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True})
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
    if not recorder or not getattr(recorder, "last_run", None):
        connection.send_error(msg["id"], "no_recording", "No completed recording to process")
        return

    last_run: dict[str, Any] = recorder.last_run
    data = last_run.get("data", [])
    head_trim: float = msg.get("head_trim", 0.0)
    tail_trim: float = msg.get("tail_trim", 0.0)
    profile_name: str = msg["profile_name"].strip()
    save_mode: str = msg["save_mode"]

    if not profile_name:
        connection.send_error(msg["id"], "invalid_format", "Profile name must not be empty")
        return

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
        duration = max(0.0, keep_end - keep_start)

        trimmed_data = [
            (dt_util.utc_from_timestamp(t).isoformat(), p)
            for t, p in parsed
            if keep_start <= t <= keep_end
        ]

        cycle_data: dict[str, Any] = {
            "id": f"rec_{int(time.time())}",
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
        await recorder.clear_last_run()
        manager.notify_update()

        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True})
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

    connection.send_result(msg["id"], {"feedbacks": feedbacks})


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
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True, "dismissed": len(pending)})
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
        connection.send_result(msg["id"], {"stats": stats})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/reprocess_history", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_reprocess_history(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Full processing pass over historical data.

    Runs, in order: reprocess (rematch + rebuild envelopes) -> backfill the
    recorded/golden flag -> refresh tuning suggestions -> on-device ML training
    (when enabled) -> recompute per-cycle ML health against the resulting model.
    This is the single "Process history" trigger in Diagnostics.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    store = manager.profile_store
    summary: dict[str, Any] = {"success": True}
    try:
        summary["count"] = await store.async_reprocess_all_data()

        # Recorded cycles are golden references - backfill the single flag.
        try:
            summary["golden_backfilled"] = await store.async_backfill_recorded_golden()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("golden backfill failed for %s: %s", entry_id, exc)

        # Refresh tuning suggestions.
        learning = getattr(manager, "learning_manager", None)
        if learning is not None and hasattr(learning, "async_run_full_analysis"):
            try:
                res = await learning.async_run_full_analysis()
                summary["suggestions"] = (res or {}).get("count", 0)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("suggestion analysis failed for %s: %s", entry_id, exc)

        # On-device ML training (bypasses schedule guards; safe when disabled).
        if ENABLE_ML_TRAINING:
            try:
                tr = await manager.async_run_ml_training(force=True)
                summary["ml_training"] = {
                    "ok": bool(tr.get("ok")),
                    "promoted": tr.get("promoted", []),
                    "reason": tr.get("reason"),
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("ML training failed for %s: %s", entry_id, exc)

        # Recompute per-cycle health against the (possibly retrained) model.
        # Skip when training already recomputed it (a promotion refreshes health).
        if not (summary.get("ml_training", {}) or {}).get("promoted"):
            try:
                summary["health_recomputed"] = await manager.async_recompute_cycle_health()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("health recompute failed for %s: %s", entry_id, exc)

        # Re-validate the manager is still live after a potentially long chain of
        # awaits (reprocess + ML training can take tens of seconds).  If the entry
        # was reloaded in the meantime the original manager is detached and its
        # store saves would overwrite the new manager's data.
        current_manager = _get_manager(hass, entry_id)
        if current_manager is not manager:
            _LOGGER.warning(
                "Manager replaced during reprocess for %s; skipping notify", entry_id
            )
            connection.send_result(msg["id"], summary)
            return
        manager.notify_update()
        connection.send_result(msg["id"], summary)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


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
        connection.send_result(msg["id"], {"success": True, "count": count})
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
        manager.notify_update()
        connection.send_result(msg["id"], {"success": True})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


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
        # Offload serialization to executor — power traces can be megabytes
        json_str = await hass.async_add_executor_job(
            lambda: json.dumps(payload, indent=2)
        )
        connection.send_result(msg["id"], {"json_data": json_str})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


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

    entry = _get_entry(hass, entry_id)
    try:
        payload = await hass.async_add_executor_job(json.loads, msg["json_data"])
        config_updates = await manager.profile_store.async_import_data(payload)

        # Re-validate after awaits — entry may have been reloaded during import.
        current_manager = _get_manager(hass, entry_id)
        if current_manager is not manager:
            _LOGGER.warning(
                "Manager replaced during import for %s; aborting notify", entry_id
            )
            connection.send_result(msg["id"], {"success": True})
            return

        if entry and config_updates:
            entry_data_updates = config_updates.get("entry_data", {})
            entry_options_updates = config_updates.get("entry_options", {})
            if entry_data_updates or entry_options_updates:
                new_options = {**entry.data, **entry.options}
                new_options.update(entry_options_updates)
                hass.config_entries.async_update_entry(entry, options=new_options)

        manager.notify_update()
        connection.send_result(msg["id"], {"success": True})
    except json.JSONDecodeError as exc:
        connection.send_error(msg["id"], "invalid_json", str(exc))
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
    connection.send_result(
        msg["id"],
        {
            "device_types": device_types,
            "state_colors": dict(STATE_COLORS),
            "ml_lab_enabled": SHOW_ML_LAB,
            "ml_suggestions_enabled": ENABLE_ML_SUGGESTIONS,
            "ml_training_available": ENABLE_ML_TRAINING,
            "PROFILE_MIN_WARMUP_CYCLES": CONF_PROFILE_MIN_WARMUP_CYCLES,
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
        raw: dict[str, Any] = manager.profile_store.get_suggestions() or {}
        for key in _SUGGESTION_KEYS:
            item = raw.get(key)
            if not isinstance(item, dict) or item.get("value") is None:
                continue
            val = item["value"]
            suggested = (
                int(float(val)) if key in _SUGGESTION_INT_KEYS else round(float(val), 4)
            )
            current = merged.get(key)
            # Hide suggestions that would not change the current value.
            if _suggestion_equivalent(suggested, current):
                continue
            out.append(
                {
                    "key": key,
                    "suggested": suggested,
                    "reason": item.get("reason", ""),
                    "current": current,
                    "updated": item.get("updated"),
                }
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error reading suggestions for %s: %s", entry_id, exc)

    connection.send_result(msg["id"], {"suggestions": out})


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
        raw: dict[str, Any] = manager.profile_store.get_suggestions() or {}
        updates: dict[str, Any] = {}
        for key in msg["keys"]:
            if key not in _SUGGESTION_KEYS:
                continue
            item = raw.get(key)
            if not isinstance(item, dict) or item.get("value") is None:
                continue
            val = item["value"]
            updates[key] = (
                int(float(val)) if key in _SUGGESTION_INT_KEYS else float(val)
            )

        if updates:
            # Clear before updating the entry: async_update_entry schedules a
            # reload that rebuilds the store, so persist the cleared state first.
            cycle_count = len(manager.profile_store.get_past_cycles())
            manager.profile_store.set_suggestion_apply_cycle_count(cycle_count)
            await manager.profile_store.clear_suggestions()
            new_options = {**entry.data, **entry.options, **updates}
            hass.config_entries.async_update_entry(entry, options=new_options)
            manager.notify_update()

        connection.send_result(
            msg["id"], {"success": True, "applied": list(updates.keys())}
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
        connection.send_result(msg["id"], {"success": True})
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
        connection.send_result(msg["id"], {"success": True, **(result or {})})
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
        cycle = next(
            (c for c in store.get_past_cycles() if c.get("id") == cycle_id), None
        )
        if cycle:
            meta = {
                "start_time": cycle.get("start_time"),
                "end_time": cycle.get("end_time"),
                "duration": cycle.get("duration"),
                "profile_name": cycle.get("profile_name"),
                "status": cycle.get("status"),
                "energy_kwh": _cycle_kwh(cycle),
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
            meta["artifacts"] = artifacts or []
            # HA restart gaps recorded during this cycle (for panel shading).
            meta["restart_gaps"] = cycle.get("restart_gaps") or []
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error getting cycle power data %s: %s", cycle_id, exc)

    connection.send_result(
        msg["id"],
        {
            "cycle_id": cycle_id,
            "samples": _downsample(samples),
            "full_duration_s": round(float(samples[-1][0]), 1) if samples else 0.0,
            **meta,
        },
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/trim_cycle",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Required("start_s"): vol.Coerce(float),
        vol.Required("end_s"): vol.Coerce(float),
    }
)
@websocket_api.async_response
async def ws_trim_cycle(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Trim a cycle's power data to the [start_s, end_s] offset window."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        ok = await manager.profile_store.trim_cycle_power_data(
            msg["cycle_id"], float(msg["start_s"]), float(msg["end_s"])
        )
        if ok:
            manager.notify_update()
            connection.send_result(msg["id"], {"success": True})
        else:
            connection.send_error(
                msg["id"], "trim_failed", "Trim produced no data or cycle not found"
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


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
        connection.send_result(
            msg["id"],
            {
                "segments": [
                    [round(float(a), 1), round(float(b), 1)] for a, b in (segs or [])
                ],
                "split_offsets": split_offsets,
                "samples": _downsample(samples),
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
@websocket_api.async_response
async def ws_apply_split(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Split a cycle at the given offsets, optionally labeling each segment."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    store = manager.profile_store
    cycle_id: str = msg["cycle_id"]
    try:
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

        new_ids = await store.apply_split_interactive(cycle_id, segments)
        await store.async_rebuild_all_envelopes()
        manager.notify_update()
        connection.send_result(msg["id"], {"success": True, "new_ids": new_ids})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/apply_merge",
        vol.Required("entry_id"): str,
        vol.Required("cycle_ids"): [str],
        vol.Optional("target_profile"): vol.Any(str, None),
        vol.Optional("new_profile_name"): vol.Any(str, None),
    }
)
@websocket_api.async_response
async def ws_apply_merge(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Merge two or more cycles into one, optionally labeling the result."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    store = manager.profile_store
    ids: list[str] = msg["cycle_ids"]
    if len(ids) < 2:
        connection.send_error(
            msg["id"], "merge_failed", "Select at least two cycles to merge"
        )
        return

    target = msg.get("target_profile")
    try:
        if target == "__create_new__":
            name = (msg.get("new_profile_name") or "").strip()
            if not name:
                connection.send_error(
                    msg["id"], "invalid_format", "New profile name required"
                )
                return
            await store.create_profile_standalone(name)
            target = name
        elif target in ("", "none", "__none__"):
            target = None

        new_id = await store.apply_merge_interactive(ids, target)
        if not new_id:
            connection.send_error(msg["id"], "merge_failed", "Cycles could not be merged")
            return

        await store.async_rebuild_all_envelopes()
        manager.notify_update()
        connection.send_result(msg["id"], {"success": True, "new_id": new_id})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        connection.send_error(msg["id"], "unknown_error", str(exc))


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

    connection.send_result(msg["id"], {"envelope": env_out})


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
    connection.send_result(msg["id"], {"cycles": result})


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
    connection.send_result(msg["id"], out)


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
        connection.send_result(msg["id"], {"success": True})
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
    for k in ("show_expected", "show_raw", "show_raw_active", "show_debug", "onboarding_dismissed"):
        if k in p:
            cur[k] = bool(p[k])
    # F2: per-user Basic/Advanced settings disclosure level.
    if p.get("settings_level") in ("basic", "advanced"):
        cur["settings_level"] = p["settings_level"]
    prefs[user.id] = cur
    await _save_panel_data(hass)
    connection.send_result(msg["id"], {"success": True})


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
        mr = getattr(manager, "_last_match_result", None)
        conf = getattr(manager, "_last_match_confidence", None)
        out["confidence"] = round(float(conf), 4) if conf is not None else None
        out["ambiguous"] = bool(getattr(manager, "_last_match_ambiguous", False))
        if mr is not None:
            out["candidates"] = manager.profile_store.get_match_candidates_summary(mr, 5)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error building match debug for %s: %s", entry_id, exc)

    connection.send_result(msg["id"], out)


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
        else:
            manager.set_manual_program(prog)
        manager.notify_update()
        connection.send_result(msg["id"], {"success": True})
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
            live = [[round((t - start_dt).total_seconds(), 1), round(float(p), 1)] for t, p in trace]
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

    connection.send_result(msg["id"], out)


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
    connection.send_result(msg["id"], {"logs": recs[-limit:]})


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


def _health_model_sig(store: Any) -> str:
    """Signature of the active health/end models, so persisted per-cycle health
    can be invalidated (recomputed) whenever the model changes or is retrained.
    """
    versions = store.get_ml_model_versions() or {}
    parts = []
    for cap in ("quality", "end"):
        v = versions.get(cap)
        ts = v.get("trained_at") if isinstance(v, dict) else None
        parts.append(f"{cap}:{ts or 'base'}")
    return "|".join(parts)


def _compute_ml_comparison(
    store: Any, current_off_delay: int, *, force_recompute: bool = False
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
        return {"enabled": False, "error": "ML models not available", "cycles": [], "settings_comparison": {}}

    # Prefer on-device trained models when present, else the embedded baseline.
    quality_score_fn, quality_source = resolve_scorer("quality", store)
    end_score_fn, end_source = resolve_scorer("end", store)
    if quality_score_fn is None and end_score_fn is None:
        return {"enabled": False, "error": "ML models not available", "cycles": [], "settings_comparison": {}}

    model_sig = _health_model_sig(store)
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

    # --- Evaluate cycles (most recent 200 for display; all for pause analysis) ---
    intra_pauses: list[float] = []
    evaluated: list[dict[str, Any]] = []

    for cycle in cycles:
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

        points = decompress_power_data(cycle)

        # Collect intra-cycle pauses for off_delay recommendation (all cycles).
        if points and len(points) >= 4 and status == "completed":
            powers = [p for _, p in points]
            peak = max(powers) if powers else 0.0
            low_thresh = max(1.0, 0.05 * peak)
            in_pause = False
            pause_start_s = 0.0
            for offset_s, pwr in points:
                if not in_pause and pwr < low_thresh:
                    in_pause = True
                    pause_start_s = offset_s
                elif in_pause and pwr >= low_thresh:
                    pause_dur = offset_s - pause_start_s
                    if 5.0 <= pause_dur <= 1800.0:
                        intra_pauses.append(pause_dur)
                    in_pause = False

        # Only score and display the most recent 200 cycles.
        if len(evaluated) >= 200:
            continue

        # Reuse persisted per-cycle health when it was computed against the
        # current model (unless a recompute is forced). This is what keeps the
        # panel from re-scoring every cycle on every load.
        cached = cycle.get("ml_health")
        if (
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
            "has_power_data": len(points) > 0,
            "events": events,
            "ml_review": cycle.get("ml_review") or {},
        })

    # --- Settings comparison: off_delay ---
    settings_comparison: dict[str, Any] = {}
    ml_off_delay: int | None = None
    pause_p95: float | None = None
    if intra_pauses:
        intra_pauses.sort()
        p95_idx = min(int(len(intra_pauses) * 0.95), len(intra_pauses) - 1)
        pause_p95 = intra_pauses[p95_idx]
        ml_off_delay = max(60, int(pause_p95) + 60)

    original_suggestions = store.get_suggestions() or {}
    orig_off = original_suggestions.get(CONF_OFF_DELAY)
    settings_comparison["off_delay"] = {
        "key": CONF_OFF_DELAY,
        "label": "Off Delay",
        "unit": "s",
        "current_value": current_off_delay,
        "original_suggestion": orig_off.get("value") if isinstance(orig_off, dict) else None,
        "original_reason": orig_off.get("reason") if isinstance(orig_off, dict) else None,
        "ml_recommendation": ml_off_delay,
        "ml_reasoning": (
            f"p95 intra-cycle pause: {int(pause_p95)}s + 60s buffer = {ml_off_delay}s "
            f"(from {len(intra_pauses)} observed pauses)"
            if ml_off_delay is not None else "Not enough pause data to recommend"
        ),
        "pause_count": len(intra_pauses),
    }

    return {
        "enabled": True,
        "cycle_count": len(cycles),
        "evaluated_count": sum(1 for e in evaluated if e["ml_quality_score"] is not None),
        "cycles": evaluated,
        "settings_comparison": settings_comparison,
        "model_source": {"quality": quality_source, "end": end_source},
        "_health_dirty": health_dirty,
        "_health_updates": health_updates,
        "profile_stats": {
            name: {"count": int(m["count"]), "median_duration_s": int(m["duration_s"]), "median_energy_wh": round(m["energy_wh"], 1)}
            for name, m in profile_medians.items()
        },
    }


def _build_settings_comparison(manager: Any, merged: dict[str, Any]) -> dict[str, Any]:
    """Build the enriched Classic-vs-ML settings comparison (executor-safe).

    Runs both the classic :class:`SuggestionEngine` and the
    :class:`MLSuggestionEngine` against clean cycle history and merges their
    recommendations with the current option values. Gated by the caller behind
    ``ENABLE_ML_SUGGESTIONS``.
    """
    try:
        from .suggestion_engine import (  # pylint: disable=import-outside-toplevel
            MLSuggestionEngine,
            select_clean_cycles,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return {}

    learning = getattr(manager, "learning_manager", None)
    classic = getattr(learning, "suggestion_engine", None)
    if classic is None:
        return {}

    stop_thr = classic._current_stop_threshold(merged)  # pylint: disable=protected-access
    raw_cycles = classic.profile_store.get_past_cycles()

    # Classic recommendations pooled from every classic pass.
    classic_vals: dict[str, Any] = {}
    for producer in (
        classic.generate_detection_suggestions,
        classic.generate_model_suggestions,
    ):
        try:
            classic_vals.update(producer() or {})
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    try:
        classic_vals.update(classic.run_batch_simulation(raw_cycles) or {})
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    # Classic off_delay via the (Stage 2) pause analysis.
    try:
        clean, _excl = select_clean_cycles(raw_cycles[-200:], stop_threshold_w=stop_thr)
        device_floor = DEFAULT_OFF_DELAY_BY_DEVICE.get(
            classic.device_type, DEFAULT_OFF_DELAY
        ) if classic.device_type else DEFAULT_OFF_DELAY
        od = classic._suggest_off_delay_from_pauses(clean, stop_thr, device_floor)  # pylint: disable=protected-access
        if od is not None:
            classic_vals[CONF_OFF_DELAY] = {"value": od[0], "reason": od[1]}
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    # ML recommendations.
    ml_vals: dict[str, Any] = {}
    try:
        ml_vals = MLSuggestionEngine(classic).generate_ml_suggestions() or {}
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    def _val(entry: Any) -> Any:
        return entry.get("value") if isinstance(entry, dict) else None

    def _reason(entry: Any) -> str:
        return entry.get("reason", "") if isinstance(entry, dict) else ""

    comparison: dict[str, Any] = {}
    for key, label, unit in _ML_COMPARE_SETTINGS:
        cv, mv = classic_vals.get(key), ml_vals.get(key)
        if cv is None and mv is None:
            continue
        current = merged.get(key)
        classic_value = _val(cv)
        ml_value = _val(mv)
        # Hide a recommendation that would not change the current value.
        if classic_value is not None and _suggestion_equivalent(classic_value, current):
            classic_value = None
        if ml_value is not None and _suggestion_equivalent(ml_value, current):
            ml_value = None
        if classic_value is None and ml_value is None:
            continue
        comparison[key] = {
            "key": key,
            "label": label,
            "unit": unit,
            "current_value": current,
            "classic_value": classic_value,
            "classic_reason": _reason(cv) if classic_value is not None else "",
            "ml_value": ml_value,
            "ml_reason": _reason(mv) if ml_value is not None else "",
        }
    return comparison


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
        entry = _get_entry(hass, entry_id)
        merged: dict[str, Any] = {**(entry.data if entry else {}), **(entry.options if entry else {})}
        current_off_delay = int(merged.get(CONF_OFF_DELAY, 120))

        result = await hass.async_add_executor_job(
            _compute_ml_comparison, store, current_off_delay
        )
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
        result["ml_suggestions_enabled"] = ENABLE_ML_SUGGESTIONS

        # Stage 3: replace the basic off_delay-only comparison with the full
        # Classic-vs-ML settings table when ML suggestions are unlocked.
        if ENABLE_ML_SUGGESTIONS:
            enriched = await hass.async_add_executor_job(
                _build_settings_comparison, manager, merged
            )
            if enriched:
                result["settings_comparison"] = enriched

        connection.send_result(msg["id"], result)
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
        MATCH_CORR_WEIGHT,
        MATCH_DTW_ENSEMBLE_W,
        MATCH_DURATION_WEIGHT,
        MATCH_ENERGY_WEIGHT,
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
        "end": ("Cycle-end detection", "Knowing when a cycle has truly finished"),
        "quality": ("Cycle quality check", "Spotting mis-detected or corrupted cycles"),
        "live_match": ("Program matching", "Identifying the running program sooner"),
        "remaining_time": ("Time-remaining estimate", "Predicting how long is left"),
        "total_energy": ("Energy estimate", "Predicting total energy and cost"),
    }
    models: dict[str, Any] = {}
    for cap, v in versions.items():
        if not isinstance(v, dict):
            continue
        spec = v.get("spec") if isinstance(v.get("spec"), dict) else {}
        label, blurb = _cap_labels.get(cap, (cap, ""))
        info: dict[str, Any] = {
            "trained_at": v.get("trained_at"),
            "cycle_count": v.get("cycle_count"),
            "kind": spec.get("kind"),
            "label": label,
            "blurb": blurb,
        }
        # Raw metric numbers so the panel can render a humanized quality indicator
        # (a bar + word) with the exact figure on hover: held-out AUC for
        # classifiers, MAE-vs-naive for regressors.
        if v.get("new_auc") is not None:
            info["auc"] = round(float(v["new_auc"]), 4)
            info["metric"] = f"AUC {float(v['new_auc']):.2f} on held-out data"
        elif v.get("model_mae") is not None and v.get("naive_mae") is not None:
            info["model_mae"] = round(float(v["model_mae"]), 5)
            info["naive_mae"] = round(float(v["naive_mae"]), 5)
            info["metric"] = f"error {float(v['model_mae']):.3f} vs {float(v['naive_mae']):.3f} baseline"
        models[cap] = info

    # Fit trend across recent training runs (drift): compare the mean held-out
    # score of the most-recent third of runs to the oldest third, respecting each
    # metric's direction. Only meaningful with a few runs of history.
    history = store.get_ml_training_history()
    for cap, info in models.items():
        series = history.get(cap) if isinstance(history, dict) else None
        if not isinstance(series, list) or len(series) < 4:
            continue
        scores = [float(e["score"]) for e in series if isinstance(e, dict) and "score" in e]
        if len(scores) < 4:
            continue
        higher_better = bool(series[-1].get("higher_better", True))
        third = max(1, len(scores) // 3)
        old_mean = sum(scores[:third]) / third
        new_mean = sum(scores[-third:]) / third
        if abs(old_mean) < 1e-9:
            continue
        rel = (new_mean - old_mean) / abs(old_mean)
        if not higher_better:
            rel = -rel  # for error metrics, a decrease is an improvement
        info["trend"] = "improving" if rel > 0.03 else "declining" if rel < -0.03 else "steady"

    # Matcher scoring-weight tuning (Stage 4/5): current shipped defaults, the
    # on-device tuned record (if promoted), and which set is actually in use.
    tuned_rec = store.get_matching_config()
    tuned_cfg = tuned_rec.get("config") if isinstance(tuned_rec, dict) else None
    matching = {
        "defaults": {
            "corr_weight": MATCH_CORR_WEIGHT,
            "duration_weight": MATCH_DURATION_WEIGHT,
            "energy_weight": MATCH_ENERGY_WEIGHT,
            "dtw_ensemble_w": MATCH_DTW_ENSEMBLE_W,
        },
        "tuned": tuned_rec or None,
        "active": "tuned" if tuned_cfg else "default",
    }
    connection.send_result(
        msg["id"],
        {
            "available": ENABLE_ML_TRAINING,
            "enabled": bool(merged.get(CONF_ML_TRAINING_ENABLED, DEFAULT_ML_TRAINING_ENABLED)),
            "running": bool(getattr(manager, "_ml_training_running", False)),
            "last_trained": last.isoformat() if last else None,
            "cycle_count": len(store.get_past_cycles()),
            "min_cycles": int(merged.get(CONF_ML_TRAINING_MIN_CYCLES, DEFAULT_ML_TRAINING_MIN_CYCLES)),
            "interval_days": int(merged.get(CONF_ML_TRAINING_INTERVAL_DAYS, DEFAULT_ML_TRAINING_INTERVAL_DAYS)),
            "hour": int(merged.get(CONF_ML_TRAINING_HOUR, DEFAULT_ML_TRAINING_HOUR)),
            "on_device_models": models,
            "matching": matching,
        },
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/trigger_ml_training",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_trigger_ml_training(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Run on-device ML training now (manual, bypasses the schedule guards)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    if not ENABLE_ML_TRAINING:
        connection.send_error(msg["id"], "not_available", "ML training is not enabled in this build")
        return
    summary = await manager.async_run_ml_training(force=True)
    connection.send_result(msg["id"], summary)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/revert_matching_config",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_revert_matching_config(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Revert the matcher's scoring weights to the shipped defaults.

    Drops the on-device tuned override so matching falls back to the const
    defaults. The next training pass may re-promote a new one if it wins.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    await manager.profile_store.clear_matching_config()
    manager.notify_update()
    connection.send_result(msg["id"], {"success": True})


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

    Drops every promoted spec in ``ml_model_versions`` so ``resolve_scorer`` /
    ``resolve_regressor`` fall back to the baseline (or, for the baseline-less
    remaining-time regressor, become inert). The next training pass may
    re-promote models if they beat the baseline again. Mirrors
    ``revert_matching_config`` for the matcher weights.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return
    await manager.profile_store.clear_ml_model_versions()
    manager.notify_update()
    connection.send_result(msg["id"], {"success": True})


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

    This is the write-back that turns the read-only shadow view into a feedback
    loop: reviews become strong training labels for the on-device quality model.
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
        connection.send_result(msg["id"], {"success": True})
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
    await manager.async_pause_cycle()
    connection.send_result(msg["id"], {"ok": True})


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
    await manager.async_resume_cycle()
    connection.send_result(msg["id"], {"ok": True})


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
    connection.send_result(msg["id"], {"ok": True})


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
    opts: dict[str, Any] = {}
    if entry is not None:
        opts = {**getattr(entry, "data", {}), **getattr(entry, "options", {})}
    min_power = float(opts.get(CONF_MIN_POWER, 5.0) or 5.0)
    return CycleDetectorConfig(
        min_power=min_power,
        off_delay=int(opts.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY)),
        device_type=str(opts.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE)),
        completion_min_seconds=int(opts.get(CONF_COMPLETION_MIN_SECONDS, 600)),
        end_repeat_count=int(opts.get(CONF_END_REPEAT_COUNT, 1)),
        min_off_gap=int(opts.get(CONF_MIN_OFF_GAP, 60)),
        running_dead_zone=int(opts.get(CONF_RUNNING_DEAD_ZONE, 0)),
        start_threshold_w=float(opts.get(CONF_START_THRESHOLD_W, min_power)),
        stop_threshold_w=float(
            opts.get(CONF_STOP_THRESHOLD_W, min_power * 0.6 if min_power else 2.0)
        ),
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/run_playground_simulation",
        vol.Required("entry_id"): str,
        vol.Optional("cycle_ids", default=list): [str],
        vol.Optional("settings_override", default=dict): dict,
        vol.Optional("concurrency", default=1): vol.Coerce(int),
    }
)
@websocket_api.async_response
async def ws_run_playground_simulation(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Replay stored cycles through a headless detector with overridden settings.

    Returns ``{results: [...], summary: {...}}`` - a per-cycle event log +
    outcome plus aggregate counts. Nothing is persisted; this is a pure what-if.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    store = getattr(manager, "profile_store", None)
    if store is None:
        connection.send_error(msg["id"], "unavailable", "Profile store unavailable")
        return

    try:
        base_config = _playground_base_config(manager, _get_entry(hass, entry_id))
        cycle_ids = list(msg.get("cycle_ids") or [])
        settings_override = dict(msg.get("settings_override") or {})
        concurrency = int(msg.get("concurrency", 1))
        payload = await hass.async_add_executor_job(
            playground.run_playground_batch,
            store,
            cycle_ids,
            base_config,
            settings_override,
            concurrency,
        )
        connection.send_result(msg["id"], payload)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Playground simulation failed for %s: %s", entry_id, exc)
        connection.send_error(msg["id"], "unknown_error", str(exc))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_dtw_debug",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
        vol.Optional("profile_name"): vol.Any(str, None),
    }
)
@websocket_api.async_response
async def ws_get_dtw_debug(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the Stage 2 / DTW / Stage 4 score breakdown + resampled traces +
    DTW warp path for one cycle vs one profile (defaults to the cycle's label)."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    store = getattr(manager, "profile_store", None)
    if store is None:
        connection.send_error(msg["id"], "unavailable", "Profile store unavailable")
        return

    try:
        payload = await hass.async_add_executor_job(
            playground.dtw_debug_payload,
            store,
            msg["cycle_id"],
            msg.get("profile_name"),
        )
        if isinstance(payload, dict) and payload.get("error"):
            connection.send_error(
                msg["id"], payload["error"], payload.get("detail", payload["error"])
            )
            return
        connection.send_result(msg["id"], payload)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("DTW debug failed for %s: %s", entry_id, exc)
        connection.send_error(msg["id"], "unknown_error", str(exc))
