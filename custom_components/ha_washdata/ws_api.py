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
    CONF_APPLY_SUGGESTIONS,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_DEVICE_TYPE,
    CONF_DOOR_SENSOR_ENTITY,
    CONF_DURATION_TOLERANCE,
    CONF_END_ENERGY_THRESHOLD,
    CONF_EXTERNAL_END_TRIGGER,
    CONF_LINKED_DEVICE,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PUMP_STUCK_DURATION,
    CONF_RUNNING_DEAD_ZONE,
    CONF_SAMPLING_INTERVAL,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_SWITCH_ENTITY,
    CONF_WATCHDOG_INTERVAL,
    DEFAULT_DEVICE_TYPE,
    DEPRECATED_DEVICE_TYPES,
    DEVICE_TYPE_PUMP,
    DEVICE_TYPES,
    DOMAIN,
    STATE_COLORS,
)

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
)

# Suggestion keys coerced to int when applied (mirrors the OptionsFlow).
_SUGGESTION_INT_KEYS: frozenset[str] = frozenset({
    CONF_OFF_DELAY,
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_MIN_OFF_GAP,
    CONF_RUNNING_DEAD_ZONE,
})


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


# ─── Panel config + RBAC ────────────────────────────────────────────────────────

_PANEL_STORE_VERSION = 1
_PANEL_STORE_FILE = "ha_washdata_panel"
_PANEL_DATA_KEY = "ha_washdata_panel_cfg"

_LEVEL_RANK = {"none": 0, "read": 1, "edit": 2, "full": 3}
_PANEL_TABS = ("status", "history", "profiles", "settings", "tools", "panel")

# Commands that require 'full' (destructive or full-data export/import).
_FULL_COMMANDS = frozenset({
    "wipe_history", "import_config", "export_config", "clear_debug_data", "reprocess_history",
})
# Commands allowed for any authenticated user regardless of device permissions.
_OPEN_COMMANDS = frozenset({
    "get_constants", "get_notify_services", "get_panel_config", "set_user_prefs",
})
# Admin-only commands.
_ADMIN_COMMANDS = frozenset({"set_panel_config", "get_logs"})

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
        "panel": {"poll_interval_s": 5, "default_tab": "status", "hidden_tabs": [], "hide_deprecated": False},
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
    required = "full" if cmd in _FULL_COMMANDS else ("read" if cmd.startswith("get_") else "edit")
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
    if "hide_deprecated" in p:
        out["hide_deprecated"] = bool(p["hide_deprecated"])
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
        ws_get_options, ws_set_options, ws_get_notify_services,
        # Profiles
        ws_get_profiles, ws_create_profile, ws_rename_profile, ws_delete_profile,
        ws_rebuild_envelopes, ws_get_profile_phases, ws_set_profile_phases,
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
        ws_get_suggestions, ws_apply_suggestions, ws_clear_suggestions,
        # Cycle curve / interactive editing
        ws_get_cycle_power_data, ws_trim_cycle, ws_analyze_split, ws_apply_split, ws_apply_merge,
        # Profile envelope / member cycles
        ws_get_profile_envelope, ws_get_profile_cycles,
        # Feedback comparison
        ws_get_feedback_detail,
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
    }
)
@callback
def ws_get_device_cycles(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return recent cycles for a device, stripping large binary fields."""
    entry_id: str = msg["entry_id"]
    limit: int = msg.get("limit", 50)

    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    cycles: list[dict[str, Any]] = []
    try:
        store = getattr(manager, "profile_store", None)
        if store is not None:
            raw: list[Any] = store.get_past_cycles()
            for c in reversed(raw[-limit:]):
                cycles.append(_strip_cycle(c))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error fetching cycles for entry %s: %s", entry_id, exc)

    connection.send_result(msg["id"], {"entry_id": entry_id, "cycles": cycles})


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

    new_options.pop(CONF_APPLY_SUGGESTIONS, None)

    hass.config_entries.async_update_entry(entry, options=new_options)
    connection.send_result(msg["id"], {"success": True})


@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_notify_services"}
)
@callback
def ws_get_notify_services(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return all registered notify.* service names."""
    services: list[str] = []
    try:
        notify_domain = hass.services.async_services().get("notify", {})
        services = sorted(f"notify.{name}" for name in notify_domain)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error listing notify services: %s", exc)
    connection.send_result(msg["id"], {"services": services})


# ─── Profiles ─────────────────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {vol.Required("type"): "ha_washdata/get_profiles", vol.Required("entry_id"): str}
)
@callback
def ws_get_profiles(
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

    profiles: list[dict[str, Any]] = []
    try:
        profiles = manager.profile_store.list_profiles()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error listing profiles for %s: %s", entry_id, exc)

    connection.send_result(msg["id"], {"profiles": profiles})


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
            start = dt_util.parse_datetime(last_run["start_time"])
            end = dt_util.parse_datetime(last_run["end_time"])
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

        start_ts = dt_util.parse_datetime(rec_start_str).timestamp() if rec_start_str else data_start_ts
        end_ts = dt_util.parse_datetime(rec_end_str).timestamp() if rec_end_str else data_end_ts

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
    """Reprocess all historical cycle data."""
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    try:
        count = await manager.profile_store.async_reprocess_all_data()
        manager.notify_update()
        connection.send_result(msg["id"], {"success": True, "count": count})
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
@callback
def ws_export_config(
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
        connection.send_result(msg["id"], {"json_data": json.dumps(payload, indent=2)})
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
        payload = json.loads(msg["json_data"])
        config_updates = await manager.profile_store.async_import_data(payload)

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
        {"id": key, "label": label, "deprecated": key in DEPRECATED_DEVICE_TYPES}
        for key, label in DEVICE_TYPES.items()
    ]
    connection.send_result(
        msg["id"],
        {"device_types": device_types, "state_colors": dict(STATE_COLORS)},
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
            out.append(
                {
                    "key": key,
                    "suggested": suggested,
                    "reason": item.get("reason", ""),
                    "current": merged.get(key),
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


# ─── Cycle curve / interactive editing ─────────────────────────────────────────

@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_cycle_power_data",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
    }
)
@callback
def ws_get_cycle_power_data(
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
@callback
def ws_get_profile_cycles(
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

    connection.send_result(msg["id"], {"cycles": out})


# ─── Feedback comparison ───────────────────────────────────────────────────────

@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_feedback_detail",
        vol.Required("entry_id"): str,
        vol.Required("cycle_id"): str,
    }
)
@callback
def ws_get_feedback_detail(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return everything needed to compare a feedback cycle against candidates.

    Includes the actual cycle curve, the candidate profiles' average envelopes
    (for overlay) and the ranked candidates table (confidence/MAE/correlation/
    duration), reconstructed from the stored feedback ranking.
    """
    entry_id: str = msg["entry_id"]
    manager = _get_manager(hass, entry_id)
    if manager is None:
        _err_not_found(connection, msg["id"], entry_id)
        return

    cycle_id: str = msg["cycle_id"]
    out: dict[str, Any] = {
        "detected_profile": None, "confidence": None,
        "estimated_duration": None, "actual_duration": None,
        "candidates": [], "overlays": [], "actual_samples": [], "full_duration_s": 0.0,
    }
    try:
        store = manager.profile_store
        item = (store.get_pending_feedback() or {}).get(cycle_id) or {}
        out["detected_profile"] = item.get("detected_profile")
        out["confidence"] = item.get("confidence")
        out["estimated_duration"] = item.get("estimated_duration")
        out["actual_duration"] = item.get("actual_duration")

        ranking = item.get("ranking") or []
        if ranking:
            mr = _RankingMatchResult(ranking, item.get("estimated_duration") or 0)
            out["candidates"] = store.get_match_candidates_summary(mr, 5)

        samples = store.get_cycle_power_data(cycle_id)
        out["actual_samples"] = _downsample(samples)
        out["full_duration_s"] = round(float(samples[-1][0]), 1) if samples else 0.0

        names: list[str] = []
        det = item.get("detected_profile")
        if det:
            names.append(det)
        for c in out["candidates"]:
            n = c.get("profile_name")
            if n and n not in names:
                names.append(n)
        for n in names[:4]:
            env = store.get_envelope(n)
            if env and env.get("avg"):
                out["overlays"].append({"profile_name": n, "avg": _downsample(env["avg"])})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Error building feedback detail for %s: %s", cycle_id, exc)

    connection.send_result(msg["id"], out)


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
    for k in ("show_expected", "show_raw", "show_debug"):
        if k in p:
            cur[k] = bool(p[k])
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
    out: dict[str, Any] = {"cycle_active": False, "cycle_elapsed_s": 0.0, "live": [], "raw": []}
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
