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
"""Manager for WashData."""

# pylint: disable=broad-exception-caught

from __future__ import annotations

import logging
import hashlib
import json
import math
import re
import traceback
import uuid
import asyncio
import functools
from asyncio import Task
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
import numpy as np

if TYPE_CHECKING:
    from .store import StoreBridge

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Context, Event, HomeAssistant, State, callback
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_state_report_event,
    async_track_time_interval,
)
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.exceptions import HomeAssistantError
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, STATE_UNAVAILABLE, STATE_HOME
from homeassistant.util import dt as dt_util
import voluptuous as vol

import homeassistant.helpers.event as evt
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import script as script_helper
from homeassistant.helpers import translation
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.storage import Store

from .const import NOTIFY_QUEUE_STORE_SUFFIX, STORAGE_KEY
from .const import (
    resolve_off_delay_default,
    CADENCE_RESET_FROM_STATES,
    DOMAIN,
    CONF_POWER_SENSOR,
    CONF_PROFILE_EVIDENCE_SOURCES,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_ACTIONS,
    CONF_NOTIFY_START_SERVICES,
    CONF_NOTIFY_FINISH_SERVICES,
    CONF_NOTIFY_LIVE_SERVICES,
    CONF_NOTIFY_CYCLE_TIMERS,
    CONF_NOTIFY_PEOPLE,
    CONF_NOTIFY_ONLY_WHEN_HOME,
    CONF_NOTIFY_FIRE_EVENTS,
    CONF_NOTIFY_EVENTS,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_LOW_POWER_NO_UPDATE_TIMEOUT, # Import new constant
    CONF_PROGRESS_RESET_DELAY,
    CONF_LEARNING_CONFIDENCE,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_AUTO_MAINTENANCE,
    CONF_MAINTENANCE_REMINDER_CYCLES,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_WATCHDOG_INTERVAL,
    CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_PROFILE_UNMATCH_THRESHOLD,
    CONF_DEVICE_TYPE,
    CONF_SAMPLING_INTERVAL,
    CONF_SAVE_DEBUG_TRACES,
    CONF_DTW_BANDWIDTH,
    CONF_EXTERNAL_END_TRIGGER_ENABLED,
    CONF_EXTERNAL_END_TRIGGER,
    CONF_EXTERNAL_END_TRIGGER_INVERTED,
    CONF_PUMP_STUCK_DURATION,
    DEFAULT_PUMP_STUCK_DURATION,
    EVENT_PUMP_STUCK,
    DEVICE_TYPE_PUMP,
    SIGNAL_WASHER_UPDATE,
    NOTIFY_EVENT_START,
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
    NOTIFY_EVENT_CLEAN,
    NOTIFY_EVENT_TIMER,
    EVENT_CYCLE_STARTED,
    EVENT_CYCLE_ENDED,
    EVENT_CYCLE_STALLED,
    CYCLE_ANOMALY_STALLED,
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
    DEFAULT_PROGRESS_RESET_DELAY,
    DEFAULT_PROFILE_EVIDENCE_SOURCES,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_AUTO_MAINTENANCE,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_NOTIFY_TITLE,
    CONF_NOTIFY_ICON,
    CONF_NOTIFY_ICON_COLOR,
    CONF_NOTIFY_START_MESSAGE,
    CONF_NOTIFY_FINISH_MESSAGE,
    CONF_NOTIFY_PRE_COMPLETE_MESSAGE,
    CONF_NOTIFY_LIVE_INTERVAL_SECONDS,
    CONF_NOTIFY_LIVE_OVERRUN_PERCENT,
    CONF_NOTIFY_LIVE_CHRONOMETER,
    CONF_NOTIFY_LIVE_STICKY,
    CONF_NOTIFY_LIVE_CLICK_ACTION,
    CONF_NOTIFY_LIVE_SILENT,
    DEFAULT_NOTIFY_LIVE_STICKY,
    DEFAULT_NOTIFY_LIVE_CLICK_ACTION,
    DEFAULT_NOTIFY_LIVE_SILENT,
    CONF_NOTIFY_REMINDER_MESSAGE,
    CONF_NOTIFY_TIMEOUT_SECONDS,
    CONF_NOTIFY_CHANNEL,
    CONF_NOTIFY_FINISH_CHANNEL,
    CONF_ENERGY_PRICE_STATIC,
    CONF_ENERGY_PRICE_ENTITY,
    CONF_ENERGY_PRICE_DYNAMIC,
    DEFAULT_ENERGY_PRICE_DYNAMIC,
    PRICE_TIMELINE_MAX_POINTS,
    PRICE_TIMELINE_PRICE_DECIMALS,
    CONF_ENERGY_SENSOR,
    CONF_PEAK_RATE_THRESHOLD,
    CONF_PEAK_RATE_MESSAGE,
    DEFAULT_PEAK_RATE_MESSAGE,
    CONF_DOOR_SENSOR_ENTITY,
    CONF_DOOR_OPENS_AT_END,
    CONF_DOOR_END_DWELL_SECONDS,
    DEFAULT_DOOR_OPENS_AT_END,
    DEFAULT_DOOR_END_DWELL_SECONDS,
    CONF_PAUSE_CUTS_POWER,
    CONF_SWITCH_ENTITY,
    CONF_NOTIFY_UNLOAD_DELAY_MINUTES,
    CONF_NOTIFY_UNLOAD_MESSAGE,
    CONF_NOTIFY_UNLOAD_REPEAT,
    DEFAULT_NOTIFY_UNLOAD_DELAY_MINUTES,
    DEFAULT_NOTIFY_UNLOAD_MESSAGE,
    DEFAULT_NOTIFY_UNLOAD_REPEAT,
    NOTIFY_UNLOAD_REPEAT_MAX_REMINDERS,
    CONF_UNLOAD_CONFIRM_ENTITY,
    CONF_UNLOAD_TRACK_WITHOUT_DOOR,
    UNLOAD_CONFIRM_REPLAY_GRACE_S,
    DEFAULT_UNLOAD_TRACK_WITHOUT_DOOR,
    CONF_NOTIFY_MILESTONES,
    CONF_NOTIFY_MILESTONE_MESSAGE,
    DEFAULT_NOTIFY_MILESTONES,
    DEFAULT_NOTIFY_MILESTONE_MESSAGE,
    STATE_CLEAN,
    STATE_FINISHED,
    STATE_INTERRUPTED,
    STATE_FORCE_STOPPED,
    DEFAULT_NOTIFY_TITLE,
    DEFAULT_NOTIFY_START_MESSAGE,
    DEFAULT_NOTIFY_FINISH_MESSAGE,
    DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE,
    DEFAULT_NOTIFY_LIVE_WAITING_MESSAGE,
    DEFAULT_NOTIFY_ONLY_WHEN_HOME,
    DEFAULT_NOTIFY_FIRE_EVENTS,
    DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS,
    DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT,
    DEFAULT_NOTIFY_LIVE_CHRONOMETER,
    DEFAULT_NOTIFY_REMINDER_MESSAGE,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_CHANNEL,
    DEFAULT_NOTIFY_FINISH_CHANNEL,

    DEFAULT_DTW_BANDWIDTH,
    WATCHDOG_LATE_TICK_FACTOR,
    resolve_sampling_interval_default,
    resolve_watchdog_interval_default,
    CONF_MATCH_PERSISTENCE,
    DEFAULT_MATCH_PERSISTENCE,
    MATCH_LABEL_MIN_MARGIN,
    ENABLE_ML_END_GUARD,
    DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_UNMATCHED_WATCHDOG_CEILING,
    DEFAULT_UNMATCHED_WATCHDOG_CEILING_BY_DEVICE,
    DEFAULT_MAX_DEFERRAL_SECONDS,
    CYCLE_UNDERRUN_ANOMALY_RATIO,
    ENERGY_ANOMALY_Z_THRESHOLD,
    STATE_RUNNING,
    STATE_OFF,
    STATE_STARTING,
    STATE_PAUSED,
    STATE_USER_PAUSED,
    STATE_ENDING,
    STATE_ANTI_WRINKLE,
    STATE_DELAY_WAIT,
    STATE_IDLE,
    STATE_UNKNOWN,
)
from .detector_config import (
    apply_detector_config,
    build_detector_config,
    terminal_drop_baseline_for,
    terminal_drop_enabled,
    terminal_drop_fires,
    terminal_drop_may_fire,
)
from .cycle_detector import (
    MatchContext,
    CycleDetector,
    STANDBY_LEVEL_RECENT_CYCLES,
    TERMINAL_PROBE_RETURNS,
    learned_standby_level_w,
    standby_near_stop_ceiling,
    terminal_high_for_guards,
)
from .learning import LearningManager
from .profile_store import (
    MatchResult,
    ProfileStore,
    decompress_power_data,
)
from .signal_processing import (
    median_fast,
    percentile_linear,
    integrate_wh,
    energy_gap_threshold_s,
    compact_price_timeline,
    cycle_cost,
)
from .recorder import CycleRecorder
from .diag_buffer import DiagBuffer
from .log_utils import DeviceLoggerAdapter

# Per-entity anchors for the unload-confirm replay window (register items 367, 368).
# `{entity_id: datetime}` in `hass.data`, so it survives entry reloads, resets on an
# HA restart, and never carries over between different configured entities.
_UNLOAD_CONFIRM_ANCHOR_KEY = f"{DOMAIN}_unload_confirm_anchors"
from .options_utils import option_float, option_int
from .time_utils import power_data_to_offsets, utc_now
from . import analysis
from . import progress as progress_mod
from . import notification_rules as notif_rules
from . import match_rules
from .maintenance import effective_reminders
from .frontend import PANEL_URL_PATH

_LOGGER = logging.getLogger(__name__)

# Sentinel "message" understood by the Home Assistant companion app as "dismiss the
# card carrying this tag" rather than as text to display. It is only meaningful
# alongside a `tag`, and only on mobile_app targets - see _send_notification_service,
# which must never deliver it as a visible message.
_CLEAR_NOTIFICATION_MARKER = "clear_notification"

# A notification accent colour (#454) typed without its leading "#". Matches the
# three CSS hex forms the companion apps accept (RGB, RRGGBB, AARRGGBB) so the
# "#" can be added back; anything else is left alone.
_HEX_COLOR_RE = re.compile(r"[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8}")

# Finish-type notification events that would wake someone and are therefore gated by
# the quiet-hours (do-not-disturb) window. Live-progress ticks (NOTIFY_EVENT_LIVE)
# and the start notification (NOTIFY_EVENT_START) are intentionally excluded.
_QUIET_HOURS_EVENT_TYPES = frozenset(
    {NOTIFY_EVENT_FINISH, NOTIFY_EVENT_CLEAN, "pre_complete"}
)

# Held notifications persisted across a restart (audit MANAGER-16). Not persisted:
# a live update (the next tick replaces it), a cycle timer (its Resume action is
# wired for one session) and the unload nag (the Clean state it belongs to is not
# restored, and the door may have opened meanwhile). Start and pre-complete belong
# to the cycle under way, so they are restored only while one still is.
_NOTIFY_QUEUE_TRANSIENT_EVENTS = frozenset(
    {NOTIFY_EVENT_LIVE, NOTIFY_EVENT_TIMER, NOTIFY_EVENT_CLEAN}
)
_NOTIFY_QUEUE_CYCLE_EVENTS = frozenset({NOTIFY_EVENT_START, "pre_complete"})
# A saved queue older than this is dropped on restore instead of delivered: a
# "finished" from days ago is noise. Longer than any quiet window.
_NOTIFY_QUEUE_MAX_AGE_S = 24 * 3600


# Detector states in which the power sensor must not be swapped out. Every state
# with an in-flight cycle, plus ANTI_WRINKLE: its tumble pulses are still being
# attributed to the cycle that just finished, so re-pointing the listener there
# would splice a different appliance into that tail.
_SENSOR_SWAP_BLOCKED_STATES = frozenset(
    {
        STATE_STARTING,
        STATE_RUNNING,
        STATE_PAUSED,
        STATE_ENDING,
        STATE_ANTI_WRINKLE,
    }
)

# States in which a cycle is under way, so a manually picked program applies to it
# right now rather than being armed for the next one (#411). ANTI_WRINKLE is
# excluded on purpose: its tumble pulses belong to the cycle that already ended,
# so a program chosen there is meant for the next run.
_CYCLE_IN_PROGRESS_STATES = frozenset(
    {
        STATE_STARTING,
        STATE_RUNNING,
        STATE_PAUSED,
        STATE_ENDING,
    }
)


# Device classes and units that prove a configured "energy price entity" is not a
# price at all (#439). The panel's picker lists every `sensor.`, so the obvious
# mistake is to point it at the plug's own energy counter - and because a price
# entity outranks the static price, the cycle is then costed at
# `kWh_used * current_meter_reading`, which looks like a plausible number and is
# nonsense. A price is never measured in W or kWh; `monetary` and unitless or
# "EUR/kWh"-style sensors are left alone.
_NON_PRICE_DEVICE_CLASSES = frozenset(
    {"energy", "energy_storage", "power", "gas", "water", "current", "voltage"}
)
_NON_PRICE_UNITS = frozenset(
    {"w", "kw", "mw", "wh", "kwh", "mwh", "va", "kva", "varh", "a", "ma", "v", "mv"}
)


def _finite_power(raw: Any) -> float | None:
    """Parse a power sensor's state string, rejecting non-finite values.

    ``float()`` accepts ``"nan"``, ``"inf"`` and ``"infinity"``, and a power
    reading is compared against thresholds exactly like the options
    ``options_utils.option_float`` already guards: every comparison against
    ``nan`` is False, so a single such reading does not raise, it silently
    switches gates OFF. With ``_current_power`` set to ``nan`` both the
    unmatched-cycle watchdog (`< start_threshold_w`) and the high-power silence
    deferral (`> min_power`) evaluate False, so a running cycle loses the guards
    that decide whether it ends at all.

    Returns None for anything unusable, which is the "sensor is non-numeric"
    outcome every caller already handles. A real meter never reports either
    value, so no valid reading changes behaviour.
    """
    try:
        power = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(power):
        return None
    return power


def _snapshot_time(raw: Any) -> datetime | None:
    """A timestamp from the active-cycle snapshot as aware UTC, else None.

    A naive value is a legacy local stamp, read in HA's zone like every other
    snapshot field. Never raises: a junk value restores as "unknown".
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = dt_util.parse_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_util.now().tzinfo)
    return dt_util.as_utc(parsed)


def _coerce_price_timeline(raw: Any) -> list[tuple[float, float]]:
    """Coerce a persisted price timeline back into ``(ts, price)`` tuples (#426).

    JSON round-trips the pairs as lists, and a hand-edited or truncated store must
    not be able to break cycle restoration, so anything unparseable is dropped
    rather than raised on.
    """
    result: list[tuple[float, float]] = []
    if not isinstance(raw, (list, tuple)):
        return result
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            result.append((float(entry[0]), float(entry[1])))
        except (TypeError, ValueError, OverflowError):
            continue
    result.sort(key=lambda item: item[0])
    return result


def _sanitize_ranking(raw_list: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Top-N ranking candidates stripped of the heavy `current`/`sample` power
    arrays, safe to persist on cycle_data and to include in the 32KB-limited
    EVENT_CYCLE_ENDED payload."""
    out: list[dict[str, Any]] = []
    for cand in (raw_list or [])[:limit]:
        out.append({
            "name": cand.get("name"),
            "score": round(float(cand.get("score", 0.0)), 3),
            "profile_duration": cand.get("profile_duration"),
        })
    return out


def _apply_post_cycle_anomalies(cycle_data: dict[str, Any], store: Any) -> None:
    """Stamp the post-cycle A1 underrun and A2 energy anomalies onto ``cycle_data``.

    Runs at cycle end after the runtime overrun anomaly is frozen onto the cycle.
    ``store`` is the device's ProfileStore (only its median-duration and energy-stats
    lookups are used, and only when the guards pass). Each rule is independent and
    never raises: a failure leaves that rule's fields unset. Purely informational,
    never a notification.
    """
    # A1: Underrun check - computed post-cycle only, not a live signal.
    # Only applied when no runtime anomaly was detected (underrun and overrun are mutually exclusive).
    try:
        if not cycle_data.get("anomaly") or cycle_data["anomaly"] == "none":
            _uc_profile = cycle_data.get("profile_name")
            _uc_dur = float(cycle_data.get("duration", 0))
            if _uc_profile and _uc_dur > 0:
                _uc_median = store.get_profile_median_duration(_uc_profile)
                if (
                    isinstance(_uc_median, (int, float))
                    and not isinstance(_uc_median, bool)
                    and _uc_median > 0
                    and _uc_dur < _uc_median * CYCLE_UNDERRUN_ANOMALY_RATIO
                ):
                    cycle_data["anomaly"] = "underrun"
                    cycle_data["underrun_ratio"] = round(_uc_dur / _uc_median, 3)
    except Exception:  # noqa: BLE001
        pass

    # A2: Energy spike/low anomaly - stored separately from duration anomaly.
    try:
        _ea_profile = cycle_data.get("profile_name")
        _ea_energy = float(cycle_data.get("energy_wh", 0))
        if _ea_profile and _ea_energy > 0:
            _ea_stats = store.get_profile_energy_stats(_ea_profile)
            if (
                isinstance(_ea_stats, dict)
                and isinstance(_ea_stats.get("std_wh"), (int, float))
                and _ea_stats["std_wh"] > 0
            ):
                _ea_z = (_ea_energy - _ea_stats["avg_wh"]) / _ea_stats["std_wh"]
                cycle_data["energy_z_score"] = round(_ea_z, 2)
                if _ea_z > ENERGY_ANOMALY_Z_THRESHOLD:
                    cycle_data["energy_anomaly"] = "energy_spike"
                elif _ea_z < -ENERGY_ANOMALY_Z_THRESHOLD:
                    cycle_data["energy_anomaly"] = "energy_low"
    except Exception:  # noqa: BLE001
        pass

# Notification-data keys that may only be forwarded to mobile_app_* notify targets.
# Strict-schema platforms (e.g. Signal) reject unknown keys, so these are added per
# service only when the target is a mobile app. Includes the iOS Live Activity
# enrichment keys (subtitle/content_state/activity) so they never reach non-mobile
# platforms.
_MOBILE_ONLY_EXTRA_KEYS = (
    "tag",
    "timeout",
    "channel",
    "priority",
    "actions",
    "sticky",
    "clickAction",
    "url",
    "subtitle",
    "content_state",
    "activity",
    "silent",
    "push",
)


def _pn_create(
    hass: HomeAssistant,
    message: str,
    *,
    title: str | None = None,
    notification_id: str | None = None,
) -> bool:
    """Best-effort persistent notification creation; True when it was posted.

    Calls ``homeassistant.components.persistent_notification`` directly. This used
    to go through the ``components`` accessor on ``hass``, which Home Assistant
    removed: ``getattr`` found nothing and the helper returned silently, so every
    sidebar card (the fallback for users with no notify target, the auto-pause
    timer card) was dropped while the caller logged it as delivered - and the 12
    test modules that mocked that accessor kept passing (audit PLATFORM-01).
    """
    try:
        persistent_notification.async_create(
            hass, message, title=title, notification_id=notification_id
        )
        return True
    except Exception:  # noqa: BLE001 - best-effort; surface the failure in logs
        _LOGGER.warning(
            "persistent_notification create failed (id=%s)", notification_id, exc_info=True
        )
        return False


def _pn_dismiss(hass: HomeAssistant, notification_id: str) -> None:
    """Best-effort persistent notification dismissal (see :func:`_pn_create`)."""
    try:
        persistent_notification.async_dismiss(hass, notification_id)
    except Exception:  # noqa: BLE001 - best-effort; surface the failure in logs
        _LOGGER.warning(
            "persistent_notification dismiss failed (id=%s)", notification_id, exc_info=True
        )


def _read_switch_state(mgr: Any) -> match_rules.SwitchState:
    """The manager's switching fields as the shared rules read them.

    Module-level, not a method, so a test binding only
    ``_async_do_perform_matching`` onto a stub manager still runs the real path.
    The two dicts are passed by reference, as the rules always mutated them.
    """
    return match_rules.SwitchState(
        current_program=mgr._current_program,
        matched_duration=mgr._matched_profile_duration,
        last_confidence=mgr._last_match_confidence,
        last_member_confidence=mgr._last_member_confidence,
        score_history=mgr._score_history,
        persistence_counter=mgr._match_persistence_counter,
        unmatch_counter=mgr._unmatch_persistence_counter,
        current_candidate=mgr._current_match_candidate,
    )


def _write_switch_state(
    mgr: Any, state: match_rules.SwitchState, log: list[match_rules.LogLine]
) -> None:
    """Write back what the shared rules decided, then emit their log lines."""
    mgr._current_program = state.current_program
    mgr._matched_profile_duration = state.matched_duration
    mgr._last_match_confidence = state.last_confidence
    mgr._last_member_confidence = state.last_member_confidence
    mgr._score_history = state.score_history
    mgr._match_persistence_counter = state.persistence_counter
    mgr._unmatch_persistence_counter = state.unmatch_counter
    mgr._current_match_candidate = state.current_candidate
    _emit_rule_log(mgr._logger, log)


def _emit_rule_log(logger: Any, log: list[match_rules.LogLine]) -> None:
    for level, msg, args in log:
        logger.log(level, msg, *args)


def _option_then_data(config_entry: Any, key: str, default: Any) -> Any:
    """``options[key]``, else ``data[key]``, else ``default``.

    A device added after its last schema migration keeps its structural keys
    (min_power, off_delay) in ``entry.data`` only, and ``ws_set_options`` writes
    ``{**options, **changes}`` - so options-only reads on reload reset them to the
    defaults at the first unrelated settings save (register item 388a).
    """
    return config_entry.options.get(key, config_entry.data.get(key, default))


class WashDataManager:
    """Manages a single washing machine instance."""

    @property
    def store_bridge(self) -> "StoreBridge":
        """Lazy community-store bridge (kept for the entry so the token cache persists)."""
        if self._store_bridge is None:
            from .store import StoreBridge
            self._store_bridge = StoreBridge(self.hass, self.profile_store)
        return self._store_bridge

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the manager."""
        self.hass = hass
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        self._logger = DeviceLoggerAdapter(_LOGGER, config_entry.title)
        self.diag_buffer = DiagBuffer(config_entry.title)

        # Prioritize options -> data for power sensor (allows changing it)
        self.power_sensor_entity_id = config_entry.options.get(
            CONF_POWER_SENSOR, config_entry.data.get(CONF_POWER_SENSOR)
        )
        # A sensor change saved while a cycle was under way, applied once the
        # detector leaves _SENSOR_SWAP_BLOCKED_STATES (audit MANAGER-11).
        self._pending_power_sensor: str | None = None
        self.device_type = config_entry.options.get(
            CONF_DEVICE_TYPE,
            config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
        )
        # The sensor platform's add callback, kept so an in-place device type change
        # can add the pump-only sensor (sensor.async_reconcile_device_type_sensors).
        self.sensor_add_entities: Any = None

        # Initialize attributes to satisfy pylint
        self._off_delay = float(DEFAULT_OFF_DELAY)
        self._no_update_active_timeout = float(DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT)
        self._low_power_no_update_timeout = 3600.0 # Default 1h
        self._notify_before_end_minutes = float(DEFAULT_NOTIFY_BEFORE_END_MINUTES)
        self._notify_start_services: list[str] = []
        self._notify_finish_services: list[str] = []
        self._notify_live_services: list[str] = []
        self._notify_actions: list[dict[str, Any]] = []
        self._notify_script: Any = None  # cached Script; invalidated on options reload
        self._notify_people: list[str] = []
        self._notify_cycle_timers: list[dict[str, Any]] = []
        self._fired_cycle_timers: set[int] = set()
        self._timer_pause_pn_id: str | None = None
        self._timer_pause_mobile_tag: str | None = None
        self._remove_timer_action_listener: Any | None = None
        self._timer_ui_strings: dict[str, str] = {}
        self._notify_only_when_home = DEFAULT_NOTIFY_ONLY_WHEN_HOME
        self._notify_fire_events = DEFAULT_NOTIFY_FIRE_EVENTS
        self._notify_live_interval_seconds = DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS
        self._notify_live_overrun_percent = DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT
        self._notify_live_chronometer = DEFAULT_NOTIFY_LIVE_CHRONOMETER
        self._notify_live_sticky = DEFAULT_NOTIFY_LIVE_STICKY
        self._notify_live_click_action = DEFAULT_NOTIFY_LIVE_CLICK_ACTION
        self._notify_live_silent = DEFAULT_NOTIFY_LIVE_SILENT
        self._notify_timeout_seconds = DEFAULT_NOTIFY_TIMEOUT_SECONDS
        self._pending_notifications: list[dict[str, Any]] = []
        # Quiet-hours (do-not-disturb) hold queue + release timer. Finish-type
        # notifications that would fire inside the window are parked here and flushed
        # at the end of the window by a single async_call_later timer.
        self._quiet_pending_notifications: list[dict[str, Any]] = []
        self._remove_quiet_hours_timer: Any | None = None
        # Both queues outlive a restart through this file (audit MANAGER-16).
        self._notify_queue_store: Store[dict[str, Any]] | None = None
        self._notify_queue_on_disk = False
        self._remove_ha_stop_listener: Callable[[], None] | None = None
        self._remove_notify_queue_restore: Callable[[], None] | None = None
        self._remove_notify_people_listener = None
        self._live_notification_sent_count = 0

        # HA restart gap tracking: gaps in the power trace caused by integration
        # restarts during an active cycle.  Each entry is a dict:
        #   start_ts: ISO timestamp of gap start (= last snapshot save time)
        #   end_ts:   ISO timestamp of gap end (= restoration time)
        #   gap_seconds: duration in seconds
        #   profile: matched profile name at restoration time, or None
        #   match_confidence: match confidence at restoration time, or None
        # Cleared and stored into cycle_data["restart_gaps"] at cycle end.
        # Matching always uses real readings only; this list is for display/anomaly.
        self._restart_gaps: list[dict[str, Any]] = []

        # External energy-meter snapshot for the current cycle (issue #316).
        # Captured at cycle start, read back at cycle end for the start->end delta.
        # Both survive a restart via the active-cycle snapshot.
        self._energy_meter_start: float | None = None
        self._energy_meter_source: str | None = None

        # Dynamic energy price timeline for the current cycle (#426): the price in
        # force at each point of the cycle, as ``(unix_ts, price_per_kwh)`` pairs,
        # appended by the price-entity listener below. Absolute timestamps, not
        # offsets: the stored cycle's start time can end up later than the detector's
        # (leading-zero trim, a split), and converting once at cycle end against the
        # figure actually persisted is the only way the two axes cannot drift apart.
        # Survives a restart via the active-cycle snapshot; any price change that
        # happened while HA was down is recovered from the recorder at cycle end.
        self._price_timeline: list[tuple[float, float]] = []
        self._remove_price_listener: Callable[[], None] | None = None
        # Entity id already reported as "not a price" (#439), so the rejection is
        # logged once per misconfiguration instead of on every cycle.
        self._warned_price_entity: str | None = None

        # Pause tracking (user-triggered)
        self._user_pause_start: datetime | None = None
        self._total_user_paused_seconds: float = 0.0
        self._user_paused_flag: bool = False
        self._pause_cuts_power: bool = bool(
            config_entry.options.get(CONF_PAUSE_CUTS_POWER, False)
        )

        # Door sensor + clean state
        self._door_sensor_entity: str | None = config_entry.options.get(
            CONF_DOOR_SENSOR_ENTITY
        ) or None
        # Auto-open dishwasher (#342): a sustained door-open at cycle end finalizes
        # the cycle after a dwell instead of setting the sticky user-pause.
        self._door_opens_at_end: bool = bool(
            config_entry.options.get(CONF_DOOR_OPENS_AT_END, DEFAULT_DOOR_OPENS_AT_END)
        )
        self._door_end_dwell_seconds: int = int(
            config_entry.options.get(CONF_DOOR_END_DWELL_SECONDS, DEFAULT_DOOR_END_DWELL_SECONDS)
        )
        self._remove_door_end_dwell: Any = None
        self._remove_door_sensor_listener = None
        # Unload confirmation without a door sensor (#451): an entity whose
        # activation means "unloaded", and/or a plain opt-in for the Mark Unloaded
        # button and the mark_unloaded service.
        self._unload_confirm_entity: str | None = config_entry.options.get(
            CONF_UNLOAD_CONFIRM_ENTITY
        ) or None
        self._unload_track_without_door: bool = bool(
            config_entry.options.get(
                CONF_UNLOAD_TRACK_WITHOUT_DOOR, DEFAULT_UNLOAD_TRACK_WITHOUT_DOOR
            )
        )
        self._remove_unload_confirm_listener = None
        self._is_clean_state: bool = False
        self._clean_state_start: datetime | None = None
        self._notified_clean_laundry: bool = False
        # Set by _dispatch_notification when a call is queued for later (quiet
        # hours / presence) rather than sent; read by dedup-flag callers.
        self._last_dispatch_deferred: bool = False
        self._notify_unload_delay_minutes: int = int(
            config_entry.options.get(
                CONF_NOTIFY_UNLOAD_DELAY_MINUTES, DEFAULT_NOTIFY_UNLOAD_DELAY_MINUTES
            )
        )
        # Repeat the unload reminder until dismissed / door-open (opt-in, #374).
        self._notify_unload_repeat: bool = bool(
            config_entry.options.get(
                CONF_NOTIFY_UNLOAD_REPEAT, DEFAULT_NOTIFY_UNLOAD_REPEAT
            )
        )
        # Set when the user taps the reminder's "stop reminding" action; timestamp of
        # the last reminder sent, used to pace the repeats; and the mobile action
        # listener remover (mirrors the timer-pause interactive-notification wiring).
        self._unload_nag_dismissed: bool = False
        self._last_unload_nag_time: datetime | None = None
        self._unload_nag_count: int = 0  # repeat-mode safety bound (#374)
        self._remove_unload_action_listener: Any | None = None
        self._live_notification_cap = 0
        self._last_live_notification_time: datetime | None = None
        self._live_waiting_notification_sent = False
        self._live_chronometer_overrun_sent = False
        # iOS Live Activity: whether the "start" lifecycle marker has been emitted on
        # the first live notification of the current cycle. Reset per cycle.
        self._live_activity_started = False
        # Single per-device identity shared by start/live/reminder/finished so each
        # replaces the previous on the mobile app (and collapses to one entry on the
        # persistent-notification fallback). The clean-laundry nag uses its own tag
        # since it fires up to an hour after finish and should not clobber the thread.
        self._lifecycle_tag = f"ha_washdata_{self.entry_id}_lifecycle"
        self._clean_tag = f"ha_washdata_{self.entry_id}_clean"
        # #446: the live progress updates need their OWN tag, because on iOS a Live
        # Activity is a separate UI surface from the notification and is ended only
        # by `clear_notification` with the activity's tag. While this was an alias
        # for the lifecycle tag there was no way to end it: clearing would have
        # dismissed the finished card that shares the tag, which is why the cycle-end
        # path deliberately skipped the service clear - and so the activity was never
        # ended at all. Reporter's lock screen sat frozen at 98% / 0:00 for an hour
        # after the cycle finished, and on an earlier run the chronometer counted
        # upward to 4:12:20; it survives until Apple's ~8 h expiry or a manual
        # dismiss. Handover keeps the mobile app to one visible entry at a time: the
        # first live tick clears the lifecycle tag (dropping the start alert), and
        # cycle end clears this one after the finished alert has been delivered.
        self._live_notification_tag = f"ha_washdata_{self.entry_id}_live"
        self._start_event_fired = False
        self._cycle_start_time: datetime | None = None
        # Per-cycle UUID: the identity token the live match and the cycle-end tail
        # check so work for one cycle never lands on the next, even when both share
        # a second-resolution start_time. (Named for the live_match ranking
        # snapshots it first keyed; those were removed in 0.5.8.)
        self._ranking_snapshot_cycle_id: str = ""

        # State
        self._current_power = 0.0
        # Power-based Off detection (issue #284): timestamp at which power first fell
        # below the power-off threshold while in a terminal state. None = not currently
        # below (or feature disabled). Cleared on new cycle / when power rises.
        self._power_off_below_since: datetime | None = None
        # One-shot cancellable timer armed when power first drops below the power-off
        # threshold, so the terminal->Off reset fires promptly after power_off_delay
        # instead of waiting for the next 60s expiry poll. Cancelled on power rise /
        # nag hold / terminal reset / new cycle.
        self._remove_power_off_timer: Any | None = None
        self._last_reading_time: datetime | None = None
        self._last_real_reading_time: datetime | None = None # Track last real sensor update
        # Register item 266, restart hazard. The silence clock and sensor value a
        # restored cycle's snapshot carried, consumed once by the setup read; and
        # the report that read took, which the resync must not mistake for a
        # missed one (it would hand the restart's own write back as fresh).
        self._restored_sensor_clock: tuple[datetime, float | None] | None = None
        self._setup_report_ts: datetime | None = None
        self._noise_events: list[datetime] = []
        self._noise_max_powers: list[float] = []
        self._last_match_result = None
        self._last_phase_estimate_time = None
        self._matching_task: Task[Any] | None = None
        # True while a power reading is being handled: the handler refreshes the
        # entities once when it returns, so what it calls does not (register item 456).
        self._in_power_event = False
        self._cycle_end_task: Task[Any] | None = None
        self._banked_tail_repair_task: Task[Any] | None = None
        # Detached store-touching tasks (matching trigger, active-cycle clear,
        # post-cycle processing) tracked so async_shutdown can cancel them before a
        # reload/unload swaps the ProfileStore out from under them.
        self._background_tasks: set[Task[Any]] = set()
        self._is_shutdown: bool = False
        self._last_state_save = 0.0
        self._last_cycle_end_time: datetime | None = None
        self._remove_state_expiry_timer = None

        # Components
        # Coerced (audit F7 finding): a non-numeric stored value raised inside every
        # match tick (`float()` in the initial commit, `<` in the unmatch check).
        unmatch_threshold = option_float(
            config_entry.options.get(CONF_PROFILE_UNMATCH_THRESHOLD),
            DEFAULT_PROFILE_UNMATCH_THRESHOLD,
        )
        self._unmatch_threshold = unmatch_threshold

        self.profile_store = ProfileStore(
            hass,
            self.entry_id,
            min_duration_ratio=config_entry.options.get(
                CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
            ),
            max_duration_ratio=config_entry.options.get(
                CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
            ),
            save_debug_traces=config_entry.options.get(CONF_SAVE_DEBUG_TRACES, False),
            unmatch_threshold=unmatch_threshold,
            device_name=config_entry.title,
        )
        self.profile_store.dtw_bandwidth = float(
            config_entry.options.get(CONF_DTW_BANDWIDTH, DEFAULT_DTW_BANDWIDTH)
        )
        # Stage-4 energy discriminator: integrated energy for WM/washer-dryer,
        # mean power elsewhere (see analysis.stage4_energy_mode).
        self.profile_store.energy_mode = analysis.stage4_energy_mode(self.device_type)
        # Which cycle categories may shape a profile. The store cannot read entry
        # options, so the manager pushes this in (same as energy_mode above).
        self.profile_store.evidence_sources = config_entry.options.get(
            CONF_PROFILE_EVIDENCE_SOURCES, DEFAULT_PROFILE_EVIDENCE_SOURCES
        )
        self.learning_manager = LearningManager(
            hass, self.entry_id, self.profile_store, self.device_type,
            device_name=config_entry.title,
            # Its store saves and suggestion passes are cancelled with ours on an
            # unload (audit MANAGER-13), not left writing the swapped-out store.
            spawn=self._spawn_tracked,
        )
        self.recorder = CycleRecorder(hass, self.entry_id, device_name=config_entry.title)
        self._store_bridge: Any = None  # lazy community-store bridge (online features)

        # Priority: Options > Data > Default
        min_power = config_entry.options.get(
            CONF_MIN_POWER, config_entry.data.get(CONF_MIN_POWER, DEFAULT_MIN_POWER)
        )
        off_delay = _option_then_data(
            config_entry, CONF_OFF_DELAY, resolve_off_delay_default(self.device_type)
        )
        progress_reset_delay = config_entry.options.get(
            CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY
        )
        self._load_runtime_options(config_entry)
        # Device-scaled ceiling for the unmatched (expected == 0) zombie guard (#404).
        # Not a user option; purely a function of device_type, so it is recomputed
        # alongside device_type on reconfigure.
        self._unmatched_watchdog_ceiling = float(
            DEFAULT_UNMATCHED_WATCHDOG_CEILING_BY_DEVICE.get(
                self.device_type, DEFAULT_UNMATCHED_WATCHDOG_CEILING
            )
        )
        # Coerced here rather than at the point of use. Both thresholds are compared
        # against a match confidence inside the cycle-end tail, and that tail runs as
        # a spawned task: a non-numeric option (an import file is hand-editable, and
        # strip_null_options only removes nulls) raised there instead, killing the task
        # before async_add_cycle and losing the whole cycle. Same #389 failure shape,
        # one step later. Falls back to the default rather than to 0, which would
        # silently auto-label everything.

        self._profile_match_interval = int(
            config_entry.options.get(
                CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
            )
        )
        self._notify_before_end_minutes = int(
            config_entry.options.get(
                CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES
            )
        )
        self._load_notify_services(config_entry)
        self._notify_actions = list(
            cast(list[dict[str, Any]], config_entry.options.get(CONF_NOTIFY_ACTIONS, []) or [])
        )
        self._notify_people = list(
            config_entry.options.get(CONF_NOTIFY_PEOPLE, []) or []
        )
        self._notify_only_when_home = bool(
            config_entry.options.get(
                CONF_NOTIFY_ONLY_WHEN_HOME, DEFAULT_NOTIFY_ONLY_WHEN_HOME
            )
        )
        self._notify_fire_events = bool(
            config_entry.options.get(CONF_NOTIFY_FIRE_EVENTS, DEFAULT_NOTIFY_FIRE_EVENTS)
        )
        self._notify_live_interval_seconds = int(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_INTERVAL_SECONDS,
                DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS,
            )
        )
        self._notify_live_overrun_percent = int(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_OVERRUN_PERCENT,
                DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT,
            )
        )
        self._notify_live_chronometer = bool(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_CHRONOMETER,
                DEFAULT_NOTIFY_LIVE_CHRONOMETER,
            )
        )
        self._notify_live_sticky = bool(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_STICKY, DEFAULT_NOTIFY_LIVE_STICKY
            )
        )
        self._notify_live_click_action = str(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_CLICK_ACTION, DEFAULT_NOTIFY_LIVE_CLICK_ACTION
            )
            or ""
        ).strip()
        self._notify_live_silent = bool(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_SILENT, DEFAULT_NOTIFY_LIVE_SILENT
            )
        )
        self._notify_timeout_seconds = int(
            config_entry.options.get(
                CONF_NOTIFY_TIMEOUT_SECONDS, DEFAULT_NOTIFY_TIMEOUT_SECONDS
            )
        )

        self._logger.info(
            "Manager init: min_power=%sW, off_delay=%ss, type=%s",
            min_power,
            off_delay,
            self.device_type,
        )

        # One builder for the constructor, the options reload and every replay
        # harness (audit F2 / DETECT-12; item 351 was these copies drifting).
        config = build_detector_config(
            config_entry.options, config_entry.data, self.device_type
        )
        self._config = config


        def profile_matcher_wrapper(
            readings: list[tuple[datetime, float]],
        ) -> tuple[str | None, float, float, str | None] | None:
            """Wraps profile store matching logic with detector callback signature.

            The real match is offloaded to an async task that calls
            ``detector.update_match`` later, so this returns ``None`` in that case -
            the detector's contract is "None == async offload, I'll be called back"
            (see ``_try_profile_match``). It must NOT return a placeholder tuple:
            a non-empty tuple is truthy, so the detector would feed it straight into
            ``update_match`` on every match tick, spuriously logging the
            "invalid raw_expected_duration 0.0" debug line and momentarily zeroing
            ``_last_match_confidence`` between real async updates. Only the manual
            override path below returns a genuine synchronous tuple.
            """
            # Manual program override
            if self._manual_program_active and self._current_program:
                elapsed_seconds = 0.0
                if len(readings) > 1:
                    elapsed_seconds = max(
                        0.0,
                        (readings[-1][0] - readings[0][0]).total_seconds(),
                    )

                expected_duration = float(self._matched_profile_duration or 0.0)
                manual_phase = self.profile_store.check_phase_match(
                    self._current_program,
                    elapsed_seconds,
                )
                # Elements 9 and 10 matter even here. update_match CLEARS
                # _matched_tail_power and _matched_terminal_high for any tuple
                # shorter than this, on the sound reasoning that a newly matched
                # profile must not inherit the previous one's tail - but a manual
                # pin names its profile, so the answer is to supply that profile's
                # own values rather than nothing. Left empty, the #364 tail guard
                # and the #399 anti-crease spin wait both sat inert for every
                # hand-picked program, so a washer could finalize in the quiet
                # before its terminal spin and record the spin as a second cycle.
                # Elements 5-8 stay False: a manual pin is certain by definition,
                # so there is no mismatch or ambiguity to report.
                terminal_high = self._terminal_high_for_guards(self._current_program)
                return (
                    self._current_program,
                    1.0,
                    expected_duration,
                    manual_phase or "Manual",
                    False,
                    False,
                    False,
                    False,
                    self.profile_store.profile_tail_power(self._current_program),
                    terminal_high,
                    # Element 11 (register item 297): same reasoning as elements 9 and 10 - a
                    # manual pin names its profile, so it must supply that
                    # profile's own measurements rather than leaving the guard
                    # inert. Without it a hand-picked program would still bank
                    # Smart Termination's confirmation delay as cycle time.
                    self.profile_store.profile_terminal_quiet_seconds(
                        self._current_program
                    ),
                    # Element 12: no candidates on a manual pin; 0.0 is "none".
                    0.0,
                    # Element 13 (register item 384): the pinned profile's
                    # user-vouched length, which floors a dishwasher's kept tail.
                    self.profile_store.profile_trusted_min_duration(
                        self._current_program
                    ),
                )

            if not readings:
                return None

            # Which cycle this match is FOR, captured NOW: the detector calls this
            # synchronously from an active state. The task body runs later, and on
            # the reading that finishes the cycle the detector has already reset by
            # then, so identity captured inside the task described the NEXT state
            # and the stale result re-armed the finished detector (audit LIVE-01).
            identity = self._match_identity()
            self._spawn_tracked(
                self._async_perform_combined_matching(readings, identity)
            )
            return None

        self.detector = CycleDetector(
            config,
            self._on_state_change,
            self._on_cycle_end,
            profile_matcher=profile_matcher_wrapper,
            device_name=config_entry.title,
            end_confidence_provider=self._ml_end_confidence,
            terminal_drop_provider=self._terminal_drop_provider,
        )
        self._ml_end_expectation_cache: tuple[str, dict[str, float]] | None = None
        # (cycle_count, earliest_quiet_offset|None, peak_range|None) for the
        # terminal-drop baselines; keyed by cycle count so it auto-invalidates
        # when history grows.
        self._terminal_drop_cache: (
            tuple[int, float | None, tuple[float, float] | None] | None
        ) = None
        # Cycle count an executor refresh of the baseline is in-flight/done for, so
        # the loop never recomputes it (issue #311) and never double-schedules.
        self._terminal_drop_refresh_n: int | None = None

        self._remove_listener = None
        self._remove_report_listener = None  # state_reported (unchanged re-reports) #363/#329
        self._remove_external_trigger_listener = None  # External cycle end trigger
        self._remove_watchdog = None
        self._watchdog_interval = int(
            config_entry.options.get(
                CONF_WATCHDOG_INTERVAL,
                resolve_watchdog_interval_default(self.device_type),
            )
        )
        self._sampling_interval = float(
            config_entry.options.get(
                CONF_SAMPLING_INTERVAL,
                resolve_sampling_interval_default(self.device_type),
            )
        )
        self._current_program: str = "off"
        self._time_remaining: float | None = None
        self._total_duration: float | None = None
        self._cycle_progress: float = 0.0
        self._smoothed_progress: float = 0.0  # Smoothed progress tracking for EMA
        self._smoothed_for_program: str | None = None  # the program that EMA tracks
        # Live projected total energy/cost for the running cycle (None until a
        # reliable progress estimate exists). Derived from accumulated energy and
        # the (ML-blended) progress fraction; surfaced as progress-sensor attrs.
        self._projected_energy_wh: float | None = None
        self._projected_cost: float | None = None
        # Runtime overrun anomaly (soft, visible; never a notification). "none"
        # or "overrun" once a running cycle exceeds its matched profile's expected
        # duration by CYCLE_OVERRUN_ANOMALY_RATIO. Surfaced as a state-sensor attr
        # and frozen onto the cycle at end for panel badging.
        self._cycle_anomaly: str = "none"
        self._overrun_ratio: float = 0.0
        # Where this run maps onto its matched profile's envelope, 0-1, from the
        # DTW alignment that already runs for the verified-pause decision. Visible
        # only: nothing reads it back. None until an alignment has produced one.
        self._envelope_position: float | None = None
        # Post-cycle anomaly cache: holds energy/underrun anomaly from the last
        # completed cycle so sensor attributes can surface them after idle.
        self._last_cycle_post_anomaly: dict = {}
        self._cycle_completed_time: datetime | None = None  # Track when cycle finished
        self._progress_reset_delay: int = int(
            progress_reset_delay
        )  # Reset progress after idle
        self._last_reading_time: datetime | None = None
        self._current_power: float = 0.0
        self._last_estimate_time: datetime | None = None
        self._last_match_ambiguous: bool = False
        self._matched_profile_duration: float | None = None
        self._last_match_confidence: float = 0.0
        # Stage-5 companion to the above (item 206): what the SELECTED group member
        # earned on its own curve, vs the group's score in _last_match_confidence.
        # None for every non-group match, and the label gate then behaves exactly as
        # before. Kept as a separate field rather than replacing the confidence,
        # because the confidence is what the detector's end-detection gates read.
        self._last_member_confidence: float | None = None


        self._remove_maintenance_scheduler = None
        self._remove_ml_training_scheduler = None
        self._ml_training_failures = 0  # consecutive gate failures for auto-disable
        self._ml_training_running = False  # True while a training run is in flight
        self._profile_sample_repair_stats: dict[str, int] | None = None


        # Pump Monitor state
        self._pump_stuck_duration: int = int(
            config_entry.options.get(CONF_PUMP_STUCK_DURATION, DEFAULT_PUMP_STUCK_DURATION)
        )
        self._pump_stuck: bool = False  # True once the stuck threshold has fired for this cycle

        self._manual_program_active: bool = False
        # The user's standing program choice for the current or next cycle (#411).
        # Rehydrated from the store during setup so arming survives a restart.
        self._armed_program: str | None = None
        self._notified_start: bool = False
        self._notified_pre_completion: bool = False
        self._last_match_result: Any = None  # Stores full MatchResult object
        self._score_history: dict[str, list[float]] = {}  # Tracks recent scores for trend analysis
        self._match_persistence_counter: dict[str, int] = {}  # Tracks consecutive matches
        self._unmatch_persistence_counter: int = 0  # Tracks consecutive low-confidence matches
        self._current_match_candidate: str | None = None  # Pending profile name

    _MATCH_ACTIVE_STATES = (STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING)

    def _match_identity(self) -> tuple[str, datetime | None, bool]:
        """The cycle a live match belongs to: snapshot id, cycle start, active."""
        return (
            self._ranking_snapshot_cycle_id,
            self.detector.current_cycle_start,
            self.detector.state in self._MATCH_ACTIVE_STATES,
        )

    def _match_still_current(self, identity: tuple[str, datetime | None, bool]) -> bool:
        """True while the cycle a live match was dispatched for is still running.

        Checked after EVERY await in the match task (items 388e, audit LIVE-01/02):
        the cycle can end, or the next one start, while the matcher or the
        alignment check is in the executor. A match dispatched while no cycle was
        active (a direct call) only requires the same cycle identity.
        """
        token, start, was_active = identity
        cur_token, cur_start, is_active = self._match_identity()
        if cur_token != token or cur_start != start:
            return False
        return is_active or not was_active

    async def _async_perform_combined_matching(
        self,
        readings: list[tuple[datetime, float]],
        identity: tuple[str, datetime | None, bool] | None = None,
    ) -> None:
        """PRIMARY matching task: Updates both Manager and Detector using best method."""
        self._logger.debug(
            "Matching trigger: readings=%d, task_exists=%s",
            len(readings) if readings else 0,
            getattr(self, "_matching_task", None) is not None
        )
        # Prevent concurrent matching tasks
        current_task = self._matching_task
        if current_task is not None and not current_task.done():
            self._logger.debug("Matching skipped: previous task still running")
            return

        try:
            if not readings:
                self._logger.debug("Matching skipped: no readings")
                return

            # Skip match entirely when no real profiles exist — nothing to match against.
            if not self.profile_store.has_real_profiles:
                self._logger.debug("Matching skipped: no real profiles configured yet")
                return

            self._matching_task = self.hass.async_create_task(
                self._async_do_perform_matching(
                    readings, identity if identity is not None else self._match_identity()
                )
            )
        except Exception as e:
            self._logger.error("Perform combined matching trigger failed: %s", e)

    async def _async_do_perform_matching(
        self,
        readings: list[tuple[datetime, float]],
        identity: tuple[str, datetime | None, bool] | None = None,
    ) -> None:
        """Inner task to handle actual matching logic."""
        if identity is None:
            identity = self._match_identity()
        try:
            end_time = readings[-1][0]
            start_time = readings[0][0]
            current_duration = (end_time - start_time).total_seconds()
            # Which cycle this match is FOR (item 388e) is `identity`, captured at
            # dispatch. The await below yields the loop, and the cycle can end - or
            # the next one start - before it returns; applying the result then
            # rewrote `_current_program` and the expected duration on a finished
            # cycle, and pushed it into the detector, so the next cycle reached
            # RUNNING already "matched".

            # 1. RUN BETTER ASYNC MATCHING
            # in_progress: this is the live match on a cycle that is still running,
            # so Stage 4 grades each candidate on the same elapsed stretch instead
            # of on its complete duration/energy (#400). The two final-match paths
            # (_run_final_match_from_cycle_data, _async_process_cycle_end) leave it
            # off - there the cycle really is complete.
            result = await self.profile_store.async_match_profile(
                 readings,
                 current_duration,
                 in_progress=True,
                 # Lets the #288 prefix term ignore longer programmes that never
                 # pause below this threshold, so cannot explain a quiet (#424).
                 stop_threshold_w=float(self.detector.config.stop_threshold_w),
            )

            if not self._match_still_current(identity):
                self._logger.debug(
                    "Discarding a live match that returned after its cycle ended "
                    "(detector %s)", self.detector.state,
                )
                return

            # 2. UPDATE MANAGER STATE (Estimates, Program Name, etc.)
            self._last_match_result = result
            self._last_match_ambiguous = result.is_ambiguous

            # --- Switching Logic (Temporal Persistence) ---
            # The rules live in `match_rules` (audit PLAYGROUND-01/03), shared with
            # the Playground replay so it makes the decisions made here. Step 1:
            # margin, divergence revert, persistence. `profile_name` is the tick's
            # name from here on - "detecting..." after a divergence revert, which is
            # also what the detector is handed below.
            switch_state = _read_switch_state(self)
            tick = match_rules.begin_tick(
                switch_state, result, self._match_persistence, current_duration
            )
            _write_switch_state(self, switch_state, tick.log)
            profile_name = tick.profile_name
            confidence = tick.confidence
            matched_duration = tick.matched_duration
            phase_name = tick.phase_name

            # Step 2 (match_rules.decide_switch): Case 1 initial commit, Case 2
            # decisive-margin / trend switch, Case 3 unmatch, and the switch itself.
            switch_state = _read_switch_state(self)
            switch_log = match_rules.decide_switch(
                switch_state,
                tick,
                result,
                self._match_persistence,
                self._unmatch_threshold,
            )
            _write_switch_state(self, switch_state, switch_log)

            self._last_estimate_time = utc_now()

            # Update score history for all candidates to track trends
            match_rules.record_scores(switch_state, result.candidates)

            # Note: _update_remaining_only() and notify move to end of flow

            # 3. UPDATE DETECTOR (Envelopes, Deferral, State Transitions)
            current_matched = self.detector.matched_profile
            verified_pause = getattr(self.detector, "_verified_pause", False)
            current_power = readings[-1][1] if readings else 0.0

            # --- Envelope Verification for Mismatches & Pauses ---
            # Check alignment if we have a match and power is low, to confirm if
            # this is a legitimate (auto-detected) pause or a mismatch.  Skipped
            # while the user has explicitly paused (issue #306): the user pause is
            # authoritative and must not be re-judged by the envelope heuristic
            # (see the verified_pause override below).
            stop_thresh = float(self.detector.config.stop_threshold_w)
            alignment: tuple[bool, float] | None = None
            if match_rules.needs_alignment_check(
                current_matched, current_power, stop_thresh, self._is_user_paused
            ):
                formatted = power_data_to_offsets(cast(list[list[Any] | tuple[Any, ...]], readings))
                try:
                    profile_store_any = cast(Any, self.profile_store)
                    verify_alignment = profile_store_any.async_verify_alignment
                    is_confirmed, mapped_time, _ = (
                        await verify_alignment(current_matched, formatted)
                    )
                    if not self._match_still_current(identity):
                        # Same race as the matcher await above (audit LIVE-02): a
                        # verified pause and the old match must not land on a
                        # finished detector and carry into the next cycle.
                        self._logger.debug(
                            "Discarding a live match whose alignment check returned "
                            "after its cycle ended"
                        )
                        return
                except Exception as e: # pylint: disable=broad-exception-caught
                    self._logger.error(
                        "Alignment verification crashed for profile %s: %s",
                        current_matched, e, exc_info=True
                    )
                    is_confirmed = False
                    mapped_time = 0.0
                alignment = (is_confirmed, mapped_time)

            # Confirmed alignment, the 95%-of-span release, the high-power clear,
            # the #375 sustained-quiet release and the user-pause override
            # (match_rules.decide_verified_pause). The quiet tally is the GAP-FREE
            # one: a telemetry outage is unobserved time and must not satisfy the
            # #375 quiet floor (falls back to the plain tally on an older detector).
            pause = match_rules.decide_alignment_pause(
                verified_pause=verified_pause,
                current_matched=current_matched,
                alignment=alignment,
                envelope_span=self.profile_store.envelope_time_span,
            )
            if pause.envelope_position is not None:
                self._envelope_position = pause.envelope_position
            _emit_rule_log(self._logger, pause.log)
            pause = match_rules.decide_pause_release(
                verified_pause=pause.verified_pause,
                current_matched=current_matched,
                current_power=current_power,
                stop_threshold_w=getattr(self.detector.config, "stop_threshold_w", 5.0),
                user_paused=self._is_user_paused,
                expected_duration=self.detector.expected_duration_seconds,
                current_duration=current_duration,
                time_below=getattr(
                    self.detector,
                    "_time_below_threshold_gapfree",
                    getattr(self.detector, "_time_below_threshold", 0.0),
                ),
                program=self._current_program,
            )
            _emit_rule_log(self._logger, pause.log)
            verified_pause = pause.verified_pause

            # --- Consistency Override (verified pause / confident mismatch) ---
            switch_state = _read_switch_state(self)
            override_log = match_rules.consistency_override(
                switch_state, tick, result, verified_pause, self.profile_store.get_profile
            )
            _write_switch_state(self, switch_state, override_log)

            # Register item 469(b): an ambiguous tick in ENDING engages no new
            # verified pause (the detector refuses its match if it would defer).
            pause = match_rules.hold_in_ending(
                ending=self.detector.state == STATE_ENDING,
                is_ambiguous=bool(result.is_ambiguous),
                current_matched=current_matched,
                prev_verified=getattr(self.detector, "_verified_pause", False),
                verified_pause=verified_pause,
                user_paused=self._is_user_paused,
            )
            _emit_rule_log(self._logger, pause.log)
            # Push updates to detector
            self.detector.set_verified_pause(pause.verified_pause)
            # A divergence revert revokes the detector's match (no name, revoke flag).
            profile_name, revoke = match_rules.detector_match(tick, result)
            # Built by name (audit DETECT-15); see CycleDetector.MatchContext.
            terminal_high = self._terminal_high_for_guards(profile_name)
            # Half-interval matching until the first commit (audit LIVE-17).
            self.detector.set_match_committed(
                match_rules.program_is_committed(self._current_program)
            )
            self.detector.update_match(MatchContext(
                profile_name=profile_name,
                confidence=confidence,
                expected_duration=matched_duration,
                phase_name=phase_name,
                is_confident_mismatch=revoke,
                is_ambiguous=result.is_ambiguous,
                is_prefix_ambiguous_full_shape=result.is_prefix_ambiguous_full_shape,
                tail_power=(
                    self.profile_store.profile_tail_power(profile_name) if profile_name else None
                ),
                terminal_high=terminal_high,
                # Item 297: the profile's measured post-activity quiet span, which
                # bounds how much of Smart Termination's delay the tail may keep.
                terminal_quiet_s=(
                    self.profile_store.profile_terminal_quiet_seconds(profile_name)
                    if profile_name else None
                ),
                # Item 330: from the FULL candidate population, carried on the
                # result (`result.candidates` is the top 5 and would hide the very
                # programme an ambiguous match may be warning about).
                longest_candidate_s=float(
                    getattr(result, "longest_candidate_duration_s", 0.0) or 0.0
                ),
                # Item 384: the shortest length the user has vouched for.
                trusted_min_s=(
                    self.profile_store.profile_trusted_min_duration(profile_name)
                    if profile_name else None
                ),
                # Audit DETECT-16: the pause catalogue for the hazard end gate.
                pause_catalogue=(
                    self.profile_store.profile_pause_catalogue(
                        profile_name, float(self.detector.config.stop_threshold_w)
                    ) if profile_name else None
                ),
                # #452: the same below the near-stop ceiling, for the stall display.
                # Lazy: read only when a flat run is long enough to be judged.
                stall_catalogue=(
                    functools.partial(
                        self.profile_store.profile_pause_catalogue,
                        profile_name,
                        standby_near_stop_ceiling(self.detector.config.stop_threshold_w),
                    ) if profile_name else None
                ),
            ))

            # --- LOGGING (Unified) ---
            self._logger.info(
                "Profile match attempt: name=%s, confidence=%.3f, duration=%.0fs, samples=%d",
                profile_name, confidence, current_duration, len(readings),
            )

            self._update_remaining_only()

            # --- START NOTIFICATION LOGIC ---
            # Fallback for restart-recovery: fires only if the immediate notification in
            # _on_state_change was missed (e.g., HA restarted mid-cycle before snapshot).
            if not getattr(self, "_notified_start", False):
                if self._notify_fire_events and not self._start_event_fired:
                    self.hass.bus.async_fire(
                        EVENT_CYCLE_STARTED,
                        {
                            "entry_id": self.entry_id,
                            "device_name": self.config_entry.title,
                            "device_type": self.device_type,
                            "program": self._current_program,
                            "start_time": (
                                self._cycle_start_time or utc_now()
                            ).isoformat(),
                        },
                    )
                    self._start_event_fired = True

                if self._notify_start_services or self._notify_actions:
                    msg_template = self.config_entry.options.get(
                        CONF_NOTIFY_START_MESSAGE, DEFAULT_NOTIFY_START_MESSAGE
                    )
                    msg = self._safe_format_template(
                        msg_template,
                        fallback_template=DEFAULT_NOTIFY_START_MESSAGE,
                        device=self.config_entry.title,
                        program=self._current_program,
                    )
                    # B4: append a peak-rate advisory tip when the current price is
                    # at/above the configured threshold. Purely informational.
                    tip = self._peak_rate_tip(
                        self.config_entry.options, self._resolve_energy_price()
                    )
                    if tip:
                        msg = f"{msg}\n{tip}"

                    self._dispatch_notification(
                        msg,
                        event_type=NOTIFY_EVENT_START,
                        extra_vars={
                            "program": self._current_program,
                            "tag": self._lifecycle_tag,
                        },
                    )
                    self._notified_start = True
                    self._logger.info("Sent start notification for program '%s'", self._current_program)

                    # Ensure pre-completion notifications never precede cycle-start signaling.
                    self._check_pre_completion_notification()

            self._check_live_progress_notification()
            self._notify_update_deferrable()

        except Exception as e:
            self._logger.error("Perform combined matching failed: %s", e, exc_info=True)

    @property
    def top_candidates(self) -> list[dict[str, Any]]:
        """Return a lightweight list of top candidates from the last match."""
        if not self._last_match_result:
            return []

        # Get raw list from ranking (best) or candidates
        raw_list: list[dict[str, Any]] = []
        if hasattr(self._last_match_result, "ranking") and self._last_match_result.ranking:
            raw_list = self._last_match_result.ranking
        elif hasattr(self._last_match_result, "candidates"):
            raw_list = self._last_match_result.candidates

        # SANITIZE: Remove heavy power arrays before sending to Home Assistant attributes
        return _sanitize_ranking(raw_list)

    @property
    def phase_description(self) -> str | None:
        """The current phase: a name from the matched profile's ranges, or None.

        Only the *functional* progress-driven phase (the visual per-profile phase
        configurator's ranges, indexed by the live ML-blended progress). None -
        the sensor's ``unknown`` - when no range applies (audit PROGRESS-11): the
        old fallbacks were the matcher's nearest-range guess, power heuristics
        ("Spinning" over 200 W, so a 2 kW heater read Spinning) and the detector
        sub-state, all English free text no translation could reach. The cycle
        state itself stays on the state sensor (a translated slug).
        """
        return self._current_phase_from_progress()

    def _current_phase_from_progress(self) -> str | None:
        """Live phase from the profile's configured ranges + ML-blended progress.

        This is the *merge* of the visual phase configurator with the runtime
        estimator: one phase definition (the per-profile ranges the user draws),
        indexed by the smoothed progress fraction rather than raw elapsed seconds,
        so overrun/underrun cycles still name the phase correctly. The fraction
        maps onto the matched profile's expected duration (or the ranges' end if
        they run longer), so ranges read at their real minutes (audit
        PROGRESS-10). Returns None when not running, no profile is matched, the
        profile has no phase ranges, or no range covers this point. Never raises.
        """
        return progress_mod.current_phase(
            self.profile_store,
            self.detector.state,
            self._current_program,
            self._cycle_progress,
            self._matched_profile_duration,
        )

    @property
    def match_ambiguity(self) -> bool:
        """Return True if the last match was ambiguous."""
        if self._last_match_result and hasattr(self._last_match_result, "is_ambiguous"):
            return self._last_match_result.is_ambiguous
        return False

    @property
    def last_ambiguity_margin(self) -> float | None:
        """Return the score margin between top-1 and top-2 candidates, or None."""
        result = self._last_match_result
        if result is None:
            return None
        return getattr(result, "ambiguity_margin", None)

    @property
    def match_uncertainty(self) -> dict[str, Any] | None:
        """Top two of an undecided live match, for the panel (MATCH-DECIDE-15).

        Display only (``match_rules.live_match_uncertainty``). None while idle, on
        a hand-picked program, or once the match is decided.
        """
        try:
            if self.manual_program_active:
                return None
            if self.detector.state not in (
                STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_USER_PAUSED, STATE_ENDING,
            ):
                return None
            return match_rules.live_match_uncertainty(
                self._last_match_result, self._current_program
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    # Note: last_match_details property is defined later in the class
    # It returns MatchResult from _last_match_result
    async def _attempt_state_restoration(self) -> None:
        """Attempt to restore active cycle state from storage."""
        active_snapshot = self.profile_store.get_active_cycle()

        # Check current power state first
        state = self.hass.states.get(self.power_sensor_entity_id)
        current_power = 0.0
        power_is_valid = False

        if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            _restore_power = _finite_power(state.state)
            if _restore_power is None:
                # Not numeric, or nan/inf: treat as 0W and do not restore by power.
                # Leaving power_is_valid False is what keeps a non-finite reading out
                # of the restore decision, whose comparisons would silently be False.
                self._logger.debug(
                    "Power sensor %s state %r is not a finite number during "
                    "restoration; treating as 0W and not restoring by power",
                    self.power_sensor_entity_id,
                    getattr(state, "state", None),
                )
            else:
                current_power = _restore_power
                power_is_valid = True

        should_restore = False
        active_snapshot_to_restore: dict[str, Any] | None = (
            active_snapshot if isinstance(active_snapshot, dict) else None
        )

        snap = active_snapshot_to_restore or {}
        snap_last_real = _snapshot_time(snap.get("last_real_reading_time"))

        def is_live_silent_tail(now: datetime) -> bool:
            """A cycle waiting out a silent low-power tail (register item 266).

            The windows in ``is_viable_restore`` age the snapshot, but a silent
            tail is not stale: the watchdog keeps such a cycle for as long as its
            staleness budget allows silence (1 h, a dishwasher 4 h, the profile's
            remaining time, a verified pause), and saves came only with real
            readings, so a drying dishwasher's snapshot aged past 30 min while
            Home Assistant was watching it. Restore what the watchdog would hold:
            same budget, judged on the sensor's real silence, and only while the
            sensor still reads low. Past that budget the watchdog would have
            force-ended it, so it is genuinely stale and is still dropped.
            """
            if snap_last_real is None or not power_is_valid:
                return False
            if current_power >= self._config.min_power:
                return False
            if snap.get("state") not in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
                return False
            try:
                waiting = float(snap.get("time_below") or 0.0) > 0.0
                expected = float(snap.get("expected_duration") or 0.0)
            except (TypeError, ValueError, OverflowError):
                return False
            if not waiting:
                return False
            start = _snapshot_time(snap.get("current_cycle_start"))
            elapsed = (now - start).total_seconds() if start is not None else 0.0
            budget = self._low_power_silence_budget_s(
                elapsed, expected, bool(snap.get("is_user_paused"))
            )
            silence = (now - snap_last_real).total_seconds()
            if silence > budget:
                return False
            self._logger.info(
                "Restoring a cycle in a silent low-power tail: sensor silent "
                "%.0fs, within the watchdog's %.0fs budget",
                silence,
                budget,
            )
            return True

        # Helper to check if a snapshot is viable
        def is_viable_restore(last_save_time: datetime) -> bool:
            now = utc_now()
            # Handle timezone mismatch gracefully
            if last_save_time.tzinfo is None:
                # Assume naive means local system time, convert to aware
                last_save_time = last_save_time.replace(tzinfo=now.tzinfo)

            age = (now - last_save_time).total_seconds()

            # Unconditional restore window (30 mins)
            if age < 1800:
                return True
            # Extended window if power is confirmed HIGH (60 mins)
            if (
                age < 3600
                and power_is_valid
                and current_power >= self._config.min_power
            ):
                return True
            return is_live_silent_tail(now)

        last_save = self.profile_store.get_last_active_save()
        if last_save and last_save.tzinfo is None:
            # Normalize naive legacy timestamps to system time
            last_save = last_save.replace(tzinfo=dt_util.now().tzinfo)

        if active_snapshot_to_restore is not None and last_save and is_viable_restore(last_save):
            should_restore = True
            age = (utc_now() - last_save).total_seconds()
            age = (utc_now() - last_save).total_seconds()
            self._logger.info(
                "Found recently saved active cycle (last_save=%s, age=%.0fs), restoring...",
                last_save,
                age
            )
            # strict extension logic unless the user wants to enforce it.
            active_snapshot_to_restore["sub_state"] = (
                active_snapshot_to_restore.get("sub_state") or "Restored"
            )
            # NOTE: We disable dynamic min duration enforcement on recovery since we
            # might have missed data
            active_snapshot_to_restore["dynamic_min_duration"] = None

        # A snapshot of a cycle that is already stored is not an active cycle: an
        # HA restart between the cycle-end tail persisting the cycle and clearing
        # the snapshot used to restore it, so the cycle ended twice - two stored
        # copies, two "finished" pushes and lifetime energy counted twice on a
        # TOTAL_INCREASING sensor (audit MANAGER-04).
        if should_restore and active_snapshot_to_restore is not None:
            snap_start = dt_util.parse_datetime(
                str(active_snapshot_to_restore.get("current_cycle_start") or "")
            )
            if snap_start is not None and snap_start.tzinfo is None:
                snap_start = snap_start.replace(tzinfo=dt_util.now().tzinfo)
            if snap_start is not None:
                for stored in self.profile_store.get_past_cycles()[-5:]:
                    stored_start = dt_util.parse_datetime(
                        str(stored.get("start_time") or "")
                    )
                    if stored_start is not None and stored_start.tzinfo is None:
                        stored_start = stored_start.replace(tzinfo=dt_util.now().tzinfo)
                    if stored_start is not None and abs(
                        (stored_start - snap_start).total_seconds()
                    ) < 1.0:
                        self._logger.info(
                            "Active-cycle snapshot belongs to stored cycle %s; "
                            "not restoring it",
                            stored.get("id"),
                        )
                        should_restore = False
                        active_snapshot_to_restore = None
                        await self.profile_store.async_clear_active_cycle()
                        break

        # (The "resurrection" fallback that re-opened the last stored cycle when it
        # was interrupted or force-stopped less than 20 min ago is gone: it popped
        # the stored cycle and re-ended it on any restart or settings save, so the
        # lifetime energy was counted twice, a second "finished" push went out and
        # a force_stopped cycle was re-stored as completed - audit MANAGER-03. The
        # 60 s active-cycle snapshot is what covers a mid-cycle restart.)

        if should_restore and active_snapshot_to_restore:
            try:
                # A cycle paused by the user while still in STARTING is promoted
                # to PAUSED before restoration so that (a) the false-start abort
                # cannot fire on the first low-power reading, and (b) the PAUSED
                # branch of the restore block below re-applies the user-pause state
                # and re-asserts verified_pause (issue #306).
                if (
                    active_snapshot_to_restore.get("state") == STATE_STARTING
                    and active_snapshot_to_restore.get("is_user_paused")
                ):
                    active_snapshot_to_restore = {
                        **active_snapshot_to_restore,
                        "state": STATE_PAUSED,
                    }

                if self.detector.restore_state_snapshot(active_snapshot_to_restore) is False:
                    # The detector already logged the traceback.
                    await self._async_keep_failed_restore(
                        active_snapshot_to_restore,
                        last_save,
                        self.detector.restore_error or "restore_state_snapshot failed",
                        traceback_logged=True,
                    )
                    return

                # Anti-wrinkle keepalive anchor (#339). The keepalive in
                # _handle_state_expiry needs a "sensor last spoke" timestamp, but
                # _last_real_reading_time is only ever set by a live reading, so a
                # restart into ANTI_WRINKLE with an already-silent plug leaves it
                # None and the keepalive can never fire -- pinning the mode until
                # the next cycle. The snapshot save is driven by real readings, so
                # last_save is the best available proxy. Scoped to ANTI_WRINKLE so
                # no other timer sees a synthetic anchor.
                if self.detector.state == STATE_ANTI_WRINKLE and last_save:
                    self._last_real_reading_time = snap_last_real or last_save
                    # The cycle end started the expiry timer that runs that
                    # keepalive; a restart into the tail must start it too.
                    self._start_state_expiry_timer()

                # Restore if in any active state (Running, Paused, Ending)
                if self.detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
                    # The sensor's pre-restart silence clock (register item 266).
                    # The setup read then decides whether the entity's startup
                    # write is a new report or just the old value written again.
                    if snap_last_real is not None:
                        self._last_real_reading_time = snap_last_real
                        self._restored_sensor_clock = (
                            snap_last_real,
                            _finite_power(snap.get("last_sensor_power")),
                        )

                    # Restore manual program flag if present
                    self._manual_program_active = active_snapshot_to_restore.get(
                        "manual_program", False
                    )

                    # Restore the external energy-meter snapshot (issue #316) so a
                    # restart mid-cycle keeps an accurate start->end delta.
                    self._energy_meter_start = active_snapshot_to_restore.get(
                        "energy_meter_start"
                    )
                    self._energy_meter_source = active_snapshot_to_restore.get(
                        "energy_meter_source"
                    )

                    # Restore the dynamic price timeline (#426). Prices recorded
                    # before the restart stay valid; a change that happened while HA
                    # was down is recovered from the recorder at cycle end, guided by
                    # the restart gap recorded a few lines below.
                    self._price_timeline = _coerce_price_timeline(
                        active_snapshot_to_restore.get("price_timeline")
                    )

                    # If we restored into a low-power state, ensure we don't
                    # immediately quit. For now we just log this; the cycle
                    # detector's off_delay will handle actual shutdown.
                    if power_is_valid and current_power < self._config.min_power:
                        self._logger.debug(
                            "Restored active cycle in low-power state "
                            "(power=%.2fW < min_power=%.2fW); waiting for "
                            "detector off_delay before marking as finished",
                            current_power,
                            self._config.min_power,
                        )

                    # The committed program and its expected duration. Until the
                    # snapshot carried them, the detector's match was all there was:
                    # the last tick's raw winner, not necessarily the displayed
                    # program, and no `_matched_profile_duration`, so the ETA was blank
                    # and the first tick another candidate led reverted the program to
                    # "detecting..." (decide_switch's no-duration branch).
                    if "committed_program" in active_snapshot_to_restore:
                        committed = active_snapshot_to_restore.get("committed_program")
                        committed_duration = active_snapshot_to_restore.get(
                            "committed_program_duration"
                        )
                    else:  # snapshot from an older version
                        committed = self.detector.matched_profile
                        committed_duration = self.detector.expected_duration_seconds
                    committed_profile = (
                        self.profile_store.get_profile(committed)
                        if isinstance(committed, str) and committed
                        else None
                    )
                    if committed_profile is not None:
                        self._current_program = committed
                        self._matched_profile_duration = self._profile_duration(
                            committed_duration
                        ) or self._profile_duration(committed_profile.get("avg_duration"))
                        self._logger.info(
                            "Restored washer cycle with profile: %s (duration=%.0fs)",
                            self._current_program,
                            self._matched_profile_duration or 0.0,
                        )
                    else:
                        self._current_program = "detecting..."

                    # Re-pin a manual program override across the restart (#404
                    # secondary bug). _manual_program_active was restored above, but
                    # _matched_profile_duration was not; without this the manual
                    # matcher tuple would feed expected_duration=0 on the next tick,
                    # wipe the detector's matched_profile, and drop the cycle to
                    # "detecting..." + the unmatched watchdog guard. The duration is
                    # re-read from the (possibly since-learned) profile, so an empty
                    # profile stays active with a None duration rather than being lost.
                    if self._manual_program_active:
                        manual_name = (
                            active_snapshot_to_restore.get("manual_program_name")
                            or self.detector.matched_profile
                        )
                        profile = (
                            self.profile_store.get_profile(manual_name)
                            if manual_name
                            else None
                        )
                        if profile is not None:
                            self._current_program = manual_name
                            self._matched_profile_duration = self._profile_duration(
                                profile.get("avg_duration")
                            )
                            self._logger.info(
                                "Restored manual program override: %s (duration=%.0fs)",
                                manual_name,
                                self._matched_profile_duration or 0.0,
                            )
                        else:
                            # The chosen profile was deleted while HA was down.
                            self._manual_program_active = False
                            self._logger.info(
                                "Manual program %r no longer exists after restart; "
                                "reverting to auto-detect",
                                manual_name,
                            )

                    # Restore persisted start-notification/event flags from snapshot.
                    self._notified_start = bool(
                        active_snapshot_to_restore.get("notified_start", False)
                    )
                    self._start_event_fired = bool(
                        active_snapshot_to_restore.get("start_event_fired", False)
                    )
                    # One-shot per-cycle state (audit MANAGER-08); junk is dropped.
                    _fired = active_snapshot_to_restore.get("fired_cycle_timers")
                    self._fired_cycle_timers = {
                        int(i) for i in (_fired if isinstance(_fired, list) else [])
                        if isinstance(i, int) and not isinstance(i, bool)
                    }
                    self._notified_pre_completion = bool(
                        active_snapshot_to_restore.get("notified_pre_completion", False)
                    )
                    _gaps = active_snapshot_to_restore.get("restart_gaps")
                    self._restart_gaps = [
                        g for g in (_gaps if isinstance(_gaps, list) else [])
                        if isinstance(g, dict)
                    ]
                    self._live_activity_started = bool(
                        active_snapshot_to_restore.get("live_activity_started", False)
                    )

                    # Restore user-pause state from snapshot.
                    self._is_user_paused = bool(
                        active_snapshot_to_restore.get("is_user_paused", False)
                    )
                    _pause_start_raw = active_snapshot_to_restore.get("user_pause_start")
                    self._user_pause_start = (
                        dt_util.parse_datetime(_pause_start_raw)
                        if isinstance(_pause_start_raw, str) and _pause_start_raw
                        else None
                    )
                    self._total_user_paused_seconds = float(
                        active_snapshot_to_restore.get("total_user_paused_seconds", 0.0)
                    )
                    # get_state_snapshot() does not persist the detector's
                    # verified-pause flag, so a user-paused cycle would otherwise be
                    # finalized on the first ENDING timeout after a restart (issue
                    # #306).  Re-assert it so the pause survives the reload.
                    if self._is_user_paused:
                        self.detector.set_verified_pause(True)

                    # Auto-open dishwasher: arm the dwell if we restored into ENDING
                    # with the door already open. _on_state_change is not called
                    # during restoration, so the transition guard there would not fire.
                    self._maybe_arm_door_end_dwell_if_open()

                    # Record the restart gap so the Cycles tab can shade it and
                    # anomaly detection can surface it.  Only meaningful when
                    # last_save is known and the dark period exceeds 30 s.
                    # Matching always uses real readings only (Option B from the
                    # gap-fill analysis); synthetic fill is intentionally NOT added
                    # to _power_readings to prevent circular-bias inflation.
                    if last_save:
                        gap_end = utc_now()
                        gap_secs = (gap_end - last_save).total_seconds()
                        if gap_secs > 30:
                            self._restart_gaps.append({
                                "start_ts": last_save.isoformat(),
                                "end_ts": gap_end.isoformat(),
                                "gap_seconds": round(gap_secs, 1),
                                "profile": self.detector.matched_profile,
                                # `_last_match_confidence`: the detector has no
                                # `match_confidence` attribute, so this was always
                                # None (audit DETECT-09).
                                "match_confidence": getattr(
                                    self.detector, "_last_match_confidence", None
                                ),
                            })
                            self._logger.info(
                                "HA restart gap recorded: %.0fs (%.1f min) in active cycle; "
                                "power trace has a hole — no synthetic fill (matching integrity)",
                                gap_secs,
                                gap_secs / 60,
                            )

                    self._start_watchdog()
                else:
                    await self.profile_store.async_clear_active_cycle()
            except Exception:  # noqa: BLE001
                await self._async_keep_failed_restore(
                    active_snapshot_to_restore, last_save, traceback.format_exc()
                )
        else:
            if last_save:
                age = (utc_now() - last_save).total_seconds()
                self._logger.info("Active cycle too stale (age=%.0fs), clearing", age)
            await self.profile_store.async_clear_active_cycle()

    async def _async_keep_failed_restore(
        self,
        snapshot: dict[str, Any],
        last_save: datetime | None,
        error: str,
        *,
        traceback_logged: bool = False,
    ) -> None:
        """A snapshot that could not be restored: say so, keep it, start from OFF.

        It used to be deleted after one log line, so the running cycle vanished
        without a trace (register item 266 follow-up). It is kept, the last one per
        device, for the diagnostics download; the active slot is cleared so the
        next restart does not trip over it again, and the detector starts OFF so
        the appliance's next cycle is detected normally. Never raises.
        """
        now = utc_now()
        age = (now - last_save).total_seconds() if last_save is not None else None
        self._logger.warning(
            "Could not restore the active cycle (state %r, snapshot age %s); kept it "
            "for diagnostics and starting from OFF%s",
            snapshot.get("state"),
            f"{age:.0f}s" if age is not None else "unknown",
            "" if traceback_logged else f":\n{error}",
        )
        try:
            self.detector.reset()
        except Exception:  # noqa: BLE001
            self._logger.debug("Detector reset after a failed restore raised", exc_info=True)
        await self.profile_store.async_keep_failed_restore({
            "failed_at": now.isoformat(),
            "last_active_save": last_save.isoformat() if last_save is not None else None,
            "age_s": round(age, 1) if age is not None else None,
            "error": error,
            "snapshot": snapshot,
        })
        try:
            await self.profile_store.async_clear_active_cycle()
        except Exception:  # noqa: BLE001
            self._logger.debug("Clearing a failed snapshot raised", exc_info=True)

    async def _async_repair_banked_tails(self) -> None:
        """One-time repair of cycles that banked the end-of-cycle confirmation
        delay as cycle time (register item 297).

        Runs in the background after setup. Never raises: the store method already
        swallows its own failures and leaves the history untouched, and this
        wrapper exists only so a scheduling error cannot surface as an unhandled
        task exception.
        """
        try:
            result = await self.profile_store.async_repair_banked_tails(
                float(self.detector.config.stop_threshold_w), self.device_type
            )
            if result.get("repaired"):
                self._logger.info(
                    "Repaired %d of %d stored cycles that had banked the "
                    "end-of-cycle confirmation delay (%.0f min reclaimed); profile "
                    "averages and envelopes rebuilt.",
                    result["repaired"],
                    result["examined"],
                    result["reclaimed_s"] / 60.0,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._logger.warning("Banked-tail repair could not run: %s", exc)

    def async_schedule_banked_tail_repair(self) -> None:
        """Run the one-time banked-tail repair now if the marker is armed.

        `async_setup` is one caller, and covers the storage migration that arms
        the marker at v12->v13. An **import** arms it too - both
        `async_import_data` and `async_import_data_selective` set it for a payload
        old enough to carry banked tails - and an import does not reliably reload
        the entry: the WS handlers only call `async_update_entry` when the payload
        brings options with it, so a cycles-only import, or any selective import
        with `apply_settings=False`, left the marker set and the imported tails
        feeding `avg_duration`, the ETA and Smart Termination until the next
        restart.

        Cheap when there is nothing to do - the marker is the whole test, and the
        repair clears it. A second call while the first is still in flight is a
        no-op rather than a second walk of the history: the repair only clears the
        marker at the end, so the check alone would not stop two concurrent runs
        from rebuilding the same envelopes.
        """
        if self.profile_store.banked_tail_repair_pending() is not True:
            return
        existing = self._banked_tail_repair_task
        if existing is not None and not existing.done():
            return
        self._banked_tail_repair_task = self._spawn_tracked(
            self._async_repair_banked_tails()
        )

    def _terminal_high_for_guards(self, profile_name: str | None) -> Any:
        """Element 10: the matched profile's last high-power block, or None.

        Two callers with two different bars, and the bar has to travel with the
        block (register item 351):

        * **anti-crease** (#399) measures against ``anti_wrinkle_max_power``, the
          dryer's "a tumble is below this" level. Only meaningful while
          anti-wrinkle is on, and it sends a triple.
        * **the standby-band finalise** (#296 / #445) shares the same predicate,
          and used to get nothing at all: element 10 was supplied ONLY when
          anti-wrinkle was enabled, and `DEFAULT_ANTI_WRINKLE_ENABLED` is False,
          so `_anticrease_spin_pending` returned False immediately and a washer
          could finalise on the quiet plateau before its final spin - recording
          that spin as a second cycle. Measured over the 273-cycle replay corpus:
          the standby band fires on 14 cycles, 11 of them with the guard inert,
          and 6 of those 11 have a reading above `min_power` still ahead, i.e.
          would split. This arms it against a share of the cycle's own peak, the
          same `STANDBY_BAND_MAX_FRACTION` the plateau test uses, which recovers
          5 of the 6 for one extra bounded wait. Sent as a QUAD so
          `_high_power_seconds_since` counts against that bar too.

        Returns None when nothing applies, which leaves the guard exactly as
        inert as it was - the fail-open direction every input here takes.
        """
        # One implementation, shared with the Playground's sim tuple - see
        # `cycle_detector.terminal_high_for_guards` for why it is not inlined here.
        return terminal_high_for_guards(
            self.profile_store,
            self.detector.config,
            getattr(self.detector, "_cycle_max_power", 0.0),
            profile_name,
        )

    def _load_runtime_options(self, config_entry: Any) -> None:
        """Manager-level tunables, read the same way at setup and on reload (388c)."""
        options = config_entry.options
        self._no_update_active_timeout = float(
            options.get(CONF_NO_UPDATE_ACTIVE_TIMEOUT, DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT)
        )
        self._low_power_no_update_timeout = float(
            options.get(CONF_LOW_POWER_NO_UPDATE_TIMEOUT, 3600.0)
        )
        self._off_delay = float(
            _option_then_data(
                config_entry, CONF_OFF_DELAY, resolve_off_delay_default(self.device_type)
            )
        )
        # Coerced here rather than at the point of use. Both thresholds are compared
        # against a match confidence inside the cycle-end tail, and that tail runs as
        # a spawned task: a non-numeric option (an import file is hand-editable, and
        # strip_null_options only removes nulls) raised there instead, killing the task
        # before async_add_cycle and losing the whole cycle. Same #389 failure shape,
        # one step later. Falls back to the default rather than to 0, which would
        # silently auto-label everything.
        self._learning_confidence = option_float(
            options.get(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE),
            DEFAULT_LEARNING_CONFIDENCE,
        )
        self._auto_label_confidence = option_float(
            options.get(CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE),
            DEFAULT_AUTO_LABEL_CONFIDENCE,
        )
        # Clamped to >= 1, the same floor `SuggestionEngine` applies to this key:
        # its interval cap is computed FROM this number, and zero would disable the
        # gate rather than tighten it (every `counter >= 0` is true).
        self._match_persistence = option_int(
            options.get(CONF_MATCH_PERSISTENCE, DEFAULT_MATCH_PERSISTENCE),
            DEFAULT_MATCH_PERSISTENCE,
            minimum=1,
        )
        self._unmatch_threshold = option_float(
            options.get(CONF_PROFILE_UNMATCH_THRESHOLD), DEFAULT_PROFILE_UNMATCH_THRESHOLD
        )
        store = getattr(self, "profile_store", None)
        if store is not None and hasattr(store, "_unmatch_threshold"):
            store._unmatch_threshold = self._unmatch_threshold  # pylint: disable=protected-access
        self._progress_reset_delay = int(
            options.get(CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY)
        )
        # Read here, not only in the constructor (audit MANAGER-15): the last
        # manager tunable an options reload left stale.
        self._noise_events_threshold = option_int(
            options.get(
                CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
                DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
            ),
            DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
        )

    async def async_setup(self) -> None:
        """Set up the manager."""
        await self.profile_store.async_load()
        try:
            _trans = await translation.async_get_translations(
                self.hass, self.hass.config.language, "options", {DOMAIN}
            )
            # Cache of manager-side fixed UI-string templates resolved from the
            # options.error.* translation namespace (timer notifications, the
            # duration-vs-typical finish variable, and the live "waiting" message).
            # The inline English mirrors the strings.json values so the fallback is
            # never the sole source and the code stays behaviour-identical in English.
            self._timer_ui_strings = {
                k: _trans.get(f"component.{DOMAIN}.options.error.{k}", v)
                for k, v in {
                    "timer_default_message": "{device}: {minutes} min timer",
                    "timer_pause_action_title": "Resume Cycle",
                    "timer_pause_body_suffix": "The cycle is paused. Open the WashData panel to resume.",
                    "unload_dismiss_action_title": "Stop reminding",
                    "vs_typical_longer": "{pct}% longer than usual",
                    "vs_typical_shorter": "{pct}% shorter than usual",
                    "notify_live_waiting_message": "{device}: No profile matched yet.",
                }.items()
            }
        except Exception:  # noqa: BLE001
            pass

        # Re-scope custom phases stranded under another device type (#450). Cheap,
        # idempotent and saves only on a change, so it runs on every setup rather
        # than behind a one-shot marker: a reconfigure that changes device_type
        # would otherwise strand the phases all over again.
        try:
            rescoped = await self.profile_store.async_repair_custom_phase_scope(
                self.device_type
            )
            if rescoped:
                self._logger.info(
                    "Re-scoped %d custom phase(s) that were stored under another "
                    "device type and could not be shown, edited or deleted.",
                    rescoped,
                )
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.exception("Failed re-scoping custom phases for %s", self.entry_id)

        # Preset maintenance reminders count from when they first apply (#461).
        self._sync_maintenance_baselines()

        # Repair broken sample_cycle_id references (can happen after aggressive retention)
        try:
            stats = await self.profile_store.async_repair_profile_samples()
            self._profile_sample_repair_stats = stats
            if stats.get("profiles_repaired", 0) or stats.get(
                "cycles_labeled_as_sample", 0
            ):
                self._logger.warning(
                    "Repaired profile sample references for %s: %s",
                    self.entry_id,
                    stats,
                )
                await self.profile_store.async_save()
        except Exception:
            self._logger.exception(
                "Failed repairing profile sample references for %s", self.entry_id
            )

        # The idle display's standby level, from the stored cycles (#452).
        await self._async_refresh_standby_level()

        # Subscribe to power sensor updates (state changes AND unchanged re-reports)
        self._subscribe_power_sensor()

        # Attempt to restore state (BEFORE starting listener)
        await self._attempt_state_restoration()

        # Restore last cycle end time to ensure ghost cycle suppression works after restart
        try:
            cycles = self.profile_store.get_past_cycles()
            if cycles:
                # Find last completed cycle with a valid end time
                for cycle in reversed(cycles):
                    if cycle.get("end_time") and cycle.get("status") == "completed":
                        ts = dt_util.parse_datetime(cycle["end_time"])
                        if ts:
                            self._last_cycle_end_time = ts
                            self._logger.debug("Restored last cycle end time: %s", ts)
                            break
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.debug("Failed to restore last cycle end time")

        # Load recorder state
        await self.recorder.async_load()

        # Rehydrate a program armed before a restart (#411). Arming is an
        # idle-time action, so the wait between picking a program and starting the
        # machine can easily span a Home Assistant restart.
        try:
            self._armed_program = self.profile_store.get_armed_program()
            if self._armed_program:
                self._logger.info(
                    "Program %r is armed for the next cycle", self._armed_program
                )
        except Exception:  # pylint: disable=broad-exception-caught
            self._armed_program = None

        # Force initial update from current state (in case it's already stable)
        self._read_power_state_at_setup()

        # Trigger migration/compression of old cycle format
        # This is safe to run repeatedly (it skips already compressed cycles)
        await self.profile_store.async_migrate_cycles_to_compressed()

        # Backfill match_confidence for labeled cycles that predate the field.
        # Tracked (audit MANAGER-13): it saves the store.
        self._spawn_tracked(self.profile_store.async_backfill_match_confidence())

        # Subscribe to external cycle end trigger (if enabled)
        await self._setup_external_end_trigger()

        # Subscribe to door sensor (if configured)
        await self._setup_door_sensor_listener()

        # Subscribe to the unload confirmation entity (if configured, #451)
        await self._setup_unload_confirm_listener()

        # Subscribe to the dynamic energy price entity (if configured, #426)
        await self._setup_price_listener()

        # Subscribe to person presence changes for notification gating
        await self._setup_notify_people_listener()

        # HA does not unload entries on a stop, so the stop gets its own hook to keep
        # the held notifications and a fresh snapshot (audit MANAGER-16); what it
        # kept is re-dispatched once HA has started and notify services exist.
        self._remove_ha_stop_listener = self.hass.bus.async_listen(
            EVENT_HOMEASSISTANT_STOP, self._async_on_ha_stop
        )
        self._remove_notify_queue_restore = async_at_started(
            self.hass, self._schedule_notify_queue_restore
        )

        # Register schedulers (maintenance + ML training). These are also re-
        # registered on every config reload; calling them here ensures they
        # survive HA restarts without requiring the user to re-save settings.
        await self._setup_maintenance_scheduler()
        self._setup_ml_training_scheduler()

        # One-time repair of cycles that banked Smart Termination's confirmation
        # delay as cycle time (register item 297). Flagged by the v12->v13 storage
        # migration and done here rather than in the migration itself, because
        # deciding where a cycle's real activity ended needs stop_threshold_w and
        # that lives in entry.options. Idempotent, marked done in the store, and it
        # never raises - a failed repair leaves the history untouched.
        # `is True` rather than a truthiness check: the flag is written as a real
        # bool by the migration, so anything else here is a stub or a hand-edited
        # store and must not trigger a rewrite of the user's history.
        #
        # LAST in async_setup, deliberately. The repair's own cycle loop takes no
        # awaits, but it then awaits an envelope rebuild per touched profile, and
        # every await hands the loop back to the rest of setup - which rewrites the
        # very cycles it is rebuilding from. `async_repair_profile_samples` can
        # drop a profile or re-point its sample, and
        # `async_migrate_cycles_to_compressed` replaces `power_data` wholesale.
        # Running last also means those legacy ISO-offset traces are already
        # converted and therefore trimmable: started earlier, such a cycle gets its
        # duration corrected and its trace left as it was, because `_safe_offset`
        # rejects an ISO string and `kept` comes back empty.
        # Backgrounded, not awaited. It walks up to 200 stored traces and rebuilds
        # envelopes, and anything awaited inside async_setup is billed to the
        # integration's reported startup time (register item 158 / #408). Nothing
        # needs it before the first cycle ends. Tracked, because it writes to the
        # ProfileStore: an untracked task would keep writing to the store a reload
        # had already swapped out.
        self.async_schedule_banked_tail_repair()

    def _load_notify_services(self, config_entry: ConfigEntry) -> None:
        """Load notification service lists, migrating legacy single-service config."""
        self._notify_start_services = list(config_entry.options.get(CONF_NOTIFY_START_SERVICES, []) or [])
        self._notify_finish_services = list(config_entry.options.get(CONF_NOTIFY_FINISH_SERVICES, []) or [])
        self._notify_live_services = list(config_entry.options.get(CONF_NOTIFY_LIVE_SERVICES, []) or [])
        raw_timers = config_entry.options.get(CONF_NOTIFY_CYCLE_TIMERS, []) or []
        self._notify_cycle_timers = [
            t for t in raw_timers
            if isinstance(t, dict) and isinstance(t.get("offset_minutes"), (int, float)) and t["offset_minutes"] > 0
        ]
        # Backward compat: migrate old single notify_service + notify_events to new per-event lists
        if not (self._notify_start_services or self._notify_finish_services or self._notify_live_services):
            _old_svc = config_entry.options.get(CONF_NOTIFY_SERVICE, "")
            _old_events = list(config_entry.options.get(CONF_NOTIFY_EVENTS, []) or [])
            if _old_svc:
                if not _old_events or NOTIFY_EVENT_START in _old_events:
                    self._notify_start_services = [_old_svc]
                if not _old_events or NOTIFY_EVENT_FINISH in _old_events:
                    self._notify_finish_services = [_old_svc]
                if not _old_events or NOTIFY_EVENT_LIVE in _old_events:
                    self._notify_live_services = [_old_svc]

    async def async_reload_config(self, config_entry: ConfigEntry) -> None:
        """
        Reload configuration options without interrupting running cycle detection.

        Updates detector config in-place.
        Handles Power Sensor entity change by reconnecting listener.
        """
        self._logger.info("Reloading configuration for %s", self.entry_id)
        # Replace reference
        self.config_entry = config_entry

        # Check if power sensor changed
        self._apply_power_sensor_option()

        # Update device type
        self.device_type = config_entry.options.get(
            CONF_DEVICE_TYPE,
            config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
        )
        # Propagate to learning pipeline (captured at construction time)
        self.learning_manager.device_type = self.device_type
        self.learning_manager.suggestion_engine.device_type = self.device_type
        # A saved reminder change (or an import) may switch a preset maintenance
        # reminder on or off: re-stamp its counting origin (#461).
        self._sync_maintenance_baselines()
        # Recompute the device-scaled unmatched-guard ceiling (#404): it tracks
        # device_type, which the reconfigure flow can change.
        self._unmatched_watchdog_ceiling = float(
            DEFAULT_UNMATCHED_WATCHDOG_CEILING_BY_DEVICE.get(
                self.device_type, DEFAULT_UNMATCHED_WATCHDOG_CEILING
            )
        )

        # Update detector config in-place
        old_min_power = self.detector.config.min_power
        old_off_delay = self.detector.config.off_delay
        old_interrupted_min = self.detector.config.interrupted_min_seconds

        # Every detector field from the one builder the constructor uses, applied
        # onto the live config object in place (audit F2): the field-by-field copy
        # that lived here is how the reload drifted from setup (items 351, 388a).
        new_detector_config = build_detector_config(
            config_entry.options, config_entry.data, self.device_type
        )
        new_min_power = new_detector_config.min_power
        new_off_delay = new_detector_config.off_delay
        new_interrupted_min = new_detector_config.interrupted_min_seconds
        apply_detector_config(self.detector.config, new_detector_config)

        self.profile_store.dtw_bandwidth = float(
            config_entry.options.get(CONF_DTW_BANDWIDTH, DEFAULT_DTW_BANDWIDTH)
        )
        # Re-plumbed on every reload, not just at construction (issue #407): this is
        # a panel checkbox, and an options change reloads in place without rebuilding
        # the store, so without this line the toggle only took effect on an HA restart.
        self.profile_store.save_debug_traces = config_entry.options.get(
            CONF_SAVE_DEBUG_TRACES, False
        )
        # Stage-4 energy discriminator: integrated energy for WM/washer-dryer,
        # mean power elsewhere (see analysis.stage4_energy_mode).
        self.profile_store.energy_mode = analysis.stage4_energy_mode(self.device_type)
        # Which cycle categories may shape a profile. Changing it changes every
        # profile's curve, so rebuild them all now: envelopes are otherwise only rebuilt
        # on a cycle end or a label change, so the user would tick the box and see
        # nothing happen for days.
        _prev_evidence = self.profile_store.evidence_sources
        self.profile_store.evidence_sources = config_entry.options.get(
            CONF_PROFILE_EVIDENCE_SOURCES, DEFAULT_PROFILE_EVIDENCE_SOURCES
        )
        if self.profile_store.evidence_sources != _prev_evidence:
            self._logger.info(
                "Profile evidence sources changed %s -> %s; rebuilding all envelopes",
                list(_prev_evidence), list(self.profile_store.evidence_sources),
            )
            # Tracked, not fire-and-forget: this writes to the ProfileStore, so a
            # reload/unload mid-rebuild must be able to cancel it before the store is
            # swapped out (see _spawn_tracked).
            self._spawn_tracked(self.profile_store.async_rebuild_all_envelopes())

        # Pump Monitor setting
        self._pump_stuck_duration = int(
            config_entry.options.get(CONF_PUMP_STUCK_DURATION, DEFAULT_PUMP_STUCK_DURATION)
        )

        if (
            old_min_power != new_min_power
            or old_off_delay != new_off_delay
            or old_interrupted_min != new_interrupted_min
        ):
            self._logger.info(
                "Updated detector config: min_power %.1fW->%.1fW, off_delay %ds->%ds, "
                "interrupted_min %ds->%ds",
                old_min_power,
                new_min_power,
                old_off_delay,
                new_off_delay,
                old_interrupted_min,
                new_interrupted_min,
            )

        # Update profile matching parameters
        old_min_ratio, old_max_ratio = self.profile_store.get_duration_ratio_limits()

        new_min_ratio = float(
            config_entry.options.get(
                CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
            )
        )
        new_max_ratio = float(
            config_entry.options.get(
                CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
            )
        )

        if old_min_ratio != new_min_ratio or old_max_ratio != new_max_ratio:
            self.profile_store.set_duration_ratio_limits(
                min_ratio=new_min_ratio, max_ratio=new_max_ratio
            )
            self._logger.info(
                "Updated duration ratios: min %.2f→%.2f, max %.2f→%.2f",
                old_min_ratio,
                new_min_ratio,
                old_max_ratio,
                new_max_ratio,
            )

        # Update match interval
        old_interval = self._profile_match_interval
        new_interval = int(
            config_entry.options.get(
                CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
            )
        )
        if old_interval != new_interval:
            self._profile_match_interval = new_interval
            self._logger.info("Updated match interval: %ds→%ds", old_interval, new_interval)

        # Update other configurable options


        # Update notification settings
        self._load_notify_services(config_entry)
        self._notify_actions = list(
            cast(list[dict[str, Any]], config_entry.options.get(CONF_NOTIFY_ACTIONS, []) or [])
        )
        self._notify_script = None
        self._notify_people = list(
            config_entry.options.get(CONF_NOTIFY_PEOPLE, []) or []
        )
        self._notify_only_when_home = bool(
            config_entry.options.get(
                CONF_NOTIFY_ONLY_WHEN_HOME, DEFAULT_NOTIFY_ONLY_WHEN_HOME
            )
        )
        self._notify_fire_events = bool(
            config_entry.options.get(CONF_NOTIFY_FIRE_EVENTS, DEFAULT_NOTIFY_FIRE_EVENTS)
        )
        self._notify_before_end_minutes = int(
            config_entry.options.get(
                CONF_NOTIFY_BEFORE_END_MINUTES, DEFAULT_NOTIFY_BEFORE_END_MINUTES
            )
        )
        self._notify_live_interval_seconds = int(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_INTERVAL_SECONDS,
                DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS,
            )
        )
        self._notify_live_overrun_percent = int(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_OVERRUN_PERCENT,
                DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT,
            )
        )
        self._notify_live_chronometer = bool(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_CHRONOMETER,
                DEFAULT_NOTIFY_LIVE_CHRONOMETER,
            )
        )
        self._notify_live_sticky = bool(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_STICKY, DEFAULT_NOTIFY_LIVE_STICKY
            )
        )
        self._notify_live_click_action = str(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_CLICK_ACTION, DEFAULT_NOTIFY_LIVE_CLICK_ACTION
            )
            or ""
        ).strip()
        self._notify_live_silent = bool(
            config_entry.options.get(
                CONF_NOTIFY_LIVE_SILENT, DEFAULT_NOTIFY_LIVE_SILENT
            )
        )
        self._notify_timeout_seconds = int(
            config_entry.options.get(
                CONF_NOTIFY_TIMEOUT_SECONDS, DEFAULT_NOTIFY_TIMEOUT_SECONDS
            )
        )

        # Reload door sensor / pause config
        self._pause_cuts_power = bool(config_entry.options.get(CONF_PAUSE_CUTS_POWER, False))
        self._door_sensor_entity = config_entry.options.get(CONF_DOOR_SENSOR_ENTITY) or None
        self._door_opens_at_end = bool(
            config_entry.options.get(CONF_DOOR_OPENS_AT_END, DEFAULT_DOOR_OPENS_AT_END)
        )
        self._door_end_dwell_seconds = int(
            config_entry.options.get(CONF_DOOR_END_DWELL_SECONDS, DEFAULT_DOOR_END_DWELL_SECONDS)
        )
        self._notify_unload_delay_minutes = int(
            config_entry.options.get(
                CONF_NOTIFY_UNLOAD_DELAY_MINUTES, DEFAULT_NOTIFY_UNLOAD_DELAY_MINUTES
            )
        )
        self._notify_unload_repeat = bool(
            config_entry.options.get(
                CONF_NOTIFY_UNLOAD_REPEAT, DEFAULT_NOTIFY_UNLOAD_REPEAT
            )
        )
        self._unload_confirm_entity = config_entry.options.get(
            CONF_UNLOAD_CONFIRM_ENTITY
        ) or None
        self._unload_track_without_door = bool(
            config_entry.options.get(
                CONF_UNLOAD_TRACK_WITHOUT_DOOR, DEFAULT_UNLOAD_TRACK_WITHOUT_DOOR
            )
        )

        # Re-subscribe to external cycle end trigger
        await self._setup_external_end_trigger()

        # Re-subscribe to door sensor. Cancel any dwell armed for the previous door
        # config first — after a sensor/auto-open/dwell change the old sensor may no
        # longer emit the close event that cancels it, so a stale timer could finalize
        # the cycle on outdated config. Re-evaluate for the new configuration after.
        self._cancel_door_end_dwell()
        await self._setup_door_sensor_listener()
        self._maybe_arm_door_end_dwell_if_open()

        # Re-subscribe to the unload confirmation entity (#451).
        await self._setup_unload_confirm_listener()

        # Re-subscribe to the dynamic energy price entity (#426). A changed entity
        # (or the toggle being turned off) takes effect from here on; the samples
        # already recorded for a running cycle stay - they were true when taken.
        await self._setup_price_listener()

        # Re-subscribe to person presence changes for notification gating
        await self._setup_notify_people_listener()

        # If a cycle is currently active and live notifications are now enabled,
        # reset counters and fire the first live notification immediately so the
        # user doesn't have to wait for the next power sensor poll.
        if self.detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            if self._notify_live_services or self._notify_actions:
                # Counters and timers only: a settings save is not a cycle
                # boundary, so the live activity must stay "already started".
                self._reset_live_notification_state(keep_activity_started=True)
                self._check_live_progress_notification()

        # Trigger entity updates to reflect any changes
        async_dispatcher_send(self.hass, f"ha_washdata_update_{self.entry_id}")

        # Schedule midnight maintenance if enabled
        await self._setup_maintenance_scheduler()

        # Schedule on-device ML retraining if enabled (Stage 4, gated)
        self._setup_ml_training_scheduler()

        # Update sampling interval
        old_sampling = self._sampling_interval
        new_sampling = float(
            config_entry.options.get(
                CONF_SAMPLING_INTERVAL,
                resolve_sampling_interval_default(self.device_type),
            )
        )
        if old_sampling != new_sampling:
            self._sampling_interval = new_sampling
            self._logger.info(
                "Updated sampling interval: %.1fs -> %.1fs", old_sampling, new_sampling
            )

        # Watchdog cadence: like sampling above, this was only read at construction, so a
        # changed CONF_WATCHDOG_INTERVAL (or a device-type change selecting a new default)
        # otherwise kept the old cadence until the manager was recreated. Re-arm an active
        # watchdog so the new interval takes effect mid-cycle.
        old_watchdog = self._watchdog_interval
        new_watchdog = int(
            config_entry.options.get(
                CONF_WATCHDOG_INTERVAL,
                resolve_watchdog_interval_default(self.device_type),
            )
        )
        if old_watchdog != new_watchdog:
            self._watchdog_interval = new_watchdog
            self._logger.info(
                "Updated watchdog interval: %ds -> %ds", old_watchdog, new_watchdog
            )
            if self._remove_watchdog:  # active cycle: cancel and re-register at the new cadence
                self._stop_watchdog()
                self._start_watchdog()

        # Manager-level settings the reload used to skip (item 388c): one loader
        # shared with __init__, so the two cannot drift again.
        self._load_runtime_options(config_entry)

        # RESTORE STATE (only if recent enough, otherwise treat as stale) - but
        # never over a cycle that is running right now (item 388b). An options
        # reload keeps the live detector; restoring the snapshot (up to 60 s old)
        # over it rolled the cycle back: readings dropped, a phantom restart gap
        # written onto the cycle, the quiet timers reset.
        # The in-place reload never restores the snapshot at all (audit MANAGER-04):
        # the live detector is always fresher than a snapshot of up to 60 s ago, and
        # an IDLE detector with a snapshot still present means the cycle-end tail
        # has not cleared it yet - restoring then re-opened the finished cycle and
        # ended it a second time (two stored copies, two pushes, double energy).
        self._logger.debug(
            "Options reload: keeping the live detector state (%s), not restoring "
            "the snapshot",
            self.detector.state,
        )
        # The idle display's standby level depends on the stop/start thresholds.
        await self._async_refresh_standby_level()

        self._logger.info("Configuration reloaded successfully")

    def _apply_power_sensor_option(self) -> None:
        """Re-point the power listener at the configured sensor, if it changed.

        Never while a cycle is under way (_SENSOR_SWAP_BLOCKED_STATES): the swap is
        parked in ``_pending_power_sensor`` and ``_on_state_change`` applies it once
        the detector leaves those states (audit MANAGER-11). It used to be dropped
        until the next reload, while the panel already showed the new sensor.
        """
        new_sensor = self.config_entry.options.get(
            CONF_POWER_SENSOR, self.config_entry.data.get(CONF_POWER_SENSOR)
        )
        if not new_sensor or new_sensor == self.power_sensor_entity_id:
            self._pending_power_sensor = None
            return
        d_state = self.detector.state
        if d_state in _SENSOR_SWAP_BLOCKED_STATES:
            # Park the change but continue with the other config updates: returning
            # from the reload would silently drop every setting saved alongside the
            # sensor in the same submission.
            self._pending_power_sensor = new_sensor
            self._logger.warning(
                "Power sensor change %s -> %s deferred: the detector is in state %s. "
                "It takes effect when the current cycle ends.",
                self.power_sensor_entity_id,
                new_sensor,
                d_state,
            )
            return
        self._pending_power_sensor = None
        self._logger.info(
            "Power sensor changed: %s -> %s", self.power_sensor_entity_id, new_sensor
        )
        self.power_sensor_entity_id = new_sensor
        # Re-attach change + report listeners to the new sensor
        # (helper removes the old ones first).
        self._subscribe_power_sensor()
        # Force update from new sensor
        state = self.hass.states.get(self.power_sensor_entity_id)
        if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            _reload_power = _finite_power(state.state)
            if _reload_power is not None:
                self.detector.process_reading(_reload_power, utc_now())
            else:
                self._logger.debug(
                    "Initial power value for %s after config reload is not a "
                    "finite number: %r",
                    self.power_sensor_entity_id,
                    state.state,
                )

    async def _async_apply_pending_power_sensor(self) -> None:
        """Apply a parked sensor swap (see _apply_power_sensor_option).

        A task, not inline in _on_state_change: that callback runs inside the
        detector's own process_reading, and the swap feeds the new sensor's
        reading straight back into it. HA starts tasks eagerly, so yield once
        first or the body would still run inside that call.
        """
        await asyncio.sleep(0)
        if self._is_shutdown or self._pending_power_sensor is None:
            return
        self._apply_power_sensor_option()
        self._notify_update()

    def _spawn_tracked(self, coro: Coroutine[Any, Any, Any]) -> Task[Any]:
        """Create a detached task and track it so shutdown can cancel it.

        Use for fire-and-forget tasks that touch the ProfileStore (matching
        trigger, active-cycle clear, post-cycle processing): if a reload/unload
        swaps the store out mid-flight, an untracked task would keep writing to the
        stale store. The task auto-removes itself from the set when it finishes.
        """
        task = self.hass.async_create_task(coro)
        # Real HA always returns a Task; guard for degenerate returns (e.g. a
        # mocked hass in tests) so tracking never breaks the caller.
        if task is not None and hasattr(task, "add_done_callback"):
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        return task

    async def async_shutdown(self) -> None:
        """Shutdown."""
        self._is_shutdown = True
        # Cancel in-flight matching and cycle-end tasks so they don't race a
        # freshly-loaded ProfileStore on reload_config_entry.
        _to_await: list[Task[Any]] = []
        if self._matching_task and not self._matching_task.done():
            self._matching_task.cancel()
            _to_await.append(self._matching_task)
        if self._cycle_end_task and not self._cycle_end_task.done():
            self._cycle_end_task.cancel()
            _to_await.append(self._cycle_end_task)
        # Cancel every other tracked detached task (matching trigger, active-cycle
        # clear, post-cycle processing) for the same reason.
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
                _to_await.append(task)
        # Drain cancelled tasks so they don't race the freshly-reloaded ProfileStore.
        if _to_await:
            await asyncio.gather(*_to_await, return_exceptions=True)
        for _unsub_name in ("_remove_ha_stop_listener", "_remove_notify_queue_restore"):
            _unsub = getattr(self, _unsub_name, None)
            if _unsub is not None:
                _unsub()
                setattr(self, _unsub_name, None)
        # Keep the held notifications for the entry's next setup (audit MANAGER-16)
        # before the queues are cleared below.
        await self._async_persist_notification_queues()
        if self._remove_listener:
            self._remove_listener()
        if self._remove_report_listener:
            self._remove_report_listener()
            self._remove_report_listener = None
        if self._remove_external_trigger_listener:
            self._remove_external_trigger_listener()
        if self._remove_door_sensor_listener:
            self._remove_door_sensor_listener()
            self._remove_door_sensor_listener = None
        if self._remove_unload_confirm_listener:
            self._remove_unload_confirm_listener()
            self._remove_unload_confirm_listener = None
        if self._remove_price_listener:
            self._remove_price_listener()
            self._remove_price_listener = None
        self._cancel_door_end_dwell()
        # Drop the repeat-unload-reminder dismiss action listener (#374).
        self._remove_unload_dismiss_listener()
        if self._remove_notify_people_listener:
            self._remove_notify_people_listener()
            self._remove_notify_people_listener = None
            self._pending_notifications = []
        # Cancel any pending quiet-hours release timer so it doesn't fire after unload.
        self._cancel_quiet_hours_timer()
        self._quiet_pending_notifications = []
        # Cancel the power-off one-shot reset timer so it can't fire post-unload.
        self._cancel_power_off_timer()
        if self._remove_watchdog:
            self._remove_watchdog()
        if (
            hasattr(self, "_remove_state_expiry_timer")
            and self._remove_state_expiry_timer
        ):
            self._remove_state_expiry_timer()
        if self._remove_maintenance_scheduler:
            self._remove_maintenance_scheduler()
        if self._remove_ml_training_scheduler:
            self._remove_ml_training_scheduler()
            self._remove_ml_training_scheduler = None

        self.diag_buffer.uninstall()

        # Dismiss the timer-pause notification so it doesn't linger on mobile or
        # sidebar after HA restarts / integration unloads.
        try:
            self._clear_timer_pause_notification()
        except Exception:  # noqa: BLE001
            pass

        # Dismiss any active live/progress notification so it doesn't linger on
        # mobile devices across HA restarts or integration unloads with a stale
        # (and eventually negative) chronometer.
        try:
            self._clear_live_progress_notification()
        except Exception:  # noqa: BLE001
            self._logger.debug("Failed to clear live notification on shutdown", exc_info=True)

        # Save active state before shutdown
        if (
            self.detector.state in {STATE_RUNNING, STATE_PAUSED, STATE_STARTING, STATE_ENDING}
            or self._in_anticrease_tail()
        ):
            snapshot = self._augment_active_snapshot(self.detector.get_state_snapshot())
            await self.profile_store.async_save_active_cycle(snapshot)

        # A debounced cycle-end write still pending must land before a reload loads
        # this store again from disk (HA's own final write covers only a stop), and
        # nothing may be debounced past this point.
        # A task cancelled above may have been on its way to a save (the learning
        # pass's feedback request, a suggestion cleanup, audit MANAGER-13): its
        # change is in memory but never reached disk, so write the store once now
        # rather than lose it to the reload that reads the file next.
        try:
            self.profile_store.coalesce_saves(0)
            if _to_await:
                await self.profile_store.async_save()
            else:
                await self.profile_store.async_flush_saves()
        except Exception:  # noqa: BLE001 - never block an unload on a save
            self._logger.debug("Flushing pending store writes failed", exc_info=True)

        self._last_reading_time = None

    async def _setup_external_end_trigger(self) -> None:
        """Set up listener for external cycle end trigger binary sensor."""
        # Remove existing listener if any
        if self._remove_external_trigger_listener:
            self._remove_external_trigger_listener()
            self._remove_external_trigger_listener = None

        # Check if enabled
        enabled = self.config_entry.options.get(
            CONF_EXTERNAL_END_TRIGGER_ENABLED, False
        )
        if not enabled:
            self._logger.debug("External cycle end trigger is disabled")
            return

        # Get entity ID
        entity_id = self.config_entry.options.get(CONF_EXTERNAL_END_TRIGGER, "")
        if not entity_id:
            self._logger.debug("External cycle end trigger: no entity configured")
            return

        self._logger.info(
            "Setting up external cycle end trigger: %s", entity_id
        )

        # Subscribe to state changes
        self._remove_external_trigger_listener = async_track_state_change_event(
            self.hass, [entity_id], self._handle_external_trigger_change
        )

    async def _setup_door_sensor_listener(self) -> None:
        """Set up listener for optional door sensor binary sensor."""
        if self._remove_door_sensor_listener:
            self._remove_door_sensor_listener()
            self._remove_door_sensor_listener = None

        entity_id = self._door_sensor_entity
        if not entity_id:
            self._logger.debug("Door sensor not configured")
            return

        self._logger.info("Setting up door sensor listener: %s", entity_id)
        self._remove_door_sensor_listener = async_track_state_change_event(
            self.hass, [entity_id], self._handle_door_sensor_change
        )

    async def _setup_unload_confirm_listener(self) -> None:
        """Subscribe to the optional unload confirmation entity (#451).

        Deliberately domain-agnostic: the point of the option is that a door sensor
        is not available, so whatever the user already has - a Zigbee button
        (``event.*`` or a ``sensor.*`` action), an ``input_button`` helper, a motion
        sensor, a scene - can say "the load has been taken out".
        """
        if self._remove_unload_confirm_listener:
            self._remove_unload_confirm_listener()
            self._remove_unload_confirm_listener = None

        entity_id = self._unload_confirm_entity
        if not entity_id:
            return

        self._logger.info("Setting up unload confirmation listener: %s", entity_id)
        self._remove_unload_confirm_listener = async_track_state_change_event(
            self.hass, [entity_id], self._handle_unload_confirm_change
        )
        # Anchor the replay window PER ENTITY, in `hass.data` so it survives entry
        # reloads and resets on an HA restart.
        #
        # Not per subscribe: a settings save is a full entry reload here (the log
        # shows a fresh `Manager init`), and re-arming on it would cost the user a
        # press for two minutes after every save - the same lost-press bug this
        # window exists beside, just narrower. A reload cannot produce a replay
        # anyway, because MQTT is not reloaded with us and the entity keeps its
        # state, so no `unknown -> value` transition occurs.
        #
        # But not per PROCESS either: keyed on the entity, a newly CONFIGURED
        # confirmation entity gets its own window instead of inheriting an expired
        # one from whatever was configured before it. Without that, pointing the
        # option at a fresh `unknown` entity hours into a session left it with no
        # protection at all, and its first retained value would clear a waiting
        # Clean state. Old keys are left behind deliberately - the dict is bounded
        # by the distinct entities a user has ever chosen here.
        anchors = self.hass.data.setdefault(_UNLOAD_CONFIRM_ANCHOR_KEY, {})
        if isinstance(anchors, dict):
            anchors.setdefault(entity_id, utc_now())

    async def _setup_price_listener(self) -> None:
        """Subscribe to the dynamic energy price entity (#426).

        Only when a price *entity* is configured and dynamic pricing is on: a
        static price cannot move, so there is nothing to track. Registered for the
        entity's whole lifetime rather than per cycle - the appended samples are
        gated on the detector being active, and a subscription that only exists
        while a cycle runs would miss the price in force at the moment it starts.
        """
        if self._remove_price_listener:
            self._remove_price_listener()
            self._remove_price_listener = None

        if not self._dynamic_pricing_enabled():
            return

        entity_id = self._price_entity_id()
        if not entity_id:
            return
        self._logger.debug("Setting up dynamic price listener: %s", entity_id)
        self._remove_price_listener = async_track_state_change_event(
            self.hass, [entity_id], self._handle_price_change
        )

    def _dynamic_pricing_enabled(self) -> bool:
        """Whether cost should be integrated against a moving price (#426)."""
        options = self.config_entry.options
        if not self._price_entity_id():
            return False
        return bool(
            options.get(CONF_ENERGY_PRICE_DYNAMIC, DEFAULT_ENERGY_PRICE_DYNAMIC)
        )

    @callback
    def _handle_price_change(self, event: Event[evt.EventStateChangedData]) -> None:
        """Record a price change onto the running cycle's timeline (#426)."""
        new_state = event.data.get("new_state")
        if new_state is None:
            return
        try:
            price = float(new_state.state)
        except (ValueError, TypeError, OverflowError):
            # unknown/unavailable/non-numeric: carry the last known price forward
            # rather than charging the cycle at zero for the outage.
            return
        self._append_price_sample(price)

    def _append_price_sample(self, price: float | None) -> None:
        """Append ``price`` to the current cycle's timeline, deduplicated.

        No-op when no cycle is running - the timeline describes one cycle - and
        when the price is unchanged, so a template sensor that re-emits the same
        number every few seconds costs one comparison and nothing else.
        """
        if price is None:
            return
        if self.detector.state not in (
            STATE_STARTING,
            STATE_RUNNING,
            STATE_PAUSED,
            STATE_ENDING,
        ):
            return
        try:
            value = round(float(price), PRICE_TIMELINE_PRICE_DECIMALS)
        except (ValueError, TypeError, OverflowError):
            return
        if not math.isfinite(value):
            # Same rule as _finite_power: "nan"/"inf" parse cleanly and would ride
            # into the stored timeline. nan also defeats the dedup below, since it
            # compares unequal to itself, so every report would append a point.
            return
        if self._price_timeline and self._price_timeline[-1][1] == value:
            return
        self._price_timeline.append((utc_now().timestamp(), value))
        # Hard bound so a pathologically chatty price entity cannot grow the
        # in-memory list without limit during a long cycle; the stored timeline is
        # compacted again (by price step) at cycle end.
        if len(self._price_timeline) > PRICE_TIMELINE_MAX_POINTS * 4:
            self._price_timeline = [
                (offset, price_val)
                for offset, price_val in compact_price_timeline(
                    self._price_timeline,
                    max_points=PRICE_TIMELINE_MAX_POINTS * 2,
                    decimals=PRICE_TIMELINE_PRICE_DECIMALS,
                )
            ]

    def _start_price_timeline(self) -> None:
        """Open a fresh price timeline for a cycle that just started (#426)."""
        self._price_timeline = []
        if not self._dynamic_pricing_enabled():
            return
        self._append_price_sample(self._resolve_energy_price())

    @callback
    def _handle_door_sensor_change(self, event: Event[evt.EventStateChangedData]) -> None:
        """Handle door sensor state changes.

        Opening the door during an active cycle confirms an intentional pause (verified_pause).
        Opening the door after a cycle clears the 'Clean' state.
        Note: door closing does NOT auto-resume a cycle - the user must do this explicitly.
        """
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")

        if new_state is None:
            return

        new_val = new_state.state
        old_val = old_state.state if old_state else None

        # Ignore unavailability transitions
        if new_val in ("unavailable", "unknown") or (
            old_val in ("unavailable", "unknown")
        ):
            return

        door_open = new_val == "on"  # binary_sensor: on = open

        if door_open:
            if self._is_clean_state:
                # User opened the door after the cycle - laundry retrieved
                self.mark_unloaded("door opened")
            elif (
                self._door_opens_at_end
                and self.detector.state in (STATE_RUNNING, STATE_ENDING)
            ):
                # Auto-open dishwasher (#342): the machine pops its door at the end.
                # A sustained open means the cycle finished; a brief open (adding an
                # item) does not. Arm a dwell timer instead of the sticky user-pause
                # (which would strand the cycle in user_paused). If the door stays
                # open past the dwell we finalize; if it closes first we cancel.
                self._logger.debug(
                    "Door opened on auto-open device: arming %ss end dwell",
                    self._door_end_dwell_seconds,
                )
                self._arm_door_end_dwell()
                self._notify_update()
            elif self.detector.state in (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING):
                # Door opened during active cycle → soft pause confirmation
                self._logger.debug(
                    "Door opened during active cycle: setting verified_pause=True"
                )
                self.detector.set_verified_pause(True)
                if not self._is_user_paused:
                    self._is_user_paused = True
                    self._user_pause_start = utc_now()
                self._notify_update()
        else:
            # Door closed: cancel a pending auto-open finalize (it was a brief open,
            # not the end-of-cycle door pop). No auto-resume otherwise (#342).
            if self._remove_door_end_dwell is not None:
                self._logger.debug("Door closed before end dwell: cancelling finalize")
                self._cancel_door_end_dwell()
                self._notify_update()

    @callback
    def _handle_unload_confirm_change(self, event: Event[evt.EventStateChangedData]) -> None:
        """Treat an activation of the unload confirmation entity as "unloaded" (#451).

        Any change to a real state counts, because the entity is whatever the user
        had to hand: an ``event.*`` button writes a fresh timestamp per press, an
        ``input_button`` the same, a ``sensor.*`` action sensor writes "single" and
        resets.

        Excluded: a transition *to* unknown/unavailable/``off``/empty (the release
        half of a contact or motion sensor, or a device dropping off), a transition
        *out of* ``unavailable`` (a flat battery coming back is not a press), and an
        entity that has only just appeared (``old_state is None``, which is what a
        restored last-press timestamp looks like on HA start).

        **``unknown`` is the interesting one, and it is time-scoped rather than
        excluded outright (register item 367).** A fresh ``event.*`` / ``button.*``
        / ``input_button.*`` sits at ``unknown`` until it is first pressed, so a
        blanket exclusion swallowed the FIRST EVER press - and that reads as "the
        feature does not work", which is the #445 failure one layer down. Accepting
        it outright is not safe either: a z2m action sensor publishes its action as
        a RETAINED MQTT message, replayed by the broker on reconnect, arriving as
        precisely this transition. The restart case is already covered above, so
        the only gap left is the moment just after we subscribe - hence
        ``UNLOAD_CONFIRM_REPLAY_GRACE_S`` from the subscription, after which an
        ``unknown -> value`` change is taken at face value.
        """
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")

        # No old state = the entity was just added or HA has just started. A button
        # entity's restored last-press timestamp must not count as a press.
        if new_state is None or old_state is None:
            return

        new_val = new_state.state
        old_val = old_state.state
        if new_val == old_val:
            return
        if new_val in ("unavailable", "unknown"):
            # Going away RE-ARMS the window. The startup anchor alone only covers
            # the reconnect that follows an HA restart; a broker restart hours
            # later replays retained values just the same, and the entity passes
            # through `unavailable`/`unknown` on its way out. Re-anchoring here
            # means the value that comes back is judged as the replay it may well
            # be. Costs nothing on the press path: a value arriving straight after
            # `unavailable` is excluded outright either way.
            self._rearm_unload_confirm_window()
            return
        if new_val in ("off", ""):
            return
        if old_val == "unavailable":
            return
        if old_val == "unknown" and self._in_unload_confirm_replay_window():
            self._logger.debug(
                "Ignoring %s -> %s within the unload-confirm replay window; a "
                "retained value can arrive this soon after subscribing",
                old_val,
                new_val,
            )
            return

        self.mark_unloaded(f"{self._unload_confirm_entity} -> {new_val}")

    def _rearm_unload_confirm_window(self) -> None:
        """Restart the replay window for the configured entity.

        Called when it drops to ``unavailable``/``unknown``, because whatever it
        reports on the way back may be a retained value rather than a press.
        """
        entity_id = self._unload_confirm_entity
        if not entity_id:
            return
        anchors = self.hass.data.setdefault(_UNLOAD_CONFIRM_ANCHOR_KEY, {})
        if isinstance(anchors, dict):
            anchors[entity_id] = utc_now()

    def _in_unload_confirm_replay_window(self) -> bool:
        """Whether this entity came back too recently to trust ``unknown -> value``.

        Measured from the entity's own anchor: set when this process first
        subscribed to it, and restarted every time it drops out (see
        `_rearm_unload_confirm_window`). Survives entry reloads, resets on an HA
        restart, and does not carry over between different configured entities.

        Fails CLOSED (True) if the anchor is missing or unusable: "we do not know
        when this entity came back" carries the same risk as "it just did".
        """
        entity_id = self._unload_confirm_entity
        anchors = self.hass.data.get(_UNLOAD_CONFIRM_ANCHOR_KEY)
        anchor = anchors.get(entity_id) if isinstance(anchors, dict) else None
        if not isinstance(anchor, datetime):
            return True
        try:
            elapsed = (utc_now() - anchor).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return True
        return elapsed < UNLOAD_CONFIRM_REPLAY_GRACE_S

    def mark_unloaded(self, source: str = "manual") -> bool:
        """Clear the Clean state: the load has been taken out (#153, #451).

        Single owner of the "laundry retrieved" transition, shared by the door-open
        handler, the unload confirmation entity, the Mark Unloaded button and the
        ``mark_unloaded`` service, so the four can never drift on what clearing it
        entails. Idempotent: a confirmation arriving when nothing is waiting is a
        no-op, which is what an automation that fires on every button press needs.

        Returns True when a Clean state was actually cleared.
        """
        if not self._is_clean_state:
            return False

        self._logger.debug("Unload confirmed (%s): clearing Clean state", source)
        self._is_clean_state = False
        self._clean_state_start = None
        self._notified_clean_laundry = False
        self._reset_unload_nag_tracking()
        # Dismiss a delivered clean reminder (and purge any queued ones) so it does
        # not linger on the phone after the laundry is taken.
        self._clear_clean_notification()
        self._notify_update()
        return True

    def _unload_confirmable_without_door(self) -> bool:
        """Whether unload can be confirmed with no door sensor configured (#451).

        Also the opt-in for entering the Clean state at all on such a device: with
        neither option set there would be no way to clear it, so the reminder would
        nag until the progress-reset window expired.
        """
        return bool(self._unload_confirm_entity) or self._unload_track_without_door

    def _maybe_arm_door_end_dwell_if_open(self) -> None:
        """Arm the end-dwell timer when in RUNNING or ENDING with the door already open.

        Called from both ``_on_state_change`` (live transition) and the
        snapshot-restoration path (where ``_on_state_change`` is not invoked).
        Matches the state set accepted by ``_handle_door_sensor_change`` so that
        restoring a RUNNING snapshot with the door already open re-arms the dwell.
        """
        if not (
            self.detector.state in (STATE_RUNNING, STATE_ENDING)
            and self._door_opens_at_end
            and self._door_sensor_entity
            and self._remove_door_end_dwell is None
        ):
            return
        door_state = self.hass.states.get(self._door_sensor_entity)
        if door_state and door_state.state == "on":
            self._logger.debug(
                "Door already open on ENDING: arming %ss end dwell",
                self._door_end_dwell_seconds,
            )
            self._arm_door_end_dwell()

    def _arm_door_end_dwell(self) -> None:
        """(Re)arm the auto-open door-end dwell timer (#342)."""
        self._cancel_door_end_dwell()
        self._remove_door_end_dwell = async_call_later(
            self.hass,
            float(max(1, self._door_end_dwell_seconds)),
            self._door_end_dwell_fired,
        )

    def _cancel_door_end_dwell(self) -> None:
        """Cancel a pending auto-open door-end dwell timer, if any (#342)."""
        if self._remove_door_end_dwell is not None:
            self._remove_door_end_dwell()
            self._remove_door_end_dwell = None

    @callback
    def _door_end_dwell_fired(self, _now: Any) -> None:
        """The door stayed open past the dwell on an auto-open device: the cycle has
        finished, so finalize it as completed (same path as the External End
        Trigger). The dwell is normally cancelled on door-close, but re-validate the
        live conditions here defensively — a close event could have been missed, or a
        user pause could have landed mid-dwell — before finalizing (#342)."""
        self._remove_door_end_dwell = None
        door_state = (
            self.hass.states.get(self._door_sensor_entity)
            if self._door_sensor_entity
            else None
        )
        if (
            self.detector.state in (STATE_RUNNING, STATE_ENDING)
            and self._door_opens_at_end
            and not self._is_user_paused
            and door_state is not None
            and door_state.state == "on"
        ):
            # The end-of-cycle door pop follows the power drop, so an appliance still
            # drawing above the stop threshold means the door was opened mid-cycle
            # (loading a dish and walking off) rather than at the end - finalizing
            # there would record a running cycle as completed.
            #
            # Re-arm rather than abandon: the dwell is one-shot, so dropping it here
            # would permanently lose the door-based finalize for a machine that just
            # happens to be mid-pulse when the timer lands (fan/zeolite drying can
            # draw with the door already popped), leaving the cycle to the power
            # timeout that #342 exists to short-circuit. Re-arming is asymmetric: it
            # can only ever delay the finalize, never skip it.
            stop_thr = 0.0
            try:
                stop_thr = float(getattr(self.detector.config, "stop_threshold_w", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                stop_thr = 0.0
            if stop_thr > 0.0 and self._current_power >= stop_thr:
                self._logger.debug(
                    "Door-end dwell fired but power %.1fW is still at/above the stop "
                    "threshold %.1fW: treating as a mid-cycle door open, re-arming",
                    self._current_power,
                    stop_thr,
                )
                self._arm_door_end_dwell()
                return
            self._logger.info(
                "Door held open %ss on auto-open device: finalizing cycle",
                self._door_end_dwell_seconds,
            )
            self.detector.user_stop()
            self._notify_update()
        else:
            self._logger.debug(
                "Door-end dwell fired but conditions no longer hold "
                "(state=%s, door=%s, user_paused=%s): not finalizing",
                self.detector.state,
                getattr(door_state, "state", None),
                self._is_user_paused,
            )

    async def _setup_notify_people_listener(self) -> None:
        """Set up listener for person presence changes used by notification gating."""
        if self._remove_notify_people_listener:
            self._remove_notify_people_listener()
            self._remove_notify_people_listener = None

        if self._notify_only_when_home and self._notify_people:
            self._remove_notify_people_listener = async_track_state_change_event(
                self.hass, self._notify_people, self._handle_notify_person_change
            )
            # If someone is already home when (re-)attaching, flush any queued
            # notifications immediately so they aren't stranded.
            if self._pending_notifications and self._is_any_notify_person_home():
                person_entity_id: str | None = None
                person_name: str | None = None
                for eid in self._notify_people:
                    state = self.hass.states.get(eid)
                    if state and state.state == STATE_HOME:
                        person_entity_id = eid
                        person_name = state.name or state.attributes.get(
                            "friendly_name", eid
                        )
                        break
                self._flush_pending_notifications(person_entity_id, person_name)
        else:
            self._pending_notifications = []

    @callback
    def _handle_external_trigger_change(self, event: Event[evt.EventStateChangedData]) -> None:
        """Handle external trigger sensor state change."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")

        if new_state is None:
            return

        inverted = self.config_entry.options.get(
            CONF_EXTERNAL_END_TRIGGER_INVERTED, False
        )

        new_value = new_state.state
        old_value = old_state.state if old_state else None

        # Ignore unavailability/unknown transitions (reconnects, disconnects)
        if old_value is None or old_value in ("unavailable", "unknown") or new_value in (
            "unavailable",
            "unknown",
        ):
            return

        # Determine if triggered based on inversion setting
        triggered = False
        if not inverted:
            # Normal: Trigger on transition to "on"
            if new_value == "on" and old_value != "on":
                triggered = True
        else:
            # Inverted: Trigger on transition to "off"
            if new_value == "off" and old_value != "off":
                triggered = True

        if triggered:
            self._logger.info(
                "External cycle end trigger activated by %s (inverted=%s)",
                event.data.get("entity_id"),
                inverted
            )
            # End cycle with "completed" status (not interrupted)
            if self.detector.state in (STATE_ANTI_WRINKLE, STATE_DELAY_WAIT):
                self.detector.reset(STATE_OFF)
                self._logger.info("%s exited via external trigger", self.detector.state)
            elif self.detector.state != STATE_OFF:
                self.detector.user_stop()
                self._logger.info("Cycle completed via external trigger")

    def _external_end_trigger_available(self) -> bool:
        """True when an authoritative external end trigger is wired and reporting.

        Used by the unmatched zombie guard (#404): if the user has configured an
        external end-of-cycle binary sensor and it currently has a usable state, an
        authoritative end signal already exists, so the time-based failsafe would only
        do harm (it would truncate a long programme that the trigger will end cleanly).
        Returns False if disabled, unconfigured, or the entity is missing/unavailable.
        """
        if not self.config_entry.options.get(CONF_EXTERNAL_END_TRIGGER_ENABLED, False):
            return False
        entity_id = self.config_entry.options.get(CONF_EXTERNAL_END_TRIGGER, "")
        if not entity_id:
            return False
        state = self.hass.states.get(entity_id)
        return state is not None and state.state not in ("unavailable", "unknown")

    async def _setup_maintenance_scheduler(self) -> None:
        """Set up daily maintenance task at midnight."""
        auto_maintenance = self.config_entry.options.get(
            CONF_AUTO_MAINTENANCE,
            self.config_entry.data.get(CONF_AUTO_MAINTENANCE, DEFAULT_AUTO_MAINTENANCE),
        )

        # Cancel existing scheduler if any
        if self._remove_maintenance_scheduler:
            self._remove_maintenance_scheduler()
            self._remove_maintenance_scheduler = None

        if not auto_maintenance:
            self._logger.debug("Auto-maintenance disabled")
            return

        async def run_maintenance(_now: datetime | None = None) -> None:
            """Run maintenance task."""
            self._logger.info("Running scheduled maintenance")
            try:
                stats = await self.profile_store.async_run_maintenance()
                self._logger.info("Maintenance completed: %s", stats)
                # Refresh persisted cycle health as part of nightly maintenance.
                await self.async_recompute_cycle_health()
            except Exception as err:  # pylint: disable=broad-exception-caught
                self._logger.error("Maintenance failed: %s", err, exc_info=True)

        # Fire daily at local midnight with a single, cleanly-cancellable handle.
        # async_track_time_change auto-repeats every day, so there is no manual
        # rescheduling that could leak handles or double-register the callback.
        self._remove_maintenance_scheduler = evt.async_track_time_change(
            self.hass, run_maintenance, hour=0, minute=0, second=0
        )
        self._logger.info("Scheduled daily maintenance at local midnight")

    def _setup_ml_training_scheduler(self) -> None:
        """Schedule the daily on-device ML retraining (Stage 4, gated).

        Uses ``async_track_time_change`` which fires every day at the configured
        hour with a single, cleanly-cancellable handle (no manual rescheduling).
        No-op unless the ``ENABLE_ML_TRAINING`` build flag and the per-device
        opt-in are both set.
        """
        from .const import (
            ENABLE_ML_TRAINING,
            CONF_ML_TRAINING_ENABLED,
            CONF_ML_TRAINING_HOUR,
            DEFAULT_ML_TRAINING_ENABLED,
            DEFAULT_ML_TRAINING_HOUR,
        )

        if self._remove_ml_training_scheduler:
            self._remove_ml_training_scheduler()
            self._remove_ml_training_scheduler = None

        if not ENABLE_ML_TRAINING:
            return
        opts = {**self.config_entry.data, **self.config_entry.options}
        if not opts.get(CONF_ML_TRAINING_ENABLED, DEFAULT_ML_TRAINING_ENABLED):
            self._logger.debug("On-device ML training disabled")
            return

        try:
            hour = int(opts.get(CONF_ML_TRAINING_HOUR, DEFAULT_ML_TRAINING_HOUR))
        except (TypeError, ValueError, OverflowError):
            hour = DEFAULT_ML_TRAINING_HOUR
        hour = max(0, min(23, hour))

        async def _scheduled(_now: datetime) -> None:
            await self.async_run_ml_training(force=False)

        self._remove_ml_training_scheduler = evt.async_track_time_change(
            self.hass, _scheduled, hour=hour, minute=0, second=0
        )
        self._logger.info("Scheduled on-device ML training daily at %02d:00", hour)

    async def async_run_ml_training(self, force: bool = False) -> dict[str, Any]:
        """Retrain the ML models from this device's own cycles (gated + guarded).

        Returns a summary dict. ``force`` bypasses the min-cycle / interval /
        idle guards (used by the manual service). Never raises to the caller.
        """
        from .const import (
            ENABLE_ML_TRAINING,
            CONF_ML_TRAINING_MIN_CYCLES,
            CONF_ML_TRAINING_INTERVAL_DAYS,
            DEFAULT_ML_TRAINING_MIN_CYCLES,
            DEFAULT_ML_TRAINING_INTERVAL_DAYS,
            EVENT_ML_TRAINING_COMPLETE,
        )

        if not ENABLE_ML_TRAINING:
            return {"ok": False, "reason": "ml_training_disabled"}

        opts = {**self.config_entry.data, **self.config_entry.options}
        # Snapshot on the event loop before any executor offload (training):
        # get_past_cycles() returns the live mutable list, so a
        # concurrent cycle add / retention trim could otherwise change the input
        # mid-run.
        cycles = list(self.profile_store.get_past_cycles())

        if not force:
            # Don't train mid-cycle; wait for a quiet moment.
            if self.detector and self.detector.state in {
                STATE_RUNNING, STATE_PAUSED, STATE_STARTING, STATE_ENDING
            }:
                self._logger.debug("Skipping scheduled ML training: device active")
                return {"ok": False, "reason": "device_active"}
            min_cycles = int(opts.get(CONF_ML_TRAINING_MIN_CYCLES, DEFAULT_ML_TRAINING_MIN_CYCLES))
            if len(cycles) < min_cycles:
                self._logger.debug(
                    "Skipping scheduled ML training: need %d cycles, have %d",
                    min_cycles,
                    len(cycles),
                )
                return {"ok": False, "reason": f"need {min_cycles} cycles, have {len(cycles)}"}
            # Respect the minimum retrain interval.
            interval_days = int(
                opts.get(CONF_ML_TRAINING_INTERVAL_DAYS, DEFAULT_ML_TRAINING_INTERVAL_DAYS)
            )
            last = self._last_ml_training_at()
            if last is not None:
                age_days = (utc_now() - last).total_seconds() / 86400.0
                if age_days < interval_days:
                    self._logger.debug(
                        "Skipping scheduled ML training: retrained %.1fd ago (<%dd)",
                        age_days,
                        interval_days,
                    )
                    return {"ok": False, "reason": f"retrained {age_days:.1f}d ago (<{interval_days}d)"}

        if self._ml_training_running:
            return {"ok": False, "reason": "already_running"}
        self._ml_training_running = True
        self.notify_update()
        try:
            from .ml.training_task import async_run_training

            summary = await async_run_training(self.hass, self)
        except Exception as err:  # noqa: BLE001 - training must never break the integration
            self._logger.error("On-device ML training failed: %s", err, exc_info=True)
            return {"ok": False, "reason": "exception", "error": str(err)}
        finally:
            self._ml_training_running = False
            self.notify_update()

        promoted = list(summary.get("promoted", {}).keys())
        if promoted:
            self._ml_training_failures = 0
            # Consumers read the trained specs live from the store via
            # ml.engine.resolve_regressor, so no refresh is needed.
            self._logger.info("On-device ML training promoted models: %s", promoted)
        else:
            self._ml_training_failures += 1
            self._logger.info(
                "On-device ML training produced no promotable models (attempt %d)",
                self._ml_training_failures,
            )

        # Record that training *ran* now, regardless of whether anything was
        # promoted, so "Last trained" advances on every run (a run that doesn't
        # beat the baseline previously left the timestamp stuck at the last
        # promotion). Never let a persistence hiccup break the run.
        try:
            _run_iso = utc_now().isoformat()
            await self.profile_store.set_ml_last_training_run(_run_iso)
            # Track each capability's held-out score over time (drift/fit trend).
            await self.profile_store.append_ml_training_history(
                _run_iso, summary.get("results", [])
            )
        except Exception as err:  # noqa: BLE001
            self._logger.debug("Failed to persist last-training-run timestamp: %s", err)

        self.hass.bus.async_fire(
            EVENT_ML_TRAINING_COMPLETE,
            {
                "entry_id": self.entry_id,
                "device_name": self.config_entry.title,
                "promoted": promoted,
                "results": summary.get("results", []),
            },
        )
        # (No health recompute: the per-cycle health reads the quality / end
        # models, and since 0.5.8 training promotes only total_energy.)

        return {
            "ok": True,
            "promoted": promoted,
            "results": summary.get("results", []),
        }

    async def async_recompute_cycle_health(self) -> int:
        """Recompute + persist per-cycle ML health against the current model.

        Health is cached on each cycle and only recomputed at defined triggers
        (this method): on-device retraining, scheduled auto-maintenance and the
        Diagnostics "Process History" action. Panel loads reuse the cache. Runs
        the CPU work in an executor and never raises to the caller.
        """
        import functools  # pylint: disable=import-outside-toplevel

        try:
            from .ws_api import _compute_ml_comparison  # pylint: disable=import-outside-toplevel
        except Exception:  # pylint: disable=broad-exception-caught
            return 0
        try:
            result = await self.hass.async_add_executor_job(
                functools.partial(
                    _compute_ml_comparison, self.profile_store, force_recompute=True
                )
            )
        except Exception as err:  # noqa: BLE001
            self._logger.debug("Cycle-health recompute failed: %s", err)
            return 0
        health_updates = result.get("_health_updates", {})
        if health_updates:
            for cycle in self.profile_store.get_past_cycles():
                cid = cycle.get("id")
                if cid in health_updates:
                    cycle["ml_health"] = health_updates[cid]
        if result.get("_health_dirty") or health_updates:
            await self.profile_store.async_save()
        return int(result.get("evaluated_count", 0))

    async def async_recompute_cycle_costs(self) -> int:
        """Recost stored cycles from the recorder's price history (#426).

        Existing cycles were costed at the single price in force when they ended -
        including everything imported from raw history (#344), which was costed at
        whatever the tariff happened to be at import time. Where the recorder still
        holds the price entity's history, those cycles can be recosted properly
        after the fact. Returns the number of cycles rewritten.

        Deliberately conservative, because it overwrites a figure the user has
        already seen:

        * only ``past_cycles`` and ``backfill_cycles`` - reference cycles are other
          people's recordings and were never the user's energy to pay for;
        * only cycles the recorder can actually answer for (a price row at or before
          the cycle's start); anything older than the recorder's retention keeps the
          cost it has;
        * cycles already costed dynamically are left alone, so the pass is
          idempotent and never re-derives a live-tracked timeline from a coarser
          recorder view.
        """
        if not self._dynamic_pricing_enabled():
            return 0
        candidates: list[tuple[dict[str, Any], datetime, datetime]] = []
        for cycle in list(self.profile_store.get_past_cycles()) + list(
            self.profile_store.get_backfill_cycles()
        ):
            if cycle.get("energy_price_mode") == "dynamic" and cycle.get("price_timeline"):
                continue
            start_dt = dt_util.parse_datetime(str(cycle.get("start_time") or ""))
            end_dt = dt_util.parse_datetime(str(cycle.get("end_time") or ""))
            # An ISO string without an offset parses naive, which every record this
            # device writes is not, but an import or a hand-edited file can be. Read
            # as UTC, the same way the odometer scan does: one such cycle otherwise
            # raises on the aware `horizon` comparison below, and the WS caller only
            # debug-logs that, so the whole pass silently recosts nothing.
            if start_dt is not None and start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=dt_util.UTC)
            if end_dt is not None and end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=dt_util.UTC)
            if start_dt is None or end_dt is None or end_dt <= start_dt:
                continue
            candidates.append((cycle, start_dt, end_dt))
        if not candidates:
            return 0

        # Bound the query to what the recorder can still answer. Reading further
        # back returns nothing but makes the executor walk the whole retained
        # window of a frequently-updating price entity for it.
        keep_days = 10.0
        try:
            from homeassistant.components.recorder import get_instance  # noqa: PLC0415

            keep_days = float(getattr(get_instance(self.hass), "keep_days", 10) or 10)
        except Exception:  # noqa: BLE001 - recorder optional; the default stands
            pass
        horizon = utc_now() - timedelta(days=min(max(keep_days, 1.0), 365.0))
        candidates = [c for c in candidates if c[2] >= horizon]
        if not candidates:
            return 0

        window_start = min(start for _, start, _ in candidates)
        window_end = max(end for _, _, end in candidates)
        rows = await self._async_price_history(max(window_start, horizon), window_end)
        if not rows:
            return 0

        updated = 0
        for cycle, start_dt, end_dt in candidates:
            start_ts = start_dt.timestamp()
            end_ts = end_dt.timestamp()
            # Require an anchor at or before the cycle: without one the first known
            # price would be back-applied to energy bought before it existed.
            anchor: tuple[float, float] | None = None
            window: list[tuple[float, float]] = []
            for ts, price in rows:
                if ts <= start_ts:
                    # Keep only the newest pre-start row. Older ones all collapse
                    # onto offset 0 and, once there are more of them than
                    # PRICE_TIMELINE_MAX_POINTS, compaction can spend the whole
                    # budget on prices this cycle never ran at and evict the real
                    # anchor or an in-cycle transition.
                    anchor = (ts, price)
                elif ts <= end_ts:
                    window.append((ts, price))
            if anchor is None:
                continue
            window.insert(0, anchor)
            points = compact_price_timeline(
                [(max(0.0, ts - start_ts), price) for ts, price in window],
                max_points=PRICE_TIMELINE_MAX_POINTS,
                decimals=PRICE_TIMELINE_PRICE_DECIMALS,
            )
            if not points:
                continue
            result = self._cost_from_timeline(cycle, points)
            if result is None:
                continue
            cost, effective_price = result
            cycle["cost"] = round(cost, 4)
            cycle["energy_price"] = round(effective_price, 6)
            cycle["energy_price_mode"] = "dynamic"
            cycle["price_timeline"] = [[round(offset, 1), price] for offset, price in points]
            updated += 1

        if updated:
            await self.profile_store.async_save()
            self._logger.info("Recosted %d cycle(s) from recorder price history", updated)
        return updated

    def _last_ml_training_at(self) -> datetime | None:
        """When on-device training last *ran* (not just last promoted a model).

        Prefers the persisted last-run timestamp so a manual/scheduled run that
        produced no promotable model still advances "Last trained" and the retrain
        interval. Falls back to the newest promoted model's ``trained_at`` for
        installs from before run-time tracking existed.
        """
        run_iso = self.profile_store.get_ml_last_training_run()
        if isinstance(run_iso, str):
            parsed = dt_util.parse_datetime(run_iso)
            if parsed is not None:
                return parsed
        latest: datetime | None = None
        for record in (self.profile_store.get_ml_model_versions() or {}).values():
            ts = record.get("trained_at") if isinstance(record, dict) else None
            if not isinstance(ts, str):
                continue
            try:
                parsed = dt_util.parse_datetime(ts)
            except (ValueError, TypeError, OverflowError):
                parsed = None
            if parsed is not None and (latest is None or parsed > latest):
                latest = parsed
        return latest

    @callback
    def _subscribe_power_sensor(self) -> None:
        """(Re)subscribe to the power sensor's state changes AND unchanged reports.

        HA fires EVENT_STATE_CHANGED only when the value (or attributes) change.
        A plug that periodically re-reports the same value (Tasmota TelePeriod,
        Zigbee max reporting interval) fires EVENT_STATE_REPORTED instead, which a
        state_changed-only subscription never receives. Missing those reports means
        a flat sub-threshold tail never advances the detector's end-of-cycle timer
        (#363) and a finished cycle lags by the plug's reporting interval (#329).

        Both events route into the same handler. Per HA, a single write fires either
        state_changed (value/attrs differ) or state_reported (unchanged), never both,
        so there is no double-counting. Report events carry ``new_state`` without an
        ``old_state``; ``_async_power_changed`` already tolerates a missing old_state.
        """
        if self._remove_listener:
            self._remove_listener()
        if self._remove_report_listener:
            self._remove_report_listener()
        self._remove_listener = async_track_state_change_event(
            self.hass, [self.power_sensor_entity_id], self._async_power_changed
        )
        self._remove_report_listener = async_track_state_report_event(
            self.hass, [self.power_sensor_entity_id], self._async_power_changed
        )

    @callback
    def _async_power_changed(self, event: Any) -> None:
        """Handle power sensor state change."""
        event_data = cast(dict[str, Any], getattr(event, "data", {}))
        new_state = cast(State | None, event_data.get("new_state"))
        # A state with no usable value is a dead sensor, not a silent one: record
        # it, so the detector stops crediting quiet until the next real reading
        # (register item 266, audit DETECT-13). Returning without a trace left
        # the watchdog's keepalives to run out the end gates during a dropout.
        if new_state is None or new_state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            self.detector.mark_sensor_unavailable(utc_now())
            return

        power = _finite_power(new_state.state)
        if power is None:
            self.detector.mark_sensor_unavailable(utc_now())
            return

        # Capture every raw sensor reading before any throttling or processing.
        # Use the sensor's own report timestamp so the trace reflects when the plug
        # actually reported the value, not when we received it. last_reported
        # advances on every write (incl. unchanged re-reports #363), whereas
        # last_updated only advances on a value change, so an unchanged re-report
        # would otherwise be stamped with a stale time.
        report_ts = getattr(new_state, "last_reported", None) or new_state.last_updated
        self.diag_buffer.record_power(power, report_ts)

        # RECORD MODE INTERCEPTION
        if self.recorder.is_recording:
            self.recorder.process_reading(power)
            self._current_power = power
            self._last_reading_time = utc_now()
            self._notify_update()
            return

        now = utc_now()

        # Throttle updates to avoid CPU overload on noisy sensors.
        # Low-power readings bypass throttling when:
        #   (a) a cycle is active (RUNNING/ENDING/PAUSED) — critical end-of-cycle signal, or
        #   (b) this is a genuine power DROP from above min_power — captures power-off events
        #       that occur before the detector has processed the previous above-threshold reading.
        # Without the guard, an idle device at 0W fires an update on every sensor poll (typically
        # every 1–5 s), flooding the detector with zero-value no-ops.
        min_p = float(self.detector.config.min_power)
        # For the "genuine drop" bypass, compare against the previous RAW sensor
        # value (old_state), not _current_power: the latter is only updated after a
        # reading passes the throttle, so a suppressed high reading would leave it
        # low and the following low reading would be throttled too, missing a short
        # high->low transition. old_state reflects the plug's actual prior value.
        prev_raw_power = self._current_power
        old_state = cast(State | None, event_data.get("old_state"))
        if old_state is not None and old_state.state not in (
            STATE_UNKNOWN, STATE_UNAVAILABLE
        ):
            _prev = _finite_power(old_state.state)
            if _prev is not None:
                prev_raw_power = _prev
        is_low_power = power < min_p and (
            self.detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
            or prev_raw_power >= min_p  # genuine drop from active power
        )

        if (
            not is_low_power
            and self._last_reading_time
            and (now - self._last_reading_time).total_seconds() < self._sampling_interval
        ):
            return

        # Track observed power readings for learning - only while a cycle is
        # active (#394). An appliance is idle ~98% of the time; running the 5-min
        # auto-tune pass and training the sample-interval cadence model on the
        # standby heartbeat is constant background work (a store rewrite every few
        # minutes) that buys nothing AND skews every operational suggestion, since
        # the idle publish-on-change heartbeat is not the in-cycle sampling
        # cadence those suggestions are sized from. The detector below still
        # receives EVERY reading, so the next cycle's start is never missed - only
        # the learning call is gated.
        if self.detector.state in (
            STATE_STARTING,
            STATE_RUNNING,
            STATE_PAUSED,
            STATE_ENDING,
        ):
            self.learning_manager.process_power_reading(
                power, now, self._last_reading_time
            )
        self._last_reading_time = now
        if self._restored_sensor_clock is not None:
            # The entity appeared only after setup, so this is its startup write
            # (register item 266): same rule as the setup read.
            self._seed_real_reading_clock(
                power, report_ts if isinstance(report_ts, datetime) else now
            )
        else:
            self._last_real_reading_time = now # Track real update
        self._current_power = power
        # One entity refresh per reading, the one at the end (register item 456):
        # _update_estimates and a match that completes inside process_reading (a
        # trace still too short to score returns without awaiting) each sent their
        # own first, all of ~20 entities rewritten 2-4 times for one reading.
        self._in_power_event = True
        try:
            self.detector.process_reading(power, now)
            self._check_stall_event()  # #452: only a real reading can start a stall

            if self._cycle_start_time is None and self.detector.current_cycle_start is not None:
                self._cycle_start_time = self.detector.current_cycle_start

            # If running (or paused/ending), try to match profile and update estimates
            if self.detector.state in (
                STATE_RUNNING,
                STATE_PAUSED,
                STATE_ENDING,
                STATE_STARTING,
            ):
                self._update_estimates()
                # Periodically save state every 60s to avoid flash wear
                # We need a tracker.
                self._check_state_save(now)
        finally:
            self._in_power_event = False

        self._notify_update()

    def _check_stall_event(self) -> None:
        """Fire EVENT_CYCLE_STALLED once per stall (discussion #452).

        Display and automation only, never a notification. The payload is the
        detector's small ``stall_info`` (no trace), far under the 32 KB limit.
        """
        if getattr(self.detector, "stalled", False) is not True:
            return
        info = self.detector.stall_info()
        if not isinstance(info, dict):
            return
        key = info.get("stalled_since")
        if key == getattr(self, "_stall_event_key", None):
            return
        self._stall_event_key = key
        if not self._notify_fire_events:
            return
        self.hass.bus.async_fire(
            EVENT_CYCLE_STALLED,
            {
                "entry_id": self.entry_id,
                "device_name": self.config_entry.title,
                "device_type": self.device_type,
                "program": self._current_program,
                **info,
            },
        )

    async def _async_refresh_standby_level(self) -> None:
        """Re-learn the standby level the idle display reads (#452). Never raises."""
        try:
            cfg = self.detector.config
            cycles = [
                dict(c) for c in list(self.profile_store.get_past_cycles() or [])[
                    -STANDBY_LEVEL_RECENT_CYCLES:
                ]
                if isinstance(c, dict)
            ]
            level = await self.hass.async_add_executor_job(
                learned_standby_level_w,
                cycles,
                float(cfg.stop_threshold_w),
                float(cfg.start_threshold_w),
            )
            self.detector.set_standby_level(level)
            self._logger.debug("Idle display standby level: %s W", level)
        except Exception:  # noqa: BLE001 - a display statistic must never break setup
            self._logger.debug("Could not learn the standby level", exc_info=True)

    def _check_state_save(self, now: datetime) -> None:
        """Periodically save active state."""
        last_save = getattr(self, "_last_state_save", None)
        if not last_save or (now - last_save).total_seconds() > 60:
            # Fire and forget save task
            # Inject manual program flag into snapshot before saving
            snapshot = self._augment_active_snapshot(self.detector.get_state_snapshot())

            # Tracked (audit MANAGER-13): an unload cancels it and writes its own.
            self._spawn_tracked(self.profile_store.async_save_active_cycle(snapshot))
            self._last_state_save = now

    def _save_snapshot_while_silent(self, now: datetime) -> None:
        """Keep the active snapshot current through a silence (register item 266).

        Saves were driven by real readings only, so a cycle waiting out a silent
        tail kept the snapshot of its last report: a crash then restored that
        moment's quiet tally, recorded the watched silence as a restart gap, and
        aged the snapshot by time Home Assistant had in fact been watching - past
        the restore window on a long drying tail. Same 60 s throttle. Never
        raises: a lost save only costs a restore, the watchdog tick must finish.
        """
        if self.detector.state not in (
            STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING
        ):
            return
        try:
            self._check_state_save(now)
        except Exception:  # noqa: BLE001
            self._logger.debug("Could not save the active cycle in a silence", exc_info=True)

    async def _run_final_match_from_cycle_data(
        self, cycle_data: dict[str, Any]
    ) -> MatchResult | None:
        """Match the COMPLETE cycle once, before it is saved; None if too short.

        No side effects: the caller decides what to adopt, because a new cycle can
        start during the await and the live fields would then belong to it (B1).
        """
        # Cycle data from detector stores power_data as [[offset_seconds, power], ...],
        # where offsets are relative to cycle start. Shared with the Playground's
        # would_label (match_rules.final_match_input).
        final_input = match_rules.final_match_input(cycle_data)
        if final_input is None:
            self._logger.debug("Insufficient power data for final match (< 10 readings)")
            return None
        power_data, duration = final_input

        self._logger.debug(
            "Running final match from cycle data: %s samples, %.0fs duration",
            len(power_data),
            duration,
        )
        return await self.profile_store.async_match_profile(power_data, duration)

    def _start_watchdog(self) -> None:
        """Start the watchdog timer when a cycle begins."""
        if self._remove_watchdog:
            return  # Already running

        interval = self._watchdog_interval
        self._logger.debug(
            "Starting watchdog timer (configured=%ss)",
            self._watchdog_interval,
        )
        self._remove_watchdog = async_track_time_interval(
            self.hass, self._watchdog_check_stuck_cycle, timedelta(seconds=interval)
        )

    def _stop_watchdog(self) -> None:
        """Stop the watchdog timer when cycle ends."""
        if self._remove_watchdog:
            self._logger.debug("Stopping watchdog timer")
            self._remove_watchdog()
            self._remove_watchdog = None

    def _start_state_expiry_timer(self) -> None:
        """Start timer to reset state to OFF and progress to 0% after idle period."""
        if not hasattr(self, "_remove_state_expiry_timer"):
            self._remove_state_expiry_timer = None

        if self._remove_state_expiry_timer:
            return  # Already running

        self._logger.debug(
            "Starting state expiry timer (will reset after %ss)",
            self._progress_reset_delay,
        )
        self._remove_state_expiry_timer = async_track_time_interval(
            self.hass,
            self._handle_state_expiry,
            timedelta(seconds=60),  # Check every minute
        )

    def _stop_state_expiry_timer(self) -> None:
        """Stop the state expiry timer."""
        if (
            hasattr(self, "_remove_state_expiry_timer")
            and self._remove_state_expiry_timer
        ):
            self._logger.debug("Stopping state expiry timer")
            self._remove_state_expiry_timer()
            self._remove_state_expiry_timer = None

    async def _handle_state_expiry(self, now: datetime) -> None:
        """Check if state and progress should be reset (auto-expiration)."""
        # Anti-wrinkle keepalive (#339): the mode's idle-timeout and 2 h safety cap
        # only advance inside CycleDetector.process_reading, and the watchdog is
        # stopped for the whole anti-wrinkle tail. A publish-on-change plug can send
        # one final 0 W reading and then go fully silent, so with no further events
        # the mode is pinned in ANTI_WRINKLE for hours. This timer keeps ticking, so
        # when the real sensor has been silent longer than off_delay we inject a
        # synthetic 0 W reading, letting the detector's own logic exit the mode.
        # 0 W and NOT the sensor's last value - unlike the two watchdog sites;
        # the block comment at the injection site below has the reason. Gate
        # on _last_real_reading_time (a genuine tumble pulse still resets the idle
        # timer via the normal handler) and never bump it here, so real silence stays
        # detectable and a still-reporting plug drives itself.
        if self.detector.state == STATE_ANTI_WRINKLE:
            last_real = self._last_real_reading_time
            if (
                last_real is not None
                and (now - last_real).total_seconds() > self._off_delay
            ):
                _ka_w, _ka_obs = self._keepalive_reading()
                self._logger.debug(
                    "Anti-wrinkle keepalive: sensor silent for %.0fs (> off_delay %ss), "
                    "injecting 0 W (sensor last read %.2fW, observed=%s) so the "
                    "idle/2h-cap timer can advance",
                    (now - last_real).total_seconds(),
                    self._off_delay,
                    _ka_w,
                    _ka_obs,
                )
                # 0 W here, NOT the sensor's last value - deliberately different
                # from the two watchdog sites. This keepalive's whole contract is
                # "silence means idle": the detector only advances
                # `_anti_wrinkle_idle_time` while `power < effective_exit`
                # (`cycle_detector.py:1499`), so injecting a stale tumble pulse
                # freezes the timer this call exists to advance - and can start a
                # new-cycle burst candidate on top. The round-11 argument for
                # carrying the real value does not reach here either: appends to
                # `_power_readings` happen in STARTING / RUNNING / PAUSED /
                # ENDING only, so nothing from this site enters the stored trace
                # and there is no fabricated sample to worry about.
                self.detector.process_reading(
                    0.0, now, synthetic=True, observed=_ka_obs
                )
                self._notify_update()
            return
        if (
            not self._cycle_completed_time
            or self.detector.state == STATE_RUNNING
            or self.detector.state == STATE_DELAY_WAIT
            # A probe out of a terminal state (item 515): the overlay waits for its
            # outcome. Resetting the detector here would kill a real start (#267),
            # and a nag must not fire into a new load; an abort resumes the timers.
            or self.detector.state == STATE_STARTING
        ):
            # Cycle is running or not completed, don't reset
            return

        # Keep the cached power honest in the terminal states too (#409). The
        # watchdog is stopped here, so nothing else refreshes it: the panel/entity
        # would keep reporting the last value seen before the cycle ended, and
        # power-based Off detection (#284) - which compares that same cache against
        # power_off_threshold_w - could never see the appliance being switched off.
        # Display/decision cache only; the detector is not fed in a terminal state.
        self._resync_power_from_state(now, feed_detector=False)

        time_since_complete = (now - self._cycle_completed_time).total_seconds()

        # Clean laundry nag notification. In repeat mode (#374) the reminder re-fires
        # every delay-minutes and carries a "stop reminding" action; otherwise it is a
        # single one-shot (the default, unchanged).
        if (
            self._is_clean_state
            and self._clean_state_start is not None
            and self._notify_unload_delay_minutes > 0
            and not self._unload_nag_dismissed
        ):
            delay_s = self._notify_unload_delay_minutes * 60
            first_due = (
                not self._notified_clean_laundry
                and (now - self._clean_state_start).total_seconds() >= delay_s
            )
            repeat_due = (
                self._notify_unload_repeat
                and self._notified_clean_laundry
                and self._unload_nag_count < NOTIFY_UNLOAD_REPEAT_MAX_REMINDERS
                and self._last_unload_nag_time is not None
                and (now - self._last_unload_nag_time).total_seconds() >= delay_s
            )
            if first_due or repeat_due:
                if self._notify_finish_services or self._notify_actions:
                    duration_min = int(time_since_complete / 60)
                    msg_template = self.config_entry.options.get(
                        CONF_NOTIFY_UNLOAD_MESSAGE, DEFAULT_NOTIFY_UNLOAD_MESSAGE
                    )
                    msg = self._safe_format_template(
                        msg_template,
                        fallback_template=DEFAULT_NOTIFY_UNLOAD_MESSAGE,
                        device=self.config_entry.title,
                        duration=duration_min,
                        duration_hm=self._format_duration_hm(duration_min),
                        delay=self._notify_unload_delay_minutes,
                    )
                    extra_vars: dict[str, Any] = {"tag": self._clean_tag}
                    if self._notify_unload_repeat:
                        # Actionable "stop reminding" button + sticky so the user can
                        # end the repeats from the notification (mobile_app targets;
                        # ignored by other platforms). Door-open still ends it too.
                        extra_vars["actions"] = [
                            {
                                "action": self._unload_dismiss_action_id,
                                "title": self._timer_ui_strings.get(
                                    "unload_dismiss_action_title", "Stop reminding"
                                ),
                            }
                        ]
                        extra_vars["sticky"] = "true"
                    sent = self._dispatch_notification(
                        msg,
                        event_type=NOTIFY_EVENT_CLEAN,
                        extra_vars=extra_vars,
                    )
                    if sent:
                        self._notified_clean_laundry = True
                        self._last_unload_nag_time = now
                        self._unload_nag_count += 1
                        if self._notify_unload_repeat:
                            self._ensure_unload_dismiss_listener()
                        self._logger.info(
                            "Sent clean laundry nag notification (%.0f min after "
                            "cycle end)%s",
                            time_since_complete / 60,
                            " [repeat]" if repeat_due else "",
                        )
                    elif self._last_dispatch_deferred:
                        # Held for quiet-hours / presence delivery; the queued copy
                        # fires later. Mark handled so the 60s expiry tick doesn't
                        # enqueue a duplicate nag every minute for the whole window.
                        self._notified_clean_laundry = True
                        self._last_unload_nag_time = now
                        self._unload_nag_count += 1
                        if self._notify_unload_repeat:
                            self._ensure_unload_dismiss_listener()
                else:
                    self._notified_clean_laundry = True
                    self._last_unload_nag_time = now
                    self._unload_nag_count += 1

        # Defer leaving the terminal state while a clean-state unload notification is
        # still pending. Without this guard the 30-min progress reset (or an early
        # power-off) fires before the unload nag, clearing _is_clean_state before the
        # notification can fire. In repeat mode the hold persists across every repeat
        # until the user dismisses it or opens the door. Both expiry modes below honour
        # it (as does the power-off one-shot timer).
        nag_pending = self._unload_nag_active(now)

        # Power-based Off detection (issue #284): opt-in, and only valid when the
        # threshold sits below stop_threshold_w (so it cannot fire while a cycle could
        # still be running, and cannot re-trigger the #267 spin-down ghost cycle). It is
        # evaluated ONLY in a terminal state; active states never reach here because
        # _cycle_completed_time is None until cycle end.
        cfg = self.detector.config
        pot = cfg.power_off_threshold_w
        stop_w = cfg.stop_threshold_w
        power_off_enabled = (
            isinstance(pot, (int, float))
            and isinstance(stop_w, (int, float))
            and 0.0 < pot < stop_w
            and self.detector.state
            in (STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED)
        )

        if power_off_enabled:
            # Power owns the Off transition. The classic timer still zeroes the progress
            # bar after progress_reset_delay, but the terminal state PERSISTS until the
            # machine is actually switched off (no timer fallback, by design: a machine
            # whose standby never drops below the threshold stays "Finished").
            if (
                time_since_complete > self._progress_reset_delay
                and self._cycle_progress != 0.0
            ):
                self._cycle_progress = 0.0
                self._notify_update()

            if nag_pending:
                # Hold the terminal state (and pause power sampling) until the nag fires.
                self._power_off_below_since = None
                self._cancel_power_off_timer()
                return

            if self._current_power < cfg.power_off_threshold_w:
                if self._power_off_below_since is None:
                    self._power_off_below_since = now
                    # Arm a precise one-shot reset instead of waiting for the next
                    # 60s poll (the poll below stays as a backstop).
                    self._arm_power_off_timer(cfg.power_off_delay)
                elif (
                    now - self._power_off_below_since
                ).total_seconds() >= cfg.power_off_delay:
                    self._logger.debug(
                        "Power-based Off: %.2fW below %.2fW for >= %.0fs in %s. "
                        "Resetting to OFF.",
                        self._current_power,
                        cfg.power_off_threshold_w,
                        cfg.power_off_delay,
                        self.detector.state,
                    )
                    self._reset_terminal_to_off()
            else:
                # Power rose back above the threshold: restart the debounce window.
                self._power_off_below_since = None
                self._cancel_power_off_timer()
            return

        # Timer-based Off (feature disabled): classic behaviour, unchanged.
        self._power_off_below_since = None
        self._cancel_power_off_timer()
        if time_since_complete > self._progress_reset_delay:
            if nag_pending:
                return
            # Auto-expire the "Finished" (or other terminal) state
            self._logger.debug(
                "State expiry: cycle idle for %.0fs (threshold: %ss). Resetting to OFF.",
                time_since_complete,
                self._progress_reset_delay,
            )
            self._reset_terminal_to_off()

    def _reset_terminal_to_off(self) -> None:
        """Return a terminal state (Finished/Interrupted/Force-Stopped, incl. the Clean
        overlay) to OFF and clear all post-cycle bookkeeping.

        Single owner of the terminal -> OFF transition, shared by the timer-based and
        the power-based (issue #284) expiry paths so the two can never diverge.
        """
        self._cycle_progress = 0.0
        self._cycle_completed_time = None
        # Clear the Clean overlay too, or check_state() keeps reporting "Clean".
        self._is_clean_state = False
        self._clean_state_start = None
        self._notified_clean_laundry = False
        self._reset_unload_nag_tracking()
        self._power_off_below_since = None
        self._cancel_power_off_timer()
        self.detector.reset(STATE_OFF)
        self._stop_state_expiry_timer()
        self._notify_update()

    def _cancel_power_off_timer(self) -> None:
        """Cancel the pending power-off one-shot reset timer, if armed."""
        if self._remove_power_off_timer is not None:
            self._remove_power_off_timer()
            self._remove_power_off_timer = None

    def _arm_power_off_timer(self, delay: float) -> None:
        """Arm a single cancellable one-shot power-off reset timer.

        Fires ``delay`` seconds after power first fell below the power-off
        threshold, so the terminal->Off transition does not have to wait for the
        next 60s expiry poll. The callback re-verifies the condition before acting,
        so a timer left armed after power rose (before the next poll cancels it) is
        a harmless no-op.
        """
        self._cancel_power_off_timer()

        @callback
        def _fire(_now: datetime) -> None:
            self._remove_power_off_timer = None
            self._power_off_timer_check()

        self._remove_power_off_timer = async_call_later(
            self.hass, max(0.0, float(delay)), _fire
        )

    def _power_off_timer_check(self) -> None:
        """One-shot power-off timer callback: reset to Off only if still valid."""
        cfg = self.detector.config
        pot = cfg.power_off_threshold_w
        stop_w = cfg.stop_threshold_w
        # Same enable + terminal-state guard as the poll path.
        if not (
            isinstance(pot, (int, float))
            and isinstance(stop_w, (int, float))
            and 0.0 < pot < stop_w
            and self.detector.state
            in (STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED)
        ):
            self._power_off_below_since = None
            return
        # Re-verify the below-threshold debounce (power may have risen since arming).
        if self._power_off_below_since is None or self._current_power >= pot:
            return
        if (
            utc_now() - self._power_off_below_since
        ).total_seconds() < cfg.power_off_delay:
            return
        # Honour the clean-laundry unload nag hold (mirrors the poll path).
        if self._unload_nag_active(utc_now()):
            return
        self._logger.debug(
            "Power-based Off (one-shot timer): %.2fW below %.2fW for >= %.0fs in %s. "
            "Resetting to OFF.",
            self._current_power,
            pot,
            cfg.power_off_delay,
            self.detector.state,
        )
        self._reset_terminal_to_off()

    def _keepalive_reading(self) -> tuple[float, bool]:
        """The ``(power, observed)`` pair a watchdog keepalive should carry.

        Two things the watchdog used to get wrong, in one place because they
        come from the same read:

        * **The value.** A synthetic reading is appended to ``_power_readings``
          like any other (there is no guard at the append sites), so it lands in
          the stored ``power_data``. Injecting a hard ``0.0`` therefore writes a
          sample the appliance never produced. On a machine that idles ABOVE its
          stop threshold - the #445 pathology - that fabricates a quiet tail and
          silently defeats ``detect_standby_above_stop``, which reads exactly
          that final sample. The sensor's own last reported value is the honest
          one, and for a machine that really is at 0 W it IS 0.0, so this only
          differs where the old value was a lie.
        * **Whether it was observed.** ``_resync_power_from_state`` returns
          early when the sensor is unavailable / unknown / non-finite, but the
          watchdog injects on the silence interval alone. An unread sensor is an
          outage, and the gap-free tally must not count quiet nobody saw.
        """
        live = self._live_power_state()
        return (live[0], True) if live is not None else (0.0, False)

    def _live_power_state(self) -> tuple[float, datetime] | None:
        """Return ``(power, report_ts)`` from the power sensor's CURRENT state.

        ``hass.states`` is the authoritative record of what the plug last said, and
        unlike the manager's own cache it cannot go stale: it is written on every
        report, including the ones this manager deliberately drops (the
        sampling-interval throttle) or never receives (an event lost across a
        reload). ``last_reported`` advances on every write - unchanged re-reports
        included - so it is the honest "when did the sensor last speak" clock.

        Returns None when the sensor is missing, unavailable or non-numeric.
        """
        state = self.hass.states.get(self.power_sensor_entity_id)
        if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return None
        power = _finite_power(state.state)
        if power is None:
            return None
        report_ts = getattr(state, "last_reported", None) or state.last_updated
        if not isinstance(report_ts, datetime):
            return None
        return power, report_ts

    def _read_power_state_at_setup(self) -> None:
        """Feed the power entity's current state once at setup and seed the caches."""
        state = self.hass.states.get(self.power_sensor_entity_id)
        if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            # A non-finite reading is skipped entirely, which leaves the cache
            # unset, i.e. the pre-#409 behaviour. Seeding it with a nan instead
            # would be PERMANENT: _resync_power_from_state returns early precisely
            # when the sensor is non-finite, so the healing path could never
            # overwrite it, and every later watchdog comparison would be False.
            power = _finite_power(state.state)
            if power is not None:
                try:
                    now = utc_now()
                    self.detector.process_reading(power, now)
                    # Seed the reading cache from the sensor itself (#409). The
                    # reading was already fed to the detector; leaving the manager's
                    # own cache unset meant every reload started with
                    # _current_power = 0 and _last_reading_time = None, so (a) the
                    # power tile/entity reported a value the sensor never had until
                    # the next event and (b) the watchdog - which returns early while
                    # _last_reading_time is None - could neither keepalive nor close a
                    # restored cycle whose plug went silent across the reload.
                    self._current_power = power
                    self._last_reading_time = now
                    self._seed_real_reading_clock(
                        power,
                        getattr(state, "last_reported", None) or state.last_updated,
                    )
                except (ValueError, TypeError, OverflowError):
                    pass

    def _seed_real_reading_clock(self, power: float, report_ts: datetime) -> None:
        """Set the silence clock from the power entity's first state after a
        restart (register item 266): the setup read, or the first event when the
        entity only appears after setup.

        After a restart every entity is written afresh, so the entity's timestamp
        says "the sensor just spoke" even when the plug has been silent for an
        hour. For a restored cycle whose sensor still holds the value it held
        before the restart, that write carries nothing new: keep the restored
        clock, so the watchdog's keepalive, ghost and staleness rules go on
        measuring the real silence. A different value is a genuine report and
        takes the entity's time, as does every setup with nothing restored. The
        report is remembered so the watchdog's resync does not hand it back as
        a missed one and reseed the clock a tick later.
        """
        restored, self._restored_sensor_clock = self._restored_sensor_clock, None
        if (
            restored is not None
            and restored[1] is not None
            and math.isclose(power, restored[1], rel_tol=0.0, abs_tol=1e-6)
            and self.detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
        ):
            self._last_real_reading_time = restored[0]
            self._setup_report_ts = report_ts
            self._logger.info(
                "Power sensor still reads %.2fW, as before the restart: keeping its "
                "last report at %s as the silence clock",
                power,
                restored[0],
            )
            return
        self._last_real_reading_time = report_ts

    def _resync_power_from_state(self, now: datetime, feed_detector: bool) -> None:
        """Re-anchor the cached power on the sensor's live state (#409).

        ``_current_power`` is otherwise a pure event cache: one dropped or missed
        state event and it stays wrong indefinitely, because nothing ever compares
        it against the sensor again. That single stale number then drives the
        watchdog's high-power branch, the low-power keepalive gate and the #284
        power-off detection, so the divergence does not merely show a wrong value
        in the panel - it decides whether a cycle ends at all.

        Called from the two periodic timers (the in-cycle watchdog and the terminal
        state-expiry poll), so the cache can never be more than one tick out of
        step with the sensor. When ``feed_detector`` is set and the sensor has
        reported since our last real reading, that report is also handed to the
        detector: it is a genuine observation we simply never processed. Timestamped
        with ``now`` rather than the report time so the detector's dt can never run
        backwards (a negative dt is discarded, taking the reading with it).
        """
        live = self._live_power_state()
        if live is None:
            return
        power, report_ts = live
        missed = (
            self._last_real_reading_time is None
            or report_ts > self._last_real_reading_time
        ) and report_ts != self._setup_report_ts  # setup already took it (item 266)
        if feed_detector and missed:
            self._logger.debug(
                "Resync: sensor reported %.2fW at %s but the last processed reading "
                "was %s (cached %.2fW); processing it now",
                power,
                report_ts,
                self._last_real_reading_time,
                self._current_power,
            )
            self.detector.process_reading(power, now)
            self._last_reading_time = now
            self._last_real_reading_time = report_ts
        self._current_power = power

    def _low_power_silence_budget_s(
        self, elapsed: float, expected: float, verified_pause: bool
    ) -> float:
        """How long a low-power wait may go without a real reading before the
        watchdog force-ends it as stale. The restart path asks the same question
        of a snapshot (register item 266), so the two cannot drift apart.
        """
        # Dishwashers can have very long silent drying phases (up to 2h)
        # We use the device-specific timeout as the floor for this effective timeout.
        # The floor is applied unconditionally - dishwashers have passive drying phases
        # even when no profile has been matched yet.  The original restriction to matched
        # cycles caused premature kills: with the default 3600s timeout, an unmatched
        # dishwasher cycle was killed ~1h after the last sensor update, while the
        # physical drying phase could still have 1-2h of silent runtime remaining.
        budget = max(
            float(DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE.get(self.device_type, 0)),
            float(self._low_power_no_update_timeout),
        )
        # Profile-Aware Extension: with a matched profile, never kill during the
        # expected duration - remaining + 1800 s (30 min buffer for drying/pause).
        if expected > 0 and elapsed < expected:
            budget = max(budget, expected - elapsed + 1800)
        # Verified Pause Extension: a confirmed legitimate pause (e.g. drying) gets
        # up to the global deferral limit + the same buffer.
        if verified_pause:
            budget = max(budget, DEFAULT_MAX_DEFERRAL_SECONDS + 1800)
        return budget

    async def _watchdog_check_stuck_cycle(self, now: datetime) -> None:
        """Watchdog: check if cycle is stuck (no updates for too long)."""
        if self.detector.state not in (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING):
            return

        if not self._last_reading_time:
            return

        # Re-anchor the cached power on the sensor before any branch below reads it
        # (#409): every decision from here on (keepalive vs force-end, low- vs
        # high-power handling) is made against _current_power, and an event cache
        # that has drifted from the sensor turns those decisions into fiction.
        self._resync_power_from_state(now, feed_detector=True)
        if self.detector.state not in (
            STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING
        ):
            # A missed reading can legitimately end the cycle (that is the point of
            # picking it up). Every other injection path in this method returns
            # immediately for the same reason: none of the branches below are
            # meaningful once the cycle is over.
            self._notify_update()
            return

        # Refresh the remaining-time / progress estimate on the watchdog cadence
        # (sampling-derived: max(30, 2*sampling+1)s), independently of incoming power
        # events. A publish-on-change plug emits nothing during a flat low-power tail
        # (e.g. a dishwasher's ~30 min drying phase at 0 W), so the event-driven
        # _update_remaining_only never runs and the displayed countdown freezes at
        # whatever value it last showed. The estimate is wall-clock based
        # (net_elapsed_seconds), so this tick advances it correctly with zero new
        # readings; it no-ops until a profile is matched. Kept ahead of the
        # keepalive/force-end branches below so even a verified-pause drying tail
        # (which skips those branches) still ticks down.
        # The "almost done" reminder reads that same estimate, so it is checked here
        # too: on the power path alone a silent tail delayed it to the next reading,
        # typically the final pump-out, alongside the finish (audit PROGRESS-07).
        if self.detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            self._update_remaining_only()
            self._check_pre_completion_notification()
            self._notify_update()

        time_since_any_update = (now - self._last_reading_time).total_seconds()

        # Calculate time since REAL update (if available, else fallback to any update)
        last_real = self._last_real_reading_time or self._last_reading_time
        time_since_real_update = (now - last_real).total_seconds()

        elapsed = self.detector.get_elapsed_seconds()
        expected = getattr(self.detector, "expected_duration_seconds", 0)

        # 0a. PUMP STUCK DETECTION (Pump Monitor only)
        # If a pump cycle has been running longer than the configured stuck threshold,
        # fire a single warning event so the user can wire an automation/alert.
        # Skip while user-paused or detector-verified-pause to avoid false positives.
        _verified_pause = getattr(self.detector, "_verified_pause", False)
        if self.device_type == DEVICE_TYPE_PUMP and not self._pump_stuck and not self._is_user_paused and not _verified_pause:
            adjusted_elapsed = elapsed - self._total_user_paused_seconds
            if adjusted_elapsed >= self._pump_stuck_duration:
                self._pump_stuck = True
                self._logger.warning(
                    "Pump stuck detected: cycle has been running for %.0fs net "
                    "(threshold: %ds). Firing %s event.",
                    adjusted_elapsed,
                    self._pump_stuck_duration,
                    EVENT_PUMP_STUCK,
                )
                self.hass.bus.async_fire(
                    EVENT_PUMP_STUCK,
                    {
                        "device": self.config_entry.title,
                        "entry_id": self.entry_id,
                        "elapsed_seconds": round(adjusted_elapsed),
                        "threshold_seconds": self._pump_stuck_duration,
                    },
                )
                self._notify_update()

        # 0. ZOMBIE KILLER (Hard Limit)
        # If cycle has run significantly longer than expected (300%), kill it.
        # Only applies if we have a profile match. Skip while user-paused or
        # detector-verified-pause to avoid killing legitimately paused cycles.
        _verified_pause_zombie = getattr(self.detector, "_verified_pause", False)
        if (
            expected > 0
            and not self._is_user_paused
            and not _verified_pause_zombie
        ):
            adjusted_elapsed = elapsed - self._total_user_paused_seconds
            if adjusted_elapsed > (expected * 3.0) and adjusted_elapsed > 14400:
                self._logger.warning(
                    "Watchdog: Zombie cycle detected (%.0fs net > 300%% of expected %.0fs). Force-ending.",
                    adjusted_elapsed, expected
                )
                self.detector.force_end(now)
                self._current_power = 0.0  # Force 0W
                self._notify_update()
                return

        # Secondary zombie guard for unmatched cycles (expected == 0 when no profile
        # has been matched, or a hand-created profile has no learned avg_duration yet).
        # This is a failsafe against a stuck FALSE START, not a cap on real programmes.
        # The detector already hard-caps any cycle at 8h (cycle_detector 28800s), so
        # this only ever fires *earlier*, and only when the cycle looks like it is not
        # really running any more. Its old form force-ended on elapsed time alone, which
        # truncated genuine long runs (issue #404: a >4h washer-dryer wash+dry drawing
        # hundreds of watts was cut off at exactly 4h; the empty profile it was learning
        # against was then stored force_stopped/truncated and could never learn its true
        # duration, so every subsequent run repeated the deadlock). Three gates keep it
        # honest without losing the false-start failsafe:
        #   - a device-scaled ceiling (wet/long appliances get a longer fuse), never
        #     above the detector's 8h absolute cap;
        #   - only when the appliance is effectively idle (current draw below the
        #     running threshold) - a cycle still pulling real power is not a stuck false
        #     start (reporter's evidence: the plug never dropped below 64 W);
        #   - skipped when an authoritative external end trigger is wired and reporting,
        #     since that signal will end the cycle cleanly.
        # Gate on the active detector state, not _current_program: the latter is only set
        # to "detecting..." on the RUNNING transition, so a cycle stuck in STARTING keeps
        # a stale program and would never hit this guard.
        elif (
            expected == 0
            and not self._is_user_paused
            and not _verified_pause_zombie
            and elapsed > self._unmatched_watchdog_ceiling
            and self._current_power < self.detector.config.start_threshold_w
            and not self._external_end_trigger_available()
            and self.detector.state in (
                STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING
            )
        ):
            self._logger.warning(
                "Watchdog: Unmatched idle cycle exceeded %.0fs (elapsed %.0fs, "
                "power %.1fW). Force-ending.",
                self._unmatched_watchdog_ceiling,
                elapsed,
                self._current_power,
            )
            self.detector.force_end(now)
            self._current_power = 0.0
            self._notify_update()
            return

        # 1. GHOST CYCLE SUPPRESSOR
        # If we are "detecting" for more than 10 minutes and haven't seen an update for 5 minutes,
        # it's likely a pump-out spike or an accidental start (ghost cycle).
        # We end it aggressively ONLY if it started shortly after another cycle ended (Suspicious Window).
        cycle_start = self.detector.current_cycle_start
        is_suspicious = False
        if cycle_start and self._last_cycle_end_time:
            # Dishwashers have a drain pump-out that fires 3-8 min after the main
            # cycle ends; use a wider suspicious window so the ghost suppressor can
            # catch it without false-positives on washing machines / dryers.
            suspicious_window = 600 if self.device_type == "dishwasher" else 180
            if (cycle_start - self._last_cycle_end_time).total_seconds() < suspicious_window:
                is_suspicious = True

        # For dishwashers in the suspicious window, kill pump-out ghosts faster.
        # Pump-outs last 1-3 min then go silent; the standard 10-min wait allows
        # them to accumulate too much runtime before suppression fires.
        dishwasher_pump_out = (
            is_suspicious
            and self.device_type == "dishwasher"
            and elapsed > 180   # 3 minutes
            and time_since_real_update > 60  # 1 minute of silence
        )

        if (
            self._current_program == "detecting..."
            and is_suspicious
            and (
                dishwasher_pump_out
                or (elapsed > 600 and time_since_real_update > 300)
            )
        ):
            self._logger.warning(
                "Watchdog: Ghost cycle suppressed (within suspicious window). Detecting for %.0fs with %.0fs silence.",
                elapsed, time_since_real_update
            )
            self.detector.force_end(now)
            self._current_power = 0.0
            self._notify_update()
            return

        # --- LOW POWER HANDLING ---
        # If we are in a low power state (waiting for off_delay or drying profile),
        # we treat silence leniently. We inject keepalives until the stricter
        # low_power_no_update_timeout is reached.
        effective_low_power_timeout = self._low_power_silence_budget_s(
            elapsed, expected, bool(getattr(self.detector, "_verified_pause", False))
        )

        if self.detector.is_waiting_low_power():

            # 2. Staleness Check
            if time_since_real_update > effective_low_power_timeout:
                self._logger.warning(
                    "Watchdog: Force-ending cycle. Low-power state stale for %.0fs (> %.0fs).",
                    time_since_real_update,
                    effective_low_power_timeout
                )
                self.detector.force_end(now)
                self._last_reading_time = now
                self._current_power = 0.0
                self._notify_update()
                return

            # 3. Injection Check (Keepalive) - on the WATCHDOG cadence (#427).
            #
            # This used to be two gates: real-update silence past
            # `no_update_active_timeout`, else any-update silence past
            # `off_delay`. Both are *stall-detection* timeouts, sized at roughly
            # `p95_cadence * 20`; neither has anything to do with how fast the
            # end accumulator should be advanced. A publish-on-change plug going
            # quiet at standby is not a stall - it is the exact condition this
            # keepalive exists for - and it was precisely then that nothing was
            # injected for minutes at a time.
            #
            # Measured on the #427 reporter's v0.5.6 cycle (AEG L8FE74485,
            # no_update_active_timeout 387 s, watchdog_interval 30 s): the
            # accumulator froze twice, 383 s before PAUSED and ~390 s inside
            # ENDING - about 13 of the reported ~20 minutes was nothing but "no
            # reading arrived, so no gate was evaluated".
            #
            # This CANNOT end a cycle early. `_time_below_threshold` accumulates
            # wall-clock `dt` between readings, so the total after N seconds of
            # quiet is the same whether that arrived as one reading or twenty;
            # injecting more often changes only how promptly a crossing is
            # noticed, never the value compared against `effective_off_delay`.
            # With the fix the reporter's cycle finishes 8.2 min after the last
            # active reading, which is exactly their configured
            # `max(off_delay 480, min_off_gap 480)`.
            #
            # A verified pause is no longer excluded, and that is not a loosening:
            # the old 3b gate injected during verified pauses anyway (just on the
            # slower off_delay cadence), and every guard that a verified pause is
            # meant to hold off - the ENDING hard finalize, the terminal-drop
            # finalize, the zombie killer - reads `_verified_pause` directly and
            # is untouched by how often we sample.
            if time_since_real_update > self._watchdog_interval:
                _ka_w, _ka_obs = self._keepalive_reading()
                # On time, this tick closes at most two intervals (the first one
                # after a real reading) and then one. A longer one means the tick
                # itself was late - host suspend, event-loop stall, a restart -
                # and nobody watched the sensor in between, so the gap-free tally
                # must not bank it (register item 391). Only the two shorten-only
                # consumers of that tally read `observed`, so a false "late"
                # costs end lag, never an early end.
                if time_since_any_update > WATCHDOG_LATE_TICK_FACTOR * self._watchdog_interval:
                    _ka_obs = False
                self._logger.debug(
                    "Watchdog: Low-power sensor silence (%.0fs > watchdog interval "
                    "%ss). Injecting %.2fW keepalive (observed=%s) to advance "
                    "accumulator.",
                    time_since_real_update,
                    self._watchdog_interval,
                    _ka_w,
                    _ka_obs,
                )
                # Do NOT update _last_real_reading_time here, and tell the
                # detector this reading is ours: it must still advance the
                # quiet timers (that is the whole point of injecting it) but
                # must not count as the sensor having reported (items 238, 289).
                self.detector.process_reading(
                    _ka_w, now, synthetic=True, observed=_ka_obs
                )
                self._last_reading_time = now
                self._current_power = _ka_w
                self._save_snapshot_while_silent(now)
                self._notify_update()
                return

            return

        # Fallback for old "Case 1.5" logic (Low Power but NOT is_waiting_low_power)
        # Check this BEFORE High Power timeout to prevent trapping "Not Yet Waiting" states
        # Inject as soon as the earliest of: off_delay silence OR no_update_active_timeout.
        if self._current_power <= self.detector.config.min_power and (
            time_since_any_update > self._config.off_delay
            or time_since_real_update > self._no_update_active_timeout
        ):
            # Treating as start of low power wait
            _ka_w, _ka_obs = self._keepalive_reading()
            self._logger.debug(
                "Watchdog: Silence at low power (%.0fs). Injecting %.2fW "
                "(observed=%s).",
                time_since_any_update, _ka_w, _ka_obs,
            )
            self.detector.process_reading(
                _ka_w, now, synthetic=True, observed=_ka_obs
            )
            self._last_reading_time = now
            self._current_power = _ka_w
            self._save_snapshot_while_silent(now)
            self._notify_update()
            return

        # --- HIGH POWER HANDLING (Normal) ---
        # If power is high, we expect frequent updates.

        if time_since_any_update > self._no_update_active_timeout:

            # Check if high power (running)
            if self._current_power > self.detector.config.min_power:
                # Allow extended silence if within reasonable cycle bounds
                expected = getattr(self.detector, "expected_duration_seconds", 0)
                elapsed = self.detector.get_elapsed_seconds()
                limit = (expected + 14400) if expected > 0 else 14400 # 4h default

                if elapsed < limit:
                    # Silence at high power buys the cycle more time, but it must NOT
                    # be turned into data (#409). This branch used to re-feed the
                    # cached power to the detector on every tick, which
                    #   - wrote a reading the sensor never reported into the cycle
                    #     trace, producing a dead-flat non-zero tail that inflates
                    #     the stored duration and energy and skews matching;
                    #   - reset _time_below_threshold and _last_active_time, so the
                    #     detector's own end-of-cycle timers could never accumulate
                    #     and the fabricated tail was self-sustaining until this
                    #     limit expired hours later (force_stopped);
                    #   - fed that inflated duration back into the profile once the
                    #     cycle was labelled, growing avg_duration and therefore
                    #     `limit`, so each subsequent stuck cycle ran longer still.
                    # The keepalive was never what kept the cycle alive - the
                    # detector only ends a cycle on quiet time it has actually
                    # observed - so deferring the force-end is enough. The trace
                    # keeps an honest hole (integrate_wh drops outage-sized
                    # segments), _update_remaining_only above keeps the ETA ticking
                    # on wall-clock, and the resync at the top of this method picks
                    # the real value up the moment the plug speaks again.
                    self._logger.info(
                        "Watchdog: High power (%.1fW) stale (%.0fs, sensor silent "
                        "%.0fs). Deferring end (elapsed %.0fs < limit %.0fs).",
                        self._current_power,
                        time_since_any_update,
                        time_since_real_update,
                        elapsed,
                        limit,
                    )
                    self._last_reading_time = now
                    self._save_snapshot_while_silent(now)
                    self._notify_update()
                    return

            # If we get here, it's truly stuck/offline
            self._logger.warning(
                "Watchdog: Force-ending cycle. Active state stale for %.0fs (> timeout).",
                time_since_any_update
            )
            self.detector.force_end(now)
            self._current_power = 0.0  # FIX: Reset current power
            self._notify_update()
            return


    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """Handle state change from detector."""
        self._logger.debug("Washer state changed: %s -> %s", old_state, new_state)
        self.diag_buffer.record_state(
            old_state, new_state, self._current_program, utc_now()
        )
        # A start from idle owns no update intervals yet: drop anything a false
        # start (STARTING -> OFF, which ends no cycle) left pending, so it is not
        # committed with this cycle (#458). DELAY_WAIT is idle too, and since item
        # 504 its false starts return there, so hours of standby probes would
        # otherwise reach the next completed cycle; since item 515 so do the
        # terminal states' (the cycle end already closed the previous cycle's).
        if new_state == STATE_STARTING and old_state in CADENCE_RESET_FROM_STATES:
            self.learning_manager.discard_cycle_cadence()
        # The completed/Clean overlay (the cycle end, Clean, the unload nag, the
        # 100 % progress) is cleared when a new cycle COMMITS, in the RUNNING
        # branch below, not when a probe begins (register item 515): most probes
        # out of Finished abort, the detector returns to the terminal state, and
        # clearing here lost Clean and the nag to a blip. The expiry timer keeps
        # running through the probe and skips STARTING (#267, _handle_state_expiry).
        if (
            new_state == STATE_STARTING
            and not TERMINAL_PROBE_RETURNS
            and self._cycle_completed_time is not None
        ):
            self._cycle_completed_time = None
            self._is_clean_state = False
            self._clean_state_start = None
            self._notified_clean_laundry = False
            self._reset_unload_nag_tracking()
            self._cycle_progress = 0.0
            self._power_off_below_since = None
            self._cancel_power_off_timer()
            self._stop_state_expiry_timer()
        if old_state == STATE_STARTING and new_state in (
            STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED
        ):
            # Item 515: a false start back in its terminal state ended no cycle.
            self._cycle_start_time = None
        if new_state == STATE_RUNNING:
            new_cycle_detected = old_state in (STATE_OFF, STATE_STARTING, STATE_UNKNOWN)
            # Only reset estimates if we are truly starting a NEW cycle (from off or starting)
            # If we transition from PAUSED or ENDING, it's a resume - keep estimates!
            if new_cycle_detected:
                # The previous cycle's completed/Clean overlay ends here (item 515;
                # Clean and the nag tracking are reset further down).
                self._cycle_completed_time = None
                self._stop_state_expiry_timer()
                self._power_off_below_since = None
                self._cancel_power_off_timer()

                self._current_program = "detecting..."
                self._manual_program_active = False
                # Confidence belongs to the match that produced it, so it cannot
                # carry into the next cycle. It was only ever assigned by a real
                # match update, never cleared, and the cycle-end tail stamps it onto
                # cycle_data["match_confidence"] - so a cycle that never matched
                # (most obviously one running a hand-pinned program, where
                # _update_estimates returns early and the matcher never runs)
                # persisted the PREVIOUS cycle's confidence as its own. That number
                # then feeds the learning feedback,
                # i.e. fabricated match provenance on a cycle that has none - the
                # #400 class of bug. Zero means "no opinion" and is not stored.
                self._last_match_confidence = 0.0
                self._last_member_confidence = None
                self._notified_pre_completion = False
                self._time_remaining = None
                self._total_duration = None
                self._cycle_progress = 0
                # ...and the EMA behind it (item 388d): an armed program sets the
                # duration before the "no profile" reset can run, so the new
                # cycle started from the previous one's smoothed figure (89% two
                # minutes into a two-hour wash).
                self._smoothed_progress = 0.0
                self._matched_profile_duration = None
                # The ML expectation is the median of the profile's last 20 cycles;
                # cached per profile only, it stayed frozen until another programme
                # was matched or HA restarted (audit PROGRESS-16). Once per cycle.
                self._ml_end_expectation_cache = None
                self._last_estimate_time = None
                self._score_history = {}  # Reset score history on new cycle
                self._match_persistence_counter = {}  # Reset persistence counter
                self._unmatch_persistence_counter = 0  # Reset unmatch counter
                self._current_match_candidate = None  # Reset candidate
                self._notified_start = False # Reset start notification state
                self._start_event_fired = False
                self._last_cycle_post_anomaly = {}  # Clear previous cycle's anomaly cache
                self._cycle_start_time = self.detector.current_cycle_start or utc_now()
                self._ranking_snapshot_cycle_id = str(uuid.uuid4())
                self._reset_live_notification_state()
                # Snapshot the external energy meter (issue #316) so cycle end can
                # take an accurate start->end delta. No-op when none is configured.
                self._snapshot_energy_meter_start()
                # Open the dynamic price timeline (#426) with the price in force
                # right now. No-op when no price entity is configured.
                self._start_price_timeline()

                # Reset pause tracking and clean state for new cycle
                self._is_user_paused = False
                self._user_pause_start = None
                self._total_user_paused_seconds = 0.0
                self._is_clean_state = False
                self._fired_cycle_timers = set()
                self._clear_timer_pause_notification()
                self._clean_state_start = None
                self._notified_clean_laundry = False
                self._reset_unload_nag_tracking()

                # A program the user armed while idle - or pinned during STARTING,
                # which the reset above would otherwise have wiped - takes effect
                # here (#411). Placed after the reset for two reasons: the duration
                # it sets must not be nulled a line later, and it applies the pin,
                # which refreshes the estimate - so it has to run once the pause
                # totals belong to THIS cycle. Reading them a few lines earlier
                # computed the new cycle's first ETA from the previous cycle's
                # paused seconds, which the back-to-back case makes reachable: when
                # the cycle-end tail returns early on the new-cycle token guard, it
                # never clears them either, and the live progress notification is
                # interval-throttled, so that wrong ETA sits on the phone until the
                # next allowed tick. Still before the start event, so the event
                # carries the real program name rather than "detecting...".
                self._consume_armed_program()

                self._start_watchdog()  # Start watchdog when cycle starts

                # Fire the start event immediately on cycle detection so listeners always
                # receive it, even when no profile match occurs yet.
                if self._notify_fire_events:
                    self.hass.bus.async_fire(
                        EVENT_CYCLE_STARTED,
                        {
                            "entry_id": self.entry_id,
                            "device_name": self.config_entry.title,
                            "device_type": self.device_type,
                            "program": self._current_program or "unknown",
                            "start_time": self._cycle_start_time.isoformat(),
                        },
                    )
                    self._start_event_fired = True
                    # Mark the start fully handled ONLY when there is no push to send, so
                    # the restart-recovery fallback does not re-enter for event-only
                    # configs. When a push service/action IS configured, leave
                    # _notified_start False so the push block below still fires: a config
                    # with both events and push must get both (event delivery is tracked
                    # separately by _start_event_fired).
                    if not (self._notify_start_services or self._notify_actions):
                        self._notified_start = True

                # Fire push notification immediately - do not wait for profile matching.
                if not self._notified_start and (self._notify_start_services or self._notify_actions):
                    msg_template = self.config_entry.options.get(
                        CONF_NOTIFY_START_MESSAGE, DEFAULT_NOTIFY_START_MESSAGE
                    )
                    msg = self._safe_format_template(
                        msg_template,
                        fallback_template=DEFAULT_NOTIFY_START_MESSAGE,
                        device=self.config_entry.title,
                        program=self._current_program,
                    )
                    # B4: append a peak-rate advisory tip when the current price is
                    # at/above the configured threshold. Purely informational.
                    tip = self._peak_rate_tip(
                        self.config_entry.options, self._resolve_energy_price()
                    )
                    if tip:
                        msg = f"{msg}\n{tip}"
                    self._dispatch_notification(
                        msg,
                        event_type=NOTIFY_EVENT_START,
                        extra_vars={
                            "program": self._current_program,
                            "tag": self._lifecycle_tag,
                        },
                    )
                    self._notified_start = True
                    self._logger.info(
                        "Sent start notification for program '%s'", self._current_program
                    )
                    self._check_pre_completion_notification()
            else:
                self._logger.debug("Cycle resumed from %s, preserving estimates", old_state)
                # Ensure watchdog is running
                self._start_watchdog()

        # Auto-open dishwasher: arm the dwell if ENDING while the door is already
        # open (door event missed, or door popped just as ENDING fired) (#342).
        self._maybe_arm_door_end_dwell_if_open()

        # Stop watchdog when transitioning to OFF from any active state
        if new_state == STATE_OFF:
            self._stop_watchdog()  # Stop watchdog regardless of previous state
            self._cycle_start_time = None

        # A power-sensor change saved mid-cycle lands now (audit MANAGER-11).
        if (
            getattr(self, "_pending_power_sensor", None) is not None
            and new_state not in _SENSOR_SWAP_BLOCKED_STATES
        ):
            self._spawn_tracked(self._async_apply_pending_power_sensor())

        self._notify_update()

    def _discard_cycle_cleanup(self) -> None:
        """Discard cleanup for a ghost/noise blip that is never persisted.

        The detector still transitions into a terminal state (FINISHED/INTERRUPTED)
        when it fires the cycle-end callback, and a live active-cycle snapshot may be
        sitting in the store. The normal cycle-end tail clears that snapshot and arms
        the terminal-state expiry so the UI returns to Off; a suppressed ghost skips
        that tail, so mirror the essential parts here — otherwise the device is
        stranded in a terminal state with a stale active snapshot until the next
        cycle. Deliberately does NOT persist, notify, or run the learning pipeline.
        """
        self._spawn_tracked(self.profile_store.async_clear_active_cycle())
        # Anchor the terminal state so _handle_state_expiry (and power-off) can act,
        # then arm the expiry timer that resets terminal -> Off after the reset delay.
        self._cycle_completed_time = utc_now()
        self._start_state_expiry_timer()

    def _on_cycle_end(self, cycle_data: dict[str, Any]) -> None:
        """Handle cycle end - clear all active timers and state."""
        duration = cycle_data["duration"]
        max_power = cycle_data.get("max_power", 0)

        # Coalesce this cycle end's store writes from its first one: the cadence
        # commit below already spawns a suggestion pass that saves (item 456).
        try:
            self.profile_store.coalesce_saves()
        except Exception:  # noqa: BLE001 - a save policy must never break cycle end
            self._logger.debug("Could not coalesce cycle-end saves", exc_info=True)

        # First, and synchronously: every end - ghost, pump-out, persisted or not
        # - commits this cycle's update intervals to the cadence model or drops
        # them, so they can never ride into the next cycle (#458). It used to run
        # at the end of the async pipeline, which the ghost and pump-out branches
        # below return before.
        try:
            self.learning_manager.close_cycle_cadence(cycle_data)
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.debug("Cadence commit failed", exc_info=True)

        # IMMEDIATELY stop all active timers when cycle determined to have ended
        self._stop_watchdog()  # Stop active cycle watchdog
        self._stop_state_expiry_timer()  # Cancel any pending progress reset
        self._clear_timer_pause_notification()
        self._cancel_door_end_dwell()  # Discard stale auto-open dwell (#342)
        prev_cycle_end_time = self._last_cycle_end_time
        self._last_cycle_end_time = utc_now()
        self._pump_stuck = False  # Reset for next pump cycle

        # Auto-Tune: Check for ghost cycles (short duration AND low energy)
        # Ghost = duration < 60s AND total energy < 0.05 Wh (avoids killing pump-out spikes)
        power_data = cycle_data.get("power_data", [])
        cycle_energy_wh = 0.0
        if power_data and len(power_data) >= 2:
            valid: list[tuple[float, float]] = []
            for p in power_data:
                try:
                    valid.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError, IndexError, OverflowError):
                    pass
            if len(valid) >= 2:
                try:
                    valid.sort(key=lambda x: x[0])
                    ts = np.array([v[0] for v in valid])
                    ps = np.array([v[1] for v in valid])
                    # Shared trapezoidal integrator with a data-driven outage gap
                    # (single source with ProfileStore.async_add_cycle).
                    cycle_energy_wh = integrate_wh(
                        ts, ps, max_gap_s=energy_gap_threshold_s(ts)
                    )
                except (TypeError, ValueError, ArithmeticError):
                    cycle_energy_wh = 0.0

        # Ghost cycle: short AND low energy (real cycles have energy even if short).
        # Suppress it exactly like the dishwasher pump-out branch below: feed the
        # auto-tune counter but do NOT store it or run the cycle-end pipeline. Without
        # the return a sub-60 s / sub-0.05 Wh blip would fall through to persistence
        # and the (un-gated) finish notification, firing a phantom "cycle finished".
        if duration < 60 and cycle_energy_wh < 0.05:
            self._handle_noise_cycle(max_power)
            self._discard_cycle_cleanup()
            return  # Do not store this as a real cycle
        if self.device_type == "dishwasher" and prev_cycle_end_time is not None:
            # Pump-out suppression: dishwashers end cycles with a brief drain pump
            # (typically 30-300 s, < 1 Wh) a few minutes after the main cycle
            # finishes.  If a short, low-energy cycle starts within 10 minutes of
            # the previous cycle, treat it as a pump-out ghost and do not store it.
            cycle_start_str = cycle_data.get("start_time")
            cycle_start_dt = (
                dt_util.parse_datetime(cycle_start_str) if cycle_start_str else None
            )
            if cycle_start_dt is not None:
                gap = (cycle_start_dt - prev_cycle_end_time).total_seconds()
                if 0 < gap < 600 and duration < 300 and cycle_energy_wh < 1.0:
                    self._logger.info(
                        "Suppressing dishwasher pump-out ghost: "
                        "gap=%.0fs, duration=%.0fs, energy=%.3f Wh",
                        gap,
                        duration,
                        cycle_energy_wh,
                    )
                    self._handle_noise_cycle(max_power)
                    self._discard_cycle_cleanup()
                    return  # Do not store this as a real cycle

        # Store energy for notification and persistence (calculated above for ghost detection)
        cycle_data["energy_wh"] = round(cycle_energy_wh, 3)

        # External energy meter (issue #316): when an accurate start->end delta is
        # available, record it alongside the integrated value and mark the source.
        # The integrated energy_wh above is left untouched so matching / ML / anomaly
        # stats stay internally consistent; only user-facing figures prefer the meter.
        cycle_data["energy_source"] = "integration"
        meter_wh = self._compute_meter_energy_wh()
        if meter_wh is not None:
            cycle_data["energy_meter_wh"] = round(meter_wh, 3)
            cycle_data["energy_source"] = "meter"

        # Schedule heavy post-processing asynchronously. Capture this cycle's identity
        # token so the async tail can tell if a NEW cycle started while it was awaiting
        # (power changes are handled synchronously, so a back-to-back load can drive the
        # detector into a fresh RUNNING before post-processing completes). See B1 in
        # _async_process_cycle_end.
        end_token = self._ranking_snapshot_cycle_id
        # Freeze the price timeline alongside the token, for the same reason: the
        # back-to-back start that changes the token also calls _start_price_timeline,
        # which replaces this cycle's timeline with the NEW cycle's opening sample
        # before the task reaches the costing step (#426).
        end_price_timeline = list(self._price_timeline)
        if not self._is_shutdown:
            self._cycle_end_task = self._spawn_tracked(
                self._async_process_cycle_end(
                    cycle_data,
                    cycle_token=end_token,
                    price_timeline=end_price_timeline,
                )
            )

    def _ml_end_confidence(
        self, points: list[tuple[float, float]], expected_duration: float
    ) -> float | None:
        """Opt-in ML end-guard provider handed to the CycleDetector.

        Returns P(the latest low-power event is the true cycle end) from the
        shipped or on-device-trained cycle-end model, or ``None`` when ML models
        are disabled for this device, no profile is matched, or the model /
        features are unavailable. ``None`` means the detector keeps its existing
        power/energy-based behavior, so this can only ever *defer* a completion.
        """
        if not ENABLE_ML_END_GUARD:
            return None
        try:
            from .ml.engine import ml_models_enabled, resolve_scorer

            if not ml_models_enabled(self.config_entry.options):
                return None
            profile_name = self._current_program
            if (
                not profile_name
                or profile_name in ("off", "detecting...", "restored...")
                or profile_name not in self.profile_store.get_profiles()
            ):
                return None
            end_fn, _ = resolve_scorer("end")
            if end_fn is None:
                return None
            expectation = self._profile_end_expectation(profile_name, expected_duration)
            if expectation is None:
                return None
            from .ml.feature_extraction import latest_end_event_features

            features = latest_end_event_features(points, expectation)
            if features is None:
                return None
            return float(end_fn(features))
        except Exception as err:  # noqa: BLE001 - ML must never break detection
            self._logger.debug("ML end-guard scoring skipped: %s", err)
            return None

    def _profile_end_expectation(
        self, profile_name: str, expected_duration: float
    ) -> dict[str, float] | None:
        """Median duration/energy/peak for a matched profile, for end features.

        Cached per profile so the guard does not re-decompress history on every
        low-power reading during ENDING. The detector's authoritative expected
        duration overrides the median when available.
        """
        expectation, self._ml_end_expectation_cache = progress_mod.profile_end_expectation(
            self.profile_store,
            profile_name,
            expected_duration,
            self._ml_end_expectation_cache,
        )
        return expectation

    def _terminal_drop_provider(
        self, points: list[tuple[float, float]], expected_duration: float
    ) -> bool:
        """Opt-in terminal-drop detector handed to the CycleDetector.

        Returns ``True`` when the current low-power event is a hard cliff-to-~0
        that began at an elapsed offset EARLIER than this device has ever
        legitimately gone quiet (learned from its own completed cycles) - i.e. an
        anomalously-early drop that is almost certainly a real stop (plug pulled /
        cancelled), not a soak pause.  The detector then finalizes quickly instead
        of waiting out the full soak-bridging ``min_off_gap``.

        A very early drop is below the matcher's duration gate, so match
        confidence is not available to confirm familiarity; instead the cycle's
        **power level** must be one this device has produced before (see
        ``is_terminal_drop``) - a cycle drawing power unlike anything in its
        history is treated as a possible new program and deferred.

        Returns ``False`` (keep the proven slow path) when it is off for this
        device (``detector_config.terminal_drop_enabled``: always on for
        dishwashers, behind the "Apply smart models" toggle otherwise - audit
        ML-08), a default-on dishwasher has no committed unambiguous match yet
        (``terminal_drop_may_fire``), there is too little history to trust the
        baseline, the cycle looks novel, or the drop is not anomalously early.
        Never raises - the
        anomaly signal must never break detection.
        """
        try:
            options = self.config_entry.options
            if not terminal_drop_enabled(self.device_type, options):
                return False
            # Default-on dishwashers fire only on a committed, unambiguous match.
            if not terminal_drop_may_fire(
                self.device_type, options, self.detector,
                pinned=bool(self._manual_program_active),
            ):
                return False
            return terminal_drop_fires(
                points,
                self._terminal_drop_baseline(),
                float(self.detector.config.stop_threshold_w),
            )
        except Exception as err:  # noqa: BLE001 - anomaly signal must never break detection
            self._logger.debug("Terminal-drop detection skipped: %s", err)
            return False

    def _terminal_drop_baseline(self) -> tuple[float | None, tuple[float, float] | None]:
        """Cached (earliest-quiet-offset, historical-peak-range) for this device.

        Both are learned from the device's completed cycles and used by the
        terminal-drop detector (anomaly + familiarity gates).  Keyed by cycle
        count so it refreshes as history grows.

        The recompute decompresses every completed trace, which is too heavy to run
        on the event loop inside the detector's reading path (issue #311). So this
        NEVER recomputes synchronously: on a miss/stale cache it schedules an
        executor refresh and serves the last known baseline in the meantime (one
        cycle stale is harmless for an anomaly heuristic). Until the first refresh
        lands there is no baseline, so it returns ``(None, None)`` and
        ``is_terminal_drop`` defers to the proven slow end-detection."""
        cycles = self.profile_store.get_past_cycles()
        n = len(cycles)
        cache = self._terminal_drop_cache
        if cache is not None and cache[0] == n:
            return cache[1], cache[2]
        self._schedule_terminal_drop_refresh(n)
        if cache is not None:
            return cache[1], cache[2]
        return None, None

    def _schedule_terminal_drop_refresh(self, n: int) -> None:
        """Kick a one-shot executor refresh of the terminal-drop baseline for the
        current cycle count, unless one is already in-flight/done for it."""
        if self._terminal_drop_refresh_n == n:
            return
        self._terminal_drop_refresh_n = n
        self.hass.async_create_task(self._async_refresh_terminal_drop_baseline(n))

    async def _async_refresh_terminal_drop_baseline(self, n: int) -> None:
        """Recompute the baseline off the event loop and cache it. Never raises -
        the anomaly signal must never break detection."""
        try:
            # Snapshot the list on the loop before handing it to the executor.
            cycles = list(self.profile_store.get_past_cycles())
            stop_threshold = float(self.detector.config.stop_threshold_w)
            earliest, peak_range = await self.hass.async_add_executor_job(
                terminal_drop_baseline_for, cycles, stop_threshold
            )
            self._terminal_drop_cache = (len(cycles), earliest, peak_range)
        except Exception as err:  # noqa: BLE001 - anomaly signal must never break detection
            self._logger.debug("Terminal-drop baseline refresh failed: %s", err)
            # Allow a later reading to retry the refresh for this count.
            if self._terminal_drop_refresh_n == n:
                self._terminal_drop_refresh_n = None

    def _price_entity_reject_reason(self, entity_id: str) -> str | None:
        """Why ``entity_id`` cannot be a price per kWh, or None (#439).

        Only *positive* evidence rejects: an entity that has not loaded yet carries
        no attributes, and refusing it would silence a perfectly good tariff sensor
        that HA sets up after us.
        """
        options = self.config_entry.options
        if entity_id == self.power_sensor_entity_id:
            return "it is this device's power sensor"
        if entity_id == options.get(CONF_ENERGY_SENSOR):
            return "it is this device's energy meter"
        state = self.hass.states.get(entity_id)
        if state is None:
            return None
        device_class = str(state.attributes.get("device_class") or "").strip().lower()
        if device_class in _NON_PRICE_DEVICE_CLASSES:
            return f"its device class is '{device_class}'"
        unit = str(state.attributes.get("unit_of_measurement") or "").strip().lower()
        if unit in _NON_PRICE_UNITS:
            return f"its unit is '{unit}'"
        return None

    def _price_entity_id(self) -> str | None:
        """The configured price entity, or None when it is provably not a price.

        Guards the single trap the cost feature has (#439): the panel picker lists
        every sensor, a price entity outranks the static price, and pointing it at
        the plug's own kWh counter silently charges every cycle
        ``energy * meter_reading`` instead of ``energy * tariff``. Rejecting it here
        - rather than in the panel alone - also repairs entries that are already
        misconfigured, which fall back to the static price.
        """
        entity_id = self.config_entry.options.get(CONF_ENERGY_PRICE_ENTITY)
        if not entity_id:
            return None
        reason = self._price_entity_reject_reason(entity_id)
        if reason is None:
            self._warned_price_entity = None
            return entity_id
        if self._warned_price_entity != entity_id:
            self._warned_price_entity = entity_id
            self._logger.warning(
                "Energy price entity %s is not a price per kWh (%s); ignoring it and "
                "using the static energy price instead. Set a tariff sensor there, or "
                "clear the field to cost cycles at the static price",
                entity_id,
                reason,
            )
        return None

    def _resolve_energy_price(self) -> float | None:
        """Current energy price per kWh, or None when none is configured.

        A price entity (e.g. a dynamic tariff) takes precedence over the static
        value. Used to freeze each cycle's cost at completion time.
        """
        options = self.config_entry.options
        price_entity = self._price_entity_id()
        if price_entity:
            state = self.hass.states.get(price_entity)
            if state is not None:
                try:
                    value = float(state.state)
                except (ValueError, TypeError, OverflowError):
                    pass
                else:
                    # A non-finite reading is treated as no reading, exactly like an
                    # unparseable one: returning it would freeze an infinite cost
                    # onto the cycle (register item 211).
                    if math.isfinite(value):
                        return value
        static = options.get(CONF_ENERGY_PRICE_STATIC)
        if static is not None:
            try:
                value = float(static)
            except (ValueError, TypeError, OverflowError):
                pass
            else:
                if math.isfinite(value):
                    return value
        return None

    async def _async_price_history(
        self, start: datetime, end: datetime
    ) -> list[tuple[float, float]]:
        """``(unix_ts, price)`` rows for the price entity over a window, via the
        recorder (#426).

        Used to recover price changes the live listener could not see - the period
        HA was down mid-cycle, a cycle that predates the feature, a device whose
        history was imported from raw recorder data. Returns ``[]`` on any failure
        (recorder disabled, entity excluded from recording, data purged) so the
        caller falls back to whatever it already had.
        """
        entity_id = self._price_entity_id()
        if not entity_id:
            return []
        try:
            from homeassistant.components.recorder import (  # noqa: PLC0415
                get_instance,
                history,
            )
        except Exception:  # noqa: BLE001 - recorder is an optional component
            return []

        def _query() -> list[tuple[float, float]]:
            res = history.state_changes_during_period(
                self.hass, start, end, entity_id, include_start_time_state=True
            )
            start_ts = start.timestamp()
            rows: list[tuple[float, float]] = []
            for state in res.get(entity_id, []) or []:
                try:
                    price = float(state.state)
                except (ValueError, TypeError, OverflowError):
                    # unknown/unavailable: the previous price stays in force.
                    continue
                ts = state.last_changed.timestamp()
                # include_start_time_state hands back the state in force at the
                # window start, whose last_changed can predate it by hours. Clamp
                # so it anchors the timeline instead of sorting before the cycle.
                rows.append((max(ts, start_ts), price))
            rows.sort(key=lambda item: item[0])
            return rows

        try:
            return await get_instance(self.hass).async_add_executor_job(_query)
        except Exception as exc:  # noqa: BLE001 - cost must never break cycle end
            self._logger.debug("Price history lookup failed for %s: %s", entity_id, exc)
            return []

    async def _async_apply_cycle_cost(
        self,
        cycle_data: dict[str, Any],
        price_timeline: list[tuple[float, float]] | None = None,
    ) -> None:
        """Freeze ``cost`` / ``energy_price`` onto a finished cycle (#426).

        Dynamic mode integrates the stored power trace against the price timeline
        recorded while the cycle ran; ``energy_price`` then carries the *effective*
        price per kWh the cycle paid (cost / kWh), which is the only figure that
        stays meaningful once the tariff moved. ``energy_price_mode`` says which of
        the two produced the number so the panel can label it honestly.

        Falls back to the single current price whenever dynamic costing cannot
        produce an answer - no price entity, the toggle off, no timeline, or a
        trace with no energy in it. Never raises: a cost figure is display-only and
        must not be able to lose a finished cycle.
        """
        price = self._resolve_energy_price()
        if self._dynamic_pricing_enabled():
            try:
                if await self._async_apply_dynamic_cost(
                    cycle_data, price_timeline=price_timeline
                ):
                    return
            except Exception as exc:  # noqa: BLE001 - never break cycle storage
                self._logger.debug("Dynamic cost calculation failed: %s", exc)
        if price is not None:
            cycle_data["energy_price"] = price
            cycle_data["energy_price_mode"] = "fixed"
            cycle_data["cost"] = round(
                self._cycle_report_energy_wh(cycle_data) / 1000.0 * price, 4
            )

    async def _async_apply_dynamic_cost(
        self,
        cycle_data: dict[str, Any],
        price_timeline: list[tuple[float, float]] | None = None,
    ) -> bool:
        """Cost the cycle against its price timeline. True when it succeeded.

        ``price_timeline`` is the finished cycle's own history, frozen at cycle end.
        Falling back to the live ``_price_timeline`` is only correct while no new
        cycle has started since.
        """
        start_dt = dt_util.parse_datetime(str(cycle_data.get("start_time") or ""))
        end_dt = dt_util.parse_datetime(str(cycle_data.get("end_time") or ""))
        if start_dt is None or end_dt is None or end_dt <= start_dt:
            return False

        timeline = list(
            self._price_timeline if price_timeline is None else price_timeline
        )
        # The live listener is complete whenever HA stayed up for the whole cycle.
        # Consult the recorder only when it cannot have been: nothing recorded at
        # all (the feature was switched on mid-cycle), or a restart gap where price
        # changes would have gone unseen.
        if not timeline or cycle_data.get("restart_gaps"):
            recorded = await self._async_price_history(start_dt, end_dt)
            if recorded:
                # The recorder saw the downtime too, so it supersedes rather than
                # merges - interleaving the two would double-count a change that
                # both captured at slightly different timestamps.
                timeline = recorded

        if not timeline:
            return False

        start_ts = start_dt.timestamp()
        duration = (end_dt - start_dt).total_seconds()
        points = compact_price_timeline(
            [(ts - start_ts, price) for ts, price in timeline],
            max_points=PRICE_TIMELINE_MAX_POINTS,
            decimals=PRICE_TIMELINE_PRICE_DECIMALS,
        )
        # A price recorded before the trace's first sample still sets the opening
        # price; clamp rather than drop it, and discard anything past the end.
        points = [(max(0.0, offset), price) for offset, price in points if offset <= duration]
        points = compact_price_timeline(
            points, max_points=PRICE_TIMELINE_MAX_POINTS,
            decimals=PRICE_TIMELINE_PRICE_DECIMALS,
        )
        if not points:
            return False

        result = self._cost_from_timeline(cycle_data, points)
        if result is None:
            return False
        cost, effective_price = result
        cycle_data["cost"] = round(cost, 4)
        cycle_data["energy_price"] = round(effective_price, 6)
        cycle_data["energy_price_mode"] = "dynamic"
        cycle_data["price_timeline"] = [
            [round(offset, 1), price] for offset, price in points
        ]
        return True

    def _cost_from_timeline(
        self, cycle_data: dict[str, Any], points: list[tuple[float, float]]
    ) -> tuple[float, float] | None:
        """``(cost, effective_price)`` for a cycle and its price timeline, or None.

        Pure apart from reading the cycle; the meter-vs-integrated decision is the
        same ``_cycle_report_energy_wh`` every other user-facing energy figure uses,
        so the cost and the kWh shown beside it are computed from one number.
        """
        # decompress_power_data, not the raw list: a cycle stored before the
        # offset migration still carries ISO timestamps, and recosting must work
        # on exactly the cycles that are old enough to need it.
        points_xy = decompress_power_data(cast(Any, cycle_data))
        if len(points_xy) < 2:
            return None
        timestamps = np.asarray([t for t, _ in points_xy], dtype=float)
        power = np.asarray([p for _, p in points_xy], dtype=float)
        return cycle_cost(
            timestamps,
            power,
            points,
            max_gap_s=energy_gap_threshold_s(timestamps),
            report_wh=self._cycle_report_energy_wh(cycle_data),
        )

    def _read_energy_meter(self) -> tuple[float, str] | None:
        """Read the configured external energy meter, normalized to Wh.

        Returns ``(value_wh, entity_id)`` or ``None`` when no meter is configured
        or its reading is not usable (unknown/unavailable/non-numeric state, or an
        unrecognised unit). Never raises -- every failure path returns ``None`` so
        the caller falls back to the integrated energy (issue #316).
        """
        entity_id = self.config_entry.options.get(CONF_ENERGY_SENSOR)
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in (None, "unknown", "unavailable", ""):
            return None
        try:
            value = float(state.state)
        except (ValueError, TypeError, OverflowError):
            return None
        unit = str(state.attributes.get("unit_of_measurement") or "").strip().lower()
        # Normalize to Wh. An unrecognised unit is treated as unusable so a
        # mis-configured entity falls back rather than reporting a wrong figure.
        scale = {"wh": 1.0, "kwh": 1000.0, "mwh": 1_000_000.0}.get(unit)
        if scale is None:
            return None
        return value * scale, entity_id

    def _snapshot_energy_meter_start(self) -> None:
        """Capture the meter reading at cycle start (issue #316)."""
        snap = self._read_energy_meter()
        if snap is None:
            self._energy_meter_start = None
            self._energy_meter_source = None
        else:
            self._energy_meter_start, self._energy_meter_source = snap

    def _compute_meter_energy_wh(self) -> float | None:
        """Cycle energy from the external meter's start->end delta, or None.

        Falls back (returns ``None``) when: no start was captured; no meter is
        configured now; the configured entity differs from the one snapshotted at
        start (source changed mid-cycle -- no cross-meter delta); the current
        reading is unusable; or the delta is <= 0 (counter reset or stuck plug).
        """
        start = self._energy_meter_start
        source = self._energy_meter_source
        if start is None or source is None:
            return None
        cur = self._read_energy_meter()
        if cur is None:
            return None
        cur_wh, cur_source = cur
        if cur_source != source:
            return None
        delta = cur_wh - start
        if delta <= 0:
            return None
        return delta

    @staticmethod
    def _cycle_report_energy_wh(cycle_data: dict[str, Any]) -> float:
        """User-facing reported energy (Wh): meter value if present, else integrated.

        The integrated ``energy_wh`` is always stored and used internally (matching,
        ML, anomaly, envelopes); only cost/lifetime/notifications/panel display
        prefer the more accurate meter figure when one was captured (issue #316).
        """
        meter = cycle_data.get("energy_meter_wh")
        if meter is not None:
            try:
                return float(meter)
            except (ValueError, TypeError, OverflowError):
                pass
        try:
            return float(cycle_data.get("energy_wh", 0.0))
        except (ValueError, TypeError, OverflowError):
            return 0.0

    def _in_anticrease_tail(self) -> bool:
        """The detector sits in a #296 anti-crease tail (register item 393a).

        Saved at stop and unload like an active cycle: a restart that starts from
        OFF reads the tail's next drum bursts as a new cycle (one ~20 min cycle on
        the item-393 shape). The cycle end has already cleared the active slot, and
        the restore consumes it again, so a stale tail cannot come back later.
        """
        return (
            self.detector.state == STATE_ANTI_WRINKLE
            and getattr(self.detector, "in_anticrease_tail", False) is True
        )

    def _augment_active_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Add manager-owned fields to a detector snapshot before persisting.

        The detector snapshot only carries detector state; these fields are owned
        by the manager and must survive a restart alongside it. Kept in one place
        so every save site (shutdown, periodic, pause, resume) stays consistent.
        """
        snapshot["manual_program"] = self._manual_program_active
        # Persist the chosen program name too (#404 secondary bug): the detector
        # snapshot's matched_profile is wiped on the first post-restart match tick
        # because _matched_profile_duration is not restored, so the name must be
        # carried explicitly to re-pin the override.
        snapshot["manual_program_name"] = (
            self._current_program if self._manual_program_active else None
        )
        # The auto-detected program on display and its expected duration, for the
        # same reason: the detector holds only the last tick's raw winner.
        committed = not self._manual_program_active and match_rules.program_is_committed(
            self._current_program
        )
        snapshot["committed_program"] = self._current_program if committed else None
        snapshot["committed_program_duration"] = (
            self._matched_profile_duration if committed else None
        )
        snapshot["notified_start"] = self._notified_start
        snapshot["start_event_fired"] = self._start_event_fired
        snapshot["is_user_paused"] = self._is_user_paused
        snapshot["user_pause_start"] = (
            self._user_pause_start.isoformat() if self._user_pause_start else None
        )
        snapshot["total_user_paused_seconds"] = self._total_user_paused_seconds
        snapshot["energy_meter_start"] = self._energy_meter_start
        snapshot["energy_meter_source"] = self._energy_meter_source
        # Dynamic price timeline (#426): absolute (unix_ts, price) pairs, as lists
        # so the JSON round-trip is lossless.
        snapshot["price_timeline"] = [
            [ts, price] for ts, price in self._price_timeline
        ]
        # One-shot per-cycle state (audit MANAGER-08): without it a restart re-fired
        # every passed cycle timer ("Add softener" twice; an auto_pause timer paused
        # again and, with pause_cuts_power, switched the appliance off), could send
        # the pre-completion reminder twice, and a second restart lost the first gap.
        snapshot["fired_cycle_timers"] = sorted(self._fired_cycle_timers)
        snapshot["notified_pre_completion"] = bool(self._notified_pre_completion)
        snapshot["restart_gaps"] = list(self._restart_gaps)
        snapshot["live_activity_started"] = bool(self._live_activity_started)
        # When the power sensor last really reported, and the value the manager
        # held for it (register item 266). Without them a restart took the
        # entity's own startup write for a report, so the watchdog's silence
        # clock restarted at zero and a cycle in a long silent tail looked as if
        # its plug had just spoken.
        snapshot["last_real_reading_time"] = (
            self._last_real_reading_time.isoformat()
            if self._last_real_reading_time is not None
            else None
        )
        snapshot["last_sensor_power"] = self._current_power
        return snapshot

    @staticmethod
    def _format_duration_hm(minutes: Any) -> str:
        """The ``{duration_hm}`` template variable: ``"1 h 05 min"``, ``"45 min"``.

        Unit symbols a voice assistant reads correctly (#93, #117: it read
        ``{duration}m`` as metres). Whole minutes in; never raises.
        """
        try:
            total = max(0, int(minutes))
        except (TypeError, ValueError, OverflowError):
            return ""
        hours, mins = divmod(total, 60)
        return f"{hours} h {mins:02d} min" if hours else f"{mins} min"

    @staticmethod
    def _format_vs_typical(
        duration: float,
        median: float | None,
        *,
        longer_template: str = "{pct}% longer than usual",
        shorter_template: str = "{pct}% shorter than usual",
    ) -> str:
        """Human comparison of a cycle's duration to its profile median.

        Returns "" when there is no usable median or the difference is under 1%.
        This fills the ``vs_typical`` variable of the finish-message template. The
        text itself is fixed (not user-editable), so it is localizable: callers pass
        the resolved ``options.error.vs_typical_*`` templates; the English defaults
        here mirror strings.json and are used as the resilient fallback.
        """
        try:
            if not median or float(median) <= 0:
                return ""
            pct = round((float(duration) - float(median)) / float(median) * 100)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return ""
        try:
            if pct >= 1:
                return longer_template.format(pct=pct)
            if pct <= -1:
                return shorter_template.format(pct=abs(pct))
        except (KeyError, IndexError, ValueError, OverflowError):
            # Malformed translation template; fall back to the English default.
            if pct >= 1:
                return f"{pct}% longer than usual"
            if pct <= -1:
                return f"{abs(pct)}% shorter than usual"
        return ""

    def _peak_rate_tip(self, options: dict[str, Any], price: float | None) -> str:
        """Return a peak-rate advisory tip for the start notification, or "".

        Appended only when a positive ``peak_rate_threshold`` is configured and the
        current price meets/exceeds it. Purely informational — no scheduling or
        appliance control. Never raises; a bad threshold is skipped silently.
        """
        try:
            raw = options.get(CONF_PEAK_RATE_THRESHOLD)
            if raw in (None, ""):
                return ""
            threshold = float(raw)
            if threshold <= 0 or price is None or float(price) < threshold:
                return ""
            tip_template = options.get(CONF_PEAK_RATE_MESSAGE) or DEFAULT_PEAK_RATE_MESSAGE
            return self._safe_format_template(
                tip_template,
                fallback_template=DEFAULT_PEAK_RATE_MESSAGE,
                device=self.config_entry.title,
                price=f"{float(price):.3f}",
            )
        except (ValueError, TypeError, OverflowError):
            return ""

    async def _async_process_cycle_end(
        self,
        cycle_data: dict[str, Any],
        cycle_token: str | None = None,
        price_timeline: list[tuple[float, float]] | None = None,
    ) -> None:
        """Process cycle completion, then close it whatever failed (MANAGER-12).

        An exception anywhere in the steps used to end the task before the flush
        and the terminal reset: the UI stayed on the finished cycle with no expiry
        timer. On shutdown the task is cancelled and ``async_shutdown`` flushes the
        store itself; arming the expiry timer then would outlive the unload.
        """
        failed = False
        try:
            await self._async_cycle_end_steps(cycle_data, cycle_token, price_timeline)
        except Exception:  # pylint: disable=broad-exception-caught
            failed = True
            self._logger.exception("Cycle-end processing failed; closing the cycle anyway")
        finally:
            if not self._is_shutdown:
                await self._async_close_cycle_end(
                    cycle_token, failed, cycle_status=cycle_data.get("status")
                )

    async def _async_cycle_end_steps(
        self,
        cycle_data: dict[str, Any],
        cycle_token: str | None = None,
        price_timeline: list[tuple[float, float]] | None = None,
    ) -> None:
        """Process cycle completion asynchronously (heavy tasks).

        ``cycle_token`` is the ``_ranking_snapshot_cycle_id`` captured when this cycle
        ended. The terminal-state reset at the tail is skipped if a new cycle has
        started since (token changed), so back-to-back cycles are not clobbered (B1).
        ``price_timeline`` is that same cycle's tariff history, frozen at the same
        moment and for the same reason; None means "read the live one".
        """

        # B1: freeze THIS cycle's live context into immutable locals BEFORE the first
        # await. A new cycle can start synchronously during any await below (a
        # back-to-back load drives the detector into a fresh RUNNING via
        # _on_state_change, which rolls _current_program back to "detecting..." and
        # resets the match fields). The tail (event payload, finish notification,
        # learning inputs) must describe the cycle that just finished, not whatever
        # the live fields hold by the time each await returns.
        program = self._current_program
        live_result = self._last_match_result
        match_confidence = self._last_match_confidence
        member_confidence = self._last_member_confidence
        matched_profile_duration = self._matched_profile_duration
        manual_program = self._manual_program_active
        cycle_anomaly = self._cycle_anomaly
        overrun_ratio = self._overrun_ratio

        # ONE match over the complete trace; every label decision below reads it
        # (audit MATCH-DECIDE-02). The last live tick is a PREFIX match (prefix
        # shapes, the in-progress duration kernel, up to profile_match_interval
        # stale, the ENDING quiet tail inside its duration), and its winner differed
        # from the complete-cycle winner on 17.5% of corpus cycles, while
        # MATCH_LABEL_MIN_MARGIN was calibrated on complete folds (item 310).
        # It runs BEFORE the cycle is stored, so a raise here used to lose the
        # cycle and strand the device on it (audit MATCH-CORE-06 / MANAGER-12). A
        # failed match leaves an empty result: the cycle is stored unlabelled, not
        # labelled from the live prefix match instead.
        try:
            final_result = await self._run_final_match_from_cycle_data(cycle_data)
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.exception(
                "Final match failed; storing the cycle without a label"
            )
            final_result = MatchResult(None, 0.0, 0.0, None, [], False, 0.0)
        same_cycle = cycle_token is None or self._ranking_snapshot_cycle_id == cycle_token
        if program in ("detecting...", "restored...") and final_result is not None:
            # Never committed live: the complete match names the program for DISPLAY
            # at a low floor, since the trace is complete. Labelling is decided below.
            if final_result.best_profile and final_result.confidence >= 0.15:
                program = final_result.best_profile
                match_confidence = final_result.confidence
                member_confidence = final_result.member_confidence
                if same_cycle:
                    self._current_program = program
                    self._last_match_result = final_result
                    self._last_match_confidence = match_confidence
                    self._last_member_confidence = member_confidence
                self._logger.info(
                    "Final match from cycle data: '%s' with confidence %.3f",
                    program, match_confidence,
                )
            else:
                self._logger.info(
                    "No confident match from cycle data (best: %s, conf=%.3f)",
                    final_result.best_profile, final_result.confidence,
                )
        match_result = final_result if final_result is not None else live_result

        # The number every label / persistence decision gates on: the member-aware
        # score (item 206), not a Stage-5 group's, which is its best SIBLING's.
        try:
            label_confidence = float(getattr(match_result, "label_confidence", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            label_confidence = 0.0
        if label_confidence > 0 and (
            not manual_program or getattr(match_result, "best_profile", None) == program
        ):
            # Recorded whether or not we label, so the panel can show what WashData
            # suspected without the cycle claiming it as its program. Not on a
            # hand-picked cycle the matcher would have called something else: the
            # number would read as confidence in the user's pick.
            cycle_data["match_confidence"] = label_confidence

        # A label is not a display value: it makes the cycle evidence for that
        # profile, moving avg_duration / target_duration, which arm Smart
        # Termination and the anti-crease finalize - so a weak guess recorded as fact
        # seeds the next mis-detection (#400). The verdict is the learning floor (the
        # panel's ladder: unmatch < match < learning < auto-label) plus the margin
        # and Stage-5 checks, on the complete match; the label is that match's own
        # winner. A hand-picked program bypasses it and is stamped "manual", since
        # "auto_match" is what lets bulk auto-labelling overwrite a label later.
        # The verdict also reaches learning.process_cycle_end, which must not
        # auto-label a cycle refused here (audit MANAGER-02 / MATCH-DECIDE-01).
        profiles = self.profile_store.get_profiles()
        learning_floor = float(self._learning_confidence or 0.0)
        label_gate_ok = False
        if manual_program and program and program in profiles:
            cycle_data["profile_name"] = program
            cycle_data["label_source"] = "manual"
            label_gate_ok = True
        else:
            # Shared with the Playground's would_label (match_rules).
            verdict, reason = match_rules.cycle_end_label_verdict(
                match_result, learning_floor, profiles
            )
            best = getattr(match_result, "best_profile", None)
            if verdict:
                cycle_data["profile_name"] = verdict
                cycle_data["label_source"] = "auto_match"
                label_gate_ok = True
                if verdict != program:
                    self._logger.info(
                        "Labelled cycle as '%s' (the complete-cycle winner, %.2f) "
                        "although '%s' was shown while it ran.",
                        verdict, label_confidence, program,
                    )
            elif reason == "below_floor":
                _group_conf = float(getattr(match_result, "confidence", 0.0) or 0.0)
                self._logger.info(
                    "Not labeling cycle as '%s': match confidence %.2f is below the "
                    "learning threshold %.2f.%s",
                    best, label_confidence, learning_floor,
                    f" (its profile group scored {_group_conf:.2f}, but that was a "
                    "different member of the group)"
                    if _group_conf > label_confidence + 1e-9 else "",
                )
            elif reason == "ambiguous":
                self._logger.info(
                    "Not labeling cycle as '%s': the matcher flagged its own pick as "
                    "uncertain (a profile-group member that fits poorly, or a run "
                    "past that member's length).",
                    best,
                )
            elif reason == "margin":
                self._logger.info(
                    "Not labeling cycle as '%s': confident enough (%.2f) but only "
                    "%.3f clear of the next candidate, under the %.2f a label needs.",
                    best, label_confidence,
                    float(getattr(match_result, "ambiguity_margin", 0.0) or 0.0),
                    MATCH_LABEL_MIN_MARGIN,
                )
            elif reason == "unknown_profile":
                self._logger.info(
                    "Not labeling cycle as '%s': that profile no longer exists.", best
                )

        # Attach extensive debug data if available (and configured). From the
        # complete match, so the stored ranking describes the finished cycle.
        if match_result:
            ranking = getattr(match_result, "ranking", [])
            # Top-5 ranking stored unconditionally (small, high training value).
            # SANITIZE: strip heavy current/sample arrays — this field is NOT in the
            # EVENT_CYCLE_ENDED exclusion set, so it must stay small (32KB limit).
            cycle_data["match_ranking_top5"] = _sanitize_ranking(ranking)
            cycle_data["debug_data"] = {
                "ranking": ranking,
                "details": getattr(match_result, "debug_details", {}),
                "ambiguous": getattr(match_result, "is_ambiguous", False),
            }

        # Compute envelope conformance for the matched profile.
        # Stored as cycle_data["envelope_conformance"] so the panel and quality
        # gate can display/use it.  Only runs when we have a profile + power trace.
        _ep = cycle_data.get("profile_name")
        _pd = cycle_data.get("power_data")
        _start_iso = cycle_data.get("start_time")
        if _ep and isinstance(_pd, list) and len(_pd) >= 4:
            try:
                from .time_utils import power_data_to_offsets  # noqa: PLC0415
                _pts = [(float(o), float(p)) for o, p in power_data_to_offsets(_pd, _start_iso)]
                if len(_pts) >= 4:
                    # Offloaded: both reach `analysis.align_trace_to_envelope`,
                    # which re-derives the envelope's DTW warp (item 324). The
                    # cost matrix is vectorised and bounded, but a bounded NumPy
                    # DTW is still CPU work and this runs on the event loop at
                    # every cycle end. The WS twin `expected_curve_for_cycle`
                    # already goes through the executor; these two did not.
                    # One job, not two: they share `_pts` and must describe the
                    # same alignment of the same cycle.
                    def _conformance_and_artifacts() -> tuple[Any, Any]:
                        return (
                            self.profile_store.compute_envelope_conformance(_ep, _pts),
                            self.profile_store.detect_cycle_artifacts(_ep, _pts),
                        )

                    conformance_rec, artifacts = await self.hass.async_add_executor_job(
                        _conformance_and_artifacts
                    )
                    if conformance_rec is not None:
                        cycle_data["envelope_conformance"] = conformance_rec.get("conformance")
                    # Transient artifacts (door-open pauses, out-of-band dips/spikes)
                    # for graph markers + a Cycles-list badge; [] when none.
                    if artifacts:
                        cycle_data["artifacts"] = artifacts
            except Exception:  # noqa: BLE001
                pass

        # Freeze the runtime overrun anomaly onto the cycle for panel badging.
        # "overrun" means the cycle ran materially longer than its matched
        # profile's typical duration; purely informational (never a notification).
        if cycle_anomaly and cycle_anomaly != "none":
            cycle_data["anomaly"] = cycle_anomaly
            if overrun_ratio > 0:
                cycle_data["overrun_ratio"] = round(float(overrun_ratio), 3)

        # A1 underrun + A2 energy spike/low (post-cycle only, never raises).
        _apply_post_cycle_anomalies(cycle_data, self.profile_store)

        # Cache post-cycle anomaly data so sensor attributes surface it while idle.
        self._last_cycle_post_anomaly = {
            k: cycle_data[k]
            for k in ("anomaly", "underrun_ratio", "energy_anomaly", "energy_z_score")
            if k in cycle_data
        }

        # Store any HA restart gaps that occurred during this cycle.
        # The panel shades these regions in the power trace and shows a badge.
        # Matching always uses real readings only (no synthetic fill in power_data).
        # Copy them onto the cycle now but keep the source list intact until the
        # cycle is confirmed persisted (below) — clearing here would lose them if
        # async_add_cycle() fails.
        restart_gaps_snapshot: list[dict[str, Any]] | None = None
        if self._restart_gaps:
            restart_gaps_snapshot = list(self._restart_gaps)
            cycle_data["restart_gaps"] = restart_gaps_snapshot

        # Freeze the energy cost onto the cycle. With a dynamic tariff this is the
        # power trace integrated against the price in force at each moment (#426);
        # otherwise the single price in effect NOW. Either way it is frozen here, so
        # later price changes never rewrite historical costs.
        await self._async_apply_cycle_cost(cycle_data, price_timeline=price_timeline)

        # Add cycle to store immediately (still sync but offloadable parts optimized
        # internally if possible)
        # Note: add_cycle is mostly safe (signature calc is O(N) but fast enough for
        # single cycle).
        # We could offload signature calc to analysis logic if really needed, but let's
        # stick to match profile optimization first.
        cycle_persisted = False
        # Read the odometer BEFORE the add: its getter floors at len(past_cycles), so
        # read afterwards it already counted this cycle and `+ 1` double-stepped - a
        # fresh install read 2/3/4 after 1/2/3 cycles and milestones fired one cycle
        # early (audit MANAGER-05).
        try:
            odometer_before_add: int | None = self._lifetime_cycle_count()
        except Exception:  # noqa: BLE001 - counter must never break cycle end
            odometer_before_add = None
        # Every save from here until the follow-up work has settled is debounced,
        # and what has to be durable is written once by async_flush_saves below:
        # this pipeline and the tasks it spawns used to rewrite the whole store six
        # times (register item 456). Re-armed here (_on_cycle_end opened it) so
        # the window runs from the add, however long the final match took.
        self.profile_store.coalesce_saves()
        # The envelopes this cycle end changed: the labelled profile's, and those of
        # any profile retention trimmed. Nothing else needs rebuilding (the nightly
        # maintenance still rebuilds them all).
        touched_profiles: list[str] = []
        try:
            retained = await self.profile_store.async_add_cycle(cycle_data)
            cycle_persisted = True
            # The cycle (with its restart_gaps) is now durably stored, so it is safe
            # to drop the live buffer. Doing this only after a confirmed persist means
            # a failed save keeps the gaps for the next cycle-end attempt.
            if restart_gaps_snapshot is not None:
                self._restart_gaps.clear()
            profile_name = cycle_data.get("profile_name")
            if profile_name:
                touched_profiles.append(profile_name)
                await self.profile_store.async_rebuild_envelope(profile_name)
            if isinstance(retained, (set, frozenset, list, tuple)):
                for name in sorted(p for p in retained if isinstance(p, str)):
                    if not name or name in touched_profiles:
                        continue
                    touched_profiles.append(name)
                    try:
                        await self.profile_store.async_rebuild_envelope(name)
                    except Exception:  # pylint: disable=broad-exception-caught
                        self._logger.debug(
                            "Envelope rebuild after retention failed for %s",
                            name, exc_info=True,
                        )
        except Exception as e: # pylint: disable=broad-exception-caught
            self._logger.error("Failed to add cycle to store: %s", e)

        # C2: bump the persisted lifetime cycle counter. Unlike ``cycle_count``
        # (== len(history), which regresses when history is trimmed/merged), this
        # monotonic counter only ever increments — and only on a real persisted
        # cycle — so milestones stay correct across retention limits. Captured here
        # for the milestone check below. The write is persisted by the lifetime-energy
        # save immediately after (same store, one save).
        prev_lifetime_count: int | None = None
        cur_lifetime_count: int | None = None
        if cycle_persisted and odometer_before_add is not None:
            try:
                prev_lifetime_count = odometer_before_add
                cur_lifetime_count = prev_lifetime_count + 1
                # In-memory only; persisted by the batched lifetime-energy save below.
                self.profile_store.set_lifetime_cycle_count(cur_lifetime_count)
            except Exception:  # noqa: BLE001 - counter must never break cycle end
                prev_lifetime_count = None
                cur_lifetime_count = None

        # B1: accumulate lifetime energy for the HA Energy dashboard sensor. Runs
        # exactly once per persisted cycle so the TOTAL_INCREASING meter never
        # double-counts. Never breaks cycle end.
        if cycle_persisted:
            try:
                await self.profile_store.async_add_lifetime_energy_wh(
                    self._cycle_report_energy_wh(cycle_data)
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                self._logger.debug("Failed to accumulate lifetime energy: %s", e)

        # Ensure cycle has a stable ID even if store add failed (or did not mutate).
        if not cycle_data.get("id"):
            try:
                unique_str = f"{cycle_data['start_time']}_{cycle_data['duration']}"
                cycle_data["id"] = hashlib.sha256(unique_str.encode()).hexdigest()[:12]
            except Exception:  # noqa: BLE001
                pass

        # B1: only clear the active-cycle snapshot if it still belongs to THIS cycle.
        # If a new cycle started during the awaits above, it now owns the active
        # snapshot; clearing it here would strip the new cycle's restart-resilience.
        if cycle_token is None or self._ranking_snapshot_cycle_id == cycle_token:
            self._spawn_tracked(self.profile_store.async_clear_active_cycle())

        # Refresh the artifacts that read the envelopes rebuilt above.
        self._spawn_tracked(self._run_post_cycle_processing(touched_profiles))

        # Prepare cycle data for event (enrich if needed)
        # IMPORTANT: Exclude large fields to prevent exceeding HA's 32KB event data limit
        excluded_fields = {
            "power_data", "debug_data", "power_trace",
            # A chatty dynamic tariff can add hundreds of entries (#426); the
            # cost and effective price it produced ride along instead.
            "price_timeline",
        }
        event_cycle_data = {
            k: v for k, v in cycle_data.items() if k not in excluded_fields
        }
        event_cycle_data["device_type"] = self.device_type
        # The program to SHOW: the stored label, else THIS cycle's captured live
        # program (not the live field, which may already belong to a newly-started
        # cycle). `_add_cycle_data` always writes `profile_name` (None when the
        # label gate refused), so the old "key missing" fill-in never ran and every
        # unlabelled cycle announced "Washer finished None" (audit MANAGER-06).
        # The stored cycle keeps profile_name None: this is display, not a label.
        display_program = event_cycle_data.get("profile_name")
        if not display_program and program and program not in (
            "off", "detecting...", "restored...", "starting", "unknown"
        ):
            display_program = program
        display_program = display_program or "unknown"
        # MATCH-DECIDE-15: how sure the complete-cycle match was, and whether the
        # stored cycle was labelled with `program` or it is only the best guess
        # shown for display. The margin is None with no winner (1.0 when only one
        # programme was a candidate); `label_applied` covers a hand-picked one.
        match_margin: float | None = None
        if getattr(match_result, "best_profile", None):
            try:
                match_margin = round(
                    float(getattr(match_result, "ambiguity_margin", 0.0) or 0.0), 3
                )
            except (TypeError, ValueError, OverflowError):
                match_margin = None

        if self._notify_fire_events:
            self.hass.bus.async_fire(
                EVENT_CYCLE_ENDED,
                {
                    "entry_id": self.entry_id,
                    "device_name": self.config_entry.title,
                    "cycle_data": event_cycle_data,
                    "program": display_program,
                    "match_margin": match_margin,
                    "label_applied": bool(label_gate_ok),
                    "duration": event_cycle_data.get("duration"),
                    "start_time": event_cycle_data.get("start_time"),
                    "end_time": event_cycle_data.get("end_time") or utc_now().isoformat(),
                },
            )

        # Purge pending live entries and reset counters. No service-level clear
        # here: the activity is ended below, AFTER the finished notification has
        # been delivered, so the lock screen is never momentarily empty. The
        # action-based clear marker still fires for action templates.
        # _clear_live_progress_notification resets _live_activity_started, so the
        # flag has to be read before it runs (#446).
        #
        # Gated on the cycle token, the same test the terminal-state reset below
        # uses. Everything here runs AFTER the persistence / envelope / cost /
        # lifetime-energy awaits, and a new cycle can start during them: its
        # `_on_state_change` calls `_reset_live_notification_state()` and its
        # first live tick sets `_live_activity_started` again. Ungated, this tail
        # then reads the NEW cycle's flag, purges the NEW cycle's live counters
        # and pending start entries, and - because `_live_notification_tag` is
        # per DEVICE, not per cycle - ends the activity the new cycle is running.
        # The user watches it vanish and its start card get cleared a second
        # time, and the next tick restarts it. If a newer cycle owns the tag,
        # leave the activity alone: it continues on the same tag.
        _same_cycle = (
            cycle_token is None or self._ranking_snapshot_cycle_id == cycle_token
        )
        live_activity_running = _same_cycle and self._live_activity_started
        if _same_cycle:
            self._clear_live_progress_notification(clear_services=False)

        # No "finished" push for an interrupted cycle (audit MANAGER-10): a false
        # start or a cancelled programme finished nothing. Nothing replaces the
        # start card on the lifecycle tag then, so clear it the way the shutdown
        # path does when no finished notification follows - unless a newer cycle
        # already owns that tag.
        cycle_status = cycle_data.get("status")
        announce_finish = notif_rules.cycle_end_is_finish(cycle_status)
        if not announce_finish and _same_cycle:
            self._send_tag_clear(self._lifecycle_tag)

        # Send notification if enabled
        if announce_finish and (self._notify_finish_services or self._notify_actions):
            msg_template = self.config_entry.options.get(CONF_NOTIFY_FINISH_MESSAGE, DEFAULT_NOTIFY_FINISH_MESSAGE)
            duration_min = int(cycle_data['duration'] / 60)
            program_name = display_program
            # `completed` or `force_stopped` (interrupted cycles never get here).
            status_str = str(cycle_status or "completed")

            energy_kwh = round(self._cycle_report_energy_wh(cycle_data) / 1000, 3)

            # Reuse the cost frozen onto the cycle above (same price resolution).
            cost_val = cycle_data.get("cost")
            cost_str = f"{cost_val:.2f}" if cost_val is not None else ""

            # B3: extra finish-notification template variables. All are safe to
            # ignore in a template — str.format drops unused kwargs.
            time_finished = dt_util.now().strftime("%H:%M")
            # Prefer the monotonic lifetime counter (falls back to the retained count).
            finished_cycle_count = (
                cur_lifetime_count if cur_lifetime_count is not None else self.cycle_count
            )
            vs_typical = ""
            matched_name = cycle_data.get("profile_name")
            if matched_name:
                _median = self.profile_store.get_profile_median_duration(matched_name)
                vs_typical = self._format_vs_typical(
                    cycle_data.get("duration", 0.0),
                    _median,
                    longer_template=self._timer_ui_strings.get(
                        "vs_typical_longer", "{pct}% longer than usual"
                    ),
                    shorter_template=self._timer_ui_strings.get(
                        "vs_typical_shorter", "{pct}% shorter than usual"
                    ),
                )

            msg = self._safe_format_template(
                msg_template,
                fallback_template=DEFAULT_NOTIFY_FINISH_MESSAGE,
                device=self.config_entry.title,
                duration=duration_min,
                duration_hm=self._format_duration_hm(duration_min),
                program=program_name,
                energy_kwh=f"{energy_kwh:.3f}",
                cost=cost_str,
                time_finished=time_finished,
                cycle_count=finished_cycle_count,
                vs_typical=vs_typical,
                status=status_str,
            )
            self._dispatch_notification(
                msg,
                event_type=NOTIFY_EVENT_FINISH,
                extra_vars={
                    "duration_minutes": duration_min,
                    "duration_seconds": cycle_data["duration"],
                    "program": program_name,
                    "energy_kwh": energy_kwh,
                    "cost": cost_str,
                    "time_finished": time_finished,
                    "cycle_count": finished_cycle_count,
                    "vs_typical": vs_typical,
                    "status": status_str,
                    # Same lifecycle tag as start/live so the finished alert replaces
                    # the live notification in place. No live_update/alert_once here,
                    # so the companion app surfaces it with sound.
                    "tag": self._lifecycle_tag,
                    # C3: retained, but it is NOT what ends the activity - there is
                    # no `activity` key in the companion notification API and this
                    # was never acted on (#446). Kept because it is inert and this
                    # code cannot be exercised against a real device here; the
                    # documented clear below is the mechanism that works.
                    "activity": "end",
                },
            )

        # #446: end the iOS Live Activity now that the finished alert has gone out.
        # Only when one was actually started, so a device that never ran an activity
        # gets no stray service call.
        if live_activity_running:
            self._end_live_activity()

        # C2: milestone (cycle-count achievement) notification. Fires at most once per
        # cycle, only when the cycle actually persisted (so the lifetime count is real)
        # and a finish delivery channel is configured. Respects quiet hours via
        # _dispatch_notification's finish-type gate.
        if cycle_persisted:
            self._maybe_notify_milestone(prev_lifetime_count, cur_lifetime_count)

        # Request user feedback if we had a confident match.
        # AND perform learning analysis on the completed cycle.
        # IMPORTANT: this must happen before we clear match state.
        # Only run when the cycle was actually persisted — an unpersisted cycle
        # has no store entry to reference, so a pending-feedback record would
        # dangle forever. Use THIS cycle's captured match context (not the live
        # fields, which may already belong to a newly-started cycle after the awaits).
        if cycle_persisted:
            # Ask about what the complete match picked, not the live tick's program.
            feedback_profile = (
                cycle_data.get("profile_name")
                or getattr(match_result, "best_profile", None)
                or program
            )
            feedback_duration = (
                matched_profile_duration
                if feedback_profile == program
                else (profiles.get(feedback_profile) or {}).get("avg_duration")
            )
            self.learning_manager.process_cycle_end(
                cycle_data,
                detected_profile=feedback_profile,
                confidence=label_confidence,
                predicted_duration=feedback_duration,
                match_result=match_result,
                label_allowed=label_gate_ok or bool(cycle_data.get("profile_name")),
            )

    async def _async_close_cycle_end(
        self,
        cycle_token: str | None,
        tail_failed: bool = False,
        cycle_status: str | None = None,
    ) -> None:
        """The end of the cycle-end tail, run whatever failed before it (MANAGER-12).

        Flushes the coalesced cycle-end write (item 456) and resets the terminal
        state, unless a newer cycle has started (B1). ``tail_failed`` means the
        follow-up stopped part-way, possibly before the live notification was
        handed over, so the live tag is cleared here instead of being left to
        count its chronometer into negative numbers. ``cycle_status`` is the
        ended cycle's status: an interrupted one never enters the Clean state
        (audit MANAGER-10, ``notification_rules.cycle_end_is_finish``).
        """
        # The one immediate write of this cycle end: the cycle, its counters, the
        # rebuilt envelopes and the feedback request the learning pass just queued.
        # No await since the lifetime-energy save that used to write first, so the
        # cycle is durable at the same point as before. Only what the follow-ups
        # derive (refreshed artifacts, suggestions) waits for the debounced write.
        try:
            await self.profile_store.async_flush_saves()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Failed to save the finished cycle: %s", e)
        # The idle display's standby level now includes this cycle (#452).
        await self._async_refresh_standby_level()

        if tail_failed and (
            cycle_token is None or self._ranking_snapshot_cycle_id == cycle_token
        ):
            try:
                self._clear_live_progress_notification()
            except Exception:  # noqa: BLE001 - cleanup must not stop the reset
                self._logger.debug("Clearing the live notification failed", exc_info=True)

        # B1: a new cycle may have started while the heavy post-processing above was
        # awaiting. If so, the manager's live-cycle fields (_current_program,
        # _cycle_start_time, _ranking_snapshot_cycle_id, progress) now belong to the
        # NEW cycle. Zeroing them here — and re-arming the state-expiry timer — used to
        # clobber the running cycle and, once it hit PAUSED/ENDING past the reset delay,
        # reset it to Off mid-run. Detect the new cycle via the identity token and skip
        # the terminal-state reset; cycle A was already persisted/learned/notified above.
        if cycle_token is not None and self._ranking_snapshot_cycle_id != cycle_token:
            self._logger.debug(
                "Cycle-end post-processing completed after a new cycle started "
                "(token %s -> %s); skipping terminal-state reset to preserve the "
                "live cycle.",
                cycle_token,
                self._ranking_snapshot_cycle_id,
            )
            self._notify_update()
            return

        # Clear all state and timers - zero everything out
        self._current_program = "off"
        self._manual_program_active = False
        # A pin is for the cycle it was made for, so it does not carry over (#411).
        self.clear_armed_program()
        self._notified_pre_completion = False
        self._time_remaining = None
        self._matched_profile_duration = None
        self._last_estimate_time = None
        self._last_match_result = None  # Clear so phase sensor resets to "Off" (issue #192)
        self._cycle_progress = 100.0  # 100% = cycle complete
        self._cycle_completed_time = utc_now()
        self._cycle_start_time = None
        self._ranking_snapshot_cycle_id = ""
        self._reset_live_notification_state()

        # Reset pause tracking for the next cycle
        self._is_user_paused = False
        self._user_pause_start = None
        self._total_user_paused_seconds = 0.0

        # Enter Clean state if door sensor is configured and door is currently closed
        self._is_clean_state = False
        self._clean_state_start = None
        self._notified_clean_laundry = False
        self._reset_unload_nag_tracking()
        if not notif_rules.cycle_end_is_finish(cycle_status):
            # An interrupted cycle finished nothing, so there is nothing to unload
            # and no reminder to nag with (audit MANAGER-10).
            self._logger.debug("Cycle ended %s: not entering Clean state", cycle_status)
        elif self._door_sensor_entity:
            door_state = self.hass.states.get(self._door_sensor_entity)
            if door_state and door_state.state == "off":  # binary_sensor: off = closed
                self._is_clean_state = True
                self._clean_state_start = utc_now()
                self._logger.debug(
                    "Cycle ended with door closed: entering Clean state"
                )
        elif self._unload_confirmable_without_door():
            # No door sensor, but the user opted into confirming the unload some
            # other way (#451): a button entity, or the Mark Unloaded button /
            # service driven by their own automation.
            self._is_clean_state = True
            self._clean_state_start = utc_now()
            self._logger.debug(
                "Cycle ended, unload confirmation configured: entering Clean state"
            )

        # Start progress reset timer to go back to 0% after user unload window
        self._start_state_expiry_timer()

        self._notify_update()

    @property
    def profile_sample_repair_stats(self) -> dict[str, int] | None:
        """Return statistics from profile sample repair operation."""
        return self._profile_sample_repair_stats

    @property
    def suggestions(self) -> dict[str, Any]:
        """Suggested settings computed by learning/heuristics (never auto-applied)."""
        return self.profile_store.get_suggestions()

    # ------------------------------------------------------------------
    # C1 - Quiet hours (do-not-disturb window)
    # ------------------------------------------------------------------
    def _quiet_hours_bounds(self) -> tuple[int, int] | None:
        """Return validated (start_hour, end_hour) or None when the feature is off.

        Off when either hour is unset/None/non-int/out-of-range, or start == end.
        """
        return notif_rules.quiet_hours_bounds(self.config_entry.options)

    def _in_quiet_hours(self, when: datetime | None = None) -> bool:
        """Return True when ``when`` (default now) falls inside the quiet window.

        Supports windows that wrap midnight (start > end, e.g. 22 -> 7 means
        22:00-06:59). The end hour is exclusive at the hour granularity, so a window
        of start=22, end=7 covers hours 22, 23, 0..6.
        """
        # Quiet hours are local clock hours; interval stamps in this module are
        # UTC (audit DETECT-01), so convert whatever the caller passes.
        return notif_rules.in_quiet_hours(
            self._quiet_hours_bounds(), dt_util.as_local(when or utc_now())
        )

    def _seconds_until_quiet_end(self, when: datetime | None = None) -> float:
        """Seconds from ``when`` until the next end-of-quiet-window boundary (end:00).

        Returns 0.0 when the feature is off or when not currently in quiet hours.
        """
        # Quiet hours are local clock hours; interval stamps in this module are
        # UTC (audit DETECT-01), so convert whatever the caller passes.
        return notif_rules.seconds_until_quiet_end(
            self._quiet_hours_bounds(), dt_util.as_local(when or utc_now())
        )

    def _queue_quiet_hours_notification(
        self,
        message: str,
        *,
        title: str | None,
        icon: str | None,
        event_type: str | None,
        extra_vars: dict[str, Any] | None,
    ) -> None:
        """Park a finish-type notification until the quiet window ends."""
        self._quiet_pending_notifications.append(
            {
                "message": message,
                "title": title,
                "icon": icon,
                "event_type": event_type,
                "extra_vars": extra_vars,
            }
        )
        self._schedule_quiet_hours_flush()

    def _schedule_quiet_hours_flush(self) -> None:
        """(Re)arm the single async_call_later timer that flushes the quiet queue."""
        if self._remove_quiet_hours_timer is not None:
            # A timer is already pending; keep it (all queued items share one release).
            return
        delay = self._seconds_until_quiet_end()
        if delay <= 0:
            # Not actually in quiet hours (defensive) -> flush immediately.
            self._flush_quiet_hours_notifications()
            return

        @callback
        def _fire(_now: datetime) -> None:
            self._remove_quiet_hours_timer = None
            self._flush_quiet_hours_notifications()

        self._remove_quiet_hours_timer = async_call_later(self.hass, delay, _fire)

    def _flush_quiet_hours_notifications(self) -> None:
        """Deliver every queued quiet-hours notification (same service/message/tag)."""
        if self._remove_quiet_hours_timer is not None:
            self._remove_quiet_hours_timer()
            self._remove_quiet_hours_timer = None
        if not self._quiet_pending_notifications:
            return
        pending = list(self._quiet_pending_notifications)
        self._quiet_pending_notifications = []
        for entry in pending:
            # Disable ONLY the quiet-hours re-hold (the window is closing), but keep
            # presence gating on: if nobody is home and notify_only_when_home is set,
            # the item must stay queued in the presence queue rather than fire into an
            # empty house. (Previously allow_deferral=False disabled both, delivering
            # to nobody.)
            self._dispatch_notification(
                entry["message"],
                title=entry.get("title"),
                icon=entry.get("icon"),
                event_type=entry.get("event_type"),
                extra_vars=entry.get("extra_vars"),
                allow_deferral=False,
                allow_presence_deferral=True,
            )

    def _cancel_quiet_hours_timer(self) -> None:
        """Cancel the pending quiet-hours release timer (shutdown/unload)."""
        if self._remove_quiet_hours_timer is not None:
            self._remove_quiet_hours_timer()
            self._remove_quiet_hours_timer = None

    # ------------------------------------------------------------------
    # Held notifications across a restart (audit MANAGER-16)
    # ------------------------------------------------------------------
    def _get_notify_queue_store(self) -> Store[dict[str, Any]]:
        if getattr(self, "_notify_queue_store", None) is None:
            self._notify_queue_store = Store(
                self.hass, 1, f"{STORAGE_KEY}.{self.entry_id}.{NOTIFY_QUEUE_STORE_SUFFIX}"
            )
        return self._notify_queue_store

    @staticmethod
    def _persistable_notifications(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The queued entries worth keeping, as JSON-safe copies."""
        out: list[dict[str, Any]] = []
        for entry in queue:
            if entry.get("event_type") in _NOTIFY_QUEUE_TRANSIENT_EVENTS:
                continue
            try:
                json.dumps(entry)
            except (TypeError, ValueError, OverflowError):
                continue
            out.append(dict(entry))
        return out

    async def _async_persist_notification_queues(self) -> None:
        """Write the quiet-hours and presence queues to disk. Never raises.

        Touches storage only when there is something to keep, or a file this
        manager wrote earlier is now stale.
        """
        try:
            quiet = self._persistable_notifications(
                getattr(self, "_quiet_pending_notifications", None) or []
            )
            presence = self._persistable_notifications(
                getattr(self, "_pending_notifications", None) or []
            )
            if quiet or presence:
                await self._get_notify_queue_store().async_save(
                    {
                        "saved_at": utc_now().isoformat(),
                        "quiet": quiet,
                        "presence": presence,
                    }
                )
                self._notify_queue_on_disk = True
                self._logger.info(
                    "Kept %d held notification(s) for after the restart",
                    len(quiet) + len(presence),
                )
            elif getattr(self, "_notify_queue_on_disk", False):
                await self._get_notify_queue_store().async_remove()
                self._notify_queue_on_disk = False
        except Exception:  # noqa: BLE001 - a stop or unload must not fail on this
            self._logger.debug("Could not persist held notifications", exc_info=True)

    async def _async_on_ha_stop(self, _event: Event) -> None:
        """Persist what lives only in memory before Home Assistant stops.

        HA does not unload config entries on a stop, so ``async_shutdown`` never runs
        on a restart: the held notifications were lost and the active-cycle snapshot
        was up to a minute old. Nothing is sent from here.
        """
        if self._is_shutdown:
            return
        try:
            if self.detector.state in {
                STATE_RUNNING, STATE_PAUSED, STATE_STARTING, STATE_ENDING
            } or self._in_anticrease_tail():
                snapshot = self._augment_active_snapshot(
                    self.detector.get_state_snapshot()
                )
                await self.profile_store.async_save_active_cycle(snapshot)
        except Exception:  # noqa: BLE001
            self._logger.debug("Could not save the active cycle at stop", exc_info=True)
        await self._async_persist_notification_queues()

    @callback
    def _schedule_notify_queue_restore(self, _hass: HomeAssistant) -> None:
        """Restore the held notifications once HA has started (notify services exist)."""
        self._remove_notify_queue_restore = None
        if not self._is_shutdown:
            self._spawn_tracked(self._async_restore_notification_queues())

    async def _async_restore_notification_queues(self) -> None:
        """Re-dispatch the notifications held when HA last stopped. Never raises.

        Each one goes back through the normal gates, so it is held again if quiet
        hours are still on or nobody is home, and delivered otherwise. The file is
        deleted on read, so a later restart cannot deliver it twice.
        """
        try:
            store = self._get_notify_queue_store()
            data = await store.async_load()
            if data is None:
                return
            await store.async_remove()
        except Exception:  # noqa: BLE001
            self._logger.debug("Could not restore held notifications", exc_info=True)
            return
        if self._is_shutdown or not isinstance(data, dict):
            return
        saved_at = dt_util.parse_datetime(str(data.get("saved_at") or ""))
        if (
            saved_at is None
            or (utc_now() - dt_util.as_utc(saved_at)).total_seconds()
            > _NOTIFY_QUEUE_MAX_AGE_S
        ):
            self._logger.info("Dropped held notifications saved at %s: too old", saved_at)
            return
        in_progress = self.detector.state in _CYCLE_IN_PROGRESS_STATES
        restored = 0
        for key in ("quiet", "presence"):
            entries = data.get(key)
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict) or not isinstance(entry.get("message"), str):
                    continue
                event_type = entry.get("event_type")
                if event_type in _NOTIFY_QUEUE_TRANSIENT_EVENTS:
                    continue
                if event_type in _NOTIFY_QUEUE_CYCLE_EVENTS and not in_progress:
                    continue
                extra = entry.get("extra_vars")
                self._dispatch_notification(
                    entry["message"],
                    title=entry.get("title"),
                    icon=entry.get("icon"),
                    event_type=event_type,
                    extra_vars=extra if isinstance(extra, dict) else None,
                )
                restored += 1
        if restored:
            self._logger.info("Restored %d held notification(s) after restart", restored)

    # ------------------------------------------------------------------
    # C2 - Milestone (cycle-count achievement) notifications
    # ------------------------------------------------------------------
    @staticmethod
    def _milestone_crossed(
        prev_count: int, cur_count: int, milestones: Any
    ) -> int | None:
        """Return the milestone just crossed, or None.

        A milestone ``m`` is crossed when ``prev_count < m <= cur_count``. Empty or
        malformed ``milestones`` is a no-op (returns None). If several are crossed in
        one step the largest is returned so a single, most-significant notification
        fires.
        """
        return notif_rules.milestone_crossed(prev_count, cur_count, milestones)

    def _lifetime_cycle_count(self) -> int:
        """Persisted monotonic lifetime completed-cycle count.

        Unlike ``cycle_count`` (== len(retained history)), this only ever increments
        and never regresses when history is trimmed/merged, so it is the correct basis
        for milestone crossings. Falls back to ``cycle_count`` if the persisted value
        is unavailable. Never raises.
        """
        try:
            return self.profile_store.get_lifetime_cycle_count()
        except Exception:  # noqa: BLE001
            try:
                return int(self.cycle_count)
            except Exception:  # noqa: BLE001
                return 0

    def _maybe_notify_milestone(
        self, prev_count: int | None = None, cur_count: int | None = None
    ) -> int | None:
        """Fire one milestone notification if the lifetime count just crossed one.

        Called at cycle end AFTER the cycle has persisted. ``prev_count``/``cur_count``
        are the persisted lifetime counter's values from before/after this cycle's
        persist; when omitted they are resolved from the persisted counter
        (previous = current - 1). Using the monotonic lifetime counter (not
        ``cycle_count`` == len(history)) keeps milestones correct across retention
        trims/merges. Returns the crossed milestone value (for tests/logging) or None.
        Never raises.
        """
        try:
            if not (self._notify_finish_services or self._notify_actions):
                return None
            milestones = self.config_entry.options.get(
                CONF_NOTIFY_MILESTONES, DEFAULT_NOTIFY_MILESTONES
            )
            if cur_count is None:
                cur_count = self._lifetime_cycle_count()
            if prev_count is None:
                prev_count = cur_count - 1
            crossed = self._milestone_crossed(prev_count, cur_count, milestones)
            if crossed is None:
                return None
            msg_template = self.config_entry.options.get(
                CONF_NOTIFY_MILESTONE_MESSAGE, DEFAULT_NOTIFY_MILESTONE_MESSAGE
            )
            msg = self._safe_format_template(
                msg_template,
                fallback_template=DEFAULT_NOTIFY_MILESTONE_MESSAGE,
                device=self.config_entry.title,
                cycle_count=crossed,
            )
            self._dispatch_notification(
                msg,
                event_type=NOTIFY_EVENT_FINISH,
                extra_vars={
                    "cycle_count": crossed,
                    # Distinct tag so a milestone alert does not clobber (or get
                    # clobbered by) the lifecycle finish thread.
                    "tag": f"{self._lifecycle_tag}_milestone",
                },
            )
            self._logger.info(
                "Sent milestone notification: %s cycles", crossed
            )
            return crossed
        except Exception as err:  # pylint: disable=broad-exception-caught
            self._logger.debug("Milestone notification check failed: %s", err)
            return None

    # ------------------------------------------------------------------
    # C3 - iOS Live Activity enrichment (HA Companion beta, mobile_app_* only)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_ios_live_activity_extras(
        *,
        state: str,
        progress_pct: float,
        eta_timestamp: Any,
        program: str | None,
        device: str,
        activity: str | None = None,
    ) -> dict[str, Any]:
        """Build the iOS Live Activity payload additions (mobile-only keys).

        Returns a dict containing ``content_state`` (always), ``subtitle`` (only when
        a program is matched) and ``activity`` (only when a lifecycle marker is
        supplied). These keys are forwarded to mobile_app_* targets only by
        ``_send_notification_service``; other platforms never receive them.
        """
        try:
            pct = int(round(float(progress_pct)))
        except (TypeError, ValueError, OverflowError):
            pct = 0
        pct = max(0, min(100, pct))
        extras: dict[str, Any] = {
            "content_state": {
                "state": state,
                "progress_pct": pct,
                "eta_timestamp": eta_timestamp,
                "program": program or "",
                "device": device,
            }
        }
        if program:
            extras["subtitle"] = program
        if activity:
            extras["activity"] = activity
        return extras

    @staticmethod
    def _mobile_service_extras(
        ev: dict[str, Any], notify_service: str | None
    ) -> dict[str, Any]:
        """Return extra_vars keys allowed only on mobile_app_* targets.

        For non-mobile services this is always empty, so strict-schema platforms and
        the iOS Live Activity enrichment keys stay isolated to mobile targets.
        """
        if not WashDataManager._is_mobile_notify_service(notify_service):
            return {}
        return {k: ev[k] for k in _MOBILE_ONLY_EXTRA_KEYS if k in ev}

    def _safe_format_template(
        self,
        template: Any,
        *,
        fallback_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Format templates safely and return a resilient fallback on any error."""
        text_template = str(template)
        try:
            return text_template.format(**kwargs)
        except Exception as err:  # pylint: disable=broad-exception-caught
            self._logger.debug(
                "Failed to format notification template %r with %s: %s",
                text_template,
                kwargs,
                err,
            )

        if fallback_template:
            try:
                return fallback_template.format(**kwargs)
            except Exception as err:  # pylint: disable=broad-exception-caught
                self._logger.debug(
                    "Failed to format fallback notification template %r with %s: %s",
                    fallback_template,
                    kwargs,
                    err,
                )

        device = str(kwargs.get("device") or self.config_entry.title)
        program = kwargs.get("program")
        if program:
            return f"{device}: {program}"
        return device

    def _get_services_for_event(self, event_type: str | None) -> list[str]:
        """Return the configured notify service list for the given event type."""
        if event_type == NOTIFY_EVENT_START:
            return self._notify_start_services
        if event_type in (NOTIFY_EVENT_FINISH, "pre_complete", NOTIFY_EVENT_CLEAN):
            return self._notify_finish_services
        if event_type == NOTIFY_EVENT_LIVE:
            return self._notify_live_services
        if event_type == NOTIFY_EVENT_TIMER:
            # Cycle timers go to all configured services (start union finish, deduped).
            return list(dict.fromkeys(
                self._notify_start_services + self._notify_finish_services
            ))
        return []

    def _resolve_channel(self, event_type: str | None) -> str | None:
        """Resolve the Android notification channel name for an event type.

        Finished, the clean-laundry nag, and the pre-completion reminder route to the
        dedicated finish channel (so they can carry their own sound), falling back to
        the status channel. Start/live use the status channel. An empty configured
        value means "omit channel" so existing setups are unchanged.
        """
        status_channel = self.config_entry.options.get(
            CONF_NOTIFY_CHANNEL, DEFAULT_NOTIFY_CHANNEL
        )
        finish_channel = self.config_entry.options.get(
            CONF_NOTIFY_FINISH_CHANNEL, DEFAULT_NOTIFY_FINISH_CHANNEL
        )
        if event_type in (NOTIFY_EVENT_FINISH, NOTIFY_EVENT_CLEAN, "pre_complete"):
            return (finish_channel or status_channel) or None
        return status_channel or None

    def _log_notification(
        self,
        event_type: str | None,
        message: str,
        *,
        targets: str = "",
        deferred_reason: str | None = None,
        delivered: bool = True,
    ) -> None:
        """Emit log lines for a notification's send / defer / drop.

        Every user-facing notification funnels through ``_dispatch_notification``,
        so this is the single place that records what WashData notified about,
        where it went, and whether it was delivered, deferred (quiet-hours /
        presence hold), or dropped.

        Log contract (regression-locked by ``test_manager_notification_logging``):
        - **INFO** ``"Notification sent (<event>)"`` — one line per discrete
          notification (start/finish/milestone/clean/pause/…). Target list and
          message body are omitted to prevent entity-ID PII from leaking into
          bug-report logs.
        - **DEBUG** ``"Notification sent (<event>) via <targets>: <summary>"`` —
          full target list and truncated message body for troubleshooting.
        - **DEBUG** for live-progress ticks (high-frequency in-place updates).
        - **DEBUG** for deferred (quiet-hours/presence hold) and not-delivered paths.

        Deferred items are re-dispatched when the hold clears and log again as
        "sent" on actual delivery.
        """
        label = event_type or "notification"
        # Countdown / finish messages can span multiple lines - collapse to one.
        summary = " ".join(str(message).split())
        if len(summary) > 200:
            summary = summary[:197] + "..."
        is_live = event_type == NOTIFY_EVENT_LIVE
        if deferred_reason:
            self._logger.debug(
                "Notification deferred (%s) - %s: %s", label, deferred_reason, summary
            )
        elif not delivered:
            self._logger.debug(
                "Notification not delivered (%s) - no matching target: %s",
                label, summary,
            )
        elif is_live:
            self._logger.debug(
                "Notification sent (%s) via %s: %s", label, targets, summary
            )
        else:
            self._logger.info("Notification sent (%s)", label)
            self._logger.debug(
                "Notification sent (%s) via %s: %s", label, targets, summary
            )

    def _dispatch_notification(
        self,
        message: str,
        *,
        title: str | None = None,
        icon: str | None = None,
        event_type: str | None = None,
        person_entity_id: str | None = None,
        person_name: str | None = None,
        extra_vars: dict[str, Any] | None = None,
        allow_deferral: bool = True,
        allow_presence_deferral: bool = True,
    ) -> bool:
        """Route notification via actions or notify service with optional gating.

        ``allow_deferral`` gates the quiet-hours (do-not-disturb) hold; a
        quiet-window flush passes ``allow_deferral=False`` so the released item is
        not re-held by the still-closing window. ``allow_presence_deferral`` gates
        the "notify only when home" presence hold *independently* — a quiet-hours
        flush must keep presence gating on (nobody home => stay queued), so it
        leaves ``allow_presence_deferral=True``. Only the presence flush (which
        runs *because* someone is now home) disables both.
        """
        # Signals whether this call *queued* the notification for later delivery
        # (quiet-hours / presence hold) instead of sending or dropping it. Callers
        # that use a "fire once" dedup flag (e.g. the clean-laundry nag) must treat
        # a deferral as handled, otherwise they re-queue a duplicate on every retry
        # tick for the whole quiet/away window.
        self._last_dispatch_deferred = False
        if not title:
            title_template = self.config_entry.options.get(CONF_NOTIFY_TITLE, DEFAULT_NOTIFY_TITLE)
            title = self._safe_format_template(
                title_template,
                fallback_template=DEFAULT_NOTIFY_TITLE,
                device=self.config_entry.title,
            )

        if not icon:
            icon = self.config_entry.options.get(CONF_NOTIFY_ICON)

        if person_entity_id is None and self._notify_people:
            for candidate in self._notify_people:
                state = self.hass.states.get(candidate)
                if state and state.state == STATE_HOME:
                    person_entity_id = candidate
                    person_name = state.name or state.attributes.get(
                        "friendly_name", candidate
                    )
                    break

        variables: dict[str, Any] = {
            "device": self.config_entry.title,
            "program": self._current_program,
            "message": message,
            "title": title,
            "icon": icon,
            "event_type": event_type,
            "person_entity_id": person_entity_id,
            "person_name": person_name,
        }
        if extra_vars:
            variables.update(extra_vars)

        # Channel + auto-dismiss timeout apply to every event type. Inject into both
        # the action variables and the notify-service extra_vars so both delivery
        # paths honour them. Empty channel / zero timeout are omitted (no-op default).
        channel = self._resolve_channel(event_type)
        if channel:
            variables["channel"] = channel
            extra_vars = {**(extra_vars or {}), "channel": channel}
        if self._notify_timeout_seconds > 0:
            variables["timeout"] = self._notify_timeout_seconds
            extra_vars = {**(extra_vars or {}), "timeout": self._notify_timeout_seconds}
        tap_target = self._notification_tap_target()
        if tap_target and message != _CLEAR_NOTIFICATION_MARKER:
            # A dismiss marker is a command, not a card - it has nothing to tap.
            variables["clickAction"] = tap_target
            variables["url"] = tap_target
            extra_vars = {
                **(extra_vars or {}),
                "clickAction": tap_target,
                "url": tap_target,
            }

        # Quiet hours (do-not-disturb): hold finish-type notifications that would
        # wake someone and deliver them at the end of the window. Live-progress ticks
        # and the start notification are never delayed. Guarded by allow_deferral so a
        # quiet-window flush (allow_deferral=False) cannot re-defer.
        if (
            allow_deferral
            and event_type in _QUIET_HOURS_EVENT_TYPES
            and self._in_quiet_hours()
        ):
            self._queue_quiet_hours_notification(
                message,
                title=title,
                icon=icon,
                event_type=event_type,
                extra_vars=extra_vars,
            )
            self._last_dispatch_deferred = True
            self._log_notification(event_type, message, deferred_reason="quiet hours")
            return False

        if (
            allow_presence_deferral
            and self._notify_only_when_home
            and self._notify_people
        ):
            if not self._is_any_notify_person_home():
                if event_type == NOTIFY_EVENT_LIVE:
                    self._pending_notifications = [
                        entry
                        for entry in self._pending_notifications
                        if entry.get("event_type") != NOTIFY_EVENT_LIVE
                    ]
                self._pending_notifications.append(
                    {
                        "message": message,
                        "title": title,
                        "icon": icon,
                        "event_type": event_type,
                        "extra_vars": extra_vars,
                    }
                )
                self._last_dispatch_deferred = True
                self._log_notification(
                    event_type, message, deferred_reason="nobody home"
                )
                return False

        actions_sent = False
        if self._notify_actions:
            actions_sent = bool(self._run_notification_actions(variables))

        # If actions fired and there are no per-event services, skip the
        # service/persistent-notification path entirely.
        services = self._get_services_for_event(event_type)
        if actions_sent and not services:
            self._log_notification(event_type, message, targets="actions")
            return True

        service_sent = self._send_notification_service(
            message,
            services=services,
            title=title,
            icon=icon,
            event_type=event_type,
            extra_vars=extra_vars,
        )

        if actions_sent or service_sent:
            targets: list[str] = []
            if actions_sent:
                targets.append("actions")
            if service_sent:
                # _send_notification_service falls back to a persistent
                # notification only when no notify services are configured.
                targets.extend(services if services else ["persistent_notification"])
            self._log_notification(
                event_type, message, targets=", ".join(targets)
            )
        else:
            self._log_notification(event_type, message, delivered=False)
        return actions_sent or service_sent

    def _send_notification_service(
        self,
        message: str,
        *,
        services: list[str],
        title: str | None = None,
        icon: str | None = None,
        event_type: str | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> bool:
        """Send a notification to each configured notify service, or fall back to persistent notification."""
        ev = extra_vars or {}
        # Base payload shared by all notification platforms.
        data: dict[str, Any] = {}
        if icon:
            data["icon"] = icon
        icon_color = self._notification_icon_color()

        # Live-progress-only payload keys (countdown, progress bar, throttle markers).
        # Live updates are already gated to mobile_app targets by the guard below,
        # so these keys never reach strict-schema platforms.
        if event_type == NOTIFY_EVENT_LIVE:
            for key in (
                "progress",
                "progress_max",
                "live_update",
                "alert_once",
                "cycle_seconds",
                "time_remaining_seconds",
                "minutes_left",
                "live_updates_sent",
                "live_updates_cap",
                "chronometer",
                "when",
                "countdown",
            ):
                if key in ev:
                    data[key] = ev[key]

        sent = False
        for notify_service in services:
            if event_type == NOTIFY_EVENT_LIVE and not self._is_mobile_notify_service(
                notify_service
            ):
                self._logger.debug(
                    "Skipping live notification for non-mobile notify service: %s",
                    notify_service,
                )
                continue

            # Mobile-app-specific keys (tag/timeout/channel/priority) plus the iOS
            # Live Activity enrichment keys (subtitle/content_state/activity) are
            # rejected by some strict-schema platforms such as Signal Messenger.
            # Only add them for mobile_app targets; all other platforms receive
            # the base payload only.
            svc_data = dict(data)
            svc_data.update(self._mobile_service_extras(ev, notify_service))

            # #435: the companion app reads the notification icon from
            # `notification_icon`, NOT from `icon` - so the configured mdi icon was
            # being sent under a key no companion platform looks at. `icon` stays in
            # the base payload for the platforms that do use it (notify.html5 and
            # friends); the mobile-only alias is added here. The same key now covers
            # both platforms: Android draws it in the status bar, and iOS renders it
            # as a communication-notification avatar in place of the app icon from
            # companion app 2026.8.0 (home-assistant/iOS#4672). Older iOS builds
            # ignore the key rather than failing, so there is nothing to gate on.
            if icon and self._is_mobile_notify_service(notify_service):
                svc_data["notification_icon"] = icon

            # #454: with a washer, a dryer and a dishwasher live at once, every
            # card on the Lock Screen looks the same. One configured colour maps
            # to the three keys the companion apps actually read: `color` is the
            # Android notification accent, `notification_icon_color` tints the iOS
            # icon glyph, and `progress_bar_color` recolours the iOS Live Activity
            # bar (it falls back to notification_icon_color, but is set explicitly
            # so the two stay in step). Mobile-only, same as the icon above; unset
            # leaves the payload byte-identical to before.
            #
            # iOS is where this reliably shows. On Android the companion app does
            # `builder.color = parseColor(notification_icon_color ?: color)` and
            # never calls setColorized, so Android 12+ applies it per-OEM: a Pixel
            # tints the icon, Samsung One UI shows nothing. Nothing we can send
            # changes that, which is why the setting's help text says so (item 372).
            # Note the app reads the iOS-named key FIRST - harmless only because
            # both carry one value here, so do not let them diverge.
            #
            # #465: on an ordinary iOS notification the two keys mean different
            # things: `color` is the avatar disc and `notification_icon_color` the
            # glyph on it (white by default), so sending one value to both drew the
            # icon in the disc's own colour and it vanished. Only a Live Activity
            # reads `notification_icon_color` as the icon tint, so only the live
            # update carries it; Android falls back to `color` either way.
            if icon_color and self._is_mobile_notify_service(notify_service):
                svc_data["color"] = icon_color
                if event_type == NOTIFY_EVENT_LIVE:
                    svc_data["notification_icon_color"] = icon_color
                svc_data["progress_bar_color"] = icon_color

            state = (
                self.hass.states.get(notify_service)
                if notify_service.startswith("notify.")
                else None
            )
            # A notify *entity* (has a state, domain "notify") only registers the
            # entity service notify.send_message — NOT a legacy notify.<object_id>
            # domain service. So route entity targets through send_message
            # regardless of svc_data; that schema accepts only entity_id/message/
            # title, so drop any unsupported extras (icon, mobile/iOS keys) rather
            # than fall through to a legacy service that would raise ServiceNotFound
            # and silently drop the notification. Legacy notify.mobile_app_* targets
            # have no entity state (state is None) and correctly take the else path.
            if state is not None and getattr(state, "domain", None) == "notify":
                # A dismiss marker is carried *by* its tag, and send_message cannot
                # carry one - delivering it anyway would show the user a card whose
                # body literally reads "clear_notification". There is no way to
                # dismiss an entity-target card, so skip the send entirely. (Before
                # entity targets were routed here, these calls always had a tag and
                # so took the legacy-service path, where the tag works.)
                if message == _CLEAR_NOTIFICATION_MARKER and ev.get("tag"):
                    self._logger.debug(
                        "Notify entity %s: skipping dismiss marker (tag=%s) - "
                        "notify.send_message cannot carry a tag, so it would be "
                        "delivered as visible text",
                        notify_service, ev.get("tag"),
                    )
                    continue
                if svc_data:
                    self._logger.debug(
                        "Notify entity %s: dropping %d unsupported payload key(s) "
                        "(%s) - notify.send_message accepts only message/title",
                        notify_service, len(svc_data), ", ".join(sorted(svc_data)),
                    )
                service_data: dict[str, Any] = {
                    "entity_id": notify_service,
                    "message": message,
                }
                if title:
                    service_data["title"] = title
                self.hass.async_create_task(
                    self.hass.services.async_call(
                        "notify", "send_message", service_data
                    )
                )
            else:
                domain, service = (
                    notify_service.split(".", 1)
                    if "." in notify_service
                    else ("notify", notify_service)
                )
                # `title` is OPTIONAL in notify's service schema but validated as a
                # string, so passing it as None fails validation outright
                # ("string value is None at 'title'") and the call never reaches the
                # platform. _dispatch_notification always resolves a title, but the
                # four dismiss-marker senders do not pass one - so every tag clear
                # (live-activity end, lifecycle hand-over, clean reminder, timer
                # pause) was rejected before delivery and nothing was ever
                # dismissed on the phone (#446 follow-up).
                service_data = {"message": message}
                if title is not None:
                    service_data["title"] = title
                if svc_data:
                    service_data["data"] = svc_data
                self.hass.async_create_task(
                    self.hass.services.async_call(domain, service, service_data)
                )
            sent = True

        if not sent:
            if event_type == NOTIFY_EVENT_LIVE:
                return False
            # A dismiss marker is not content: if no target could carry it, there is
            # nothing to show. Falling through would post a persistent-notification
            # card whose body reads "clear_notification" - the same leak the entity
            # branch above guards against, just via the other exit. The callers that
            # send this marker dismiss their own persistent notification separately
            # (_pn_dismiss), so nothing is lost by returning early.
            if message == _CLEAR_NOTIFICATION_MARKER:
                return False
            # Reuse the notification's tag as a stable persistent-notification id so
            # the HA notifications tab collapses the lifecycle thread to one entry
            # instead of accumulating a new card per cycle (issue #248/#249 clutter).
            return _pn_create(
                self.hass,
                message,
                title=title,
                notification_id=ev.get("tag"),
            )

        return sent

    def _run_notification_actions(self, variables: dict[str, Any]) -> bool:
        """Run configured notification actions."""
        actions: list[dict[str, Any]] = self._notify_actions
        if not actions:
            return False

        if self._notify_script is None:
            try:
                # Validate through cv.SCRIPT_SCHEMA before handing the sequence to
                # Script, because that is what turns a templated `data` value into
                # a Template: `cv.template_complex` converts the strings that look
                # like templates, and at run time `render_complex` renders only
                # Template instances and passes plain strings through untouched.
                # Built straight from the stored options - as this did - every
                # `{{ device }}` in a user's action was delivered to their phone
                # as the literal text `{{ device }}`, which makes the documented
                # notification variables useless. Found by the test box's
                # check_notify_actions.sh; register item 323.
                self._notify_script = script_helper.Script(
                    self.hass,
                    cv.SCRIPT_SCHEMA(actions),
                    name=f"{self.config_entry.title} notification",
                    domain=DOMAIN,
                    logger=_LOGGER,
                )
            except vol.Invalid as err:
                self._logger.error(
                    "Invalid notification action configuration for %s: %s",
                    self.config_entry.title,
                    err,
                )
                return False
            except (ValueError, TypeError, HomeAssistantError, OverflowError) as err:
                self._logger.error(
                    "Invalid notification action configuration for %s: %s",
                    self.config_entry.title,
                    err,
                )
                return False
            except Exception as err:
                self._logger.exception(
                    "Unexpected error while building notification actions for %s: %s",
                    self.config_entry.title,
                    err,
                )
                return False
        script = self._notify_script

        try:
            action_task = self.hass.async_create_task(
                script.async_run(variables, context=Context())
            )
            # This method is synchronous, so the script runs fire-and-forget: a
            # failure inside async_run() would otherwise land after we return True
            # and be swallowed. Surface it via a done-callback so an action-only
            # setup at least logs the drop. (A user-visible fallback would require
            # awaiting, i.e. making the whole notification-dispatch chain async.)
            def _log_action_failure(task: Task[Any]) -> None:
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    self._logger.warning(
                        "Notification action execution failed for %s: %s",
                        self.config_entry.title,
                        exc,
                    )

            action_task.add_done_callback(_log_action_failure)
            return True
        except HomeAssistantError as err:
            self._logger.warning(
                "Notification action execution failed for %s: %s",
                self.config_entry.title,
                err,
            )
            return False
        except Exception as err:
            self._logger.exception(
                "Unexpected error while scheduling notification actions for %s: %s",
                self.config_entry.title,
                err,
            )
            return False

    def _is_any_notify_person_home(self) -> bool:
        """Return True when any configured person is home."""
        for person_entity_id in self._notify_people:
            state = self.hass.states.get(person_entity_id)
            if state and state.state == STATE_HOME:
                return True
        return False

    @callback
    def _handle_notify_person_change(self, event: Event[evt.EventStateChangedData]) -> None:
        """Handle person state changes to release pending notifications."""
        new_state = event.data.get("new_state")
        if not new_state or new_state.state != STATE_HOME:
            return

        if not self._pending_notifications:
            return

        self._flush_pending_notifications(
            new_state.entity_id,
            new_state.name
            or new_state.attributes.get("friendly_name", new_state.entity_id),
        )

    def _flush_pending_notifications(
        self, person_entity_id: str | None, person_name: str | None
    ) -> None:
        """Deliver every notification presence gating queued, and record it.

        Two callers reach this: a person arriving home, and the listener finding
        somebody already home when it (re-)attaches after a reload. They were
        two copies of the same loop and drifted - only one of them recorded that
        a Live Activity had started, so a queued live card delivered by the other
        left `_live_activity_started` False, the cycle-end tail skipped
        `_end_live_activity()`, and the card stayed frozen on the phone (#446).
        One body now, so they cannot disagree again.
        """
        pending: list[dict[str, Any]] = list(self._pending_notifications)
        self._pending_notifications = []

        # The #446 handover has to run BEFORE the queue is delivered, not after
        # the live entry inside it. It `_send_tag_clear`s `_lifecycle_tag`, and on
        # this path entries that RIDE that tag are delivered first: a deferred
        # LIVE entry replaces any earlier live one and is appended last (see
        # `_dispatch_notification`), so a `pre_complete` reminder queued earlier in
        # the cycle sits ahead of it. Flushed in order, the reminder was delivered
        # and then dismissed off the phone a moment later by the handover - and it
        # is a `priority: high` "nearly done" card, i.e. the one worth having.
        # The direct path cannot hit this: there the first live tick happens early,
        # long before any reminder exists. Only presence deferral can put the two
        # in this order.
        handover_done = False

        def _deliver(entry: dict[str, Any]) -> None:
            sent = self._dispatch_notification(
                entry["message"],
                title=entry.get("title"),
                icon=entry.get("icon"),
                event_type=entry.get("event_type"),
                person_entity_id=person_entity_id,
                person_name=person_name,
                extra_vars=entry.get("extra_vars"),
                allow_deferral=False,
                allow_presence_deferral=False,
            )
            if sent and entry.get("event_type") == NOTIFY_EVENT_LIVE:
                ev_raw = entry.get("extra_vars")
                ev: dict[str, Any] = ev_raw if isinstance(ev_raw, dict) else {}
                if "progress" not in ev:
                    self._live_waiting_notification_sent = True
                else:
                    self._live_notification_sent_count += 1
                    self._last_live_notification_time = utc_now()
                # The queued entry carries the same `activity: "start"` the direct
                # paths send, so the phone has a live activity either way and the
                # cycle-end teardown has to know about it (#446). Recorded only
                # once the dispatch actually SENT, which is why the flag is not set
                # next to the hoisted handover: a failed delivery would otherwise
                # claim an activity that is not on the phone.
                if handover_done:
                    self._live_activity_started = True
                else:
                    self._record_live_activity_started()

        if not self._live_activity_started and any(
            entry.get("event_type") == NOTIFY_EVENT_LIVE for entry in pending
        ):
            # START entries go out FIRST, *before* the clear, and are not dropped.
            # "Superseded by the live activity" is true only of a mobile target
            # that also receives the live card: `_send_tag_clear` returns early
            # when `_notify_live_services` is empty and only ever addresses that
            # list, while START has its own `_notify_start_services` and may be a
            # telegram or e-mail target the clear can never reach - and a
            # notification ACTION fires on delivery, so an automation branching on
            # `event_type == "start"` needs the dispatch to happen at all.
            # Delivering them ahead of the clear gets every case right at once: a
            # mobile start card is delivered and then swept up by the handover
            # exactly as it was before, and every other target simply keeps its
            # notification.
            # SPLIT the queue at the last START; do not pull STARTs forward.
            # Round 32 hoisted them to the front, which silently reordered every
            # lifecycle-tagged entry queued BEFORE one - and FINISH survives the
            # `_clear_live_progress_notification` purge (which drops LIVE, START
            # and `pre_complete`, not FINISH). So `[FINISH(A), START(B), LIVE(B)]`
            # is reachable: nobody home, cycle A ends, cycle B starts. Hoisting
            # sent START(B) first, cleared the tag, and then delivered FINISH(A)
            # AFTER it, leaving "cycle A finished" pinned to the lifecycle tag for
            # the whole of cycle B while B's own start card had already been
            # cleared. Delivering in order gets it right for free: FINISH(A)
            # lands, START(B) replaces it on the same tag, and the handover then
            # clears the one card that is left.
            _last_start = max(
                (
                    i
                    for i, e in enumerate(pending)
                    if e.get("event_type") == NOTIFY_EVENT_START
                ),
                default=-1,
            )
            _head = pending[: _last_start + 1]
            for entry in [
                e for e in _head if e.get("event_type") != NOTIFY_EVENT_LIVE
            ]:
                _deliver(entry)
            # A LIVE entry sitting before that START (possible across cycles, since
            # the dedup only re-appends within one queue) still belongs after the
            # clear, with the rest of the tail.
            pending = [
                e for e in _head if e.get("event_type") == NOTIFY_EVENT_LIVE
            ] + pending[_last_start + 1 :]
            self._hand_over_lifecycle_to_live_activity()
            handover_done = True

        for entry in pending:
            _deliver(entry)

    def _handle_noise_cycle(self, max_power: float) -> None:
        """Handle a detected noise cycle."""
        # Clean up old noise events > 24h
        now = utc_now()
        self._noise_events = [
            t
            for t in getattr(self, "_noise_events", [])
            if (now - t).total_seconds() < 86400
        ]
        self._noise_events.append(now)

        # Track max power of noise
        self._noise_max_powers = getattr(self, "_noise_max_powers", [])
        self._noise_max_powers.append(max_power)

        # If noise events exceed threshold in 24h, trigger tune
        if len(self._noise_events) >= self._noise_events_threshold:
            # Tracked (audit MANAGER-13): it saves the store.
            self._spawn_tracked(self._tune_threshold())

    async def _tune_threshold(self) -> None:
        """Increase the minimum power threshold."""
        current_min = self.detector.config.min_power

        # Calculate new suggested threshold
        # Max of observed noise * 1.2 safety factor
        noise_max = max(self._noise_max_powers)
        new_min = noise_max * 1.2

        # Cap absolute max to avoid runaway (e.g. 50W)
        if new_min > 50.0:
            new_min = 50.0

        if new_min <= current_min:
            # Clear events so we don't loop try to update
            self._noise_events = []
            self._noise_max_powers = []
            return

        self._logger.info(
            "Auto-Tune suggestion: min_power from %.1fW -> %.1fW due to noise",
            current_min,
            new_min,
        )

        # Store a suggestion (do not mutate user-set options). The suggestion is
        # surfaced in the panel (Settings suggestions banner / per-field pill);
        # WashData intentionally does not raise a persistent notification here.
        self.profile_store.set_suggestion(
            CONF_MIN_POWER,
            float(new_min),
            f"Auto-tune: {len(self._noise_events)} ghost cycles detected in 24h",
        )
        await self.profile_store.async_save()

        # Reset trackers
        self._noise_events = []
        self._noise_max_powers = []

    def _update_estimates(self) -> None:
        """Update time remaining and profile estimates."""
        if self.detector.state in (
            STATE_OFF,
            STATE_UNKNOWN,
            STATE_IDLE,
            STATE_STARTING,
            STATE_ANTI_WRINKLE,
            STATE_DELAY_WAIT,
        ):
            self._current_program = "off"
            self._time_remaining = None
            self._total_duration = None
            # A probe out of a completed cycle keeps its 100 % (item 515): most
            # abort back to Finished; RUNNING zeroes it when one commits.
            if not (
                self.detector.state == STATE_STARTING
                and self._cycle_completed_time is not None
            ):
                self._cycle_progress = 0.0
            self._projected_energy_wh = None
            self._projected_cost = None
            self._cycle_anomaly = "none"
            self._overrun_ratio = 0.0
            self._envelope_position = None
            self._last_match_result = None
            self._notify_update_deferrable()
            return

        now = utc_now()

        # Throttle heavy matching to configured interval (default: 5 minutes)
        effective_match_interval = self._profile_match_interval
        if (
            self._last_estimate_time
            and (now - self._last_estimate_time).total_seconds()
            < effective_match_interval
        ):
            # Still update remaining/progress if we already have a match
            self._update_remaining_only()
            self._check_pre_completion_notification()
            self._check_live_progress_notification()
            return

        # SKIP matching if manual program is active
        if self._manual_program_active:
            self._last_estimate_time = now  # touch timestamp to throttle estimates loop
            self._update_remaining_only()
            # Also check notifications in loop
            self._check_pre_completion_notification()
            self._check_live_progress_notification()
            self._notify_update_deferrable()
            return

        # No matching task trigger here anymore!
        # The detector callback handles it.
        # Just update progress/remaining based on existing match.
        self._update_remaining_only()
        self._check_pre_completion_notification()
        self._check_live_progress_notification()
        self._notify_update_deferrable()

    def _reset_live_notification_state(
        self, *, keep_activity_started: bool = False
    ) -> None:
        """Reset per-cycle live notification counters and timers.

        ``keep_activity_started`` exists for the single caller that is NOT a cycle
        boundary. Every other call sites here is one - cycle start, cycle end, and
        the live-progress clear - so resetting the flag is exactly right for them.
        `async_reload_config` is different: it re-arms live notifications MID-cycle
        when the user saves any option, and clearing the flag there makes the next
        live tick look like the first of a new cycle. Three things follow, all
        wrong, and none of them visible from this function alone:

        * `_record_live_activity_started` re-runs the #446 handover, which
          `_send_tag_clear`s `_lifecycle_tag` - and the pre-completion reminder
          rides that same tag at `priority: high`, so an already-delivered
          reminder is dismissed off the user's phone.
        * `_apply_live_notification_prefs` gates `silent` / `push` on this flag, so
          the next live update alerts audibly even with `notify_live_silent` on:
          #417 re-entering through the reload door.
        * the handover is a once-per-cycle event by design, and a settings save
          does not start a cycle.

        Preserving is safe in both directions: the flag is only ever kept at the
        value it already had, so a reload that ENABLES live notifications mid-cycle
        still leaves it False and the handover still runs on the first real tick.
        """
        self._live_notification_sent_count = 0
        self._live_notification_cap = 0
        self._last_live_notification_time = None
        self._live_waiting_notification_sent = False
        self._live_chronometer_overrun_sent = False
        if not keep_activity_started:
            self._live_activity_started = False

    @staticmethod
    def _is_mobile_notify_service(notify_service: str | None) -> bool:
        """Return True when configured notify target is a mobile app service."""
        if not notify_service:
            return False
        _, service = (
            notify_service.split(".", 1)
            if "." in notify_service
            else ("notify", notify_service)
        )
        return service.startswith("mobile_app")

    def _notification_tap_target(self) -> str:
        """Where a tap on this device's notifications should land (#438).

        Blank (the shipped default) resolves to this appliance's own panel deep link
        ``/ha-washdata?device=<entry_id>``, so a notification about the dryer opens
        the dryer instead of whichever appliance the panel happened to show last.
        The entry id is used rather than the title because it survives a rename, and
        it needs no URL escaping. A user-supplied value wins, and the literal
        ``none`` turns the tap target off entirely.

        Emitted as both ``clickAction`` (Android) and ``url`` (iOS), which are the
        two companion-app keys for the same thing; both are mobile-only and are
        filtered to ``mobile_app_*`` targets by ``_mobile_service_extras``.
        """
        configured = (self._notify_live_click_action or "").strip()
        if configured:
            return "" if configured.lower() == "none" else configured
        return f"/{PANEL_URL_PATH}?device={self.entry_id}"

    def _notification_icon_color(self) -> str | None:
        """Resolve the per-device notification accent colour (#454).

        Returns ``None`` when unset, which is the shipped default and leaves the
        payload exactly as it was. A bare hex value is accepted without the leading
        ``#`` because that is how a user pastes one out of a colour picker; anything
        else (a CSS colour name, which Android accepts and iOS ignores) is passed
        through untouched rather than rejected - a colour the companion app does not
        understand is ignored by the app, so there is nothing to fail on here.
        """
        configured = str(
            self.config_entry.options.get(CONF_NOTIFY_ICON_COLOR) or ""
        ).strip()
        if not configured:
            return None
        if _HEX_COLOR_RE.fullmatch(configured):
            return f"#{configured}"
        return configured

    @property
    def _timer_pause_action_id(self) -> str:
        """Stable mobile action ID for timer-pause Resume button, unique per device."""
        return f"RESUME_WD_{self.entry_id[:8].upper()}"

    def _estimate_live_notification_cap(self) -> int:
        """Compute hard cap for live updates from estimated cycle duration and overrun margin."""
        interval = max(30, int(self._notify_live_interval_seconds))
        estimated_duration = float(
            self._matched_profile_duration
            or self._total_duration
            or max(float(self.detector.get_elapsed_seconds()), float(interval))
        )
        estimated_updates = max(1, int(np.ceil(estimated_duration / interval)))
        overrun_ratio = max(0, float(self._notify_live_overrun_percent)) / 100.0
        return max(1, int(np.ceil(estimated_updates * (1.0 + overrun_ratio))))

    def _apply_live_notification_prefs(self, extra_vars: dict[str, Any]) -> None:
        """Inject the user's live-notification data keys (#347, #417).

        ``sticky`` keeps the live notification on screen when tapped. It is a
        mobile-only key forwarded only to ``mobile_app_*`` live targets, and its
        default (off) adds nothing, so the payload is byte-identical to before
        unless the user opts in. The tap target itself is no longer applied here:
        it belongs to every event type, so ``_dispatch_notification`` injects
        ``_notification_tap_target()`` centrally instead (#438).

        ``silent`` (#417) marks a *refresh* of the running Live Activity as a
        non-alerting, lower-priority push, which is what stops iOS playing a sound and
        vibrating on every interval tick; ``push.interruption-level: passive`` is the
        companion's generic quiet key and covers the case where the update is rendered
        as an ordinary banner instead. Neither is applied to the update that STARTS the
        activity: that one is the "cycle is now on your Lock Screen" cue and stays
        audible (per the companion docs ``silent`` has no effect there in any case).
        """
        if self._notify_live_sticky:
            extra_vars["sticky"] = "true"
        if self._notify_live_silent and self._live_activity_started:
            extra_vars["silent"] = True
            extra_vars["push"] = {"interruption-level": "passive"}

    def _check_live_progress_notification(self) -> None:
        """Send throttled live progress notifications for compatible mobile targets."""
        if not self._notify_live_services and not self._notify_actions:
            return
        if self.detector.state not in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            return

        # #437: a match landing is not the same as an ETA existing.
        # _update_remaining_only() is throttled to one estimate per 5 s, so the tick
        # that accepts a match can find _matched_profile_duration set while
        # _time_remaining is still None - and the progress branch below renders that
        # `float(self._time_remaining or 0.0)` as remaining 0, i.e. elapsed == total
        # (a full 100 % bar) and minutes_left == 1 ("less than 1 minute") at the very
        # start of the cycle. Require a real estimate; the waiting latch keeps that
        # tick silent rather than re-sending the waiting message.
        has_profile_match = bool(
            self._matched_profile_duration
            and self._matched_profile_duration > 0
            and self._time_remaining is not None
        )
        if has_profile_match:
            # A profile has been matched - reset the waiting latch so future
            # "no profile yet" phases (e.g. after a cycle restart) will send
            # the waiting message again.
            self._live_waiting_notification_sent = False
        if not has_profile_match:
            # Suppress the waiting notification when no profiles exist at all —
            # the setup card explains the state instead.
            if not self.profile_store.has_real_profiles:
                return
            if self._live_waiting_notification_sent:
                return

            # Fixed (non user-editable) live message: localize via the cached
            # options.error template, falling back to the English default.
            waiting_template = self._timer_ui_strings.get(
                "notify_live_waiting_message", DEFAULT_NOTIFY_LIVE_WAITING_MESSAGE
            )
            msg = self._safe_format_template(
                waiting_template,
                fallback_template=DEFAULT_NOTIFY_LIVE_WAITING_MESSAGE,
                device=self.config_entry.title,
                program=self._current_program,
            )
            waiting_extra_vars: dict[str, Any] = {
                "tag": self._live_notification_tag,
                "live_update": True,
                "alert_once": True,
            }
            # C3: mark the first live notification of the cycle so iOS can begin a
            # Live Activity even before a profile is matched (mobile-only key).
            if not self._live_activity_started:
                waiting_extra_vars["activity"] = "start"
            self._apply_live_notification_prefs(waiting_extra_vars)
            sent = self._dispatch_notification(
                msg,
                event_type=NOTIFY_EVENT_LIVE,
                extra_vars=waiting_extra_vars,
            )
            self._live_waiting_notification_sent = sent
            if sent:
                self._record_live_activity_started()
            return

        interval = max(30, int(self._notify_live_interval_seconds))
        now = utc_now()
        if self._last_live_notification_time and (
            now - self._last_live_notification_time
        ).total_seconds() < interval:
            return

        cap_candidate = self._estimate_live_notification_cap()
        if cap_candidate > self._live_notification_cap:
            self._live_notification_cap = cap_candidate

        total_seconds = int(
            max(
                1,
                round(
                    float(
                        self._total_duration
                        or self._matched_profile_duration
                        or self.detector.get_elapsed_seconds()
                    )
                ),
            )
        )
        remaining_seconds = int(max(0, round(float(self._time_remaining or 0.0))))
        elapsed_seconds = max(0, total_seconds - remaining_seconds)

        # When a chronometer notification is on the phone but the estimate has
        # expired, bypass the cap once to replace the frozen "0:00" countdown
        # with a plain text update so the user isn't left with a stale timer.
        chronometer_overrun = (
            self._notify_live_chronometer
            and remaining_seconds <= 0
            and self._live_notification_sent_count > 0
            and not self._live_chronometer_overrun_sent
        )
        if not chronometer_overrun and self._live_notification_sent_count >= self._live_notification_cap:
            return
        minutes_left = max(1, math.ceil(remaining_seconds / 60))

        msg_template = self.config_entry.options.get(
            CONF_NOTIFY_PRE_COMPLETE_MESSAGE,
            DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE,
        )
        msg = self._safe_format_template(
            msg_template,
            fallback_template=DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE,
            device=self.config_entry.title,
            minutes=minutes_left,
            program=self._current_program,
        )

        extra_vars: dict[str, Any] = {
            "tag": self._live_notification_tag,
            "progress": elapsed_seconds,
            "progress_max": total_seconds,
            "live_update": True,
            "alert_once": True,
            "cycle_seconds": total_seconds,
            "time_remaining_seconds": remaining_seconds,
            "minutes_left": minutes_left,
            "live_updates_sent": self._live_notification_sent_count + 1,
            "live_updates_cap": self._live_notification_cap,
        }
        if self._notify_live_chronometer and remaining_seconds > 0:
            extra_vars["chronometer"] = True
            extra_vars["when"] = int(now.timestamp()) + remaining_seconds
            extra_vars["countdown"] = True

        # C3: iOS Live Activity enrichment. Derived from the SAME values feeding the
        # flat progress/when keys above. Forwarded to mobile_app_* targets only (see
        # _send_notification_service); non-mobile live targets are already skipped.
        eta_timestamp = int(now.timestamp()) + remaining_seconds
        progress_pct = (
            100.0 * elapsed_seconds / total_seconds if total_seconds > 0 else 0.0
        )
        activity_marker = None if self._live_activity_started else "start"
        extra_vars.update(
            self._build_ios_live_activity_extras(
                state="paused" if self.detector.state == STATE_PAUSED else "running",
                progress_pct=progress_pct,
                eta_timestamp=eta_timestamp,
                program=self._current_program,
                device=self.config_entry.title,
                activity=activity_marker,
            )
        )
        self._apply_live_notification_prefs(extra_vars)
        sent = self._dispatch_notification(
            msg,
            event_type=NOTIFY_EVENT_LIVE,
            extra_vars=extra_vars,
        )
        if sent:
            self._record_live_activity_started()
            if chronometer_overrun:
                self._live_chronometer_overrun_sent = True
            else:
                self._live_notification_sent_count += 1
            self._last_live_notification_time = now

    def _send_tag_clear(self, tag: str) -> None:
        """Send the companion app's documented ``clear_notification`` for ``tag``.

        This is the ONLY way to end an iOS Live Activity (HA companion docs, "Live
        Activities and Live Updates"): an activity is started with
        ``live_update: true``, updated by re-sending the same tag, and ended by
        this. There is no ``activity`` key in that API - the one this integration
        sent on the finished notification was never acted on, which is #446.
        """
        if not self._notify_live_services:
            return
        self._send_notification_service(
            _CLEAR_NOTIFICATION_MARKER,
            services=self._notify_live_services,
            event_type=NOTIFY_EVENT_LIVE,
            extra_vars={"tag": tag},
        )

    def _record_live_activity_started(self) -> None:
        """Mark that a live activity is running, handing over the lifecycle card.

        Three paths deliver the first live notification of a cycle - the waiting
        card, the progress card, and ``_handle_notify_person_change`` releasing
        either of them after presence gating deferred it - and all three have to
        record it identically. The deferred one did not, so a cycle whose only
        live delivery came through presence left ``_live_activity_started`` False,
        the cycle-end path skipped ``_end_live_activity``, and the activity stayed
        frozen on the phone: #446's own bug, reached from the other side.
        """
        if not self._live_activity_started:
            self._hand_over_lifecycle_to_live_activity()
        self._live_activity_started = True

    def _hand_over_lifecycle_to_live_activity(self) -> None:
        """Drop the lifecycle-tagged card as the live activity takes over (#446).

        Live updates used to share the lifecycle tag, so each one replaced the
        start alert in place and the mobile app showed a single entry. With the
        live activity on its own tag that replacement no longer happens, so clear
        the lifecycle tag explicitly the first time an activity starts. Two things
        fall out of it for free: an Android user still sees one entry rather than
        a stale "cycle started" beside the live one, and a Live Activity left
        running under the OLD shared tag by a pre-0.5.7 build is ended here, so
        the upgrade heals a frozen card instead of stranding it.
        """
        self._send_tag_clear(self._lifecycle_tag)

    def _end_live_activity(self) -> None:
        """End the iOS Live Activity at cycle end (#446).

        Called AFTER the finished notification has been dispatched, so the lock
        screen is never momentarily empty: the finished alert lands on the
        lifecycle tag, then the activity on its own tag goes away.
        """
        self._send_tag_clear(self._live_notification_tag)

    def _clear_live_progress_notification(self, clear_services: bool = True) -> None:
        """Clear active live/progress notifications and purge stale deferred alerts.

        On cycle finish (``clear_services=False``) the caller ends the activity
        itself, after the finished notification has been delivered - see
        ``_end_live_activity`` (#446). Only the pending-purge, the action-based
        clear marker (kept for backward compatibility with custom action templates)
        and the state reset run here.

        On shutdown (``clear_services=True``) the two tags are NOT treated alike,
        and the difference is load-bearing (register item 350(c)). The live tag is
        cleared unconditionally, because a Live Activity left behind counts its
        chronometer into negative numbers once nothing updates it. The lifecycle
        tag is cleared only while ``self.detector.state`` is in
        ``_CYCLE_IN_PROGRESS_STATES``: it carries the FINISHED alert, so an unload
        or a restart after a cycle ended used to dismiss the very card the user was
        still reading. "No finished notification follows" is true of a cycle in
        progress and false of one already over - while both tags shared a value
        that could not be distinguished, and now it can. See the comment at the
        clear itself before making this unconditional again.
        """
        # Purge queued live-progress entries and stale start/pre-complete entries
        # so a completed cycle cannot replay them later.
        live_tag = self._live_notification_tag
        self._pending_notifications = [
            entry
            for entry in self._pending_notifications
            if not (
                (
                    entry.get("event_type") == NOTIFY_EVENT_LIVE
                    and isinstance(entry.get("extra_vars"), dict)
                    and entry["extra_vars"].get("tag") == live_tag
                    and entry["extra_vars"].get("live_update") is True
                )
                or entry.get("event_type") in {NOTIFY_EVENT_START, "pre_complete"}
            )
        ]
        # ...and a "minutes left" reminder parked by quiet hours (audit MANAGER-09):
        # it was delivered at the window's end, hours after the cycle, right
        # before "finished".
        self._quiet_pending_notifications = [
            entry for entry in self._quiet_pending_notifications
            if entry.get("event_type") != "pre_complete"
        ]

        # Always emit the clear when the user has any live channel configured.
        # The in-memory sent-count is unreliable after an HA restart (it resets
        # to 0 while the notification still lives on the phone), and a no-op
        # clear for a non-existent tag is harmless on the mobile_app side.
        if not self._notify_live_services and not self._notify_actions:
            self._reset_live_notification_state()
            return

        # Invoke notification actions to clear live notification in action-based setups
        # Include full context variables expected by notification action handlers
        self._run_notification_actions(
            {
                "device": self.config_entry.title,
                "program": "",  # Cleared marker
                "message": _CLEAR_NOTIFICATION_MARKER,  # Clear marker for action handlers
                "title": "",  # Clear title
                "icon": None,
                "event_type": NOTIFY_EVENT_LIVE,
                "person_entity_id": None,
                "person_name": None,
                "tag": self._live_notification_tag,
                "live_update": True,
                "alert_once": True,
                # C3: tell iOS to end the Live Activity (mobile-only key downstream).
                "activity": "end",
            }
        )

        if clear_services:
            # The live tag unconditionally: it carries a Live Activity with a
            # chronometer that goes negative once nothing updates it, so a stale
            # one is worse than none (#446).
            self._send_tag_clear(self._live_notification_tag)
            # The lifecycle tag only while a cycle is actually running. The
            # finished alert uses this SAME tag, so on the shutdown path - a HA
            # restart or an entry unload after a cycle ended - clearing it
            # unconditionally dismisses the finished card the user still wants.
            # "No finished notification follows" is true of a cycle in progress
            # and false of one already over; while the two tags were the same
            # value this could not be distinguished, and now it can.
            if self.detector.state in _CYCLE_IN_PROGRESS_STATES:
                self._send_tag_clear(self._lifecycle_tag)

        # Reset live-update state flags and counters.
        self._reset_live_notification_state()

    def _clear_clean_notification(self) -> None:
        """Dismiss a delivered clean-laundry reminder and purge any queued ones.

        The clean nag uses its own tag (``_clean_tag``) rather than the lifecycle
        tag, so nothing replaces it once the clean state resolves. Mirror the
        lifecycle clear here so a delivered reminder is removed from the mobile
        app instead of lingering. A clear for a non-existent tag is harmless, so
        this runs whenever the user has any clean/finish delivery configured.
        """
        # Drop the repeat-reminder dismiss action listener too, so no stale mobile
        # action stays wired once the reminder is gone (#374).
        self._remove_unload_dismiss_listener()
        # Drop any still-queued clean entries so they cannot replay later — from
        # both the presence-hold queue and the quiet-hours queue (the nag can be
        # deferred into either).
        self._pending_notifications = [
            n for n in self._pending_notifications
            if n.get("event_type") != NOTIFY_EVENT_CLEAN
        ]
        self._quiet_pending_notifications = [
            n for n in self._quiet_pending_notifications
            if n.get("event_type") != NOTIFY_EVENT_CLEAN
        ]

        # Only mobile_app targets understand the "clear_notification" marker;
        # non-mobile targets (email, Telegram, etc.) would receive it as a
        # literal message. Mirror the pattern from _cancel_timer_mobile_notification.
        mobile_services = [
            s for s in self._get_services_for_event(NOTIFY_EVENT_CLEAN)
            if self._is_mobile_notify_service(s)
        ]
        if not mobile_services and not self._notify_actions:
            return

        if self._notify_actions:
            self._run_notification_actions(
                {
                    "device": self.config_entry.title,
                    "program": "",
                    "message": _CLEAR_NOTIFICATION_MARKER,
                    "title": "",
                    "icon": None,
                    "event_type": NOTIFY_EVENT_CLEAN,
                    "person_entity_id": None,
                    "person_name": None,
                    "tag": self._clean_tag,
                }
            )
        if mobile_services:
            self._send_notification_service(
                _CLEAR_NOTIFICATION_MARKER,
                services=mobile_services,
                event_type=NOTIFY_EVENT_CLEAN,
                extra_vars={"tag": self._clean_tag},
            )

    @property
    def _unload_dismiss_action_id(self) -> str:
        """Stable mobile action ID for the unload-reminder dismiss button, per device."""
        return f"UNLOAD_STOP_WD_{self.entry_id[:8].upper()}"

    def _unload_nag_active(self, now: datetime) -> bool:
        """Whether the terminal state must be held alive for the unload reminder.

        Default (one-shot) behaviour: hold only until the single reminder is due, so
        the 30-min progress reset / power-based Off cannot clear the Clean state before
        the reminder fires. Repeat mode (``CONF_NOTIFY_UNLOAD_REPEAT``, #374): hold
        indefinitely so the reminder keeps re-firing, until the user dismisses it from
        the notification or opens the door (both release the hold).
        """
        if (
            not self._is_clean_state
            or self._notify_unload_delay_minutes <= 0
            or self._cycle_completed_time is None
        ):
            return False
        if self._notify_unload_repeat:
            # Only hold while a delivery channel exists — otherwise no reminder (and
            # no dismiss button) is ever sent (dispatch is gated on the same
            # condition), so the hold would strand the Clean state forever.
            # Also bounded by NOTIFY_UNLOAD_REPEAT_MAX_REMINDERS: the "Stop
            # reminding" button is mobile_app-only, so a non-mobile target leaves
            # the door sensor as the only escape, and a sensor that never reports
            # open would pin Clean state (and suppress power-based Off) forever.
            return (
                bool(self._notify_finish_services or self._notify_actions)
                and not self._unload_nag_dismissed
                and self._unload_nag_count < NOTIFY_UNLOAD_REPEAT_MAX_REMINDERS
            )
        return (
            not self._notified_clean_laundry
            and (now - self._cycle_completed_time).total_seconds()
            < self._notify_unload_delay_minutes * 60
        )

    def _ensure_unload_dismiss_listener(self) -> None:
        """Register the mobile action listener for the reminder's dismiss button (once).

        Mirrors the timer-pause interactive notification wiring: a single
        ``mobile_app_notification_action`` listener that matches this device's action
        ID, marks the reminder dismissed, and clears the delivered card.
        """
        if self._remove_unload_action_listener is not None:
            return
        action_id = self._unload_dismiss_action_id

        @callback
        def _on_unload_action(event: Any) -> None:
            if event.data.get("action") == action_id:
                self._logger.debug(
                    "Unload reminder dismissed via notification action"
                )
                self._unload_nag_dismissed = True
                self._clear_clean_notification()
                self._notify_update()

        self._remove_unload_action_listener = self.hass.bus.async_listen(
            "mobile_app_notification_action",
            _on_unload_action,
        )

    def _remove_unload_dismiss_listener(self) -> None:
        """Drop the unload-reminder dismiss action listener, if registered."""
        if self._remove_unload_action_listener is not None:
            self._remove_unload_action_listener()
            self._remove_unload_action_listener = None

    def _reset_unload_nag_tracking(self) -> None:
        """Reset repeat-reminder tracking and drop the dismiss listener.

        Called wherever the Clean state is cleared so the next cycle's reminder starts
        fresh and no stale mobile action listener leaks.
        """
        self._remove_unload_dismiss_listener()
        self._unload_nag_dismissed = False
        self._last_unload_nag_time = None
        self._unload_nag_count = 0

    def _check_pre_completion_notification(self) -> None:
        """Check and send pre-completion notification."""
        if notif_rules.should_notify_pre_completion(
            self._notify_before_end_minutes,
            self._notified_pre_completion,
            self._time_remaining,
            self._cycle_progress,
            self._last_match_ambiguous,
        ):
            # Send notification!
            self._notified_pre_completion = True

            # Distinct reminder message (not the live-update template) so the one-time
            # "X minutes left" alert is not confused with the recurring live ticks that
            # reuse CONF_NOTIFY_PRE_COMPLETE_MESSAGE.
            msg_template = self.config_entry.options.get(
                CONF_NOTIFY_REMINDER_MESSAGE, DEFAULT_NOTIFY_REMINDER_MESSAGE
            )
            minutes_left = self._notify_before_end_minutes

            msg = self._safe_format_template(
                msg_template,
                fallback_template=DEFAULT_NOTIFY_REMINDER_MESSAGE,
                device=self.config_entry.title,
                minutes=minutes_left,
                program=self._current_program,
            )
            self._dispatch_notification(
                msg,
                event_type="pre_complete",
                extra_vars={
                    # Share the lifecycle tag so the reminder updates the live thread in
                    # place. No alert_once -> the companion app makes a sound once; it is
                    # routed to the finish channel (see _resolve_channel) for audibility.
                    "tag": self._lifecycle_tag,
                    "minutes_left": minutes_left,
                    "minutes": minutes_left,
                    "priority": "high",
                },
            )
            self._logger.info("Sent pre-completion notification: %s", msg)

    def _update_projected_energy(self) -> None:
        """Project total energy/cost for the running cycle.

        Prefers the on-device ``total_energy`` regressor (which models energy's
        non-linear accumulation); otherwise falls back to
        ``energy_so_far / progress_fraction`` (progress already carries the ML
        remaining-time blend, so it personalizes to this device's real cycle
        length). Cost uses the same price resolution that freezes each completed
        cycle's cost, so a running estimate and the final frozen value are
        consistent. Clears to ``None`` when progress is too low, there is no energy
        yet, or projection would be implausible. Never raises — a projection
        failure must not disturb the estimate loop.
        """
        try:
            trace = self.detector.get_power_trace()
            energy_so_far = float(
                getattr(self.detector, "_energy_since_idle_wh", 0.0) or 0.0
            )
            price = self._resolve_energy_price()
        except Exception:  # noqa: BLE001 - projection must never break estimates
            self._projected_energy_wh = None
            self._projected_cost = None
            return
        live_cost = self._live_cost_so_far(trace)
        wh, cost = progress_mod.projected_energy(
            self.profile_store,
            self.config_entry.options,
            float(self._matched_profile_duration or 0.0),
            trace,
            self._current_program,
            float(self._cycle_progress or 0.0),
            energy_so_far,
            price,
            self._profile_end_expectation,
            self._logger,
            cost_so_far=live_cost[0] if live_cost else None,
            cost_so_far_wh=live_cost[1] if live_cost else None,
        )
        self._projected_energy_wh = wh
        self._projected_cost = cost

    def _live_cost_so_far(
        self, trace: list[tuple[datetime, float]]
    ) -> tuple[float, float] | None:
        """``(cost, charged_wh)`` incurred so far at the prices the cycle ran through.

        Returns ``None`` when dynamic pricing is off or nothing has been recorded
        yet, which puts :func:`progress.projected_energy` back on the flat-price
        formula. The second element is the energy that cost was charged for, which
        the projection must subtract instead of ``energy_so_far``: this integrates
        the trace, while ``energy_so_far`` is the detector's per-reading
        accumulator, and the two treat outages and sub-threshold intervals
        differently. Never raises.
        """
        if not self._dynamic_pricing_enabled() or not self._price_timeline:
            return None
        try:
            if len(trace) < 2 or self._cycle_start_time is None:
                return None
            start_ts = self._cycle_start_time.timestamp()
            timestamps = np.asarray([t.timestamp() - start_ts for t, _ in trace], dtype=float)
            power = np.asarray([p for _, p in trace], dtype=float)
            points = compact_price_timeline(
                [(ts - start_ts, price) for ts, price in self._price_timeline],
                max_points=PRICE_TIMELINE_MAX_POINTS,
                decimals=PRICE_TIMELINE_PRICE_DECIMALS,
            )
            max_gap_s = energy_gap_threshold_s(timestamps)
            result = cycle_cost(timestamps, power, points, max_gap_s=max_gap_s)
            if result is None:
                return None
            # integrate_wh is the same total the price segments sum to, by the
            # documented contract of integrate_wh_by_price.
            return result[0], float(integrate_wh(timestamps, power, max_gap_s=max_gap_s))
        except Exception:  # noqa: BLE001 - projection must never break estimates
            return None

    def _update_cycle_anomaly(self, duration_so_far: float) -> None:
        """Flag a *soft* runtime overrun anomaly for the running cycle.

        Sets ``_overrun_ratio = elapsed / expected`` and ``_cycle_anomaly`` to
        ``"overrun"`` once the ratio crosses ``CYCLE_OVERRUN_ANOMALY_RATIO``. This
        is purely a visible signal (state-sensor attribute + cycle metadata); it
        never notifies and never terminates (the zombie-killer owns hard limits).
        No-op / cleared when no profile duration is known. Never raises.
        """
        self._overrun_ratio, self._cycle_anomaly = progress_mod.cycle_anomaly(
            self._matched_profile_duration, duration_so_far
        )

    def _update_remaining_only(self) -> None:
        """Recompute remaining/progress using phase-aware estimation."""
        # Throttle updates and only clear on truly dead states
        if self.detector.state in (STATE_OFF, STATE_UNKNOWN, STATE_IDLE):
            self._time_remaining = None
            self._total_duration = None
            self._cycle_progress = 0.0
            self._smoothed_progress = 0.0
            self._projected_energy_wh = None
            self._projected_cost = None
            self._cycle_anomaly = "none"
            self._overrun_ratio = 0.0
            self._envelope_position = None
            return

        now = utc_now()
        # The 5 s throttle guards the heavy phase estimate, but the FIRST estimate
        # after a match must not wait it out (#437): the match callback calls this
        # and then _check_live_progress_notification(), which now stays in the
        # waiting branch while _time_remaining is None. Bypassing once per match
        # means the live notification switches to a real countdown immediately
        # instead of holding the waiting message for another interval.
        first_estimate_after_match = (
            self._time_remaining is None
            and bool(self._matched_profile_duration)
            and self._matched_profile_duration > 0
        )
        prev_estimate_at = self._last_phase_estimate_time
        if (
            prev_estimate_at
            and (now - prev_estimate_at).total_seconds() < 5.0
            and not first_estimate_after_match
        ):
            return
        self._last_phase_estimate_time = now

        # Use net elapsed (wall-clock minus user-paused time) for all time estimates
        # so that paused time is excluded from progress / remaining / total duration.
        duration_so_far = float(self.net_elapsed_seconds)
        self._check_cycle_timers(duration_so_far)

        # An async match result can land AFTER the detector has closed the cycle:
        # the cycle start is cleared by then, so elapsed reads 0 while the matched
        # duration and the smoothed progress are still set. The back-calculation
        # below would then publish a full fresh "remaining" (raw progress 0 damped
        # against a ~90% EMA) over the finished cycle's terminal values - measured
        # live: remaining jumped from 15 min to 28 min and total duration from
        # 164 min to 28 min, 7 s before the finish notification. There is nothing to
        # estimate without an open cycle, and the terminal values must stand.
        if duration_so_far <= 0.0:
            return

        # Item 514: a halt is not progress either. The detector's programme view
        # leaves its stalls out of the elapsed time and the trace, like the
        # Playground replay (`progress_elapsed_s` / `progress_trace`); the cycle
        # timers above and the projected energy below keep the real figures.
        prog_elapsed = getattr(self.detector, "progress_elapsed_s", None)
        prog_elapsed = prog_elapsed(duration_so_far, now) if callable(prog_elapsed) else None
        if isinstance(prog_elapsed, (int, float)):
            duration_so_far = float(prog_elapsed)

        if not (self._matched_profile_duration and self._matched_profile_duration > 0):
            # No profile matched - don't provide misleading time estimates.
            self._time_remaining = None
            self._total_duration = None
            self._cycle_progress = 0.0
            self._smoothed_progress = 0.0
            self._projected_energy_wh = None
            self._projected_cost = None
            self._cycle_anomaly = "none"
            self._overrun_ratio = 0.0
            self._envelope_position = None
            self._logger.debug(
                "No profile matched yet, elapsed=%smin", int(duration_so_far / 60)
            )
            return

        # Compute the phase-aware progress input via the manager's own wrapper
        # (so per-call caching + test mocks apply), then hand it to the shared
        # pure smoothing/back-calc in :mod:`progress` - the identical math the
        # Playground simulation runs.
        trace = self.detector.get_power_trace()
        prog_trace = getattr(self.detector, "progress_trace", None)
        prog_trace = prog_trace(now) if callable(prog_trace) else None
        if isinstance(prog_trace, list):
            trace = prog_trace  # item 514, see above
        phase_result = None
        if len(trace) >= 10 and self._current_program != "detecting...":
            phase_result = self._estimate_phase_progress(
                trace, duration_so_far, self._current_program
            )

        result = progress_mod.compute_progress(
            self.device_type,
            float(self._matched_profile_duration),
            duration_so_far,
            progress_mod.ema_seed(
                self._smoothed_progress, self._smoothed_for_program, self._current_program
            ),
            phase_result,
            self._logger,
            # Real gap since the previous estimate, so the progress EMA keeps its
            # time constant instead of its step count - a plug that reports every
            # 30 s must not lag 6x further behind than one reporting every 5 s.
            dt_seconds=(
                (now - prev_estimate_at).total_seconds() if prev_estimate_at else None
            ),
        )

        self._cycle_progress = result.progress
        self._smoothed_progress = result.smoothed
        self._smoothed_for_program = self._current_program
        self._time_remaining = result.remaining
        self._total_duration = result.total
        self._update_projected_energy()
        self._update_cycle_anomaly(duration_so_far)

    def _check_cycle_timers(self, elapsed_seconds: float) -> None:
        """Fire any user-configured cycle timers whose offset has been reached."""
        if not self._notify_cycle_timers:
            return
        if self.detector.state not in (STATE_RUNNING, STATE_PAUSED):
            return

        elapsed_minutes = elapsed_seconds / 60.0
        for idx, timer in enumerate(self._notify_cycle_timers):
            if idx in self._fired_cycle_timers:
                continue
            offset = float(timer.get("offset_minutes", 0))
            if elapsed_minutes < offset:
                continue

            self._fired_cycle_timers.add(idx)
            raw_msg = timer.get("message") or ""
            fmt_kwargs = {
                "device": self.config_entry.title,
                "program": self._current_program or "",
                "minutes": int(offset),
            }
            msg = self._safe_format_template(
                raw_msg or self._timer_ui_strings.get("timer_default_message", "{device}: {minutes} min timer"),
                **fmt_kwargs,
            )
            auto_pause = bool(timer.get("auto_pause", False))
            timer_tag = f"{self._lifecycle_tag}_timer_{idx}"
            if auto_pause:
                # Defer the ENTIRE interactive notification until the pause takes
                # effect. The Resume action + sticky flag (and the action listener
                # that makes the button work) are all created together in
                # _setup_timer_pause_notification only on pause success, so a
                # no-op/failed pause never leaves a sticky "Resume Cycle" card with
                # a dead button. _check_cycle_timers is sync, so bridge via a task.
                self.hass.async_create_task(
                    self._async_auto_pause_and_notify(msg, timer_tag)
                )
            else:
                self._dispatch_notification(
                    msg,
                    event_type=NOTIFY_EVENT_TIMER,
                    extra_vars={"tag": timer_tag},
                    allow_deferral=False,
                    allow_presence_deferral=False,
                )
            self._logger.info(
                "Cycle timer #%d fired at %.0fs (%.1f min): %s",
                idx, elapsed_seconds, offset, msg,
            )

    async def _async_auto_pause_and_notify(self, msg: str, tag: str) -> None:
        """Pause the cycle for an auto-pause timer, then show the pause UI on success.

        The interactive pause notification is created only after the pause actually
        takes effect, so a no-op/failed pause never leaves a stale "paused" card.
        """
        if await self.async_pause_cycle():
            self._setup_timer_pause_notification(msg, tag)

    def _setup_timer_pause_notification(self, msg: str, tag: str) -> None:
        """Create the interactive pause notification: mobile card + HA sidebar + action listener.

        Called from _async_auto_pause_and_notify only after async_pause_cycle() has
        actually taken effect. Sends the interactive mobile notification (Resume
        action + sticky), creates the HA persistent notification for sidebar
        visibility, and registers the mobile action listener — all together, so the
        Resume button always has a live listener behind it and only ever appears
        when the cycle is genuinely paused.
        """
        self._clear_timer_pause_notification()

        # Interactive mobile notification — dispatched now (post-pause) rather than
        # at timer-fire time, so a failed/no-op pause never shows a dead Resume card.
        self._dispatch_notification(
            msg,
            event_type=NOTIFY_EVENT_TIMER,
            extra_vars={
                "tag": tag,
                "actions": [
                    {
                        "action": self._timer_pause_action_id,
                        "title": self._timer_ui_strings.get(
                            "timer_pause_action_title", "Resume Cycle"
                        ),
                    }
                ],
                "sticky": "true",
            },
            allow_deferral=False,
            allow_presence_deferral=False,
        )

        self._timer_pause_pn_id = tag
        self._timer_pause_mobile_tag = tag

        _body_suffix = self._timer_ui_strings.get(
            "timer_pause_body_suffix", "The cycle is paused. Open the WashData panel to resume."
        )
        _pn_create(
            self.hass,
            f"{msg}\n\n{_body_suffix}",
            title=f"WashData: {self.config_entry.title}",
            notification_id=tag,
        )

        action_id = self._timer_pause_action_id

        @callback
        def _on_mobile_action(event: Any) -> None:
            if event.data.get("action") == action_id:
                self.hass.async_create_task(self.async_resume_cycle())

        self._remove_timer_action_listener = self.hass.bus.async_listen(
            "mobile_app_notification_action",
            _on_mobile_action,
        )

    def _clear_timer_pause_notification(self) -> None:
        """Dismiss the active timer-pause notification (both HA persistent and mobile)."""
        if self._remove_timer_action_listener is not None:
            self._remove_timer_action_listener()
            self._remove_timer_action_listener = None

        if self._timer_pause_pn_id:
            _pn_dismiss(self.hass, self._timer_pause_pn_id)
            self._timer_pause_pn_id = None

        if self._timer_pause_mobile_tag:
            services = self._get_services_for_event(NOTIFY_EVENT_TIMER)
            mobile_services = [s for s in services if self._is_mobile_notify_service(s)]
            if mobile_services:
                self._send_notification_service(
                    _CLEAR_NOTIFICATION_MARKER,
                    services=mobile_services,
                    event_type=NOTIFY_EVENT_TIMER,
                    extra_vars={"tag": self._timer_pause_mobile_tag},
                )
            self._timer_pause_mobile_tag = None

    def _estimate_phase_progress(
        self,
        current_power_data: list[tuple[datetime, float]] | list[tuple[str, float]],
        current_duration: float,
        profile_name: str,
    ) -> tuple[float, float] | None:
        """Phase-aware progress estimate. Thin wrapper over :mod:`progress`."""
        return progress_mod.estimate_phase_progress(
            self.profile_store,
            current_power_data,
            current_duration,
            profile_name,
            self._logger,
            quiet_threshold_w=float(
                getattr(self.detector.config, "stop_threshold_w", 0.0) or 0.0
            ),
        )

    def _notify_update(self) -> None:
        """Notify entities of update."""
        async_dispatcher_send(self.hass, SIGNAL_WASHER_UPDATE.format(self.entry_id))

    def _notify_update_deferrable(self) -> None:
        """Notify, unless a power reading is being handled: its handler notifies
        once when it returns, after everything this caller changed."""
        if not getattr(self, "_in_power_event", False):
            self._notify_update()

    def notify_update(self) -> None:
        """Public method to notify entities of update."""
        self._notify_update()

    @property
    def is_user_paused(self) -> bool:
        """Return True if cycle is currently user-paused."""
        return self._is_user_paused

    @property
    def _is_user_paused(self) -> bool:
        return self._user_paused_flag

    @_is_user_paused.setter
    def _is_user_paused(self, value: bool) -> None:
        # Mirrored into the detector so Smart Termination honours a user pause too:
        # it was the one finisher that did not, and the pause itself carried the
        # elapsed time past the ratio (audit DETECT-05).
        self._user_paused_flag = bool(value)
        detector = getattr(self, "detector", None)
        setter = getattr(detector, "set_user_paused", None)
        if callable(setter):
            setter(self._user_paused_flag)

    @property
    def is_clean_state(self) -> bool:
        """Return True if machine is in Clean state (cycle ended, door not yet opened)."""
        return self._is_clean_state

    @property
    def net_elapsed_seconds(self) -> float:
        """Elapsed seconds in the current cycle, excluding user-paused time."""
        raw = float(self.detector.get_elapsed_seconds())
        paused = self._total_user_paused_seconds
        if self._user_pause_start is not None:
            paused += (utc_now() - self._user_pause_start).total_seconds()
        return max(0.0, raw - paused)

    def check_state(self):
        """Return the state entities show (the detector's exposed state)."""
        if self.recorder.is_recording:
            return STATE_RUNNING
        state = self.detector.state
        # The detector's display layer: a standby re-probe reads as off until it
        # has evidence (item 501), a stalled cycle as paused and a two-level
        # appliance at its standby level as idle (#452). A stand-in detector
        # without one shows its raw state.
        exposed = getattr(self.detector, "exposed_state", None)
        if isinstance(exposed, str):
            state = exposed
        # A completed cycle ends in STATE_FINISHED, not STATE_OFF; accept both
        # or the door-sensor Clean state (#153) is never surfaced (#282).
        if self._is_clean_state and state in (
            STATE_OFF,
            STATE_IDLE,
            STATE_FINISHED,
        ):
            return STATE_CLEAN
        if self._is_user_paused:
            return STATE_USER_PAUSED
        return state

    def list_phase_catalog(self, device_type: str) -> list[dict[str, Any]]:
        """Return the merged phase catalog for a device type."""
        return self.profile_store.list_phase_catalog(device_type)

    def get_profile_phase_ranges_for_device(
        self,
        profile_name: str,
        device_type: str,
    ) -> list[dict[str, Any]]:
        """Return phase ranges assigned to a profile for a given device type."""
        return self.profile_store.get_profile_phase_ranges_for_device(
            profile_name,
            device_type,
        )

    @property
    def sub_state(self) -> str | None:
        """Return more granular state info (e.g. current phase)."""
        if self.recorder.is_recording:
            return "Recording"
        # Item 501 / #452, as in check_state.
        exposed = getattr(self.detector, "exposed_sub_state", NotImplemented)
        if exposed is None or isinstance(exposed, str):
            return exposed
        return self.detector.sub_state

    @property
    def current_program(self):
        """Return the current program name."""
        return self._current_program

    @property
    def time_remaining(self):
        """Return estimated time remaining in seconds."""
        return self._time_remaining

    @property
    def total_duration(self) -> float | None:
        """Return total predicted duration in seconds."""
        return self._total_duration

    @property
    def cycle_progress(self):
        """Return cycle progress as a percentage."""
        return self._cycle_progress

    @property
    def projected_energy_wh(self) -> float | None:
        """Projected total energy (Wh) for the running cycle, or None."""
        return self._projected_energy_wh

    @property
    def projected_cost(self) -> float | None:
        """Projected total cost for the running cycle, or None when no price."""
        return self._projected_cost

    @property
    def cycle_anomaly(self) -> str:
        """Runtime anomaly state for the current cycle ("none" | "overrun" | "stalled").

        ``stalled`` (#452) wins while the detector shows the cycle stalled; like
        ``overrun`` it is visible only and never a notification.
        """
        if getattr(self.detector, "stalled", False) is True:
            return CYCLE_ANOMALY_STALLED
        return self._cycle_anomaly

    @property
    def overrun_ratio(self) -> float:
        """Elapsed / expected duration for the running cycle (0.0 when unknown)."""
        return self._overrun_ratio

    @property
    def envelope_position(self) -> float | None:
        """How far this run has mapped onto its profile's envelope, 0-1, or None.

        Produced by the DTW alignment that runs for the verified-pause decision,
        so it is only refreshed while power is below the stop threshold and a
        profile is matched - which is exactly the phase where elapsed time says
        least (a dishwasher sitting in its drying phase). Visible only; no
        detection path reads it.
        """
        return self._envelope_position

    @property
    def last_cycle_post_anomaly(self) -> dict:
        """Post-cycle anomaly data from the last completed cycle.

        Contains subset of keys present: anomaly (underrun/overrun/none),
        underrun_ratio, energy_anomaly (energy_spike/energy_low), energy_z_score.
        Empty dict when no completed cycle or no anomaly detected.
        """
        return self._last_cycle_post_anomaly

    @property
    def restart_gaps(self) -> list[dict]:
        """HA restart gaps recorded during the current active cycle (may be empty)."""
        return self._restart_gaps

    @property
    def maintenance_due(self) -> list[str]:
        """Ids of the maintenance tasks that are due (E2, #461): built-in types, then
        custom tasks.

        Surfaced as a state-sensor attribute, the Maintenance-due binary sensor and
        the panel banner. Never a notification. Returns an empty list on any error.
        """
        try:
            return self.profile_store.get_maintenance_due(
                effective_reminders(
                    self.device_type,
                    self.config_entry.options.get(CONF_MAINTENANCE_REMINDER_CYCLES),
                )
            )
        except Exception:  # noqa: BLE001
            return []

    @property
    def maintenance_status(self) -> list[dict[str, Any]]:
        """Every active maintenance reminder with its progress (#461). Never raises.

        Rows from ``ProfileStore.get_maintenance_status`` against the device-type
        aware reminder config (``maintenance.effective_reminders``).
        """
        try:
            return self.profile_store.get_maintenance_status(
                effective_reminders(
                    self.device_type,
                    self.config_entry.options.get(CONF_MAINTENANCE_REMINDER_CYCLES),
                )
            )
        except Exception:  # noqa: BLE001
            return []

    def _sync_maintenance_baselines(self) -> None:
        """Stamp/clear the preset reminder types' counting origin (#461). Never raises.

        Run on every setup and config reload, after the store is loaded, so a
        preset reminder that starts applying (an upgrade, or the user switching it
        on) counts from now rather than opening as due.
        """
        try:
            self.profile_store.sync_maintenance_baselines(
                effective_reminders(
                    self.device_type,
                    self.config_entry.options.get(CONF_MAINTENANCE_REMINDER_CYCLES),
                )
            )
        except Exception:  # noqa: BLE001
            self._logger.debug("Maintenance baseline sync failed", exc_info=True)

    @property
    def current_power(self):
        """Return current power reading in watts.

        Prefers the sensor's live state over the event cache (#409): the cache is
        only refreshed while a cycle is active or expiring, so an idle appliance
        whose plug reports rarely would otherwise show whatever value was last seen
        - which is what users compared against their HA sensor and found wrong.
        """
        live = self._live_power_state()
        return live[0] if live is not None else self._current_power

    @property
    def cycle_start_time(self) -> datetime | None:
        """Return the start time of the current cycle."""
        return self.detector.current_cycle_start

    @property
    def last_cycle_end_time(self) -> datetime | None:
        """Return when the most recent completed cycle ended (or None).

        Set at cycle end and restored from stored history on startup. Consumed by
        the conversation intent handler to answer "how long ago did it finish".
        """
        return self._last_cycle_end_time

    @property
    def last_match_details(self) -> dict[str, Any] | None:
        """Return details of the last profile match."""
        res = getattr(self, "_last_match_result", None)
        return res.to_dict() if res else None

    @property
    def samples_recorded(self):
        """Return the number of power samples recorded in current cycle."""
        return self.detector.samples_recorded

    @property
    def sample_interval_stats(self):
        """The detector's own cadence window (audit MANAGER-14: this dict was never
        filled, so the debug sensor's `sampling_p95` and diagnostics were empty)."""
        dts = list(getattr(self.detector, "_recent_dts", None) or [])
        if not dts:
            return {}
        return {
            "p95": round(percentile_linear(dts, 95), 2),
            "median": round(median_fast(dts), 2),
            "count": len(dts),
        }

    @property
    def pump_stuck(self) -> bool:
        """Return True if the pump stuck threshold has fired for the current cycle."""
        return self._pump_stuck

    @property
    def pump_runs_today(self) -> int:
        """Return the number of completed pump cycles that started in the last 24 hours.

        Counts all past cycles whose ``start_time`` falls within the rolling 24-hour
        window ending now.  Returns 0 for non-pump device types.
        """
        if self.device_type != DEVICE_TYPE_PUMP:
            return 0
        cutoff = utc_now().timestamp() - 86400.0
        count = 0
        for cycle in self.profile_store.get_past_cycles():
            start_raw = cycle.get("start_time")
            if not start_raw:
                continue
            try:
                if isinstance(start_raw, str):
                    parsed = dt_util.parse_datetime(start_raw)
                    if parsed is None:
                        continue
                    ts = parsed.timestamp()
                else:
                    ts = float(start_raw)
                if ts >= cutoff:
                    count += 1
            except (TypeError, ValueError, OverflowError):
                continue
        return count

    @property
    def cycle_count(self) -> int:
        """Return the total number of completed cycles stored for this device."""
        return len(self.profile_store.get_past_cycles())

    @property
    def lifetime_energy_kwh(self) -> float:
        """Lifetime accumulated energy (kWh) for the HA Energy dashboard sensor."""
        return round(self.profile_store.get_lifetime_energy_wh() / 1000.0, 3)

    @property
    def lifetime_cycle_count(self) -> int:
        """Cycles this appliance has run, ever - the odometer behind the count sensor.

        Distinct from :attr:`cycle_count`, which is ``len(retained history)`` and is
        the right basis for "do I have enough data yet" gates. This one only ever
        rises: it survives retention trimming, record deletion and a data wipe, so an
        "every N cycles" maintenance schedule (ours or an external integration's) can
        be built on it (#414).
        """
        return self._lifetime_cycle_count()

    @property
    def manual_program_active(self) -> bool:
        """Return True if a manual program override is active."""
        return getattr(self, "_manual_program_active", False)

    @property
    def armed_program(self) -> str | None:
        """The program pinned for the next cycle, when one is not under way (#411)."""
        return getattr(self, "_armed_program", None)

    def _resolve_profiles(self) -> dict[str, Any]:
        """Return the stored profiles mapping, falling back to the raw store dict.

        Shared by the manual-program set and re-arm paths so both agree on what
        counts as an existing program. Never raises.
        """
        profiles_raw: Any = None
        try:
            profiles_raw = self.profile_store.get_profiles()
        except Exception:  # pylint: disable=broad-exception-caught
            profiles_raw = None

        if isinstance(profiles_raw, dict):
            return cast(dict[str, Any], profiles_raw)
        profiles_fallback = getattr(self.profile_store, "_data", {}).get("profiles", {})
        return (
            cast(dict[str, Any], profiles_fallback)
            if isinstance(profiles_fallback, dict)
            else {}
        )

    def set_manual_program(self, profile_name: str) -> bool:
        """Pin a program to the current cycle, or arm it for the next one (#411).

        Returns True when the choice was accepted. It used to return silently
        unless the detector was exactly in ``running``, which meant selecting a
        program on an idle appliance (the overwhelmingly common case: the panel
        offers the dropdown at all times) did nothing at all, reported success to
        the caller because there was no return value to test, and logged nothing.
        The selection then snapped back to auto-detect on the next refresh.

        Now every state is accepted. While a cycle is under way the program is
        applied to it immediately; otherwise it is armed and applied the moment
        the next cycle starts, which is what someone picking a program on an idle
        machine means by it. The only rejection left is a program that does not
        exist, and that one is reported rather than swallowed.
        """
        profiles = self._resolve_profiles()

        if profile_name not in profiles:
            self._logger.warning("Cannot set manual program: '%s' not found", profile_name)
            return False

        in_progress = self.detector.state in _CYCLE_IN_PROGRESS_STATES
        # The arm survives only where the next cycle still needs it. Applied to a
        # cycle already under way the pin belongs to THAT cycle, and leaving it armed
        # let a back-to-back load inherit it: the cycle-end tail does try to clear it
        # ("a pin is for the cycle it was made for") but sits behind the new-cycle
        # token guard and returns early in exactly that case, so _consume_armed_program
        # would stamp an unrelated cycle `label_source = "manual"` and let it reshape
        # that program's envelope.
        #
        # STARTING is the one in-progress state that must KEEP the arm: the
        # STARTING -> RUNNING transition resets the live pin as it starts the new
        # cycle, and _consume_armed_program is what puts it back.
        keep_arm = (not in_progress) or self.detector.state == STATE_STARTING
        armed = profile_name if keep_arm else None
        # Resolved before persisting, so this is ONE store write. Setting and then
        # clearing scheduled two async_set_armed_program tasks and two saves for
        # every mid-cycle pin.
        self._armed_program = armed
        self._persist_armed_program(armed)

        if in_progress:
            self._apply_manual_program(profile_name, profiles.get(profile_name))
        else:
            self._logger.info(
                "Program %r armed; it will be applied when the next cycle starts",
                profile_name,
            )
        return True

    @staticmethod
    def _profile_duration(value: Any) -> float | None:
        """A profile's expected duration in seconds, or None when unusable.

        None is the field's declared "unknown" and every reader of
        ``_matched_profile_duration`` already guards for it. Non-finite is rejected
        for the same reason ``_finite_power`` rejects it: ``inf`` survives a plain
        ``> 0`` test, and ``sensor.py``'s ``int(time_remaining / 60)`` then raises
        OverflowError on every update. A profile written by this device is always
        finite; an imported or hand-edited one need not be (register items 211/229).

        OverflowError is caught alongside the rest because ``json`` keeps an
        oversized integer literal as an unbounded ``int``: ``float(10**400)``
        raises instead of returning ``inf``, so the non-finite filter below is
        never reached and the raise escapes a ``@callback`` WS handler. ``1e400``
        parses to ``inf`` and is the case the filter covers; the two are different
        inputs (register items 279/280). The rule is ``match_rules.profile_duration``,
        shared with the switching rules and the Playground replay.
        """
        return match_rules.profile_duration(value)

    def _apply_manual_program(
        self, profile_name: str, profile: dict[str, Any] | None
    ) -> None:
        """Pin *profile_name* to the cycle in progress. Shared by set and re-arm."""
        self._current_program = profile_name
        self._manual_program_active = True

        # Update expected duration immediately. A profile with nothing learned yet
        # (hand-created, or imported before its first cycle) must CLEAR the duration
        # rather than leave the previously matched program's behind: the pin is
        # applied mid-cycle, so the ETA and progress would go on describing the
        # program the user just replaced. None is this field's established "unknown"
        # value and every reader guards for it. Same shape as the restart path that
        # re-pins a manual program (see the #404 secondary-bug block above), which
        # already got this right.
        avg = self._profile_duration(profile.get("avg_duration")) if profile else None
        self._matched_profile_duration = avg
        if avg:
            self._logger.info(
                "Manual program set to %s, duration=%.0fs", profile_name, avg
            )
        else:
            self._logger.info(
                "Manual program set to %s; it has no learned duration yet, so the "
                "time estimate stays unknown until it does",
                profile_name,
            )

        # Refresh whatever estimate the current state exposes, then publish. Runs for
        # the cleared case too, so a stale remaining time is not left on display until
        # the next tick.
        #
        # Every live state, not just RUNNING: set_manual_program applies the pin in
        # PAUSED and ENDING as well (_CYCLE_IN_PROGRESS_STATES), and neither
        # select.py nor _update_remaining_only publishes on its own, so on the
        # select-entity path nothing reached the sensors at all. The phase
        # estimator's 5 s throttle is bypassed because this is a user action, not a
        # tick, and the value it invalidates is on screen right now.
        #
        # STARTING is deliberately excluded from the refresh: it exposes no estimate,
        # and _update_estimates() treats it as a dead state - it would reset
        # _current_program to "off" and undo the pin we just applied.
        state = self.detector.state
        if state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            self._last_phase_estimate_time = None
            if state == STATE_RUNNING:
                self._update_estimates()
            else:
                self._update_remaining_only()
        self._notify_update()

    def _persist_armed_program(self, profile_name: str | None) -> None:
        """Persist the armed program without blocking the caller. Never raises."""
        try:
            # Tracked, not bare async_create_task: this writes to the ProfileStore
            # and saves, so a reload/unload that swaps the store out mid-flight must
            # be able to cancel it rather than let it write to the stale one. That is
            # the documented rule for every store-touching fire-and-forget in this
            # class; this call site was the one that did not follow it.
            self._spawn_tracked(
                self.profile_store.async_set_armed_program(profile_name)
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.debug("Could not persist the armed program", exc_info=True)

    def clear_armed_program(self) -> bool:
        """Drop any program armed for the next cycle. True if one was armed.

        `_armed_program` is the authoritative copy; the store key is only there so
        an arm survives a restart. So every site that retires an arm has to clear
        BOTH, and there were three inline copies of this pair plus one place that
        cleared only the store - the wipe (`clear_all_data` pops `armed_program`,
        `ws_wipe_history` never touched the field), which left a pre-wipe pin ready
        to be re-applied as soon as a profile of the same name existed again.
        """
        if self._armed_program is None:
            return False
        self._armed_program = None
        self._persist_armed_program(None)
        return True

    def _consume_armed_program(self) -> bool:
        """Apply an armed program to the cycle that just started, if there is one.

        Called from the new-cycle reset, which is also what would otherwise wipe a
        program pinned during STARTING. Returns True when one was applied, so the
        caller knows not to fall back to "detecting...".
        """
        name = self._armed_program
        if not name:
            return False
        profiles = self._resolve_profiles()
        if name not in profiles:
            # Deleted between arming and starting: drop it rather than pinning a
            # program that no longer exists.
            self._logger.info(
                "Armed program %r no longer exists; reverting to auto-detect", name
            )
            self.clear_armed_program()
            return False
        self._apply_manual_program(name, profiles.get(name))
        self.clear_armed_program()
        self._logger.info("Applied armed program %r to the cycle just started", name)
        return True

    async def async_pause_cycle(self) -> bool:
        """Pause the current cycle (user-triggered).

        Sets verified_pause so the cycle is not finalized when power drops.
        Optionally cuts power to the switch entity if CONF_PAUSE_CUTS_POWER is enabled.

        Returns True if the cycle was paused, False if it was a no-op (wrong state).
        """
        if self.detector.state not in (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING):
            self._logger.debug(
                "async_pause_cycle: ignored (detector state=%s)", self.detector.state
            )
            return False

        if self._is_user_paused:
            self._logger.debug("async_pause_cycle: already user-paused, ignoring")
            return False

        self._logger.info("Cycle paused by user")
        prev_verified = self.detector._verified_pause
        self._is_user_paused = True
        self._user_pause_start = utc_now()
        self.detector.set_verified_pause(True)

        if self._pause_cuts_power:
            switch_entity = self.config_entry.options.get(
                CONF_SWITCH_ENTITY
            ) or self.config_entry.data.get(CONF_SWITCH_ENTITY)
            if switch_entity:
                self._logger.info(
                    "pause_cuts_power: turning off switch %s", switch_entity
                )
                try:
                    await self.hass.services.async_call(
                        "switch", "turn_off", {"entity_id": switch_entity}, blocking=True
                    )
                except HomeAssistantError as err:
                    self._logger.warning(
                        "pause_cuts_power: failed to turn off %s: %s - rolling back pause state",
                        switch_entity, err,
                    )
                    self._is_user_paused = False
                    self._user_pause_start = None
                    self.detector.set_verified_pause(prev_verified)
                    return False

        # A user pause overrides an auto-open dwell: cancel it so the pending timer
        # can't finalize the cycle out from under the pause (#342).
        self._cancel_door_end_dwell()

        snapshot = self._augment_active_snapshot(self.detector.get_state_snapshot())
        self._spawn_tracked(self.profile_store.async_save_active_cycle(snapshot))
        self._notify_update()
        return True

    async def async_resume_cycle(self) -> bool:
        """Resume a user-paused cycle.

        Accumulates elapsed paused time and clears the verified pause flag.
        Optionally restores power via the switch entity if CONF_PAUSE_CUTS_POWER is enabled.

        Returns True if the cycle was resumed, False if it was a no-op (not paused).
        """
        if not self._is_user_paused:
            self._logger.debug("async_resume_cycle: not user-paused, ignoring")
            return False

        now = utc_now()
        prev_pause_start = self._user_pause_start
        accumulated = (
            (now - prev_pause_start).total_seconds()
            if prev_pause_start is not None else 0.0
        )

        self._total_user_paused_seconds += accumulated
        self._user_pause_start = None
        self._is_user_paused = False
        self.detector.set_verified_pause(False)
        self._logger.info(
            "Cycle resumed by user (total paused: %.0fs)", self._total_user_paused_seconds
        )

        if self._pause_cuts_power:
            switch_entity = self.config_entry.options.get(
                CONF_SWITCH_ENTITY
            ) or self.config_entry.data.get(CONF_SWITCH_ENTITY)
            if switch_entity:
                self._logger.info(
                    "pause_cuts_power: turning on switch %s", switch_entity
                )
                try:
                    await self.hass.services.async_call(
                        "switch", "turn_on", {"entity_id": switch_entity}, blocking=True
                    )
                except HomeAssistantError as err:
                    self._logger.warning(
                        "pause_cuts_power: failed to turn on %s: %s - rolling back resume state",
                        switch_entity, err,
                    )
                    self._total_user_paused_seconds -= accumulated
                    self._user_pause_start = prev_pause_start
                    self._is_user_paused = True
                    self.detector.set_verified_pause(True)
                    return False

        # Dismiss the interactive pause notification only after the resume (incl. the
        # switch turn-on) has actually succeeded — a rolled-back resume above returns
        # early with the card still up, matching the real (still-paused) state.
        self._clear_timer_pause_notification()

        snapshot = self._augment_active_snapshot(self.detector.get_state_snapshot())
        self._spawn_tracked(self.profile_store.async_save_active_cycle(snapshot))
        self._notify_update()
        return True

    async def async_terminate_cycle(self) -> None:
        """Force terminate the current cycle via user request."""
        self._logger.warning("Force terminating cycle by user request")

        # A manual recording pins the shown state at running (check_state) while the
        # detector is fed nothing, so user_stop() alone was a no-op and a forgotten
        # recording kept the device "running" through restarts (#376, #383). Stop it
        # as the Stop Recording button does: the run is kept for processing.
        if self.recorder.is_recording:
            self._logger.info("Force terminate: stopping the active manual recording")
            await self.recorder.stop_recording()

        # Trigger natural cycle end via detector
        # This will call _on_cycle_end callback, which handles:
        # - Saving to profile store
        # - Clearing active cycle persistence
        # - Post-processing/Merging
        # - Notifications
        self.detector.user_stop()

        # We DO NOT clear manager state manually here (e.g. self._current_program)
        # because we want the UI to show the "Clean" state with the just-finished
        # program info. The standard reset timers in _on_cycle_end /
        # _async_power_changed will handle cleanup after delay.

        # Force a state update to reflect the change immediately
        self._notify_update()

    async def async_start_recording(self) -> None:
        """Start manual recording of a cycle."""
        if self.recorder.is_recording:
            self._logger.warning("Already recording")
            return

        # Ensure we are in a clean state (stop any running cycle first?)
        # If running, user should probably stop it? Or force stop?
        # Plan said "unregulated", so we just start recording.
        # But if cycle_detector thinks it's running, we should probably "pause" it
        # or just override state. My override in checks_state handles UI.
        # But should we clear current program?
        if self.detector.state != "off":
            self._logger.info("Forcing detector reset before recording")
            self.detector.reset()

        await self.recorder.start_recording()
        self._notify_update()

    async def async_stop_recording(self) -> None:
        """Stop manual recording."""
        if not self.recorder.is_recording:
            return

        await self.recorder.stop_recording()
        self._notify_update()

    def clear_manual_program(self) -> None:
        """Clear the manual program override, live or merely armed (#411).

        Previously bailed out unless a pin was active on a running cycle, which
        left an armed program stuck: picking "Auto-detect" on an idle appliance
        could not undo a choice made a moment earlier.
        """
        had_arm = self._armed_program is not None
        if had_arm:
            self._armed_program = None
            self._persist_armed_program(None)

        if not self._manual_program_active:
            if had_arm:
                self._notify_update()
                self._logger.info("Armed program cleared, reverting to auto-detection")
            return

        self._manual_program_active = False
        # A live cycle goes back to auto-detection. PAUSED and ENDING are live too
        # (audit MANAGER-15): comparing against "running" alone showed "off" for a
        # cycle still under way. The refresh mirrors set_manual_program's.
        state = self.detector.state
        self._matched_profile_duration = None
        if state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            self._current_program = "detecting..."
            self._last_phase_estimate_time = None
            if state == STATE_RUNNING:
                self._update_estimates()  # Trigger immediate re-detection attempt
            else:
                self._update_remaining_only()
        else:
            # No cycle under way: clear the forced program
            self._current_program = "off"

        self._notify_update()
        self._logger.info("Manual program cleared, reverting to auto-detection")

    async def _run_post_cycle_processing(self, profiles: Any = ()) -> None:
        """Refresh what the cycle that just ended invalidated (``profiles``' artifacts).

        Not the full maintenance any more (register item 456): that rebuilt every
        envelope, recomputed every cycle's artifacts and re-matched every unlabelled
        cycle at each cycle end. Those global passes run nightly.
        """
        try:
            stats = await self.profile_store.async_post_cycle_refresh(profiles)
            if stats.get("orphaned_profiles"):
                self._logger.info(
                    "Post-cycle processing: removed %s orphaned profile(s)",
                    stats["orphaned_profiles"],
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Post-cycle processing failed: %s", e)