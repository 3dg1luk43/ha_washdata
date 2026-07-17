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
import inspect
import math
import uuid
from asyncio import Task
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
import numpy as np

if TYPE_CHECKING:
    from .store import StoreBridge

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Context, Event, HomeAssistant, State, callback
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.exceptions import HomeAssistantError
from homeassistant.const import STATE_UNAVAILABLE, STATE_HOME
from homeassistant.util import dt as dt_util
import homeassistant.helpers.event as evt
from homeassistant.helpers import script as script_helper
from homeassistant.helpers import translation

from .const import (
    DOMAIN,
    CONF_POWER_SENSOR,
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
    CONF_SMOOTHING_WINDOW,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_INTERRUPTED_MIN_SECONDS,
    CONF_ABRUPT_DROP_WATTS,
    CONF_ABRUPT_DROP_RATIO,
    CONF_ABRUPT_HIGH_LOAD_FACTOR,
    CONF_PROGRESS_RESET_DELAY,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_AUTO_MAINTENANCE,
    CONF_MAINTENANCE_REMINDER_CYCLES,
    DEFAULT_MAINTENANCE_REMINDER_CYCLES,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_MAX_PAST_CYCLES,
    CONF_MAX_FULL_TRACES_PER_PROFILE,
    CONF_MAX_FULL_TRACES_UNLABELED,
    CONF_WATCHDOG_INTERVAL,
    CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_PROFILE_UNMATCH_THRESHOLD,
    CONF_DEVICE_TYPE,
    CONF_START_DURATION_THRESHOLD,
    CONF_RUNNING_DEAD_ZONE,
    CONF_END_REPEAT_COUNT,
    CONF_MIN_OFF_GAP,
    CONF_START_ENERGY_THRESHOLD,
    CONF_END_ENERGY_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_POWER_OFF_DELAY,
    CONF_SAMPLING_INTERVAL,
    CONF_SAVE_DEBUG_TRACES,
    CONF_DTW_BANDWIDTH,
    CONF_EXTERNAL_END_TRIGGER_ENABLED,
    CONF_EXTERNAL_END_TRIGGER,
    CONF_EXTERNAL_END_TRIGGER_INVERTED,
    CONF_ANTI_WRINKLE_ENABLED,
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_ANTI_WRINKLE_MAX_DURATION,
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_DELAY_START_DETECT_ENABLED,
    CONF_DELAY_CONFIRM_SECONDS,
    CONF_DELAY_TIMEOUT_HOURS,
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
    DEFAULT_MIN_POWER,
    DEFAULT_OFF_DELAY,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_PROFILE_DURATION_TOLERANCE,
    DEFAULT_INTERRUPTED_MIN_SECONDS,
    DEFAULT_ABRUPT_DROP_WATTS,
    DEFAULT_ABRUPT_DROP_RATIO,
    DEFAULT_ABRUPT_HIGH_LOAD_FACTOR,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEFAULT_NOTIFY_BEFORE_END_MINUTES,
    DEFAULT_PROFILE_MATCH_THRESHOLD,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
    DEFAULT_SAMPLING_INTERVAL,
    DEFAULT_PROGRESS_RESET_DELAY,
    DEFAULT_POWER_OFF_THRESHOLD_W,
    DEFAULT_POWER_OFF_DELAY,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_AUTO_MAINTENANCE,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE,
    DEFAULT_ANTI_WRINKLE_ENABLED,
    DEFAULT_ANTI_WRINKLE_MAX_POWER,
    DEFAULT_ANTI_WRINKLE_MAX_DURATION,
    DEFAULT_ANTI_WRINKLE_EXIT_POWER,
    DEFAULT_DELAY_START_DETECT_ENABLED,
    DEFAULT_DELAY_CONFIRM_SECONDS,
    DEFAULT_DELAY_TIMEOUT_HOURS,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_MAX_PAST_CYCLES,
    DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
    CONF_NOTIFY_TITLE,
    CONF_NOTIFY_ICON,
    CONF_NOTIFY_START_MESSAGE,
    CONF_NOTIFY_FINISH_MESSAGE,
    CONF_NOTIFY_PRE_COMPLETE_MESSAGE,
    CONF_NOTIFY_LIVE_INTERVAL_SECONDS,
    CONF_NOTIFY_LIVE_OVERRUN_PERCENT,
    CONF_NOTIFY_LIVE_CHRONOMETER,
    CONF_NOTIFY_REMINDER_MESSAGE,
    CONF_NOTIFY_TIMEOUT_SECONDS,
    CONF_NOTIFY_CHANNEL,
    CONF_NOTIFY_FINISH_CHANNEL,
    CONF_ENERGY_PRICE_STATIC,
    CONF_ENERGY_PRICE_ENTITY,
    CONF_PEAK_RATE_THRESHOLD,
    CONF_PEAK_RATE_MESSAGE,
    DEFAULT_PEAK_RATE_MESSAGE,
    CONF_DOOR_SENSOR_ENTITY,
    CONF_PAUSE_CUTS_POWER,
    CONF_SWITCH_ENTITY,
    CONF_NOTIFY_UNLOAD_DELAY_MINUTES,
    CONF_NOTIFY_UNLOAD_MESSAGE,
    DEFAULT_NOTIFY_UNLOAD_DELAY_MINUTES,
    DEFAULT_NOTIFY_UNLOAD_MESSAGE,
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

    DEFAULT_MAX_FULL_TRACES_UNLABELED,
    DEFAULT_DTW_BANDWIDTH,
    DEFAULT_WATCHDOG_INTERVAL,
    CONF_MATCH_PERSISTENCE,
    DEFAULT_MATCH_PERSISTENCE,
    DEFAULT_MATCH_REVERT_RATIO,
    DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_START_DURATION_THRESHOLD,
    DEFAULT_RUNNING_DEAD_ZONE,
    DEFAULT_END_REPEAT_COUNT,
    DEFAULT_MIN_OFF_GAP,
    DEFAULT_MIN_OFF_GAP_BY_DEVICE,
    DEFAULT_MAX_DEFERRAL_SECONDS,
    DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE,
    DEFAULT_END_ENERGY_THRESHOLD,
    DEVICE_COMPLETION_THRESHOLDS,
    CYCLE_UNDERRUN_ANOMALY_RATIO,
    ENERGY_ANOMALY_Z_THRESHOLD,
    TERMINAL_DROP_MIN_CLEAN_CYCLES,
    TERMINAL_DROP_MIN_QUIET_SPAN_S,
    TERMINAL_DROP_EARLINESS_RATIO,
    TERMINAL_DROP_MIN_PEAK_RATIO,
    TERMINAL_DROP_PEAK_FAMILIAR_TOL,
    ML_MATCH_COMMIT_THRESHOLD,
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
from .cycle_detector import CycleDetector, CycleDetectorConfig
from .learning import LearningManager
from .profile_store import (
    ProfileStore,
    decompress_power_data,
    is_terminal_drop,
    terminal_drop_baseline,
)
from .signal_processing import integrate_wh, energy_gap_threshold_s
from .recorder import CycleRecorder
from .diag_buffer import DiagBuffer
from .log_utils import DeviceLoggerAdapter
from .time_utils import power_data_to_offsets
from . import progress as progress_mod
from . import notification_rules as notif_rules

_LOGGER = logging.getLogger(__name__)

# Finish-type notification events that would wake someone and are therefore gated by
# the quiet-hours (do-not-disturb) window. Live-progress ticks (NOTIFY_EVENT_LIVE)
# and the start notification (NOTIFY_EVENT_START) are intentionally excluded.
_QUIET_HOURS_EVENT_TYPES = frozenset(
    {NOTIFY_EVENT_FINISH, NOTIFY_EVENT_CLEAN, "pre_complete"}
)


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
    "subtitle",
    "content_state",
    "activity",
)


def _pn_create(
    hass: HomeAssistant,
    message: str,
    *,
    title: str | None = None,
    notification_id: str | None = None,
) -> None:
    """Best-effort persistent notification creation.

    Deliberately goes through ``hass.components.persistent_notification`` rather than
    a direct ``homeassistant.components.persistent_notification`` import: the test
    suite stubs out the whole ``homeassistant`` module and mocks this dynamic
    attribute, so a direct import would both fail under test and bypass those mocks.
    Failures are logged at debug (not silently swallowed) so a stuck notification is
    at least visible in the logs.
    """
    try:
        components = getattr(cast(Any, hass), "components", None)
        pn = getattr(cast(Any, components), "persistent_notification", None)
        if pn is None:
            return
        result = pn.async_create(message, title=title, notification_id=notification_id)
        if inspect.iscoroutine(result):
            hass.async_create_task(result)
    except Exception:  # noqa: BLE001 - best-effort; surface the failure in logs
        _LOGGER.debug("persistent_notification create failed (id=%s)", notification_id, exc_info=True)


def _pn_dismiss(hass: HomeAssistant, notification_id: str) -> None:
    """Best-effort persistent notification dismissal.

    Uses the ``hass.components`` accessor for the same test-mocking reason as
    :func:`_pn_create`; failures are logged at debug rather than swallowed.
    """
    try:
        components = getattr(cast(Any, hass), "components", None)
        pn = getattr(cast(Any, components), "persistent_notification", None)
        if pn is None:
            return
        result = pn.async_dismiss(notification_id)
        if inspect.iscoroutine(result):
            hass.async_create_task(result)
    except Exception:  # noqa: BLE001 - best-effort; surface the failure in logs
        _LOGGER.debug("persistent_notification dismiss failed (id=%s)", notification_id, exc_info=True)


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
        self.device_type = config_entry.options.get(
            CONF_DEVICE_TYPE,
            config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
        )

        # Initialize attributes to satisfy pylint
        self._off_delay = float(DEFAULT_OFF_DELAY)
        self._no_update_active_timeout = float(DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT)
        self._low_power_no_update_timeout = 3600.0 # Default 1h
        self._notify_before_end_minutes = float(DEFAULT_NOTIFY_BEFORE_END_MINUTES)
        self._notify_start_services: list[str] = []
        self._notify_finish_services: list[str] = []
        self._notify_live_services: list[str] = []
        self._notify_actions: list[dict[str, Any]] = []
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
        self._notify_timeout_seconds = DEFAULT_NOTIFY_TIMEOUT_SECONDS
        self._pending_notifications: list[dict[str, Any]] = []
        # Quiet-hours (do-not-disturb) hold queue + release timer. Finish-type
        # notifications that would fire inside the window are parked here and flushed
        # at the end of the window by a single async_call_later timer.
        self._quiet_pending_notifications: list[dict[str, Any]] = []
        self._remove_quiet_hours_timer: Any | None = None
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

        # Pause tracking (user-triggered)
        self._user_pause_start: datetime | None = None
        self._total_user_paused_seconds: float = 0.0
        self._is_user_paused: bool = False
        self._pause_cuts_power: bool = bool(
            config_entry.options.get(CONF_PAUSE_CUTS_POWER, False)
        )

        # Door sensor + clean state
        self._door_sensor_entity: str | None = config_entry.options.get(
            CONF_DOOR_SENSOR_ENTITY
        ) or None
        self._remove_door_sensor_listener = None
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
        self._lifecycle_pn_id = self._lifecycle_tag
        self._clean_tag = f"ha_washdata_{self.entry_id}_clean"
        # Backwards-compatible alias for existing live-notification call sites/tests.
        self._live_notification_tag = self._lifecycle_tag
        self._start_event_fired = False
        self._cycle_start_time: datetime | None = None
        # Per-cycle UUID used to key ranking snapshots; prevents cross-contamination
        # between cycles that happen to share the same second-resolution start_time.
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
        self._noise_events: list[datetime] = []
        self._noise_max_powers: list[float] = []
        self._last_match_result = None
        self._last_phase_estimate_time = None
        self._sample_intervals: list[float] = []
        self._sample_interval_stats: dict[str, Any] = {}
        self._matching_task: Task[Any] | None = None
        self._last_state_save = 0.0
        self._last_cycle_end_time: datetime | None = None
        self._remove_state_expiry_timer = None

        # Components
        match_threshold = config_entry.options.get(
            CONF_PROFILE_MATCH_THRESHOLD, DEFAULT_PROFILE_MATCH_THRESHOLD
        )
        unmatch_threshold = config_entry.options.get(
            CONF_PROFILE_UNMATCH_THRESHOLD, DEFAULT_PROFILE_UNMATCH_THRESHOLD
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
            match_threshold=match_threshold,
            unmatch_threshold=unmatch_threshold,
            device_name=config_entry.title,
        )
        self.profile_store.dtw_bandwidth = float(
            config_entry.options.get(CONF_DTW_BANDWIDTH, DEFAULT_DTW_BANDWIDTH)
        )
        self.learning_manager = LearningManager(
            hass, self.entry_id, self.profile_store, self.device_type,
            device_name=config_entry.title,
        )
        self.recorder = CycleRecorder(hass, self.entry_id, device_name=config_entry.title)
        self._store_bridge: Any = None  # lazy community-store bridge (online features)

        # Priority: Options > Data > Default
        min_power = config_entry.options.get(
            CONF_MIN_POWER, config_entry.data.get(CONF_MIN_POWER, DEFAULT_MIN_POWER)
        )
        off_delay = config_entry.options.get(
            CONF_OFF_DELAY, config_entry.data.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY)
        )
        progress_reset_delay = config_entry.options.get(
            CONF_PROGRESS_RESET_DELAY, DEFAULT_PROGRESS_RESET_DELAY
        )
        self._no_update_active_timeout = float(
            config_entry.options.get(
                CONF_NO_UPDATE_ACTIVE_TIMEOUT,
                DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
            )
        )
        self._low_power_no_update_timeout = float(
            config_entry.options.get(CONF_LOW_POWER_NO_UPDATE_TIMEOUT, 3600.0)
        )
        self._off_delay = float(config_entry.options.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY))
        self._learning_confidence = config_entry.options.get(
            CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE
        )
        self._duration_tolerance = config_entry.options.get(
            CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE
        )
        self._auto_label_confidence = config_entry.options.get(
            CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE
        )

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
        self._notify_timeout_seconds = int(
            config_entry.options.get(
                CONF_NOTIFY_TIMEOUT_SECONDS, DEFAULT_NOTIFY_TIMEOUT_SECONDS
            )
        )

        # Advanced options
        smoothing_window = int(config_entry.options.get("smoothing_window", 5))
        interrupted_min_seconds = int(
            config_entry.options.get("interrupted_min_seconds", 150)
        )
        abrupt_drop_watts = float(config_entry.options.get("abrupt_drop_watts", 500.0))
        abrupt_drop_ratio = float(config_entry.options.get("abrupt_drop_ratio", 0.6))
        abrupt_high_load_factor = float(
            config_entry.options.get("abrupt_high_load_factor", 5.0)
        )

        # Get device specific default for completion threshold
        device_default_completion = DEVICE_COMPLETION_THRESHOLDS.get(
            self.device_type, DEFAULT_COMPLETION_MIN_SECONDS
        )
        completion_min_seconds = int(
            config_entry.options.get(
                CONF_COMPLETION_MIN_SECONDS, device_default_completion
            )
        )

        start_duration_threshold = float(
            config_entry.options.get(
                CONF_START_DURATION_THRESHOLD, DEFAULT_START_DURATION_THRESHOLD
            )
        )
        running_dead_zone = int(
            config_entry.options.get(CONF_RUNNING_DEAD_ZONE, DEFAULT_RUNNING_DEAD_ZONE)
        )
        end_repeat_count = int(
            config_entry.options.get(CONF_END_REPEAT_COUNT, DEFAULT_END_REPEAT_COUNT)
        )

        self._logger.info(
            "Manager init: min_power=%sW, off_delay=%ss, type=%s",
            min_power,
            off_delay,
            self.device_type,
        )

        config = CycleDetectorConfig(
            min_power=float(min_power),
            off_delay=int(off_delay),
            smoothing_window=smoothing_window,
            interrupted_min_seconds=interrupted_min_seconds,
            abrupt_drop_watts=abrupt_drop_watts,
            abrupt_drop_ratio=abrupt_drop_ratio,
            abrupt_high_load_factor=abrupt_high_load_factor,
            completion_min_seconds=completion_min_seconds,
            start_duration_threshold=start_duration_threshold,
            running_dead_zone=running_dead_zone,
            end_repeat_count=end_repeat_count,
            min_off_gap=int(
                config_entry.options.get(
                    CONF_MIN_OFF_GAP,
                    DEFAULT_MIN_OFF_GAP_BY_DEVICE.get(
                        self.device_type, DEFAULT_MIN_OFF_GAP
                    ),
                )
            ),
            start_energy_threshold=float(
                config_entry.options.get(
                    CONF_START_ENERGY_THRESHOLD,
                    DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE.get(self.device_type, 0.2)
                )
            ),
            end_energy_threshold=float(
                config_entry.options.get(CONF_END_ENERGY_THRESHOLD, DEFAULT_END_ENERGY_THRESHOLD)
            ),
            start_threshold_w=float(
                config_entry.options.get(
                    CONF_START_THRESHOLD_W,
                    float(min_power) + max(1.0, 0.1 * float(min_power)),
                )
            ),
            stop_threshold_w=float(
                config_entry.options.get(
                    CONF_STOP_THRESHOLD_W,
                    float(min_power) * 0.6 if float(min_power) > 0 else 2.0,
                )
            ),
            power_off_threshold_w=float(
                config_entry.options.get(
                    CONF_POWER_OFF_THRESHOLD_W, DEFAULT_POWER_OFF_THRESHOLD_W
                )
            ),
            power_off_delay=float(
                config_entry.options.get(CONF_POWER_OFF_DELAY, DEFAULT_POWER_OFF_DELAY)
            ),
            min_duration_ratio=float(
                config_entry.options.get(
                    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE.get(
                        self.device_type, DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO
                    ),
                )
            ),
            match_interval=int(
                config_entry.options.get(
                    CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
                )
            ),
            anti_wrinkle_enabled=bool(
                config_entry.options.get(
                    CONF_ANTI_WRINKLE_ENABLED, DEFAULT_ANTI_WRINKLE_ENABLED
                )
            ),
            anti_wrinkle_max_power=float(
                config_entry.options.get(
                    CONF_ANTI_WRINKLE_MAX_POWER, DEFAULT_ANTI_WRINKLE_MAX_POWER
                )
            ),
            anti_wrinkle_max_duration=float(
                config_entry.options.get(
                    CONF_ANTI_WRINKLE_MAX_DURATION, DEFAULT_ANTI_WRINKLE_MAX_DURATION
                )
            ),
            anti_wrinkle_exit_power=float(
                config_entry.options.get(
                    CONF_ANTI_WRINKLE_EXIT_POWER, DEFAULT_ANTI_WRINKLE_EXIT_POWER
                )
            ),
            delay_detect_enabled=bool(
                config_entry.options.get(
                    CONF_DELAY_START_DETECT_ENABLED, DEFAULT_DELAY_START_DETECT_ENABLED
                )
            ),
            delay_confirm_seconds=float(
                config_entry.options.get(
                    CONF_DELAY_CONFIRM_SECONDS, DEFAULT_DELAY_CONFIRM_SECONDS
                )
            ),
            delay_timeout_seconds=float(
                config_entry.options.get(
                    CONF_DELAY_TIMEOUT_HOURS, DEFAULT_DELAY_TIMEOUT_HOURS
                )
            ) * 3600.0,
        )
        self._config = config


        def profile_matcher_wrapper(
            readings: list[tuple[datetime, float]],
        ) -> tuple[str | None, float, float, str | None]:
            """Wraps profile store matching logic with detector callback signature.

            Returns: None (async offload)
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
                return (
                    self._current_program,
                    1.0,
                    expected_duration,
                    manual_phase or "Manual",
                )

            if not readings:
                return (None, 0.0, 0.0, None)

            # Snapshotted for thread safety indirectly by task logic
            # We don't need a wrapper task if we unify with _update_estimates matching
            # but for now let's keep the detector callback as a trigger
            self.hass.async_create_task(self._async_perform_combined_matching(readings))
            return (None, 0.0, 0.0, None)

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
        self._remove_external_trigger_listener = None  # External cycle end trigger
        self._remove_watchdog = None
        self._watchdog_interval = int(
            config_entry.options.get(CONF_WATCHDOG_INTERVAL, DEFAULT_WATCHDOG_INTERVAL)
        )
        self._match_persistence = int(
            config_entry.options.get(CONF_MATCH_PERSISTENCE, DEFAULT_MATCH_PERSISTENCE)
        )
        self._sampling_interval = float(
            config_entry.options.get(CONF_SAMPLING_INTERVAL, DEFAULT_SAMPLING_INTERVAL)
        )
        self._noise_events_threshold = int(
            config_entry.options.get(
                CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
                DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD,
            )
        )
        self._current_program: str = "off"
        self._time_remaining: float | None = None
        self._total_duration: float | None = None
        self._last_total_duration_update: datetime | None = None
        self._cycle_progress: float = 0.0
        self._smoothed_progress: float = 0.0  # Smoothed progress tracking for EMA
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
        # Sample interval tracking (seconds) for adaptive timing
        # Profile matching duration tolerance (0.25 = ±25%)
        self._profile_duration_tolerance: float = float(
            config_entry.options.get("profile_duration_tolerance", 0.25)
        )


        self._remove_maintenance_scheduler = None
        self._remove_ml_training_scheduler = None
        self._ml_training_failures = 0  # consecutive gate failures for auto-disable
        self._ml_training_running = False  # True while a training run is in flight
        self._profile_sample_repair_stats: dict[str, int] | None = None

        self._last_suggestion_update: datetime | None = None

        # Pump Monitor state
        self._pump_stuck_duration: int = int(
            config_entry.options.get(CONF_PUMP_STUCK_DURATION, DEFAULT_PUMP_STUCK_DURATION)
        )
        self._pump_stuck: bool = False  # True once the stuck threshold has fired for this cycle

        self._manual_program_active: bool = False
        self._notified_start: bool = False
        self._notified_pre_completion: bool = False
        self._last_match_result: Any = None  # Stores full MatchResult object
        self._score_history: dict[str, list[float]] = {}  # Tracks recent scores for trend analysis
        self._match_persistence_counter: dict[str, int] = {}  # Tracks consecutive matches
        self._unmatch_persistence_counter: int = 0  # Tracks consecutive low-confidence matches
        self._current_match_candidate: str | None = None  # Pending profile name

    async def _async_perform_combined_matching(
        self, readings: list[tuple[datetime, float]]
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

            self._matching_task = self.hass.async_create_task(self._async_do_perform_matching(readings))
        except Exception as e:
            self._logger.error("Perform combined matching trigger failed: %s", e)

    async def _async_do_perform_matching(self, readings: list[tuple[datetime, float]]) -> None:
        """Inner task to handle actual matching logic."""
        try:
            end_time = readings[-1][0]
            start_time = readings[0][0]
            current_duration = (end_time - start_time).total_seconds()

            # 1. RUN BETTER ASYNC MATCHING
            result = await self.profile_store.async_match_profile(
                 readings,
                 current_duration
            )

            # 2. UPDATE MANAGER STATE (Estimates, Program Name, etc.)
            self._last_match_result = result
            self._last_match_ambiguous = result.is_ambiguous

            profile_name = result.best_profile
            confidence = result.confidence
            matched_duration = result.expected_duration
            phase_name = result.matched_phase

            # --- Switching Logic (Temporal Persistence) ---
            should_switch = False
            switch_reason = ""

            # Identify current program score from results
            current_program_score = 0.0
            for c in result.candidates:
                if c.get("name") == self._current_program:
                    current_program_score = c.get("score", 0.0)
                    break

            # CASE: Divergence Detection (Score Drop)
            # If current matched program has a significant drop from its own peak score,
            # we should consider unmatching it even if it's still the "best" candidate.
            if (
                self._current_program not in ("detecting...", "off", "starting", "unknown")
                and profile_name == self._current_program
            ):
                history: list[float] = self._score_history.get(self._current_program, [])
                if len(history) > 3:
                    peak_score = max(history)
                    # If score drops by more than 40% from peak AND is below threshold, unmatch.
                    # This catches divergence faster than waiting for fixed unmatch_threshold.
                    if confidence < peak_score * (1.0 - DEFAULT_MATCH_REVERT_RATIO):
                        self._unmatch_persistence_counter += 1
                        if self._unmatch_persistence_counter >= self._match_persistence:
                            self._current_program = "detecting..."
                            self._matched_profile_duration = None
                            self._unmatch_persistence_counter = 0
                            self._logger.info(
                                "Divergence detected for profile '%s' (confidence %.3f < 60%% of peak %.3f). "
                                "Reverting to detection.",
                                profile_name, confidence, peak_score
                            )
                            # Reset profile_name so Case 3 doesn't re-trigger
                            profile_name = "detecting..."

            # Update persistence for the best profile
            if profile_name and profile_name != "detecting...":
                self._match_persistence_counter[profile_name] = self._match_persistence_counter.get(profile_name, 0) + 1

                # Check if this is the same candidate as before
                if profile_name != self._current_match_candidate:
                    # Reset counter for old candidate if it wasn't locked in
                    self._current_match_candidate = profile_name
                    self._match_persistence_counter[profile_name] = 1
            else:
                self._current_match_candidate = None

            is_persistent = profile_name and self._match_persistence_counter.get(profile_name, 0) >= self._match_persistence

            # --- Live-match features: compute always for ranking history + ML gate ---
            # Features are cheap scalars derived from the current trace.  We compute
            # them whenever there is a non-ambiguous candidate so they can be recorded
            # as a training snapshot regardless of whether ML models are opted in.
            # The opt-in ML commit check then uses the same features when enabled.
            ml_commit_score: float | None = None
            _live_feat: dict[str, float] | None = None
            _top2_score: float | None = None
            if (
                profile_name
                and profile_name != "detecting..."
                and not result.is_ambiguous
                and self._current_program in ("detecting...",)
            ):
                try:
                    from .ml.feature_extraction import live_match_features  # noqa: PLC0415

                    first_ts = readings[0][0]
                    pts = [
                        ((ts - first_ts).total_seconds(), float(pw))
                        for ts, pw in readings
                    ]
                    top1_dist = max(0.0, 1.0 - confidence)
                    top2_raw: float | None = None
                    if len(result.candidates) > 1:
                        s2 = result.candidates[1].get("score", 0.0) or 0.0
                        top2_raw = max(0.0, 1.0 - float(s2))
                        _top2_score = float(result.candidates[1].get("score", 0.0) or 0.0)
                    n_profiles = len(self.profile_store.get_profiles())
                    _live_feat = live_match_features(
                        points=pts,
                        elapsed_s=current_duration,
                        top1_distance=top1_dist,
                        top2_distance=top2_raw,
                        top1_median_duration_s=float(result.expected_duration or 0),
                        candidate_count=max(1, n_profiles),
                    )
                    # --- ML early-commit gate (opt-in) ---
                    # When the user has opted into experimental ML models, query the
                    # live_match_commit model for P(top-1 is correct).
                    try:
                        from .ml.engine import ml_models_enabled, resolve_scorer  # noqa: PLC0415
                        if ml_models_enabled(self.config_entry.options):
                            match_fn, _ = resolve_scorer("live_match", self.profile_store)
                            if match_fn is not None:
                                ml_commit_score = float(match_fn(_live_feat))
                                self._logger.debug(
                                    "Live-match ML commit score for '%s': %.3f (threshold %.2f)",
                                    profile_name,
                                    ml_commit_score,
                                    ML_MATCH_COMMIT_THRESHOLD,
                                )
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001 - ML must never break matching
                    pass

            # --- Record ranking snapshot for live_match on-device training ---
            # Snapshot is recorded unconditionally (not gated on ML opt-in) so that
            # training data accumulates even before the user enables ML models.
            # Confirmed labels are back-filled at cycle end.
            if _live_feat is not None and self._cycle_start_time:
                try:
                    self.profile_store.record_match_ranking_snapshot(
                        start_time_iso=self._cycle_start_time.isoformat(),
                        features=_live_feat,
                        top1_profile=profile_name or "",
                        top1_score=float(confidence),
                        top2_score=_top2_score,
                        candidate_count=max(1, len(self.profile_store.get_profiles())),
                        cycle_id=self._ranking_snapshot_cycle_id,
                    )
                except Exception:  # noqa: BLE001 - never break matching
                    pass

            ml_early_commit = (
                ml_commit_score is not None
                and ml_commit_score >= ML_MATCH_COMMIT_THRESHOLD
                and confidence >= 0.30
            )

            # Case 1: Initial Match from "detecting..."
            if (
                profile_name
                and confidence >= 0.15
                and (not result.is_ambiguous or is_persistent)
                and (not self._matched_profile_duration or self._current_program == "detecting...")
            ):
                if is_persistent:
                    should_switch = True
                    switch_reason = f"initial_match (persistent {self._match_persistence_counter[profile_name]}x)"
                elif ml_early_commit:
                    should_switch = True
                    switch_reason = (
                        f"initial_match (ML commit score {ml_commit_score:.3f} >= {ML_MATCH_COMMIT_THRESHOLD})"
                    )
                else:
                    self._logger.debug(
                        "Match persistence: %s at %d/%d matches. Stay at detecting...",
                        profile_name, self._match_persistence_counter.get(profile_name, 0), self._match_persistence
                    )

            # Case 2: Mid-cycle override (different profile)
            elif (
                profile_name
                and self._current_program != profile_name
                and self._current_program not in ("detecting...", "off", "starting", "unknown")
            ):
                # High Confidence Override: Bypass persistence if match is VERY strong
                if confidence > 0.8 and (confidence - current_program_score) > 0.15:
                    should_switch = True
                    switch_reason = f"high_confidence_override ({confidence:.3f} vs {current_program_score:.3f})"

                # Normal Switch: Requires persistence AND either better score + trend
                elif is_persistent:
                    if confidence > current_program_score and self._analyze_trend(profile_name):
                        # Add a minimum score gap for mid-cycle switching (0.05) to prevent flapping
                        if (confidence - current_program_score) > 0.05:
                            should_switch = True
                            switch_reason = f"positive_trend_persistent ({confidence:.3f} > {current_program_score:.3f})"

            # Case 3: Unmatching (confidence drop)
            elif (
                self._current_program not in ("detecting...", "off", "starting", "unknown")
                and profile_name == self._current_program
                and confidence < self._unmatch_threshold
            ):
                self._unmatch_persistence_counter += 1
                is_unmatch_persistent = self._unmatch_persistence_counter >= self._match_persistence

                if is_unmatch_persistent:
                    self._current_program = "detecting..."
                    self._matched_profile_duration = None
                    self._unmatch_persistence_counter = 0
                    self._logger.info(
                        "Unmatched profile '%s' (confidence %.3f < threshold %.3f persistent %dx). "
                        "Reverting to detection.",
                        profile_name,
                        confidence,
                        self._unmatch_threshold,
                        self._match_persistence
                    )
                else:
                    self._logger.debug(
                        "Unmatch persistence: %s at %d/%d low-confidence matches. Stay at %s...",
                        profile_name, self._unmatch_persistence_counter, self._match_persistence, profile_name
                    )

            # Reset unmatch counter if confidence is healthy
            # AND we didn't just detect a divergence
            elif (
                profile_name == self._current_program
                and confidence >= self._unmatch_threshold
                and not (len(self._score_history.get(self._current_program, [])) > 3 and confidence < max(self._score_history[self._current_program]) * (1.0 - DEFAULT_MATCH_REVERT_RATIO))
            ):
                self._unmatch_persistence_counter = 0

            if should_switch:
                if profile_name is None:
                    self._current_program = "detecting..."
                else:
                    self._current_program = profile_name
                self._last_match_confidence = confidence
                self._unmatch_persistence_counter = 0 # Reset on switch
                if profile_name in self._match_persistence_counter:
                    self._match_persistence_counter[profile_name] = self._match_persistence # Lock it in

                avg_duration = float(matched_duration)
                self._matched_profile_duration = avg_duration if avg_duration > 0 else None
                self._logger.info(
                     "Switching to profile '%s' (reason: %s). Expected duration: %.0fs (%smin)",
                     profile_name, switch_reason, avg_duration, int(avg_duration / 60),
                )
            elif profile_name == self._current_program:
                # Same program, but update confidence for sensors
                self._last_match_confidence = confidence
            elif not self._matched_profile_duration:
                self._current_program = "detecting..."

            self._last_estimate_time = dt_util.now()

            # Update score history for all candidates to track trends
            for cand in result.candidates:
                cname = cand.get("name")
                if cname:
                    history = self._score_history.setdefault(cname, [])
                    history.append(float(cand.get("score", 0.0)))
                    if len(history) > 20:
                        history.pop(0)

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
            if current_matched and current_power < stop_thresh and not self._is_user_paused:
                formatted = power_data_to_offsets(cast(list[list[Any] | tuple[Any, ...]], readings))
                try:
                    profile_store_any = cast(Any, self.profile_store)
                    verify_alignment = profile_store_any.async_verify_alignment
                    is_confirmed, mapped_time, _ = (
                        await verify_alignment(current_matched, formatted)
                    )
                except Exception as e: # pylint: disable=broad-exception-caught
                    self._logger.error(
                        "Alignment verification crashed for profile %s: %s",
                        current_matched, e, exc_info=True
                    )
                    is_confirmed = False
                    mapped_time = 0.0

                if is_confirmed:
                    if not verified_pause:
                        self._logger.info(
                            "Envelope verified expected low power phase for %s. Enabling verified pause.",
                            current_matched
                        )
                    verified_pause = True
                    # Smart Termination within Envelope block
                    try:
                        profile = self.profile_store.get_profile(current_matched)
                        if profile:
                            avg_dur = profile.get("avg_duration", 0)
                            if avg_dur > 0 and (mapped_time / avg_dur) > 0.95:
                                verified_pause = False
                                self._logger.info("Smart Termination: Near end of profile. Releasing pause lock.")
                    except Exception as e:
                        self._logger.debug("Smart Termination alignment verification failed: %s", e)
                else:
                    if verified_pause:
                        self._logger.info(
                            "Envelope indicates UNEXPECTED low power for %s. Disabling verified pause.",
                            current_matched
                        )
                    verified_pause = False

            # --- High Power Clear ---
            stop_threshold = getattr(self.detector.config, "stop_threshold_w", 5.0)

            if current_power > stop_threshold * 10:
                verified_pause = False

            # A user-initiated pause (Pause Cycle button, or the door-open soft
            # pause) stays in force until the user resumes (issue #306).  The
            # heuristics above only govern *auto-detected* low-power phases; without
            # this override they clear verified_pause and the cycle is finalized
            # (leaving the "Paused by user" state, e.g. a dishwasher closes at the
            # 1 h min-off-gap timeout) instead of waiting for Resume.  Re-asserting
            # here also repairs the flag after a restart, since the detector state
            # snapshot does not persist _verified_pause.
            if self._is_user_paused:
                verified_pause = True

            # --- Consistency Override ---
            # If envelope verified or mismatched, ensure manager program matches
            if profile_name != self._current_program and (verified_pause or result.is_confident_mismatch):
                if profile_name:
                    self._current_program = profile_name
                    self._last_match_confidence = confidence
                    # Try to fetch duration if we switched back to matched
                    try:
                        prof = self.profile_store.get_profile(profile_name)
                        if prof:
                            self._matched_profile_duration = float(prof.get("avg_duration", 0))
                    except Exception as e:
                        self._logger.debug("Failed to fetch profile duration on switch: %s", e)
                else:
                    self._current_program = "detecting..."
                    self._matched_profile_duration = None

            # --- HEURISTICS (Descriptive Phases) ---
            if not phase_name:
                if self.device_type == "dishwasher" and self.detector.is_waiting_low_power():
                    phase_name = "Drying"
                elif self.device_type == "washing_machine" and current_power > 200:
                    phase_name = "Spinning"
                elif self.device_type == "washing_machine" and self.detector.is_waiting_low_power():
                    phase_name = "Rinsing/Soaking"

            # Push updates to detector
            self.detector.set_verified_pause(verified_pause)
            self.detector.update_match(
                (profile_name, confidence, matched_duration, phase_name,
                 result.is_confident_mismatch, result.is_ambiguous,
                 result.is_prefix_ambiguous)
            )

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
                                self._cycle_start_time or dt_util.now()
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
            self._notify_update()

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
    def phase_description(self) -> str:
        """Return a description of the current phase.

        Prefers the *functional* progress-driven phase (the visual per-profile
        phase configurator's ranges, indexed by the live ML-blended progress) so
        the readout stays accurate even when a cycle runs longer/shorter than the
        profile's nominal timeline. Falls back to the matcher's phase, then the
        detector sub-state/state.
        """
        live = self._current_phase_from_progress()
        if live:
            return live
        if self._last_match_result and self._last_match_result.matched_phase:
            return self._last_match_result.matched_phase
        if self.detector.sub_state:
            return self.detector.sub_state
        return self.detector.state

    def _current_phase_from_progress(self) -> str | None:
        """Live phase from the profile's configured ranges + ML-blended progress.

        This is the *merge* of the visual phase configurator with the runtime
        estimator: one phase definition (the per-profile ranges the user draws),
        indexed by the smoothed progress fraction rather than raw elapsed seconds,
        so overrun/underrun cycles still name the phase correctly. Returns None
        (caller falls back) when not running, no profile is matched, or the
        profile has no configured phase ranges. Never raises.
        """
        return progress_mod.current_phase(
            self.profile_store,
            self.detector.state,
            self._current_program,
            self._cycle_progress,
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
            try:
                current_power = float(state.state)
                power_is_valid = True
            except (ValueError, TypeError):
                # Power sensor state is not numeric during restoration; treat as 0W
                self._logger.debug(
                    "Power sensor %s state %r is not numeric during restoration; "
                    "treating as 0W and not restoring by power",
                    self.power_sensor_entity_id,
                    getattr(state, "state", None),
                )

        should_restore = False
        active_snapshot_to_restore: dict[str, Any] | None = (
            active_snapshot if isinstance(active_snapshot, dict) else None
        )

        # Helper to check if a snapshot is viable
        def is_viable_restore(last_save_time: datetime) -> bool:
            now = dt_util.now()
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
            return False

        last_save = self.profile_store.get_last_active_save()
        if last_save and last_save.tzinfo is None:
            # Normalize naive legacy timestamps to system time
            last_save = last_save.replace(tzinfo=dt_util.now().tzinfo)

        if active_snapshot_to_restore is not None and last_save and is_viable_restore(last_save):
            should_restore = True
            age = (dt_util.now() - last_save).total_seconds()
            age = (dt_util.now() - last_save).total_seconds()
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

        # FALLBACK: Resurrection Logic
        if not should_restore:
            past_cycles = self.profile_store.get_past_cycles()
            if past_cycles:
                last_cycle = past_cycles[-1]
                last_end_str = last_cycle.get("end_time")
                if last_end_str:
                    last_end = dt_util.parse_datetime(last_end_str)
                    if last_end:
                        gap = (dt_util.now() - last_end).total_seconds()
                        is_recent = gap < 1200  # 20 mins
                        status = last_cycle.get("status")

                        if is_recent and status != "completed":
                            self._logger.info(
                                "Found recent interrupted cycle in history "
                                "(id=%s, gap=%.0fs). Resurrecting...",
                                last_cycle["id"],
                                gap,
                            )
                            try:
                                power_data = decompress_power_data(last_cycle)
                                if power_data:
                                    # decompress_power_data returns (offset_seconds, watts)
                                    # tuples, but restore_state_snapshot parses reading[0]
                                    # as an ISO datetime. Convert offsets to absolute ISO
                                    # timestamps (base = cycle start) so the resurrected
                                    # trace is not silently dropped (B4).
                                    _res_start = dt_util.parse_datetime(
                                        last_cycle["start_time"]
                                    )
                                    if _res_start is not None:
                                        if _res_start.tzinfo is None:
                                            _res_start = _res_start.replace(
                                                tzinfo=dt_util.now().tzinfo
                                            )
                                        power_readings = [
                                            (
                                                (
                                                    _res_start
                                                    + timedelta(seconds=float(off))
                                                ).isoformat(),
                                                p,
                                            )
                                            for off, p in power_data
                                        ]
                                    else:
                                        power_readings = power_data
                                    active_snapshot_to_restore = {
                                        # Reconstruct basic running state
                                        "state": "running",
                                        "sub_state": "Resurrected",
                                        "current_cycle_start": last_cycle["start_time"],
                                        "last_active_time": last_cycle["end_time"],
                                        "low_power_start": None,
                                        "cycle_max_power": (
                                            max([p for _, p in power_data])
                                            if power_data
                                            else 0
                                        ),
                                        "power_readings": power_readings,
                                        "ma_buffer": (
                                            [p for _, p in power_data[-10:]]
                                            if power_data
                                            else []
                                        ),
                                        "end_condition_count": 0,
                                        "extension_count": 0,
                                        "dynamic_min_duration": None,
                                        "matched_profile": last_cycle.get(
                                            "profile_name"
                                        ),
                                    }
                                    should_restore = True
                                    past_cycles.pop()
                                    await self.profile_store.async_save()
                            except Exception as e:
                                self._logger.error("Failed to resurrect cycle: %s", e)

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

                self.detector.restore_state_snapshot(active_snapshot_to_restore)

                # Restore if in any active state (Running, Paused, Ending)
                if self.detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
                    # Restore manual program flag if present
                    self._manual_program_active = active_snapshot_to_restore.get(
                        "manual_program", False
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

                    if self.detector.matched_profile:
                        self._current_program = self.detector.matched_profile
                        self._logger.info(
                            "Restored/Resurrected washer cycle with profile: %s",
                            self._current_program,
                        )
                    else:
                        self._current_program = "detecting..."
                    
                    # Restore persisted start-notification/event flags from snapshot.
                    self._notified_start = bool(
                        active_snapshot_to_restore.get("notified_start", False)
                    )
                    self._start_event_fired = bool(
                        active_snapshot_to_restore.get("start_event_fired", False)
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

                    # Record the restart gap so the Cycles tab can shade it and
                    # anomaly detection can surface it.  Only meaningful when
                    # last_save is known and the dark period exceeds 30 s.
                    # Matching always uses real readings only (Option B from the
                    # gap-fill analysis); synthetic fill is intentionally NOT added
                    # to _power_readings to prevent circular-bias inflation.
                    if last_save:
                        gap_end = dt_util.now()
                        gap_secs = (gap_end - last_save).total_seconds()
                        if gap_secs > 30:
                            self._restart_gaps.append({
                                "start_ts": last_save.isoformat(),
                                "end_ts": gap_end.isoformat(),
                                "gap_seconds": round(gap_secs, 1),
                                "profile": self.detector.matched_profile,
                                "match_confidence": getattr(
                                    self.detector, "match_confidence", None
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
            except Exception as err:
                self._logger.warning("Failed to restore active cycle: %s, clearing", err)
                await self.profile_store.async_clear_active_cycle()
        else:
            if last_save:
                age = (dt_util.now() - last_save).total_seconds()
                self._logger.info("Active cycle too stale (age=%.0fs), clearing", age)
            await self.profile_store.async_clear_active_cycle()

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
                    "vs_typical_longer": "{pct}% longer than usual",
                    "vs_typical_shorter": "{pct}% shorter than usual",
                    "notify_live_waiting_message": "{device}: No profile matched yet.",
                }.items()
            }
        except Exception:  # noqa: BLE001
            pass
        # Apply configurable duration tolerance to profile store
        try:
            self.profile_store.set_duration_tolerance(self._profile_duration_tolerance)
            self.profile_store.set_retention_limits(
                max_past_cycles=int(
                    self.config_entry.options.get(
                        CONF_MAX_PAST_CYCLES, DEFAULT_MAX_PAST_CYCLES
                    )
                ),
                max_full_traces_per_profile=int(
                    self.config_entry.options.get(
                        CONF_MAX_FULL_TRACES_PER_PROFILE,
                        DEFAULT_MAX_FULL_TRACES_PER_PROFILE,
                    )
                ),
                max_full_traces_unlabeled=int(
                    self.config_entry.options.get(
                        CONF_MAX_FULL_TRACES_UNLABELED,
                        DEFAULT_MAX_FULL_TRACES_UNLABELED,
                    )
                ),
            )
        except Exception:
            pass

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

        # Subscribe to power sensor updates
        self._remove_listener = async_track_state_change_event(
            self.hass, [self.power_sensor_entity_id], self._async_power_changed
        )

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

        # Force initial update from current state (in case it's already stable)
        state = self.hass.states.get(self.power_sensor_entity_id)
        if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            try:
                power = float(state.state)
                now = dt_util.now()
                self.detector.process_reading(power, now)
            except (ValueError, TypeError):
                pass

        # Trigger migration/compression of old cycle format
        # This is safe to run repeatedly (it skips already compressed cycles)
        await self.profile_store.async_migrate_cycles_to_compressed()

        # Backfill match_confidence for labeled cycles that predate the field
        self.hass.async_create_task(
            self.profile_store.async_backfill_match_confidence()
        )

        # Subscribe to external cycle end trigger (if enabled)
        await self._setup_external_end_trigger()

        # Subscribe to door sensor (if configured)
        await self._setup_door_sensor_listener()

        # Subscribe to person presence changes for notification gating
        await self._setup_notify_people_listener()

        # Register schedulers (maintenance + ML training). These are also re-
        # registered on every config reload; calling them here ensures they
        # survive HA restarts without requiring the user to re-save settings.
        await self._setup_maintenance_scheduler()
        self._setup_ml_training_scheduler()

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
        new_sensor = config_entry.options.get(
            CONF_POWER_SENSOR, config_entry.data.get(CONF_POWER_SENSOR)
        )
        if new_sensor and new_sensor != self.power_sensor_entity_id:
            # Block sensor changes when a cycle is active to prevent inconsistent state
            d_state = self.detector.state
            self._logger.debug(
                "Reloading config: detector.state=%r (type=%s), RUNNING=%r",
                d_state,
                type(d_state),
                STATE_RUNNING,
            )
            if d_state == STATE_RUNNING:
                self._logger.warning(
                    "Cannot change power sensor from %s to %s while a cycle "
                    "is active. Please wait for the current cycle to complete "
                    "before changing the power sensor.",
                    self.power_sensor_entity_id,
                    new_sensor,
                )
                # Skip sensor change but continue with other config updates
                return

            self._logger.info(
                "Power sensor changed: %s -> %s", self.power_sensor_entity_id, new_sensor
            )
            self.power_sensor_entity_id = new_sensor
            # Remove old listener
            if self._remove_listener:
                self._remove_listener()
            # Attach new listener
            self._remove_listener = async_track_state_change_event(
                self.hass, [self.power_sensor_entity_id], self._async_power_changed
            )
            # Force update from new sensor
            state = self.hass.states.get(self.power_sensor_entity_id)
            if state and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                try:
                    power = float(state.state)
                    self.detector.process_reading(power, dt_util.now())
                except ValueError:
                    self._logger.debug(
                        "Initial power value for %s after config reload is not numeric: %r",
                        self.power_sensor_entity_id,
                        state.state,
                    )

        # Update device type
        self.device_type = config_entry.options.get(
            CONF_DEVICE_TYPE,
            config_entry.data.get(CONF_DEVICE_TYPE, DEFAULT_DEVICE_TYPE),
        )
        # Propagate to learning pipeline (captured at construction time)
        self.learning_manager.device_type = self.device_type
        self.learning_manager.suggestion_engine.device_type = self.device_type

        # Update detector config in-place
        old_min_power = self.detector.config.min_power
        old_off_delay = self.detector.config.off_delay
        old_smoothing = self.detector.config.smoothing_window
        old_interrupted_min = self.detector.config.interrupted_min_seconds
        old_abrupt_drop_watts = self.detector.config.abrupt_drop_watts
        old_abrupt_drop_ratio = self.detector.config.abrupt_drop_ratio
        old_abrupt_high_load = self.detector.config.abrupt_high_load_factor

        # Get new values from config
        new_min_power = float(
            config_entry.options.get(CONF_MIN_POWER, DEFAULT_MIN_POWER)
        )
        new_off_delay = int(config_entry.options.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY))
        new_smoothing = int(
            config_entry.options.get(CONF_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW)
        )
        new_interrupted_min = int(
            config_entry.options.get(
                CONF_INTERRUPTED_MIN_SECONDS, DEFAULT_INTERRUPTED_MIN_SECONDS
            )
        )
        new_abrupt_drop_watts = float(
            config_entry.options.get(CONF_ABRUPT_DROP_WATTS, DEFAULT_ABRUPT_DROP_WATTS)
        )
        new_abrupt_drop_ratio = float(
            config_entry.options.get(CONF_ABRUPT_DROP_RATIO, DEFAULT_ABRUPT_DROP_RATIO)
        )
        self.detector.config.match_interval = int(
            config_entry.options.get(
                CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL
            )
        )
        self.profile_store.dtw_bandwidth = float(
            config_entry.options.get(CONF_DTW_BANDWIDTH, DEFAULT_DTW_BANDWIDTH)
        )
        new_abrupt_high_load = float(
            config_entry.options.get(
                CONF_ABRUPT_HIGH_LOAD_FACTOR, DEFAULT_ABRUPT_HIGH_LOAD_FACTOR
            )
        )

        # Device default
        dev_def = DEVICE_COMPLETION_THRESHOLDS.get(
            self.device_type, DEFAULT_COMPLETION_MIN_SECONDS
        )
        new_completion_min = int(
            config_entry.options.get(CONF_COMPLETION_MIN_SECONDS, dev_def)
        )

        new_start_threshold = float(
            config_entry.options.get(
                CONF_START_DURATION_THRESHOLD, DEFAULT_START_DURATION_THRESHOLD
            )
        )
        new_running_dead_zone = int(
            config_entry.options.get(CONF_RUNNING_DEAD_ZONE, DEFAULT_RUNNING_DEAD_ZONE)
        )
        new_end_repeat_count = int(
            config_entry.options.get(CONF_END_REPEAT_COUNT, DEFAULT_END_REPEAT_COUNT)
        )

        # Power Hysteresis Thresholds
        new_start_threshold_w = float(
            config_entry.options.get(
                CONF_START_THRESHOLD_W,
                float(new_min_power) + max(1.0, 0.1 * float(new_min_power)),
            )
        )
        new_stop_threshold_w = float(
            config_entry.options.get(
                CONF_STOP_THRESHOLD_W,
                max(0.0, float(new_min_power) - max(0.5, 0.1 * float(new_min_power))),
            )
        )
        new_power_off_threshold_w = float(
            config_entry.options.get(
                CONF_POWER_OFF_THRESHOLD_W, DEFAULT_POWER_OFF_THRESHOLD_W
            )
        )
        new_power_off_delay = float(
            config_entry.options.get(CONF_POWER_OFF_DELAY, DEFAULT_POWER_OFF_DELAY)
        )

        new_start_energy = float(
            config_entry.options.get(
                CONF_START_ENERGY_THRESHOLD,
                DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE.get(self.device_type, 0.2)
            )
        )
        new_end_energy = float(
            config_entry.options.get(CONF_END_ENERGY_THRESHOLD, DEFAULT_END_ENERGY_THRESHOLD)
        )

        new_anti_wrinkle_enabled = bool(
            config_entry.options.get(
                CONF_ANTI_WRINKLE_ENABLED, DEFAULT_ANTI_WRINKLE_ENABLED
            )
        )
        new_anti_wrinkle_max_power = float(
            config_entry.options.get(
                CONF_ANTI_WRINKLE_MAX_POWER, DEFAULT_ANTI_WRINKLE_MAX_POWER
            )
        )
        new_anti_wrinkle_max_duration = float(
            config_entry.options.get(
                CONF_ANTI_WRINKLE_MAX_DURATION, DEFAULT_ANTI_WRINKLE_MAX_DURATION
            )
        )
        new_anti_wrinkle_exit_power = float(
            config_entry.options.get(
                CONF_ANTI_WRINKLE_EXIT_POWER, DEFAULT_ANTI_WRINKLE_EXIT_POWER
            )
        )
        new_delay_detect_enabled = bool(
            config_entry.options.get(
                CONF_DELAY_START_DETECT_ENABLED, DEFAULT_DELAY_START_DETECT_ENABLED
            )
        )
        new_delay_confirm_seconds = float(
            config_entry.options.get(
                CONF_DELAY_CONFIRM_SECONDS, DEFAULT_DELAY_CONFIRM_SECONDS
            )
        )
        new_delay_timeout_seconds = float(
            config_entry.options.get(
                CONF_DELAY_TIMEOUT_HOURS, DEFAULT_DELAY_TIMEOUT_HOURS
            )
        ) * 3600.0

        # Apply all detector config updates
        self.detector.config.min_power = new_min_power
        self.detector.config.off_delay = new_off_delay
        self.detector.config.smoothing_window = new_smoothing
        self.detector.config.interrupted_min_seconds = new_interrupted_min
        self.detector.config.abrupt_drop_watts = new_abrupt_drop_watts
        self.detector.config.abrupt_drop_ratio = new_abrupt_drop_ratio
        self.detector.config.abrupt_high_load_factor = new_abrupt_high_load
        self.detector.config.completion_min_seconds = new_completion_min
        self.detector.config.start_duration_threshold = new_start_threshold
        self.detector.config.running_dead_zone = new_running_dead_zone
        self.detector.config.end_repeat_count = new_end_repeat_count
        self.detector.config.start_threshold_w = new_start_threshold_w
        self.detector.config.stop_threshold_w = new_stop_threshold_w
        self.detector.config.power_off_threshold_w = new_power_off_threshold_w
        self.detector.config.power_off_delay = new_power_off_delay
        self.detector.config.start_energy_threshold = new_start_energy
        self.detector.config.end_energy_threshold = new_end_energy
        self.detector.config.anti_wrinkle_enabled = new_anti_wrinkle_enabled
        self.detector.config.anti_wrinkle_max_power = new_anti_wrinkle_max_power
        self.detector.config.anti_wrinkle_max_duration = new_anti_wrinkle_max_duration
        self.detector.config.anti_wrinkle_exit_power = new_anti_wrinkle_exit_power
        self.detector.config.delay_detect_enabled = new_delay_detect_enabled
        self.detector.config.delay_confirm_seconds = new_delay_confirm_seconds
        self.detector.config.delay_timeout_seconds = new_delay_timeout_seconds

        # Pump Monitor setting
        self._pump_stuck_duration = int(
            config_entry.options.get(CONF_PUMP_STUCK_DURATION, DEFAULT_PUMP_STUCK_DURATION)
        )

        if (
            old_min_power != new_min_power
            or old_off_delay != new_off_delay
            or old_smoothing != new_smoothing
            or old_interrupted_min != new_interrupted_min
            or old_abrupt_drop_watts != new_abrupt_drop_watts
            or old_abrupt_drop_ratio != new_abrupt_drop_ratio
            or old_abrupt_high_load != new_abrupt_high_load
        ):
            self._logger.info(
                "Updated detector config: min_power %.1fW→%.1fW, off_delay %ds→%ds, "
                "smoothing %d→%d, interrupted_min %ds→%ds, abrupt_drop %.0fW→%.0fW, "
                "abrupt_ratio %.2f→%.2f, high_load %.1f→%.1f",
                old_min_power,
                new_min_power,
                old_off_delay,
                new_off_delay,
                old_smoothing,
                new_smoothing,
                old_interrupted_min,
                new_interrupted_min,
                old_abrupt_drop_watts,
                new_abrupt_drop_watts,
                old_abrupt_drop_ratio,
                new_abrupt_drop_ratio,
                old_abrupt_high_load,
                new_abrupt_high_load,
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
        self._profile_duration_tolerance = float(
            config_entry.options.get(
                CONF_PROFILE_DURATION_TOLERANCE, DEFAULT_PROFILE_DURATION_TOLERANCE
            )
        )


        # Update notification settings
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
        self._notify_timeout_seconds = int(
            config_entry.options.get(
                CONF_NOTIFY_TIMEOUT_SECONDS, DEFAULT_NOTIFY_TIMEOUT_SECONDS
            )
        )

        # Reload door sensor / pause config
        self._pause_cuts_power = bool(config_entry.options.get(CONF_PAUSE_CUTS_POWER, False))
        self._door_sensor_entity = config_entry.options.get(CONF_DOOR_SENSOR_ENTITY) or None
        self._notify_unload_delay_minutes = int(
            config_entry.options.get(
                CONF_NOTIFY_UNLOAD_DELAY_MINUTES, DEFAULT_NOTIFY_UNLOAD_DELAY_MINUTES
            )
        )

        # Re-subscribe to external cycle end trigger
        await self._setup_external_end_trigger()

        # Re-subscribe to door sensor
        await self._setup_door_sensor_listener()

        # Re-subscribe to person presence changes for notification gating
        await self._setup_notify_people_listener()

        # If a cycle is currently active and live notifications are now enabled,
        # reset counters and fire the first live notification immediately so the
        # user doesn't have to wait for the next power sensor poll.
        if self.detector.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            if self._notify_live_services or self._notify_actions:
                self._reset_live_notification_state()
                self._check_live_progress_notification()

        self._logger.info("Configuration reloaded successfully")

        # Trigger entity updates to reflect any changes
        async_dispatcher_send(self.hass, f"ha_washdata_update_{self.entry_id}")

        if self.detector:
            self.detector.config.profile_duration_tolerance = self._profile_duration_tolerance

        # Schedule midnight maintenance if enabled
        await self._setup_maintenance_scheduler()

        # Schedule on-device ML retraining if enabled (Stage 4, gated)
        self._setup_ml_training_scheduler()

        # Update sampling interval
        old_sampling = self._sampling_interval
        new_sampling = float(
            config_entry.options.get(CONF_SAMPLING_INTERVAL, DEFAULT_SAMPLING_INTERVAL)
        )
        if old_sampling != new_sampling:
            self._sampling_interval = new_sampling
            self._logger.info(
                "Updated sampling interval: %.1fs -> %.1fs", old_sampling, new_sampling
            )

        # RESTORE STATE (only if recent enough, otherwise treat as stale)
        await self._attempt_state_restoration()

        self._logger.info("Configuration reloaded successfully")

    async def async_shutdown(self) -> None:
        """Shutdown."""
        if self._remove_listener:
            self._remove_listener()
        if self._remove_external_trigger_listener:
            self._remove_external_trigger_listener()
        if self._remove_door_sensor_listener:
            self._remove_door_sensor_listener()
            self._remove_door_sensor_listener = None
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
        if self.detector.state in {STATE_RUNNING, STATE_PAUSED, STATE_STARTING, STATE_ENDING}:
            snapshot = self.detector.get_state_snapshot()
            snapshot["manual_program"] = self._manual_program_active
            snapshot["notified_start"] = self._notified_start
            snapshot["start_event_fired"] = self._start_event_fired
            snapshot["is_user_paused"] = self._is_user_paused
            snapshot["user_pause_start"] = (
                self._user_pause_start.isoformat() if self._user_pause_start else None
            )
            snapshot["total_user_paused_seconds"] = self._total_user_paused_seconds
            await self.profile_store.async_save_active_cycle(snapshot)

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
                self._logger.debug("Door opened: clearing Clean state")
                self._is_clean_state = False
                self._clean_state_start = None
                self._notified_clean_laundry = False
                # Dismiss a delivered clean reminder (and purge any queued ones)
                # so it does not linger on the phone after the laundry is taken.
                self._clear_clean_notification()
                self._notify_update()
            elif self.detector.state in (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING):
                # Door opened during active cycle → soft pause confirmation
                self._logger.debug(
                    "Door opened during active cycle: setting verified_pause=True"
                )
                self.detector.set_verified_pause(True)
                if not self._is_user_paused:
                    self._is_user_paused = True
                    self._user_pause_start = dt_util.now()
                self._notify_update()
        # Door closing is intentionally not handled - no auto-resume

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
                pending = list(self._pending_notifications)
                self._pending_notifications = []
                for entry in pending:
                    self._dispatch_notification(
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
        except (TypeError, ValueError):
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
        cycles = self.profile_store.get_past_cycles()

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
                age_days = (dt_util.now() - last).total_seconds() / 86400.0
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
            # Consumers (ML Lab, MLSuggestionEngine) read the trained specs live
            # from the store via ml.engine.resolve_scorer, so no refresh is needed.
            self._logger.info("On-device ML training promoted models: %s", promoted)
        else:
            self._ml_training_failures += 1
            self._logger.info(
                "On-device ML training produced no promotable models (attempt %d)",
                self._ml_training_failures,
            )

        # Stage 4/5: tune the matcher's scoring weights from this device's own
        # cycles (same held-out promotion discipline as the models). Independent
        # of model promotion; runs on every training pass.
        matching = await self._tune_matching_config(cycles)

        # Record that training *ran* now, regardless of whether anything was
        # promoted, so "Last trained" advances on every run (a run that doesn't
        # beat the baseline previously left the timestamp stuck at the last
        # promotion). Never let a persistence hiccup break the run.
        try:
            _run_iso = dt_util.now().isoformat()
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
                "matching": matching,
            },
        )
        # A promoted model changes the health-model signature, so recompute the
        # persisted per-cycle health now rather than lazily on the next view.
        if promoted:
            try:
                await self.async_recompute_cycle_health()
            except Exception as err:  # noqa: BLE001
                self._logger.debug("Post-training health recompute failed: %s", err)

        return {
            "ok": True,
            "promoted": promoted,
            "results": summary.get("results", []),
            "matching": matching,
        }

    async def _tune_matching_config(self, cycles: list[dict[str, Any]]) -> dict[str, Any]:
        """Tune + (if it beats the shipped defaults on a held-out split) persist
        the matcher's scoring weights for this device. Executor-offloaded and
        never raises. Returns the tuner status dict for logging / the UI event.
        """
        try:
            from .ml.matching_tuner import tune_matching_config

            result = await self.hass.async_add_executor_job(tune_matching_config, cycles)
        except Exception as err:  # noqa: BLE001 - tuning must never break training
            self._logger.debug("Matching-config tuning failed: %s", err)
            return {"promoted": False, "reason": "exception", "error": str(err)}

        if result.get("promoted") and result.get("config"):
            record = {
                "config": result["config"],
                "trained_at": dt_util.now().isoformat(),
                "cycle_count": len(cycles),
                "baseline_test_top1": result.get("baseline_test_top1"),
                "tuned_test_top1": result.get("tuned_test_top1"),
            }
            await self.profile_store.set_matching_config(record)
            self._logger.info(
                "On-device matcher tuning promoted (top-1 %.3f -> %.3f): %s",
                result.get("baseline_test_top1") or 0.0,
                result.get("tuned_test_top1") or 0.0,
                result["config"],
            )
        else:
            self._logger.debug(
                "On-device matcher tuning not promoted: %s", result.get("reason")
            )
        return result

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
        opts = {**self.config_entry.data, **self.config_entry.options}
        off_delay = int(opts.get(CONF_OFF_DELAY, DEFAULT_OFF_DELAY))
        try:
            result = await self.hass.async_add_executor_job(
                functools.partial(
                    _compute_ml_comparison, self.profile_store, off_delay, force_recompute=True
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
            except (ValueError, TypeError):
                parsed = None
            if parsed is not None and (latest is None or parsed > latest):
                latest = parsed
        return latest

    @callback
    def _async_power_changed(self, event: Any) -> None:
        """Handle power sensor state change."""
        event_data = cast(dict[str, Any], getattr(event, "data", {}))
        new_state = cast(State | None, event_data.get("new_state"))
        if new_state is None or new_state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return

        try:
            power = float(new_state.state)
        except ValueError:
            return

        # Capture every raw sensor reading before any throttling or processing.
        # Use the sensor's own last_updated timestamp so the trace reflects
        # when the plug actually reported the value, not when we received it.
        self.diag_buffer.record_power(power, new_state.last_updated)

        # RECORD MODE INTERCEPTION
        if self.recorder.is_recording:
            self.recorder.process_reading(power)
            self._current_power = power
            self._last_reading_time = dt_util.now()
            self._notify_update()
            return

        now = dt_util.now()

        # Throttle updates to avoid CPU overload on noisy sensors
        # BUT always allow updates if power is below min_power (critical end-of-cycle signal).
        min_p = float(self.detector.config.min_power)
        is_low_power = power < min_p

        if (
            not is_low_power
            and self._last_reading_time
            and (now - self._last_reading_time).total_seconds() < self._sampling_interval
        ):
            return

        # Track observed power readings for learning
        self.learning_manager.process_power_reading(power, now, self._last_reading_time)
        self._last_reading_time = now
        self._last_real_reading_time = now # Track real update
        self._current_power = power
        self.detector.process_reading(power, now)

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

        self._notify_update()

    def _check_state_save(self, now: datetime) -> None:
        """Periodically save active state."""
        last_save = getattr(self, "_last_state_save", None)
        if not last_save or (now - last_save).total_seconds() > 60:
            # Fire and forget save task
            # Inject manual program flag into snapshot before saving
            snapshot = self.detector.get_state_snapshot()
            snapshot["manual_program"] = self._manual_program_active
            snapshot["notified_start"] = self._notified_start
            snapshot["start_event_fired"] = self._start_event_fired
            snapshot["is_user_paused"] = self._is_user_paused
            snapshot["user_pause_start"] = (
                self._user_pause_start.isoformat() if self._user_pause_start else None
            )
            snapshot["total_user_paused_seconds"] = self._total_user_paused_seconds

            self.hass.async_create_task(
                self.profile_store.async_save_active_cycle(snapshot)
            )
            self._last_state_save = now

    async def _run_final_match_from_cycle_data(self, cycle_data: dict[str, Any]) -> None:
        """Run final profile match using the cycle's power data before it's saved.

        This is called from _on_cycle_end when _current_program is still 'detecting...'
        to ensure we try matching with complete cycle data before persistence.
        """
        # Cycle data from detector stores power_data as [[offset_seconds, power], ...],
        # where offsets are relative to cycle start.
        power_data = cycle_data.get("power_data", [])
        duration = cycle_data.get("duration", 0)

        if not power_data or len(power_data) < 10:
            self._logger.debug("Insufficient power data for final match (< 10 readings)")
            return

        # power_data is already in [[offset_seconds, power], ...] format for matching.
        self._logger.info(
            "Running final match from cycle data: %s samples, %.0fs duration",
            len(power_data),
            duration,
        )

        result = await self.profile_store.async_match_profile(power_data, duration)
        profile_name = result.best_profile
        confidence = result.confidence

        # Store result for debug data
        self._last_match_result = result

        # Accept match at lower threshold since cycle is complete
        # Also ignore ambiguity for completed cycles - pick the best match
        if profile_name and confidence >= 0.15:
            self._logger.info(
                "Final match from cycle data: '%s' with confidence %.3f",
                profile_name,
                confidence,
            )
            self._current_program = profile_name
            self._last_match_confidence = confidence
        else:
            self._logger.info(
                "No confident match from cycle data (best: %s, conf=%.3f)",
                profile_name,
                confidence,
            )

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
        if (
            not self._cycle_completed_time
            or self.detector.state == STATE_RUNNING
            or self.detector.state == STATE_ANTI_WRINKLE
            or self.detector.state == STATE_DELAY_WAIT
        ):
            # Cycle is running or not completed, don't reset
            return

        time_since_complete = (now - self._cycle_completed_time).total_seconds()

        # Clean laundry nag notification
        if (
            self._is_clean_state
            and not self._notified_clean_laundry
            and self._clean_state_start is not None
            and self._notify_unload_delay_minutes > 0
        ):
            time_in_clean = (now - self._clean_state_start).total_seconds()
            if time_in_clean >= self._notify_unload_delay_minutes * 60:
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
                        delay=self._notify_unload_delay_minutes,
                    )
                    sent = self._dispatch_notification(
                        msg,
                        event_type=NOTIFY_EVENT_CLEAN,
                        extra_vars={"tag": self._clean_tag},
                    )
                    if sent:
                        self._notified_clean_laundry = True
                        self._logger.info(
                            "Sent clean laundry nag notification (%.0f min after cycle end)",
                            time_since_complete / 60,
                        )
                    elif self._last_dispatch_deferred:
                        # Held for quiet-hours / presence delivery; the queued copy
                        # fires later. Mark handled so the 60s expiry tick doesn't
                        # enqueue a duplicate nag every minute for the whole window.
                        self._notified_clean_laundry = True
                else:
                    self._notified_clean_laundry = True

        # Defer leaving the terminal state while a clean-state unload notification is
        # still pending. Without this guard the 30-min progress reset (or an early
        # power-off) fires before the unload nag, clearing _is_clean_state before the
        # notification can fire. Both expiry modes below honour it.
        nag_pending = (
            self._is_clean_state
            and not self._notified_clean_laundry
            and self._notify_unload_delay_minutes > 0
            and time_since_complete < self._notify_unload_delay_minutes * 60
        )

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
            dt_util.now() - self._power_off_below_since
        ).total_seconds() < cfg.power_off_delay:
            return
        # Honour the clean-laundry unload nag hold (mirrors the poll path).
        if (
            self._is_clean_state
            and not self._notified_clean_laundry
            and self._notify_unload_delay_minutes > 0
            and self._cycle_completed_time is not None
            and (dt_util.now() - self._cycle_completed_time).total_seconds()
            < self._notify_unload_delay_minutes * 60
        ):
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

    async def _watchdog_check_stuck_cycle(self, now: datetime) -> None:
        """Watchdog: check if cycle is stuck (no updates for too long)."""
        if self.detector.state not in (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING):
            return

        if not self._last_reading_time:
            return

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

        # Dishwashers can have very long silent drying phases (up to 2h)
        # We use the device-specific timeout as the floor for this effective timeout.
        # The floor is applied unconditionally - dishwashers have passive drying phases
        # even when no profile has been matched yet.  The original restriction to matched
        # cycles caused premature kills: with the default 3600s timeout, an unmatched
        # dishwasher cycle was killed ~1h after the last sensor update, while the
        # physical drying phase could still have 1-2h of silent runtime remaining.
        low_power_floor = DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE.get(
            self.device_type, 0
        )
        effective_low_power_timeout = max(
            low_power_floor, self._low_power_no_update_timeout
        )

        # Profile-Aware Extension:
        # If we have a matched profile, ensure we don't kill during the expected duration.
        if expected > 0 and elapsed < expected:
            # Extend timeout to cover the remaining expected duration + buffer
            remaining = expected - elapsed
            # Allow silence up to remaining + 1800s (30m buffer for drying/pause)
            extended_timeout = remaining + 1800
            if extended_timeout > effective_low_power_timeout:
                effective_low_power_timeout = extended_timeout

        # Verified Pause Extension:
        # If the manager/store has confirmed this is a legitimate pause (e.g. Drying),
        # allow even more leniency up to the global deferral limit.
        if getattr(self.detector, "_verified_pause", False):
            # Allow silence up to DEFAULT_MAX_DEFERRAL_SECONDS (default 2h) + buffer
            pause_limit = DEFAULT_MAX_DEFERRAL_SECONDS + 1800
            if pause_limit > effective_low_power_timeout:
                effective_low_power_timeout = pause_limit
                self._logger.debug(
                    "Watchdog: Extending timeout to %.0fs due to verified pause",
                    effective_low_power_timeout
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

            # 3. Injection Check (Keepalive)
            # 3a. Honour the user-configured no_update_active_timeout for low-power silence.
            # Publish-on-change sensors go completely silent once they stabilise at a low
            # standby value (e.g. 1 W).  The existing off_delay-based injection fires
            # every 2 watchdog ticks, which is fine with a short watchdog interval but can
            # take many minutes with a larger one.  Respecting no_update_active_timeout
            # here gives users a predictable upper bound on how long a cycle lingers after
            # the appliance reaches standby, consistent with what the setting implies.
            # Verified pauses (e.g. dishwasher drying confirmed by envelope) are excluded
            # so that legitimate long silent phases are not prematurely terminated.
            if (
                not getattr(self.detector, "_verified_pause", False)
                and time_since_real_update > self._no_update_active_timeout
            ):
                self._logger.debug(
                    "Watchdog: Low-power real-update silence (%.0fs) > no_update_active_timeout (%.0fs). "
                    "Injecting 0W keepalive to advance accumulator.",
                    time_since_real_update,
                    self._no_update_active_timeout,
                )
                self.detector.process_reading(0.0, now)
                self._last_reading_time = now
                self._current_power = 0.0
                self._notify_update()
                return

            # 3b. Fallback: inject 0W when any-update silence exceeds off_delay.
            # This keeps the accumulator moving even when no_update_active_timeout has
            # not been exceeded (e.g. the user left it at the default 600 s).
            if time_since_any_update > self._config.off_delay:
                self._logger.debug(
                    "Watchdog: Low power silence (%.0fs). Injecting 0W keepalive.",
                    time_since_any_update
                )
                # Ensure we handle the injection cleanly
                # Do NOT update _last_real_reading_time here
                self.detector.process_reading(0.0, now)
                self._last_reading_time = now # Resets 'any' timer so we don't spam
                self._current_power = 0.0
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
            self._logger.debug("Watchdog: Silence at low power (%.0fs). Injecting 0W.", time_since_any_update)
            self.detector.process_reading(0.0, now)
            self._last_reading_time = now
            self._current_power = 0.0
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
                    self._logger.info(
                        "Watchdog: High power (%.1fW) stale (%.0fs). Injecting refresh.",
                        self._current_power, time_since_any_update
                    )
                    self.detector.process_reading(self._current_power, now)
                    self._last_reading_time = now
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
            old_state, new_state, self._current_program, dt_util.now()
        )
        # A new cycle starting while we are still showing the completed/Clean
        # overlay (the progress-reset window) must clear that overlay and cancel
        # the expiry timer right away, so the UI leaves "Finished" and the unload
        # nag stops immediately instead of waiting for the reset window - and so
        # the expiry timer cannot race the new cycle and reset us to OFF (#267).
        if new_state == STATE_STARTING and self._cycle_completed_time is not None:
            self._cycle_completed_time = None
            self._is_clean_state = False
            self._clean_state_start = None
            self._notified_clean_laundry = False
            self._cycle_progress = 0.0
            self._power_off_below_since = None
            self._cancel_power_off_timer()
            self._stop_state_expiry_timer()
        if new_state == STATE_RUNNING:
            new_cycle_detected = old_state in (STATE_OFF, STATE_STARTING, STATE_UNKNOWN)
            # Only reset estimates if we are truly starting a NEW cycle (from off or starting)
            # If we transition from PAUSED or ENDING, it's a resume - keep estimates!
            if new_cycle_detected:
                self._cycle_completed_time = None
                self._stop_state_expiry_timer()

                self._current_program = "detecting..."
                self._manual_program_active = False
                self._notified_pre_completion = False
                self._time_remaining = None
                self._total_duration = None
                self._cycle_progress = 0
                self._matched_profile_duration = None
                self._last_estimate_time = None
                self._score_history = {}  # Reset score history on new cycle
                self._match_persistence_counter = {}  # Reset persistence counter
                self._unmatch_persistence_counter = 0  # Reset unmatch counter
                self._current_match_candidate = None  # Reset candidate
                self._notified_start = False # Reset start notification state
                self._start_event_fired = False
                self._last_cycle_post_anomaly = {}  # Clear previous cycle's anomaly cache
                self._cycle_start_time = self.detector.current_cycle_start or dt_util.now()
                self._ranking_snapshot_cycle_id = str(uuid.uuid4())
                self._reset_live_notification_state()

                # Reset pause tracking and clean state for new cycle
                self._is_user_paused = False
                self._user_pause_start = None
                self._total_user_paused_seconds = 0.0
                self._is_clean_state = False
                self._fired_cycle_timers = set()
                self._clear_timer_pause_notification()
                self._clean_state_start = None
                self._notified_clean_laundry = False

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

        # Stop watchdog when transitioning to OFF from any active state
        if new_state == STATE_OFF:
            self._stop_watchdog()  # Stop watchdog regardless of previous state
            self._cycle_start_time = None

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
        self.hass.async_create_task(self.profile_store.async_clear_active_cycle())
        # Anchor the terminal state so _handle_state_expiry (and power-off) can act,
        # then arm the expiry timer that resets terminal -> Off after the reset delay.
        self._cycle_completed_time = dt_util.now()
        self._start_state_expiry_timer()

    def _on_cycle_end(self, cycle_data: dict[str, Any]) -> None:
        """Handle cycle end - clear all active timers and state."""
        duration = cycle_data["duration"]
        max_power = cycle_data.get("max_power", 0)

        # IMMEDIATELY stop all active timers when cycle determined to have ended
        self._stop_watchdog()  # Stop active cycle watchdog
        self._stop_state_expiry_timer()  # Cancel any pending progress reset
        self._clear_timer_pause_notification()
        prev_cycle_end_time = self._last_cycle_end_time
        self._last_cycle_end_time = dt_util.now()
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
                except (TypeError, ValueError, IndexError):
                    pass
            if len(valid) >= 2:
                try:
                    valid.sort(key=lambda x: x[0])
                    ts = np.array([v[0] for v in valid])
                    ps = np.array([v[1] for v in valid])
                    # Shared trapezoidal integrator with a data-driven outage gap
                    # (single source with ProfileStore.add_cycle).
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

        # Schedule heavy post-processing asynchronously. Capture this cycle's identity
        # token so the async tail can tell if a NEW cycle started while it was awaiting
        # (power changes are handled synchronously, so a back-to-back load can drive the
        # detector into a fresh RUNNING before post-processing completes). See B1 in
        # _async_process_cycle_end.
        end_token = self._ranking_snapshot_cycle_id
        self.hass.async_create_task(
            self._async_process_cycle_end(cycle_data, cycle_token=end_token)
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
            end_fn, _ = resolve_scorer("end", self.profile_store)
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

        Returns ``False`` (keep the proven slow path) when the ML/anomaly opt-in
        is off, there is too little history to trust the baseline, the cycle
        looks novel, or the drop is not anomalously early.  Never raises - the
        anomaly signal must never break detection.
        """
        try:
            from .ml.engine import ml_models_enabled

            if not ml_models_enabled(self.config_entry.options):
                return False
            earliest, peak_range = self._terminal_drop_baseline()
            return is_terminal_drop(
                points,
                earliest,
                peak_range,
                float(self.detector.config.stop_threshold_w),
                TERMINAL_DROP_EARLINESS_RATIO,
                TERMINAL_DROP_MIN_PEAK_RATIO,
                TERMINAL_DROP_PEAK_FAMILIAR_TOL,
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
                terminal_drop_baseline,
                cycles,
                stop_threshold,
                TERMINAL_DROP_MIN_QUIET_SPAN_S,
                TERMINAL_DROP_MIN_CLEAN_CYCLES,
            )
            self._terminal_drop_cache = (len(cycles), earliest, peak_range)
        except Exception as err:  # noqa: BLE001 - anomaly signal must never break detection
            self._logger.debug("Terminal-drop baseline refresh failed: %s", err)
            # Allow a later reading to retry the refresh for this count.
            if self._terminal_drop_refresh_n == n:
                self._terminal_drop_refresh_n = None

    def _ml_progress_percent(
        self,
        trace: list[tuple[datetime, float]],
        profile_name: str,
    ) -> float | None:
        """ML completion-fraction estimate (0-100) for the running cycle, or None.

        Uses the on-device ``remaining_time`` regressor (a ``standardized_linear``
        head with no shipped baseline) to predict how far through the cycle we
        are, learning this device's own progress curve rather than assuming the
        matched profile's median duration. Gated on the ML opt-in and only active
        once training has promoted a regressor; otherwise returns ``None`` so the
        caller keeps the proven phase-aware estimate untouched. Never raises.
        """
        return progress_mod.ml_progress_percent(
            self.profile_store,
            self.config_entry.options,
            float(self._matched_profile_duration or 0.0),
            trace,
            profile_name,
            self._profile_end_expectation,
            self._logger,
        )

    def _ml_energy_total(
        self,
        trace: list[tuple[datetime, float]],
        profile_name: str,
    ) -> float | None:
        """Predicted total cycle energy (Wh) from the on-device ``total_energy``
        regressor, or None.

        The regressor predicts the *energy-completion fraction* (energy so far ÷
        final energy); total = ``energy_so_far / fraction``. Because energy
        accumulates non-linearly (heating front-loads it), this is more stable —
        especially early — than dividing accumulated energy by *time* progress
        (the fallback in :meth:`_update_projected_energy`). Same gating as the
        remaining-time regressor: opt-in, inert until a model is promoted. Never
        raises.
        """
        return progress_mod.ml_energy_total(
            self.profile_store,
            self.config_entry.options,
            float(self._matched_profile_duration or 0.0),
            trace,
            profile_name,
            self._profile_end_expectation,
            self._logger,
        )

    def _compute_cycle_quality_score(self, cycle_data: dict[str, Any]) -> None:
        """Score a just-finished cycle with the hybrid_curve_quality model (opt-in).

        When ML models are enabled for this device, computes P(cycle is a problem)
        and stores it under ``cycle_data["ml_quality_score"]``.  A high score means
        the cycle may be mis-detected or corrupt; the learning manager uses it to
        downgrade auto-labeling to a feedback request so the user can confirm.
        Never raises — scoring failure is silently ignored to keep cycle storage safe.
        """
        try:
            from .ml.engine import ml_models_enabled, resolve_scorer
            from .ml.feature_extraction import quality_features

            if not ml_models_enabled(self.config_entry.options):
                return
            quality_fn, _ = resolve_scorer("quality", self.profile_store)
            if quality_fn is None:
                return

            profile_name = cycle_data.get("profile_name")
            if not profile_name:
                return

            points = decompress_power_data(cycle_data)
            if not points or len(points) < 4:
                return

            # Build profile median stats from stored labeled cycles.
            durations: list[float] = []
            energies: list[float] = []
            peaks: list[float] = []
            for c in self.profile_store.get_past_cycles():
                if c.get("profile_name") != profile_name:
                    continue
                if c.get("duration") is not None:
                    durations.append(float(c["duration"]))
                if c.get("energy_wh") is not None:
                    energies.append(float(c["energy_wh"]))
                if c.get("max_power") is not None:
                    peaks.append(float(c["max_power"]))

            if not durations:
                return

            med_dur = float(np.median(durations))
            med_energy = float(np.median(energies)) if energies else 500.0
            med_peak = float(np.median(peaks)) if peaks else 500.0

            match_conf = float(cycle_data.get("match_confidence") or 0.0)
            conf_known = match_conf > 0
            proxy_dist = max(0.0, 1.0 - match_conf) if conf_known else 0.25
            proxy_margin = match_conf if conf_known else 0.30
            proxy_fit = match_conf if conf_known else 0.75

            feat = quality_features(
                points=points,
                profile_median_duration_s=med_dur,
                profile_median_energy_wh=med_energy,
                profile_median_peak_w=med_peak,
                profile_distance=proxy_dist,
                label_margin=proxy_margin,
                profile_fit_score=proxy_fit,
                flag_count=len(cycle_data.get("artifacts", [])),
            )
            score = round(float(quality_fn(feat)), 3)
            cycle_data["ml_quality_score"] = score
            self._logger.debug(
                "ML quality score (profile=%s): %.3f", profile_name, score
            )
        except Exception:  # noqa: BLE001 - never break cycle storage
            pass

    def _resolve_energy_price(self) -> float | None:
        """Current energy price per kWh, or None when none is configured.

        A price entity (e.g. a dynamic tariff) takes precedence over the static
        value. Used to freeze each cycle's cost at completion time.
        """
        options = self.config_entry.options
        price_entity = options.get(CONF_ENERGY_PRICE_ENTITY)
        if price_entity:
            state = self.hass.states.get(price_entity)
            if state is not None:
                try:
                    return float(state.state)
                except (ValueError, TypeError):
                    pass
        static = options.get(CONF_ENERGY_PRICE_STATIC)
        if static is not None:
            try:
                return float(static)
            except (ValueError, TypeError):
                pass
        return None

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
        except (ValueError, TypeError, ZeroDivisionError):
            return ""
        try:
            if pct >= 1:
                return longer_template.format(pct=pct)
            if pct <= -1:
                return shorter_template.format(pct=abs(pct))
        except (KeyError, IndexError, ValueError):
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
        except (ValueError, TypeError):
            return ""

    async def _async_process_cycle_end(
        self, cycle_data: dict[str, Any], cycle_token: str | None = None
    ) -> None:
        """Process cycle completion asynchronously (heavy tasks).

        ``cycle_token`` is the ``_ranking_snapshot_cycle_id`` captured when this cycle
        ended. The terminal-state reset at the tail is skipped if a new cycle has
        started since (token changed), so back-to-back cycles are not clobbered (B1).
        """

        # FINAL PROFILE MATCH: If still detecting, try one last match with complete cycle data
        if self._current_program in ("detecting...", "restored..."):
            await self._run_final_match_from_cycle_data(cycle_data)

        # B1: freeze THIS cycle's live context into immutable locals now, before the
        # persistence / auto-label / lifetime-energy awaits below. A new cycle can
        # start synchronously during any of those awaits (a back-to-back load drives
        # the detector into a fresh RUNNING via _on_state_change, which rolls
        # _current_program back to "detecting..." and resets the match fields). The
        # tail (event payload, finish notification, learning inputs) must describe the
        # cycle that just finished, not whatever the live fields hold by the time each
        # await returns. The final match above is what determines these values for
        # this cycle, so capture right after it.
        program = self._current_program
        match_result = self._last_match_result
        match_confidence = self._last_match_confidence
        matched_profile_duration = self._matched_profile_duration
        cycle_anomaly = self._cycle_anomaly
        overrun_ratio = self._overrun_ratio

        # If we had a runtime match, attach the profile name for persistence
        if (
            program
            and program not in ("off", "detecting...", "restored...")
            and program in self.profile_store.get_profiles()
        ):
            cycle_data["profile_name"] = program
            cycle_data["label_source"] = "auto_match"
            if match_confidence:
                cycle_data["match_confidence"] = float(match_confidence)

        # Attach extensive debug data if available (and configured)
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

        # Post-Cycle Auto-Labeling (if not already matched)
        # Offload this match too if needed
        if not cycle_data.get("profile_name") and self._auto_label_confidence > 0:
            res = await self.profile_store.async_match_profile(
                cycle_data["power_data"], cycle_data["duration"]
            )
            if res.best_profile and res.confidence >= self._auto_label_confidence:
                cycle_data["profile_name"] = res.best_profile
                cycle_data["label_source"] = "auto_label_post"
                cycle_data["match_confidence"] = float(res.confidence)
                # Top-5 from post-cycle match (may differ from live match ranking).
                # Sanitize: strip heavy current/sample arrays (these fields are NOT
                # in the fired-event exclusion set, so they must stay small to keep
                # EVENT_CYCLE_ENDED under HA's 32KB event-data limit).
                cycle_data["match_ranking_top5"] = _sanitize_ranking(
                    getattr(res, "ranking", [])
                )
                self._logger.info(
                    "Post-cycle auto-labeled as '%s' (confidence: %.2f)",
                    res.best_profile,
                    res.confidence,
                )

        # Back-fill confirmed label on any ranking snapshots captured during this cycle
        # so the live_match on-device trainer can use them as labelled examples.
        _start_iso = cycle_data.get("start_time")
        _confirmed_profile = cycle_data.get("profile_name")
        if _start_iso and _confirmed_profile:
            try:
                self.profile_store.confirm_match_ranking_snapshots(
                    _start_iso,
                    _confirmed_profile,
                    # Use the token captured when THIS cycle ended, not the live
                    # field, which may already have rolled to a newly-started cycle
                    # during the awaits above (else this cycle's snapshots go
                    # unlabelled and a new cycle's snapshot gets mislabelled).
                    cycle_id=cycle_token or None,
                )
            except Exception:  # noqa: BLE001
                pass

        # Compute envelope conformance for the matched profile.
        # Stored as cycle_data["envelope_conformance"] so the panel and quality
        # gate can display/use it.  Only runs when we have a profile + power trace.
        _ep = cycle_data.get("profile_name")
        _pd = cycle_data.get("power_data")
        if _ep and isinstance(_pd, list) and len(_pd) >= 4:
            try:
                from .time_utils import power_data_to_offsets  # noqa: PLC0415
                _pts = [(float(o), float(p)) for o, p in power_data_to_offsets(_pd, _start_iso)]
                if len(_pts) >= 4:
                    conformance_rec = self.profile_store.compute_envelope_conformance(_ep, _pts)
                    if conformance_rec is not None:
                        cycle_data["envelope_conformance"] = conformance_rec.get("conformance")
                    # Transient artifacts (door-open pauses, out-of-band dips/spikes)
                    # for graph markers + a Cycles-list badge; [] when none.
                    artifacts = self.profile_store.detect_cycle_artifacts(_ep, _pts)
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

        # A1: Underrun check — computed post-cycle only, not a live signal.
        # Only applied when no runtime anomaly was detected (underrun and overrun are mutually exclusive).
        try:
            if not cycle_data.get("anomaly") or cycle_data["anomaly"] == "none":
                _uc_profile = cycle_data.get("profile_name")
                _uc_dur = float(cycle_data.get("duration", 0))
                if _uc_profile and _uc_dur > 0:
                    _uc_median = self.profile_store.get_profile_median_duration(_uc_profile)
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

        # A2: Energy spike/low anomaly — stored separately from duration anomaly.
        try:
            _ea_profile = cycle_data.get("profile_name")
            _ea_energy = float(cycle_data.get("energy_wh", 0))
            if _ea_profile and _ea_energy > 0:
                _ea_stats = self.profile_store.get_profile_energy_stats(_ea_profile)
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

        # Freeze the energy cost onto the cycle using the price in effect NOW, so
        # later price changes never rewrite historical costs. Stored as a number
        # (currency per the configured price's unit); absent when no price is set.
        price = self._resolve_energy_price()
        if price is not None:
            cycle_data["energy_price"] = price
            cycle_data["cost"] = round(cycle_data.get("energy_wh", 0.0) / 1000.0 * price, 4)

        # Score cycle quality with the ML model before persisting so the score is
        # stored on the cycle record and available to the learning manager immediately.
        # Must run BEFORE async_add_cycle so get_past_cycles() inside the scorer does
        # not yet include the current cycle, keeping reference statistics uncontaminated.
        # Only opted-in devices reach the scorer, and only then is it offloaded to the
        # executor: on a long trace its NumPy feature extraction is O(N) and must not
        # block the event loop (mirrors the profile matcher). Gating here avoids a
        # pointless thread-hop for the default (ML-off) case where the scorer no-ops.
        # The scorer mutates only cycle_data (nothing else touches it here) and never
        # raises, so this is executor-safe.
        from .ml.engine import ml_models_enabled  # noqa: PLC0415

        if ml_models_enabled(self.config_entry.options):
            await self.hass.async_add_executor_job(
                self._compute_cycle_quality_score, cycle_data
            )

        # Add cycle to store immediately (still sync but offloadable parts optimized
        # internally if possible)
        # Note: add_cycle is mostly safe (signature calc is O(N) but fast enough for
        # single cycle).
        # We could offload signature calc to analysis logic if really needed, but let's
        # stick to match profile optimization first.
        cycle_persisted = False
        try:
            await self.profile_store.async_add_cycle(cycle_data)
            cycle_persisted = True
            # The cycle (with its restart_gaps) is now durably stored, so it is safe
            # to drop the live buffer. Doing this only after a confirmed persist means
            # a failed save keeps the gaps for the next cycle-end attempt.
            if restart_gaps_snapshot is not None:
                self._restart_gaps.clear()
            profile_name = cycle_data.get("profile_name")
            if profile_name:
                await self.profile_store.async_rebuild_envelope(profile_name)
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
        if cycle_persisted:
            try:
                prev_lifetime_count = self._lifetime_cycle_count()
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
                    cycle_data.get("energy_wh", 0.0)
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
            self.hass.async_create_task(self.profile_store.async_clear_active_cycle())

        # Auto post-process: merge fragmented cycles from last 3 hours
        self.hass.async_create_task(self._run_post_cycle_processing())

        # Prepare cycle data for event (enrich if needed)
        # IMPORTANT: Exclude large fields to prevent exceeding HA's 32KB event data limit
        excluded_fields = {"power_data", "debug_data", "power_trace"}
        event_cycle_data = {
            k: v for k, v in cycle_data.items() if k not in excluded_fields
        }
        event_cycle_data["device_type"] = self.device_type
        # Add program if missing or generic (use THIS cycle's captured program, not
        # the live field which may already belong to a newly-started cycle).
        if "profile_name" not in event_cycle_data and program:
            event_cycle_data["profile_name"] = program

        if self._notify_fire_events:
            self.hass.bus.async_fire(
                EVENT_CYCLE_ENDED,
                {
                    "entry_id": self.entry_id,
                    "device_name": self.config_entry.title,
                    "cycle_data": event_cycle_data,
                    "program": event_cycle_data.get("profile_name", "unknown"),
                    "duration": event_cycle_data.get("duration"),
                    "start_time": event_cycle_data.get("start_time"),
                    "end_time": event_cycle_data.get("end_time") or dt_util.now().isoformat(),
                },
            )

        # Purge pending live entries and reset counters, but don't send a service-level
        # clear: the finished notification below reuses the lifecycle tag and replaces
        # the live card in place (sending a clear first would cause a dismiss/recreate
        # flicker). The action-based clear marker still fires for action templates.
        self._clear_live_progress_notification(clear_services=False)

        # Send notification if enabled
        if self._notify_finish_services or self._notify_actions:
            msg_template = self.config_entry.options.get(CONF_NOTIFY_FINISH_MESSAGE, DEFAULT_NOTIFY_FINISH_MESSAGE)
            duration_min = int(cycle_data['duration'] / 60)
            program_name = event_cycle_data.get("profile_name", "unknown")

            energy_kwh = round(cycle_data.get("energy_wh", 0.0) / 1000, 3)

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
                program=program_name,
                energy_kwh=f"{energy_kwh:.3f}",
                cost=cost_str,
                time_finished=time_finished,
                cycle_count=finished_cycle_count,
                vs_typical=vs_typical,
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
                    # Same lifecycle tag as start/live so the finished alert replaces
                    # the live notification in place. No live_update/alert_once here,
                    # so the companion app surfaces it with sound.
                    "tag": self._lifecycle_tag,
                    # C3: end the iOS Live Activity (mobile_app_* only downstream).
                    "activity": "end",
                },
            )

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
            self.learning_manager.process_cycle_end(
                cycle_data,
                detected_profile=program,
                confidence=match_confidence or 0.0,
                predicted_duration=matched_profile_duration,
                match_result=match_result,
            )

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
        self._notified_pre_completion = False
        self._time_remaining = None
        self._matched_profile_duration = None
        self._last_estimate_time = None
        self._last_match_result = None  # Clear so phase sensor resets to "Off" (issue #192)
        self._cycle_progress = 100.0  # 100% = cycle complete
        self._cycle_completed_time = dt_util.now()
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
        if self._door_sensor_entity:
            door_state = self.hass.states.get(self._door_sensor_entity)
            if door_state and door_state.state == "off":  # binary_sensor: off = closed
                self._is_clean_state = True
                self._clean_state_start = dt_util.now()
                self._logger.debug(
                    "Cycle ended with door closed: entering Clean state"
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
        return notif_rules.in_quiet_hours(
            self._quiet_hours_bounds(), when or dt_util.now()
        )

    def _seconds_until_quiet_end(self, when: datetime | None = None) -> float:
        """Seconds from ``when`` until the next end-of-quiet-window boundary (end:00).

        Returns 0.0 when the feature is off or when not currently in quiet hours.
        """
        return notif_rules.seconds_until_quiet_end(
            self._quiet_hours_bounds(), when or dt_util.now()
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
        except (TypeError, ValueError):
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
                return False

        actions_sent = False
        if self._notify_actions:
            actions_sent = bool(self._run_notification_actions(variables))

        # If actions fired and there are no per-event services, skip the
        # service/persistent-notification path entirely.
        services = self._get_services_for_event(event_type)
        if actions_sent and not services:
            return True

        service_sent = self._send_notification_service(
            message,
            services=services,
            title=title,
            icon=icon,
            event_type=event_type,
            extra_vars=extra_vars,
        )
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

            state = (
                self.hass.states.get(notify_service)
                if notify_service.startswith("notify.")
                else None
            )
            if state is not None and getattr(state, "domain", None) == "notify":
                service_data: dict[str, Any] = {
                    "entity_id": notify_service,
                    "message": message,
                }
                if title:
                    service_data["title"] = title
                if svc_data:
                    service_data["data"] = svc_data
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
                service_data = {"message": message, "title": title}
                if svc_data:
                    service_data["data"] = svc_data
                self.hass.async_create_task(
                    self.hass.services.async_call(domain, service, service_data)
                )
            sent = True

        if not sent:
            if event_type == NOTIFY_EVENT_LIVE:
                return False
            # Reuse the notification's tag as a stable persistent-notification id so
            # the HA notifications tab collapses the lifecycle thread to one entry
            # instead of accumulating a new card per cycle (issue #248/#249 clutter).
            _pn_create(
                self.hass,
                message,
                title=title,
                notification_id=ev.get("tag"),
            )
            return True

        return sent

    def _run_notification_actions(self, variables: dict[str, Any]) -> bool:
        """Run configured notification actions."""
        actions: list[dict[str, Any]] = self._notify_actions
        if not actions:
            return False

        try:
            script = script_helper.Script(
                self.hass,
                actions,
                name=f"{self.config_entry.title} notification",
                domain=DOMAIN,
                logger=_LOGGER,
            )
        except (ValueError, TypeError, HomeAssistantError) as err:
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

        try:
            self.hass.async_create_task(
                script.async_run(variables, context=Context())
            )
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

        person_entity_id = new_state.entity_id
        person_name = new_state.name or new_state.attributes.get(
            "friendly_name", person_entity_id
        )
        pending: list[dict[str, Any]] = list(self._pending_notifications)
        self._pending_notifications = []
        for entry in pending:
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
                    self._last_live_notification_time = dt_util.now()

    def _handle_noise_cycle(self, max_power: float) -> None:
        """Handle a detected noise cycle."""
        # Clean up old noise events > 24h
        now = dt_util.now()
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
            self.hass.async_create_task(self._tune_threshold())

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
            self._cycle_progress = 0.0
            self._projected_energy_wh = None
            self._projected_cost = None
            self._cycle_anomaly = "none"
            self._overrun_ratio = 0.0
            self._last_match_result = None
            self._notify_update()
            return

        now = dt_util.now()

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
            self._notify_update()
            return

        # No matching task trigger here anymore!
        # The detector callback handles it.
        # Just update progress/remaining based on existing match.
        self._update_remaining_only()
        self._check_pre_completion_notification()
        self._check_live_progress_notification()
        self._notify_update()

    # _async_run_matching removed in favor of _async_perform_combined_matching

    def _analyze_trend(self, profile_name: str) -> bool:
        """Analyze score history to detect positive trend.

        Returns True if score has increased in at least 7 of the last 10 intervals.
        Requires at least 5 samples history to make a determination.
        """
        history = self._score_history.get(profile_name, [])
        if len(history) < 5:
            return False

        # Use last 11 points to get 10 intervals (or fewer if history short)
        recent = history[-11:]
        if len(recent) < 2:
            return False

        up_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        total_intervals = len(recent) - 1

        # Proportional threshold (7/10 => 0.7)
        return (up_count / total_intervals) >= 0.70

    def _reset_live_notification_state(self) -> None:
        """Reset per-cycle live notification counters and timers."""
        self._live_notification_sent_count = 0
        self._live_notification_cap = 0
        self._last_live_notification_time = None
        self._live_waiting_notification_sent = False
        self._live_chronometer_overrun_sent = False
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

    def _check_live_progress_notification(self) -> None:
        """Send throttled live progress notifications for compatible mobile targets."""
        if not self._notify_live_services and not self._notify_actions:
            return
        if self.detector.state not in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            return

        has_profile_match = bool(
            self._matched_profile_duration and self._matched_profile_duration > 0
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
            sent = self._dispatch_notification(
                msg,
                event_type=NOTIFY_EVENT_LIVE,
                extra_vars=waiting_extra_vars,
            )
            self._live_waiting_notification_sent = sent
            if sent:
                self._live_activity_started = True
            return

        interval = max(30, int(self._notify_live_interval_seconds))
        now = dt_util.now()
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
        sent = self._dispatch_notification(
            msg,
            event_type=NOTIFY_EVENT_LIVE,
            extra_vars=extra_vars,
        )
        if sent:
            self._live_activity_started = True
            if chronometer_overrun:
                self._live_chronometer_overrun_sent = True
            else:
                self._live_notification_sent_count += 1
            self._last_live_notification_time = now

    def _clear_live_progress_notification(self, clear_services: bool = True) -> None:
        """Clear active live/progress notifications and purge stale deferred alerts.

        On cycle finish (``clear_services=False``) the finished notification carries
        the same lifecycle tag and replaces the live notification in place, so we must
        NOT also send a service-level ``clear_notification`` (it would briefly dismiss
        then re-create the card). The pending-purge, the action-based clear marker
        (kept for backward compatibility with custom action templates), and the state
        reset still run. On shutdown (``clear_services=True``) no finished notification
        follows, so the explicit service clear is required to dismiss the live card.
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
                "message": "clear_notification",  # Clear marker for action handlers
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
            self._send_notification_service(
                "clear_notification",
                services=self._notify_live_services,
                event_type=NOTIFY_EVENT_LIVE,
                extra_vars={
                    "tag": self._live_notification_tag,
                    "live_update": True,
                    "alert_once": True,
                    # C3: iOS Live Activity end marker (mobile_app_* only downstream).
                    "activity": "end",
                },
            )

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

        services = self._get_services_for_event(NOTIFY_EVENT_CLEAN)
        if not services and not self._notify_actions:
            return

        if self._notify_actions:
            self._run_notification_actions(
                {
                    "device": self.config_entry.title,
                    "program": "",
                    "message": "clear_notification",
                    "title": "",
                    "icon": None,
                    "event_type": NOTIFY_EVENT_CLEAN,
                    "person_entity_id": None,
                    "person_name": None,
                    "tag": self._clean_tag,
                }
            )
        if services:
            self._send_notification_service(
                "clear_notification",
                services=services,
                event_type=NOTIFY_EVENT_CLEAN,
                extra_vars={"tag": self._clean_tag},
            )

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
        )
        self._projected_energy_wh = wh
        self._projected_cost = cost

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
            return

        now = dt_util.now()
        if (
            self._last_phase_estimate_time
            and (now - self._last_phase_estimate_time).total_seconds() < 5.0
        ):
            return
        self._last_phase_estimate_time = now

        # Use net elapsed (wall-clock minus user-paused time) for all time estimates
        # so that paused time is excluded from progress / remaining / total duration.
        duration_so_far = float(self.net_elapsed_seconds)
        self._check_cycle_timers(duration_so_far)

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
            self._logger.debug(
                "No profile matched yet, elapsed=%smin", int(duration_so_far / 60)
            )
            return

        # Compute the phase-aware and ML progress inputs via the manager's own
        # wrappers (so per-call caching + test mocks apply), then hand them to the
        # shared pure smoothing/back-calc in :mod:`progress` - the identical math
        # the Playground simulation runs.
        trace = self.detector.get_power_trace()
        phase_result = None
        if len(trace) >= 10 and self._current_program != "detecting...":
            phase_result = self._estimate_phase_progress(
                trace, duration_so_far, self._current_program
            )
        ml_pct = self._ml_progress_percent(trace, self._current_program)
        result = progress_mod.compute_progress(
            self.device_type,
            float(self._matched_profile_duration),
            duration_so_far,
            self._smoothed_progress,
            phase_result,
            ml_pct,
            self._logger,
        )

        self._cycle_progress = result.progress
        self._smoothed_progress = result.smoothed
        self._time_remaining = result.remaining
        self._total_duration = result.total
        self._last_total_duration_update = now
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
                    "clear_notification",
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
        )

    def _notify_update(self) -> None:
        """Notify entities of update."""
        async_dispatcher_send(self.hass, SIGNAL_WASHER_UPDATE.format(self.entry_id))

    def notify_update(self) -> None:
        """Public method to notify entities of update."""
        self._notify_update()

    @property
    def is_user_paused(self) -> bool:
        """Return True if cycle is currently user-paused."""
        return self._is_user_paused

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
            paused += (dt_util.now() - self._user_pause_start).total_seconds()
        return max(0.0, raw - paused)

    def check_state(self):
        """Return current detector state."""
        if self.recorder.is_recording:
            return STATE_RUNNING
        # A completed cycle ends in STATE_FINISHED, not STATE_OFF; accept both
        # or the door-sensor Clean state (#153) is never surfaced (#282).
        if self._is_clean_state and self.detector.state in (
            STATE_OFF,
            STATE_FINISHED,
        ):
            return STATE_CLEAN
        if self._is_user_paused:
            return STATE_USER_PAUSED
        return self.detector.state

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
    def last_total_duration_update(self) -> datetime | None:
        """Return when total duration was last refined."""
        return self._last_total_duration_update

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
        """Runtime anomaly state for the current cycle ("none" | "overrun")."""
        return self._cycle_anomaly

    @property
    def overrun_ratio(self) -> float:
        """Elapsed / expected duration for the running cycle (0.0 when unknown)."""
        return self._overrun_ratio

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
        """Maintenance event types whose reminder threshold has been reached (E2).

        Surfaced as a state-sensor attribute + read by the panel banner. Never a
        notification. Returns an empty list on any error.
        """
        try:
            cfg = self.config_entry.options.get(CONF_MAINTENANCE_REMINDER_CYCLES)
            if not isinstance(cfg, dict) or not cfg:
                cfg = DEFAULT_MAINTENANCE_REMINDER_CYCLES
            return self.profile_store.get_maintenance_due(cfg)
        except Exception:  # noqa: BLE001
            return []

    @property
    def current_power(self):
        """Return current power reading in watts."""
        return self._current_power

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
        return len(self.detector.get_power_trace())

    @property
    def sample_interval_stats(self):
        """Return statistics about sampling intervals."""
        return self._sample_interval_stats

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
        cutoff = dt_util.now().timestamp() - 86400.0
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
            except (TypeError, ValueError):
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
    def manual_program_active(self) -> bool:
        """Return True if a manual program override is active."""
        return getattr(self, "_manual_program_active", False)

    def set_manual_program(self, profile_name: str) -> None:
        """Manually set the current program."""
        if self.detector.state != "running":
            return
        profiles_raw: Any = None
        try:
            profiles_raw = self.profile_store.get_profiles()
        except Exception:  # pylint: disable=broad-exception-caught
            profiles_raw = None

        if isinstance(profiles_raw, dict):
            profiles: dict[str, Any] = cast(dict[str, Any], profiles_raw)
        else:
            profiles_fallback = getattr(self.profile_store, "_data", {}).get(
                "profiles", {}
            )
            profiles = (
                cast(dict[str, Any], profiles_fallback)
                if isinstance(profiles_fallback, dict)
                else {}
            )

        if profile_name not in profiles:
            self._logger.warning("Cannot set manual program: '%s' not found", profile_name)
            return

        self._current_program = profile_name
        self._manual_program_active = True

        # Update expected duration immediately
        profile = profiles.get(profile_name)
        if profile:
            avg = float(profile.get("avg_duration", 0.0))
            if avg > 0:
                self._matched_profile_duration = avg
                self._logger.info(
                    "Manual program set to %s, duration=%.0fs", profile_name, avg
                )

                # Update estimates if running
                if self.detector.state == "running":
                    self._update_estimates()

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
        self._user_pause_start = dt_util.now()
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

        snapshot = self.detector.get_state_snapshot()
        snapshot["manual_program"] = self._manual_program_active
        snapshot["notified_start"] = self._notified_start
        snapshot["start_event_fired"] = self._start_event_fired
        snapshot["is_user_paused"] = self._is_user_paused
        snapshot["user_pause_start"] = (
            self._user_pause_start.isoformat() if self._user_pause_start else None
        )
        snapshot["total_user_paused_seconds"] = self._total_user_paused_seconds
        self.hass.async_create_task(self.profile_store.async_save_active_cycle(snapshot))
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

        now = dt_util.now()
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

        snapshot = self.detector.get_state_snapshot()
        snapshot["manual_program"] = self._manual_program_active
        snapshot["notified_start"] = self._notified_start
        snapshot["start_event_fired"] = self._start_event_fired
        snapshot["is_user_paused"] = self._is_user_paused
        snapshot["user_pause_start"] = (
            self._user_pause_start.isoformat() if self._user_pause_start else None
        )
        snapshot["total_user_paused_seconds"] = self._total_user_paused_seconds
        self.hass.async_create_task(self.profile_store.async_save_active_cycle(snapshot))
        self._notify_update()
        return True

    async def async_terminate_cycle(self) -> None:
        """Force terminate the current cycle via user request."""
        self._logger.warning("Force terminating cycle by user request")

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
        """Clear manual program override."""
        if not self._manual_program_active:
            return

        self._manual_program_active = False
        # If running, revert to detecting so auto-detection can resume?
        if self.detector.state == "running":
            self._current_program = "detecting..."
            self._matched_profile_duration = None
            self._update_estimates()  # Trigger immediate re-detection attempt
        else:
            # If not running, clear the forced program
            self._current_program = "off"
            self._matched_profile_duration = None

        self._notify_update()
        self._logger.info("Manual program cleared, reverting to auto-detection")

    async def _run_post_cycle_processing(self) -> None:
        """Run post-cycle processing (merge fragments, split anomalies)."""
        try:
            # User Feedback: Use 5 hour lookback and configured gap settings
            stats = await self.profile_store.async_run_maintenance()

            # Log significant actions
            merged = stats.get("merged_cycles", 0)
            split = stats.get("split_cycles", 0)
            if merged > 0 or split > 0:
                self._logger.info(
                    "Post-cycle processing: Merged %s, Split %s cycle(s)", merged, split
                )

            # Note: async_run_maintenance saves automatically if changes occur
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Post-cycle processing failed: %s", e)