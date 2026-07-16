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
"""Constants for the WashData integration."""

from enum import StrEnum

DOMAIN = "ha_washdata"


class TerminationReason(StrEnum):
    """Why a cycle ended. StrEnum members equal their string value, so existing
    string comparisons and JSON serialisation keep working unchanged."""

    TIMEOUT = "timeout"          # low-power off_delay elapsed (normal completion)
    SMART = "smart"              # smart-termination heuristic finished the cycle
    FORCE_STOPPED = "force_stopped"  # watchdog / no-update force end
    USER = "user"                # user manually stopped the cycle
    TERMINAL_DROP = "terminal_drop"  # anomalously-early hard cliff-to-0 (opt-in)


# Completed cycles stay eligible for anti-wrinkle handling only for these
# reasons (a user-stopped cycle is intentionally excluded).
ANTI_WRINKLE_ELIGIBLE_REASONS = frozenset(
    {TerminationReason.TIMEOUT, TerminationReason.SMART}
)

# Configuration keys
CONF_POWER_SENSOR = "power_sensor"
CONF_NAME = "name"
CONF_MIN_POWER = "min_power"
CONF_OFF_DELAY = "off_delay"
CONF_NOTIFY_SERVICE = "notify_service"  # Deprecated - kept for migration only
CONF_NOTIFY_ACTIONS = "notify_actions"
CONF_NOTIFY_PEOPLE = "notify_people"
CONF_NOTIFY_ONLY_WHEN_HOME = "notify_only_when_home"
CONF_NOTIFY_FIRE_EVENTS = "notify_fire_events"
CONF_NOTIFY_EVENTS = "notify_events"  # Deprecated - kept for migration only
CONF_NOTIFY_START_SERVICES = "notify_start_services"
CONF_NOTIFY_FINISH_SERVICES = "notify_finish_services"
CONF_NOTIFY_LIVE_SERVICES = "notify_live_services"
CONF_NOTIFY_CYCLE_TIMERS = "notify_cycle_timers"
CONF_NO_UPDATE_ACTIVE_TIMEOUT = "no_update_active_timeout"
CONF_LOW_POWER_NO_UPDATE_TIMEOUT = "low_power_no_update_timeout"
CONF_SMOOTHING_WINDOW = "smoothing_window"
CONF_SAMPLING_INTERVAL = "sampling_interval"
CONF_START_DURATION_THRESHOLD = (
    "start_duration_threshold"  # Debounce for start detection
)
CONF_DEVICE_TYPE = "device_type"
CONF_PROFILE_DURATION_TOLERANCE = "profile_duration_tolerance"


CONF_INTERRUPTED_MIN_SECONDS = "interrupted_min_seconds"  # Internal use only
CONF_PROGRESS_RESET_DELAY = "progress_reset_delay"
CONF_LEARNING_CONFIDENCE = "learning_confidence"
CONF_DURATION_TOLERANCE = "duration_tolerance"
CONF_AUTO_LABEL_CONFIDENCE = "auto_label_confidence"
CONF_AUTO_MAINTENANCE = "auto_maintenance"
CONF_PROFILE_MATCH_INTERVAL = "profile_match_interval"
CONF_PROFILE_MATCH_MIN_DURATION_RATIO = "profile_match_min_duration_ratio"
CONF_PROFILE_MATCH_MAX_DURATION_RATIO = "profile_match_max_duration_ratio"
CONF_MAX_PAST_CYCLES = "max_past_cycles"
CONF_MAX_FULL_TRACES_PER_PROFILE = "max_full_traces_per_profile"
CONF_MAX_FULL_TRACES_UNLABELED = "max_full_traces_unlabeled"
CONF_WATCHDOG_INTERVAL = "watchdog_interval"  # Derived from sampling_interval
CONF_MATCH_PERSISTENCE = "match_persistence"
CONF_COMPLETION_MIN_SECONDS = "completion_min_seconds"
CONF_NOTIFY_BEFORE_END_MINUTES = "notify_before_end_minutes"
CONF_RUNNING_DEAD_ZONE = "running_dead_zone"  # Seconds after start to ignore power dips
CONF_END_REPEAT_COUNT = "end_repeat_count"  # Number of times end condition must be met
CONF_MIN_OFF_GAP = "min_off_gap"  # Minimum gap to separate cycles (seconds)
CONF_START_ENERGY_THRESHOLD = "start_energy_threshold"  # Wh required to confirm start
CONF_END_ENERGY_THRESHOLD = "end_energy_threshold"  # Wh allowed during end candidates
CONF_START_THRESHOLD_W = "start_threshold_w"  # Custom power threshold for STARTING
CONF_STOP_THRESHOLD_W = (
    "stop_threshold_w"  # Custom power threshold for ENDING (hysteresis)
)
CONF_POWER_OFF_THRESHOLD_W = (
    "power_off_threshold_w"  # W; 0 = disabled. Terminal Finished/Clean -> Off when
)  # smoothed power stays below this (must sit below stop_threshold_w when > 0)
CONF_POWER_OFF_DELAY = (
    "power_off_delay"  # Seconds below the power-off threshold before Finished/Clean -> Off
)
CONF_EXPOSE_DEBUG_ENTITIES = "expose_debug_entities"  # Expose detailed debug sensors
CONF_SAVE_DEBUG_TRACES = (
    "save_debug_traces"  # Improve historical cycle data with rich debug info
)
# Cycle interruption detection settings (not exposed in UI, but used internally)
CONF_ABRUPT_DROP_WATTS = "abrupt_drop_watts"  # Power cliff threshold for interrupted status
CONF_ABRUPT_DROP_RATIO = "abrupt_drop_ratio"  # Relative drop ratio for interrupted status
CONF_ABRUPT_HIGH_LOAD_FACTOR = "abrupt_high_load_factor"  # High load factor threshold
CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD = "auto_tune_noise_events_threshold"  # Noise events before auto-tune
CONF_EXTERNAL_END_TRIGGER_ENABLED = "external_end_trigger_enabled"  # Enable external cycle end trigger
CONF_EXTERNAL_END_TRIGGER = "external_end_trigger"  # Binary sensor entity for external cycle end
CONF_EXTERNAL_END_TRIGGER_INVERTED = "external_end_trigger_inverted"  # Invert external trigger logic (trigger on OFF)
CONF_ANTI_WRINKLE_ENABLED = "anti_wrinkle_enabled"  # Dryer anti-wrinkle shielding
CONF_ANTI_WRINKLE_MAX_POWER = "anti_wrinkle_max_power"  # W threshold for anti-wrinkle spikes
CONF_ANTI_WRINKLE_MAX_DURATION = "anti_wrinkle_max_duration"  # Seconds to treat as anti-wrinkle
CONF_ANTI_WRINKLE_EXIT_POWER = "anti_wrinkle_exit_power"  # W threshold for true-off exit
CONF_DELAY_START_DETECT_ENABLED = "delay_start_detect_enabled"  # Enable delayed-start detection
CONF_DELAY_CONFIRM_SECONDS = "delay_confirm_seconds"  # Seconds power must stay in standby band before DELAY_WAIT engages
CONF_DELAY_TIMEOUT_HOURS = "delay_timeout_hours"  # Safety timeout (hours) while waiting to start
# Note: the deprecated 0.4.5 drain-spike keys (delay_drain_*) are stripped during
# config migration in __init__.py using raw string literals; no constants needed.


NOTIFY_EVENT_START = "cycle_start"
NOTIFY_EVENT_FINISH = "cycle_finish"
NOTIFY_EVENT_LIVE = "cycle_live"
NOTIFY_EVENT_CLEAN = "cycle_clean"  # Laundry still inside after cycle ends
NOTIFY_EVENT_TIMER = "cycle_timer"  # User-configured mid-cycle countdown timer

CONF_NOTIFY_TITLE = "notify_title"
CONF_NOTIFY_ICON = "notify_icon"
CONF_NOTIFY_START_MESSAGE = "notify_start_message"
CONF_NOTIFY_FINISH_MESSAGE = "notify_finish_message"
CONF_NOTIFY_PRE_COMPLETE_MESSAGE = "notify_pre_complete_message"
CONF_NOTIFY_LIVE_INTERVAL_SECONDS = "notify_live_interval_seconds"
CONF_NOTIFY_LIVE_OVERRUN_PERCENT = "notify_live_overrun_percent"
CONF_NOTIFY_LIVE_CHRONOMETER = "notify_live_chronometer"
CONF_NOTIFY_REMINDER_MESSAGE = "notify_reminder_message"  # Distinct one-time pre-end alert
CONF_NOTIFY_TIMEOUT_SECONDS = "notify_timeout_seconds"  # Auto-dismiss after N seconds (0 = never)
CONF_NOTIFY_CHANNEL = "notify_channel"  # Android channel for status/live/reminder
CONF_NOTIFY_FINISH_CHANNEL = "notify_finish_channel"  # Distinct Android channel for finished/clean
CONF_ENERGY_PRICE_STATIC = "energy_price_static"
CONF_ENERGY_PRICE_ENTITY = "energy_price_entity"
# Peak-rate awareness: when the current price meets/exceeds this threshold, the
# start notification gets an informational tip appended (purely advisory).
CONF_PEAK_RATE_THRESHOLD = "peak_rate_threshold"
CONF_PEAK_RATE_MESSAGE = "peak_rate_message"

# Door sensor & pause
CONF_DOOR_SENSOR_ENTITY = "door_sensor_entity"  # Optional binary_sensor for machine door
CONF_PAUSE_CUTS_POWER = "pause_cuts_power"  # Also turn off switch entity when pausing
CONF_SWITCH_ENTITY = "switch_entity"  # Optional switch entity toggled on pause/resume
CONF_NOTIFY_UNLOAD_DELAY_MINUTES = "notify_unload_delay_minutes"  # Minutes before "laundry waiting" nag
CONF_NOTIFY_UNLOAD_MESSAGE = "notify_unload_message"  # Template for the clean-laundry nag message

# Quiet hours (do-not-disturb window). Both hours 0-23; unset/None (or start==end)
# = feature off. When configured, finish-type notifications (finish, clean-laundry
# nag, pre-complete/reminder, milestone) that would fire inside the window are held
# and delivered at the end of the window. Live-progress ticks and the start
# notification are never delayed.
CONF_NOTIFY_QUIET_START_HOUR = "notify_quiet_start_hour"
CONF_NOTIFY_QUIET_END_HOUR = "notify_quiet_end_hour"

# Milestone (cycle-count achievement) notifications. A list of lifetime completed-
# cycle counts; a single milestone notification fires when the device's lifetime
# count crosses one of these values. Empty/malformed list = no-op.
CONF_NOTIFY_MILESTONES = "notify_milestones"
CONF_NOTIFY_MILESTONE_MESSAGE = "notify_milestone_message"

# Optional link to an existing HA device (e.g. the smart plug or appliance).
# When set, the WashData device is exposed as "Connected via <device>" through
# the device registry's via_device relationship. Stores a device registry id.
CONF_LINKED_DEVICE = "linked_device"

DEFAULT_NOTIFY_TITLE = "WashData: {device}"
DEFAULT_NOTIFY_START_MESSAGE = "{device} started."
DEFAULT_NOTIFY_FINISH_MESSAGE = "{device} finished. Duration: {duration}m."
DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE = "{device}: Less than {minutes} minutes remaining."
DEFAULT_NOTIFY_REMINDER_MESSAGE = "{device}: about {minutes} minutes left."
DEFAULT_NOTIFY_LIVE_WAITING_MESSAGE = "{device}: No profile matched yet."
DEFAULT_NOTIFY_ONLY_WHEN_HOME = False
DEFAULT_NOTIFY_FIRE_EVENTS = True
DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS = 300
DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT = 20
DEFAULT_NOTIFY_LIVE_CHRONOMETER = False
DEFAULT_NOTIFY_TIMEOUT_SECONDS = 0  # 0 = notifications never auto-dismiss
DEFAULT_NOTIFY_CHANNEL = ""  # Empty = omit channel (companion app default)
DEFAULT_NOTIFY_FINISH_CHANNEL = ""  # Empty = reuse status channel
DEFAULT_NOTIFY_UNLOAD_DELAY_MINUTES = 60  # 1 hour before "still waiting" nag notification
DEFAULT_NOTIFY_UNLOAD_MESSAGE = "{device} finished {duration}m ago - laundry is still inside."
DEFAULT_PEAK_RATE_MESSAGE = "Running at peak rate ({price}/kWh)."

# Quiet hours default: feature off (both hours unset). See CONF_NOTIFY_QUIET_*.
DEFAULT_NOTIFY_QUIET_START_HOUR = None
DEFAULT_NOTIFY_QUIET_END_HOUR = None

# Milestone notification defaults.
DEFAULT_NOTIFY_MILESTONES = [50, 100, 500, 1000]
DEFAULT_NOTIFY_MILESTONE_MESSAGE = "{device} has completed {cycle_count} cycles!"

# Defaults
DEFAULT_MIN_POWER = 2.0  # Watts
DEFAULT_OFF_DELAY = 180  # Seconds (3 minutes, safer for 60s polling)
DEFAULT_NAME = "Washing Machine"
# Seconds without updates while active before forced stop (publish-on-change sockets)
DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT = 600  # 10 minutes
DEFAULT_SMOOTHING_WINDOW = 2
DEFAULT_SAMPLING_INTERVAL = 30.0  # Seconds
DEFAULT_START_DURATION_THRESHOLD = 5.0  # Seconds (debounce)
DEFAULT_END_ENERGY_THRESHOLD = 0.05  # Wh - Require effectively zero energy to end
DEFAULT_DEVICE_TYPE = "washing_machine"
DEFAULT_PROFILE_DURATION_TOLERANCE = 0.25

DEFAULT_INTERRUPTED_MIN_SECONDS = 150  # Internal use only, not exposed

DEFAULT_PROGRESS_RESET_DELAY = 1800  # Seconds (30 minutes state expiry/unload window)

# Power-based Off detection (issue #284; opt-in, default off). Threshold 0 = disabled
# (the enable marker); when > 0 it must sit BELOW stop_threshold_w (beneath the idle/
# standby floor) or it is ignored. The delay is a short debounce that is safe to keep
# small because it only applies in the terminal state (no soak risk there). When enabled,
# power-off owns the terminal -> Off transition and the progress-reset timer no longer
# forces Off (the terminal state persists until the machine is actually switched off).
DEFAULT_POWER_OFF_THRESHOLD_W = 0.0  # Disabled
DEFAULT_POWER_OFF_DELAY = 30  # Seconds
DEFAULT_LEARNING_CONFIDENCE = 0.6  # Minimum confidence to request user verification
DEFAULT_DURATION_TOLERANCE = 0.10  # Allow ±10% duration variance before flagging
DEFAULT_AUTO_LABEL_CONFIDENCE = 0.9  # High confidence auto-label threshold
DEFAULT_AUTO_MAINTENANCE = True  # Enable nightly cleanup by default
DEFAULT_COMPLETION_MIN_SECONDS = 600  # 10 minutes
DEFAULT_NOTIFY_BEFORE_END_MINUTES = 0  # Disabled
DEFAULT_PROFILE_MATCH_INTERVAL = (
    300  # Seconds between profile matching attempts (5 minutes)
)
DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO = 0.10  # Allow match after 10% of expected duration
# 1.5 = up to 150% of the profile's average duration. Tuned via the precision
# harness in devtools/dtw_ab_eval.py: widening 1.3->1.5 lifts commit-recall
# 71.6%->73.4% for a negligible false-positive change; 1.3 was rejecting normal
# longer-than-average runs (extended/anti-wrinkle variants).
DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO = 1.5
DEFAULT_MAX_PAST_CYCLES = 200
DEFAULT_MAX_FULL_TRACES_PER_PROFILE = 20
DEFAULT_MAX_FULL_TRACES_UNLABELED = 20
DEFAULT_WATCHDOG_INTERVAL = 30  # Derived: 2 * sampling_interval + 1
DEFAULT_MATCH_PERSISTENCE = 3
DEFAULT_RUNNING_DEAD_ZONE = 3  # Seconds after start to ignore power dips
DEFAULT_END_REPEAT_COUNT = 1  # 1 = current behavior (no repeat required)

# Matching & Termination Stability
DEFAULT_MATCH_REVERT_RATIO = 0.4  # Drop from peak score to revert to detecting
DEFAULT_DEFER_FINISH_CONFIDENCE = 0.55  # Minimum confidence to defer cycle finish

# ML live-match commit gate: P(top-1 is correct) threshold to commit a match
# before the persistence counter is satisfied.  Set high to avoid false-early
# commits; the model's owner-holdout precision is ~0.87 at this score.
ML_MATCH_COMMIT_THRESHOLD = 0.85

# ML quality gate: P(cycle is a problem) threshold above which even a high-
# confidence auto-label is downgraded to a feedback request.  Tuned for a
# specificity of ~0.84 (few false positives) so users are not flooded.
ML_QUALITY_SUSPICIOUS_THRESHOLD = 0.65

# Match ranking history: maximum number of per-cycle snapshots retained on-device.
# Each snapshot stores pre-computed live_match feature scalars (not traces) so
# footprint is small; 500 snapshots cover ~6–12 months of typical usage and are
# enough to build a per-device live_match training dataset.
MATCH_RANKING_HISTORY_MAX = 500

# Runtime overrun anomaly: a *soft, visible* signal (attribute + cycle metadata,
# never a notification) flagged once a running cycle exceeds its matched
# profile's expected duration by this ratio. Distinct from the 300% zombie-kill
# hard limit: this only surfaces "running longer than usual" for the UI. Kept
# below the zombie threshold so it lights up well before any termination.
CYCLE_OVERRUN_ANOMALY_RATIO = 1.5
# Underrun anomaly: a cycle that finishes in less than this fraction of its
# matched profile's median duration is flagged "underrun" (post-cycle only,
# never a live signal — computed in _async_process_cycle_end after the cycle
# ends). Mutually exclusive with overrun: only set when no runtime anomaly fired.
CYCLE_UNDERRUN_ANOMALY_RATIO = 0.55   # below 55% of expected duration = underrun

# Energy anomaly thresholds: a cycle whose energy deviates by more than this
# many standard deviations from the profile's historical average is flagged
# "energy_spike" or "energy_low". Stored separately from the duration anomaly
# so both can coexist. Requires at least 3 labeled cycles for the reference stats.
ENERGY_ANOMALY_Z_THRESHOLD = 2.5   # |z-score| above this = energy anomaly

# Profile warm-up mode: a newly-created profile with fewer than this many
# labeled cycles skips auto-labeling and always requests manual confirmation.
# Prevents the system from confidently mis-labeling cycles before it has seen
# enough examples of the program.
CONF_PROFILE_MIN_WARMUP_CYCLES = 5   # labeled cycles before auto-matching is enabled

# Shape drift detection: compares the average power-curve envelope of the
# earliest third of a profile's cycles against the most recent third.
# A Pearson correlation below SHAPE_DRIFT_THRESHOLD signals drift.
SHAPE_DRIFT_THRESHOLD = 0.85          # envelope correlation below this = shape drifting
SHAPE_DRIFT_MIN_CYCLES = 10           # minimum labeled cycles to check drift
SHAPE_DRIFT_RESAMPLE_N = 50           # points for envelope comparison

# Unlabeled-cycle shape clustering (A3): when suggest_coverage_gaps finds
# duration-bucketed clusters of unmatched cycles, it also checks whether the
# power-curve shapes within each bucket are similar enough to suggest a new
# profile.  Uses a normalized cross-correlation on resampled traces.
CLUSTER_SHAPE_SIMILARITY_THRESHOLD = 0.75   # min correlation for shape-similar cluster
CLUSTER_RESAMPLE_N = 50                      # points for pairwise comparison

# Terminal-drop fast finalize (opt-in; gated on CONF_ENABLE_ML_MODELS via the
# manager provider). A hard cliff-to-~0 at an elapsed offset EARLIER than this
# device has ever legitimately gone quiet (learned from its own completed
# cycles) is an anomaly - almost certainly a real stop (plug pulled / cancelled)
# rather than a soak pause - so the cycle is finalized quickly instead of waiting
# out the full soak-bridging min_off_gap (up to 8 min for washers, 1 h for
# dishwashers). Asymmetric like the ML end-guard, but the opposite direction: it
# can only SHORTEN the end wait, and only for anomalously-early drops.
TERMINAL_DROP_OFF_DELAY_SECONDS = 90    # shortened below-threshold wait once terminal
TERMINAL_DROP_MIN_CLEAN_CYCLES = 3      # completed cycles needed before we trust the baseline
TERMINAL_DROP_MIN_QUIET_SPAN_S = 60     # sustained sub-threshold span that counts as a legit quiet period
TERMINAL_DROP_EARLINESS_RATIO = 0.8     # fire only if drop starts < ratio * earliest-ever-quiet offset
TERMINAL_DROP_MIN_PEAK_RATIO = 5.0      # cycle must have been clearly ON (peak >= ratio * stop_threshold)
# Familiarity/novelty gate: an early hard drop is only trusted as terminal when
# the cycle's power level is one this device has produced before.  A very early
# drop (below the matcher's duration gate) can't be confirmed by match
# confidence, so power level is the signal available that early: a cycle peaking
# outside the device's historical peak range (widened by this tolerance) is
# treated as potentially a NEW program and DEFERRED to the proven slow path
# rather than assumed to be a stop.
TERMINAL_DROP_PEAK_FAMILIAR_TOL = 0.4

# Cycle interruption detection defaults (internal)
DEFAULT_ABRUPT_DROP_WATTS = 500.0  # Power cliff detection threshold (W)
DEFAULT_ABRUPT_DROP_RATIO = 0.6  # 60% drop considered abrupt
DEFAULT_ABRUPT_HIGH_LOAD_FACTOR = 5.0  # High load factor threshold
DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD = 3  # Ghost cycles before threshold adjustment

# Anti-wrinkle defaults (advanced; disabled by default)
DEFAULT_ANTI_WRINKLE_ENABLED = False
DEFAULT_ANTI_WRINKLE_MAX_POWER = 400.0  # W
DEFAULT_ANTI_WRINKLE_MAX_DURATION = 60.0  # s
DEFAULT_ANTI_WRINKLE_EXIT_POWER = 0.8  # W

# Delayed-start detection defaults (disabled by default).
#
# The detector watches for sustained power between stop_threshold_w and
# start_threshold_w (the "standby band"): a machine sitting in that band
# for at least DEFAULT_DELAY_CONFIRM_SECONDS is in delayed-start mode, not
# off and not running. Short menu-navigation peaks above the band are
# ignored because they don't sustain long enough to satisfy the normal
# start-duration gate.
DEFAULT_DELAY_START_DETECT_ENABLED = False
DEFAULT_DELAY_CONFIRM_SECONDS = 60.0  # s - sustained standby before DELAY_WAIT engages
DEFAULT_DELAY_TIMEOUT_HOURS = 8.0  # h - give up waiting after this long

# Pump Monitor settings (pump device type only)
CONF_PUMP_STUCK_DURATION = "pump_stuck_duration"  # Seconds before a running pump is flagged as stuck
DEFAULT_PUMP_STUCK_DURATION = 1800  # 30 min - typical sump pump runs <60 s; 30 min implies motor is jammed
EVENT_PUMP_STUCK = "ha_washdata_pump_stuck"  # Fired when stuck-pump threshold is exceeded

# Profile Matching Thresholds
CONF_PROFILE_MATCH_THRESHOLD = "profile_match_threshold"
CONF_PROFILE_UNMATCH_THRESHOLD = "profile_unmatch_threshold"

DEFAULT_PROFILE_MATCH_THRESHOLD = 0.4
DEFAULT_PROFILE_UNMATCH_THRESHOLD = 0.35

CONF_DTW_BANDWIDTH = "dtw_bandwidth"
DEFAULT_DTW_BANDWIDTH = 0.20  # 20% Sakoe-Chiba constraint

# ─── Matching pipeline scoring constants (analysis.py) ────────────────────────
# Previously scattered as magic numbers in analysis.py / profile_store.py.
# Centralised here so the scoring formula is auditable in one place and the
# ambiguity threshold cannot drift between its two call sites.
#
# Core similarity (Stage 2): score = CORR_WEIGHT*max(0,corr) + MAE_WEIGHT*mae_score
# where mae_score = MAE_SCALE / (MAE_SCALE + scaled_mae). See MATCH_MAE_SCALE_MODE
# in analysis.py for how scaled_mae is normalised across device power scales.
# 0.45 tuned via devtools/dtw_ab_eval.py: weighting MAE more (0.6->0.45 corr)
# lifted leave-one-out top-1 74%->79.5% AND the recall/FP net 10.7%->13.7% (FP
# flat), i.e. a genuine discrimination gain, not confidence inflation. 0.35-0.45
# is a broad plateau; 0.45 is best on top-1/MRR.
MATCH_CORR_WEIGHT = 0.45           # MAE weight is (1 - MATCH_CORR_WEIGHT), computed inline
MATCH_MAE_SCALE = 100.0            # half-saturation point of the MAE score curve
# Scale-invariant MAE (5c): the raw MAE is expressed relative to the current
# cycle's peak power before scoring, so the same *proportional* error yields the
# same confidence on a 200 W dishwasher and a 2000 W dryer. Calibrated to be
# behaviour-neutral at MATCH_MAE_REF_PEAK: at that peak scaled_mae == raw mae, so
# existing thresholds keep their meaning. The current cycle's peak is common to
# every candidate in a match, so this does not change candidate ranking.
MATCH_MAE_REF_PEAK = 1000.0        # peak (W) at which scoring matches the legacy formula
MATCH_MAE_PEAK_FLOOR = 50.0        # floor so tiny/idle traces don't explode the ratio
MATCH_KEEP_MIN_SCORE = 0.1         # candidates scoring below this are discarded
# DTW refinement (Stage 3): blended = DTW_BLEND*core + (1-DTW_BLEND)*dtw_score,
# dtw_score = DIST_SCALE / (DIST_SCALE + scaled_dtw_distance).
MATCH_DTW_BLEND = 0.5
MATCH_DTW_DIST_SCALE = 50.0
MATCH_DTW_REFINE_TOP_N = 5         # DTW is applied to this many top candidates
                                   # (5 tuned via dtw_ab_eval: rescues correct
                                   #  profiles Stage-2 ranked 4th-5th; +1.8pp)
# Stage-3 DTW modes (config key "dtw_mode"):
#   "legacy" - original: raw sequences, distance / len(current), fixed 50 W scale.
#   "scaled" - both sequences resampled to MATCH_DTW_RESAMPLE_N and the distance
#              expressed relative to the current peak (behaviour-neutral at
#              MATCH_MAE_REF_PEAK), matching the Stage-2 MAE treatment. Default.
#   "ddtw"   - like "scaled" but warps on the first derivative (slope) of the
#              curves, so alignment is driven by shape rather than absolute level.
#   "ensemble" - blend of "scaled" and "ddtw": ENSEMBLE_W*L1 + (1-W)*DDTW.
# Defaults tuned via devtools/dtw_ab_eval.py on cycle_data/ (leave-one-out top-1):
# off 62.4%, legacy 66.4%, scaled 69.9%, ddtw 69.0%, ensemble(w=0.7,dd=30) 70.7%.
DEFAULT_DTW_MODE = "ensemble"
MATCH_DTW_RESAMPLE_N = 200         # common grid length for "scaled"/"ddtw" DTW
MATCH_DDTW_DIST_SCALE = 30.0       # half-saturation for derivative-DTW distance
MATCH_DTW_ENSEMBLE_W = 0.7         # weight on L1 vs DDTW in "ensemble" mode
# Ambiguity: top1-top2 score gap below this flags the match as ambiguous.
MATCH_AMBIGUITY_MARGIN = 0.05
# Smart Termination landscape guard: when a non-winning candidate is at least this
# much longer than the matched profile AND has a decent shape score (before Stage-4
# duration penalty), the current trace may be a *prefix* of that longer program
# rather than a completed short one. Smart Termination is blocked; the power-based
# fallback timeout decides instead. Ratio chosen so that programmes within ~50% of
# each other (e.g. Quick 46 min vs Eco 60 min, ratio 1.30) do not trigger the guard
# but genuine prefix pairs like Quick 46 vs Normal 88 min (ratio 1.91) always do.
SMART_TERM_LANDSCAPE_RATIO = 1.5       # candidate must be >= 1.5× the matched duration
SMART_TERM_LANDSCAPE_MIN_SHAPE = 0.40  # minimum shape score (pre-Stage-4) to qualify

# Number of points in the compact reference-profile curve exposed on the
# `_program` sensor (`profile_store.reference_curve`). Chosen so the resulting
# `[[offset_s, watts], ...]` attribute stays comfortably under ~1 KB regardless
# of cycle length; the raw envelope can be hundreds to thousands of points.
REFERENCE_PROFILE_CURVE_POINTS = 50
# Duration + energy agreement blended into the final score. Shape correlation
# alone cannot separate profiles that differ mainly in duration/energy (a real
# weakness on multi-program washing machines), so the final score is
# (1 - dur_w - en_w)*shape + dur_w*dur_agreement + en_w*energy_agreement, where
# agreement = 1/(1 + |ln(observed/expected)| / scale) is 1.0 on a perfect match.
# Weights 0.22 and scales tuned via devtools/dtw_ab_eval.py (weight x scale grid):
# a SHARPER agreement scale (halved) plus a moderately higher weight separates
# near-duplicate profiles on the same device rather than inflating confidence.
# This lifted the recall/FP net 13.7%->17.4% with the false-positive rate
# actually DROPPING (62.7%->59.9%). Raising weight alone at the old loose scale
# inflated both recall and FP (net-negative), so both knobs move together.
MATCH_DURATION_WEIGHT = 0.22
MATCH_ENERGY_WEIGHT = 0.22
MATCH_DURATION_SCALE = 0.175       # ~ln ratio at which duration agreement halves
MATCH_ENERGY_SCALE = 0.25          # ~ln ratio at which energy agreement halves

CONF_SUPPRESS_FEEDBACK_NOTIFICATIONS = "suppress_feedback_notifications"
DEFAULT_SUPPRESS_FEEDBACK_NOTIFICATIONS = True  # Feedback stays in-panel by default; no persistent notifications

# States
STATE_OFF = "off"
STATE_DELAY_WAIT = "delay_wait"
STATE_IDLE = "idle"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_USER_PAUSED = "user_paused"
STATE_ENDING = "ending"
STATE_FINISHED = "finished"
STATE_ANTI_WRINKLE = "anti_wrinkle"
STATE_INTERRUPTED = "interrupted"
STATE_FORCE_STOPPED = "force_stopped"
STATE_RINSE = "rinse"
STATE_UNKNOWN = "unknown"
STATE_CLEAN = "clean"  # Cycle ended but door not yet opened (laundry still inside)

# Authoritative state -> display color map. Single source of truth for the
# full-screen panel (and any other frontend), surfaced over the WebSocket
# get_constants command so colors are defined in exactly one place. Values are
# CSS colors using Home Assistant theme variables with a hex fallback, so they
# adapt to the active theme. The "recording" key covers the manual recorder state.
STATE_COLORS = {
    STATE_OFF: "var(--state-inactive-color, #9e9e9e)",
    STATE_IDLE: "var(--state-inactive-color, #9e9e9e)",
    STATE_DELAY_WAIT: "var(--secondary-text-color, #757575)",
    STATE_STARTING: "var(--warning-color, #ff9800)",
    STATE_RUNNING: "var(--success-color, #4caf50)",
    STATE_PAUSED: "var(--warning-color, #ff9800)",
    STATE_USER_PAUSED: "var(--warning-color, #ff9800)",
    STATE_ENDING: "var(--info-color, #2196f3)",
    STATE_FINISHED: "var(--success-color, #4caf50)",
    STATE_ANTI_WRINKLE: "var(--info-color, #2196f3)",
    STATE_INTERRUPTED: "var(--error-color, #f44336)",
    STATE_FORCE_STOPPED: "var(--error-color, #f44336)",
    STATE_RINSE: "var(--info-color, #2196f3)",
    STATE_CLEAN: "var(--teal-color, #009688)",
    STATE_UNKNOWN: "var(--disabled-color, #bdbdbd)",
    "recording": "var(--error-color, #f44336)",
}

# Device Types
DEVICE_TYPE_WASHING_MACHINE = "washing_machine"
DEVICE_TYPE_DRYER = "dryer"
DEVICE_TYPE_WASHER_DRYER = "washer_dryer"
DEVICE_TYPE_DISHWASHER = "dishwasher"
DEVICE_TYPE_AIR_FRYER = "air_fryer"
DEVICE_TYPE_BREAD_MAKER = "bread_maker"
DEVICE_TYPE_PUMP = "pump"
# Full-featured generic type for predictable appliances that don't fit any of the
# named categories. Participates in profile matching/learning like any other
# device type. Ships with neutral/safe defaults; the user tunes from there.
DEVICE_TYPE_GENERIC = "generic"
# Threshold-only bucket. No profile matching. Ships intentionally generic
# defaults; the user must configure thresholds and timeouts themselves.
# Config entries whose stored device_type is no longer supported are migrated
# to this bucket on load (see __init__.py), preserving their tuned options.
DEVICE_TYPE_OTHER = "other"

DEVICE_TYPES = {
    DEVICE_TYPE_WASHING_MACHINE: "Washing Machine",
    DEVICE_TYPE_DRYER: "Dryer",
    DEVICE_TYPE_WASHER_DRYER: "Washer-Dryer Combo",
    DEVICE_TYPE_DISHWASHER: "Dishwasher",
    DEVICE_TYPE_AIR_FRYER: "Air Fryer",
    DEVICE_TYPE_BREAD_MAKER: "Bread Maker",
    DEVICE_TYPE_PUMP: "Pump / Sump Pump",
    DEVICE_TYPE_GENERIC: "Other (Advanced)",
    DEVICE_TYPE_OTHER: "Threshold Device",
}

# Device Type Defaults
# Device Type Defaults (Maps)

DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE = {
    DEVICE_TYPE_DISHWASHER: 14400,  # 4 hours (Drying can be long)
    DEVICE_TYPE_BREAD_MAKER: 7200,  # 2 hours (Proving/Rising is very low-power for extended periods)
    DEVICE_TYPE_PUMP: DEFAULT_PUMP_STUCK_DURATION + 60,  # Must exceed stuck-alarm threshold so the alarm fires before the watchdog
}

DEFAULT_MAX_DEFERRAL_SECONDS = 14400  # 4 hours max safe deferral

# Issue #43: dishwasher end-of-cycle pump-out handling.
#
# A dishwasher's wash→drying drain wind-down produces brief power spikes mid
# ENDING that, prior to the issue #43 fix, would set _end_spike_seen=True and
# pre-arm Smart Termination - so the cycle closed at 99% of expected, BEFORE
# the real end-of-cycle pump-out at ~99.5% of expected.  The pump-out then
# registered as a brand-new cycle.
#
# Two coordinated thresholds gate the fix.  They MUST agree: the wait window
# in _should_defer_finish (DISHWASHER_END_SPIKE_WAIT_SECONDS) is the upper
# bound for keeping the cycle open without an end spike, and Smart
# Termination's own wait branch in STATE_ENDING uses the SAME constant so the
# two paths release the cycle at the same moment.
#
# A spike at < DISHWASHER_END_SPIKE_MIN_PROGRESS of expected duration is
# ignored for end-spike tracking (the cycle still stays in ENDING via the
# existing long_ending_tail path - this only governs the smart-termination
# pre-arming).
DISHWASHER_END_SPIKE_MIN_PROGRESS = 0.85
# Widened from 300s to 1800s after issue #43 follow-up:
# real-world user reports showed Smart Termination misfiring ~4 min before the
# end-of-cycle pump-out, and the original 5-min escape hatch wasn't generous
# enough to cover that gap before the next reading arrived.  30 min is plenty
# to capture even the latest pump-outs while still guaranteeing the cycle
# closes eventually for dishwashers that have no pump-out at all.
DISHWASHER_END_SPIKE_WAIT_SECONDS = 1800.0
# Minimum reasonable dishwasher cycle duration (seconds).  Even the shortest
# quick programmes take at least 30 min; defer _should_defer_finish for any
# dishwasher whose cycle has not yet crossed this floor, regardless of whether
# a profile match is available yet.
DISHWASHER_MIN_CYCLE_DURATION_S = 1800.0
# Once a dishwasher is in ENDING and power has been sustained-quiet for this
# long, the active cycle is over - only the passive drain/dry tail remains.
# Live re-matching is frozen past this point: continuing to re-match on the
# ever-growing idle tail inflates the observed duration and drifts the Stage-4
# duration-agreement score toward LONGER near-duplicate profiles, which would 
# flip the stored label and stall smart-termination on the ambiguity gate.  
# The active-phase match is complete
# by now, so freezing it preserves the correct program identity.  A real
# resume (mid-cycle soak) sends a high reading that leaves ENDING and re-arms
# matching, so this is self-correcting.
DISHWASHER_MATCH_FREEZE_QUIET_SECONDS = 300.0
# Release the end-of-cycle pump-out wait early once a dishwasher has BOTH reached
# its expected duration AND been sustained-quiet this long afterwards.  This lets a
# cycle that ran slightly shorter than the profile's (drifted-up) average - and whose
# terminal pump-out landed before the drop into ENDING, so no in-ENDING end-spike ever
# armed - finalise near its expected end instead of hanging the full
# DISHWASHER_END_SPIKE_WAIT_SECONDS (30 min) past expected.  Gated on reaching the
# expected duration so a long passive-drying phase that still precedes a genuinely-late
# pump-out (quiet from ~50%-99% of expected) keeps waiting and its real pump-out is
# caught by the end-spike arm first.  Smaller than the 30-min window but large enough
# to confirm a terminal tail rather than an inter-phase gap.
DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS = 600.0

# Confirmation window a dishwasher must spend in ENDING before Smart Termination
# fires.  This is deliberately a FIXED constant and NOT derived from off_delay:
# off_delay must be large (up to ~30 min) to bridge a dishwasher's long passive
# drying "pause" so a single cycle is not split by the fallback timeout, but that
# large value must NOT delay Smart Termination - which ends the cycle near the
# matched profile's expected duration so the finish notification is timely.  A
# previous formula (max(300, off_delay*0.25)) coupled the two: a suggested
# off_delay of 1800-1999 s inflated this window to 450-500 s, and on the sparsely
# sampled near-zero drying tail the eligibility instant could fall in a gap
# between samples, slipping the cycle's end by 20+ min or leaving it to only end
# via the fallback timeout (which snaps the trace back and drops the drying tail)
# or a manual stop.  300 s (the old floor, proven on a hand-tuned production
# dishwasher running off_delay=180) settles transient dips without starving the
# end.  Smart Termination is independently gated on duration >= expected*ratio, so
# a shorter window can never fire it mid-cycle.
DISHWASHER_SMART_TERMINATION_DEBOUNCE_SECONDS = 300.0

DEFAULT_OFF_DELAY_BY_DEVICE = {
    DEVICE_TYPE_DISHWASHER: 1800,  # 30 min (Drying)
    DEVICE_TYPE_BREAD_MAKER: 300,  # 5 min (Keep-warm phase after baking)
    DEVICE_TYPE_PUMP: 20,  # 20 s (Pumps cut off sharply; no warm-down phase)
}

# Device-specific progress smoothing thresholds (percentage points)
# These control how much backward progress is allowed before heavy damping kicks in
DEVICE_SMOOTHING_THRESHOLDS = {
    DEVICE_TYPE_WASHING_MACHINE: 5.0,  # Can have repeating phases (rinse cycles)
    DEVICE_TYPE_DRYER: 3.0,  # More linear, less phase repetition
    DEVICE_TYPE_WASHER_DRYER: 5.0,  # Combined washer+dryer, use washer defaults
    DEVICE_TYPE_DISHWASHER: 5.0,  # Similar to washing machine with distinct phases
    DEVICE_TYPE_AIR_FRYER: 2.0,  # Constant load with sudden drop
    DEVICE_TYPE_BREAD_MAKER: 5.0,  # Large power swings between kneading, proving, baking
    DEVICE_TYPE_PUMP: 2.0,  # Binary on/off spikes; minimal smoothing needed
    DEVICE_TYPE_GENERIC: 3.0,  # Neutral middle ground for unknown appliance types
}

# Device specific completion thresholds (min run time to be considered a valid "completed" cycle)
DEVICE_COMPLETION_THRESHOLDS = {
    DEVICE_TYPE_WASHING_MACHINE: 600,  # 10 min
    DEVICE_TYPE_DRYER: 600,  # 10 min
    DEVICE_TYPE_WASHER_DRYER: 600,  # 10 min (same as washer)
    DEVICE_TYPE_DISHWASHER: 900,  # 15 min
    DEVICE_TYPE_AIR_FRYER: 300,  # 5 min minimum
    DEVICE_TYPE_BREAD_MAKER: 1800,  # 30 min (even express bread takes 30+ min)
    DEVICE_TYPE_PUMP: 5,  # 5 s - pump cycles can be under 30 seconds
}

# Default min_off_gap by device type (seconds)
# If gap between cycles is larger than this, force new cycle.
# If smaller, and we deemed previous as 'ended' but technically could be same,
# we might want to handle that (though strict state machine usually suffices if tuned well).
# Default min_off_gap by device type (seconds)
# Default min_off_gap by device type (seconds)
DEFAULT_MIN_OFF_GAP_BY_DEVICE = {
    DEVICE_TYPE_WASHING_MACHINE: 480,  # 8 min (Soak handling)
    DEVICE_TYPE_DRYER: 300,  # 5 min (Cool down gaps?)
    DEVICE_TYPE_WASHER_DRYER: 600,  # 10 min (longer for combined cycles)
    DEVICE_TYPE_DISHWASHER: 3600,  # 1 hour (Drying pauses)
    DEVICE_TYPE_AIR_FRYER: 120,  # 2 min (Shaking food)
    DEVICE_TYPE_BREAD_MAKER: 600,  # 10 min (Resting between knead/prove keeps same cycle together)
    DEVICE_TYPE_PUMP: 60,  # 1 min (Pumps can cycle every 3-5 min in heavy rain)
}
DEFAULT_MIN_OFF_GAP = 60  # Scalar fallback

# Default start energy threshold by device type (Wh)
# Filter noise spikes (1000W * 0.01s = 0.002Wh).
# Must be significant enough to imply mechanical work.
DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE = {
    DEVICE_TYPE_WASHING_MACHINE: 0.2,  # ~50W for 15s or 200W for 3s
    DEVICE_TYPE_DRYER: 0.5,  # Heater kicks in hard
    DEVICE_TYPE_WASHER_DRYER: 0.3,  # Mix of washer and dryer
    DEVICE_TYPE_DISHWASHER: 0.2,  # Pump/Heater
    DEVICE_TYPE_AIR_FRYER: 0.2,  # Heater kicks in
    DEVICE_TYPE_BREAD_MAKER: 0.2,  # Kneading motor starts (~200W for a few seconds)
    DEVICE_TYPE_PUMP: 0.003,  # ~100W motor for ~0.1 s is enough to confirm a pump cycle
}
# Default sampling interval by device type
DEFAULT_SAMPLING_INTERVAL_BY_DEVICE = {
    # 2s captures the rapid 0<->150W motor/heater oscillation in wet appliances;
    # the 30s global default discards those spikes and undersamples the cycle.
    DEVICE_TYPE_WASHING_MACHINE: 2.0,
    DEVICE_TYPE_WASHER_DRYER: 2.0,
    DEVICE_TYPE_DISHWASHER: 2.0,
    DEVICE_TYPE_PUMP: 10.0,  # 10s - pump cycles can be <30 s; 30s default would miss them
}

# Default profile match min duration ratio by device type
DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE = {
    DEVICE_TYPE_DISHWASHER: 0.10,
}

# Profile groups (Stage 5): the matcher only collapses a group into one
# aggregate candidate when its members' minimum pairwise shape similarity is at
# least this. Similarity is DTW/Sakoe-Chiba on peak-normalised envelopes, so it
# tolerates the duration (longer heating/draining) and amplitude (temp/spin)
# variation between real members. Looser groups stay individual (a blurry generic
# aggregate could out-match unrelated profiles) and are flagged in the UI.
# Calibrated on real profiles: genuine temp/spin variants score ~0.86-0.95,
# distinct programs <~0.6; 0.80 leaves margin below the 0.85 suggestion bar.
GROUP_MIN_COHESION = 0.80

# Storage
# v6: backfill ml_review.golden=True for manually-recorded cycles (recorded ==
# golden reference; a single flag, no duplicate "recorded" field).
# v7: re-run that backfill (broadened to the meta.original_samples marker) so
# installs already at v6 that carry unflagged recorded cycles are caught too —
# the v6 step only ran for installs upgrading from below v6.
# v8: re-run again after _is_recorded_cycle gained the structural fallback
# (completed + no max_power/termination_reason) so OLD recordings that carry
# only meta:None — which the marker-only v6/v7 backfill missed — are tagged.
# v9: pre-initialize additive top-level keys (lifetime_energy_wh,
# settings_changelog, maintenance_log) so they are present from first load
# rather than only appearing lazily on first use.
STORAGE_VERSION = 10
STORAGE_KEY = "ha_washdata"

# Notification events
EVENT_CYCLE_STARTED = "ha_washdata_cycle_started"
EVENT_CYCLE_ENDED = "ha_washdata_cycle_ended"

# Signals
SIGNAL_WASHER_UPDATE = "ha_washdata_update_{}"

# Learning & Feedback

SERVICE_SUBMIT_FEEDBACK = (
    "ha_washdata.submit_cycle_feedback"  # Service to submit feedback
)

# ─── Feature flags (staged rollout) ───────────────────────────────────────────
# These gate preproduction / ML features so they can be shipped dark and unlocked
# in stages. When a flag is False the corresponding UI *and* logic stay hidden:
# no panel sections render and no background work runs.
#
#   SHOW_ML_LAB           ML Lab comparison tab in the WashData panel.
#   ENABLE_ML_SUGGESTIONS ML-model-driven setting suggestions (Stage 3), shown
#                         side-by-side with the classic statistical suggestions.
#   ENABLE_ML_TRAINING    On-device model training loop (Stage 4): scheduled
#                         retraining on the user's own labeled cycles.
#
# Stage 1 (new statistical suggestions) and Stage 2 (fixed classic algorithms)
# are always on - they only improve the existing suggestion engine and add no
# new surfaces, so they need no flag.
SHOW_ML_LAB = True
ENABLE_ML_SUGGESTIONS = True
ENABLE_ML_TRAINING = True

# ─── Community store (online features) ────────────────────────────────────────
# Opt-in browsing/importing/sharing of reference cycles via the WashData Store.
# When the option is off the Store tab and all network calls stay inert.
CONF_ENABLE_ONLINE_FEATURES = "enable_online_features"   # master gate, default False
CONF_STORE_BRAND = "store_brand"                          # declared appliance brand
CONF_STORE_MODEL = "store_model"                          # declared appliance model
DEFAULT_ENABLE_ONLINE_FEATURES = False

# Device-level settings that may be shared/adopted with a device bundle (Stage 3).
# These are recognition/matching thresholds intrinsic to the appliance MODEL (the
# same for everyone with that machine), never environment/plug/identity settings:
# no entity ids, notify services, energy price, sampling cadence, smoothing,
# housekeeping timers, plug-robustness (end_repeat_count) or device-behaviour
# toggles (anti-wrinkle, delay-start). Kept as one editable allow-list so share and
# adopt agree on exactly what travels. All values are plain numbers -> nothing here
# can leak PII or a user's HA topology.
SHAREABLE_SETTING_KEYS: tuple[str, ...] = (
    # Detection / recognition
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_ENERGY_THRESHOLD,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_RUNNING_DEAD_ZONE,
    CONF_MIN_OFF_GAP,
    CONF_END_ENERGY_THRESHOLD,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_POWER_OFF_DELAY,
    # Matching
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_PROFILE_UNMATCH_THRESHOLD,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_DURATION_TOLERANCE,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
)

# Public Firebase web config for the community store (NOT secret - identifies the
# project; access is enforced by the store's Firestore rules).
STORE_PROJECT_ID = "washdata-store"
STORE_API_KEY = "AIzaSyDzq0MoWdU_21CSohZUhIIV7ZwfWppjcAk"
STORE_WEB_ORIGIN = "https://3dg1luk43.github.io/washdata-store"

# Reference-cycle trace format versions this integration can import.
SUPPORTED_CYCLE_SCHEMA_VERSIONS = {1}

# Obfuscated provenance codes stamped on an uploaded cycle (see store.derive_qc).
QC_RECORDING = 1   # pure recorder capture
QC_EDITED = 2      # trimmed/edited from a detected cycle
QC_MANUAL = 3      # a plain detected cycle flagged golden by hand

# ─── On-device ML training (Stage 4) ──────────────────────────────────────────
# Config keys for the scheduled, opt-in retraining loop. All gated behind
# ENABLE_ML_TRAINING; nothing runs and no options render when that flag is False.
CONF_ML_TRAINING_ENABLED = "ml_training_enabled"        # per-device opt-in
CONF_ML_TRAINING_HOUR = "ml_training_hour"              # local hour (0-23) to train
CONF_ML_TRAINING_MIN_CYCLES = "ml_training_min_cycles"  # min labelled clean cycles before training
CONF_ML_TRAINING_INTERVAL_DAYS = "ml_training_interval_days"  # min days between retrains

DEFAULT_ML_TRAINING_ENABLED = False
DEFAULT_ML_TRAINING_HOUR = 2          # 02:00 local - quiet hour
DEFAULT_ML_TRAINING_MIN_CYCLES = 30   # need a meaningful corpus first
DEFAULT_ML_TRAINING_INTERVAL_DAYS = 7 # retrain at most weekly

# A newly trained model is only promoted over the shipped baseline when its
# held-out AUC is at least (baseline AUC - this margin). Small negative slack is
# allowed so personalisation can win even at a tiny AUC cost.
ML_TRAINING_AUC_MARGIN = 0.02
# Separate tolerance for the calibration gate: a retrained classifier must not
# degrade balanced accuracy AT the live operating cutoff by more than this. Kept
# distinct from ML_TRAINING_AUC_MARGIN because it bounds a different metric (decision
# quality at a fixed threshold, not overall rank quality); same 0.02 default today.
ML_TRAINING_BACC_MARGIN = 0.02
ML_TRAINING_MIN_POSITIVES = 20  # need at least this many positive examples to trust a fit

# Per-capability held-out-score history kept across training runs, so the panel
# can show whether a model's fit is improving, steady, or declining over time
# (drift). Compact (one number per capability per run); this caps how many runs
# are retained.
ML_TRAINING_HISTORY_MAX = 30

# Remaining-time regressor (standardized_linear). Unlike the classifier heads it
# has no shipped baseline; it is only promoted when its held-out mean-absolute
# error on the completion-fraction target beats the naive elapsed/expected
# estimate by at least this relative margin (5% lower MAE). Trained from prefixes
# of the device's own clean cycles.
ML_TRAINING_REGRESSION_MARGIN = 0.05
ML_TRAINING_MIN_REGRESSION_ROWS = 30  # synthesized prefix rows needed to fit
# How strongly a promoted remaining-time regressor influences the live progress
# estimate. The ML completion-fraction is blended with the phase-aware estimate
# at this weight before the existing EMA smoothing/monotonicity guards run, so a
# bad model can never wholly override the proven phase estimator.
ML_PROGRESS_BLEND_WEIGHT = 0.5

# Service + event names for the training loop.
SERVICE_TRIGGER_ML_TRAINING = "trigger_ml_training"
EVENT_ML_TRAINING_COMPLETE = "ha_washdata_ml_training_complete"

# ─── Suggestion quality gates ──────────────────────────────────────────────────
# A suggestion is only stored / surfaced when it clears both thresholds:
#   (a) relative delta >= MIN_SUGGESTION_REL_DELTA  OR
#       absolute delta >= per-key absolute minimum (see _suggestion_min_abs_delta)
# Suggestions that are below BOTH thresholds are deleted so they don't clutter
# the panel with noise (e.g. 0.67 → 0.68).
MIN_SUGGESTION_REL_DELTA = 0.08  # 8% minimum relative change

# After the user applies suggestions, suppress new suggestions for this many
# completed cycles. Prevents the engine from immediately re-suggesting
# slightly-different values based on a single new cycle.
MIN_SUGGESTION_COOLDOWN_CYCLES = 3

# ─── Appliance health & predictive maintenance (Group E) ───────────────────────
# Per-device maintenance-reminder thresholds: a dict {event_type: cycle_threshold}
# persisted via ws_set_options. When the number of completed cycles since the most
# recent maintenance event of a given type reaches its threshold, the event type is
# surfaced (sensor attribute + panel banner). A threshold of 0 (or an absent key)
# disables reminders for that event type.
CONF_MAINTENANCE_REMINDER_CYCLES = "maintenance_reminder_cycles"
DEFAULT_MAINTENANCE_REMINDER_CYCLES = {
    "descale": 30,
    "filter_clean": 50,
    "drum_clean": 100,
}
# Recognised maintenance event types. bearing_service / other default off (absent
# from the default reminder dict) and are opt-in.
MAINTENANCE_EVENT_TYPES = (
    "descale",
    "filter_clean",
    "drum_clean",
    "bearing_service",
    "other",
)
# A logged maintenance event of a matching type within this many days suppresses
# the "needs maintenance" nag advisory (duration-trend / shape-drift).
MAINTENANCE_RECENT_SUPPRESS_DAYS = 30
