# WashData `const.py` — Complete Constant Catalog

Source: `/root/ha_washdata/custom_components/ha_washdata/const.py` (1068 lines).
Integration version (`manifest.json`): **0.5.4**.

**Counts (approximate, from the 2026-07-18 pass):** ~298 module-level assignments + 5 `TerminationReason` enum members = ~**303 named constants/keys cataloged** (of which `CONF_*` ≈ 102, `DEFAULT_*` ≈ 83). These totals predate the 0.5.4 additions and are no longer exact.

> Content-corrected 2026-08-17 for 0.5.4 (header metadata; `running_dead_zone` retired). The constant **values** below were re-verified against `const.py` and are accurate; only stale metadata and the `running_dead_zone` entries were changed.

**Important cross-file note:** `CONF_ENABLE_ML_MODELS` (`"enable_ml_models"`) is referenced by `const.py` comments but is **defined in `ml/engine.py:46`, not in `const.py`.** The panel toggle "Apply smart models during a cycle" writes this key; `ml_models_enabled(options)` (`ml/engine.py:56`) reads it. It is documented in the Feature-flag section below for completeness but does not add to the const.py count.

---

## Block index (section headers found in `const.py`, in file order)

| # | Header / logical block | Anchor |
|---|---|---|
| 1 | Module docstring + `DOMAIN` | const.py:17-21 |
| 2 | `class TerminationReason(StrEnum)` + `ANTI_WRINKLE_ELIGIBLE_REASONS` | const.py:24-39 |
| 3 | `# Configuration keys` (detection/matching/housekeeping CONF_*) | const.py:41-97 |
| 4 | Phase-matching opt-in + phase-consistency advisory | const.py:99-117 |
| 5 | `CONF_SAVE_DEBUG_TRACES` + `# Cycle interruption detection settings` + anti-wrinkle/delay-start CONF_* | const.py:118-137 |
| 6 | `NOTIFY_EVENT_*` event-type ids | const.py:140-144 |
| 7 | Notification content CONF_* (title/icon/messages/channels/price) + `# Door sensor & pause` | const.py:146-170 |
| 8 | `# Quiet hours` + `# Milestone` + `# Optional link to an existing HA device` | const.py:172-189 |
| 9 | `DEFAULT_NOTIFY_*` (message templates + notify defaults) | const.py:191-215 |
| 10 | `# Defaults` (core detection defaults) | const.py:217-263 |
| 11 | `# Power-based Off detection` | const.py:234-241 |
| 12 | `# Matching & Termination Stability` | const.py:265-267 |
| 13 | ML live-match / quality gates + match-ranking-history cap | const.py:269-283 |
| 14 | Runtime overrun / underrun / energy anomaly ratios | const.py:285-301 |
| 15 | Profile warm-up mode | const.py:303-307 |
| 16 | Shape drift detection | const.py:309-314 |
| 17 | Unlabeled-cycle shape clustering (A3) | const.py:316-321 |
| 18 | `# Terminal-drop fast finalize` (`TERMINAL_DROP_*`) | const.py:323-343 |
| 19 | `# Cycle interruption detection defaults` | const.py:345-349 |
| 20 | `# Anti-wrinkle defaults` | const.py:351-355 |
| 21 | `# Delayed-start detection defaults` | const.py:357-367 |
| 22 | `# Pump Monitor settings` | const.py:369-372 |
| 23 | `# Profile Matching Thresholds` + DTW bandwidth | const.py:374-382 |
| 24 | `# Matching pipeline scoring constants (analysis.py)` (`MATCH_*`, DTW, ambiguity, smart-term guard, duration/energy agreement) | const.py:384-459 |
| 25 | `# States` (`STATE_*`) | const.py:461-476 |
| 26 | `STATE_COLORS` map | const.py:478-500 |
| 27 | `# Device Types` + `DEVICE_TYPES` map | const.py:502-530 |
| 28 | `# Device Type Defaults (Maps)` (incl. dishwasher end-spike constants, `DEFAULT_MAX_DEFERRAL_SECONDS`, per-device dicts, `GROUP_MIN_COHESION`) | const.py:532-696 |
| 29 | `# Storage` (`STORAGE_VERSION`/`STORAGE_KEY`) | const.py:698-714 |
| 30 | `# Notification events` (HA bus events) | const.py:716-718 |
| 31 | `# Signals` | const.py:720-721 |
| 32 | `# Learning & Feedback` (service const) | const.py:723-727 |
| 33 | `# Feature flags (staged rollout)` | const.py:729-745 |
| 34 | `# Community store (online features)` | const.py:747-802 |
| 35 | `# On-device ML training (Stage 4)` (`CONF_ML_TRAINING_*`, `ML_TRAINING_*`, `ML_PROGRESS_BLEND_WEIGHT`) | const.py:803-848 |
| 36 | `# Suggestion quality gates` | const.py:850-861 |
| 37 | `# Appliance health & predictive maintenance (Group E)` | const.py:863-886 |

---

## 1. Domain & termination reasons (const.py:17-39)

| Name | Value | Description / consumers |
|---|---|---|
| `DOMAIN` | `"ha_washdata"` | Integration domain; used everywhere (entities, storage, events, services). |
| `TerminationReason.TIMEOUT` | `"timeout"` | Low-power off_delay elapsed = normal completion. |
| `TerminationReason.SMART` | `"smart"` | Smart-termination heuristic finished the cycle. |
| `TerminationReason.FORCE_STOPPED` | `"force_stopped"` | Watchdog / no-update force end. |
| `TerminationReason.USER` | `"user"` | User manually stopped the cycle. |
| `TerminationReason.TERMINAL_DROP` | `"terminal_drop"` | Anomalously-early hard cliff-to-0 finalize (opt-in). Stamped by terminal-drop fast finalize. |
| `ANTI_WRINKLE_ELIGIBLE_REASONS` | `frozenset({TIMEOUT, SMART})` | Only these termination reasons keep a completed cycle eligible for dryer anti-wrinkle handling (user-stopped intentionally excluded). |

`TerminationReason` is a `StrEnum` (members == their string) so JSON/string comparisons keep working. Consumed in `cycle_detector.py`, `playground.py`.

---

## 2. Configuration keys — detection / matching / housekeeping (const.py:41-97)

All are `CONF_*` string-key constants. "Panel-settable" = editable in the panel Settings tab via `ws_set_options` (the 180+ tunables live in the panel, not the HA config flow). The config flow itself only edits device type, power sensor, min power.

| CONF key | String value | Default constant | Panel-settable | Controls |
|---|---|---|---|---|
| `CONF_POWER_SENSOR` | `power_sensor` | (identity, in `entry.data`) | Config flow | Power sensor entity id being monitored. |
| `CONF_NAME` | `name` | `DEFAULT_NAME` = `"Washing Machine"` | Config flow | Device display name. |
| `CONF_MIN_POWER` | `min_power` | `DEFAULT_MIN_POWER` = `2.0` W | Yes | Minimum power to consider "on". Shareable. |
| `CONF_OFF_DELAY` | `off_delay` | `DEFAULT_OFF_DELAY` = `180` s (per-device via `DEFAULT_OFF_DELAY_BY_DEVICE`) | Yes | Low-power seconds before a cycle is deemed ended. Shareable. |
| `CONF_NOTIFY_SERVICE` | `notify_service` | — | **Deprecated** (migration only) | Old single notify service; migrated to per-event lists. |
| `CONF_NOTIFY_ACTIONS` | `notify_actions` | — | Yes | Actionable notification buttons config. |
| `CONF_NOTIFY_PEOPLE` | `notify_people` | — | Yes | People/presence entities for "only when home". |
| `CONF_NOTIFY_ONLY_WHEN_HOME` | `notify_only_when_home` | `DEFAULT_NOTIFY_ONLY_WHEN_HOME` = `False` | Yes | Suppress notifications when nobody home. |
| `CONF_NOTIFY_FIRE_EVENTS` | `notify_fire_events` | `DEFAULT_NOTIFY_FIRE_EVENTS` = `True` | Yes | Fire HA bus events on cycle start/end. |
| `CONF_NOTIFY_EVENTS` | `notify_events` | — | **Deprecated** (migration only) | Old event-list; migrated to per-event service lists. |
| `CONF_NOTIFY_START_SERVICES` | `notify_start_services` | — | Yes | notify.* services fired on cycle start. |
| `CONF_NOTIFY_FINISH_SERVICES` | `notify_finish_services` | — | Yes | notify.* services fired on cycle finish. |
| `CONF_NOTIFY_LIVE_SERVICES` | `notify_live_services` | — | Yes | notify.* services for live-progress ticks. |
| `CONF_NOTIFY_CYCLE_TIMERS` | `notify_cycle_timers` | — | Yes | User-configured mid-cycle countdown timers. |
| `CONF_NO_UPDATE_ACTIVE_TIMEOUT` | `no_update_active_timeout` | `DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT` = `600` s (per-device via `DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE`) | Yes | Seconds without updates while active → forced stop (publish-on-change plugs). |
| `CONF_LOW_POWER_NO_UPDATE_TIMEOUT` | `low_power_no_update_timeout` | *(no DEFAULT in const.py — derived/handled elsewhere)* | Yes | No-update timeout while in low-power state. |
| `CONF_SMOOTHING_WINDOW` | `smoothing_window` | `DEFAULT_SMOOTHING_WINDOW` = `2` | Yes | Rolling-average window (samples) for power smoothing. |
| `CONF_SAMPLING_INTERVAL` | `sampling_interval` | `DEFAULT_SAMPLING_INTERVAL` = `30.0` s (per-device via `DEFAULT_SAMPLING_INTERVAL_BY_DEVICE`) | Yes | Expected sensor cadence; drives watchdog derivation. |
| `CONF_START_DURATION_THRESHOLD` | `start_duration_threshold` | `DEFAULT_START_DURATION_THRESHOLD` = `5.0` s | Yes | Debounce: sustained-on time before STARTING. Shareable. |
| `CONF_DEVICE_TYPE` | `device_type` | `DEFAULT_DEVICE_TYPE` = `"washing_machine"` | Config flow | Appliance category; selects per-device defaults. |
| `CONF_PROFILE_DURATION_TOLERANCE` | `profile_duration_tolerance` | `DEFAULT_PROFILE_DURATION_TOLERANCE` = `0.25` | Yes | Duration tolerance for profile matching. Shareable. |
| `CONF_INTERRUPTED_MIN_SECONDS` | `interrupted_min_seconds` | `DEFAULT_INTERRUPTED_MIN_SECONDS` = `150` s | **Internal only** | Min runtime before a cycle can be flagged "interrupted". |
| `CONF_PROGRESS_RESET_DELAY` | `progress_reset_delay` | `DEFAULT_PROGRESS_RESET_DELAY` = `1800` s (30 min) | Yes | State-expiry / unload window after a cycle ends. |
| `CONF_LEARNING_CONFIDENCE` | `learning_confidence` | `DEFAULT_LEARNING_CONFIDENCE` = `0.6` | Yes | Min match confidence to request user verification. Shareable. |
| `CONF_DURATION_TOLERANCE` | `duration_tolerance` | `DEFAULT_DURATION_TOLERANCE` = `0.10` (±10%) | Yes | Duration variance before flagging anomaly. Shareable. |
| `CONF_AUTO_LABEL_CONFIDENCE` | `auto_label_confidence` | `DEFAULT_AUTO_LABEL_CONFIDENCE` = `0.9` | Yes | Confidence to auto-label a cycle without asking. Shareable. |
| `CONF_AUTO_MAINTENANCE` | `auto_maintenance` | `DEFAULT_AUTO_MAINTENANCE` = `True` | Yes | Enable nightly cleanup/housekeeping. |
| `CONF_PROFILE_MATCH_INTERVAL` | `profile_match_interval` | `DEFAULT_PROFILE_MATCH_INTERVAL` = `300` s | Yes | Seconds between profile-match attempts. Shareable. |
| `CONF_PROFILE_MATCH_MIN_DURATION_RATIO` | `profile_match_min_duration_ratio` | `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO` = `0.10` (per-device dict) | Yes | Stage-1 gate lower bound (match after 10% of expected duration). Shareable. |
| `CONF_PROFILE_MATCH_MAX_DURATION_RATIO` | `profile_match_max_duration_ratio` | `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO` = `1.5` | Yes | Stage-1 gate upper bound (up to 150% of avg). Shareable. |
| `CONF_MAX_PAST_CYCLES` | `max_past_cycles` | `DEFAULT_MAX_PAST_CYCLES` = `200` | Yes | Retention cap for stored past cycles. |
| `CONF_MAX_FULL_TRACES_PER_PROFILE` | `max_full_traces_per_profile` | `DEFAULT_MAX_FULL_TRACES_PER_PROFILE` = `20` | Yes | Cap of full traces kept per profile. |
| `CONF_MAX_FULL_TRACES_UNLABELED` | `max_full_traces_unlabeled` | `DEFAULT_MAX_FULL_TRACES_UNLABELED` = `20` | Yes | Cap of full traces kept for unlabeled cycles. |
| `CONF_WATCHDOG_INTERVAL` | `watchdog_interval` | `DEFAULT_WATCHDOG_INTERVAL` = `30` s (derived: `2*sampling_interval+1`) | Derived | Force-stop watchdog cadence. |
| `CONF_MATCH_PERSISTENCE` | `match_persistence` | `DEFAULT_MATCH_PERSISTENCE` = `3` | Yes | Consecutive matches needed to commit a match (bypassed by ML early-commit). |
| `CONF_COMPLETION_MIN_SECONDS` | `completion_min_seconds` | `DEFAULT_COMPLETION_MIN_SECONDS` = `600` s (per-device via `DEVICE_COMPLETION_THRESHOLDS`) | Yes | Min runtime to count as a completed cycle. Shareable. |
| `CONF_NOTIFY_BEFORE_END_MINUTES` | `notify_before_end_minutes` | `DEFAULT_NOTIFY_BEFORE_END_MINUTES` = `0` (disabled) | Yes | Pre-completion "almost done" notification lead time. |
| `CONF_RUNNING_DEAD_ZONE` | `running_dead_zone` | (no default; removed) | **Removed 0.5.3** | Migration-only stub (`const.py:87`, "REMOVED in 0.5.3 - was never wired to detection"). No longer panel-settable or shareable. |
| `CONF_END_REPEAT_COUNT` | `end_repeat_count` | `DEFAULT_END_REPEAT_COUNT` = `1` (no repeat) | Yes | Times the end condition must repeat before ending (plug-robustness). |
| `CONF_MIN_OFF_GAP` | `min_off_gap` | `DEFAULT_MIN_OFF_GAP` = `60` s (per-device via `DEFAULT_MIN_OFF_GAP_BY_DEVICE`) | Yes | Min quiet gap to separate two cycles (soak-bridging). Shareable. |
| `CONF_START_ENERGY_THRESHOLD` | `start_energy_threshold` | per-device via `DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE` | Yes | Wh required to confirm a real start (noise rejection). Shareable. |
| `CONF_END_ENERGY_THRESHOLD` | `end_energy_threshold` | `DEFAULT_END_ENERGY_THRESHOLD` = `0.05` Wh | Yes | Wh allowed during end candidates (≈zero energy to end). Shareable. |
| `CONF_START_THRESHOLD_W` | `start_threshold_w` | *(no DEFAULT const; falls back to min_power logic)* | Yes | Custom power threshold for STARTING. Shareable. |
| `CONF_STOP_THRESHOLD_W` | `stop_threshold_w` | *(no DEFAULT const)* | Yes | Custom power threshold for ENDING (hysteresis). Shareable. |
| `CONF_POWER_OFF_THRESHOLD_W` | `power_off_threshold_w` | `DEFAULT_POWER_OFF_THRESHOLD_W` = `0.0` (disabled) | Yes | Power-based Off: below this (must sit below stop_threshold_w) terminal→Off. Shareable. |
| `CONF_POWER_OFF_DELAY` | `power_off_delay` | `DEFAULT_POWER_OFF_DELAY` = `30` s | Yes | Debounce below power-off threshold before Finished/Clean→Off. Shareable. |
| `CONF_EXPOSE_DEBUG_ENTITIES` | `expose_debug_entities` | (default off) | Yes | Gate: expose detailed debug sensors. |

---

## 3. Phase-matching opt-in + phase-consistency advisory (const.py:99-117)

| Name | Value | Panel-settable | Description / consumers |
|---|---|---|---|
| `CONF_ENABLE_PHASE_MATCHING` | `"enable_phase_matching"` | Yes (per-device, default off) | Blends phase-resolved (per-role budget) ETA into time-remaining for washing machine / washer-dryer. Consumed in `phase_segmenter.py`. |
| `PHASE_CONSISTENCY_MIN_CYCLES` | `4` | — | Min cycles before the phase-structure consistency advisory runs. `profile_store.py`/`phase_segmenter.py`. |
| `PHASE_HEAT_CV_WARN` | `0.45` | — | Heating-time std/mean above this → likely mixed temperatures under one label (clean ~0.2, mixed 30/40/90°C ~0.45-0.6). |
| `PHASE_HEAT_OCC_MIXED_LO` | `0.25` | — | Heating present in only 25%–75% of cycles → mixed with a non-heating program (lower bound). |
| `PHASE_HEAT_OCC_MIXED_HI` | `0.75` | — | Upper bound of the mixed-occupancy band. |

---

## 4. Debug traces + interruption / anti-wrinkle / delay-start CONF keys (const.py:118-137)

| CONF key | String value | Default constant | Panel-settable | Controls |
|---|---|---|---|---|
| `CONF_SAVE_DEBUG_TRACES` | `save_debug_traces` | — | Yes | Store rich debug info to improve historical cycle data. |
| `CONF_ABRUPT_DROP_WATTS` | `abrupt_drop_watts` | `DEFAULT_ABRUPT_DROP_WATTS` = `500.0` W | Internal | Power-cliff threshold for "interrupted" status. |
| `CONF_ABRUPT_DROP_RATIO` | `abrupt_drop_ratio` | `DEFAULT_ABRUPT_DROP_RATIO` = `0.6` | Internal | Relative drop ratio (60%) for interrupted status. |
| `CONF_ABRUPT_HIGH_LOAD_FACTOR` | `abrupt_high_load_factor` | `DEFAULT_ABRUPT_HIGH_LOAD_FACTOR` = `5.0` | Internal | High-load-factor threshold for interruption. |
| `CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD` | `auto_tune_noise_events_threshold` | `DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD` = `3` | Internal | Ghost cycles before auto-tune adjusts thresholds. |
| `CONF_EXTERNAL_END_TRIGGER_ENABLED` | `external_end_trigger_enabled` | — | Yes | Enable external cycle-end trigger. |
| `CONF_EXTERNAL_END_TRIGGER` | `external_end_trigger` | — | Yes | Binary-sensor entity for external cycle end. |
| `CONF_EXTERNAL_END_TRIGGER_INVERTED` | `external_end_trigger_inverted` | — | Yes | Invert external trigger (fire on OFF). |
| `CONF_ANTI_WRINKLE_ENABLED` | `anti_wrinkle_enabled` | `DEFAULT_ANTI_WRINKLE_ENABLED` = `False` | Yes | Dryer anti-wrinkle shielding. |
| `CONF_ANTI_WRINKLE_MAX_POWER` | `anti_wrinkle_max_power` | `DEFAULT_ANTI_WRINKLE_MAX_POWER` = `400.0` W | Yes | Power ceiling for anti-wrinkle spikes. |
| `CONF_ANTI_WRINKLE_MAX_DURATION` | `anti_wrinkle_max_duration` | `DEFAULT_ANTI_WRINKLE_MAX_DURATION` = `60.0` s | Yes | Max duration treated as anti-wrinkle. |
| `CONF_ANTI_WRINKLE_EXIT_POWER` | `anti_wrinkle_exit_power` | `DEFAULT_ANTI_WRINKLE_EXIT_POWER` = `0.8` W | Yes | Below this = true-off exit from anti-wrinkle. |
| `CONF_DELAY_START_DETECT_ENABLED` | `delay_start_detect_enabled` | `DEFAULT_DELAY_START_DETECT_ENABLED` = `False` | Yes | Enable delayed-start detection. |
| `CONF_DELAY_CONFIRM_SECONDS` | `delay_confirm_seconds` | `DEFAULT_DELAY_CONFIRM_SECONDS` = `60.0` s | Yes | Sustained standby-band time before DELAY_WAIT engages. |
| `CONF_DELAY_TIMEOUT_HOURS` | `delay_timeout_hours` | `DEFAULT_DELAY_TIMEOUT_HOURS` = `8.0` h | Yes | Safety timeout while waiting to start. |

Note: deprecated 0.4.5 drain-spike keys (`delay_drain_*`) are stripped during config migration in `__init__.py` via raw string literals — no constants remain.

---

## 5. Notification event ids + content keys + door/pause (const.py:140-170)

**Event-type identifiers** (`NOTIFY_EVENT_*`):

| Name | Value | Meaning |
|---|---|---|
| `NOTIFY_EVENT_START` | `"cycle_start"` | Cycle start notification type. |
| `NOTIFY_EVENT_FINISH` | `"cycle_finish"` | Cycle finish notification type. |
| `NOTIFY_EVENT_LIVE` | `"cycle_live"` | Live-progress tick type. |
| `NOTIFY_EVENT_CLEAN` | `"cycle_clean"` | Laundry still inside after end. |
| `NOTIFY_EVENT_TIMER` | `"cycle_timer"` | User-configured mid-cycle countdown. |

**Notification content / channel CONF keys:**

| CONF key | String value | Default | Panel-settable | Controls |
|---|---|---|---|---|
| `CONF_NOTIFY_TITLE` | `notify_title` | `DEFAULT_NOTIFY_TITLE` = `"WashData: {device}"` | Yes | Notification title template. |
| `CONF_NOTIFY_ICON` | `notify_icon` | — | Yes | Notification icon. |
| `CONF_NOTIFY_START_MESSAGE` | `notify_start_message` | `DEFAULT_NOTIFY_START_MESSAGE` = `"{device} started."` | Yes | Start message template. |
| `CONF_NOTIFY_FINISH_MESSAGE` | `notify_finish_message` | `DEFAULT_NOTIFY_FINISH_MESSAGE` = `"{device} finished. Duration: {duration}m."` | Yes | Finish message template. |
| `CONF_NOTIFY_PRE_COMPLETE_MESSAGE` | `notify_pre_complete_message` | `DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE` = `"{device}: Less than {minutes} minutes remaining."` | Yes | Pre-completion message. |
| `CONF_NOTIFY_LIVE_INTERVAL_SECONDS` | `notify_live_interval_seconds` | `DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS` = `300` | Yes | Seconds between live-progress ticks. |
| `CONF_NOTIFY_LIVE_OVERRUN_PERCENT` | `notify_live_overrun_percent` | `DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT` = `20` | Yes | Overrun % that triggers a live warning. |
| `CONF_NOTIFY_LIVE_CHRONOMETER` | `notify_live_chronometer` | `DEFAULT_NOTIFY_LIVE_CHRONOMETER` = `False` | Yes | Show a running chronometer in live notifications. |
| `CONF_NOTIFY_REMINDER_MESSAGE` | `notify_reminder_message` | `DEFAULT_NOTIFY_REMINDER_MESSAGE` = `"{device}: about {minutes} minutes left."` | Yes | Distinct one-time pre-end alert. |
| `CONF_NOTIFY_TIMEOUT_SECONDS` | `notify_timeout_seconds` | `DEFAULT_NOTIFY_TIMEOUT_SECONDS` = `0` (never) | Yes | Auto-dismiss notifications after N s. |
| `CONF_NOTIFY_CHANNEL` | `notify_channel` | `DEFAULT_NOTIFY_CHANNEL` = `""` | Yes | Android channel for status/live/reminder. |
| `CONF_NOTIFY_FINISH_CHANNEL` | `notify_finish_channel` | `DEFAULT_NOTIFY_FINISH_CHANNEL` = `""` | Yes | Distinct Android channel for finished/clean. |
| `CONF_ENERGY_PRICE_STATIC` | `energy_price_static` | — | Yes | Static kWh price for cost estimates. |
| `CONF_ENERGY_PRICE_ENTITY` | `energy_price_entity` | — | Yes | Dynamic-price entity id. |
| `CONF_PEAK_RATE_THRESHOLD` | `peak_rate_threshold` | — | Yes | Price at/above which start notification appends a peak-rate tip. |
| `CONF_PEAK_RATE_MESSAGE` | `peak_rate_message` | `DEFAULT_PEAK_RATE_MESSAGE` = `"Running at peak rate ({price}/kWh)."` | Yes | Peak-rate advisory text. |
| `CONF_DOOR_SENSOR_ENTITY` | `door_sensor_entity` | — | Yes | Optional binary_sensor for machine door. |
| `CONF_PAUSE_CUTS_POWER` | `pause_cuts_power` | — | Yes | Also turn off switch entity when pausing. |
| `CONF_SWITCH_ENTITY` | `switch_entity` | — | Yes | Optional switch toggled on pause/resume. |
| `CONF_NOTIFY_UNLOAD_DELAY_MINUTES` | `notify_unload_delay_minutes` | `DEFAULT_NOTIFY_UNLOAD_DELAY_MINUTES` = `60` min | Yes | Minutes before "laundry waiting" nag. |
| `CONF_NOTIFY_UNLOAD_MESSAGE` | `notify_unload_message` | `DEFAULT_NOTIFY_UNLOAD_MESSAGE` = `"{device} finished {duration}m ago - laundry is still inside."` | Yes | Clean-laundry nag template. |

---

## 6. Quiet hours + milestones + linked device (const.py:172-189)

| CONF key | String value | Default | Panel-settable | Controls |
|---|---|---|---|---|
| `CONF_NOTIFY_QUIET_START_HOUR` | `notify_quiet_start_hour` | `DEFAULT_NOTIFY_QUIET_START_HOUR` = `None` | Yes | Quiet-hours (DND) window start hour (0-23). `None`/start==end = off. Finish-type notifications inside the window are held and delivered at window end; live ticks + start are never delayed. |
| `CONF_NOTIFY_QUIET_END_HOUR` | `notify_quiet_end_hour` | `DEFAULT_NOTIFY_QUIET_END_HOUR` = `None` | Yes | Quiet-hours window end hour. |
| `CONF_NOTIFY_MILESTONES` | `notify_milestones` | `DEFAULT_NOTIFY_MILESTONES` = `[50, 100, 500, 1000]` | Yes | Lifetime completed-cycle counts that trigger a milestone notification. Empty/malformed = no-op. |
| `CONF_NOTIFY_MILESTONE_MESSAGE` | `notify_milestone_message` | `DEFAULT_NOTIFY_MILESTONE_MESSAGE` = `"{device} has completed {cycle_count} cycles!"` | Yes | Milestone message template. |
| `CONF_LINKED_DEVICE` | `linked_device` | — | Yes | Device-registry id to expose "Connected via <device>" (via_device relationship). |

---

## 7. Notification default templates & flags (const.py:191-215)

| Name | Value |
|---|---|
| `DEFAULT_NOTIFY_TITLE` | `"WashData: {device}"` |
| `DEFAULT_NOTIFY_START_MESSAGE` | `"{device} started."` |
| `DEFAULT_NOTIFY_FINISH_MESSAGE` | `"{device} finished. Duration: {duration}m."` |
| `DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE` | `"{device}: Less than {minutes} minutes remaining."` |
| `DEFAULT_NOTIFY_REMINDER_MESSAGE` | `"{device}: about {minutes} minutes left."` |
| `DEFAULT_NOTIFY_LIVE_WAITING_MESSAGE` | `"{device}: No profile matched yet."` |
| `DEFAULT_NOTIFY_ONLY_WHEN_HOME` | `False` |
| `DEFAULT_NOTIFY_FIRE_EVENTS` | `True` |
| `DEFAULT_NOTIFY_LIVE_INTERVAL_SECONDS` | `300` |
| `DEFAULT_NOTIFY_LIVE_OVERRUN_PERCENT` | `20` |
| `DEFAULT_NOTIFY_LIVE_CHRONOMETER` | `False` |
| `DEFAULT_NOTIFY_TIMEOUT_SECONDS` | `0` (never auto-dismiss) |
| `DEFAULT_NOTIFY_CHANNEL` | `""` (omit channel → companion default) |
| `DEFAULT_NOTIFY_FINISH_CHANNEL` | `""` (reuse status channel) |
| `DEFAULT_NOTIFY_UNLOAD_DELAY_MINUTES` | `60` |
| `DEFAULT_NOTIFY_UNLOAD_MESSAGE` | `"{device} finished {duration}m ago - laundry is still inside."` |
| `DEFAULT_PEAK_RATE_MESSAGE` | `"Running at peak rate ({price}/kWh)."` |
| `DEFAULT_NOTIFY_QUIET_START_HOUR` | `None` |
| `DEFAULT_NOTIFY_QUIET_END_HOUR` | `None` |
| `DEFAULT_NOTIFY_MILESTONES` | `[50, 100, 500, 1000]` |
| `DEFAULT_NOTIFY_MILESTONE_MESSAGE` | `"{device} has completed {cycle_count} cycles!"` |

---

## 8. Core detection defaults (const.py:217-263)

| Name | Value | Notes |
|---|---|---|
| `DEFAULT_MIN_POWER` | `2.0` W | On threshold. |
| `DEFAULT_OFF_DELAY` | `180` s | 3 min, safer for 60 s polling. |
| `DEFAULT_NAME` | `"Washing Machine"` | — |
| `DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT` | `600` s | 10 min without updates while active → forced stop. |
| `DEFAULT_SMOOTHING_WINDOW` | `2` | — |
| `DEFAULT_SAMPLING_INTERVAL` | `30.0` s | — |
| `DEFAULT_START_DURATION_THRESHOLD` | `5.0` s | Debounce. |
| `DEFAULT_END_ENERGY_THRESHOLD` | `0.05` Wh | Effectively-zero energy to end. |
| `DEFAULT_DEVICE_TYPE` | `"washing_machine"` | — |
| `DEFAULT_PROFILE_DURATION_TOLERANCE` | `0.25` | — |
| `DEFAULT_INTERRUPTED_MIN_SECONDS` | `150` s | Internal, not exposed. |
| `DEFAULT_PROGRESS_RESET_DELAY` | `1800` s | 30-min state expiry/unload window. |
| `DEFAULT_POWER_OFF_THRESHOLD_W` | `0.0` | Disabled (enable marker; when >0 must sit below stop_threshold_w). |
| `DEFAULT_POWER_OFF_DELAY` | `30` s | Terminal-state debounce. |
| `DEFAULT_LEARNING_CONFIDENCE` | `0.6` | Min confidence to request verification. |
| `DEFAULT_DURATION_TOLERANCE` | `0.10` | ±10% duration variance before flagging. |
| `DEFAULT_AUTO_LABEL_CONFIDENCE` | `0.9` | High-confidence auto-label threshold. |
| `DEFAULT_AUTO_MAINTENANCE` | `True` | Nightly cleanup on by default. |
| `DEFAULT_COMPLETION_MIN_SECONDS` | `600` s | 10 min. |
| `DEFAULT_NOTIFY_BEFORE_END_MINUTES` | `0` | Disabled. |
| `DEFAULT_PROFILE_MATCH_INTERVAL` | `300` s | 5 min. |
| `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO` | `0.10` | Match after 10% of expected duration. |
| `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO` | `1.5` | **Tuned 1.3→1.5** (commit-recall 71.6%→73.4%, negligible FP change; 1.3 rejected extended/anti-wrinkle variants). |
| `DEFAULT_MAX_PAST_CYCLES` | `200` | — |
| `DEFAULT_MAX_FULL_TRACES_PER_PROFILE` | `20` | — |
| `DEFAULT_MAX_FULL_TRACES_UNLABELED` | `20` | — |
| `DEFAULT_WATCHDOG_INTERVAL` | `30` s | Derived: `2*sampling_interval + 1`. |
| `DEFAULT_MATCH_PERSISTENCE` | `3` | — |
| `DEFAULT_RUNNING_DEAD_ZONE` | *(removed in 0.5.3)* | Constant no longer exists; the setting was retired. |
| `DEFAULT_END_REPEAT_COUNT` | `1` | No repeat required. |

### Matching & Termination Stability (const.py:265-267)

| Name | Value | Description / consumers |
|---|---|---|
| `DEFAULT_MATCH_REVERT_RATIO` | `0.4` | Drop from peak score that reverts a committed match back to "detecting". Consumed in `cycle_detector.py`, `manager.py`, `ml/training_task.py`. |
| `DEFAULT_DEFER_FINISH_CONFIDENCE` | `0.55` | Min ML end-guard confidence to defer a cycle finish (anti-premature-stop). Gates `_should_defer_finish` / `_ml_end_confidence`. |

---

## 9. ML live-decision gates + match-ranking history (const.py:269-283)

| Name | Value | Description / consumers |
|---|---|---|
| `ML_MATCH_COMMIT_THRESHOLD` | `0.85` | P(top-1 correct) needed to commit a match *before* the persistence counter is met (cuts time-to-first-match). Owner-holdout precision ~0.87 at this score. Consumed in `manager.py` (`_async_do_perform_matching`) and `ml/training_task.py`. Gated on `CONF_ENABLE_ML_MODELS`. |
| `ML_QUALITY_SUSPICIOUS_THRESHOLD` | `0.65` | P(cycle is a problem) above which a high-confidence auto-label is downgraded to a feedback request. Tuned for specificity ~0.84. |
| `MATCH_RANKING_HISTORY_MAX` | `500` | Max per-cycle live_match feature snapshots retained on-device (~6-12 months). Training dataset for on-device `live_match` retraining. |

---

## 10. Runtime overrun / underrun / energy anomaly ratios (const.py:285-301)

| Name | Value | Description / consumers |
|---|---|---|
| `CYCLE_OVERRUN_ANOMALY_RATIO` | `1.5` | Soft, visible (never a notification) overrun signal: a running cycle exceeding matched-profile expected duration by this ratio is flagged. Below the 300% zombie-kill limit. Consumed in `progress.py`, `manager.py`, `playground.py`. |
| `CYCLE_UNDERRUN_ANOMALY_RATIO` | `0.55` | Post-cycle-only: a cycle finishing below 55% of median expected duration is flagged "underrun" (mutually exclusive with overrun). |
| `ENERGY_ANOMALY_Z_THRESHOLD` | `2.5` | \|z-score\| above this vs the profile's historical energy = "energy_spike"/"energy_low". Needs ≥3 labeled cycles. |

---

## 11. Profile warm-up / shape-drift / cluster-shape (const.py:303-321)

| Name | Value | Description |
|---|---|---|
| `CONF_PROFILE_MIN_WARMUP_CYCLES` | `5` | Labeled cycles before auto-matching enabled; a new profile below this always requests manual confirmation. (Named `CONF_*` but is a numeric constant, not a stored option key.) |
| `SHAPE_DRIFT_THRESHOLD` | `0.85` | Pearson correlation (earliest third vs recent third envelope) below this = shape drift. |
| `SHAPE_DRIFT_MIN_CYCLES` | `10` | Min labeled cycles to check drift. |
| `SHAPE_DRIFT_RESAMPLE_N` | `50` | Points for envelope comparison. |
| `CLUSTER_SHAPE_SIMILARITY_THRESHOLD` | `0.75` | Min normalized cross-correlation for a shape-similar unmatched cluster (coverage-gap suggestion A3). |
| `CLUSTER_RESAMPLE_N` | `50` | Points for pairwise cluster comparison. |

---

## 12. Terminal-drop fast finalize `TERMINAL_DROP_*` (const.py:323-343)

Opt-in, gated on `CONF_ENABLE_ML_MODELS` via the manager provider. Pure statistics (no trained model). Asymmetric: can only *shorten* the end wait, and only for anomalously-early hard cliffs on a familiar cycle. Consumed in `manager.py` (`_terminal_drop_provider`, `_terminal_drop_baseline`), decided by `profile_store.is_terminal_drop`.

| Name | Value | Description |
|---|---|---|
| `TERMINAL_DROP_OFF_DELAY_SECONDS` | `90` | Shortened below-threshold wait once a drop is deemed terminal (vs full soak-bridging min_off_gap: up to 8 min washers / 1 h dishwashers). |
| `TERMINAL_DROP_MIN_CLEAN_CYCLES` | `3` | Completed clean cycles needed before the learned baselines are trusted. |
| `TERMINAL_DROP_MIN_QUIET_SPAN_S` | `60` | Sustained sub-threshold span that counts as a legit quiet period. |
| `TERMINAL_DROP_EARLINESS_RATIO` | `0.8` | Fire only if the drop starts < ratio × earliest-ever-quiet offset. |
| `TERMINAL_DROP_MIN_PEAK_RATIO` | `5.0` | Cycle must have been clearly ON (peak ≥ ratio × stop_threshold). |
| `TERMINAL_DROP_PEAK_FAMILIAR_TOL` | `0.4` | Familiarity gate: peak must sit within the device's historical peak range widened by this tolerance, else it may be a NEW program → deferred to the slow path. |

---

## 13. Interruption / anti-wrinkle / delay-start / pump defaults (const.py:345-372)

| Name | Value | Notes |
|---|---|---|
| `DEFAULT_ABRUPT_DROP_WATTS` | `500.0` W | Power-cliff interruption threshold. |
| `DEFAULT_ABRUPT_DROP_RATIO` | `0.6` | 60% drop considered abrupt. |
| `DEFAULT_ABRUPT_HIGH_LOAD_FACTOR` | `5.0` | High-load-factor threshold. |
| `DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD` | `3` | Ghost cycles before threshold auto-adjust. |
| `DEFAULT_ANTI_WRINKLE_ENABLED` | `False` | — |
| `DEFAULT_ANTI_WRINKLE_MAX_POWER` | `400.0` W | — |
| `DEFAULT_ANTI_WRINKLE_MAX_DURATION` | `60.0` s | — |
| `DEFAULT_ANTI_WRINKLE_EXIT_POWER` | `0.8` W | True-off exit threshold. |
| `DEFAULT_DELAY_START_DETECT_ENABLED` | `False` | — |
| `DEFAULT_DELAY_CONFIRM_SECONDS` | `60.0` s | Sustained standby before DELAY_WAIT. |
| `DEFAULT_DELAY_TIMEOUT_HOURS` | `8.0` h | Give-up timeout. |
| `CONF_PUMP_STUCK_DURATION` | `"pump_stuck_duration"` | Pump device type only; panel-settable. Seconds before a running pump is flagged stuck. |
| `DEFAULT_PUMP_STUCK_DURATION` | `1800` s | 30 min (typical sump pump runs <60 s → 30 min implies jam). |
| `EVENT_PUMP_STUCK` | `"ha_washdata_pump_stuck"` | HA bus event fired on stuck-pump threshold. Consumed in `manager.py`. |

---

## 14. Profile matching thresholds + DTW bandwidth (const.py:374-382)

| CONF/Name | Value | Panel-settable | Controls |
|---|---|---|---|
| `CONF_PROFILE_MATCH_THRESHOLD` | `"profile_match_threshold"` | Yes | Score to commit a match. Shareable. |
| `CONF_PROFILE_UNMATCH_THRESHOLD` | `"profile_unmatch_threshold"` | Yes | Score below which a match is dropped. Shareable. |
| `DEFAULT_PROFILE_MATCH_THRESHOLD` | `0.4` | — | Commit threshold (near-optimal per precision harness; `MATCH_KEEP_MIN_SCORE`=0.1 sits below it). |
| `DEFAULT_PROFILE_UNMATCH_THRESHOLD` | `0.35` | — | Un-match threshold. |
| `CONF_DTW_BANDWIDTH` | `"dtw_bandwidth"` | Yes | Sakoe-Chiba band width. |
| `DEFAULT_DTW_BANDWIDTH` | `0.20` | — | 20% Sakoe-Chiba constraint. |

---

## 15. Matching pipeline scoring constants `MATCH_*` (const.py:384-459)

Centralised out of `analysis.py`/`profile_store.py`. Consumed in `analysis.py`, `profile_store.py`, `playground.py`, `ws_api.py`. Most are overridable via matcher config keys (shown in parens).

**Stage 2 — Core similarity:** `score = MATCH_CORR_WEIGHT·max(0,corr) + (1−MATCH_CORR_WEIGHT)·mae_score`

| Name | Value | Meaning / tuning history |
|---|---|---|
| `MATCH_CORR_WEIGHT` | `0.45` | Correlation weight (MAE weight = 0.55, inline). **Tuned 0.6→0.45**: more MAE weight lifted leave-one-out top-1 74%→79.5% and recall/FP net 10.7%→13.7% (FP flat). 0.35-0.45 is a broad plateau. Overridable `corr_weight`. |
| `MATCH_MAE_SCALE` | `100.0` | Half-saturation of the MAE-score curve `MAE_SCALE/(MAE_SCALE+scaled_mae)`. |
| `MATCH_MAE_REF_PEAK` | `1000.0` W | Peak at which scale-invariant MAE == raw MAE (behaviour-neutral calibration point). |
| `MATCH_MAE_PEAK_FLOOR` | `50.0` W | Floor so tiny/idle traces don't explode the peak ratio. |
| `MATCH_KEEP_MIN_SCORE` | `0.1` | Candidates below this are discarded. Overridable `keep_min_score`. |

**Stage 3 — DTW-lite refinement:** `blended = MATCH_DTW_BLEND·core + (1−MATCH_DTW_BLEND)·dtw_score`

| Name | Value | Meaning / tuning history |
|---|---|---|
| `MATCH_DTW_BLEND` | `0.5` | DTW/core blend (near-optimal). Overridable `dtw_blend`. |
| `MATCH_DTW_DIST_SCALE` | `50.0` | Half-saturation for DTW distance score. Overridable `dtw_l1_scale`. |
| `MATCH_DTW_REFINE_TOP_N` | `5` | DTW applied to top-N candidates. **Tuned to 5** (rescues correct profiles Stage-2 ranked 4th-5th; +1.8pp). Overridable `dtw_refine_top_n`. |
| `DEFAULT_DTW_MODE` | `"ensemble"` | DTW variant (config key `dtw_mode`): `legacy`/`scaled`/`ddtw`/`ensemble`. Leave-one-out top-1: off 62.4%, legacy 66.4%, scaled 69.9%, ddtw 69.0%, ensemble(w=0.7,dd=30) 70.7%; ensemble + top_n=5 → 72.5%. |
| `MATCH_DTW_RESAMPLE_N` | `200` | Common grid length for scaled/ddtw DTW. |
| `MATCH_DDTW_DIST_SCALE` | `30.0` | Half-saturation for derivative-DTW distance. Overridable `dtw_ddtw_scale`. |
| `MATCH_DTW_ENSEMBLE_W` | `0.7` | Weight on L1(scaled) vs DDTW in ensemble mode. Overridable `dtw_ensemble_w` (also on-device tunable via matching_tuner). |

**Ambiguity + Smart-Termination landscape guard:**

| Name | Value | Meaning |
|---|---|---|
| `MATCH_AMBIGUITY_MARGIN` | `0.05` | `is_ambiguous = (top1−top2) < margin`. **Single source** used by both match paths + the Match Ambiguity diagnostic sensor. |
| `SMART_TERM_LANDSCAPE_RATIO` | `1.5` | A non-winning candidate ≥1.5× longer (with decent shape) means the trace may be a *prefix* of a longer program → Smart Termination blocked (Quick 46 vs Normal 88 min ratio 1.91 triggers; Quick 46 vs Eco 60 ratio 1.30 doesn't). Consumed in `profile_store.py`. |
| `SMART_TERM_LANDSCAPE_MIN_SHAPE` | `0.40` | Min shape score (pre-Stage-4) to qualify as a prefix candidate. |
| `REFERENCE_PROFILE_CURVE_POINTS` | `50` | Points in the compact reference curve on the `_program` sensor (keeps the attribute <~1 KB). |

**Stage 4 — duration/energy agreement:** `final = (1−dur_w−en_w)·shape + dur_w·dur_agreement + en_w·energy_agreement`, `agreement = 1/(1 + |ln(obs/exp)|/scale)`

| Name | Value | Meaning / tuning history |
|---|---|---|
| `MATCH_DURATION_WEIGHT` | `0.22` | Duration-agreement weight. **Tuned 0.15→0.22** with scale halved. Overridable `duration_weight` (on-device tunable). |
| `MATCH_ENERGY_WEIGHT` | `0.22` | Energy-agreement weight. Same tuning. Overridable `energy_weight` (on-device tunable, independent axis). |
| `MATCH_DURATION_SCALE` | `0.175` | ln-ratio at which duration agreement halves. **Halved 0.35→0.175** (sharper) — net 13.7%→17.4% with FP dropping 62.7%→59.9%. Overridable `duration_scale`. |
| `MATCH_ENERGY_SCALE` | `0.25` | ln-ratio at which energy agreement halves. **Halved 0.5→0.25.** Overridable `energy_scale`. |

`MatchResult.confidence` is the raw Stage-2/3 similarity of the top candidate (0-1) — a similarity score, not a calibrated probability.

---

## 16. States `STATE_*` + STATE_COLORS (const.py:461-500)

State machine: `OFF → STARTING → RUNNING ↔ PAUSED → ENDING → OFF`.

| Name | Value |
|---|---|
| `STATE_OFF` | `"off"` |
| `STATE_DELAY_WAIT` | `"delay_wait"` |
| `STATE_IDLE` | `"idle"` |
| `STATE_STARTING` | `"starting"` |
| `STATE_RUNNING` | `"running"` |
| `STATE_PAUSED` | `"paused"` |
| `STATE_USER_PAUSED` | `"user_paused"` |
| `STATE_ENDING` | `"ending"` |
| `STATE_FINISHED` | `"finished"` |
| `STATE_ANTI_WRINKLE` | `"anti_wrinkle"` |
| `STATE_INTERRUPTED` | `"interrupted"` |
| `STATE_FORCE_STOPPED` | `"force_stopped"` |
| `STATE_RINSE` | `"rinse"` |
| `STATE_UNKNOWN` | `"unknown"` |
| `STATE_CLEAN` | `"clean"` (cycle ended, door not yet opened) |

`STATE_COLORS` — dict mapping each state (+ `"recording"`) to a CSS color using HA theme variables with hex fallbacks (e.g. `STATE_RUNNING → "var(--success-color, #4caf50)"`). Single source of truth surfaced over the WebSocket `get_constants` command; consumed by `ws_api.py`/panel.

---

## 17. Device types + `DEVICE_TYPES` map (const.py:502-530)

| Constant | Value | Label (in `DEVICE_TYPES`) |
|---|---|---|
| `DEVICE_TYPE_WASHING_MACHINE` | `"washing_machine"` | Washing Machine |
| `DEVICE_TYPE_DRYER` | `"dryer"` | Dryer |
| `DEVICE_TYPE_WASHER_DRYER` | `"washer_dryer"` | Washer-Dryer Combo |
| `DEVICE_TYPE_DISHWASHER` | `"dishwasher"` | Dishwasher |
| `DEVICE_TYPE_AIR_FRYER` | `"air_fryer"` | Air Fryer |
| `DEVICE_TYPE_BREAD_MAKER` | `"bread_maker"` | Bread Maker |
| `DEVICE_TYPE_PUMP` | `"pump"` | Pump / Sump Pump |
| `DEVICE_TYPE_GENERIC` | `"generic"` | Other (Advanced) — full profile matching/learning, neutral defaults |
| `DEVICE_TYPE_OTHER` | `"other"` | Threshold Device — threshold-only, no profile matching; migration target for removed types |

(Coffee machine, EV, heat pump, oven were removed in 0.5.0; existing entries migrate to `other`.)

---

## 18. Device-type default maps + dishwasher end-spike constants + group cohesion (const.py:532-696)

### `DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE` (overrides scalar `600`)

| Device | Value |
|---|---|
| dishwasher | `14400` (4 h — long drying) |
| bread_maker | `7200` (2 h — low-power proving) |
| pump | `DEFAULT_PUMP_STUCK_DURATION + 60` = `1860` (must exceed stuck-alarm so the alarm fires first) |

### Dishwasher end-of-cycle handling (issue #43) — standalone constants

| Name | Value | Meaning |
|---|---|---|
| `DEFAULT_MAX_DEFERRAL_SECONDS` | `14400` (4 h) | Max safe deferral. |
| `DISHWASHER_END_SPIKE_MIN_PROGRESS` | `0.85` | Spike before 85% of expected is ignored for end-spike pre-arming. |
| `DISHWASHER_END_SPIKE_WAIT_SECONDS` | `1800.0` (30 min) | Upper bound to keep the cycle open awaiting pump-out. **Widened 300→1800 s** after real-world misfires ~4 min early. Same constant used by `_should_defer_finish` AND the STATE_ENDING wait branch. |
| `DISHWASHER_MIN_CYCLE_DURATION_S` | `1800.0` | Defer finish for any dishwasher below this floor (even shortest quick programs take 30 min). |
| `DISHWASHER_MATCH_FREEZE_QUIET_SECONDS` | `300.0` | Once ENDING + sustained-quiet this long, freeze live re-matching (idle tail would drift Stage-4 duration score toward longer near-duplicates). Self-correcting on real resume. |
| `DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS` | `600.0` | Release pump-out wait early once BOTH expected duration reached AND sustained-quiet this long (avoids hanging the full 30 min past expected). |
| `DISHWASHER_SMART_TERMINATION_DEBOUNCE_SECONDS` | `300.0` | Confirmation window in ENDING before Smart Termination fires. **Deliberately fixed, NOT derived from off_delay** (old `max(300, off_delay*0.25)` coupling slipped ends by 20+ min). |

Consumed in `cycle_detector.py`.

### `DEFAULT_OFF_DELAY_BY_DEVICE` (overrides scalar `180`)

| Device | Value |
|---|---|
| dishwasher | `1800` (30 min drying) |
| bread_maker | `300` (5 min keep-warm) |
| pump | `20` (sharp cutoff) |

### `DEVICE_SMOOTHING_THRESHOLDS` (progress backward-damping, %-points)

| Device | Value |
|---|---|
| washing_machine | `5.0` |
| dryer | `3.0` |
| washer_dryer | `5.0` |
| dishwasher | `5.0` |
| air_fryer | `2.0` |
| bread_maker | `5.0` |
| pump | `2.0` |
| generic | `3.0` |

### `DEVICE_COMPLETION_THRESHOLDS` (min runtime for a valid completed cycle, s)

| Device | Value |
|---|---|
| washing_machine | `600` |
| dryer | `600` |
| washer_dryer | `600` |
| dishwasher | `900` |
| air_fryer | `300` |
| bread_maker | `1800` |
| pump | `5` |

### `DEFAULT_MIN_OFF_GAP_BY_DEVICE` (cycle-separation gap, s; scalar fallback `DEFAULT_MIN_OFF_GAP` = `60`)

| Device | Value |
|---|---|
| washing_machine | `480` (8 min soak) |
| dryer | `300` |
| washer_dryer | `600` |
| dishwasher | `3600` (1 h drying pauses) |
| air_fryer | `120` |
| bread_maker | `600` |
| pump | `60` |

### `DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE` (Wh to confirm start)

| Device | Value |
|---|---|
| washing_machine | `0.2` |
| dryer | `0.5` |
| washer_dryer | `0.3` |
| dishwasher | `0.2` |
| air_fryer | `0.2` |
| bread_maker | `0.2` |
| pump | `0.003` |

### `DEFAULT_SAMPLING_INTERVAL_BY_DEVICE` (s; scalar default `30.0`)

| Device | Value |
|---|---|
| washing_machine | `2.0` |
| washer_dryer | `2.0` |
| dishwasher | `2.0` |
| pump | `10.0` |

(2 s captures rapid 0↔150 W motor oscillation in wet appliances; 30 s default undersamples.)

### `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE`

| Device | Value |
|---|---|
| dishwasher | `0.10` |

### Profile groups (Stage 5)

| Name | Value | Meaning |
|---|---|---|
| `GROUP_MIN_COHESION` | `0.80` | Min pairwise (DTW/Sakoe-Chiba, peak-normalised) shape similarity for the matcher to collapse a group into one aggregate candidate; looser groups stay individual + flagged. Genuine temp/spin variants score ~0.86-0.95, distinct programs <~0.6; 0.80 leaves margin below the 0.85 suggestion bar. Consumed in `profile_store.py`, `ws_api.py`. |

---

## 19. Storage (const.py:698-714)

| Name | Value | Description |
|---|---|---|
| `STORAGE_VERSION` | `11` | Current storage schema version (migrations in `profile_store.WashDataStore._async_migrate_func`). Documented steps: v1→2 signature; v2→3 ISO→offset + status + profile device_type; v3→4 phases; v4→5 custom_phases canonicalize; v5→6 golden backfill (recorded==golden); v6→7 broadened backfill; v7→8 structural fallback backfill; v9 pre-init additive keys (lifetime_energy_wh, settings_changelog, maintenance_log); v11 marker-only bump for derived per-phase profile cache (self-populates on envelope rebuild). |
| `STORAGE_KEY` | `"ha_washdata"` | Store file key. |

---

## 20. HA bus events, signals, learning service (const.py:716-727)

| Name | Value | Description |
|---|---|---|
| `EVENT_CYCLE_STARTED` | `"ha_washdata_cycle_started"` | Fired on cycle start (32 KB event limit — power_data/debug excluded). |
| `EVENT_CYCLE_ENDED` | `"ha_washdata_cycle_ended"` | Fired on cycle end. |
| `SIGNAL_WASHER_UPDATE` | `"ha_washdata_update_{}"` | Dispatcher signal template (per entry_id) for entity updates. |
| `SERVICE_SUBMIT_FEEDBACK` | `"ha_washdata.submit_cycle_feedback"` | Feedback-submission service id. |

---

## 21. Feature flags (const.py:729-745) + the per-device ML-models gate

Staged-rollout gates: when `False`, the corresponding panel UI *and* background logic stay inert.

| Name | Value | Gates |
|---|---|---|
| `SHOW_ML_LAB` | `True` | ML insights in the panel (per-cycle ML health/review inline in Cycles; Classic-vs-ML comparison; the ML Training tab surface). Consumed in `ws_api.py`. |
| `ENABLE_ML_SUGGESTIONS` | `True` | `MLSuggestionEngine` + Classic-vs-ML settings comparison. |
| `ENABLE_ML_TRAINING` | `True` | On-device scheduled/manual training loop + `trigger_ml_training` service + `ml_training_*` options. |

**`CONF_ENABLE_ML_MODELS`** (`"enable_ml_models"`) — **NOT in const.py**; defined in `ml/engine.py:46`, read by `ml_models_enabled(options)` (`ml/engine.py:56`, default `False`). Per-device opt-in gate for feeding ML/anomaly signals into live decisions (ML end-guard, ML early match commit, ML quality gate, ML remaining-time regressor, terminal-drop fast finalize). Toggle lives in the panel's ML Training tab.

---

## 22. Community store (const.py:747-802)

| Name | Value | Panel-settable | Description |
|---|---|---|---|
| `CONF_ENABLE_ONLINE_FEATURES` | `"enable_online_features"` | Yes | Master gate for the WashData Store (browse/import/share reference cycles); default off → tab + network inert. |
| `CONF_STORE_BRAND` | `"store_brand"` | Yes | Declared appliance brand. |
| `CONF_STORE_MODEL` | `"store_model"` | Yes | Declared appliance model. |
| `DEFAULT_ENABLE_ONLINE_FEATURES` | `False` | — | Online features off by default. |
| `SHAREABLE_SETTING_KEYS` | tuple (20 keys) | — | Allow-list of recognition/matching thresholds intrinsic to the appliance model that travel with a shared device bundle (Stage 3). All plain numbers; **no** entity ids, notify services, energy price, sampling/smoothing, housekeeping timers, or behaviour toggles. Consumed in `store_client.py`, `ws_api.py`. |
| `STORE_PROJECT_ID` | `"washdata-store"` | — | Public Firebase project id (not secret). |
| `STORE_API_KEY` | `"AIzaSyDzq0MoWdU_21CSohZUhIIV7ZwfWppjcAk"` | — | Public Firebase web API key (not secret; access enforced by Firestore rules). Consumed in `store_client.py`, `ws_api.py`. |
| `STORE_WEB_ORIGIN` | `"https://3dg1luk43.github.io/washdata-store"` | — | Store web origin. |
| `SUPPORTED_CYCLE_SCHEMA_VERSIONS` | `{1}` | — | Reference-cycle trace formats this integration can import. |
| `QC_RECORDING` | `1` | — | Provenance code: pure recorder capture. Consumed in `store.py`/`store_client.py`. |
| `QC_EDITED` | `2` | — | Provenance code: trimmed/edited from a detected cycle. |
| `QC_MANUAL` | `3` | — | Provenance code: plain detected cycle flagged golden by hand. |

**`SHAREABLE_SETTING_KEYS` members (20):** `CONF_MIN_POWER`, `CONF_OFF_DELAY`, `CONF_START_THRESHOLD_W`, `CONF_STOP_THRESHOLD_W`, `CONF_START_DURATION_THRESHOLD`, `CONF_START_ENERGY_THRESHOLD`, `CONF_COMPLETION_MIN_SECONDS`, `CONF_MIN_OFF_GAP`, `CONF_END_ENERGY_THRESHOLD`, `CONF_POWER_OFF_THRESHOLD_W`, `CONF_POWER_OFF_DELAY`, `CONF_PROFILE_MATCH_THRESHOLD`, `CONF_PROFILE_UNMATCH_THRESHOLD`, `CONF_PROFILE_MATCH_INTERVAL`, `CONF_PROFILE_MATCH_MIN_DURATION_RATIO`, `CONF_PROFILE_MATCH_MAX_DURATION_RATIO`, `CONF_PROFILE_DURATION_TOLERANCE`, `CONF_DURATION_TOLERANCE`, `CONF_AUTO_LABEL_CONFIDENCE`, `CONF_LEARNING_CONFIDENCE`. (`CONF_RUNNING_DEAD_ZONE` was removed from the allow-list in 0.5.3.)

---

## 23. On-device ML training (Stage 4) (const.py:803-848)

Gated behind `ENABLE_ML_TRAINING`. Consumed in `ml/training_task.py`, `manager.py`, `progress.py`, `ws_api.py`.

| Name | Value | Panel-settable | Description |
|---|---|---|---|
| `CONF_ML_TRAINING_ENABLED` | `"ml_training_enabled"` | Yes (per-device, "Learn from this machine") | Opt-in scheduled retraining. |
| `CONF_ML_TRAINING_HOUR` | `"ml_training_hour"` | Yes | Local hour (0-23) to train. |
| `CONF_ML_TRAINING_MIN_CYCLES` | `"ml_training_min_cycles"` | Yes | Min labelled clean cycles before training. |
| `CONF_ML_TRAINING_INTERVAL_DAYS` | `"ml_training_interval_days"` | Yes | Min days between retrains. |
| `DEFAULT_ML_TRAINING_ENABLED` | `False` | — | Off by default. |
| `DEFAULT_ML_TRAINING_HOUR` | `2` | — | 02:00 local quiet hour. |
| `DEFAULT_ML_TRAINING_MIN_CYCLES` | `30` | — | Meaningful corpus first. |
| `DEFAULT_ML_TRAINING_INTERVAL_DAYS` | `7` | — | Retrain at most weekly. |
| `ML_TRAINING_AUC_MARGIN` | `0.02` | — | Classifier promoted when held-out AUC ≥ (baseline − margin); small negative slack lets personalisation win at tiny AUC cost. |
| `ML_TRAINING_BACC_MARGIN` | `0.02` | — | Calibration gate: retrained classifier must not degrade balanced accuracy at the live cutoff by more than this (distinct metric from AUC). |
| `ML_TRAINING_MIN_POSITIVES` | `20` | — | Min positive examples to trust a classifier fit. |
| `ML_TRAINING_HISTORY_MAX` | `30` | — | Per-capability held-out-score history retained (fit-trend badge in panel). |
| `ML_TRAINING_REGRESSION_MARGIN` | `0.05` | — | Remaining-time regressor promoted only when held-out MAE beats naive elapsed/expected by ≥5%. |
| `ML_TRAINING_MIN_REGRESSION_ROWS` | `30` | — | Synthesized prefix rows needed to fit a regressor. |
| `ML_PROGRESS_BLEND_WEIGHT` | `0.5` | — | Weight blending the ML completion-fraction with the phase-aware estimate *before* EMA/monotonicity guards (so a bad model can't wholly override). Consumed in `progress.py`. |
| `SERVICE_TRIGGER_ML_TRAINING` | `"trigger_ml_training"` | — | Manual-training service (behind `ENABLE_ML_TRAINING`). |
| `EVENT_ML_TRAINING_COMPLETE` | `"ha_washdata_ml_training_complete"` | — | Bus event fired when a training run finishes. |

---

## 24. Suggestion quality gates (const.py:850-861)

| Name | Value | Description |
|---|---|---|
| `MIN_SUGGESTION_REL_DELTA` | `0.08` | 8% min relative change; a suggestion is only stored when it clears this OR a per-key absolute minimum (else deleted as noise). |
| `MIN_SUGGESTION_COOLDOWN_CYCLES` | `3` | After the user applies suggestions, suppress new ones for this many completed cycles. |

---

## 25. Appliance health & predictive maintenance (Group E) (const.py:863-886)

| Name | Value | Panel-settable | Description |
|---|---|---|---|
| `CONF_MAINTENANCE_REMINDER_CYCLES` | `"maintenance_reminder_cycles"` | Yes (dict `{event_type: cycle_threshold}` via `ws_set_options`) | Per-device maintenance reminders; when completed cycles since the last event of a type reach its threshold, it surfaces (sensor attr + panel banner). Threshold 0 / absent = disabled. Consumed in `manager.py`, `profile_store.py`, `ws_api.py`. |
| `DEFAULT_MAINTENANCE_REMINDER_CYCLES` | `{"descale": 30, "filter_clean": 50, "drum_clean": 100}` | — | Default reminder thresholds (bearing_service/other absent → opt-in). |
| `MAINTENANCE_EVENT_TYPES` | `("descale", "filter_clean", "drum_clean", "bearing_service", "other")` | — | Recognised maintenance event types. |
| `MAINTENANCE_RECENT_SUPPRESS_DAYS` | `30` | — | A logged event of a matching type within this many days suppresses the "needs maintenance" nag (duration-trend / shape-drift). |

---

## Assessment notes for a doc-writer

- **Single-source discipline:** several constants are explicitly the *only* definition of a value — `MATCH_AMBIGUITY_MARGIN` (both match paths + sensor), `STATE_COLORS` (panel + WS `get_constants`), the `progress.py` blend (`ML_PROGRESS_BLEND_WEIGHT`), and the dishwasher end-spike wait (`DISHWASHER_END_SPIKE_WAIT_SECONDS` shared by two code paths). Do not fork these.
- **`CONF_ENABLE_ML_MODELS` lives in `ml/engine.py`, not `const.py`** — the most likely doc trap.
- **`CONF_PROFILE_MIN_WARMUP_CYCLES` is misnamed** — it is a numeric constant (`5`), not a stored option key, despite the `CONF_` prefix.
- Some `CONF_*` keys have **no `DEFAULT_*` constant** in const.py (`low_power_no_update_timeout`, `start_threshold_w`, `stop_threshold_w`) — their effective defaults are derived in code.
- Deprecated keys retained only for migration: `CONF_NOTIFY_SERVICE`, `CONF_NOTIFY_EVENTS`.
- Many defaults are **per-device dicts** that override the scalar default; always mention the device override alongside the scalar.
