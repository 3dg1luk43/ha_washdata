# WashData `manager.py` — Authoritative Engineering Reference

**File:** `/root/ha_washdata/custom_components/ha_washdata/manager.py`
**Size:** 6359 lines (CLAUDE.md says "~5200 lines" — **stale**; the file is ~22% larger).
**Class:** `WashDataManager` (one instance per config entry / appliance).
**Role:** Central orchestrator — subscribes to the power sensor, feeds readings into `CycleDetector`, orchestrates async profile matching, computes progress/remaining/phase/energy (via `progress.py`), delivers notifications, wires the opt-in ML subsystem, persists state, and produces the attributes consumed by the entity platforms.

Two module-level helper functions and two module-level constants precede the class:
- `_sanitize_ranking(raw_list, limit=5)` (L274) — strips heavy `current`/`sample` power arrays from candidate ranking so it's safe to persist on `cycle_data` and to include under the 32 KB event-data limit. Returns `[{name, score(rounded 3dp), profile_duration}]`.
- `_pn_create(hass, message, *, title, notification_id)` (L305) / `_pn_dismiss(hass, notification_id)` (L333) — best-effort persistent-notification create/dismiss via the **dynamic** `hass.components.persistent_notification` accessor (deliberately not a direct import, so the test suite's mock is honoured). Failures logged at debug, never raised.
- `_QUIET_HOURS_EVENT_TYPES = frozenset({NOTIFY_EVENT_FINISH, NOTIFY_EVENT_CLEAN, "pre_complete"})` (L269) — the finish-type events gated by quiet hours. START and LIVE are intentionally excluded.
- `_MOBILE_ONLY_EXTRA_KEYS` (L292) — `tag, timeout, channel, priority, actions, sticky, subtitle, content_state, activity`. These are forwarded ONLY to `mobile_app_*` notify targets (strict-schema platforms like Signal reject unknown keys).

Cross-module imports of note: `cycle_detector` (CycleDetector/Config), `learning` (LearningManager), `profile_store` (ProfileStore + `decompress_power_data`, `is_terminal_drop`, `terminal_drop_baseline`), `signal_processing` (`integrate_wh`, `energy_gap_threshold_s`), `recorder` (CycleRecorder), `diag_buffer` (DiagBuffer), `progress as progress_mod`, `notification_rules as notif_rules`, `phase_segmenter.phase_matching_enabled`, and (lazily) `store.StoreBridge`, `ml.engine`, `ml.feature_extraction`, `ml.training_task`, `ml.matching_tuner`, `ws_api._compute_ml_comparison`.

---

## 0. Method Inventory (grep of every `def`/`class`, with line anchors)

| Line | Symbol | Kind |
|---|---|---|
| 274 | `_sanitize_ranking` | module fn |
| 305 | `_pn_create` | module fn |
| 333 | `_pn_dismiss` | module fn |
| 351 | `class WashDataManager` | class |
| 355 | `store_bridge` | @property (lazy StoreBridge) |
| 362 | `__init__` | ctor |
| 746 | `profile_matcher_wrapper` | nested (detector callback) |
| 881 | `_async_perform_combined_matching` | async |
| 910 | `_async_do_perform_matching` | async (match orchestration core) |
| 1345 | `top_candidates` | @property |
| 1361 | `phase_description` | @property |
| 1379 | `_current_phase_from_progress` | method |
| 1397 | `match_ambiguity` | @property |
| 1404 | `last_ambiguity_margin` | @property |
| 1413 | `_attempt_state_restoration` | async |
| 1441 | `is_viable_restore` | nested |
| 1675 | `async_setup` | async lifecycle |
| 1803 | `_load_notify_services` | method |
| 1825 | `async_reload_config` | async lifecycle |
| 2231 | `async_shutdown` | async lifecycle |
| 2294 | `_setup_external_end_trigger` | async |
| 2324 | `_setup_door_sensor_listener` | async |
| 2341 | `_handle_door_sensor_change` | @callback |
| 2388 | `_setup_notify_people_listener` | async |
| 2429 | `_handle_external_trigger_change` | @callback |
| 2476 | `_setup_maintenance_scheduler` | async |
| 2492 | `run_maintenance` | nested async |
| 2511 | `_setup_ml_training_scheduler` | method |
| 2544 | `_scheduled` | nested async |
| 2552 | `async_run_ml_training` | async (ML) |
| 2675 | `_tune_matching_config` | async (ML) |
| 2709 | `async_recompute_cycle_health` | async (ML) |
| 2744 | `_last_ml_training_at` | method |
| 2771 | `_async_power_changed` | @callback (power ingestion) |
| 2834 | `_check_state_save` | method |
| 2855 | `_run_final_match_from_cycle_data` | async |
| 2901 | `_start_watchdog` / 2915 `_stop_watchdog` | method |
| 2922 | `_start_state_expiry_timer` / 2940 `_stop_state_expiry_timer` | method |
| 2950 | `_handle_state_expiry` | async (terminal expiry + power-off + clean nag) |
| 3086 | `_reset_terminal_to_off` | method |
| 3105 | `_cancel_power_off_timer` / 3111 `_arm_power_off_timer` / 3131 `_power_off_timer_check` | method |
| 3173 | `_watchdog_check_stuck_cycle` | async (watchdog) |
| 3428 | `_on_state_change` | detector callback |
| 3545 | `_discard_cycle_cleanup` | method |
| 3562 | `_on_cycle_end` | detector callback (sync front-half) |
| 3644 | `_ml_end_confidence` | ML end-guard provider |
| 3683 | `_profile_end_expectation` | method (cache bridge) |
| 3700 | `_terminal_drop_provider` | ML anomaly provider |
| 3742 | `_terminal_drop_baseline` / 3766 `_schedule_terminal_drop_refresh` / 3774 `_async_refresh_terminal_drop_baseline` | method/async |
| 3795 | `_ml_progress_percent` / 3819 `_ml_energy_total` | wrappers |
| 3845 | `_compute_cycle_quality_score` | ML quality gate |
| 3917 | `_resolve_energy_price` | method |
| 5215 | `_async_price_history` | async (recorder price lookup, #426) |
| 5264 | `_async_apply_cycle_cost` / 5292 `_async_apply_dynamic_cost` / 5344 `_cost_from_timeline` | cycle-end cost freeze (#426) |
| 2941 | `_setup_price_listener` / 2963 `_dynamic_pricing_enabled` / 2973 `_handle_price_change` / 2986 `_append_price_sample` / 3022 `_start_price_timeline` | dynamic price timeline (#426) |
| 3576 | `async_recompute_cycle_costs` | async (recorder recost pass, #426) |
| 7494 | `_live_cost_so_far` | live projection input (#426) |
| 3941 | `_format_vs_typical` | @staticmethod |
| 3975 | `_peak_rate_tip` | method |
| 3999 | `_async_process_cycle_end` | async (cycle-end heavy tail) |
| 4450 | `profile_sample_repair_stats` | @property |
| 4455 | `suggestions` | @property |
| 4462 | `_quiet_hours_bounds` / 4469 `_in_quiet_hours` / 4480 `_seconds_until_quiet_end` | notif predicates |
| 4489 | `_queue_quiet_hours_notification` / 4510 `_schedule_quiet_hours_flush` / 4528 `_flush_quiet_hours_notifications` / 4553 `_cancel_quiet_hours_timer` | quiet-hours queue |
| 4562 | `_milestone_crossed` @staticmethod / 4575 `_lifetime_cycle_count` / 4591 `_maybe_notify_milestone` | milestone |
| 4647 | `_build_ios_live_activity_extras` @staticmethod / 4684 `_mobile_service_extras` @staticmethod | iOS LiveActivity |
| 4697 | `_safe_format_template` | method |
| 4733 | `_get_services_for_event` / 4748 `_resolve_channel` | notif routing |
| 4766 | `_dispatch_notification` | notif core |
| 4903 | `_send_notification_service` | notif delivery |
| 5009 | `_run_notification_actions` | notif delivery (script) |
| 5058 | `_is_any_notify_person_home` / 5067 `_handle_notify_person_change` | presence |
| 5103 | `_handle_noise_cycle` / 5122 `_tune_threshold` | auto-tune |
| 5161 | `_update_estimates` | estimate loop |
| 5218 | `_analyze_trend` | method |
| 5239 | `_reset_live_notification_state` | method |
| 5248 | `_is_mobile_notify_service` @staticmethod | method |
| 5261 | `_timer_pause_action_id` @property | method |
| 5265 | `_estimate_live_notification_cap` | method |
| 5277 | `_check_live_progress_notification` | live notif |
| 5429 | `_clear_live_progress_notification` / 5502 `_clear_clean_notification` | notif clear |
| 5549 | `_check_pre_completion_notification` | notif |
| 5591 | `_update_projected_energy` / 5629 `_update_cycle_anomaly` / 5642 `_update_remaining_only` | progress wrappers |
| 5730 | `_check_cycle_timers` | user cycle timers |
| 5781 | `_async_auto_pause_and_notify` / 5790 `_setup_timer_pause_notification` / 5848 `_clear_timer_pause_notification` | timer pause |
| 5870 | `_estimate_phase_progress` | progress wrapper |
| 5885 | `_notify_update` / 5889 `notify_update` | dispatcher |
| 5893–6106 | Public properties (see §9) | |
| 6108 | `set_manual_program` | method |
| 6151 | `async_pause_cycle` / 6210 `async_resume_cycle` / 6278 `async_terminate_cycle` | async user actions |
| 6298 | `async_start_recording` / 6317 `async_stop_recording` | async recorder |
| 6325 | `clear_manual_program` | method |
| 6344 | `_run_post_cycle_processing` | async (merge/split maintenance) |

---

## 1. Lifecycle: setup, wiring, listeners, teardown

### `__init__` (L362–879)
Reads all config with the precedence **options → data → default**. Notable:
- `power_sensor_entity_id`, `device_type` (default `DEFAULT_DEVICE_TYPE`).
- Constructs `ProfileStore` (L500) with duration-ratio limits, `save_debug_traces`, `match_threshold`, `unmatch_threshold`; sets `profile_store.dtw_bandwidth`.
- Constructs `LearningManager`, `CycleRecorder`, `DiagBuffer`.
- Builds a full `CycleDetectorConfig` (L645–742) from ~30 options, several with **device-type-keyed defaults**: `min_off_gap` (`DEFAULT_MIN_OFF_GAP_BY_DEVICE`), `start_energy_threshold` (`DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE`), `min_duration_ratio` (`DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE`), `completion_min_seconds` (`DEVICE_COMPLETION_THRESHOLDS`). `start_threshold_w`/`stop_threshold_w` are derived from `min_power` when not set; `delay_timeout_seconds = DELAY_TIMEOUT_HOURS × 3600`.
- Defines nested **`profile_matcher_wrapper`** (L746) handed to the detector as `profile_matcher`. If a manual program is active it returns `(program, 1.0, expected_duration, phase)` synchronously; otherwise it **fires-and-forgets** `_async_perform_combined_matching(readings)` and returns `(None,0,0,None)` — i.e. the detector callback is only a *trigger* for async matching.
- Constructs `CycleDetector` (L783) with callbacks `_on_state_change`, `_on_cycle_end`, `profile_matcher=profile_matcher_wrapper`, `end_confidence_provider=self._ml_end_confidence`, `terminal_drop_provider=self._terminal_drop_provider`.
- Initializes large state surface: live-cycle fields (`_current_program="off"`, progress/remaining/duration, `_smoothed_progress`, `_projected_energy_wh`, `_cycle_anomaly`, `_overrun_ratio`), match bookkeeping (`_score_history`, `_match_persistence_counter`, `_unmatch_persistence_counter`, `_current_match_candidate`), notification state, pause tracking, door/clean state, `_restart_gaps`, terminal-drop caches, and per-device notification tags (`_lifecycle_tag = ha_washdata_{entry_id}_lifecycle`, `_clean_tag = ha_washdata_{entry_id}_clean`; `_live_notification_tag` aliases the lifecycle tag).
- `_ranking_snapshot_cycle_id` (L465) — per-cycle UUID keying ranking snapshots so two cycles sharing a second-resolution start time don't cross-contaminate. Also used as the **back-to-back race token**.

### `store_bridge` property (L355)
Lazily instantiates `StoreBridge` (community online features); cached on `self._store_bridge`.

### `async_setup` (L1675)
Order matters:
1. `await profile_store.async_load()`.
2. Load `options.error.*` translation templates into `_timer_ui_strings` (fixed manager-side UI strings: timer default message, Resume title/body, `vs_typical_*`, live waiting message). English mirrors are inlined as fallbacks.
3. Apply duration tolerance + retention limits (`set_retention_limits`: max_past_cycles, max_full_traces_per_profile, max_full_traces_unlabeled).
4. **Repair** broken sample references (`async_repair_profile_samples`), stash stats on `_profile_sample_repair_stats`, save if repaired.
5. Subscribe to power sensor via `async_track_state_change_event` → `_async_power_changed` (**before** restoration).
6. `_attempt_state_restoration()`.
7. Restore `_last_cycle_end_time` from last completed cycle (ghost-cycle suppression across restart).
8. `recorder.async_load()`.
9. Force an initial `detector.process_reading` from current sensor state.
10. `async_migrate_cycles_to_compressed()`; background `async_backfill_match_confidence()`.
11. Set up external-end-trigger, door-sensor, notify-people listeners.
12. Register maintenance + ML-training schedulers (also re-registered on reload, so they survive restart without a settings re-save).

### `async_reload_config` (L1825)
In-place config reload without interrupting a running cycle.
- **Power-sensor change is blocked while `detector.state == STATE_RUNNING`** (returns early — other options are NOT updated in that case). Otherwise it re-attaches the listener and force-reads the new sensor.
- Rewrites the ENTIRE `detector.config` in place (min_power, off_delay, smoothing, abrupt-drop trio, completion, thresholds W, power-off, energy thresholds, anti-wrinkle, delay-start), duration-ratio limits, match interval, `dtw_bandwidth`, pump-stuck duration, notification config, door/pause config, sampling interval.
- Propagates `device_type` to `learning_manager` and `suggestion_engine`.
- If a cycle is active and live notifications now configured: resets live counters and fires one live tick immediately.
- Re-registers external/door/notify-people listeners + maintenance + ML schedulers.
- Calls `_attempt_state_restoration()` again at the end, and dispatches `ha_washdata_update_{entry_id}`.
- **Edge case:** logs "Configuration reloaded successfully" twice (L2201, L2229). Harmless.

### `async_shutdown` (L2231)
Removes all listeners (power, external, door, notify-people), cancels quiet-hours timer + power-off timer + watchdog + state-expiry + maintenance + ML schedulers, uninstalls diag_buffer, clears timer-pause + live-progress notifications. **Persists the active-cycle snapshot** (enriched with `manual_program`, `notified_start`, `start_event_fired`, `is_user_paused`, `user_pause_start`, `total_user_paused_seconds`) if state ∈ {RUNNING, PAUSED, STARTING, ENDING}.

### Listener setup helpers
- `_setup_external_end_trigger` (L2294) — subscribes to a binary sensor; on trigger (respecting `CONF_EXTERNAL_END_TRIGGER_INVERTED`) calls `detector.reset(OFF)` from ANTI_WRINKLE/DELAY_WAIT or `detector.user_stop()` otherwise. `_handle_external_trigger_change` (L2429) ignores unavailable/unknown transitions.
- `_setup_door_sensor_listener` (L2324) + `_handle_door_sensor_change` (L2341) — door OPEN during active cycle sets `verified_pause=True` and marks `_is_user_paused`; door OPEN after cycle clears Clean state and dismisses the clean nag. **Door close never auto-resumes.**
- `_setup_notify_people_listener` (L2388) — only attaches when `notify_only_when_home AND notify_people`; flushes pending notifications immediately if someone is already home.
- `_setup_maintenance_scheduler` (L2476) — `async_track_time_change` at local **midnight**; `run_maintenance` runs `profile_store.async_run_maintenance()` then `async_recompute_cycle_health()`.
- `_setup_ml_training_scheduler` (L2511) — no-op unless `ENABLE_ML_TRAINING` build flag AND `ml_training_enabled` opt-in; schedules `async_run_ml_training(force=False)` daily at `ml_training_hour` (clamped 0–23).

---

## 2. Power-change handling, sampling, watchdog

### `_async_power_changed` (L2771, `@callback`)
The hot path. Steps:
1. Ignore `None`/UNKNOWN/UNAVAILABLE; non-numeric → return.
2. **Always** record the raw reading into `diag_buffer.record_power(power, new_state.last_updated)` — uses the sensor's own timestamp, captured **before** throttling.
3. **Record-mode interception:** if `recorder.is_recording`, feed the recorder, update `_current_power`/`_last_reading_time`, notify, return (no detection).
4. **Throttle:** if power ≥ min_power AND less than `_sampling_interval` since `_last_reading_time`, return. **Low-power readings bypass the throttle** (critical end-of-cycle signal).
5. `learning_manager.process_power_reading(power, now, last)`, then `detector.process_reading(power, now)`.
6. Capture `_cycle_start_time` from detector on first RUNNING.
7. If state ∈ {RUNNING, PAUSED, ENDING, STARTING}: `_update_estimates()` + `_check_state_save(now)`.
8. `_notify_update()`.

There is **no server-side smoothing here** — smoothing is inside `CycleDetector` (`smoothing_window`). The manager only throttles by sampling interval.

### `_check_state_save` (L2834)
Every ~60 s, fires-and-forgets `async_save_active_cycle(snapshot)` with the enriched snapshot (manual_program + start-notification flags + user-pause fields). Flash-wear mitigation.

### Watchdog — `_start_watchdog`/`_stop_watchdog` (L2901/2915) + `_watchdog_check_stuck_cycle` (L3173, async)
Started on cycle RUNNING; period = `CONF_WATCHDOG_INTERVAL`. Runs only in active states. Ordered checks:
- **0a. Pump-stuck** (device_type == pump): once `net elapsed >= _pump_stuck_duration`, sets `_pump_stuck=True` and fires **`EVENT_PUMP_STUCK`** (`{device, entry_id, elapsed_seconds, threshold_seconds}`). Skipped while user/verified-paused.
- **0. Zombie killer:** if a profile is matched and `net elapsed > expected×3.0 AND > 14400s`, `detector.force_end(now)`. Skipped while paused.
- **1. Ghost-cycle suppressor:** if `_current_program == "detecting..."` within a "suspicious window" after the previous cycle (600 s dishwasher / 180 s else), force-end. Dishwasher pump-out has a faster path (`elapsed>180 AND silence>60`); general path needs `elapsed>600 AND silence>300`.
- **Low-power handling:** computes `effective_low_power_timeout = max(device floor from DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE, _low_power_no_update_timeout)`, extended by remaining expected duration +1800 s, and by `DEFAULT_MAX_DEFERRAL_SECONDS + 1800` when `_verified_pause`. In `is_waiting_low_power()`: force-end when stale beyond the effective timeout; otherwise **inject a 0 W keepalive** to advance the detector accumulator (two triggers: real-update silence > `no_update_active_timeout` (skipped if verified pause), or any-update silence > `off_delay`).
- **High-power handling:** silence > `no_update_active_timeout`: if still high power and within `expected + 14400` (else 14400) inject a refresh reading; otherwise force-end.

Key nuance: keepalive injection uses `detector.process_reading(0.0, now)` and updates `_last_reading_time` but **not** `_last_real_reading_time` (so real-silence detection stays honest).

### State-expiry timer — `_start/_stop_state_expiry_timer` (L2922/2940) + `_handle_state_expiry` (L2950, async)
Polls every 60 s once a cycle has completed (`_cycle_completed_time` set). Responsibilities:
- **Clean-laundry nag** (NOTIFY_EVENT_CLEAN, uses `_clean_tag`) fired once, `_notify_unload_delay_minutes` after cycle end; sets `_notified_clean_laundry` on send OR on deferral (so it isn't re-queued every tick).
- **Power-based Off (issue #284)** — enabled only when `0 < power_off_threshold_w < stop_threshold_w` AND state ∈ {FINISHED, INTERRUPTED, FORCE_STOPPED}. Progress bar zeroes after `progress_reset_delay`, but the terminal STATE persists until power actually drops below the threshold for `power_off_delay` — then `_reset_terminal_to_off()`. Arms a precise one-shot `_arm_power_off_timer` in addition to the 60 s poll backstop.
- **Timer-based Off (feature off):** classic reset to OFF after `progress_reset_delay`.
- **Nag hold:** while a clean nag is still pending, BOTH modes defer leaving the terminal state (`nag_pending` guard).

### Power-off one-shot timer (L3105–3171)
`_cancel_power_off_timer`, `_arm_power_off_timer(delay)` (`async_call_later`), `_power_off_timer_check` — re-verifies enable + terminal-state + below-threshold debounce + nag-hold before `_reset_terminal_to_off()`. A timer left armed after power rises is a harmless no-op.

### `_reset_terminal_to_off` (L3086)
**Single owner** of terminal→OFF. Zeroes progress, clears clean-state + nag flag + power-off tracking, `detector.reset(OFF)`, stops expiry timer, notifies. Shared by timer- and power-based paths so they never diverge.

---

## 3. Matching orchestration

### Trigger chain
`profile_matcher_wrapper` (detector callback, ~every `match_interval`) → `_async_perform_combined_matching(readings)` (L881) → `_async_do_perform_matching(readings)` (L910).

### `_async_perform_combined_matching` (L881, async)
- Guards against concurrent matching (`_matching_task` not done → skip).
- Skips if no readings, or **skips entirely when `profile_store.has_real_profiles` is False** (nothing to match).
- Otherwise stores `_matching_task = hass.async_create_task(_async_do_perform_matching(...))`.

### `_async_do_perform_matching` (L910, async) — the brain of program identification
1. Computes duration; calls `await profile_store.async_match_profile(readings, current_duration)` (executor-offloaded NumPy pipeline). Stores `_last_match_result`, `_last_match_ambiguous`.
2. **Switching logic (temporal persistence):**
   - *Divergence detection* (L946): if current matched program's confidence drops below `peak × (1 - DEFAULT_MATCH_REVERT_RATIO)` (0.4) with >3 history samples, bump `_unmatch_persistence_counter`; when it reaches `_match_persistence`, revert to `detecting...`.
   - *Persistence counters* per candidate (`_match_persistence_counter`), `is_persistent = counter >= _match_persistence`.
   - **Case 1 — initial match** from `detecting...` (confidence ≥ 0.15, not ambiguous or persistent): switch if persistent OR `ml_early_commit`.
   - **Case 2 — mid-cycle override**: high-confidence override when `confidence > 0.8 AND (confidence - current) > 0.15`; else persistence + positive trend + `>0.05` gap.
   - **Case 3 — unmatch** when confidence < `_unmatch_threshold` persistently.
   - Else reset unmatch counter when confidence healthy and no divergence.
3. **ML early-commit path** (L988–1062):
   - Computes `live_match_features` (via `ml.feature_extraction.live_match_features`) whenever there is a non-ambiguous candidate while still `detecting...` — **unconditionally** (not gated on ML opt-in) so training data accrues.
   - **Records ranking snapshot** (`profile_store.record_match_ranking_snapshot`, keyed by `_ranking_snapshot_cycle_id`) unconditionally — back-filled with confirmed label at cycle end.
   - If `ml_models_enabled(options)`: `resolve_scorer("live_match")` → `ml_commit_score`. `ml_early_commit = score >= ML_MATCH_COMMIT_THRESHOLD (0.85) AND confidence >= 0.30`. All ML wrapped in try/except (must never break matching).
4. Applies the switch (updates `_current_program`, `_last_match_confidence`, `_matched_profile_duration`), updates `_score_history` (capped 20 per candidate).
5. **Detector update block** (L1175–1277): envelope-verification of low-power phases (`async_verify_alignment`) sets `verified_pause`; **Smart Termination** releases the pause lock past 95% of profile duration; high power (>10× stop threshold) clears pause; **user pause is authoritative** (`if self._is_user_paused: verified_pause = True`, issue #306). Adds descriptive heuristic phases (dishwasher "Drying", washer "Spinning"/"Rinsing/Soaking"). Pushes `detector.set_verified_pause(...)` + `detector.update_match((name, conf, dur, phase, is_confident_mismatch, is_ambiguous, is_prefix_ambiguous))`.
6. `_update_remaining_only()`.
7. **Restart-recovery start-notification fallback** (L1290): fires `EVENT_CYCLE_STARTED` + start notification if `_notified_start` was missed (e.g. HA restart mid-cycle). Appends a peak-rate tip. Then `_check_pre_completion_notification()`, `_check_live_progress_notification()`, `_notify_update()`.

### `_run_final_match_from_cycle_data` (L2855, async)
Called at cycle end (from `_async_process_cycle_end`) when still `detecting...`/`restored...`. Runs `async_match_profile` on the complete power data (offset format), accepts at the lower `0.15` threshold **ignoring ambiguity** (complete cycle → pick best). Sets `_current_program`, `_last_match_confidence`.

Related properties: `top_candidates` (L1345, sanitized ranking), `phase_description` (L1361), `_current_phase_from_progress` (L1379 → `progress_mod.current_phase`), `match_ambiguity` (L1397), `last_ambiguity_margin` (L1404).

---

## 4. Cycle-end processing

### `_on_cycle_end` (L3562, sync — detector callback front-half)
- Immediately stops watchdog + state-expiry timers, clears timer-pause notification, resets `_pump_stuck`, records `_last_cycle_end_time`.
- Computes cycle energy via the **shared** `integrate_wh(ts, ps, max_gap_s=energy_gap_threshold_s(ts))` (single source with `ProfileStore.add_cycle`; no inline trapezoid — CLAUDE.md rule honoured).
- **Ghost/noise suppression (returns without persisting):**
  - `duration < 60 AND energy < 0.05 Wh` → `_handle_noise_cycle(max_power)` + `_discard_cycle_cleanup()` + return.
  - Dishwasher pump-out: gap 0–600 s after previous end AND duration < 300 s AND energy < 1.0 Wh → same suppress.
- Stores `cycle_data["energy_wh"]`.
- **Captures the race token** `end_token = self._ranking_snapshot_cycle_id`, then schedules `_async_process_cycle_end(cycle_data, cycle_token=end_token)` as a task.

### `_discard_cycle_cleanup` (L3545)
For suppressed ghosts: clears the active-cycle snapshot, anchors `_cycle_completed_time = now`, and arms the expiry timer so the UI returns to OFF. Deliberately does NOT persist/notify/learn.

### `_async_process_cycle_end` (L3999, async — heavy tail)
1. Final match if still detecting (`_run_final_match_from_cycle_data`).
2. **B1 freeze** — captures `program`, `match_result`, `match_confidence`, `matched_profile_duration`, `cycle_anomaly`, `overrun_ratio` into locals **immediately** so the tail describes THIS cycle even if a back-to-back cycle rolls the live fields during the awaits.
3. Attaches `profile_name`/`label_source="auto_match"`/`match_confidence` when a runtime match exists and is a real profile.
4. Attaches `match_ranking_top5` (**sanitized**, small) + full `debug_data` (`ranking`/`details`/`ambiguous`).
5. **Post-cycle auto-labeling** if unlabeled and `_auto_label_confidence > 0`: `async_match_profile` on complete data; accept ≥ threshold → `label_source="auto_label_post"`.
6. **Back-fills ranking-snapshot labels** via `confirm_match_ranking_snapshots(start_iso, confirmed_profile, cycle_id=cycle_token)` (uses the captured token, not the live field).
7. **Envelope conformance + artifacts** (`compute_envelope_conformance`, `detect_cycle_artifacts`) stored on `cycle_data["envelope_conformance"]` / `["artifacts"]`.
8. **Anomalies:**
   - Freezes runtime overrun (`anomaly="overrun"` + `overrun_ratio`).
   - **A1 underrun** (post-cycle only, mutually exclusive with overrun): `duration < median × CYCLE_UNDERRUN_ANOMALY_RATIO (0.55)` → `anomaly="underrun"` + `underrun_ratio`.
   - **A2 energy anomaly:** z-score vs profile energy stats; `|z| > ENERGY_ANOMALY_Z_THRESHOLD (2.5)` → `energy_anomaly="energy_spike"|"energy_low"` + `energy_z_score`.
   - Caches these into `_last_cycle_post_anomaly` for idle sensor attrs.
9. Restart gaps: snapshot onto `cycle_data["restart_gaps"]`; source list cleared **only after** confirmed persist.
10. **Energy cost frozen** (`_async_apply_cycle_cost`, #426). Dynamic mode (a price *entity* + `CONF_ENERGY_PRICE_DYNAMIC`): the stored trace integrated against the cycle's price timeline (`signal_processing.cycle_cost`), writing `cost`, `energy_price` (the **effective** price, cost / kWh), `energy_price_mode="dynamic"` and the compacted `price_timeline`. The timeline comes from the live price-entity listener, or from the recorder (`_async_price_history`) when the listener cannot have been complete (empty, or `restart_gaps` present). Otherwise, and on any failure, the classic single price: `cost = report_wh/1000 × _resolve_energy_price()` with `energy_price_mode="fixed"`.
11. **ML quality score** (`_compute_cycle_quality_score`) — only when `ml_models_enabled`, and offloaded to the executor. Runs BEFORE `async_add_cycle` so reference stats exclude the current cycle.
12. `async_add_cycle(cycle_data)` → `cycle_persisted=True`; `async_rebuild_envelope(profile_name)`.
13. **C2 lifetime counter** (`set_lifetime_cycle_count`, in-memory; persisted by lifetime-energy save) — monotonic, correct across retention trims.
14. **B1 lifetime energy** (`async_add_lifetime_energy_wh`) once per persisted cycle (TOTAL_INCREASING never double-counts).
15. Ensures a stable cycle `id` (sha256 of start_time+duration).
16. **B1 active-snapshot clear** only if the token still matches (else the new cycle owns it).
17. Schedules `_run_post_cycle_processing()` (merge/split maintenance).
18. **Fires `EVENT_CYCLE_ENDED`** — excludes `{power_data, debug_data, power_trace}` (32 KB limit), adds `device_type` + `profile_name`.
19. Clears live progress notification (`clear_services=False`), fires **finish notification** (NOTIFY_EVENT_FINISH) with template vars (`duration`, `program`, `energy_kwh`, `cost`, `time_finished`, `cycle_count`, `vs_typical`), tag=`_lifecycle_tag`, `activity="end"`.
20. **C2 milestone** (`_maybe_notify_milestone(prev, cur)`) only when persisted.
21. **Learning** (`learning_manager.process_cycle_end(...)`) only when persisted, using CAPTURED context.
22. **B1 back-to-back guard:** if `cycle_token != self._ranking_snapshot_cycle_id` (a new cycle started during the awaits), skip the terminal-state reset and return — the new cycle's live fields are preserved.
23. Terminal-state reset: `_current_program="off"`, clears match/progress fields, `_cycle_progress=100.0`, `_cycle_completed_time=now`, resets user-pause and clean-state, enters Clean state if door configured + closed, arms the state-expiry timer.

---

## 5. Progress / remaining / phase / energy wrappers (all thin over `progress.py`)

These confirm the "single source of truth" contract — **none forks the math**:
- `_estimate_phase_progress` (L5870) → `progress_mod.estimate_phase_progress(...)`.
- `_ml_progress_percent` (L3795) → `progress_mod.ml_progress_percent(...)` (opt-in `remaining_time` regressor; None until a model is promoted).
- `_ml_energy_total` (L3819) → `progress_mod.ml_energy_total(...)` (opt-in `total_energy` regressor).
- `_update_projected_energy` → `progress_mod.projected_energy(...)` (prefers ML energy regressor, else `energy_so_far / progress_fraction`; cost via the same price resolution). With dynamic pricing it also passes `cost_so_far=_live_cost_so_far(trace)`, so only the energy still to come is charged at the current price (#426). Wrapped in try/except.
- `_update_cycle_anomaly(duration_so_far)` (L5629) → `progress_mod.cycle_anomaly(...)` → sets `_overrun_ratio`, `_cycle_anomaly` ("overrun" at `CYCLE_OVERRUN_ANOMALY_RATIO`=1.5).
- `_current_phase_from_progress` (L1379) → `progress_mod.current_phase(...)` (indexes per-profile ranges by ML-blended progress, not raw elapsed).

### `_update_remaining_only` (L5642) — the estimate assembler (NOT a pure wrapper; it orchestrates)
- Clears estimates on OFF/UNKNOWN/IDLE, or when no profile duration.
- Throttled to once per 5 s (`_last_phase_estimate_time`).
- Uses **net elapsed** (`net_elapsed_seconds` = wall minus paused) for all estimates; calls `_check_cycle_timers(duration_so_far)`.
- Gathers three progress inputs: `phase_result` (from `_estimate_phase_progress` when ≥10 pts + matched), `ml_pct` (`_ml_progress_percent`), and **opt-in `phase_remaining_s`** (Phase-matching: only when `phase_matching_enabled(options, device_type)` and ≥10 pts; from `profile_store.phase_remaining(...)`).
- Hands all to **`progress_mod.compute_progress(device_type, matched_duration, duration_so_far, smoothed, phase_result, ml_pct, logger, phase_remaining_s=...)`** — the single blend+EMA+back-calc. Assigns `_cycle_progress`, `_smoothed_progress`, `_time_remaining`, `_total_duration`, `_last_total_duration_update`, then `_update_projected_energy()` + `_update_cycle_anomaly(...)`.

### `_update_estimates` (L5161) — the outer loop
Clears everything on dead states. Throttles heavy work to `_profile_match_interval`; between intervals still calls `_update_remaining_only`/pre-complete/live checks. **Matching itself is triggered by the detector callback, not here** (comment "No matching task trigger here anymore").

---

## 6. Notification DELIVERY (all delivery lives here; decisions in `notification_rules.py`)

**Decision predicates delegated to `notif_rules`:** `_quiet_hours_bounds` (L4462), `_in_quiet_hours` (L4469), `_seconds_until_quiet_end` (L4480), `_milestone_crossed` (L4562), `_check_pre_completion_notification` uses `notif_rules.should_notify_pre_completion(...)` (L5549).

### Core dispatch — `_dispatch_notification` (L4766)
Builds title/icon, resolves a home person, assembles `variables`, injects channel (`_resolve_channel`) + auto-dismiss `timeout` into both action variables and service extra_vars. Gating order:
1. **Quiet hours** — if `allow_deferral` AND event ∈ `_QUIET_HOURS_EVENT_TYPES` AND in quiet hours → `_queue_quiet_hours_notification`, sets `_last_dispatch_deferred=True`, returns False.
2. **Presence** — if `allow_presence_deferral` AND `notify_only_when_home` AND nobody home → append to `_pending_notifications` (LIVE entries dedup: only newest LIVE kept), sets deferred, returns False.
3. Runs `_notify_actions` (script) + per-event services. Returns actions_sent OR service_sent.
`_last_dispatch_deferred` lets "fire once" callers (clean nag) treat a deferral as handled.

### Delivery — `_send_notification_service` (L4903)
Per configured service: LIVE-only extras (progress/countdown/chronometer) added; **LIVE targets must be `mobile_app_*`** (others skipped). `_mobile_service_extras` adds `_MOBILE_ONLY_EXTRA_KEYS` only for mobile targets (protects strict-schema platforms + isolates iOS LiveActivity keys). Supports both entity-style (`notify.*` entity → `notify.send_message`) and legacy `domain.service`. Falls back to **persistent notification** (using the tag as the stable id) if no services sent (LIVE never falls back).

### Actions — `_run_notification_actions` (L5009)
Builds a `script_helper.Script` from `_notify_actions`; fires-and-forgets `script.async_run(variables, Context())`. Errors logged, never raised.

### Quiet-hours queue (L4489–4557)
`_queue_quiet_hours_notification` parks finish-type items; `_schedule_quiet_hours_flush` arms ONE `async_call_later` at the window end; `_flush_quiet_hours_notifications` re-dispatches with `allow_deferral=False` but **`allow_presence_deferral=True`** (nobody home → stays in presence queue, not delivered into an empty house). `_cancel_quiet_hours_timer` on shutdown.

### Presence (L5058–5101)
`_is_any_notify_person_home`; `_handle_notify_person_change` releases `_pending_notifications` when a person comes HOME (disables both deferrals on release), updating LIVE counters.

### Milestone (L4562–4642)
Uses the **monotonic lifetime counter** (`_lifetime_cycle_count`), largest crossed milestone → one NOTIFY_EVENT_FINISH with a distinct `{lifecycle_tag}_milestone` tag.

### iOS Live Activity (L4647–4695)
`_build_ios_live_activity_extras` builds `content_state`/`subtitle`/`activity`; `_mobile_service_extras` restricts them to mobile targets.

### Live progress — `_check_live_progress_notification` (L5277)
Only in RUNNING/PAUSED/ENDING with live services/actions. Sends a one-time "waiting" message (localized) when no profile yet (suppressed if no real profiles). Throttled by `max(30, live_interval)`; hard-capped by `_estimate_live_notification_cap` (duration/interval × overrun margin). Emits progress bar + countdown/chronometer + iOS LiveActivity enrichment. Chronometer-overrun logic replaces a frozen 0:00 once.
- `_clear_live_progress_notification` (L5429) — purges queued LIVE/start/pre-complete entries; on finish (`clear_services=False`) does NOT send a service clear (finish reuses the lifecycle tag, avoiding flicker); on shutdown sends the explicit clear.
- `_clear_clean_notification` (L5502) — clears the `_clean_tag` nag from both queues + delivered.

### Pre-completion — `_check_pre_completion_notification` (L5549)
One-shot "X minutes left" via `notif_rules.should_notify_pre_completion`; uses `CONF_NOTIFY_REMINDER_MESSAGE`, tag=`_lifecycle_tag`, `priority:high`, routed to finish channel.

### User cycle timers — `_check_cycle_timers` (L5730)
Fires user timers by net-elapsed minutes (RUNNING/PAUSED only). Auto-pause timers bridge to `_async_auto_pause_and_notify` (L5781) → `_setup_timer_pause_notification` (L5790) which shows the interactive Resume card **only after the pause actually takes effect** (mobile action + sticky + HA persistent + `mobile_app_notification_action` listener). `_clear_timer_pause_notification` (L5848) tears it all down.

### Peak-rate tip — `_peak_rate_tip` (L3975)
Appended to the START notification when configured `peak_rate_threshold` is met by the current price. Purely informational.

**Events fired** (all under `_notify_fire_events` except EVENT_PUMP_STUCK/EVENT_ML_TRAINING_COMPLETE): `EVENT_CYCLE_STARTED`, `EVENT_CYCLE_ENDED`, `EVENT_PUMP_STUCK`, `EVENT_ML_TRAINING_COMPLETE`, plus the dispatcher signal `SIGNAL_WASHER_UPDATE` / `ha_washdata_update_{entry_id}` (entity refresh).

---

## 7. ML wiring (all live paths gated on `CONF_ENABLE_ML_MODELS` via `ml_models_enabled(options)`)

- **`_ml_end_confidence`** (L3644) — end-guard provider handed to the detector. Returns `P(latest low-power event is true end)` from `resolve_scorer("end")` + `latest_end_event_features` + `_profile_end_expectation`; `None` when ML off / no matched profile / no features. Asymmetric: only ever *defers*.
- **`_profile_end_expectation`** (L3683) → `progress_mod.profile_end_expectation(...)`, caches `_ml_end_expectation_cache` per profile.
- **`_terminal_drop_provider`** (L3700) — anomaly (no trained model — pure statistics). Calls `is_terminal_drop(points, earliest, peak_range, stop_threshold_w, EARLINESS_RATIO=0.8, MIN_PEAK_RATIO=5.0, PEAK_FAMILIAR_TOL=0.4)`. Returns False when ML off / insufficient history / novel cycle / not anomalously early. Only ever *shortens* the wait.
- **`_terminal_drop_baseline`** (L3742) — cached `(earliest_quiet_offset, peak_range)`, keyed by cycle count. **Never recomputes synchronously** (issue #311): schedules `_schedule_terminal_drop_refresh` → `_async_refresh_terminal_drop_baseline` (executor via `terminal_drop_baseline(cycles, stop_thresh, MIN_QUIET_SPAN_S=60, MIN_CLEAN_CYCLES=3)`); serves last-known baseline meanwhile, `(None, None)` until first refresh.
- **`_compute_cycle_quality_score`** (L3845) — `resolve_scorer("quality")` + `quality_features` built from profile median stats; stores `cycle_data["ml_quality_score"]` (P(problem)). Runs pre-persist, executor-offloaded, never raises.
- **`async_run_ml_training`** (L2552) — gated `ENABLE_ML_TRAINING`; guards (unless `force`): skip if device active, need ≥ `ml_training_min_cycles`, respect `ml_training_interval_days` (via `_last_ml_training_at`). Sets `_ml_training_running`, calls `ml.training_task.async_run_training`, tracks promotions/`_ml_training_failures`, runs `_tune_matching_config`, persists `set_ml_last_training_run` + `append_ml_training_history`, fires **`EVENT_ML_TRAINING_COMPLETE`**, recomputes cycle health on promotion. Never raises to caller.
- **`_tune_matching_config`** (L2675) — executor `ml.matching_tuner.tune_matching_config(cycles)`; persists `set_matching_config(record)` only on held-out promotion.
- **`async_recompute_cycle_health`** (L2709) — executor `ws_api._compute_ml_comparison(..., force_recompute=True)`; writes `cycle["ml_health"]`, saves. Triggered by nightly maintenance / post-training / Process History.
- **`_last_ml_training_at`** (L2744) — prefers persisted last-run, falls back to newest promoted `trained_at`.

**Note:** the terminal-drop *finalization* (`TERMINAL_DROP_OFF_DELAY_SECONDS`=90) and end-guard bound (`ML_END_GUARD_MAX_DEFER_SECONDS`) live in `cycle_detector.py`; the manager only supplies the providers/baselines.

---

## 8. Background tasks

Manager background work uses **`hass.async_create_task` / `async_add_executor_job` directly — NOT `task_registry.py`** (grep confirms zero `task_registry` references in this file). The registry-tracked long ops (Playground history/sweep, Process-history/reprocess, on-device ML training pills) are wired in `ws_api.py`, not here. **This is a discrepancy vs CLAUDE.md's task_registry description** (which lists "Process history/reprocess, and on-device ML training" as running through the registry) and vs the assessment prompt's framing of §8. In manager, ML training/health recompute/matching-tune are plain fire-and-forget executor jobs; they are not surfaced as registry activity pills.

Actual manager-driven background work:
- **Merge/split maintenance** — `_run_post_cycle_processing` (L6344) → `profile_store.async_run_maintenance()` (5 h lookback; auto-saves). Also runs from `_setup_maintenance_scheduler`'s midnight `run_maintenance`.
- **Rebuild envelopes** — `profile_store.async_rebuild_envelope(profile_name)` after each persisted cycle.
- **Cycle compression/backfill** — `async_migrate_cycles_to_compressed`, background `async_backfill_match_confidence` (setup).
- **Trim/split** — trimming is a service in `__init__.py`; merge/split live inside `async_run_maintenance` (profile_store).
- **Active-cycle snapshot saves** — `_check_state_save`, pause/resume, shutdown.

---

## 9. Entity attribute production (what feeds `sensor.py`)

`sensor.py` (986 lines) reads these manager members (per grep):
- **State sensor:** `check_state()` (L5912 → Recording/Clean/UserPaused overlay over `detector.state`), `samples_recorded`, `current_program`, `sub_state`, `device_type`, `pump_stuck`, `cycle_anomaly`, `overrun_ratio`, `last_cycle_post_anomaly` (dict: anomaly/underrun_ratio/energy_anomaly/energy_z_score), `restart_gaps`, `maintenance_due`.
- **Remaining/duration/progress:** `time_remaining`, `total_duration`, `last_total_duration_update`, `cycle_progress`, `projected_energy_wh`, `projected_cost`.
- **Program/phase:** `current_program`, `phase_description`, `list_phase_catalog`, `get_profile_phase_ranges_for_device`, `profile_store.reference_curve`.
- **Power/timing:** `current_power`, `cycle_start_time`, `sample_interval_stats`, `detector`.
- **Match diagnostics:** `top_candidates`, `last_match_details`, `last_ambiguity_margin`, `match_ambiguity`.
- **Counts/energy:** `cycle_count`, `lifetime_energy_kwh`, `pump_runs_today`.
- **Suggestions:** `suggestions` (→ `profile_store.get_suggestions()`), `profile_sample_repair_stats`.

Other public surface: `is_user_paused`, `is_clean_state`, `net_elapsed_seconds`, `manual_program_active`, `last_cycle_end_time`, `maintenance_due`.
User actions: `async_pause_cycle` (L6151, rollback on switch-off failure), `async_resume_cycle` (L6210, rollback on switch-on failure, accumulates paused time), `async_terminate_cycle` (L6278, `detector.user_stop()`), `set_manual_program` (L6108), `clear_manual_program` (L6325), `async_start/stop_recording` (L6298/6317).

---

## 10. Cross-cutting notes & CLAUDE.md discrepancies

1. **Line count stale:** CLAUDE.md says manager.py "~5200 lines"; actual **6359**.
2. **task_registry not used in manager** (see §8) — CLAUDE.md and the §8 prompt imply reprocess/ML training run through `task_registry`; in manager these are plain tasks/executor jobs. Registry wiring is in `ws_api.py`. Memory note `project_background_task_registry.md` ("wired for playground, reprocess/ML are next") corroborates that ML/reprocess registry integration is NOT complete in manager.
3. **Duplicate log line** in `async_reload_config` (L2201 and L2229) and a duplicated `age = ...` line at L1469–1470 in `_attempt_state_restoration` — cosmetic.
4. **Energy integration** correctly routes through the shared `integrate_wh` + `energy_gap_threshold_s` in both `_on_cycle_end` and (via profile_store) `add_cycle` — CLAUDE.md rule honoured.
5. **32 KB event-data exclusion** correctly excludes `{power_data, debug_data, power_trace}`; `match_ranking_top5` is deliberately sanitized because it is NOT in the exclusion set.
6. **Progress math never forked** — all progress/energy/anomaly/phase paths delegate to `progress.py`; Playground parity is preserved (CLAUDE.md contract holds).
7. **Datetime discipline** — every timestamp uses `dt_util.now()` / `dt_util.parse_datetime`; naive legacy timestamps normalized on restore.
8. **ML never breaks core** — every ML/anomaly path is wrapped in broad try/except returning `None`/`False`/no-op; opt-in gated on `CONF_ENABLE_ML_MODELS`, except ranking-snapshot recording + live-match feature extraction which run unconditionally to accrue training data.
9. **Back-to-back race handling** via `_ranking_snapshot_cycle_id` token (B1) is threaded through `_on_cycle_end` → `_async_process_cycle_end` and guards both the active-snapshot clear and the terminal-state reset.
10. **User pause is authoritative** (issue #306) — re-asserted in the matcher block, on restore, and via the door sensor; detector snapshot does not persist `_verified_pause`, so the manager re-asserts it.

**Key constants used by manager:** `DEFAULT_MATCH_REVERT_RATIO`=0.4, `ML_MATCH_COMMIT_THRESHOLD`=0.85, `CYCLE_OVERRUN_ANOMALY_RATIO`=1.5, `CYCLE_UNDERRUN_ANOMALY_RATIO`=0.55, `ENERGY_ANOMALY_Z_THRESHOLD`=2.5, `TERMINAL_DROP_*` (MIN_CLEAN_CYCLES=3, MIN_QUIET_SPAN_S=60, EARLINESS_RATIO=0.8, MIN_PEAK_RATIO=5.0, PEAK_FAMILIAR_TOL=0.4, OFF_DELAY_SECONDS=90), `DEFAULT_MAX_DEFERRAL_SECONDS`, plus the device-keyed default maps (`DEFAULT_MIN_OFF_GAP_BY_DEVICE`, `DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE`, `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE`, `DEVICE_COMPLETION_THRESHOLDS`, `DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE`).
