# WashData HA-Integration Surface Reference

**Scope:** Entry point, config flow, entity platforms, services, intents, recorder, setup advisor, diagnostics.
**Files read:** `__init__.py`, `config_flow.py`, `sensor.py`, `binary_sensor.py`, `select.py`, `button.py`, `services.yaml`, `intents.py`, `recorder.py`, `setup_advisor.py`, `diagnostics.py`, `diag_buffer.py`, `log_utils.py`
**Integration constants source:** `const.py`

> Content-corrected 2026-08-17 for 0.5.4 (config schema 3.8, running_dead_zone retired). Line anchors may still reflect the original 2026-07-18 / 0.5.1 pass.

---

## 1. Config Entry Schema

### Versions

- **Current version:** `3.8` (class `ConfigFlow`: `VERSION = 3`, `MINOR_VERSION = 8`)
- **Storage version:** `11` (`STORAGE_VERSION` in `const.py:713`)

### Identity keys (stored in `entry.data`)

| Key | Description |
|-----|-------------|
| `name` (via `CONF_NAME`) | Display name / entry title |
| `power_sensor` (`CONF_POWER_SENSOR`) | Moved to options post-migration |

After migration, `entry.data` holds only `name` plus any keys not yet moved; the integration reads all structural tunables from `entry.options` first (`options.get(k, data.get(k, default))`).

### Tunables (stored in `entry.options`)

All 180+ tunables live in `entry.options` and are edited via the WashData panel (`ws_set_options`), NOT via multi-step HA flows. The config-flow / options-flow exposes only three structural fields:

- `device_type` (`CONF_DEVICE_TYPE`)
- `power_sensor` (`CONF_POWER_SENSOR`)
- `min_power` (`CONF_MIN_POWER`)

---

## 2. Config Flow (`config_flow.py`)

### `ConfigFlow` — Initial Setup (`async_step_user`)

**Schema:** `STEP_USER_DATA_SCHEMA` (lines 126-143)

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `name` | string | yes | `DEFAULT_NAME` |
| `device_type` | SelectSelector (dropdown, `translation_key="device_type"`) | yes | `DEFAULT_DEVICE_TYPE` = `"washing_machine"` |
| `power_sensor` | EntitySelector (domain="sensor") | yes | — |
| `min_power` | float (Coerce) | optional | `DEFAULT_MIN_POWER` |

**Validation:** `min_power <= 0` → error `"invalid_power"`. Creates entry with `title=name`, `data=user_input`.

### `ConfigFlow` — Reconfigure (`async_step_reconfigure`)

Uses `_structural_schema(entry)` which resolves values **options-first** (mirrors manager behaviour post-3.6 remap). On success calls `async_update_reload_and_abort` — writes only `options`, not `data`. Does NOT store `CONF_NAME` in `entry.data`.

### `OptionsFlowHandler` — Minimal Options (`async_step_init`)

Docstring: "stub that redirects users to the WashData panel." Presents the same structural schema (`_structural_schema`). On submit calls `_merge_structural_options` which merges device_type/power_sensor/min_power into `entry.options` (preserving all others, stripping any `CONF_NAME` that leaked in). Updates entry title if name changed. Returns `async_create_entry(title="", data=new_options)`.

### Helper `_resolve_options_first`

`entry.options.get(key, entry.data.get(key, default))` — used by both reconfigure and options flows to avoid surfacing stale pre-3.6 remap values.

---

## 3. Config Entry Migration (`async_migrate_entry`, `__init__.py:133-300`)

The function migrates any version ≤ 3.7 → 3.8. Returns `False` for version > 3. All steps are deterministic and idempotent.

### Step-by-step

| Step | What happens |
|------|-------------|
| **3.6 → 3.7** (lines 145-151) | Remove `"initial_profile"` stub key from `entry.data`. |
| **3.7 → 3.8** (lines 153-161) | Drop `running_dead_zone` from `entry.options` (setting retired in 0.5.3; was never wired to detection). Fast path: if already 3.8, returns True immediately. |
| **Core settings: data → options** (lines 164-174) | Move `min_power`, `off_delay`, `device_type`, `power_sensor`, `notify_service` from `entry.data` into `entry.options` if missing from options. |
| **Legacy notify_service → per-event lists** (lines 176-189) | If a single `CONF_NOTIFY_SERVICE` string exists, populate `notify_start_services`, `notify_finish_services`, and (if `NOTIFY_EVENT_LIVE` was in legacy `notify_events`) `notify_live_services`. |
| **Option defaults backfill** (lines 190-250) | `setdefault` for ~30 options: progress_reset_delay, learning_confidence, duration_tolerance, auto_label_confidence, no_update_active_timeout, smoothing_window, profile_duration_tolerance, interrupted_min_seconds, abrupt_drop_watts, abrupt_drop_ratio, abrupt_high_load_factor, device_type, start_duration_threshold, profile_match_interval, profile_match_min/max_duration_ratio, max_past_cycles, max_full_traces_per_profile, max_full_traces_unlabeled, watchdog_interval, auto_tune_noise_events_threshold, completion_min_seconds, notify_before_end_minutes, notify_actions, notify_people, notify_only_when_home, notify_fire_events, notify_live_interval_seconds, notify_live_overrun_percent, notify_timeout_seconds, notify_channel, notify_finish_channel, notify_reminder_message |
| **Keys removed from data** (lines 251-259) | Pop `min_power`, `off_delay`, `device_type`, `power_sensor`, `notify_service` from `entry.data`. |
| **Drain-spike key removal** (lines 261-269) | Pop `delay_drain_min_power`, `delay_drain_max_power`, `delay_drain_max_duration` from options (3.4 obsolete). |
| **Suppress-feedback removal** (lines 271-274) | Pop `suppress_feedback_notifications` from options (3.6 inert). |
| **Removed device-type remap** (lines 276-288) | If `device_type` is `coffee_machine`, `ev`, `heat_pump`, or `oven` → remap to `DEVICE_TYPE_OTHER` ("other" / Threshold Device). All tuned options preserved. |
| **Final write** (lines 290-302) | `async_update_entry(data=data, options=options, version=3, minor_version=8)` |

---

## 4. Device Types

Defined in `const.py:503-529`. All nine types registered in `DEVICE_TYPES` dict.

| Key | Display name | Notes |
|-----|-------------|-------|
| `washing_machine` | Washing Machine | Default (`DEFAULT_DEVICE_TYPE`) |
| `dryer` | Dryer | |
| `washer_dryer` | Washer-Dryer Combo | |
| `dishwasher` | Dishwasher | Longest `min_off_gap` (3600 s), Smart Termination end-spike logic |
| `air_fryer` | Air Fryer | |
| `bread_maker` | Bread Maker | Long watchdog (7200 s), long completion_min (1800 s) |
| `pump` | Pump / Sump Pump | Very short `min_off_gap` (60 s), creates `PumpRunsTodaySensor`; `pump_stuck` attr on state sensor |
| `generic` | Other (Advanced) | Full profile matching/learning with neutral defaults; for predictable non-listed appliances |
| `other` | Threshold Device | Threshold-based detection only, no profile matching; intentionally generic defaults for user tuning; migration target for deprecated types |

**Removed/deprecated types** (migrated to `other`): `coffee_machine`, `ev`, `heat_pump`, `oven`.

### Key per-device-type default differences (from `const.py`)

| Constant | washing_machine | dishwasher | pump | bread_maker | generic |
|----------|----------------|------------|------|-------------|---------|
| `DEFAULT_WATCHDOG_INTERVAL_BY_DEVICE` | 1800 | 14400 | (stuck+60) | 7200 | — |
| `DEFAULT_MIN_OFF_GAP_BY_DEVICE` | 480 | 3600 | 60 | 600 | — |
| `DEFAULT_SMOOTHING_BY_DEVICE` | 5.0 | 5.0 | 2.0 | 5.0 | 3.0 |
| `DEFAULT_COMPLETION_MIN_SECONDS_BY_DEVICE` | 600 | 900 | 5 | 1800 | — |
| `DEFAULT_MIN_OFF_GAP_BY_DEVICE` | 480 | 3600 | 60 | 600 | — |
| `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE` | 1.0 | 0.10 | — | — | — |

---

## 5. Entity Platforms

Platforms registered: `SENSOR`, `BINARY_SENSOR`, `SELECT`, `BUTTON` (`__init__.py:116-121`).

---

### 5.1 Sensor Platform (`sensor.py`)

All sensors inherit `WasherBaseSensor` (has_entity_name=True, device_info set, unique_id = `{entry_id}_{key}`, dispatcher-connected to `SIGNAL_WASHER_UPDATE`).

#### Always-created sensors

| Class | key | Translation key | Device class | Unit | State class | Notes |
|-------|-----|-----------------|-------------|------|------------|-------|
| `WasherStateSensor` | `washer_state` | `washer_state` | ENUM | — | — | 15 possible states; icon varies by device_type |
| `WasherProgramSensor` | `washer_program` | `washer_program` | ENUM | — | — | Options = profile list + "none"/"unknown"; `reference_profile` is `_unrecorded_attributes` |
| `WasherCurrentPhaseSensor` | `current_phase` | `current_phase` | — | — | — | Current cycle phase description |
| `WasherTimeRemainingSensor` | `time_remaining` | `time_remaining` | DURATION | `min` (static) | — | Null in OFF/ANTI_WRINKLE/DELAY_WAIT states |
| `WasherTotalDurationSensor` | `total_duration` | `total_duration` | DURATION | `min` (static) | — | Null in OFF state |
| `WasherProgressSensor` | `cycle_progress` | `cycle_progress` | — | `%` | — | 1 decimal precision |
| `WasherPowerSensor` | `current_power` | `current_power` | POWER | `W` | — | Live power reading |
| `WasherElapsedTimeSensor` | `elapsed_time` | `elapsed_time` | DURATION | `s` | — | 0 when off |
| `WasherDebugSensor` | `debug_info` | `debug_info` | — | — | — | EntityCategory.DIAGNOSTIC; `entity_registry_enabled_default=False` |
| `WasherSuggestionsSensor` | `suggestions` | `suggestions` | — | — | — | EntityCategory.DIAGNOSTIC; value = count of actionable suggestions |
| `WasherCycleCountSensor` | `cycle_count` | `cycle_count` | — | `cycles` | — | Total completed cycles |
| `WasherEnergyTotalSensor` | `energy_total` | `energy_total` | ENERGY | `kWh` | TOTAL_INCREASING | Lifetime accumulating; HA Energy dashboard compatible |

#### Conditionally created sensors

| Class | Condition | key | Notes |
|-------|-----------|-----|-------|
| `PumpRunsTodaySensor` | `device_type == "pump"` | `pump_runs_today` | Cycles in last 24h |
| `WasherMatchConfidenceSensor` | `CONF_EXPOSE_DEBUG_ENTITIES` option | `match_confidence` | DIAGNOSTIC; % |
| `WasherTopCandidatesSensor` | `CONF_EXPOSE_DEBUG_ENTITIES` option | `top_candidates` | DIAGNOSTIC |
| `WasherAmbiguitySensor` (sensor) | `CONF_EXPOSE_DEBUG_ENTITIES` option | `ambiguity` | DIAGNOSTIC; MEASUREMENT; % margin |
| `WasherProfileCountSensor` (dynamic) | One per profile, managed by `WasherProfileSensorManager` | `profile_count_{sha256[:8]}` | DIAGNOSTIC; TOTAL state class; auto-created/removed on profile changes |

#### State Sensor Full Attribute List (`WasherStateSensor.extra_state_attributes`)

| Attribute | Always present | Condition |
|-----------|---------------|-----------|
| `samples_recorded` | yes | — |
| `current_program_guess` | yes | — |
| `sub_state` | yes | — |
| `pump_stuck` | only when `device_type == "pump"` | — |
| `cycle_anomaly` | only when anomaly != "none" | runtime overrun signal |
| `overrun_ratio` | only when anomaly != "none" | rounded to 2dp |
| `last_cycle_anomaly` | only when last cycle had underrun | `"underrun"` |
| `last_cycle_underrun_ratio` | only when underrun_ratio present | — |
| `last_cycle_energy_anomaly` | only when energy_anomaly present | — |
| `last_cycle_energy_z_score` | only when energy_z_score present | — |
| `ha_restart_gaps` | only when gaps exist | count of HA-restart gaps during current cycle |
| `maintenance_due` | only when non-empty | list of event types whose cycle threshold reached |

#### Program Sensor Attributes (`WasherProgramSensor.extra_state_attributes`)

Returns `None` when program is off/detecting/starting/unknown.

| Attribute | Description |
|-----------|-------------|
| `active_phase` | Current phase description string |
| `phase_catalog` | List of `{name, description, is_default}` dicts for all phases of this device type |
| `phase_ranges` | Assigned phase ranges for current profile |
| `reference_profile` | `[[offset_s, watts], ...]` reference power curve (only when matched; excluded from recorder DB via `_unrecorded_attributes`) |

#### Total Duration Sensor Attributes

| Attribute | Description |
|-----------|-------------|
| `last_updated` | Timestamp of last duration update |

#### Progress Sensor Attributes (`WasherProgressSensor.extra_state_attributes`)

Present only when projection is available (omitted when idle/early in cycle).

| Attribute | Description |
|-----------|-------------|
| `projected_energy_kwh` | Projected total energy kWh (3dp); accumulated energy / ML-blended progress |
| `projected_cost` | Projected cycle cost (2dp); present only when cost calculation available |

#### Debug Sensor Attributes (`WasherDebugSensor.extra_state_attributes`)

| Attribute | Description |
|-----------|-------------|
| `sub_state` | Detector sub-state |
| `match_confidence` | `_last_match_confidence` (0.0–1.0) |
| `cycle_id` | `detector._current_cycle_start` |
| `samples` | `detector.samples_recorded` |
| `energy_accum` | `detector._energy_since_idle_wh` |
| `time_below` | `detector._time_below_threshold` |
| `sampling_p95` | 95th-percentile sample interval |
| `noise_events` | Count of noise events |
| `top_candidates` | Full top-candidates list |
| `last_match_details` | Full last match details dict |

#### Suggestions Sensor Attributes

| Attribute | Description |
|-----------|-------------|
| `has_actionable_suggestions` | bool |
| `suggestions_count` | int |
| `suggested_option_keys` | sorted list of option keys with suggestions |
| `suggestions` | full suggestions dict |

Suggestion keys monitored: `min_power`, `off_delay`, `watchdog_interval`, `no_update_active_timeout`, `sampling_interval`, `profile_match_interval`, `auto_label_confidence`, `duration_tolerance`, `profile_duration_tolerance`, `profile_match_min/max_duration_ratio`, `min_off_gap`, `stop_threshold_w`, `start_threshold_w`, `end_energy_threshold`. (`running_dead_zone` was removed in 0.5.3.)

#### Profile Count Sensor Attributes

| Attribute | Description |
|-----------|-------------|
| `average_consumption_kwh` | Profile average energy |
| `total_consumption_kwh` | avg_energy × cycle_count |
| `last_run` | Timestamp of last run |
| `average_length_min` | int minutes |
| `min_length_min` | int minutes |
| `max_length_min` | int minutes |
| `consistency_min` | duration std-dev in minutes |

#### Ambiguity Sensor (debug-gated) Attributes

| Attribute | Description |
|-----------|-------------|
| `is_ambiguous` | bool |

#### Top Candidates Sensor (debug-gated) Attributes

| Attribute | Description |
|-----------|-------------|
| `candidates` | full top-candidates list |

---

### 5.2 Binary Sensor Platform (`binary_sensor.py`)

| Class | unique_id suffix | Translation key | Notes |
|-------|-----------------|-----------------|-------|
| `WasherRunningBinarySensor` | `_running` | `running` | `is_on` = `state == STATE_RUNNING` |
| `WasherAmbiguitySensor` | `_ambiguity` | `match_ambiguity` | Only when `CONF_EXPOSE_DEBUG_ENTITIES`; DIAGNOSTIC; `is_on` = `match_ambiguity` |

`WasherAmbiguitySensor` (binary_sensor) extra attributes: `{"margin": ambiguity_margin_float}`.

Note: There is also a `WasherAmbiguitySensor` sensor in `sensor.py` (measurement %, debug-gated). These are two separate entities on two platforms, both gated by `CONF_EXPOSE_DEBUG_ENTITIES`. The binary sensor reports the boolean flag; the sensor reports the margin percentage.

---

### 5.3 Select Platform (`select.py`)

| Class | unique_id suffix | Translation key | Options |
|-------|-----------------|-----------------|---------|
| `WashDataProgramSelect` | `_program_select` | `program_select` | `["auto_detect"] + sorted(profile_names)` |

Selecting `"auto_detect"` calls `manager.clear_manual_program()`; selecting any profile name calls `manager.set_manual_program(option)`. Options list refreshes on every `SIGNAL_WASHER_UPDATE`. Icon varies by device_type.

---

### 5.4 Button Platform (`button.py`)

| Class | unique_id suffix | Translation key | Icon | Availability |
|-------|-----------------|-----------------|------|--------------|
| `WashDataTerminateButton` | `_force_end` | `force_end_cycle` | `mdi:stop-circle-outline` | Always available |
| `WashDataPauseCycleButton` | `_pause_cycle` | `pause_cycle` | `mdi:pause-circle-outline` | Active cycle (RUNNING/STARTING/PAUSED/ENDING) and NOT user-paused |
| `WashDataResumeCycleButton` | `_resume_cycle` | `resume_cycle` | `mdi:play-circle-outline` | Only when `is_user_paused` |
| `WashDataRecordStartButton` | `_record_start` | `record_start` | `mdi:record-circle-outline` | Not recording AND detector.state == "off" |
| `WashDataRecordStopButton` | `_record_stop` | `record_stop` | `mdi:stop-circle` | Only when `recorder.is_recording` |

---

## 6. Registered Services (`__init__.py:392-998`)

All services registered once per HA instance (guarded with `hass.services.has_service`). All require `device_id` (resolved to config entry via device registry) except `submit_cycle_feedback` which accepts `device_id` OR `entry_id`.

### `label_cycle` (`__init__.py:394-430`)

**Description:** Assign an existing profile to a past cycle, or remove the label.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `device_id` | string | yes | — |
| `cycle_id` | string | yes | — |
| `profile_name` | string | no | — (empty = remove label) |

**Action:** Calls `profile_store.assign_profile_to_cycle(cycle_id, profile_name_or_None)`. Raises `ServiceValidationError` with key `assign_profile_failed` on ValueError.

---

### `create_profile` (`__init__.py:433-464`)

**Description:** Create a new profile (standalone or based on a reference cycle).

| Param | Type | Required |
|-------|------|----------|
| `device_id` | string | yes |
| `profile_name` | string | yes |
| `reference_cycle_id` | string | no |

**Action:** Calls `profile_store.create_profile_standalone(profile_name, reference_cycle_id)`. Raises `ServiceValidationError(create_profile_failed)` on ValueError.

---

### `delete_profile` (`__init__.py:467-489`)

**Description:** Delete a profile and optionally unlabel cycles using it.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `device_id` | string | yes | — |
| `profile_name` | string | yes | — |
| `unlabel_cycles` | bool | no | `true` |

**Action:** Calls `profile_store.delete_profile(profile_name, unlabel_cycles)`.

---

### `auto_label_cycles` (`__init__.py:492-523`)

**Description:** Retroactively label unlabeled cycles using profile matching.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `device_id` | string | yes | — |
| `confidence_threshold` | float | no | `0.75` (services.yaml says 0.70) |

**Action:** Calls `profile_store.auto_label_cycles(confidence_threshold)`. Logs stats (labeled/skipped counts).

Note: `__init__.py:498` uses default `0.75`; `services.yaml:89` shows `0.70`. Minor discrepancy.

---

### `trim_cycle` (`__init__.py:526-584`)

**Description:** Trim the power data of a past cycle to a specific time window. Offsets renormalized to start from 0; cycle duration updated.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `device_id` | string | yes | — |
| `cycle_id` | string | yes | — |
| `trim_start_s` | float | no | `0.0` (clamped ≥ 0) |
| `trim_end_s` | float | no | full cycle duration |

**Action:** Validates cycle exists via `store.get_cycle_power_data`. Calls `store.trim_cycle_power_data(cycle_id, trim_start_s, trim_end_s)`. Raises on device_not_found, no_config_entry, integration_not_loaded, cycle_not_found_or_no_power, trim_invalid_range, trim_failed_empty_window.

---

### `export_config` (`__init__.py:715-792`)

**Description:** Export device's profiles and cycles to a JSON file.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `device_id` | string | yes | — |
| `path` | string | no | `/config/ha_washdata_export_{entry_id}.json` |

**Action:** Calls `profile_store.export_data(entry_data, entry_options)` then writes JSON (executor-offloaded). Path must be in HA-allowed dirs (`hass.config.is_allowed_path`). Custom path uses exclusive-create (`"x"` mode) to prevent overwrite. Raises path_not_allowed, export_path_exists, export_write_failed.

---

### `import_config` (`__init__.py:795-870`)

**Description:** Import profiles and cycles for a device from a JSON export file.

| Param | Type | Required |
|-------|------|----------|
| `device_id` | string | yes |
| `path` | string | yes |

**Action:** Reads and parses JSON (executor-offloaded), calls `profile_store.async_import_data(payload)`. Applies returned `entry_data`/`entry_options` updates to config entry (only `min_power`/`off_delay` from data; all from options). Path must be in HA-allowed dirs.

---

### `submit_cycle_feedback` (`__init__.py:650-712`)

**Description:** Confirm or correct an auto-detected program after a completed cycle.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `device_id` | string | no (one of) | Preferred |
| `entry_id` | string | no (one of) | Alternative |
| `cycle_id` | string | yes | — |
| `user_confirmed` | bool | yes | True = detected program is correct |
| `corrected_profile` | string | no | Correct profile if not confirmed |
| `corrected_duration` | int | no | Corrected duration in seconds |
| `notes` | string | no | — |
| `dismiss` | bool | no | Implicit default |

**Action:** Calls `learning_manager.async_submit_cycle_feedback(...)`. On success, dismisses persistent notification `ha_washdata_feedback_{entry_id}_{cycle_id}` (best-effort).

---

### `record_start` (`__init__.py:873-887`)

**Description:** Start manually recording a clean cycle (bypasses all matching logic).

| Param | Type | Required |
|-------|------|----------|
| `device_id` | string | yes |

**Action:** Calls `manager.async_start_recording()`.

---

### `record_stop` (`__init__.py:889-903`)

**Description:** Stop manual recording.

| Param | Type | Required |
|-------|------|----------|
| `device_id` | string | yes |

**Action:** Calls `manager.async_stop_recording()`.

---

### `pause_cycle` (`__init__.py:930-963`)

**Description:** Pause the active cycle so a power drop won't finalize it.

| Param | Type | Required |
|-------|------|----------|
| `device_id` | string | yes |

**Action:** Calls `manager.async_pause_cycle()`. Raises `ServiceValidationError(no_active_cycle)` on failure.

---

### `resume_cycle` (`__init__.py:965-998`)

**Description:** Resume a previously paused cycle.

| Param | Type | Required |
|-------|------|----------|
| `device_id` | string | yes |

**Action:** Calls `manager.async_resume_cycle()`. Raises `ServiceValidationError(no_active_cycle)` on failure.

---

### `trigger_ml_training` (`__init__.py:908-927`)

**Description:** Manually retrain on-device ML models from this device's own labelled cycles.

| Param | Type | Required |
|-------|------|----------|
| `device_id` | string | yes |

**Gate:** Only registered when `ENABLE_ML_TRAINING = True` (`const.py:745`) AND `not hass.services.has_service(DOMAIN, SERVICE_TRIGGER_ML_TRAINING)`.

**Action:** Calls `manager.async_run_ml_training(force=True)`.

---

### Services summary table

| Service | `services.yaml` | `strings.json` required | Gated |
|---------|-----------------|------------------------|-------|
| `label_cycle` | yes | yes | no |
| `create_profile` | yes | yes | no |
| `delete_profile` | yes | yes | no |
| `auto_label_cycles` | yes | yes | no |
| `trim_cycle` | yes | yes | no |
| `export_config` | yes | yes | no |
| `import_config` | yes | yes | no |
| `submit_cycle_feedback` | yes (as `submit_cycle_feedback`) | yes | no |
| `record_start` | yes | yes | no |
| `record_stop` | yes | yes | no |
| `pause_cycle` | yes | yes | no |
| `resume_cycle` | yes | yes | no |
| `trigger_ml_training` | yes | yes | `ENABLE_ML_TRAINING` flag |

---

## 7. Assist / Conversation Intents (`intents.py`)

### Intent: `HaWashdataStatus`

Constant: `INTENT_STATUS = "HaWashdataStatus"` (`intents.py:97`)

**Handler class:** `WashDataStatusIntentHandler`
**Slot:** `name` (optional, `cv.string`) — appliance name to disambiguate.
**Registration:** `intent.async_register(hass, WashDataStatusIntentHandler())` called once per HA instance from `async_setup_intents` (guarded by `hass.data["ha_washdata_intents_registered"]`).

**Response behaviour:**
- Iterates all loaded `WashDataManager` instances via `_iter_managers`.
- If `name` slot provided: exact → word → substring match on entry title; falls back to `unknown_device` template.
- If single device: describes it directly.
- If multiple devices, none named: summarizes running devices; `none_running` if none.

**Active states** (reports as running): `running`, `starting`, `ending`, `paused`, `user_paused`, `rinse`, `anti_wrinkle`

**Finished states**: `finished`, `clean`

**Recent finish window**: 720 minutes (12 hours) — within this window, an idle/off device still reports "finished N minutes ago".

**Response templates** (English defaults in `DEFAULT_TEMPLATES`):
- `running_with_estimate`: "Your {device} is still running. About {minutes} minutes left."
- `running_no_estimate`: "Your {device} is still running."
- `finished_recently`: "Your {device} finished {minutes} minutes ago."
- `just_finished`: "Your {device} just finished."
- `not_running`: "Your {device} is not running."
- `no_devices`, `unknown_device`, `none_running`, `error`

**Localization:** Templates loaded from `translations/intent/{lang}.json` (cached, sanitized path, not validated by hassfest). Overlaid in order: `en` → base subtag → full tag.

**Trigger sentences:** NOT auto-registered. Users copy `docs/custom_sentences/en/ha_washdata.yaml` to `<config>/custom_sentences/en/ha_washdata.yaml`. Example triggers:
- "is my {name} done"
- "is the {name} finished"
- "how long until the {name} finishes"
- "is the laundry done"

---

## 8. Recorder (`recorder.py`)

### `CycleRecorder`

**Storage key:** `ha_washdata.recorder.{entry_id}` (`STORAGE_KEY_RECORDER` = `ha_washdata.recorder`)
**Persistence:** `RecorderStore` (HA `Store`, version `STORAGE_VERSION`).

**State:**
- `is_recording: bool`
- `start_time: datetime | None`
- `_buffer: list[tuple[str, float]]` — `(iso_timestamp, power_watts)`
- `_last_run: dict | None` — last completed recording

**Public API:**
- `async_load()` — loads from storage; validates: if `is_recording=True` but no valid start_time, resets to not-recording.
- `start_recording()` — sets `is_recording=True`, sets `start_time`, clears buffer, saves.
- `stop_recording()` → `dict` — returns `{start_time, end_time, data}`, saves last_run, clears active state.
- `process_reading(power: float)` — appends `(now.isoformat(), power)` to buffer; auto-saves every 60 s.
- `clear_last_run()` — clears `_last_run` and saves.

**Properties:** `is_recording`, `start_time`, `current_duration` (seconds since start).

**Integration:** Buffer flow: `manager.async_start_recording()` → `recorder.start_recording()`; each `async_handle_power_change` calls `recorder.process_reading(power)` while active; `manager.async_stop_recording()` → `recorder.stop_recording()` → result fed to `profile_store`.

---

## 9. Setup Advisor (`setup_advisor.py`)

Pure Python (no HA imports), testable in isolation. Computes the current "adoption phase" for a device.

### `compute_setup_phase(...)` → `SetupPhaseResult`

**Inputs:**
- `device_type: str`
- `profile_names: list[str]`
- `past_cycles: list[dict]`
- `ref_profile_names: set[str]` — profiles with reference/store cycles
- `coverage_gap: dict | None` — from `profile_store.suggest_coverage_gaps()`
- `suggestions: list[dict]` — actionable suggestions
- `profile_groups: list[dict]`
- `skipped_steps: dict[str, str | None]` — `step_key -> "never" | ISO_timestamp | None`
- `now: datetime`

**Phase logic (priority order):**

| Phase | Trigger condition | CTA |
|-------|-----------------|-----|
| `phase0` | No real profiles AND no store profiles | Start recording / label detected cycle |
| `phase1c` | Has store profiles but no self cycles yet | View profiles |
| `phase2` | Coverage gap active (not suppressed) | Create from cluster / create profile |
| `phase3` | Pending tuning item (suggestions or profile groups, not suppressed) | Review suggestions / organise profiles |
| `phase1b` | Has real profiles, no coverage gap, has recorded cycles | Record more / browse cycles |
| `phase1a` | Has real profiles, no coverage gap, no recorded cycles | Record / browse cycles |
| `phase4` | Healthy — all steps done or gap not yet triggered | — (dismissible) |

**Suppression:** Each step can be permanently suppressed (`"never"`) or snoozed (ISO timestamp); `_is_step_suppressed` checks both.

**`SetupPhaseResult` fields:** `phase`, `message_key`, `message_params`, `cta_label_key`, `cta_action`, `secondary_label_key`, `secondary_action`, `skippable`, `dismissible`, `step_key`

---

## 10. Diagnostics (`diagnostics.py`)

Implements `async_get_config_entry_diagnostics`. Returned dict:

```
{
  "entry": redacted_entry_as_dict,
  "manager_state": {
    "current_state", "current_program", "time_remaining", "cycle_progress",
    "sample_interval_stats", "profile_sample_repair_stats", "suggestions",
    "feature_flags": {auto_maintenance, save_debug_traces, notify_fire_events}
  },
  "store_export": redacted_profile_store_export,
  "live_diagnostics": diag_buffer.redacted_snapshot()
}
```

**Redacted keys** (set to `"**REDACTED**"`): `auth`, `entry_id`, `flow_id`, `flow_title`, `handler`, `name`, `source`, `title`, `unique_id`, `user_id`, `refresh_token`, `id_token`, `uid`, `notify_service`, `notify_start_services`, `notify_finish_services`, `notify_live_services`, `notify_people`, `notify_actions`, `power_sensor`, `external_end_trigger`, `door_sensor_entity`, `switch_entity`, `energy_price_entity`.

---

## 11. DiagBuffer (`diag_buffer.py`)

Per-device in-memory rolling 24-hour ring buffer. **No disk writes.**

**Three independent buffers:**
- `power_trace` — raw power readings: `deque(maxlen=100_000)` of `(unix_ts, watts)`
- `state_history` — detector state transitions: `deque(maxlen=2_000)` of `(unix_ts, from, to, program)`
- `logs` — log records matching device tag: `deque(maxlen=5_000)` of `(created_float, levelname, msg)`

**Log filtering:** `_LogHandler` installed on `custom_components.ha_washdata` logger; keeps only records whose `getMessage()` contains `[{device_name}]` (prefix from `DeviceLoggerAdapter`).

**Methods:**
- `record_power(watts, ts)` — appends to power buffer
- `record_state(from, to, program, ts)` — appends to state buffer
- `snapshot()` → full 24h window (includes msg in logs)
- `redacted_snapshot()` → same but `device_name` removed, `msg` stripped from log entries
- `uninstall()` — removes log handler; **must** be called from `manager.async_shutdown()` to avoid handler accumulation across reloads

---

## 12. Log Utils (`log_utils.py`)

### `DeviceLoggerAdapter`

Wraps a `logging.Logger`, prepends `[{device_name}] ` to all messages. Used throughout `manager.py` and `profile_store.py`. The `[device_name]` prefix is also the filter tag used by `DiagBuffer._LogHandler`.

---

## 13. Entry Point (`__init__.py`) — Additional Details

### `async_setup_entry` flow summary (`__init__.py:351-1000`)

1. Duplicate-setup guard (`entry.entry_id` already in `hass.data[DOMAIN]`).
2. Remove deprecated `auto_maintenance` switch entity from entity registry (migration cleanup).
3. Instantiate `WashDataManager`, call `manager.async_setup()`.
4. Call `_migrate_online_to_global` (hoist per-device online-features flag to integration-wide store; best-effort, never aborts setup).
5. `async_forward_entry_setups` for SENSOR, BINARY_SENSOR, SELECT, BUTTON.
6. `_apply_device_link` — sync `via_device_id` in device registry from `CONF_LINKED_DEVICE` option.
7. Register `async_reload_entry` as update listener.
8. Register frontend card (`WashDataCardRegistration`) — once per HA instance; state machine: registering → registered OR deferred.
9. Register sidebar panel (`async_register_panel`) — once per HA instance.
10. Register WebSocket API commands (`async_register_commands`) — **re-registered on every setup/reload** (idempotent, enables new commands without full HA restart).
11. Load panel config (`async_load_panel_config`) and account store (`store_account.async_load`).
12. Register intents (`async_setup_intents`) — once per HA instance.
13. Register all services (each guarded with `has_service` check).

### `async_reload_entry`

If manager found: calls `manager.async_reload_config(entry)` (in-place, no entity recreation) and re-applies device link. If no manager: full unload + setup.

### `async_unload_entry`

Unloads platforms, pops manager, calls `manager.async_shutdown()`. When last entry removed: calls `async_unregister_panel`.

### `_apply_device_link`

Syncs `via_device_id` on the WashData device in the device registry. Stale targets (device no longer exists) are treated as no-link.

---

## 14. Key Constants Referenced

| Constant | Location | Value |
|----------|----------|-------|
| `DOMAIN` | `const.py` | `"ha_washdata"` |
| `STORAGE_VERSION` | `const.py:713` | `11` |
| `STORAGE_KEY` | `const.py:714` | `"ha_washdata"` |
| `DEFAULT_DEVICE_TYPE` | `const.py:227` | `"washing_machine"` |
| `DEVICE_TYPE_OTHER` | `const.py:518` | `"other"` |
| `DEVICE_TYPE_GENERIC` | `const.py:513` | `"generic"` |
| `DEVICE_TYPE_PUMP` | `const.py:509` | `"pump"` |
| `ENABLE_ML_TRAINING` | `const.py:745` | `True` |
| `SERVICE_TRIGGER_ML_TRAINING` | `const.py:847` | `"trigger_ml_training"` |
| `SERVICE_SUBMIT_FEEDBACK` | `const.py:725` | `"ha_washdata.submit_cycle_feedback"` |
| `CONF_EXPOSE_DEBUG_ENTITIES` | `const.py:98` | `"expose_debug_entities"` |
| `CONF_LINKED_DEVICE` | `const.py:189` | `"linked_device"` |
| `SIGNAL_WASHER_UPDATE` | `const.py` | `"washdata_update_{}"` |

---

## 15. CLAUDE.md vs. Code Discrepancies

1. **`auto_label_cycles` confidence default**: CLAUDE.md does not specify the default. `__init__.py:498` uses `0.75` as Python default; `services.yaml:89` shows `0.70`. The Python handler default wins when the service is called without the param; the services.yaml default is displayed in the UI.

2. **Config-entry version**: CLAUDE.md says "config schema v1→3.6"; code is actually at 3.7 (minor_version=7). The 3.6→3.7 step (removing `initial_profile` stub) is not described in CLAUDE.md.

3. **Service `record_stop`**: CLAUDE.md lists only `record_start`/`record_stop` (not `record_stop` explicitly in some descriptions). Both are present in both `__init__.py` and `services.yaml`.

4. **`WasherAmbiguitySensor` exists on two platforms**: `sensor.py` (measurement %, debug-gated) and `binary_sensor.py` (bool, debug-gated). The CLAUDE.md mentions the sensor but not the binary sensor by name. Both share the unique_id suffix `_ambiguity` on their respective platforms.

5. **`setup_advisor.py` labelled as `setup_advisor.py` in CLAUDE.md but described as "setup guidance"**: The file is purely phase-computation logic, no HA imports. Not mentioned in architecture section of CLAUDE.md.

6. **`diag_buffer.py` / `diagnostics.py`**: Not covered in the architecture section of CLAUDE.md.
