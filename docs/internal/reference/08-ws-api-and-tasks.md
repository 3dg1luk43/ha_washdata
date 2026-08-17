# WashData WebSocket API — Engineering Reference

**Files covered:**
- `custom_components/ha_washdata/ws_api.py` (5988 lines) — all WS command handlers + RBAC
- `custom_components/ha_washdata/ws_schema.py` (1024 lines) — typed request/response contract
- `custom_components/ha_washdata/frontend.py` (400 lines) — panel + static path registration
- `custom_components/ha_washdata/task_registry.py` (288 lines) — background-task lifecycle

> Content-corrected 2026-08-17 for 0.5.4 (command inventory, counts, task-kind list). Line anchors below may still reflect the original 2026-07-18 / 0.5.1 pass.

---

## 1. Overview

The integration exposes **106 WebSocket commands** under the `ha_washdata/` namespace, all registered via `ws_api.async_register_commands()`. The count of 106 comes from the `WS_COMMANDS` dict in `ws_schema.py`, and it is confirmed to match exactly by `tests/test_ws_contract.py` (which asserts one-to-one correspondence between commands in `ws_api.py` and keys in `WS_COMMANDS`).

All commands use the wire format `{"type": "ha_washdata/<command>", ...params}`.

---

## 2. Registration Lifecycle

### `async_register_commands` (`ws_api.py:1025-1106`)

Called from `__init__.async_setup_entry` on **every integration load/reload**:

```python
# __init__.py:624-637
# Register WebSocket API commands for the panel. Re-run on every setup/reload:
# HA's async_register_command overwrites the handler per command type, so this
# is idempotent AND means NEW commands become available after an integration
# reload, not only after a full Home Assistant restart (previously the
# once-per-instance guard forced a full restart for any newly-added command).
async_register_commands(hass)
hass.data["ha_washdata_ws_registered"] = True
```

**Key points:**
- `websocket_api.async_register_command` overwrites any prior handler for the same command string, making re-registration safe and idempotent.
- A new command added to the code becomes available **after an integration reload** (not a full HA restart). This replaced an old once-per-instance guard.
- Every handler is wrapped by `_guard(handler)` at registration time (`ws_api.py:1106`), injecting RBAC checks transparently via `functools.wraps`.
- `async_load_panel_config(hass)` is called before command registration to load the panel-config + RBAC store from disk (`ws_api.py:634`). It self-guards against double-load.

### `@async_response` vs `@callback`

HA's WS framework requires one of two decorators on every handler:

- `@websocket_api.async_response` — for **async** handlers (awaits I/O, executor calls, or stores). The framework spawns a task for the handler coroutine. **Required** for any handler that does `await`; omitting it causes the handler to silently hang.
- `@callback` — for **synchronous** handlers. Runs immediately on the event loop without spawning a task. Used only for pure in-memory reads or commands that immediately kick off a detached background task and return a task_id.

Both decorators carry through `functools.wraps` in `_guard`, so the WS framework sees the original attributes.

---

## 3. RBAC / Access Control (`ws_api.py:555-639`)

### Access Levels

Four levels in ascending order: `none` / `read` / `edit` / `full`.

- When RBAC is **disabled** (default): every authenticated user resolves to `"full"`.
- HA admins always resolve to `"full"` regardless of RBAC config.
- Non-admins resolve via per-user, per-device overrides in the panel RBAC store (`set_panel_config`).

### Command Classification

| Set | Required level | Commands |
|---|---|---|
| `_OPEN_COMMANDS` | any authenticated user | `get_constants`, `get_panel_config`, `set_user_prefs` |
| `_ADMIN_COMMANDS` | HA admin always (even when RBAC disabled) | `set_panel_config`, `get_logs`, `wipe_history`, `import_config`, `export_config`, `reprocess_history`, `clear_debug_data`, `store_connect`, `store_disconnect`, `store_set_online`, `store_set_prefs` |
| `_FULL_COMMANDS` | `full` access | `wipe_history`, `import_config`, `export_config`, `clear_debug_data`, `reprocess_history`, `trigger_ml_training`, `revert_matching_config`, `revert_ml_models` |
| `_READ_WRITE_COMMANDS` | `read` access | `set_program`, `run_playground_*`, `list_tasks`, `subscribe_tasks`, `cancel_task`, `get_task_result`, `start_playground_*`, `store_status`, `store_search_devices`, `store_list_brands`, `store_get_*` |
| `get_*` (by name prefix) | `read` | all query commands not in the above sets |
| everything else | `edit` | all mutation commands not listed above |

### RBAC Guard

`_rbac_ok()` (`ws_api.py:576-624`) runs before every handler via the `_guard` wrapper. Sends `"unauthorized"` / `"forbidden"` error and returns `False` without calling the handler. For task commands with no `entry_id` under RBAC, it resolves the entry_id from the task registry to authorize correctly.

### Contract Debug Mode (`ws_api.py:90-148`)

`_send_result()` wraps `connection.send_result()`. When `_WS_CONTRACT_CHECK=True` (env var `HA_WASHDATA_WS_CONTRACT=1` or test flag), it validates the response payload against the `TypedDict` in `WS_RESPONSE_TYPES` and logs any mismatches. No-op in production.

---

## 4. WebSocket Commands by Feature Area

### 4.1 Devices

| Command | Line | Params (R=required, O=optional) | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_devices` | 1111 | — | `@callback` | `{devices: [DeviceInfo]}` | RBAC-filtered list; includes live state (power, progress, program) |
| `get_device_cycles` | 1198 | `entry_id` R, `limit` O (1-200, default 50), `offset` O (default 0) | `@callback` | `{entry_id, cycles, reference_cycles, total, has_more}` | Paginated, most-recent-first; `reference_cycles` on page 0 only; strips `power_data/power_trace/debug_data/samples` |

### 4.2 Settings / Options

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_options` | 1267 | `entry_id` R | `@callback` | `{options}` | Merges `entry.data + entry.options` |
| `set_options` | 1285 | `entry_id` R, `options` R (dict) | `@async_response` | `{success}` | Normalizes cleared selectors (external_end_trigger, door_sensor, linked_device, switch_entity) to `None`; drops pump-only keys for non-pump devices; diffs changes into a settings changelog; triggers reload via `async_update_entry` |
| `get_settings_changelog` | 1388 | `entry_id` R | `@async_response` | `{changelog}` | Returns per-key history of changed option values |
| `get_setup_status` | 1420 | `entry_id` R | `@async_response` | `{phase, message_key, ...}` | Adoption-phase status for the setup guidance card |

### 4.3 Profiles

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_profiles` | 1495 | `entry_id` R | `@async_response` | `{profiles, profile_health, profile_trends, coverage_gaps, profile_advisories}` | Executor-offloads `compute_profile_health`, `compute_profile_trends`, `suggest_coverage_gaps`, `compute_profile_advisories` |
| `create_profile` | 1554 | `entry_id` R, `name` R, `reference_cycle` O, `manual_duration_min` O | `@async_response` | `{success, name}` | Creates from existing cycle or blank |
| `rename_profile` | 1599 | `entry_id` R, `profile_name` R, `new_name` R, `manual_duration_min` O | `@async_response` | `{success}` | Also updates optional duration |
| `delete_profile` | 1641 | `entry_id` R, `profile_name` R, `unlabel_cycles` O | `@async_response` | `{success}` | Optionally clears labels on linked cycles |
| `rebuild_envelopes` | 1798 | `entry_id` R | `@callback` | `{task_id}` | **Background task** (`kind="rebuild"`); rebuilds one profile envelope per step in executor; serialized under per-entry write lock |
| `get_profile_phases` | 1864 | `entry_id` R, `profile_name` R | `@callback` | `{phases}` | Phase ranges for a profile |
| `set_profile_phases` | 1893 | `entry_id` R, `profile_name` R, `phases` R (list) | `@async_response` | `{success}` | Saves phase ranges into the profile store |
| `get_profile_envelope` | 3503 | `entry_id` R, `profile_name` R | `@callback` | `{envelope}` | Returns avg/min/max/target_duration/avg_energy/cycle_count |
| `get_profile_cycles` | 3542 | `entry_id` R, `profile_name` R, `limit` O | `@async_response` | `{cycles}` | Cycles tagged to a profile; artifacts computed on demand for older cycles |

### 4.4 Profile Groups (Stage 5)

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_profile_groups` | 1674 | `entry_id` R | `@async_response` | `{groups, min_cohesion, suggestions}` | Cohesion computed per-group in executor; suggestions via `suggest_profile_groups` |
| `save_profile_group` | 1713 | `entry_id` R, `name` R, `members` R (list[str]) | `@async_response` | `{success}` | Creates or updates a group |
| `rename_profile_group` | 1745 | `entry_id` R, `name` R, `new_name` R | `@async_response` | `{success}` | |
| `delete_profile_group` | 1772 | `entry_id` R, `name` R | `@async_response` | `{success}` | |

### 4.5 Maintenance Log

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_maintenance_log` | 1925 | `entry_id` R | `@async_response` | `{log, due, event_types, reminders}` | Includes reminder config from device options |
| `add_maintenance_event` | 1952 | `entry_id` R, `event_type` R, `date` O, `notes` O | `@async_response` | `{success, event}` | Requires `edit` access |
| `delete_maintenance_event` | 1985 | `entry_id` R, `event_id` R | `@async_response` | `{success}` | Requires `edit` access |

### 4.6 Cycles

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `label_cycle` | 2015 | `entry_id` R, `cycle_id` R, `profile_name` O, `new_profile_name` O | `@async_response` | `{success}` | Can create a new profile on the fly via `new_profile_name` |
| `delete_cycle` | 2061 | `entry_id` R, `cycle_id` R | `@async_response` | `{success}` | |
| `auto_label_cycles` | 2089 | `entry_id` R, `confidence_threshold` O | `@async_response` | `{success}` | Bulk-labels unmatched cycles above threshold |
| `get_cycle_power_data` | 3125 | `entry_id` R, `cycle_id` R | `@async_response` | `{cycle_id, samples, full_duration_s, ...metadata}` | Downsampled to 240 pts; computes artifacts on-demand for older cycles (executor-offloaded) |
| `trim_cycle` | 3196 | `entry_id` R, `cycle_id` R, `start_s` R, `end_s` R | `@callback` | `{task_id}` | **Background task** (`kind="trim"`); serialized under per-entry write lock |
| `analyze_split` | 3262 | `entry_id` R, `cycle_id` R, `gap_seconds` O (30-21600, default 900) | `@async_response` | `{segments, split_offsets, samples, full_duration_s}` | Executor-offloads gap detection |
| `apply_split` | 3316 | `entry_id` R, `cycle_id` R, `split_offsets` R (list[float]), `segment_profiles` O | `@callback` | `{task_id}` | **Background task** (`kind="split"`); fast validation synchronously, heavy apply as task |
| `apply_merge` | 3411 | `entry_id` R, `cycle_ids` R (list[str]), `target_profile` O, `new_profile_name` O | `@callback` | `{task_id}` | **Background task** (`kind="merge"`); serialized under per-entry write lock |

### 4.7 Phase Catalog

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_phase_catalog` | 2122 | `entry_id` R, `device_type` O | `@callback` | `{phases, device_type}` | |
| `create_phase` | 2152 | `entry_id` R, `device_type` R, `name` R, `description` O | `@async_response` | `{success}` | |
| `update_phase` | 2185 | `entry_id` R, `phase_id` R, `new_name` R, `description` O | `@async_response` | `{success}` | |
| `delete_phase` | 2218 | `entry_id` R, `phase_id` R | `@async_response` | `{success}` | |

### 4.8 Recording

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_recording_state` | 2247 | `entry_id` R | `@callback` | `{state, duration_s?, sample_count?, start_time?, end_time?}` | State is one of: idle / recording / ready |
| `start_recording` | 2296 | `entry_id` R | `@async_response` | `{success}` | |
| `stop_recording` | 2319 | `entry_id` R | `@async_response` | `{success}` | |
| `process_recording` | 2342 | `entry_id` R, `profile_name` R, `save_mode` R (`new_profile`/`existing_profile`), `head_trim` O, `tail_trim` O | `@async_response` | `{success}` | Serialized under per-entry write lock; heavy persist |
| `discard_recording` | 2504 | `entry_id` R | `@async_response` | `{success}` | |

### 4.9 Feedbacks

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_feedbacks` | 2531 | `entry_id` R | `@callback` | `{feedbacks}` | |
| `resolve_feedback` | 2561 | `entry_id` R, `cycle_id` R, `action` R (`confirm`/`correct`/`ignore`/`delete`), `corrected_profile` O, `corrected_duration_min` O | `@async_response` | `{success}` | |
| `dismiss_all_feedbacks` | 2613 | `entry_id` R | `@async_response` | `{success, dismissed}` | |

### 4.10 Diagnostics / Maintenance

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_diagnostics` | 2649 | `entry_id` R | `@async_response` | `{stats}` | Storage statistics |
| `reprocess_history` | 2748 | `entry_id` R | `@callback` | `{task_id}` | **Background task** (`kind="reprocess"`); 5-phase sequence: rematch -> backfill golden -> refresh suggestions -> ML training (if `ENABLE_ML_TRAINING`) -> recompute health; serialized under write lock; admin-only |
| `clear_debug_data` | 2771 | `entry_id` R | `@async_response` | `{success, count}` | Clears stored debug traces; admin-only |
| `wipe_history` | 2794 | `entry_id` R | `@async_response` | `{success}` | Destructive: clears all cycles + profiles; admin-only |
| `export_config` | 2818 | `entry_id` R | `@async_response` | `{json_data}` | Executor-offloads JSON serialization; admin-only |
| `import_config` | 2849 | `entry_id` R, `json_data` R (str) | `@async_response` | `{success}` | Serialized under write lock; admin-only |

### 4.11 Shared Constants

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_constants` | 2918 | — | `@callback` | `{device_types, state_colors, ml_lab_enabled, ml_suggestions_enabled, ml_training_available, PROFILE_MIN_WARMUP_CYCLES, store_online_available, store_online_enabled, store_web_origin}` | Open to all authenticated users; surfaced `SHOW_ML_LAB`, `ENABLE_ML_SUGGESTIONS`, `ENABLE_ML_TRAINING` flags |

### 4.12 Suggestions

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_suggestions` | 2959 | `entry_id` R | `@callback` | `{suggestions}` | Filtered to `_SUGGESTION_KEYS` |
| `apply_suggestions` | 3015 | `entry_id` R, `keys` R (list[str]) | `@async_response` | `{success, applied}` | Only keys in `_SUGGESTION_KEYS` are accepted; coerces int types; triggers reload |
| `clear_suggestions` | 3072 | `entry_id` R | `@async_response` | `{success}` | |
| `run_suggestion_analysis` | 3096 | `entry_id` R | `@async_response` | `{success, count?}` | Runs the full suggestion analysis pass |

### 4.13 Panel Config + RBAC

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_panel_config` | 3600 | — | `@async_response` | `{panel, is_admin, user, prefs, rbac?, users?}` | `rbac` and `users` only returned to admins; open to all authenticated users |
| `set_panel_config` | 3640 | `panel` O (dict), `rbac` O (dict) | `@async_response` | `{success}` | Admin-only; sanitized via `_sanitize_panel` / `_sanitize_rbac` before persist |
| `set_user_prefs` | 3670 | `prefs` R (dict) | `@async_response` | `{success}` | Per-user prefs: `date_format` (`relative`/`absolute`), `lang_override` (BCP-47 tag or empty string); open to all authenticated users |

### 4.14 Live Monitoring

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_match_debug` | 3725 | `entry_id` R | `@callback` | `{confidence, ambiguous, candidates}` | Live match debug from running detector |
| `get_power_history` | 3796 | `entry_id` R, `with_raw` O | `@async_response` | `{cycle_active, cycle_elapsed_s, live, raw?, restart_gaps, cycle_start_iso?}` | Raw only if `with_raw=True`; uses recorder for history |
| `set_program` | 3762 | `entry_id` R, `program` O (str\|null) | `@callback` | `{success}` | Override the live program selection; read-level (gated in `_READ_WRITE_COMMANDS`) |
| `get_logs` | 3859 | `level` O, `limit` O (1-500, default 200) | `@callback` | `{logs}` | In-memory ring buffer of last 500 `ha_washdata` logger records; admin-only |

### 4.15 ML Lab

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `get_ml_comparison` | 4350 | `entry_id` R | `@async_response` | `{enabled, cycles, settings_comparison, cycle_count, evaluated_count, model_source, profile_stats, ml_suggestions_enabled, error?}` | Executor-offloads ML model evaluation against cycle history; updates per-cycle `ml_health` if needed; includes Classic-vs-ML settings comparison when `ENABLE_ML_SUGGESTIONS` |
| `get_ml_training_status` | 4410 | `entry_id` R | `@async_response` | `{available, enabled, running, last_trained, cycle_count, min_cycles, interval_days, hour, on_device_models, matching}` | Includes per-capability fit-trend from `ml_training_history`; gated by `ENABLE_ML_TRAINING` for `available` field |
| `trigger_ml_training` | 4574 | `entry_id` R | `@callback` | `{task_id}` | **Background task** (`kind="ml_training"`); gated — returns `"not_available"` if `ENABLE_ML_TRAINING` is False; `full`-level RBAC; admin-only |
| `revert_matching_config` | 4601 | `entry_id` R | `@async_response` | `{success}` | Drops on-device tuned matcher weights; `full`-level RBAC |
| `revert_ml_models` | 4628 | `entry_id` R | `@async_response` | `{success}` | Drops all on-device promoted ML model specs; `full`-level RBAC |
| `set_ml_review` | 4661 | `entry_id` R, `cycle_id` R, `quality` O (`""`/`"bad"`/`"good"`/`"unusable"`), `golden` O, `tags` O, `notes` O | `@async_response` | `{success}` | ML-Lab review write-back; gated on `SHOW_ML_LAB` at UI level only (no server-side flag gate) |

### 4.16 Cycle Controls

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `pause_cycle` | 4706 | `entry_id` R | `@async_response` | `{ok}` | User-pauses the active cycle |
| `resume_cycle` | 4725 | `entry_id` R | `@async_response` | `{ok}` | Resumes a user-paused cycle |
| `terminate_cycle` | 4744 | `entry_id` R | `@async_response` | `{ok}` | Force-terminates the active cycle |

### 4.17 Playground

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `run_playground_cycle_detail` | 4866 | `entry_id` R, `cycle_id` R, `settings_override` O | `@async_response` | `{cycle_id, series, events, alerts, outcome, error?}` | Single-cycle faithful replay; inline (synchronous); superseded by `start_playground_cycle_detail` for long cycles |
| `run_playground_history` | 4906 | `entry_id` R, `cycle_ids` O, `settings_override` O, `concurrency` O | `@async_response` | `{rows, summary, baseline_rows, baseline_summary, diff?}` | Inline executor batch; superseded by `start_playground_history` |
| `run_playground_sweep` | 4950 | `entry_id` R, `param` R, `values` R, `objective` R, `cycle_ids` O, `concurrency` O, `param_y` O, `values_y` O | `@async_response` | `{param, objective, points, current_value, best_value, best_metric?, ...}` or 2D grid | Inline; superseded by `start_playground_sweep` |
| `start_playground_history` | 5271 | `entry_id` R, `cycle_ids` O, `settings_override` O | `@callback` | `{task_id}` | **Background task** (`kind="pg_history"`); chunked executor, progress via registry; read-level RBAC |
| `start_playground_sweep` | 5299 | `entry_id` R, `param` R, `values` R, `objective` R, `param_y` O, `values_y` O | `@callback` | `{task_id}` | **Background task** (`kind="pg_sweep"`); chunked 1D or 2D sweep; read-level RBAC |
| `start_playground_cycle_detail` | 5389 | `entry_id` R, `cycle_id` R, `settings_override` O | `@callback` | `{task_id}` | **Background task** (`kind="pg_detail"`); chunked per-5s replay; read-level RBAC |
| `get_dtw_debug` | 5009 | `entry_id` R, `cycle_id` R, `profile_name` O | `@callback` | `{cycle_id, profile_name, grid_n, stage2, dtw, stage4, warp_path, cycle_trace, profile_trace}` | Full DTW stage scores for a specific cycle+profile pair |

### 4.18 Background Task Registry

| Command | Line | Params | Handler type | Response | Notes |
|---|---|---|---|---|---|
| `list_tasks` | 5057 | `entry_id` O | `@callback` | `{tasks}` | Snapshot of all active + recently-finished tasks (optionally filtered to one device); read-level RBAC |
| `subscribe_tasks` | 5075 | `entry_id` O | `@callback` | `{}` then push events | Sends current snapshot as `{type: "task", task: TaskSnapshot}` events on subscribe, then one event per change; subscription cancelled on socket close; read-level RBAC |
| `cancel_task` | 5110 | `task_id` R | `@callback` | `{cancelled}` | Sets cancel flag; consumer polls `Task.cancel_requested` between chunks; read-level RBAC |
| `get_task_result` | 5127 | `task_id` R | `@callback` | `TaskSnapshot` with `result` field | Reads finished task result (reloadable until evicted); read-level RBAC |

### 4.19 Community Store (Online Features)

All store commands gate on `online_features_enabled(hass)` — returns `{"enabled": False}` or `{"disabled": True}` when the integration-wide online flag is off.

| Command | Line | Params | Handler type | Response | RBAC | Notes |
|---|---|---|---|---|---|---|
| `store_status` | 690 | `entry_id` R | `@async_response` | `StoreStatusResponse` | read | Identity (uid/name/brand/model), enabled/connected |
| `store_connect` | 704 | `entry_id` R, `refresh_token` R, `uid` R, `name` O | `@async_response` | `StoreSimpleResponse` | admin | Sets integration-wide account |
| `store_disconnect` | 719 | `entry_id` R | `@async_response` | `StoreSimpleResponse` | admin | Clears account |
| `store_search_devices` | 730 | `entry_id` R, `query` O, `appliance_type` O, `model_query` O, `include_pending` O | `@async_response` | `StoreItemsResponse` | read | Browse community device catalog |
| `store_list_brands` | 749 | `entry_id` R, `query` O, `include_pending` O | `@async_response` | `StoreItemsResponse` | read | |
| `store_get_device_quality` | 764 | `entry_id` R, `device_id` R | `@async_response` | `StoreQualityResponse` | read | 5-star avg + count |
| `store_get_device_profiles` | 778 | `entry_id` R, `brand` R, `model` R, `appliance_type` R | `@async_response` | `StoreDeviceProfilesResponse` | read | Share dialog picker |
| `store_confirm_device` | 793 | `entry_id` R, `device_id` R | `@async_response` | `StoreConfirmResponse` | edit | Community confirm-count auto-promote (threshold 5) |
| `store_rate_device` | 807 | `entry_id` R, `device_id` R, `rating` R (int) | `@async_response` | `StoreOnlineResponse` | edit | |
| `store_set_online` | 822 | `entry_id` R, `enabled` R | `@async_response` | `StoreOnlineResponse` | admin | Integration-wide online toggle |
| `store_set_prefs` | 837 | `entry_id` R, `prefs` R (dict) | `@async_response` | `StorePrefsResponse` | admin | Integration-wide community prefs |
| `store_get_profiles` | 852 | `entry_id` R, `device_id` R | `@async_response` | `StoreItemsResponse` | read | |
| `store_get_cycles` | 867 | `entry_id` R, `profile_id` R | `@async_response` | `StoreItemsResponse` | read | |
| `store_import_cycle` | 882 | `entry_id` R, `cycle_id` R, `target_profile` O, `new_profile_name` O | `@async_response` | `StoreImportResponse` | edit | Import reference cycle from store |
| `store_upload_cycle` | 901 | `entry_id` R, `local_cycle_id` R, `program` R, `description` O | `@async_response` | `StoreUploadResponse` | edit | Share a local cycle |
| `store_upload_device` | 927 | `entry_id` R, `items` R (list), `include_phases` O, `include_settings` O | `@async_response` | `StoreUploadDeviceResponse` | edit | Share a whole-device bundle |
| `store_download_device` | 964 | `entry_id` R, `device_id` R, `include_settings` O | `@async_response` | `StoreDownloadDeviceResponse` | edit | Adopt a community bundle |
| `get_shareable_cycles` | 1001 | `entry_id` R | `@async_response` | `{items, phase_programs, all_programs}` | edit | Recorded/golden cycles eligible for sharing |

### 4.20 Selective Export/Import, Playground Presets, Suggestion Lock

Eight commands that postdate the 0.5.1 pass and were missing from the inventory above:

| Command | Line | Notes |
|---|---|---|
| `get_export_inventory` | 3009 | Per-category data inventory (profiles/cycles/settings counts) for the selective-export wizard tree |
| `export_config_selective` | 3097 | Exports only the user-chosen subset instead of the whole store |
| `analyze_import` | 3042 | Dry-run: parses an export payload and describes what it contains, no mutation |
| `import_config_selective` | 3141 | Imports only the user-chosen subset from an analyzed payload |
| `get_playground_settings` | 5489 | Live effective Playground settings + saved presets + publishable-key list |
| `save_playground_preset` | 5528 | Persists a named Playground override preset (bounded by `PLAYGROUND_PRESET_MAX`) |
| `delete_playground_preset` | 5567 | Removes a saved Playground preset |
| `set_suggestion_lock` | 3434 | Locks/unlocks a suggestion key so the analyser stops re-proposing it |

---

## 5. Per-Entry Write Lock (`ws_api.py:309-325`)

A per-entry `asyncio.Lock` stored in `hass.data[DOMAIN + "_ws_write_locks"]` serializes all store-mutating handlers for the same device:

- `process_recording` (claim+persist)
- `reprocess_history` (_reprocess_task)
- `import_config` (JSON parse + store rewrite)
- `rebuild_envelopes` (_rebuild_envelopes_task)
- `trim_cycle` (_trim_task)
- `apply_split` (_apply_split_task)
- `apply_merge` (_apply_merge_task)

Prevents interleaving mutations that would corrupt cycle IDs or clobber the store.

---

## 6. Large-Payload Safety (`ws_api.py:151`)

`_CYCLE_STRIP_KEYS = frozenset({"power_data", "power_trace", "debug_data", "samples"})` — stripped from every cycle before WS transmission (enforced by `_strip_cycle()`). Prevents the 32 KB HA WS event limit from being hit.

`_downsample()` (`ws_api.py:225-259`) — reduces `(offset_s, watts)` series to ≤240 points by striding, preserving first and last samples.

---

## 7. Background Task Internals

Nine task kinds run through the registry:

| Kind | Spawned by | Phases / chunks |
|---|---|---|
| `reprocess` | `ws_reprocess_history` | 5 phases (match, golden, suggestions, ML training, health) |
| `rebuild` | `ws_rebuild_envelopes` | 1 step per profile, executor per step |
| `trim` | `ws_trim_cycle` | 1 step (trim + envelope rebuild), write-lock held |
| `split` | `ws_apply_split` | N segment steps, write-lock held |
| `merge` | `ws_apply_merge` | Multi-step, write-lock held |
| `ml_training` | `ws_trigger_ml_training` | Gated on `ENABLE_ML_TRAINING` |
| `pg_history` | `ws_start_playground_history` | `_PG_HISTORY_CHUNK=2` cycles per executor call |
| `pg_sweep` | `ws_start_playground_sweep` | 1 (param, value) point per executor call |
| `pg_detail` | `ws_start_playground_cycle_detail` | `_PG_DETAIL_CHUNK` steps per executor call |

All tasks: check `task.cancel_requested` between chunks; call `reg.update(task, done=N)` for progress; call `reg.finish(task, ...)` when done/cancelled/error. The per-entry write lock may be held across the full task (reprocess, rebuild, trim, split, merge) to prevent interleaved mutations.

---

## 8. `task_registry.py` — API Reference

### `Task` dataclass (`task_registry.py:54-119`)

| Field | Type | Description |
|---|---|---|
| `id` | `str` | 12-char hex UUID |
| `entry_id` | `str` | Config entry this task belongs to |
| `kind` | `str` | `reprocess` / `ml_training` / `pg_history` / `pg_sweep` / `pg_detail` / `rebuild` / `trim` / `split` / `merge` |
| `label` | `str` | English fallback displayed in the header pill |
| `label_key` | `str | None` | i18n key for the panel's `_t()` |
| `label_params` | `dict` | Substitution params for `label_key` |
| `total` | `int` | Total steps (0 = unknown) |
| `done` | `int` | Completed steps |
| `state` | `str` | `running` / `done` / `error` / `cancelled` |
| `error` | `str | None` | Error message if state=error |
| `started_at` | `float` | Unix timestamp |
| `updated_at` | `float` | Unix timestamp of last change |
| `finished_at` | `float | None` | Unix timestamp when state changed to done/error/cancelled |
| `result` | `Any` | Stored payload (only when finished) |
| `_cancelled` | `bool` | Internal cancel flag; read via `cancel_requested` property |

**Methods:**
- `progress() -> float | None` — fraction [0,1] when `total > 0`, else `None`
- `eta_s() -> float | None` — rough remaining seconds from elapsed/progress ratio; `None` when not running or no progress
- `snapshot(include_result=False) -> dict` — JSON-safe view; `include_result=True` embeds the `result` payload

### `TaskRegistry` class (`task_registry.py:122-235`)

| Method | Description |
|---|---|
| `create(entry_id, kind, label, total, *, label_key, label_params) -> Task` | Allocates a task, notifies listeners, evicts old finished tasks |
| `update(task, *, done, total, label, label_key, label_params)` | Updates progress fields, notifies listeners |
| `finish(task, *, state, result, error)` | Sets final state/result, notifies, evicts |
| `cancel(task_id) -> bool` | Sets `_cancelled=True` on a running task; returns `True` if found |
| `get(task_id) -> Task | None` | Direct lookup |
| `snapshot(entry_id=None) -> list[dict]` | All tasks (optionally filtered by device) as snapshots |
| `add_listener(cb) -> unsubscribe_fn` | Registers a sync callback called on any state change |

**Eviction:** `_MAX_FINISHED = 30` finished tasks retained; oldest evicted on each `create` or `finish`. Running tasks are never evicted.

**Singleton:** `get_registry(hass)` (`task_registry.py:238-244`) lazily creates and stores one `TaskRegistry` per `hass` instance in `hass.data[DOMAIN + "_task_registry"]`.

---

## 9. `frontend.py` — Panel Registration

### Static Paths

Registered once per HA instance at `async_setup_entry` time, guarded by `PANEL_STATIC_REGISTERED` flag:

| URL path | Filesystem path | Description |
|---|---|---|
| `/ha_washdata/ha-washdata-panel.js` | `www/ha-washdata-panel.js` | Full-screen panel JS module; served with cache headers |
| `/ha_washdata/panel-translations` | `translations/panel/` | One `{lang}.json` per language; the panel fetches the user's language + `en` fallback on demand |
| `/ha_washdata/ha-washdata-card.js` | `www/ha-washdata-card.js` | Lovelace card; auto-registered as a Lovelace resource |

### Panel Registration (`frontend.py:255-365`)

- `async_register_panel()` is **idempotent** via `hass.data[PANEL_REGISTERED_KEY]` flag.
- Registers a HA built-in `"custom"` component panel at URL path `ha-washdata`, sidebar title `"WashData"`, icon `mdi:washing-machine`, `require_admin=False`.
- Passes `_panel_custom` config with `module_url` including a mtime-based cache-buster (`?v=<mtime>`).
- The static path registration is guarded by a separate `PANEL_STATIC_REGISTERED` flag to prevent duplicate route registration on multi-device setups.
- Falls back to sync `register_static_path` for older HA versions that lack `async_register_static_paths`.

### Panel Translation Serving (`frontend.py:296-314`)

- Registered at `PANEL_TRANSLATIONS_URL = /ha_washdata/panel-translations` pointing to `translations/panel/`.
- The panel fetches `/{lang}.json` + `en.json` on demand — no build step, no bundle. Languages not present fall back to `en.json`.
- This replaced the old monolithic `www/panel-translations.json` bundle (removed in 0.5.0).

### Lovelace Card Registration (`frontend.py:172-252`)

- `WashDataCardRegistration.async_register()` handles deferred registration if Lovelace is not yet loaded at setup time (listens for `EVENT_COMPONENT_LOADED`).
- For storage-backed Lovelace: updates an existing resource entry if the URL changed (version bump), or creates a new one.
- For non-storage Lovelace: calls `add_extra_js_url`.
- Returns `"registered"` / `"deferred"` / `"failed"`.

### Panel Unregistration (`frontend.py:368-400`)

- `async_unregister_panel()` removes the sidebar panel via `frontend.async_remove_panel()` and clears the guard flags.
- Called when the final WashData config entry is removed.
- HA has no public API to unregister a static path; clearing the guard flags allows re-registration on the next setup.

---

## 10. `ws_schema.py` — Typed Contract

### Purpose

Dependency-free (imports nothing from HA or `ws_api`); safe to import from tooling and tests without import cycles.

### `WS_COMMANDS` (`ws_schema.py:750`)

Map `command_name -> {"params": [_p(...)]}`. Hand-maintained to mirror the voluptuous schemas in `ws_api.py`. Used by `devtools/generate_ws_types.py` to emit TypeScript types and Markdown docs.

### `WS_RESPONSE_TYPES` (`ws_schema.py:609`)

Map `command_name -> TypedDict class`. Used by `_send_result` in debug-contract mode and by `tests/test_ws_contract.py` to assert command coverage.

### `WS_OPEN_RESPONSES` (`ws_schema.py:714`)

Commands whose response splats an upstream dict and has an open-ended key set — the debug validator skips extra-key checks for these:
- `run_suggestion_analysis`
- `run_playground_cycle_detail`
- `run_playground_history`
- `run_playground_sweep`
- `get_task_result`

### TypedDict totality convention

- `total=True` (default): every key always present in the response.
- `total=False`: response has conditional keys. Used for commands with state-dependent shapes (e.g. `GetRecordingStateResponse`, `GetPanelConfigResponse`, `RunPlaygroundHistoryResponse`, all store responses).

### Contract test

`tests/test_ws_contract.py` asserts that the set of commands registered by `async_register_commands` matches `WS_COMMANDS` keys exactly. Adding, removing, or renaming a command in `ws_api.py` without mirroring in `ws_schema.py` fails the test suite.

---

## 11. Feature Flag Gating

| Flag | Location | Commands gated |
|---|---|---|
| `ENABLE_ML_TRAINING` (`const.py`) | Build-time bool | `trigger_ml_training` returns `"not_available"` if False; `reprocess_history` skips ML training step; `get_ml_training_status` reports `available=False`; `get_constants` reports `ml_training_available=False` |
| `ENABLE_ML_SUGGESTIONS` (`const.py`) | Build-time bool | `get_ml_comparison` skips Classic-vs-ML settings table; `get_constants` reports `ml_suggestions_enabled=False` |
| `SHOW_ML_LAB` (`const.py`) | Build-time bool | `get_constants` reports `ml_lab_enabled=False`; panel hides ML Lab tab (no server-side gate on individual commands) |
| `online_features_enabled(hass)` | Runtime (integration-wide flag) | All `store_*` commands return `{"enabled": False}` / `{"disabled": True}` when offline |

---

## 12. CLAUDE.md vs Code Discrepancies

1. **"Needs full HA restart for new commands"** — `CLAUDE.md` (`project_panel_ws_gotchas.md` memory) says new WS commands need a full HA restart. The code comment at `__init__.py:624-628` explicitly documents this was **fixed**: `async_register_command` overwrites handlers per-command, so an integration reload now suffices. The CLAUDE.md memory note is outdated.

2. **`@async_response` requirement** — CLAUDE.md memory (`project_panel_ws_gotchas.md`) correctly notes this; confirmed in code.

3. **Task kinds in CLAUDE.md** — CLAUDE.md lists task kinds as `'reprocess' | 'ml_training' | 'pg_history' | 'pg_sweep'` but the actual code also uses `'pg_detail'`, `'rebuild'`, `'trim'`, `'split'`, `'merge'` as task kinds. The doc comment in `task_registry.py:60` is also outdated (lists only the first four).

4. **`ws_get_setup_status`** — listed in `WS_COMMANDS` and `WS_RESPONSE_TYPES` but not mentioned in CLAUDE.md. It calls `compute_setup_phase` from `setup_advisor.py` for the onboarding guidance card.

5. **Per-entry write lock** — not documented in CLAUDE.md; critical for correctness of concurrent write operations.

---

## 13. Summary Reference Table (count)

Total commands in `WS_COMMANDS`: **106**. Confirmed by `tests/test_ws_contract.py`, which asserts the registered commands match `WS_COMMANDS` exactly.

By area:

| Area | Count |
|---|---|
| Devices | 2 |
| Settings | 4 |
| Profiles | 7 |
| Profile groups | 4 |
| Maintenance log | 3 |
| Cycles (label/delete/trim/split/merge/power-data) | 8 |
| Phase catalog | 4 |
| Recording | 5 |
| Feedbacks | 3 |
| Diagnostics/maintenance | 6 |
| Shared constants | 1 |
| Suggestions | 4 |
| Panel config + RBAC + prefs | 3 |
| Live monitoring (match, power history, logs, set_program) | 4 |
| ML Lab (comparison, training status, trigger, revert, review) | 6 |
| Cycle controls (pause/resume/terminate) | 3 |
| Playground (inline + background tasks + DTW debug) | 8 |
| Background task registry | 4 |
| Community store | 19 |
| **Total** | **106** |

> The per-area figures above are approximate and predate the 0.5.4 additions (selective export/import, playground presets, suggestion lock); the authoritative total is 106, enforced by `tests/test_ws_contract.py`.
