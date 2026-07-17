# WashData WebSocket API

<!-- AUTO-GENERATED — do not edit; run devtools/generate_ws_types.py -->

This document is generated from `custom_components/ha_washdata/ws_schema.py`. Every command is prefixed with `ha_washdata/` on the wire. Do not edit by hand — run `python3 devtools/generate_ws_types.py`.

**97 commands.**

| Command | Request params | Response type |
| --- | --- | --- |
| `get_devices` | — | `GetDevicesResponse` |
| `get_device_cycles` | entry_id, limit?, offset? | `GetDeviceCyclesResponse` |
| `get_options` | entry_id | `GetOptionsResponse` |
| `set_options` | entry_id, options | `SuccessResponse` |
| `get_settings_changelog` | entry_id | `GetSettingsChangelogResponse` |
| `get_profiles` | entry_id | `GetProfilesResponse` |
| `create_profile` | entry_id, name, reference_cycle?, manual_duration_min? | `CreateProfileResponse` |
| `rename_profile` | entry_id, profile_name, new_name, manual_duration_min? | `SuccessResponse` |
| `delete_profile` | entry_id, profile_name, unlabel_cycles? | `SuccessResponse` |
| `get_profile_groups` | entry_id | `GetProfileGroupsResponse` |
| `save_profile_group` | entry_id, name, members | `SuccessResponse` |
| `rename_profile_group` | entry_id, name, new_name | `SuccessResponse` |
| `delete_profile_group` | entry_id, name | `SuccessResponse` |
| `rebuild_envelopes` | entry_id | `SuccessResponse` |
| `get_profile_phases` | entry_id, profile_name | `GetProfilePhasesResponse` |
| `set_profile_phases` | entry_id, profile_name, phases | `SuccessResponse` |
| `get_maintenance_log` | entry_id | `GetMaintenanceLogResponse` |
| `add_maintenance_event` | entry_id, event_type, date?, notes? | `AddMaintenanceEventResponse` |
| `delete_maintenance_event` | entry_id, event_id | `SuccessResponse` |
| `label_cycle` | entry_id, cycle_id, profile_name?, new_profile_name? | `SuccessResponse` |
| `delete_cycle` | entry_id, cycle_id | `SuccessResponse` |
| `auto_label_cycles` | entry_id, confidence_threshold? | `SuccessResponse` |
| `get_phase_catalog` | entry_id, device_type? | `GetPhaseCatalogResponse` |
| `create_phase` | entry_id, device_type, name, description? | `SuccessResponse` |
| `update_phase` | entry_id, phase_id, new_name, description? | `SuccessResponse` |
| `delete_phase` | entry_id, phase_id | `SuccessResponse` |
| `get_recording_state` | entry_id | `GetRecordingStateResponse` |
| `start_recording` | entry_id | `SuccessResponse` |
| `stop_recording` | entry_id | `SuccessResponse` |
| `process_recording` | entry_id, profile_name, save_mode, head_trim?, tail_trim? | `SuccessResponse` |
| `discard_recording` | entry_id | `SuccessResponse` |
| `get_feedbacks` | entry_id | `GetFeedbacksResponse` |
| `resolve_feedback` | entry_id, cycle_id, action, corrected_profile?, corrected_duration_min? | `SuccessResponse` |
| `dismiss_all_feedbacks` | entry_id | `DismissAllFeedbacksResponse` |
| `get_diagnostics` | entry_id | `GetDiagnosticsResponse` |
| `reprocess_history` | entry_id | `StartTaskResponse` |
| `clear_debug_data` | entry_id | `ClearDebugDataResponse` |
| `wipe_history` | entry_id | `SuccessResponse` |
| `export_config` | entry_id | `ExportConfigResponse` |
| `import_config` | entry_id, json_data | `SuccessResponse` |
| `get_constants` | — | `GetConstantsResponse` |
| `get_suggestions` | entry_id | `GetSuggestionsResponse` |
| `apply_suggestions` | entry_id, keys | `ApplySuggestionsResponse` |
| `clear_suggestions` | entry_id | `SuccessResponse` |
| `run_suggestion_analysis` | entry_id | `RunSuggestionAnalysisResponse` |
| `get_cycle_power_data` | entry_id, cycle_id | `GetCyclePowerDataResponse` |
| `trim_cycle` | entry_id, cycle_id, start_s, end_s | `SuccessResponse` |
| `analyze_split` | entry_id, cycle_id, gap_seconds? | `AnalyzeSplitResponse` |
| `apply_split` | entry_id, cycle_id, split_offsets, segment_profiles? | `ApplySplitResponse` |
| `apply_merge` | entry_id, cycle_ids, target_profile?, new_profile_name? | `ApplyMergeResponse` |
| `get_profile_envelope` | entry_id, profile_name | `GetProfileEnvelopeResponse` |
| `get_profile_cycles` | entry_id, profile_name, limit? | `GetProfileCyclesResponse` |
| `get_panel_config` | — | `GetPanelConfigResponse` |
| `set_panel_config` | panel?, rbac? | `SuccessResponse` |
| `set_user_prefs` | prefs | `SuccessResponse` |
| `get_match_debug` | entry_id | `GetMatchDebugResponse` |
| `set_program` | entry_id, program | `SuccessResponse` |
| `get_power_history` | entry_id, with_raw? | `GetPowerHistoryResponse` |
| `get_logs` | level?, limit? | `GetLogsResponse` |
| `get_ml_comparison` | entry_id | `GetMlComparisonResponse` |
| `get_ml_training_status` | entry_id | `GetMlTrainingStatusResponse` |
| `trigger_ml_training` | entry_id | `StartTaskResponse` |
| `revert_matching_config` | entry_id | `SuccessResponse` |
| `revert_ml_models` | entry_id | `SuccessResponse` |
| `set_ml_review` | entry_id, cycle_id, quality?, golden?, tags?, notes? | `SuccessResponse` |
| `pause_cycle` | entry_id | `OkResponse` |
| `resume_cycle` | entry_id | `OkResponse` |
| `terminate_cycle` | entry_id | `OkResponse` |
| `run_playground_simulation` | entry_id, cycle_ids?, settings_override?, concurrency? | `RunPlaygroundSimulationResponse` |
| `run_playground_cycle_detail` | entry_id, cycle_id, settings_override? | `RunPlaygroundCycleDetailResponse` |
| `run_playground_history` | entry_id, cycle_ids?, settings_override?, concurrency? | `RunPlaygroundHistoryResponse` |
| `run_playground_sweep` | entry_id, param, values, objective, cycle_ids?, concurrency?, param_y?, values_y? | `RunPlaygroundSweepResponse` |
| `get_dtw_debug` | entry_id, cycle_id, profile_name? | `GetDtwDebugResponse` |
| `list_tasks` | entry_id? | `ListTasksResponse` |
| `subscribe_tasks` | entry_id? | `SubscribeTasksResponse` |
| `cancel_task` | task_id | `CancelTaskResponse` |
| `get_task_result` | task_id | `TaskSnapshot` |
| `start_playground_history` | entry_id, cycle_ids?, settings_override? | `StartTaskResponse` |
| `start_playground_sweep` | entry_id, param, values, objective, param_y?, values_y? | `StartTaskResponse` |
| `store_status` | entry_id | `StoreStatusResponse` |
| `store_connect` | entry_id, refresh_token, uid, name? | `StoreSimpleResponse` |
| `store_disconnect` | entry_id | `StoreSimpleResponse` |
| `store_search_devices` | entry_id, query?, appliance_type?, model_query?, include_pending? | `StoreItemsResponse` |
| `store_list_brands` | entry_id, query?, include_pending? | `StoreItemsResponse` |
| `store_get_profiles` | entry_id, device_id | `StoreItemsResponse` |
| `store_get_cycles` | entry_id, profile_id | `StoreItemsResponse` |
| `store_get_device_quality` | entry_id, device_id | `StoreQualityResponse` |
| `store_get_device_profiles` | entry_id, brand, model, appliance_type | `StoreDeviceProfilesResponse` |
| `store_confirm_device` | entry_id, device_id | `StoreConfirmResponse` |
| `store_rate_device` | entry_id, device_id, rating | `StoreOnlineResponse` |
| `store_set_online` | entry_id, enabled | `StoreOnlineResponse` |
| `store_set_prefs` | entry_id, prefs | `StorePrefsResponse` |
| `store_import_cycle` | entry_id, cycle_id, target_profile?, new_profile_name? | `StoreImportResponse` |
| `store_upload_cycle` | entry_id, local_cycle_id, program, description? | `StoreUploadResponse` |
| `store_upload_device` | entry_id, items, include_phases?, include_settings? | `StoreUploadDeviceResponse` |
| `store_download_device` | entry_id, device_id, include_settings? | `StoreDownloadDeviceResponse` |
| `get_shareable_cycles` | entry_id | `GetShareableCyclesResponse` |

## `ha_washdata/get_devices`

**Request parameters**

_None._

**Response** (`GetDevicesResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `devices` | yes | list[DeviceInfo] |

## `ha_washdata/get_device_cycles`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `limit` | no | int |
| `offset` | no | int |

**Response** (`GetDeviceCyclesResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycles` | yes | list[dict[str, any]] |
| `reference_cycles` | yes | list[dict[str, any]] |
| `total` | yes | number |
| `has_more` | yes | bool |

## `ha_washdata/get_options`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetOptionsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `options` | yes | dict[str, any] |

## `ha_washdata/set_options`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `options` | yes | dict |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_settings_changelog`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetSettingsChangelogResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `changelog` | yes | list[dict[str, any]] |

## `ha_washdata/get_profiles`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetProfilesResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `profiles` | yes | list[dict[str, any]] |
| `profile_health` | yes | dict[str, any] |
| `profile_trends` | yes | dict[str, any] |
| `coverage_gaps` | yes | dict[str, any] |
| `profile_advisories` | yes | list[dict[str, any]] |

## `ha_washdata/create_profile`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `name` | yes | str |
| `reference_cycle` | no | str\|null |
| `manual_duration_min` | no | float\|null |

**Response** (`CreateProfileResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |
| `name` | yes | str |

## `ha_washdata/rename_profile`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `profile_name` | yes | str |
| `new_name` | yes | str |
| `manual_duration_min` | no | float\|null |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/delete_profile`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `profile_name` | yes | str |
| `unlabel_cycles` | no | bool |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_profile_groups`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetProfileGroupsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `groups` | yes | list[ProfileGroupInfo] |
| `min_cohesion` | yes | number |
| `suggestions` | yes | list[dict[str, any]] |

## `ha_washdata/save_profile_group`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `name` | yes | str |
| `members` | yes | list[str] |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/rename_profile_group`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `name` | yes | str |
| `new_name` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/delete_profile_group`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `name` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/rebuild_envelopes`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_profile_phases`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `profile_name` | yes | str |

**Response** (`GetProfilePhasesResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `phases` | yes | list[dict[str, any]] |

## `ha_washdata/set_profile_phases`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `profile_name` | yes | str |
| `phases` | yes | list |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_maintenance_log`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetMaintenanceLogResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `log` | yes | list[dict[str, any]] |
| `due` | yes | any |
| `event_types` | yes | list[str] |
| `reminders` | yes | dict[str, any] |

## `ha_washdata/add_maintenance_event`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `event_type` | yes | str |
| `date` | no | str\|null |
| `notes` | no | str |

**Response** (`AddMaintenanceEventResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |
| `event` | yes | dict[str, any] |

## `ha_washdata/delete_maintenance_event`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `event_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/label_cycle`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `profile_name` | no | str\|null |
| `new_profile_name` | no | str\|null |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/delete_cycle`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/auto_label_cycles`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `confidence_threshold` | no | float |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_phase_catalog`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `device_type` | no | str\|null |

**Response** (`GetPhaseCatalogResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `phases` | yes | list[dict[str, any]] |
| `device_type` | yes | str \| null |

## `ha_washdata/create_phase`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `device_type` | yes | str |
| `name` | yes | str |
| `description` | no | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/update_phase`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `phase_id` | yes | str |
| `new_name` | yes | str |
| `description` | no | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/delete_phase`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `phase_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_recording_state`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetRecordingStateResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `state` | no | str |
| `duration_s` | no | number |
| `sample_count` | no | number |
| `start_time` | no | str \| null |
| `end_time` | no | str \| null |

## `ha_washdata/start_recording`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/stop_recording`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/process_recording`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `profile_name` | yes | str |
| `save_mode` | yes | str ('new_profile', 'existing_profile') |
| `head_trim` | no | float |
| `tail_trim` | no | float |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/discard_recording`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_feedbacks`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetFeedbacksResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `feedbacks` | yes | list[dict[str, any]] |

## `ha_washdata/resolve_feedback`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `action` | yes | str ('confirm', 'correct', 'ignore', 'delete') |
| `corrected_profile` | no | str\|null |
| `corrected_duration_min` | no | float\|null |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/dismiss_all_feedbacks`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`DismissAllFeedbacksResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |
| `dismissed` | yes | number |

## `ha_washdata/get_diagnostics`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetDiagnosticsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `stats` | yes | dict[str, any] |

## `ha_washdata/reprocess_history`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`StartTaskResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `task_id` | yes | str |

## `ha_washdata/clear_debug_data`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`ClearDebugDataResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |
| `count` | yes | number |

## `ha_washdata/wipe_history`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/export_config`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`ExportConfigResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `json_data` | yes | str |

## `ha_washdata/import_config`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `json_data` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_constants`

**Request parameters**

_None._

**Response** (`GetConstantsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `device_types` | yes | list[dict[str, any]] |
| `state_colors` | yes | dict[str, any] |
| `ml_lab_enabled` | yes | bool |
| `ml_suggestions_enabled` | yes | bool |
| `ml_training_available` | yes | bool |
| `PROFILE_MIN_WARMUP_CYCLES` | yes | any |
| `store_online_available` | yes | bool |
| `store_online_enabled` | yes | bool |
| `store_web_origin` | yes | str |

## `ha_washdata/get_suggestions`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetSuggestionsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `suggestions` | yes | list[dict[str, any]] |

## `ha_washdata/apply_suggestions`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `keys` | yes | list[str] |

**Response** (`ApplySuggestionsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |
| `applied` | yes | list[str] |

## `ha_washdata/clear_suggestions`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/run_suggestion_analysis`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`RunSuggestionAnalysisResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | no | bool |
| `count` | no | number |

_Open-ended: additional top-level keys from an upstream summary may be present._

## `ha_washdata/get_cycle_power_data`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |

**Response** (`GetCyclePowerDataResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `cycle_id` | no | str |
| `samples` | no | list[list[number]] |
| `full_duration_s` | no | number |
| `start_time` | no | str \| null |
| `end_time` | no | str \| null |
| `duration` | no | number \| null |
| `profile_name` | no | str \| null |
| `status` | no | str \| null |
| `energy_kwh` | no | number \| null |
| `artifacts` | no | list[dict[str, any]] |
| `restart_gaps` | no | list[any] |

## `ha_washdata/trim_cycle`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `start_s` | yes | float |
| `end_s` | yes | float |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/analyze_split`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `gap_seconds` | no | int |

**Response** (`AnalyzeSplitResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `segments` | yes | list[list[number]] |
| `split_offsets` | yes | list[number] |
| `samples` | yes | list[list[number]] |
| `full_duration_s` | yes | number |

## `ha_washdata/apply_split`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `split_offsets` | yes | list[float] |
| `segment_profiles` | no | list |

**Response** (`ApplySplitResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |
| `new_ids` | yes | list[str] |

## `ha_washdata/apply_merge`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_ids` | yes | list[str] |
| `target_profile` | no | str\|null |
| `new_profile_name` | no | str\|null |

**Response** (`ApplyMergeResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |
| `new_id` | yes | str |

## `ha_washdata/get_profile_envelope`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `profile_name` | yes | str |

**Response** (`GetProfileEnvelopeResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `envelope` | yes | ProfileEnvelope \| null |

## `ha_washdata/get_profile_cycles`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `profile_name` | yes | str |
| `limit` | no | int |

**Response** (`GetProfileCyclesResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `cycles` | yes | list[dict[str, any]] |

## `ha_washdata/get_panel_config`

**Request parameters**

_None._

**Response** (`GetPanelConfigResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `panel` | no | dict[str, any] |
| `is_admin` | no | bool |
| `user` | no | dict[str, any] |
| `prefs` | no | dict[str, any] |
| `rbac` | no | dict[str, any] |
| `users` | no | list[dict[str, any]] |

## `ha_washdata/set_panel_config`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `panel` | no | dict |
| `rbac` | no | dict |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/set_user_prefs`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `prefs` | yes | dict |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_match_debug`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetMatchDebugResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `confidence` | yes | number \| null |
| `ambiguous` | yes | bool |
| `candidates` | yes | list[dict[str, any]] |

## `ha_washdata/set_program`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `program` | yes | str\|null |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/get_power_history`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `with_raw` | no | bool |

**Response** (`GetPowerHistoryResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `cycle_active` | no | bool |
| `cycle_elapsed_s` | no | number |
| `live` | no | list[list[number]] |
| `raw` | no | list[list[number]] |
| `restart_gaps` | no | list[any] |
| `cycle_start_iso` | no | str |

## `ha_washdata/get_logs`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `level` | no | str\|null |
| `limit` | no | int |

**Response** (`GetLogsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `logs` | yes | list[dict[str, any]] |

## `ha_washdata/get_ml_comparison`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetMlComparisonResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `enabled` | no | bool |
| `error` | no | str |
| `cycles` | no | list[dict[str, any]] |
| `settings_comparison` | no | dict[str, any] |
| `cycle_count` | no | number |
| `evaluated_count` | no | number |
| `model_source` | no | dict[str, any] |
| `profile_stats` | no | dict[str, any] |
| `ml_suggestions_enabled` | no | bool |

## `ha_washdata/get_ml_training_status`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetMlTrainingStatusResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `available` | yes | bool |
| `enabled` | yes | bool |
| `running` | yes | bool |
| `last_trained` | yes | str \| null |
| `cycle_count` | yes | number |
| `min_cycles` | yes | number |
| `interval_days` | yes | number |
| `hour` | yes | number |
| `on_device_models` | yes | dict[str, any] |
| `matching` | yes | dict[str, any] |

## `ha_washdata/trigger_ml_training`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`StartTaskResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `task_id` | yes | str |

## `ha_washdata/revert_matching_config`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/revert_ml_models`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/set_ml_review`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `quality` | no | str ('', 'bad', 'good', 'unusable') |
| `golden` | no | bool |
| `tags` | no | list[str] |
| `notes` | no | str |

**Response** (`SuccessResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `success` | yes | bool |

## `ha_washdata/pause_cycle`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`OkResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `ok` | yes | bool |

## `ha_washdata/resume_cycle`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`OkResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `ok` | yes | bool |

## `ha_washdata/terminate_cycle`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`OkResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `ok` | yes | bool |

## `ha_washdata/run_playground_simulation`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_ids` | no | list[str] |
| `settings_override` | no | dict |
| `concurrency` | no | int |

**Response** (`RunPlaygroundSimulationResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `results` | yes | list[dict[str, any]] |
| `summary` | yes | PlaygroundSummary |

## `ha_washdata/run_playground_cycle_detail`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `settings_override` | no | dict |

**Response** (`RunPlaygroundCycleDetailResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `cycle_id` | no | any |
| `label` | no | str \| null |
| `duration_s` | no | number \| null |
| `config_summary` | no | dict[str, any] |
| `series` | no | list[dict[str, any]] |
| `events` | no | list[dict[str, any]] |
| `alerts` | no | list[dict[str, any]] |
| `outcome` | no | dict[str, any] |
| `error` | no | str |

_Open-ended: additional top-level keys from an upstream summary may be present._

## `ha_washdata/run_playground_history`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_ids` | no | list[str] |
| `settings_override` | no | dict |
| `concurrency` | no | int |

**Response** (`RunPlaygroundHistoryResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `rows` | no | list[dict[str, any]] |
| `summary` | no | dict[str, any] |
| `baseline_rows` | no | list[dict[str, any]] |
| `baseline_summary` | no | dict[str, any] |
| `diff` | no | dict[str, list[str]] |

_Open-ended: additional top-level keys from an upstream summary may be present._

## `ha_washdata/run_playground_sweep`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `param` | yes | str |
| `values` | yes | list[float] |
| `objective` | yes | str |
| `cycle_ids` | no | list[str] |
| `concurrency` | no | int |
| `param_y` | no | str |
| `values_y` | no | list[float] |

**Response** (`RunPlaygroundSweepResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `param` | no | str |
| `objective` | no | str |
| `points` | no | list[dict[str, any]] |
| `current_value` | no | any |
| `best_value` | no | any |
| `best_metric` | no | number \| null |
| `param_x` | no | str |
| `param_y` | no | str |
| `x_values` | no | list[number] |
| `y_values` | no | list[number] |
| `grid` | no | list[list[any]] |
| `best` | no | dict[str, any] |
| `current` | no | dict[str, any] |
| `error` | no | str |

_Open-ended: additional top-level keys from an upstream summary may be present._

## `ha_washdata/get_dtw_debug`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `profile_name` | no | str\|null |

**Response** (`GetDtwDebugResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `cycle_id` | yes | any |
| `profile_name` | yes | str |
| `grid_n` | yes | number |
| `cycle_duration_s` | yes | number |
| `profile_duration_s` | yes | number |
| `cycle_trace` | yes | list[list[number]] |
| `profile_trace` | yes | list[list[number]] |
| `stage2` | yes | DtwStage2Scores |
| `dtw` | yes | DtwScores |
| `stage4` | yes | DtwStage4Scores |
| `warp_path` | yes | list[list[number]] |

## `ha_washdata/list_tasks`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | no | str\|null |

**Response** (`ListTasksResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `tasks` | yes | list[TaskSnapshot] |

## `ha_washdata/subscribe_tasks`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | no | str\|null |

**Response** (`SubscribeTasksResponse`)

_Empty object._

## `ha_washdata/cancel_task`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `task_id` | yes | str |

**Response** (`CancelTaskResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `cancelled` | yes | bool |

## `ha_washdata/get_task_result`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `task_id` | yes | str |

**Response** (`TaskSnapshot`)

| Field | Always present | Type |
| --- | --- | --- |
| `id` | no | str |
| `entry_id` | no | str |
| `kind` | no | str |
| `label` | no | str |
| `state` | no | str |
| `done` | no | number |
| `total` | no | number |
| `progress` | no | number \| null |
| `eta_s` | no | number \| null |
| `started_at` | no | number |
| `updated_at` | no | number |
| `finished_at` | no | number \| null |
| `error` | no | str \| null |
| `has_result` | no | bool |
| `result` | no | any |

_Open-ended: additional top-level keys from an upstream summary may be present._

## `ha_washdata/start_playground_history`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_ids` | no | list[str] |
| `settings_override` | no | dict |

**Response** (`StartTaskResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `task_id` | yes | str |

## `ha_washdata/start_playground_sweep`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `param` | yes | str |
| `values` | yes | list[float] |
| `objective` | yes | str |
| `param_y` | no | str\|null |
| `values_y` | no | list[float] |

**Response** (`StartTaskResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `task_id` | yes | str |

## `ha_washdata/store_status`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`StoreStatusResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `enabled` | no | bool |
| `connected` | no | bool |
| `uid` | no | str \| null |
| `name` | no | str \| null |
| `brand` | no | str \| null |
| `model` | no | str \| null |
| `disabled` | no | bool |

## `ha_washdata/store_connect`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `refresh_token` | yes | str |
| `uid` | yes | str |
| `name` | no | str\|null |

**Response** (`StoreSimpleResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `connected` | no | bool |
| `uid` | no | str \| null |
| `name` | no | str \| null |
| `brand` | no | str \| null |
| `model` | no | str \| null |
| `error` | no | str |
| `disabled` | no | bool |

## `ha_washdata/store_disconnect`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`StoreSimpleResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `connected` | no | bool |
| `uid` | no | str \| null |
| `name` | no | str \| null |
| `brand` | no | str \| null |
| `model` | no | str \| null |
| `error` | no | str |
| `disabled` | no | bool |

## `ha_washdata/store_search_devices`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `query` | no | str\|null |
| `appliance_type` | no | str\|null |
| `model_query` | no | str\|null |
| `include_pending` | no | bool |

**Response** (`StoreItemsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `items` | no | list |
| `disabled` | no | bool |

## `ha_washdata/store_list_brands`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `query` | no | str\|null |
| `include_pending` | no | bool |

**Response** (`StoreItemsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `items` | no | list |
| `disabled` | no | bool |

## `ha_washdata/store_get_profiles`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `device_id` | yes | str |

**Response** (`StoreItemsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `items` | no | list |
| `disabled` | no | bool |

## `ha_washdata/store_get_cycles`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `profile_id` | yes | str |

**Response** (`StoreItemsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `items` | no | list |
| `disabled` | no | bool |

## `ha_washdata/store_get_device_quality`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `device_id` | yes | str |

**Response** (`StoreQualityResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `avg` | no | number \| null |
| `count` | no | number |
| `disabled` | no | bool |

## `ha_washdata/store_get_device_profiles`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `brand` | yes | str |
| `model` | yes | str |
| `appliance_type` | yes | str |

**Response** (`StoreDeviceProfilesResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `device_id` | no | str |
| `items` | no | list |
| `disabled` | no | bool |

## `ha_washdata/store_confirm_device`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `device_id` | yes | str |

**Response** (`StoreConfirmResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `confirmed` | no | bool |
| `confirmCount` | no | number |
| `status` | no | str \| null |
| `error` | no | str |
| `disabled` | no | bool |

## `ha_washdata/store_rate_device`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `device_id` | yes | str |
| `rating` | yes | int |

**Response** (`StoreOnlineResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `enabled` | no | bool |
| `ok` | no | bool |
| `error` | no | str |
| `disabled` | no | bool |

## `ha_washdata/store_set_online`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `enabled` | yes | bool |

**Response** (`StoreOnlineResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `enabled` | no | bool |
| `ok` | no | bool |
| `error` | no | str |
| `disabled` | no | bool |

## `ha_washdata/store_set_prefs`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `prefs` | yes | dict |

**Response** (`StorePrefsResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `prefs` | no | dict[str, any] |
| `error` | no | str |
| `disabled` | no | bool |

## `ha_washdata/store_import_cycle`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `cycle_id` | yes | str |
| `target_profile` | no | str\|null |
| `new_profile_name` | no | str\|null |

**Response** (`StoreImportResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `profile` | no | str |
| `cycle_id` | no | str |
| `error` | no | str |
| `disabled` | no | bool |

## `ha_washdata/store_upload_cycle`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `local_cycle_id` | yes | str |
| `program` | yes | str |
| `description` | no | str\|null |

**Response** (`StoreUploadResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `store_cycle_id` | no | str |
| `error` | no | str |
| `detail` | no | str \| null |
| `disabled` | no | bool |

## `ha_washdata/store_upload_device`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `items` | yes | list |
| `include_phases` | no | list |
| `include_settings` | no | bool |

**Response** (`StoreUploadDeviceResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `ok` | no | bool |
| `cycle_ids` | no | list |
| `errors` | no | list |
| `error` | no | str |
| `detail` | no | str \| null |
| `disabled` | no | bool |

## `ha_washdata/store_download_device`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |
| `device_id` | yes | str |
| `include_settings` | no | bool |

**Response** (`StoreDownloadDeviceResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `profiles_adopted` | no | number |
| `cycles_imported` | no | number |
| `phases_applied` | no | number |
| `settings_applied` | no | number |
| `error` | no | str |
| `disabled` | no | bool |

## `ha_washdata/get_shareable_cycles`

**Request parameters**

| Param | Required | Type |
| --- | --- | --- |
| `entry_id` | yes | str |

**Response** (`GetShareableCyclesResponse`)

| Field | Always present | Type |
| --- | --- | --- |
| `items` | no | list |
| `phase_programs` | no | list |
