// AUTO-GENERATED — do not edit; run devtools/generate_ws_types.py
// WashData WebSocket API type contract (Group H1).
//
// Response payloads for every `ha_washdata/*` WebSocket command, plus the
// request parameters each accepts. Import these into the panel for a typed
// `hass.callWS` layer.

// ── Response payloads ──────────────────────────────────────────────────────

export interface AddMaintenanceEventResponse {
  success: boolean;
  event: Record<string, unknown>;
}

export interface AnalyzeSplitResponse {
  segments: number[][];
  split_offsets: number[];
  samples: number[][];
  full_duration_s: number;
}

export interface ApplyMergeResponse {
  success: boolean;
  new_id: string;
}

export interface ApplySplitResponse {
  success: boolean;
  new_ids: string[];
}

export interface ApplySuggestionsResponse {
  success: boolean;
  applied: string[];
}

export interface CancelTaskResponse {
  cancelled: boolean;
}

export interface ClearDebugDataResponse {
  success: boolean;
  count: number;
}

export interface CreateProfileResponse {
  success: boolean;
  name: string;
}

export interface DeviceInfo {
  entry_id: string;
  perm: string;
  title: string;
  detector_state: string;
  sub_state: string | null;
  current_program: string | null;
  time_remaining_s: number | null;
  total_duration_s: number | null;
  current_power_w: number | null;
  cycle_progress_pct: number | null;
  suggestions_count: number;
  feedback_count: number;
  recording: boolean;
  is_user_paused: boolean;
  manual_program: boolean;
  options: Record<string, unknown>;
}

export interface DismissAllFeedbacksResponse {
  success: boolean;
  dismissed: number;
}

export interface DtwScores {
  l1_score: number;
  ddtw_score: number;
  ensemble_score: number;
  blend_weight: number;
  blended_score: number;
}

export interface DtwStage2Scores {
  correlation: number;
  mae_score: number;
  score: number;
}

export interface DtwStage4Scores {
  duration_agreement: number;
  energy_agreement: number;
  final_score: number;
}

export interface ExportConfigResponse {
  json_data: string;
}

export interface GetConstantsResponse {
  device_types: Record<string, unknown>[];
  state_colors: Record<string, unknown>;
  ml_lab_enabled: boolean;
  ml_suggestions_enabled: boolean;
  ml_training_available: boolean;
  PROFILE_MIN_WARMUP_CYCLES: unknown;
  store_online_available: boolean;
  store_online_enabled: boolean;
  store_web_origin: string;
}

export interface GetCyclePowerDataResponse {
  cycle_id?: string;
  samples?: number[][];
  full_duration_s?: number;
  start_time?: string | null;
  end_time?: string | null;
  duration?: number | null;
  profile_name?: string | null;
  status?: string | null;
  energy_kwh?: number | null;
  artifacts?: Record<string, unknown>[];
  restart_gaps?: unknown[];
}

export interface GetDeviceCyclesResponse {
  entry_id: string;
  cycles: Record<string, unknown>[];
  reference_cycles: Record<string, unknown>[];
  total: number;
  has_more: boolean;
}

export interface GetDevicesResponse {
  devices: DeviceInfo[];
}

export interface GetDiagnosticsResponse {
  stats: Record<string, unknown>;
}

export interface GetDtwDebugResponse {
  cycle_id: unknown;
  profile_name: string;
  grid_n: number;
  cycle_duration_s: number;
  profile_duration_s: number;
  cycle_trace: number[][];
  profile_trace: number[][];
  stage2: DtwStage2Scores;
  dtw: DtwScores;
  stage4: DtwStage4Scores;
  warp_path: number[][];
}

export interface GetFeedbacksResponse {
  feedbacks: Record<string, unknown>[];
}

export interface GetLogsResponse {
  logs: Record<string, unknown>[];
}

export interface GetMaintenanceLogResponse {
  log: Record<string, unknown>[];
  due: unknown;
  event_types: string[];
  reminders: Record<string, unknown>;
}

export interface GetMatchDebugResponse {
  confidence: number | null;
  ambiguous: boolean;
  candidates: Record<string, unknown>[];
}

export interface GetMlComparisonResponse {
  enabled?: boolean;
  error?: string;
  cycles?: Record<string, unknown>[];
  settings_comparison?: Record<string, unknown>;
  cycle_count?: number;
  evaluated_count?: number;
  model_source?: Record<string, unknown>;
  profile_stats?: Record<string, unknown>;
  ml_suggestions_enabled?: boolean;
}

export interface GetMlTrainingStatusResponse {
  available: boolean;
  enabled: boolean;
  running: boolean;
  last_trained: string | null;
  cycle_count: number;
  min_cycles: number;
  interval_days: number;
  hour: number;
  on_device_models: Record<string, unknown>;
  matching: Record<string, unknown>;
}

export interface GetOptionsResponse {
  options: Record<string, unknown>;
}

export interface GetPanelConfigResponse {
  panel?: Record<string, unknown>;
  is_admin?: boolean;
  user?: Record<string, unknown>;
  prefs?: Record<string, unknown>;
  rbac?: Record<string, unknown>;
  users?: Record<string, unknown>[];
}

export interface GetPhaseCatalogResponse {
  phases: Record<string, unknown>[];
  device_type: string | null;
}

export interface GetPowerHistoryResponse {
  cycle_active?: boolean;
  cycle_elapsed_s?: number;
  live?: number[][];
  raw?: number[][];
  restart_gaps?: unknown[];
  cycle_start_iso?: string;
}

export interface GetProfileCyclesResponse {
  cycles: Record<string, unknown>[];
}

export interface GetProfileEnvelopeResponse {
  envelope: ProfileEnvelope | null;
}

export interface GetProfileGroupsResponse {
  groups: ProfileGroupInfo[];
  min_cohesion: number;
  suggestions: Record<string, unknown>[];
}

export interface GetProfilePhasesResponse {
  phases: Record<string, unknown>[];
}

export interface GetProfilesResponse {
  profiles: Record<string, unknown>[];
  profile_health: Record<string, unknown>;
  profile_trends: Record<string, unknown>;
  coverage_gaps: Record<string, unknown>;
  profile_advisories: Record<string, unknown>[];
}

export interface GetRecordingStateResponse {
  state?: string;
  duration_s?: number;
  sample_count?: number;
  start_time?: string | null;
  end_time?: string | null;
}

export interface GetSettingsChangelogResponse {
  changelog: Record<string, unknown>[];
}

export interface GetShareableCyclesResponse {
  items?: unknown[];
}

export interface GetSuggestionsResponse {
  suggestions: Record<string, unknown>[];
}

export interface ListTasksResponse {
  tasks: TaskSnapshot[];
}

export interface OkResponse {
  ok: boolean;
}

export interface PlaygroundSummary {
  cycles: number;
  requested: number;
  concurrency: number;
  detected: number;
  missed: number;
  false_end: number;
  match_correct: number;
  match_wrong: number;
  unmatched: number;
  skipped_ids: string[];
}

export interface ProfileEnvelope {
  avg: number[][];
  min: number[][];
  max: number[][];
  target_duration: number | null;
  avg_energy: number | null;
  duration_std_dev: number | null;
  cycle_count: number;
}

export interface ProfileGroupInfo {
  name: string;
  members: string[];
  cohesion: number;
  cohesive: boolean;
}

export interface RunPlaygroundCycleDetailResponse {
  cycle_id?: unknown;
  label?: string | null;
  duration_s?: number | null;
  config_summary?: Record<string, unknown>;
  series?: Record<string, unknown>[];
  events?: Record<string, unknown>[];
  alerts?: Record<string, unknown>[];
  outcome?: Record<string, unknown>;
  error?: string;
}

export interface RunPlaygroundHistoryResponse {
  rows?: Record<string, unknown>[];
  summary?: Record<string, unknown>;
  baseline_rows?: Record<string, unknown>[];
  baseline_summary?: Record<string, unknown>;
  diff?: Record<string, string[]>;
}

export interface RunPlaygroundSimulationResponse {
  results: Record<string, unknown>[];
  summary: PlaygroundSummary;
}

export interface RunPlaygroundSweepResponse {
  param?: string;
  objective?: string;
  points?: Record<string, unknown>[];
  current_value?: unknown;
  best_value?: unknown;
  best_metric?: number | null;
  param_x?: string;
  param_y?: string;
  x_values?: number[];
  y_values?: number[];
  grid?: unknown[][];
  best?: Record<string, unknown>;
  current?: Record<string, unknown>;
  error?: string;
}

export interface RunSuggestionAnalysisResponse {
  success?: boolean;
  count?: number;
}

export interface StartTaskResponse {
  task_id: string;
}

export interface StoreConfirmResponse {
  confirmed?: boolean;
  confirmCount?: number;
  status?: string | null;
  error?: string;
  disabled?: boolean;
}

export interface StoreDeviceProfilesResponse {
  device_id?: string;
  items?: unknown[];
  disabled?: boolean;
}

export interface StoreDownloadDeviceResponse {
  profiles_adopted?: number;
  cycles_imported?: number;
  error?: string;
  disabled?: boolean;
}

export interface StoreImportResponse {
  profile?: string;
  cycle_id?: string;
  error?: string;
  disabled?: boolean;
}

export interface StoreItemsResponse {
  items?: unknown[];
  disabled?: boolean;
}

export interface StoreOnlineResponse {
  enabled?: boolean;
  ok?: boolean;
  error?: string;
  disabled?: boolean;
}

export interface StorePrefsResponse {
  prefs?: Record<string, unknown>;
  error?: string;
  disabled?: boolean;
}

export interface StoreQualityResponse {
  avg?: number | null;
  count?: number;
  disabled?: boolean;
}

export interface StoreSimpleResponse {
  connected?: boolean;
  uid?: string | null;
  name?: string | null;
  brand?: string | null;
  model?: string | null;
  error?: string;
  disabled?: boolean;
}

export interface StoreStatusResponse {
  enabled?: boolean;
  connected?: boolean;
  uid?: string | null;
  name?: string | null;
  brand?: string | null;
  model?: string | null;
  disabled?: boolean;
}

export interface StoreUploadDeviceResponse {
  ok?: boolean;
  cycle_ids?: unknown[];
  errors?: unknown[];
  error?: string;
  detail?: string | null;
  disabled?: boolean;
}

export interface StoreUploadResponse {
  store_cycle_id?: string;
  error?: string;
  detail?: string | null;
  disabled?: boolean;
}

export interface SubscribeTasksResponse {
}

export interface SuccessResponse {
  success: boolean;
}

export interface TaskSnapshot {
  id?: string;
  entry_id?: string;
  kind?: string;
  label?: string;
  state?: string;
  done?: number;
  total?: number;
  progress?: number | null;
  eta_s?: number | null;
  started_at?: number;
  updated_at?: number;
  finished_at?: number | null;
  error?: string | null;
  has_result?: boolean;
  result?: unknown;
}

// ── Request parameters ─────────────────────────────────────────────────────

export interface GetDevicesRequest {
}

export interface GetDeviceCyclesRequest {
  entry_id: string;
  limit?: number;
  offset?: number;
}

export interface GetOptionsRequest {
  entry_id: string;
}

export interface SetOptionsRequest {
  entry_id: string;
  options: Record<string, unknown>;
}

export interface GetSettingsChangelogRequest {
  entry_id: string;
}

export interface GetProfilesRequest {
  entry_id: string;
}

export interface CreateProfileRequest {
  entry_id: string;
  name: string;
  reference_cycle?: string | null;
  manual_duration_min?: number | null;
}

export interface RenameProfileRequest {
  entry_id: string;
  profile_name: string;
  new_name: string;
  manual_duration_min?: number | null;
}

export interface DeleteProfileRequest {
  entry_id: string;
  profile_name: string;
  unlabel_cycles?: boolean;
}

export interface GetProfileGroupsRequest {
  entry_id: string;
}

export interface SaveProfileGroupRequest {
  entry_id: string;
  name: string;
  members: string[];
}

export interface RenameProfileGroupRequest {
  entry_id: string;
  name: string;
  new_name: string;
}

export interface DeleteProfileGroupRequest {
  entry_id: string;
  name: string;
}

export interface RebuildEnvelopesRequest {
  entry_id: string;
}

export interface GetProfilePhasesRequest {
  entry_id: string;
  profile_name: string;
}

export interface SetProfilePhasesRequest {
  entry_id: string;
  profile_name: string;
  phases: unknown[];
}

export interface GetMaintenanceLogRequest {
  entry_id: string;
}

export interface AddMaintenanceEventRequest {
  entry_id: string;
  event_type: string;
  date?: string | null;
  notes?: string;
}

export interface DeleteMaintenanceEventRequest {
  entry_id: string;
  event_id: string;
}

export interface LabelCycleRequest {
  entry_id: string;
  cycle_id: string;
  profile_name?: string | null;
  new_profile_name?: string | null;
}

export interface DeleteCycleRequest {
  entry_id: string;
  cycle_id: string;
}

export interface AutoLabelCyclesRequest {
  entry_id: string;
  confidence_threshold?: number;
}

export interface GetPhaseCatalogRequest {
  entry_id: string;
  device_type?: string | null;
}

export interface CreatePhaseRequest {
  entry_id: string;
  device_type: string;
  name: string;
  description?: string;
}

export interface UpdatePhaseRequest {
  entry_id: string;
  phase_id: string;
  new_name: string;
  description?: string;
}

export interface DeletePhaseRequest {
  entry_id: string;
  phase_id: string;
}

export interface GetRecordingStateRequest {
  entry_id: string;
}

export interface StartRecordingRequest {
  entry_id: string;
}

export interface StopRecordingRequest {
  entry_id: string;
}

export interface ProcessRecordingRequest {
  entry_id: string;
  profile_name: string;
  save_mode: "new_profile" | "existing_profile";
  head_trim?: number;
  tail_trim?: number;
}

export interface DiscardRecordingRequest {
  entry_id: string;
}

export interface GetFeedbacksRequest {
  entry_id: string;
}

export interface ResolveFeedbackRequest {
  entry_id: string;
  cycle_id: string;
  action: "confirm" | "correct" | "ignore" | "delete";
  corrected_profile?: string | null;
  corrected_duration_min?: number | null;
}

export interface DismissAllFeedbacksRequest {
  entry_id: string;
}

export interface GetDiagnosticsRequest {
  entry_id: string;
}

export interface ReprocessHistoryRequest {
  entry_id: string;
}

export interface ClearDebugDataRequest {
  entry_id: string;
}

export interface WipeHistoryRequest {
  entry_id: string;
}

export interface ExportConfigRequest {
  entry_id: string;
}

export interface ImportConfigRequest {
  entry_id: string;
  json_data: string;
}

export interface GetConstantsRequest {
}

export interface GetSuggestionsRequest {
  entry_id: string;
}

export interface ApplySuggestionsRequest {
  entry_id: string;
  keys: string[];
}

export interface ClearSuggestionsRequest {
  entry_id: string;
}

export interface RunSuggestionAnalysisRequest {
  entry_id: string;
}

export interface GetCyclePowerDataRequest {
  entry_id: string;
  cycle_id: string;
}

export interface TrimCycleRequest {
  entry_id: string;
  cycle_id: string;
  start_s: number;
  end_s: number;
}

export interface AnalyzeSplitRequest {
  entry_id: string;
  cycle_id: string;
  gap_seconds?: number;
}

export interface ApplySplitRequest {
  entry_id: string;
  cycle_id: string;
  split_offsets: number[];
  segment_profiles?: unknown[];
}

export interface ApplyMergeRequest {
  entry_id: string;
  cycle_ids: string[];
  target_profile?: string | null;
  new_profile_name?: string | null;
}

export interface GetProfileEnvelopeRequest {
  entry_id: string;
  profile_name: string;
}

export interface GetProfileCyclesRequest {
  entry_id: string;
  profile_name: string;
  limit?: number;
}

export interface GetPanelConfigRequest {
}

export interface SetPanelConfigRequest {
  panel?: Record<string, unknown>;
  rbac?: Record<string, unknown>;
}

export interface SetUserPrefsRequest {
  prefs: Record<string, unknown>;
}

export interface GetMatchDebugRequest {
  entry_id: string;
}

export interface SetProgramRequest {
  entry_id: string;
  program: string | null;
}

export interface GetPowerHistoryRequest {
  entry_id: string;
  with_raw?: boolean;
}

export interface GetLogsRequest {
  level?: string | null;
  limit?: number;
}

export interface GetMlComparisonRequest {
  entry_id: string;
}

export interface GetMlTrainingStatusRequest {
  entry_id: string;
}

export interface TriggerMlTrainingRequest {
  entry_id: string;
}

export interface RevertMatchingConfigRequest {
  entry_id: string;
}

export interface RevertMlModelsRequest {
  entry_id: string;
}

export interface SetMlReviewRequest {
  entry_id: string;
  cycle_id: string;
  quality?: "" | "bad" | "good" | "unusable";
  golden?: boolean;
  tags?: string[];
  notes?: string;
}

export interface PauseCycleRequest {
  entry_id: string;
}

export interface ResumeCycleRequest {
  entry_id: string;
}

export interface TerminateCycleRequest {
  entry_id: string;
}

export interface RunPlaygroundSimulationRequest {
  entry_id: string;
  cycle_ids?: string[];
  settings_override?: Record<string, unknown>;
  concurrency?: number;
}

export interface RunPlaygroundCycleDetailRequest {
  entry_id: string;
  cycle_id: string;
  settings_override?: Record<string, unknown>;
}

export interface RunPlaygroundHistoryRequest {
  entry_id: string;
  cycle_ids?: string[];
  settings_override?: Record<string, unknown>;
  concurrency?: number;
}

export interface RunPlaygroundSweepRequest {
  entry_id: string;
  param: string;
  values: number[];
  objective: string;
  cycle_ids?: string[];
  concurrency?: number;
  param_y?: string;
  values_y?: number[];
}

export interface GetDtwDebugRequest {
  entry_id: string;
  cycle_id: string;
  profile_name?: string | null;
}

export interface ListTasksRequest {
  entry_id?: string | null;
}

export interface SubscribeTasksRequest {
  entry_id?: string | null;
}

export interface CancelTaskRequest {
  task_id: string;
}

export interface GetTaskResultRequest {
  task_id: string;
}

export interface StartPlaygroundHistoryRequest {
  entry_id: string;
  cycle_ids?: string[];
  settings_override?: Record<string, unknown>;
}

export interface StartPlaygroundSweepRequest {
  entry_id: string;
  param: string;
  values: number[];
  objective: string;
  param_y?: string | null;
  values_y?: number[];
}

export interface StoreStatusRequest {
  entry_id: string;
}

export interface StoreConnectRequest {
  entry_id: string;
  refresh_token: string;
  uid: string;
  name?: string | null;
}

export interface StoreDisconnectRequest {
  entry_id: string;
}

export interface StoreSearchDevicesRequest {
  entry_id: string;
  query?: string | null;
  appliance_type?: string | null;
  model_query?: string | null;
  include_pending?: boolean;
}

export interface StoreListBrandsRequest {
  entry_id: string;
  query?: string | null;
  include_pending?: boolean;
}

export interface StoreGetProfilesRequest {
  entry_id: string;
  device_id: string;
}

export interface StoreGetCyclesRequest {
  entry_id: string;
  profile_id: string;
}

export interface StoreGetDeviceQualityRequest {
  entry_id: string;
  device_id: string;
}

export interface StoreGetDeviceProfilesRequest {
  entry_id: string;
  brand: string;
  model: string;
  appliance_type: string;
}

export interface StoreConfirmDeviceRequest {
  entry_id: string;
  device_id: string;
}

export interface StoreRateDeviceRequest {
  entry_id: string;
  device_id: string;
  rating: number;
}

export interface StoreSetOnlineRequest {
  entry_id: string;
  enabled: boolean;
}

export interface StoreSetPrefsRequest {
  entry_id: string;
  prefs: Record<string, unknown>;
}

export interface StoreImportCycleRequest {
  entry_id: string;
  cycle_id: string;
  target_profile?: string | null;
  new_profile_name?: string | null;
}

export interface StoreUploadCycleRequest {
  entry_id: string;
  local_cycle_id: string;
  program: string;
  description?: string | null;
}

export interface StoreUploadDeviceRequest {
  entry_id: string;
  items: unknown[];
}

export interface StoreDownloadDeviceRequest {
  entry_id: string;
  device_id: string;
}

export interface GetShareableCyclesRequest {
  entry_id: string;
}

// ── Command maps ───────────────────────────────────────────────────────────

export interface WashDataWsRequests {
  "ha_washdata/get_devices": GetDevicesRequest;
  "ha_washdata/get_device_cycles": GetDeviceCyclesRequest;
  "ha_washdata/get_options": GetOptionsRequest;
  "ha_washdata/set_options": SetOptionsRequest;
  "ha_washdata/get_settings_changelog": GetSettingsChangelogRequest;
  "ha_washdata/get_profiles": GetProfilesRequest;
  "ha_washdata/create_profile": CreateProfileRequest;
  "ha_washdata/rename_profile": RenameProfileRequest;
  "ha_washdata/delete_profile": DeleteProfileRequest;
  "ha_washdata/get_profile_groups": GetProfileGroupsRequest;
  "ha_washdata/save_profile_group": SaveProfileGroupRequest;
  "ha_washdata/rename_profile_group": RenameProfileGroupRequest;
  "ha_washdata/delete_profile_group": DeleteProfileGroupRequest;
  "ha_washdata/rebuild_envelopes": RebuildEnvelopesRequest;
  "ha_washdata/get_profile_phases": GetProfilePhasesRequest;
  "ha_washdata/set_profile_phases": SetProfilePhasesRequest;
  "ha_washdata/get_maintenance_log": GetMaintenanceLogRequest;
  "ha_washdata/add_maintenance_event": AddMaintenanceEventRequest;
  "ha_washdata/delete_maintenance_event": DeleteMaintenanceEventRequest;
  "ha_washdata/label_cycle": LabelCycleRequest;
  "ha_washdata/delete_cycle": DeleteCycleRequest;
  "ha_washdata/auto_label_cycles": AutoLabelCyclesRequest;
  "ha_washdata/get_phase_catalog": GetPhaseCatalogRequest;
  "ha_washdata/create_phase": CreatePhaseRequest;
  "ha_washdata/update_phase": UpdatePhaseRequest;
  "ha_washdata/delete_phase": DeletePhaseRequest;
  "ha_washdata/get_recording_state": GetRecordingStateRequest;
  "ha_washdata/start_recording": StartRecordingRequest;
  "ha_washdata/stop_recording": StopRecordingRequest;
  "ha_washdata/process_recording": ProcessRecordingRequest;
  "ha_washdata/discard_recording": DiscardRecordingRequest;
  "ha_washdata/get_feedbacks": GetFeedbacksRequest;
  "ha_washdata/resolve_feedback": ResolveFeedbackRequest;
  "ha_washdata/dismiss_all_feedbacks": DismissAllFeedbacksRequest;
  "ha_washdata/get_diagnostics": GetDiagnosticsRequest;
  "ha_washdata/reprocess_history": ReprocessHistoryRequest;
  "ha_washdata/clear_debug_data": ClearDebugDataRequest;
  "ha_washdata/wipe_history": WipeHistoryRequest;
  "ha_washdata/export_config": ExportConfigRequest;
  "ha_washdata/import_config": ImportConfigRequest;
  "ha_washdata/get_constants": GetConstantsRequest;
  "ha_washdata/get_suggestions": GetSuggestionsRequest;
  "ha_washdata/apply_suggestions": ApplySuggestionsRequest;
  "ha_washdata/clear_suggestions": ClearSuggestionsRequest;
  "ha_washdata/run_suggestion_analysis": RunSuggestionAnalysisRequest;
  "ha_washdata/get_cycle_power_data": GetCyclePowerDataRequest;
  "ha_washdata/trim_cycle": TrimCycleRequest;
  "ha_washdata/analyze_split": AnalyzeSplitRequest;
  "ha_washdata/apply_split": ApplySplitRequest;
  "ha_washdata/apply_merge": ApplyMergeRequest;
  "ha_washdata/get_profile_envelope": GetProfileEnvelopeRequest;
  "ha_washdata/get_profile_cycles": GetProfileCyclesRequest;
  "ha_washdata/get_panel_config": GetPanelConfigRequest;
  "ha_washdata/set_panel_config": SetPanelConfigRequest;
  "ha_washdata/set_user_prefs": SetUserPrefsRequest;
  "ha_washdata/get_match_debug": GetMatchDebugRequest;
  "ha_washdata/set_program": SetProgramRequest;
  "ha_washdata/get_power_history": GetPowerHistoryRequest;
  "ha_washdata/get_logs": GetLogsRequest;
  "ha_washdata/get_ml_comparison": GetMlComparisonRequest;
  "ha_washdata/get_ml_training_status": GetMlTrainingStatusRequest;
  "ha_washdata/trigger_ml_training": TriggerMlTrainingRequest;
  "ha_washdata/revert_matching_config": RevertMatchingConfigRequest;
  "ha_washdata/revert_ml_models": RevertMlModelsRequest;
  "ha_washdata/set_ml_review": SetMlReviewRequest;
  "ha_washdata/pause_cycle": PauseCycleRequest;
  "ha_washdata/resume_cycle": ResumeCycleRequest;
  "ha_washdata/terminate_cycle": TerminateCycleRequest;
  "ha_washdata/run_playground_simulation": RunPlaygroundSimulationRequest;
  "ha_washdata/run_playground_cycle_detail": RunPlaygroundCycleDetailRequest;
  "ha_washdata/run_playground_history": RunPlaygroundHistoryRequest;
  "ha_washdata/run_playground_sweep": RunPlaygroundSweepRequest;
  "ha_washdata/get_dtw_debug": GetDtwDebugRequest;
  "ha_washdata/list_tasks": ListTasksRequest;
  "ha_washdata/subscribe_tasks": SubscribeTasksRequest;
  "ha_washdata/cancel_task": CancelTaskRequest;
  "ha_washdata/get_task_result": GetTaskResultRequest;
  "ha_washdata/start_playground_history": StartPlaygroundHistoryRequest;
  "ha_washdata/start_playground_sweep": StartPlaygroundSweepRequest;
  "ha_washdata/store_status": StoreStatusRequest;
  "ha_washdata/store_connect": StoreConnectRequest;
  "ha_washdata/store_disconnect": StoreDisconnectRequest;
  "ha_washdata/store_search_devices": StoreSearchDevicesRequest;
  "ha_washdata/store_list_brands": StoreListBrandsRequest;
  "ha_washdata/store_get_profiles": StoreGetProfilesRequest;
  "ha_washdata/store_get_cycles": StoreGetCyclesRequest;
  "ha_washdata/store_get_device_quality": StoreGetDeviceQualityRequest;
  "ha_washdata/store_get_device_profiles": StoreGetDeviceProfilesRequest;
  "ha_washdata/store_confirm_device": StoreConfirmDeviceRequest;
  "ha_washdata/store_rate_device": StoreRateDeviceRequest;
  "ha_washdata/store_set_online": StoreSetOnlineRequest;
  "ha_washdata/store_set_prefs": StoreSetPrefsRequest;
  "ha_washdata/store_import_cycle": StoreImportCycleRequest;
  "ha_washdata/store_upload_cycle": StoreUploadCycleRequest;
  "ha_washdata/store_upload_device": StoreUploadDeviceRequest;
  "ha_washdata/store_download_device": StoreDownloadDeviceRequest;
  "ha_washdata/get_shareable_cycles": GetShareableCyclesRequest;
}

export interface WashDataWsResponses {
  "ha_washdata/get_devices": GetDevicesResponse;
  "ha_washdata/get_device_cycles": GetDeviceCyclesResponse;
  "ha_washdata/get_options": GetOptionsResponse;
  "ha_washdata/set_options": SuccessResponse;
  "ha_washdata/get_settings_changelog": GetSettingsChangelogResponse;
  "ha_washdata/get_profiles": GetProfilesResponse;
  "ha_washdata/create_profile": CreateProfileResponse;
  "ha_washdata/rename_profile": SuccessResponse;
  "ha_washdata/delete_profile": SuccessResponse;
  "ha_washdata/get_profile_groups": GetProfileGroupsResponse;
  "ha_washdata/save_profile_group": SuccessResponse;
  "ha_washdata/rename_profile_group": SuccessResponse;
  "ha_washdata/delete_profile_group": SuccessResponse;
  "ha_washdata/rebuild_envelopes": SuccessResponse;
  "ha_washdata/get_profile_phases": GetProfilePhasesResponse;
  "ha_washdata/set_profile_phases": SuccessResponse;
  "ha_washdata/get_maintenance_log": GetMaintenanceLogResponse;
  "ha_washdata/add_maintenance_event": AddMaintenanceEventResponse;
  "ha_washdata/delete_maintenance_event": SuccessResponse;
  "ha_washdata/label_cycle": SuccessResponse;
  "ha_washdata/delete_cycle": SuccessResponse;
  "ha_washdata/auto_label_cycles": SuccessResponse;
  "ha_washdata/get_phase_catalog": GetPhaseCatalogResponse;
  "ha_washdata/create_phase": SuccessResponse;
  "ha_washdata/update_phase": SuccessResponse;
  "ha_washdata/delete_phase": SuccessResponse;
  "ha_washdata/get_recording_state": GetRecordingStateResponse;
  "ha_washdata/start_recording": SuccessResponse;
  "ha_washdata/stop_recording": SuccessResponse;
  "ha_washdata/process_recording": SuccessResponse;
  "ha_washdata/discard_recording": SuccessResponse;
  "ha_washdata/get_feedbacks": GetFeedbacksResponse;
  "ha_washdata/resolve_feedback": SuccessResponse;
  "ha_washdata/dismiss_all_feedbacks": DismissAllFeedbacksResponse;
  "ha_washdata/get_diagnostics": GetDiagnosticsResponse;
  "ha_washdata/reprocess_history": StartTaskResponse;
  "ha_washdata/clear_debug_data": ClearDebugDataResponse;
  "ha_washdata/wipe_history": SuccessResponse;
  "ha_washdata/export_config": ExportConfigResponse;
  "ha_washdata/import_config": SuccessResponse;
  "ha_washdata/get_constants": GetConstantsResponse;
  "ha_washdata/get_suggestions": GetSuggestionsResponse;
  "ha_washdata/apply_suggestions": ApplySuggestionsResponse;
  "ha_washdata/clear_suggestions": SuccessResponse;
  "ha_washdata/run_suggestion_analysis": RunSuggestionAnalysisResponse;
  "ha_washdata/get_cycle_power_data": GetCyclePowerDataResponse;
  "ha_washdata/trim_cycle": SuccessResponse;
  "ha_washdata/analyze_split": AnalyzeSplitResponse;
  "ha_washdata/apply_split": ApplySplitResponse;
  "ha_washdata/apply_merge": ApplyMergeResponse;
  "ha_washdata/get_profile_envelope": GetProfileEnvelopeResponse;
  "ha_washdata/get_profile_cycles": GetProfileCyclesResponse;
  "ha_washdata/get_panel_config": GetPanelConfigResponse;
  "ha_washdata/set_panel_config": SuccessResponse;
  "ha_washdata/set_user_prefs": SuccessResponse;
  "ha_washdata/get_match_debug": GetMatchDebugResponse;
  "ha_washdata/set_program": SuccessResponse;
  "ha_washdata/get_power_history": GetPowerHistoryResponse;
  "ha_washdata/get_logs": GetLogsResponse;
  "ha_washdata/get_ml_comparison": GetMlComparisonResponse;
  "ha_washdata/get_ml_training_status": GetMlTrainingStatusResponse;
  "ha_washdata/trigger_ml_training": StartTaskResponse;
  "ha_washdata/revert_matching_config": SuccessResponse;
  "ha_washdata/revert_ml_models": SuccessResponse;
  "ha_washdata/set_ml_review": SuccessResponse;
  "ha_washdata/pause_cycle": OkResponse;
  "ha_washdata/resume_cycle": OkResponse;
  "ha_washdata/terminate_cycle": OkResponse;
  "ha_washdata/run_playground_simulation": RunPlaygroundSimulationResponse;
  "ha_washdata/run_playground_cycle_detail": RunPlaygroundCycleDetailResponse;
  "ha_washdata/run_playground_history": RunPlaygroundHistoryResponse;
  "ha_washdata/run_playground_sweep": RunPlaygroundSweepResponse;
  "ha_washdata/get_dtw_debug": GetDtwDebugResponse;
  "ha_washdata/list_tasks": ListTasksResponse;
  "ha_washdata/subscribe_tasks": SubscribeTasksResponse;
  "ha_washdata/cancel_task": CancelTaskResponse;
  "ha_washdata/get_task_result": TaskSnapshot;
  "ha_washdata/start_playground_history": StartTaskResponse;
  "ha_washdata/start_playground_sweep": StartTaskResponse;
  "ha_washdata/store_status": StoreStatusResponse;
  "ha_washdata/store_connect": StoreSimpleResponse;
  "ha_washdata/store_disconnect": StoreSimpleResponse;
  "ha_washdata/store_search_devices": StoreItemsResponse;
  "ha_washdata/store_get_profiles": StoreItemsResponse;
  "ha_washdata/store_get_cycles": StoreItemsResponse;
  "ha_washdata/store_import_cycle": StoreImportResponse;
  "ha_washdata/store_upload_cycle": StoreUploadResponse;
  "ha_washdata/store_list_brands": StoreItemsResponse;
  "ha_washdata/store_get_device_quality": StoreQualityResponse;
  "ha_washdata/store_confirm_device": StoreConfirmResponse;
  "ha_washdata/store_rate_device": StoreOnlineResponse;
  "ha_washdata/store_set_online": StoreOnlineResponse;
  "ha_washdata/store_set_prefs": StorePrefsResponse;
  "ha_washdata/store_get_device_profiles": StoreDeviceProfilesResponse;
  "ha_washdata/store_upload_device": StoreUploadDeviceResponse;
  "ha_washdata/store_download_device": StoreDownloadDeviceResponse;
  "ha_washdata/get_shareable_cycles": GetShareableCyclesResponse;
}
