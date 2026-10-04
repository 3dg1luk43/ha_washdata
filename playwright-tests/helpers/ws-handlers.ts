/**
 * WS handler factory for WashData Playwright tests.
 *
 * DEFAULT_HANDLERS is a minimal set that satisfies the panel's initial boot sequence.
 * Individual tests can override specific commands via the second argument to buildHandlers().
 */

import constants from '../fixtures/mock-data/constants.json';
import panelConfig from '../fixtures/mock-data/panel-config.json';
import deviceIdle from '../fixtures/mock-data/device-idle.json';
import cycles from '../fixtures/mock-data/cycles.json';
import profiles from '../fixtures/mock-data/profiles.json';
import options from '../fixtures/mock-data/options.json';

/** Minimal power history (no active cycle, just a flat idle line). Idle `live` is
 * [offset_s, watts] pairs from the oldest recent reading, as ws_get_power_history sends. */
const IDLE_POWER_HISTORY = {
  live: Array.from({ length: 20 }, (_, i) => [i * 30, 1.2]),
  raw: [],
  restart_gaps: [],
  cycle_active: false,
  cycle_elapsed_s: 0,
};

/** Device-resolved defaults (ws_api._resolved_option_defaults('washing_machine')). */
const OPTION_DEFAULTS = {
  sampling_interval: 2.0,
  watchdog_interval: 30,
  start_duration_threshold: 5.0,
  smart_termination_duration_ratio: 0.98,
  anti_crease_finalize_ratio: 0.98,
  profile_match_max_duration_ratio: 1.8,
  min_off_gap: 480,
  off_delay: 180,
};

/** Minimal suggestions (none to keep Settings tab clean by default). */
const NO_SUGGESTIONS = { suggestions: [], locked_suggestions: [] };

const EMPTY_MAINTENANCE = {
  log: [],
  due: [],
  event_types: ['descale', 'filter_clean', 'drum_clean', 'bearing_service', 'other'],
  reminders: { descale: 30 },
  // #414: cycles run since each task, and the odometer they are measured against.
  cycles_since: { descale: 12, filter_clean: 12, drum_clean: 12, bearing_service: 12, other: 12 },
  lifetime_cycle_count: 212,
};

const EMPTY_CHANGELOG = { changelog: [] };

const EMPTY_FEEDBACKS = { feedbacks: [] };

const EMPTY_PHASE_CATALOG = { phases: [], device_type: '' };

const EMPTY_PROFILE_GROUPS = { groups: [], min_cohesion: 0.85 };

const EMPTY_DIAGNOSTICS = {
  stats: {
    storage_size_bytes: 102400,
    cycle_count: 5,
    profile_count: 3,
    store_version: 8,
  },
  // No restore point: "Undo last import" stays hidden until a replace import
  // leaves one behind (register item 195).
  import_undo: null,
};

const EMPTY_ML_STATUS = {
  available: true,
  on_device_models: {},
  cycle_count: 0,
  min_cycles: 20,
  last_trained: null,
  enabled: false,
  interval_days: 7,
  hour: 2,
  running: false,
  last_run: {},
};

const EMPTY_ML_COMPARISON = {
  enabled: true,
  cycle_count: 0,
  evaluated_count: 0,
  cycles: [],
  model_source: { quality: 'baseline', end: 'baseline' },
  profile_stats: {},
};

const EMPTY_LOGS = { logs: [] };

const RECORDING_STATE = { state: 'idle', duration_s: 0, sample_count: 0 };

/** Default handler map — sufficient to boot and render every tab. */
export const DEFAULT_HANDLERS: Record<string, unknown> = {
  'ha_washdata/get_constants': constants,
  'ha_washdata/get_panel_config': panelConfig,
  'ha_washdata/get_devices': deviceIdle,
  'ha_washdata/get_power_history': IDLE_POWER_HISTORY,
  'ha_washdata/get_profile_envelope': { envelope: null },
  'ha_washdata/get_recording_state': RECORDING_STATE,
  'ha_washdata/get_device_cycles': cycles,
  'ha_washdata/get_feedbacks': EMPTY_FEEDBACKS,
  'ha_washdata/get_phase_catalog': EMPTY_PHASE_CATALOG,
  'ha_washdata/get_profiles': profiles,
  'ha_washdata/get_profile_groups': EMPTY_PROFILE_GROUPS,
  'ha_washdata/get_options': { options, defaults: OPTION_DEFAULTS },
  'ha_washdata/get_settings_changelog': EMPTY_CHANGELOG,
  'ha_washdata/get_ml_comparison': EMPTY_ML_COMPARISON,
  'ha_washdata/get_ml_training_status': EMPTY_ML_STATUS,
  'ha_washdata/get_diagnostics': EMPTY_DIAGNOSTICS,
  'ha_washdata/get_maintenance_log': EMPTY_MAINTENANCE,
  'ha_washdata/get_logs': EMPTY_LOGS,
  'ha_washdata/get_suggestions': NO_SUGGESTIONS,
  // Write commands — return success so form submissions don't throw.
  'ha_washdata/set_options': { success: true },
  'ha_washdata/undo_import': { success: true, summary: { restored_from: '2026-10-01T10:00:00+00:00', counts: { profiles: 2, real_cycles: 5, reference_cycles: 0, backfill_cycles: 0 } } },
  'ha_washdata/set_lifetime_cycle_count': { success: true, lifetime_cycle_count: 250 },
  'ha_washdata/set_user_prefs': { success: true },
  'ha_washdata/set_panel_config': { success: true },
  // Task result (mock-hass TASK_START): manager.async_run_ml_training's summary.
  'ha_washdata/trigger_ml_training': { ok: true, promoted: [], results: [] },
  // Historical power-data import (#344). The two `__history_import_*_result` keys are
  // not real commands: they are the payloads TASK_START hands back as each detached
  // task's result, mirroring how the playground task keys work.
  'ha_washdata/history_import_begin': { token: 'tok-1', max_bytes: 33554432, chunk_bytes: 524288 },
  'ha_washdata/history_import_chunk': { received_bytes: 128, next_seq: 1 },
  'ha_washdata/history_import_recorder': { token: 'tok-rec', rows: 2400, entity_id: 'sensor.washer_power', days: 10, start_date: '2026-07-18', truncated: false },
  'ha_washdata/__history_import_scan_result': {
    segments: [
      { index: 0, start_time: '2026-07-21T09:14:00+00:00', end_time: '2026-07-21T10:28:00+00:00',
        duration_s: 4464, status: 'completed', termination_reason: 'timeout', samples: 661,
        peak_w: 2165, energy_wh: 1096.3, accept: true, reason: null, below_minimum: false,
        curve: [10, 900, 1800, 1200, 400, 60, 5] },
      { index: 1, start_time: '2026-07-21T11:34:00+00:00', end_time: '2026-07-21T11:41:00+00:00',
        duration_s: 420, status: 'interrupted', termination_reason: 'timeout', samples: 84,
        peak_w: 410, energy_wh: 22.5, accept: false, reason: 'shorter_than_minimum', below_minimum: true,
        curve: [5, 300, 410, 120, 4] },
    ],
    skipped: [{ reason: 'idle', span_s: 3600, samples: 1 }, { reason: 'sparse', span_s: 604800, samples: 114 }],
    parse: { rows_total: 2358, rows_parsed: 2358, breaks: 8, rows_other_entity: 0,
             first: '2026-07-21T09:14:00+00:00', last: '2026-07-28T08:22:00+00:00', peak_w: 2165 },
    settings: { min_power: 2, off_delay: 300, min_off_gap: 480, device_type: 'washing_machine' },
    found: 2, capped: false, truncated_blocks: 0, partial: false, token: 'tok-1',
  },
  'ha_washdata/__history_import_apply_result': { imported: 1, duplicates: 0, capped: false, total_backfill: 1 },
  'ha_washdata/revert_ml_models': { success: true },
  'ha_washdata/label_cycle': { success: true },
  'ha_washdata/create_profile': { success: true, name: 'New Profile' },
  'ha_washdata/delete_profile': { success: true },
  'ha_washdata/add_maintenance_event': {
    success: true,
    event: { id: 'maint-001', date: '2026-07-20T09:00:00+00:00', event_type: 'descale', notes: '', cycle_count_at_log: 212 },
  },
  'ha_washdata/delete_maintenance_event': { success: true },
  // Split/trim run as background tasks; these are the payloads get_task_result returns.
  'ha_washdata/apply_split': { success: true, new_ids: ['cyc-split-a', 'cyc-split-b'] },
  'ha_washdata/trim_cycle': { success: true },
  'ha_washdata/apply_merge': { success: true, new_id: 'cyc-merged' },
  'ha_washdata/rebuild_envelopes': { success: true, rebuilt: 3 },
  'ha_washdata/analyze_split': { segments: [[0, 600], [900, 1740]], split_offsets: [600], samples: [], sample_count: 0, decimated: false, full_duration_s: 1740 },
  'ha_washdata/get_cycle_power_data': {
    cycle_id: 'cyc-001',
    samples: Array.from({ length: 30 }, (_, i) => [i * 60, i < 2 || i > 27 ? 3 : 900]),
    sample_count: 30,
    decimated: false,
    full_duration_s: 1740,
  },
  'ha_washdata/run_playground_cycle_detail': {
    cycle_id: 'cyc-001', label: 'Cotton 40°C', duration_s: 1740,
    config_summary: { device_type: 'washing_machine', off_delay: 180 },
    series: [
      { t: 0, power: 3, energy_wh: 0, state: 'starting', progress: null, remaining_s: null, phase: null, confidence: null, matched_profile: null },
      { t: 600, power: 900, energy_wh: 120, state: 'running', progress: 35, remaining_s: 1100, phase: 'Wash', confidence: 0.74, matched_profile: 'Cotton 40°C', projected_energy_wh: 480, projected_cost: 2.4 },
      { t: 1700, power: 3, energy_wh: 470, state: 'ending', progress: 97, remaining_s: 40, phase: 'Spin', confidence: 0.82, matched_profile: 'Cotton 40°C', projected_energy_wh: 485, projected_cost: 2.5 },
    ],
    events: [
      { t: 30, type: 'detected', detail: 'cycle detected', severity: 'info' },
      { t: 300, type: 'match_commit', detail: 'Cotton 40°C (0.74)', severity: 'info' },
      { t: 1700, type: 'finished', detail: 'reason=smart', severity: 'info' },
    ],
    alerts: [],
    outcome: { detected: true, detected_count: 1, termination_reason: 'smart', status: 'completed', final_duration_s: 1740, matched_profile: 'Cotton 40°C', match_correct: true, confidence: 0.82, expected_s: 1720, overrun_ratio: 1.01, projected_energy_wh: 485, projected_cost: 2.5, would_label: true, label_profile: 'Cotton 40°C', label_reason: 'ok' },
  },
  'ha_washdata/run_playground_history': {
    rows: [
      { cycle_id: 'cyc-001', label: 'Cotton 40°C', detected: true, detected_count: 1, matched_profile: 'Cotton 40°C', match_correct: true, confidence: 0.82, termination_reason: 'smart', duration_s: 1740, expected_s: 1720, overrun_ratio: 1.01, alerts: [], would_label: false, label_profile: null, label_reason: 'margin' },
    ],
    summary: { cycles: 1, detected: 1, labelled: 1, match_correct: 1, match_wrong: 0, unmatched: 0, false_end: 0 },
  },
  'ha_washdata/run_playground_sweep': {
    param: 'off_delay', objective: 'match_accuracy', current_value: 180, best_value: 120, best_metric: 0.9,
    points: [{ value: 120, metric: 0.9, summary: {} }, { value: 180, metric: 0.8, summary: {} }],
  },
  // Playground settings control panel: the live effective values the sandbox opens
  // on, plus this device's saved presets. `publishable` mirrors the backend
  // allow-list (every key is a real option).
  'ha_washdata/get_playground_settings': {
    effective: {
      min_power: 2,
      off_delay: 120,
      min_off_gap: 180,
      start_threshold_w: 10,
      stop_threshold_w: 3,
      completion_min_seconds: 600,
      start_duration_threshold: 5,
      end_repeat_count: 1,
      interrupted_min_seconds: 150,
      anti_wrinkle_enabled: false,
      anti_wrinkle_max_power: 200,
      anti_wrinkle_max_duration: 300,
      anti_wrinkle_exit_power: 10,
      anti_wrinkle_idle_timeout: 900,
      dishwasher_end_spike_quiet_release: 600,
      profile_match_min_duration_ratio: 0.1,
      profile_match_max_duration_ratio: 1.5,
    },
    presets: [
      { name: 'Quiet nights', values: { off_delay: 300, min_off_gap: 240 }, created_at: '2026-08-01T10:00:00+00:00', updated_at: '2026-08-01T10:00:00+00:00' },
    ],
    publishable: [
      'min_power', 'off_delay', 'min_off_gap', 'start_threshold_w', 'stop_threshold_w',
      'completion_min_seconds', 'start_duration_threshold', 'end_repeat_count',
      'interrupted_min_seconds', 'anti_wrinkle_enabled', 'anti_wrinkle_max_power',
      'anti_wrinkle_max_duration', 'anti_wrinkle_exit_power', 'anti_wrinkle_idle_timeout',
      'dishwasher_end_spike_quiet_release',
      'profile_match_min_duration_ratio', 'profile_match_max_duration_ratio',
    ],
    preset_limit: 30,
    classic_suggestions: { off_delay: 90, min_off_gap: 240 },
  },
  'ha_washdata/save_playground_preset': {
    success: true,
    presets: [
      { name: 'My preset', values: { off_delay: 222 }, created_at: '2026-08-16T10:00:00+00:00', updated_at: '2026-08-16T10:00:00+00:00' },
      { name: 'Quiet nights', values: { off_delay: 300 }, created_at: '2026-08-01T10:00:00+00:00', updated_at: '2026-08-01T10:00:00+00:00' },
    ],
  },
  'ha_washdata/delete_playground_preset': { success: true, presets: [] },
};

/**
 * Build a final handler map merging defaults with test-specific overrides.
 * Handlers can be a static value (returned as-is) or a function (msg → value).
 */
export function buildHandlers(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return { ...DEFAULT_HANDLERS, ...overrides };
}

/** Convenience: return a devices payload with a single running device. */
export { deviceIdle, cycles, profiles, options };
export { IDLE_POWER_HISTORY, OPTION_DEFAULTS, EMPTY_MAINTENANCE, EMPTY_CHANGELOG, EMPTY_ML_STATUS };
