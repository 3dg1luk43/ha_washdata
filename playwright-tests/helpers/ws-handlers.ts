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

/** Minimal power history (no active cycle, just a flat idle line). */
const IDLE_POWER_HISTORY = {
  live: Array.from({ length: 20 }, (_, i) => ({ t: Date.now() / 1000 - (19 - i) * 30, p: 1.2 })),
  raw: [],
  cycle_active: false,
  cycle_elapsed_s: 0,
  profile_envelope: null,
};

/** Minimal suggestions (none to keep Settings tab clean by default). */
const NO_SUGGESTIONS = { suggestions: [] };

const EMPTY_MAINTENANCE = {
  log: [],
  due: [],
  event_types: ['descale', 'filter_clean', 'drum_clean', 'bearing_service', 'other'],
  reminders: {},
};

const EMPTY_CHANGELOG = { changelog: [] };

const EMPTY_FEEDBACKS = { feedbacks: [] };

const EMPTY_PHASE_CATALOG = { phases: [], device_type: '' };

const EMPTY_PROFILE_GROUPS = { groups: [], suggestions: [], min_cohesion: 0.85 };

const EMPTY_DIAGNOSTICS = {
  stats: {
    storage_size_bytes: 102400,
    cycle_count: 5,
    profile_count: 3,
    store_version: 8,
  },
};

const EMPTY_ML_STATUS = {
  on_device_models: {},
  cycle_count: 0,
  min_cycles: 20,
  last_trained: null,
  enabled: false,
  hour: 2,
  running: false,
};

const EMPTY_ML_COMPARISON = { comparisons: [] };

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
  'ha_washdata/get_options': { options },
  'ha_washdata/get_settings_changelog': EMPTY_CHANGELOG,
  'ha_washdata/get_ml_comparison': EMPTY_ML_COMPARISON,
  'ha_washdata/get_ml_training_status': EMPTY_ML_STATUS,
  'ha_washdata/get_diagnostics': EMPTY_DIAGNOSTICS,
  'ha_washdata/get_maintenance_log': EMPTY_MAINTENANCE,
  'ha_washdata/get_logs': EMPTY_LOGS,
  'ha_washdata/get_suggestions': NO_SUGGESTIONS,
  // Write commands — return success so form submissions don't throw.
  'ha_washdata/set_options': { success: true },
  'ha_washdata/set_user_prefs': { success: true },
  'ha_washdata/set_panel_config': { success: true },
  'ha_washdata/trigger_ml_training': { ok: true, message: 'Training started' },
  'ha_washdata/revert_ml_models': { ok: true },
  'ha_washdata/revert_matching_config': { ok: true },
  'ha_washdata/label_cycle': { success: true },
  'ha_washdata/delete_cycles': { success: true },
  'ha_washdata/create_profile': { success: true },
  'ha_washdata/delete_profile': { success: true },
  'ha_washdata/add_maintenance_event': { success: true, id: 'maint-001' },
  'ha_washdata/delete_maintenance_event': { success: true },
  'ha_washdata/save_maintenance_reminders': { success: true },
  'ha_washdata/run_playground_simulation': { results: {}, summary: {} },
  // Split/trim run as background tasks; these are the payloads get_task_result returns.
  'ha_washdata/apply_split': { success: true, new_ids: ['cyc-split-a', 'cyc-split-b'] },
  'ha_washdata/trim_cycle': { success: true },
  'ha_washdata/analyze_split': { segments: [[0, 600], [900, 1740]], split_offsets: [600], samples: [], full_duration_s: 1740 },
  'ha_washdata/get_cycle_power_data': {
    samples: Array.from({ length: 30 }, (_, i) => [i * 60, i < 2 || i > 27 ? 3 : 900]),
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
    outcome: { detected: true, detected_count: 1, termination_reason: 'smart', status: 'completed', final_duration_s: 1740, matched_profile: 'Cotton 40°C', match_correct: true, confidence: 0.82, expected_s: 1720, overrun_ratio: 1.01, projected_energy_wh: 485, projected_cost: 2.5 },
  },
  'ha_washdata/run_playground_history': {
    rows: [
      { cycle_id: 'cyc-001', label: 'Cotton 40°C', detected: true, detected_count: 1, matched_profile: 'Cotton 40°C', match_correct: true, confidence: 0.82, termination_reason: 'smart', duration_s: 1740, expected_s: 1720, overrun_ratio: 1.01, alerts: [] },
    ],
    summary: { cycles: 1, detected: 1, labelled: 1, match_correct: 1, match_wrong: 0, unmatched: 0, false_end: 0 },
  },
  'ha_washdata/run_playground_sweep': {
    param: 'off_delay', objective: 'match_accuracy', current_value: 180, best_value: 120, best_metric: 0.9,
    points: [{ value: 120, metric: 0.9, summary: {} }, { value: 180, metric: 0.8, summary: {} }],
  },
  'ha_washdata/get_dtw_debug': {
    stage2: { correlation: 0.91, mae_score: 0.88, score: 0.90 },
    dtw: { l1_score: 0.87, ddtw_score: 0.85, blend_weight: 0.7, blended_score: 0.86 },
    stage4: { duration_agreement: 0.94, energy_agreement: 0.88, final_score: 0.91 },
    cycle_trace: Array.from({ length: 30 }, (_, i) => Math.sin(i / 5) * 500 + 600),
    profile_trace: Array.from({ length: 30 }, (_, i) => Math.sin(i / 5) * 480 + 590),
    warp_path: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
  },
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
export { IDLE_POWER_HISTORY, EMPTY_MAINTENANCE, EMPTY_CHANGELOG, EMPTY_ML_STATUS };
