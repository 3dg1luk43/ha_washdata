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
