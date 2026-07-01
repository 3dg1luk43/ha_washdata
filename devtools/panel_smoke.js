#!/usr/bin/env node
/*
 * Panel render smoke test. Stubs a minimal DOM, loads the panel module (catches
 * top-level TDZ/init errors), instantiates the element, seeds representative
 * state, and calls every tab renderer + the modals so template ReferenceErrors
 * surface without a browser. Exit non-zero on any failure.
 *
 *   node devtools/panel_smoke.js
 */
'use strict';
const path = require('path');

function fakeEl() {
  const el = {
    style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
    children: [], attributes: {},
    appendChild() {}, setAttribute() {}, removeAttribute() {}, addEventListener() {},
    querySelector() { return null; }, querySelectorAll() { return []; },
    getElementById() { return null; }, focus() {}, setSelectionRange() {},
    getContext() { return null; }, getBoundingClientRect() { return { width: 300, height: 150 }; },
    set innerHTML(_v) {}, get innerHTML() { return ''; },
  };
  return el;
}

global.HTMLElement = class { attachShadow() { return fakeEl(); } addEventListener() {} };
global.customElements = { _cls: null, get() { return null; }, define(_n, c) { this._cls = c; } };
global.document = {
  createElement: () => fakeEl(), createElementNS: () => fakeEl(),
  head: fakeEl(), body: fakeEl(), addEventListener() {},
};
global.window = { matchMedia: () => ({ matches: false, addEventListener() {} }), addEventListener() {}, location: {} };
global.navigator = { language: 'en' };

let failures = 0;
function check(label, fn) {
  try { fn(); console.log('  ok   ' + label); }
  catch (e) { failures++; console.log('  FAIL ' + label + ' -> ' + e.message); }
}

// 1. Module load (top-level init / TDZ).
let Panel;
try {
  require(path.resolve(__dirname, '../custom_components/ha_washdata/www/ha-washdata-panel.js'));
  Panel = global.customElements._cls;
  if (!Panel) throw new Error('customElements.define was not called');
  console.log('module load: ok');
} catch (e) {
  console.log('module load: FAIL -> ' + e.message);
  process.exit(1);
}

// 2. Instantiate + seed representative state.
const el = new Panel();
el.shadowRoot = fakeEl();
el._container = fakeEl();
el._hass = { states: { 'sensor.p': { state: '5' } } };
el._constants = { stateColors: {}, deviceTypes: [['washing_machine', 'Washing Machine']], mlLabEnabled: true, mlSuggestionsEnabled: true, mlTrainingAvailable: true };
el._panelCfg = { is_admin: true, panel: {}, prefs: {}, rbac: { enabled: false }, users: [] };
el._devices = [{ entry_id: 'e', title: 'Washer', detector_state: 'running', options: { device_type: 'washing_machine' }, current_power_w: 500, cycle_progress_pct: 40, time_remaining_s: 600, suggestions_count: 1, feedback_count: 1 }];
el._selIdx = 0;
el._profiles = [{ name: 'Cotton 60', avg_duration: 1000, avg_energy: 0.5, cycle_count: 4 }, { name: 'Cotton 40', avg_duration: 1010, avg_energy: 0.4, cycle_count: 3 }, { name: 'Quick', avg_duration: 300, avg_energy: 0.1, cycle_count: 2 }];
el._profileGroups = { groups: [{ name: 'Cotton 2:47', members: ['Cotton 60', 'Cotton 40'], cohesion: 0.93, cohesive: true }, { name: 'Loose', members: ['Cotton 60', 'Quick'], cohesion: 0.4, cohesive: false }], suggestions: [{ members: ['Cotton 60', 'Cotton 40'], existing_group: null }], min_cohesion: 0.85 };
el._opts = { device_type: 'washing_machine', min_power: 2, off_delay: 180, abrupt_drop_watts: 500, notify_actions: [{ action: 'X', title: 'Y' }], notify_people: ['person.a'], notify_start_services: ['notify.phone'] };
el._cycles = [{ id: 'c1', profile_name: 'Cotton 60', status: 'completed', duration: 1000, start_time: new Date().toISOString(), energy_wh: 500, match_confidence: 0.8 }, { id: 'c2', profile_name: null, status: 'force_stopped', duration: 200, start_time: new Date().toISOString() }];
el._suggestions = [{ key: 'off_delay', suggested: 120, current: 180, reason: 'test' }];
el._mlSettings = { off_delay: { ml_value: 130, ml_reason: 'ml' } };
el._mlById = {};
el._mlTrainingStatus = { available: true, enabled: true, running: false, last_trained: null, cycle_count: 10, min_cycles: 30, hour: 2, on_device_models: {},
  matching: { defaults: { corr_weight: 0.45, duration_weight: 0.22, energy_weight: 0.22, dtw_ensemble_w: 0.7 }, tuned: { config: { corr_weight: 0.5, duration_weight: 0.15, energy_weight: 0.15, dtw_ensemble_w: 0.85 }, trained_at: new Date().toISOString(), cycle_count: 40, baseline_test_top1: 0.7, tuned_test_top1: 0.8 }, active: 'tuned' } };
el._powerData = { live: [], raw: [], cycle_active: true };
el._diag = { total_cycles: 10, total_profiles: 3, debug_traces_count: 0, file_size_kb: 12.3 };
el._logs = [];
el._feedbacks = [{ cycle_id: 'c2', detected_profile: 'Cotton 60', confidence: 0.5 }];

// 3. Exercise the renderers.
check('_htmlHeader', () => el._htmlHeader());
check('_htmlDeviceBar', () => el._htmlDeviceBar());
check('_htmlBody (all panes)', () => el._htmlBody());
check('_htmlStatus', () => el._htmlStatus());
check('_htmlHistory', () => el._htmlHistory());
check('_htmlProfiles', () => el._htmlProfiles());
check('_htmlSettings', () => el._htmlSettings());
check('_htmlSettings (search)', () => { el._settingsSearch = 'threshold'; const h = el._htmlSettings(); el._settingsSearch = ''; return h; });
check('_htmlPanel (advanced)', () => el._htmlPanel());
check('_htmlDiagnostics', () => el._htmlDiagnostics());
check('_htmlLogs', () => el._htmlLogs());
check('_htmlMlTrainingCard', () => el._htmlMlTrainingCard());
check('_htmlMatchingTuningCard', () => el._htmlMatchingTuningCard());
check('_htmlMatchingTuningCard (default)', () => { const s = el._mlTrainingStatus; el._mlTrainingStatus = { ...s, matching: { ...s.matching, tuned: null, active: 'default' } }; const h = el._htmlMatchingTuningCard(); el._mlTrainingStatus = s; return h; });
check('_buildHtml', () => el._buildHtml());

// Modals
check('modal: profile-group (new)', () => { el._modal = { type: 'profile-group', orig: null, name: '', members: [] }; return el._htmlModal(); });
check('modal: profile-group (edit)', () => { el._modal = { type: 'profile-group', orig: 'Cotton 2:47', name: 'Cotton 2:47', members: ['Cotton 60', 'Cotton 40'] }; return el._htmlModal(); });
check('modal: cycle-detail review', () => { el._modal = { type: 'cycle-detail', mode: 'review', loaded: true, cycleId: 'c1', curve: { samples: [[0, 1], [10, 2]], full_duration_s: 1000, duration: 1000, profile_name: 'Cotton 60', start_time: new Date().toISOString() }, ml: null }; return el._htmlModal(); });
check('modal: profile-panel stats', () => { el._modal = { type: 'profile-panel', name: 'Cotton 60', tab: 'stats', loaded: true, stats: el._profiles[0], env: {} }; return el._htmlModal(); });
check('modal: notify-actions (rows)', () => { el._modal = { type: 'notify-actions', mode: 'rows', rows: [{ service: 'notify.phone', data: '{"message":"hi"}' }], raw: '[]' }; return el._htmlModal(); });
check('modal: notify-actions (raw)', () => { el._modal = { type: 'notify-actions', mode: 'raw', rows: [], raw: '[{"service":"notify.phone"}]' }; return el._htmlModal(); });
check('_naBuildActions rows', () => { el._modal = { type: 'notify-actions', mode: 'rows', rows: [{ service: 'notify.phone', data: '{"message":"x"}' }] }; const a = el._naBuildActions(el._modal); if (!Array.isArray(a) || a[0].service !== 'notify.phone') throw new Error('bad build'); });
el._modal = null;

console.log(failures ? `\nSMOKE FAILED (${failures})` : '\nSMOKE OK');
process.exit(failures ? 1 : 0);
