#!/usr/bin/env node
// WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
// Copyright (C) 2026 Lukas Bandura
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
/*
 * Panel render smoke test. Stubs a minimal DOM, loads the panel module (catches
 * top-level TDZ/init errors), instantiates the element, seeds representative
 * state, and calls every tab renderer + the modals so template ReferenceErrors
 * surface without a browser. Exit non-zero on any failure.
 *
 *   node devtools/panel_smoke.js
 *
 * ESM, not CommonJS: devtools/package.json declares "type": "module" (the panel build
 * script needs it), which makes Node parse every .js in this directory as a module.
 */
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

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
function near(a, b, msg) { if (Math.abs(a - b) > 1e-6) throw new Error((msg || 'value') + ': ' + a + ' != ' + b); }

// Load the panel as an ES module (matching production, where HA imports it via
// import()), so import.meta.url resolves. Wrap the rest in an async IIFE.
(async () => {
// 1. Module load (top-level init / TDZ).
let Panel;
try {
  await import(pathToFileURL(path.resolve(__dirname, '../custom_components/ha_washdata/www/ha-washdata-panel.js')).href);
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
el._mlTrainingStatus = { available: true, enabled: true, running: false, last_trained: new Date().toISOString(), cycle_count: 35, min_cycles: 30, hour: 2,
  on_device_models: {
    end: { trained_at: new Date().toISOString(), cycle_count: 40, kind: 'standardized_logistic', label: 'Cycle-end detection', blurb: "Knowing when a cycle has truly finished", auc: 0.91, metric: 'AUC 0.91 on held-out data', trend: 'improving' },
    remaining_time: { trained_at: new Date().toISOString(), cycle_count: 40, kind: 'standardized_linear', label: 'Time-remaining estimate', blurb: 'Predicting how long is left', model_mae: 0.02, naive_mae: 0.12, metric: 'error 0.020 vs 0.120 baseline', trend: 'declining' },
  },
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
check('_htmlAutomations', () => { el._deviceAutomations = [{ id: 'abc', name: 'Notify on finish', enabled: true }, { id: 'def', name: 'Off automation', enabled: false }]; return el._htmlAutomations(); });
check('_htmlAutomations (legacy actions)', () => { const o = el._opts; el._opts = { ...o, notify_actions: [{ service: 'notify.phone', data: { message: 'done' } }] }; const h = el._htmlAutomations(); el._opts = o; return h; });
check('_htmlSettings (notifications section)', () => { const s = el._settingsSec; el._settingsSec = 'notifications'; const h = el._htmlSettings(); el._settingsSec = s; return h; });
check('_htmlPanel (advanced)', () => el._htmlPanel());
check('_htmlDiagnostics', () => el._htmlDiagnostics());
check('_htmlLogDrawer', () => { const o = el._logOpen; el._logOpen = true; const h = el._htmlLogDrawer(); el._logOpen = o; return h; });
check('_htmlMlTab', () => el._htmlMlTab());
check('_htmlMlStatusSection', () => el._htmlMlStatusSection(el._mlTrainingStatus, 'entry-1'));
check('_htmlMlLearnedSection', () => el._htmlMlLearnedSection(el._mlTrainingStatus));
check('_htmlMlLearnedSection (empty)', () => { const s = el._mlTrainingStatus; el._mlTrainingStatus = { ...s, on_device_models: {} }; const h = el._htmlMlLearnedSection(el._mlTrainingStatus); el._mlTrainingStatus = s; return h; });
check('_htmlMatchingTuningCard', () => el._htmlMatchingTuningCard());
check('_htmlMatchingTuningCard (default)', () => { const s = el._mlTrainingStatus; el._mlTrainingStatus = { ...s, matching: { ...s.matching, tuned: null, active: 'default' } }; const h = el._htmlMatchingTuningCard(); el._mlTrainingStatus = s; return h; });
// Playground (unified workbench + drawer, with and without backend data)
check('_htmlPlayground (workbench, empty)', () => { el._pgAnalysisTab = 'history'; return el._htmlPlayground(); });
check('_htmlPlayground (workbench, with detail)', () => {
  el._pgAnalysisTab = 'history';
  el._pgCycleId = 'c1';
  el._pgPowerPts = [{ t: 0, w: 5 }, { t: 500, w: 900 }, { t: 1000, w: 3 }];
  el._pgDetail = {
    cycle_id: 'c1', label: 'Cotton 60', duration_s: 1000,
    series: [
      { t: 0, power: 5, energy_wh: 0, state: 'starting', progress: null, remaining_s: null, phase: null, confidence: null, matched_profile: null },
      { t: 300, power: 900, energy_wh: 80, state: 'running', progress: 30, remaining_s: 700, phase: 'Wash', confidence: 0.72, matched_profile: 'Cotton 60', projected_energy_wh: 500, projected_cost: 3 },
      { t: 1000, power: 3, energy_wh: 500, state: 'ending', progress: 96, remaining_s: 40, phase: 'Spin', confidence: 0.8, matched_profile: 'Cotton 60', projected_energy_wh: 505, projected_cost: 3 },
    ],
    events: [
      { t: 30, type: 'detected', detail: 'cycle detected', severity: 'info' },
      { t: 200, type: 'match_commit', detail: 'Cotton 60 (0.7)', severity: 'info' },
      { t: 900, type: 'notify_pre_complete', detail: 'almost done', severity: 'info' },
      { t: 1000, type: 'finished', detail: 'reason=smart', severity: 'info' },
    ],
    alerts: [{ code: 'overrun', severity: 'warn', detail: 'Ran 110% of typical.' }],
    outcome: { detected: true, detected_count: 1, termination_reason: 'smart', status: 'completed', final_duration_s: 1000, matched_profile: 'Cotton 60', match_correct: true, overrun_ratio: 1.1, projected_energy_wh: 505, projected_cost: 3 },
  };
  const h = el._htmlPlayground(); el._pgUpdateStripAt(500); el._pgUpdateStripAt(null); return h;
});
check('_htmlPlayground (drawer: history)', () => {
  el._pgAnalysisTab = 'history';
  el._pgHistory = {
    rows: [
      { cycle_id: 'c1', label: 'Cotton 60', detected: true, detected_count: 1, matched_profile: 'Cotton 60', match_correct: true, confidence: 0.8, termination_reason: 'smart', duration_s: 1000, expected_s: 1000, overrun_ratio: 1.0, alerts: [] },
      { cycle_id: 'c2', label: 'Cotton 40', detected: true, detected_count: 1, matched_profile: 'Cotton 60', match_correct: false, confidence: 0.5, termination_reason: 'timeout', duration_s: 1400, expected_s: 1010, overrun_ratio: 1.38, alerts: ['overrun'] },
    ],
    summary: { cycles: 2, detected: 2, labelled: 2, match_correct: 1, match_wrong: 1, unmatched: 0, false_end: 0 },
    baseline_rows: [
      { cycle_id: 'c1', label: 'Cotton 60', match_correct: true, termination_reason: 'smart', duration_s: 1000 },
      { cycle_id: 'c2', label: 'Cotton 40', match_correct: false, termination_reason: 'timeout', duration_s: 1400 },
    ],
    baseline_summary: { cycles: 2, detected: 2, labelled: 2, match_correct: 1 },
    diff: { newly_correct: [], regressed: [], end_timing_changed: ['c2'] },
  };
  return el._htmlPlayground();
});
check('_htmlPlayground (drawer: sweep 1D)', () => {
  el._pgAnalysisTab = 'sweep';
  el._pgSweepNew = { param: 'off_delay', objective: 'match_accuracy', current_value: 180, best_value: 120, best_metric: 0.9,
    points: [{ value: 120, metric: 0.9, summary: {} }, { value: 180, metric: 0.8, summary: {} }, { value: 240, metric: 0.7, summary: {} }] };
  return el._htmlPlayground();
});
el._pgAnalysisTab = 'history';

check('_buildHtml', () => el._buildHtml());

// Modals
check('modal: profile-group (new)', () => { el._modal = { type: 'profile-group', orig: null, name: '', members: [] }; return el._htmlModal(); });
check('modal: profile-group (edit)', () => { el._modal = { type: 'profile-group', orig: 'Cotton 2:47', name: 'Cotton 2:47', members: ['Cotton 60', 'Cotton 40'] }; return el._htmlModal(); });
check('modal: cycle-detail review', () => { el._modal = { type: 'cycle-detail', mode: 'review', loaded: true, cycleId: 'c1', curve: { samples: [[0, 1], [10, 2]], full_duration_s: 1000, duration: 1000, profile_name: 'Cotton 60', start_time: new Date().toISOString(), artifacts: [{ type: 'pause', start_s: 300, end_s: 420, detail: 'Power dropped to near zero for ~120s then resumed — likely the door was opened.', severity: 0.4 }] }, ml: null }; return el._htmlModal(); });
check('modal: cycle-detail inspect (draw w/ artifacts)', () => { el._modal = { type: 'cycle-detail', mode: 'view', loaded: true, cycleId: 'c1', curve: { samples: [[0, 900], [300, 950], [360, 3], [420, 940], [1000, 900]], full_duration_s: 1000, profile_name: 'Cotton 60', start_time: new Date().toISOString(), artifacts: [{ type: 'pause', start_s: 300, end_s: 420, detail: 'door opened', severity: 0.4 }] }, ml: null }; const h = el._htmlModal(); el._drawCycleEditor(); return h; });
check('modal: cycle-detail inspect (with restart gap)', () => { const now = new Date(); el._modal = { type: 'cycle-detail', mode: 'view', loaded: true, cycleId: 'c2', curve: { samples: [[0, 900], [200, 950], [600, 920], [1000, 10]], full_duration_s: 1000, profile_name: 'Cotton 60', start_time: new Date(now - 1000000).toISOString(), restart_gaps: [{ start_ts: new Date(now - 600000).toISOString(), end_ts: new Date(now - 400000).toISOString(), gap_seconds: 200, profile: 'Cotton 60', match_confidence: 0.78 }] }, ml: null }; const h = el._htmlModal(); el._drawCycleEditor(); return h; });
check('modal: profile-panel stats', () => { el._modal = { type: 'profile-panel', name: 'Cotton 60', tab: 'stats', loaded: true, stats: el._profiles[0], env: {} }; return el._htmlModal(); });
check('modal: compare-cycles (html + draw)', () => {
  el._cycles = [
    { id: 'c1', start_time: new Date().toISOString(), duration: 1000, profile_name: 'Cotton 60' },
    { id: 'c2', start_time: new Date().toISOString(), duration: 1200, profile_name: 'Cotton 40' },
  ];
  el._modal = {
    type: 'compare-cycles', ids: ['c1', 'c2'], hidden: new Set(['c2']), overlays: ['Cotton 60'], loaded: true,
    cycles: { c1: { samples: [[0, 900], [500, 800], [1000, 5]], full_duration_s: 1000 }, c2: { samples: [[0, 950], [1200, 3]], full_duration_s: 1200 } },
  };
  const h = el._htmlModal(); el._drawCompareCanvas(); return h;
});
check('modal: store-share-device (tree)', () => {
  el._shareableCycles = [
    { id: 'g1', start_time: new Date().toISOString(), duration: 3600, profile_name: 'Cotton 60', source: 'recorder' },
    { id: 'g2', start_time: new Date().toISOString(), duration: 1800, profile_name: 'Eco 40', source: 'recorder' },
  ];
  el._sharePhasePrograms = ['Cotton 60'];
  el._modal = { type: 'store-share-device', selected: new Set(['g1']), includePhases: new Set(['Cotton 60']), includeSettings: true };
  return el._htmlModal();
});
check('modal: store-share-device (empty)', () => { el._shareableCycles = []; el._sharePhasePrograms = []; el._modal = { type: 'store-share-device', selected: new Set(), includePhases: new Set(), includeSettings: false }; return el._htmlModal(); });

// #344 import wizard: every step, including the empty-result and error paths.
const _histBase = { type: 'history-import', csvText: '', days: 10, token: 't', scanTaskId: null, applyTaskId: null, result: null, accept: new Set(), done: null, error: null };
check('modal: history-import input', () => { el._modal = { ..._histBase, step: 'input' }; return el._htmlModal(); });
check('modal: history-import input (error)', () => { el._modal = { ..._histBase, step: 'input', error: 'entity_not_in_file' }; return el._htmlModal(); });
check('modal: history-import scanning', () => { el._modal = { ..._histBase, step: 'scan', scanTaskId: 'task1' }; el._tasks = { task1: { id: 'task1', kind: 'history_import', done: 40, total: 100, state: 'running' } }; return el._htmlModal(); });
check('modal: history-import review', () => {
  el._modal = { ..._histBase, step: 'review', accept: new Set([0]), result: {
    segments: [
      { index: 0, start_time: new Date().toISOString(), duration_s: 4464, status: 'completed', peak_w: 2165, energy_wh: 1096, accept: true, reason: null, curve: [10, 900, 1800, 400, 5] },
      { index: 1, start_time: new Date().toISOString(), duration_s: 300, status: 'interrupted', peak_w: 400, energy_wh: 20, accept: false, reason: 'shorter_than_minimum', curve: [5, 40, 10] },
    ],
    skipped: [{ reason: 'idle', span_s: 3600 }, { reason: 'sparse', span_s: 86400 }],
    parse: { rows_total: 2358, first: new Date().toISOString(), last: new Date().toISOString(), breaks: 8, rows_other_entity: 0 },
    settings: { min_power: 2, off_delay: 300, min_off_gap: 480, device_type: 'washing_machine' },
    found: 2, capped: false,
  } };
  const h = el._htmlModal();
  el._drawHistorySparklines();
  return h;
});
check('modal: history-import review (nothing found)', () => { el._modal = { ..._histBase, step: 'review', result: { segments: [], skipped: [{ reason: 'sparse', span_s: 86400 }], parse: { rows_total: 114 } } }; return el._htmlModal(); });
check('modal: history-import done', () => { el._modal = { ..._histBase, step: 'done', done: { imported: 3, duplicates: 1, capped: false } }; return el._htmlModal(); });

// Trim clock-mode + sample-snapping guards (issue #373). Build start_time from
// local Date components so getHours()/setHours() stay timezone-agnostic.
check('trim: _clockToOffset day-shift guard (#373 B)', () => {
  const base = new Date(2026, 7, 8, 21, 0, 0);  // local 21:00:00, cycle runs 2h33m
  el._modal = { type: 'cycle-detail', mode: 'trim', loaded: true, curve: { start_time: base.toISOString(), full_duration_s: 9180 } };
  near(el._clockToOffset('21:00:00'), 0, 'exact start');
  near(el._clockToOffset('21:30:00'), 1800, 'after start');
  near(el._clockToOffset('20:59:58'), 0, 'two seconds early must clamp to 0, not wrap to end');
  near(el._clockToOffset('20:44:00'), 0, 'early start (the #366 field value) must clamp to 0');
});
check('trim: _clockToOffset legit past-midnight still shifts (#373 B)', () => {
  const base = new Date(2026, 7, 8, 23, 0, 0);  // local 23:00, cycle runs 2h -> 01:00 next day
  el._modal = { type: 'cycle-detail', mode: 'trim', loaded: true, curve: { start_time: base.toISOString(), full_duration_s: 7200 } };
  near(el._clockToOffset('00:30:00'), 5400, 'past-midnight time inside the cycle must shift +24h');
});
check('trim: _snapTrimBounds snaps to real sample offsets (#373 A)', () => {
  el._modal = { type: 'cycle-detail', mode: 'trim', loaded: true, trim: { start: 9, end: 29 },
    curve: { samples: [[0, 1], [10.3, 2], [20.6, 3], [30.9, 4]], full_duration_s: 30.9 } };
  el._snapTrimBounds();
  near(el._modal.trim.start, 10.3, 'start snapped');
  near(el._modal.trim.end, 30.9, 'end snapped');
});
el._modal = null;

console.log(failures ? `\nSMOKE FAILED (${failures})` : '\nSMOKE OK');
process.exit(failures ? 1 : 0);
})();
