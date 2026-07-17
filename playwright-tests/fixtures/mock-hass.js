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
'use strict';

/**
 * Mock HA hass object for WashData Playwright tests.
 *
 * Usage from Playwright tests:
 *   await page.evaluate((data) => window.__boot_panel(data.handlers, data.hassExtra), { handlers, hassExtra });
 *
 * The panel's only external dependency is:
 *   this._hass.connection.sendMessagePromise(msg)  →  Promise
 * We mock that by a simple lookup in window.__ws_handlers.
 *
 * Poll interference prevention: tests should override __ws_handlers after boot
 * so polls return stable data. Or call window.__freeze_poll() to pause polling.
 */

window.__ws_handlers = {};
window.__ws_calls = [];       // log of all WS calls for test assertions
window.__ws_errors = {};      // command → error to throw (simulates backend failures)
window.__ws_tasks = [];       // background-task snapshots (registry mock)
window.__ws_task_subs = [];   // active subscribe_tasks callbacks

// Push/replace a task snapshot and notify subscribers (mirrors the real registry).
window.__emit_task = function (task) {
  const i = window.__ws_tasks.findIndex((t) => t.id === task.id);
  if (i >= 0) window.__ws_tasks[i] = task; else window.__ws_tasks.push(task);
  (window.__ws_task_subs || []).forEach((cb) => { try { cb({ type: 'task', task: task }); } catch (e) { /* ignore */ } });
};

window.__create_mock_hass = function (extra) {
  extra = extra || {};
  return {
    connection: {
      sendMessagePromise: function (msg) {
        window.__ws_calls.push(msg);
        const err = window.__ws_errors[msg.type];
        if (err) return Promise.reject(err);
        // Background-task registry mock: task-start commands kick off a task that
        // completes ~immediately (setTimeout 0), emitting a done `task` event to
        // subscribers and storing the result for get_task_result. Reuses existing
        // static handler data as the task payload where applicable.
        var TASK_START = {
          'ha_washdata/start_playground_history': { kind: 'pg_history', resKey: 'ha_washdata/run_playground_history' },
          'ha_washdata/start_playground_sweep': { kind: 'pg_sweep', resKey: 'ha_washdata/run_playground_sweep' },
          'ha_washdata/start_playground_cycle_detail': { kind: 'pg_detail', resKey: 'ha_washdata/run_playground_cycle_detail' },
          'ha_washdata/reprocess_history': { kind: 'reprocess', resKey: 'ha_washdata/reprocess_history' },
          'ha_washdata/trigger_ml_training': { kind: 'ml_training', resKey: 'ha_washdata/trigger_ml_training' },
        };
        if (TASK_START[msg.type]) {
          const spec = TASK_START[msg.type];
          // Monotonic counter (incremented synchronously per start) so consecutive
          // same-kind starts get distinct ids and never overwrite __task_results.
          window.__task_seq = (window.__task_seq || 0) + 1;
          const id = 'task-' + window.__task_seq + '-' + spec.kind;
          // Register a RUNNING task synchronously and return its id (mirrors the real
          // registry's reg.create running before the start request's WS reply), so an
          // immediate list_tasks after awaiting the start promise already includes it.
          const running = {
            id: id, entry_id: msg.entry_id, kind: spec.kind, label: spec.kind,
            state: 'running', done: 0, total: 1, progress: 0, eta_s: null,
            has_result: false, updated_at: 0,
          };
          window.__task_results[id] = running;
          const ti = window.__ws_tasks.findIndex((t) => t.id === id);
          if (ti >= 0) window.__ws_tasks[ti] = running; else window.__ws_tasks.push(running);
          // Resolve the result payload with the SAME function-vs-static dispatch as the
          // generic path, normalized through Promise.resolve so a sync value, a Promise,
          // or a synchronous throw are all handled. Then flip the task to done (or error)
          // and emit the terminal event - mirroring the detached runner's lifecycle.
          const rh = window.__ws_handlers[spec.resKey];
          Promise.resolve().then(function () { return typeof rh === 'function' ? rh(msg) : rh; }).then(
            function (res) {
              const done = Object.assign({}, running, {
                state: 'done', done: 1, progress: 1, has_result: true,
                updated_at: Date.now() / 1000, finished_at: Date.now() / 1000, result: res || {},
              });
              window.__task_results[id] = done;
              window.__emit_task(done);
            },
            function (err) {
              const errored = Object.assign({}, running, {
                state: 'error', has_result: false,
                updated_at: Date.now() / 1000, finished_at: Date.now() / 1000,
                error: String((err && err.message) || err),
              });
              window.__task_results[id] = errored;
              window.__emit_task(errored);
            }
          );
          return Promise.resolve({ task_id: id });
        }
        if (msg.type === 'ha_washdata/get_task_result') {
          const t = window.__task_results[msg.task_id];
          return t ? Promise.resolve(t) : Promise.reject({ code: 'not_found', message: 'Task not found' });
        }
        if (msg.type === 'ha_washdata/list_tasks') {
          return Promise.resolve({ tasks: (window.__ws_tasks || []).slice() });
        }
        if (msg.type === 'ha_washdata/cancel_task') {
          return Promise.resolve({ cancelled: true });
        }
        const h = window.__ws_handlers[msg.type];
        if (h == null) {
          console.warn('[mock-hass] No handler for', msg.type);
          return Promise.reject({ code: 'unknown_command', message: 'No mock handler: ' + msg.type });
        }
        try {
          const result = typeof h === 'function' ? h(msg) : h;
          return Promise.resolve(result);
        } catch (e) {
          return Promise.reject(e);
        }
      },
      subscribeMessage: function (cb, msg) {
        if (msg && msg.type === 'ha_washdata/subscribe_tasks') {
          window.__ws_task_subs.push(cb);
          // Emit the current snapshot (like the real subscribe_tasks handler).
          (window.__ws_tasks || []).forEach((t) => { try { cb({ type: 'task', task: t }); } catch (e) { /* ignore */ } });
          return Promise.resolve(function () {
            window.__ws_task_subs = (window.__ws_task_subs || []).filter((f) => f !== cb);
          });
        }
        return Promise.resolve(function () {});
      },
    },
    localize: function () { return ''; },
    locale: { language: 'en', time_format: '24' },
    config: { currency: 'EUR', unit_system: { temperature: '°C', length: 'km' } },
    user: { name: 'Admin', is_admin: true },
    themes: { theme: 'default', darkMode: false },
    devices: {},
    states: {},
    entities: {},
  };
};

window.__boot_panel = function (handlers, hassExtra) {
  handlers = handlers || {};
  hassExtra = hassExtra || {};

  // Remove any existing panel to allow re-boot in the same page.
  const old = document.querySelector('ha-washdata-panel');
  if (old) old.remove();

  window.__ws_handlers = handlers;
  window.__ws_calls = [];
  window.__ws_errors = {};
  window.__ws_tasks = [];
  window.__ws_task_subs = [];
  window.__task_results = {};
  window.__task_seq = 0;

  const el = document.createElement('ha-washdata-panel');
  el.setAttribute('id', 'wd-panel');
  document.body.appendChild(el);
  el.hass = window.__create_mock_hass(hassExtra);
  return el;
};

/** Replace a handler at runtime (for post-boot state changes). */
window.__set_handler = function (type, handler) {
  window.__ws_handlers[type] = handler;
};

/** Simulate an error response for a command. */
window.__set_error = function (type, err) {
  window.__ws_errors[type] = err || { code: 'unknown_error', message: 'Simulated error' };
};

/** Return calls made to a specific WS command. */
window.__get_calls = function (type) {
  return window.__ws_calls.filter(function (c) { return c.type === type; });
};

/**
 * Pause the panel's 5-second polling loop so tests don't fight re-renders.
 * Call this after the initial render settles.
 */
window.__freeze_poll = function () {
  const el = document.getElementById('wd-panel');
  if (el && el._pollTimer) { clearTimeout(el._pollTimer); el._pollTimer = null; }
  // Patch _scheduleRefresh so future calls are no-ops.
  if (el) el._scheduleRefresh = function () {};
};
