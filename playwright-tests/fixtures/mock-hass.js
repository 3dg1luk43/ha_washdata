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
        // Background-task registry mock: start_* kicks off a task that completes
        // ~immediately (setTimeout 0), emitting a done `task` event to subscribers
        // and storing the result for get_task_result. Reuses the existing static
        // run_playground_* result data as the task payload.
        if (msg.type === 'ha_washdata/start_playground_history' ||
            msg.type === 'ha_washdata/start_playground_sweep') {
          const isHist = msg.type.slice(-7) === 'history';
          const resKey = isHist ? 'ha_washdata/run_playground_history' : 'ha_washdata/run_playground_sweep';
          const result = window.__ws_handlers[resKey] || {};
          const id = 'task-' + (window.__ws_tasks.length + 1) + (isHist ? '-h' : '-s');
          const task = {
            id: id, entry_id: msg.entry_id, kind: isHist ? 'pg_history' : 'pg_sweep',
            label: isHist ? 'Test on history' : 'Optimize', state: 'done',
            done: 1, total: 1, progress: 1, eta_s: null, has_result: true,
            updated_at: Date.now() / 1000, result: result,
          };
          window.__task_results[id] = task;
          setTimeout(function () { window.__emit_task(task); }, 0);
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
