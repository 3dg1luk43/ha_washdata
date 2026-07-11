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

window.__create_mock_hass = function (extra) {
  extra = extra || {};
  return {
    connection: {
      sendMessagePromise: function (msg) {
        window.__ws_calls.push(msg);
        const err = window.__ws_errors[msg.type];
        if (err) return Promise.reject(err);
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
      subscribeMessage: function (_cb, _msg) {
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
