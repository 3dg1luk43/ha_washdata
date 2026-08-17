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
const CARD_TAG = "ha-washdata-card";
const EDITOR_TAG = "ha-washdata-card-editor";
const DOMAIN = "ha_washdata";

// Gesture timing (ms) and movement tolerance (px) for tap / hold / double-tap,
// chosen to match Home Assistant's own action handler conventions.
const HOLD_MS = 500;
const DOUBLE_TAP_MS = 250;
const TAP_MOVE_TOLERANCE = 10;

// Sibling entities of a WashData device are resolved from the device registry by
// their stable translation_key (which never changes even if the user renames the
// entity), so pointing the card at any one entity auto-wires all the rest.
const ROLE_TKEYS = {
  state: "washer_state",
  program: "washer_program",
  time: "time_remaining",
  progress: "cycle_progress",
  power: "current_power",
  energy: "energy_total",
  phase: "current_phase",
};
const BUTTON_TKEYS = {
  pause: "pause_cycle",
  resume: "resume_cycle",
  terminate: "force_end_cycle",
  record_start: "record_start",
  record_stop: "record_stop",
};
const BUTTON_ICONS = {
  pause: "mdi:pause",
  resume: "mdi:play",
  terminate: "mdi:stop",
  record_start: "mdi:record-circle-outline",
  record_stop: "mdi:stop-circle-outline",
  open_panel: "mdi:open-in-new",
};
const BUTTON_ORDER = ["pause", "resume", "terminate", "record_start", "record_stop", "program", "open_panel"];

const ACTIVE_STATES = ["running", "paused", "user_paused", "ending", "starting", "anti_wrinkle", "rinse"];
const INACTIVE_STATES = ["off", "unknown", "unavailable", "idle"];
const SPARK_MAX_POINTS = 60;

// ── Shared, module-level resource caches ────────────────────────────────────
// One WebSocket round-trip per HA connection for the shared display constants
// (STATE_COLORS is the single source of truth in const.py), and one fetch per
// language for the panel translation dictionaries the card reuses for its own
// chrome. Both are shared across every card instance on the page.
const _constantsCache = new WeakMap(); // hass.connection -> Promise<{stateColors}>
function loadConstants(hass) {
  const conn = hass && hass.connection;
  if (!conn) return Promise.resolve({ stateColors: {} });
  if (_constantsCache.has(conn)) return _constantsCache.get(conn);
  const p = hass
    .callWS({ type: DOMAIN + "/get_constants" })
    .then((c) => ({ stateColors: (c && c.state_colors) || {} }))
    .catch(() => ({ stateColors: {} }));
  _constantsCache.set(conn, p);
  return p;
}

const _panelTrans = {}; // lang -> dict
const _panelTransPending = {}; // lang -> Promise
function _panelTransUrl(lang) {
  return "/" + DOMAIN + "/panel-translations/" + encodeURIComponent(lang) + ".json";
}
async function _fetchPanelLang(lang) {
  if (!lang) return null;
  const candidates = [lang];
  const dash = lang.indexOf("-");
  if (dash > 0) candidates.push(lang.slice(0, dash));
  for (const cand of candidates) {
    try {
      const r = await fetch(_panelTransUrl(cand));
      if (r.ok) {
        const j = await r.json();
        if (j && typeof j === "object") return j;
      }
    } catch (_) {
      /* try next candidate */
    }
  }
  return null;
}
async function loadPanelLang(lang) {
  if (!lang || _panelTrans[lang]) return;
  if (_panelTransPending[lang]) {
    await _panelTransPending[lang];
    return;
  }
  const p = _fetchPanelLang(lang).then((d) => {
    if (d) _panelTrans[lang] = d;
  });
  _panelTransPending[lang] = p;
  await p;
}

class WashDataCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._cfg = null;
    this._hass = null;
    this._builtSig = null; // structure signature the current shell was built for
    this._constantsLoaded = false;
    this._langEnsured = "";
    this._constants = null;
    // Rolling in-memory buffer of {t, w} power samples for the live sparkline.
    this._spark = [];
    this._sparkState = null;
    // Gesture state for tap / hold / double-tap recognition.
    this._holdTimer = null;
    this._holdTriggered = false;
    this._tapTimer = null;
    this._lastTapTime = 0;
    this._pointerStart = null;
    this._pointerCanceled = false;
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onPointerCancel = this._onPointerCancel.bind(this);
  }

  static getStubConfig() {
    return {
      entity: "sensor.washing_machine_state",
      layout: "tile",
      tap_action: { action: "more-info" },
      hold_action: { action: "none" },
      double_tap_action: { action: "none" },
    };
  }

  static getConfigElement() {
    return document.createElement(EDITOR_TAG);
  }

  setConfig(config) {
    if (!config) throw new Error("Invalid configuration");
    const layout = config.layout || "tile";
    if (layout !== "glance" && !config.entity) {
      throw new Error("Please define an entity");
    }
    if (layout === "glance" && !(Array.isArray(config.entities) && config.entities.length) && !config.entity) {
      throw new Error("Please define entities for the glance layout");
    }
    this._cfg = { ...config, layout };
    // Rebuild the shell when the structural signature changes - not only on a
    // layout switch, but also when the progress bar, sparkline, buttons, or the
    // glance entity count change (HA reuses the element and re-calls setConfig on
    // editor edits / Lovelace reloads). Otherwise the _update* helpers early-return
    // because their elements don't exist and the new options only appear after a
    // layout switch or page reload (removed entities also leave orphan glance rows).
    if (this._builtSig !== this._structureSig()) {
      this._builtSig = null;
      if (this.shadowRoot) this.shadowRoot.innerHTML = "";
    }
    this._render();
  }

  // Signature of every config field that changes which elements the shell emits
  // (see _buildTile/_buildDetail/_buildGlance). A change here forces a rebuild.
  _structureSig() {
    const c = this._cfg || {};
    const f = this._flags();
    const layout = c.layout || "tile";
    return [
      layout,
      f.showBar ? "bar" : "",
      f.showSparkline ? "spark" : "",
      f.buttons.join("+"),
      layout === "glance" ? String(this._glanceEntities().length) : "",
    ].join("|");
  }

  set hass(hass) {
    this._hass = hass;
    this._ensureResources();
    this._render();
  }

  // Reserve enough vertical space for whatever the detail layout is showing, so
  // the sections-view grid (which clips to the reserved rows) never cuts off the
  // action buttons or the sparkline at the bottom of the card.
  _detailRows() {
    const f = this._flags();
    let rows = 2; // header + progress bar + meta chips (content-hugging card)
    if (f.buttons.length) rows += 1;
    if (f.showSparkline) rows += 1;
    return rows;
  }

  getCardSize() {
    const layout = this._cfg && this._cfg.layout;
    if (layout === "detail") return this._detailRows();
    if (layout === "glance") return this._glanceEntities().length || 1;
    return 1;
  }

  getGridOptions() {
    const layout = this._cfg && this._cfg.layout;
    if (layout === "detail") {
      const rows = this._detailRows();
      return { rows, min_rows: rows, columns: 12, min_columns: 6 };
    }
    if (layout === "glance") {
      const n = Math.max(1, this._glanceEntities().length);
      return { rows: n, min_rows: 1, columns: 12, min_columns: 6 };
    }
    return { rows: 1, min_rows: 1, columns: 6, min_columns: 3 };
  }

  disconnectedCallback() {
    this._clearHoldTimer();
    if (this._tapTimer) {
      window.clearTimeout(this._tapTimer);
      this._tapTimer = null;
    }
  }

  // ── Localization ──────────────────────────────────────────────────────────
  _lang() {
    const raw =
      (this._hass && this._hass.locale && this._hass.locale.language) ||
      (this._hass && this._hass.language) ||
      "en";
    return typeof raw === "string" && raw ? raw : "en";
  }

  _tLookup(key, lang) {
    const dict = _panelTrans[lang];
    if (!dict) return null;
    const val = key.split(".").reduce((o, k) => (o && o[k] !== undefined ? o[k] : null), dict);
    return val && typeof val === "string" ? val : null;
  }

  _t(key, vars, fallback) {
    const lang = this._lang();
    let s = (lang && this._tLookup(key, lang)) || this._tLookup(key, "en") || fallback || key;
    if (vars) {
      for (const [k, v] of Object.entries(vars)) {
        s = s.replace(new RegExp("\\{" + k + "\\}", "g"), String(v));
      }
    }
    return s;
  }

  _ensureResources() {
    if (!this._hass) return;
    const lang = this._lang();
    const needsConstants = !this._constantsLoaded;
    const needsLang = lang !== this._langEnsured;
    if (!needsConstants && !needsLang) return;
    const tasks = [];
    if (needsConstants) {
      this._constantsLoaded = true;
      tasks.push(
        loadConstants(this._hass).then((c) => { this._constants = c; }),
        loadPanelLang("en"),
      );
    }
    if (needsLang) {
      this._langEnsured = lang;
      if (lang && lang !== "en") tasks.push(loadPanelLang(lang));
    }
    Promise.all(tasks)
      .then(() => this._update())
      .catch(() => {
        /* card degrades gracefully to English fallbacks + theme colors */
      });
  }

  // ── Entity discovery ──────────────────────────────────────────────────────
  _resolveRoles(primaryEntity) {
    const hass = this._hass;
    const cfg = this._cfg || {};
    const reg = (hass && hass.entities) || {};
    const primary = primaryEntity || cfg.entity;
    const primaryReg = reg[primary];
    const deviceId = cfg.device_id || (primaryReg && primaryReg.device_id) || null;

    const roles = { state: primary };
    const buttons = {};
    let selectEntity = null;

    if (deviceId) {
      for (const [eid, ent] of Object.entries(reg)) {
        if (!ent || ent.device_id !== deviceId) continue;
        if (ent.platform && ent.platform !== DOMAIN) continue;
        const tk = ent.translation_key;
        // Display roles are always sensors; restrict the match so a like-named
        // entity in another domain can never hijack a display role.
        if (eid.startsWith("sensor.")) {
          for (const [role, want] of Object.entries(ROLE_TKEYS)) {
            if (tk === want && !roles[role]) roles[role] = eid;
          }
        }
        for (const [btn, want] of Object.entries(BUTTON_TKEYS)) {
          if (tk === want && !buttons[btn]) buttons[btn] = eid;
        }
        if (!selectEntity && eid.startsWith("select.")) selectEntity = eid;
      }
    }

    // Legacy explicit overrides always win over auto-discovery.
    if (cfg.program_entity) roles.program = cfg.program_entity;
    if (cfg.time_entity) roles.time = cfg.time_entity;
    if (cfg.pct_entity) roles.progress = cfg.pct_entity;
    if (cfg.power_entity) roles.power = cfg.power_entity;
    if (cfg.energy_entity) roles.energy = cfg.energy_entity;
    if (cfg.phase_entity) roles.phase = cfg.phase_entity;

    // Suffix fallback for the common case where the primary is the state sensor
    // but the registry lookup came up empty (e.g. template entity, no device).
    if (!roles.time || !roles.progress || !roles.program) {
      const m = /^sensor\.(.+)_state$/.exec(primary);
      if (m) {
        const base = "sensor." + m[1] + "_";
        const st = hass && hass.states;
        const guess = (suffix) => (st && st[base + suffix] ? base + suffix : null);
        roles.time = roles.time || guess("time_remaining");
        roles.progress = roles.progress || guess("cycle_progress");
        roles.program = roles.program || guess("program");
      }
    }

    roles._buttons = buttons;
    roles._select = selectEntity;
    roles._deviceId = deviceId;
    return roles;
  }

  // ── View model ────────────────────────────────────────────────────────────
  _fmtState(stateObj) {
    if (!stateObj) return "";
    const hass = this._hass;
    if (hass && typeof hass.formatEntityState === "function") {
      try {
        return hass.formatEntityState(stateObj);
      } catch (_) {
        /* fall through */
      }
    }
    const s = stateObj.state || "";
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
  }

  _vm(roles) {
    const hass = this._hass;
    const cfg = this._cfg;
    const states = hass.states;
    const stateObj = states[roles.state];
    if (!stateObj) {
      return { missing: true, missingEntity: roles.state };
    }

    const sk = String(stateObj.state || "").toLowerCase();
    const isInactive = INACTIVE_STATES.includes(sk);
    const isActive = ACTIVE_STATES.includes(sk);
    const attr = stateObj.attributes || {};

    // Program: prefer the dedicated program entity, then the state sensor's guess.
    let program = "";
    const progObj = roles.program ? states[roles.program] : null;
    if (progObj) {
      const pv = String(progObj.state || "").toLowerCase();
      if (!["unknown", "none", "off", "unavailable", ""].includes(pv)) program = this._fmtState(progObj);
    }
    if (!program && attr.current_program_guess) {
      const pv = String(attr.current_program_guess).toLowerCase();
      if (!["unknown", "none", "off", "unavailable"].includes(pv)) program = attr.current_program_guess;
    }

    // Phase: dedicated phase entity, else the program sensor's active_phase attr.
    let phase = "";
    const phaseObj = roles.phase ? states[roles.phase] : null;
    if (phaseObj) {
      const pv = String(phaseObj.state || "").toLowerCase();
      if (!["unknown", "none", "off", "unavailable", ""].includes(pv)) phase = this._fmtState(phaseObj);
    }
    if (!phase && progObj && progObj.attributes && progObj.attributes.active_phase) {
      phase = progObj.attributes.active_phase;
    }

    // Sub-state (e.g. "Running (Rinsing)" -> "Rinsing").
    let subState = "";
    if (sk === "running" && attr.sub_state) {
      const m = String(attr.sub_state).match(/Running \((.*)\)/);
      subState = m && m[1] ? m[1] : attr.sub_state;
    }

    // Progress percentage.
    let pct = null;
    const pctObj = roles.progress ? states[roles.progress] : null;
    if (pctObj && !isNaN(parseFloat(pctObj.state))) {
      pct = Math.max(0, Math.min(100, Math.round(parseFloat(pctObj.state))));
    }

    // Remaining time (respects the entity's Display Precision + unit, #354).
    let timeText = "";
    const timeObj = roles.time ? states[roles.time] : null;
    if (timeObj && !INACTIVE_STATES.includes(String(timeObj.state).toLowerCase())) {
      if (hass && typeof hass.formatEntityState === "function") {
        timeText = this._fmtState(timeObj);
      } else if (!isNaN(parseFloat(timeObj.state))) {
        timeText = timeObj.state + " " + this._t("card.min", null, "min");
      } else {
        timeText = timeObj.state;
      }
    }

    // Projected energy + cost (live on the progress entity while running).
    let energyText = "";
    let costText = "";
    if (pctObj && pctObj.attributes) {
      const kwh = pctObj.attributes.projected_energy_kwh;
      if (kwh !== undefined && kwh !== null) energyText = kwh + " kWh";
      const cost = pctObj.attributes.projected_cost;
      if (cost !== undefined && cost !== null) costText = this._fmtCost(cost);
    }

    // Current power.
    let powerW = null;
    const powerObj = roles.power ? states[roles.power] : null;
    if (powerObj && !isNaN(parseFloat(powerObj.state))) powerW = parseFloat(powerObj.state);

    // Overrun / anomaly (visible-only; never a notification).
    let anomaly = null;
    if (attr.cycle_anomaly && attr.cycle_anomaly !== "none") {
      anomaly = { kind: attr.cycle_anomaly, ratio: attr.overrun_ratio };
    }

    const color = this._stateColor(sk, isActive, cfg.active_color);

    return {
      missing: false,
      sk,
      stateObj,
      isInactive,
      isActive,
      isRunning: sk === "running",
      stateLabel: this._fmtState(stateObj),
      color,
      program,
      phase,
      subState,
      pct,
      timeText,
      energyText,
      costText,
      powerW,
      anomaly,
      buttons: roles._buttons || {},
      select: roles._select || null,
    };
  }

  _fmtCost(cost) {
    const hass = this._hass;
    const currency = hass && hass.config && hass.config.currency;
    if (currency) {
      try {
        return new Intl.NumberFormat(this._lang(), { style: "currency", currency }).format(Number(cost));
      } catch (_) {
        /* fall through */
      }
    }
    return String(cost);
  }

  _stateColor(sk, isActive, override) {
    // Honour an explicit per-card override colour first (legacy active_color).
    if (isActive && override) {
      if (Array.isArray(override)) {
        const [r, g, b] = override;
        return { fg: "rgb(" + r + "," + g + "," + b + ")", bg: "rgba(" + r + "," + g + "," + b + ",0.2)" };
      }
      return { fg: override, bg: "rgba(128,128,128,0.15)" };
    }
    const colors = (this._constants && this._constants.stateColors) || {};
    const c = colors[sk];
    if (c) return { fg: c, bg: this._alpha(c, sk) };
    // Fallback to theme semantics when constants have not loaded yet.
    if (INACTIVE_STATES.includes(sk)) {
      return { fg: "var(--disabled-text-color, grey)", bg: "rgba(128,128,128,0.1)" };
    }
    return { fg: "var(--primary-color)", bg: "rgba(var(--rgb-primary-color, 33,150,243),0.2)" };
  }

  _alpha(cssColor, sk) {
    // The STATE_COLORS values are "var(--x, #hex)" strings; a translucent tint is
    // built from the hex fallback so the icon chip reads on any theme.
    const hex = /#([0-9a-fA-F]{6})/.exec(cssColor);
    if (hex) {
      const n = parseInt(hex[1], 16);
      const r = (n >> 16) & 255;
      const g = (n >> 8) & 255;
      const b = n & 255;
      return "rgba(" + r + "," + g + "," + b + ",0.18)";
    }
    return INACTIVE_STATES.includes(sk) ? "rgba(128,128,128,0.1)" : "rgba(128,128,128,0.15)";
  }

  _deviceTypeIcon() {
    const hass = this._hass;
    const cfg = this._cfg;
    if (cfg.icon) return cfg.icon;
    const stateObj = hass && hass.states[cfg.entity];
    if (stateObj && stateObj.attributes && stateObj.attributes.icon) return stateObj.attributes.icon;
    return "mdi:washing-machine";
  }

  _titleText() {
    const cfg = this._cfg;
    if (cfg.title) return cfg.title;
    const hass = this._hass;
    const reg = (hass && hass.entities) || {};
    const devices = (hass && hass.devices) || {};
    const ent = reg[cfg.entity];
    if (ent && ent.device_id && devices[ent.device_id]) {
      const dev = devices[ent.device_id];
      const name = dev.name_by_user || dev.name;
      if (name) return name;
    }
    const stateObj = hass && hass.states[cfg.entity];
    if (stateObj && stateObj.attributes && stateObj.attributes.friendly_name) {
      return String(stateObj.attributes.friendly_name).replace(/ (State|Status)$/i, "");
    }
    return "WashData";
  }

  _flags() {
    const c = this._cfg || {};
    return {
      showState: c.show_state !== false,
      showProgram: c.show_program !== false,
      showDetails: c.show_details !== false, // tile: show the time / percentage line
      showBar: c.show_progress_bar !== false,
      showPhase: c.show_phase !== false,
      showEnergy: c.show_energy !== false,
      showAnomaly: c.show_anomaly !== false,
      showSparkline: !!c.show_sparkline,
      displayMode: c.display_mode || "time",
      buttons: Array.isArray(c.buttons) ? c.buttons.filter((b) => BUTTON_ORDER.includes(b)) : [],
    };
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  _render() {
    if (!this.shadowRoot || !this._cfg) return;
    const sig = this._structureSig();
    if (this._builtSig !== sig) {
      this._build(this._cfg.layout || "tile");
      this._builtSig = sig;
    }
    this._update();
  }

  _baseStyle() {
    return (
      ":host{display:block;height:100%}" +
      "ha-card{padding:0;background:var(--ha-card-background,var(--card-background-color,white));" +
      "border-radius:var(--ha-card-border-radius,12px);box-shadow:var(--ha-card-box-shadow,none);" +
      "overflow:hidden;cursor:pointer;height:100%;box-sizing:border-box;" +
      "border:var(--ha-card-border-width,1px) solid var(--ha-card-border-color,var(--divider-color))}" +
      ".icon-container{border-radius:12px;display:flex;align-items:center;justify-content:center;flex-shrink:0;" +
      "transition:background-color .3s,color .3s;position:relative}" +
      "ha-icon{--mdc-icon-size:24px}" +
      ".primary{font-weight:500;color:var(--primary-text-color);white-space:nowrap;text-overflow:ellipsis;overflow:hidden;line-height:1.2}" +
      ".secondary{color:var(--secondary-text-color);white-space:nowrap;text-overflow:ellipsis;overflow:hidden;line-height:1.2}" +
      ".bar{height:4px;border-radius:2px;background:var(--divider-color,rgba(128,128,128,.25));overflow:hidden}" +
      ".bar>i{display:block;height:100%;width:0;border-radius:2px;transition:width .5s ease,background-color .3s}" +
      ".chip{display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:1px 8px;border-radius:10px;" +
      "background:rgba(128,128,128,.14);color:var(--secondary-text-color);white-space:nowrap}" +
      ".chip.warn{background:rgba(255,152,0,.16);color:var(--warning-color,#ff9800)}" +
      ".acts{display:flex;gap:4px;align-items:center}" +
      ".wd-act{border:none;background:rgba(128,128,128,.12);color:var(--primary-text-color);border-radius:10px;" +
      "width:34px;height:34px;display:flex;align-items:center;justify-content:center;cursor:pointer;padding:0}" +
      ".wd-act:hover{background:rgba(128,128,128,.22)}" +
      ".wd-act[disabled]{opacity:.35;pointer-events:none}" +
      ".wd-act ha-icon{--mdc-icon-size:20px}" +
      "select.wd-prog{max-width:150px;font:inherit;color:var(--primary-text-color);background:var(--secondary-background-color);" +
      "border:1px solid var(--divider-color);border-radius:8px;padding:4px 6px}"
    );
  }

  _build(layout) {
    if (layout === "detail") return this._buildDetail();
    if (layout === "glance") return this._buildGlance();
    return this._buildTile();
  }

  _attachGestures(cardEl) {
    cardEl.addEventListener("pointerdown", this._onPointerDown);
    cardEl.addEventListener("pointermove", this._onPointerMove);
    cardEl.addEventListener("pointerup", this._onPointerUp);
    cardEl.addEventListener("pointercancel", this._onPointerCancel);
    cardEl.addEventListener("pointerleave", this._onPointerCancel);
  }

  _buildTile() {
    const flags = this._flags();
    const style = this._baseStyle() + ".tile{display:flex;flex-direction:row;align-items:center;padding:8px 12px;gap:12px;width:100%;box-sizing:border-box}" +
      ".tile .icon-container{width:40px;height:40px}" +
      ".tile .info{display:flex;flex-direction:column;justify-content:center;overflow:hidden;flex:1;min-width:0}" +
      ".tile .primary{font-size:14px}.tile .secondary{font-size:12px;margin-top:2px}" +
      ".tile .bar{margin-top:6px}";
    this.shadowRoot.innerHTML =
      "<style>" + style + "</style>" +
      '<ha-card id="card"><div class="tile">' +
      '<div class="icon-container" id="iconc"><ha-icon id="icon"></ha-icon></div>' +
      '<div class="info"><div class="primary" id="title"></div>' +
      '<div class="secondary" id="state"></div>' +
      (flags.showBar ? '<div class="bar" id="bar"><i id="barfill"></i></div>' : "") +
      "</div></div></ha-card>";
    this._attachGestures(this.shadowRoot.getElementById("card"));
  }

  _buildDetail() {
    const flags = this._flags();
    const style = this._baseStyle() +
      // The detail card hugs its content (height:auto) so a short cycle does not
      // leave a large empty gap; the sections grid reserves rows via getGridOptions.
      "ha-card.wd-detail{height:auto}" +
      ".detail{display:flex;flex-direction:column;gap:8px;padding:12px 14px;width:100%;box-sizing:border-box}" +
      ".detail .top{display:flex;align-items:center;gap:12px}" +
      ".detail .icon-container{width:48px;height:48px}.detail .icon-container ha-icon{--mdc-icon-size:30px}" +
      ".detail .info{flex:1;min-width:0}" +
      ".detail .primary{font-size:16px}.detail .secondary{font-size:13px;margin-top:2px}" +
      ".detail .eta{text-align:right;flex-shrink:0}" +
      ".detail .eta .big{font-size:22px;font-weight:600;color:var(--primary-text-color);line-height:1.1}" +
      ".detail .eta .lbl{font-size:11px;color:var(--secondary-text-color);text-transform:uppercase;letter-spacing:.04em}" +
      ".detail .bar{height:6px}" +
      ".detail .meta{display:flex;flex-wrap:wrap;gap:6px;align-items:center}" +
      ".detail .meta:empty{display:none}" +
      ".detail canvas{width:100%;height:34px;display:block}";
    this.shadowRoot.innerHTML =
      "<style>" + style + "</style>" +
      '<ha-card id="card" class="wd-detail"><div class="detail">' +
      '<div class="top">' +
      '<div class="icon-container" id="iconc"><ha-icon id="icon"></ha-icon></div>' +
      '<div class="info"><div class="primary" id="title"></div><div class="secondary" id="state"></div></div>' +
      '<div class="eta" id="etawrap"><div class="big" id="eta"></div><div class="lbl" id="etalbl"></div></div>' +
      "</div>" +
      (flags.showBar ? '<div class="bar" id="bar"><i id="barfill"></i></div>' : "") +
      '<div class="meta" id="meta"></div>' +
      (flags.showSparkline ? '<canvas id="spark"></canvas>' : "") +
      (flags.buttons.length ? '<div class="acts" id="acts"></div>' : "") +
      "</div></ha-card>";
    this._attachGestures(this.shadowRoot.getElementById("card"));
    if (flags.buttons.length) this._buildButtons(this.shadowRoot.getElementById("acts"), flags.buttons);
  }

  _buildButtons(container, buttons) {
    if (!container) return;
    container.innerHTML = "";
    for (const b of buttons) {
      if (b === "program") {
        const sel = document.createElement("select");
        sel.className = "wd-prog";
        sel.id = "prog-select";
        sel.addEventListener("click", (ev) => ev.stopPropagation());
        sel.addEventListener("pointerdown", (ev) => ev.stopPropagation());
        sel.addEventListener("change", (ev) => {
          ev.stopPropagation();
          this._onProgramChange(ev.target.value);
        });
        container.appendChild(sel);
        continue;
      }
      const btn = document.createElement("button");
      btn.className = "wd-act";
      btn.dataset.btn = b;
      btn.title = this._t("card.btn." + b, null, this._defaultBtnLabel(b));
      btn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
      btn.addEventListener("pointerup", (ev) => ev.stopPropagation());
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        this._onActionButton(b);
      });
      const ic = document.createElement("ha-icon");
      ic.setAttribute("icon", BUTTON_ICONS[b] || "mdi:gesture-tap-button");
      btn.appendChild(ic);
      container.appendChild(btn);
    }
  }

  _defaultBtnLabel(b) {
    const map = {
      pause: "Pause",
      resume: "Resume",
      terminate: "End cycle",
      record_start: "Start recording",
      record_stop: "Stop recording",
      open_panel: "Open WashData",
    };
    return map[b] || b;
  }

  _buildGlance() {
    const style = this._baseStyle() +
      ".glance{display:flex;flex-direction:column;padding:6px 0}" +
      ".row{display:flex;align-items:center;gap:12px;padding:8px 14px;cursor:pointer}" +
      ".row:hover{background:rgba(128,128,128,.06)}" +
      ".dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}" +
      ".row .info{flex:1;min-width:0}.row .primary{font-size:14px}.row .secondary{font-size:12px;margin-top:1px}" +
      ".row .rt{font-size:13px;color:var(--secondary-text-color);flex-shrink:0;text-align:right}";
    const rows = this._glanceEntities()
      .map(
        (_, i) =>
          '<div class="row" data-idx="' + i + '"><span class="dot" data-dot></span>' +
          '<div class="info"><div class="primary" data-title></div><div class="secondary" data-sub></div></div>' +
          '<div class="rt" data-rt></div></div>'
      )
      .join("");
    this.shadowRoot.innerHTML = "<style>" + style + "</style>" + '<ha-card><div class="glance">' + rows + "</div></ha-card>";
    this.shadowRoot.querySelectorAll(".row").forEach((rowEl) => {
      rowEl.addEventListener("click", () => {
        const idx = parseInt(rowEl.dataset.idx, 10);
        const ent = this._glanceEntities()[idx];
        if (ent) this._moreInfo(ent);
      });
    });
  }

  _glanceEntities() {
    const cfg = this._cfg || {};
    if (Array.isArray(cfg.entities) && cfg.entities.length) {
      return cfg.entities.map((e) => (typeof e === "string" ? e : e && e.entity)).filter(Boolean);
    }
    return cfg.entity ? [cfg.entity] : [];
  }

  _update() {
    if (!this.shadowRoot || !this._hass || !this._cfg) return;
    const layout = this._cfg.layout || "tile";
    if (layout === "glance") return this._updateGlance();
    const roles = this._resolveRoles();
    const vm = this._vm(roles);
    if (layout === "detail") return this._updateDetail(vm);
    return this._updateTile(vm);
  }

  _applyIcon(vm) {
    const iconEl = this.shadowRoot.getElementById("icon");
    const iconc = this.shadowRoot.getElementById("iconc");
    if (!iconEl || !iconc) return;
    iconEl.setAttribute("icon", this._deviceTypeIcon());
    iconc.style.color = vm.color.fg;
    iconc.style.background = vm.color.bg;
  }

  _applyBar(vm) {
    const fill = this.shadowRoot.getElementById("barfill");
    const bar = this.shadowRoot.getElementById("bar");
    if (!fill || !bar) return;
    if (!vm.isActive || vm.pct === null) {
      bar.style.visibility = "hidden";
      fill.style.width = "0%";
      return;
    }
    bar.style.visibility = "visible";
    fill.style.width = vm.pct + "%";
    fill.style.background = vm.color.fg;
  }

  _updateTile(vm) {
    const titleEl = this.shadowRoot.getElementById("title");
    const stateEl = this.shadowRoot.getElementById("state");
    if (vm.missing) {
      if (titleEl) titleEl.textContent = this._t("card.entity_not_found", null, "Entity not found");
      if (stateEl) stateEl.textContent = vm.missingEntity || "";
      return;
    }
    titleEl.textContent = this._titleText();
    this._applyIcon(vm);
    this._applyBar(vm);

    const flags = this._flags();
    const parts = [];
    if (flags.showState) {
      if (vm.isRunning) {
        if (vm.subState) parts.push(vm.subState);
      } else {
        parts.push(vm.stateLabel);
      }
    }
    if (flags.showProgram && vm.program) parts.push(vm.program);
    if (!vm.isInactive && flags.showDetails) {
      if (flags.displayMode === "percentage" && vm.pct !== null) parts.push(vm.pct + "%");
      else if (vm.timeText) parts.push(vm.timeText);
      else if (vm.pct !== null) parts.push(vm.pct + "%");
    }
    if (flags.showAnomaly && vm.anomaly) parts.push(this._t("card.running_long", null, "running long"));
    stateEl.textContent = parts.join(" • ");
  }

  _updateDetail(vm) {
    const titleEl = this.shadowRoot.getElementById("title");
    const stateEl = this.shadowRoot.getElementById("state");
    const etaEl = this.shadowRoot.getElementById("eta");
    const etaLbl = this.shadowRoot.getElementById("etalbl");
    const etaWrap = this.shadowRoot.getElementById("etawrap");
    const meta = this.shadowRoot.getElementById("meta");
    if (vm.missing) {
      if (titleEl) titleEl.textContent = this._t("card.entity_not_found", null, "Entity not found");
      if (stateEl) stateEl.textContent = vm.missingEntity || "";
      return;
    }
    titleEl.textContent = this._titleText();
    this._applyIcon(vm);
    this._applyBar(vm);

    const flags = this._flags();
    // Header secondary line: state + program.
    const sub = [];
    if (flags.showState) sub.push(vm.isRunning && vm.subState ? vm.subState : vm.stateLabel);
    if (flags.showProgram && vm.program) sub.push(vm.program);
    stateEl.textContent = sub.join(" • ");

    // Big ETA (or percentage when running with no time).
    if (etaWrap && etaEl && etaLbl) {
      if (vm.isActive && vm.timeText) {
        etaEl.textContent = vm.timeText;
        etaLbl.textContent = this._t("card.remaining", null, "remaining");
        etaWrap.style.visibility = "visible";
      } else if (vm.isActive && vm.pct !== null) {
        etaEl.textContent = vm.pct + "%";
        etaLbl.textContent = this._t("card.progress", null, "progress");
        etaWrap.style.visibility = "visible";
      } else {
        etaWrap.style.visibility = "hidden";
      }
    }

    // Meta chips.
    if (meta) {
      meta.innerHTML = "";
      const addChip = (text, warn) => {
        if (!text) return;
        const c = document.createElement("span");
        c.className = "chip" + (warn ? " warn" : "");
        c.textContent = text;
        meta.appendChild(c);
      };
      if (flags.showPhase && vm.phase && vm.isActive) addChip(vm.phase);
      if (flags.showEnergy && vm.energyText) addChip(vm.energyText);
      if (flags.showEnergy && vm.costText) addChip(vm.costText);
      if (vm.isRunning && vm.powerW !== null) addChip(Math.round(vm.powerW) + " W");
      if (flags.showAnomaly && vm.anomaly) {
        const ratio = vm.anomaly.ratio ? " (" + Math.round((vm.anomaly.ratio - 1) * 100) + "%)" : "";
        addChip(this._t("card.running_long", null, "Running long") + ratio, true);
      }
    }

    this._updateButtons(vm);
    if (flags.showSparkline) this._updateSparkline(vm);
  }

  _updateButtons(vm) {
    const acts = this.shadowRoot.getElementById("acts");
    if (!acts) return;
    const sk = vm.sk;
    const enabled = {
      pause: sk === "running",
      resume: sk === "paused" || sk === "user_paused",
      terminate: vm.isActive,
      record_start: !vm.isActive,
      record_stop: true,
      open_panel: true,
    };
    acts.querySelectorAll(".wd-act").forEach((btn) => {
      const b = btn.dataset.btn;
      const target = b === "open_panel" ? true : b === "program" ? true : vm.buttons[b];
      const ok = enabled[b] !== false && (b === "open_panel" || target);
      if (ok) btn.removeAttribute("disabled");
      else btn.setAttribute("disabled", "");
    });
    const sel = this.shadowRoot.getElementById("prog-select");
    if (sel && vm.select) {
      const selObj = this._hass.states[vm.select];
      const opts = (selObj && selObj.attributes && selObj.attributes.options) || [];
      const cur = selObj && selObj.state;
      const sig = opts.join("|");
      if (sel._sig !== sig) {
        sel._sig = sig;
        sel.innerHTML = "";
        for (const o of opts) {
          const opt = document.createElement("option");
          opt.value = o;
          opt.textContent = o;
          sel.appendChild(opt);
        }
      }
      if (cur !== undefined) sel.value = cur;
    } else if (sel) {
      sel.setAttribute("disabled", "");
    }
  }

  _updateSparkline(vm) {
    const canvas = this.shadowRoot.getElementById("spark");
    if (!canvas) return;
    // Reset the buffer when a new cycle starts / the machine goes idle.
    if (vm.sk !== this._sparkState) {
      if (!vm.isActive) this._spark = [];
      this._sparkState = vm.sk;
    }
    if (vm.isActive && vm.powerW !== null) {
      const last = this._spark[this._spark.length - 1];
      const now = Date.now();
      if (!last || now - last.t > 2000) {
        this._spark.push({ t: now, w: vm.powerW });
        if (this._spark.length > SPARK_MAX_POINTS) this._spark.shift();
      }
    }
    const pts = this._spark;
    // Hide the canvas until there is a line to draw, so an early / idle cycle
    // does not leave a blank 34px band above the buttons.
    if (pts.length < 2) {
      canvas.style.display = "none";
      return;
    }
    canvas.style.display = "block";
    const w = canvas.clientWidth || 300;
    const h = 34;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    const max = Math.max.apply(null, pts.map((p) => p.w)) || 1;
    const pad = 3;
    const dx = (w - pad * 2) / (pts.length - 1);
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = pad + i * dx;
      const y = h - pad - (p.w / max) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = vm.color.fg;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  _updateGlance() {
    const entities = this._glanceEntities();
    const rows = this.shadowRoot.querySelectorAll(".row");
    entities.forEach((ent, i) => {
      const rowEl = rows[i];
      if (!rowEl) return;
      const roles = this._resolveRoles(ent);
      const vm = this._vm(roles);
      const dot = rowEl.querySelector("[data-dot]");
      const title = rowEl.querySelector("[data-title]");
      const sub = rowEl.querySelector("[data-sub]");
      const rt = rowEl.querySelector("[data-rt]");
      if (vm.missing) {
        if (title) title.textContent = ent;
        if (sub) sub.textContent = this._t("card.entity_not_found", null, "Entity not found");
        if (dot) dot.style.background = "var(--disabled-text-color, grey)";
        if (rt) rt.textContent = "";
        return;
      }
      if (dot) dot.style.background = vm.color.fg;
      if (title) title.textContent = this._deviceNameFor(ent);
      const parts = [];
      if (vm.isRunning && vm.subState) parts.push(vm.subState);
      else parts.push(vm.stateLabel);
      if (vm.program) parts.push(vm.program);
      if (sub) sub.textContent = parts.join(" • ");
      if (rt) {
        if (vm.isActive && vm.timeText) rt.textContent = vm.timeText;
        else if (vm.isActive && vm.pct !== null) rt.textContent = vm.pct + "%";
        else rt.textContent = "";
      }
    });
  }

  _deviceNameFor(entityId) {
    const hass = this._hass;
    const reg = (hass && hass.entities) || {};
    const devices = (hass && hass.devices) || {};
    const ent = reg[entityId];
    if (ent && ent.device_id && devices[ent.device_id]) {
      const dev = devices[ent.device_id];
      const name = dev.name_by_user || dev.name;
      if (name) return name;
    }
    const stateObj = hass && hass.states[entityId];
    if (stateObj && stateObj.attributes && stateObj.attributes.friendly_name) {
      return String(stateObj.attributes.friendly_name).replace(/ (State|Status)$/i, "");
    }
    return entityId;
  }

  // ── Actions ───────────────────────────────────────────────────────────────
  _onActionButton(b) {
    const hass = this._hass;
    if (!hass) return;
    if (b === "open_panel") {
      this._navigate("/" + "ha-washdata");
      return;
    }
    const roles = this._resolveRoles();
    const entityId = roles._buttons && roles._buttons[b];
    if (!entityId) return;
    this._fireHaptic("light");
    hass.callService("button", "press", { entity_id: entityId });
  }

  _onProgramChange(value) {
    const hass = this._hass;
    if (!hass || !value) return;
    const roles = this._resolveRoles();
    const sel = roles._select;
    if (!sel) return;
    hass.callService("select", "select_option", { entity_id: sel, option: value });
  }

  _moreInfo(entityId) {
    if (!entityId) return;
    this.dispatchEvent(new CustomEvent("hass-more-info", { detail: { entityId }, bubbles: true, composed: true }));
  }

  _navigate(path) {
    window.history.pushState(null, "", path);
    window.dispatchEvent(new CustomEvent("location-changed", { detail: { replace: false } }));
  }

  // ── Gesture handling (tap / hold / double-tap) ──────────────────────────────
  _clearHoldTimer() {
    if (this._holdTimer) {
      window.clearTimeout(this._holdTimer);
      this._holdTimer = null;
    }
  }

  _onPointerDown(ev) {
    if (ev.button !== undefined && ev.button !== 0) return;
    this._holdTriggered = false;
    this._pointerCanceled = false;
    this._pointerStart = { x: ev.clientX, y: ev.clientY };
    const holdCfg = this._cfg && this._cfg.hold_action;
    if (holdCfg && holdCfg.action && holdCfg.action !== "none") {
      this._clearHoldTimer();
      this._holdTimer = window.setTimeout(() => {
        this._holdTimer = null;
        this._holdTriggered = true;
        this._fireHaptic("success");
        this._executeAction(holdCfg);
      }, HOLD_MS);
    }
  }

  _onPointerMove(ev) {
    if (!this._pointerStart) return;
    const dx = ev.clientX - this._pointerStart.x;
    const dy = ev.clientY - this._pointerStart.y;
    if (dx * dx + dy * dy > TAP_MOVE_TOLERANCE * TAP_MOVE_TOLERANCE) {
      this._clearHoldTimer();
      this._pointerStart = null;
      this._pointerCanceled = true;
    }
  }

  _onPointerCancel() {
    this._clearHoldTimer();
  }

  _onPointerUp() {
    this._clearHoldTimer();
    if (this._pointerCanceled) {
      this._pointerCanceled = false;
      return;
    }
    if (this._holdTriggered) {
      this._holdTriggered = false;
      return;
    }
    const tapCfg = (this._cfg && this._cfg.tap_action) || { action: "more-info" };
    const doubleCfg = this._cfg && this._cfg.double_tap_action;
    const hasDouble = doubleCfg && doubleCfg.action && doubleCfg.action !== "none";
    if (!hasDouble) {
      this._executeAction(tapCfg);
      return;
    }
    const now = Date.now();
    if (this._tapTimer && now - this._lastTapTime < DOUBLE_TAP_MS) {
      window.clearTimeout(this._tapTimer);
      this._tapTimer = null;
      this._lastTapTime = 0;
      this._executeAction(doubleCfg);
      return;
    }
    this._lastTapTime = now;
    this._tapTimer = window.setTimeout(() => {
      this._tapTimer = null;
      this._executeAction(tapCfg);
    }, DOUBLE_TAP_MS);
  }

  _fireHaptic(type) {
    this.dispatchEvent(new CustomEvent("haptic", { detail: type, bubbles: true, composed: true }));
  }

  _executeAction(actionCfg) {
    if (!actionCfg) return;
    const action = actionCfg.action || "more-info";
    const entityId = actionCfg.entity || (this._cfg && this._cfg.entity);
    switch (action) {
      case "none":
        return;
      case "more-info":
        this._moreInfo(entityId);
        return;
      case "toggle":
        if (!this._hass || !entityId) return;
        this._hass.callService("homeassistant", "toggle", { entity_id: entityId });
        return;
      case "call-service":
      case "perform-action": {
        const svc = actionCfg.perform_action || actionCfg.service;
        if (!svc || !this._hass) return;
        const [svcDomain, svcName] = svc.split(".");
        if (!svcDomain || !svcName) return;
        const data = { ...(actionCfg.data || actionCfg.service_data || {}) };
        this._hass.callService(svcDomain, svcName, data, actionCfg.target);
        return;
      }
      case "navigate": {
        const path = actionCfg.navigation_path;
        if (!path) return;
        if (actionCfg.navigation_replace) {
          window.history.replaceState(window.history.state, "", path);
          window.dispatchEvent(new CustomEvent("location-changed", { detail: { replace: true } }));
        } else {
          this._navigate(path);
        }
        return;
      }
      case "url": {
        const url = actionCfg.url_path;
        if (!url) return;
        window.open(url, "_blank", "noopener,noreferrer");
        return;
      }
      default:
        return;
    }
  }
}

class WashDataCardEditor extends HTMLElement {
  _lang() {
    const raw =
      (this._hass && this._hass.locale && this._hass.locale.language) ||
      (this._hass && this._hass.language) ||
      "en";
    return typeof raw === "string" && raw ? raw : "en";
  }

  _t(key, fallback) {
    const lang = this._lang();
    const dict = _panelTrans[lang] || _panelTrans["en"];
    if (dict) {
      const val = key.split(".").reduce((o, k) => (o && o[k] !== undefined ? o[k] : null), dict);
      if (val && typeof val === "string") return val;
    }
    return fallback || key;
  }

  setConfig(config) {
    this._cfg = { layout: "tile", ...config };
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    const lang = this._lang();
    const jobs = [];
    if (!_panelTrans["en"]) jobs.push(loadPanelLang("en"));
    if (lang && lang !== "en" && !_panelTrans[lang]) jobs.push(loadPanelLang(lang));
    if (jobs.length) Promise.all(jobs).then(() => this._render());
    if (this._form) this._form.hass = hass;
  }

  _render() {
    if (!this.shadowRoot) this.attachShadow({ mode: "open" });
    if (!this._form) {
      this.shadowRoot.innerHTML =
        "<style>.editor-container{padding:8px 4px}ha-form{display:block}</style>" +
        '<div class="editor-container" id="editor-container"></div>';
      this._form = document.createElement("ha-form");
      this.shadowRoot.getElementById("editor-container").appendChild(this._form);
      this._form.addEventListener("value-changed", (ev) => this._valueChanged(ev));
      this._form.computeLabel = (schema) => this._t("card.editor." + schema.name, this._humanize(schema.name));
    }
    this._form.schema = this._schema();
    this._form.data = this._cfg;
    if (this._hass) this._form.hass = this._hass;
  }

  _humanize(name) {
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  _schema() {
    // Flat schema (no nested/expandable groups) so config keys always stay at the
    // top level regardless of the running Home Assistant version.
    const layout = (this._cfg && this._cfg.layout) || "tile";
    const layoutSel = {
      name: "layout",
      selector: {
        select: {
          mode: "dropdown",
          options: [
            { value: "tile", label: this._t("card.layout.tile", "Tile (compact)") },
            { value: "detail", label: this._t("card.layout.detail", "Detail (rich)") },
            { value: "glance", label: this._t("card.layout.glance", "Glance (multiple devices)") },
          ],
        },
      },
    };
    if (layout === "glance") {
      return [
        layoutSel,
        { name: "entities", selector: { entity: { domain: "sensor", multiple: true } } },
        { name: "title", selector: { text: {} } },
      ];
    }
    const schema = [
      layoutSel,
      { name: "entity", selector: { entity: { domain: "sensor" } } },
      { name: "title", selector: { text: {} } },
      { name: "icon", selector: { icon: {} } },
      { name: "show_state", selector: { boolean: {} } },
      { name: "show_program", selector: { boolean: {} } },
      { name: "show_progress_bar", selector: { boolean: {} } },
      {
        name: "display_mode",
        selector: {
          select: {
            mode: "dropdown",
            options: [
              { value: "time", label: this._t("card.show_time_remaining", "Show Time Remaining") },
              { value: "percentage", label: this._t("card.show_percentage", "Show Percentage") },
            ],
          },
        },
      },
      { name: "active_color", selector: { color_rgb: {} } },
    ];
    if (layout === "detail") {
      schema.push(
        { name: "show_phase", selector: { boolean: {} } },
        { name: "show_energy", selector: { boolean: {} } },
        { name: "show_anomaly", selector: { boolean: {} } },
        { name: "show_sparkline", selector: { boolean: {} } },
        {
          name: "buttons",
          selector: {
            select: {
              multiple: true,
              mode: "list",
              options: BUTTON_ORDER.map((b) => ({ value: b, label: this._t("card.btn." + b, this._humanize(b)) })),
            },
          },
        }
      );
    }
    schema.push(
      { name: "program_entity", selector: { entity: { domain: ["sensor", "select", "input_select", "input_text"] } } },
      { name: "time_entity", selector: { entity: { domain: "sensor" } } },
      { name: "pct_entity", selector: { entity: { domain: "sensor" } } },
      { name: "power_entity", selector: { entity: { domain: "sensor" } } },
      { name: "tap_action", selector: { ui_action: {} } },
      { name: "hold_action", selector: { ui_action: {} } },
      { name: "double_tap_action", selector: { ui_action: {} } }
    );
    return schema;
  }

  _valueChanged(ev) {
    if (!this._cfg) return;
    const val = ev.detail.value;
    const layoutChanged = val.layout && val.layout !== this._cfg.layout;
    this._cfg = { ...this._cfg, ...val };
    this.dispatchEvent(
      new CustomEvent("config-changed", { detail: { config: this._cfg }, bubbles: true, composed: true })
    );
    // Re-render the schema when the layout changes so its layout-specific fields appear.
    if (layoutChanged) this._render();
  }
}

customElements.define(CARD_TAG, WashDataCard);
customElements.define(EDITOR_TAG, WashDataCardEditor);

window.customCards = window.customCards || [];
window.customCards.push({
  type: CARD_TAG,
  name: "WashData Card",
  preview: true,
  description: "Adaptive card for WashData appliances: compact tile, rich detail, or multi-device glance.",
  documentationURL: "https://github.com/3dg1luk43/ha_washdata",
});
