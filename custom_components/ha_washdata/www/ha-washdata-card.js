const CARD_TAG = "ha-washdata-card";
const EDITOR_TAG = "ha-washdata-card-editor";

class WashDataCard extends HTMLElement {
  static getStubConfig() {
    return {
      entity: "sensor.washing_machine_state",
      title: "Washing Machine",
      icon: "mdi:washing-machine",
      display_mode: "time",
      active_color: [33, 150, 243],
      show_state: true,
      show_program: true,
      show_details: true
    };
  }

  static getConfigElement() {
    return document.createElement(EDITOR_TAG);
  }

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._rendered = false;
    this._handleClick = this._handleClick.bind(this);
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error("Please define an entity");
    }
    this._cfg = { ...WashDataCard.getStubConfig(), ...config };
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    this._update();
  }

  getCardSize() {
    return 1;
  }

  _handleClick() {
    const entityId = this._cfg.entity;
    const event = new CustomEvent("hass-more-info", {
      detail: { entityId },
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }

  _render() {
    if (!this.shadowRoot) return;

    // Only create the DOM once to avoid memory leaks from duplicate event listeners
    if (!this._rendered) {
      this.shadowRoot.innerHTML = `
        <style>
          :host {
            display: block;
            height: 100%;
          }
          ha-card {
            padding: 0;
            background: var(--ha-card-background, var(--card-background-color, white));
            border-radius: var(--ha-card-border-radius, 12px);
            box-shadow: var(--ha-card-box-shadow, none);
            overflow: hidden;
            cursor: pointer;
            height: 100%;
            display: flex;
            align-items: center;
            box-sizing: border-box;
            border: var(--ha-card-border-width, 1px) solid var(--ha-card-border-color, var(--divider-color));
          }
          .tile {
            display: flex;
            flex-direction: row;
            align-items: center;
            padding: 0 12px;
            gap: 12px;
            width: 100%;
            height: 100%;
            min-height: 56px; /* standard tile height */
            max-height: 56px;
            box-sizing: border-box;
          }
          .icon-container {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--tile-icon-bg, rgba(128, 128, 128, 0.1));
            color: var(--tile-icon-color, var(--primary-text-color));
            flex-shrink: 0;
            transition: background-color 0.3s, color 0.3s;
          }
          ha-icon {
            --mdc-icon-size: 24px;
          }
          .info {
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow: hidden;
            flex: 1;
          }
          .primary {
            font-weight: 500;
            font-size: 14px;
            color: var(--primary-text-color);
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
            line-height: 1.2;
          }
          .secondary {
            font-size: 12px;
            color: var(--secondary-text-color);
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
            line-height: 1.2;
            margin-top: 2px;
          }
        </style>
        <ha-card id="card">
          <div class="tile">
            <div class="icon-container" id="icon-container">
              <ha-icon id="icon"></ha-icon>
            </div>
            <div class="info">
              <div class="primary" id="title"></div>
              <div class="secondary" id="state"></div>
            </div>
          </div>
        </ha-card>
      `;

      this.shadowRoot.getElementById("card").addEventListener("click", this._handleClick);
      this._rendered = true;
    }

    this._update();
  }

  _update() {
    if (!this.shadowRoot || !this._hass || !this._cfg) return;

    const entityId = this._cfg.entity;
    const stateObj = this._hass.states[entityId];

    const titleEl = this.shadowRoot.getElementById("title");
    const stateEl = this.shadowRoot.getElementById("state");
    const iconEl = this.shadowRoot.getElementById("icon");
    const iconContainer = this.shadowRoot.getElementById("icon-container");

    if (!stateObj) {
      if (titleEl) titleEl.textContent = "Entity not found";
      if (stateEl) stateEl.textContent = entityId;
      return;
    }

    const title = this._cfg.title || "Washing Machine";
    const icon = this._cfg.icon || stateObj.attributes.icon || "mdi:washing-machine";
    const activeColor = this._cfg.active_color;

    const state = stateObj.state;
    // Treat as inactive if off, unknown, unavailable, idle
    const isInactive = ['off', 'unknown', 'unavailable', 'idle'].includes(state.toLowerCase());

    if (isInactive) {
      iconContainer.style.background = `rgba(128, 128, 128, 0.1)`;
      iconContainer.style.color = `var(--disabled-text-color, grey)`;
    } else {
      let colorCss = "var(--primary-color)";
      let bgCss = "rgba(var(--rgb-primary-color, 33, 150, 243), 0.2)";

      if (Array.isArray(activeColor)) {
        const [r, g, b] = activeColor;
        colorCss = `rgb(${r}, ${g}, ${b})`;
        bgCss = `rgba(${r}, ${g}, ${b}, 0.2)`;
      } else if (activeColor) {
        colorCss = activeColor;
        bgCss = `rgba(128, 128, 128, 0.15)`;
      }

      iconContainer.style.color = colorCss;
      iconContainer.style.background = bgCss;
    }

    iconEl.setAttribute("icon", icon);
    titleEl.textContent = title;

    const attr = stateObj.attributes;
    const parts = [];

    // 1. State / Sub-State
    // Default show_state to true if undefined
    if (this._cfg.show_state !== false) {
      if (state.toLowerCase() === 'running') {
        const subState = attr.sub_state;
        if (subState) {
          // If sub_state is "Running (Rinsing)", extract "Rinsing"
          const match = subState.match(/Running \((.*)\)/);
          if (match && match[1]) {
            parts.push(match[1]);
          } else {
            parts.push(subState);
          }
        }
        // If no sub_state (or just "Running"), we show NOTHING (redundant)
      } else {
        // Not running (e.g. Off, Completed, etc) - show standard state
        parts.push(state.charAt(0).toUpperCase() + state.slice(1));
      }
    }

    // 2. Program
    if (this._cfg.show_program !== false) {
      let program = "";
      if (this._cfg.program_entity) {
        const progState = this._hass.states[this._cfg.program_entity];
        if (progState) program = progState.state;
      } else if (attr.program) {
        program = attr.program;
      }
      if (program && !["unknown", "none", "off", "unavailable"].includes(program.toLowerCase())) {
        parts.push(program);
      }
    }

    // 3. Details (Time / Pct)
    if (this._cfg.show_details !== false) {
      let remaining = "";
      if (this._cfg.time_entity) {
        remaining = this._hass.states[this._cfg.time_entity]?.state;
      } else if (attr.time_remaining) {
        remaining = attr.time_remaining;
      }

      let pct = "";
      if (this._cfg.pct_entity) {
        pct = this._hass.states[this._cfg.pct_entity]?.state;
      } else if (attr.cycle_progress) {
        pct = attr.cycle_progress;
      }

      if (this._cfg.display_mode === 'percentage' && pct) {
        parts.push(`${Math.round(pct)}%`);
      } else if (remaining) {
        parts.push(remaining);
      }
    }

    stateEl.textContent = parts.length > 0 ? parts.join(" • ") : "";
  }
}

class WashDataCardEditor extends HTMLElement {
  setConfig(config) {
    this._cfg = { ...WashDataCard.getStubConfig(), ...config };
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    if (this._form) {
      this._form.hass = hass;
    }
  }

  _render() {
    if (!this.shadowRoot) {
      this.attachShadow({ mode: "open" });
    }

    if (!this._form) {
      this.shadowRoot.innerHTML = `
        <style>
          .editor-container {
            padding: 16px;
            max-width: 400px; /* Constrain editor width */
          }
          ha-form {
            display: block;
          }
        </style>
        <div class="editor-container" id="editor-container"></div>
      `;
      this._form = document.createElement("ha-form");
      this.shadowRoot.getElementById("editor-container").appendChild(this._form);

      this._form.addEventListener("value-changed", (ev) => this._valueChanged(ev));

      this._form.schema = [
        { name: "title", selector: { text: {} } },
        { name: "entity", selector: { entity: { domain: "sensor" } } },
        { name: "icon", selector: { icon: {} } },
        { name: "active_color", selector: { color_rgb: {} } },
        { name: "show_state", selector: { boolean: {} } },
        { name: "show_program", selector: { boolean: {} } },
        { name: "show_details", selector: { boolean: {} } },
        {
          name: "display_mode",
          selector: {
            select: {
              options: [
                { value: "time", label: "Show Time Remaining" },
                { value: "percentage", label: "Show Percentage" }
              ],
              mode: "dropdown"
            }
          }
        },
        { name: "program_entity", selector: { entity: { domain: ["sensor", "select", "input_select", "input_text"] } } },
        { name: "pct_entity", selector: { entity: { domain: "sensor" } } },
        { name: "time_entity", selector: { entity: { domain: "sensor" } } },
      ];

      this._form.computeLabel = (schema) => {
        const labels = {
          title: "Title",
          entity: "Status Entity (Main)",
          icon: "Icon",
          active_color: "Icon Color (Active)",
          show_state: "Show State / Phase",
          show_program: "Show Program",
          show_details: "Show Rate (Time/%)",
          program_entity: "Program Entity (Override)",
          pct_entity: "Percentage Entity (Override)",
          time_entity: "Time Entity (Override)",
          display_mode: "Detail Display Mode"
        };
        return labels[schema.name] || schema.name;
      };
    }

    this._form.data = this._cfg;
    if (this._hass) {
      this._form.hass = this._hass;
    }
  }

  _valueChanged(ev) {
    if (!this._cfg || !this._hass) return;
    const val = ev.detail.value;
    this._cfg = { ...this._cfg, ...val };

    const event = new CustomEvent("config-changed", {
      detail: { config: this._cfg },
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }
}

customElements.define(CARD_TAG, WashDataCard);
customElements.define(EDITOR_TAG, WashDataCardEditor);

window.customCards = window.customCards || [];
window.customCards.push({
  type: CARD_TAG,
  name: "WashData Tile Card",
  preview: true,
  description: "A compact tile-style card for washing machines.",
});
