class WashDataCard extends HTMLElement {
    set hass(hass) {
        this._hass = hass;
        if (!this.content) {
            const card = document.createElement('ha-card');
            card.header = this.config.title;
            this.content = document.createElement('div');
            this.content.style.padding = '0 16px 16px';
            card.appendChild(this.content);
            this.appendChild(card);
        }

        const entityId = this.config.entity;
        const stateObj = hass.states[entityId];

        if (stateObj) {
            const state = stateObj.state;
            const attributes = stateObj.attributes;
            const icon = this.config.icon || attributes.icon || 'mdi:washing-machine';

            // Status mapping
            let statusText = state;
            if (state === 'running') statusText = 'Running';
            if (state === 'off') statusText = 'Off';
            if (state === 'idle') statusText = 'Idle';

            // Extract attributes
            const program = attributes.program || '';
            const remaining = attributes.time_remaining ? `${attributes.time_remaining} remaining` : '';
            const progress = attributes.cycle_progress || 0;

            let color = 'var(--primary-text-color)';
            if (state === 'running') color = 'var(--primary-color)';

            this.content.innerHTML = `
                <style>
                    .wash-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
                    .wash-icon { --mdc-icon-size: 48px; color: ${color}; }
                    .wash-status { font-size: 1.2em; font-weight: 500; }
                    .wash-sub { color: var(--secondary-text-color); font-size: 0.9em; }
                    .progress-bar { height: 6px; background: var(--divider-color); border-radius: 3px; overflow: hidden; margin-top: 8px; }
                    .progress-fill { height: 100%; background: var(--primary-color); transition: width 0.5s ease; }
                </style>
                <div class="wash-row">
                    <div>
                        <ha-icon icon="${icon}" class="wash-icon"></ha-icon>
                    </div>
                    <div style="text-align: right;">
                        <div class="wash-status">${statusText}</div>
                        ${program ? `<div class="wash-sub">${program}</div>` : ''}
                        ${remaining ? `<div class="wash-sub">${remaining}</div>` : ''}
                    </div>
                </div>
                ${state === 'running' ? `
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${progress}%"></div>
                </div>` : ''}
            `;
        } else {
            this.content.innerHTML = `<div>Entity not found: ${entityId}</div>`;
        }
    }

    setConfig(config) {
        if (!config.entity) {
            throw new Error('You need to define an entity');
        }
        this.config = config;
    }

    getCardSize() {
        return 3;
    }

    static getConfigElement() {
        return document.createElement("ha-washdata-card-editor");
    }

    static getStubConfig() {
        return { entity: "sensor.washing_machine", title: "WashData" };
    }
}

class WashDataCardEditor extends HTMLElement {
    setConfig(config) {
        this._config = config;
        // Re-render if elements exist
        if (this._entityPicker) {
            this._entityPicker.value = config.entity;
        }
    }

    configChanged(newConfig) {
        const event = new CustomEvent("config-changed", {
            detail: { config: newConfig },
            bubbles: true,
            composed: true,
        });
        this.dispatchEvent(event);
    }

    set hass(hass) {
        this._hass = hass;

        if (this._elementsCreated) {
            // Update hass on picker if it exists
            if (this._entityPicker) {
                this._entityPicker.hass = hass;
            }
            return;
        }

        this._elementsCreated = true;
        this.innerHTML = '';

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.gap = '16px';

        // Entity Picker
        this._entityPicker = document.createElement('ha-entity-picker');
        this._entityPicker.label = 'Entity';
        this._entityPicker.hass = hass;
        this._entityPicker.value = this._config.entity;
        this._entityPicker.domain = 'sensor'; // Correct property for checking domain is often implicit or needs checking component docs, "include-domains" is common
        // Note: ha-entity-picker properties vary by version. .hass and .value are standard. 
        // Some versions use .includeDomains = ['sensor']

        this._entityPicker.addEventListener('value-changed', (ev) => {
            this._config = { ...this._config, entity: ev.detail.value };
            this.configChanged(this._config);
        });

        // Title Input
        // Using paper-input or ha-textfield depending on availability. 
        // Try ha-textfield first (newer), fallback to input if complex.
        // Actually, simple input is safest or ha-textfield if available.
        // Let's use ha-textfield as standard modern HA component.
        const titleInput = document.createElement('ha-textfield');
        titleInput.label = 'Title (Optional)';
        titleInput.value = this._config.title || '';
        titleInput.addEventListener('input', (ev) => {
            this._config = { ...this._config, title: ev.target.value };
            this.configChanged(this._config);
        });

        container.appendChild(this._entityPicker);
        container.appendChild(titleInput);
        this.appendChild(container);
    }
}

customElements.define('ha-washdata-card-editor', WashDataCardEditor);
customElements.define('ha-washdata-card', WashDataCard);

window.customCards = window.customCards || [];
window.customCards.push({
    type: "ha-washdata-card",
    name: "WashData Card",
    description: "A minimal status card for WashData entities",
});
