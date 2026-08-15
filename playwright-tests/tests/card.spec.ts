/**
 * Bundled Lovelace card (ha-washdata-card) tests.
 *
 * Boots the real card resource in a browser against a mock `hass`, and verifies
 * the three layouts (tile / detail / glance), zero-config entity auto-discovery,
 * state coloring from STATE_COLORS, the progress bar, action buttons, and the
 * editor schema. The card renders into an open shadow root, which Playwright
 * pierces automatically.
 */

import { test, expect } from '@playwright/test';

// A washing machine mid-cycle, wired as a real WashData device: only the state
// sensor is configured on the card; every sibling is discovered by device +
// translation_key.
const RUN: Record<string, unknown> = {
  currency: 'EUR',
  stateColors: {
    running: 'var(--success-color, #4caf50)',
    off: 'var(--state-inactive-color, #9e9e9e)',
    clean: 'var(--teal-color, #009688)',
  },
  states: {
    'sensor.wm_state': { state: 'running', attributes: { sub_state: 'Running (Rinsing)', current_program_guess: 'Cotton 40', cycle_anomaly: 'overrun', overrun_ratio: 1.2 } },
    'sensor.wm_cycle_progress': { state: '64', attributes: { projected_energy_kwh: 0.42, projected_cost: 0.13 } },
    'sensor.wm_time_remaining': { state: '16', attributes: { unit_of_measurement: 'min' } },
    'sensor.wm_program': { state: 'Cotton 40', attributes: { active_phase: 'Rinse' } },
    'sensor.wm_current_power': { state: '420', attributes: { unit_of_measurement: 'W' } },
    'select.wm_program': { state: 'Cotton 40', attributes: { options: ['Eco', 'Cotton 40'] } },
    'button.wm_pause': { state: '2026-01-01T00:00:00+00:00', attributes: {} },
    'button.wm_resume': { state: '2026-01-01T00:00:00+00:00', attributes: {} },
    'button.wm_terminate': { state: '2026-01-01T00:00:00+00:00', attributes: {} },
  },
  entities: {
    'sensor.wm_state': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'washer_state' },
    'sensor.wm_cycle_progress': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'cycle_progress' },
    'sensor.wm_time_remaining': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'time_remaining' },
    'sensor.wm_program': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'washer_program' },
    'sensor.wm_current_power': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'current_power' },
    'select.wm_program': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'program_select' },
    'button.wm_pause': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'pause_cycle' },
    'button.wm_resume': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'resume_cycle' },
    'button.wm_terminate': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'force_end_cycle' },
  },
  devices: { d1: { name: 'Washing Machine' } },
};

function withState(state: string): Record<string, unknown> {
  const clone = JSON.parse(JSON.stringify(RUN));
  clone.states['sensor.wm_state'].state = state;
  return clone;
}

test.beforeEach(async ({ page }) => {
  await page.goto('/card.html');
  await page.waitForFunction(() => (window as any).__ready === true, { timeout: 10_000 });
});

async function mount(page: any, config: unknown, data: unknown) {
  await page.evaluate(
    ({ c, d }: any) => (window as any).__mountCard(c, d),
    { c: config, d: data },
  );
}

test('tile: zero-config auto-discovery fills program, sub-state and time', async ({ page }) => {
  // Only the state entity is given; program/time/progress must be discovered.
  await mount(page, { entity: 'sensor.wm_state', layout: 'tile' }, RUN);
  const vm = await page.evaluate(() => {
    const sr = (window as any).__card.shadowRoot;
    return {
      title: sr.getElementById('title').textContent,
      secondary: sr.getElementById('state').textContent,
      barWidth: sr.getElementById('barfill') ? sr.getElementById('barfill').style.width : null,
    };
  });
  expect(vm.title).toBe('Washing Machine');
  expect(vm.secondary).toContain('Rinsing');
  expect(vm.secondary).toContain('Cotton 40');
  expect(vm.secondary).toContain('16 min');
  expect(vm.barWidth).toBe('64%');
});

test('tile: state color comes from STATE_COLORS (running = success color)', async ({ page }) => {
  await mount(page, { entity: 'sensor.wm_state', layout: 'tile' }, RUN);
  const bg = await page.evaluate(() => {
    const sr = (window as any).__card.shadowRoot;
    return sr.getElementById('barfill').style.background;
  });
  expect(bg).toContain('--success-color');
});

test('tile: inactive appliance hides the progress bar and shows the state', async ({ page }) => {
  await mount(page, { entity: 'sensor.wm_state', layout: 'tile' }, withState('off'));
  const vm = await page.evaluate(() => {
    const sr = (window as any).__card.shadowRoot;
    return {
      barVisibility: sr.getElementById('bar').style.visibility,
      secondary: sr.getElementById('state').textContent,
    };
  });
  expect(vm.barVisibility).toBe('hidden');
  expect(vm.secondary.toLowerCase()).toContain('off');
});

test('detail: big ETA, phase/energy/power chips and overrun warning render', async ({ page }) => {
  await mount(
    page,
    { entity: 'sensor.wm_state', layout: 'detail', show_sparkline: true, buttons: ['pause', 'resume', 'terminate', 'program', 'open_panel'] },
    RUN,
  );
  const d = await page.evaluate(() => {
    const sr = (window as any).__card.shadowRoot;
    const meta = sr.getElementById('meta');
    const chips = meta ? Array.from(meta.querySelectorAll('.chip')).map((c: any) => c.textContent) : [];
    const warnChips = meta ? Array.from(meta.querySelectorAll('.chip.warn')).map((c: any) => c.textContent) : [];
    const acts = Array.from(sr.querySelectorAll('.wd-act')).map((b: any) => b.dataset.btn);
    return {
      eta: sr.getElementById('eta').textContent,
      etalbl: sr.getElementById('etalbl').textContent,
      chips,
      warnChips,
      acts,
      hasSpark: !!sr.getElementById('spark'),
      hasSelect: !!sr.getElementById('prog-select'),
    };
  });
  expect(d.eta).toBe('16 min');
  expect(d.etalbl).toBe('remaining'); // localized from the real en.json (card.remaining)
  const chipStr = d.chips.join('|');
  expect(chipStr).toContain('Rinse'); // phase
  expect(chipStr).toContain('0.42 kWh'); // projected energy
  expect(chipStr).toContain('0.13'); // projected cost (currency symbol varies)
  expect(chipStr).toContain('420 W'); // current power
  expect(d.warnChips.join('|').toLowerCase()).toContain('running long'); // overrun accent
  expect(d.acts).toEqual(expect.arrayContaining(['pause', 'resume', 'terminate', 'open_panel']));
  expect(d.hasSpark).toBeTruthy();
  expect(d.hasSelect).toBeTruthy();
});

test('detail: action buttons are state-gated (pause enabled, resume disabled while running)', async ({ page }) => {
  await mount(page, { entity: 'sensor.wm_state', layout: 'detail', buttons: ['pause', 'resume', 'terminate'] }, RUN);
  const gated = await page.evaluate(() => {
    const sr = (window as any).__card.shadowRoot;
    const q = (b: string) => sr.querySelector(`.wd-act[data-btn="${b}"]`);
    return {
      pauseDisabled: q('pause').hasAttribute('disabled'),
      resumeDisabled: q('resume').hasAttribute('disabled'),
      terminateDisabled: q('terminate').hasAttribute('disabled'),
    };
  });
  expect(gated.pauseDisabled).toBe(false);
  expect(gated.resumeDisabled).toBe(true);
  expect(gated.terminateDisabled).toBe(false);
});

test('detail: pressing Pause calls button.press on the discovered pause entity', async ({ page }) => {
  await mount(page, { entity: 'sensor.wm_state', layout: 'detail', buttons: ['pause'] }, RUN);
  await page.evaluate(() => (window as any).__card.shadowRoot.querySelector('.wd-act[data-btn="pause"]').click());
  const svc = await page.evaluate(() => (window as any).__svcCalls());
  expect(
    svc.some((c: any) => c.domain === 'button' && c.service === 'press' && c.data.entity_id === 'button.wm_pause'),
  ).toBeTruthy();
});

test('glance: renders one row per device with state dot and time', async ({ page }) => {
  await mount(page, { entity: 'sensor.wm_state', entities: ['sensor.wm_state'], layout: 'glance' }, RUN);
  const rows = await page.evaluate(() => {
    const sr = (window as any).__card.shadowRoot;
    return Array.from(sr.querySelectorAll('.row')).map((r: any) => ({
      title: r.querySelector('[data-title]').textContent,
      sub: r.querySelector('[data-sub]').textContent,
      rt: r.querySelector('[data-rt]').textContent,
      dot: r.querySelector('[data-dot]').style.background,
    }));
  });
  expect(rows.length).toBe(1);
  expect(rows[0].title).toBe('Washing Machine');
  expect(rows[0].sub).toContain('Rinsing');
  expect(rows[0].rt).toBe('16 min');
  expect(rows[0].dot).toContain('--success-color');
});

test('editor: exposes a layout-aware flat schema', async ({ page }) => {
  await page.evaluate(
    ({ c, d }: any) => (window as any).__mountEditor(c, d),
    { c: { entity: 'sensor.wm_state', layout: 'detail' }, d: RUN },
  );
  const ed = await page.evaluate(() => {
    const f = (window as any).__editor._form;
    const names = f && f.schema ? f.schema.map((s: any) => s.name) : [];
    return { hasForm: !!f, names };
  });
  expect(ed.hasForm).toBeTruthy();
  expect(ed.names).toContain('layout');
  expect(ed.names).toContain('entity');
  expect(ed.names).toContain('buttons'); // detail-only field
  // Flat schema: no nested/expandable groups that would break the config shape.
  expect(ed.names).not.toContain('appearance');
});
