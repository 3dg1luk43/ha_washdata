/**
 * #478: the Automations tab.
 *
 * The Notifications > Automations subsection moved to its own top-level tab, which
 * also offers the blueprint templates shipped with the integration
 * (ha_washdata/blueprints/list), filtered to the device type. An administrator
 * installs one with HA's blueprint/save; a template whose only open input is the
 * device ("one_click") then becomes an automation for this device in the same
 * click. Everyone may read a template's YAML.
 *
 * v2: "Start from a trigger" lists every WashData integration trigger the device
 * type offers, and everything the panel creates gets the "WashData" label and the
 * "WashData" automation category (found by name or created), without ever failing
 * the creation over it.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled } from '../helpers/panel';
import optionsData from '../fixtures/mock-data/options.json';
import panelConfig from '../fixtures/mock-data/panel-config.json';

const ENTRY = 'test-entry-001';
const HASS = {
  devices: {
    plug: { id: 'plug', identifiers: [['tasmota', 'abc']], config_entries: ['other-entry'] },
    wd: { id: 'wd-device', identifiers: [['ha_washdata', ENTRY]], config_entries: [ENTRY] },
  },
};
const SRC = 'https://github.com/3dg1luk43/ha_washdata/blob/main/custom_components/ha_washdata/blueprints/automation/';

const DRYER = {
  id: 'dryer_anti_crease_power_cut',
  name: 'WashData: end a long dryer anti-crease phase',
  description: 'For dryers that keep tumbling.\n\nNeeds WashData 0.5.9 or newer.',
  device_types: ['dryer', 'washer_dryer'],
  min_version: '0.5.9',
  tags: ['anti-crease', 'smart plug'],
  source_url: SRC + 'dryer_anti_crease_power_cut.yaml',
  path: 'ha_washdata/dryer_anti_crease_power_cut.yaml',
  yaml: 'blueprint:\n  name: "WashData: end a long dryer anti-crease phase"\n  domain: automation\n',
  one_click: true,
  inputs: [
    { key: 'washdata_device', name: 'Dryer', description: '', selector: 'device', has_default: false },
    { key: 'wait_minutes', name: 'Wait first', description: '', selector: 'number', has_default: true },
  ],
};
const INDICATOR = {
  id: 'cycle_running_indicator',
  name: 'WashData: turn something on while a cycle runs',
  description: 'Turns a light on while a cycle runs.',
  device_types: ['all'],
  min_version: '0.5.9',
  tags: ['light'],
  source_url: SRC + 'cycle_running_indicator.yaml',
  path: 'ha_washdata/cycle_running_indicator.yaml',
  yaml: 'blueprint:\n  name: "WashData: turn something on while a cycle runs"\n  domain: automation\n',
  one_click: false,
  inputs: [
    { key: 'washdata_device', name: 'Appliance', description: '', selector: 'device', has_default: false },
    { key: 'target_entity', name: 'Turn on while running', description: '', selector: 'entity', has_default: false },
  ],
};
const CATALOG = { blueprints: [DRYER, INDICATOR] };
const DRYER_OPTIONS = { 'ha_washdata/get_options': { options: { ...optionsData, device_type: 'dryer' }, defaults: {} } };

/** HA's blueprint/list entry for an installed copy of `bp` (metadata only). */
function installed(bp: typeof DRYER, change: Record<string, unknown> = {}) {
  const input: Record<string, unknown> = {};
  for (const i of bp.inputs) input[i.key] = i.has_default ? { name: i.name, default: 1 } : { name: i.name };
  return { [bp.path]: { metadata: { name: bp.name, description: bp.description, source_url: bp.source_url, domain: 'automation', input, ...change } } };
}

async function openTab(
  page,
  overrides: Record<string, unknown> = {},
  hassExtra: Record<string, unknown> = HASS,
  related: Record<string, unknown> | null = null,
) {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/blueprints/list': CATALOG,
    'blueprint/list': {},
    'blueprint/save': { overrides_existing: false },
    'config/label_registry/list': [],
    'config/category_registry/list': [],
    'config/entity_registry/update': {},
    ...overrides,
  }, hassExtra);
  await page.evaluate((rel) => {
    const el: any = document.querySelector('ha-washdata-panel');
    const w = window as any;
    w.__api = [];
    w.__apiFail = null;
    w.__notified = [];
    window.addEventListener('hass-notification', (e: any) => w.__notified.push(e.detail));
    // HA's registries answer a create with the new entry.
    w.__ws_handlers['config/label_registry/create'] = (m: any) => ({ label_id: 'washdata', name: m.name, icon: m.icon, color: null, description: null });
    w.__ws_handlers['config/category_registry/create'] = (m: any) => ({ category_id: 'cat_washdata', name: m.name, icon: m.icon });
    // Record what the panel POSTs to HA's automation config API. Like HA, the
    // automation entity appears a moment after the POST returns.
    el._hass.callApi = (method: string, path: string, body: any) => {
      w.__api.push({ method, path, body });
      const fail = w.__apiFail;
      if (fail) return Promise.reject(fail);
      if (method === 'POST') {
        const id = path.split('/').pop();
        setTimeout(() => {
          const entityId = 'automation.' + id;
          el._hass.states[entityId] = { entity_id: entityId, state: 'on', attributes: { id, friendly_name: body.alias } };
          el._hass.entities[entityId] = { entity_id: entityId, labels: w.__newEntityLabels || [] };
        }, 300);
      }
      return Promise.resolve({});
    };
    el._hass.callWS = (msg: any) => (msg.type === 'search/related' && rel
      ? Promise.resolve(rel) : Promise.reject({ code: 'unknown_command', message: msg.type }));
  }, related);
  await clickTab(page, 'automations');
  await expect(page.locator('.wd-auto-card')).toBeVisible({ timeout: 8_000 });
}

const card = (page, id: string) => page.locator(`.wd-bp-card[data-bpid="${id}"]`);
const apiCalls = (page) => page.evaluate(() => (window as any).__api);

test('the tab follows Settings and Notifications no longer carries the automations', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/blueprints/list': CATALOG, 'blueprint/list': {} }, HASS, { settingsLevel: 'advanced' });
  const ids = await page.locator('button.wd-tab').evaluateAll((els) => els.map((e) => (e as HTMLElement).dataset.tab));
  expect(ids.indexOf('automations')).toBe(ids.indexOf('settings') + 1);

  await clickTab(page, 'settings');
  await page.locator('button[data-sec="notifications"]').first().click();
  await expect(page.locator('input[data-opt="notify_only_when_home"]').first()).toBeAttached({ timeout: 8_000 });
  await expect(page.locator('[data-action="auto-new"]')).toHaveCount(0);
  await expect(page.locator('.wd-auto-pills')).toHaveCount(0);

  await clickTab(page, 'automations');
  await expect(page.locator('[data-action="auto-new"]')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('.wd-bp-section')).toBeVisible();
});

test('a washer sees only the templates for every device type', async ({ page }) => {
  await openTab(page);
  await expect(card(page, 'cycle_running_indicator')).toBeVisible();
  await expect(card(page, 'dryer_anti_crease_power_cut')).toHaveCount(0);
});

test('a dryer sees both templates, with name, tags and inputs', async ({ page }) => {
  await openTab(page, DRYER_OPTIONS);
  const dryer = card(page, 'dryer_anti_crease_power_cut');
  await expect(dryer).toBeVisible();
  await expect(card(page, 'cycle_running_indicator')).toBeVisible();
  await expect(dryer.locator('.wd-bp-name')).toHaveText('End a long dryer anti-crease phase');
  await expect(dryer.locator('.wd-bp-tags .wd-tag')).toHaveText(['anti-crease', 'smart plug']);
  await expect(dryer.locator('.wd-bp-inputs')).toContainText('Wait first');
  // The device input is the panel's to fill, never listed.
  await expect(dryer.locator('.wd-bp-inputs')).not.toContainText('Dryer');
  await expect(card(page, 'cycle_running_indicator').locator('.wd-bp-needed')).toContainText('Turn on while running');
  await expect(page.locator('.wd-bp-footer a')).toHaveAttribute('href', /CONTRIBUTING\.md#automation-templates$/);
});

test('one click installs the blueprint, creates the automation for this device and opens it', async ({ page }) => {
  await openTab(page, DRYER_OPTIONS);
  await card(page, 'dryer_anti_crease_power_cut').locator('[data-action="auto-bp-use"]').click();

  await expect(page).toHaveURL(/\/config\/automation\/edit\/washdata_[a-z0-9]+$/, { timeout: 8_000 });
  const [save] = await assertWsCalled(page, 'blueprint/save');
  expect(save).toMatchObject({ domain: 'automation', path: DRYER.path, yaml: DRYER.yaml, source_url: DRYER.source_url });
  expect(save).not.toHaveProperty('allow_override');

  const [call] = await apiCalls(page);
  expect(call.method).toBe('POST');
  expect(call.path).toMatch(/^config\/automation\/config\/washdata_[a-z0-9]+$/);
  expect(page.url().endsWith(call.path.split('/').pop())).toBe(true);
  expect(call.body.use_blueprint).toEqual({ path: DRYER.path, input: { washdata_device: 'wd-device' } });
  expect(call.body.alias).toBe('Test Washer: End a long dryer anti-crease phase');
  const [update] = await assertWsCalled(page, 'config/entity_registry/update');
  expect(update).toMatchObject({ entity_id: 'automation.' + call.path.split('/').pop(), labels: ['washdata'], categories: { automation: 'cat_washdata' } });
});

test('an installed copy of the same version is not saved again', async ({ page }) => {
  await openTab(page, { ...DRYER_OPTIONS, 'blueprint/list': installed(DRYER) });
  const dryer = card(page, 'dryer_anti_crease_power_cut');
  await expect(dryer.locator('.wd-bp-status')).toHaveText('Installed');
  await dryer.locator('[data-action="auto-bp-use"]').click();

  await expect.poll(() => apiCalls(page).then((c) => c.length)).toBe(1);
  await assertWsNotCalled(page, 'blueprint/save');
});

test('a different installed copy is replaced only after a confirm', async ({ page }) => {
  await openTab(page, { ...DRYER_OPTIONS, 'blueprint/list': installed(DRYER, { description: 'edited by hand' }) });
  const dryer = card(page, 'dryer_anti_crease_power_cut');
  await expect(dryer.locator('.wd-bp-status')).toHaveText('Other version installed');
  await dryer.locator('[data-action="auto-bp-use"]').click();

  await expect(page.locator('.wd-modal')).toContainText('different version');
  await assertWsNotCalled(page, 'blueprint/save');
  await page.locator('.wd-modal [data-maction="ok"]').click();

  await expect.poll(() => apiCalls(page).then((c) => c.length)).toBe(1);
  const [save] = await assertWsCalled(page, 'blueprint/save');
  expect(save.allow_override).toBe(true);
});

test('a failed automation step leaves the blueprint installed and says so', async ({ page }) => {
  await openTab(page, DRYER_OPTIONS);
  await page.evaluate(() => { (window as any).__apiFail = { status_code: 400, body: { message: 'Message malformed' } }; });
  await card(page, 'dryer_anti_crease_power_cut').locator('[data-action="auto-bp-use"]').click();

  await expect(page.locator('.wd-toast')).toContainText('template is installed', { timeout: 8_000 });
  await expect(page.locator('.wd-toast')).toContainText('Message malformed');
  await assertWsCalled(page, 'blueprint/save');
  // Nothing rolled back: one POST, no DELETE; nothing to label.
  expect((await apiCalls(page)).map((c) => c.method)).toEqual(['POST']);
  await expect(page).not.toHaveURL(/\/config\//);
  await assertWsNotCalled(page, 'config/entity_registry/update');
});

test('a template with open inputs is installed and finished in Home Assistant', async ({ page }) => {
  await openTab(page);
  const ind = card(page, 'cycle_running_indicator');
  await expect(ind.locator('[data-action="auto-bp-use"]')).toHaveCount(0);
  await ind.locator('[data-action="auto-bp-install"]').click();

  await expect(page).toHaveURL(/\/config\/blueprint\/dashboard$/, { timeout: 8_000 });
  const [save] = await assertWsCalled(page, 'blueprint/save');
  expect(save).toMatchObject({ domain: 'automation', path: INDICATOR.path, yaml: INDICATOR.yaml });
  expect(await apiCalls(page)).toEqual([]);
});

test('a non-admin reads templates but cannot install them', async ({ page }) => {
  await openTab(page, { ...DRYER_OPTIONS, 'ha_washdata/get_panel_config': { ...panelConfig, is_admin: false } });
  await expect(card(page, 'dryer_anti_crease_power_cut')).toBeVisible();
  await expect(page.locator('.wd-bp-admin-hint')).toBeVisible();
  await expect(page.locator('[data-action="auto-bp-use"], [data-action="auto-bp-install"]')).toHaveCount(0);
  await expect(page.locator('[data-action="auto-new"]')).toHaveCount(0);
  await expect(page.locator('[data-action="auto-toggle-triggers"]')).toHaveCount(0);
  // HA's automation list is under /config, which is admin-only.
  await expect(page.locator('a.wd-auto-all')).toHaveCount(0);
  await expect(page.locator('[data-action="auto-bp-yaml"]')).toHaveCount(2);
  await assertWsNotCalled(page, 'blueprint/list');
});

test('View YAML shows the blueprint, copies it and links to My Home Assistant', async ({ page }) => {
  await openTab(page);
  await page.evaluate(() => {
    (window as any).__copied = null;
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: (t: string) => { (window as any).__copied = t; return Promise.resolve(); } },
    });
  });
  await card(page, 'cycle_running_indicator').locator('[data-action="auto-bp-yaml"]').click();
  const modal = page.locator('.wd-modal');
  await expect(modal.locator('.wd-bp-yaml')).toHaveText(INDICATOR.yaml.trimEnd());
  await expect(modal.locator('a.wd-bp-import')).toHaveAttribute('href',
    'https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=' + encodeURIComponent(INDICATOR.source_url));

  await modal.locator('[data-maction="bp-copy"]').click();
  await expect.poll(() => page.evaluate(() => (window as any).__copied)).toBe(INDICATOR.yaml);
  await expect(page.locator('.wd-toast')).toContainText('Copied');

  // The modal stays inside the viewport and scrolls its own content.
  const box = await modal.boundingBox();
  const vp = page.viewportSize()!;
  expect(box!.y).toBeGreaterThanOrEqual(0);
  expect(box!.y + box!.height).toBeLessThanOrEqual(vp.height + 0.5);
  expect(box!.x + box!.width).toBeLessThanOrEqual(vp.width + 0.5);
  await modal.locator('[data-maction="cancel"]').click();
  await expect(modal).toHaveCount(0);
});

test('legacy custom actions still offer the converter on the tab', async ({ page }) => {
  const step = { action: 'notify.notify', data: { message: 'done' } };
  await openTab(page, { 'ha_washdata/get_options': { options: { ...optionsData, notify_actions: [step] }, defaults: {} } });
  await expect(page.locator('.wd-auto-legacy')).toBeVisible();
  await expect(page.locator('[data-action="auto-convert-legacy"]')).toBeVisible();
});

// ── v2: labels, category, the full trigger list, the default tab ──

const WASHER_ID = /\/config\/automation\/edit\/washdata_[a-z0-9]+$/;
const triggerKeys = (page) => page.locator('[data-action="auto-new-trigger"]').evaluateAll(
  (els) => els.map((e) => (e as HTMLElement).dataset.trigger));
const typeOptions = (type: string) => ({ 'ha_washdata/get_options': { options: { ...optionsData, device_type: type }, defaults: {} } });

async function startFromTrigger(page, key: string) {
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  await page.locator(`[data-action="auto-new-trigger"][data-trigger="${key}"]`).click();
}

test('a trigger creates a labelled automation, creating the label and the category', async ({ page }) => {
  await openTab(page);
  await startFromTrigger(page, 'finishing_soon');
  await expect(page).toHaveURL(WASHER_ID, { timeout: 8_000 });

  const [call] = await apiCalls(page);
  expect(call.method).toBe('POST');
  expect(call.body).toMatchObject({
    alias: 'Test Washer: Finishing soon',
    triggers: [{ trigger: 'ha_washdata.finishing_soon', target: { device_id: ['wd-device'] }, options: { minutes: 10 } }],
    conditions: [],
    actions: [],
  });
  const id = call.path.split('/').pop();
  expect(page.url().endsWith(id)).toBe(true);

  const [label] = await assertWsCalled(page, 'config/label_registry/create');
  expect(label).toMatchObject({ name: 'WashData', icon: 'mdi:washing-machine' });
  const [cat] = await assertWsCalled(page, 'config/category_registry/create');
  expect(cat).toMatchObject({ scope: 'automation', name: 'WashData' });
  const [list] = await assertWsCalled(page, 'config/category_registry/list');
  expect(list.scope).toBe('automation');
  const [update] = await assertWsCalled(page, 'config/entity_registry/update');
  expect(update).toEqual({
    type: 'config/entity_registry/update',
    entity_id: 'automation.' + id,
    labels: ['washdata'],
    categories: { automation: 'cat_washdata' },
  });
});

test('an existing label and category are reused by name and existing labels kept', async ({ page }) => {
  await openTab(page, {
    'config/label_registry/list': [{ label_id: 'other', name: 'Kitchen' }, { label_id: 'wd_label', name: 'washdata' }],
    'config/category_registry/list': [{ category_id: 'wd_cat', name: 'WASHDATA ' }],
  });
  await page.evaluate(() => { (window as any).__newEntityLabels = ['other']; });
  await startFromTrigger(page, 'cycle_finished');
  await expect(page).toHaveURL(WASHER_ID, { timeout: 8_000 });

  await assertWsNotCalled(page, 'config/label_registry/create');
  await assertWsNotCalled(page, 'config/category_registry/create');
  const [update] = await assertWsCalled(page, 'config/entity_registry/update');
  expect(update).toMatchObject({ labels: ['other', 'wd_label'], categories: { automation: 'wd_cat' } });
});

test('a failed labelling still creates and opens the automation, with a warning', async ({ page }) => {
  await openTab(page);
  await page.evaluate(() => {
    (window as any).__set_error('config/label_registry/list', { code: 'unauthorized', message: 'Unauthorized' });
    (window as any).__set_error('config/category_registry/list', { code: 'unauthorized', message: 'Unauthorized' });
  });
  await startFromTrigger(page, 'cycle_started');

  await expect(page).toHaveURL(WASHER_ID, { timeout: 8_000 });
  expect((await apiCalls(page)).map((c) => c.method)).toEqual(['POST']);
  await assertWsNotCalled(page, 'config/entity_registry/update');
  await expect(page.locator('.wd-toast')).toContainText('label or category could not be added');
  // Home Assistant's own toast carries it past the navigation into the editor.
  const notified = await page.evaluate(() => (window as any).__notified);
  expect(notified.map((n) => n.message)).toEqual([expect.stringContaining('could not be added')]);
});

test('"New automation" creates a blank labelled automation and opens it', async ({ page }) => {
  await openTab(page);
  await page.locator('[data-action="auto-new"]').click();
  await expect(page).toHaveURL(WASHER_ID, { timeout: 8_000 });
  const [call] = await apiCalls(page);
  expect(call.body).toMatchObject({ alias: 'Test Washer: new automation', triggers: [], actions: [] });
  await assertWsCalled(page, 'config/entity_registry/update');
});

test('the trigger list follows the device type', async ({ page }) => {
  // Washer: matching and anti-wrinkle triggers, no pump.
  await openTab(page);
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  let keys = await triggerKeys(page);
  expect(keys).toContain('anti_wrinkle_started');
  expect(keys).toContain('finishing_soon');
  expect(keys).not.toContain('pump_stuck');
  expect(keys).toHaveLength(17);
  await expect(page.locator('[data-trigger="still_tumbling"]')).toHaveText('Still tumbling after the cycle');

  await openTab(page, typeOptions('dryer'));
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  keys = await triggerKeys(page);
  expect(keys).toContain('still_tumbling');
  expect(keys).not.toContain('pump_stuck');

  await openTab(page, typeOptions('pump'));
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  keys = await triggerKeys(page);
  expect(keys).toContain('pump_stuck');
  expect(keys).toContain('program_recognised');
  expect(keys).not.toContain('anti_wrinkle_started');

  // A Threshold Device never recognises a program.
  await openTab(page, typeOptions('other'));
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  keys = await triggerKeys(page);
  for (const k of ['program_recognised', 'phase_started', 'finishing_soon', 'cycle_overrun', 'anti_wrinkle_started', 'still_tumbling', 'pump_stuck']) {
    expect(keys).not.toContain(k);
  }
  expect(keys).toEqual(['cycle_started', 'cycle_paused', 'cycle_resumed', 'cycle_stalled', 'cycle_finished',
    'cycle_ended', 'unload_pending', 'unloaded', 'state_changed', 'maintenance_due', 'milestone_reached']);
});

test('a trigger with options is saved with their defaults', async ({ page }) => {
  await openTab(page, typeOptions('dryer'));
  await startFromTrigger(page, 'still_tumbling');
  await expect.poll(() => apiCalls(page).then((c) => c.length)).toBe(1);
  const [call] = await apiCalls(page);
  expect(call.body.triggers).toEqual([
    { trigger: 'ha_washdata.still_tumbling', target: { device_id: ['wd-device'] }, options: { minutes: 15 } },
  ]);
});

test('automations carrying the WashData label get a badge, and the list link filters by it', async ({ page }) => {
  const hass = {
    ...HASS,
    states: {
      'automation.wd': { entity_id: 'automation.wd', state: 'on', attributes: { id: 'a1', friendly_name: 'Dryer done' } },
      'automation.own': { entity_id: 'automation.own', state: 'on', attributes: { id: 'a2', friendly_name: 'My own' } },
    },
    entities: {
      'automation.wd': { entity_id: 'automation.wd', labels: ['wd_label'] },
      'automation.own': { entity_id: 'automation.own', labels: [] },
    },
  };
  await openTab(page, { 'config/label_registry/list': [{ label_id: 'wd_label', name: 'WashData' }] }, hass,
    { automation: ['automation.wd', 'automation.own'] });
  await expect(page.locator('.wd-auto-pill[data-autoid="a1"] .wd-auto-badge')).toHaveText('WashData');
  await expect(page.locator('.wd-auto-pill[data-autoid="a2"]')).toBeVisible();
  await expect(page.locator('.wd-auto-pill[data-autoid="a2"] .wd-auto-badge')).toHaveCount(0);
  const link = page.locator('a.wd-auto-all');
  await expect(link).toHaveAttribute('href', '/config/automation/dashboard?historyBack=1&label=wd_label');
  await expect(link).toHaveAttribute('title', /WashData label/);
});

test('without the label the list link opens the unfiltered automation list', async ({ page }) => {
  await openTab(page);
  await expect(page.locator('a.wd-auto-all')).toHaveAttribute('href', '/config/automation/dashboard');
});

test('Automations is offered as the default tab and as a tab to hide', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/blueprints/list': CATALOG,
    'ha_washdata/get_panel_config': { ...panelConfig, prefs: { ...(panelConfig as any).prefs, default_tab: 'automations' } },
  }, HASS);
  // The saved preference opens the panel on the tab.
  await expect(page.locator('button.wd-tab[data-tab="automations"]')).toHaveClass(/active/, { timeout: 8_000 });

  await page.locator('#wd-settings-btn').click();
  await expect(page.locator('#wd-pref-tab option[value="automations"]')).toHaveCount(1, { timeout: 8_000 });
  await expect(page.locator('#wd-pref-tab')).toHaveValue('automations');
  await page.locator('.wd-modal [data-gtab="panel"]').click();
  await expect(page.locator('#wd-ps-deftab option[value="automations"]')).toHaveCount(1, { timeout: 8_000 });
  await expect(page.locator('[data-hidetab="automations"]')).toHaveCount(1);
});

test.describe('phone', () => {
  test.skip(({ isMobile }) => !isMobile, 'mobile-only layout');

  test('template cards stack and their buttons stay inside the card', async ({ page }) => {
    await openTab(page, DRYER_OPTIONS);
    const a = await card(page, 'dryer_anti_crease_power_cut').boundingBox();
    const b = await card(page, 'cycle_running_indicator').boundingBox();
    expect(Math.abs(a!.x - b!.x)).toBeLessThan(1);
    expect(b!.y).toBeGreaterThanOrEqual(a!.y + a!.height);
    for (const btn of await card(page, 'dryer_anti_crease_power_cut').locator('.wd-btn').all()) {
      const r = await btn.boundingBox();
      expect(r!.x + r!.width).toBeLessThanOrEqual(a!.x + a!.width + 0.5);
    }
  });

  test('a long labelled automation pill stays inside the card', async ({ page }) => {
    const name = 'Test Washer: cycle finished, tell everyone in the house at once';
    const hass = {
      ...HASS,
      states: { 'automation.wd': { entity_id: 'automation.wd', state: 'on', attributes: { id: 'a1', friendly_name: name } } },
      entities: { 'automation.wd': { entity_id: 'automation.wd', labels: ['wd_label'] } },
    };
    await openTab(page, { 'config/label_registry/list': [{ label_id: 'wd_label', name: 'WashData' }] }, hass,
      { automation: ['automation.wd'] });
    const pill = page.locator('.wd-auto-pill[data-autoid="a1"]');
    await expect(pill.locator('.wd-auto-badge')).toBeVisible();
    const p = await pill.boundingBox();
    const c = await page.locator('.wd-auto-pills').boundingBox();
    expect(p!.x + p!.width).toBeLessThanOrEqual(c!.x + c!.width + 0.5);
  });

  test('the trigger list stays inside the card', async ({ page }) => {
    await openTab(page, DRYER_OPTIONS);
    await page.locator('[data-action="auto-toggle-triggers"]').click();
    const c = await page.locator('.wd-auto-card').boundingBox();
    const btns = await page.locator('.wd-auto-trig').all();
    expect(btns.length).toBeGreaterThan(10);
    for (const btn of btns) {
      const r = await btn.boundingBox();
      expect(r!.x).toBeGreaterThanOrEqual(c!.x - 0.5);
      expect(r!.x + r!.width).toBeLessThanOrEqual(c!.x + c!.width + 0.5);
    }
  });
});
