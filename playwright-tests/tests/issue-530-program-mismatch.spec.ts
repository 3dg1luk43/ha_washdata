/**
 * #530: the running cycle does not look like the program picked by hand.
 *
 * The backend's background check sets `program_mismatch` on the device (the pick,
 * the program it looks like, the margin). The Status card says so next to the
 * program selector; the pick itself stays selected. Display only. The
 * Automations tab offers the matching integration trigger.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import deviceRunning from '../fixtures/mock-data/device-running.json';

const ENTRY = 'test-entry-001';
const HASS = {
  devices: {
    wd: { id: 'wd-device', identifiers: [['ha_washdata', ENTRY]], config_entries: [ENTRY] },
  },
};

function running(fields: Record<string, unknown>) {
  const dev = JSON.parse(JSON.stringify(deviceRunning));
  Object.assign(dev.devices[0], fields);
  return dev;
}

const MISMATCH = { selected: 'Cotton 40°C', looks_like: 'Eco 60°C', margin: 0.31 };

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('a hand-picked program the cycle does not look like gets a badge', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({
    manual_program: true, program_mismatch: MISMATCH,
  }) });
  const badge = page.locator('[data-testid="program-mismatch"]');
  await expect(badge).toBeVisible({ timeout: 8_000 });
  await expect(badge).toContainText('Looks more like Eco 60°C');
  // The tip names the pick and says nothing was changed.
  const tip = page.locator('.wd-prog-ctl .wd-tip').last();
  await expect(tip).toHaveAttribute('aria-label', /does not look like Cotton 40°C/);
  await expect(tip).toHaveAttribute('aria-label', /Nothing is changed/);
  // The pick stays selected and is still marked as the user's.
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C');
  await expect(page.locator('.wd-prog-tag.manual')).toBeVisible();
});

test('no badge without a mismatch, or on an auto-detected program', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);

  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_devices': running({
    manual_program: false, program_mismatch: MISMATCH,
  }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);
});

test('no badge once the cycle is over', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({
    detector_state: 'off', manual_program: true, program_mismatch: MISMATCH,
  }) });
  await expect(page.locator('#wd-status-prog')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);
});

test('a slower device list sent before a program change does not bring the badge back', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const stale = running({ manual_program: true, program_mismatch: MISMATCH });
  const fresh = running({ manual_program: true });
  const shown = await page.evaluate(async ({ stale, fresh }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let release: () => void = () => {};
    const slow = new Promise<void>((r) => { release = r; });
    let n = 0;
    el._ws = (msg: any) => {
      if (msg.type !== 'ha_washdata/get_devices') return real(msg);
      n += 1;
      return n === 1 ? slow.then(() => stale) : Promise.resolve(fresh);
    };
    const older = el._fetchAll();   // a poll in flight when the program changed
    await el._fetchAll();           // the refresh after the change
    release();
    await older;
    return el._devices[el._selIdx].program_mismatch ?? null;
  }, { stale, fresh });
  expect(shown).toBeNull();
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);
});

test('the program it looks like renders as text, never as markup', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({
    manual_program: true,
    program_mismatch: { ...MISMATCH, looks_like: '<img src=x id=wd-xss>' },
  }) });
  await expect(page.locator('[data-testid="program-mismatch"]'))
    .toContainText('<img src=x id=wd-xss>', { timeout: 8_000 });
  await expect(page.locator('#wd-xss')).toHaveCount(0);
});

test('the Automations tab starts an automation from the trigger', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/blueprints/list': { blueprints: [] },
    'config/label_registry/list': [],
    'config/category_registry/list': [],
  }, HASS);
  await page.evaluate(() => {
    const el: any = document.querySelector('ha-washdata-panel');
    (window as any).__api = [];
    el._hass.callApi = (method: string, path: string, body: unknown) => {
      (window as any).__api.push({ method, path, body });
      return Promise.resolve({});
    };
  });
  await clickTab(page, 'automations');
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  const item = page.locator('[data-action="auto-new-trigger"][data-trigger="program_mismatch"]');
  await expect(item).toHaveText('Running program differs from the selected one');
  await item.click();
  await expect.poll(() => page.evaluate(() => (window as any).__api.length)).toBeGreaterThan(0);
  const [call] = await page.evaluate(() => (window as any).__api);
  expect(call.body.triggers).toEqual([
    { trigger: 'ha_washdata.program_mismatch', target: { device_id: ['wd-device'] }, options: {} },
  ]);
});
