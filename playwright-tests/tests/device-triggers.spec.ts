/**
 * Automation helpers build WashData triggers (#478).
 *
 * "Start from a trigger" creates an automation prefilled with a WashData
 * integration trigger (`trigger: ha_washdata.<type>` targeting the device). It
 * used to be an `ha_washdata_cycle_*` event trigger, which only ran while the
 * (now retired) "Fire HA events" option was on. With no device registry entry in
 * sight it falls back to the bus event, which always fires, for the two triggers
 * that have one. The legacy-actions converter keeps the native device triggers.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import optionsData from '../fixtures/mock-data/options.json';

const ENTRY = 'test-entry-001';
const WITH_DEVICE = {
  devices: {
    plug: { id: 'plug', identifiers: [['tasmota', 'abc']], config_entries: ['other-entry'] },
    wd: { id: 'wd-device', identifiers: [['ha_washdata', ENTRY]], config_entries: [ENTRY] },
  },
};

async function openAutomations(page, hassExtra = {}, overrides = {}) {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/blueprints/list': { blueprints: [] }, ...overrides }, hassExtra);
  // Record what the panel would POST to HA's automation config API.
  await page.evaluate(() => {
    const el: any = document.querySelector('ha-washdata-panel');
    (window as any).__api = [];
    el._hass.callApi = (method: string, path: string, body: unknown) => {
      (window as any).__api.push({ method, path, body });
      return Promise.resolve({});
    };
  });
  await clickTab(page, 'automations');
  await expect(page.locator('.wd-auto-card')).toBeVisible({ timeout: 8_000 });
}

async function posted(page) {
  await expect.poll(() => page.evaluate(() => (window as any).__api.length)).toBeGreaterThan(0);
  return page.evaluate(() => (window as any).__api);
}

async function startFromTrigger(page, key: string) {
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  await page.locator(`[data-action="auto-new-trigger"][data-trigger="${key}"]`).click();
}

test('"Cycle finished" creates an integration-trigger automation', async ({ page }) => {
  await openAutomations(page, WITH_DEVICE);
  await startFromTrigger(page, 'cycle_finished');

  const [call] = await posted(page);
  expect(call.method).toBe('POST');
  expect(call.path).toMatch(/^config\/automation\/config\/washdata_/);
  expect(call.body.triggers).toEqual([
    { trigger: 'ha_washdata.cycle_finished', target: { device_id: ['wd-device'] }, options: {} },
  ]);
  expect(call.body.actions).toEqual([]);
});

test('"Cycle started" uses the cycle_started integration trigger', async ({ page }) => {
  await openAutomations(page, WITH_DEVICE);
  await startFromTrigger(page, 'cycle_started');

  const [call] = await posted(page);
  expect(call.body.triggers[0]).toEqual(
    { trigger: 'ha_washdata.cycle_started', target: { device_id: ['wd-device'] }, options: {} },
  );
});

test('without a known device the bus event is the fallback', async ({ page }) => {
  await openAutomations(page);
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  // Only the two triggers with a bus event can be built without the device.
  const keys = await page.locator('[data-action="auto-new-trigger"]').evaluateAll(
    (els) => els.map((e) => (e as HTMLElement).dataset.trigger));
  expect(keys).toEqual(['cycle_started', 'cycle_finished']);
  await page.locator('[data-action="auto-new-trigger"][data-trigger="cycle_finished"]').click();

  const [call] = await posted(page);
  expect(call.body.triggers).toEqual([
    { trigger: 'event', event_type: 'ha_washdata_cycle_ended', event_data: { entry_id: ENTRY } },
  ]);
});

test('legacy custom actions convert to an automation on both device triggers', async ({ page }) => {
  const step = { action: 'notify.notify', data: { message: 'done' } };
  await openAutomations(page, WITH_DEVICE, {
    'ha_washdata/get_options': { options: { ...optionsData, notify_actions: [step] }, defaults: {} },
  });
  await page.locator('[data-action="auto-convert-legacy"]').click();

  const [call] = await posted(page);
  expect(call.body.triggers.map((t: any) => [t.trigger, t.device_id, t.type])).toEqual([
    ['device', 'wd-device', 'cycle_started'],
    ['device', 'wd-device', 'cycle_finished'],
  ]);
  expect(call.body.actions).toEqual([step]);
  expect(call.body.description).toContain('{{ trigger.device_name }}');
});
