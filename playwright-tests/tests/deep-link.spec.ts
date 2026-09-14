/**
 * Device deep linking (issue #428).
 *
 * `/ha-washdata?device=<entry_id|name>` opens the panel with that appliance
 * selected, so an automation's live-progress notification can link to the device
 * it is about instead of whichever one was viewed last.
 */

import { test, expect } from '@playwright/test';
import { bootPanel } from '../helpers/panel';

const base = require('../fixtures/mock-data/device-idle.json').devices[0];

// Two devices, so the picker renders and "which one is active" is observable.
const TWO_DEVICES = {
  'ha_washdata/get_devices': {
    devices: [
      { ...base },
      { ...base, entry_id: 'test-entry-002', title: 'Trockner Küche' },
    ],
  },
};

/** Name shown on the selected device pill. */
const activeName = (page: any) => page.locator('.wd-devcard.active .wd-devname');

test('?device=<entry_id> selects that device at boot', async ({ page }) => {
  await page.goto('/?device=test-entry-002');
  await bootPanel(page, TWO_DEVICES);
  await expect(activeName(page)).toHaveText('Trockner Küche', { timeout: 5_000 });
});

test('?device=<name> matches the visible title case-insensitively', async ({ page }) => {
  await page.goto('/?device=' + encodeURIComponent('trockner küche'));
  await bootPanel(page, TWO_DEVICES);
  await expect(activeName(page)).toHaveText('Trockner Küche', { timeout: 5_000 });
});

test('?device=<slug> matches a name with spaces and accents', async ({ page }) => {
  await page.goto('/?device=trockner-kuche');
  await bootPanel(page, TWO_DEVICES);
  await expect(activeName(page)).toHaveText('Trockner Küche', { timeout: 5_000 });
});

test('a deep link outranks the remembered device', async ({ page }) => {
  await page.goto('/?device=test-entry-001');
  // Pretend the user last viewed the second device.
  await page.evaluate(() => localStorage.setItem('wd-last-device', 'test-entry-002'));
  await bootPanel(page, TWO_DEVICES);
  await expect(activeName(page)).toHaveText('Test Washer', { timeout: 5_000 });
  // …and the link becomes the new "last used", so the next plain visit agrees.
  expect(await page.evaluate(() => localStorage.getItem('wd-last-device'))).toBe('test-entry-001');
});

test('an unknown ?device= falls back to the remembered device', async ({ page }) => {
  await page.goto('/?device=does-not-exist');
  await page.evaluate(() => localStorage.setItem('wd-last-device', 'test-entry-002'));
  const warnings: string[] = [];
  page.on('console', (m) => { if (m.type() === 'warning') warnings.push(m.text()); });
  await bootPanel(page, TWO_DEVICES);
  await expect(activeName(page)).toHaveText('Trockner Küche', { timeout: 5_000 });
  expect(warnings.join('\n')).toContain('does-not-exist');
});

test('no ?device= leaves the remembered device selected', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.setItem('wd-last-device', 'test-entry-002'));
  await bootPanel(page, TWO_DEVICES);
  await expect(activeName(page)).toHaveText('Trockner Küche', { timeout: 5_000 });
});

test('a second notification tap re-selects while the panel stays mounted', async ({ page }) => {
  await page.goto('/?device=test-entry-002');
  await bootPanel(page, TWO_DEVICES);
  await expect(activeName(page)).toHaveText('Trockner Küche', { timeout: 5_000 });

  // The user switches back by hand, then taps a notification for the dryer again:
  // HA navigates in-app (same element, new query) and fires `location-changed`.
  await page.locator('.wd-devcard[data-idx="0"]').click();
  await expect(activeName(page)).toHaveText('Test Washer', { timeout: 5_000 });
  await page.evaluate(() => {
    history.pushState(null, '', '/?device=test-entry-002');
    window.dispatchEvent(new CustomEvent('location-changed'));
  });
  await expect(activeName(page)).toHaveText('Trockner Küche', { timeout: 5_000 });
});

test('a manual switch survives the next poll', async ({ page }) => {
  // The link must not be re-applied on every refresh, or picking another device
  // from the bar would be undone a few seconds later.
  await page.goto('/?device=test-entry-002');
  await bootPanel(page, TWO_DEVICES);
  await expect(activeName(page)).toHaveText('Trockner Küche', { timeout: 5_000 });

  await page.locator('.wd-devcard[data-idx="0"]').click();
  await expect(activeName(page)).toHaveText('Test Washer', { timeout: 5_000 });

  await page.evaluate(async () => {
    const el = document.getElementById('wd-panel') as any;
    await el._fetchAll();
  });
  await expect(activeName(page)).toHaveText('Test Washer', { timeout: 5_000 });
});
