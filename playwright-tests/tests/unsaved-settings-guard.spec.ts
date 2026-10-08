/**
 * Unsaved Settings edits are guarded everywhere the user can leave them: a tab
 * switch (audit UI-15), a device switch, a link that leaves the integration (the
 * HA sidebar is a plain link in a shadow root) and a browser reload or close.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

const base = require('../fixtures/mock-data/device-idle.json').devices[0];

const TWO_DEVICES = {
  'ha_washdata/get_devices': {
    devices: [
      { ...base },
      { ...base, entry_id: 'test-entry-002', title: 'Second Washer' },
    ],
  },
};

const activeName = (page: any) => page.locator('.wd-devcard.active .wd-devname');

async function editMinPower(page: any) {
  await clickTab(page, 'settings');
  const inp = page.locator('input[data-opt="min_power"]').first();
  await expect(inp).toBeVisible({ timeout: 8_000 });
  await inp.fill('3.5');
  await inp.dispatchEvent('change');
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, TWO_DEVICES);
});

test('switching device with an unsaved edit asks first', async ({ page }) => {
  const first = (await activeName(page).textContent()) || '';
  await editMinPower(page);
  await page.locator('.wd-devcard[data-idx="1"]').click();
  await expect(page.locator('.wd-modal h2')).toContainText(/unsaved/i);
  await page.locator('[data-maction="cancel"]').click();
  await expect(activeName(page)).toHaveText(first);
  await expect(page.locator('input[data-opt="min_power"]').first()).toHaveValue('3.5');

  await page.locator('.wd-devcard[data-idx="1"]').click();
  await page.locator('[data-maction="ok"]').click();
  await expect(activeName(page)).toHaveText('Second Washer');
});

test('switching device without edits does not ask', async ({ page }) => {
  await clickTab(page, 'settings');
  await expect(page.locator('input[data-opt="min_power"]').first()).toBeVisible({ timeout: 8_000 });
  await page.locator('.wd-devcard[data-idx="1"]').click();
  await expect(activeName(page)).toHaveText('Second Washer');
  await expect(page.locator('.wd-modal')).toHaveCount(0);
});

test('a link that leaves the integration is held until the user confirms', async ({ page }) => {
  await editMinPower(page);
  // Stand-in for the HA sidebar: a same-origin link inside another shadow root.
  await page.evaluate(() => {
    const host = document.createElement('div');
    host.id = 'fake-sidebar';
    document.body.appendChild(host);
    const root = host.attachShadow({ mode: 'open' });
    root.innerHTML = '<a id="out" href="/lovelace/0">Overview</a>';
  });
  const before = await page.evaluate(() => window.location.pathname);
  await page.evaluate(() => (document.querySelector('#fake-sidebar') as any).shadowRoot.querySelector('#out').click());
  await expect(page.locator('.wd-modal h2')).toContainText(/unsaved/i);
  expect(await page.evaluate(() => window.location.pathname)).toBe(before);

  await page.locator('[data-maction="ok"]').click();
  await expect.poll(() => page.evaluate(() => window.location.pathname)).toBe('/lovelace/0');
});

test('a reload or close with an unsaved edit asks the browser to confirm', async ({ page }) => {
  const prevented = () => page.evaluate(() => {
    const e = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(e);
    return e.defaultPrevented;
  });
  await clickTab(page, 'settings');
  await expect(page.locator('input[data-opt="min_power"]').first()).toBeVisible({ timeout: 8_000 });
  expect(await prevented()).toBe(false);
  await editMinPower(page);
  expect(await prevented()).toBe(true);
});
