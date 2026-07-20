/**
 * Header gear "Settings" overlay: integration-wide (device-agnostic) sections.
 * My Preferences (all users) + Panel Settings / Access Control / Online (admin).
 */

import { test, expect } from '@playwright/test';
import { bootPanel, assertWsCalled } from '../helpers/panel';
import constants from '../fixtures/mock-data/constants.json';

async function openGear(page) {
  await page.locator('#wd-settings-btn').click();
  await expect(page.locator('.wd-modal [data-gtab="prefs"]')).toBeVisible({ timeout: 8_000 });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

test('gear button opens the settings overlay with My Preferences by default', async ({ page }) => {
  await openGear(page);
  const prefsTab = page.locator('.wd-modal [data-gtab="prefs"]');
  await expect(prefsTab).toHaveClass(/active/);
  await expect(page.locator('.wd-modal button[data-action="save-prefs"]').first()).toBeVisible({ timeout: 8_000 });
});

test('gear exposes Panel Settings and Access Control for admins', async ({ page }) => {
  await openGear(page);
  await expect(page.locator('.wd-modal [data-gtab="panel"]')).toBeVisible();
  await expect(page.locator('.wd-modal [data-gtab="access"]')).toBeVisible();
});

test('saving preferences from the gear calls set_user_prefs', async ({ page }) => {
  await openGear(page);
  const saveBtn = page.locator('.wd-modal button[data-action="save-prefs"]').first();
  await expect(saveBtn).toBeVisible({ timeout: 8_000 });
  await saveBtn.click();
  await assertWsCalled(page, 'ha_washdata/set_user_prefs');
});

test('switching to Access Control renders the RBAC save button', async ({ page }) => {
  await openGear(page);
  await page.locator('.wd-modal [data-gtab="access"]').click();
  await expect(page.locator('.wd-modal button[data-action="save-rbac"]').first()).toBeVisible({ timeout: 8_000 });
});

test('gear closes via the close button', async ({ page }) => {
  await openGear(page);
  await page.locator('.wd-modal [data-maction="cancel"]').first().click();
  await expect(page.locator('.wd-modal [data-gtab="prefs"]')).toHaveCount(0);
});

test('Online & Community tab appears only when online is available', async ({ page }) => {
  // Default boot: online not advertised -> no Online tab.
  await openGear(page);
  await expect(page.locator('.wd-modal [data-gtab="online"]')).toHaveCount(0);
});

test('Online & Community tab shows when the backend advertises it', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_constants': { ...constants, store_online_available: true, store_online_enabled: true, store_web_origin: 'http://localhost:4567' },
    'ha_washdata/store_status': { enabled: true, connected: false },
  });
  await openGear(page);
  await page.locator('.wd-modal [data-gtab="online"]').click();
  await expect(page.locator('.wd-modal .wd-switch-lbl:has(input[data-action="store-toggle-online"])')).toBeVisible({ timeout: 8_000 });
});
