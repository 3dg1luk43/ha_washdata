/**
 * Community Store tab tests.
 *
 * The Store tab is gated behind BOTH a backend capability
 * (get_constants.store_online_available) AND a per-device opt-in option
 * (enable_online_features). These tests boot with both enabled and mock the
 * store_* WS commands.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import constants from '../fixtures/mock-data/constants.json';
import optionsData from '../fixtures/mock-data/options.json';

// Constants with the community store advertised as available.
const STORE_CONSTANTS = {
  ...constants,
  store_online_available: true,
  store_web_origin: 'http://localhost:4567',
};

// Device options with online features opted in + a declared appliance.
function storeOptions(extra: Record<string, unknown> = {}) {
  return { options: { ...optionsData, enable_online_features: true, store_brand: 'Bosch', store_model: 'WAT28401', ...extra } };
}

const SEARCH_RESULTS = {
  items: [
    { id: 'dev-1', brand: 'Bosch', model: 'WAT28401', applianceType: 'washing_machine', profileCount: 3, favoriteCount: 12 },
    { id: 'dev-2', brand: 'Miele', model: 'WCG370', applianceType: 'washing_machine', profileCount: 5, favoriteCount: 30 },
  ],
};

const STORE_PROFILES = {
  items: [
    { id: 'sp-1', program: 'Cotton 40°C', cycleCount: 4 },
    { id: 'sp-2', program: 'Eco 60°C', cycleCount: 2 },
  ],
};

const STORE_CYCLES = {
  items: [
    {
      id: 'sc-1',
      stats: { duration: 3600, energy_wh: 850, peak_w: 2100 },
      trace: { points: [[0, 3], [600, 900], [1800, 1500], [3000, 700], [3600, 3]] },
      uploaderName: 'octocat',
      downloads: 7,
    },
  ],
};

// Full override map that enables + fully mocks the community store.
function storeHandlers(extra: Record<string, unknown> = {}) {
  return {
    'ha_washdata/get_constants': STORE_CONSTANTS,
    'ha_washdata/get_options': storeOptions(),
    'ha_washdata/store_status': { enabled: true, connected: true, uid: 'gh_123', name: 'octocat', brand: 'Bosch', model: 'WAT28401' },
    'ha_washdata/store_search_devices': SEARCH_RESULTS,
    'ha_washdata/store_get_profiles': STORE_PROFILES,
    'ha_washdata/store_get_cycles': STORE_CYCLES,
    'ha_washdata/store_import_cycle': { profile: 'Cotton 40°C', cycle_id: 'local-9' },
    'ha_washdata/store_disconnect': { connected: false },
    ...extra,
  };
}

test('store tab is hidden when online features are disabled', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_constants': STORE_CONSTANTS,
    'ha_washdata/get_options': storeOptions({ enable_online_features: false }),
  });
  await expect(page.locator('button.wd-tab[data-tab="store"]')).toHaveCount(0);
});

test('store tab is visible when online features are enabled', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await expect(page.locator('button.wd-tab[data-tab="store"]')).toBeVisible();
});

test('store tab renders device search results', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await clickTab(page, 'store');
  await assertWsCalled(page, 'ha_washdata/store_search_devices');
  const cards = page.locator('[data-action="store-open-device"]');
  await expect(cards.first()).toBeVisible({ timeout: 8_000 });
  await expect(cards).toHaveCount(2);
  await expect(page.locator('.wd-store-item-title').filter({ hasText: 'Bosch WAT28401' })).toBeVisible();
});

test('searching re-queries the store', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await clickTab(page, 'store');
  await expect(page.locator('[data-action="store-open-device"]').first()).toBeVisible({ timeout: 8_000 });
  const before = (await page.evaluate((t: string) => window.__get_calls(t), 'ha_washdata/store_search_devices')).length;
  await page.locator('#wd-store-q').fill('miele');
  await page.locator('[data-action="store-search"]').click();
  await expect
    .poll(async () => (await page.evaluate((t: string) => window.__get_calls(t), 'ha_washdata/store_search_devices')).length)
    .toBeGreaterThan(before);
});

test('clicking a device loads its programs', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await clickTab(page, 'store');
  await page.locator('[data-action="store-open-device"]').first().click();
  await assertWsCalled(page, 'ha_washdata/store_get_profiles');
  const programs = page.locator('[data-action="store-open-profile"]');
  await expect(programs.first()).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('.wd-store-item-title').filter({ hasText: 'Cotton 40°C' })).toBeVisible();
});

test('clicking a program loads its reference cycles with a sparkline', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await clickTab(page, 'store');
  await page.locator('[data-action="store-open-device"]').first().click();
  await page.locator('[data-action="store-open-profile"]').first().click();
  await assertWsCalled(page, 'ha_washdata/store_get_cycles');
  await expect(page.locator('[data-action="store-import"]').first()).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('svg.wd-store-spark').first()).toBeVisible();
  // uploader + download count surfaced on the reference-cycle card
  await expect(page.locator('.wd-store-cycle-stats').filter({ hasText: 'octocat' }).first()).toBeVisible();
});

test('importing a reference cycle calls store_import_cycle', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await clickTab(page, 'store');
  await page.locator('[data-action="store-open-device"]').first().click();
  await page.locator('[data-action="store-open-profile"]').first().click();
  const importBtn = page.locator('[data-action="store-import"]').first();
  await expect(importBtn).toBeVisible({ timeout: 8_000 });
  await importBtn.click();
  // Import choice modal opens (default = new profile named after the program).
  await expect(page.locator('.wd-modal')).toBeVisible({ timeout: 5_000 });
  await page.locator('[data-maction="store-import-ok"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/store_import_cycle');
  expect(calls[0]).toHaveProperty('cycle_id', 'sc-1');
  expect(calls[0]).toHaveProperty('new_profile_name', 'Cotton 40°C');
});

test('settings tab shows the connected community-store account with a disconnect action', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await clickTab(page, 'settings');
  const disconnect = page.locator('[data-action="store-disconnect"]');
  await expect(disconnect).toBeVisible({ timeout: 8_000 });
  await disconnect.click();
  await assertWsCalled(page, 'ha_washdata/store_disconnect');
});
