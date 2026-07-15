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

// Constants with the community store advertised as available AND enabled.
// Online features are now integration-wide (device-agnostic): the panel reads
// store_online_enabled from constants, not a per-device option.
const STORE_CONSTANTS = {
  ...constants,
  store_online_available: true,
  store_online_enabled: true,
  store_web_origin: 'http://localhost:4567',
};

// Device options with a declared appliance (brand/model are still per-device).
function storeOptions(extra: Record<string, unknown> = {}) {
  return { options: { ...optionsData, store_brand: 'Bosch', store_model: 'WAT28401', ...extra } };
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
      status: 'pending',
      confirmCount: 2,
      rating: { avg: 4.5, count: 3 },
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
    'ha_washdata/get_constants': { ...STORE_CONSTANTS, store_online_enabled: false },
    'ha_washdata/get_options': storeOptions(),
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
  await expect(page.locator('.wd-store-row-title').filter({ hasText: 'Bosch WAT28401' })).toBeVisible();
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
  await expect(page.locator('.wd-store-row-title').filter({ hasText: 'Cotton 40°C' })).toBeVisible();
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

test('reference-cycle card shows the approval pill, downloads and rating', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await clickTab(page, 'store');
  await page.locator('[data-action="store-open-device"]').first().click();
  await page.locator('[data-action="store-open-profile"]').first().click();
  const stats = page.locator('.wd-store-cycle-stats').first();
  await expect(stats).toBeVisible({ timeout: 8_000 });
  // Awaiting-approval pill (pending + confirmCount), download count, and rating summary.
  await expect(stats.locator('.wd-tag-pending')).toBeVisible();
  await expect(stats).toContainText('7');            // downloads
  await expect(stats).toContainText('4.5');          // rating avg
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

test('gear "Online & Community" shows the connected account with a disconnect action', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  // Online features + the GitHub connection now live in the header gear overlay.
  await page.locator('#wd-settings-btn').click();
  await page.locator('[data-gtab="online"]').click();
  const disconnect = page.locator('[data-action="store-disconnect"]');
  await expect(disconnect).toBeVisible({ timeout: 8_000 });
  await disconnect.click();
  await assertWsCalled(page, 'ha_washdata/store_disconnect');
});

test('gear online toggle persists via the global store_set_online command', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/get_constants': { ...STORE_CONSTANTS, store_online_enabled: false },
    'ha_washdata/store_set_online': { enabled: true },
  });
  await page.locator('#wd-settings-btn').click();
  await page.locator('[data-gtab="online"]').click();
  await page.locator('input[data-action="store-toggle-online"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/store_set_online');
  expect(calls[0]).toHaveProperty('enabled', true);
});

test('gear "Show contributor names" toggle persists via store_set_prefs', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/get_constants': { ...STORE_CONSTANTS, store_prefs: { show_contributor: true } },
    'ha_washdata/store_set_prefs': { prefs: { show_contributor: false } },
  });
  await page.locator('#wd-settings-btn').click();
  await page.locator('[data-gtab="online"]').click();
  const pref = page.locator('input[data-action="store-toggle-pref"][data-pref="show_contributor"]');
  await expect(pref).toBeVisible({ timeout: 8_000 });
  await pref.click();
  const calls = await assertWsCalled(page, 'ha_washdata/store_set_prefs');
  expect(calls[0].prefs).toHaveProperty('show_contributor', false);
});
