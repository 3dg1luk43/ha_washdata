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

// ── Device-bundle sharing (Stage 1) ─────────────────────────────────────────

// get_shareable_cycles returns the device's recorded/golden reference cycles
// (the backend already filters + enumerates every page; imported are excluded).
// Two programs, one with two recordings.
const SHAREABLE = {
  items: [
    { id: 'gcyc-1', profile_name: 'Cotton 40°C', start_time: '2026-07-10T08:15:00+00:00', duration: 3600, source: 'recorder' },
    { id: 'gcyc-2', profile_name: 'Cotton 40°C', start_time: '2026-05-02T08:15:00+00:00', duration: 3660, source: 'recorder' },
    { id: 'gcyc-3', profile_name: 'Eco 60°C', start_time: '2026-07-08T09:00:00+00:00', duration: 5700, source: 'recorder' },
  ],
  phase_programs: ['Cotton 40°C'],  // only Cotton 40°C has a local phase map
};

test('Profiles tab shows "Share this device" when online + connected + declared', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeHandlers());
  await clickTab(page, 'profiles');
  await expect(page.locator('[data-action="store-share-device"]')).toBeVisible({ timeout: 8_000 });
});

test('share-device tree enumerates all reference cycles and uploads the selection', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/get_shareable_cycles': SHAREABLE,
    'ha_washdata/store_upload_device': { ok: true, cycle_ids: ['a', 'b', 'c'], created: 3, duplicates: 0, errors: [] },
  });
  await clickTab(page, 'profiles');
  await page.locator('[data-action="store-share-device"]').click();
  // Two programs grouped; all three reference cycles listed (not just page 1).
  const groups = page.locator('.wd-sd-group');
  await expect(groups).toHaveCount(2, { timeout: 8_000 });
  await expect(page.locator('.wd-sd-cyc')).toHaveCount(3);
  // All checked by default -> Share enabled -> uploads every selected cycle.
  const shareBtn = page.locator('[data-maction="store-share-device-ok"]');
  await expect(shareBtn).toBeEnabled();
  await shareBtn.click();
  const calls = await assertWsCalled(page, 'ha_washdata/store_upload_device');
  const items = calls[0].items as Array<{ local_cycle_id: string; program: string }>;
  expect(items).toHaveLength(3);
  expect(items.every((i) => i.local_cycle_id && i.program)).toBe(true);
  // The program with a local phase map bundles its phases by default.
  expect(calls[0].include_phases).toEqual(['Cotton 40°C']);
});

test('share tree can bundle device settings via the include-settings toggle', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/get_shareable_cycles': SHAREABLE,
    'ha_washdata/store_upload_device': { ok: true, cycle_ids: ['a', 'b', 'c'], created: 3, duplicates: 0, errors: [] },
  });
  await clickTab(page, 'profiles');
  await page.locator('[data-action="store-share-device"]').click();
  const settingsToggle = page.locator('[data-maction="sd-toggle-settings"]');
  await expect(settingsToggle).toBeVisible({ timeout: 8_000 });
  await expect(settingsToggle).not.toBeChecked();  // off by default
  await settingsToggle.click();
  await page.locator('[data-maction="store-share-device-ok"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/store_upload_device');
  expect(calls[0].include_settings).toBe(true);
});

test('device download can adopt settings via the opt-in checkbox', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/store_download_device': { profiles_adopted: 1, cycles_imported: 2, phases_applied: 0, settings_applied: 4 },
  });
  await clickTab(page, 'store');
  await page.locator('[data-action="store-open-device"]').first().click();
  const adopt = page.locator('[data-action="store-toggle-dl-settings"]');
  await expect(adopt).toBeVisible({ timeout: 8_000 });
  await adopt.click();
  await page.locator('[data-action="store-download-device"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/store_download_device');
  expect(calls[0].include_settings).toBe(true);
});

test('phase-map toggle shows only for programs with phases and can be opted out', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/get_shareable_cycles': SHAREABLE,
    'ha_washdata/store_upload_device': { ok: true, cycle_ids: ['a', 'b', 'c'], created: 3, duplicates: 0, errors: [] },
  });
  await clickTab(page, 'profiles');
  await page.locator('[data-action="store-share-device"]').click();
  // Exactly one phase toggle (Cotton 40°C); Eco 60°C has no phase map.
  const phaseToggles = page.locator('[data-maction="sd-toggle-phases"]');
  await expect(phaseToggles).toHaveCount(1, { timeout: 8_000 });
  // Opt out, then share -> include_phases must be empty.
  await phaseToggles.first().click();
  await page.locator('[data-maction="store-share-device-ok"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/store_upload_device');
  expect(calls[0].include_phases).toEqual([]);
});

test('share-device reports cycles already in the store as duplicates', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/get_shareable_cycles': SHAREABLE,
    // Every selected cycle's trace was already uploaded -> all duplicates, none new.
    'ha_washdata/store_upload_device': { ok: true, cycle_ids: ['a', 'b', 'c'], created: 0, duplicates: 3, errors: [] },
  });
  await clickTab(page, 'profiles');
  await page.locator('[data-action="store-share-device"]').click();
  await page.locator('[data-maction="store-share-device-ok"]').click();
  await assertWsCalled(page, 'ha_washdata/store_upload_device');
  // Neutral info toast (not a green "shared N" success).
  await expect(page.locator('.wd-toast-info')).toBeVisible({ timeout: 5_000 });
});

test('share-device shows an empty state when there are no shareable cycles', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { ...storeHandlers(), 'ha_washdata/get_shareable_cycles': { items: [] } });
  await clickTab(page, 'profiles');
  await page.locator('[data-action="store-share-device"]').click();
  await expect(page.locator('.wd-sd-group')).toHaveCount(0, { timeout: 8_000 });
  await expect(page.locator('.wd-modal .wd-empty')).toBeVisible();
  // Nothing selectable -> Share is disabled.
  await expect(page.locator('[data-maction="store-share-device-ok"]')).toBeDisabled();
});

test('device view offers "Download this setup" and calls store_download_device', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/store_download_device': { profiles_adopted: 2, cycles_imported: 5 },
  });
  await clickTab(page, 'store');
  await page.locator('[data-action="store-open-device"]').first().click();
  const dl = page.locator('[data-action="store-download-device"]');
  await expect(dl).toBeVisible({ timeout: 8_000 });
  await dl.click();
  const calls = await assertWsCalled(page, 'ha_washdata/store_download_device');
  expect(calls[0]).toHaveProperty('device_id', 'dev-1');
});

test('empty device shows the onboarding banner that jumps to the store', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/get_profiles': { profiles: [] },
    'ha_washdata/get_device_cycles': { cycles: [], total: 0, has_more: false },
  });
  await clickTab(page, 'profiles');
  const onboard = page.locator('[data-action="store-onboard"]');
  await expect(onboard).toBeVisible({ timeout: 8_000 });
  await onboard.click();
  await expect(page.locator('button.wd-tab[data-tab="store"].active')).toBeVisible({ timeout: 5_000 });
});

test('partial device share reports how many uploaded and closes the tree', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/get_shareable_cycles': SHAREABLE,
    // Some cycles uploaded, one failed -> panel must report partial success, not "failed".
    'ha_washdata/store_upload_device': { ok: false, cycle_ids: ['a', 'b'], created: 2, duplicates: 0, errors: ['quota'] },
  });
  await clickTab(page, 'profiles');
  await page.locator('[data-action="store-share-device"]').click();
  await page.locator('[data-maction="store-share-device-ok"]').click();
  await assertWsCalled(page, 'ha_washdata/store_upload_device');
  // Tree closed (partial success is still success for the uploaded cycles).
  await expect(page.locator('.wd-sd-tree')).toHaveCount(0, { timeout: 8_000 });
});

test('downloading a device with nothing new does not claim success', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    ...storeHandlers(),
    'ha_washdata/store_download_device': { profiles_adopted: 0, cycles_imported: 0 },
  });
  await clickTab(page, 'store');
  await page.locator('[data-action="store-open-device"]').first().click();
  const dl = page.locator('[data-action="store-download-device"]');
  await expect(dl).toBeVisible({ timeout: 8_000 });
  await dl.click();
  await assertWsCalled(page, 'ha_washdata/store_download_device');
  // Neutral info toast, not the green "N added" success message.
  await expect(page.locator('.wd-toast-info')).toBeVisible({ timeout: 5_000 });
});
