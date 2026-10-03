/**
 * Audit 2026-10-02 UI-02: settings tiers.
 *
 * Basic shows the device's identity and the everyday settings instead of
 * suggestion-owned internals. In Advanced, ~13 internal tuning fields are tucked
 * away behind "Show internal settings" - but never out of reach: search finds
 * them, and a field the device has its own value for stays visible.
 */
import { test, expect } from '@playwright/test';
import { bootPanel, setHandler } from '../helpers/panel';
import { buildHandlers } from '../helpers/ws-handlers';
import optionsData from '../fixtures/mock-data/options.json';

async function openSettings(page: any, level: 'basic' | 'advanced', options: Record<string, unknown> = {}) {
  await page.goto('/');
  const base: any = buildHandlers({});
  await bootPanel(page, {
    'ha_washdata/get_panel_config': { ...(base['ha_washdata/get_panel_config'] || {}), prefs: { settings_level: level } },
    'ha_washdata/get_options': { options: { ...(optionsData as any), ...options }, defaults: {} },
  });
  await page.locator('button.wd-tab[data-tab="settings"]').click();
  await expect(page.locator('#wd-settings-form')).toBeVisible({ timeout: 8_000 });
}

const field = (page: any, key: string) => page.locator(`[data-opt="${key}"]`);

test('Basic shows identity and everyday settings, not detection internals', async ({ page }) => {
  await openSettings(page, 'basic');
  // store_brand / store_model: identity too, so a second device can declare its
  // appliance without switching to Advanced.
  for (const key of ['name', 'min_power', 'off_delay', 'store_brand', 'store_model']) {
    await expect(field(page, key).first()).toBeAttached();
  }
  for (const key of ['start_threshold_w', 'min_off_gap', 'sampling_interval']) {
    await expect(field(page, key)).toHaveCount(0);
  }
});

test('internal fields are tucked away in Advanced until asked for', async ({ page }) => {
  await openSettings(page, 'advanced');
  await page.locator('.wd-section-nav [data-sec="timing"]').click();
  await expect(field(page, 'watchdog_interval')).toHaveCount(0);
  await page.locator('#wd-settings-internal-chk').click();
  await expect(field(page, 'watchdog_interval').first()).toBeAttached();
});

test('search always finds an internal field', async ({ page }) => {
  await openSettings(page, 'advanced');
  await page.locator('#wd-settings-search').fill('watchdog');
  await expect(field(page, 'watchdog_interval').first()).toBeAttached();
});

test('an internal field the device has its own value for stays visible', async ({ page }) => {
  await openSettings(page, 'advanced', { watchdog_interval: 90 });
  await page.locator('.wd-section-nav [data-sec="timing"]').click();
  await expect(field(page, 'watchdog_interval').first()).toBeAttached();
});
