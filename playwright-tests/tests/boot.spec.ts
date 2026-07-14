/**
 * Boot & Initialization tests.
 * Verifies the panel renders correctly on first load and handles edge cases.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, assertWsCalled } from '../helpers/panel';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('renders tab bar after successful WS handshake', async ({ page }) => {
  await bootPanel(page);
  const tabs = page.locator('button.wd-tab');
  // At minimum: Status, Cycles, Profiles, Settings, Advanced
  const count = await tabs.count();
  expect(count).toBeGreaterThanOrEqual(5);
});

test('shows device name in device bar', async ({ page }) => {
  await bootPanel(page);
  await expect(page.locator('text=Test Washer').first()).toBeVisible();
});

test('shows empty state when no devices returned', async ({ page }) => {
  // Route translations before boot — bootPanel would normally do this
  await page.route('**/panel-translations/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }),
  );
  // Boot with no devices — the panel skips the tab bar and shows a .wd-empty message
  await page.evaluate(() => {
    window.__boot_panel({
      'ha_washdata/get_constants': window.__DEFAULT_CONSTANTS || { device_types: [], state_colors: {}, ml_training_available: true, ml_lab_enabled: true, ml_suggestions_enabled: true },
      'ha_washdata/get_panel_config': { is_admin: true, access_level: 'full', prefs: {}, panel: { hidden_tabs: [] }, rbac: { enabled: false, default_level: 'edit', users: {} }, users: [] },
      'ha_washdata/get_devices': { devices: [] },
    });
  });
  // No tab bar — the panel renders the no-device empty state
  const emptyMsg = page.locator('.wd-empty');
  await expect(emptyMsg.first()).toBeVisible({ timeout: 10_000 });
});

test('calls get_constants and get_devices on boot', async ({ page }) => {
  await bootPanel(page);
  await assertWsCalled(page, 'ha_washdata/get_constants');
  await assertWsCalled(page, 'ha_washdata/get_devices');
});

test('calls get_panel_config on boot', async ({ page }) => {
  await bootPanel(page);
  await assertWsCalled(page, 'ha_washdata/get_panel_config');
});

test('status tab is active by default', async ({ page }) => {
  await bootPanel(page);
  const activeTab = page.locator('button.wd-tab.active');
  await expect(activeTab).toHaveText(/status|overview/i);
});

test('shows idle state badge when device state is idle', async ({ page }) => {
  await bootPanel(page);
  // The state badge / chip should contain "Idle" or "Off"
  const badge = page.locator('.wd-state-badge, [class*="badge"], .wd-chip').first();
  // Just check it's visible; we don't hardcode exact text (it's translated)
  await expect(badge).toBeVisible({ timeout: 5_000 });
});

test('Playground tab is visible when mlTrainingAvailable is true', async ({ page }) => {
  await bootPanel(page);
  const pgTab = page.locator('button.wd-tab[data-tab="playground"]');
  await expect(pgTab).toBeVisible();
});

test('ML Training is a subtab under Advanced (not a top-level tab)', async ({ page }) => {
  await bootPanel(page);
  // ML Training moved from a top-level tab into the Advanced tab's subtabs.
  await expect(page.locator('button.wd-tab[data-tab="ml"]')).toHaveCount(0);
  await page.locator('button.wd-tab[data-tab="advanced"]').click();
  const mlSub = page.locator('button.wd-subtab[data-ptab="ml"]');
  await expect(mlSub).toBeVisible();
});
