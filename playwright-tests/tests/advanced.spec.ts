/**
 * Advanced tab tests (device-scoped tools only: Maintenance, Diagnostics, ML).
 * My Preferences / Panel Settings / Access Control / Online moved to the header
 * gear (see gear.spec.ts). Logs were removed entirely.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

// ─── Default subtab ───────────────────────────────────────────────────────────

test('advanced tab renders Maintenance by default', async ({ page }) => {
  await clickTab(page, 'advanced');
  const maintTab = page.locator('[data-ptab="maintenance"]').first();
  await expect(maintTab).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('button[data-action="maint-add"]').first()).toBeVisible({ timeout: 8_000 });
});

test('Logs subtab is gone from Advanced', async ({ page }) => {
  await clickTab(page, 'advanced');
  await expect(page.locator('[data-ptab="logs"]')).toHaveCount(0);
});

test('Logs header button is gone (replaced by the settings gear)', async ({ page }) => {
  await expect(page.locator('[data-action="toggle-log-drawer"]')).toHaveCount(0);
  await expect(page.locator('[data-action="open-settings"]')).toBeVisible();
});

// ─── Diagnostics ────────────────────────────────────────────────────────────

test('diagnostics section fetches diagnostics data', async ({ page }) => {
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
  await assertWsCalled(page, 'ha_washdata/get_diagnostics');
});

test('diagnostics shows storage statistics', async ({ page }) => {
  await page.evaluate(() => {
    window.__ws_handlers['ha_washdata/get_diagnostics'] = () => ({
      stats: { file_size_kb: 200, total_cycles: 42, total_profiles: 3, debug_traces_count: 0 },
    });
  });
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
  await expect(page.locator('text=42').first()).toBeVisible({ timeout: 8_000 });
});

// ─── Export / Import ────────────────────────────────────────────────────────

test('export config button is present in diagnostics subtab', async ({ page }) => {
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
  const exportBtn = page.locator('button[data-action="export-config"]').first();
  await expect(exportBtn).toBeVisible({ timeout: 8_000 });
});

test('import config button is present in diagnostics subtab', async ({ page }) => {
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
  const importBtn = page.locator('button[data-action="import-config-open"]').first();
  await expect(importBtn).toBeVisible({ timeout: 8_000 });
});

// ─── Maintenance ──────────────────────────────────────────────────────────────

test('maintenance subtab is visible in advanced tab', async ({ page }) => {
  await clickTab(page, 'advanced');
  const maintTab = page.locator('[data-ptab="maintenance"]').first();
  await expect(maintTab).toBeVisible({ timeout: 8_000 });
  await maintTab.click();
  const maintContent = page.locator('button[data-action="maint-add"]').first();
  await expect(maintContent).toBeVisible({ timeout: 5_000 });
});

// ─── Mobile ─────────────────────────────────────────────────────────────────

test('advanced tab renders without overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await clickTab(page, 'advanced');
  const overflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const body = el.shadowRoot.querySelector('.wd-body');
    return body ? body.scrollWidth - body.clientWidth : 0;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});
