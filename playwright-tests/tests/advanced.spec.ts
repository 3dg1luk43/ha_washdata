/**
 * Advanced / Maintenance / Logs tab tests.
 * Covers: diagnostics section, log viewer, preferences, CRUD operations on maintenance items.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, setHandler } from '../helpers/panel';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

// ─── Diagnostics ────────────────────────────────────────────────────────────

test('advanced tab renders Preferences section by default', async ({ page }) => {
  await clickTab(page, 'advanced');
  // The default subtab in the Advanced panel is "My Preferences"
  const prefsTitle = page.locator('.wd-card-title').filter({ hasText: /Preferences/i }).first();
  await expect(prefsTitle).toBeVisible({ timeout: 8_000 });
});

test('diagnostics section fetches diagnostics data', async ({ page }) => {
  await clickTab(page, 'advanced');
  // Navigate to the diagnostics subtab
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
  await assertWsCalled(page, 'ha_washdata/get_diagnostics');
});

test('diagnostics shows storage statistics', async ({ page }) => {
  await page.evaluate(() => {
    window.__ws_handlers['ha_washdata/get_diagnostics'] = () => ({
      stats: {
        file_size_kb: 200,
        total_cycles: 42,
        total_profiles: 3,
        debug_traces_count: 0,
      },
    });
  });
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
  // 42 cycles should appear somewhere in the diagnostics section
  await expect(page.locator('text=42').first()).toBeVisible({ timeout: 8_000 });
});

// ─── Logs ───────────────────────────────────────────────────────────────────

test('log viewer fetches logs on navigation (admin only subtab)', async ({ page }) => {
  await clickTab(page, 'advanced');
  // Logs subtab is admin-only (data-ptab="logs")
  const logsSubtab = page.locator('[data-ptab="logs"]').first();
  await expect(logsSubtab).toBeVisible({ timeout: 5_000 });
  await logsSubtab.click();
  await assertWsCalled(page, 'ha_washdata/get_logs');
});

test('log entries render in the log viewer', async ({ page }) => {
  await page.evaluate(() => {
    window.__ws_handlers['ha_washdata/get_logs'] = () => ({
      logs: [
        { ts: '2026-07-11T09:00:00+00:00', level: 'info', msg: 'Cycle started' },
        { ts: '2026-07-11T09:01:00+00:00', level: 'debug', msg: 'Power reading: 820W' },
        { ts: '2026-07-11T09:05:00+00:00', level: 'info', msg: 'Matched profile: Cotton 40°C' },
      ],
    });
  });
  await clickTab(page, 'advanced');
  const logsSubtab = page.locator('[data-ptab="logs"]').first();
  await expect(logsSubtab).toBeVisible({ timeout: 5_000 });
  await logsSubtab.click();
  await expect(page.locator('text=Cycle started').first()).toBeVisible({ timeout: 8_000 });
});

test('log viewer filters by component and search (client-side)', async ({ page }) => {
  await page.evaluate(() => {
    window.__ws_handlers['ha_washdata/get_logs'] = () => ({
      logs: [
        { ts: 1752200000, level: 'INFO', logger: 'manager', device: 'Dishwasher', msg: 'Cycle started' },
        { ts: 1752200100, level: 'INFO', logger: 'playground', device: 'Dishwasher', msg: 'Optimize sweep value 120' },
      ],
    });
  });
  await clickTab(page, 'advanced');
  await page.locator('[data-ptab="logs"]').first().click();
  await expect(page.locator('text=Cycle started').first()).toBeVisible({ timeout: 8_000 });
  // Component filter -> only playground records remain.
  await page.locator('.wd-log-filter[data-logfilter="component"][data-ctx="page"]').selectOption('playground');
  await expect(page.locator('text=Optimize sweep value 120').first()).toBeVisible();
  await expect(page.locator('text=Cycle started')).toHaveCount(0);
  // Reset component; search narrows instead.
  await page.locator('.wd-log-filter[data-logfilter="component"][data-ctx="page"]').selectOption('');
  await page.locator('.wd-log-filter[data-logfilter="search"][data-ctx="page"]').fill('sweep');
  await expect(page.locator('text=Optimize sweep value 120').first()).toBeVisible();
  await expect(page.locator('text=Cycle started')).toHaveCount(0);
});

// ─── Preferences / Panel Config ─────────────────────────────────────────────

test('preferences section renders by default', async ({ page }) => {
  await clickTab(page, 'advanced');
  // "My Preferences" is the default subtab content
  const prefsSection = page.locator('text=My Preferences').first();
  await expect(prefsSection).toBeVisible({ timeout: 8_000 });
});

test('preferences fetches panel config', async ({ page }) => {
  await clickTab(page, 'advanced');
  await assertWsCalled(page, 'ha_washdata/get_panel_config');
});

test('saving preferences calls set_user_prefs WS command', async ({ page }) => {
  await clickTab(page, 'advanced');
  // Preferences are the default subtab; save button uses data-action="save-prefs"
  const saveBtn = page.locator('button[data-action="save-prefs"]').first();
  await expect(saveBtn).toBeVisible({ timeout: 8_000 });
  await saveBtn.click();
  await assertWsCalled(page, 'ha_washdata/set_user_prefs');
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

// ─── Recorded / Manual Recording ────────────────────────────────────────────

test('maintenance subtab is visible in advanced tab', async ({ page }) => {
  await clickTab(page, 'advanced');
  const maintTab = page.locator('[data-ptab="maintenance"]').first();
  await expect(maintTab).toBeVisible({ timeout: 8_000 });
  await maintTab.click();
  // Maintenance section renders an "Add maintenance event" button
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
