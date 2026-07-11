/**
 * Settings tab tests.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled } from '../helpers/panel';
import optionsData from '../fixtures/mock-data/options.json';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await clickTab(page, 'settings');
});

test('settings tab renders the device name field', async ({ page }) => {
  const nameInput = page.locator('input[data-opt="name"], input[data-opt="name"]').first();
  await expect(nameInput).toBeVisible({ timeout: 8_000 });
  await expect(nameInput).toHaveValue('Test Washer');
});

test('settings tab fetches options on navigation', async ({ page }) => {
  await assertWsCalled(page, 'ha_washdata/get_options');
});

test('settings tab shows Basic/Advanced toggle', async ({ page }) => {
  const toggle = page.locator('.wd-level-toggle, [data-action="set-settings-level"], button[data-slevel]').first();
  await expect(toggle).toBeVisible({ timeout: 5_000 });
});

test('advanced mode shows more fields than basic mode', async ({ page }) => {
  // Switch to basic mode
  const basicBtn = page.locator('button[data-slevel="basic"]').first();
  await expect(basicBtn).toBeVisible({ timeout: 5_000 });
  await basicBtn.click();
  const basicCount = await page.locator('.wd-field').count();

  // Switch to advanced mode
  const advBtn = page.locator('button[data-slevel="advanced"]').first();
  await advBtn.click();
  const advCount = await page.locator('.wd-field').count();

  expect(advCount).toBeGreaterThan(basicCount);
});

test('editing a field marks the form as dirty (enables save)', async ({ page }) => {
  const nameInput = page.locator('input[data-opt="name"]').first();
  await expect(nameInput).toBeVisible({ timeout: 8_000 });
  await nameInput.fill('Modified Washer');
  // Save button should become enabled
  const saveBtn = page.locator('#wd-settings-save').first();
  await expect(saveBtn).not.toBeDisabled({ timeout: 3_000 });
});

test('saving settings calls set_options WS command', async ({ page }) => {
  const nameInput = page.locator('input[data-opt="name"]').first();
  await expect(nameInput).toBeVisible({ timeout: 8_000 });
  await nameInput.fill('New Washer Name');
  const saveBtn = page.locator('#wd-settings-save').first();
  await saveBtn.click();
  await assertWsCalled(page, 'ha_washdata/set_options');
});

test('search input filters settings to matching fields only', async ({ page }) => {
  const searchInput = page.locator('#wd-settings-search, input[placeholder*="search"], input[placeholder*="filter"]').first();
  await expect(searchInput).toBeVisible({ timeout: 5_000 });
  await searchInput.fill('delay');
  // Fields with "delay" in the label should appear
  const visibleFields = page.locator('.wd-field');
  const count = await visibleFields.count();
  expect(count).toBeGreaterThan(0);
  // Fields without "delay" should not appear — check device name is gone
  await expect(page.locator('input[data-opt="name"]')).not.toBeVisible();
});

test('suggestion banner appears when device has suggestions', async ({ page }) => {
  // Re-boot with suggestions
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        { key: 'off_delay', suggested: 371, current: 180, reason: 'Your device consistently goes off for longer' },
      ],
    },
  });
  await clickTab(page, 'settings');
  const banner = page.locator('.wd-sug-banner').first();
  await expect(banner).toBeVisible({ timeout: 5_000 });
});

test('suggestion widget appears beside the relevant field', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        { key: 'off_delay', suggested: 371, current: 180, reason: 'Test reason' },
      ],
    },
  });
  await clickTab(page, 'settings');
  // Navigate to show the off_delay field
  const field = page.locator('.wd-field[data-field="off_delay"], [data-field="off_delay"]').first();
  await expect(field).toBeVisible({ timeout: 8_000 });
  // The suggestion widget should be inside the field
  const sug = field.locator('.wd-sug').first();
  await expect(sug).toBeVisible({ timeout: 3_000 });
});

test('Use button in suggestion applies the value', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        { key: 'off_delay', suggested: 371, current: 180, reason: 'Test reason' },
      ],
    },
  });
  await clickTab(page, 'settings');
  const useBtn = page.locator('.wd-sug-use').first();
  await expect(useBtn).toBeVisible({ timeout: 8_000 });
  await useBtn.click();
  // The off_delay input should now show 371
  const offDelayInput = page.locator('input[data-opt="off_delay"]').first();
  await expect(offDelayInput).toHaveValue('371', { timeout: 3_000 });
});

test('settings changelog dot appears for changed settings', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_settings_changelog': {
      changelog: [
        { key: 'off_delay', old: '180', new: '371', timestamp: '2026-07-10T14:00:00+00:00' },
      ],
    },
  });
  await clickTab(page, 'settings');
  const dot = page.locator('.wd-chg-dot').first();
  await expect(dot).toBeVisible({ timeout: 5_000 });
});

test('split suggestion widget (Observed vs Calibrated) shows two option rows', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        // suggested must differ from current off_delay (120 in options.json)
        { key: 'off_delay', suggested: 180, current: 120, reason: 'Classic reason' },
      ],
    },
    // Panel reads d.settings_comparison (keyed by field key), not d.comparisons
    'ha_washdata/get_ml_comparison': {
      settings_comparison: {
        off_delay: { ml_value: 371, ml_reason: 'ML reason' },
      },
    },
  });
  await clickTab(page, 'settings');
  // The split case: both classic+ML diverge → two wd-sug-opt divs
  const opts = page.locator('.wd-sug-split .wd-sug-opt');
  await expect(opts).toHaveCount(2, { timeout: 8_000 });
});

test('settings tab renders without overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  // Settings should stack to single column
  const overflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const body = el.shadowRoot.querySelector('.wd-body');
    return body ? body.scrollWidth - body.clientWidth : 0;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});

test('notification fields are present in settings', async ({ page }) => {
  // Navigate to the Notifications section (separate from the default basic section)
  const notifSec = page.locator('button[data-sec="notifications"]').first();
  await expect(notifSec).toBeVisible({ timeout: 8_000 });
  await notifSec.click();
  // notify_fire_events is a checkbox switch; target the visible field container
  const notifyField = page.locator('.wd-field-switch:has(input[data-opt="notify_fire_events"])').first();
  await expect(notifyField).toBeVisible({ timeout: 8_000 });
});

test('revert button appears when changes have been staged', async ({ page }) => {
  const nameInput = page.locator('input[data-opt="name"]').first();
  await expect(nameInput).toBeVisible({ timeout: 8_000 });
  await nameInput.fill('Changed Name');
  const revertBtn = page.locator('#wd-settings-revert').first();
  await expect(revertBtn).toBeVisible({ timeout: 3_000 });
});
