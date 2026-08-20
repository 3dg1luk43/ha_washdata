/**
 * Status tab (Overview) tests.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import deviceRunning from '../fixtures/mock-data/device-running.json';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('status tab shows device name', async ({ page }) => {
  await bootPanel(page);
  await expect(page.locator('text=Test Washer').first()).toBeVisible();
});

test('status tab shows idle state chip when device is idle', async ({ page }) => {
  await bootPanel(page);
  // The state chip contains the state label. Look for a badge element.
  const stateBadge = page.locator('.wd-badge, .wd-chip, [class*="state"]').first();
  await expect(stateBadge).toBeVisible({ timeout: 5_000 });
});

test('status tab shows running state and program when cycle is active', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': deviceRunning,
  });
  // Cotton 40°C should be the selected value in the program selector
  const progSelect = page.locator('#wd-status-prog');
  await expect(progSelect).toHaveValue('Cotton 40°C', { timeout: 5_000 });
});

test('progress bar is visible during a running cycle', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': deviceRunning,
  });
  await expect(page.locator('.wd-prog-bg, .wd-prog-fill').first()).toBeVisible({ timeout: 5_000 });
});

test('power curve canvas is present with live power data', async ({ page }) => {
  await bootPanel(page);
  // The status canvas should be present in the DOM
  const canvas = page.locator('#wd-status-canvas, canvas').first();
  await expect(canvas).toBeVisible({ timeout: 8_000 });
});

test('status tab fetches power history on load', async ({ page }) => {
  await bootPanel(page);
  await assertWsCalled(page, 'ha_washdata/get_power_history');
});

test('attention card with suggestions appears when device has suggestions', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        suggestions_count: 3,
        suggestion_keys: ['off_delay', 'stop_threshold_w', 'min_off_gap'],
      }],
    },
  });
  // An attention card with the suggestion count should appear
  const attnCard = page.locator('.wd-attn-card').filter({ hasText: '3' });
  await expect(attnCard).toBeVisible({ timeout: 5_000 });
});

test('clicking the suggestion attention card switches to settings tab', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        suggestions_count: 2,
        suggestion_keys: ['off_delay', 'stop_threshold_w'],
      }],
    },
  });
  const attnCard = page.locator('.wd-attn-card[data-action="goto-suggestions"]').first();
  await expect(attnCard).toBeVisible({ timeout: 5_000 });
  await attnCard.click();
  // Should end up on the settings tab
  const settingsTab = page.locator('button.wd-tab[data-tab="settings"].active');
  await expect(settingsTab).toBeVisible({ timeout: 5_000 });
});

test('device pill badge counts a Calibrated (ML) suggestion', async ({ page }) => {
  // Regression: the pill badge used to render the backend classic count only, so a
  // device whose only tuning suggestions were Calibrated (ML) ones showed no bulb
  // while the Settings tab banner announced them.
  const base = require('../fixtures/mock-data/device-idle.json').devices[0];
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [
        { ...base, suggestions_count: 0, suggestion_keys: [] },
        { ...base, entry_id: 'test-entry-002', title: 'Second Washer', suggestions_count: 0, suggestion_keys: [] },
      ],
    },
    'ha_washdata/get_suggestions': { suggestions: [] },
    // off_delay is 120 in options.json, so 371 is a live ML recommendation.
    'ha_washdata/get_ml_comparison': {
      settings_comparison: { off_delay: { ml_value: 371, ml_reason: 'ML reason' } },
    },
  });
  // Opening Settings is what loads the (expensive) Calibrated comparison.
  await clickTab(page, 'settings');
  const activeBadge = page.locator('.wd-devcard.active .wd-dbadge.sug');
  await expect(activeBadge).toHaveText(/1/, { timeout: 8_000 });
  // Scoped per device: the other pill has no comparison fetched, so no bulb.
  await expect(page.locator('.wd-devcard:not(.active) .wd-dbadge.sug')).toHaveCount(0);
});

test('device pill badge does not double-count a key both engines suggest', async ({ page }) => {
  const base = require('../fixtures/mock-data/device-idle.json').devices[0];
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [
        { ...base, suggestions_count: 1, suggestion_keys: ['off_delay'] },
        { ...base, entry_id: 'test-entry-002', title: 'Second Washer', suggestions_count: 0, suggestion_keys: [] },
      ],
    },
    'ha_washdata/get_suggestions': {
      suggestions: [{ key: 'off_delay', suggested: 371, current: 120, reason: 'Classic reason' }],
    },
    'ha_washdata/get_ml_comparison': {
      settings_comparison: { off_delay: { ml_value: 371, ml_reason: 'ML reason' } },
    },
  });
  await clickTab(page, 'settings');
  const banner = page.locator('.wd-sug-banner').first();
  await expect(banner).toContainText('1 tuning suggestion', { timeout: 8_000 });
  await expect(page.locator('.wd-devcard.active .wd-dbadge.sug')).toHaveText(/1/);
});

test('feedback attention card appears when device has pending feedbacks', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        feedback_count: 2,
      }],
    },
  });
  const feedbackCard = page.locator('[data-action="goto-feedbacks"]');
  await expect(feedbackCard).toBeVisible({ timeout: 5_000 });
});

// Mobile responsiveness
test('status tab renders without horizontal overflow on mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 }); // iPhone 14
  await bootPanel(page);
  const body = page.locator('.wd-body');
  await expect(body).toBeVisible({ timeout: 5_000 });
  // Check no horizontal scrollbar (scrollWidth == clientWidth)
  const overflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const body = el.shadowRoot.querySelector('.wd-body');
    return body ? body.scrollWidth - body.clientWidth : 0;
  });
  expect(overflow).toBeLessThanOrEqual(1); // Allow 1px rounding
});
