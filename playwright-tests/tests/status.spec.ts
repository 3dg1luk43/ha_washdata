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

test('onboarding card appears when no profiles exist and onboarding not dismissed', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_profiles': { profiles: [], advisories: [], coverage_gap: null },
    'ha_washdata/get_panel_config': {
      ...require('../fixtures/mock-data/panel-config.json'),
      prefs: { onboarding_dismissed: false },
    },
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        cycle_count: 1,
      }],
    },
    'ha_washdata/get_power_history': { live: [], raw: [], cycle_active: false },
  });
  await expect(page.locator('.wd-onboard').first()).toBeVisible({ timeout: 5_000 });
});

test('skip setup button dismisses the onboarding card', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_profiles': { profiles: [], advisories: [], coverage_gap: null },
    'ha_washdata/get_panel_config': {
      ...require('../fixtures/mock-data/panel-config.json'),
      prefs: { onboarding_dismissed: false },
    },
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        cycle_count: 1,
      }],
    },
    'ha_washdata/get_power_history': { live: [], raw: [], cycle_active: false },
  });
  const card = page.locator('.wd-onboard').first();
  await expect(card).toBeVisible({ timeout: 5_000 });
  await page.locator('[data-action="skip-onboarding"]').click();
  await expect(card).not.toBeVisible({ timeout: 3_000 });
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
