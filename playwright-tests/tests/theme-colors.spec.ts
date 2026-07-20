/**
 * Theme color robustness (issues #314 / #315).
 *
 * A custom theme (e.g. graphite) or HA's accent-color picker can declare
 * --primary-color in rgb() form. getPropertyValue returns that raw token, and
 * the old canvas code built fills by concatenating a hex alpha suffix
 * (`col + '55'`), producing an invalid "rgb(238, 147, 0)55" that made
 * addColorStop throw -- freezing the panel on the loading hourglass. These
 * tests boot with an rgb() primary color and assert the canvases render without
 * an uncaught color-parsing error.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import deviceRunning from '../fixtures/mock-data/device-running.json';

const COLOR_ERROR = /addColorStop|did not match the expected pattern|could not be parsed as a color/i;

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  // Emulate a theme that declares the primary color as rgb() rather than hex.
  await page.addStyleTag({
    content:
      ':root, html, body, ha-washdata-panel { --primary-color: rgb(238, 147, 0); }',
  });
});

test('status curve renders under an rgb() primary color (no addColorStop crash)', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));

  await bootPanel(page, { 'ha_washdata/get_devices': deviceRunning });

  // The status canvas draws a primary-colored fill gradient for the live trace.
  await expect(page.locator('#wd-status-canvas').first()).toBeVisible({ timeout: 8_000 });
  // Re-render via tab switches (the crash in #314 fired on tab navigation).
  await clickTab(page, 'history');
  await clickTab(page, 'status');

  expect(errors.join('\n')).not.toMatch(COLOR_ERROR);
});

test('cycle detail canvas renders under an rgb() primary color (#315)', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));

  await bootPanel(page);
  await clickTab(page, 'history');

  // Open the first cycle -> draws the cycle editor's primary-colored fill.
  const row = page.locator('tr[data-cid]').first();
  await expect(row).toBeVisible({ timeout: 8_000 });
  await row.click();
  await expect(page.locator('.wd-modal')).toBeVisible({ timeout: 5_000 });

  expect(errors.join('\n')).not.toMatch(COLOR_ERROR);
});
