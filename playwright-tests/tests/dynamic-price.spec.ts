/**
 * Time-weighted energy cost (#426).
 *
 * Two surfaces: the Settings toggle that turns it on, and the Cycles table, where
 * a dynamically-costed cycle has to be distinguishable from one billed at a single
 * frozen price - otherwise the two numbers look identical and the user cannot tell
 * which claim the figure is making.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

test('the time-weighted cost toggle renders under Notifications -> Energy', async ({ page }) => {
  await clickTab(page, 'settings');
  const sec = page.locator('button[data-sec="notifications"]').first();
  await expect(sec).toBeVisible({ timeout: 8_000 });
  await sec.click();
  const toggle = page.locator('input[data-opt="energy_price_dynamic"]').first();
  await expect(toggle).toHaveCount(1, { timeout: 8_000 });
  // Default on: a frozen end-of-cycle price is simply wrong for a moving tariff.
  await expect(toggle).toBeChecked();
});

test('turning the toggle off saves it as false', async ({ page }) => {
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="notifications"]').first().click();
  const field = page.locator('.wd-field-switch:has(input[data-opt="energy_price_dynamic"])').first();
  await expect(field).toBeVisible({ timeout: 8_000 });
  await field.locator('label').first().click();
  await expect(page.locator('input[data-opt="energy_price_dynamic"]').first()).not.toBeChecked();

  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;
  expect(options.energy_price_dynamic).toBe(false);
});

test('a time-weighted cycle shows its effective price, a fixed-price one does not', async ({ page }) => {
  await clickTab(page, 'history');
  const dynamicRow = page.locator('tr[data-cid="cyc-001"]');
  await expect(dynamicRow).toBeVisible({ timeout: 8_000 });

  const marked = dynamicRow.locator('td.wd-tc-num span[title*="Time-weighted"]');
  await expect(marked).toHaveCount(1);
  await expect(marked).toHaveText('0.21 EUR');
  await expect(marked).toHaveAttribute('title', /0\.2471/);

  // The frozen-price cycle shows the same kind of number with no such claim on it.
  const fixedRow = page.locator('tr[data-cid="cyc-002"]');
  await expect(fixedRow.locator('td.wd-tc-num span[title*="Time-weighted"]')).toHaveCount(0);
  await expect(fixedRow.locator('td.wd-tc-num').nth(2)).toHaveText('0.34 EUR');
});
