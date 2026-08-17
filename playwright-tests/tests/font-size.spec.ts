/**
 * Panel font-size slider (accessibility). Users who find the panel text too small
 * can scale everything from My Preferences. The panel is em-relative and :host has
 * no explicit font-size, so the slider sets the host font-size and the whole panel
 * scales. This verifies the live preview actually changes the host font size.
 */

import { test, expect } from '@playwright/test';
import { bootPanel } from '../helpers/panel';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

test('font-size slider scales the panel live', async ({ page }) => {
  await page.locator('#wd-settings-btn').click();
  await expect(page.locator('.wd-modal [data-gtab="prefs"]')).toBeVisible({ timeout: 8_000 });

  const slider = page.locator('#wd-pref-fontscale');
  await expect(slider).toBeVisible({ timeout: 8_000 });

  // Host starts at the inherited size (no explicit font-size).
  const before = await page.evaluate(
    () => document.querySelector('ha-washdata-panel')!.style.fontSize,
  );
  expect(before === '' || before === '100%').toBeTruthy();

  // Drag the slider up and fire input (live preview path).
  await slider.evaluate((el: HTMLInputElement) => {
    el.value = '1.4';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });

  const after = await page.evaluate(
    () => document.querySelector('ha-washdata-panel')!.style.fontSize,
  );
  // 1.4 -> 140% on the host; everything downstream is em so it all scales.
  expect(after).toBe('140%');

  // The readout reflects the new value.
  await expect(page.locator('#wd-pref-fontscale-val')).toHaveText('140%');
});
