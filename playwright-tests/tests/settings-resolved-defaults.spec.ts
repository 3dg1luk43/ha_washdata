/**
 * Settings: device-resolved defaults (#396/#393) feeding conflict checks and save.
 *
 * The cadence/ratio fields (sampling_interval, watchdog_interval, start_duration_threshold,
 * smart_termination_duration_ratio) resolve per device type and are sent by the backend as
 * `defaults` (get_options) / `option_defaults` (get_devices), never stored in options. Two
 * behaviours depend on the panel using those defaults for an unset field:
 *   1. A cross-section conflict (watchdog >= 2*sampling) must fire even when the partner
 *      field's section was never opened - otherwise the unset partner reads as undefined and
 *      the rule silently short-circuits.
 *   2. Saving must NOT persist an untouched field that still equals its resolved default -
 *      that would pin the value against future default changes.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import optionsData from '../fixtures/mock-data/options.json';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('watchdog<2*sampling conflict fires from the resolved sampling default (partner section never opened)', async ({ page }) => {
  // sampling_interval is unset in options; the integration would use 30. watchdog 30 < 2*30,
  // so the conflict must show on the Settings tab without visiting the Detection section.
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: { ...optionsData, watchdog_interval: 30 },
      defaults: {
        sampling_interval: 30,
        watchdog_interval: 61,
        start_duration_threshold: 30,
        smart_termination_duration_ratio: 0.98,
      },
    },
  });
  await clickTab(page, 'settings');
  // The Settings-tab conflict banner (distinct from the hidden Overview attention card,
  // which shares the "Setting conflicts" prefix) carries this unique tail.
  await expect(page.getByText(/Check the highlighted sections/i)).toBeVisible({ timeout: 8_000 });
});

test('no conflict when the resolved sampling default keeps watchdog>=2*sampling', async ({ page }) => {
  // Same unset-watchdog=30, but sampling resolves to 2 (a wet appliance), so 30 >= 2*2 and no
  // conflict may appear - proving the check reads the resolved default, not a fixed number.
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: { ...optionsData, watchdog_interval: 30 },
      defaults: {
        sampling_interval: 2,
        watchdog_interval: 30,
        start_duration_threshold: 5,
        smart_termination_duration_ratio: 0.98,
      },
    },
  });
  await clickTab(page, 'settings');
  await expect(page.getByText(/Check the highlighted sections/i)).toHaveCount(0);
});

test('saving an untouched section does not persist fields still equal to their resolved default', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: { ...optionsData },
      defaults: {
        sampling_interval: 2,
        watchdog_interval: 30,
        start_duration_threshold: 5,
        smart_termination_duration_ratio: 0.98,
      },
    },
  });
  await clickTab(page, 'settings');

  // Advanced mode so the Signal Processing / Cycle End fields render.
  const chk = page.locator('#wd-settings-level-chk');
  await expect(chk).toHaveCount(1, { timeout: 5_000 });
  if (!(await chk.isChecked())) {
    await page.locator('.wd-mode-switch .wd-toggle-track').first().click();
    await expect(chk).toBeChecked();
  }

  // Open Detection (renders sampling_interval, start_duration_threshold,
  // smart_termination_duration_ratio - all showing their resolved defaults - plus
  // smoothing_window, which we change to make the form dirty).
  await page.locator('button[data-sec="detection"]').first().click();
  const smoothing = page.locator('input[data-opt="smoothing_window"]').first();
  await expect(smoothing).toBeVisible({ timeout: 8_000 });
  await smoothing.fill('4');

  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;

  // The changed, non-resolved-default field is saved...
  expect(options.smoothing_window).toBe(4);
  // ...but untouched fields still equal to their resolved default are NOT persisted,
  // so they keep resolving per device type.
  expect(options).not.toHaveProperty('sampling_interval');
  expect(options).not.toHaveProperty('start_duration_threshold');
  expect(options).not.toHaveProperty('smart_termination_duration_ratio');
});
