/**
 * PR #529: under 1 W, Min Power switches a kW Power Sensor to raw readings
 * (`const.power_unit_scale`), so the Min Power <= Stop fix (0.8 x stop) would stop
 * every cycle from starting. Reconcile holds Min Power at 1 there; the panel must
 * not flag, and offer to fix, what reconcile deliberately left.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import { options } from '../helpers/ws-handlers';

const LOW_STOP = { ...options, min_power: 1.0, start_threshold_w: 5, stop_threshold_w: 0.8 };

function statesIn(unit: string) {
  return {
    states: {
      'sensor.washer_power': { state: unit === 'kW' ? '0.002' : '2', attributes: { device_class: 'power', unit_of_measurement: unit } },
    },
  };
}

async function minPowerError(page: any) {
  await clickTab(page, 'settings');
  await expect(page.locator('input[data-opt="min_power"]').first()).toBeVisible({ timeout: 8_000 });
  return page.locator('[data-cerr="min_power"]');
}

test('a kW sensor: Min Power 1 over a 0.8 W stop is not a conflict', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_options': { options: LOW_STOP } }, statesIn('kW'));
  const err = await minPowerError(page);
  await expect(err).toBeHidden();
  await expect(page.locator('.wd-field[data-field="min_power"]')).not.toHaveClass(/wd-has-conflict/);
});

test('a W sensor: the same values are still flagged', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_options': { options: LOW_STOP } }, statesIn('W'));
  const err = await minPowerError(page);
  await expect(err).toBeVisible({ timeout: 8_000 });
});
