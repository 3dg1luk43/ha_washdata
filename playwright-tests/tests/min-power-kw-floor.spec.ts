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

function statesIn(unit: string, state?: string) {
  const attributes: Record<string, string> = { device_class: 'power' };
  if (unit) attributes.unit_of_measurement = unit;
  return {
    states: {
      'sensor.washer_power': { state: state ?? (unit === 'kW' ? '0.002' : '2'), attributes },
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

test('a kW sensor: Min Power 2 over a 1.1 W stop is flagged, its fix never under 1 W', async ({ page }) => {
  await page.goto('/');
  const opts = { ...options, min_power: 2.0, start_threshold_w: 10, stop_threshold_w: 1.1 };
  await bootPanel(page, { 'ha_washdata/get_options': { options: opts } }, statesIn('kW'));
  const err = await minPowerError(page);
  await expect(err).toBeVisible({ timeout: 8_000 });
  // Not waived (1 W fits under a 1.1 W stop), and the fix stays at the 1 W floor.
  await expect(page.locator('.wd-conflict-fix[data-ckey="min_power"]')).toHaveAttribute('data-cval', '1');
});

test('an unavailable sensor without a unit counts as one that could be kW', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_options': { options: LOW_STOP } }, statesIn('', 'unavailable'));
  const err = await minPowerError(page);
  await expect(err).toBeHidden();
});

test('no Stop fix that would reach an explicit Start', async ({ page }) => {
  await page.goto('/');
  const opts = { ...options, min_power: 2.0, start_threshold_w: 2.0, stop_threshold_w: 1.1 };
  await bootPanel(page, { 'ha_washdata/get_options': { options: opts } }, statesIn('W'), { settingsLevel: 'advanced' });
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="detection"]').first().click();
  await expect(page.locator('input[data-opt="stop_threshold_w"]').first()).toBeVisible({ timeout: 8_000 });
  // The conflict shows on Stop, without a fix that would push it to Start.
  await expect(page.locator('[data-cerr="stop_threshold_w"]')).toBeVisible();
  await expect(page.locator('.wd-conflict-fix[data-ckey="stop_threshold_w"]')).toHaveCount(0);
});

test('a Stop fix that stays under Start is still offered', async ({ page }) => {
  await page.goto('/');
  const opts = { ...options, min_power: 2.0, start_threshold_w: 10.0, stop_threshold_w: 1.1 };
  await bootPanel(page, { 'ha_washdata/get_options': { options: opts } }, statesIn('W'), { settingsLevel: 'advanced' });
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="detection"]').first().click();
  await expect(page.locator('.wd-conflict-fix[data-ckey="stop_threshold_w"]')).toHaveAttribute('data-cval', '2.5', { timeout: 8_000 });
});

test('a pending Stop suggestion over the unsaved Start is not recommended', async ({ page }) => {
  await page.goto('/');
  const opts = { ...options, min_power: 6.0, start_threshold_w: 7.0, stop_threshold_w: 5.0 };
  await bootPanel(page, {
    'ha_washdata/get_options': { options: opts },
    'ha_washdata/get_suggestions': {
      suggestions: [{ key: 'stop_threshold_w', current: 5.0, suggested: 8.0, reason: 'r' }],
      locked_suggestions: [],
    },
  }, statesIn('W'), { settingsLevel: 'advanced' });
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="detection"]').first().click();
  await expect(page.locator('[data-cerr="stop_threshold_w"]')).toBeVisible({ timeout: 8_000 });
  // 8 W would resolve Min Power, but invert Start/Stop: neither it nor 7.5 W is offered.
  await expect(page.locator('[data-cerr="stop_threshold_w"] .wd-conflict-sug-note')).toHaveCount(0);
  await expect(page.locator('.wd-conflict-fix[data-ckey="stop_threshold_w"]')).toHaveCount(0);
});

test('a pending Stop suggestion under Start is still recommended', async ({ page }) => {
  await page.goto('/');
  const opts = { ...options, min_power: 6.0, start_threshold_w: 12.0, stop_threshold_w: 5.0 };
  await bootPanel(page, {
    'ha_washdata/get_options': { options: opts },
    'ha_washdata/get_suggestions': {
      suggestions: [{ key: 'stop_threshold_w', current: 5.0, suggested: 8.0, reason: 'r' }],
      locked_suggestions: [],
    },
  }, statesIn('W'), { settingsLevel: 'advanced' });
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="detection"]').first().click();
  await expect(page.locator('[data-cerr="stop_threshold_w"] .wd-conflict-sug-note')).toBeVisible({ timeout: 8_000 });
});
