/**
 * #535: Min Power is no limit for an explicit Stop Threshold (an unset one is 60% of
 * it, and once Start and Stop are set no detection threshold reads Min Power), and
 * an emptied Start or Stop Threshold saves as unset so it follows Min Power again.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import { options } from '../helpers/ws-handlers';

function statesIn(unit: string) {
  return {
    states: {
      'sensor.washer_power': {
        state: unit === 'kW' ? '0.002' : '2',
        attributes: { device_class: 'power', unit_of_measurement: unit },
      },
    },
  };
}

async function openDetection(page: any, opts: Record<string, unknown>, unit = 'W') {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_options': { options: opts } }, statesIn(unit), { settingsLevel: 'advanced' });
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="detection"]').first().click();
  await expect(page.locator('input[data-opt="stop_threshold_w"]').first()).toBeVisible({ timeout: 8_000 });
}

for (const unit of ['W', 'kW']) {
  test(`a Stop Threshold under Min Power is not a conflict (${unit} sensor)`, async ({ page }) => {
    const opts = { ...options, min_power: 2.0, start_threshold_w: 10, stop_threshold_w: 1.1 };
    await page.goto('/');
    await bootPanel(page, { 'ha_washdata/get_options': { options: opts } }, statesIn(unit), { settingsLevel: 'advanced' });
    await clickTab(page, 'settings');
    await expect(page.locator('input[data-opt="min_power"]').first()).toBeVisible({ timeout: 8_000 });
    await expect(page.locator('[data-cerr="min_power"]')).toBeHidden();
    await page.locator('button[data-sec="detection"]').first().click();
    await expect(page.locator('input[data-opt="stop_threshold_w"]').first()).toBeVisible({ timeout: 8_000 });
    await expect(page.locator('.wd-field[data-field="stop_threshold_w"]')).not.toHaveClass(/wd-has-conflict/);
    await expect(page.locator('[data-cerr="stop_threshold_w"]')).toBeHidden();
    await expect(page.locator('.wd-conflict-fix[data-ckey="stop_threshold_w"]')).toHaveCount(0);
  });
}

test('emptied Start and Stop Thresholds save as unset', async ({ page }) => {
  await openDetection(page, { ...options, min_power: 2.0, start_threshold_w: 10, stop_threshold_w: 1.1 });
  await page.locator('input[data-opt="start_threshold_w"]').first().fill('');
  await page.locator('input[data-opt="stop_threshold_w"]').first().fill('');
  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const sent = calls[calls.length - 1].options as Record<string, unknown>;
  expect(sent).toHaveProperty('start_threshold_w', null);
  expect(sent).toHaveProperty('stop_threshold_w', null);
});

test('an emptied Stop no longer counts against Start', async ({ page }) => {
  await openDetection(page, { ...options, min_power: 2.0, start_threshold_w: 5, stop_threshold_w: 6 });
  await expect(page.locator('[data-cerr="start_threshold_w"]')).toBeVisible();
  await page.locator('input[data-opt="stop_threshold_w"]').first().fill('');
  await expect(page.locator('[data-cerr="start_threshold_w"]')).toBeHidden();
});
