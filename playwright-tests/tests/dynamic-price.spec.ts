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
import { options } from '../helpers/ws-handlers';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

/**
 * #439: the price picker lists every sensor and a price entity outranks the static
 * price, so picking the plug's own kWh counter costs each cycle at
 * `energy * meter_reading` (the report: 1.30 kWh billed as 7.22 EUR on a 0.37 tariff).
 */
const PRICE_STATES = {
  states: {
    'sensor.washer_energy': { state: '5.554', attributes: { device_class: 'energy', unit_of_measurement: 'kWh' } },
    'sensor.washer_power': { state: '2100', attributes: { device_class: 'power', unit_of_measurement: 'W' } },
    'sensor.spot_tariff': { state: '0.37', attributes: { unit_of_measurement: 'EUR/kWh' } },
  },
};

async function openEnergySection(page) {
  await clickTab(page, 'settings');
  const sec = page.locator('button[data-sec="notifications"]').first();
  await expect(sec).toBeVisible({ timeout: 8_000 });
  await sec.click();
}

test('the price picker offers tariff sensors and hides energy/power ones', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_options': { options: { ...options, energy_sensor: 'sensor.washer_energy' } } }, PRICE_STATES);
  await openEnergySection(page);

  const inp = page.locator('input[data-opt="energy_price_entity"]').first();
  await expect(inp).toHaveCount(1, { timeout: 8_000 });
  await inp.click();
  await inp.fill('sensor.');

  const drop = page.locator('.wd-combo:has(input[data-opt="energy_price_entity"]) .wd-combo-drop');
  await expect(drop.locator('.wd-combo-item[data-val="sensor.spot_tariff"]')).toHaveCount(1, { timeout: 8_000 });
  await expect(drop.locator('.wd-combo-item[data-val="sensor.washer_energy"]')).toHaveCount(0);
  await expect(drop.locator('.wd-combo-item[data-val="sensor.washer_power"]')).toHaveCount(0);
});

test('an energy meter typed into the price field is flagged, and clearing it resolves', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_options': { options: { ...options, energy_sensor: 'sensor.washer_energy' } } }, PRICE_STATES);
  await openEnergySection(page);

  const inp = page.locator('input[data-opt="energy_price_entity"]').first();
  await inp.fill('sensor.washer_energy');
  await inp.dispatchEvent('input');

  const err = page.locator('[data-cerr="energy_price_entity"]');
  await expect(err).toBeVisible({ timeout: 8_000 });
  await expect(err).toContainText('not a price');
  await expect(page.locator('.wd-field[data-field="energy_price_entity"]')).toHaveClass(/wd-has-conflict/);

  await inp.fill('');
  await inp.dispatchEvent('input');
  await expect(err).toBeHidden({ timeout: 8_000 });
});

test('a real tariff sensor in the price field raises nothing', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_options': { options: { ...options, energy_price_entity: 'sensor.spot_tariff' } } }, PRICE_STATES);
  await openEnergySection(page);

  await expect(page.locator('input[data-opt="energy_price_entity"]').first()).toHaveValue('sensor.spot_tariff', { timeout: 8_000 });
  await expect(page.locator('[data-cerr="energy_price_entity"]')).toBeHidden();
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
