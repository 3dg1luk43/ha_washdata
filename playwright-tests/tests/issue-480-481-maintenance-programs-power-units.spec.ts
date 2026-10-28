/**
 * #480: a maintenance task can be marked done by the programs that run it (the
 * drum clean by Cotton 90). #481: a power sensor that is really an energy counter
 * (kWh) is flagged in Settings and hidden from the picker.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import { options } from '../helpers/ws-handlers';

const MAINT = {
  log: [
    { id: 'm1', date: '2026-10-18T09:00:00+00:00', event_type: 'drum_clean', notes: '',
      cycle_count_at_log: 200, program: 'Cotton 90', cycle_id: 'c9' },
  ],
  due: [],
  event_types: ['descale', 'filter_clean', 'drum_clean', 'bearing_service', 'other'],
  reminders: { drum_clean: 30 },
  cycles_since: { descale: 12, filter_clean: 12, drum_clean: 4, bearing_service: 12, other: 12 },
  custom_tasks: [],
  status: [
    { id: 'drum_clean', custom: false, name: null, cycles_interval: 30, days_interval: 0,
      cycles_since: 4, days_since: 4, due: false, programs: ['Eco'] },
  ],
  programs_available: ['Cotton 40', 'Cotton 90', 'Eco'],
  programs: { drum_clean: ['Eco'] },
  lifetime_cycle_count: 204,
  limits: { tasks_max: 20, name_max: 60, cycles_max: 100000, days_max: 3650 },
};

async function openMaintenance(page) {
  await clickTab(page, 'history');
  const sub = page.locator('[data-hsub="maintenance"]').first();
  await expect(sub).toBeVisible({ timeout: 8_000 });
  await sub.click();
  await expect(page.locator('button[data-action="maint-save-reminders"]')).toBeVisible({ timeout: 8_000 });
}

test('a task names the programs that mark it done, and saving sends only that change', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_maintenance_log': MAINT });
  await openMaintenance(page);

  await expect(page.locator('[data-mstatus="drum_clean"]')).toContainText('Done automatically by: Eco');
  await expect(page.getByText('Done by a Cotton 90 cycle')).toBeVisible();

  const row = page.locator('details[data-mprog-row="drum_clean"]');
  await expect(row.locator('summary')).toContainText('Eco');
  await row.locator('summary').click();
  await row.locator('label.wd-switch-lbl:has(input[data-prog="Cotton 90"])').click();
  await expect(row.locator('input[data-prog="Cotton 90"]')).toBeChecked();

  await page.locator('button[data-action="maint-save-reminders"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_maintenance_programs');
  expect(calls).toHaveLength(1);
  expect(calls[0]).toMatchObject({ task_id: 'drum_clean' });
  expect([...calls[0].programs].sort()).toEqual(['Cotton 90', 'Eco']);
});

const ENERGY_STATES = {
  states: {
    'sensor.washer_energy': { state: '412.7', attributes: { device_class: 'energy', unit_of_measurement: 'kWh' } },
    'sensor.washer_power': { state: '2.1', attributes: { device_class: 'power', unit_of_measurement: 'kW' } },
  },
};

async function openBasic(page) {
  await clickTab(page, 'settings');
  const sec = page.locator('button[data-sec="basic"]').first();
  await expect(sec).toBeVisible({ timeout: 8_000 });
  await sec.click();
}

test('an energy counter as the power sensor is flagged, and the picker hides it', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_options': { options: { ...options, power_sensor: 'sensor.washer_energy' } } }, ENERGY_STATES);
  await openBasic(page);

  const err = page.locator('[data-cerr="power_sensor"]');
  await expect(err).toBeVisible({ timeout: 8_000 });
  await expect(err).toContainText('counts energy (kWh)');

  const inp = page.locator('input[data-opt="power_sensor"]').first();
  await inp.fill('sensor.');
  const drop = page.locator('.wd-combo:has(input[data-opt="power_sensor"]) .wd-combo-drop');
  await expect(drop.locator('.wd-combo-item[data-val="sensor.washer_power"]')).toHaveCount(1, { timeout: 8_000 });
  await expect(drop.locator('.wd-combo-item[data-val="sensor.washer_energy"]')).toHaveCount(0);

  // A kW power sensor is fine: no warning.
  await inp.fill('sensor.washer_power');
  await inp.dispatchEvent('input');
  await expect(err).toBeHidden({ timeout: 8_000 });
});
