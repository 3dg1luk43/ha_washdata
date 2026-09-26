/**
 * Unload confirmation without a door sensor (#451).
 *
 * The reporter cannot fit a contact sensor on the machine door and has a Zigbee
 * button instead, so the two new settings have to be reachable and the
 * confirmation picker has to offer the domains such a button actually lives in -
 * a picker restricted to binary_sensor, like the door one, would be useless here.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

const BUTTON_STATES = {
  states: {
    'event.laundry_button': { state: '2026-09-24T09:00:00+00:00', attributes: { device_class: 'button' } },
    'input_button.laundry_out': { state: '2026-09-24T08:00:00+00:00', attributes: {} },
    'binary_sensor.washer_door': { state: 'off', attributes: { device_class: 'door' } },
  },
};

async function openTriggersSection(page) {
  await clickTab(page, 'settings');
  const sec = page.locator('button[data-sec="triggers"]').first();
  await expect(sec).toBeVisible({ timeout: 8_000 });
  await sec.click();
}

test('the unload confirmation settings render in Triggers & Door', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, BUTTON_STATES);
  await openTriggersSection(page);

  await expect(page.locator('.wd-field[data-field="unload_confirm_entity"]')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('input[data-opt="unload_track_without_door"]')).toHaveCount(1);
});

test('the confirmation picker offers button-ish domains, not just binary_sensor', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, BUTTON_STATES);
  await openTriggersSection(page);

  const inp = page.locator('input[data-opt="unload_confirm_entity"]').first();
  await expect(inp).toHaveCount(1, { timeout: 8_000 });
  await inp.click();
  await inp.fill('la');

  const drop = page.locator('.wd-combo:has(input[data-opt="unload_confirm_entity"]) .wd-combo-drop');
  await expect(drop.locator('.wd-combo-item[data-val="event.laundry_button"]')).toHaveCount(1, { timeout: 8_000 });
  await expect(drop.locator('.wd-combo-item[data-val="input_button.laundry_out"]')).toHaveCount(1);
});

test('the door picker is still restricted to contact-sensor domains', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, BUTTON_STATES);
  await openTriggersSection(page);

  const inp = page.locator('input[data-opt="door_sensor_entity"]').first();
  await inp.click();
  await inp.fill('la');

  const drop = page.locator('.wd-combo:has(input[data-opt="door_sensor_entity"]) .wd-combo-drop');
  await expect(drop.locator('.wd-combo-item[data-val="event.laundry_button"]')).toHaveCount(0);
});
