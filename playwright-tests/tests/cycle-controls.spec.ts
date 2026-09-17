/**
 * Status-header cycle controls (Pause / Resume / Force Stop).
 *
 * Regression: the controls were gated on their own hand-copied state list that
 * omitted 'paused', so a cycle the detector had auto-paused (power dropped below
 * the running threshold mid-cycle) showed NO buttons at all -- the user could
 * neither convert it into a user pause nor force-stop it -- while a device in
 * 'ending' still showed them. All three call sites now share _ACTIVE_STATES.
 */

import { test, expect } from '@playwright/test';
import { bootPanel } from '../helpers/panel';
import deviceRunning from '../fixtures/mock-data/device-running.json';

function device(overrides: Record<string, unknown>) {
  return { devices: [{ ...deviceRunning.devices[0], ...overrides }] };
}

const pause = '[data-action="pause-cycle"]';
const resume = '[data-action="resume-cycle"]';
const stop = '[data-action="terminate-cycle"]';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('running cycle offers Pause and Force Stop', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': device({ detector_state: 'running' }) });
  await expect(page.locator(pause)).toBeVisible({ timeout: 5_000 });
  await expect(page.locator(stop)).toBeVisible();
  await expect(page.locator(resume)).toHaveCount(0);
});

test('auto-paused cycle still offers Pause and Force Stop', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': device({ detector_state: 'paused', sub_state: 'Paused' }) });
  await expect(page.locator(stop)).toBeVisible({ timeout: 5_000 });
  await expect(page.locator(pause)).toBeVisible();
  // Resume only applies to a *user* pause; the backend no-ops otherwise.
  await expect(page.locator(resume)).toHaveCount(0);
});

test('ending cycle offers Pause and Force Stop', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': device({ detector_state: 'ending' }) });
  await expect(page.locator(pause)).toBeVisible({ timeout: 5_000 });
  await expect(page.locator(stop)).toBeVisible();
});

test('user-paused cycle offers Resume and Force Stop, not Pause', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': device({ detector_state: 'paused', is_user_paused: true }),
  });
  await expect(page.locator(resume)).toBeVisible({ timeout: 5_000 });
  await expect(page.locator(stop)).toBeVisible();
  await expect(page.locator(pause)).toHaveCount(0);
});

test('idle device offers no cycle controls', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': device({ detector_state: 'off', sub_state: 'Off' }) });
  await expect(page.locator('.wd-badge').first()).toBeVisible({ timeout: 5_000 });
  await expect(page.locator(pause)).toHaveCount(0);
  await expect(page.locator(resume)).toHaveCount(0);
  await expect(page.locator(stop)).toHaveCount(0);
});
