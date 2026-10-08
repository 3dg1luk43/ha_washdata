/**
 * Audit 2026-10-02 UI-11 / UI-03 (PERF-05): the panel's error, stale and render states.
 *
 * UI-11: a failed get_devices read "No WashData devices configured yet.", a failed
 * get_options left "Loading settings..." up forever, and a lost connection kept
 * showing the last state with no sign it was stale.
 * UI-03: every render built all seven panes; only the active one is built now.
 */
import { test, expect } from '@playwright/test';
import { buildHandlers } from '../helpers/ws-handlers';
import { bootPanel, clickTab } from '../helpers/panel';

test('a failed device list shows an error with Retry, not "no devices"', async ({ page }) => {
  await page.goto('/');
  await page.route('**/panel-translations/**', (route: any) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));
  const handlers: any = buildHandlers({});
  const devices = handlers['ha_washdata/get_devices'];
  delete handlers['ha_washdata/get_devices'];   // the HA-restart unknown_command race
  await page.evaluate(([h]: any) => (window as any).__boot_panel(h, {}), [handlers] as any);
  const err = page.locator('.wd-error-state [data-action="retry-load"]');
  await expect(err).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText('No WashData devices configured yet.')).toHaveCount(0);
  await page.evaluate((d: any) => (window as any).__set_handler('ha_washdata/get_devices', d), devices);
  await err.click();
  await expect(page.locator('button.wd-tab').first()).toBeVisible({ timeout: 10_000 });
});

test('a failed get_options shows Retry instead of loading forever', async ({ page }) => {
  await page.goto('/');
  // get_options fails from boot on: the backend always advertises
  // store_online_available, so a successful boot already primes the options and
  // only a failure that starts at boot can leave Settings with nothing to show.
  const options = buildHandlers({})['ha_washdata/get_options'];
  await bootPanel(page, { 'ha_washdata/get_options': null });
  await page.locator('button.wd-tab[data-tab="settings"]').click();
  const retry = page.locator('.wd-error-state [data-action="retry-tab"]');
  await expect(retry).toBeVisible({ timeout: 8_000 });
  await page.evaluate((o: any) => (window as any).__set_handler('ha_washdata/get_options', o), options);
  await retry.click();
  await expect(page.locator('input[data-opt="name"]').first()).toBeVisible({ timeout: 8_000 });
});

test('a lost connection shows a stale-data chip that clears on recovery', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await expect(page.locator('#wd-stale.wd-stale')).toHaveCount(0);
  await page.evaluate(async () => {
    (window as any).__set_error('ha_washdata/get_devices');
    await (document.getElementById('wd-panel') as any)._fetchAll();
  });
  await expect(page.locator('#wd-stale.wd-stale')).toBeVisible();
  await page.evaluate(async () => {
    delete (window as any).__ws_errors['ha_washdata/get_devices'];
    await (document.getElementById('wd-panel') as any)._fetchAll();
  });
  await expect(page.locator('#wd-stale.wd-stale')).toHaveCount(0);
});

test('only the active pane is built', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  const empty = async () => page.evaluate(() => {
    const sr = (document.getElementById('wd-panel') as any).shadowRoot;
    return Array.from(sr.querySelectorAll('.wd-pane:not(.active)'))
      .every((p: any) => p.childElementCount === 0);
  });
  expect(await empty()).toBe(true);
  await clickTab(page, 'history');
  expect(await empty()).toBe(true);
});
