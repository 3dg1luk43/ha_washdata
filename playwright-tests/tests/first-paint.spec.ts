/**
 * Audit 2026-10-02 UI-19: the first paint used to wait for 7-12 sequential WS
 * round trips (constants, panel config, devices, power history, cycles,
 * suggestions, profiles, setup status, ...) before anything but a spinner showed.
 * The Status tab now paints once the device list and its live data are in, and
 * the per-device reads go out together.
 */

import { test, expect } from '@playwright/test';
import { buildHandlers } from '../helpers/ws-handlers';

test('the Status tab paints while the primed reads are still in flight, and they run in parallel', async ({ page }) => {
  await page.route('**/panel-translations/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));
  await page.goto('/');
  await page.evaluate((h: Record<string, unknown>) => {
    const w = window as any;
    w.__boot_panel(h);
    // Hold the device's cycle list back until the test releases it. _fetchAll only
    // starts after the translation fetch settles, so swapping the handler here,
    // synchronously after boot, is in time.
    const cycles = h['ha_washdata/get_device_cycles'];
    w.__ws_handlers['ha_washdata/get_device_cycles'] = () =>
      new Promise((res) => { w.__releaseCycles = () => res(cycles); });
  }, buildHandlers({}) as any);

  // Painted: the device's status card, not the boot spinner.
  await expect(page.locator('.wd-stat-lbl').first()).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('.wd-body .wd-empty .wd-icon', { hasText: '⏳' })).toHaveCount(0);
  // ...while get_device_cycles is still pending, its siblings were already sent.
  const pending = await page.evaluate(() => typeof (window as any).__releaseCycles === 'function');
  expect(pending).toBe(true);
  for (const t of ['ha_washdata/get_suggestions', 'ha_washdata/get_profiles', 'ha_washdata/get_setup_status']) {
    const n = await page.evaluate((type) => (window as any).__get_calls(type).length, t);
    expect(n, `${t} should not wait for get_device_cycles`).toBeGreaterThan(0);
  }

  // Release: the primed data lands and the panel settles with it. Later reads
  // (the Cycles tab fetches its own page) get the data straight away.
  await page.evaluate((c) => {
    const w = window as any;
    w.__releaseCycles();
    w.__ws_handlers['ha_washdata/get_device_cycles'] = c;
  }, buildHandlers({})['ha_washdata/get_device_cycles'] as any);
  await page.evaluate(() => (window as any).__freeze_poll());
  await page.locator('button.wd-tab[data-tab="history"]').click();
  await expect(page.locator('tr[data-cid="cyc-001"]')).toBeVisible({ timeout: 8_000 });
});

test('a non-Status first tab keeps the spinner until its data is primed', async ({ page }) => {
  // default_tab = history: the Cycles table is built from the primed list, so an
  // early paint would flash "no cycles" before it arrives.
  const base = buildHandlers({});
  const cfg = base['ha_washdata/get_panel_config'] as Record<string, any>;
  const handlers = { ...base, 'ha_washdata/get_panel_config': { ...cfg, prefs: { ...cfg.prefs, default_tab: 'history' } } };
  await page.route('**/panel-translations/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));
  await page.goto('/');
  await page.evaluate((h: Record<string, unknown>) => {
    const w = window as any;
    w.__boot_panel(h);
    const cycles = h['ha_washdata/get_device_cycles'];
    w.__ws_handlers['ha_washdata/get_device_cycles'] = () =>
      new Promise((res) => { w.__releaseCycles = () => res(cycles); });
  }, handlers as any);

  await expect.poll(() => page.evaluate(() => typeof (window as any).__releaseCycles === 'function')).toBe(true);
  // The header is up over the spinner (not a blank panel), and no empty Cycles table.
  await expect(page.locator('.wd-body .wd-empty', { hasText: '⏳' })).toBeVisible();
  await expect(page.locator('tr[data-cid]')).toHaveCount(0);
  await page.evaluate(() => (window as any).__releaseCycles());
  await expect(page.locator('tr[data-cid="cyc-001"]')).toBeVisible({ timeout: 8_000 });
});
