/**
 * Playground tab tests — unified workbench.
 *
 * The interactive graph + shared detection/matching settings are ALWAYS present;
 * "Test on history" and "Optimize" live in a bottom "Across your cycles" drawer
 * (sub-tabs, not full-page modes) and drive the same graph in place. Every view
 * runs the real backend WS commands — there is no client-side detection copy.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  // The playground WS commands (and get_cycle_power_data) have sensible defaults
  // in helpers/ws-handlers.ts, so no per-test overrides are needed.
  await bootPanel(page, {});
});

// ─── Basic rendering ─────────────────────────────────────────────────────────

test('playground tab renders without errors', async ({ page }) => {
  await clickTab(page, 'playground');
  const body = page.locator('.wd-body, .wd-tab-pane').first();
  await expect(body).toBeVisible({ timeout: 8_000 });
});

test('playground tab fetches cycles and profiles', async ({ page }) => {
  await clickTab(page, 'playground');
  await assertWsCalled(page, 'ha_washdata/get_device_cycles');
  await assertWsCalled(page, 'ha_washdata/get_profiles');
});

// ─── Persistent workbench: graph + Run/Cancel + settings ─────────────────────

test('workbench: cycle/profile selectors, Run control, canvas and strip present', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('#wd-pg-cyc-sel')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('#wd-pg-prof-sel')).toBeVisible();
  await expect(page.locator('button[data-action="pg-run"]')).toBeVisible();
  await expect(page.locator('canvas#wd-pg-canvas')).toBeVisible();
  await expect(page.locator('#wd-pg-strip')).toBeVisible();
  // The removed JS-replay controls must be gone.
  await expect(page.locator('button[data-action="pg-play"]')).toHaveCount(0);
  await expect(page.locator('#wd-pg-dur')).toHaveCount(0);
});

test('workbench: Run calls the faithful backend sim', async ({ page }) => {
  await clickTab(page, 'playground');
  const firstCycleOpt = page.locator('#wd-pg-cyc-sel option[value]:not([value=""])').first();
  await expect(firstCycleOpt).toBeAttached({ timeout: 8_000 });
  await page.locator('button[data-action="pg-run"]').click();
  await assertWsCalled(page, 'ha_washdata/start_playground_cycle_detail');
});

test('workbench: model time-left readout and phase field are present', async ({ page }) => {
  await clickTab(page, 'playground');
  await page.locator('button[data-action="pg-run"]').click();
  // The strip carries the model-estimated remaining time + live phase (not a
  // static countdown).
  await expect(page.locator('#wd-pg-rem')).toBeAttached({ timeout: 8_000 });
  await expect(page.locator('#wd-pg-phase')).toBeAttached();
});

test('workbench: outcome + alerts card appears after a sim run', async ({ page }) => {
  await clickTab(page, 'playground');
  await page.locator('button[data-action="pg-run"]').click();
  await expect(page.locator('.wd-pg-alerts-card')).toBeVisible({ timeout: 8_000 });
});

test('workbench: detection + matching param inputs and reset present', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('.wd-pg-param-inp[data-pgkey]').first()).toBeVisible({ timeout: 8_000 });
  // The matching group exposes only user-configurable matching options.
  await expect(page.locator('.wd-pg-param-inp[data-pgkey="profile_match_min_duration_ratio"]')).toBeAttached();
  await expect(page.locator('.wd-pg-param-inp[data-pgkey="profile_match_max_duration_ratio"]')).toBeAttached();
  await expect(page.locator('button[data-action="pg-reset-params"]')).toBeVisible();
});

test('workbench: editing a param reveals Save-to-settings and it persists', async ({ page }) => {
  await clickTab(page, 'playground');
  const inp = page.locator('.wd-pg-param-inp[data-pgkey="off_delay"]');
  await expect(inp).toBeVisible({ timeout: 8_000 });
  await inp.fill('222');
  const saveBtn = page.locator('button[data-action="pg-apply-settings"]');
  await expect(saveBtn).toBeVisible({ timeout: 8_000 });
  page.once('dialog', d => d.accept());   // confirm()
  await saveBtn.click();
  await assertWsCalled(page, 'ha_washdata/set_options');
});

// ─── "Across your cycles" drawer: History + Optimize sub-tabs ────────────────

test('drawer: History and Optimize sub-tabs are present; History is default', async ({ page }) => {
  await clickTab(page, 'playground');
  const drawer = page.locator('.wd-pg-drawer');
  await expect(drawer).toBeVisible({ timeout: 8_000 });
  const tabs = drawer.locator('.wd-pg-subtabs');
  await expect(tabs.locator('button[data-subtab="history"]')).toHaveClass(/active/);
  await expect(tabs.locator('button[data-subtab="sweep"]')).toBeVisible();
  // The old 3-way mode switch is gone.
  await expect(page.locator('.wd-pg-modeswitch')).toHaveCount(0);
});

test('drawer/history: Run starts a task and shows the result table', async ({ page }) => {
  await clickTab(page, 'playground');
  const runBtn = page.locator('button[data-action="pg-run-history"]');
  await expect(runBtn).toBeVisible({ timeout: 8_000 });
  await runBtn.click();
  // Kicks off a detached, registry-tracked task (not a blocking call), then the
  // completed result loads via the task registry.
  await assertWsCalled(page, 'ha_washdata/start_playground_history');
  await expect(page.locator('table.wd-pg-htable')).toBeVisible({ timeout: 8_000 });
});

test('drawer/history: clicking a row loads that cycle into the graph', async ({ page }) => {
  await clickTab(page, 'playground');
  await page.locator('button[data-action="pg-run-history"]').click();
  const firstRow = page.locator('tr.wd-pg-hrow').first();
  await expect(firstRow).toBeVisible({ timeout: 8_000 });
  await firstRow.click();
  // Drilling a row re-runs the single-cycle sim (drives the graph above) rather
  // than switching to a separate page.
  await assertWsCalled(page, 'ha_washdata/start_playground_cycle_detail');
  await expect(page.locator('.wd-pg-drawer')).toBeVisible();
});

test('drawer/optimize: param + objective selectors and run present', async ({ page }) => {
  await clickTab(page, 'playground');
  await page.locator('.wd-pg-subtabs button[data-subtab="sweep"]').click();
  await expect(page.locator('#wd-pg-sw-param')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('#wd-pg-sw-obj')).toBeVisible();
  await expect(page.locator('button[data-action="pg-sweep-run2"]')).toBeVisible();
});

test('drawer/optimize: running a 1D sweep starts a sweep task', async ({ page }) => {
  await clickTab(page, 'playground');
  await page.locator('.wd-pg-subtabs button[data-subtab="sweep"]').click();
  await page.locator('#wd-pg-sw-from').fill('60');
  await page.locator('#wd-pg-sw-to').fill('240');
  await page.locator('#wd-pg-sw-steps').fill('3');
  await page.locator('button[data-action="pg-sweep-run2"]').click();
  await assertWsCalled(page, 'ha_washdata/start_playground_sweep');
});

// ─── Mobile ──────────────────────────────────────────────────────────────────

test('playground tab renders without overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await clickTab(page, 'playground');
  const overflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const body = el.shadowRoot.querySelector('.wd-body');
    return body ? body.scrollWidth - body.clientWidth : 0;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});
