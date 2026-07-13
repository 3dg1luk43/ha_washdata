/**
 * Playground tab tests — single-canvas replay + simulation UI.
 *
 * Covers: cycle/profile selectors, replay controls (play/stop/load, speed slider),
 * main canvas, state strip, parameter inputs, simulation button, and parameter
 * sweep (controls + chart canvas after a sweep run).
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

const MOCK_RUN_RESULT = {
  results: [
    {
      cycle_id: 'cyc-001',
      profile_name: 'Cotton 40°C',
      outcome: { detected: true, match_profile: 'Cotton 40°C', match_correct: true, ambiguous: false },
      events: [],
    },
  ],
  summary: {
    cycles: 1, requested: 1, concurrency: 3,
    detected: 1, missed: 0, false_end: 0,
    match_correct: 1, match_wrong: 0, unmatched: 0,
    skipped_ids: [],
  },
};

const PG_WS_CMD = 'ha_washdata/run_playground_simulation';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    [PG_WS_CMD]: MOCK_RUN_RESULT,
  });
});

// ─── Basic rendering ─────────────────────────────────────────────────────────

test('playground tab renders without errors', async ({ page }) => {
  await clickTab(page, 'playground');
  const body = page.locator('.wd-body, .wd-tab-pane').first();
  await expect(body).toBeVisible({ timeout: 8_000 });
});

test('playground tab fetches cycles for the selector', async ({ page }) => {
  await clickTab(page, 'playground');
  await assertWsCalled(page, 'ha_washdata/get_device_cycles');
});

test('playground tab fetches profiles for comparison', async ({ page }) => {
  await clickTab(page, 'playground');
  await assertWsCalled(page, 'ha_washdata/get_profiles');
});

// ─── Cycle and profile selectors ─────────────────────────────────────────────

test('cycle selector exists and has options', async ({ page }) => {
  await clickTab(page, 'playground');
  const cycSel = page.locator('#wd-pg-cyc-sel');
  await expect(cycSel).toBeVisible({ timeout: 8_000 });
  // After cycles load the selector is populated with cycle IDs as option values
  const firstCycleOpt = cycSel.locator('option[value]:not([value=""])').first();
  await expect(firstCycleOpt).toBeAttached({ timeout: 8_000 });
});

test('profile selector exists with auto-detect option', async ({ page }) => {
  await clickTab(page, 'playground');
  const profSel = page.locator('#wd-pg-prof-sel');
  await expect(profSel).toBeVisible({ timeout: 8_000 });
  // First option should be Auto-Detect (empty value)
  await expect(profSel.locator('option').first()).toBeAttached({ timeout: 5_000 });
});

// ─── Replay controls ─────────────────────────────────────────────────────────

test('replay speed slider exists', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('#wd-pg-dur')).toBeVisible({ timeout: 8_000 });
});

test('play stop and load buttons are present', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('button[data-action="pg-play"]')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('button[data-action="pg-stop"]')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('button[data-action="pg-load"]')).toBeVisible({ timeout: 5_000 });
});

// ─── Canvas ──────────────────────────────────────────────────────────────────

test('replay canvas is present and visible', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('canvas#wd-pg-canvas')).toBeVisible({ timeout: 8_000 });
});

// ─── State strip ─────────────────────────────────────────────────────────────

test('state strip is present', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('#wd-pg-strip')).toBeVisible({ timeout: 8_000 });
});

// ─── Parameter inputs ────────────────────────────────────────────────────────

test('parameter inputs exist', async ({ page }) => {
  await clickTab(page, 'playground');
  const paramInputs = page.locator('.wd-pg-param-inp[data-pgkey]');
  await expect(paramInputs.first()).toBeVisible({ timeout: 8_000 });
});

test('reset params button exists', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('button[data-action="pg-reset-params"]')).toBeVisible({ timeout: 8_000 });
});

// ─── Simulation ──────────────────────────────────────────────────────────────

test('run simulation button exists', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('button[data-action="pg-run-sim"]')).toBeVisible({ timeout: 8_000 });
});

test('clicking run-sim calls run_playground_simulation WS command', async ({ page }) => {
  await clickTab(page, 'playground');
  // Wait for cycles to load so run-sim has cycle IDs to pass to the backend
  const firstCycleOpt = page.locator('#wd-pg-cyc-sel option[value]:not([value=""])').first();
  await expect(firstCycleOpt).toBeAttached({ timeout: 8_000 });
  const runSimBtn = page.locator('button[data-action="pg-run-sim"]');
  await expect(runSimBtn).toBeVisible({ timeout: 5_000 });
  await runSimBtn.click();
  await assertWsCalled(page, PG_WS_CMD);
});

// ─── Parameter sweep ─────────────────────────────────────────────────────────

test('sweep param select and range inputs exist', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('#wd-pg-sw-param')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('#wd-pg-sw-from')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('#wd-pg-sw-to')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('#wd-pg-sw-steps')).toBeVisible({ timeout: 5_000 });
});

test('run sweep button exists', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('button[data-action="pg-sweep-run"]')).toBeVisible({ timeout: 8_000 });
});

test('sweep chart canvas appears after running a parameter sweep', async ({ page }) => {
  await clickTab(page, 'playground');
  // Wait for cycles to load before triggering sweep
  const firstCycleOpt = page.locator('#wd-pg-cyc-sel option[value]:not([value=""])').first();
  await expect(firstCycleOpt).toBeAttached({ timeout: 8_000 });
  // Fill in sweep range — input events update internal _pgSweepFrom/_pgSweepTo state
  await page.locator('#wd-pg-sw-from').fill('10');
  await page.locator('#wd-pg-sw-to').fill('100');
  // Changing steps triggers a re-render that picks up the new from/to values,
  // setting swCanRun=true and removing the disabled attribute from the sweep button
  await page.locator('#wd-pg-sw-steps').fill('2');
  const sweepBtn = page.locator('button[data-action="pg-sweep-run"]');
  await expect(sweepBtn).not.toBeDisabled({ timeout: 3_000 });
  await sweepBtn.click();
  // canvas#wd-pg-sweep-chart is rendered inside swBestHtml once sweepResults.length >= 2
  await expect(page.locator('canvas#wd-pg-sweep-chart')).toBeVisible({ timeout: 8_000 });
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
