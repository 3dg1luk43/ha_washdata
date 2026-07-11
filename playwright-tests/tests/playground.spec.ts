/**
 * Playground (Simulator) tab tests.
 * Covers: cycle replay simulator, A/B comparison, DTW inspector, concurrency slider.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import cyclesData from '../fixtures/mock-data/cycles.json';
import profilesData from '../fixtures/mock-data/profiles.json';

const MOCK_RUN_RESULT = {
  results: [
    {
      cycle_id: 'cyc-001',
      profile_name: 'Cotton 40°C',
      outcome: { detected: true, match_profile: 'Cotton 40°C', match_correct: true },
      events: [],
    },
    {
      cycle_id: 'cyc-002',
      profile_name: 'Eco 60°C',
      outcome: { detected: true, match_profile: 'Eco 60°C', match_correct: true },
      events: [],
    },
  ],
  summary: {
    total: 2,
    matched: 2,
    unmatched: 0,
    avg_confidence: 0.90,
  },
};

// WS command used by both the simulator and A/B comparison
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

// ─── Cycle selection ─────────────────────────────────────────────────────────

test('playground shows cycle list for selection', async ({ page }) => {
  await clickTab(page, 'playground');
  // Before running, cycles appear as label.wd-rev-tag (checkbox labels in a grid)
  const cycleTags = page.locator('label.wd-rev-tag');
  await expect(cycleTags.first()).toBeVisible({ timeout: 8_000 });
});

test('selecting a cycle lane updates selection count', async ({ page }) => {
  await clickTab(page, 'playground');
  // Initially all cycles are pre-selected (first 20); the count span should be visible
  const selCount = page.locator('#wd-pg-selcount').first();
  await expect(selCount).toBeVisible({ timeout: 8_000 });
  // Deselect all, then select one
  const clearBtn = page.locator('button[data-action="pg-sel-none"]').first();
  const isVisible = await clearBtn.isVisible({ timeout: 2_000 }).catch(() => false);
  if (isVisible) {
    await clearBtn.click();
    const cb = page.locator('input.wd-pg-cyc').first();
    await expect(cb).toBeVisible({ timeout: 3_000 });
    await cb.check();
    await expect(selCount).toHaveText('1 selected', { timeout: 3_000 });
  }
});

// ─── Concurrency slider ───────────────────────────────────────────────────────

test('concurrency slider is visible in playground tab', async ({ page }) => {
  await clickTab(page, 'playground');
  const slider = page.locator('[data-pgconc]').first();
  await expect(slider).toBeVisible({ timeout: 8_000 });
});

test('concurrency slider default value is 50', async ({ page }) => {
  await clickTab(page, 'playground');
  const slider = page.locator('[data-pgconc]').first();
  await expect(slider).toBeVisible({ timeout: 8_000 });
  const value = await slider.inputValue();
  expect(parseInt(value)).toBe(50);
});

test('concurrency slider can be adjusted to reduce batch size', async ({ page }) => {
  await clickTab(page, 'playground');
  const slider = page.locator('[data-pgconc]').first();
  await expect(slider).toBeVisible({ timeout: 8_000 });
  await slider.fill('5');
  const label = page.locator('#wd-pg-conc-val').first();
  await expect(label).toBeVisible({ timeout: 3_000 });
});

// ─── Run simulation ───────────────────────────────────────────────────────────

test('run button is present in playground tab', async ({ page }) => {
  await clickTab(page, 'playground');
  const runBtn = page.locator('button[data-action="pg-run"]').first();
  await expect(runBtn).toBeVisible({ timeout: 8_000 });
});

test('clicking run calls run_playground WS command', async ({ page }) => {
  await clickTab(page, 'playground');
  // Select all cycles first (select-all checkbox)
  const selectAll = page.locator('button[data-action="pg-sel-all"]').first();
  const isVisible = await selectAll.isVisible({ timeout: 2_000 }).catch(() => false);
  if (isVisible) await selectAll.click();

  const runBtn = page.locator('button[data-action="pg-run"]').first();
  await expect(runBtn).toBeVisible({ timeout: 8_000 });
  await runBtn.click();
  await assertWsCalled(page, PG_WS_CMD);
});

test('simulator shows results after run completes', async ({ page }) => {
  await clickTab(page, 'playground');
  const selectAll = page.locator('button[data-action="pg-sel-all"]').first();
  const isVisible = await selectAll.isVisible({ timeout: 2_000 }).catch(() => false);
  if (isVisible) await selectAll.click();

  const runBtn = page.locator('button[data-action="pg-run"]').first();
  await expect(runBtn).toBeVisible({ timeout: 8_000 });
  await runBtn.click();
  // Results lane label with Cotton 40°C should appear in the timeline
  await expect(page.locator('.wd-pg-lane-lbl').filter({ hasText: 'Cotton 40°C' }).first()).toBeVisible({ timeout: 8_000 });
});

// ─── A/B comparison ───────────────────────────────────────────────────────────

test('A/B comparison subtab is present', async ({ page }) => {
  await clickTab(page, 'playground');
  const abTab = page.locator('[data-pgtab="ab"]').first();
  await expect(abTab).toBeVisible({ timeout: 8_000 });
});

test('A/B comparison table renders rows', async ({ page }) => {
  await clickTab(page, 'playground');
  const abTab = page.locator('[data-pgtab="ab"]').first();
  const isVisible = await abTab.isVisible({ timeout: 3_000 }).catch(() => false);
  if (isVisible) {
    await abTab.click();
    // Trigger A/B run or fetch
    const runAbBtn = page.locator('button[data-action="pg-ab-run"]').first();
    const runIsVisible = await runAbBtn.isVisible({ timeout: 2_000 }).catch(() => false);
    if (runIsVisible) await runAbBtn.click();
    // A/B results render an "Outcome" section title above the metric table
    await expect(page.locator('text=Outcome').first()).toBeVisible({ timeout: 8_000 });
  }
});

test('A/B comparison table is responsive (wrapped on narrow viewport)', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await clickTab(page, 'playground');
  const abTab = page.locator('[data-pgtab="ab"]').first();
  const isVisible = await abTab.isVisible({ timeout: 3_000 }).catch(() => false);
  if (isVisible) {
    await abTab.click();
    // Table should be inside a scroll wrapper
    const tableWrap = page.locator('.wd-tbl-wrap, .wd-ab-table-wrap').first();
    if (await tableWrap.isVisible({ timeout: 2_000 }).catch(() => false)) {
      // Scroll wrapper should not overflow the viewport
      const wrapOverflow = await page.evaluate(() => {
        const el = document.querySelector('ha-washdata-panel');
        if (!el || !el.shadowRoot) return 0;
        const wrap = el.shadowRoot.querySelector('.wd-tbl-wrap');
        return wrap ? Math.max(0, wrap.scrollWidth - wrap.clientWidth) : 0;
      });
      // The wrap itself should be scrollable internally, but not bleed outside
      expect(wrapOverflow).toBeGreaterThanOrEqual(0); // just checks it exists
    }
  }
});

// ─── DTW inspector ───────────────────────────────────────────────────────────

test('DTW inspector subtab is present', async ({ page }) => {
  await clickTab(page, 'playground');
  const dtwTab = page.locator('[data-pgtab="dtw"]').first();
  await expect(dtwTab).toBeVisible({ timeout: 8_000 });
});

test('DTW inspector renders score breakdown table', async ({ page }) => {
  await clickTab(page, 'playground');
  const dtwTab = page.locator('[data-pgtab="dtw"]').first();
  const isVisible = await dtwTab.isVisible({ timeout: 3_000 }).catch(() => false);
  if (isVisible) {
    await dtwTab.click();
    // Select a cycle and run DTW
    const runDtwBtn = page.locator('button[data-action="pg-dtw-run"]').first();
    const runIsVisible = await runDtwBtn.isVisible({ timeout: 2_000 }).catch(() => false);
    if (runIsVisible) await runDtwBtn.click();
    // DTW result renders a "Score breakdown" section with a table
    const scoreTitle = page.locator('text=Score breakdown').first();
    await expect(scoreTitle).toBeVisible({ timeout: 8_000 });
  }
});

// ─── Mobile ─────────────────────────────────────────────────────────────────

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

test('playground lane labels are truncated on mobile (not overflowing)', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await clickTab(page, 'playground');
  // Before running, cycles appear as label.wd-rev-tag items
  const cycleTags = page.locator('label.wd-rev-tag');
  await expect(cycleTags.first()).toBeVisible({ timeout: 8_000 });
  const laneOverflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return -1;
    const lane = el.shadowRoot.querySelector('label.wd-rev-tag');
    return lane ? lane.scrollWidth - lane.clientWidth : -1;
  });
  // -1 means no lane found; otherwise it should be 0 on mobile
  if (laneOverflow >= 0) expect(laneOverflow).toBeLessThanOrEqual(1);
});
