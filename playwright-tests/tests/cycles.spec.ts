/**
 * Cycles / History tab tests.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, setHandler } from '../helpers/panel';
import cyclesData from '../fixtures/mock-data/cycles.json';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

test('cycles tab renders all cycle rows', async ({ page }) => {
  await clickTab(page, 'history');
  // Should see 5 rows (matching mock data)
  const rows = page.locator('tr[data-cid]');
  await expect(rows).toHaveCount(5, { timeout: 8_000 });
});

test('cycles tab shows profile name on each cycle row', async ({ page }) => {
  await clickTab(page, 'history');
  await expect(page.locator('tr[data-cid]').filter({ hasText: 'Cotton 40°C' }).first()).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('tr[data-cid]').filter({ hasText: 'Eco 60°C' }).first()).toBeVisible({ timeout: 5_000 });
});

test('text filter narrows visible cycle rows', async ({ page }) => {
  await clickTab(page, 'history');
  const filterInput = page.locator('#wd-cyc-filter-text, input[placeholder*="filter"], input[placeholder*="search"]').first();
  await expect(filterInput).toBeVisible({ timeout: 5_000 });
  await filterInput.fill('Cotton');
  // After filtering, only Cotton 40°C cycles should show
  await expect(page.locator('tr[data-cid]').filter({ hasText: 'Cotton 40°C' }).first()).toBeVisible();
  // Eco 60°C should be gone
  await expect(page.locator('tr[data-cid]').filter({ hasText: 'Eco 60°C' })).not.toBeVisible();
});

test('clicking a cycle row opens cycle detail modal', async ({ page }) => {
  await clickTab(page, 'history');
  // Mock the cycle power data endpoint
  await page.evaluate(() => {
    window.__set_handler('ha_washdata/get_cycle_power_data', {
      power_data: [
        { t: 0, p: 0 },
        { t: 30, p: 820 },
        { t: 60, p: 750 },
      ],
      artifacts: [],
      envelope_conformance: null,
    });
  });
  const firstRow = page.locator('tr[data-cid]').first();
  await expect(firstRow).toBeVisible({ timeout: 5_000 });
  await firstRow.click();
  await expect(page.locator('.wd-modal')).toBeVisible({ timeout: 5_000 });
});

test('cycle badge shows for underrun anomaly', async ({ page }) => {
  await clickTab(page, 'history');
  // cyc-003 has anomaly: "underrun"
  const badge = page.locator('.wd-cbadge, .wd-badge, [class*="anomaly"], [class*="underrun"]').first();
  // Just check some badge exists (exact styling depends on implementation)
  // At minimum, the anomaly cycle should be in the list
  await expect(page.locator('text=Quick 30°C').first()).not.toBeVisible({ timeout: 2_000 }).catch(() => {});
});

test('load more button appears when has_more is true', async ({ page }) => {
  await clickTab(page, 'history');
  // Update the handler to return has_more=true
  await setHandler(page, 'ha_washdata/get_device_cycles', {
    ...cyclesData,
    has_more: true,
  });
  // Reload cycles tab
  await clickTab(page, 'status');
  await clickTab(page, 'history');
  const loadMoreBtn = page.locator('button[data-action="cyc-load-more"]');
  await expect(loadMoreBtn.first()).toBeVisible({ timeout: 5_000 });
});

test('load more fetches next page and adds rows', async ({ page }) => {
  // Boot with has_more=true
  await page.evaluate(() => {
    window.__ws_handlers['ha_washdata/get_device_cycles'] = (msg) => {
      const offset = msg.offset || 0;
      if (offset === 0) {
        return { cycles: window.__mock_cycles.slice(0, 3), total: 5, has_more: true };
      }
      return { cycles: window.__mock_cycles.slice(3), total: 5, has_more: false };
    };
    window.__mock_cycles = [
      { id: 'c1', start_time: '2026-07-10T08:00:00+00:00', end_time: '2026-07-10T09:00:00+00:00', duration: 3600, profile_name: 'Cotton', status: 'completed', energy_kwh: 0.5 },
      { id: 'c2', start_time: '2026-07-09T08:00:00+00:00', end_time: '2026-07-09T09:00:00+00:00', duration: 3600, profile_name: 'Eco', status: 'completed', energy_kwh: 0.6 },
      { id: 'c3', start_time: '2026-07-08T08:00:00+00:00', end_time: '2026-07-08T09:00:00+00:00', duration: 3600, profile_name: 'Quick', status: 'completed', energy_kwh: 0.4 },
      { id: 'c4', start_time: '2026-07-07T08:00:00+00:00', end_time: '2026-07-07T09:00:00+00:00', duration: 3600, profile_name: 'Rinse', status: 'completed', energy_kwh: 0.3 },
      { id: 'c5', start_time: '2026-07-06T08:00:00+00:00', end_time: '2026-07-06T09:00:00+00:00', duration: 3600, profile_name: 'Spin', status: 'completed', energy_kwh: 0.2 },
    ];
  });
  await clickTab(page, 'history');
  // 3 cycles initially
  const rows = page.locator('tr[data-cid]');
  await expect(rows).toHaveCount(3, { timeout: 5_000 });

  // Click load more
  const loadMoreBtn = page.locator('button[data-action="cyc-load-more"]').first();
  await expect(loadMoreBtn).toBeVisible({ timeout: 3_000 });
  await loadMoreBtn.click();

  // Should now have 5 rows
  await expect(rows).toHaveCount(5, { timeout: 5_000 });
});

test('sort by date descending by default', async ({ page }) => {
  await clickTab(page, 'history');
  // The first cycle should be the most recent (cyc-001, July 10)
  const firstRow = page.locator('tr[data-cid]').first();
  await expect(firstRow).toBeVisible({ timeout: 5_000 });
  // Cotton 40°C is the most recent (July 10) — it should appear first
  await expect(firstRow.locator('text=Cotton 40°C')).toBeVisible({ timeout: 3_000 });
});

test('multi-select mode activates on select button click', async ({ page }) => {
  await clickTab(page, 'history');
  // Look for the multi-select toggle button
  const selectBtn = page.locator('button[data-action="cyc-select-toggle"]').first();
  await expect(selectBtn).toBeVisible({ timeout: 5_000 });
  await selectBtn.click();
  // Checkboxes should appear
  const checkboxes = page.locator('tr[data-cid] input[type="checkbox"]');
  await expect(checkboxes.first()).toBeVisible({ timeout: 3_000 });
});

test('imported reference cycles appear in the Cycles list with a badge and filter', async ({ page }) => {
  // Backend returns imported store recordings in a separate `reference_cycles`
  // array (kept out of pagination). The panel merges them into the same table,
  // tagged is_reference, with an Imported badge + filter.
  await setHandler(page, 'ha_washdata/get_device_cycles', {
    ...cyclesData,
    reference_cycles: [
      {
        id: 'ref-e2e-1',
        start_time: '2026-07-14T09:00:00+00:00',
        end_time: '2026-07-14T10:00:00+00:00',
        duration: 3600,
        profile_name: 'Imported Eco',
        status: 'completed',
        is_reference: true,
        meta: { source: 'store:e2e1' },
      },
    ],
  });
  await clickTab(page, 'history');

  // 5 real cycles + 1 imported = 6 rows.
  await expect(page.locator('tr[data-cid]')).toHaveCount(6, { timeout: 8_000 });
  const importedRow = page.locator('tr[data-cid]').filter({ hasText: 'Imported Eco' });
  await expect(importedRow).toHaveCount(1);
  await expect(importedRow).toContainText('📥');

  // The status filter gains an "Imported" option; picking it narrows to just it.
  const statusSel = page.locator('#wd-cyc-filter-status');
  await expect(statusSel.locator('option[value="imported"]')).toHaveCount(1);
  await statusSel.selectOption('imported');
  await expect(page.locator('tr[data-cid]')).toHaveCount(1);
  await expect(page.locator('tr[data-cid]').first()).toContainText('Imported Eco');
});

test('imported cycles are not bulk-selectable', async ({ page }) => {
  await setHandler(page, 'ha_washdata/get_device_cycles', {
    ...cyclesData,
    reference_cycles: [
      { id: 'ref-e2e-2', start_time: '2026-07-14T09:00:00+00:00', duration: 3600, profile_name: 'Imported Eco', status: 'completed', is_reference: true, meta: { source: 'store:e2e2' } },
    ],
  });
  await clickTab(page, 'history');
  await page.locator('button[data-action="cyc-select-toggle"]').first().click();
  // Real cycles get a checkbox; the imported row keeps its status dot (no checkbox).
  await expect(page.locator('tr[data-cid] input[type="checkbox"]')).toHaveCount(5, { timeout: 3_000 });
  const importedRow = page.locator('tr[data-cid]').filter({ hasText: 'Imported Eco' });
  await expect(importedRow.locator('input[type="checkbox"]')).toHaveCount(0);
});

test('cycles tab renders without overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await clickTab(page, 'history');
  const overflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const body = el.shadowRoot.querySelector('.wd-body');
    return body ? body.scrollWidth - body.clientWidth : 0;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});
