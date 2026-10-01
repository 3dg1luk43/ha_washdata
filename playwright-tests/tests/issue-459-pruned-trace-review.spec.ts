/**
 * Issue #459 - a cycle whose power trace retention pruned must not sit in the
 * review queue on an ML health label alone.
 *
 * The nightly health recompute used to score traceless cycles to ~1% health
 * ("review"), so one old cycle joined the queue per new cycle once a program
 * reached its trace cap. The backend no longer does that, but the panel also
 * refuses to treat an ML label as a review finding when the trace it judged is
 * gone (`has_power_data: false`).
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

const ML_COMPARISON = {
  cycles: [
    // Pruned: an ML label from before the fix, no trace behind it.
    { id: 'cyc-001', ml_quality_label: 'review', ml_quality_score: 0.989, has_power_data: false, ml_review: {} },
    // Traced: a real finding, still flagged.
    { id: 'cyc-002', ml_quality_label: 'review', ml_quality_score: 0.8, has_power_data: true, ml_review: {} },
  ],
};

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_feedbacks': { feedbacks: [] },
    'ha_washdata/get_ml_comparison': ML_COMPARISON,
  });
});

test('a pruned cycle is not in the review queue, a traced one is (#459)', async ({ page }) => {
  await clickTab(page, 'history');
  const statusSel = page.locator('#wd-cyc-filter-status');
  await expect(statusSel).toBeVisible({ timeout: 5_000 });
  await statusSel.selectOption('needs_review');
  await expect(page.locator('tr[data-cid="cyc-002"]')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('tr[data-cid="cyc-001"]')).toHaveCount(0);
});

test('the trace caps are editable in Housekeeping (#459)', async ({ page }) => {
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="timing"]').first().click();
  await expect(page.locator('input[data-opt="max_full_traces_per_profile"]').first()).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('input[data-opt="max_full_traces_unlabeled"]').first()).toBeVisible();
});
