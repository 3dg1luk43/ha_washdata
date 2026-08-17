/**
 * Issue #355 - the "needs review" counter and the Cycles review list must agree.
 *
 * A cycle can carry BOTH an unresolved pending feedback (user_response: null) AND
 * a completed ML quality review (ml_review.reviewed_at set). The header counter
 * counts it (pending feedback), but the list's needsReview() short-circuited on
 * reviewed_at and hid it - counter says N, list empty. Unresolved pending feedback
 * must take priority so the cycle stays visible (and resolvable) in the list.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

// cyc-005 in the fixture already has ml_review.reviewed_at set; attach an
// unresolved pending feedback to it (the #355 collision).
const FB_ON_REVIEWED = {
  feedbacks: [
    {
      cycle_id: 'cyc-005',
      detected_profile: 'Cotton 40°C',
      confidence: 0.55,
      user_response: null,
      created_at: '2026-07-23T15:32:38+02:00',
    },
  ],
};

// Populate the ML index so isReviewed(cyc-005) is true (reviewed_at set) - this is
// what collides with the pending feedback and used to hide the cycle.
const ML_COMPARISON = {
  cycles: [
    { id: 'cyc-005', ml_review: { reviewed_at: '2026-07-07T09:00:00+00:00' }, ml_quality_label: 'good' },
  ],
};

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_feedbacks': FB_ON_REVIEWED,
    'ha_washdata/get_ml_comparison': ML_COMPARISON,
  });
});

test('a reviewed cycle with unresolved pending feedback still needs review (#355)', async ({ page }) => {
  await clickTab(page, 'history');
  const statusSel = page.locator('#wd-cyc-filter-status');
  await expect(statusSel).toBeVisible({ timeout: 5_000 });
  await statusSel.selectOption('needs_review');
  // Pending feedback wins over reviewed_at: cyc-005 must appear in the list,
  // matching the header count.
  await expect(page.locator('tr[data-cid="cyc-005"]')).toBeVisible({ timeout: 5_000 });
});
