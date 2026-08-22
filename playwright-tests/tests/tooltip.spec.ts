/**
 * Issue #385 - the "i" help popovers (`_tip()`) must stay fully inside whatever
 * clips them.
 *
 * `.wd-tip-pop` is a fixed-width bubble centred on its 15px anchor. When the
 * anchor sits closer than half the bubble width to a container edge, the
 * overflowing half used to be clipped away (`.wd-modal` declares only
 * `overflow-y: auto`, but CSS promotes the unset axis to `auto` too, so it clips
 * horizontally as well) and the explanation was cut off mid-word. Reported
 * against the cycle modal's Review tab, where every label sits hard against the
 * modal's left edge.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

const CYCLE_POWER = {
  power_data: [
    { t: 0, p: 0 },
    { t: 30, p: 820 },
    { t: 60, p: 750 },
  ],
  artifacts: [],
  envelope_conformance: null,
};

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_cycle_power_data': CYCLE_POWER });
});

/** Open the cycle modal in Review mode. */
async function openReview(page: import('@playwright/test').Page) {
  await clickTab(page, 'history');
  const row = page.locator('tr[data-cid="cyc-001"]');
  await expect(row).toBeVisible({ timeout: 5_000 });
  await row.click();
  const modal = page.locator('.wd-modal');
  await expect(modal).toBeVisible({ timeout: 5_000 });
  await modal.locator('button[data-maction="cyc-review"]').click();
  await expect(modal.locator('#wd-cyc-rev-label')).toBeVisible({ timeout: 5_000 });
  return modal;
}

test('review-tab help popovers stay inside the modal (#385)', async ({ page }) => {
  const modal = await openReview(page);
  const tips = modal.locator('.wd-tip');
  const n = await tips.count();
  expect(n).toBeGreaterThan(0);

  const box = await modal.boundingBox();
  expect(box).not.toBeNull();
  const modalLeft = box!.x;
  const modalRight = box!.x + box!.width;

  for (let i = 0; i < n; i++) {
    const tip = tips.nth(i);
    await tip.scrollIntoViewIfNeeded();
    await tip.hover();
    const pop = tip.locator('.wd-tip-pop');
    await expect(pop).toBeVisible();
    const pb = await pop.boundingBox();
    expect(pb, `tip #${i} popover has no box`).not.toBeNull();
    // 1px slack for sub-pixel rounding.
    expect(pb!.x, `tip #${i} popover overflows the modal's left edge`).toBeGreaterThanOrEqual(modalLeft - 1);
    expect(pb!.x + pb!.width, `tip #${i} popover overflows the modal's right edge`).toBeLessThanOrEqual(modalRight + 1);
  }
});

test('the edge nudge is idempotent across repeat hovers (#385)', async ({ page }) => {
  const modal = await openReview(page);
  // The Profile label is the leftmost tip, so it is the one that gets nudged.
  const tip = modal.locator('.wd-tip').first();
  const pop = tip.locator('.wd-tip-pop');
  const other = modal.locator('button[data-maction="cyc-review-save"]');

  await tip.hover();
  await expect(pop).toBeVisible();
  const first = await pop.boundingBox();
  expect(first).not.toBeNull();

  // Hover away and back twice: the shift is recomputed from the unshifted
  // position each time, so it must land in exactly the same place.
  for (let i = 0; i < 2; i++) {
    await other.hover();
    await tip.hover();
    await expect(pop).toBeVisible();
    const again = await pop.boundingBox();
    expect(again!.x, `hover #${i + 2} moved the popover`).toBeCloseTo(first!.x, 0);
  }
});
