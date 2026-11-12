/**
 * #518 part 3: a cycle found in imported power history could only be deleted.
 *
 * Its trace can hold an end wait the user wants gone, so the cycle dialog now
 * offers Inspect and Trim for it (the backend says `trimmable`); Split and Review
 * stay real-cycle only, and the note says so. A community-store recording is
 * still read-only.
 */

import { test, expect, Page } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

const EID = 'test-entry-001';

async function bootWith(page: Page, origin: 'backfill' | 'reference') {
  await page.goto('/');
  await bootPanel(page);
  await page.evaluate((cycleOrigin) => {
    (window as any).__ws_handlers['ha_washdata/get_cycle_power_data'] = (msg: any) => ({
      cycle_id: msg.cycle_id,
      samples: Array.from({ length: 30 }, (_, i) => [i * 60, i < 2 || i > 27 ? 3 : 900]),
      sample_count: 30, decimated: false, full_duration_s: 1740,
      profile_name: null, status: 'completed', artifacts: [], restart_gaps: [],
      is_reference: true, labelable: true, editable: false,
      trimmable: cycleOrigin === 'backfill', cycle_origin: cycleOrigin,
    });
  }, origin);
}

async function openCycle(page: Page, cid = 'cyc-001') {
  await clickTab(page, 'history');
  const row = page.locator(`tr[data-cid="${cid}"]`);
  await expect(row).toBeVisible({ timeout: 5_000 });
  await row.click();
  await expect(page.locator('#wd-cyc-canvas')).toBeVisible({ timeout: 5_000 });
}

test('an imported history cycle offers Trim, not Split or Review, and trims', async ({ page }) => {
  await bootWith(page, 'backfill');
  await openCycle(page);
  const modal = page.locator('.wd-modal');
  await expect(modal.locator('button[data-maction="cyc-view"]')).toBeVisible();
  await expect(modal.locator('button[data-maction="cyc-trim"]')).toBeVisible();
  await expect(modal.locator('button[data-maction="cyc-split"]')).toHaveCount(0);
  await expect(modal.locator('button[data-maction="cyc-review"]')).toHaveCount(0);
  const note = modal.locator('[data-cyc-ref-note]');
  await expect(note).toContainText('cannot be split');
  await expect(note).not.toContainText('trimmed');

  await modal.locator('button[data-maction="cyc-trim"]').click();
  await expect(modal.locator('#wd-trim-end')).toBeVisible();
  await modal.locator('#wd-trim-end').fill('1500');
  await modal.locator('#wd-trim-end').dispatchEvent('change');
  await modal.locator('button[data-maction="cyc-apply-trim"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/trim_cycle', 1);
  expect(calls[0]).toMatchObject({ entry_id: EID, cycle_id: 'cyc-001', start_s: 0 });
});

test('a community-store cycle stays read-only', async ({ page }) => {
  await bootWith(page, 'reference');
  await openCycle(page);
  const modal = page.locator('.wd-modal');
  await expect(modal.locator('button[data-maction="cyc-trim"]')).toHaveCount(0);
  await expect(modal.locator('button[data-maction="cyc-view"]')).toHaveCount(0);
  await expect(modal.locator('[data-cyc-ref-note]')).toContainText('community store');
});
