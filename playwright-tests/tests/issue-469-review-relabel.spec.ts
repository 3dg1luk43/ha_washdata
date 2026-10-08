/**
 * #469: after Save Review relabelled a cycle, the still-open dialog showed the
 * label it was opened with ("Unlabelled"), so the save looked lost. Worse, the
 * Review form's snapshot wrote the UNSAVED dropdown choice into the same field the
 * save compares against, so a save after any re-render (an overlay toggle) saw "no
 * change" and skipped label_cycle while the toast still said "Review saved".
 *
 * The unsaved choice now lives apart from the label on record, and a successful
 * save updates the open dialog.
 */

import { test, expect, Page } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

async function boot(page: Page) {
  await page.goto('/');
  await bootPanel(page);
  await page.evaluate(() => {
    const w = window as any;
    w.__ws_handlers['ha_washdata/set_ml_review'] = () => ({ success: true });
    w.__ws_handlers['ha_washdata/get_profile_envelope'] = () => ({ envelope: null });
    w.__ws_handlers['ha_washdata/get_cycle_power_data'] = (msg: any) => ({
      cycle_id: msg.cycle_id,
      samples: Array.from({ length: 30 }, (_, i) => [i * 60, i < 2 || i > 27 ? 3 : 900]),
      sample_count: 30, decimated: false, full_duration_s: 1740,
      profile_name: msg.cycle_id === 'cyc-001' ? 'Cotton 40°C' : null,
      status: 'completed', artifacts: [], restart_gaps: [], is_reference: false,
    });
  });
}

async function openReview(page: Page, cid: string) {
  await clickTab(page, 'history');
  await page.locator(`tr[data-cid="${cid}"]`).click();
  await expect(page.locator('#wd-cyc-canvas')).toBeVisible({ timeout: 5_000 });
  await page.locator('button[data-maction="cyc-review"]').click();
  await expect(page.locator('#wd-cyc-rev-label')).toBeVisible();
}

const labelCalls = (page: Page) =>
  page.evaluate(() => (window as any).__get_calls('ha_washdata/label_cycle')) as Promise<any[]>;

async function save(page: Page) {
  await page.locator('button[data-maction="cyc-review-save"]').click();
  await expect(page.locator('button[data-maction="cyc-review-save"]')).toBeEnabled({ timeout: 5_000 });
}

test('the open dialog shows the new label after Save Review', async ({ page }) => {
  await boot(page);
  await openReview(page, 'cyc-003');
  await page.locator('#wd-cyc-rev-label').selectOption('Eco 60°C');
  await save(page);
  expect((await labelCalls(page)).map(c => c.profile_name)).toEqual(['Eco 60°C']);
  await expect(page.locator('#wd-cyc-rev-label')).toHaveValue('Eco 60°C');
  await expect(page.locator('.wd-modal .wd-kv-item').filter({ hasText: 'Profile' }).first()).toContainText('Eco 60°C');
});

test('relabelling back to the original label is saved, not skipped', async ({ page }) => {
  await boot(page);
  await openReview(page, 'cyc-001');
  await page.locator('#wd-cyc-rev-label').selectOption('Eco 60°C');
  await save(page);
  await page.locator('#wd-cyc-rev-label').selectOption('Cotton 40°C');
  await save(page);
  expect((await labelCalls(page)).map(c => c.profile_name)).toEqual(['Eco 60°C', 'Cotton 40°C']);
});

test('an unsaved choice survives a re-render and is still saved', async ({ page }) => {
  await boot(page);
  await openReview(page, 'cyc-003');
  await page.locator('#wd-cyc-rev-label').selectOption('Quick 30°C');
  await page.locator('.wd-cyc-overlay').first().check();  // snapshots the form and re-renders
  await expect(page.locator('#wd-cyc-rev-label')).toHaveValue('Quick 30°C');
  // The label on record is still none until the save.
  await expect(page.locator('.wd-modal .wd-kv-item').filter({ hasText: 'Profile' }).first()).toContainText('Unlabelled');
  await save(page);
  expect((await labelCalls(page)).map(c => c.profile_name)).toEqual(['Quick 30°C']);
});
