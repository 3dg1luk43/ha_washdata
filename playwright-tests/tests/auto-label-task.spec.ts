/**
 * Audit 2026-10-02 PLATFORM-05: Auto-label runs as a registry task.
 *
 * It re-matches every unlabelled cycle (56 s for 201) and used to hold the WS
 * request open with no progress or cancel. The command now returns a task id and
 * the panel waits on the task before reporting.
 */
import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

test('auto-label starts a task and reports when it finishes', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/auto_label_cycles': { labeled: 3, relabeled: 0, skipped: 1, total: 4 } });
  await clickTab(page, 'history');
  await page.locator('[data-action="cyc-auto-open"]').click();
  await page.locator('[data-maction="auto-run"]').click();
  await expect(page.locator('.wd-toast')).toContainText('Auto-label complete', { timeout: 8_000 });
  await assertWsCalled(page, 'ha_washdata/auto_label_cycles');
});
