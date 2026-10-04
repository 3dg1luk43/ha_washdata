/**
 * Audit 2026-10-02 PLATFORM-05: Auto-label runs as a registry task.
 *
 * It re-matches every unlabelled cycle (56 s for 201) and used to hold the WS
 * request open with no progress or cancel. The command now returns a task id and
 * the panel waits on the task before reporting.
 *
 * Audit UI-10: the modal opened on a fixed 0.75, below the device's own
 * Auto-Label Confidence (default 0.9), and the toast said "complete" with no
 * counts. It now opens on the device's setting and reports what it did.
 */
import { test, expect, type Page } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import options from '../fixtures/mock-data/options.json';
import en from '../../custom_components/ha_washdata/translations/panel/en.json';

const RESULT = { labeled: 3, relabeled: 0, skipped: 1, total: 4 };

async function openModal(page: Page): Promise<void> {
  await clickTab(page, 'history');
  await page.locator('[data-action="cyc-auto-open"]').click();
  await expect(page.locator('#wd-al-thr')).toBeVisible({ timeout: 8_000 });
}

test('auto-label starts a task and reports when it finishes', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/auto_label_cycles': RESULT });
  await openModal(page);
  await page.locator('[data-maction="auto-run"]').click();
  await expect(page.locator('.wd-toast')).toContainText('Auto-label complete', { timeout: 8_000 });
  await assertWsCalled(page, 'ha_washdata/auto_label_cycles');
});

test('UI-10: the toast says how many cycles were labelled and left alone', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/auto_label_cycles': RESULT }, {}, { translations: { en } });
  await openModal(page);
  await page.locator('[data-maction="auto-run"]').click();
  await expect(page.locator('.wd-toast')).toContainText(
    'Auto-label complete: 3 cycles labelled, 1 left unlabelled.', { timeout: 8_000 },
  );
});

test('UI-10: one labelled cycle reads in the singular', async ({ page }) => {
  await page.goto('/');
  await bootPanel(
    page, { 'ha_washdata/auto_label_cycles': { labeled: 1, relabeled: 0, skipped: 0, total: 1 } },
    {}, { translations: { en } },
  );
  await openModal(page);
  await page.locator('[data-maction="auto-run"]').click();
  await expect(page.locator('.wd-toast')).toContainText(
    'Auto-label complete: 1 cycle labelled, 0 left unlabelled.', { timeout: 8_000 },
  );
});

test('UI-10: the modal opens on the device Auto-Label Confidence and sends it', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_options': { options: { ...options, auto_label_confidence: 0.85 }, defaults: {} },
    'ha_washdata/auto_label_cycles': RESULT,
  });
  await openModal(page);
  await expect(page.locator('#wd-al-thr')).toHaveValue('0.85');
  await page.locator('[data-maction="auto-run"]').click();
  await expect(page.locator('.wd-toast')).toBeVisible({ timeout: 8_000 });
  const calls = await assertWsCalled(page, 'ha_washdata/auto_label_cycles');
  expect(calls[calls.length - 1].confidence_threshold).toBe(0.85);
});

test('UI-10: an unset setting opens on the 0.9 default, not 0.75', async ({ page }) => {
  await page.goto('/');
  const { auto_label_confidence: _drop, ...withoutAuto } = options as Record<string, unknown>;
  await bootPanel(page, { 'ha_washdata/get_options': { options: withoutAuto, defaults: {} } });
  await openModal(page);
  await expect(page.locator('#wd-al-thr')).toHaveValue('0.9');
});

test('UI-10: a setting outside the bulk range is clamped to it', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_options': { options: { ...options, auto_label_confidence: 0.99 }, defaults: {} },
  });
  await openModal(page);
  await expect(page.locator('#wd-al-thr')).toHaveValue('0.95');
});

test('UI-10: an emptied field lets the backend apply the device setting', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/auto_label_cycles': RESULT });
  await openModal(page);
  await page.locator('#wd-al-thr').fill('');
  await page.locator('[data-maction="auto-run"]').click();
  await expect(page.locator('.wd-toast')).toBeVisible({ timeout: 8_000 });
  const calls = await assertWsCalled(page, 'ha_washdata/auto_label_cycles');
  expect('confidence_threshold' in calls[calls.length - 1]).toBe(false);
});
