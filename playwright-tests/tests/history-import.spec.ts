/**
 * Import power history (issue #344), Advanced -> Diagnostics.
 *
 * Covers the four steps: stage the CSV, scan it as a background task, review what was
 * found with per-candidate keep/discard, then apply only what was kept. Also covers the
 * two places the review must be honest: a stream with nothing usable in it, and a
 * candidate the detector could not end cleanly.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled } from '../helpers/panel';

const CSV = [
  'entity_id,state,last_changed',
  'sensor.washer_power,1800,2026-07-21T09:14:00+00:00',
  'sensor.washer_power,400,2026-07-21T09:20:00+00:00',
  'sensor.washer_power,0,2026-07-21T10:28:00+00:00',
].join('\n');

async function openDiagnostics(page) {
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
}

async function openWizard(page) {
  await openDiagnostics(page);
  await page.locator('button[data-action="hist-import-open"]').first().click();
  await expect(page.locator('#wd-hist-csv')).toBeVisible({ timeout: 8_000 });
}

async function scanCsv(page) {
  await openWizard(page);
  await page.locator('#wd-hist-csv').fill(CSV);
  await page.locator('button[data-maction="hist-scan"]').first().click();
  await expect(page.locator('table.wd-table tr[data-hist-row]').first()).toBeVisible({ timeout: 8_000 });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {});
});

test('the wizard opens from Diagnostics with both ingest routes', async ({ page }) => {
  await openWizard(page);
  await expect(page.locator('#wd-hist-file')).toBeVisible();
  await expect(page.locator('button[data-maction="hist-recorder"]')).toBeVisible();
  // The recorder read is bounded by a start DATE, defaulting to 10 days back (HA's
  // default purge_keep_days), not by a day count the user has to work out.
  const since = page.locator('#wd-hist-since');
  await expect(since).toHaveAttribute('type', 'date');
  const tenDaysAgo = new Date();
  tenDaysAgo.setDate(tenDaysAgo.getDate() - 10);
  const pad = (n: number) => String(n).padStart(2, '0');
  const expected = `${tenDaysAgo.getFullYear()}-${pad(tenDaysAgo.getMonth() + 1)}-${pad(tenDaysAgo.getDate())}`;
  await expect(since).toHaveValue(expected);
});

test('scanning stages the text in chunks and starts a background task', async ({ page }) => {
  await scanCsv(page);
  await assertWsCalled(page, 'ha_washdata/history_import_begin');
  const chunks = await assertWsCalled(page, 'ha_washdata/history_import_chunk');
  // Small file: one chunk, carrying the token from begin and starting at seq 0.
  expect(chunks).toHaveLength(1);
  expect(chunks[0]).toHaveProperty('seq', 0);
  expect(chunks[0]).toHaveProperty('token', 'tok-1');
  expect(chunks[0].text).toContain('entity_id,state,last_changed');
  const scans = await assertWsCalled(page, 'ha_washdata/start_history_import_scan');
  expect(scans[0]).toHaveProperty('token', 'tok-1');
});

test('scanning without any data refuses to upload', async ({ page }) => {
  await openWizard(page);
  await page.locator('button[data-maction="hist-scan"]').first().click();
  await assertWsNotCalled(page, 'ha_washdata/history_import_begin');
});

test('the review step lists candidates and preselects only the clean ones', async ({ page }) => {
  await scanCsv(page);
  const rows = page.locator('table.wd-table tr[data-hist-row]');
  await expect(rows).toHaveCount(2);
  // A completed cycle defaults to kept; one the detector could not end cleanly does not.
  await expect(page.locator('input[data-hist-pick="0"]')).toBeChecked();
  await expect(page.locator('input[data-hist-pick="1"]')).not.toBeChecked();
  // The reason is shown rather than the row being hidden.
  await expect(rows.nth(1)).toContainText('shorter');
  // Every candidate gets a shape preview.
  await expect(page.locator('canvas[data-hist-spark]')).toHaveCount(2);
  // The import button counts what is actually ticked.
  await expect(page.locator('button[data-maction="hist-apply"]')).toContainText('1');
});

test('the review step accounts for the stretches it skipped', async ({ page }) => {
  await scanCsv(page);
  const modal = page.locator('.wd-modal');
  await expect(modal).toContainText('2358');           // readings read
  await expect(modal).toContainText('readings too far apart');  // the hourly-average region
  await expect(modal).toContainText('minimum power 2 W');       // which settings were used
});

test('ticking a candidate updates the import count', async ({ page }) => {
  await scanCsv(page);
  await page.locator('input[data-hist-pick="1"]').click();
  await expect(page.locator('button[data-maction="hist-apply"]')).toContainText('2');
  await page.locator('button[data-maction="hist-toggle-all"]').first().click();
  await expect(page.locator('input[data-hist-pick="0"]')).not.toBeChecked();
  await expect(page.locator('button[data-maction="hist-apply"]')).toBeDisabled();
});

test('applying sends only the kept candidates and reports the outcome', async ({ page }) => {
  await scanCsv(page);
  await page.locator('button[data-maction="hist-apply"]').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/apply_history_import');
  expect(calls[0].accept).toEqual([0]);
  expect(calls[0]).toHaveProperty('scan_task_id');
  await expect(page.locator('.wd-modal')).toContainText('1 cycles imported', { timeout: 8_000 });
  await expect(page.locator('button[data-maction="hist-goto-cycles"]')).toBeVisible();
});

test('reading from Home Assistant skips the upload and scans directly', async ({ page }) => {
  await openWizard(page);
  await page.locator('#wd-hist-since').fill('2026-01-05');
  await page.locator('button[data-maction="hist-recorder"]').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/history_import_recorder');
  expect(calls[0]).toHaveProperty('start_date', '2026-01-05');
  await assertWsNotCalled(page, 'ha_washdata/history_import_chunk');
  await expect(page.locator('table.wd-table tr[data-hist-row]').first()).toBeVisible({ timeout: 8_000 });
});

test('a history with nothing usable in it explains itself', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/__history_import_scan_result': {
      segments: [], found: 0,
      skipped: [{ reason: 'sparse', span_s: 604800, samples: 114 }],
      parse: { rows_total: 114, breaks: 0 },
      settings: { min_power: 2, off_delay: 300 },
    },
  });
  await openWizard(page);
  await page.locator('#wd-hist-csv').fill(CSV);
  await page.locator('button[data-maction="hist-scan"]').first().click();
  const modal = page.locator('.wd-modal');
  await expect(modal).toContainText('No cycles could be detected', { timeout: 8_000 });
  await expect(modal).toContainText('readings too far apart');
  await expect(page.locator('button[data-maction="hist-back"]')).toBeVisible();
});

test('an expired scan asks for a rescan instead of failing silently', async ({ page }) => {
  await scanCsv(page);
  await page.locator('button[data-maction="hist-apply"]').first().click();
  await expect(page.locator('button[data-maction="hist-goto-cycles"]')).toBeVisible({ timeout: 8_000 });
  // The registry keeps only the last finished tasks per device and an entry reload
  // clears the staging area, so the backend can legitimately answer "scan_expired".
  // Replay that terminal snapshot the way the task subscription would.
  await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel') as any;
    el._histSettled = new Set();
    (window as any).__emit_task({
      id: el._modal.applyTaskId, entry_id: 'entry-1', kind: 'history_import_apply',
      state: 'error', error: 'scan_expired', updated_at: Date.now() / 1000,
      finished_at: Date.now() / 1000, has_result: false,
    });
  });
  await expect(page.locator('.wd-modal')).toContainText('scan again', { timeout: 8_000 });
});
