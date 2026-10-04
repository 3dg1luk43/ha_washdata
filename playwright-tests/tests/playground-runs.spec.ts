/**
 * Playground run hygiene (audit wave 6, PLAYGROUND-08/-12/-13/-14/-18, PLATFORM-04,
 * PLAYGROUND-11/-24 in the import review).
 *
 * - A superseded or cancelled Simulate is cancelled on the backend, not just dropped.
 * - A settings edit marks the outcome stale and offers a re-run.
 * - The profile picker says it only picks the overlay.
 * - "Last N" is capped at the backend limit and sent as a count, to Optimize too.
 * - Optimize shows N, keeps the current value on a tie, and flags guarded values.
 * - A busy device refuses a second batch run with a readable message.
 * - The import review says when the read was cut short, and flags kW / naive stamps.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, setHandler } from '../helpers/panel';
import { DEFAULT_HANDLERS } from '../helpers/ws-handlers';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {});
});

/** Make every Simulate task hang in `running`, like a long cycle replaying. */
async function holdDetailTasks(page) {
  await page.evaluate(() => {
    (window as any).__set_handler('ha_washdata/run_playground_cycle_detail', () => new Promise(() => {}));
  });
}

async function startedDetailTaskIds(page): Promise<string[]> {
  return page.evaluate(() => (window as any).__ws_tasks
    .filter((t: any) => t.kind === 'pg_detail').map((t: any) => t.id));
}

test('PLAYGROUND-12: Cancel sends cancel_task for the running Simulate', async ({ page }) => {
  await holdDetailTasks(page);
  await clickTab(page, 'playground');
  await page.locator('button[data-action="pg-run"]').click();
  await expect.poll(() => startedDetailTaskIds(page), { timeout: 8_000 }).toHaveLength(1);
  const [tid] = await startedDetailTaskIds(page);
  await page.locator('button[data-action="pg-cancel-run"]').click();
  const cancels = await assertWsCalled(page, 'ha_washdata/cancel_task');
  expect(cancels.map(c => c.task_id)).toContain(tid);
  await expect(page.locator('button[data-action="pg-cancel-run"]')).toHaveCount(0);
});

test('PLAYGROUND-12: picking another cycle cancels the Simulate it supersedes', async ({ page }) => {
  await holdDetailTasks(page);
  await clickTab(page, 'playground');
  await page.locator('button[data-action="pg-run"]').click();
  await expect.poll(() => startedDetailTaskIds(page), { timeout: 8_000 }).toHaveLength(1);
  const [first] = await startedDetailTaskIds(page);
  // The cycle select is disabled while busy; drilling a cycle in is the other path.
  await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel') as any;
    el._pgSelectCycle('cyc-002');
  });
  await expect.poll(() => startedDetailTaskIds(page), { timeout: 8_000 }).toHaveLength(2);
  await expect.poll(async () => (await page.evaluate(() => (window as any).__get_calls('ha_washdata/cancel_task')))
    .map((c: any) => c.task_id), { timeout: 8_000 }).toContain(first);
});

test('PLAYGROUND-13: an edit marks the outcome stale and Update re-runs it with the edit', async ({ page }) => {
  await clickTab(page, 'playground');
  await page.locator('button[data-action="pg-run"]').click();
  const card = page.locator('.wd-pg-alerts-card');
  await expect(card).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('.wd-pg-stale')).toHaveCount(0);
  await page.locator('.wd-pg-param-inp[data-pgkey="off_delay"]').fill('222');
  const banner = page.locator('.wd-pg-stale');
  await expect(banner).toBeVisible();
  await expect(banner).toContainText('Settings changed');
  await banner.locator('button[data-action="pg-rerun"]').click();
  await expect.poll(async () => (await page.evaluate(() => (window as any).__get_calls('ha_washdata/start_playground_cycle_detail'))).length,
    { timeout: 8_000 }).toBe(2);
  const calls = await assertWsCalled(page, 'ha_washdata/start_playground_cycle_detail', 2);
  expect((calls[1].settings_override as Record<string, unknown>).off_delay).toBe(222);
  // The re-run used the edited settings, so the outcome is current again.
  await expect(page.locator('.wd-pg-stale')).toHaveCount(0, { timeout: 8_000 });
  await expect(card).toBeVisible();
});

test('PLAYGROUND-14: the profile picker says it only picks what to compare with', async ({ page }) => {
  await clickTab(page, 'playground');
  const field = page.locator('.wd-field', { has: page.locator('#wd-pg-prof-sel') });
  await expect(field.locator('label')).toHaveText('Compare with profile', { timeout: 8_000 });
  await expect(field).toHaveAttribute('title', /simulation always matches/);
});

test('PLAYGROUND-18: Last N is capped at 50 and sent as a count', async ({ page }) => {
  await clickTab(page, 'playground');
  const n = page.locator('#wd-pg-simn');
  await expect(n).toHaveAttribute('max', '50', { timeout: 8_000 });
  await n.fill('200');
  await page.locator('button[data-action="pg-run-history"]').click();
  const [call] = await assertWsCalled(page, 'ha_washdata/start_playground_history');
  expect(call.count).toBe(50);
  expect(call).not.toHaveProperty('cycle_ids');
});

test('PLAYGROUND-18: Optimize replays the same Last N', async ({ page }) => {
  await clickTab(page, 'playground');
  await page.locator('#wd-pg-simn').fill('7');
  await page.locator('.wd-pg-subtabs button[data-subtab="sweep"]').click();
  await expect(page.locator('#wd-pg-sw-n')).toHaveValue('7', { timeout: 8_000 });
  await page.locator('#wd-pg-sw-from').fill('60');
  await page.locator('#wd-pg-sw-to').fill('240');
  await page.locator('#wd-pg-sw-steps').fill('3');
  await page.locator('button[data-action="pg-sweep-run2"]').click();
  const [call] = await assertWsCalled(page, 'ha_washdata/start_playground_sweep');
  expect(call.count).toBe(7);
});

async function runSweep(page) {
  await clickTab(page, 'playground');
  await page.locator('.wd-pg-subtabs button[data-subtab="sweep"]').click();
  await page.locator('#wd-pg-sw-obj').selectOption('end_lag');
  await page.locator('#wd-pg-sw-from').fill('60');
  await page.locator('#wd-pg-sw-to').fill('240');
  await page.locator('#wd-pg-sw-steps').fill('3');
  await page.locator('button[data-action="pg-sweep-run2"]').click();
}

test('PLAYGROUND-08: a tie keeps the current value, shows N and offers no Apply', async ({ page }) => {
  await setHandler(page, 'ha_washdata/run_playground_sweep', {
    param: 'off_delay', objective: 'end_lag', current_value: 150,
    current_metric: 600, best_value: 150, best_metric: 600, keep_current: true, cycles: 12,
    lower_is_better: true,
    points: [
      { value: 60, metric: 570, summary: {}, guarded: false },
      { value: 150, metric: 600, summary: {}, guarded: false },
      { value: 240, metric: 300, summary: {}, guarded: true },
    ],
  });
  await runSweep(page);
  const head = page.locator('.wd-pg-sweep-head');
  await expect(head).toContainText('Keep the current value', { timeout: 8_000 });
  await expect(head).toContainText('over 12 cycles');
  await expect(page.locator('button[data-action="pg-sweep-apply2"]')).toHaveCount(0);
  // The fastest value ended cycles early: flagged, never recommended.
  await expect(page.locator('.wd-pg-sweep-row[data-value="240"] .wd-pg-sweep-guarded')).toHaveCount(1);
  await expect(page.locator('.wd-pg-sweep-row[data-value="240"]')).toContainText('5.0 min');
});

test('PLAYGROUND-08: a real gain offers the backend\'s pick, not the first extreme', async ({ page }) => {
  await setHandler(page, 'ha_washdata/run_playground_sweep', {
    param: 'off_delay', objective: 'end_lag', current_value: 240,
    current_metric: 900, best_value: 150, best_metric: 600, keep_current: false, cycles: 12,
    lower_is_better: true,
    points: [
      { value: 60, metric: 300, summary: {}, guarded: true },
      { value: 150, metric: 600, summary: {}, guarded: false },
      { value: 240, metric: 900, summary: {}, guarded: false },
    ],
  });
  await runSweep(page);
  await expect(page.locator('.wd-pg-sweep-head')).toContainText('Best value found', { timeout: 8_000 });
  await expect(page.locator('button[data-action="pg-sweep-apply2"]')).toHaveAttribute('data-val', '150');
});

test('PLATFORM-04: a second batch run on a busy device explains itself', async ({ page }) => {
  await page.evaluate(() => (window as any).__set_error('ha_washdata/start_playground_history',
    { code: 'task_busy', message: 'A Test-on-history or Optimize run is already in progress for this device' }));
  await clickTab(page, 'playground');
  await page.locator('button[data-action="pg-run-history"]').click();
  await expect(page.locator('.wd-toast')).toContainText('already in progress', { timeout: 8_000 });
  await expect(page.locator('button[data-action="pg-run-history"]')).toBeEnabled();
});

// ─── Import review (PLAYGROUND-11 / -24) ─────────────────────────────────────

async function scanWithParse(page, parse: Record<string, unknown>) {
  const base = DEFAULT_HANDLERS['ha_washdata/__history_import_scan_result'] as Record<string, any>;
  await setHandler(page, 'ha_washdata/__history_import_scan_result', { ...base, parse: { ...base.parse, ...parse } });
  await clickTab(page, 'advanced');
  await page.locator('[data-ptab="diagnostics"]').first().click();
  await page.locator('button[data-action="hist-import-open"]').first().click();
  await page.locator('#wd-hist-csv').fill('entity_id,state,last_changed\nsensor.washer_power,1.8,2026-07-21T09:14:00');
  await page.locator('button[data-maction="hist-scan"]').first().click();
  await expect(page.locator('table.wd-table tr[data-hist-row]').first()).toBeVisible({ timeout: 8_000 });
}

test('PLAYGROUND-11: the review says when the read stopped at the row limit', async ({ page }) => {
  await scanWithParse(page, { truncated: true, rows_total: 500000 });
  await expect(page.locator('.wd-modal')).toContainText('500000-reading limit');
});

test('PLAYGROUND-24: kW-looking data and timestamps without an offset are flagged', async ({ page }) => {
  await scanWithParse(page, { peak_w: 2.2, rows_naive_time: 3, warnings: ['looks_like_kw', 'naive_timestamps'] });
  const warns = page.locator('.wd-hist-warn');
  await expect(warns).toHaveCount(2);
  await expect(warns.nth(0)).toContainText('kilowatts');
  await expect(warns.nth(1)).toContainText('3 timestamps have no time zone');
});

test('PLAYGROUND-07: a trimmed end wait is marked on the candidate', async ({ page }) => {
  const base = DEFAULT_HANDLERS['ha_washdata/__history_import_scan_result'] as Record<string, any>;
  const segs = base.segments.map((s: any, i: number) => (i === 0 ? { ...s, banked_tail_s: 3600 } : s));
  await setHandler(page, 'ha_washdata/__history_import_scan_result', { ...base, segments: segs });
  await clickTab(page, 'advanced');
  await page.locator('[data-ptab="diagnostics"]').first().click();
  await page.locator('button[data-action="hist-import-open"]').first().click();
  await page.locator('#wd-hist-csv').fill('entity_id,state,last_changed\nsensor.washer_power,1800,2026-07-21T09:14:00+00:00');
  await page.locator('button[data-maction="hist-scan"]').first().click();
  const cell = page.locator('tr[data-hist-row="0"] td').nth(2);
  await expect(cell).toHaveAttribute('title', /60 min of waiting/, { timeout: 8_000 });
});
