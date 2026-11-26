/**
 * #536: a cycle ended too early can have its end moved past the stored trace.
 *
 * Trim asks get_cycle_context for up to 4 h of the power sensor's history after
 * the end, draws it, and lets the end handle move into it, short of the next
 * run. Applying sends the usual trim_cycle; the backend reads the recorder and
 * appends the readings.
 */

import { test, expect, Page } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

const PANEL = 'ha-washdata-panel';
const EID = 'test-entry-001';

async function bootWith(page: Page, ctx: Record<string, unknown> | null) {
  await page.goto('/');
  await bootPanel(page);
  await page.evaluate((extra) => {
    const w = window as any;
    if (!extra) return;
    w.__ws_handlers['ha_washdata/get_cycle_context'] = (msg: any) => ({
      cycle_id: msg.cycle_id, available: true, reason: null, entity_id: 'sensor.washer_power',
      before_s: msg.before_s, after_s: msg.after_s, trace_end_s: 1740, after_end_s: 1740 + msg.after_s,
      next_start_s: null, before: [],
      after: [[1740, 3.0], [1800, 3.3], [2400, 6.3], [2440, 3.2]],
      ...extra,
    });
  }, ctx);
}

async function openTrim(page: Page) {
  await clickTab(page, 'history');
  const row = page.locator('tr[data-cid="cyc-001"]');
  await expect(row).toBeVisible({ timeout: 5_000 });
  await row.click();
  await expect(page.locator('#wd-cyc-canvas')).toBeVisible({ timeout: 5_000 });
  await page.locator('button[data-maction="cyc-trim"]').click();
  await expect(page.locator('#wd-trim-end')).toBeVisible();
}

async function setEnd(page: Page, value: string) {
  const end = page.locator('#wd-trim-end');
  await end.fill(value);
  await end.dispatchEvent('change');
}

test('the end moves into the recorder history after the cycle', async ({ page }) => {
  await bootWith(page, {});
  await openTrim(page);
  const calls = await assertWsCalled(page, 'ha_washdata/get_cycle_context');
  expect(calls[calls.length - 1]).toMatchObject({ entry_id: EID, cycle_id: 'cyc-001', before_s: 0, after_s: 14400 });
  const modal = page.locator('.wd-modal');
  await expect(modal.locator('#wd-trim-ext-note')).toContainText('Drag the end into it');

  // Snaps to the recorded reading nearest the typed value, past the trace's end.
  await setEnd(page, '2430');
  await expect(page.locator('#wd-trim-end')).toHaveValue('2440');
  await expect(modal.locator('#wd-trim-ext-note')).toContainText('readings Home Assistant recorded up to it are added');

  await modal.locator('button[data-maction="cyc-apply-trim"]').click();
  const trims = await assertWsCalled(page, 'ha_washdata/trim_cycle', 1);
  expect(trims[0]).toMatchObject({ entry_id: EID, cycle_id: 'cyc-001', start_s: 0, end_s: 2440 });
  await expect(page.locator('.wd-toast').last()).toContainText('Cycle end moved');
});

test('the next run bounds the end', async ({ page }) => {
  await bootWith(page, { next_start_s: 2000 });
  await openTrim(page);
  await expect(page.locator('#wd-trim-ext-note')).toContainText('Drag the end into it');
  await setEnd(page, '5000');
  // Clamped short of the next run, then onto the last reading before it.
  await expect(page.locator('#wd-trim-end')).toHaveValue('1800');
});

test('without recorder history the end stays inside the trace', async ({ page }) => {
  await bootWith(page, null);
  await openTrim(page);
  await expect(page.locator('#wd-trim-ext-note')).toBeEmpty();
  await setEnd(page, '5000');
  await expect(page.locator('#wd-trim-end')).toHaveValue('1740');
});

test('a refused extension is worded', async ({ page }) => {
  await bootWith(page, {});
  const texts = await page.evaluate(`(() => {
    const el = document.querySelector('${PANEL}');
    return ['extend_overlaps', 'extend_no_history', 'extend_too_far', 'extend_not_live', 'trim_failed']
      .map(c => el._trimErrorText(c));
  })()`) as (string | null)[];
  expect(texts[0]).toBe('The end cannot move past the start of the next cycle.');
  expect(texts[1]).toContain('no readings');
  expect(texts[2]).toContain('4 h');
  expect(texts[3]).toContain('recorded');
  expect(texts[4]).toBeNull();
});

test('cutting most of the start still asks, though the end moves past the trace', async ({ page }) => {
  // 1500 of the 1740 s trace cut from the start: about 14% of the original is
  // kept, whatever the extension adds (PR #539 review: it counted 54%).
  await bootWith(page, {});
  await openTrim(page);
  await setEnd(page, '2440');
  const start = page.locator('#wd-trim-start');
  await start.fill('1500');
  await start.dispatchEvent('change');
  const dialogs: string[] = [];
  page.on('dialog', (d) => { dialogs.push(d.message()); void d.dismiss(); });
  await page.locator('.wd-modal button[data-maction="cyc-apply-trim"]').click();
  await expect.poll(() => dialogs.length).toBe(1);
  expect(dialogs[0]).toMatch(/keeps only \d+% of the cycle/);
  const sent = await page.evaluate(() => (window as any).__get_calls('ha_washdata/trim_cycle').length);
  expect(sent).toBe(0);
});

test('the start lands on a recorded reading past the trace too', async ({ page }) => {
  // Start and end both past the 1740 s trace: the start used to snap back onto
  // the trace's last sample, keeping readings meant to go (PR #539 review).
  await bootWith(page, {});
  await openTrim(page);
  await setEnd(page, '2440');
  const start = page.locator('#wd-trim-start');
  await start.fill('2400');
  await start.dispatchEvent('change');
  await expect(start).toHaveValue('2400');
  await expect(page.locator('#wd-trim-end')).toHaveValue('2440');
});

test('re-entering Trim asks the recorder again', async ({ page }) => {
  // The recorder keeps recording after an early end: a later visit to Trim must
  // see the readings since, not the first answer (PR #539 review).
  await bootWith(page, {});
  await openTrim(page);
  const first = (await assertWsCalled(page, 'ha_washdata/get_cycle_context')).length;
  await page.locator('button[data-maction="cyc-view"]').click();
  await page.locator('button[data-maction="cyc-trim"]').click();
  await expect(page.locator('#wd-trim-end')).toBeVisible();
  await expect.poll(async () => (await assertWsCalled(page, 'ha_washdata/get_cycle_context')).length)
    .toBe(first + 1);
});

test('a refetch that brings the limit in pulls the end back', async ({ page }) => {
  // The end sat at 2440; a run has started at 2000 since: the refetch must not
  // leave the end past it, off the chart and still applied (PR #539 review).
  await bootWith(page, {});
  await openTrim(page);
  await setEnd(page, '2440');
  await expect(page.locator('#wd-trim-end')).toHaveValue('2440');
  await page.locator('button[data-maction="cyc-view"]').click();
  await page.evaluate(() => {
    const w = window as any;
    const prev = w.__ws_handlers['ha_washdata/get_cycle_context'];
    w.__ws_handlers['ha_washdata/get_cycle_context'] = (msg: any) => ({ ...prev(msg), next_start_s: 2000 });
  });
  await page.locator('button[data-maction="cyc-trim"]').click();
  await expect(page.locator('#wd-trim-end')).toHaveValue('1800');
});
