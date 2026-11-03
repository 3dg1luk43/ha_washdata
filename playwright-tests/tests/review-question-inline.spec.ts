/**
 * Audit item 1 (review budget follow-up): getting the few review questions that
 * matter answered.
 *
 * 1. The waiting question is asked on the Overview itself: the cycle's trace with
 *    both candidates' curves, when it started, and its duration and energy next
 *    to each candidate's usual values.
 * 2. One-tap answers named after the request's two candidates, then "Something
 *    else..." (the existing Correct dialog) and "Not sure" (the existing ignore),
 *    in the Overview question and the cycle dialog's banner alike. No confidence
 *    figure anywhere (it is not a probability, register item 305).
 * 4. While a live match is torn between two programmes, "X or Y?" chips on the
 *    Status card pin the program, the same set_program the dropdown sends.
 *
 * The backend halves (learning.review_choices / async_answer_review,
 * match_rules.live_torn_choices, the get_feedbacks fields) are unit-tested in
 * tests/test_review_answers.py.
 */

import { test, expect, Page } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled, setHandler } from '../helpers/panel';
import deviceIdle from '../fixtures/mock-data/device-idle.json';
import deviceRunning from '../fixtures/mock-data/device-running.json';
import panelConfig from '../fixtures/mock-data/panel-config.json';

const PANEL = 'ha-washdata-panel';

// cyc-003 is the fixture's unlabelled cycle. The request offers Eco 60°C (the
// detected programme) and Quick 30°C (the stored ranking's runner-up).
const REQUEST = {
  cycle_id: 'cyc-003',
  detected_profile: 'Eco 60°C',
  confidence: 0.611,
  user_response: null,
  created_at: '2026-07-08T20:16:00+00:00',
  ranking: [
    { name: 'Eco 60°C', score: 0.611, metrics: {}, profile_duration: 9600 },
    { name: 'Quick 30°C', score: 0.58, metrics: {}, profile_duration: 2400 },
  ],
  choices: [
    { name: 'Eco 60°C', avg_duration_s: 9600, avg_energy_kwh: 1.1 },
    { name: 'Quick 30°C', avg_duration_s: 2400, avg_energy_kwh: 0.22 },
  ],
  cycle: { start_time: '2026-07-08T19:30:00+00:00', duration_s: 2700, energy_kwh: 0.31, profile_name: null },
};

// The boot handlers are plain data (the question loads during the first paint),
// so both candidates get the same flat curve; which profile each series is comes
// from its name and the envelope requests.
const ENV_PTS = Array.from({ length: 10 }, (_, i) => [i * 270, 600]);
const ENVELOPE = { envelope: { avg: ENV_PTS, min: ENV_PTS, max: ENV_PTS, target_duration: 2430 } };

function devices(base: any, patch: Record<string, unknown>) {
  const d = JSON.parse(JSON.stringify(base));
  Object.assign(d.devices[0], patch);
  return d;
}

async function bootWithQuestion(page: Page, feedbacks: unknown[] = [REQUEST], device: any = deviceIdle) {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_devices': devices(device, { feedback_count: feedbacks.length }),
    'ha_washdata/get_feedbacks': { feedbacks },
    'ha_washdata/get_profile_envelope': ENVELOPE,
  });
}

/** The review chart's drawn series (name + first y value), Power excluded. */
async function drawnReviewSeries(page: Page): Promise<{ name: string; w: number }[]> {
  return page.evaluate(`(() => {
    const sr = document.querySelector('${PANEL}').shadowRoot;
    const c = sr.getElementById('wd-review-canvas');
    if (!c || !c._wd) return [];
    return c._wd.series.filter(s => s.stroke !== 'primary')
      .map(s => ({ name: s.name, w: s.points && s.points[0] ? s.points[0][1] : null }));
  })()`) as Promise<{ name: string; w: number }[]>;
}

test('the waiting question is asked on the Overview with both candidates', async ({ page }) => {
  await bootWithQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect(q).toBeVisible({ timeout: 8_000 });
  await expect(q).toHaveAttribute('data-review-cid', 'cyc-003');
  await expect(q).toContainText('Which program was this?');
  await expect(q).toContainText('Started');
  // Duration and energy against each candidate's usual values.
  const rows = q.locator('.wd-review-q-cmp tbody tr');
  await expect(rows).toHaveCount(3);
  await expect(rows.nth(0)).toContainText('This cycle');
  await expect(rows.nth(0)).toContainText('0.31');
  await expect(rows.nth(1)).toContainText('Usual for Eco 60°C');
  await expect(rows.nth(1)).toContainText('1.1');
  await expect(rows.nth(2)).toContainText('Usual for Quick 30°C');
  await expect(rows.nth(2)).toContainText('0.22');
  // The #462 overlay: both candidates' curves behind the cycle's trace.
  await expect(page.locator('#wd-review-canvas')).toBeVisible();
  await expect.poll(() => drawnReviewSeries(page)).toEqual([
    { name: 'Eco 60°C', w: 600 },
    { name: 'Quick 30°C', w: 600 },
  ]);
  const envCalls = await assertWsCalled(page, 'ha_washdata/get_profile_envelope', 2);
  expect(envCalls.map(c => c.profile_name).sort()).toEqual(['Eco 60°C', 'Quick 30°C']);
  // One-tap answers, named after the two candidates, and no confidence figure.
  const answers = q.locator('.wd-review-answers button');
  await expect(answers).toHaveText(['Eco 60°C', 'Quick 30°C', 'Something else…', 'Not sure']);
  await expect(q).not.toContainText('%');
  // The count card is replaced by the question, which keeps a way to the queue.
  await expect(page.locator('.wd-attn-card[data-action="goto-feedbacks"]')).toHaveCount(0);
  await expect(q.locator('[data-action="goto-feedbacks"]')).toBeVisible();
});

test('picking the runner-up corrects to it and the next question takes its place', async ({ page }) => {
  const second = { ...REQUEST, cycle_id: 'cyc-004', created_at: '2026-07-07T12:00:00+00:00',
    cycle: { ...REQUEST.cycle, start_time: '2026-07-07T11:00:00+00:00' } };
  await bootWithQuestion(page, [REQUEST, second]);
  const q = page.locator('.wd-review-q');
  await expect(q).toHaveAttribute('data-review-cid', 'cyc-003', { timeout: 8_000 });
  await expect(q.locator('[data-action="goto-feedbacks"]')).toHaveText('1 more to review');

  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [second] });
  await q.locator('.wd-review-answers button', { hasText: 'Quick 30°C' }).click();

  const calls = await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  expect(calls[0]).toMatchObject({ cycle_id: 'cyc-003', action: 'correct', corrected_profile: 'Quick 30°C' });
  await expect(page.locator('.wd-review-q')).toHaveAttribute('data-review-cid', 'cyc-004', { timeout: 5_000 });
  await expect(page.locator('.wd-review-q [data-action="goto-feedbacks"]')).toHaveText('Open the Cycles review queue');
});

test('picking the detected programme confirms it', async ({ page }) => {
  await bootWithQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect(q).toBeVisible({ timeout: 8_000 });
  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [] });
  await q.locator('.wd-review-answers button', { hasText: 'Eco 60°C' }).click();
  const calls = await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  expect(calls[0]).toMatchObject({ cycle_id: 'cyc-003', action: 'confirm' });
  expect('corrected_profile' in calls[0]).toBe(false);
  await expect(page.locator('.wd-review-q')).toHaveCount(0, { timeout: 5_000 });
});

test('"Not sure" dismisses and "Something else…" opens the Correct dialog', async ({ page }) => {
  await bootWithQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect(q).toBeVisible({ timeout: 8_000 });
  await q.locator('.wd-review-answers button', { hasText: 'Something else…' }).click();
  await expect(page.locator('.wd-modal #wd-fb-profile')).toBeVisible({ timeout: 5_000 });
  await page.keyboard.press('Escape');
  await expect(page.locator('.wd-modal')).toHaveCount(0, { timeout: 5_000 });

  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await page.locator('.wd-review-q .wd-review-answers button', { hasText: 'Not sure' }).click();
  const calls = await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  expect(calls[0]).toMatchObject({ cycle_id: 'cyc-003', action: 'ignore' });
});

test('the count card stands in when no request carries its cycle', async ({ page }) => {
  // An older backend (no `cycle` field) or only orphaned requests.
  const { cycle: _c, choices: _ch, ...bare } = REQUEST;
  await bootWithQuestion(page, [bare]);
  await expect(page.locator('.wd-attn-card[data-action="goto-feedbacks"]')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('.wd-review-q')).toHaveCount(0);
});

test('while a cycle runs the live card comes first, the question below it', async ({ page }) => {
  await bootWithQuestion(page, [REQUEST], deviceRunning as any);
  await expect(page.locator('.wd-review-q')).toBeVisible({ timeout: 8_000 });
  const order = await page.evaluate(`(() => {
    const sr = document.querySelector('${PANEL}').shadowRoot;
    const sel = sr.getElementById('wd-status-prog');
    const q = sr.querySelector('.wd-review-q');
    return sel.compareDocumentPosition(q) & Node.DOCUMENT_POSITION_FOLLOWING ? 'after' : 'before';
  })()`);
  expect(order).toBe('after');
});

test('the question fits a phone screen without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 780 });
  await bootWithQuestion(page);
  await expect(page.locator('.wd-review-q')).toBeVisible({ timeout: 8_000 });
  const overflow = await page.evaluate(`(() => {
    const sr = document.querySelector('${PANEL}').shadowRoot;
    const body = sr.querySelector('.wd-body');
    const q = sr.querySelector('.wd-review-q');
    return { body: body.scrollWidth - body.clientWidth, q: q.scrollWidth - q.clientWidth };
  })()`) as { body: number; q: number };
  expect(overflow.body).toBeLessThanOrEqual(1);
  expect(overflow.q).toBeLessThanOrEqual(1);
});

test('the cycle dialog banner offers the same answers without a confidence figure', async ({ page }) => {
  await bootWithQuestion(page);
  await clickTab(page, 'history');
  const row = page.locator('tr[data-cid="cyc-003"]');
  await expect(row).toBeVisible({ timeout: 5_000 });
  await row.click();
  const modal = page.locator('.wd-modal');
  await expect(modal.getByText('Pending detection feedback')).toBeVisible({ timeout: 5_000 });
  await expect(modal.locator('.wd-review-answers button')).toHaveText(['Eco 60°C', 'Quick 30°C', 'Something else…', 'Not sure']);
  await expect(modal.getByText('Which program was it?')).toBeVisible();
  await expect(modal).not.toContainText('confidence');
  await expect(modal).not.toContainText('61%');

  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [] });
  await modal.locator('.wd-review-answers button', { hasText: 'Quick 30°C' }).click();
  const calls = await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  expect(calls[0]).toMatchObject({ cycle_id: 'cyc-003', action: 'correct', corrected_profile: 'Quick 30°C' });
  // Answered: the dialog's overlay is the new label, no longer "suspected".
  const legend = page.locator('.wd-modal [data-cyc-legend]');
  await expect(legend.locator('[data-leg="suspected"], [data-leg="runner_up"]')).toHaveCount(0, { timeout: 5_000 });
  await expect(legend.locator('[data-leg="expected"]')).toHaveText('Expected (Quick 30°C)');
});

test('a viewer sees no question and no answers', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_panel_config': { ...panelConfig, is_admin: false },
    'ha_washdata/get_devices': devices(deviceIdle, { feedback_count: 1, perm: 'read' }),
    'ha_washdata/get_feedbacks': { feedbacks: [REQUEST] },
  });
  await expect(page.locator('#wd-status-prog')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.wd-review-q')).toHaveCount(0);
  await assertWsNotCalled(page, 'ha_washdata/get_feedbacks');
});

// ─── 4. "X or Y?" while the live match is torn ──────────────────────────────

function torn(patch: Record<string, unknown>) {
  return devices(deviceRunning, { current_program: 'detecting...', feedback_count: 0, ...patch });
}

test('a torn live match offers both programmes as chips that pin the program', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_devices': torn({
      current_program: null,
      match_uncertainty: { top: 'Cotton 40°C', runner_up: 'Eco 60°C', margin: 0.03, sure_pct: 45 },
      match_choices: ['Cotton 40°C', 'Eco 60°C'],
    }),
    'ha_washdata/set_program': { success: true },
  });
  const chips = page.locator('.wd-torn .wd-torn-chip');
  await expect(chips).toHaveText(['Cotton 40°C', 'Eco 60°C'], { timeout: 8_000 });
  await expect(page.locator('.wd-torn')).toContainText('Cotton 40°C or Eco 60°C?');
  // The chips stand in for the "Uncertain" line, which names the same two.
  await expect(page.locator('.wd-prog-unc')).toHaveCount(0);
  await chips.nth(1).click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_program', 1);
  expect(calls[0]).toMatchObject({ entry_id: 'test-entry-001', program: 'Eco 60°C' });
});

test('no chips once the match is clear, on a hand-picked program, or when idle', async ({ page }) => {
  await page.goto('/');
  // Clear: the backend sends no choices; the uncertainty line (if any) is unchanged.
  await bootPanel(page, { 'ha_washdata/get_devices': torn({
    current_program: null,
    match_uncertainty: { top: 'Cotton 40°C', runner_up: 'Eco 60°C', margin: 0.2, sure_pct: 90 },
    match_choices: null,
  }) });
  await expect(page.locator('.wd-prog-unc')).toContainText('Uncertain: Cotton 40°C or Eco 60°C', { timeout: 8_000 });
  await expect(page.locator('.wd-torn')).toHaveCount(0);

  // Hand-picked: even a stale choices list draws nothing.
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_devices': torn({
    current_program: 'Cotton 40°C', manual_program: true, match_choices: ['Cotton 40°C', 'Eco 60°C'],
  }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  await expect(page.locator('.wd-torn')).toHaveCount(0);

  // Idle.
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_devices': devices(deviceIdle, { match_choices: ['Cotton 40°C', 'Eco 60°C'] }) });
  await expect(page.locator('#wd-status-prog')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('.wd-torn')).toHaveCount(0);
});
