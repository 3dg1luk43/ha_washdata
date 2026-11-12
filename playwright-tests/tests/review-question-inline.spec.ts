/**
 * Audit item 1 (review budget follow-up): getting the few review questions that
 * matter answered.
 *
 * 1. The waiting questions are asked on the Overview itself, collapsed by default
 *    to one line with the count (remembered once opened); opened, one tile per
 *    request, side by side and stepped through sideways: the cycle's trace with
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

/** Open the collapsed question card (it starts collapsed). */
async function openQuestion(page: Page) {
  const toggle = page.locator('.wd-review-q-toggle');
  await expect(toggle).toHaveAttribute('aria-expanded', 'false', { timeout: 8_000 });
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
}

/** A tile's drawn series (name + first y value), Power excluded. */
async function drawnReviewSeries(page: Page, i = 0): Promise<{ name: string; w: number }[]> {
  return page.evaluate(`(() => {
    const sr = document.querySelector('${PANEL}').shadowRoot;
    const c = sr.getElementById('wd-review-canvas-${i}');
    if (!c || !c._wd) return [];
    return c._wd.series.filter(s => s.stroke !== 'primary')
      .map(s => ({ name: s.name, w: s.points && s.points[0] ? s.points[0][1] : null }));
  })()`) as Promise<{ name: string; w: number }[]>;
}

/** A second request, a day older, on another stored cycle. */
const SECOND = { ...REQUEST, cycle_id: 'cyc-004', created_at: '2026-07-07T12:00:00+00:00',
  cycle: { ...REQUEST.cycle, start_time: '2026-07-07T11:00:00+00:00' } };

test('the question starts collapsed to one line and fetches nothing until opened', async ({ page }) => {
  await bootWithQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect(q).toBeVisible({ timeout: 8_000 });
  await expect(q).toContainText('Which program was this?');
  await expect(q.locator('.wd-review-q-count')).toHaveText('To review: 1');
  await expect(q.locator('.wd-review-tile')).toHaveCount(0);
  await expect(q.locator('.wd-review-answers')).toHaveCount(0);
  // The card stands in for the count card.
  await expect(page.locator('.wd-attn-card[data-action="goto-feedbacks"]')).toHaveCount(0);
  await assertWsNotCalled(page, 'ha_washdata/get_cycle_power_data');
  await assertWsNotCalled(page, 'ha_washdata/get_profile_envelope');
});

test('the waiting question is asked on the Overview with both candidates', async ({ page }) => {
  await bootWithQuestion(page);
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
  const tile = q.locator('.wd-review-tile');
  await expect(tile).toHaveCount(1);
  await expect(tile).toHaveAttribute('data-review-cid', 'cyc-003');
  await expect(tile).toContainText('Started');
  // A single request has no stepping buttons.
  await expect(q.locator('.wd-review-q-step')).toHaveCount(0);
  // Duration and energy against each candidate's usual values.
  const rows = tile.locator('.wd-review-q-cmp tbody tr');
  await expect(rows).toHaveCount(3);
  await expect(rows.nth(0)).toContainText('This cycle');
  await expect(rows.nth(0)).toContainText('0.31');
  await expect(rows.nth(1)).toContainText('Usual for Eco 60°C');
  await expect(rows.nth(1)).toContainText('1.1');
  await expect(rows.nth(2)).toContainText('Usual for Quick 30°C');
  await expect(rows.nth(2)).toContainText('0.22');
  // The #462 overlay: both candidates' curves behind the cycle's trace.
  await expect(page.locator('#wd-review-canvas-0')).toBeVisible();
  await expect.poll(() => drawnReviewSeries(page)).toEqual([
    { name: 'Eco 60°C', w: 600 },
    { name: 'Quick 30°C', w: 600 },
  ]);
  const envCalls = await assertWsCalled(page, 'ha_washdata/get_profile_envelope', 2);
  expect(envCalls.map(c => c.profile_name).sort()).toEqual(['Eco 60°C', 'Quick 30°C']);
  // One-tap answers, named after the two candidates, and no confidence figure.
  const answers = tile.locator('.wd-review-answers button');
  await expect(answers).toHaveText(['Eco 60°C', 'Quick 30°C', 'Something else…', 'Not sure']);
  await expect(q).not.toContainText('%');
  await expect(q.locator('[data-action="goto-feedbacks"]')).toHaveText('Open the Cycles review queue');
});

test('opening is remembered, and closing hides the tiles again', async ({ page }) => {
  await bootWithQuestion(page);
  await openQuestion(page);
  await expect(page.locator('.wd-review-tile')).toHaveCount(1);
  await bootWithQuestion(page);
  await expect(page.locator('.wd-review-q-toggle')).toHaveAttribute('aria-expanded', 'true', { timeout: 8_000 });
  await expect(page.locator('.wd-review-tile')).toHaveCount(1);
  await page.locator('.wd-review-q-toggle').click();
  await expect(page.locator('.wd-review-q-toggle')).toHaveAttribute('aria-expanded', 'false');
  await expect(page.locator('.wd-review-tile')).toHaveCount(0);
  await bootWithQuestion(page);
  await expect(page.locator('.wd-review-q-toggle')).toHaveAttribute('aria-expanded', 'false', { timeout: 8_000 });
});

test('several questions are tiles side by side, stepped through sideways', async ({ page }) => {
  await page.setViewportSize({ width: 700, height: 900 });
  const third = { ...REQUEST, cycle_id: 'cyc-005', created_at: '2026-07-06T12:00:00+00:00',
    cycle: { ...REQUEST.cycle, start_time: '2026-07-06T11:00:00+00:00' } };
  await bootWithQuestion(page, [REQUEST, SECOND, third]);
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect(q.locator('.wd-review-q-count')).toHaveText('To review: 3');
  const tiles = q.locator('.wd-review-tile');
  await expect(tiles).toHaveCount(3);
  expect(await tiles.evaluateAll(els => els.map(e => e.getAttribute('data-review-cid'))))
    .toEqual(['cyc-003', 'cyc-004', 'cyc-005']);
  // Each tile draws its own chart; the envelopes are fetched once per programme.
  await expect.poll(() => drawnReviewSeries(page, 2)).toHaveLength(2);
  const envCalls = await assertWsCalled(page, 'ha_washdata/get_profile_envelope', 2);
  expect(envCalls).toHaveLength(2);
  await assertWsCalled(page, 'ha_washdata/get_cycle_power_data', 3);

  const strip = q.locator('.wd-review-tiles');
  const scrollLeft = () => strip.evaluate(el => el.scrollLeft);
  expect(await strip.evaluate(el => el.scrollWidth > el.clientWidth)).toBe(true);
  const prev = q.locator('.wd-review-q-step[data-dir="-1"]');
  const next = q.locator('.wd-review-q-step[data-dir="1"]');
  await expect(prev).toBeDisabled();
  await expect(next).toBeEnabled();
  await expect(next).toHaveAttribute('aria-label', 'Next cycle');
  await next.click();
  // Settled on the second tile (the step scrolls smoothly).
  const settled = async () => {
    const a = await scrollLeft();
    await page.waitForTimeout(120);
    return a > 100 && (await scrollLeft()) === a;
  };
  await expect.poll(settled).toBe(true);
  await expect(prev).toBeEnabled();
  // A re-render (the poll) keeps the strip where it was.
  const before = await scrollLeft();
  await page.evaluate(`document.querySelector('${PANEL}')._render()`);
  expect(Math.abs((await scrollLeft()) - before)).toBeLessThanOrEqual(1);
  await prev.click();
  await expect.poll(scrollLeft).toBeLessThanOrEqual(1);
});

test('picking the runner-up corrects to it and its tile goes', async ({ page }) => {
  await bootWithQuestion(page, [REQUEST, SECOND]);
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect(q.locator('.wd-review-tile')).toHaveCount(2);

  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [SECOND] });
  await q.locator('.wd-review-tile[data-review-cid="cyc-003"] .wd-review-answers button', { hasText: 'Quick 30°C' }).click();

  const calls = await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  expect(calls[0]).toMatchObject({ cycle_id: 'cyc-003', action: 'correct', corrected_profile: 'Quick 30°C' });
  const tiles = page.locator('.wd-review-q .wd-review-tile');
  await expect(tiles).toHaveCount(1, { timeout: 5_000 });
  await expect(tiles).toHaveAttribute('data-review-cid', 'cyc-004');
  await expect(page.locator('.wd-review-q-count')).toHaveText('To review: 1');
});

test('a zoomed tile does not pass its zoom to the cycle that moves into its place', async ({ page }) => {
  await bootWithQuestion(page, [REQUEST, SECOND]);
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect(q.locator('.wd-review-tile')).toHaveCount(2);
  await expect.poll(() => drawnReviewSeries(page)).toHaveLength(2);
  const zoom0 = () => page.evaluate(() =>
    (document.getElementById('wd-panel') as any)._canvasZoom['wd-review-canvas-0'] || null);
  await page.evaluate(() =>
    (document.getElementById('wd-panel') as any)._setCanvasViewport('wd-review-canvas-0', 0, 600));
  await expect.poll(zoom0).not.toBeNull();

  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [SECOND] });
  await q.locator('.wd-review-tile[data-review-cid="cyc-003"] .wd-review-answers button', { hasText: 'Quick 30°C' }).click();
  const tiles = page.locator('.wd-review-q .wd-review-tile');
  await expect(tiles).toHaveCount(1, { timeout: 5_000 });
  await expect(tiles).toHaveAttribute('data-review-cid', 'cyc-004');
  await expect.poll(zoom0).toBeNull();
});

test('a failed request-list refetch is retried on the next poll', async ({ page }) => {
  await bootWithQuestion(page, [REQUEST, SECOND]);
  await openQuestion(page);
  const tiles = page.locator('.wd-review-q .wd-review-tile');
  await expect(tiles).toHaveCount(2);
  const poll = () => page.evaluate(() => (document.getElementById('wd-panel') as any)._fetchAll());

  // cyc-003 is answered outside the panel; the refetch that follows fails.
  await setHandler(page, 'ha_washdata/get_devices', devices(deviceIdle, { feedback_count: 1 }));
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [SECOND] });
  await page.evaluate(() => (window as any).__set_error('ha_washdata/get_feedbacks'));
  await poll();
  // The next poll (same count) fetches again and drops the answered request.
  await page.evaluate(() => { delete (window as any).__ws_errors['ha_washdata/get_feedbacks']; });
  await poll();
  await expect(tiles).toHaveCount(1, { timeout: 5_000 });
  await expect(tiles).toHaveAttribute('data-review-cid', 'cyc-004');
});

test('a failed refetch after an answer is retried on the next poll', async ({ page }) => {
  await bootWithQuestion(page, [REQUEST, SECOND]);
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect(q.locator('.wd-review-tile')).toHaveCount(2);

  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await page.evaluate(() => (window as any).__set_error('ha_washdata/get_feedbacks'));
  await q.locator('.wd-review-tile[data-review-cid="cyc-003"] .wd-review-answers button', { hasText: 'Quick 30°C' }).click();
  await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);

  // The poll that follows sees the backend's count (1) and must still refetch.
  await page.evaluate(() => { delete (window as any).__ws_errors['ha_washdata/get_feedbacks']; });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [SECOND] });
  await setHandler(page, 'ha_washdata/get_devices', devices(deviceIdle, { feedback_count: 1 }));
  await page.evaluate(() => (document.getElementById('wd-panel') as any)._fetchAll());
  const tiles = page.locator('.wd-review-q .wd-review-tile');
  await expect(tiles).toHaveCount(1, { timeout: 5_000 });
  await expect(tiles).toHaveAttribute('data-review-cid', 'cyc-004');
});

test('an envelope fetched before an answer does not land in the cache the answer reset', async ({ page }) => {
  await bootWithQuestion(page, [REQUEST, SECOND]);
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect.poll(() => drawnReviewSeries(page)).toHaveLength(2);
  // Every envelope response is held until the test releases it.
  await page.evaluate(() => {
    const w = window as any;
    w.__envQ = [];
    w.__ws_handlers['ha_washdata/get_profile_envelope'] = () => new Promise(r => w.__envQ.push(r));
  });
  const held = () => page.evaluate(() => (window as any).__envQ.length);
  // A poll refetches the candidate curves; they stay in flight.
  await page.evaluate(() => {
    const el = document.getElementById('wd-panel') as any;
    el._reviewQ.envs = {};
    el._fetchAll();
  });
  await expect.poll(held).toBe(2);

  // The answer rebuilds the envelopes: the cache is reset and fetched again.
  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [SECOND] });
  await q.locator('.wd-review-tile[data-review-cid="cyc-003"] .wd-review-answers button', { hasText: 'Quick 30°C' }).click();
  await expect.poll(held).toBe(4);
  // The fresh responses land first, then the stale ones from before the answer.
  await page.evaluate(() => {
    const w = window as any;
    const env = (watts: number) => ({ envelope: { avg: [[0, watts], [2430, watts]], min: [[0, watts], [2430, watts]], max: [[0, watts], [2430, watts]], target_duration: 2430 } });
    w.__envQ.slice(2).forEach((r: any) => r(env(600)));
    w.__envQ.slice(0, 2).forEach((r: any) => r(env(999)));
  });
  const cached = () => page.evaluate(() => {
    const envs = (document.getElementById('wd-panel') as any)._reviewQ.envs;
    return Object.values(envs).map((e: any) => (e && e.avg ? e.avg[0][1] : null)).sort();
  });
  await expect.poll(cached, { timeout: 5_000 }).toEqual([600, 600]);
  await page.waitForTimeout(200);
  expect(await cached()).toEqual([600, 600]);
});

test('a request list fetched on another tab does not leave the Overview on a stale count', async ({ page }) => {
  await bootWithQuestion(page);
  await openQuestion(page);
  await expect(page.locator('.wd-review-q .wd-review-tile')).toHaveAttribute('data-review-cid', 'cyc-003');
  await clickTab(page, 'history');
  // Labelling cyc-003 on the Cycles tab refetches the list there: now empty.
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [] });
  await page.evaluate(() => (document.getElementById('wd-panel') as any)._fetchFeedbacks('test-entry-001'));
  // A new request brings the count back to 1 before the Overview polls again.
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [SECOND] });
  await setHandler(page, 'ha_washdata/get_devices', devices(deviceIdle, { feedback_count: 1 }));
  await clickTab(page, 'status');
  await page.evaluate(() => (document.getElementById('wd-panel') as any)._fetchAll());
  const tiles = page.locator('.wd-review-q .wd-review-tile');
  await expect(tiles).toHaveCount(1, { timeout: 5_000 });
  await expect(tiles).toHaveAttribute('data-review-cid', 'cyc-004');
});

test('an answered tile goes at once, not after the next tiles\' curves load', async ({ page }) => {
  await bootWithQuestion(page, [REQUEST, SECOND]);
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
  await expect.poll(() => drawnReviewSeries(page)).toHaveLength(2);
  // Every envelope response from here on is held (a slow remote link).
  await page.evaluate(() => {
    const w = window as any;
    w.__envQ = [];
    w.__ws_handlers['ha_washdata/get_profile_envelope'] = () => new Promise(r => w.__envQ.push(r));
  });
  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [SECOND] });
  await q.locator('.wd-review-tile[data-review-cid="cyc-003"] .wd-review-answers button', { hasText: 'Quick 30°C' }).click();
  await expect.poll(() => page.evaluate(() => (window as any).__envQ.length)).toBeGreaterThan(0);
  // Still loading, and the answered tile is already gone.
  await expect(q.locator('.wd-review-tile[data-review-cid="cyc-003"]')).toHaveCount(0, { timeout: 2_000 });
  await page.evaluate(() => (window as any).__envQ.forEach((r: any) => r(null)));
});

test('a second tap while an answer is in flight sends nothing', async ({ page }) => {
  await bootWithQuestion(page);
  await openQuestion(page);
  // The answer's round trip is held, as on a slow remote link.
  await page.evaluate(() => {
    const w = window as any;
    w.__resolveQ = [];
    w.__ws_handlers['ha_washdata/resolve_feedback'] = () => new Promise(r => w.__resolveQ.push(r));
  });
  const btn = page.locator('.wd-review-q .wd-review-answers button', { hasText: 'Eco 60°C' });
  await btn.click();
  await btn.click();
  await page.locator('.wd-review-q .wd-review-answers button', { hasText: 'Not sure' }).click();
  await expect.poll(() => page.evaluate(() => (window as any).__resolveQ.length)).toBe(1);
  await page.waitForTimeout(200);
  const calls = await page.evaluate(() => (window as any).__get_calls('ha_washdata/resolve_feedback'));
  expect(calls).toHaveLength(1);
  expect(calls[0]).toMatchObject({ cycle_id: 'cyc-003', action: 'confirm' });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [] });
  await page.evaluate(() => (window as any).__resolveQ.forEach((r: any) => r({ success: true })));
  await expect(page.locator('.wd-review-q')).toHaveCount(0, { timeout: 5_000 });
});

test('a failed candidate-curve fetch is retried on the next poll', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_devices': devices(deviceIdle, { feedback_count: 1 }),
    'ha_washdata/get_feedbacks': { feedbacks: [REQUEST] },
    'ha_washdata/get_profile_envelope': ENVELOPE,
  });
  await page.evaluate(() => (window as any).__set_error('ha_washdata/get_profile_envelope'));
  await openQuestion(page);
  await expect(page.locator('#wd-review-canvas-0')).toBeVisible({ timeout: 5_000 });
  await expect.poll(() => drawnReviewSeries(page)).toEqual([]);

  await page.evaluate(() => { delete (window as any).__ws_errors['ha_washdata/get_profile_envelope']; });
  await page.evaluate(() => (document.getElementById('wd-panel') as any)._fetchAll());
  await expect.poll(() => drawnReviewSeries(page), { timeout: 5_000 }).toHaveLength(2);
});

test('a request answered outside the panel is not asked again when the next one arrives', async ({ page }) => {
  await bootWithQuestion(page);
  await openQuestion(page);
  const tiles = page.locator('.wd-review-q .wd-review-tile');
  await expect(tiles).toHaveAttribute('data-review-cid', 'cyc-003');
  const poll = () => page.evaluate(() => (document.getElementById('wd-panel') as any)._fetchAll());

  // A notification action answers it: the next poll sees no request waiting.
  await setHandler(page, 'ha_washdata/get_devices', devices(deviceIdle, { feedback_count: 0 }));
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [] });
  await poll();
  await expect(page.locator('.wd-review-q')).toHaveCount(0, { timeout: 5_000 });

  // A new request brings the count back to 1: it is fetched and asked.
  await setHandler(page, 'ha_washdata/get_devices', devices(deviceIdle, { feedback_count: 1 }));
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [SECOND] });
  await poll();
  await expect(tiles).toHaveCount(1, { timeout: 5_000 });
  await expect(tiles).toHaveAttribute('data-review-cid', 'cyc-004');
});

test('picking the detected programme confirms it', async ({ page }) => {
  await bootWithQuestion(page);
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
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
  await openQuestion(page);
  const q = page.locator('.wd-review-q');
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

test('the questions fit a phone screen; only the tile strip scrolls sideways', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 780 });
  await bootWithQuestion(page, [REQUEST, SECOND]);
  await openQuestion(page);
  await expect(page.locator('.wd-review-tile')).toHaveCount(2);
  const m = await page.evaluate(`(() => {
    const sr = document.querySelector('${PANEL}').shadowRoot;
    const body = sr.querySelector('.wd-body');
    const q = sr.querySelector('.wd-review-q');
    const strip = q.querySelector('.wd-review-tiles');
    const tile = strip.querySelector('.wd-review-tile');
    return { body: body.scrollWidth - body.clientWidth, q: q.scrollWidth - q.clientWidth,
      tileOverflow: tile.scrollWidth - tile.clientWidth, peek: strip.clientWidth - tile.offsetWidth };
  })()`) as { body: number; q: number; tileOverflow: number; peek: number };
  expect(m.body).toBeLessThanOrEqual(1);
  expect(m.q).toBeLessThanOrEqual(1);
  expect(m.tileOverflow).toBeLessThanOrEqual(1);
  // The next tile peeks in, so the strip reads as scrollable.
  expect(m.peek).toBeGreaterThan(20);
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

test('answering the other programme in the dialog drops the old programme\'s curve', async ({ page }) => {
  await bootWithQuestion(page);
  await clickTab(page, 'history');
  await page.locator('tr[data-cid="cyc-003"]').click();
  const modal = page.locator('.wd-modal');
  await expect(modal.locator('.wd-review-answers button')).toHaveCount(4, { timeout: 5_000 });
  const env = () => page.evaluate(() => {
    const m = (document.getElementById('wd-panel') as any)._modal;
    return m && m.profileEnv ? 'set' : 'none';
  });
  await expect.poll(env).toBe('set');   // the suspected Eco 60°C curve

  // The new programme's envelope cannot be fetched: Eco 60°C's curve must not stay
  // on the chart under the name Quick 30°C.
  await page.evaluate(() => (window as any).__set_error('ha_washdata/get_profile_envelope'));
  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [] });
  await modal.locator('.wd-review-answers button', { hasText: 'Quick 30°C' }).click();
  await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  await expect.poll(env).toBe('none');
});

test('a one-tap answer replaces an unsaved Review-form label', async ({ page }) => {
  await bootWithQuestion(page);
  await clickTab(page, 'history');
  await page.locator('tr[data-cid="cyc-003"]').click();
  const modal = page.locator('.wd-modal');
  await expect(modal.locator('.wd-review-answers button')).toHaveCount(4, { timeout: 5_000 });
  await page.locator('button[data-maction="cyc-review"]').click();
  await expect(page.locator('#wd-cyc-rev-label')).toHaveValue('');
  // Any form re-render snapshots the select, storing the unlabelled '' choice.
  await page.evaluate(() => {
    const el = document.getElementById('wd-panel') as any;
    el._snapshotCycleReviewForm(el.shadowRoot);
  });

  await setHandler(page, 'ha_washdata/resolve_feedback', { success: true });
  await setHandler(page, 'ha_washdata/get_feedbacks', { feedbacks: [] });
  await modal.locator('.wd-review-answers button', { hasText: 'Quick 30°C' }).click();
  await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  // Save Review would otherwise send '' and unlabel the cycle.
  await expect(page.locator('#wd-cyc-rev-label')).toHaveValue('Quick 30°C', { timeout: 5_000 });
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

// A get_feedbacks answer for a device switched away from meanwhile used to become
// the shared list, which the Cycles view reads without checking whose it is.
test('a review list that arrives for another device is dropped', async ({ page }) => {
  await bootWithQuestion(page);
  const out = await page.evaluate(async () => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const before = el._feedbacks;
    const real = el._ws.bind(el);
    el._ws = (msg: any) => (msg.type === 'ha_washdata/get_feedbacks' && msg.entry_id === 'other-entry'
      ? Promise.resolve({ feedbacks: [{ cycle_id: 'foreign' }] })
      : real(msg));
    const ok = await el._fetchFeedbacks('other-entry');
    return { ok, same: el._feedbacks === before, eid: el._feedbacksEid };
  });
  expect(out).toEqual({ ok: false, same: true, eid: 'test-entry-001' });
});


// Answered from the notification, a request leaves the count lower without this
// panel answering: that answer rebuilt its programme's envelope, which the
// remaining tile may draw, so the cached candidate curves are dropped.
test('a review count that moved elsewhere drops the cached candidate curves', async ({ page }) => {
  await bootWithQuestion(page);
  const replaced = await page.evaluate(async () => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const dev = el._devices[el._selIdx];
    const stale = { stale: true };
    el._reviewQ = { eid: dev.entry_id, curves: {}, envs: { 'Eco 60°C': stale } };
    el._feedbacksSeenCount = (dev.feedback_count || 0) + 1;   // another answer moved it
    await el._ensureReviewQuestion(dev);
    return el._reviewQ.envs['Eco 60°C'] !== stale;
  });
  expect(replaced).toBe(true);
});

// A one-tap answer in the cycle dialog re-rendered it through its toast before
// the Review form was snapshotted, so unsaved notes were gone for Save Review.
test('a one-tap answer keeps the unsaved Review notes', async ({ page }) => {
  await bootWithQuestion(page);
  await clickTab(page, 'history');
  await page.locator('tr[data-cid="cyc-003"]').click();
  await expect(page.locator('#wd-cyc-canvas')).toBeVisible({ timeout: 5_000 });
  await page.locator('button[data-maction="cyc-review"]').click();
  await page.locator('#wd-cyc-rev-notes').fill('drum squeaked');
  const pick = page.locator('.wd-modal button[data-action="fb-pick"]').first();
  await expect(pick).toBeVisible({ timeout: 5_000 });
  await pick.click();
  await expect(page.locator('.wd-toast')).toBeVisible();
  await expect(page.locator('#wd-cyc-rev-notes')).toHaveValue('drum squeaked');
});

// A request from an older backend has no `choices`; its ranking can carry a
// Stage-5 group key, which names no programme.
test('the legacy answer fallback names a group by its scoring member', async ({ page }) => {
  await bootWithQuestion(page);
  const names = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel') as any;
    return [
      el._reviewChoiceNames({ detected_profile: 'Eco 60°C', ranking: [
        { name: 'Eco 60°C' }, { name: '__group__Cottons', group_best_member: 'Cotton 40°C' }] }),
      el._reviewChoiceNames({ detected_profile: 'Eco 60°C', ranking: [
        { name: '__group__Cottons' }, { name: 'Quick 30°C' }] }),
    ];
  });
  expect(names).toEqual([['Eco 60°C', 'Cotton 40°C'], ['Eco 60°C', 'Quick 30°C']]);
});

// The Previous and Next steps share data-action: a background re-render put focus
// back on the first of them, not the one the keyboard was on.
test('a re-render keeps focus on the same review step', async ({ page }) => {
  await bootWithQuestion(page, [REQUEST, { ...REQUEST, cycle_id: 'cyc-002' }]);
  await openQuestion(page);
  const steps = page.locator('button[data-action="review-q-step"]');
  await expect(steps).toHaveCount(2, { timeout: 5_000 });
  const second = steps.nth(1);
  const dir = await second.getAttribute('data-dir');
  await second.focus();
  await page.evaluate(() => (document.querySelector('ha-washdata-panel') as any)._render());
  const focused = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const a = el.shadowRoot.activeElement;
    return a ? a.getAttribute('data-dir') : null;
  });
  expect(focused).toBe(dir);
});
