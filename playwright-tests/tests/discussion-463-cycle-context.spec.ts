/**
 * Discussion #463 / register item 513: the cycle chart opens at the first reading
 * over the start threshold, glued to the y axis, so a cycle looks cut off and a
 * user labelling by noted start times cannot see what came before it.
 *
 * The cycle dialog now asks get_cycle_context for the power sensor's recorder
 * history either side of the stored trace and draws it grey on a shaded
 * background, read as "Before start" / "After end". The length is a per-device
 * pref (10 min by default, Off turns it off), it is absent when the recorder has
 * nothing, and Trim/Split never show it (they edit the trace itself).
 */

import { test, expect, Page } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled } from '../helpers/panel';
import panelConfig from '../fixtures/mock-data/panel-config.json';

const PANEL = 'ha-washdata-panel';
const EID = 'test-entry-001';

/**
 * Boot the panel, then install a recorder reply (functions do not cross into the
 * page through bootPanel): idle before with one unavailable row, a tail after.
 */
async function bootWith(page: Page, overrides: Record<string, unknown> = {}, opts: { context?: boolean; reference?: boolean } = {}) {
  await page.goto('/');
  await bootPanel(page, overrides);
  await page.evaluate(({ context, reference }) => {
    const w = window as any;
    if (context) {
      w.__ws_handlers['ha_washdata/get_cycle_context'] = (msg: any) => ({
        cycle_id: msg.cycle_id, available: true, reason: null, entity_id: 'sensor.washer_power',
        before_s: msg.before_s, after_s: msg.after_s, trace_end_s: 1740, after_end_s: 1740 + msg.after_s,
        before: [[-msg.before_s, 0.0], [-msg.before_s / 2, null], [-msg.before_s / 3, 1.5], [-30, 2.0]],
        after: [[1740, 3.0], [1800, 0.5]],
      });
    }
    if (reference) {
      w.__ws_handlers['ha_washdata/get_cycle_power_data'] = (msg: any) => ({
        cycle_id: msg.cycle_id,
        samples: Array.from({ length: 30 }, (_, i) => [i * 60, i < 2 || i > 27 ? 3 : 900]),
        sample_count: 30, decimated: false, full_duration_s: 1740,
        profile_name: null, status: 'completed', artifacts: [], restart_gaps: [],
        is_reference: true, labelable: true, editable: false, cycle_origin: 'reference',
      });
    }
  }, { context: !!opts.context, reference: !!opts.reference });
}

async function openCycle(page: Page, cid = 'cyc-001') {
  await clickTab(page, 'history');
  const row = page.locator(`tr[data-cid="${cid}"]`);
  await expect(row).toBeVisible({ timeout: 5_000 });
  await row.click();
  await expect(page.locator('#wd-cyc-canvas')).toBeVisible({ timeout: 5_000 });
}

/** What the cycle chart last drew: its left edge and the context series. */
async function chart(page: Page): Promise<{ xStart: number; xMax: number; ctx: [number, number | null][] | null; kinds: string[] }> {
  return page.evaluate(`(() => {
    const sr = document.querySelector('${PANEL}').shadowRoot;
    const c = sr.getElementById('wd-cyc-canvas');
    if (!c || !c._wd) return { xStart: NaN, xMax: NaN, ctx: null, kinds: [] };
    const s = c._wd.series.find(x => x.kind === 'context');
    return { xStart: c._wd.xStart, xMax: c._wd.xMax, ctx: s ? s.points : null, kinds: c._wd.series.map(x => x.kind || '') };
  })()`) as any;
}

/** Hover the chart at x seconds and return the readout lines (HTML stripped). */
async function readoutAt(page: Page, xs: number): Promise<string[]> {
  const pos = await page.evaluate(`(() => {
    const sr = document.querySelector('${PANEL}').shadowRoot;
    const c = sr.getElementById('wd-cyc-canvas');
    c.scrollIntoView({ block: 'center' });
    const r = c.getBoundingClientRect();
    return { x: r.left + c._wd.xToCss(${xs}), y: r.top + r.height / 2 };
  })()`) as { x: number; y: number };
  await page.mouse.move(pos.x, pos.y);
  await expect.poll(() => page.evaluate(`(() => (document.querySelector('${PANEL}')._gtipLines || []).length)()`)).toBeGreaterThan(0);
  return page.evaluate(`(() => (document.querySelector('${PANEL}')._gtipLines || []).map(l => l.replace(/<[^>]+>/g, '')))()`) as Promise<string[]>;
}

test('draws the recorder history before and after the cycle, grey and labelled', async ({ page }) => {
  await bootWith(page, {}, { context: true });
  await openCycle(page);
  const calls = await assertWsCalled(page, 'ha_washdata/get_cycle_context', 1);
  expect(calls[0]).toMatchObject({ entry_id: EID, cycle_id: 'cyc-001', before_s: 600, after_s: 600 });

  const modal = page.locator('.wd-modal');
  await expect(modal.locator('[data-cyc-legend] [data-leg="context"]')).toHaveText('Recorder history');
  await expect(modal.locator('[data-cyc-ctx-note="shown"]')).toContainText('not part of the cycle');
  await expect(modal.locator('#wd-cyc-ctx')).toHaveValue('10');

  const c = await chart(page);
  expect(c.xStart).toBe(-600);
  expect(c.xMax).toBe(2340);
  const ctx = c.ctx!;
  // Steps, broken at the unavailable row, rising into the trace at 0 and starting
  // again from the trace's last reading after the end.
  expect(ctx[0]).toEqual([-600, 0]);
  expect(ctx).toContainEqual([-300, null]);
  expect(ctx).toContainEqual([0, 2.0]);
  expect(ctx).toContainEqual([0, 3]);  // the trace's first sample (3 W)
  expect(ctx).toContainEqual([1740, null]);  // no line across the cycle itself
  expect(ctx[ctx.length - 1]).toEqual([2340, 0.5]);

  // The readout says where the cursor is relative to the cycle, and the trace has
  // no value in the lead-in (it is not "Power" there).
  const before = await readoutAt(page, -120);
  expect(before[0]).toBe('Before start: 2:00');
  expect(before).toContain('Recorder history: 1.5 W');
  expect(before.some(l => l.startsWith('Power:'))).toBe(false);
  const inside = await readoutAt(page, 900);
  expect(inside[0]).toBe('From start: 15:00');
  expect(inside.some(l => l.startsWith('Power:'))).toBe(true);
  expect(inside.some(l => l.startsWith('Recorder history:'))).toBe(false);
  const after = await readoutAt(page, 1900);
  expect(after[0]).toBe('After end: 2:40');
});

test('the length is adjustable, remembered per device, and Off removes it', async ({ page }) => {
  await bootWith(page, {}, { context: true });
  await openCycle(page);
  await expect.poll(async () => (await chart(page)).xStart).toBe(-600);

  await page.locator('#wd-cyc-ctx').selectOption('30');
  await expect.poll(async () => (await chart(page)).xStart).toBe(-1800);
  const calls = await assertWsCalled(page, 'ha_washdata/get_cycle_context', 2);
  expect(calls[calls.length - 1]).toMatchObject({ before_s: 1800, after_s: 1800 });
  const prefs = await assertWsCalled(page, 'ha_washdata/set_user_prefs', 1);
  expect(prefs[prefs.length - 1]).toMatchObject({ prefs: { cycle_context_min: { [EID]: 30 } } });

  await page.locator('#wd-cyc-ctx').selectOption('0');
  await expect.poll(async () => (await chart(page)).xStart).toBe(0);
  const c = await chart(page);
  expect(c.ctx).toBeNull();
  expect(c.xMax).toBe(1740);
  await expect(page.locator('.wd-modal [data-leg="context"]')).toHaveCount(0);
  await expect(page.locator('.wd-modal [data-cyc-ctx-note]')).toHaveCount(0);
  // Off asks the recorder nothing.
  expect((await assertWsCalled(page, 'ha_washdata/get_cycle_context', 2)).length).toBe(2);
});

test('a device whose pref is Off opens without asking the recorder', async ({ page }) => {
  const cfg = { ...panelConfig, prefs: { ...panelConfig.prefs, cycle_context_min: { [EID]: 0 } } };
  await bootWith(page, { 'ha_washdata/get_panel_config': cfg }, { context: true });
  await openCycle(page);
  await expect(page.locator('#wd-cyc-ctx')).toHaveValue('0');
  await assertWsNotCalled(page, 'ha_washdata/get_cycle_context');
  expect((await chart(page)).xStart).toBe(0);
});

test('absent when the recorder has nothing, and says so', async ({ page }) => {
  await bootWith(page);  // default handler: available false
  await openCycle(page);
  await assertWsCalled(page, 'ha_washdata/get_cycle_context', 1);
  await expect(page.locator('.wd-modal [data-cyc-ctx-note="none"]')).toContainText('no readings around this cycle');
  const c = await chart(page);
  expect(c.xStart).toBe(0);
  expect(c.ctx).toBeNull();
  expect(c.xMax).toBe(1740);
  await expect(page.locator('.wd-modal [data-leg="context"]')).toHaveCount(0);
  // The readout is the unchanged in-cycle one.
  const lines = await readoutAt(page, 0);
  expect(lines[0]).toBe('From start: 0:00');
});

test('Trim never shows the context, and Inspect brings it back', async ({ page }) => {
  await bootWith(page, {}, { context: true });
  await openCycle(page);
  await expect.poll(async () => (await chart(page)).xStart).toBe(-600);
  await page.locator('button[data-maction="cyc-trim"]').click();
  await expect.poll(async () => (await chart(page)).xStart).toBe(0);
  expect((await chart(page)).ctx).toBeNull();
  await expect(page.locator('#wd-cyc-ctx')).toHaveCount(0);
  await page.locator('button[data-maction="cyc-view"]').click();
  await expect.poll(async () => (await chart(page)).xStart).toBe(-600);
});

test('an imported cycle neither asks for nor draws context', async ({ page }) => {
  await bootWith(page, {}, { context: true, reference: true });
  await openCycle(page);
  await expect(page.locator('#wd-cyc-ctx')).toHaveCount(0);
  await assertWsNotCalled(page, 'ha_washdata/get_cycle_context');
  expect((await chart(page)).xStart).toBe(0);
});
