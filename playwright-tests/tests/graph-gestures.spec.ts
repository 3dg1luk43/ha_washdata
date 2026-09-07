/**
 * Graph gesture tests (issue #413).
 *
 * Covers the touch contract that makes the charts usable on a phone without
 * taking anything away from mouse users:
 *   - a vertical swipe over a chart scrolls the PAGE (the chart no longer
 *     swallows it, which is what "touch-action: none" used to do)
 *   - a two-finger pinch zooms, and a two-finger drag pans
 *   - double tap zooms in, and resets when already zoomed
 *   - the reset button appears only while zoomed
 *   - the readout survives lifting the finger and is never placed under it
 *   - trim-handle drags and split taps still work on touch
 *
 * Touch gestures are dispatched through CDP because page.touchscreen only
 * supports taps, and a pinch needs two simultaneous touch points.
 */

import { test, expect, type Page, type CDPSession } from '@playwright/test';
import { bootPanel, clickTab, setHandler } from '../helpers/panel';
import { buildHandlers } from '../helpers/ws-handlers';
import de from '../../custom_components/ha_washdata/translations/panel/de.json';

const PANEL = 'ha-washdata-panel';

async function panelState<T>(page: Page, fn: string): Promise<T> {
  return page.evaluate(`(() => {
    const root = document.querySelector('${PANEL}');
    const sr = root.shadowRoot;
    return (${fn})(root, sr);
  })()`) as Promise<T>;
}

/** Bounding rect of a canvas, after scrolling it fully into view. */
async function canvasBox(page: Page, id: string) {
  return panelState<{ x: number; y: number; width: number; height: number; top: number; bottom: number }>(
    page,
    `(root, sr) => { const c = sr.getElementById('${id}'); c.scrollIntoView({ block: 'center' }); return c.getBoundingClientRect().toJSON(); }`,
  );
}

async function swipe(cdp: CDPSession, x: number, y: number, dx: number, dy: number, id = 1) {
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y, id }] });
  for (let i = 1; i <= 10; i++) {
    await cdp.send('Input.dispatchTouchEvent', {
      type: 'touchMove',
      touchPoints: [{ x: x + (dx * i) / 10, y: y + (dy * i) / 10, id }],
    });
  }
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
}

/** Symmetric two-finger gesture. spread > 0 pinches out (zoom in). */
async function pinch(cdp: CDPSession, cx: number, cy: number, from: number, spread: number, driftX = 0) {
  const pt = (off: number, dx: number, id: number) => ({ x: cx + dx + off, y: cy, id });
  await cdp.send('Input.dispatchTouchEvent', {
    type: 'touchStart',
    touchPoints: [pt(-from, 0, 21), pt(from, 0, 22)],
  });
  for (let i = 1; i <= 8; i++) {
    const off = from + (spread * i) / 8;
    const dx = (driftX * i) / 8;
    await cdp.send('Input.dispatchTouchEvent', {
      type: 'touchMove',
      touchPoints: [pt(-off, dx, 21), pt(off, dx, 22)],
    });
  }
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
}

async function doubleTap(cdp: CDPSession, x: number, y: number, page: Page) {
  for (const id of [31, 32]) {
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y, id }] });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await page.waitForTimeout(60);
  }
}

const scrollTop = (page: Page) => page.evaluate(() => document.scrollingElement!.scrollTop);
const zoomOf = (page: Page, id: string) =>
  panelState<{ xMin: number; xMax: number } | null>(page, `(root) => root._canvasZoom['${id}'] || null`);

/** A cycle trace long enough to have a meaningful x axis to zoom into. */
const CYCLE_CURVE = {
  samples: Array.from({ length: 120 }, (_, i) => [i * 30, i < 3 || i > 116 ? 2 : 400 + (i % 7) * 90]),
  full_duration_s: 3600,
  artifacts: [],
  envelope_conformance: null,
};

/** Open the cycle-detail modal, whose canvas has a real (non-degenerate) x axis. */
async function openCycleModal(page: Page) {
  await setHandler(page, 'ha_washdata/get_cycle_power_data', CYCLE_CURVE);
  await clickTab(page, 'history');
  const firstRow = page.locator('tr[data-cid]').first();
  await expect(firstRow).toBeVisible({ timeout: 8_000 });
  await firstRow.click();
  await expect(page.locator('#wd-cyc-canvas')).toBeVisible({ timeout: 8_000 });
  // Wait for _drawCurves to have published its hit-test map.
  await expect
    .poll(() => panelState<number>(page, `(root, sr) => { const c = sr.getElementById('wd-cyc-canvas'); return c && c._wd ? c._wd.xMax : 0; }`))
    .toBeGreaterThan(100);
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {});
});

// ─── Navigation: the chart must not swallow page scrolling ───────────────────

test('graph canvases declare touch-action pan-y so a swipe scrolls the page', async ({ page }) => {
  const ta = await panelState<string>(
    page,
    `(root, sr) => getComputedStyle(sr.getElementById('wd-status-canvas')).touchAction`,
  );
  expect(ta).toBe('pan-y');
});

test('playground canvas declares touch-action pan-y as well', async ({ page }) => {
  await clickTab(page, 'playground');
  await expect(page.locator('#wd-pg-canvas')).toBeVisible({ timeout: 8_000 });
  const ta = await panelState<string>(
    page,
    `(root, sr) => getComputedStyle(sr.getElementById('wd-pg-canvas')).touchAction`,
  );
  expect(ta).toBe('pan-y');
});

test.describe('touch', () => {
  test.skip(({ hasTouch }) => !hasTouch, 'touch-only gestures');

  test('a vertical swipe starting on a chart scrolls the page (#413)', async ({ page }) => {
    // Overview, not a modal: this is the case users hit, and the page here is
    // genuinely taller than the viewport. Before the fix this swipe moved
    // nothing at all, because touch-action: none handed every touch that began
    // on a chart to the canvas and the scroller never saw it.
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-status-canvas');
    const before = await scrollTop(page);
    await swipe(cdp, Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2), 0, -170);
    await page.waitForTimeout(400);
    expect(await scrollTop(page)).toBeGreaterThan(before);
  });

  test('the same swipe cancels the chart pointer instead of scrubbing', async ({ page }) => {
    // The browser claiming the gesture shows up as pointercancel. Handlers have
    // to treat that as "gesture over", or an interrupted drag stays armed and
    // the next hover keeps dragging with no finger down.
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-status-canvas');
    await panelState(page, `(root, sr) => { const c = sr.getElementById('wd-status-canvas');
      window.__ev = []; ['pointerdown','pointercancel'].forEach(t => c.addEventListener(t, () => window.__ev.push(t), { passive: true })); }`);
    await swipe(cdp, Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2), 0, -170);
    await page.waitForTimeout(300);
    const ev = await page.evaluate(() => (window as unknown as { __ev: string[] }).__ev.join(' '));
    expect(ev).toContain('pointercancel');
  });

  test('a two-finger pinch zooms the x axis and reveals the reset button', async ({ page }) => {
    await openCycleModal(page);
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    expect(await zoomOf(page, 'wd-cyc-canvas')).toBeNull();

    await pinch(cdp, Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2), 40, 90);
    await page.waitForTimeout(300);

    const zoom = await zoomOf(page, 'wd-cyc-canvas');
    expect(zoom).not.toBeNull();
    expect(zoom!.xMax - zoom!.xMin).toBeLessThan(3600 * 0.99);
    await expect(page.locator('.wd-modal .wd-zoom-reset.wd-zoom-reset--on')).toBeVisible();
  });

  test('pinching back out clears the zoom', async ({ page }) => {
    await openCycleModal(page);
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    const cx = Math.round(box.x + box.width / 2);
    const cy = Math.round(box.y + box.height / 2);
    await pinch(cdp, cx, cy, 30, 120);
    await page.waitForTimeout(250);
    expect(await zoomOf(page, 'wd-cyc-canvas')).not.toBeNull();
    // Pinch in repeatedly: each gesture shrinks the range back towards full.
    for (let i = 0; i < 6; i++) {
      await pinch(cdp, cx, cy, 150, -130);
      await page.waitForTimeout(120);
    }
    expect(await zoomOf(page, 'wd-cyc-canvas')).toBeNull();
  });

  test('a two-finger drag pans the zoomed viewport', async ({ page }) => {
    await openCycleModal(page);
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    const cx = Math.round(box.x + box.width / 2);
    const cy = Math.round(box.y + box.height / 2);
    await pinch(cdp, cx, cy, 30, 110);
    await page.waitForTimeout(250);
    const zoomed = await zoomOf(page, 'wd-cyc-canvas');
    expect(zoomed).not.toBeNull();

    // Constant separation, centre travelling left => viewport moves later.
    await pinch(cdp, cx, cy, 60, 0, -90);
    await page.waitForTimeout(250);
    const panned = await zoomOf(page, 'wd-cyc-canvas');
    expect(panned).not.toBeNull();
    expect(panned!.xMin).toBeGreaterThan(zoomed!.xMin);
    // Panning must not change the window width.
    expect(Math.abs((panned!.xMax - panned!.xMin) - (zoomed!.xMax - zoomed!.xMin))).toBeLessThan(2);
  });

  test('double tap zooms in, and resets when already zoomed', async ({ page }) => {
    await openCycleModal(page);
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    const cx = Math.round(box.x + box.width / 2);
    const cy = Math.round(box.y + box.height / 2);

    await doubleTap(cdp, cx, cy, page);
    await page.waitForTimeout(250);
    const zoom = await zoomOf(page, 'wd-cyc-canvas');
    expect(zoom).not.toBeNull();
    expect(zoom!.xMax - zoom!.xMin).toBeLessThan(3600 * 0.8);

    // Well clear of the browser's ~300ms double-tap window, or taps 3-4 chain
    // onto 1-2 and the second dblclick never fires.
    await page.waitForTimeout(600);
    await doubleTap(cdp, cx, cy, page);
    await page.waitForTimeout(250);
    expect(await zoomOf(page, 'wd-cyc-canvas')).toBeNull();
  });

  test('the reset button clears the zoom and hides itself', async ({ page }) => {
    await openCycleModal(page);
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    await pinch(cdp, Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2), 35, 100);
    await page.waitForTimeout(250);

    const btn = page.locator('.wd-modal .wd-zoom-reset');
    await expect(btn).toBeVisible();
    await btn.click();
    expect(await zoomOf(page, 'wd-cyc-canvas')).toBeNull();
    await expect(btn).not.toBeVisible();
  });

  test('the readout stays pinned after the finger lifts, and is not under it', async ({ page }) => {
    await openCycleModal(page);
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    const tx = Math.round(box.x + box.width * 0.55);
    const ty = Math.round(box.y + box.height * 0.6);

    await swipe(cdp, tx, ty, 40, 0, 41);
    await page.waitForTimeout(300);

    const tip = await panelState<{ display: string; top: number; bottom: number; left: number; right: number; pinned: boolean; dismissHint: boolean }>(
      page,
      `(root, sr) => { const t = sr.querySelector('.wd-gtip'); const r = t.getBoundingClientRect();
        return { display: getComputedStyle(t).display, top: r.top, bottom: r.bottom, left: r.left, right: r.right,
                 pinned: t.classList.contains('wd-gtip--pinned'),
                 dismissHint: !!t.querySelector('.wd-gtip-dismiss') }; }`,
    );
    expect(tip.display).not.toBe('none');
    expect(tip.pinned).toBe(true);
    // ...and it says how to get rid of it.
    expect(tip.dismissHint).toBe(true);
    // The readout must not overlap the touch point (a fingertip is ~45px wide).
    const overlaps = ty >= tip.top - 22 && ty <= tip.bottom + 22 && tx + 40 >= tip.left && tx + 40 <= tip.right;
    expect(overlaps).toBe(false);
  });

  test('tapping outside the chart dismisses the pinned readout', async ({ page }) => {
    await openCycleModal(page);
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    await swipe(cdp, Math.round(box.x + box.width * 0.5), Math.round(box.y + box.height * 0.6), 35, 0, 42);
    await page.waitForTimeout(250);
    expect(await panelState<boolean>(page, `(root) => root._gtipPinned`)).toBe(true);

    await page.locator('.wd-modal h2').first().click({ force: true });
    await page.waitForTimeout(150);
    expect(await panelState<boolean>(page, `(root) => root._gtipPinned`)).toBe(false);
    const disp = await panelState<string>(page, `(root, sr) => getComputedStyle(sr.querySelector('.wd-gtip')).display`);
    expect(disp).toBe('none');
  });

  test('trim handles are still draggable by touch', async ({ page }) => {
    await openCycleModal(page);
    const trimBtn = page.locator('[data-maction="cyc-trim"]').first();
    // Assert, do not skip: a wrong selector here used to make all three trim
    // tests report as skipped, i.e. green while validating nothing.
    await expect(trimBtn).toBeVisible({ timeout: 5_000 });
    await trimBtn.click();
    await expect(page.locator('#wd-trim-start')).toBeVisible({ timeout: 5_000 });

    const before = await panelState<number>(page, `(root) => root._modal.trim.start`);
    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    // Grab near the left (start) handle and drag right.
    await swipe(cdp, Math.round(box.x + 50), Math.round(box.y + box.height / 2), 110, 0, 51);
    await page.waitForTimeout(300);
    const after = await panelState<number>(page, `(root) => root._modal.trim.start`);
    expect(after).toBeGreaterThan(before);
  });

  test('trim mode does not re-trap page scrolling away from the handles', async ({ page }) => {
    // The touch guard only lets go of a vertical swipe while the chart has NOT
    // claimed the gesture, so an unconditional claim in trim mode would undo
    // the #413 fix inside that mode. A touch far from both handles must claim
    // nothing.
    await openCycleModal(page);
    const trimBtn = page.locator('[data-maction="cyc-trim"]').first();
    // Assert, do not skip: a wrong selector here used to make all three trim
    // tests report as skipped, i.e. green while validating nothing.
    await expect(trimBtn).toBeVisible({ timeout: 5_000 });
    await trimBtn.click();
    await expect(page.locator('#wd-trim-start')).toBeVisible({ timeout: 5_000 });

    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    const before = await panelState<{ start: number; end: number }>(
      page, `(root) => ({ start: root._modal.trim.start, end: root._modal.trim.end })`);
    // Mid-chart: far from the start handle (left edge) and the end handle (right edge).
    await swipe(cdp, Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2), 0, -150, 61);
    await page.waitForTimeout(300);

    const claimed = await panelState<boolean>(
      page, `(root, sr) => !!sr.getElementById('wd-cyc-canvas')._wdOwnGesture`);
    expect(claimed).toBe(false);
    const after = await panelState<{ start: number; end: number }>(
      page, `(root) => ({ start: root._modal.trim.start, end: root._modal.trim.end })`);
    expect(after.start).toBeCloseTo(before.start, 1);
    expect(after.end).toBeCloseTo(before.end, 1);
  });

  test('a second finger landing on a grabbed trim handle pinches instead of dragging', async ({ page }) => {
    // The case the abort hook exists for: finger one is ON the start handle, so
    // the drag IS armed, and only then does finger two arrive. Without
    // _wdAbortEdit + the _wdPinching guard the handle follows the pinch centre.
    await openCycleModal(page);
    const trimBtn = page.locator('[data-maction="cyc-trim"]').first();
    await expect(trimBtn).toBeVisible({ timeout: 5_000 });
    await trimBtn.click();
    await expect(page.locator('#wd-trim-start')).toBeVisible({ timeout: 5_000 });

    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    // Finger 1 right on the start handle (the plot's left edge, padL = 44 CSS px).
    const hx = Math.round(box.x + 46);
    const cy = Math.round(box.y + box.height / 2);
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: hx, y: cy, id: 71 }] });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: hx + 6, y: cy, id: 71 }] });
    const armed = await panelState<string | null>(page, `(root) => root._modal.drag || null`);
    expect(armed).toBe('start');   // the drag really was armed
    // Baseline AFTER that legitimate one-finger drag: the question is only
    // whether the pinch moves the handle further, not whether the drag worked.
    const before = await panelState<{ start: number; end: number }>(
      page, `(root) => ({ start: root._modal.trim.start, end: root._modal.trim.end })`);

    // Finger 2 arrives -> becomes a pinch, and the armed drag must be abandoned.
    // Two independent mechanisms do that; this asserts the observable end state.
    // The touch tolerance in the trim pointerdown clears m.drag because finger 2
    // is nowhere near a handle (first line of defence), and the gesture layer's
    // _wdAbortEdit hook clears it wherever finger 2 lands (covered separately).
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [
      { x: hx + 6, y: cy, id: 71 }, { x: hx + 160, y: cy, id: 72 }] });
    expect(await panelState<string | null>(page, `(root) => root._modal.drag || null`)).toBeNull();
    expect(await panelState<boolean>(page,
      `(root, sr) => !!sr.getElementById('wd-cyc-canvas')._wdPinching`)).toBe(true);
    for (let i = 1; i <= 6; i++) {
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [
        { x: hx + 6 - i * 6, y: cy, id: 71 }, { x: hx + 160 + i * 10, y: cy, id: 72 }] });
    }
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await page.waitForTimeout(300);

    const after = await panelState<{ start: number; end: number }>(
      page, `(root) => ({ start: root._modal.trim.start, end: root._modal.trim.end })`);
    expect(after.start).toBeCloseTo(before.start, 1);
    expect(after.end).toBeCloseTo(before.end, 1);
    expect(await zoomOf(page, 'wd-cyc-canvas')).not.toBeNull();   // it zoomed instead
  });

  test('the pinch-abort hook drops an armed trim drag without committing it', async ({ page }) => {
    // _wdAbortEdit protects the case the tolerance cannot: finger 2 landing
    // close enough to a handle that the trim pointerdown re-arms. Called
    // directly, because reaching that state through synthesised touches needs
    // two fingers within ~22 px of each other, which is not a pinch.
    await openCycleModal(page);
    const trimBtn = page.locator('[data-maction="cyc-trim"]').first();
    await expect(trimBtn).toBeVisible({ timeout: 5_000 });
    await trimBtn.click();
    await expect(page.locator('#wd-trim-start')).toBeVisible({ timeout: 5_000 });

    const res = await panelState<{ hooked: boolean; armedBefore: string | null; armedAfter: string | null; trimAfter: number }>(
      page,
      `(root, sr) => {
         const c = sr.getElementById('wd-cyc-canvas');
         const hooked = typeof c._wdAbortEdit === 'function';
         root._modal.trim.start = 123;
         root._modal.drag = 'start';
         const armedBefore = root._modal.drag;
         if (hooked) c._wdAbortEdit();
         return { hooked, armedBefore, armedAfter: root._modal.drag || null,
                  trimAfter: root._modal.trim.start };
       }`,
    );
    expect(res.hooked).toBe(true);
    expect(res.armedBefore).toBe('start');
    expect(res.armedAfter).toBeNull();
    // Abandoned, not committed: the in-progress value is left as it was rather
    // than being snapped and persisted at wherever the pinch centre travelled.
    expect(res.trimAfter).toBe(123);
  });

  test('a pinch does not drag a trim handle', async ({ page }) => {
    await openCycleModal(page);
    const trimBtn = page.locator('[data-maction="cyc-trim"]').first();
    // Assert, do not skip: a wrong selector here used to make all three trim
    // tests report as skipped, i.e. green while validating nothing.
    await expect(trimBtn).toBeVisible({ timeout: 5_000 });
    await trimBtn.click();
    await expect(page.locator('#wd-trim-start')).toBeVisible({ timeout: 5_000 });

    const cdp = await page.context().newCDPSession(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    const before = await panelState<{ start: number; end: number }>(
      page, `(root) => ({ start: root._modal.trim.start, end: root._modal.trim.end })`);
    await pinch(cdp, Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2), 40, 90);
    await page.waitForTimeout(300);
    const after = await panelState<{ start: number; end: number }>(
      page, `(root) => ({ start: root._modal.trim.start, end: root._modal.trim.end })`);
    expect(after.start).toBeCloseTo(before.start, 1);
    expect(after.end).toBeCloseTo(before.end, 1);
    // ...and it zoomed instead.
    expect(await zoomOf(page, 'wd-cyc-canvas')).not.toBeNull();
  });
});

// ─── Mouse behaviour must be unchanged ───────────────────────────────────────

test.describe('mouse', () => {
  test.skip(({ hasTouch }) => !!hasTouch, 'mouse-only behaviour');

  test('wheel still zooms about the cursor', async ({ page }) => {
    await openCycleModal(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -240);
    await page.waitForTimeout(200);
    const zoom = await zoomOf(page, 'wd-cyc-canvas');
    expect(zoom).not.toBeNull();
    expect(zoom!.xMax - zoom!.xMin).toBeLessThan(3600);
  });

  test('wheeling back out clears the zoom entirely', async ({ page }) => {
    await openCycleModal(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -240);
    await page.waitForTimeout(150);
    expect(await zoomOf(page, 'wd-cyc-canvas')).not.toBeNull();
    for (let i = 0; i < 8; i++) await page.mouse.wheel(0, 240);
    await page.waitForTimeout(200);
    expect(await zoomOf(page, 'wd-cyc-canvas')).toBeNull();
  });

  test('the y axis rescales to the zoomed window', async ({ page }) => {
    // The trace peaks mid-cycle; zoom onto the quiet tail and the y scale must
    // follow, instead of staying pinned to an off-screen peak.
    await setHandler(page, 'ha_washdata/get_cycle_power_data', {
      samples: [...Array.from({ length: 60 }, (_, i) => [i * 30, 2000]),
                ...Array.from({ length: 60 }, (_, i) => [1800 + i * 30, 50])],
      full_duration_s: 3600,
      artifacts: [],
      envelope_conformance: null,
    });
    await clickTab(page, 'history');
    await page.locator('tr[data-cid]').first().click();
    await expect(page.locator('#wd-cyc-canvas')).toBeVisible({ timeout: 8_000 });
    const yFull = await panelState<number>(page, `(root, sr) => sr.getElementById('wd-cyc-canvas')._wd.yMax`);
    expect(yFull).toBeGreaterThan(1900);

    await panelState(page, `(root) => root._setCanvasViewport('wd-cyc-canvas', 2400, 3500)`);
    const yTail = await panelState<number>(page, `(root, sr) => sr.getElementById('wd-cyc-canvas')._wd.yMax`);
    expect(yTail).toBeLessThan(200);
  });

  test('dblclick zooms in then resets, and the reset button tracks it', async ({ page }) => {
    await openCycleModal(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    const btn = page.locator('.wd-modal .wd-zoom-reset');
    await expect(btn).not.toBeVisible();
    await page.mouse.dblclick(box.x + box.width / 2, box.y + box.height / 2);
    await page.waitForTimeout(200);
    expect(await zoomOf(page, 'wd-cyc-canvas')).not.toBeNull();
    await expect(btn).toBeVisible();
    await page.mouse.dblclick(box.x + box.width / 2, box.y + box.height / 2);
    await page.waitForTimeout(200);
    expect(await zoomOf(page, 'wd-cyc-canvas')).toBeNull();
    await expect(btn).not.toBeVisible();
  });

  test('the readout is not pinned for mouse hover, and clears on leave', async ({ page }) => {
    await openCycleModal(page);
    const box = await canvasBox(page, 'wd-cyc-canvas');
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.waitForTimeout(200);
    expect(await panelState<string>(page, `(root, sr) => getComputedStyle(sr.querySelector('.wd-gtip')).display`)).not.toBe('none');
    expect(await panelState<boolean>(page, `(root) => root._gtipPinned`)).toBe(false);
    await page.mouse.move(box.x + box.width / 2, box.y - 60);
    await page.waitForTimeout(200);
    expect(await panelState<string>(page, `(root, sr) => getComputedStyle(sr.querySelector('.wd-gtip')).display`)).toBe('none');
  });
});

// ─── Translations ────────────────────────────────────────────────────────────

/**
 * The six new gesture strings must resolve through _t() from a real shipped
 * language file, not just fall back to the English in the JS. Loads de.json for
 * real (the panel fetches /ha_washdata/panel-translations/{lang}.json on demand)
 * and asserts the button tooltip plus both touch hint variants.
 */
test('the #413 gesture strings render from the German panel translation', async ({ page }) => {
  // Boot by hand rather than via bootPanel: that helper registers its own
  // '**/panel-translations/**' route returning {}, and Playwright matches the
  // LAST registered route first, so it would shadow the German one and the test
  // would silently assert the English _t() fallbacks instead.
  await page.route('**/panel-translations/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(de) }));
  await page.goto('/');
  await page.evaluate((h) => { window.__boot_panel(h); }, buildHandlers({}) as never);
  await expect(page.locator('button.wd-tab').first()).toBeVisible({ timeout: 10_000 });
  await page.evaluate(() => window.__freeze_poll());
  await openCycleModal(page);

  // Button tooltip + aria-label come from btn.reset_zoom.
  await panelState(page, `(root) => root._setCanvasViewport('wd-cyc-canvas', 600, 2400)`);
  const btn = page.locator('.wd-modal .wd-zoom-reset');
  await expect(btn).toBeVisible();
  await expect(btn).toHaveAttribute('title', 'Zoom zurücksetzen');
  await expect(btn).toHaveAttribute('aria-label', 'Zoom zurücksetzen');

  // Both hint variants, chosen by pointer type and zoom state.
  const hints = await panelState<string[]>(page, `(root) => [
    root._t('lbl.zoom_hint_touch', {}, 'x'),
    root._t('lbl.zoom_hint_touch_zoomed', {}, 'x'),
    root._t('lbl.tap_to_dismiss', {}, 'x'),
  ]`);
  expect(hints[0]).toBe('mit zwei Fingern zoomen');
  expect(hints[1]).toContain('·');
  expect(hints[1]).toBe('mit zwei Fingern zoomen · Doppeltippen zum Zurücksetzen');
  expect(hints[2]).toBe('zum Ausblenden auf das Diagramm tippen');
});
