/**
 * A re-render must not throw the user back to the top (#443, #449).
 *
 * `_render()` replaces `this._container.innerHTML` wholesale. The element that
 * actually scrolls - `.wd-main`, since the panel scrolls internally rather than
 * letting the document scroll - lives inside that container, so the swap destroys
 * it and the replacement starts at scrollTop 0. Focus was already carried across
 * the swap; scroll was not.
 *
 * #443's reporter hit it in the Playground, which re-renders on every input.
 * #449's hit it on the Overview, which re-renders itself every few seconds while a
 * cycle is running - no input needed, so the page simply would not stay put. The
 * three groups below are the three questions the fix answers: which containers are
 * carried, which re-renders are navigation, and what happens to the position while
 * a tab is showing its loading placeholder.
 */

import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import deviceRunning from '../fixtures/mock-data/device-running.json';

/** A running cycle with a long enough trace to draw, as on the reporter's Overview. */
const RUNNING_POWER_HISTORY = {
  live: Array.from({ length: 300 }, (_, i) => [i * 30, 200 + 400 * Math.sin(i / 9)]),
  raw: [],
  cycle_active: true,
  cycle_elapsed_s: 9000,
  profile_envelope: null,
};

const MANY_LOGS = {
  logs: Array.from({ length: 300 }, (_, i) => ({
    ts: Date.now() / 1000 - i,
    level: 'INFO',
    msg: `line ${i} lorem ipsum dolor sit amet consectetur`,
    device: 'Test Washer',
    component: 'manager',
  })),
};

/** scrollTop/scrollLeft of the first element matching `sel`, or null if absent. */
async function offsetOf(page: Page, sel: string): Promise<{ top: number; left: number } | null> {
  return page.evaluate((s) => {
    const el = document.querySelector('ha-washdata-panel')?.shadowRoot?.querySelector(s) as
      | HTMLElement
      | null;
    return el ? { top: el.scrollTop, left: el.scrollLeft } : null;
  }, sel);
}

async function setOffset(page: Page, sel: string, pos: { top?: number; left?: number }): Promise<void> {
  await page.evaluate(({ s, p }) => {
    const el = document.querySelector('ha-washdata-panel')?.shadowRoot?.querySelector(s) as
      | HTMLElement
      | null;
    if (!el) throw new Error(`no ${s}`);
    if (p.top != null) el.scrollTop = p.top;
    if (p.left != null) el.scrollLeft = p.left;
  }, { s: sel, p: pos });
}

/** Run one poll the way the 5s timer and the hass-update throttle both do. */
async function poll(page: Page): Promise<void> {
  await page.evaluate(async () => {
    await (document.getElementById('wd-panel') as unknown as { _fetchAll: () => Promise<void> })._fetchAll();
  });
}

/** Boot on the Overview with a cycle running - the #449 reporter's screen. */
async function bootRunning(page: Page): Promise<void> {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_devices': deviceRunning,
    'ha_washdata/get_power_history': RUNNING_POWER_HISTORY,
    'ha_washdata/get_logs': MANY_LOGS,
  });
  await clickTab(page, 'status');
}



async function scrollerTop(page: Page): Promise<number> {
  return page.evaluate(() => {
    const panel = document.querySelector('ha-washdata-panel');
    const main = panel?.shadowRoot?.querySelector('.wd-main') as HTMLElement | null;
    return main ? main.scrollTop : -1;
  });
}

async function scrollMainTo(page: Page, top: number): Promise<void> {
  await page.evaluate((y) => {
    const panel = document.querySelector('ha-washdata-panel');
    const main = panel?.shadowRoot?.querySelector('.wd-main') as HTMLElement | null;
    if (main) main.scrollTop = y;
  }, top);
}

/**
 * A small viewport, so .wd-main overflows for real. An inline max-height would be
 * wiped by the very innerHTML swap under test, changing clientHeight mid-run and
 * making "scrolled to N" mean something different before and after.
 */
const SMALL_VIEWPORT = { width: 520, height: 400 };

test.use({ viewport: SMALL_VIEWPORT });

async function assertScrollable(page: Page): Promise<void> {
  const room = await page.evaluate(() => {
    const panel = document.querySelector('ha-washdata-panel');
    const main = panel?.shadowRoot?.querySelector('.wd-main') as HTMLElement | null;
    return main ? main.scrollHeight - main.clientHeight : 0;
  });
  if (room < 400) throw new Error(`.wd-main is not scrollable enough (room ${room}px)`);
}

test('a re-render keeps the scroll position', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await clickTab(page, 'settings');
  await assertScrollable(page);
  await scrollMainTo(page, 300);
  expect(await scrollerTop(page)).toBeGreaterThan(0);
  const before = await scrollerTop(page);

  // Any re-render: the panel re-renders itself on demand the same way a control
  // change does.
  await page.evaluate(() => {
    const panel = document.querySelector('ha-washdata-panel') as unknown as {
      _render: () => void;
    };
    panel._render();
  });

  await expect
    .poll(async () => scrollerTop(page), { timeout: 3_000 })
    .toBe(before);
});

test('the Playground stays put when a param is typed into (#443, reported case)', async ({
  page,
}) => {
  await page.goto('/');
  await bootPanel(page, {});
  await clickTab(page, 'playground');
  await page.waitForSelector('input[data-pgkey]');
  await assertScrollable(page);

  // Put a Playground param on screen the way the user would: scroll down to it.
  // (Not via locator.fill(), which scrolls the element into view itself and would
  // move the very position under test before the typing ever happens.)
  const before = await page.evaluate(() => {
    const panel = document.querySelector('ha-washdata-panel');
    const sr = panel!.shadowRoot!;
    const main = sr.querySelector('.wd-main') as HTMLElement;
    const inp = sr.querySelector('input[data-pgkey]') as HTMLInputElement;
    inp.scrollIntoView({ block: 'center' });
    return main.scrollTop;
  });
  expect(before).toBeGreaterThan(0);

  // Type one character, exactly as the panel's own handler sees it.
  await page.evaluate(() => {
    const sr = document.querySelector('ha-washdata-panel')!.shadowRoot!;
    const inp = sr.querySelector('input[data-pgkey]') as HTMLInputElement;
    inp.focus();
    inp.value = '7';
    inp.dispatchEvent(new Event('input', { bubbles: true }));
  });

  await expect.poll(async () => scrollerTop(page), { timeout: 3_000 }).toBe(before);
});

test('scroll is clamped, not restored past the end of a shorter tree', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await clickTab(page, 'settings');
  await assertScrollable(page);
  await scrollMainTo(page, 400);

  // Shrink the content, then re-render: the saved offset no longer exists.
  await page.evaluate(() => {
    const panel = document.querySelector('ha-washdata-panel') as unknown as {
      _render: () => void;
      _settingsSection?: string;
    };
    panel._render();
  });

  const top = await scrollerTop(page);
  const max = await page.evaluate(() => {
    const panel = document.querySelector('ha-washdata-panel');
    const main = panel?.shadowRoot?.querySelector('.wd-main') as HTMLElement | null;
    return main ? main.scrollHeight - main.clientHeight : 0;
  });
  expect(top).toBeLessThanOrEqual(max + 1);
  expect(top).toBeGreaterThanOrEqual(0);
});

// ── #449: the Overview refreshes itself, so nothing the user does is involved ──

test('the Overview stays put while a cycle is running (#449, reported case)', async ({ page }) => {
  await bootRunning(page);
  await assertScrollable(page);
  await scrollMainTo(page, 250);
  const before = await scrollerTop(page);

  // Three polls, because the reporter's complaint is that it keeps happening: one
  // refresh that held its position would still leave the page unusable if the next
  // one did not.
  for (let i = 0; i < 3; i++) {
    await poll(page);
    expect(await scrollerTop(page), `poll ${i + 1}`).toBe(before);
  }
});

test('a poll keeps its position while the live trace grows', async ({ page }) => {
  await bootRunning(page);
  await assertScrollable(page);
  await scrollMainTo(page, 250);
  const before = await scrollerTop(page);

  // A running cycle's graph gets longer on every poll, which is the refresh the
  // reporter was watching when the page jumped.
  await page.evaluate(() => {
    window.__set_handler('ha_washdata/get_power_history', {
      live: Array.from({ length: 600 }, (_, i) => [i * 30, 200 + 400 * Math.sin(i / 9)]),
      raw: [],
      cycle_active: true,
      cycle_elapsed_s: 18_000,
      profile_envelope: null,
    });
  });
  await poll(page);

  expect(await scrollerTop(page)).toBe(before);
});

// ── Every scroller the swap destroys, not just the page ──

test('the horizontal tab strip keeps its position across a poll', async ({ page }) => {
  await bootRunning(page);
  // The strip overflows on a narrow screen; scrolling it is how the later tabs are
  // reached at all, so snapping it back every few seconds hides them again.
  await setOffset(page, '.wd-tabs', { left: 100 });
  expect((await offsetOf(page, '.wd-tabs'))!.left).toBeGreaterThan(0);
  const before = await offsetOf(page, '.wd-tabs');

  await poll(page);

  expect(await offsetOf(page, '.wd-tabs')).toEqual(before);
});

test('the log drawer keeps its position across a poll', async ({ page }) => {
  await bootRunning(page);
  // The drawer refreshes on the same 5s poll, which is exactly when reading a log
  // is least tolerant of being sent back to the first line.
  await page.evaluate(async () => {
    const el = document.getElementById('wd-panel') as unknown as {
      _logOpen: boolean; _fetchLogs: () => Promise<void>; _render: () => void;
    };
    el._logOpen = true;
    await el._fetchLogs();
    el._render();
  });
  await setOffset(page, '.wd-logs', { top: 300 });
  const before = await offsetOf(page, '.wd-logs');
  expect(before!.top).toBe(300);

  await poll(page);

  expect(await offsetOf(page, '.wd-logs')).toEqual(before);
});

// ── The loading placeholder must not eat the position ──

test('a tab re-fetch keeps its position across the loading placeholder', async ({ page }) => {
  await bootRunning(page);
  await assertScrollable(page);
  await scrollMainTo(page, 250);
  const before = await scrollerTop(page);

  // _fetchTabData renders a spinner a few dozen pixels tall before the fetch and the
  // real content after it. Clamping to that intermediate tree - which is shorter than
  // both the tree it replaced and the one that replaces it - would drop the position
  // on the floor, and it stays dropped: the next render captures the clamped 0.
  await page.evaluate(async () => {
    await (document.getElementById('wd-panel') as unknown as {
      _fetchTabData: () => Promise<void>;
    })._fetchTabData();
  });

  expect(await scrollerTop(page)).toBe(before);
});

test('a poll landing during the loading placeholder keeps its position', async ({ page }) => {
  await bootRunning(page);
  await assertScrollable(page);
  await scrollMainTo(page, 250);
  const before = await scrollerTop(page);

  await page.evaluate(async () => {
    const el = document.getElementById('wd-panel') as unknown as {
      _fetchTabData: () => Promise<void>; _fetchAll: () => Promise<void>;
    };
    const pending = el._fetchTabData();
    await el._fetchAll();     // the 5s poll does not wait for the tab fetch
    await pending;
  });

  expect(await scrollerTop(page)).toBe(before);
});

// ── Navigation is the one thing that outranks preserving ──

test('changing tab lands at the top, but does not move the tab strip', async ({ page }) => {
  await bootRunning(page);
  await assertScrollable(page);
  await scrollMainTo(page, 250);
  await setOffset(page, '.wd-tabs', { left: 100 });
  const strip = await offsetOf(page, '.wd-tabs');

  await clickTab(page, 'settings');

  expect(await scrollerTop(page)).toBe(0);
  // Scrolling the strip to reach a tab and then having it snap back is how the tab
  // just pressed ends up off-screen.
  expect(await offsetOf(page, '.wd-tabs')).toEqual(strip);
});

test('changing subtab or settings section lands at the top', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_devices': deviceRunning });

  await clickTab(page, 'advanced');
  await scrollMainTo(page, 200);
  await page.evaluate(() => {
    const sr = document.querySelector('ha-washdata-panel')!.shadowRoot!;
    (sr.querySelector('[data-ptab="ml"]') as HTMLElement).click();
  });
  await expect.poll(async () => scrollerTop(page), { timeout: 3_000 }).toBe(0);

  await clickTab(page, 'settings');
  await assertScrollable(page);
  await scrollMainTo(page, 250);
  await page.evaluate(() => {
    const sr = document.querySelector('ha-washdata-panel')!.shadowRoot!;
    const inactive = (Array.from(sr.querySelectorAll('[data-sec]')) as HTMLElement[])
      .find((b) => !b.classList.contains('active'));
    inactive!.click();
  });
  await expect.poll(async () => scrollerTop(page), { timeout: 3_000 }).toBe(0);
});

test('switching device lands at the top', async ({ page }) => {
  await bootRunning(page);
  await assertScrollable(page);
  await scrollMainTo(page, 250);

  await page.evaluate(() => {
    const el = document.getElementById('wd-panel') as unknown as {
      _selIdx: number; _render: () => void;
    };
    el._selIdx = 1;
    el._render();
  });

  expect(await scrollerTop(page)).toBe(0);
});

test('a background batch finishing is not navigation', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_devices': deviceRunning });
  await clickTab(page, 'playground');
  await assertScrollable(page);
  await scrollMainTo(page, 200);
  const before = await scrollerTop(page);

  // The Playground flips its own analysis drawer when a history/sweep batch lands.
  // That is the panel talking to itself, not the user asking to be elsewhere - and
  // treating it as navigation would reintroduce #449 on a tab that can run for
  // minutes.
  await page.evaluate(() => {
    const el = document.getElementById('wd-panel') as unknown as {
      _pgAnalysisTab: string; _render: () => void;
    };
    el._pgAnalysisTab = el._pgAnalysisTab === 'sweep' ? 'history' : 'sweep';
    el._render();
  });

  expect(await scrollerTop(page)).toBe(before);
});
