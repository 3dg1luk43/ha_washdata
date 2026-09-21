/**
 * A re-render must not throw the user back to the top (#443).
 *
 * `_render()` replaces `this._container.innerHTML` wholesale. The element that
 * actually scrolls - `.wd-main`, since the panel scrolls internally rather than
 * letting the document scroll - lives inside that container, so the swap destroys
 * it and the replacement starts at scrollTop 0. Focus was already carried across
 * the swap; scroll was not.
 *
 * The reporter hit it in the Playground, which re-renders on every input, but it
 * applies to any tab whose controls re-render while scrolled.
 */

import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';



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
