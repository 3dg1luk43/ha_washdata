/**
 * The panel owns the full height it is given and scrolls internally.
 *
 * HA renders a non-iframe custom panel into a plain container with no height of
 * its own, so the panel's old `min-height: 100%` resolved to auto and it grew
 * with its content: the document scrolled, `.wd-main`'s overflow never engaged,
 * and a combobox dropdown opened near the bottom was cut off by the window edge
 * with no way to reach the rest of it (scrolling the page moves the anchor and
 * closes the list).
 */

import { test, expect, type Page } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import constants from '../fixtures/mock-data/constants.json';
import optionsData from '../fixtures/mock-data/options.json';

const PANEL = 'ha-washdata-panel';

async function panelState<T>(page: Page, fn: string): Promise<T> {
  return page.evaluate(`(() => {
    const root = document.querySelector('${PANEL}');
    const sr = root.shadowRoot;
    return (${fn})(root, sr);
  })()`) as Promise<T>;
}

// Enough brands that a 220px list would overflow the space below the input.
const MANY_BRANDS = Array.from({ length: 30 }, (_, i) => ({
  id: `wbrand${i}`, brand: `WBrand ${i}`, status: 'approved',
}));

function storeBoot() {
  return {
    'ha_washdata/get_constants': { ...constants, store_online_available: true, store_online_enabled: true },
    'ha_washdata/get_options': { options: { ...optionsData, store_brand: 'Bosch', store_model: 'WAT28401' } },
    'ha_washdata/store_status': { enabled: true, connected: true, brand: 'Bosch', model: 'WAT28401' },
    'ha_washdata/store_search_devices': { items: [] },
    'ha_washdata/store_get_catalog_entry': { device_id: 'washer__bosch__wat28401' },
    'ha_washdata/store_list_brands': { items: MANY_BRANDS },
  };
}

test('the panel fits the viewport instead of growing the document', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {});
  // Overview is the tallest default tab; whatever it contains, the host must not
  // be taller than the window.
  const m = await panelState<{ host: number; win: number; docScroll: number }>(
    page,
    `(root) => ({
      host: Math.round(root.getBoundingClientRect().height),
      win: window.innerHeight,
      docScroll: document.scrollingElement.scrollHeight - document.scrollingElement.clientHeight,
    })`,
  );
  expect(m.host).toBeLessThanOrEqual(m.win + 1);
  expect(m.docScroll).toBeLessThanOrEqual(1);
});

test('.wd-main is the scroller, on every tab', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {});
  for (const tab of ['status', 'history', 'profiles', 'settings', 'advanced']) {
    await clickTab(page, tab);
    const m = await panelState<{ declared: string; bounded: boolean; docScroll: number }>(
      page,
      `(root, sr) => {
        const main = sr.querySelector('.wd-main');
        const doc = document.scrollingElement;
        return {
          declared: getComputedStyle(main).overflowY,
          // The declaration alone proves nothing: while the flex chain was broken
          // .wd-main still said overflow-y:auto but was SIZED TO ITS CONTENT, so it
          // never overflowed (no scrollbar) and spilled out of the clipped host
          // instead. What matters is that the scroller is bounded by the host.
          bounded: main.clientHeight <= Math.ceil(root.getBoundingClientRect().height),
          docScroll: doc.scrollHeight - doc.clientHeight,
        };
      }`,
    );
    expect(m.declared, `tab ${tab}`).toBe('auto');
    expect(m.bounded, `tab ${tab}: .wd-main must be bounded by the host, not content-sized`).toBe(true);
    expect(m.docScroll, `tab ${tab}: the document must not scroll`).toBeLessThanOrEqual(1);
  }
});

test('a dropdown near the bottom is sized to the room it has, not cut off', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, storeBoot());
  await clickTab(page, 'store');
  const box = page.locator('#wd-store-q');
  await expect(box).toBeVisible({ timeout: 8_000 });

  // Push the input as close to the bottom of the scroller as it will go, so the
  // space below it is genuinely small.
  await panelState(page, `(root, sr) => { sr.getElementById('wd-store-q').scrollIntoView({ block: 'end' }); }`);
  await box.click();
  await box.fill('');
  await box.pressSequentially('w', { delay: 60 });

  const drop = page.locator('.wd-store-search .wd-combo-drop');
  await expect(drop.locator('.wd-combo-item').first()).toBeVisible({ timeout: 8_000 });

  const fit = await panelState<{ dropBottom: number; dropTop: number; clipBottom: number; clipTop: number }>(
    page,
    `(root, sr) => {
      const d = sr.querySelector('.wd-store-search .wd-combo-drop').getBoundingClientRect();
      const main = sr.querySelector('.wd-main').getBoundingClientRect();
      return {
        dropBottom: d.bottom, dropTop: d.top,
        clipBottom: Math.min(main.bottom, window.innerHeight),
        clipTop: Math.max(main.top, 0),
      };
    }`,
  );
  // 1px of rounding slack; the point is that it is not hanging off the edge.
  expect(fit.dropBottom).toBeLessThanOrEqual(fit.clipBottom + 1);
  expect(fit.dropTop).toBeGreaterThanOrEqual(fit.clipTop - 1);
});

test('the bottom of a long tab can be scrolled to', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {});
  await clickTab(page, 'settings');
  // Force the condition the bug needs: content taller than the window. A short
  // viewport does that for any tab, and is itself a case users hit (laptop, landscape
  // phone) rather than a contrivance.
  await page.setViewportSize({ width: 1000, height: 420 });
  await page.waitForTimeout(150);
  const reach = await panelState<{ overflow: number; strandedPx: number; scrolled: number }>(
    page,
    `(root, sr) => {
      const main = sr.querySelector('.wd-main');
      main.scrollTop = main.scrollHeight;      // scroll as far as it will go
      const body = sr.querySelector('.wd-body');
      const mainRect = main.getBoundingClientRect();
      return {
        overflow: main.scrollHeight - main.clientHeight,
        // With the scroller at its end, the content's last pixel must be visible:
        // both inside the scroller AND inside the window. A panel taller than the
        // window strands the difference where no amount of scrolling reaches it.
        strandedPx: Math.round(Math.max(
          body.getBoundingClientRect().bottom - mainRect.bottom,
          mainRect.bottom - window.innerHeight,
        )),
        scrolled: main.scrollTop,
      };
    }`,
  );
  expect(reach.overflow, 'the settings tab should be long enough to test this').toBeGreaterThan(0);
  expect(reach.scrolled).toBeGreaterThan(0);
  expect(reach.strandedPx).toBeLessThanOrEqual(1);
});

test('the panel is sized to the space it has, not to the viewport', async ({ page }) => {
  // Reproduces HA's container: a plain block wrapper with no height, offset down the
  // page. Assuming 100dvh there overhangs the window by the offset, and because the
  // host clips its overflow the last 40px of the scroller are unreachable.
  await page.goto('/');
  await bootPanel(page, {});
  const fits = await page.evaluate(() => {
    const root = document.querySelector('ha-washdata-panel') as HTMLElement;
    const box = document.createElement('div');
    box.style.paddingTop = '40px';               // like ha-panel-custom's safe-area pad
    document.body.insertBefore(box, root);
    box.appendChild(root);
    window.dispatchEvent(new Event('resize'));
    return new Promise<number>((res) =>
      requestAnimationFrame(() =>
        res(Math.round(root.getBoundingClientRect().bottom - window.innerHeight))));
  });
  expect(fits).toBeLessThanOrEqual(1);
});

test('the device list and tabs stay visible while the pane scrolls', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {});
  await clickTab(page, 'settings');
  await page.setViewportSize({ width: 1000, height: 420 });
  await page.waitForTimeout(150);
  const pinned = await panelState<{ navTopBefore: number; navTopAfter: number; scrolled: number }>(
    page,
    `(root, sr) => {
      const main = sr.querySelector('.wd-main');
      const nav = sr.querySelector('.wd-nav');
      const mainTop = () => main.getBoundingClientRect().top;
      const navTopBefore = Math.round(nav.getBoundingClientRect().top - mainTop());
      main.scrollTop = main.scrollHeight;
      const navTopAfter = Math.round(nav.getBoundingClientRect().top - mainTop());
      return { navTopBefore, navTopAfter, scrolled: main.scrollTop };
    }`,
  );
  expect(pinned.scrolled).toBeGreaterThan(0);
  // Pinned to the top of the scrollport, i.e. it did not move with the content.
  expect(pinned.navTopAfter).toBe(pinned.navTopBefore);
  expect(pinned.navTopAfter).toBeLessThanOrEqual(1);
});
