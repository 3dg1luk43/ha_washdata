/**
 * Audit UI-20: on a phone the tab strip (619 px of tabs in a 391 px strip) and the
 * Cycles table (707 px in 363 px) overflow sideways with nothing on screen saying
 * so; Duration/Energy/Cost and the Playground/Advanced tabs simply look absent.
 *
 * A horizontal scroller now shows a shadow on whichever edge hides content
 * (.wd-ovf-l / .wd-ovf-r, drawn as a background gradient). Desktop widths overflow
 * nowhere, so they must carry no shadow at all.
 */

import { test, expect, type Locator } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

async function shadowOf(el: Locator): Promise<string> {
  return el.evaluate((n) => getComputedStyle(n).backgroundImage);
}

test.describe('phone', () => {
  test.skip(({ isMobile }) => !isMobile, 'mobile-only layout');

  test('the tab strip shows a shadow on the side that still has tabs', async ({ page }) => {
    await page.goto('/');
    await bootPanel(page);
    const strip = page.locator('.wd-nav .wd-tabs');
    await expect(strip).toHaveClass(/\bwd-ovf-r\b/);
    await expect(strip).not.toHaveClass(/\bwd-ovf-l\b/);
    expect(await shadowOf(strip)).not.toBe('none');

    // Scrolled to the end: the shadow moves to the start edge.
    await strip.evaluate((n) => { n.scrollLeft = n.scrollWidth; });
    await expect(strip).toHaveClass(/\bwd-ovf-l\b/);
    await expect(strip).not.toHaveClass(/\bwd-ovf-r\b/);
  });

  test('the Cycles table shows a shadow where columns are off-screen', async ({ page }) => {
    await page.goto('/');
    await bootPanel(page);
    await clickTab(page, 'history');
    const wrap = page.locator('.wd-pane.active .wd-table-wrap').first();
    await expect(wrap).toBeVisible({ timeout: 8_000 });
    await expect(wrap).toHaveClass(/\bwd-ovf-r\b/);
    expect(await shadowOf(wrap)).not.toBe('none');

    // Halfway: both edges hide content.
    await wrap.evaluate((n) => { n.scrollLeft = (n.scrollWidth - n.clientWidth) / 2; });
    await expect(wrap).toHaveClass(/\bwd-ovf-l\b/);
    await expect(wrap).toHaveClass(/\bwd-ovf-r\b/);
  });
});

test.describe('desktop', () => {
  test.skip(({ isMobile }) => !!isMobile, 'desktop-only layout');

  test('nothing overflows, so there is no shadow', async ({ page }) => {
    await page.goto('/');
    await bootPanel(page);
    const strip = page.locator('.wd-nav .wd-tabs');
    await expect(strip).not.toHaveClass(/\bwd-ovf-[lr]\b/);
    expect(await shadowOf(strip)).toBe('none');

    await clickTab(page, 'history');
    const wrap = page.locator('.wd-pane.active .wd-table-wrap').first();
    await expect(wrap).toBeVisible({ timeout: 8_000 });
    await expect(wrap).not.toHaveClass(/\bwd-ovf-[lr]\b/);
    expect(await shadowOf(wrap)).toBe('none');
  });
});
