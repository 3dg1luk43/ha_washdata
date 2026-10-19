/**
 * #473: on a phone (HA iOS app) the Logs drawer could not be closed. Below 680px it
 * was position:fixed at top:0 with a z-index above the header, so it covered the
 * header (burger, gear, Logs toggle) and put its own close button under the status
 * bar, where iOS takes the taps. Its open state was also restored from localStorage,
 * so the full-screen drawer came back on every visit.
 */

import { test, expect, Page } from '@playwright/test';
import { bootPanel } from '../helpers/panel';

test.use({ viewport: { width: 390, height: 844 } });

const SAFE_AREA_PX = 47;

/** Wrap the panel like ha-panel-custom does on a notched phone (safe-area padding). */
async function wrapInSafeArea(page: Page) {
  await page.evaluate((pad) => {
    const root = document.querySelector('ha-washdata-panel') as HTMLElement;
    const box = document.createElement('div');
    box.style.paddingTop = `${pad}px`;
    document.body.insertBefore(box, root);
    box.appendChild(root);
    window.dispatchEvent(new Event('resize'));
  }, SAFE_AREA_PX);
}

async function openDrawer(page: Page) {
  await page.locator('.wd-header [data-action="toggle-log-drawer"]').click();
  await expect(page.locator('.wd-log-drawer.open')).toBeVisible({ timeout: 5_000 });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.removeItem('wd-log-open'));
});

test('the header stays usable while the drawer is open', async ({ page }) => {
  await bootPanel(page);
  await openDrawer(page);
  // click() refuses when another element sits on top, which the old fixed drawer did.
  await page.locator('#wd-burger').click({ timeout: 3_000 });
  await page.locator('.wd-header [data-action="toggle-log-drawer"]').click({ timeout: 3_000 });
  await expect(page.locator('.wd-log-drawer.open')).toHaveCount(0);
});

test('the close button sits below the header and the safe area, and closes the drawer', async ({ page }) => {
  await bootPanel(page);
  await wrapInSafeArea(page);
  await openDrawer(page);
  const close = page.locator('.wd-log-close-btn');
  const box = await close.boundingBox();
  const header = await page.locator('.wd-header').boundingBox();
  expect(box).not.toBeNull();
  expect(header).not.toBeNull();
  expect(box!.y).toBeGreaterThanOrEqual(SAFE_AREA_PX);
  expect(box!.y).toBeGreaterThanOrEqual(header!.y + header!.height - 1);
  expect(box!.y + box!.height).toBeLessThanOrEqual(844);
  expect(box!.height).toBeGreaterThanOrEqual(44);
  await close.click({ timeout: 3_000 });
  await expect(page.locator('.wd-log-drawer.open')).toHaveCount(0);
});

test('a saved open drawer does not reopen on a phone', async ({ page }) => {
  await page.evaluate(() => localStorage.setItem('wd-log-open', '1'));
  await bootPanel(page);
  await expect(page.locator('.wd-header [data-action="toggle-log-drawer"]')).toBeVisible();
  await expect(page.locator('.wd-log-drawer.open')).toHaveCount(0);
});

test('Escape closes the drawer', async ({ page }) => {
  await bootPanel(page);
  await openDrawer(page);
  await page.locator('.wd-log-close-btn').focus();
  await page.keyboard.press('Escape');
  await expect(page.locator('.wd-log-drawer.open')).toHaveCount(0);
});
