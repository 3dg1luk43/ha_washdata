/**
 * Issue #454: a per-device accent colour for notifications and Live Activities.
 *
 * The colour is a text field holding the stored value plus a native swatch, because
 * `<input type="color">` cannot express "unset" - it always reports a colour, so it
 * can only ever write into the text box, never be the box. Clearing the text is how
 * the user goes back to the platform default, and only the text box carries
 * `data-opt`, so the swatch is never collected as an option of its own.
 */

import { test, expect, type Page } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import optionsData from '../fixtures/mock-data/options.json';

async function openNotificationColour(page: Page) {
  const sec = page.locator('button[data-sec="notifications"]').first();
  await expect(sec).toBeVisible({ timeout: 8_000 });
  await sec.click();
  const field = page.locator('.wd-colorfield').filter({ has: page.locator('[data-opt="notify_icon_color"]') }).first();
  await expect(field).toBeVisible({ timeout: 8_000 });
  return field;
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await clickTab(page, 'settings');
});

test('the colour field renders a swatch alongside the hex text box', async ({ page }) => {
  const field = await openNotificationColour(page);
  await expect(field.locator('.wd-color-sw')).toHaveCount(1);
  await expect(field.locator('input[data-opt="notify_icon_color"]')).toHaveValue('');
});

test('a typed hex saves as the notify_icon_color option', async ({ page }) => {
  const field = await openNotificationColour(page);
  await field.locator('input[data-opt="notify_icon_color"]').fill('#4CAF50');

  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;
  expect(options.notify_icon_color).toBe('#4CAF50');
});

test('picking from the swatch fills the text box and saves that value', async ({ page }) => {
  const field = await openNotificationColour(page);
  // A native colour dialog cannot be driven, so set the swatch value the way the
  // browser would and fire the same event it fires.
  await field.locator('.wd-color-sw').evaluate((el: HTMLInputElement) => {
    el.value = '#26c6da';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(field.locator('input[data-opt="notify_icon_color"]')).toHaveValue('#26C6DA');

  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;
  expect(options.notify_icon_color).toBe('#26C6DA');
  // The swatch must not become an option of its own.
  expect(Object.keys(options).filter((k) => k.includes('color'))).toEqual(['notify_icon_color']);
});

test('clear puts a configured colour back to the platform default', async ({ page }) => {
  // Saving only sends what changed against the stored options (#447), so the
  // clear has to start from a colour that is actually persisted.
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_options': { options: { ...optionsData, notify_icon_color: '#FF9800' } },
  });
  await clickTab(page, 'settings');

  const field = await openNotificationColour(page);
  const txt = field.locator('input[data-opt="notify_icon_color"]');
  await expect(txt).toHaveValue('#FF9800');
  await field.locator('.wd-color-clear').click();
  await expect(txt).toHaveValue('');

  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;
  expect(options.notify_icon_color).toBe('');
});
