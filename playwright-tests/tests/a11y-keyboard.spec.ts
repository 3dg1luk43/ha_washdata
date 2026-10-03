/**
 * Audit UI-04 / UI-05 (0.5.8): keyboard and screen-reader basics.
 * - a background re-render keeps keyboard focus on the same control;
 * - cycle rows open from the keyboard;
 * - help tips are reachable by keyboard and carry their text as a name;
 * - settings inputs have an accessible name.
 */
import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

const active = (page: any) => page.evaluate(() => {
  const sr = (document.querySelector('ha-washdata-panel') as any).shadowRoot;
  const a = sr.activeElement;
  return a ? { tab: a.dataset.tab || null, cid: a.dataset.cid || null, tag: a.tagName } : null;
});

test('a background re-render keeps focus on the same control', async ({ page }) => {
  await clickTab(page, 'history');
  await page.locator('button.wd-tab[data-tab="profiles"]').focus();
  await page.evaluate(() => (document.querySelector('ha-washdata-panel') as any)._render());
  expect(await active(page)).toMatchObject({ tab: 'profiles' });
});

test('a cycle row opens from the keyboard', async ({ page }) => {
  await clickTab(page, 'history');
  const row = page.locator('tr[data-cid][role="button"]').first();
  await expect(row).toBeVisible({ timeout: 8_000 });
  await row.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('.wd-modal')).toBeVisible({ timeout: 5_000 });
});

test('help tips open on keyboard focus and are named by their text', async ({ page }) => {
  await clickTab(page, 'settings');
  const tip = page.locator('.wd-field .wd-tip').first();
  await expect(tip).toBeAttached({ timeout: 8_000 });
  expect((await tip.getAttribute('aria-label')) || '').not.toBe('');
  await tip.focus();
  await expect(tip.locator('.wd-tip-pop')).toBeVisible();
});

test('settings inputs carry an accessible name', async ({ page }) => {
  await clickTab(page, 'settings');
  const inp = page.locator('input[data-opt="min_power"]').first();
  await expect(inp).toBeVisible({ timeout: 8_000 });
  expect((await inp.getAttribute('aria-label')) || '').toMatch(/power/i);
});
