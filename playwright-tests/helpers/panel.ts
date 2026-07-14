/**
 * High-level helpers for WashData panel tests.
 * All helpers operate on a Playwright Page and return locators or values.
 */

import { Page, Locator, expect } from '@playwright/test';
import { buildHandlers } from './ws-handlers';

export type Handlers = Record<string, unknown>;

/**
 * Boot the WashData panel on the given page with the provided WS handlers.
 * Waits for the initial render (tab bar visible) before returning.
 *
 * The page must already be navigated to the fixture HTML (http://localhost:4567/).
 *
 * @param page       Playwright page
 * @param overrides  WS handler overrides merged on top of DEFAULT_HANDLERS
 */
export async function bootPanel(page: Page, overrides: Handlers = {}): Promise<void> {
  // Intercept the per-language translation fetches so tests don't need network.
  // Returning an empty dict makes _t() fall back to the JS-embedded English.
  await page.route('**/panel-translations/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }),
  );

  const handlers = buildHandlers(overrides);

  await page.evaluate((h: Handlers) => {
    // Serialise functions aren't transferable; only plain data reaches page scope.
    window.__boot_panel(h);
  }, handlers as any);

  // Wait for the tab bar to appear — confirms initial render completed.
  await expect(page.locator('button.wd-tab').first()).toBeVisible({ timeout: 10_000 });

  // Freeze the polling loop so background polls don't interfere with assertions.
  await page.evaluate(() => window.__freeze_poll());
}

/** Click a top-level tab by its data-tab attribute value. */
export async function clickTab(page: Page, tabId: string): Promise<void> {
  const btn = page.locator(`button.wd-tab[data-tab="${tabId}"]`);
  await expect(btn).toBeVisible({ timeout: 5_000 });
  await btn.click();
  // Wait for the pane to become active.
  await expect(page.locator(`.wd-pane[data-pane="${tabId}"].active, .wd-pane.active`)).toBeVisible({
    timeout: 5_000,
  });
}

/** Wait for a specific tab's pane to be active. */
export async function waitForTab(page: Page, tabId: string): Promise<void> {
  await expect(page.locator(`button.wd-tab[data-tab="${tabId}"].active`)).toBeVisible({ timeout: 5_000 });
}

/** Return the open modal element, or throw if none. */
export function getModal(page: Page): Locator {
  return page.locator('.wd-modal');
}

/**
 * Navigate to the ML Training subtab. ML Training is no longer a top-level tab;
 * it is a subtab nested under the Advanced tab (data-ptab="ml").
 */
export async function openMlTab(page: Page): Promise<void> {
  await clickTab(page, 'advanced');
  const sub = page.locator('button.wd-subtab[data-ptab="ml"]');
  await expect(sub).toBeVisible({ timeout: 5_000 });
  await sub.click();
}

/** Click a subtab button by its data attribute (data-ptab or data-pgtab). */
export async function clickSubtab(page: Page, attr: string, value: string): Promise<void> {
  const btn = page.locator(`[${attr}="${value}"]`);
  await expect(btn).toBeVisible({ timeout: 5_000 });
  await btn.click();
}

/**
 * Assert that a WS command was called at least once.
 * Returns the list of calls matching the type.
 */
export async function assertWsCalled(
  page: Page,
  type: string,
  minTimes = 1,
): Promise<Record<string, unknown>[]> {
  const calls = await page.evaluate((t: string) => window.__get_calls(t), type);
  expect(calls.length, `Expected ${type} to be called at least ${minTimes}×, got ${calls.length}`).toBeGreaterThanOrEqual(minTimes);
  return calls as any;
}

/**
 * Assert that a WS command was NOT called.
 */
export async function assertWsNotCalled(page: Page, type: string): Promise<void> {
  const calls = await page.evaluate((t: string) => window.__get_calls(t), type);
  expect(calls.length, `Expected ${type} NOT to be called, but it was called ${calls.length}×`).toBe(0);
}

/**
 * Set a WS handler at runtime (after boot) so subsequent calls return new data.
 * Useful for simulating state changes.
 */
export async function setHandler(
  page: Page,
  type: string,
  data: unknown,
): Promise<void> {
  await page.evaluate(
    ({ t, d }: { t: string; d: unknown }) => window.__set_handler(t, d),
    { t: type, d: data } as any,
  );
}

/** Show a toast / trigger a toast notification in the panel via evaluate. */
export async function showToast(page: Page, message: string, kind = 'info'): Promise<void> {
  await page.evaluate(
    ({ m, k }: { m: string; k: string }) => {
      const el = document.getElementById('wd-panel') as any;
      if (el && el._showToast) el._showToast(m, k);
    },
    { m: message, k: kind } as any,
  );
}

/** Dismiss a confirm-style modal by clicking its primary confirm button. */
export async function confirmModal(page: Page): Promise<void> {
  const modal = getModal(page);
  await expect(modal).toBeVisible({ timeout: 3_000 });
  const confirmBtn = modal.locator('button.wd-btn-primary, button[data-action="modal-ok"]').first();
  await confirmBtn.click();
}

/** Close a modal by clicking the × / Cancel button. */
export async function closeModal(page: Page): Promise<void> {
  const modal = getModal(page);
  await expect(modal).toBeVisible({ timeout: 3_000 });
  const closeBtn = modal.locator('button[data-action="modal-close"], button.wd-modal-close').first();
  await closeBtn.click();
}
