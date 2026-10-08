/**
 * Issue #467: a long notify service in a pill picker could not be removed.
 *
 * The pill was a flex box holding a bare text node, which cannot shrink or
 * ellipsize, so a value wider than the box pushed the remove button past the
 * pill's clipped edge. The label is its own shrinkable span now.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import options from '../fixtures/mock-data/options.json';

const LONG = "notify.mobile_app_{{ states('input_text.the_phone_of_whoever_is_home_right_now') }}";

test('a long value keeps its remove button inside the pill and removable', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: { ...options, notify_finish_services: [LONG, 'notify.mobile_app_phone'] }, defaults: {},
    },
  });
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="notifications"]').first().click();
  const box = page.locator('.wd-pillbox[data-opt="notify_finish_services"]');
  await expect(box).toBeVisible({ timeout: 8_000 });

  const pill = box.locator('.wd-pill', { hasText: 'input_text' });
  const x = pill.locator('.wd-pill-x');
  const [p, b] = [await pill.boundingBox(), await x.boundingBox()];
  expect(p && b).toBeTruthy();
  expect(b!.x + b!.width).toBeLessThanOrEqual(p!.x + p!.width + 0.5);
  expect(await pill.getAttribute('title')).toBe(LONG);

  await x.click();
  await expect(box.locator('.wd-pill')).toHaveCount(1);
  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const sent = calls[calls.length - 1].options as Record<string, unknown>;
  expect(sent.notify_finish_services).toEqual(['notify.mobile_app_phone']);
});
