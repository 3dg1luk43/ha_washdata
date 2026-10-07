/**
 * Share modal during a store outage (audit STORE-09, PR #466 round 15).
 *
 * store_get_device_profiles answers {items: [], error: "store_unreachable"} when
 * the store cannot be reached. The modal used to read only `items`, so an outage
 * showed "(no profiles yet - add one)", as if the appliance had none.
 */

import { test, expect } from '@playwright/test';
import { bootPanel } from '../helpers/panel';
import options from '../fixtures/mock-data/options.json';

async function openShare(page: any, reply: Record<string, unknown>) {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: { ...options, store_brand: 'Bosch', store_model: 'WAT' }, defaults: {},
    },
    'ha_washdata/store_get_device_profiles': reply,
  });
  await page.evaluate(async () => {
    const el = document.querySelector('ha-washdata-panel') as any;
    el._modal = { type: 'store-share', cycleId: 'c1', program: '', profiles: null, deviceId: null };
    el._render();
    await el._loadShareProfiles();
  });
  const modal = page.locator('.wd-modal');
  await expect(modal).toBeVisible({ timeout: 8_000 });
  return modal;
}

test('a store outage says so instead of "no profiles yet"', async ({ page }) => {
  const modal = await openShare(page, { items: [], error: 'store_unreachable' });
  await expect(modal.locator('.wd-error-state')).toBeVisible();
  await expect(modal).not.toContainText('no profiles yet');
});

test('a reachable store with no profiles still offers to add one', async ({ page }) => {
  const modal = await openShare(page, { items: [], device_id: null });
  await expect(modal.locator('.wd-error-state')).toHaveCount(0);
  await expect(modal).toContainText('no profiles yet');
});
