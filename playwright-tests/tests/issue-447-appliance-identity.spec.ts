/**
 * #447 - Brand/model shown in Device Info but the store reports "no appliance declared".
 *
 * The store-backed brand/model pickers wrote the typed value straight into
 * `this._opts`, the panel's copy of the SAVED options. Since 0.5.6 the save
 * collector diffs the form against `_opts` (`_changedOptions`, #406), so the key
 * looked unchanged and was dropped from the `set_options` payload - the save
 * reported "No changes to save" and never called the backend at all. The panel
 * then showed an appliance the integration had never stored, and every upload
 * failed with `no_appliance_declared`.
 *
 * The contribute popup had the same defect from the other end: creating a brand
 * or appliance on the store site patched `_opts` locally and never persisted it.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import constants from '../fixtures/mock-data/constants.json';
import optionsData from '../fixtures/mock-data/options.json';

// The store web origin must be the page's OWN origin: the contribute popup replies with
// postMessage, which drops a message whose targetOrigin does not match, and --e2e-min
// serves the panel from a different port than the readable build.
function storeConstants(origin: string) {
  return {
    ...constants,
    store_online_available: true,
    store_online_enabled: true,
    store_web_origin: origin,
  };
}

// An undeclared device - the state a user starts from.
function handlers(origin: string, extra: Record<string, unknown> = {}) {
  return {
    'ha_washdata/get_constants': storeConstants(origin),
    'ha_washdata/get_options': { options: { ...optionsData, store_brand: '', store_model: '' } },
    'ha_washdata/store_status': { enabled: true, connected: true, uid: 'gh_1', name: 'octocat' },
    'ha_washdata/store_list_brands': { items: [{ id: 'bosch', brand: 'Bosch', status: 'approved' }] },
    'ha_washdata/store_search_devices': { items: [] },
    'ha_washdata/store_get_catalog_entry': { device_id: null, brand: null, device: null },
    ...extra,
  };
}

const pageOrigin = (page: import('@playwright/test').Page) =>
  page.evaluate(() => location.origin);

async function declareAppliance(page: import('@playwright/test').Page) {
  const brand = page.locator('#wd-store-brand');
  await expect(brand).toBeVisible({ timeout: 8_000 });
  await brand.fill('Bosch');
  await brand.blur();          // the picker commits on 'change', i.e. blur/datalist pick
  const model = page.locator('#wd-store-model');
  await expect(model).toBeEnabled({ timeout: 8_000 });
  await model.fill('WAE28220');
  await model.blur();
}

test('typing brand + model in Device Info persists them via set_options', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, handlers(await pageOrigin(page)));
  await clickTab(page, 'settings');
  await declareAppliance(page);

  await page.locator('#wd-settings-save').first().click();

  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;
  expect(options.store_brand).toBe('Bosch');
  expect(options.store_model).toBe('WAE28220');
});

test('an unsaved appliance edit survives a settings section switch', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, handlers(await pageOrigin(page)));
  await clickTab(page, 'settings');
  await declareAppliance(page);

  // Leaving and re-entering the section re-renders the form from _opts +
  // _pendingSettings; the edit must come back with it, not be silently dropped.
  await page.locator('[data-sec="notifications"]').first().click();
  await page.locator('[data-sec="basic"]').first().click();

  await expect(page.locator('#wd-store-brand')).toHaveValue('Bosch');
  await expect(page.locator('#wd-store-model')).toHaveValue('WAE28220');
});

test('the share buttons stay hidden until the appliance is actually saved', async ({ page }) => {
  // _storeDeviceDeclared() must track the BACKEND's options: the upload handlers read
  // entry.options, so offering "Share this device" on an unsaved edit only produces
  // the no_appliance_declared toast this issue is about.
  await page.goto('/');
  await bootPanel(page, handlers(await pageOrigin(page)));
  await clickTab(page, 'settings');
  await declareAppliance(page);

  await expect(page.locator('[data-action="store-share-device"]')).toHaveCount(0);

  await page.locator('#wd-settings-save').first().click();
  await assertWsCalled(page, 'ha_washdata/set_options');
  await expect(page.locator('[data-action="store-share-device"]')).toHaveCount(1);
});

test('an appliance created in the contribute popup is persisted to the entry', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, handlers(await pageOrigin(page)));
  await clickTab(page, 'settings');

  // The "add appliance" + button lives on the model picker, which only renders once
  // a brand is picked.
  const brand = page.locator('#wd-store-brand');
  await expect(brand).toBeVisible({ timeout: 8_000 });
  await brand.fill('Bosch');
  await brand.blur();

  // Arm the popup listener the same way clicking "+" does, then replay the message
  // the store's create.html posts back.
  await page.locator('[data-action="store-add-appliance"]').first().click();
  await page.evaluate(() => {
    window.postMessage({ type: 'washdata-device-created', brand: 'Bosch', model: 'WAE28220' }, location.origin);
  });

  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;
  expect(options.store_brand).toBe('Bosch');
  expect(options.store_model).toBe('WAE28220');
});

test('a brand created in the contribute popup is persisted to the entry', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, handlers(await pageOrigin(page)));
  await clickTab(page, 'settings');

  await page.locator('[data-action="store-add-brand"]').first().click();
  await page.evaluate(() => {
    window.postMessage({ type: 'washdata-brand-created', brand: 'Bosch' }, location.origin);
  });

  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  expect((calls[calls.length - 1].options as Record<string, unknown>).store_brand).toBe('Bosch');
});
