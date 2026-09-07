/**
 * Issue #406: saving settings silently cleared "Group under device" (linked_device)
 * because the picker could not render the stored value.
 *
 * The HA device registry reaches the frontend asynchronously, so an early paint of
 * the Settings form can see an empty hass.devices map. The <select> then had no
 * <option> for the stored device id, the browser displayed "- None -", the
 * pending-edit snapshot froze that as an explicit null, and the next save (of any
 * unrelated field) wrote it back - dropping linked_device and the device's
 * via_device_id.
 */

import { test, expect } from '@playwright/test';
import { buildHandlers } from '../helpers/ws-handlers';
import optionsData from '../fixtures/mock-data/options.json';
import de from '../../custom_components/ha_washdata/translations/panel/de.json';

const LINKED = 'dev-plug-1';

// The shared options fixture omits the cadence keys, so Advanced mode falls back to
// the schema literals - and 30 s watchdog / 30 s sampling trips the panel's own
// watchdog >= 2x sampling rule, sending every save down the conflict branch. Supply
// a self-consistent set so these tests exercise the normal save path.
const CADENCE = {
  sampling_interval: 10,
  watchdog_interval: 30,
  no_update_active_timeout: 600,
  start_duration_threshold: 10,
  // stop_threshold_w (3 W) over off_delay (120 s) needs >= 0.1 Wh, above the 0.05
  // schema default, or End Energy conflicts too.
  end_energy_threshold: 0.2,
};

/** Boot into Advanced Settings without asserting a conflict-free start. */
async function bootSettingsRaw(page: any, opts: Record<string, unknown>, devices: Record<string, unknown>) {
  await page.goto('/');
  await page.route('**/panel-translations/**', (route: any) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));
  const handlers = buildHandlers({
    'ha_washdata/get_options': { options: { ...optionsData, ...opts } },
  });
  await page.evaluate(([h, d]: any) => { (window as any).__boot_panel(h, { devices: d }); },
    [handlers, devices] as any);
  await expect(page.locator('button.wd-tab').first()).toBeVisible({ timeout: 10_000 });
  await page.evaluate(() => (window as any).__freeze_poll());
  await page.locator('button.wd-tab[data-tab="settings"]').click();
  await expect(page.locator('input[data-opt="name"]').first()).toBeVisible({ timeout: 8_000 });
  const chk = page.locator('#wd-settings-level-chk');
  if (!(await chk.isChecked())) await page.locator('.wd-mode-switch .wd-toggle-track').first().click();
}

/** Boot the panel on the Settings tab in Advanced mode, with a controllable registry. */
async function bootSettings(page: any, opts: Record<string, unknown>, devices: Record<string, unknown>) {
  await page.goto('/');
  await page.route('**/panel-translations/**', (route: any) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));
  const handlers = buildHandlers({
    'ha_washdata/get_options': { options: { ...optionsData, ...CADENCE, ...opts } },
  });
  await page.evaluate(([h, d]: any) => { (window as any).__boot_panel(h, { devices: d }); },
    [handlers, devices] as any);
  await expect(page.locator('button.wd-tab').first()).toBeVisible({ timeout: 10_000 });
  await page.evaluate(() => (window as any).__freeze_poll());
  await page.locator('button.wd-tab[data-tab="settings"]').click();
  await expect(page.locator('input[data-opt="name"]').first()).toBeVisible({ timeout: 8_000 });
  const chk = page.locator('#wd-settings-level-chk');
  if (!(await chk.isChecked())) await page.locator('.wd-mode-switch .wd-toggle-track').first().click();
  // These tests assert on the normal save path, so the starting state must be
  // conflict-free - a held-back conflicting field would change the payload.
  const conflicts = await page.evaluate(() => {
    const el: any = document.querySelector('#wd-panel');
    return [...el._conflictKeysFromOpts()];
  });
  expect(conflicts, 'fixture must start conflict-free').toEqual([]);
}

test('the linked device stays selected when the device registry has loaded', async ({ page }) => {
  await bootSettings(page, { linked_device: LINKED }, {
    [LINKED]: { id: LINKED, name: 'ShellyPlugS04' },
    'dev-other': { id: 'dev-other', name: 'Router' },
  });
  await expect(page.locator('select[data-opt="linked_device"]')).toHaveValue(LINKED);
});

test('the linked device stays selected before the device registry arrives (#406)', async ({ page }) => {
  await bootSettings(page, { linked_device: LINKED }, {});
  // The stored id must still be selectable, so the field round-trips instead of
  // collapsing to "- None -".
  await expect(page.locator('select[data-opt="linked_device"]')).toHaveValue(LINKED);
});

test('a late device-registry update does not reset the picker to None (#406)', async ({ page }) => {
  await bootSettings(page, { linked_device: LINKED }, {});
  // Registry lands after the first paint; the panel re-renders.
  await page.evaluate((id: string) => {
    const el: any = document.querySelector('#wd-panel');
    el._hass.devices = { [id]: { id, name: 'ShellyPlugS04' } };
    el._render();
  }, LINKED);
  const sel = page.locator('select[data-opt="linked_device"]');
  await expect(sel).toHaveValue(LINKED);
  // ...and it is now the real registry entry, not the placeholder.
  await expect(sel.locator(`option[value="${LINKED}"]`)).toHaveText('ShellyPlugS04');
});

test('saving an unrelated field never submits linked_device (#406)', async ({ page }) => {
  await bootSettings(page, { linked_device: LINKED }, {});
  await page.locator('input[data-opt="min_power"]').fill('2.5');
  await page.locator('#wd-settings-save').click();

  await expect.poll(async () =>
    (await page.evaluate(() => (window as any).__get_calls('ha_washdata/set_options'))).length,
  ).toBe(1);
  const [call] = await page.evaluate(() => (window as any).__get_calls('ha_washdata/set_options'));
  expect(call.options).toEqual({ min_power: 2.5 });
  expect('linked_device' in call.options).toBe(false);
});

test('an untouched field is not rewritten by a save', async ({ page }) => {
  await bootSettings(page, { linked_device: LINKED }, { [LINKED]: { id: LINKED, name: 'Plug' } });
  await page.locator('input[data-opt="off_delay"]').fill('150');
  await page.locator('#wd-settings-save').click();
  await expect.poll(async () =>
    (await page.evaluate(() => (window as any).__get_calls('ha_washdata/set_options'))).length,
  ).toBe(1);
  const [call] = await page.evaluate(() => (window as any).__get_calls('ha_washdata/set_options'));
  expect(call.options).toEqual({ off_delay: 150 });
});

test('a save with no edits does not call set_options at all', async ({ page }) => {
  await bootSettings(page, { linked_device: LINKED }, { [LINKED]: { id: LINKED, name: 'Plug' } });
  await page.locator('#wd-settings-save').click();
  await page.waitForTimeout(300);
  const calls = await page.evaluate(() => (window as any).__get_calls('ha_washdata/set_options'));
  expect(calls).toHaveLength(0);
});

test('the device-type picker keeps the stored type when get_constants failed', async ({ page }) => {
  await page.goto('/');
  await page.route('**/panel-translations/**', (route: any) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));
  const handlers = buildHandlers({});
  await page.evaluate(([h]: any) => {
    (window as any).__boot_panel(h, { devices: {} });
    // get_constants is fetched once and its failure is swallowed, so device_types
    // can be empty for the whole session.
    const el: any = document.querySelector('#wd-panel');
    el._constants = { ...el._constants, deviceTypes: [] };
  }, [handlers] as any);
  await expect(page.locator('button.wd-tab').first()).toBeVisible({ timeout: 10_000 });
  await page.evaluate(() => (window as any).__freeze_poll());
  await page.locator('button.wd-tab[data-tab="settings"]').click();
  await expect(page.locator('input[data-opt="name"]').first()).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('select[data-opt="device_type"]')).toHaveValue('washing_machine');
});

test('an off-screen cascade fix still reaches the save payload (#406 guard)', async ({ page }) => {
  // Guards the diff introduced for #406: a cascade fix for a field in ANOTHER
  // settings section writes its value straight into _opts (so cross-section
  // validation sees it) and records it in _cascadePending. Diffing the payload
  // against the mutated _opts would call it unchanged and silently drop it, so the
  // diff must use the untouched _preCascadeOpts baseline.
  //
  // no_update_active_timeout (50) <= watchdog_interval (100) conflicts. Taking the
  // offered watchdog fix (25) then breaks watchdog >= 2x sampling (30), whose only
  // other fixable key - sampling_interval - lives in the Detection section.
  await bootSettingsRaw(page, {
    sampling_interval: 30,
    watchdog_interval: 100,
    no_update_active_timeout: 50,
    start_duration_threshold: 30,
    end_energy_threshold: 0.2,
  }, {});

  await page.locator('.wd-sec-btn', { hasText: 'Timing' }).click();
  const fix = page.locator('.wd-conflict-fix[data-ckey="watchdog_interval"]').first();
  await expect(fix).toBeVisible({ timeout: 5_000 });
  await fix.click();

  // The off-screen cascade landed on sampling_interval.
  await expect.poll(async () => page.evaluate(() =>
    JSON.stringify((document.querySelector('#wd-panel') as any)._cascadePending),
  )).toContain('sampling_interval');

  await page.locator('#wd-settings-save').click();
  await expect.poll(async () =>
    (await page.evaluate(() => (window as any).__get_calls('ha_washdata/set_options'))).length,
  ).toBe(1);
  const [call] = await page.evaluate(() => (window as any).__get_calls('ha_washdata/set_options'));
  expect(call.options.watchdog_interval).toBe(25);
  expect(call.options.sampling_interval).toBe(12);
});

/**
 * The two strings added for #406 must resolve through _t() from a real shipped
 * language file, not just fall back to English. Loads de.json for real (the panel
 * fetches /ha_washdata/panel-translations/{lang}.json on demand) and checks both the
 * dropdown option label - including its {id} substitution - and the toast.
 */
test('the #406 strings render from the German panel translation', async ({ page }) => {
  await page.goto('/');
  await page.route('**/panel-translations/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(de) }));

  const handlers = buildHandlers({
    'ha_washdata/get_options': { options: { ...optionsData, linked_device: 'dev-gone-42' } },
  });
  await page.evaluate(([h]: any) => { (window as any).__boot_panel(h, { devices: {} }); },
    [handlers] as any);
  await expect(page.locator('button.wd-tab').first()).toBeVisible({ timeout: 10_000 });
  await page.evaluate(() => (window as any).__freeze_poll());
  await page.locator('button.wd-tab[data-tab="settings"]').click();
  await expect(page.locator('input[data-opt="name"]').first()).toBeVisible({ timeout: 8_000 });

  const sel = page.locator('select[data-opt="linked_device"]');
  await expect(sel).toHaveValue('dev-gone-42');
  // German label with the id substituted, proving both _t() lookup and {id} vars.
  await expect(sel.locator('option[value="dev-gone-42"]'))
    .toHaveText('Nicht verfügbares Gerät (dev-gone-42)');

  // Save with no edits -> the German "no changes" toast.
  await page.locator('#wd-settings-save').click();
  await expect(page.locator('.wd-toast, [class*="toast"]').first())
    .toContainText('Keine Änderungen zum Speichern', { timeout: 5_000 });
});
