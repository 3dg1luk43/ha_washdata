/**
 * Phase Catalog: an edited built-in phase is reset, not deleted.
 *
 * Editing a built-in stores an override under the built-in's id. The catalog used
 * to offer Delete on it, and delete_phase rejected that every time with
 * "cannot_delete_builtin". The backend now restores the built-in on delete and the
 * catalog marks the row `is_override`, so the panel offers Reset.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

const CATALOG = {
  device_type: 'washing_machine',
  phases: [
    { id: 'washing_machine.wash', name: 'Main wash', description: '', device_type: 'washing_machine', is_default: false, is_override: true },
    { id: 'washing_machine.spin', name: 'Spin', description: '', device_type: 'washing_machine', is_default: true },
    { id: 'c-1', name: 'Steam', description: '', device_type: 'washing_machine', is_default: false },
  ],
};

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_phase_catalog': CATALOG, 'ha_washdata/delete_phase': { success: true } });
  await clickTab(page, 'profiles');
  await page.locator('[data-proftab="phase-catalog"]').click();
});

test('an edited built-in offers Reset; a custom phase offers Delete; a built-in neither', async ({ page }) => {
  const reset = page.locator('[data-action="del-phase"][data-pid="washing_machine.wash"]');
  await expect(reset).toBeVisible({ timeout: 8_000 });
  await expect(reset).toHaveAttribute('data-reset', '1');
  await expect(reset).toHaveText(/Reset/);
  await expect(page.locator('[data-action="del-phase"][data-pid="c-1"]')).toHaveText(/Delete/);
  await expect(page.locator('[data-action="del-phase"][data-pid="washing_machine.spin"]')).toHaveCount(0);
});

test('Reset asks to restore the built-in and calls delete_phase', async ({ page }) => {
  await page.locator('[data-action="del-phase"][data-pid="washing_machine.wash"]').click();
  await expect(page.locator('.wd-modal')).toContainText(/restore the built-in phase/i);
  await page.locator('.wd-modal [data-maction="ok"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/delete_phase');
  expect(calls[calls.length - 1]).toMatchObject({ phase_id: 'washing_machine.wash' });
});
