/**
 * "Undo last import" (register item 195), Advanced -> Diagnostics -> Export / Import.
 * The button exists only while get_diagnostics reports a restore point, asks for
 * confirmation, and disappears once the undo has consumed the restore point.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import panelConfig from '../fixtures/mock-data/panel-config.json';

const RESTORE_POINT = {
  created_at: '2026-10-01T10:00:00+00:00',
  source: 'import_selective_replace',
  counts: { profiles: 2, real_cycles: 5, reference_cycles: 1, backfill_cycles: 3 },
};

async function openDiagnostics(page) {
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
  await expect(page.locator('button[data-action="export-select-open"]').first()).toBeVisible({ timeout: 8_000 });
}

async function bootWithRestorePoint(page, extra = {}) {
  await page.goto('/');
  await bootPanel(page, extra);
  // Function handlers cannot cross bootPanel's serialisation; install them after.
  // The restore point is there until undo_import consumes it.
  await page.evaluate((rp) => {
    const w = window as any;
    w.__ws_handlers['ha_washdata/get_diagnostics'] = () => ({
      stats: { total_cycles: 5 }, import_undo: w.__undo_done ? null : rp,
    });
    w.__ws_handlers['ha_washdata/undo_import'] = () => {
      w.__undo_done = true;
      return { success: true, summary: { restored_from: rp.created_at, counts: rp.counts } };
    };
  }, RESTORE_POINT);
}

test('no undo button while there is no restore point', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await openDiagnostics(page);
  await expect(page.locator('button[data-action="import-undo"]')).toHaveCount(0);
});

test('a restore point shows the undo button with what it holds', async ({ page }) => {
  await bootWithRestorePoint(page);
  await openDiagnostics(page);
  const btn = page.locator('button[data-action="import-undo"]');
  await expect(btn).toBeVisible({ timeout: 5_000 });
  await expect(btn).toHaveText('Undo last import');
  // 5 real + 1 reference + 3 backfill
  await expect(page.locator('.wd-import-undo')).toContainText('2 profile(s) and 9 cycle(s)');
});

test('undo asks first, then restores and the button goes away', async ({ page }) => {
  await bootWithRestorePoint(page);
  await openDiagnostics(page);
  await page.locator('button[data-action="import-undo"]').click();

  const modal = page.locator('.wd-modal');
  await expect(modal).toBeVisible({ timeout: 5_000 });
  await expect(modal).toContainText('Undo last import');
  await expect(modal).toContainText('Cycles recorded and changes made since then are lost');
  expect(await page.evaluate(() => (window as any).__get_calls('ha_washdata/undo_import').length)).toBe(0);

  await modal.locator('button[data-maction="ok"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/undo_import');
  expect(calls[0]).toHaveProperty('entry_id');
  await expect(page.locator('.wd-toast')).toContainText('Import undone', { timeout: 5_000 });
  await expect(page.locator('button[data-action="import-undo"]')).toHaveCount(0, { timeout: 5_000 });
});

test('cancelling the confirmation changes nothing', async ({ page }) => {
  await bootWithRestorePoint(page);
  await openDiagnostics(page);
  await page.locator('button[data-action="import-undo"]').click();
  await page.locator('.wd-modal button[data-maction="cancel"]').click();
  await expect(page.locator('.wd-modal')).toHaveCount(0);
  expect(await page.evaluate(() => (window as any).__get_calls('ha_washdata/undo_import').length)).toBe(0);
  await expect(page.locator('button[data-action="import-undo"]')).toBeVisible();
});

test('the undo button is admin-only, like the command', async ({ page }) => {
  await bootWithRestorePoint(page, {
    'ha_washdata/get_panel_config': { ...panelConfig, is_admin: false },
  });
  await openDiagnostics(page);
  await expect(page.locator('button[data-action="export-select-open"]').first()).toBeVisible();
  await expect(page.locator('button[data-action="import-undo"]')).toHaveCount(0);
});

test('a replace import names the undo, and the button appears after it', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/analyze_import': {
      manifest: {
        format: 'v2', version: 17, device_type_match: true, real_history_allowed: true, warnings: [],
        source_device_type: 'washing_machine', local_device_type: 'washing_machine',
        categories: {
          profiles: { present: true, importable: true, count: 1,
            items: [{ name: 'Wool 20', real_cycles: 0, reference_cycles: 0, backfill_cycles: 0, conflict: false }] },
        },
      },
    },
  });
  await page.evaluate((rp) => {
    const w = window as any;
    w.__ws_handlers['ha_washdata/get_diagnostics'] = () => ({
      stats: { total_cycles: 5 }, import_undo: w.__imported ? rp : null,
    });
    w.__ws_handlers['ha_washdata/import_config_selective'] = () => {
      w.__imported = true;
      return { success: true, summary: { profiles_imported: 1, real_cycles_imported: 0,
        reference_cycles_imported: 0, backfill_cycles_imported: 0, restore_point_saved: true } };
    };
  }, RESTORE_POINT);
  await openDiagnostics(page);
  await expect(page.locator('button[data-action="import-undo"]')).toHaveCount(0);
  await page.locator('button[data-action="import-config-open"]').first().click();
  await page.locator('#wd-import-json').fill('{"version":17,"data":{"profiles":{"Wool 20":{}}}}');
  await page.locator('button[data-maction="import-analyze"]').first().click();
  await page.locator('button[data-maction="imp-mode-replace"]').click();
  await expect(page.locator('.wd-modal')).toContainText('you can undo this import from Export / Import');
  await page.locator('button[data-maction="import-apply-ok"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/import_config_selective');
  expect(calls[0]).toHaveProperty('mode', 'replace');
  await expect(page.locator('button[data-action="import-undo"]')).toBeVisible({ timeout: 5_000 });
});
