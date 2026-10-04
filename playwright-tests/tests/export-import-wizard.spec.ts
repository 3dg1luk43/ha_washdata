/**
 * Selective export/import wizard (Advanced -> Diagnostics).
 * Covers the export selection tree + selective export call, and the import
 * analyze -> select -> apply flow with the tri-state tree.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

const EXPORT_INVENTORY = {
  manifest: {
    profiles: { present: true, count: 1, items: [{ name: 'Cotton 40', real_cycles: 2, reference_cycles: 0 }] },
    real_cycles: {
      present: true, count: 2,
      groups: [{ profile: 'Cotton 40', count: 2, cycles: [
        { id: 'p1', date: '2023-01-01T10:00:00+00:00', duration: 3600 },
        { id: 'p2', date: '2023-01-02T10:00:00+00:00', duration: 3500 },
      ] }],
    },
    settings: { present: true, count: 1, keys: ['off_delay'] },
    maintenance_log: { present: true, count: 1 },
  },
};

const IMPORT_MANIFEST = {
  manifest: {
    format: 'v2', version: 11, device_type_match: true, real_history_allowed: true, warnings: [],
    source_device_type: 'washing_machine', local_device_type: 'washing_machine',
    categories: {
      profiles: { present: true, importable: true, count: 1,
        items: [{ name: 'Wool 20', real_cycles: 1, reference_cycles: 0, conflict: false }] },
      real_cycles: { present: true, importable: true, count: 1,
        groups: [{ profile: 'Wool 20', count: 1, cycles: [{ id: 'r1', date: '2023-02-01T10:00:00+00:00', duration: 1800 }] }] },
      settings: { present: true, importable: true, count: 1, keys: ['off_delay'] },
    },
  },
};

async function openDiagnostics(page) {
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_export_inventory': EXPORT_INVENTORY,
    'ha_washdata/export_config_selective': { json_data: '{"version":11,"data":{}}' },
    'ha_washdata/analyze_import': IMPORT_MANIFEST,
    'ha_washdata/import_config_selective': {
      success: true,
      summary: { profiles_imported: 1, real_cycles_imported: 0, reference_cycles_imported: 1, settings_applied: 0 },
    },
  });
});

// ─── Export wizard ────────────────────────────────────────────────────────────

test('export wizard opens and fetches the inventory', async ({ page }) => {
  await openDiagnostics(page);
  await page.locator('button[data-action="export-select-open"]').first().click();
  await assertWsCalled(page, 'ha_washdata/get_export_inventory');
  await expect(page.locator('.wd-modal .wd-sd-tree')).toBeVisible({ timeout: 8_000 });
  // Category rows rendered.
  await expect(page.locator('input[data-maction="wiz-toggle-cat"][data-cat="profiles"]')).toBeVisible();
});

test('export wizard sends a selection without the unticked category', async ({ page }) => {
  await openDiagnostics(page);
  await page.locator('button[data-action="export-select-open"]').first().click();
  await expect(page.locator('.wd-modal .wd-sd-tree')).toBeVisible({ timeout: 8_000 });
  // Everything is selected by default; untick "settings".
  const settingsCb = page.locator('input[data-maction="wiz-toggle-cat"][data-cat="settings"]');
  await expect(settingsCb).toBeChecked();
  await settingsCb.click();
  await expect(settingsCb).not.toBeChecked();
  await page.locator('button[data-maction="export-generate"]').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/export_config_selective');
  const selection = calls[0].selection as { categories: string[] };
  expect(selection.categories).toContain('profiles');
  expect(selection.categories).not.toContain('settings');
});

// ─── Import wizard ────────────────────────────────────────────────────────────

test('import wizard analyzes pasted JSON and renders the manifest tree', async ({ page }) => {
  await openDiagnostics(page);
  await page.locator('button[data-action="import-config-open"]').first().click();
  const ta = page.locator('#wd-import-json');
  await expect(ta).toBeVisible({ timeout: 8_000 });
  await ta.fill('{"version":11,"data":{"profiles":{"Wool 20":{}}}}');
  await page.locator('button[data-maction="import-analyze"]').first().click();
  await assertWsCalled(page, 'ha_washdata/analyze_import');
  // Select step: tree + mode/destination toggles rendered.
  await expect(page.locator('.wd-modal .wd-sd-tree')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('button[data-maction="imp-mode-merge"]')).toBeVisible();
  await expect(page.locator('button[data-maction="imp-dest-reference"]')).toBeVisible();
});

test('import wizard applies the selection with merge + reference destination', async ({ page }) => {
  await openDiagnostics(page);
  await page.locator('button[data-action="import-config-open"]').first().click();
  const ta = page.locator('#wd-import-json');
  await expect(ta).toBeVisible({ timeout: 8_000 });
  await ta.fill('{"version":11,"data":{"profiles":{"Wool 20":{}}}}');
  await page.locator('button[data-maction="import-analyze"]').first().click();
  await expect(page.locator('.wd-modal .wd-sd-tree')).toBeVisible({ timeout: 8_000 });
  await page.locator('button[data-maction="import-apply-ok"]').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/import_config_selective');
  expect(calls[0]).toHaveProperty('mode', 'merge');
  expect(calls[0]).toHaveProperty('cycle_destination', 'reference');
  const selection = calls[0].selection as { categories: string[] };
  expect(selection.categories).toContain('profiles');
});

// ─── Backfill cycles as a wizard category (register item 129e) ───────────────

const BACKFILL_INVENTORY = {
  manifest: {
    profiles: { present: true, count: 1, items: [{ name: 'Cotton 40', real_cycles: 1, reference_cycles: 0, backfill_cycles: 2 }] },
    real_cycles: {
      present: true, count: 1,
      groups: [{ profile: 'Cotton 40', count: 1, cycles: [{ id: 'p1', date: '2023-01-01T10:00:00+00:00', duration: 3600 }] }],
    },
    backfill_cycles: {
      present: true, count: 3,
      groups: [
        { profile: '', count: 1, cycles: [{ id: 'b0', date: '2022-11-30T10:00:00+00:00', duration: 3000 }] },
        { profile: 'Cotton 40', count: 2, cycles: [
          { id: 'b1', date: '2022-12-01T10:00:00+00:00', duration: 3500 },
          { id: 'b2', date: '2022-12-02T10:00:00+00:00', duration: 3400 },
        ] },
      ],
    },
  },
};

test('export wizard lists backfill cycles and sends their ids', async ({ page }) => {
  await page.evaluate((inv) => { (window as any).__ws_handlers['ha_washdata/get_export_inventory'] = inv; }, BACKFILL_INVENTORY);
  await openDiagnostics(page);
  await page.locator('button[data-action="export-select-open"]').first().click();
  const tree = page.locator('.wd-modal .wd-sd-tree');
  await expect(tree).toBeVisible({ timeout: 8_000 });
  const cat = page.locator('input[data-maction="wiz-toggle-cat"][data-cat="backfill_cycles"]');
  await expect(cat).toBeChecked();
  await expect(tree).toContainText('Cycles found in imported power history');
  await expect(tree.locator('.wd-sd-group').filter({ has: cat }).locator('.wd-sd-count')).toHaveText('3/3');
  // The per-profile count includes backfilled cycles (1 real + 2 backfill).
  await expect(tree).toContainText('Cotton 40 · 3');

  // Untick one profile group of backfill cycles: only the rest travel.
  await page.locator('input[data-maction="wiz-toggle-cycgroup"][data-cat="backfill_cycles"][data-prof="Cotton 40"]').click();
  await expect(cat).not.toBeChecked();
  await page.locator('button[data-maction="export-generate"]').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/export_config_selective');
  const selection = calls[0].selection as { categories: string[]; backfill_cycle_ids: string[] };
  expect(selection.categories).toContain('backfill_cycles');
  expect(selection.backfill_cycle_ids).toEqual(['b0']);
});

test('import wizard imports backfill cycles and counts them in the toast', async ({ page }) => {
  await page.evaluate(() => {
    const w = window as any;
    w.__ws_handlers['ha_washdata/analyze_import'] = { manifest: {
      format: 'v2', version: 17, device_type_match: true, real_history_allowed: true, warnings: [],
      source_device_type: 'washing_machine', local_device_type: 'washing_machine',
      categories: {
        backfill_cycles: { present: true, importable: true, count: 2,
          groups: [{ profile: 'Wool 20', count: 2, cycles: [
            { id: 'x1', date: '2023-02-01T10:00:00+00:00', duration: 1800 },
            { id: 'x2', date: '2023-02-02T10:00:00+00:00', duration: 1800 },
          ] }] },
      },
    } };
    w.__ws_handlers['ha_washdata/import_config_selective'] = { success: true, summary: {
      profiles_imported: 0, real_cycles_imported: 0, reference_cycles_imported: 0,
      backfill_cycles_imported: 2, settings_applied: 0, restore_point_saved: false } };
  });
  await openDiagnostics(page);
  await page.locator('button[data-action="import-config-open"]').first().click();
  await page.locator('#wd-import-json').fill('{"version":17,"data":{"backfill_cycles":[]}}');
  await page.locator('button[data-maction="import-analyze"]').first().click();
  await expect(page.locator('input[data-maction="wiz-toggle-cat"][data-cat="backfill_cycles"]')).toBeChecked({ timeout: 8_000 });
  await page.locator('button[data-maction="import-apply-ok"]').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/import_config_selective');
  const selection = calls[0].selection as { categories: string[]; backfill_cycle_ids: string[] };
  expect(selection.categories).toEqual(['backfill_cycles']);
  expect(selection.backfill_cycle_ids.sort()).toEqual(['x1', 'x2']);
  await expect(page.locator('.wd-toast')).toContainText('Imported 0 profile(s) and 2 cycle(s)', { timeout: 5_000 });
});
