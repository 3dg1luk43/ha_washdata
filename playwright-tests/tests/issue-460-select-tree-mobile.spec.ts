/**
 * #460: on a phone, "Export (choose data)" (and the Import wizard's select step,
 * which renders the same tree) cut its options off: category names were sliced
 * in half and the per-cycle rows were unreachable.
 *
 * Cause: .wd-sd-tree is a column flex box with a max-height. Its .wd-sd-group
 * children set overflow:hidden, which drops their automatic minimum height to 0,
 * so once the tree's content outgrew the cap the groups SHRANK to fit (clipping
 * their own rows) instead of the tree scrolling. A phone hits the cap at once
 * (44vh is small and labels wrap); a desktop only with a long list.
 */

import { test, expect, Page } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

const cycles = (n: number, prefix: string) => Array.from({ length: n }, (_, i) => ({
  id: `${prefix}${i}`, date: `2023-01-${String(i + 1).padStart(2, '0')}T10:00:00+00:00`, duration: 3600 - i * 60,
}));

const CATS = {
  profiles: { present: true, count: 2, items: [
    { name: 'Baumwolle 60 mit Vorwaesche', real_cycles: 6, reference_cycles: 0 },
    { name: 'Cotton 40', real_cycles: 3, reference_cycles: 1 },
  ] },
  real_cycles: { present: true, count: 9, groups: [
    { profile: 'Baumwolle 60 mit Vorwaesche', count: 6, cycles: cycles(6, 'p') },
    { profile: 'Cotton 40', count: 3, cycles: cycles(3, 'q') },
  ] },
  reference_cycles: { present: true, count: 1, groups: [{ profile: 'Cotton 40', count: 1, cycles: cycles(1, 'r') }] },
  // Longest English category label: on a phone it must wrap, not ellipsise.
  backfill_cycles: { present: true, count: 1, groups: [{ profile: 'Cotton 40', count: 1, cycles: cycles(1, 'b') }] },
  settings: { present: true, count: 1, keys: ['off_delay'] },
  maintenance_log: { present: true, count: 1 },
};

const EXPORT_INVENTORY = { manifest: CATS };
const IMPORT_MANIFEST = {
  manifest: {
    format: 'v2', version: 11, device_type_match: true, real_history_allowed: true, warnings: [],
    source_device_type: 'washing_machine', local_device_type: 'washing_machine',
    categories: Object.fromEntries(Object.entries(CATS).map(([k, v]) => [k, { ...v, importable: true }])),
  },
};

async function openDiagnostics(page: Page) {
  await clickTab(page, 'advanced');
  const diagTab = page.locator('[data-ptab="diagnostics"]').first();
  await expect(diagTab).toBeVisible({ timeout: 5_000 });
  await diagTab.click();
}

async function openExport(page: Page) {
  await openDiagnostics(page);
  await page.locator('button[data-action="export-select-open"]').first().click();
  await expect(page.locator('.wd-modal .wd-sd-tree')).toBeVisible({ timeout: 8_000 });
}

async function openImport(page: Page) {
  await openDiagnostics(page);
  await page.locator('button[data-action="import-config-open"]').first().click();
  await page.locator('#wd-import-json').fill('{"version":11,"data":{}}');
  await page.locator('button[data-maction="import-analyze"]').first().click();
  await expect(page.locator('.wd-modal .wd-sd-tree')).toBeVisible({ timeout: 8_000 });
}

/** Rows whose content is clipped by their own box, and labels outside the viewport. */
async function clipping(page: Page) {
  return page.evaluate(() => {
    const root = (document.querySelector('ha-washdata-panel') as any).shadowRoot as ShadowRoot;
    const tree = root.querySelector('.wd-modal .wd-sd-tree') as HTMLElement;
    const vw = window.innerWidth;
    const clippedGroups = Array.from(tree.querySelectorAll<HTMLElement>('.wd-sd-group'))
      .filter(g => g.scrollHeight > g.clientHeight + 1)
      .map(g => `${(g.textContent || '').trim().split('\n')[0]} ${g.clientHeight}/${g.scrollHeight}`);
    const truncatedNames = Array.from(tree.querySelectorAll<HTMLElement>('.wd-sd-prof-name, .wd-sd-cyc-meta'))
      .filter(s => s.scrollWidth > s.clientWidth + 1)
      .map(s => (s.textContent || '').trim());
    const offscreen = Array.from(tree.querySelectorAll<HTMLElement>('label'))
      .filter(l => { const r = l.getBoundingClientRect(); return r.left < -1 || r.right > vw + 1; })
      .map(l => (l.textContent || '').trim());
    return { clippedGroups, truncatedNames, offscreen };
  });
}

async function assertTreeUsable(page: Page) {
  // Expand the first cycle group so its individual cycles are listed.
  await page.locator('button[data-maction="wiz-expand"]').first().click();
  const cyc = page.locator('input[data-maction="wiz-toggle-cyc"][data-cid="p5"]');
  await expect(cyc).toBeAttached();
  expect(await clipping(page)).toEqual({ clippedGroups: [], truncatedNames: [], offscreen: [] });
  // A per-cycle checkbox can be reached and ticked off; the count follows.
  await cyc.scrollIntoViewIfNeeded();
  await expect(cyc).toBeChecked();
  await cyc.click({ timeout: 5_000 });
  await expect(page.locator('input[data-maction="wiz-toggle-cyc"][data-cid="p5"]')).not.toBeChecked();
  const realCat = page.locator('input[data-maction="wiz-toggle-cat"][data-cat="real_cycles"]');
  await expect(page.locator('.wd-sd-group').filter({ has: realCat }).locator('.wd-sd-count').first()).toHaveText('8/9');
}

for (const vp of [null, { width: 360, height: 640 }]) {
  test.describe(vp ? 'small phone 360x640' : 'project viewport', () => {
    if (vp) test.use({ viewport: vp });

    test.beforeEach(async ({ page }) => {
      await page.goto('/');
      await bootPanel(page, {
        'ha_washdata/get_export_inventory': EXPORT_INVENTORY,
        'ha_washdata/analyze_import': IMPORT_MANIFEST,
      });
    });

    test('export (choose data): every option and cycle row is visible and selectable', async ({ page }) => {
      await openExport(page);
      await assertTreeUsable(page);
    });

    test('import (choose data): every option and cycle row is visible and selectable', async ({ page }) => {
      await openImport(page);
      await assertTreeUsable(page);
    });
  });
}
