/**
 * Panel behaviour once a translation file has loaded (audit 2026-10-02 UI-08,
 * UI-09, UI-21). bootPanel() normally serves `{}` so _t() falls back to the
 * English in the JS; every bug here lived only on the loaded-translation path.
 */

import { test, expect, type Page } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import { buildHandlers } from '../helpers/ws-handlers';
import deviceIdle from '../fixtures/mock-data/device-idle.json';
import en from '../../custom_components/ha_washdata/translations/panel/en.json';
import de from '../../custom_components/ha_washdata/translations/panel/de.json';

const DE = { locale: { language: 'de', number_format: 'language', time_format: '24' } };

async function panelEval<T>(page: Page, fn: string): Promise<T> {
  return page.evaluate(`(() => { const root = document.querySelector('ha-washdata-panel'); return (${fn})(root); })()`) as Promise<T>;
}

// ── UI-21: _t() escapes substituted vars ───────────────────────────────────────

test('UI-21: markup typed into settings search renders as text with a translation loaded', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, DE, { translations: { en, de } });
  await clickTab(page, 'settings');
  const search = page.locator('#wd-settings-search');
  await expect(search).toBeVisible({ timeout: 8_000 });

  await search.fill('<img src=x onerror="window.__wdxss=1"><b id="wd-xss-probe">zz</b>');

  // The German template proves the loaded translation rendered the message, not
  // the (already escaping) JS fallback.
  const msg = page.locator('.wd-info', { hasText: 'Keine Einstellungen' });
  await expect(msg).toBeVisible();
  await expect(msg).toContainText('<b id="wd-xss-probe">zz</b>');
  await expect(page.locator('#wd-xss-probe')).toHaveCount(0);
  await expect(page.locator('ha-washdata-panel img[src="x"]')).toHaveCount(0);
  expect(await page.evaluate(() => (window as any).__wdxss)).toBeUndefined();
});

test('UI-21: _t escapes vars, _html opts out, _tText stays plain text', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, {}, { translations: { en } });
  const out = await panelEval<string[]>(page, `(root) => [
    root._t('msg.no_settings_match', {q: '<i>&'}, 'x'),
    root._tText('msg.no_settings_match', {q: '<i>&'}, 'x'),
    root._t('lbl.n_selected', {n: '$& $1'}, 'x'),
  ]`);
  expect(out[0]).toBe('No settings match "&lt;i&gt;&amp;"');
  expect(out[1]).toBe('No settings match "<i>&"');
  // A value is never read as a String.replace pattern.
  expect(out[2]).toBe('$&amp; $1 selected');
  // The two deliberate HTML vars still render as markup.
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="notifications"]').first().click();
  await expect(page.locator('.wd-info code', { hasText: 'notify.<name>' }).first()).toBeVisible({ timeout: 8_000 });
});

// ── UI-08: plural forms survive a loaded translation ───────────────────────────

test('UI-08: English plural forms render from en.json, singular and plural', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, {}, { translations: { en } });
  // The idle fixture has two pending suggestions.
  await expect(page.locator('.wd-attn-title', { hasText: 'tuning suggestion' })).toHaveText('2 tuning suggestions');

  const one = { ...deviceIdle.devices[0], suggestion_keys: ['off_delay'], suggestions_count: 1 };
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_devices': { devices: [one] } }, {}, { translations: { en } });
  await expect(page.locator('.wd-attn-title', { hasText: 'tuning suggestion' })).toHaveText('1 tuning suggestion');
});

test('UI-08: CLDR categories per language, with plain-key and English fallbacks', async ({ page }) => {
  const pl = {
    lbl: {
      n_tuning_suggestions: '{n} PLAIN',
      n_tuning_suggestions_one: '{n} ONE',
      n_tuning_suggestions_few: '{n} FEW',
      n_tuning_suggestions_many: '{n} MANY',
      n_tuning_suggestions_other: '{n} OTHER',
      n_selected: '{n} wybrano',  // single-form translation: no plural keys at all
    },
  };
  await page.goto('/');
  await bootPanel(page, {}, { locale: { language: 'pl' } }, { translations: { en, pl } });
  const out = await panelEval<string[]>(page, `(root) => [
    root._t('lbl.n_tuning_suggestions', {n: 1}, 'x'),
    root._t('lbl.n_tuning_suggestions', {n: 3}, 'x'),
    root._t('lbl.n_tuning_suggestions', {n: 5}, 'x'),
    root._t('lbl.n_tuning_suggestions', {n: 1.5}, 'x'),
    root._t('lbl.n_selected', {n: 3}, 'x'),
    root._t('msg.showing_suggestions', {count: 1}, 'x'),
    root._t('msg.showing_suggestions', {count: 4}, 'x'),
  ]`);
  expect(out).toEqual([
    '1 ONE', '3 FEW', '5 MANY', '1.5 OTHER',
    '3 wybrano',
    // Missing from the language entirely: English, with English plural rules.
    'Showing 1 setting with suggestions.', 'Showing 4 settings with suggestions.',
  ]);
});

// ── UI-08: numbers, costs and dates follow the HA user's locale ────────────────

test('UI-08: the cycles table formats energy and cost for the HA user locale', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, DE, { translations: { en, de } });
  await clickTab(page, 'history');
  const row = page.locator('tr[data-cid="cyc-001"]');
  await expect(row).toBeVisible({ timeout: 8_000 });
  await expect(row).toContainText('0,85 kWh');
  await expect(row).toContainText('0,21 €');
});

test('UI-08: HA number_format overrides the language for numbers', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, { locale: { language: 'en', number_format: 'decimal_comma' } }, { translations: { en } });
  await clickTab(page, 'history');
  const row = page.locator('tr[data-cid="cyc-001"]');
  await expect(row).toBeVisible({ timeout: 8_000 });
  await expect(row).toContainText('0,85 kWh');
});

// ── UI-09: settings search matches what the user reads ─────────────────────────

test('UI-09: settings search finds a field by its translated label', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {}, DE, { translations: { en, de } });
  await clickTab(page, 'settings');
  const search = page.locator('#wd-settings-search');
  await expect(search).toBeVisible({ timeout: 8_000 });
  const label = (de as any).setting.min_power.label as string;
  expect(label.toLowerCase()).not.toContain('minimum');  // a real translation, not the English
  await search.fill(label.toLowerCase());
  await expect(page.locator('.wd-field[data-field="min_power"]')).toBeVisible();
  await expect(page.locator('.wd-info', { hasText: 'Keine Einstellungen' })).toHaveCount(0);

  // English schema text and the raw key still match.
  await search.fill('off_delay');
  await expect(page.locator('.wd-field[data-field="off_delay"]')).toBeVisible();
});
