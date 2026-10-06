/**
 * Settings tab tests.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled } from '../helpers/panel';
import optionsData from '../fixtures/mock-data/options.json';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await clickTab(page, 'settings');
});

test('settings tab fetches options on navigation', async ({ page }) => {
  await assertWsCalled(page, 'ha_washdata/get_options');
});

test('Basic, the production default, shows the device identity and power sensor (UI-22)', async ({ page }) => {
  // The fixture now boots in Basic, as a fresh install does (audit UI-22); this
  // is the screen most users see, and it once hid name and sensor (UI-02).
  await expect(page.locator('#wd-settings-level-chk')).not.toBeChecked({ timeout: 5_000 });
  await expect(page.locator('input[data-opt="name"]')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('input[data-opt="power_sensor"]')).toBeVisible();
  await expect(page.locator('input[data-opt="name"]')).toHaveValue('Test Washer');
});

test('advanced mode shows more fields than basic mode', async ({ page }) => {
  const chk = page.locator('#wd-settings-level-chk');
  await expect(chk).toHaveCount(1, { timeout: 5_000 });
  // Ensure we start in basic mode (checkbox unchecked).
  if (await chk.isChecked()) {
    await page.locator('.wd-mode-switch').first().click();
    await expect(chk).not.toBeChecked();
  }
  const basicCount = await page.locator('.wd-field').count();

  // Flip the toggle to advanced by clicking the slider track.
  await page.locator('.wd-mode-switch .wd-toggle-track').first().click();
  await expect(chk).toBeChecked();
  const advCount = await page.locator('.wd-field').count();

  expect(advCount).toBeGreaterThan(basicCount);
});

test('editing a field marks the form as dirty (enables save)', async ({ page }) => {
  const nameInput = page.locator('input[data-opt="name"]').first();
  await expect(nameInput).toBeVisible({ timeout: 8_000 });
  await nameInput.fill('Modified Washer');
  // Save button should become enabled
  const saveBtn = page.locator('#wd-settings-save').first();
  await expect(saveBtn).not.toBeDisabled({ timeout: 3_000 });
});

test('saving settings calls set_options WS command', async ({ page }) => {
  const nameInput = page.locator('input[data-opt="name"]').first();
  await expect(nameInput).toBeVisible({ timeout: 8_000 });
  await nameInput.fill('New Washer Name');
  const saveBtn = page.locator('#wd-settings-save').first();
  await saveBtn.click();
  await assertWsCalled(page, 'ha_washdata/set_options');
});

test('search input filters settings to matching fields only', async ({ page }) => {
  const searchInput = page.locator('#wd-settings-search, input[placeholder*="search"], input[placeholder*="filter"]').first();
  await expect(searchInput).toBeVisible({ timeout: 5_000 });
  await searchInput.fill('delay');
  // Fields with "delay" in the label should appear
  const visibleFields = page.locator('.wd-field');
  const count = await visibleFields.count();
  expect(count).toBeGreaterThan(0);
  // Fields without "delay" should not appear — check device name is gone
  await expect(page.locator('input[data-opt="name"]')).not.toBeVisible();
});

test('suggestion banner appears when device has suggestions', async ({ page }) => {
  // Re-boot with suggestions
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        { key: 'off_delay', suggested: 371, current: 180, reason: 'Your device consistently goes off for longer' },
      ],
    },
  });
  await clickTab(page, 'settings');
  const banner = page.locator('.wd-sug-banner').first();
  await expect(banner).toBeVisible({ timeout: 5_000 });
});

test('mute button renders inside the suggestion card', async ({ page }) => {
  // The mute button (#343) is injected into the suggestion markup by a trailing
  // `</div>` regex in _htmlSugWidget, which implicitly assumes every branch
  // (single / split / agree-collapse) ends with a matching closing div. This
  // asserts the button actually lands INSIDE .wd-sug so a future markup change
  // to any branch cannot silently drop or misplace it.
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        { key: 'off_delay', suggested: 371, current: 180, reason: 'Test reason' },
      ],
    },
  });
  await clickTab(page, 'settings');
  const field = page.locator('.wd-field[data-field="off_delay"]').first();
  await expect(field).toBeVisible({ timeout: 8_000 });
  // Nested inside the suggestion card, not a sibling that escaped the replace().
  const mute = field.locator('.wd-sug [data-suglock="off_delay"]').first();
  await expect(mute).toBeVisible({ timeout: 3_000 });
});

test('suggestion widget appears beside the relevant field', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        { key: 'off_delay', suggested: 371, current: 180, reason: 'Test reason' },
      ],
    },
  });
  await clickTab(page, 'settings');
  // Navigate to show the off_delay field
  const field = page.locator('.wd-field[data-field="off_delay"], [data-field="off_delay"]').first();
  await expect(field).toBeVisible({ timeout: 8_000 });
  // The suggestion widget should be inside the field
  const sug = field.locator('.wd-sug').first();
  await expect(sug).toBeVisible({ timeout: 3_000 });
});

test('Use button in suggestion applies the value', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        { key: 'off_delay', suggested: 371, current: 180, reason: 'Test reason' },
      ],
    },
  });
  await clickTab(page, 'settings');
  const useBtn = page.locator('.wd-sug-use').first();
  await expect(useBtn).toBeVisible({ timeout: 8_000 });
  await useBtn.click();
  // The off_delay input should now show 371
  const offDelayInput = page.locator('input[data-opt="off_delay"]').first();
  await expect(offDelayInput).toHaveValue('371', { timeout: 3_000 });
});

test('a muted key hides its suggestion and its banner count', async ({ page }) => {
  // The mute (#343) is per-setting: a locked key must not render a pill or count.
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': { suggestions: [], locked_suggestions: ['off_delay'] },
  });
  await clickTab(page, 'settings');
  const field = page.locator('.wd-field[data-field="off_delay"]').first();
  await expect(field).toBeVisible({ timeout: 8_000 });
  await expect(field.locator('.wd-sug')).toHaveCount(0);
  await expect(page.locator('.wd-sug-banner')).not.toContainText(/tuning suggestion/i);
  await expect(page.locator('.wd-sug-banner [data-action="sug-unmute-all"]')).toBeVisible();
});

test('the suggestion pill says what the change does', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [{ key: 'off_delay', suggested: 371, current: 120, reason: 'Test reason' }],
    },
  });
  // The harness serves empty translations and impact lines have no JS fallback,
  // so seed the one key this asserts on (en.json carries it in production).
  await page.evaluate(() => {
    const el: any = document.querySelector('ha-washdata-panel');
    el._panelTrans = { en: { suggestion: { impact: { off_delay: { higher: 'More time for appliance pauses' } } } } };
  });
  await clickTab(page, 'settings');
  const field = page.locator('.wd-field[data-field="off_delay"]').first();
  await expect(field.locator('.wd-sug .wd-sug-impact-line')).toContainText(/pauses/i, { timeout: 8_000 });
});

test('Apply all previews every change before saving', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': {
      suggestions: [
        { key: 'off_delay', suggested: 371, current: 120, reason: 'a' },
        { key: 'min_power', suggested: 3, current: 2, reason: 'b' },
      ],
    },
  });
  await clickTab(page, 'settings');
  await page.locator('.wd-sug-banner [data-action="sug-apply-all"]').first().click();
  const list = page.locator('.wd-confirm-list li');
  await expect(list).toHaveCount(2, { timeout: 5_000 });
  await expect(list.first()).toContainText('120');
  await expect(list.first()).toContainText('371');
  // Nothing is sent until the preview is confirmed.
  await assertWsNotCalled(page, 'ha_washdata/apply_suggestions');
  await page.locator('[data-maction="ok"]').click();
  await assertWsCalled(page, 'ha_washdata/apply_suggestions');
});

test('settings changelog dot appears for changed settings', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_settings_changelog': {
      changelog: [
        { key: 'off_delay', old: '180', new: '371', timestamp: '2026-07-10T14:00:00+00:00' },
      ],
    },
  });
  await clickTab(page, 'settings');
  const dot = page.locator('.wd-chg-dot').first();
  await expect(dot).toBeVisible({ timeout: 5_000 });
});

test('settings tab renders without overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  // Settings should stack to single column
  const overflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const body = el.shadowRoot.querySelector('.wd-body');
    return body ? body.scrollWidth - body.clientWidth : 0;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});

// The fixture boots Basic, the production default (audit UI-22). Tests of
// Advanced-only fields re-boot in Advanced explicitly.
async function rebootAdvanced(page) {
  await bootPanel(page, {}, {}, { settingsLevel: 'advanced' });
  await clickTab(page, 'settings');
}

test('notification fields are present in settings', async ({ page }) => {
  await rebootAdvanced(page);
  // Navigate to the Notifications section (separate from the default basic section)
  const notifSec = page.locator('button[data-sec="notifications"]').first();
  await expect(notifSec).toBeVisible({ timeout: 8_000 });
  await notifSec.click();
  // notify_fire_events is a checkbox switch; target the visible field container
  const notifyField = page.locator('.wd-field-switch:has(input[data-opt="notify_fire_events"])').first();
  await expect(notifyField).toBeVisible({ timeout: 8_000 });
});

test('revert button appears when changes have been staged', async ({ page }) => {
  const nameInput = page.locator('input[data-opt="name"]').first();
  await expect(nameInput).toBeVisible({ timeout: 8_000 });
  await nameInput.fill('Changed Name');
  const revertBtn = page.locator('#wd-settings-revert').first();
  await expect(revertBtn).toBeVisible({ timeout: 3_000 });
});

// ─── Profile evidence (which cycles shape a program) ──────────────────────────
//
// A `checkboxlist` field: several checkboxes writing ONE option as a list of the ticked
// values. The inner boxes carry data-choice, never data-opt, so the save collector must
// see a single field - if that ever regresses, three bogus option keys get persisted.

async function openProfileEvidence(page) {
  await rebootAdvanced(page);
  const sec = page.locator('button[data-sec="matching"]').first();
  await expect(sec).toBeVisible({ timeout: 8_000 });
  await sec.click();
  const field = page.locator('[data-opt="profile_evidence_sources"]').first();
  await expect(field).toBeVisible({ timeout: 8_000 });
  return field;
}

test('profile evidence renders one checkbox per cycle category, all ticked by default', async ({ page }) => {
  const field = await openProfileEvidence(page);
  await expect(field.locator('[data-choice]')).toHaveCount(3);
  for (const choice of ['real_cycles', 'reference_cycles', 'backfill_cycles']) {
    await expect(field.locator(`[data-choice="${choice}"]`)).toBeChecked();
  }
});

test('unticking a category saves the option as the remaining list', async ({ page }) => {
  const field = await openProfileEvidence(page);
  // The input itself is visually hidden behind the switch slider, as every other
  // switch in this panel is; the label is the clickable affordance.
  await field.locator('label:has([data-choice="backfill_cycles"])').click();
  await expect(field.locator('[data-choice="backfill_cycles"]')).not.toBeChecked();

  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;
  expect(options.profile_evidence_sources).toEqual(['real_cycles', 'reference_cycles']);
  // One option key, not one per checkbox.
  expect(options).not.toHaveProperty('real_cycles');
  expect(options).not.toHaveProperty('backfill_cycles');
});

test('unticking every evidence category saves the full default set, not an empty list', async ({ page }) => {
  // An empty evidence selection is not a valid "none": the backend silently falls back
  // to all three (a profile with no cycles can never match), so persisting [] would
  // leave the UI showing all-unchecked while matching used every source. Saving must
  // normalise an empty pick back to the default set.
  const field = await openProfileEvidence(page);
  for (const choice of ['real_cycles', 'reference_cycles', 'backfill_cycles']) {
    await field.locator(`label:has([data-choice="${choice}"])`).click();
    await expect(field.locator(`[data-choice="${choice}"]`)).not.toBeChecked();
  }

  await page.locator('#wd-settings-save').first().click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  const options = calls[calls.length - 1].options as Record<string, unknown>;
  expect(options.profile_evidence_sources).toEqual(
    ['real_cycles', 'reference_cycles', 'backfill_cycles'],
  );
});

test('leaving Settings with an unsaved edit asks before discarding it (audit UI-15)', async ({ page }) => {
  await clickTab(page, 'settings');
  const inp = page.locator('input[data-opt="min_power"]').first();
  await expect(inp).toBeVisible({ timeout: 8_000 });
  await inp.fill('3.5');
  await inp.dispatchEvent('change');
  await page.locator('button.wd-tab[data-tab="history"]').click();
  // Still on Settings, with a confirm.
  await expect(page.locator('.wd-modal h2')).toContainText(/unsaved/i);
  await page.locator('[data-maction="cancel"]').click();
  await expect(page.locator('button.wd-tab[data-tab="settings"].active')).toBeVisible();
  await expect(page.locator('input[data-opt="min_power"]').first()).toHaveValue('3.5');
  // Confirming discards and leaves.
  await page.locator('button.wd-tab[data-tab="history"]').click();
  await page.locator('[data-maction="ok"]').click();
  await expect(page.locator('button.wd-tab[data-tab="history"].active')).toBeVisible();
});

test('leaving Settings without edits does not ask', async ({ page }) => {
  await clickTab(page, 'settings');
  await expect(page.locator('input[data-opt="min_power"]').first()).toBeVisible({ timeout: 8_000 });
  await page.locator('button.wd-tab[data-tab="history"]').click();
  await expect(page.locator('button.wd-tab[data-tab="history"].active')).toBeVisible();
  await expect(page.locator('.wd-modal')).toHaveCount(0);
});
