import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, setHandler } from '../helpers/panel';

// PR #529 review: a full reload replaced the device while a mute was written, so
// the reply is flagged `reloaded`. The panel re-reads the live device's mute
// state instead of marking the setting muted on its own.
const SUG = { key: 'off_delay', suggested: 371, current: 180, reason: 'Test reason' };

async function muteAfterReload(page: any, liveLocked: string[], liveSuggestions: unknown[]) {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_suggestions': { suggestions: [SUG], locked_suggestions: [] } });
  await clickTab(page, 'settings');
  const mute = page.locator('.wd-field[data-field="off_delay"] .wd-sug [data-suglock="off_delay"]').first();
  await expect(mute).toBeVisible({ timeout: 8_000 });
  await setHandler(page, 'ha_washdata/set_suggestion_lock', {
    success: true, locked_suggestions: liveLocked, reloaded: true,
  });
  await setHandler(page, 'ha_washdata/get_suggestions', {
    suggestions: liveSuggestions, locked_suggestions: liveLocked,
  });
  await mute.click();
}

test('a mute the reloaded device holds says muted', async ({ page }) => {
  await muteAfterReload(page, ['off_delay'], []);
  await expect(page.locator('.wd-toast')).toContainText("Won't suggest this setting again", { timeout: 5_000 });
  await expect(page.locator('[data-suglock="off_delay"]')).toHaveCount(0);
});

test('a mute the reloaded device does not hold is not reported as muted', async ({ page }) => {
  await muteAfterReload(page, [], [SUG]);
  await expect(page.locator('.wd-toast')).toContainText('Could not mute suggestion', { timeout: 5_000 });
  await expect(page.locator('.wd-field[data-field="off_delay"] .wd-sug [data-suglock="off_delay"]').first())
    .toBeVisible();
});

// "Reset muted" across a full reload: the live device may have loaded the file
// before an unlock landed, so success is said only when it holds none of them.
async function resetMutedAfterReload(page: any, liveLocked: string[]) {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': { suggestions: [], locked_suggestions: ['off_delay', 'min_power'] },
  });
  await clickTab(page, 'settings');
  const reset = page.locator('[data-action="sug-unmute-all"]').first();
  await expect(reset).toBeVisible({ timeout: 8_000 });
  await setHandler(page, 'ha_washdata/set_suggestion_lock', {
    success: true, locked_suggestions: liveLocked, reloaded: true,
  });
  await setHandler(page, 'ha_washdata/get_suggestions', { suggestions: [], locked_suggestions: liveLocked });
  await reset.click();
}

test('a reset the reloaded device did not take is not reported as done', async ({ page }) => {
  await resetMutedAfterReload(page, ['off_delay']);
  await expect(page.locator('.wd-toast')).toContainText('Reset did not complete', { timeout: 5_000 });
  await expect(page.locator('[data-action="sug-unmute-all"]').first()).toBeVisible();
});

test('a reset the reloaded device holds says reset', async ({ page }) => {
  await resetMutedAfterReload(page, []);
  await expect(page.locator('.wd-toast')).toContainText('Muted suggestions reset', { timeout: 5_000 });
  await expect(page.locator('[data-action="sug-unmute-all"]')).toHaveCount(0);
});
