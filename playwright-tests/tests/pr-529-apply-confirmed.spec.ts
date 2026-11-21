import { test, expect } from '@playwright/test';
import { assertWsCalled, bootPanel, clickTab, setHandler } from '../helpers/panel';

// PR #529 review: Apply all sends the values its preview showed, and a suggestion
// that changed since then is refused with the list as it now stands, so the user
// confirms the new value instead of it being saved unseen.
const SUGS = [
  { key: 'off_delay', suggested: 371, current: 120, reason: 'a' },
  { key: 'min_power', suggested: 3, current: 2, reason: 'b' },
];

async function applyAll(page: any) {
  await page.locator('.wd-sug-banner [data-action="sug-apply-all"]').first().click();
  await page.locator('[data-maction="ok"]').click();
}

test('Apply all sends the values the preview showed', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': { suggestions: SUGS, locked_suggestions: [] },
    'ha_washdata/apply_suggestions': { success: true, applied: ['off_delay', 'min_power'] },
  });
  await clickTab(page, 'settings');
  await applyAll(page);
  await expect(page.locator('.wd-toast')).toContainText('Suggestions applied', { timeout: 5_000 });
  const [call] = await assertWsCalled(page, 'ha_washdata/apply_suggestions');
  expect(call.keys).toEqual(['off_delay', 'min_power']);
  expect(call.values).toEqual({ off_delay: 371, min_power: 3 });
});

test('a suggestion changed since the preview is shown again, not applied', async ({ page }) => {
  const changed = [{ ...SUGS[0], suggested: 400 }, SUGS[1]];
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': { suggestions: SUGS, locked_suggestions: [] },
    'ha_washdata/apply_suggestions': {
      success: false, applied: [], changed: ['off_delay'],
      suggestions: changed, locked_suggestions: [],
    },
  });
  await clickTab(page, 'settings');
  await applyAll(page);
  await expect(page.locator('.wd-toast')).toContainText('The suggestions changed', { timeout: 5_000 });
  await expect(page.locator('.wd-toast')).not.toContainText('applied');
  await expect(page.locator('[data-sugkey="off_delay"]').first()).toHaveAttribute('data-sugval', '400');

  // The next Apply previews and confirms the new value.
  await setHandler(page, 'ha_washdata/apply_suggestions', { success: true, applied: ['off_delay', 'min_power'] });
  await page.locator('.wd-sug-banner [data-action="sug-apply-all"]').first().click();
  await expect(page.locator('.wd-confirm-list li').first()).toContainText('400', { timeout: 5_000 });
  await page.locator('[data-maction="ok"]').click();
  await expect(page.locator('.wd-toast')).toContainText('Suggestions applied', { timeout: 5_000 });
  const calls = await assertWsCalled(page, 'ha_washdata/apply_suggestions', 2);
  expect(calls[1].values).toEqual({ off_delay: 400, min_power: 3 });
});
