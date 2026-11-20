import { test, expect } from '@playwright/test';
import { assertWsCalled, bootPanel, clickTab, setHandler } from '../helpers/panel';

// PR #529 review, round 40c.

const SUGS = [
  { key: 'off_delay', suggested: 371, current: 120, reason: 'a' },
  { key: 'min_power', suggested: 3, current: 2, reason: 'b' },
];

test('Apply refused after the suggestions were cleared shows the list as it stands', async ({ page }) => {
  // cycles_changed comes after the clear: the old list stayed on screen and a
  // second Apply found nothing. The list is read again and the user is told to
  // analyse again.
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_suggestions': { suggestions: SUGS, locked_suggestions: [] },
  });
  await clickTab(page, 'settings');
  await expect(page.locator('[data-sugkey="off_delay"]').first()).toBeVisible({ timeout: 5_000 });
  await setHandler(page, 'ha_washdata/get_suggestions', { suggestions: [], locked_suggestions: [] });
  await page.evaluate(() => (window as any).__set_error('ha_washdata/apply_suggestions', {
    code: 'cycles_changed',
    message: 'A cycle was stored while the suggestions were applied, so the timing settings '
      + 'were not checked against it. The suggestions were cleared, but no setting was saved; '
      + 'try again once they are offered again.',
  }));
  const before = (await assertWsCalled(page, 'ha_washdata/get_suggestions')).length;

  await page.locator('.wd-sug-banner [data-action="sug-apply-all"]').first().click();
  await page.locator('[data-maction="ok"]').click();

  await expect(page.locator('.wd-toast')).toContainText('Run the suggestion analysis again', { timeout: 5_000 });
  await assertWsCalled(page, 'ha_washdata/get_suggestions', before + 1);
  await expect(page.locator('[data-sugkey="off_delay"]')).toHaveCount(0);
  await expect(page.locator('.wd-sug-banner [data-action="sug-apply-all"]')).toHaveCount(0);
});
