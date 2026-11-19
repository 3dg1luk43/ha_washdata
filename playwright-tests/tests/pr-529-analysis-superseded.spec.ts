import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, setHandler } from '../helpers/panel';

// PR #529 review: a manual "Analyze now" superseded on every run (options saved,
// stored data replaced) is refused with code `analysis_superseded` and an English
// message. The panel's WS wrapper shows it in the user's language, with the
// canonical English text when no translation is loaded.
test('a superseded analysis is reported in the panel language', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  const out = await page.evaluate(async () => {
    const el: any = document.querySelector('ha-washdata-panel');
    const orig = el._hass.connection.sendMessagePromise;
    el._hass.connection.sendMessagePromise = (msg: any) =>
      msg.type === 'ha_washdata/run_suggestion_analysis'
        ? Promise.reject({ code: 'analysis_superseded', message: 'raw backend text' })
        : orig(msg);
    const msgs: string[] = [];
    const run = async () => {
      try { await el._ws({ type: 'ha_washdata/run_suggestion_analysis' }); } catch (e: any) { msgs.push(`${e.code}:${e.message}`); }
    };
    el._panelTrans = { en: { msg: { analysis_superseded: 'CHANGED, AGAIN' } } };
    await run();
    el._panelTrans = { en: {} };
    await run();
    return msgs;
  });
  expect(out).toEqual([
    'analysis_superseded:CHANGED, AGAIN',
    'analysis_superseded:Settings or cycle data changed during the analysis. Run it again.',
  ]);
});

// PR #529 review: the manual analysis's joint timing check could not run. Passes
// before it may have stored suggestions, so the panel neither says nothing was
// saved (as Apply all's end_gate_check_failed does) nor keeps the old list.
test('a failed timing check reports the analysis incomplete and shows what it stored', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await clickTab(page, 'settings');
  await expect(page.locator('[data-action="sug-analyze"]')).toBeVisible({ timeout: 5_000 });
  await setHandler(page, 'ha_washdata/get_suggestions', {
    suggestions: [{ key: 'off_delay', current: 120, suggested: 600, reason: 'r' }],
    locked_suggestions: [],
  });
  await page.evaluate(() => (window as any).__set_error('ha_washdata/run_suggestion_analysis', {
    code: 'analysis_check_failed', message: 'joint end-gate check failed: scan broke',
  }));
  await page.locator('[data-action="sug-analyze"]').click();
  const toast = page.locator('.wd-toast');
  await expect(toast).toContainText(
    "The suggested timing settings could not be checked against this appliance's pauses. Run it again.",
    { timeout: 5_000 },
  );
  await expect(toast).not.toContainText('Nothing was saved');
  await expect(page.locator('.wd-sug-banner')).toBeVisible({ timeout: 5_000 });
});
