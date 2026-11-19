import { test, expect } from '@playwright/test';
import { bootPanel } from '../helpers/panel';

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
