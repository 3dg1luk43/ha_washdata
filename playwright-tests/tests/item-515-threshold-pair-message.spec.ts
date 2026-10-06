import { test, expect } from '@playwright/test';
import { bootPanel } from '../helpers/panel';

// Register item 515: the backend refuses a write that would put the Stop Threshold
// at or above the Start Threshold with code `invalid_threshold_pair` and an English
// message. The panel's WS wrapper localises it once, so every save path shows it
// in the user's language with the two values.
test('an inverted threshold pair is reported in the panel language', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  const out = await page.evaluate(async () => {
    const el: any = document.querySelector('ha-washdata-panel');
    el._panelTrans = { en: { msg: { invalid_threshold_pair: 'STOP {stop} START {start}' } } };
    const orig = el._hass.connection.sendMessagePromise;
    el._hass.connection.sendMessagePromise = (msg: any) =>
      msg.type === 'ha_washdata/set_options'
        ? Promise.reject({ code: 'invalid_threshold_pair', message: 'Stop Threshold (6 W) must be below Start Threshold (2.3 W): x' })
        : msg.type === 'ha_washdata/other'
          ? Promise.reject({ code: 'invalid_format', message: 'raw' })
          : orig(msg);
    const msgs: string[] = [];
    for (const type of ['ha_washdata/set_options', 'ha_washdata/other']) {
      try { await el._ws({ type }); } catch (e: any) { msgs.push(`${e.code}:${e.message}`); }
    }
    return msgs;
  });
  expect(out).toEqual(['invalid_threshold_pair:STOP 6 START 2.3', 'invalid_format:raw']);
});
