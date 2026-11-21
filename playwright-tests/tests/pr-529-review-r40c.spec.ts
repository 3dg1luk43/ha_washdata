import { test, expect } from '@playwright/test';
import { assertWsCalled, bootPanel, clickTab, setHandler } from '../helpers/panel';
import panelConfig from '../fixtures/mock-data/panel-config.json';

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

// The live chart's history: the tab load and the raw-power toggle wrote it with
// no request order, so an older poll reply still in flight replaced it.
async function slowFirstHistory(page: any, newer: string) {
  return page.evaluate(async (newer: string) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let release: () => void = () => {};
    const slow = new Promise<void>((r) => { release = r; });
    let asked: () => void = () => {};
    const pollAsked = new Promise<void>((r) => { asked = r; });
    let n = 0;
    el._ws = (msg: any) => {
      if (msg.type !== 'ha_washdata/get_power_history') return real(msg);
      n += 1;
      const answer = { live: [[0, 1], [5, 2]], raw: [], cycle_active: false, cycle_elapsed_s: 0, n };
      if (n === 1) { asked(); return slow.then(() => answer); }
      return Promise.resolve(answer);
    };
    el._tab = 'status';
    const older = el._fetchAll();   // the poll's read waits on the recorder
    await pollAsked;
    if (newer === 'tab') {
      await el._fetchTabData();
    } else {
      const box = el.shadowRoot.querySelector('input[data-statustoggle="show_raw_active"]');
      box.checked = true;
      box.dispatchEvent(new Event('change'));
      for (let i = 0; i < 50 && n < 2; i += 1) await new Promise((r) => setTimeout(r, 10));
      await new Promise((r) => setTimeout(r, 20));
    }
    const landed = el._powerData.n;
    release();
    await older;
    return { landed, after: el._powerData.n };
  }, newer);
}

test('the tab load\'s power history is not replaced by an older poll\'s', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {});
  await expect(page.locator('#wd-status-canvas')).toBeVisible({ timeout: 8_000 });
  expect(await slowFirstHistory(page, 'tab')).toEqual({ landed: 2, after: 2 });
});

test('the raw-power toggle\'s history is not replaced by an older poll\'s', async ({ page }) => {
  await page.goto('/');
  const cfg = JSON.parse(JSON.stringify(panelConfig));
  cfg.prefs.show_raw = true;
  await bootPanel(page, { 'ha_washdata/get_panel_config': cfg });
  await expect(page.locator('input[data-statustoggle="show_raw_active"]')).toBeVisible({ timeout: 8_000 });
  expect(await slowFirstHistory(page, 'toggle')).toEqual({ landed: 2, after: 2 });
});
