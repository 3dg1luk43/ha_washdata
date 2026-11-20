import { test, expect, Page } from '@playwright/test';
import { bootPanel } from '../helpers/panel';
import deviceRunning from '../fixtures/mock-data/device-running.json';

// PR #529 review: the Status card's phase timeline caches the matched program's
// phase ranges per appliance and program (_ensureStatusPhases). Polling is frozen
// by bootPanel; these drive _fetchAll by hand, as the poll would.

function running(entryId: string, title: string, program: string) {
  const dev = JSON.parse(JSON.stringify(deviceRunning));
  Object.assign(dev.devices[0], {
    entry_id: entryId, title, current_program: program,
    cycle_progress_pct: 10, expected_duration_s: 3600,
  });
  return dev;
}

function phasesOf(name: string) {
  return { phases: [
    { name: `${name} wash`, start: 0, end: 1800 },
    { name: `${name} rinse`, start: 1800, end: 3600 },
  ] };
}

async function poll(page: Page): Promise<void> {
  await page.evaluate(() => (document.querySelector('ha-washdata-panel') as any)._fetchAll());
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test("another appliance on the same program never shows the last one's phases", async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': running('entry-a', 'Washer A', 'Eco'),
    'ha_washdata/get_profile_phases': phasesOf('A'),
  });
  await expect(page.locator('.wd-ptl-cur')).toContainText('A wash', { timeout: 8_000 });

  // The device list now holds B, also on Eco, at the selected index (an entry
  // removed from the list: no device switch). B's phase read is held back.
  await page.evaluate(({ devs, phases }) => {
    const w: any = window;
    w.__ws_handlers['ha_washdata/get_devices'] = devs;
    w.__ws_handlers['ha_washdata/get_profile_phases'] = () =>
      new Promise((resolve) => { w.__releasePhases = () => resolve(phases); });
    const el: any = document.querySelector('ha-washdata-panel');
    w.__firstPoll = el._fetchAll();
  }, { devs: running('entry-b', 'Washer B', 'Eco'), phases: phasesOf('B') });
  await expect.poll(() => page.evaluate(() => (window as any).__get_calls('ha_washdata/get_profile_phases')
    .some((m: any) => m.entry_id === 'entry-b'))).toBe(true);

  // A second poll while B's read is out paints B's card: A's ranges are not B's.
  await poll(page);
  await expect(page.locator('text=Washer B').first()).toBeVisible();
  await expect(page.locator('.wd-ptl')).toHaveCount(0);

  // B's own ranges land with the read.
  await page.evaluate(async () => {
    const w: any = window;
    w.__releasePhases();
    await w.__firstPoll;
  });
  await poll(page);
  await expect(page.locator('.wd-ptl-cur')).toContainText('B wash', { timeout: 8_000 });
});
