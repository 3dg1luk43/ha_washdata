import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, setHandler } from '../helpers/panel';
import profilesData from '../fixtures/mock-data/profiles.json';

// #158: a profile with cycles has its duration recomputed on every envelope rebuild,
// so the Expected Duration is shown read-only there and a plain rename no longer
// resends the rounded minutes. A profile without cycles stays editable.
async function openDangerTab(page: import('@playwright/test').Page, learned: boolean): Promise<void> {
  const rows = (profilesData as any).profiles.map((p: any, i: number) => ({ ...p, duration_learned: i === 0 ? learned : false }));
  await setHandler(page, 'ha_washdata/get_profiles', { ...profilesData, profiles: rows });
  await clickTab(page, 'profiles');
  await setHandler(page, 'ha_washdata/get_profile_cycles', { cycles: [] });
  await setHandler(page, 'ha_washdata/get_profile_envelope', { envelope: null });
  await setHandler(page, 'ha_washdata/rename_profile', { success: true });
  await page.locator('.wd-profile-card').first().click();
  const tab = page.locator('.wd-modal [data-maction="pp-tab-danger"]');
  await expect(tab).toBeVisible({ timeout: 5_000 });
  await tab.click();
  await expect(page.locator('.wd-modal #wd-pp-dur')).toBeVisible({ timeout: 5_000 });
}

async function renameCalls(page: import('@playwright/test').Page): Promise<any[]> {
  return page.evaluate(() => window.__get_calls('ha_washdata/rename_profile')) as any;
}

test('#158 a learned duration is read-only and a rename does not resend it', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await openDangerTab(page, true);
  const dur = page.locator('.wd-modal #wd-pp-dur');
  await expect(dur).toBeDisabled();
  await expect(page.locator('.wd-modal')).toContainText("Worked out from this program's cycles");
  await page.locator('.wd-modal #wd-pp-rename').fill('Cotton 40 renamed');
  await page.locator('.wd-modal [data-maction="pp-rename"]').click();
  await expect.poll(async () => (await renameCalls(page)).length).toBe(1);
  expect((await renameCalls(page))[0].manual_duration_min).toBeNull();
});

test('#158 without cycles the duration is editable and only an edit is sent', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  await openDangerTab(page, false);
  const dur = page.locator('.wd-modal #wd-pp-dur');
  await expect(dur).toBeEnabled();
  await page.locator('.wd-modal [data-maction="pp-rename"]').click();
  await expect.poll(async () => (await renameCalls(page)).length).toBe(1);
  expect((await renameCalls(page))[0].manual_duration_min).toBeNull();  // unchanged

  await page.locator('.wd-modal #wd-pp-dur').fill('95');
  await page.locator('.wd-modal [data-maction="pp-rename"]').click();
  await expect.poll(async () => (await renameCalls(page)).length).toBe(2);
  expect((await renameCalls(page))[1].manual_duration_min).toBe(95);
});

// #285 #296 #325: the detector counts anti-wrinkle quiet below the HIGHER of the exit
// power and the stop threshold, so no pair of the two is a conflict (the old rule
// pushed the exit power below stop, exactly where it has no effect).
test('#285 an exit power on either side of the stop threshold is not a conflict', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
  const keys = await page.evaluate(() => {
    const el: any = document.querySelector('ha-washdata-panel');
    const out: string[][] = [];
    for (const exit of [0.8, 1.2, 4.0]) {
      out.push([...el._conflictKeysForOpts({ device_type: 'dryer', anti_wrinkle_exit_power: exit, stop_threshold_w: 1.2, start_threshold_w: 5 }, {})]);
    }
    return out;
  });
  for (const k of keys) {
    expect(k).not.toContain('anti_wrinkle_exit_power');
    expect(k).not.toContain('stop_threshold_w');
  }
});
