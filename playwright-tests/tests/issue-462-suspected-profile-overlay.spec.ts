/**
 * #462: since 0.5.7's label margin an uncertain cycle stays UNLABELLED while its
 * pending feedback still names the profile the matcher suspected. The cycle
 * dialog fetched the overlay curve only for `profile_name`, so the one case that
 * needs a visual comparison showed the banner but no profile curve.
 *
 * Now the dialog falls back to the feedback's `detected_profile`, draws it dashed
 * and names it "(suspected)" in the legend, adds the feedback ranking's runner-up,
 * and matches each envelope response against the name the modal asked for.
 */

import { test, expect, Page } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled } from '../helpers/panel';

const PANEL = 'ha-washdata-panel';

// cyc-003 is the fixture's unlabelled cycle; cyc-001 is labelled Cotton 40°C.
const FEEDBACK = {
  feedbacks: [{
    cycle_id: 'cyc-003',
    detected_profile: 'Eco 60°C',
    confidence: 0.611,
    user_response: null,
    created_at: '2026-07-23T15:32:38+02:00',
    // learning.request_cycle_verification stores the top-5 ranking here.
    ranking: [
      { name: 'Eco 60°C', score: 0.611, metrics: {}, profile_duration: 1800 },
      { name: 'Quick 30°C', score: 0.54, metrics: {}, profile_duration: 1700 },
    ],
  }],
};

// One flat level per profile, so the drawn series identify their envelope.
const LEVEL: Record<string, number> = { 'Eco 60°C': 600, 'Quick 30°C': 300, 'Cotton 40°C': 1200 };

/**
 * Per-cycle power data and per-profile envelopes. `held` names envelopes whose
 * response waits for `window.__w462_release(name)`, to put a reply in flight.
 */
async function installHandlers(page: Page, held: string[] = []) {
  await page.evaluate(({ levels, held }) => {
    const w = window as any;
    const env = (name: string) => {
      const pts = Array.from({ length: 10 }, (_, i) => [i * 180, levels[name]]);
      return { avg: pts, min: pts, max: pts, target_duration: 1620 };
    };
    const gates: Record<string, () => void> = {};
    w.__w462_release = (name: string) => { if (gates[name]) gates[name](); };
    w.__ws_handlers['ha_washdata/get_profile_envelope'] = (msg: any) => {
      const name = msg.profile_name;
      const body = { envelope: name in levels ? env(name) : null };
      if (!held.includes(name)) return body;
      return new Promise(res => { gates[name] = () => res(body); });
    };
    w.__ws_handlers['ha_washdata/get_cycle_power_data'] = (msg: any) => ({
      cycle_id: msg.cycle_id,
      samples: Array.from({ length: 30 }, (_, i) => [i * 60, i < 2 || i > 27 ? 3 : 900]),
      sample_count: 30, decimated: false, full_duration_s: 1740,
      profile_name: msg.cycle_id === 'cyc-001' ? 'Cotton 40°C' : null,
      status: 'completed', artifacts: [], restart_gaps: [], is_reference: false,
    });
  }, { levels: LEVEL, held });
}

/** The series the cycle chart last drew (name + first y value), Power excluded. */
async function drawnOverlays(page: Page): Promise<{ name: string; w: number }[]> {
  return page.evaluate(`(() => {
    const sr = document.querySelector('${PANEL}').shadowRoot;
    const c = sr.getElementById('wd-cyc-canvas');
    if (!c || !c._wd) return [];
    return c._wd.series.filter(s => s.stroke !== 'primary')
      .map(s => ({ name: s.name, w: s.points && s.points[0] ? s.points[0][1] : null }));
  })()`) as Promise<{ name: string; w: number }[]>;
}

async function openCycle(page: Page, cid: string) {
  await clickTab(page, 'history');
  const row = page.locator(`tr[data-cid="${cid}"]`);
  await expect(row).toBeVisible({ timeout: 5_000 });
  await row.click();
  await expect(page.locator('#wd-cyc-canvas')).toBeVisible({ timeout: 5_000 });
}

async function bootWith(page: Page, feedbacks: unknown, held: string[] = []) {
  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_feedbacks': feedbacks });
  await installHandlers(page, held);
}

test('unlabelled cycle with pending feedback overlays the suspected profile and runner-up (Inspect)', async ({ page }) => {
  await bootWith(page, FEEDBACK);
  await openCycle(page, 'cyc-003');
  const modal = page.locator('.wd-modal');
  await expect(modal.getByText('Pending detection feedback')).toBeVisible();

  const legend = modal.locator('[data-cyc-legend]');
  await expect(legend.locator('[data-leg="suspected"]')).toHaveText('Eco 60°C (suspected)');
  await expect(legend.locator('[data-leg="runner_up"]')).toHaveText('Quick 30°C (runner-up)');
  // Never presented as the cycle's label.
  await expect(legend.locator('[data-leg="expected"]')).toHaveCount(0);
  await expect.poll(() => drawnOverlays(page)).toEqual([
    { name: 'Eco 60°C (suspected)', w: 600 },
    { name: 'Quick 30°C (runner-up)', w: 300 },
  ]);
  const calls = await assertWsCalled(page, 'ha_washdata/get_profile_envelope', 2);
  expect(calls.map(c => c.profile_name).sort()).toEqual(['Eco 60°C', 'Quick 30°C']);
});

test('the suspected overlay lands in Review mode too, even when the reply arrives after the switch', async ({ page }) => {
  // Hold both envelopes so the replies arrive after Review re-rendered the form
  // (which rewrites curve.profile_name from the unlabelled select).
  await bootWith(page, FEEDBACK, ['Eco 60°C', 'Quick 30°C']);
  await openCycle(page, 'cyc-003');
  await page.locator('button[data-maction="cyc-review"]').click();
  await expect(page.locator('#wd-cyc-rev-label')).toBeVisible();
  await page.locator('.wd-cyc-rev-tag').first().check();  // a re-render that snapshots the form
  await page.evaluate(() => { (window as any).__w462_release('Eco 60°C'); (window as any).__w462_release('Quick 30°C'); });

  const legend = page.locator('.wd-modal [data-cyc-legend]');
  await expect(legend.locator('[data-leg="suspected"]')).toHaveText('Eco 60°C (suspected)');
  await expect(legend.locator('[data-leg="runner_up"]')).toHaveText('Quick 30°C (runner-up)');
  await expect.poll(() => drawnOverlays(page)).toEqual([
    { name: 'Eco 60°C (suspected)', w: 600 },
    { name: 'Quick 30°C (runner-up)', w: 300 },
  ]);
});

test('an unlabelled cycle without pending feedback gets no profile overlay', async ({ page }) => {
  await bootWith(page, { feedbacks: [] });
  await openCycle(page, 'cyc-003');
  await expect(page.locator('.wd-modal [data-cyc-legend]')).toHaveCount(0);
  await assertWsNotCalled(page, 'ha_washdata/get_profile_envelope');
  expect(await drawnOverlays(page)).toEqual([]);
});

test('a labelled cycle still overlays its own profile as Expected', async ({ page }) => {
  await bootWith(page, FEEDBACK);
  await openCycle(page, 'cyc-001');
  const legend = page.locator('.wd-modal [data-cyc-legend]');
  await expect(legend.locator('[data-leg="expected"]')).toHaveText('Expected (Cotton 40°C)');
  await expect(legend.locator('[data-leg="suspected"], [data-leg="runner_up"]')).toHaveCount(0);
  await expect.poll(() => drawnOverlays(page)).toEqual([{ name: 'Expected (Cotton 40°C)', w: 1200 }]);
});

test('confirming the feedback turns the suspected overlay into the label', async ({ page }) => {
  await bootWith(page, FEEDBACK);
  await openCycle(page, 'cyc-003');
  const legend = page.locator('.wd-modal [data-cyc-legend]');
  await expect(legend.locator('[data-leg="suspected"]')).toHaveText('Eco 60°C (suspected)');
  await page.evaluate(() => {
    const w = window as any;
    w.__ws_handlers['ha_washdata/resolve_feedback'] = { success: true };
    // The backend labels the cycle on confirm, and the feedback is gone.
    w.__ws_handlers['ha_washdata/get_feedbacks'] = { feedbacks: [] };
  });
  await page.locator('.wd-modal button[data-action="fb-confirm"]').click();
  await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  await expect(legend.locator('[data-leg="expected"]')).toHaveText('Expected (Eco 60°C)');
  await expect(legend.locator('[data-leg="suspected"], [data-leg="runner_up"]')).toHaveCount(0);
});

test('a suspected-profile reply that arrives after switching cycles is ignored', async ({ page }) => {
  await bootWith(page, FEEDBACK, ['Eco 60°C', 'Quick 30°C']);
  await openCycle(page, 'cyc-003');
  await page.locator('.wd-modal button[data-maction="cancel"]').first().click();
  await expect(page.locator('.wd-modal')).toHaveCount(0);
  await openCycle(page, 'cyc-001');
  const legend = page.locator('.wd-modal [data-cyc-legend]');
  await expect(legend.locator('[data-leg="expected"]')).toHaveText('Expected (Cotton 40°C)');
  // The first cycle's replies land now; they must not repaint this cycle.
  await page.evaluate(() => { (window as any).__w462_release('Eco 60°C'); (window as any).__w462_release('Quick 30°C'); });
  await page.waitForTimeout(300);
  expect(await drawnOverlays(page)).toEqual([{ name: 'Expected (Cotton 40°C)', w: 1200 }]);
  await expect(legend.locator('[data-leg]')).toHaveCount(1);
});
