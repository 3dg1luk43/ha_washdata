/**
 * Discussion #452: a stalled cycle and an idle (display-on) appliance.
 *
 * A washer that halts mid-cycle on an unbalanced load sits at its 4-5 W standby
 * draw, above its stop threshold, so the cycle stays open. The integration now
 * shows it as `paused` with sub-state "Stalled" (and `cycle_anomaly: stalled`),
 * and between cycles a two-level appliance shows `idle` at its standby level.
 * These checks cover what the panel, the Playground and the card make of that.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, setHandler } from '../helpers/panel';
import { DEFAULT_HANDLERS } from '../helpers/ws-handlers';
import deviceRunning from '../fixtures/mock-data/device-running.json';

function device(overrides: Record<string, unknown>) {
  return { devices: [{ ...deviceRunning.devices[0], ...overrides }] };
}

test.describe('panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('a stalled cycle reads Paused (Stalled) and keeps its cycle controls', async ({ page }) => {
    await bootPanel(page, {
      'ha_washdata/get_devices': device({ detector_state: 'paused', sub_state: 'Stalled' }),
    });
    const badge = page.locator('.wd-badge').first();
    await expect(badge).toBeVisible({ timeout: 5_000 });
    await expect(badge).toContainText('(Stalled)');
    await expect(page.locator('[data-action="terminate-cycle"]')).toBeVisible();
    await expect(page.locator('[data-action="pause-cycle"]')).toBeVisible();
  });

  test('an idle appliance shows no sub-state and no cycle controls', async ({ page }) => {
    await bootPanel(page, {
      'ha_washdata/get_devices': device({ detector_state: 'idle', sub_state: 'Idle' }),
    });
    const badge = page.locator('.wd-badge').first();
    await expect(badge).toBeVisible({ timeout: 5_000 });
    await expect(badge).toContainText('Idle');
    await expect(badge).not.toContainText('(Idle)');
    await expect(page.locator('[data-action="terminate-cycle"]')).toHaveCount(0);
    await expect(page.locator('[data-action="pause-cycle"]')).toHaveCount(0);
  });

  test('the Playground strip shows a replayed stall', async ({ page }) => {
    await bootPanel(page, {});
    const base = DEFAULT_HANDLERS['ha_washdata/run_playground_cycle_detail'] as Record<string, any>;
    const series = [
      ...base.series.slice(0, 2),
      { ...base.series[1], t: 1700, power: 4.5, stalled: true },
    ];
    await setHandler(page, 'ha_washdata/run_playground_cycle_detail', {
      ...base,
      series,
      events: [...base.events.slice(0, 2), { t: 1500, type: 'stalled', detail: 'stalled at 4.5 W (display only)', severity: 'warn' }],
    });
    await clickTab(page, 'playground');
    await page.locator('button[data-action="pg-run"]').click();
    await expect(page.locator('.wd-pg-alerts-card')).toBeVisible({ timeout: 8_000 });
    await expect(page.locator('#wd-pg-state-badge')).toHaveText(/stalled/i, { timeout: 8_000 });
  });
});

test.describe('card', () => {
  const STALLED: Record<string, unknown> = {
    currency: 'EUR',
    stateColors: { paused: 'var(--warning-color, #ff9800)' },
    states: {
      'sensor.wm_state': {
        state: 'paused',
        attributes: { sub_state: 'Stalled', current_program_guess: 'Cotton 40', cycle_anomaly: 'stalled', overrun_ratio: 1.3 },
      },
    },
    entities: {
      'sensor.wm_state': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'washer_state' },
    },
    devices: { d1: { name: 'Washing Machine' } },
  };

  test.beforeEach(async ({ page }) => {
    await page.goto('/card.html');
    await page.waitForFunction(() => (window as any).__ready === true, { timeout: 10_000 });
  });

  for (const layout of ['tile', 'detail']) {
    test(`${layout}: a stall is not reported as running long`, async ({ page }) => {
      await page.evaluate(
        ({ c, d }: any) => (window as any).__mountCard(c, d),
        { c: { entity: 'sensor.wm_state', layout }, d: STALLED },
      );
      const text = await page.evaluate(() => {
        const sr = (window as any).__card.shadowRoot;
        return (sr.getElementById('state')?.textContent || '') + '|' + (sr.getElementById('meta')?.textContent || '');
      });
      expect(text).toContain('Stalled');
      expect(text.toLowerCase()).not.toContain('running long');
    });
  }
});
