/**
 * Issue #453: the live-power chart's y-axis labels did not scale with the power
 * being shown.
 *
 * The tick label was `Math.round(value) + 'W'`, so an appliance idling at 0.3 W
 * got three identical "0W" labels stacked next to a perfectly legible curve. The
 * decimal count now follows the distance between ticks, so the labels always
 * resolve one grid step at any scale - without adding noise to a normal 2 kW
 * cycle, where whole watts are still what gets printed.
 *
 * Canvas text is not in the DOM, so these tests record `fillText` calls on the
 * real render path rather than inspecting pixels.
 */

import { test, expect, type Page } from '@playwright/test';
import { bootPanel } from '../helpers/panel';

const PANEL = 'ha-washdata-panel';

/** A live trace held flat at `watts`, in the [offset_s, watts] shape the WS sends. */
function flatTrace(watts: number, n = 20) {
  return {
    live: Array.from({ length: n }, (_, i) => [i * 30, watts]),
    raw: [],
    cycle_active: false,
    cycle_elapsed_s: 0,
    restart_gaps: [],
  };
}

/**
 * Redraw the status chart with `fillText` hooked, and return the watt labels it
 * painted (the axis gutter labels; the "N min" time label is filtered out).
 */
async function wattLabels(page: Page): Promise<string[]> {
  return page.evaluate((panel) => {
    const proto = CanvasRenderingContext2D.prototype;
    const orig = proto.fillText;
    const seen: string[] = [];
    proto.fillText = function (text: unknown, ...rest: unknown[]) {
      if (this.canvas && this.canvas.id === 'wd-status-canvas') seen.push(String(text));
      return (orig as (...a: unknown[]) => void).call(this, text, ...rest);
    };
    try {
      const root = document.querySelector(panel) as unknown as { _drawStatusCurve: () => void };
      root._drawStatusCurve();
    } finally {
      proto.fillText = orig;
    }
    return seen.filter((s) => /W$/.test(s));
  }, PANEL);
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('a standby-power trace gets distinct, non-zero axis labels', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_power_history': flatTrace(0.3) });

  const labels = await wattLabels(page);
  expect(labels.length).toBeGreaterThan(1);
  // The bug: every tick printed "0W".
  expect(new Set(labels).size).toBe(labels.length);
  expect(labels.filter((l) => l === '0W')).toHaveLength(1); // only the baseline
  // The top tick must actually resolve the 0.3 W the device is drawing.
  expect(parseFloat(labels[0])).toBeGreaterThan(0.2);
});

test('a normal cycle still gets whole-watt labels', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_power_history': flatTrace(2000) });

  const labels = await wattLabels(page);
  expect(labels.length).toBeGreaterThan(1);
  for (const l of labels) expect(l).toMatch(/^\d+W$/);
  expect(parseFloat(labels[0])).toBeGreaterThanOrEqual(2000);
});
