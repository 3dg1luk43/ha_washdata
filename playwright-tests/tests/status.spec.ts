/**
 * Status tab (Overview) tests.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';
import deviceRunning from '../fixtures/mock-data/device-running.json';
import deviceIdle from '../fixtures/mock-data/device-idle.json';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('status tab shows device name', async ({ page }) => {
  await bootPanel(page);
  await expect(page.locator('text=Test Washer').first()).toBeVisible();
});

test('status tab shows idle state chip when device is idle', async ({ page }) => {
  await bootPanel(page);
  // The state chip contains the state label. Look for a badge element.
  const stateBadge = page.locator('.wd-badge, .wd-chip, [class*="state"]').first();
  await expect(stateBadge).toBeVisible({ timeout: 5_000 });
});

test('status tab shows running state and program when cycle is active', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': deviceRunning,
  });
  // Cotton 40°C should be the selected value in the program selector
  const progSelect = page.locator('#wd-status-prog');
  await expect(progSelect).toHaveValue('Cotton 40°C', { timeout: 5_000 });
});

// ─── Program pre-arming on an idle appliance (#411) ──────────────────────────

test('picking a program while idle really sends the choice', async ({ page }) => {
  // The dropdown is offered on an idle appliance, and the backend used to drop
  // the write silently. Assert the command carries the picked program.
  await bootPanel(page, { 'ha_washdata/set_program': { success: true } });
  const progSelect = page.locator('#wd-status-prog');
  await expect(progSelect).toHaveValue('auto_detect', { timeout: 8_000 });
  await progSelect.selectOption('Cotton 40\u00B0C');
  const calls = await assertWsCalled(page, 'ha_washdata/set_program');
  expect(calls[0].program).toBe('Cotton 40\u00B0C');
});

test('an armed program is shown as such, not as a live match', async ({ page }) => {
  const armed = JSON.parse(JSON.stringify(deviceIdle));
  armed.devices[0].armed_program = 'Eco 60\u00B0C';
  await bootPanel(page, { 'ha_washdata/get_devices': armed });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Eco 60\u00B0C', { timeout: 8_000 });
  await expect(page.locator('.wd-prog-tag').first()).toContainText('next cycle');
});

test('progress bar is visible during a running cycle', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': deviceRunning,
  });
  await expect(page.locator('.wd-prog-bg, .wd-prog-fill').first()).toBeVisible({ timeout: 5_000 });
});

test('power curve canvas is present with live power data', async ({ page }) => {
  await bootPanel(page);
  // The status canvas should be present in the DOM
  const canvas = page.locator('#wd-status-canvas, canvas').first();
  await expect(canvas).toBeVisible({ timeout: 8_000 });
});

test('status tab fetches power history on load', async ({ page }) => {
  await bootPanel(page);
  await assertWsCalled(page, 'ha_washdata/get_power_history');
});

test('attention card with suggestions appears when device has suggestions', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        suggestions_count: 3,
        suggestion_keys: ['off_delay', 'stop_threshold_w', 'min_off_gap'],
      }],
    },
  });
  // An attention card with the suggestion count should appear
  const attnCard = page.locator('.wd-attn-card').filter({ hasText: '3' });
  await expect(attnCard).toBeVisible({ timeout: 5_000 });
});

test('clicking the suggestion attention card switches to settings tab', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        suggestions_count: 2,
        suggestion_keys: ['off_delay', 'stop_threshold_w'],
      }],
    },
  });
  const attnCard = page.locator('.wd-attn-card[data-action="goto-suggestions"]').first();
  await expect(attnCard).toBeVisible({ timeout: 5_000 });
  await attnCard.click();
  // Should end up on the settings tab
  const settingsTab = page.locator('button.wd-tab[data-tab="settings"].active');
  await expect(settingsTab).toBeVisible({ timeout: 5_000 });
});

test('the standby-above-stop card renders on the Status page (#445 cause 1)', async ({ page }) => {
  // Regression: the card was pushed onto `attn` AFTER `attnHtml` had already been
  // joined into a string, so it was computed, pushed and then never rendered.
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        standby_above_stop: {
          cycles_above: 5,
          cycles_checked: 8,
          idle_w: 3.4,
          stop_threshold_w: 2.56,
        },
      }],
    },
  });
  const card = page.locator('.wd-attn-card').filter({ hasText: '3.4' });
  await expect(card).toBeVisible({ timeout: 5_000 });
  await expect(card).toContainText('2.56');
});

test('no standby-above-stop card when the backend reports no pattern', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        standby_above_stop: null,
      }],
    },
  });
  await expect(page.locator('.wd-attn-card[data-action="goto-conflicts"]')).toHaveCount(0);
});

test('feedback attention card appears when device has pending feedbacks', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{
        ...require('../fixtures/mock-data/device-idle.json').devices[0],
        feedback_count: 2,
      }],
    },
  });
  const feedbackCard = page.locator('[data-action="goto-feedbacks"]');
  await expect(feedbackCard).toBeVisible({ timeout: 5_000 });
});

// Mobile responsiveness
test('status tab renders without horizontal overflow on mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 }); // iPhone 14
  await bootPanel(page);
  const body = page.locator('.wd-body');
  await expect(body).toBeVisible({ timeout: 5_000 });
  // Check no horizontal scrollbar (scrollWidth == clientWidth)
  const overflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const body = el.shadowRoot.querySelector('.wd-body');
    return body ? body.scrollWidth - body.clientWidth : 0;
  });
  expect(overflow).toBeLessThanOrEqual(1); // Allow 1px rounding
});

// ─── Envelope position (item 269) ────────────────────────────────────────────
//
// The one "how far through" figure that is not derived from elapsed time. The
// manager computed it, used it for the Smart Termination release and threw it
// away; it now reaches the progress row beside the time-based percentage, which
// is exactly where the two disagreeing is informative rather than confusing.

test('the matched-curve position renders beside the time-based progress', async ({ page }) => {
  const dev = JSON.parse(JSON.stringify(deviceRunning));
  dev.devices[0].envelope_position = 0.87;
  await bootPanel(page, { 'ha_washdata/get_devices': dev });

  const row = page.locator('.wd-prog-row').first();
  await expect(row).toBeVisible({ timeout: 8_000 });
  await expect(row).toContainText('45.2%');
  // Deliberately not the same number: 45% of the expected time, 87% of the way
  // along the matched curve, is the overrun the profile alignment can see.
  await expect(row).toContainText('curve 87%');
  await expect(row.locator('span[title]', { hasText: 'curve 87%' }))
    .toHaveAttribute('title', /while the appliance is quiet/);
});

test('no curve position is shown before an alignment has run', async ({ page }) => {
  // Every cycle starts this way: the verification only runs below the stop
  // threshold, so a fresh run has nothing measured yet and must show nothing
  // rather than a placeholder 0%.
  await bootPanel(page, { 'ha_washdata/get_devices': deviceRunning });
  const row = page.locator('.wd-prog-row').first();
  await expect(row).toBeVisible({ timeout: 8_000 });
  await expect(row).not.toContainText('curve');
});

test('a zero curve position is rendered, not swallowed as falsy', async ({ page }) => {
  const dev = JSON.parse(JSON.stringify(deviceRunning));
  dev.devices[0].envelope_position = 0.0;
  await bootPanel(page, { 'ha_washdata/get_devices': dev });
  await expect(page.locator('.wd-prog-row').first()).toContainText('curve 0%', { timeout: 8_000 });
});

// ─── An undecided match is shown, not silent (MATCH-DECIDE-15) ───────────────

function undecided(unc: Record<string, unknown> | null, program: string | null = null) {
  const dev = JSON.parse(JSON.stringify(deviceRunning));
  dev.devices[0].current_program = program;
  dev.devices[0].match_uncertainty = unc;
  return dev;
}

test('an undecided live match names the top two and how sure it is', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': undecided(
    { top: 'Cotton 40°C', runner_up: 'Eco 60°C', margin: 0.04, sure_pct: 45 }) });
  const unc = page.locator('.wd-prog-unc');
  await expect(unc).toBeVisible({ timeout: 8_000 });
  await expect(unc).toContainText('Uncertain: Cotton 40°C or Eco 60°C');
  await expect(unc).toContainText('~45% sure');
  // The dropdown still says Auto-detect: nothing was committed or relabelled.
  await expect(page.locator('#wd-status-prog')).toHaveValue('auto_detect');
});

test('a lone candidate reads as a maybe', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': undecided(
    { top: 'Quick 30°C', runner_up: null, margin: null, sure_pct: 60 }) });
  const unc = page.locator('.wd-prog-unc');
  await expect(unc).toContainText('Uncertain: maybe Quick 30°C', { timeout: 8_000 });
  await expect(unc).toContainText('~60% sure');
});

test('a decided match shows no uncertainty line', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': deviceRunning });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  await expect(page.locator('.wd-prog-unc')).toHaveCount(0);
});

test('profile names in the uncertainty line render as text, never as markup', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': undecided(
    { top: '<img src=x id=wd-xss>', runner_up: 'Eco', margin: 0.01, sure_pct: 25 }) });
  await expect(page.locator('.wd-prog-unc')).toContainText('<img src=x id=wd-xss>', { timeout: 8_000 });
  await expect(page.locator('#wd-xss')).toHaveCount(0);
});

// ─── Phase timeline uses the live phase sensor's scale ───────────────────────

// progress.current_phase maps progress onto max(last range end, the program's
// expected length), so partial ranges read at their real minutes (audit
// PROGRESS-10): Wash 0-30 / Rinse 30-60 min on a 100 min program.
const partialPhases = { phases: [
  { name: 'Wash', start: 0, end: 1800 },
  { name: 'Rinse', start: 1800, end: 3600 },
] };

function runningAt(progressPct: number, expectedS: number | null = 6000) {
  const dev = JSON.parse(JSON.stringify(deviceRunning));
  dev.devices[0].cycle_progress_pct = progressPct;
  dev.devices[0].expected_duration_s = expectedS;
  return dev;
}

test('the phase timeline reads partial ranges at their real minutes', async ({ page }) => {
  // Minute 45 of 100 is Rinse. Stretching the ranges over the cycle (the old
  // scale, the last range end) named Wash here.
  await bootPanel(page, {
    'ha_washdata/get_devices': runningAt(45),
    'ha_washdata/get_profile_phases': partialPhases,
  });
  await expect(page.locator('.wd-ptl-cur')).toContainText('Rinse', { timeout: 8_000 });
});

test('the phase timeline names no phase past the last range', async ({ page }) => {
  // Minute 80 of 100 is after Rinse ends at 60: no phase, as the sensor says.
  await bootPanel(page, {
    'ha_washdata/get_devices': runningAt(80),
    'ha_washdata/get_profile_phases': partialPhases,
  });
  await expect(page.locator('.wd-ptl')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('.wd-ptl-cur')).toHaveCount(0);
});

test('ranges longer than the program keep their own end', async ({ page }) => {
  // The span is the LONGER of the two: ranges to 1000 s on a 600 s program map
  // 45.2% to 452 s, which is Spin.
  await bootPanel(page, {
    'ha_washdata/get_devices': runningAt(45.2, 600),
    'ha_washdata/get_profile_phases': { phases: [
      { name: 'Wash', start: 0, end: 400 },
      { name: 'Spin', start: 400, end: 1000 },
    ] },
  });
  await expect(page.locator('.wd-ptl-cur')).toContainText('Spin', { timeout: 8_000 });
});
