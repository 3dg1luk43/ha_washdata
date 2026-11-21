/**
 * Manual recording display (issue #313).
 *
 * While a manual recording is active the CycleDetector is frozen at OFF (readings
 * go to the recorder, not the detector), so the raw detector state leaked into
 * the panel as a "Recording (Off)" badge, and the Manual Recording widget -- which
 * trusted a separately-polled _recState that is null on first load -- showed
 * "Start Recording" during an active recording. Both must reflect the
 * authoritative dev.recording flag from get_devices.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, setHandler } from '../helpers/panel';
import deviceIdle from '../fixtures/mock-data/device-idle.json';

function recordingDevice(overrides: Record<string, unknown> = {}) {
  return {
    devices: [
      {
        ...deviceIdle.devices[0],
        recording: true,
        // Detector is frozen at OFF during recording -> raw sub_state is "Off".
        detector_state: 'off',
        sub_state: 'Off',
        ...overrides,
      },
    ],
  };
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('status badge shows "Recording" without a frozen "(Off)" suffix', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': recordingDevice() });
  const badge = page.locator('.wd-badge').first();
  await expect(badge).toContainText('Recording', { timeout: 5_000 });
  await expect(badge).not.toContainText('(Off)');
});

test('recording widget shows Stop, not Start Recording, when recState is stale/idle', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': recordingDevice(),
    // Simulate the first-load condition: recording detail source reports idle.
    'ha_washdata/get_recording_state': { state: 'idle', duration_s: 0, sample_count: 0 },
  });
  await expect(page.locator('[data-action="rec-stop"]')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('[data-action="rec-start"]')).toHaveCount(0);
});

test('idle device does not show a redundant "(Off)" sub-state suffix', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': {
      devices: [{ ...deviceIdle.devices[0], detector_state: 'off', sub_state: 'Off' }],
    },
  });
  const badge = page.locator('.wd-badge').first();
  await expect(badge).toBeVisible({ timeout: 5_000 });
  await expect(badge).not.toContainText('(Off)');
});

// PR #529 review: the read after Stop or Discard fails. The poll reads the
// recording state only while dev.recording is set, so after a Stop it never read
// it again and the widget kept offering Stop; after a Discard it kept Process
// for a discarded trace. A transition whose read failed is read again.
async function failRecState(page: any, on: boolean) {
  await page.evaluate((fail: boolean) => {
    const w = window as any;
    if (fail) w.__set_error('ha_washdata/get_recording_state', { code: 'unknown_error', message: 'x' });
    else delete w.__ws_errors['ha_washdata/get_recording_state'];
  }, on);
}

async function poll(page: any) {
  await page.evaluate(() => (document.querySelector('ha-washdata-panel') as any)._fetchAll());
}

test('a failed read after Stop is retried, and the widget offers Process', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': recordingDevice(),
    'ha_washdata/get_recording_state': { state: 'recording', duration_s: 60, sample_count: 5 },
    'ha_washdata/stop_recording': { success: true },
  });
  await expect(page.locator('[data-action="rec-stop"]')).toBeVisible({ timeout: 5_000 });
  await setHandler(page, 'ha_washdata/get_devices', recordingDevice({ recording: false }));
  await failRecState(page, true);
  await page.locator('[data-action="rec-stop"]').click();
  await expect(page.locator('.wd-toast')).toContainText('Recording stopped', { timeout: 5_000 });
  await poll(page);  // the read fails again: the last snapshot stays
  await failRecState(page, false);
  await setHandler(page, 'ha_washdata/get_recording_state', { state: 'stopped', duration_s: 600, sample_count: 12 });
  await poll(page);
  await expect(page.locator('[data-action="rec-process-open"]')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('[data-action="rec-stop"]')).toHaveCount(0);
});

test('a failed read after Discard is retried, and the widget no longer offers Process', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_devices': recordingDevice({ recording: false }),
    'ha_washdata/get_recording_state': { state: 'stopped', duration_s: 600, sample_count: 12 },
    'ha_washdata/discard_recording': { success: true },
  });
  // A held trace: the Status tab's own read shows it (the poll reads only while recording).
  await page.evaluate(() => (document.querySelector('ha-washdata-panel') as any)._fetchTabData());
  await expect(page.locator('[data-action="rec-discard"]')).toBeVisible({ timeout: 5_000 });
  await failRecState(page, true);
  await page.locator('[data-action="rec-discard"]').click();
  await page.getByRole('dialog', { name: 'Discard Recording' }).getByRole('button', { name: 'Discard' }).click();
  await expect(page.locator('.wd-toast')).toContainText('Recording discarded', { timeout: 5_000 });
  await failRecState(page, false);
  await setHandler(page, 'ha_washdata/get_recording_state', { state: 'idle', duration_s: 0, sample_count: 0 });
  await poll(page);
  await expect(page.locator('[data-action="rec-start"]')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('[data-action="rec-process-open"]')).toHaveCount(0);
});

// PR #529 review: the retry mark was one for the whole panel. Another appliance's
// Stop, acknowledged after the shown one's Discard read failed, replaced it; its own
// read is not applied to the shown appliance, so nothing read the shown one again
// and the widget kept offering Process for the discarded recording.
test('another appliance\'s late Stop does not cancel the shown one\'s retry', async ({ page }) => {
  const shown = { ...deviceIdle.devices[0], recording: false };
  const other = { ...shown, entry_id: 'test-entry-002', title: 'Dryer' };
  await bootPanel(page, {
    'ha_washdata/get_devices': { devices: [shown, other] },
    'ha_washdata/get_recording_state': { state: 'stopped', duration_s: 600, sample_count: 12 },
    'ha_washdata/discard_recording': { success: true },
  });
  await page.evaluate(() => (document.querySelector('ha-washdata-panel') as any)._fetchTabData());
  await expect(page.locator('[data-action="rec-discard"]')).toBeVisible({ timeout: 5_000 });
  await failRecState(page, true);
  await page.locator('[data-action="rec-discard"]').click();
  await page.getByRole('dialog', { name: 'Discard Recording' }).getByRole('button', { name: 'Discard' }).click();
  await expect(page.locator('.wd-toast')).toContainText('Recording discarded', { timeout: 5_000 });
  await failRecState(page, false);
  await setHandler(page, 'ha_washdata/get_recording_state', { state: 'idle', duration_s: 0, sample_count: 0 });
  // The Dryer's Stop acknowledgement lands now: its read is for an appliance not shown.
  const applied = await page.evaluate(
    () => (document.querySelector('ha-washdata-panel') as any)._fetchRecState('test-entry-002', true),
  );
  expect(applied).toBe(false);
  await poll(page);
  await expect(page.locator('[data-action="rec-start"]')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('[data-action="rec-process-open"]')).toHaveCount(0);
});
