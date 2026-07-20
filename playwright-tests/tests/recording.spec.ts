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
import { bootPanel } from '../helpers/panel';
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
