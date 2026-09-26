/**
 * Settings: the effective end wait is visible (#445).
 *
 * CycleDetector waits `max(off_delay, min_off_gap)`. `min_off_gap` was not in the
 * backend's resolved-defaults payload, so an unset one rendered as an empty field
 * and every cross-field rule's `!= null` guard short-circuited - nothing anywhere
 * showed the number actually governing the end of a cycle. The #445 reporter set
 * off_delay to 180 s, waited six minutes and force-stopped three cycles because
 * their washing machine was really waiting 480 s.
 *
 * This is a note, not a conflict: the configuration is legitimate, so it must not
 * highlight the field or feed the section conflict dots.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import optionsData from '../fixtures/mock-data/options.json';

const DEFAULTS = {
  sampling_interval: 2,
  watchdog_interval: 30,
  start_duration_threshold: 5,
  smart_termination_duration_ratio: 0.98,
  min_off_gap: 480,
  off_delay: 180,
};

/** The reporter's shape: off_delay lowered by hand, min_off_gap never set. */
function optionsWithoutGap(overrides: Record<string, unknown> = {}) {
  const o: Record<string, unknown> = { ...optionsData, ...overrides };
  delete o.min_off_gap;
  return o;
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('an unset min_off_gap above off_delay is disclosed as the real wait', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: optionsWithoutGap({ off_delay: 180 }),
      defaults: DEFAULTS,
    },
  });
  await clickTab(page, 'settings');

  const note = page.locator('[data-cnote="off_delay"]');
  await expect(note).toBeVisible({ timeout: 8_000 });
  await expect(note).toContainText('480');
  await expect(note).toContainText('180');

  // A note is not a conflict: no field highlight, no error slot.
  await expect(page.locator('.wd-field[data-field="off_delay"].wd-has-conflict')).toHaveCount(0);
  await expect(page.locator('[data-cerr="off_delay"]:visible')).toHaveCount(0);
});

test('no note when off_delay already governs the wait', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: optionsWithoutGap({ off_delay: 600 }),
      defaults: DEFAULTS,
    },
  });
  await clickTab(page, 'settings');
  await expect(page.locator('input[data-opt="off_delay"]').first()).toHaveValue('600', {
    timeout: 8_000,
  });
  await expect(page.locator('[data-cnote="off_delay"]:visible')).toHaveCount(0);
});

test('the note clears live once off_delay is raised past the gap', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: optionsWithoutGap({ off_delay: 180 }),
      defaults: DEFAULTS,
    },
  });
  await clickTab(page, 'settings');

  const note = page.locator('[data-cnote="off_delay"]');
  await expect(note).toBeVisible({ timeout: 8_000 });

  const offDelay = page.locator('input[data-opt="off_delay"]').first();
  await offDelay.fill('900');
  await offDelay.dispatchEvent('input');
  await expect(note).toBeHidden({ timeout: 3_000 });
});
