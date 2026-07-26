/**
 * Issue #331 - a cycle awaiting detection feedback ("must be reviewed") must
 * expose its resolve controls without the user having to discover Review mode,
 * and (re)labelling it must resolve the pending feedback backend-side.
 *
 * These tests cover the frontend half: the Confirm/Correct/Ignore banner now
 * renders in the default Inspect view, the needs-review filter surfaces the
 * cycle, and confirming/labelling issues the right WS commands. The backend
 * resolution itself is unit-tested in tests/test_issue_331_*.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled } from '../helpers/panel';

// A dishwasher-style pending feedback on cyc-001 (Cotton 40°C in the fixture),
// mirroring the issue: a low-confidence detection awaiting the user's answer.
const PENDING_FB = {
  feedbacks: [
    {
      cycle_id: 'cyc-001',
      detected_profile: 'Cotton 40°C',
      confidence: 0.55,
      user_response: null,
      created_at: '2026-07-23T15:32:38+02:00',
    },
  ],
};

const CYCLE_POWER = {
  power_data: [
    { t: 0, p: 0 },
    { t: 30, p: 820 },
    { t: 60, p: 750 },
  ],
  artifacts: [],
  envelope_conformance: null,
};

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_feedbacks': PENDING_FB,
    'ha_washdata/get_cycle_power_data': CYCLE_POWER,
  });
});

test('needs-review filter surfaces the pending-feedback cycle (#331)', async ({ page }) => {
  await clickTab(page, 'history');
  const statusSel = page.locator('#wd-cyc-filter-status');
  await expect(statusSel).toBeVisible({ timeout: 5_000 });
  await statusSel.selectOption('needs_review');
  await expect(page.locator('tr[data-cid="cyc-001"]')).toBeVisible({ timeout: 5_000 });
});

test('review cycle exposes Confirm/Correct/Ignore in Inspect mode (#331)', async ({ page }) => {
  await clickTab(page, 'history');
  const row = page.locator('tr[data-cid="cyc-001"]');
  await expect(row).toBeVisible({ timeout: 5_000 });
  await row.click();

  const modal = page.locator('.wd-modal');
  await expect(modal).toBeVisible({ timeout: 5_000 });

  // A plain row click opens Inspect (view) mode - NOT Review mode. The resolve
  // controls must be right here, so the "obvious" path is not a dead end (#331).
  await expect(modal.getByText('Pending detection feedback')).toBeVisible();
  await expect(modal.locator('button[data-action="fb-confirm"]')).toBeVisible();
  await expect(modal.locator('button[data-action="fb-correct"]')).toBeVisible();
  await expect(modal.locator('button[data-action="fb-ignore"]')).toBeVisible();
});

test('confirming from Inspect mode resolves the feedback (#331)', async ({ page }) => {
  await clickTab(page, 'history');
  await page.locator('tr[data-cid="cyc-001"]').click();
  await expect(page.locator('.wd-modal')).toBeVisible({ timeout: 5_000 });

  await page.locator('button[data-action="fb-confirm"]').click();

  const calls = await assertWsCalled(page, 'ha_washdata/resolve_feedback', 1);
  expect(calls[0].cycle_id).toBe('cyc-001');
  expect(calls[0].action).toBe('confirm');
});
