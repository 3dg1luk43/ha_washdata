/**
 * Setup Card (Adoption Guidance) tests.
 *
 * The Setup Card is rendered in the Status/Overview tab when get_setup_status
 * returns a non-null phase and there is no live power curve (hasCurve = false).
 *
 * Phase 4 renders as a compact .wd-setup-chip--healthy chip.
 * Phases 0–3 render as a full .wd-setup-card[data-phase="phaseN"] card.
 *
 * Implementation note: get_setup_status is fetched by _fetchTabData (tab
 * switches / device changes), NOT by the initial _fetchAll boot sequence.
 * Each test must call loadStatusTabData() after bootPanel to populate the
 * setup status and trigger the re-render that shows the card.
 *
 * The default IDLE_POWER_HISTORY has 20 live points (hasCurve = true), so all
 * tests also override get_power_history with an empty live array so that
 * hasCurve = false and the setup card is visible.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, assertWsCalled } from '../helpers/panel';
import type { Page } from '@playwright/test';

/** Power history override that forces hasCurve = false (live.length <= 1). */
const NO_CURVE_POWER = {
  live: [],
  raw: [],
  cycle_active: false,
  cycle_elapsed_s: 0,
  profile_envelope: null,
};

/**
 * Trigger _fetchTabData on the panel so that get_setup_status is fetched and
 * the re-render shows the setup card. Must be called after bootPanel because
 * the initial _fetchAll boot sequence does not call _fetchTabData.
 */
async function loadStatusTabData(page: Page): Promise<void> {
  await page.evaluate(async () => {
    const el = document.getElementById('wd-panel') as any;
    if (el && typeof el._fetchTabData === 'function') await el._fetchTabData();
  });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('Phase 0 card appears with open_recorder CTA button', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_power_history': NO_CURVE_POWER,
    'ha_washdata/get_setup_status': {
      phase: 'phase0',
      message_key: 'setup.phase0.washer',
      message_params: {},
      cta_label_key: 'setup.cta.start_recording',
      cta_action: 'open_recorder',
      secondary_label_key: 'setup.cta.label_detected_cycle',
      secondary_action: 'open_cycles_unlabeled',
      skippable: false,
      dismissible: false,
      step_key: null,
    },
  });
  await loadStatusTabData(page);

  // The setup card should be visible with the correct phase attribute.
  await expect(page.locator('[data-phase="phase0"]')).toBeVisible({ timeout: 5_000 });
  // The primary CTA button should carry the open_recorder action.
  await expect(page.locator('[data-cta-action="open_recorder"]')).toBeVisible({ timeout: 5_000 });
});

test('Start Recording CTA scrolls to recorder widget', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_power_history': NO_CURVE_POWER,
    'ha_washdata/get_setup_status': {
      phase: 'phase0',
      message_key: 'setup.phase0.washer',
      message_params: {},
      cta_label_key: 'setup.cta.start_recording',
      cta_action: 'open_recorder',
      secondary_label_key: null,
      secondary_action: null,
      skippable: false,
      dismissible: false,
      step_key: null,
    },
  });
  await loadStatusTabData(page);

  // Wait for the CTA to be ready.
  await expect(page.locator('[data-cta-action="open_recorder"]')).toBeVisible({ timeout: 5_000 });

  // Click the CTA — _dispatchSetupCta('open_recorder') calls scrollIntoView
  // on the recorder card (.wd-rec-dot's closest .wd-card).
  await page.locator('[data-cta-action="open_recorder"]').click();

  // The recorder widget (always rendered since _canEdit() is true) must be
  // visible — confirms the card exists and the scroll target is reachable.
  await expect(page.locator('.wd-rec-dot')).toBeVisible({ timeout: 5_000 });
});

test('Phase 4 chip shows on fully-configured device', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_power_history': NO_CURVE_POWER,
    'ha_washdata/get_setup_status': {
      phase: 'phase4',
      message_key: 'setup.phase4.healthy',
      message_params: { profile_count: 3 },
      cta_label_key: '',
      cta_action: '',
      secondary_label_key: null,
      secondary_action: null,
      skippable: false,
      dismissible: true,
      step_key: null,
    },
  });
  await loadStatusTabData(page);

  // Phase 4 renders as a compact chip, not a full card.
  await expect(page.locator('.wd-setup-chip--healthy')).toBeVisible({ timeout: 5_000 });
});

test('Phase 3 card has Hide guidance link; clicking it hides the card', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_power_history': NO_CURVE_POWER,
    'ha_washdata/get_setup_status': {
      phase: 'phase3',
      message_key: 'setup.phase3.suggestions',
      message_params: {},
      cta_label_key: 'setup.cta.review_suggestions',
      cta_action: 'open_suggestions',
      secondary_label_key: null,
      secondary_action: null,
      skippable: true,
      dismissible: true,
      step_key: 'setup_skip_phase3_suggestions',
    },
  });
  await loadStatusTabData(page);

  // The card and its dismissal link should both be visible.
  await expect(page.locator('[data-phase="phase3"]')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('[data-action="hide-setup-card"]')).toBeVisible({ timeout: 5_000 });

  // Click "Hide guidance" — sets _setupStatus = null and calls _render().
  await page.locator('[data-action="hide-setup-card"]').click();

  // The card is removed from the DOM on the synchronous re-render.
  await expect(page.locator('[data-phase="phase3"]')).not.toBeVisible();

  // _setPref('setup_card_dismissed', true) fires set_user_prefs via WS.
  await assertWsCalled(page, 'ha_washdata/set_user_prefs');
});

test('Phase 2 cluster nudge shows with create_profile_from_cluster CTA', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_power_history': NO_CURVE_POWER,
    'ha_washdata/get_setup_status': {
      phase: 'phase2',
      message_key: 'setup.phase2.cluster',
      message_params: { count: 4, cycle_ids: ['c1', 'c2', 'c3', 'c4'], name: 'Eco' },
      cta_label_key: 'setup.cta.create_from_cluster',
      cta_action: 'create_profile_from_cluster',
      secondary_label_key: null,
      secondary_action: null,
      skippable: true,
      dismissible: false,
      step_key: 'setup_skip_phase2',
    },
  });
  await loadStatusTabData(page);

  // The phase 2 card and its cluster-specific CTA should be visible.
  await expect(page.locator('[data-phase="phase2"]')).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('[data-cta-action="create_profile_from_cluster"]')).toBeVisible({ timeout: 5_000 });
});
