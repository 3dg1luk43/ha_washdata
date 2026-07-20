/**
 * ML Training tab tests.
 * Covers: status section, settings toggles, training trigger, what-was-learned section.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, openMlTab, assertWsCalled } from '../helpers/panel';

// The panel reads: st.on_device_models (dict), st.cycle_count, st.min_cycles,
// st.last_trained, st.enabled, st.hour, st.running.
const ML_STATUS_RESPONSE = {
  on_device_models: {},
  cycle_count: 12,
  min_cycles: 20,
  last_trained: null,
  enabled: false,
  hour: 2,
  running: false,
};

const ML_STATUS_PERSONALIZED = {
  on_device_models: {
    live_match: {
      label: 'Program matching',
      blurb: 'Identifies which program is running',
      auc: 0.94,
      trained_at: '2026-07-10T14:00:00+00:00',
      trend: 'improving',
    },
  },
  cycle_count: 24,
  min_cycles: 20,
  last_trained: '2026-07-10T14:00:00+00:00',
  enabled: true,
  hour: 2,
  running: false,
};

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': ML_STATUS_RESPONSE,
  });
});

// ─── Tab visibility ───────────────────────────────────────────────────────────

test('ML Training subtab is visible under Advanced when mlTrainingAvailable', async ({ page }) => {
  await clickTab(page, 'advanced');
  const mlTab = page.locator('button.wd-subtab[data-ptab="ml"]').first();
  await expect(mlTab).toBeVisible({ timeout: 8_000 });
});

test('ML Training tab fetches status on navigation', async ({ page }) => {
  await openMlTab(page);
  await assertWsCalled(page, 'ha_washdata/get_ml_training_status');
});

// ─── Status section ───────────────────────────────────────────────────────────

test('ML status section renders data-readiness bar', async ({ page }) => {
  await openMlTab(page);
  const bar = page.locator('div[style*="height:8px"]').first();
  await expect(bar).toBeVisible({ timeout: 8_000 });
});

test('ML status section shows "built-in models" label when not personalized', async ({ page }) => {
  await openMlTab(page);
  const label = page.locator('text=built-in models').first();
  await expect(label).toBeVisible({ timeout: 8_000 });
});

test('ML status section shows "Personalized to this machine" when on-device model trained', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': ML_STATUS_PERSONALIZED,
  });
  await openMlTab(page);
  const label = page.locator('text=Personalized to this machine').first();
  await expect(label).toBeVisible({ timeout: 8_000 });
});

test('ML status section shows last-checked timestamp', async ({ page }) => {
  await openMlTab(page);
  // The paragraph contains "Last checked:" as inline text in a .wd-info paragraph
  const ts = page.locator('p.wd-info:has-text("Last checked")').first();
  await expect(ts).toBeVisible({ timeout: 8_000 });
});

// ─── Train now button ────────────────────────────────────────────────────────

test('"Train now" button is present in ML status section', async ({ page }) => {
  await openMlTab(page);
  const trainBtn = page.locator('button[data-action="ml-train-now"]').first();
  await expect(trainBtn).toBeVisible({ timeout: 8_000 });
});

test('clicking "Train now" calls trigger_ml_training WS command', async ({ page }) => {
  await openMlTab(page);
  const trainBtn = page.locator('button[data-action="ml-train-now"]').first();
  await expect(trainBtn).toBeVisible({ timeout: 8_000 });
  await trainBtn.click();
  await assertWsCalled(page, 'ha_washdata/trigger_ml_training');
});

// ─── Settings section ─────────────────────────────────────────────────────────

test('ML settings toggles are present', async ({ page }) => {
  await openMlTab(page);
  // "Apply smart models" toggle — the input is visually hidden; target the field container
  const modelToggle = page.locator('.wd-field-switch').filter({ has: page.locator('[data-opt="enable_ml_models"]') }).first();
  await expect(modelToggle).toBeVisible({ timeout: 8_000 });
});

test('"Learn from this machine" toggle is present', async ({ page }) => {
  await openMlTab(page);
  const learnToggle = page.locator('.wd-field-switch').filter({ has: page.locator('[data-opt="ml_training_enabled"]') }).first();
  await expect(learnToggle).toBeVisible({ timeout: 8_000 });
});

test('toggling "Apply smart models" calls set_options', async ({ page }) => {
  await openMlTab(page);
  // The input is visually hidden; click the visible slider instead
  const slider = page.locator('.wd-field-switch').filter({ has: page.locator('[data-opt="enable_ml_models"]') }).locator('.wd-switch-slider').first();
  await expect(slider).toBeVisible({ timeout: 8_000 });
  await slider.click();
  // ML settings use id="wd-ml-save" (not data-action)
  const saveBtn = page.locator('#wd-ml-save').first();
  const saveIsVisible = await saveBtn.isVisible({ timeout: 2_000 }).catch(() => false);
  if (saveIsVisible) {
    await saveBtn.click();
    await assertWsCalled(page, 'ha_washdata/set_options');
  } else {
    // Auto-saves on toggle
    await assertWsCalled(page, 'ha_washdata/set_options');
  }
});

// ─── What WashData has learned ─────────────────────────────────────────────────

test('"What WashData has learned" section shows no-models message when not personalized', async ({ page }) => {
  await openMlTab(page);
  // No on-device models → shows a message about built-in models
  const noModels = page.locator('text=Nothing fine-tuned yet').first();
  await expect(noModels).toBeVisible({ timeout: 8_000 });
});

test('"What WashData has learned" section shows model row when personalized', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': ML_STATUS_PERSONALIZED,
  });
  await openMlTab(page);
  // Model label should appear. Scope to the "What WashData has learned" card —
  // the Playground pane also renders a hidden "Program matching" objective label.
  const learnedCard = page.locator('.wd-card', { hasText: 'What WashData has learned' });
  await expect(learnedCard.getByText('Program matching')).toBeVisible({ timeout: 8_000 });
});

test('personalized model row shows a quality chip', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': ML_STATUS_PERSONALIZED,
  });
  await openMlTab(page);
  // AUC 0.94 → "Strong fit" quality chip. Scope to the "What WashData has learned"
  // card and match the full chip text so it can't collide with substrings like the
  // Playground's "how strongly run-length agreement..." matcher-param label.
  const learnedCard = page.locator('.wd-card', { hasText: 'What WashData has learned' });
  const chip = learnedCard.getByText('Strong fit').first();
  await expect(chip).toBeVisible({ timeout: 8_000 });
});

test('"Reset to built-in models" button reverts on-device models', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': ML_STATUS_PERSONALIZED,
    'ha_washdata/revert_ml_models': { ok: true },
  });
  await openMlTab(page);
  // The reset button only appears when there are on-device models
  const revertBtn = page.locator('button[data-action="ml-revert-models"]').first();
  await expect(revertBtn).toBeVisible({ timeout: 8_000 });
  await revertBtn.click();
  await assertWsCalled(page, 'ha_washdata/revert_ml_models');
});

// ─── Program-matching fine-tuning card ────────────────────────────────────────

test('matching tuning card is present in ML tab when st.matching is provided', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': {
      ...ML_STATUS_RESPONSE,
      matching: {
        active: 'defaults',
        defaults: { corr_weight: 0.45, duration_weight: 0.22, energy_weight: 0.22, dtw_ensemble_w: 0.7 },
        tuned: null,
      },
    },
  });
  await openMlTab(page);
  // "Program-matching fine-tuning" card header should appear
  const card = page.locator('text=Program-matching').first();
  await expect(card).toBeVisible({ timeout: 8_000 });
});

test('"Reset to defaults" button calls revert_matching_config when tuned weights are active', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': {
      ...ML_STATUS_PERSONALIZED,
      matching: {
        active: 'tuned',
        defaults: { corr_weight: 0.45, duration_weight: 0.22, energy_weight: 0.22, dtw_ensemble_w: 0.7 },
        tuned: {
          config: { corr_weight: 0.42, duration_weight: 0.25, energy_weight: 0.20, dtw_ensemble_w: 0.65 },
          trained_at: '2026-07-10T14:00:00+00:00',
          cycle_count: 24,
        },
      },
    },
    'ha_washdata/revert_matching_config': { ok: true },
  });
  await openMlTab(page);
  const revertBtn = page.locator('button[data-action="ml-revert-match"]').first();
  await expect(revertBtn).toBeVisible({ timeout: 8_000 });
  await revertBtn.click();
  await assertWsCalled(page, 'ha_washdata/revert_matching_config');
});

// ─── Mobile ─────────────────────────────────────────────────────────────────

test('ML tab renders without overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openMlTab(page);
  const overflow = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const body = el.shadowRoot.querySelector('.wd-body');
    return body ? body.scrollWidth - body.clientWidth : 0;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});
