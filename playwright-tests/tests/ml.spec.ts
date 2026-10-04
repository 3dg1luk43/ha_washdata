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
    total_energy: {
      label: 'Energy estimate',
      blurb: 'Predicting total energy and cost',
      model_mae: 0.02,
      naive_mae: 0.08,
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
  // Model label should appear. Scope to the "What WashData has learned" card so
  // a same-named label elsewhere in a hidden pane cannot satisfy it.
  const learnedCard = page.locator('.wd-card', { hasText: 'What WashData has learned' });
  await expect(learnedCard.getByText('Energy estimate')).toBeVisible({ timeout: 8_000 });
});

test('personalized model row shows a quality chip', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': ML_STATUS_PERSONALIZED,
  });
  await openMlTab(page);
  // Error 0.02 vs a 0.08 baseline (75% better) → "Strong fit" quality chip. Scope to the "What WashData has learned"
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

// ─── Program-matching fine-tuning card (removed in 0.5.8) ─────────────────────

test('no matcher-tuning card renders, even for a status payload that still carries one', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': {
      ...ML_STATUS_PERSONALIZED,
      // What a pre-0.5.8 backend sent while a tuned config was live.
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
  });
  await openMlTab(page);
  await expect(page.locator('.wd-card', { hasText: 'What WashData has learned' })).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('text=Program-matching')).toHaveCount(0);
  await expect(page.locator('button[data-action="ml-revert-match"]')).toHaveCount(0);
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

// ─── Audit ML-19 / ML-20: translated strings, why nothing was learnt ──────────

test('ML-19: "never" and the unknown fine-tune time go through the translations', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': {
      ...ML_STATUS_PERSONALIZED,
      last_trained: null,
      on_device_models: { total_energy: { ...ML_STATUS_PERSONALIZED.on_device_models.total_energy, trained_at: null } },
    },
  }, {}, { translations: { en: { ml: { never: 'XX-never', trained_unknown: 'XX-unknown' } } } });
  await openMlTab(page);
  await expect(page.locator('p.wd-info:has-text("Last checked")').first()).toContainText('XX-never', { timeout: 8_000 });
  const learnedCard = page.locator('.wd-card', { hasText: 'What WashData has learned' });
  await expect(learnedCard).toContainText('fine-tuned XX-unknown');
});

test('ML-20: with nothing fine-tuned, the last run says why', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': {
      ...ML_STATUS_RESPONSE,
      last_run: { total_energy: {
        ts: '2026-10-04T02:00:00+00:00', promoted: false, reason_code: 'insufficient_rows',
        reason_params: { rows: 12, min: 30, cycles: 2 }, reason: 'insufficient data (rows=12)',
      } },
    },
  });
  await openMlTab(page);
  const learnedCard = page.locator('.wd-card', { hasText: 'What WashData has learned' });
  const line = learnedCard.locator('.wd-ml-last-run[data-reason="insufficient_rows"]');
  await expect(line).toContainText('Not enough usable cycles yet: 2 labelled, clean cycles so far.', { timeout: 8_000 });
});

test('ML-20: a kept model shows why the last run did not replace it', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': {
      ...ML_STATUS_PERSONALIZED,
      last_run: { total_energy: {
        ts: '2026-10-04T02:00:00+00:00', promoted: false, reason_code: 'not_better_than_incumbent',
        reason_params: { model: '0.031', incumbent: '0.020' }, reason: 'kept',
      } },
    },
  });
  await openMlTab(page);
  const learnedCard = page.locator('.wd-card', { hasText: 'What WashData has learned' });
  await expect(learnedCard.locator('.wd-ml-last-run')).toContainText(
    'not better than the one in use (error 0.031 vs 0.020)', { timeout: 8_000 },
  );
});

test('ML-20: a promoting last run adds no reason line', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': {
      ...ML_STATUS_PERSONALIZED,
      last_run: { total_energy: { ts: '2026-10-04T02:00:00+00:00', promoted: true } },
    },
  });
  await openMlTab(page);
  const learnedCard = page.locator('.wd-card', { hasText: 'What WashData has learned' });
  await expect(learnedCard.getByText('Energy estimate')).toBeVisible({ timeout: 8_000 });
  await expect(learnedCard.locator('.wd-ml-last-run')).toHaveCount(0);
});

test('ML-20: the fit chip says how many held-out cycles it rests on', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': {
      ...ML_STATUS_PERSONALIZED,
      on_device_models: { total_energy: { ...ML_STATUS_PERSONALIZED.on_device_models.total_energy, held_out_cycles: 6 } },
    },
  });
  await openMlTab(page);
  const learnedCard = page.locator('.wd-card', { hasText: 'What WashData has learned' });
  const chip = learnedCard.locator('div[title*="held-out"]').first();
  await expect(chip).toHaveAttribute('title', /measured on 6 held-out cycles/, { timeout: 8_000 });
});

test('ML-20: "Train now" that promotes nothing says why in the toast', async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, {
    'ha_washdata/get_ml_training_status': ML_STATUS_RESPONSE,
    'ha_washdata/trigger_ml_training': {
      ok: true, promoted: [],
      results: [{ capability: 'total_energy', promoted: false, reason_code: 'holdout_too_small',
                  reason_params: { held_out: 2, min: 5, cycles: 9 } }],
    },
  });
  await openMlTab(page);
  await page.locator('button[data-action="ml-train-now"]').first().click();
  await expect(page.locator('.wd-toast')).toContainText(
    'Too few cycles to test a new model fairly: 2 could be set aside for testing, 5 are needed.', { timeout: 8_000 },
  );
});
