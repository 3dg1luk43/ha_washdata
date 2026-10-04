/**
 * Profiles tab tests.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled, setHandler } from '../helpers/panel';
import profilesData from '../fixtures/mock-data/profiles.json';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page);
});

test('profiles tab renders one card per profile', async ({ page }) => {
  await clickTab(page, 'profiles');
  const cards = page.locator('.wd-profile-card');
  await expect(cards).toHaveCount(3, { timeout: 8_000 });
});

test('profile cards show profile names', async ({ page }) => {
  await clickTab(page, 'profiles');
  // Scope to profile cards to avoid hidden <option> elements in other tab dropdowns
  await expect(page.locator('.wd-profile-card').filter({ hasText: 'Cotton 40°C' }).first()).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.wd-profile-card').filter({ hasText: 'Eco 60°C' }).first()).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.wd-profile-card').filter({ hasText: 'Quick 30°C' }).first()).toBeVisible({ timeout: 5_000 });
});

test('profile card shows cycle count and average duration', async ({ page }) => {
  await clickTab(page, 'profiles');
  // Cotton 40°C has 8 cycles
  const cottonCard = page.locator('.wd-profile-card').filter({ hasText: 'Cotton 40°C' });
  await expect(cottonCard).toBeVisible({ timeout: 5_000 });
  await expect(cottonCard.locator('text=8').first()).toBeVisible();
});

test('clicking a profile card opens the profile detail modal', async ({ page }) => {
  await clickTab(page, 'profiles');
  // Mock the profile cycles endpoint
  await page.evaluate(() => {
    window.__set_handler('ha_washdata/get_profile_cycles', { cycles: [] });
    window.__set_handler('ha_washdata/get_profile_envelope', { envelope: null });
  });
  const firstCard = page.locator('.wd-profile-card').first();
  await firstCard.click();
  await expect(page.locator('.wd-modal')).toBeVisible({ timeout: 5_000 });
});

test('health badge shows on profile card based on health status', async ({ page }) => {
  // The card's health badge is driven by the backend's `poor_health` advisory
  // (profile_store.compute_profile_advisories), not by profile_health directly.
  // This used to pass only because the fixture lacked profile_health, so every
  // card carried a bogus "Still learning (0/5)" badge that the locator matched.
  await setHandler(page, 'ha_washdata/get_profiles', {
    ...profilesData,
    profile_advisories: [{
      profile: 'Eco 60°C',
      severity: 'warning',
      code: 'poor_health',
      message: "'Eco 60°C' has a low fit score - its recent cycles vary a lot or match weakly.",
      message_key: 'msg.advisory_poor_health',
      message_params: { name: 'Eco 60°C' },
    }],
  });
  await clickTab(page, 'profiles');
  const ecoCard = page.locator('.wd-profile-card').filter({ hasText: 'Eco 60°C' });
  await expect(ecoCard.locator('.wd-badge', { hasText: 'poor fit' })).toBeVisible({ timeout: 5_000 });
  // Cotton 40°C is healthy: no health badge, and no warm-up badge either (8 cycles).
  const cottonCard = page.locator('.wd-profile-card').filter({ hasText: 'Cotton 40°C' });
  await expect(cottonCard.locator('.wd-badge', { hasText: 'poor fit' })).toHaveCount(0);
  await expect(cottonCard.locator('.wd-badge', { hasText: 'Still learning' })).toHaveCount(0);
});

test('warmup badge shows on profiles with few labeled cycles', async ({ page }) => {
  await clickTab(page, 'profiles');
  // Quick 30°C has profile_health.cycle_count 1 (below PROFILE_MIN_WARMUP_CYCLES = 2)
  const quickCard = page.locator('.wd-profile-card').filter({ hasText: 'Quick 30°C' });
  await expect(quickCard).toBeVisible({ timeout: 5_000 });
  // Warmup badge renders as .wd-badge with text "Still learning (n/N cycles)"
  const warmupBadge = quickCard.locator('text=Still learning').first();
  await expect(warmupBadge).toBeVisible({ timeout: 3_000 });
});

test('empty profiles state shows create profile button', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_profiles': { profiles: [], profile_health: {}, profile_trends: {}, coverage_gaps: {}, profile_advisories: [], profile_terminal: {} },
    'ha_washdata/get_profile_groups': { groups: [], min_cohesion: 0.85 },
  });
  await clickTab(page, 'profiles');
  const createBtn = page.locator('button[data-action="create-profile"]').first();
  await expect(createBtn).toBeVisible({ timeout: 5_000 });
});

test('new profile button opens create-profile modal', async ({ page }) => {
  await clickTab(page, 'profiles');
  const createBtn = page.locator('button[data-action="create-profile"]').first();
  await expect(createBtn).toBeVisible({ timeout: 5_000 });
  await createBtn.click();
  await expect(page.locator('.wd-modal')).toBeVisible({ timeout: 5_000 });
});

test('profile tab fetches profiles on navigation', async ({ page }) => {
  await clickTab(page, 'profiles');
  await assertWsCalled(page, 'ha_washdata/get_profiles');
});

test('profiles grid is responsive on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await clickTab(page, 'profiles');
  // On mobile, profiles grid should stack to 1 column
  const grid = page.locator('.wd-profiles-grid').first();
  await expect(grid).toBeVisible({ timeout: 5_000 });
  const cols = await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel');
    if (!el || !el.shadowRoot) return 0;
    const grid = el.shadowRoot.querySelector('.wd-profiles-grid');
    if (!grid) return 0;
    return getComputedStyle(grid).gridTemplateColumns.split(' ').length;
  });
  // On 390px with minmax(280px, 1fr), should be 1 column
  expect(cols).toBe(1);
});

// ── Profile panel → Cleanup tab: bulk unlabel / delete ────────────────────────

const CLEANUP_CYCLES = {
  cycles: [
    {
      cycle_id: 'clean-1', start_time: '2026-08-01T10:00:00+00:00',
      duration: 3600, status: 'completed', energy_kwh: 0.92,
      samples: [[0, 0], [60, 820], [1800, 1200], [3600, 0]],
    },
    {
      cycle_id: 'clean-2', start_time: '2026-08-02T10:00:00+00:00',
      duration: 3500, status: 'completed', energy_kwh: 0.85,
      samples: [[0, 0], [60, 760], [1750, 1140], [3500, 0]],
    },
  ],
};

/** Open the first profile's control panel and switch to its Cleanup tab. */
async function openCleanupTab(page: import('@playwright/test').Page): Promise<void> {
  await clickTab(page, 'profiles');
  await setHandler(page, 'ha_washdata/get_profile_cycles', CLEANUP_CYCLES);
  await setHandler(page, 'ha_washdata/get_profile_envelope', { envelope: null });
  await page.locator('.wd-profile-card').first().click();
  await expect(page.locator('.wd-modal')).toBeVisible({ timeout: 5_000 });
  const cleanupTab = page.locator('.wd-modal [data-maction="pp-tab-cleanup"]');
  await expect(cleanupTab).toBeVisible({ timeout: 5_000 });
  await cleanupTab.click();
  await expect(page.locator('.wd-modal [data-maction="pp-cleanup-del"]')).toBeVisible({ timeout: 5_000 });
}

test('cleanup tab offers unlabel next to delete in the same actions row', async ({ page }) => {
  await openCleanupTab(page);
  const row = page.locator('.wd-modal .wd-modal-actions').filter({
    has: page.locator('[data-maction="pp-cleanup-del"]'),
  });
  await expect(row.locator('[data-maction="pp-cleanup-unlabel"]')).toBeVisible({ timeout: 5_000 });
  await expect(row.locator('[data-maction="pp-cleanup-unlabel"]')).toContainText('Unlabel');
});

test('both cleanup actions stay disabled until a cycle is ticked', async ({ page }) => {
  await openCleanupTab(page);
  const unlabel = page.locator('.wd-modal [data-maction="pp-cleanup-unlabel"]');
  const del = page.locator('.wd-modal [data-maction="pp-cleanup-del"]');
  await expect(unlabel).toBeDisabled();
  await expect(del).toBeDisabled();

  await page.locator('.wd-modal input[data-cleanidx="0"]').check();
  await expect(unlabel).toBeEnabled();
  await expect(del).toBeEnabled();
  await expect(unlabel).toContainText('1');
  await expect(del).toContainText('1');
});

test('unlabel selected sends label_cycle with a null profile per cycle', async ({ page }) => {
  await openCleanupTab(page);
  await page.locator('.wd-modal input[data-cleanidx="0"]').check();
  await page.locator('.wd-modal input[data-cleanidx="1"]').check();
  await page.locator('.wd-modal [data-maction="pp-cleanup-unlabel"]').click();

  await expect
    .poll(async () => (await page.evaluate(() => window.__get_calls('ha_washdata/label_cycle'))).length, {
      timeout: 5_000,
    })
    .toBe(2);
  const calls = await page.evaluate(() => window.__get_calls('ha_washdata/label_cycle'));
  expect(calls.map((c: any) => c.cycle_id).sort()).toEqual(['clean-1', 'clean-2']);
  for (const c of calls as any[]) expect(c.profile_name).toBeNull();
});

test('unlabel selected never deletes cycles and refreshes the profile', async ({ page }) => {
  await openCleanupTab(page);
  const before = (await page.evaluate(() => window.__get_calls('ha_washdata/get_profile_cycles'))).length;
  await page.locator('.wd-modal input[data-cleanidx="0"]').check();
  await page.locator('.wd-modal [data-maction="pp-cleanup-unlabel"]').click();

  await expect
    .poll(async () => (await page.evaluate(() => window.__get_calls('ha_washdata/get_profile_cycles'))).length, {
      timeout: 5_000,
    })
    .toBeGreaterThan(before);
  await assertWsNotCalled(page, 'ha_washdata/delete_cycle');
  await assertWsCalled(page, 'ha_washdata/get_profiles');
});

// ── Unmatchable programs (#400 follow-up) ────────────────────────────────────
//
// A program with no cycle behind it is silently absent from every match: it can
// never win, and it cannot veto a shorter look-alike either, because the #364
// prefix guard only inspects candidates that reached the ranking. The backend
// reports it as an `unmatchable` advisory; the card is where the user sees it.

test('a program reported as unmatchable gets a warning badge', async ({ page }) => {
  await setHandler(page, 'ha_washdata/get_profiles', {
    ...profilesData,
    profile_advisories: [{
      profile: 'Quick 30°C',
      severity: 'warning',
      code: 'unmatchable',
      message: "'Quick 30°C' can never be matched: it has no cycle with power data behind it.",
      message_key: 'msg.advisory_unmatchable',
      message_params: { name: 'Quick 30°C' },
    }],
  });
  await clickTab(page, 'profiles');

  const flagged = page.locator('.wd-profile-card').filter({ hasText: 'Quick 30°C' });
  await expect(flagged).toBeVisible({ timeout: 5_000 });
  const badge = flagged.locator('.wd-badge', { hasText: "can't be matched" });
  await expect(badge).toBeVisible();
  // The advisory text is the tooltip, so the fix is actionable from the card.
  await expect(badge).toHaveAttribute('title', /no cycle with power data/);

  // And it is scoped to the program actually reported - not every card.
  const healthy = page.locator('.wd-profile-card').filter({ hasText: 'Cotton 40°C' });
  await expect(healthy.locator('.wd-badge', { hasText: "can't be matched" })).toHaveCount(0);
});

test('no unmatchable badge when the backend reports none', async ({ page }) => {
  await clickTab(page, 'profiles');
  await expect(page.locator('.wd-profile-card').first()).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.wd-badge', { hasText: "can't be matched" })).toHaveCount(0);
});

// ── Terminal signature: how a program ends (item 269) ────────────────────────
//
// `profile_terminal` measures the quiet drying phase and the pump-out that
// usually follows it, per program, from that program's own cycles. It shipped
// over the WS for a commit before anything rendered it, and the earlier bug was
// the reverse - the backend computed {} for every device and no test noticed -
// so these assert the wiring, not the statistic.

test('a measured quiet tail shows on the program that has one', async ({ page }) => {
  await setHandler(page, 'ha_washdata/get_profiles', {
    ...profilesData,
    profile_terminal: {
      'Cotton 40°C': {
        quiet_before_s: 934, event_seconds: 60, event_watts: 33.2,
        event_watts_frac: 0.013, position_frac: 0.93, seen_in: 6, measured: 15,
        consistency: 0.4,
      },
    },
  });
  await clickTab(page, 'profiles');

  const withTail = page.locator('.wd-profile-card').filter({ hasText: 'Cotton 40°C' });
  const badge = withTail.locator('.wd-badge', { hasText: 'quiet tail' });
  await expect(badge).toBeVisible({ timeout: 5_000 });
  await expect(badge).toHaveText(/~16m quiet tail/);
  // 6 of 15 is the real measured frequency: the badge must not imply every run
  // does this, so the count is stated rather than rounded away.
  await expect(badge).toHaveAttribute('title', /Seen in 6 of 15 measured cycles/);
  await expect(badge).toHaveAttribute('title', /33 W for 60 s/);

  // Scoped to the program it was measured on.
  const other = page.locator('.wd-profile-card').filter({ hasText: 'Eco 60°C' });
  await expect(other.locator('.wd-badge', { hasText: 'quiet tail' })).toHaveCount(0);
});

test('a one-off terminal event is not reported as a program trait', async ({ page }) => {
  await setHandler(page, 'ha_washdata/get_profiles', {
    ...profilesData,
    profile_terminal: {
      'Cotton 40°C': {
        quiet_before_s: 934, event_seconds: 60, event_watts: 33.2,
        event_watts_frac: 0.013, position_frac: 0.93, seen_in: 1, measured: 12,
        consistency: 0.083,
      },
    },
  });
  await clickTab(page, 'profiles');
  await expect(page.locator('.wd-profile-card').first()).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.wd-badge', { hasText: 'quiet tail' })).toHaveCount(0);
});

test('no quiet-tail badge when the backend measured none', async ({ page }) => {
  await clickTab(page, 'profiles');
  await expect(page.locator('.wd-profile-card').first()).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.wd-badge', { hasText: 'quiet tail' })).toHaveCount(0);
});

// ─── Imported programs: how often the matcher used them (STORE-21) ──────────
//
// Counted locally from this appliance's own cycles the matcher labelled, so an
// import that never fits can be pruned. Only imported programs carry it.

function withImports(counts: Record<string, number>) {
  const data = JSON.parse(JSON.stringify(profilesData));
  data.profiles[1].is_imported = true;   // Eco 60°C
  data.profiles[2].is_imported = true;   // Quick 30°C
  data.profile_matcher_counts = counts;
  return data;
}

test('an imported program says how often the matcher used it', async ({ page }) => {
  await setHandler(page, 'ha_washdata/get_profiles', withImports({ 'Eco 60°C': 3, 'Cotton 40°C': 5 }));
  await clickTab(page, 'profiles');
  const eco = page.locator('.wd-profile-card').filter({ hasText: 'Eco 60°C' });
  const used = eco.locator('.wd-matcher-used');
  await expect(used).toHaveText('Matched 3 of your cycles', { timeout: 5_000 });
  await expect(used).toHaveAttribute('title', /nothing is sent to the store/);
});

test('an imported program the matcher never used says so', async ({ page }) => {
  await setHandler(page, 'ha_washdata/get_profiles', withImports({ 'Eco 60°C': 3 }));
  await clickTab(page, 'profiles');
  const quick = page.locator('.wd-profile-card').filter({ hasText: 'Quick 30°C' });
  await expect(quick.locator('.wd-matcher-used')).toHaveText('Not matched to your cycles yet', { timeout: 5_000 });
});

test('a program of your own carries no matcher-use badge', async ({ page }) => {
  await setHandler(page, 'ha_washdata/get_profiles', withImports({ 'Cotton 40°C': 5 }));
  await clickTab(page, 'profiles');
  const cotton = page.locator('.wd-profile-card').filter({ hasText: 'Cotton 40°C' });
  await expect(cotton).toBeVisible({ timeout: 5_000 });
  await expect(cotton.locator('.wd-matcher-used')).toHaveCount(0);
});

// ─── Coverage gaps (register item 432) ───────────────────────────────────────

test('a coverage-gap cluster shows on Profiles and pre-selects its cycle', async ({ page }) => {
  await setHandler(page, 'ha_washdata/get_profiles', {
    ...profilesData,
    coverage_gaps: {
      unmatched_count: 6, suggest_create: true,
      profile_suggestions: [{ suggested_name: '~45 min program', cycle_ids: ['gap-1', 'gap-2', 'gap-3'], avg_duration_s: 2700, count: 3, similarity: 0.91 }],
    },
  });
  await clickTab(page, 'profiles');
  const banner = page.locator('.wd-sug-banner').filter({ hasText: '45 min' });
  await expect(banner).toBeVisible({ timeout: 8_000 });
  await expect(banner).toContainText('3');
  await banner.locator('[data-action="coverage-create"]').click();
  await expect(page.locator('#wd-cp-cycle')).toHaveValue('gap-1');
});

test('no coverage-gap banner when nothing is missing', async ({ page }) => {
  await clickTab(page, 'profiles');
  await expect(page.locator('.wd-profiles-grid').first()).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('[data-action="coverage-create"]')).toHaveCount(0);
});
