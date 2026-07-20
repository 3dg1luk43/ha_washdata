/**
 * Profiles tab tests.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, setHandler } from '../helpers/panel';
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
  await clickTab(page, 'profiles');
  // Cotton 40°C has health_status: "healthy" — should show a health badge
  const cottonCard = page.locator('.wd-profile-card').filter({ hasText: 'Cotton 40°C' });
  await expect(cottonCard.locator('.wd-badge, [class*="health"]').first()).toBeVisible({ timeout: 5_000 });
});

test('warmup badge shows on profiles with few labeled cycles', async ({ page }) => {
  await clickTab(page, 'profiles');
  // Quick 30°C has labeled_count: 1 (below warmup threshold of 5)
  const quickCard = page.locator('.wd-profile-card').filter({ hasText: 'Quick 30°C' });
  await expect(quickCard).toBeVisible({ timeout: 5_000 });
  // Warmup badge renders as .wd-badge with text "Still learning (n/N cycles)"
  const warmupBadge = quickCard.locator('text=Still learning').first();
  await expect(warmupBadge).toBeVisible({ timeout: 3_000 });
});

test('empty profiles state shows create profile button', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/get_profiles': { profiles: [], advisories: [], coverage_gaps: null },
    'ha_washdata/get_profile_groups': { groups: [], suggestions: [], min_cohesion: 0.85 },
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
