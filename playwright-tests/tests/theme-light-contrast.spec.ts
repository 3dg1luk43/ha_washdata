/**
 * Panel text must follow the HA theme, not the browser/OS color scheme (#444).
 *
 * The reporter runs Catppuccin Auto Latte Macchiato: HA serves the LIGHT variant
 * during the day while the OS stays dark. The profile card titles came out
 * near-white on a light card and were unreadable; everything beside them (badges,
 * cycle count, averages) was fine, because those name a theme token explicitly
 * and only the inherited color was wrong.
 *
 * Reproduced here as "light theme tokens + dark OS preference", and asserted as a
 * real contrast ratio rather than a specific color, so the test states the
 * property that matters.
 */

import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';

// Catppuccin Latte: light base, dark text.
const LATTE = {
  '--primary-background-color': '#eff1f5',
  '--card-background-color': '#ffffff',
  '--secondary-background-color': '#e6e9ef',
  '--primary-text-color': '#4c4f69',
  '--secondary-text-color': '#6c6f85',
  '--divider-color': '#dce0e8',
  '--primary-color': '#1e66f5',
};

test.use({ colorScheme: 'dark' });

async function applyLightTheme(page: Page) {
  await page.addStyleTag({
    content: `:root, html, body { ${Object.entries(LATTE)
      .map(([k, v]) => `${k}: ${v};`)
      .join(' ')} }`,
  });
}

function srgb(c: number): number {
  const x = c / 255;
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}

function luminance(rgb: string): number {
  const m = rgb.match(/\d+(\.\d+)?/g);
  if (!m) throw new Error(`unparseable color: ${rgb}`);
  const [r, g, b] = m.slice(0, 3).map(Number);
  return 0.2126 * srgb(r) + 0.7152 * srgb(g) + 0.0722 * srgb(b);
}

function contrast(fg: string, bg: string): number {
  const a = luminance(fg);
  const b = luminance(bg);
  const [hi, lo] = a > b ? [a, b] : [b, a];
  return (hi + 0.05) / (lo + 0.05);
}

test('profile card titles stay readable on a light theme under a dark OS', async ({ page }) => {
  await page.goto('/');
  await applyLightTheme(page);
  await bootPanel(page, {
    'ha_washdata/get_profiles': {
      profiles: [
        { name: 'Baumwolle 60', cycle_count: 12, avg_duration: 7200, avg_energy: 800 },
      ],
    },
  });
  await clickTab(page, 'profiles');

  const title = page.locator('.wd-profile-name').first();
  await expect(title).toBeVisible({ timeout: 8_000 });

  const { fg, bg } = await title.evaluate((el) => {
    const card = el.closest('.wd-profile-card') as HTMLElement;
    return {
      fg: getComputedStyle(el).color,
      bg: getComputedStyle(card).backgroundColor,
    };
  });

  const ratio = contrast(fg, bg);
  expect(
    ratio,
    `profile title ${fg} on card ${bg} has contrast ${ratio.toFixed(2)}:1`,
  ).toBeGreaterThan(4.5);
});

test('the profile card title tracks the theme text color, not a UA system color', async ({
  page,
}) => {
  await page.goto('/');
  await applyLightTheme(page);
  await bootPanel(page, {
    'ha_washdata/get_profiles': {
      profiles: [{ name: 'Eco', cycle_count: 4, avg_duration: 3600, avg_energy: 500 }],
    },
  });
  await clickTab(page, 'profiles');

  const title = page.locator('.wd-profile-name').first();
  await expect(title).toBeVisible({ timeout: 8_000 });

  // The card is a <button>; without an explicit `color: inherit` it falls back to
  // the UA `buttontext` system color, which follows the OS scheme.
  const { titleColor, hostColor } = await title.evaluate((el) => {
    const root = (el.getRootNode() as ShadowRoot).host as HTMLElement;
    return {
      titleColor: getComputedStyle(el).color,
      hostColor: getComputedStyle(root).color,
    };
  });
  expect(titleColor).toBe(hostColor);
});
