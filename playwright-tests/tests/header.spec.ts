/**
 * Header / sidebar-toggle tests (issue #359).
 *
 * The panel is a full-page custom panel that renders its own header. On a wide
 * tablet with the companion app's "Always hide sidebar" setting, the HA header
 * (and its hamburger) is gone, so the only way back to the sidebar is the panel's
 * own burger - which used to be shown only below 870px width, trapping the user.
 * The burger must appear whenever the sidebar is always-hidden, regardless of
 * viewport width.
 */

import { test, expect } from '@playwright/test';
import { bootPanel } from '../helpers/panel';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('burger appears on a wide viewport when the sidebar is always-hidden', async ({ page }) => {
  // Force a wide viewport so the <=870px media query does NOT show the burger.
  await page.setViewportSize({ width: 1280, height: 900 });
  await bootPanel(page);

  // Default (docked sidebar) on a wide viewport: burger hidden.
  await expect(page.locator('#wd-burger')).toBeHidden();

  // Simulate the mobile-app "Always hide sidebar" setting and re-render.
  await page.evaluate(() => {
    const el = document.querySelector('ha-washdata-panel') as any;
    el.hass = Object.assign({}, el.hass, { dockedSidebar: 'always_hidden' });
    el._render();
  });

  // Now the burger must be visible so the user can reopen the sidebar.
  await expect(page.locator('#wd-burger')).toBeVisible();
});
