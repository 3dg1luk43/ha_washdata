import { defineConfig, devices } from '@playwright/test';
import path from 'path';

// When PANEL_BUILD=min the server serves the minified artifacts instead of the
// readable sources. Use a different port so that reuseExistingServer cannot
// accidentally hand a "min" test run an already-running "readable" server (or
// vice versa) — the port mismatch makes the reuse check fail gracefully.
const PORT = process.env.PANEL_BUILD === 'min' ? 4568 : 4567;

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { open: 'never', outputFolder: 'playwright-report' }],
    ['line'],
  ],

  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    // Playwright auto-pierces open shadow DOM (mode:'open') — no extra config.
    viewport: { width: 1280, height: 800 },
  },

  webServer: {
    command: 'node serve.mjs',
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    env: { PORT: String(PORT) },
    cwd: path.join(__dirname),
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 7'] },
    },
  ],
});
