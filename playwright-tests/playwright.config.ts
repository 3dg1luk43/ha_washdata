import { defineConfig, devices } from '@playwright/test';
import path from 'path';

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
    baseURL: 'http://localhost:4567',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    // Playwright auto-pierces open shadow DOM (mode:'open') — no extra config.
    viewport: { width: 1280, height: 800 },
  },

  webServer: {
    command: 'node serve.mjs',
    url: 'http://localhost:4567',
    reuseExistingServer: !process.env.CI,
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
