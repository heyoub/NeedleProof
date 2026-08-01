import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './apps/web/tests',
  fullyParallel: false,
  retries: 0,
  reporter: 'line',
  webServer: [
    {
      command:
        'env -u OPENAI_API_KEY -u OPENAI_ADMIN_KEY uv run uvicorn needleproof_api.main:app --app-dir apps/api/src --host 127.0.0.1 --port 8000',
      url: 'http://127.0.0.1:8000/api/live',
      reuseExistingServer: true,
      timeout: 30_000,
    },
    {
      command:
        'ASTRO_DEV_BACKGROUND=0 pnpm --filter @needleproof/web dev --host 127.0.0.1 --port 4321',
      url: 'http://127.0.0.1:4321',
      reuseExistingServer: true,
      timeout: 30_000,
    },
  ],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:4321',
    channel: 'chrome',
    headless: true,
    trace: 'retain-on-failure',
  },
});
