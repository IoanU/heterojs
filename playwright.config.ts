import { defineConfig } from '@playwright/test';

const url = 'http://localhost:4173';
const software = process.env.HETEROJS_SOFTWARE_GPU === '1';
const gpuArgs = software ? ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader'] : [];

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  timeout: 45_000,
  use: {
    baseURL: url,
    channel: process.env.HETEROJS_BROWSER_CHANNEL || 'chromium',
    launchOptions: { args: gpuArgs },
  },
  webServer: { command: 'npm run preview -- --host localhost --port 4173 --strictPort', url },
});
