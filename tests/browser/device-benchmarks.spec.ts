import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import type { Report } from '../../benchmark/device/protocol.js';

test('runs the built suite and compares exported reports', async ({ page }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('link', { name: 'Cross-device benchmarks and Fractalia' }).click();
  await page.getByLabel('Device name').fill('Test laptop');
  await page.getByRole('button', { name: 'Run benchmarks' }).click();
  await expect(page.getByRole('button', { name: 'Export JSON' })).toBeEnabled({ timeout: 160_000 });

  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export JSON' }).click();
  const download = await downloaded;
  expect(download.suggestedFilename()).toMatch(/^heterojs-quick-test-laptop-.+\.json$/);
  const report = JSON.parse(await readFile((await download.path())!, 'utf-8')) as Report;
  expect(report.results).toHaveLength(4);
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(report.device.capabilities.gpu.available).toBe(true);
  for (const [index, { rows }] of report.results.entries()) {
    const adaptive = rows.find((row) => row.mode === 'adaptive')!;
    expect(adaptive.issue).toBeUndefined();
    expect(adaptive.measured).toHaveLength(8);
    if (index > 1) {
      expect(rows.find((row) => row.mode === 'cpu-only')?.quality?.exactDifferences).toBe(0);
      expect(rows.find((row) => row.mode === 'legacy')?.measured[0].frameMs).toBeGreaterThan(0);
    }
    if (report.device.capabilities.gpu.available) {
      const gpu = rows.find((row) => row.mode === 'gpu-only')!;
      expect(gpu.issue).toBeUndefined();
      expect(gpu.measured.every((sample) => sample.cpuUnits === 0 && sample.gpuUnits! > 0)).toBe(true);
    }
  }

  const second = { ...report, device: { ...report.device, label: 'Test phone' } };
  await page.getByLabel('Import JSON reports').setInputFiles([
    { name: 'phone.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(second)) },
    { name: 'invalid.json', mimeType: 'application/json', buffer: Buffer.from('{}') },
  ]);
  await expect(page.locator('#import-status')).toContainText('1 report loaded. invalid.json');
  const comparison = page.getByRole('region', { name: /Vector multiply.*device comparison/ });
  await expect(comparison.getByRole('rowheader', { name: /Test laptop/ })).toBeVisible();
  await expect(comparison.getByRole('rowheader', { name: /Test phone/ })).toBeVisible();
  await page.getByLabel('Statistic').selectOption('p95');
  await expect(comparison.getByRole('columnheader', { name: 'Adaptive p95' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('reports missing WebGPU as unavailable and keeps the CPU comparison', async ({ page }) => {
  test.setTimeout(120_000);
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true }));
  await page.goto('/device-benchmarks.html');
  await page.getByLabel('Device name').fill('CPU-only device');
  await page.getByRole('button', { name: 'Run benchmarks' }).click();
  await expect(page.locator('#status')).toContainText('8 strategy rows were unavailable', { timeout: 100_000 });
  const comparison = page.getByRole('region', { name: /Vector multiply.*device comparison/ });
  const cells = comparison.getByRole('row', { name: /CPU-only device/ }).getByRole('cell');
  await expect(cells.nth(1)).toHaveText(/×$/);
  await expect(cells.nth(2)).toHaveText('—');
  await expect(cells.nth(3)).toHaveText('—');
});
