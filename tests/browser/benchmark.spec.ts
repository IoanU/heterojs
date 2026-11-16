import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import type { Row } from '../../benchmark/runner.js';

test('built benchmark verifies baselines and exports measured data', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Vector lengths').fill('65537');
  await page.getByLabel('Timed runs').fill('2');
  const run = page.getByRole('button', { name: 'Run benchmark' });
  await expect(run).toBeEnabled();
  const gpuAvailable = !(await page.locator('#hardware').textContent())?.includes('unavailable');
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(gpuAvailable).toBe(true);
  await run.click();
  await expect(page.getByRole('status')).toHaveText(gpuAvailable ? 'Complete · 5 passed' : 'Complete · 3 passed · 2 unavailable');
  await expect(page.getByRole('row', { name: /^CPU only / })).toContainText('Pass');
  await expect(page.getByRole('rowheader', { name: 'GPU only' })).toBeVisible();
  await expect(page.getByRole('rowheader', { name: 'Static hybrid' })).toBeVisible();
  await expect(page.getByRole('row', { name: /Adaptive/ })).toContainText('Pass');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export JSON' }).click();
  const report = await download;
  expect(report.suggestedFilename()).toBe('heterojs-benchmark.json');
  const { rows } = JSON.parse(await readFile(await report.path(), 'utf8')) as { rows: Row[] };
  expect(rows.map((row) => row.strategy)).toEqual(['cpu-only', 'cpu-direct', 'gpu-only', 'static', 'adaptive']);
  for (const row of rows.filter((item) => !item.unavailable)) {
    expect(row.size).toBe(65537);
    expect(row.warmups).toHaveLength(2);
    expect(row.warm).toHaveLength(2);
    expect(row.cold?.mismatches).toBe(0);
    for (const sample of row.warm) {
      expect(sample.profile.requestedStrategy).toBe(row.strategy === 'cpu-direct' ? 'cpu-only' : row.strategy);
      const inline = row.strategy === 'adaptive' || row.strategy === 'cpu-direct';
      expect(sample.profile.strategy).toBe(inline ? 'cpu-only' : row.strategy);
      if (inline) {
        expect(sample.profile.chunks.every((chunk) => chunk.cpuExecution === 'inline')).toBe(true);
        expect(sample.profile.inlineTiming?.longestTurnComputeMs).toBeGreaterThanOrEqual(0);
      }
      expect(sample.eventLoopDelayMs).toBeGreaterThanOrEqual(0);
      expect(sample.profile.totalMs).toBeGreaterThanOrEqual(0);
      expect(sample.mismatches).toBe(0);
    }
  }
});

test('benchmark runs on CPU when WebGPU is missing', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
  await page.goto('/');
  await page.getByLabel('Vector lengths').fill('1024');
  await page.getByLabel('Timed runs').fill('1');
  await page.getByRole('button', { name: 'Run benchmark' }).click();
  await expect(page.getByRole('status')).toHaveText('Complete · 3 passed · 2 unavailable');
  await expect(page.getByRole('row', { name: /GPU only/ })).toContainText('WebGPU is unavailable');
});

test('focused comparison times both CPU baselines and adaptive with tail latency and loop delay', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Static CPU share').fill('');
  await page.getByLabel('Compare').selectOption('cpu');
  await expect(page.getByLabel('Static CPU share')).toBeDisabled();
  await page.getByLabel('Vector lengths').fill('65536');
  await page.getByLabel('Timed runs').fill('2');
  await page.getByRole('button', { name: 'Run benchmark' }).click();
  await expect(page.getByRole('status')).toHaveText('Complete · 3 passed');
  await expect(page.getByRole('columnheader', { name: 'vs. CPU only' })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: 'p95', exact: true })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: 'p95 loop delay' })).toBeVisible();
  await expect(page.getByRole('rowheader', { name: 'CPU direct' })).toBeVisible();
  await expect(page.getByRole('rowheader')).toHaveCount(3);
});

test('matrix benchmark uses square dimensions, shows GFLOP/s and verifies every backend result', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Operation').selectOption('matrixMultiply');
  await page.getByLabel('Matrix side lengths').fill('33');
  await page.getByLabel('Warm-up runs').fill('0');
  await page.getByLabel('Timed runs').fill('2');
  const run = page.getByRole('button', { name: 'Run benchmark' });
  await expect(run).toBeEnabled();
  const gpu = !(await page.locator('#hardware').textContent())?.includes('unavailable');
  await run.click();
  await expect(page.getByRole('status')).toHaveText(gpu ? 'Complete · 5 passed' : 'Complete · 3 passed · 2 unavailable');
  await expect(page.getByRole('heading', { name: '33 × 33 matrices' })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: 'GFLOP/s' })).toBeVisible();
});
