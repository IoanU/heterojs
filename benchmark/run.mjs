import { readFile, writeFile } from 'node:fs/promises';
import { cpus, platform, release } from 'node:os';
import { parseArgs } from 'node:util';
import { chromium, expect } from '@playwright/test';
import { preview } from 'vite';

const { values: args } = parseArgs({ options: {
  channel: { type: 'string', default: 'chromium' },
  comparison: { type: 'string', default: 'cpu' },
  operation: { type: 'string', default: 'vectorMultiply' },
  sizes: { type: 'string' },
  warmups: { type: 'string', default: '5' },
  samples: { type: 'string', default: '25' },
  series: { type: 'string', default: '4' },
  'load-workers': { type: 'string', default: '0' },
  output: { type: 'string', default: 'benchmark-results.json' },
} });

function integer(value, min, max, label) {
  const parsed = Number(value);
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
    throw new Error(`${label} must be an integer from ${min} to ${max}.`);
  }
  return parsed;
}

const series = integer(args.series, 1, Number.MAX_SAFE_INTEGER, 'series');
const warmups = integer(args.warmups, 0, 20, 'warmups');
const samples = integer(args.samples, 1, 30, 'samples');
const loadWorkers = integer(args['load-workers'], 0, cpus().length, 'load-workers');
if (!['cpu', 'all'].includes(args.comparison)) throw new Error('comparison must be cpu or all.');
const operations = ['vectorMultiply', 'vectorAdd', 'scalarMultiply', 'sum', 'min', 'max', 'matrixMultiply'];
if (!operations.includes(args.operation)) throw new Error(`operation must be one of ${operations.join(', ')}.`);
const matrix = args.operation === 'matrixMultiply';
const sizeList = args.sizes ?? (matrix ? '128,256,512' : '1048576,4194304,16777216');
const sizes = [...new Set(sizeList.trim().split(/[,\s]+/).map((size) => integer(size, 2, matrix ? 1024 : 16_777_216, 'size')))];
if (sizes.length > 8) throw new Error('Use at most eight vector sizes.');

function statistics(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const mid = ordered.length >> 1;
  const median = ordered.length % 2 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) / 2;
  return { median: Number(median.toFixed(3)), p95: Number(ordered[Math.ceil(ordered.length * 0.95) - 1].toFixed(3)) };
}

const server = await preview({ preview: { host: '127.0.0.1', port: 0 } });
let browser;
try {
  const address = server.httpServer.address();
  const url = `http://127.0.0.1:${address.port}/`;
  browser = await chromium.launch({ channel: args.channel });
  const runs = [];
  let gpu;
  for (let index = 0; index < series; index++) {
    const page = await browser.newPage();
    try {
      await page.goto(url);
      if (loadWorkers > 0) await page.evaluate(async (count) => {
        const code = `
          let state = 42;
          postMessage(state);
          while (true) {
            for (let i = 0; i < 16777216; i++) state = (Math.imul(state, 1664525) + 1013904223) | 0;
            postMessage(state);
          }
        `;
        const workerUrl = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
        try {
          const workers = Array.from({ length: count }, () => new Worker(workerUrl));
          globalThis.benchmarkLoadWorkers = workers;
          await Promise.all(workers.map((worker) => new Promise((resolve, reject) => {
            worker.addEventListener('message', resolve, { once: true });
            worker.addEventListener('error', reject, { once: true });
          })));
        } finally {
          URL.revokeObjectURL(workerUrl);
        }
      }, loadWorkers);
      await page.getByLabel('Compare').selectOption(args.comparison);
      await page.getByLabel('Operation').selectOption(args.operation);
      await page.getByLabel(matrix ? 'Matrix side lengths' : 'Vector lengths').fill(sizes.join(','));
      await page.getByLabel('Warm-up runs').fill(String(warmups));
      await page.getByLabel('Timed runs').fill(String(samples));
      await page.getByRole('button', { name: 'Run benchmark' }).click();
      const exported = page.getByRole('button', { name: 'Export JSON' });
      await expect(exported).toBeEnabled({ timeout: 180_000 });
      const download = page.waitForEvent('download');
      await exported.click();
      const report = JSON.parse(await readFile(await (await download).path(), 'utf8'));
      for (const row of report.rows) {
        if (row.unavailable && (row.strategy === 'gpu-only' || row.strategy === 'static')) continue;
        const measured = [row.cold, ...row.warmups, ...row.warm];
        if (row.unavailable || row.warm.length !== samples || row.warmups.length !== warmups
            || measured.some((sample) => !sample || sample.mismatches !== 0)) {
          throw new Error(`Invalid result for ${row.strategy} at ${row.size}: ${row.unavailable || 'incomplete or incorrect output'}`);
        }
      }
      runs.push(report);
      if (index === 0) gpu = report.capabilities.gpu;
      console.log(`Series ${index + 1}/${series} completed`);
    } finally {
      await page.close();
    }
  }
  const summary = runs[0].rows.flatMap(({ size, strategy, operation }) => {
    const groups = runs.map((run) => run.rows.find((row) => row.size === size && row.strategy === strategy)?.warm ?? []);
    const warm = groups.flat();
    if (!warm.length) return [];
    const inline = warm.filter(({ profile }) => profile.inlineTiming);
    return [{ size, operation, strategy, samples: warm.length,
      latencyMs: statistics(warm.map(({ profile }) => profile.totalMs)),
      eventLoopDelayMs: statistics(warm.map((sample) => sample.eventLoopDelayMs)),
      inlineTurnMs: inline.length ? statistics(inline.map(({ profile }) => profile.inlineTiming.longestTurnComputeMs)) : null,
      inlineOnlyRuns: warm.filter(({ profile }) => profile.chunks.every((chunk) => chunk.cpuExecution === 'inline')).length,
      gpuRuns: warm.filter(({ profile }) => profile.gpuElements > 0).length,
      seriesLatencyMs: groups.map((group) => group.length ? statistics(group.map(({ profile }) => profile.totalMs)) : null),
    }];
  });
  const comparisons = sizes.flatMap((size) => {
    const forSize = summary.filter((row) => row.size === size);
    const cpu = forSize.find((row) => row.strategy === 'cpu-only');
    const gpu = forSize.find((row) => row.strategy === 'gpu-only');
    const adaptive = forSize.find((row) => row.strategy === 'adaptive');
    if (!cpu || !gpu || !adaptive) return [];
    const relative = (metric) => {
      const baseline = Math.min(cpu.latencyMs[metric], gpu.latencyMs[metric]);
      const time = adaptive.latencyMs[metric];
      return { bestSingleMs: baseline, adaptiveMs: time, fasterThanBoth: time < baseline,
        speedup: time > 0 ? Number((baseline / time).toFixed(3)) : null };
    };
    return [{ size, operation: adaptive.operation, median: relative('median'), p95: relative('p95') }];
  });
  const hardware = { cpu: cpus()[0]?.model, logicalCores: cpus().length, platform: platform(), release: release(), gpu };
  await writeFile(args.output, JSON.stringify({ hardware, loadWorkers, comparisons, summary, runs }, null, 2));
  console.table(summary.map(({ size, strategy, latencyMs, eventLoopDelayMs }) => ({
    size, strategy, 'median ms': latencyMs.median, 'p95 ms': latencyMs.p95, 'loop delay p95 ms': eventLoopDelayMs.p95,
  })));
} finally {
  await browser?.close();
  await server.close();
}
