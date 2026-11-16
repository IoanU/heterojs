import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { cpus, platform, release } from 'node:os';
import { setTimeout as delay } from 'node:timers/promises';
import { parseArgs } from 'node:util';
import { chromium, devices } from '@playwright/test';
import { preview } from 'vite';
import { layouts, modes } from './estimate-work.js';

const { values: args } = parseArgs({ options: {
  channel: { type: 'string', default: 'chromium' }, sizes: { type: 'string', default: '65536,524288' },
  layouts: { type: 'string', default: layouts.join(',') }, series: { type: 'string', default: '3' },
  warmups: { type: 'string', default: '8' }, samples: { type: 'string', default: '16' },
  'load-workers': { type: 'string', default: '0' }, software: { type: 'boolean' }, mobile: { type: 'boolean' },
  diagnostics: { type: 'boolean' }, output: { type: 'string', default: 'estimate-work-results.json' },
} });

function integer(text, min, max, name) {
  const value = Number(text);
  if (!/^\d+$/.test(text) || !Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer between ${min} and ${max}.`);
  }
  return value;
}

const lengths = [...new Set(args.sizes.split(',').map((n) => integer(n.trim(), 4, 4_194_304, 'size')))];
const cases = [...new Set(args.layouts.split(',').map((name) => name.trim()))];
if (cases.some((name) => !layouts.includes(name))) throw new TypeError(`Choose layouts from ${layouts.join(', ')}.`);
const series = integer(args.series, 1, 30, 'series');
const warmups = integer(args.warmups, 0, 100, 'warmups');
const samples = integer(args.samples, 1, 200, 'samples');
const loadWorkers = integer(args['load-workers'], 0, cpus().length, 'load-workers');

const library = '/__estimate-library.js';
const example = '/__estimate/estimate-work.js';
const fixtureFiles = {
  [library]: 'dist/library/index.js',
  [example]: 'benchmark/estimate-work.js',
  '/__estimate/estimate-work.cpu.js': 'benchmark/estimate-work.cpu.js',
};
const settings = { lengths, cases, modes, warmups, samples, loadWorkers, example, library };

function statistics(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return { median: (sorted[middle] + sorted[(sorted.length - 1) >> 1]) / 2,
    p95: sorted[Math.ceil(sorted.length * .95) - 1] };
}

async function openPage(browser, address) {
  const page = await browser.newPage(args.mobile ? devices['Pixel 7'] : {});
  try {
    for (const [url, path] of Object.entries(fixtureFiles)) {
      await page.route(`**${url}`, (route) => route.fulfill({ path, contentType: 'text/javascript' }));
    }
    const cdp = await page.context().newCDPSession(page);
    if (args.mobile) await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
    await page.goto(address);
    return { page, cdp };
  } catch (error) {
    await page.close();
    throw error;
  }
}

async function runSeries(page, index) {
  return page.evaluate(async ({ index, lengths, cases, modes, warmups, samples, loadWorkers, example, library }) => {
    const { HeteroJS } = await import(library);
    const { makeInput, reference, register, work } = await import(example);
    const workers = [];
    try {
      if (loadWorkers) {
        const source = `let state = 31337; postMessage(state); while (true) {
          for (let i = 0; i < 2 ** 22; i++) state = (Math.imul(state, 1664525) + 1013904223) | 0;
          postMessage(state);
        }`;
        const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
        try {
          for (let i = 0; i < loadWorkers; i++) workers.push(new Worker(url));
          await Promise.all(workers.map((worker) => new Promise((resolve, reject) => {
            worker.addEventListener('message', resolve, { once: true });
            worker.addEventListener('error', reject, { once: true });
          })));
        } finally { URL.revokeObjectURL(url); }
      }
      const observations = [];
      let gpu;
      for (const [caseIndex, layout] of cases.entries()) {
        for (const [sizeIndex, length] of lengths.entries()) {
          const inputs = [makeInput(length, layout, 0), makeInput(length, layout, 1)];
          const expected = [reference(inputs[0])];
          expected.push(layout === 'moving' ? reference(inputs[1]) : expected[0]);
          const engines = {};
          const operations = {};
          const runs = Object.fromEntries(modes.map((mode) => [mode, []]));
          try {
            for (const mode of modes) {
              const engine = await HeteroJS.create({ cpuWorkers: 4, inlineCpu: false });
              engines[mode] = engine;
              operations[mode] = register(engine, mode);
            }
            for (let sample = 0; sample < 1 + warmups + samples; sample++) {
              const offset = (index + sample + caseIndex + sizeIndex) % modes.length;
              for (const mode of [...modes.slice(offset), ...modes.slice(0, offset)]) {
                const input = inputs[sample % 2];
                const strategy = mode === 'adaptive-work' ? 'adaptive' : mode;
                const started = performance.now();
                const { output, profile } = await engines[mode].compute({ operation: operations[mode], input, strategy });
                const elapsedMs = performance.now() - started;
                const correct = expected[sample % 2];
                if (output.length !== correct.length || output.some((value, i) => value !== correct[i])) {
                  throw new Error(`Incorrect output for ${layout}, ${length}, ${mode}, sample ${sample}.`);
                }
                if (profile.fallbackReason || (mode === 'gpu-only' && profile.gpuElements !== length)) {
                  throw new Error(profile.fallbackReason ?? 'GPU-only execution used another backend.');
                }
                runs[mode].push({ elapsedMs, totalMs: profile.totalMs, schedulingMs: profile.schedulingMs,
                  aggregationMs: profile.aggregationMs, cpuElements: profile.cpuElements, gpuElements: profile.gpuElements,
                  cpuWork: profile.chunks.reduce((sum, chunk) => sum + (chunk.backend === 'cpu' ? work(input, chunk) : 0), 0),
                  work: work(input, { start: 0, length }), chunks: profile.chunks.length, heavyStart: input.heavyStart });
              }
            }
            gpu ??= (await engines['gpu-only'].getCapabilities()).gpu;
            observations.push({ layout, length, modes: Object.fromEntries(Object.entries(runs).map(([mode, [cold, ...rest]]) =>
              [mode, { cold, warmups: rest.slice(0, warmups), timed: rest.slice(warmups) }])) });
          } finally { for (const engine of Object.values(engines)) engine.close(); }
        }
      }
      return { gpu, observations };
    } finally { for (const worker of workers) worker.terminate(); }
  }, { ...settings, index });
}

async function measureMemory(browser, address) {
  const result = {};
  for (const profile of ['full', 'summary']) {
    const { page, cdp } = await openPage(browser, address);
    try {
      await page.evaluate(async ({ example, library, profile }) => {
        const { HeteroJS } = await import(library);
        const { register, makeInput, reference } = await import(example);
        const engine = await HeteroJS.create({ cpuWorkers: 4 });
        const input = makeInput(2 ** 20, 'clustered', 0);
        const operation = register(engine, 'adaptive-work');
        const expected = reference(input);
        const execute = async () => {
          const { output, profile: metrics } = await engine.compute({ operation, input, strategy: 'adaptive', profile });
          if (output.length !== expected.length || output.some((value, i) => value !== expected[i])) throw new Error('Wrong result in memory run.');
          return metrics.chunkSummary ? metrics.chunkSummary.cpu.chunks + metrics.chunkSummary.gpu.chunks : metrics.chunks.length;
        };
        for (let i = 0; i < 5; i++) await execute();
        window.memoryRun = async () => {
          const chunks = [];
          try { for (let i = 0; i < 16; i++) chunks.push(await execute()); return chunks; }
          finally { engine.close(); }
        };
      }, { example, library, profile });
      await cdp.send('HeapProfiler.collectGarbage');
      const snapshots = [await cdp.send('Runtime.getHeapUsage')];
      let finished = false;
      const running = page.evaluate(() => window.memoryRun()).finally(() => { finished = true; });
      while (!finished) {
        snapshots.push(await cdp.send('Runtime.getHeapUsage'));
        await delay(15);
      }
      const chunks = await running;
      const peak = (metric) => Math.max(...snapshots.map((sample) => sample[metric] ?? 0));
      result[profile] = { scope: 'main renderer isolate', size: 2 ** 20, chunks, snapshots: snapshots.length,
        baseline: snapshots[0], peakUsedSize: peak('usedSize'), peakBackingStorageSize: peak('backingStorageSize') };
    } finally { await page.close(); }
  }
  return result;
}

async function measureRecovery(browser, address) {
  const { page } = await openPage(browser, address);
  try {
    return await page.evaluate(async (library) => {
      const { HeteroJS } = await import(library);
      const measurements = [];
      for (let i = 0; i < 4; i++) {
        const engine = await HeteroJS.create();
        const request = { operation: 'vectorAdd', inputs: [Float32Array.of(1, 2), Float32Array.of(4, 8)] };
        const original = navigator.gpu.requestAdapter.bind(navigator.gpu);
        try {
          const probe = engine.registerOperation('device', { gpu: (_, { device }) => device });
          const { output: device } = await engine.compute({ operation: probe, input: null, strategy: 'gpu-only' });
          await engine.compute({ ...request, strategy: 'gpu-only' });
          let attempts = 0;
          Object.defineProperty(navigator.gpu, 'requestAdapter', { configurable: true,
            value: (options) => ++attempts === 1 ? Promise.resolve(null) : original(options) });
          const lostAt = performance.now();
          device.destroy();
          await device.lost;
          const notified = performance.now();
          const fallback = await engine.compute({ ...request, strategy: 'static' });
          const fellBack = performance.now();
          if (fallback.output[1] !== 10 || fallback.profile.strategy !== 'cpu-only') throw new Error('CPU recovery failed.');
          await new Promise((resolve) => setTimeout(resolve, 1050));
          const restart = performance.now();
          const recovered = await engine.compute({ ...request, strategy: 'gpu-only' });
          const restored = performance.now();
          if (recovered.output[1] !== 10 || !recovered.profile.chunks[0].gpuPipelineStartup || attempts !== 2) {
            throw new Error('The GPU device or pipeline was not recreated.');
          }
          measurements.push({ notificationMs: notified - lostAt, cpuFallbackMs: fellBack - notified,
            cooldownWaitMs: restart - fellBack, firstRestoredComputeMs: restored - restart, totalMs: restored - lostAt });
        } finally {
          delete navigator.gpu.requestAdapter;
          engine.close();
        }
      }
      return measurements;
    }, library);
  } finally { await page.close(); }
}

const server = await preview({ preview: { host: '127.0.0.1', port: 0 } });
let browser;
try {
  const software = args.software ? ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader'] : [];
  browser = await chromium.launch({ channel: args.channel, args: software });
  const address = `http://127.0.0.1:${server.httpServer.address().port}/`;
  const runs = [];
  for (let i = 0; i < series; i++) {
    const { page } = await openPage(browser, address);
    try { runs.push(await runSeries(page, i)); } finally { await page.close(); }
    console.log(`Estimate-work series ${i + 1}/${series} completed`);
  }
  const summary = cases.flatMap((layout) => lengths.map((length) => {
    const rows = runs.map((run) => run.observations.find((row) => row.layout === layout && row.length === length));
    return { layout, length, modes: Object.fromEntries(modes.map((mode) => {
      const observations = rows.flatMap((row) => row.modes[mode].timed);
      return [mode, { samples: observations.length, elapsedMs: statistics(observations.map((sample) => sample.elapsedMs)),
        coldMs: statistics(rows.map((row) => row.modes[mode].cold.elapsedMs)),
        cpuShare: statistics(observations.map((sample) => sample.cpuElements / length)),
        cpuWorkShare: statistics(observations.map((sample) => sample.cpuWork / sample.work)),
        chunks: statistics(observations.map((sample) => sample.chunks)) }];
    })) };
  }));
  const metadata = { timestamp: new Date().toISOString(), browser: { channel: args.channel, version: browser.version() },
    host: { cpu: cpus()[0]?.model.trim(), logicalCpus: cpus().length, platform: platform(), osRelease: release() },
    series, warmups, samples, cpuWorkers: 4, loadWorkers, mobileEmulation: !!args.mobile, cpuThrottle: args.mobile ? 4 : 1,
    softwareRequested: !!args.software, gpu: runs[0].gpu,
    sha256: Object.fromEntries(await Promise.all(Object.values(fixtureFiles).map(async (file) =>
      [file, createHash('sha256').update(await readFile(file)).digest('hex')]))),
  };
  const document = { metadata, summary, runs };
  if (args.diagnostics) {
    document.memory = await measureMemory(browser, address);
    document.recovery = await measureRecovery(browser, address);
  }
  await writeFile(args.output, JSON.stringify(document, null, 2) + '\n');
} finally {
  try { await browser?.close(); } finally { await new Promise((resolve) => server.httpServer.close(resolve)); }
}
