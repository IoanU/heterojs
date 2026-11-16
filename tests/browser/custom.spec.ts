import { expect, test } from '@playwright/test';
import type { HeteroJS as Engine } from '../../src/index.js';
import type { CustomComputeResult, CustomOperation } from '../../src/types.js';

test.beforeEach(async ({ page }) => {
  await page.route('**/__test-library.js', (route) => route.fulfill({
    path: 'dist/library/index.js', contentType: 'text/javascript',
  }));
  await page.route('**/__custom-cpu.js', (route) => route.fulfill({
    path: 'tests/browser/fixtures/custom-cpu.js', contentType: 'text/javascript',
  }));
  await page.route('**/examples/weighted-histogram/*.js', (route) => {
    const filename = new URL(route.request().url()).pathname.split('/').at(-1)!;
    return route.fulfill({ path: `examples/weighted-histogram/${filename}`, contentType: 'text/javascript' });
  });
  await page.goto('/');
});

test('runs a whole custom operation in a worker with structured data and engine-local handles', async ({ page }) => {
  const observations = await page.evaluate(async (library) => {
    const { HeteroJS } = await import(library) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create({ gpu: 'disabled', cpuWorkers: 4 });
    const second = await HeteroJS.create({ gpu: 'disabled' });
    type Record = { score: number };
    type Input = { records: readonly Record[]; order: 'asc' | 'desc' };
    type Output = { sorted: Record[]; values: Map<string, number>; inWorker: boolean };
    const definition = { cpu: { module: '/__custom-cpu.js', exportName: 'sort' },
      estimateUnits: (input: Input) => input.records.length };
    try {
      const operation = engine.registerOperation<Input, Output>('globalSort', definition);
      const input = { records: [{ score: 9 }, { score: -3 }, { score: 4 }], order: 'asc' } as const;
      const result = await engine.compute({ operation, input, strategy: 'adaptive' });
      const failure = (work: Promise<unknown>) => work.then(() => 'success', (error: Error) => error.message);
      return { sorted: result.output.sorted.map((record) => record.score), unitCount: result.output.values.get('units'),
        inWorker: result.output.inWorker, input: input.records.map((record) => record.score),
        chunks: result.profile.chunks.map(({ backend, length }) => [backend, length]), name: result.profile.customOperation,
        lastProfile: engine.getProfile() === result.profile,
        crossEngine: await failure(second.compute({ operation, input })),
        forged: await failure(engine.compute({ operation: { name: operation.name } as typeof operation, input })),
        missingGpu: await failure(engine.compute({ operation, input, strategy: 'gpu-only' })),
        duplicate: await failure(Promise.resolve().then(() => engine.registerOperation('globalSort', definition))),
      };
    } finally { engine.close(); second.close(); }
  }, '/__test-library.js');
  expect(observations).toEqual({ sorted: [-3, 4, 9], unitCount: 3, inWorker: true, input: [9, -3, 4],
    chunks: [['cpu', 3]], name: 'globalSort', lastProfile: true,
    crossEngine: expect.stringContaining('this HeteroJS engine'), forged: expect.stringContaining('this HeteroJS engine'),
    missingGpu: expect.stringContaining('has no GPU implementation'), duplicate: expect.stringContaining('already registered'),
  });
});

test('assembles variable custom partials in range order and supports the empty identity', async ({ page }) => {
  const result = await page.evaluate(async (library) => {
    const { HeteroJS } = await import(library) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create({ strategy: 'cpu-only', cpuWorkers: 2 });
    type Part = { items: string[] };
    try {
      const op = engine.registerOperation<string[], string, Part, string>('ordered', {
        cpu: { module: '/__custom-cpu.js', exportName: 'ordered' },
        plan(items) {
          return { units: items.length, maxChunkUnits: 2,
            chunk: ({ start, length }) => ({ items: items.slice(start, start + length) }),
            combine: async (parts) => parts.map(({ start, value }) => `${start}=${value}`).join('|'),
          };
        },
      });
      const run = (input: string[]) => engine.compute({ operation: op, input });
      const full = await run(['x', 'yz', 'abc', '', 'd']);
      const empty = await run([]);
      const chunks = full.profile.chunks;
      const completedAt = (i: number) => chunks[i].startedAtMs + chunks[i].elapsedMs;
      return { output: full.output, ranges: chunks.map(({ start, length }) => [start, length]),
        outOfOrder: completedAt(1) < completedAt(0), empty: empty.output, emptyChunks: empty.profile.chunks.length };
    } finally { engine.close(); }
  }, '/__test-library.js');
  expect(result).toEqual({ output: '0=xyz|2=abc|4=d', ranges: [[0, 2], [2, 2], [4, 1]], outOfOrder: true,
    empty: '', emptyChunks: 0 });
});

test('releases the custom CPU lane after kernel errors, clone failures, and cancellation', async ({ page }) => {
  await page.route('**/__worker-started', (route) => route.fulfill({ status: 200, body: 'started' }));
  const workerStarted = page.waitForRequest('**/__worker-started');
  const running = page.evaluate(async (library) => {
    const { HeteroJS } = await import(library) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create({ strategy: 'cpu-only', cpuWorkers: 1 });
    const controller = new AbortController();
    const failure = (work: Promise<unknown>) => work.then(() => 'success', (error: Error) => `${error.name}: ${error.message}`);
    const register = (exportName: string) => engine.registerOperation<unknown, unknown>(exportName, {
      cpu: { module: '/__custom-cpu.js', exportName },
    });
    try {
      const throwing = await failure(engine.compute({ operation: register('throwing'), input: null }));
      const missing = await failure(engine.compute({ operation: register('missing'), input: null }));
      const op = register('identity');
      const badInput = await failure(engine.compute({ operation: op, input: () => 1 }));
      const badReturn = await failure(engine.compute({ operation: register('badOutput'), input: null }));
      const stuck = register('stuck');
      const cancelled = failure(engine.compute({ operation: stuck, input: `${location.origin}/__worker-started`, signal: controller.signal }));
      window.addEventListener('message', ({ data }) => { if (data === 'cancel-custom-test') controller.abort(); }, { once: true });
      const after = engine.compute({ operation: op, input: { value: 42 } });
      return { throwing, missing, badInput, badReturn, cancelled: await cancelled, after: (await after).output };
    } finally { engine.close(); }
  }, '/__test-library.js');
  await Promise.race([workerStarted, running]);
  await page.evaluate(() => window.postMessage('cancel-custom-test', location.origin));
  const result = await running;
  expect(result).toMatchObject({ throwing: 'RangeError: Custom kernel failed.',
    missing: expect.stringContaining('no function export named missing'), badInput: expect.stringContaining('DataCloneError'),
    badReturn: expect.stringContaining('DataCloneError'), cancelled: expect.stringContaining('AbortError'), after: { value: 42 },
  });
});

type HistogramInput = { keys: Uint32Array; weights: Uint32Array; binCount: number };

test('collects transferable CPU chunks directly into the final result', async ({ page }) => {
  const result = await page.evaluate(async (library) => {
    const { HeteroJS } = await import(library) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create({ gpu: 'disabled', cpuWorkers: 2 });
    try {
      const operation = engine.registerOperation<number, Uint32Array, number, Uint32Array>('stream', {
        cpu: { module: '/__custom-cpu.js', exportName: 'transferRange', transferOutput: true },
        plan(units) {
          const result = new Uint32Array(units);
          return { units, adaptiveKey: 'test', maxChunkUnits: 256, chunk: () => 9,
            collector: { add: ({ value, start }) => { result.set(value, start); }, finish: () => result } };
        },
      });
      const { output, profile } = await engine.compute({ operation, input: 4099, strategy: 'adaptive' });
      const empty = await engine.compute({ operation, input: 0 });
      return {
        correct: output.length === 4099 && output.every((value, index) => value === 9 + index),
        bytes: profile.chunks.reduce((bytes, chunk) => bytes + chunk.outputBytes, 0),
        largest: Math.max(...profile.chunks.map((chunk) => chunk.length)),
        mode: profile.strategy, fallback: profile.fallbackReason, empty: empty.output.length,
      };
    } finally { engine.close(); }
  }, '/__test-library.js');
  expect(result).toEqual({ correct: true, bytes: 4099 * 4, largest: 256, mode: 'cpu-only',
    fallback: expect.stringContaining('no GPU implementation'), empty: 0 });
});

test('schedules a custom integer reduction over CPU, real WebGPU and both backends', async ({ page }) => {
  const capability = await page.evaluate(async (library) => {
    const { HeteroJS } = await import(library) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create();
    try { return (await engine.getCapabilities()).gpu; }
    finally { engine.close(); }
  }, '/__test-library.js');
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(capability.available, JSON.stringify(capability)).toBe(true);
  test.skip(!capability.available, JSON.stringify(capability));

  const result = await page.evaluate(async ({ library, example }) => {
    const { HeteroJS } = await import(library) as { HeteroJS: typeof Engine };
    const { registerWeightedHistogram } = await import(example) as {
      registerWeightedHistogram: (engine: Engine) => CustomOperation<HistogramInput, Uint32Array>;
    };
    const engine = await HeteroJS.create({ cpuWorkers: 2 });
    const disabled = await HeteroJS.create({ gpu: 'disabled', cpuWorkers: 1 });
    try {
      const operation = registerWeightedHistogram(engine);
      const length = 267;
      const input = { keys: Uint32Array.from({ length }, (_, i) => i % 5),
        weights: Uint32Array.from({ length }, (_, i) => i + 1), binCount: 9 };
      const expected = new Uint32Array(input.binCount);
      input.keys.forEach((key, i) => { expected[key] += input.weights[i]; });
      const inspect = ({ output, profile }: CustomComputeResult<Uint32Array>) => {
        let end = 0;
        const ordered = profile.chunks.every(({ start, length }) => {
          if (start !== end) return false;
          end += length;
          return true;
        });
        return { matches: output.every((value, i) => value === expected[i]), covers: ordered && end === length,
          backends: [...new Set(profile.chunks.map(({ backend }) => backend))].sort(), requested: profile.requestedStrategy,
          fallback: profile.fallbackReason };
      };
      const request = { operation, input, cpuShare: .5 };
      const strategies = ['cpu-only', 'gpu-only', 'static', 'adaptive'] as const;
      const results = [];
      for (const strategy of strategies) results.push(inspect(await engine.compute({ ...request, strategy })));
      const fallback = inspect(await disabled.compute({ ...request, operation: registerWeightedHistogram(disabled), strategy: 'adaptive' }));
      const empty = await engine.compute({ ...request, input: { ...input, keys: new Uint32Array(), weights: new Uint32Array() } });
      const gpuOnly = engine.registerOperation<number, number>('gpuOnly', { gpu: (n) => n });
      const noCpu = await engine.compute({ operation: gpuOnly, input: 17, strategy: 'adaptive' });
      const invalidGpu = engine.registerOperation('invalidGpu', { async gpu(_, { device }) {
        await Promise.resolve();
        device.createBuffer({ size: 4, usage: 0 }).destroy();
      } });
      const gpuError = await engine.compute({ operation: invalidGpu, input: null }).then(() => 'success', (error: Error) => error.message);
      let gpuCalls = 0;
      const whole = engine.registerOperation<number, number>('whole', {
        cpu: { module: '/__custom-cpu.js', exportName: 'identity' },
        gpu: (n) => { gpuCalls++; return n; },
      });
      const runWhole = (input: number) => engine.compute({ operation: whole, input, strategy: 'adaptive' });
      const wholeCpu = await runWhole(7);
      const wholeGpu = await runWhole(8);
      return { results, fallback, empty: [...empty.output], noCpu: [noCpu.output, noCpu.profile.strategy],
        gpuError, whole: [wholeCpu, wholeGpu].map(({ output, profile }) => [output, profile.chunks.map((chunk) => chunk.backend)]), gpuCalls,
        inputIntact: input.keys.byteLength === length * 4 && input.weights[length - 1] === length };
    } finally { engine.close(); disabled.close(); }
  }, { library: '/__test-library.js', example: '/examples/weighted-histogram/index.js' });
  expect(result.results).toEqual([
    { requested: 'cpu-only', matches: true, covers: true, backends: ['cpu'] },
    { requested: 'gpu-only', matches: true, covers: true, backends: ['gpu'] },
    { requested: 'static', matches: true, covers: true, backends: ['cpu', 'gpu'] },
    { requested: 'adaptive', matches: true, covers: true, backends: ['cpu', 'gpu'] },
  ]);
  expect(result.fallback).toMatchObject({ matches: true, covers: true, backends: ['cpu'], fallback: expect.stringContaining('disabled') });
  expect(result.empty).toEqual(Array(9).fill(0));
  expect(result.noCpu).toEqual([17, 'gpu-only']);
  expect(result.gpuError).toContain('WebGPU error');
  expect(result.whole).toEqual([[7, ['cpu']], [8, ['gpu']]]);
  expect(result.gpuCalls).toBe(1);
  expect(result.inputIntact).toBe(true);
});
