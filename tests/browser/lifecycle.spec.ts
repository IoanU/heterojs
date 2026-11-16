import { expect, test } from '@playwright/test';
import type { GpuDeviceLostError as DeviceLoss, HeteroJS as Engine } from '../../src/index.js';

const library = '/__test-library.js';

test.beforeEach(async ({ page }) => {
  await page.route(`**${library}`, (route) => route.fulfill({ path: 'dist/library/index.js', contentType: 'text/javascript' }));
  await page.route('**/__custom-cpu.js', (route) => route.fulfill({ path: 'tests/browser/fixtures/custom-cpu.js', contentType: 'text/javascript' }));
  await page.goto('/');
});

test('serializes async collectors and bounds pending chunks while transferring inputs', async ({ page }) => {
  const result = await page.evaluate(async (url) => {
    const { HeteroJS } = await import(url) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create({ cpuWorkers: 2, gpu: 'disabled' });
    let release!: () => void;
    let notify!: () => void;
    const hold = new Promise<void>((resolve) => { release = resolve; });
    const collecting = new Promise<void>((resolve) => { notify = resolve; });
    const sent: Uint32Array[] = [];
    let adds = 0;
    let finished = false;
    try {
      const op = engine.registerOperation<number, Uint32Array, Uint32Array<ArrayBuffer>, Uint32Array>('stream', {
        cpu: { module: '/__custom-cpu.js', exportName: 'transferEcho', transferOutput: true,
          transferInput: (value: Uint32Array<ArrayBuffer>) => [value.buffer] },
        plan(units) {
          const output = new Uint32Array(units);
          return { units, maxChunkUnits: 1, estimateWork: ({ length }) => 4 * length,
            chunk({ start }) { const value = Uint32Array.of(start + 100); sent.push(value); return value; },
            collector: {
              async add({ start, value }) {
                if (++adds === 1) { notify(); await hold; }
                output.set(value, start);
              },
              finish() { finished = true; return output; },
            } };
        },
      });
      const call = engine.compute({ operation: op, input: 40, strategy: 'adaptive', profile: 'summary' });
      await collecting;
      await new Promise((resolve) => setTimeout(resolve, 35));
      const blocked = { sent: sent.length, adds, finished };
      release();
      const result = await call;
      const control = new AbortController();
      const source = Uint32Array.of(7);
      const stop = engine.registerOperation<Uint32Array<ArrayBuffer>, Uint32Array>('abort-before-transfer', {
        cpu: { module: '/__custom-cpu.js', exportName: 'identity',
          transferInput: (value) => { control.abort(); return [value.buffer]; } },
      });
      const cancelled = await engine.compute({ operation: stop, input: source, signal: control.signal })
        .then(() => 'unexpected', (error: Error) => error.name);
      const vector = new Float32Array(257).fill(2);
      const full = await engine.compute({ operation: 'sum', inputs: [vector], strategy: 'cpu-only' });
      const summary = await engine.compute({ operation: 'sum', inputs: [vector], strategy: 'cpu-only', profile: 'summary' });
      const { profile, output } = result;
      return { blocked, valid: output.every((value, index) => value === 100 + index), sent: sent.length,
        detached: sent.every((buffer) => buffer.byteLength === 0), adds, finished, cancelled, cancelledBuffer: source.byteLength,
        streamed: { details: profile.chunks.length, cpu: profile.chunkSummary?.cpu, basis: profile.estimateBasis },
        sum: { value: summary.output[0], exact: summary.output[0] === full.output[0], details: summary.profile.chunks.length,
          chunks: summary.profile.chunkSummary?.cpu.chunks, fullChunks: full.profile.chunks.length } };
    } finally { release(); engine.close(); }
  }, library);
  expect(result).toMatchObject({ blocked: { sent: 2, adds: 1, finished: false }, valid: true,
    sent: 40, detached: true, adds: 40, finished: true, cancelled: 'AbortError', cancelledBuffer: 4,
    streamed: { details: 0, cpu: { chunks: 40, elements: 40, inputBytes: 160, outputBytes: 160 }, basis: 'work' },
    sum: { value: 514, exact: true, details: 0, chunks: 2, fullChunks: 2 },
  });
});

test('evaluates weighted chunk scheduling as a plan for similar workloads', async ({ page }) => {
  const result = await page.evaluate(async (url) => {
    const { HeteroJS } = await import(url) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create({ cpuWorkers: 2 });
    const capability = (await engine.getCapabilities()).gpu;
    try {
      if (!capability.available) return { capability };
      let estimated = 0;
      const operation = engine.registerOperation<number, number, number, number>('weighted', {
        cpu: { module: '/__custom-cpu.js', exportName: 'identity' }, gpu: (length) => length,
        plan(units) {
          return { units, maxChunkUnits: 256, adaptiveKey: 'asymmetric',
            estimateWork({ start, length }) { estimated++; return length + 7 * Math.min(length, Math.max(0, 512 - start)); },
            chunk: ({ length }) => length, combine: (parts) => parts.reduce((sum, { value }) => sum + value, 0) };
        },
      });
      let weighted;
      const modes = new Set<string>();
      for (let run = 0; run < 12; run++) {
        const result = await engine.compute({ operation, input: 2048, strategy: 'adaptive' });
        if (result.output !== 2048) throw new Error('A dispatch skipped or duplicated work.');
        const dispatch = result.profile.customDispatch!;
        modes.add(dispatch.mode);
        if (dispatch.mode === 'adaptive-chunks') {
          if (dispatch.cpuUnits !== result.profile.cpuElements) throw new Error('Wrong CPU allocation in dispatch profile.');
          weighted = result.profile.estimateBasis;
        }
      }
      return { capability, modes: [...modes], weighted, hinted: estimated > 0 };
    } finally { engine.close(); }
  }, library);
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(result.capability.available).toBe(true);
  test.skip(!result.capability.available, JSON.stringify(result.capability));
  expect(result).toMatchObject({ modes: expect.arrayContaining(['cpu-only', 'gpu-only', 'static', 'adaptive-chunks']),
    weighted: 'work', hinted: true });
});

test('rejects interrupted GPU work, falls back and restores a fresh device for later work', async ({ page }) => {
  const result = await page.evaluate(async (url) => {
    const { HeteroJS, GpuDeviceLostError } = await import(url) as { HeteroJS: typeof Engine; GpuDeviceLostError: typeof DeviceLoss };
    const engine = await HeteroJS.create();
    const capability = (await engine.getCapabilities()).gpu;
    try {
      if (!capability.available) return { capability };
      const request = { operation: 'vectorAdd', inputs: [Float32Array.of(4, 9), Float32Array.of(2, 7)] } as const;
      await engine.compute({ ...request, strategy: 'gpu-only' });
      let active!: (device: GPUDevice) => void;
      const started = new Promise<GPUDevice>((resolve) => { active = resolve; });
      let invocations = 0;
      let notified = false;
      const interrupt = engine.registerOperation('interrupt', { gpu(_, { device, signal }) {
        invocations++;
        active(device);
        signal.addEventListener('abort', () => { notified = true; }, { once: true });
        return new Promise<never>(() => {});
      } });
      const wasLost = (work: Promise<unknown>) => work.then(() => false, (error) => error instanceof GpuDeviceLostError);
      const waiting = wasLost(engine.compute({ operation: interrupt, input: null, strategy: 'gpu-only' }));
      const oldDevice = await started;
      const queued = wasLost(engine.runGpuGraph((graph) => graph.read(graph.upload(Uint32Array.of(7)))));
      await Promise.resolve();
      const requestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
      let attempts = 0;
      Object.defineProperty(navigator.gpu, 'requestAdapter', { configurable: true, value: (options?: GPURequestAdapterOptions) =>
        ++attempts === 1 ? Promise.resolve(null) : requestAdapter(options) });
      oldDevice.destroy();
      await oldDevice.lost;
      const lost = await Promise.all([waiting, queued]);
      const fallback = await engine.compute({ ...request, strategy: 'static' });
      await engine.getCapabilities();
      const duringDelay = attempts;
      await new Promise((resolve) => setTimeout(resolve, 1050));
      const [recovered, graph] = await Promise.all([
        engine.compute({ ...request, strategy: 'gpu-only' }),
        engine.runGpuGraph((graph) => graph.read(graph.upload(Uint32Array.of(21)))),
      ]);
      const restored = engine.registerOperation('fresh-device', { gpu: (_, { device }) => device !== oldDevice });
      const { output: differentDevice } = await engine.compute({ operation: restored, input: null });
      return { capability, lost, invocations, notified, duringDelay, attempts, differentDevice,
        fallback: { strategy: fallback.profile.strategy, reason: fallback.profile.fallbackReason, output: [...fallback.output] },
        recovered: { strategy: recovered.profile.strategy, startup: recovered.profile.chunks[0].gpuPipelineStartup,
          output: [...recovered.output], graph: new Uint32Array(graph.outputs[0])[0] } };
    } finally { engine.close(); }
  }, library);
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(result.capability.available).toBe(true);
  test.skip(!result.capability.available, JSON.stringify(result.capability));
  expect(result).toMatchObject({ lost: [true, true], invocations: 1, notified: true, duringDelay: 1, attempts: 2,
    differentDevice: true, fallback: { strategy: 'cpu-only', reason: expect.stringContaining('No WebGPU adapter'), output: [6, 16] },
    recovered: { strategy: 'gpu-only', startup: true, output: [6, 16], graph: 21 } });
});
