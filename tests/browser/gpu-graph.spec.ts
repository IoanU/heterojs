import { expect, test } from '@playwright/test';
import type { HeteroJS as Engine } from '../../src/index.js';
import type { GpuGraphBuffer, GpuGraphBuilder } from '../../src/types.js';

const library = '/__test-library.js';
const example = '/__gpu-graph-example.js';

test.beforeEach(async ({ page }) => {
  await page.route(`**${library}`, (route) => route.fulfill({ path: 'dist/library/index.js', contentType: 'text/javascript' }));
  await page.route(`**${example}`, (route) => route.fulfill({ path: 'examples/gpu-graph/index.js', contentType: 'text/javascript' }));
  await page.goto('/');
});

test('rejects invalid GPU graphs and unavailable devices', async ({ page }) => {
  const result = await page.evaluate(async (url) => {
    const { HeteroJS } = await import(url) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create({ gpu: 'disabled' });
    const failure = (call: Promise<unknown>) => call.then(() => 'success', (error: Error) => error.message);
    try {
      let escaped: GpuGraphBuilder | undefined;
      let buffer: GpuGraphBuffer | undefined;
      const gpu = await failure(engine.runGpuGraph((graph) => { escaped = graph; buffer = graph.createBuffer(4); }));
      const closedBuilder = await failure(Promise.resolve().then(() => escaped!.read(buffer!)));
      const foreign = await failure(engine.runGpuGraph((graph) => graph.read(buffer!)));
      const alignment = await failure(engine.runGpuGraph((graph) => graph.read(graph.createBuffer(3))));
      const range = await failure(engine.runGpuGraph((graph) => graph.read(graph.createBuffer(8), { offset: 4, size: 8 })));
      const asyncBuilder = await failure(engine.runGpuGraph(async () => { throw new Error('async callback'); }));
      return { gpu, closedBuilder, foreign, alignment, range, asyncBuilder };
    } finally { engine.close(); }
  }, library);
  expect(result).toEqual({
    gpu: expect.stringContaining('WebGPU is disabled'), closedBuilder: expect.stringContaining('already been built'),
    foreign: expect.stringContaining('belong to this GPU graph'), alignment: expect.stringContaining('multiple of four bytes'),
    range: expect.stringContaining('exceeds its GPU graph buffer'), asyncBuilder: expect.stringContaining('synchronously'),
  });
});

test('keeps a chain of custom kernels on WebGPU and profiles host transfers', async ({ page }) => {
  const capability = await page.evaluate(async (url) => {
    const { HeteroJS } = await import(url) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create();
    try { return (await engine.getCapabilities()).gpu; } finally { engine.close(); }
  }, library);
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(capability.available, JSON.stringify(capability)).toBe(true);
  test.skip(!capability.available, JSON.stringify(capability));

  const result = await page.evaluate(async ({ library, example }) => {
    const { HeteroJS } = await import(library) as { HeteroJS: typeof Engine };
    const { makeInputs, runResident, runStaged } = await import(example);
    const length = 65_537;
    const { a, b, expected } = makeInputs(length);
    const data = new Uint32Array(length + 2);
    data[0] = 999;
    data.set(a, 1);
    const view = data.subarray(1, length + 1);
    const engine = await HeteroJS.create();
    const metrics = (profiles: import('../../src/types.js').GpuGraphProfile[]) =>
      (['uploadBytes', 'readbackBytes', 'submissions', 'dispatches', 'uploads', 'readbacks'] as const).map((key) =>
        profiles.reduce((sum, profile) => sum + profile[key], 0));
    try {
      const [resident, staged] = await Promise.all([runResident(engine, view, b), runStaged(engine, a, b)]);
      return { correct: resident.value === expected && staged.value === expected,
        resident: metrics(resident.profiles), staged: metrics(staged.profiles), intact: data[0] === 999 && view[1] === a[1] };
    } finally { engine.close(); }
  }, { library, example });
  expect(result).toEqual({ correct: true, intact: true,
    resident: [65_537 * 8, 4, 1, 5, 2, 1], staged: [65_537 * 16, 65_537 * 8 + 4, 3, 5, 4, 3],
  });
});

test('supports independent readbacks, shader validation errors and aborting queued graphs', async ({ page }) => {
  const result = await page.evaluate(async (url) => {
    const { HeteroJS } = await import(url) as { HeteroJS: typeof Engine };
    const engine = await HeteroJS.create();
    const capability = (await engine.getCapabilities()).gpu;
    const failure = (call: Promise<unknown>) => call.then(() => 'success', (error: Error) => error.message);
    try {
      if (!capability.available) return { capability };
      const badShader = await failure(engine.runGpuGraph((graph) => {
        graph.dispatch({ kernel: { code: 'bad WGSL', entryPoint: 'main' }, bindings: [], workgroups: [1] });
      }));
      let release!: () => void;
      let running!: () => void;
      const held = new Promise<void>((resolve) => { release = resolve; });
      const started = new Promise<void>((resolve) => { running = resolve; });
      const operation = engine.registerOperation('hold', { async gpu() { running(); await held; } });
      const blocking = engine.compute({ operation, input: null, strategy: 'gpu-only' });
      await started;
      const controller = new AbortController();
      const aborted = failure(engine.runGpuGraph((graph) => graph.read(graph.createBuffer(4)), { signal: controller.signal }));
      await Promise.resolve();
      controller.abort(new Error('Cancelled queued graph'));
      const cancelled = await aborted;
      release();
      await blocking;
      const result = await engine.runGpuGraph((graph) => {
        const data = graph.upload(Uint32Array.of(1, 2, 3, 4));
        const parameter = graph.upload(Uint32Array.of(7, 0, 0, 0), 'uniform');
        const output = graph.createBuffer(8);
        graph.dispatch({ kernel: { entryPoint: 'add', constants: { multiplier: 3 }, code: `
          @group(0) @binding(0) var<storage, read> source: array<u32>;
          @group(0) @binding(1) var<storage, read_write> result: array<u32>;
          @group(1) @binding(0) var<uniform> parameter: vec4u;
          override multiplier: u32 = 1u;
          @compute @workgroup_size(2) fn add(@builtin(local_invocation_index) index: u32) {
            result[index] = multiplier * source[index + 1u] + parameter.x;
          }
        ` }, bindings: [{ binding: 0, buffer: data }, { binding: 1, buffer: output }, { group: 1, binding: 0, buffer: parameter }],
        workgroups: [1] });
        graph.read(output);
        graph.read(data, { offset: 12 });
      });
      return { capability, badShader, cancelled, output: result.outputs.map((buffer) => [...new Uint32Array(buffer)]),
        bytes: result.profile.readbackBytes, reads: result.profile.readbacks };
    } finally { engine.close(); }
  }, library);
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(result.capability.available).toBe(true);
  test.skip(!result.capability.available, JSON.stringify(result.capability));
  expect(result.badShader).toMatch(/WGSL|shader|expected/i);
  expect(result).toMatchObject({ cancelled: 'Cancelled queued graph', output: [[13, 16], [4]], bytes: 12, reads: 2 });
});
