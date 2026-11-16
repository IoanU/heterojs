import { expect, test } from '@playwright/test';
import type { HeteroJS } from '../../src/index.js';

declare global {
  interface Window { HeteroJS: typeof HeteroJS }
}

test.beforeEach(async ({ page }) => {
  await page.route('**/__test-library.js', (route) => route.fulfill({
    path: 'dist/library/index.js', contentType: 'text/javascript',
  }));
  await page.goto('/');
  await page.evaluate(async (path) => { window.HeteroJS = (await import(path)).HeteroJS; }, '/__test-library.js');
});

test('CPU workers preserve vector views, ownership, and ordered output across concurrent calls', async ({ page }) => {
  let workers = 0;
  page.on('worker', () => { workers++; });
  const result = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create({ strategy: 'cpu-only', cpuWorkers: 2 });
    const a = Float32Array.of(99, 1.5, -2, 3, 4, 5, 9);
    const b = Float32Array.of(99, 99, 2, 3, 4, 5, 6, 9);
    const request = { operation: 'vectorMultiply', inputs: [a.subarray(1, 6), b.subarray(2, 7)] } as const;
    try {
      const completionOrder: NonNullable<ReturnType<typeof engine.getProfile>>[] = [];
      const compute = async () => {
        const result = await engine.compute(request);
        completionOrder.push(result.profile);
        return result;
      };
      const [first, second] = await Promise.all([compute(), compute()]);
      return {
        outputs: [Array.from(first.output), Array.from(second.output)],
        inputsIntact: a[1] === 1.5 && b[2] === 2 && a.byteLength === 28 && b.byteLength === 32,
        chunks: first.profile.chunks.map((chunk) => [chunk.backend, chunk.start, chunk.length]),
        profile: first.profile !== second.profile && engine.getProfile() === completionOrder.at(-1),
      };
    } finally { engine.close(); }
  });
  expect(result).toEqual({
    outputs: [[3, -6, 12, 20, 30], [3, -6, 12, 20, 30]],
    inputsIntact: true, chunks: [['cpu', 0, 3], ['cpu', 3, 2]], profile: true,
  });
  expect(workers).toBe(2);
});

test('handles empty vectors, disabled GPU, invalid inputs, abortion and shutdown', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create({ gpu: 'disabled', cpuWorkers: 1 });
    const data = Float32Array.of(1, 2, 3, 4);
    const request = { operation: 'vectorMultiply', inputs: [data, data] } as const;
    const failure = (promise: Promise<unknown>) => promise.then(() => 'unexpected success', (error: Error) => error.name);
    const wrongLength = await failure(engine.compute({ ...request, inputs: [data, data.subarray(1)] }));
    const explicitGpu = await failure(engine.compute({ ...request, strategy: 'gpu-only' }));
    const empty = await engine.compute({ ...request, inputs: [new Float32Array(), new Float32Array()] });
    const controller = new AbortController();
    const aborted = failure(engine.compute({ ...request, strategy: 'cpu-only', signal: controller.signal }));
    const fallback = engine.compute(request);
    controller.abort();
    const completed = await fallback;
    const closed = failure(engine.compute(request));
    engine.close();
    return {
      wrongLength, explicitGpu, empty: empty.output.length, emptyChunks: empty.profile.chunks.length,
      aborted: await aborted, closed: await closed, afterClose: await failure(engine.compute(request)),
      fallback: completed.profile.fallbackReason, strategy: completed.profile.strategy, output: Array.from(completed.output),
    };
  });
  expect(result).toEqual({
    wrongLength: 'RangeError', explicitGpu: 'Error', empty: 0, emptyChunks: 0, aborted: 'AbortError',
    closed: 'AbortError', afterClose: 'AbortError', fallback: 'WebGPU is disabled for this engine.',
    strategy: 'cpu-only', output: [1, 4, 9, 16],
  });
});

test('WebGPU and static hybrid cover multi-chunk offset views and reuse buffers for smaller jobs', async ({ page }) => {
  const capability = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create();
    try { return (await engine.getCapabilities()).gpu; }
    finally { engine.close(); }
  });
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(capability.available, JSON.stringify(capability)).toBe(true);
  test.skip(!capability.available, JSON.stringify(capability));

  const observations = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create({ cpuWorkers: 2 });
    const size = 2 * 2 ** 20 + 69;
    const left = new Float32Array(size + 3).subarray(3);
    const right = new Float32Array(size + 1).subarray(1);
    for (let i = 0; i < size; i++) {
      left[i] = ((i % 109) - 54) * 0.0625;
      right[i] = ((i % 31) - 15) * 0.125;
    }
    const request = { operation: 'vectorMultiply', inputs: [left, right] } as const;
    try {
      const gpu = await engine.compute({ ...request, strategy: 'gpu-only' });
      const [hybrid, small] = await Promise.all([
        engine.compute({ ...request, strategy: 'static', cpuShare: .25 }),
        engine.compute({ ...request, strategy: 'gpu-only', inputs: [left.subarray(1, 66), right.subarray(1, 66)] }),
      ]);
      const correct = (output: Float32Array, offset = 0) => output.every((value, i) => value === left[i + offset] * right[i + offset]);
      const chunks = hybrid.profile.chunks;
      const overlap = chunks.some((cpu) => cpu.backend === 'cpu' && chunks.some((gpu) => gpu.backend === 'gpu'
        && gpu.startedAtMs < cpu.startedAtMs + cpu.elapsedMs && cpu.startedAtMs < gpu.startedAtMs + gpu.elapsedMs));
      const covers = (profile: typeof gpu.profile) => {
        let offset = 0;
        return profile.chunks.every((chunk) => {
          if (chunk.start !== offset) return false;
          offset += chunk.length;
          return true;
        }) && offset === size;
      };
      return {
        gpuCorrect: correct(gpu.output), hybridCorrect: correct(hybrid.output), smallCorrect: correct(small.output, 1),
        gpuChunks: gpu.profile.chunks.length, gpuCoverage: covers(gpu.profile), hybridCoverage: covers(hybrid.profile),
        cpuShare: hybrid.profile.cpuElements, overlap,
        synchronization: gpu.profile.chunks.every((chunk) => chunk.backend === 'gpu' && chunk.synchronizationMs! >= 0),
        timestamp: gpu.profile.chunks.every((chunk) => chunk.computeMs !== null && chunk.computeMs >= 0),
      };
    } finally { engine.close(); }
  });
  expect(observations).toEqual({
    gpuCorrect: true, hybridCorrect: true, smallCorrect: true, gpuChunks: 3,
    gpuCoverage: true, hybridCoverage: true, cpuShare: 524305, overlap: true, synchronization: true,
    timestamp: capability.available && capability.timestampQueries,
  });
});

test('adaptive runs useful CPU and GPU samples, retains estimates, and returns ordered output', async ({ page }) => {
  const response = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create({ strategy: 'adaptive', cpuWorkers: 2, inlineCpu: false });
    const length = 2 ** 20 + 69;
    const a = new Float32Array(length);
    const b = new Float32Array(length);
    for (let index = 0; index < length; index++) {
      a[index] = (index % 117) / 16;
      b[index] = (index % 31) / 16;
    }
    const request = { operation: 'vectorMultiply', inputs: [a, b] } as const;
    try {
      const small = await engine.compute({ ...request, inputs: [a.subarray(0, 103), b.subarray(0, 103)] });
      // Initialize the device only, so the large run can sample it immediately.
      const gpu = (await engine.getCapabilities()).gpu;
      const first = await engine.compute(request);
      const second = await engine.compute(request);
      const correct = (result: typeof first) => result.output.every((value, index) => value === a[index] * b[index]);
      return {
        gpu, correct: [small, first, second].every(correct), small: small.profile.strategy,
        firstProbes: first.profile.chunks.filter((chunk) => chunk.exploratory).map((chunk) => chunk.backend).sort(),
        secondProbes: second.profile.chunks.filter((chunk) => chunk.exploratory).length,
        firstEstimates: first.profile.estimates, secondEstimates: second.profile.estimates,
        strategies: [first, second].map((result) => result.profile.requestedStrategy),
        assignment: first.profile.cpuElements + first.profile.gpuElements,
      };
    } finally { engine.close(); }
  });
  expect(response.correct).toBe(true);
  expect(response.small).toBe('cpu-only');
  expect(response.strategies).toEqual(['adaptive', 'adaptive']);
  expect(response.assignment).toBe(2 ** 20 + 69);
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(response.gpu.available, JSON.stringify(response.gpu)).toBe(true);
  if (response.gpu.available) {
    expect(response.firstProbes).toEqual(['cpu', 'gpu']);
    expect(response.secondProbes).toBe(0);
    expect(response.firstEstimates?.cpu?.samples).toBeGreaterThan(0);
    expect(response.firstEstimates?.gpu?.samples).toBeGreaterThan(0);
    expect(response.secondEstimates?.gpu?.samples).toBeGreaterThan(0);
  }
});

test('adaptive inline CPU writes into the final output without copying input views', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create({ strategy: 'adaptive', gpu: 'disabled' });
    const left = new Float32Array(524_301).subarray(2);
    const right = new Float32Array(524_300).subarray(1);
    left.fill(-0.25);
    right.fill(0.5);
    left[0] = 0;
    const request = { operation: 'vectorMultiply', inputs: [left, right] } as const;
    try {
      const controller = new AbortController();
      const cancelled = engine.compute({ ...request, signal: controller.signal });
      controller.abort();
      const cancellation = await cancelled.then(() => 'completed', (error: Error) => error.name);
      const [first, second] = await Promise.all([engine.compute(request), engine.compute(request)]);
      return {
        correct: [first, second].every(({ output }) => output[0] === 0 && output.length === left.length
          && output.subarray(1).every((value) => value === -0.125)),
        untouched: left[1] === -0.25 && right[1] === 0.5 && first.output.buffer !== second.output.buffer,
        inline: first.profile.chunks.every((chunk) => chunk.cpuExecution === 'inline' && chunk.inputBytes === 0 && chunk.outputBytes === 0),
        requested: first.profile.requestedStrategy, execution: first.profile.strategy, cancellation,
      };
    } finally { engine.close(); }
  });
  expect(result).toEqual({
    correct: true, untouched: true, inline: true, requested: 'adaptive', execution: 'cpu-only', cancellation: 'AbortError',
  });
});

test('vector add, scalar and reductions agree across CPU and WebGPU strategies', async ({ page }) => {
  const findings = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create({ cpuWorkers: 2 });
    try {
      const available = (await engine.getCapabilities()).gpu.available;
      const modes = available ? ['cpu-only', 'gpu-only', 'static', 'adaptive'] as const : ['cpu-only', 'adaptive'] as const;
      const a = Float32Array.of(9, 1, 2, 3, 4, 5).subarray(1);
      const b = Float32Array.of(9, -1, 2, -3, 4, -5, 9).subarray(1, 6);
      const large = new Float32Array(2 ** 20 + 7).fill(0.125);
      large[3] = -2;
      large[large.length - 1] = 1.5;
      const special = Float32Array.of(0, -0, 0);
      const empty = new Float32Array();
      const failures: string[] = [];
      for (const strategy of modes) {
        const [added, scaled, sum, min, max] = await Promise.all([
          engine.compute({ operation: 'vectorAdd', inputs: [a, b], strategy }),
          engine.compute({ operation: 'scalarMultiply', inputs: [a], scalar: -0.25, strategy }),
          engine.compute({ operation: 'sum', inputs: [large], strategy }),
          engine.compute({ operation: 'min', inputs: [large], strategy }),
          engine.compute({ operation: 'max', inputs: [large], strategy }),
        ]);
        if (`${added.output}` !== '0,4,0,8,0' || `${scaled.output}` !== '-0.25,-0.5,-0.75,-1,-1.25') failures.push(`${strategy} vector`);
        if (sum.output[0] !== (large.length - 2) * 0.125 - 0.5 || min.output[0] !== -2 || max.output[0] !== 1.5) failures.push(`${strategy} reduction`);
        if (strategy === 'gpu-only' && sum.profile.chunks.length < 2) failures.push('GPU chunking');
        const minZero = await engine.compute({ operation: 'min', inputs: [special], strategy });
        const maxZero = await engine.compute({ operation: 'max', inputs: [special], strategy });
        const nan = await engine.compute({ operation: 'min', inputs: [Float32Array.of(2, NaN, 3)], strategy });
        const infinities = await engine.compute({ operation: 'sum', inputs: [Float32Array.of(Infinity, -Infinity)], strategy });
        if (!Object.is(minZero.output[0], -0) || !Object.is(maxZero.output[0], 0)
          || !Number.isNaN(nan.output[0]) || !Number.isNaN(infinities.output[0])) failures.push(`${strategy} special values`);
        for (const [operation, expected] of [['sum', 0], ['min', Infinity], ['max', -Infinity]] as const) {
          const result = await engine.compute({ operation, inputs: [empty], strategy });
          if (result.output[0] !== expected || result.profile.chunks.length) failures.push(`${strategy} ${operation} empty`);
        }
      }
      return { failures, available };
    } finally { engine.close(); }
  });
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(findings.available).toBe(true);
  expect(findings.failures).toEqual([]);
});

test('rectangular matrices merge non-aligned rows and refresh cached right operands across requests', async ({ page }) => {
  const findings = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create({ cpuWorkers: 1 });
    try {
      const available = (await engine.getCapabilities()).gpu.available;
      const modes = available ? ['cpu-only', 'gpu-only', 'static', 'adaptive'] as const : ['cpu-only', 'adaptive'] as const;
      const shape = { rows: 139, inner: 263, columns: 137 };
      const a = new Float32Array(shape.rows * shape.inner + 2).subarray(2);
      const b = new Float32Array(shape.inner * shape.columns + 1).subarray(1);
      for (let i = 0; i < a.length; i++) a[i] = ((i % 11) - 5) / 8;
      for (let i = 0; i < b.length; i++) b[i] = ((i % 13) - 6) / 8;
      const request = { operation: 'matrixMultiply', inputs: [a, b], shape } as const;
      const reference = () => {
        const expected = new Float32Array(shape.rows * shape.columns);
        for (let i = 0; i < shape.rows; i++) for (let k = 0; k < shape.inner; k++) for (let j = 0; j < shape.columns; j++) {
          expected[i * shape.columns + j] += a[i * shape.inner + k] * b[k * shape.columns + j];
        }
        return expected;
      };
      const failures: string[] = [];
      for (const strategy of modes) {
        const gold = reference();
        const { output, profile } = await engine.compute({ ...request, strategy, cpuShare: 0.37 });
        if (output.length !== gold.length || !output.every((value, index) => value === gold[index])) failures.push(`${strategy} output`);
        if (profile.cpuElements + profile.gpuElements !== shape.rows) failures.push(`${strategy} row counts`);
        if (strategy === 'cpu-only' && (profile.chunks.length < 2 || profile.chunks[1].inputBytes !== profile.chunks[1].length * shape.inner * 4)) {
          failures.push('CPU right operand reuse');
        }
        b[0] += 1;
      }
      const zero = await engine.compute({ operation: 'matrixMultiply', inputs: [new Float32Array(), new Float32Array()],
        shape: { rows: 3, inner: 0, columns: 5 }, strategy: 'gpu-only' });
      if (zero.output.length !== 15 || zero.output.some((value) => value !== 0)) failures.push('empty inner dimension');
      return { failures, available };
    } finally { engine.close(); }
  });
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(findings.available).toBe(true);
  expect(findings.failures).toEqual([]);
});

test('adaptive matrix trials preserve every row across exclusive and GPU-first hybrid dispatch', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const engine = await window.HeteroJS.create({ cpuWorkers: 2, strategy: 'adaptive' });
    try {
      const gpu = (await engine.getCapabilities()).gpu.available;
      const size = 257;
      const data = new Float32Array(size * size).fill(0.125);
      const seen = new Set<string>();
      for (let call = 0; call < 9; call++) {
        const { output, profile } = await engine.compute({ operation: 'matrixMultiply', inputs: [data, data],
          shape: { rows: size, inner: size, columns: size } });
        const plan = profile.matrixDispatch;
        if (plan) seen.add(plan.mode);
        if (!output.every((value) => value === size / 64) || profile.cpuElements + profile.gpuElements !== size) {
          return { gpu, correct: false, seen: [...seen] };
        }
        if (plan?.mode === 'hybrid') {
          const cpu = profile.chunks.find((chunk) => chunk.backend === 'cpu')!;
          const gpu = profile.chunks.find((chunk) => chunk.backend === 'gpu')!;
          if (plan.cpuRows !== profile.cpuElements || cpu.startedAtMs < gpu.startedAtMs) {
            return { gpu: true, correct: false, seen: [...seen] };
          }
        }
      }
      return { gpu, correct: true, seen: [...seen] };
    } finally { engine.close(); }
  });
  if (process.env.HETEROJS_REQUIRE_GPU === '1') expect(result.gpu).toBe(true);
  if (result.gpu) expect(result.seen).toEqual(expect.arrayContaining(['cpu', 'gpu', 'hybrid']));
  expect(result.correct).toBe(true);
});
