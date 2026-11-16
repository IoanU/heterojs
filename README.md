# HeteroJS

HeteroJS is a TypeScript library for numerical computing in the browser. It can run work in CPU workers, on the calling thread, on WebGPU, or across CPU and GPU. The library and benchmarks run locally in the browser.

[Benchmark results](docs/benchmarks.md)

## Quick start

Use Node.js 20.19+ on the 20.x line or Node.js 22.12 or later. From the repository root, run:

```sh
npm ci
npm run dev
```

Open the local URL printed by Vite. `npm run build` checks the types and creates the ESM package with declarations in `dist/library` and the static benchmark site in `dist/site`. Serve the build locally with `npm run preview`, or deploy the site to an HTTPS subdirectory.

Download the prerelease package from [GitHub Releases](https://github.com/IoanU/heterojs/releases) or build it with `npm pack`. To install a downloaded archive from your application directory, run `npm install ./heterojs-0.2.0-beta.2.tgz`.

## TypeScript API

```ts
import { HeteroJS } from 'heterojs';

const engine = await HeteroJS.create({
  strategy: 'adaptive',
  cpuWorkers: 'auto',
  gpu: 'auto',
});

try {
  const { output, profile } = await engine.compute({
    operation: 'matrixMultiply',
    inputs: [Float32Array.of(1, 2, 3, 4, 5, 6), Float32Array.of(7, 8, 9, 10, 11, 12)],
    shape: { rows: 2, inner: 3, columns: 2 },
  });
  console.log(Array.from(output)); // [58, 64, 139, 154]
  console.log(profile.strategy, profile.totalMs);
} finally {
  engine.close();
}
```

| Operation | Parameters | Result |
| --- | --- | --- |
| `vectorMultiply`, `vectorAdd` | `inputs: [a, b]` with equal lengths | One value for each pair |
| `scalarMultiply` | `inputs: [a]` and a `scalar` rounded to `f32` | One value for each input |
| `sum`, `min`, `max` | `inputs: [a]` | A single-element array |
| `matrixMultiply` | `inputs: [a, b]` and `shape: { rows, inner, columns }` | A row-major `rows × columns` matrix |

Built-in inputs are `Float32Array` views. Keep their data intact until the call completes. The engine allocates a separate output. For matrix multiplication, `a` contains `rows × inner` values and `b` contains `inner × columns`. A zero inner dimension yields zeros. An empty reduction returns `0` for `sum`, `+Infinity` for `min`, and `-Infinity` for `max`.

CPU and GPU accumulation can give different results for sums and matrix products. Set tolerances based on the magnitude of the values and the amount of cancellation. Reductions propagate `NaN`, and `min` and `max` preserve the sign of zero. [WGSL floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation)

## Custom operations

`registerOperation` accepts a CPU module, a WebGPU function, or both. `compute({ operation, input })` returns the output type declared when the operation was registered. The operation handle belongs to its engine, and `profile.customOperation` reports its name.

For an algorithm with global dependencies, such as sorting, register an implementation that receives the whole input. Serve `sort.cpu.js` as a JavaScript module that workers can load.

```js
// sort.cpu.js
export function run({ values, descending }) {
  return values.toSorted((a, b) => descending ? b - a : a - b);
}
```

```ts
type SortInput = { values: number[]; descending: boolean };

const sort = engine.registerOperation<SortInput, number[]>('sort', {
  cpu: { module: new URL('./sort.cpu.js', import.meta.url) },
  estimateUnits: ({ values }) => Math.max(1, values.length),
});

const { output } = await engine.compute({
  operation: sort,
  input: { values: [3, 1, 2], descending: false },
});
console.log(output); // [1, 2, 3]
```

- A CPU module receives `(input, range)` and can return a value or a promise. The engine calls the `run` export by default. Use `exportName` to select another export. Worker inputs and results must support structured cloning. Objects, maps, and typed arrays are supported. A relative URL string resolves against the calling page. [Worker messaging](https://html.spec.whatwg.org/multipage/workers.html#dom-worker-postmessage)
- A GPU function receives `(input, { device, signal, start, length, totalUnits })`. It can allocate buffers or textures, build shaders and pipelines, and submit multiple dependent passes through `device`. The function manages its resources and any readback. Engine GPU operations share one queue. Use `HeteroJS.create({ gpuDevice: { requiredFeatures, requiredLimits } })` to request additional capabilities. [WebGPU device descriptor](https://www.w3.org/TR/webgpu/#dictdef-gpudevicedescriptor)
- Add `plan(input)` to split work into independent chunks. The plan defines `units`, an optional `maxChunkUnits`, a `chunk({ start, length, totalUnits })` function to prepare each chunk, and `combine(parts, signal)` to produce the final result. `combine` receives the parts in ascending `start` order. Each contains its range and a `value`. A plan with zero units passes an empty array. CPU and GPU implementations must produce equivalent partial results for all allowed ranges.
- For large outputs, a `StreamingCustomPlan` can use `collector.add(part, signal)` and `collector.finish(signal)`. The engine calls `add` serially as chunks complete. A pending promise keeps the chunk's execution slot occupied so a streaming consumer can control the producer's pace. The engine calls `finish` after all chunks complete and also calls it for empty plans.
- With `cpu.transferOutput: true`, the worker returns `{ value, transfer: [buffer] }`. The list contains the `ArrayBuffer` objects included in the result. The worker relinquishes the buffers when posting the result. By default, the result is copied using structured cloning.
- Use `cpu.transferInput(value, range)` to transfer input buffers to workers. For a view owned by a single CPU chunk, the function could return `(view) => [view.buffer]`. Sending the buffer detaches it from the caller and affects all views over its memory. Give each transferred chunk its own buffer and retain any data that other chunks still need. [Transferable objects](https://html.spec.whatwg.org/multipage/structured-data.html#structuredserializewithtransfer)
- `plan.estimateWork({ start, length })` gives the relative cost of a range. For every nonempty range, return a positive integer that is additive over its subranges. A prefix sum of per-element costs gives `prefix[start + length] - prefix[start]`. The scheduler uses this information to size and allocate chunks, then updates its estimates from observed timings. When `profile.estimateBasis === 'work'`, `msPerElement` is measured per estimated unit of work.
- For repeated calls, `plan.adaptiveKey` groups workloads with similar shapes and costs. `adaptive` times useful work with CPU-only, GPU-only, and fixed splits. When `estimateWork` is present, it also evaluates `adaptive-chunks`. The engine keeps up to 32 histories per operation, separated by key, size, chunk limit, fixed share, and estimate type. `profile.customDispatch` identifies the selected plan and actual allocation. Plans without a key use the chunk scheduler.

The [weighted histogram example](examples/weighted-histogram/index.js) partitions pairs of `Uint32Array` keys and weights. The CPU module sums weights directly, the WebGPU shader uses atomic adds, and `combine` merges the partial histograms. The same call supports all four strategies.

```js
import { registerWeightedHistogram } from './examples/weighted-histogram/index.js';

const histogram = registerWeightedHistogram(engine);
const { output } = await engine.compute({
  operation: histogram,
  input: { keys: Uint32Array.of(2, 0, 2), weights: Uint32Array.of(3, 4, 5), binCount: 3 },
  strategy: 'adaptive',
});
console.log(Array.from(output)); // [4, 0, 8]
```

| Custom strategy | Backend selection |
| --- | --- |
| `cpu-only` / `gpu-only` | Requires the corresponding backend and implementation. |
| `static` | For a partitioned operation, `cpuShare` determines the split. For a whole operation, a share of at least `0.5` prefers CPU. The engine uses the available implementation if the preferred backend is unavailable. |
| `adaptive` | For partitioned work, the scheduler can adjust chunk sizes and backend allocation or select a measured whole-call plan. For a whole operation with two implementations, it compares timings across calls. Each whole operation executes once. `estimateUnits` is an optional size hint and defaults to `1`. |

Custom profiles report chunk latency and queue time. Generic profiles use `0` for unknown byte counts and `null` for GPU kernel duration. On cancellation, the engine may terminate and replace a worker executing a custom chunk. The GPU function receives the cancellation signal and manages its own resources through completion.

## GPU graphs

`engine.runGpuGraph(build, { signal })` records a sequence of WGSL kernels. Use `upload` for inputs, `createBuffer` for device storage, `dispatch` to pass a kernel's output to later kernels, and `read` to request final readbacks. `result.outputs` follows the order of the `read` calls. The engine records the passes and readback copies in one WebGPU submission, then releases the graph's buffers.

```js
import { makeInputs, runResident } from './examples/gpu-graph/index.js';

const { a, b, expected } = makeInputs(262_144);
const { value, profiles: [profile] } = await runResident(engine, a, b);
console.log(value === expected, profile.uploadBytes, profile.readbackBytes);
```

The [complete example](examples/gpu-graph/index.js) multiplies two `Uint32Array` inputs, applies a threshold, and reduces the result to a sum. Kernels can use multiple bind groups, storage and uniform buffers, buffer ranges, and WGSL overrides. Copy sizes and offsets are in bytes and must be multiples of four. Graphs also follow the device limits for buffers, bindings, and workgroup counts. [WebGPU buffer-copy requirements](https://www.w3.org/TR/webgpu/#buffer-copy-commands)

Build the graph synchronously and preserve the inputs until the call completes. Buffer handles belong to one graph. `read(buffer, { offset, size })` can read a subrange and returns its index in `outputs`. Results are independent `ArrayBuffer` objects. Graph execution requires WebGPU. Cancellation rejects the call, and any work already submitted releases its resources as it completes.

The graph profile reports upload and readback bytes, transfers, and explicit submissions. Transfer timings measure host upload calls and copies from mapped buffers. Synchronization includes waiting for the GPU and mapping, so it can overlap with kernel execution.

```sh
npm run build
npm run benchmark:gpu-graph -- --channel chrome --output gpu-graph-results.json
```

The benchmark verifies the result against a CPU reference. It compares a GPU-resident chain with the same WGSL kernels using readback and re-upload between stages. [GPU graph benchmark results](docs/benchmarks.md#gpu-kernel-chains)

## Strategies and profiling

| Strategy | Behavior |
| --- | --- |
| `cpu-only` | Runs in persistent CPU workers. `auto` uses one fewer worker than the reported logical core count, bounded between 1 and 8. |
| `gpu-only` | Runs on WebGPU. A request with work to execute fails if the device is unavailable or the workload exceeds its limits. |
| `static` | Assigns the first `round(units × cpuShare)` units to CPU. The default CPU share is `0.25`. Units are vector elements or matrix output rows. Falls back to CPU when WebGPU is unavailable or the matrix shape exceeds the device limits. This is the engine default. |
| `adaptive` | Sizes vector and reduction chunks and selects backends based on observed timings and queue occupancy. For large matrices, it measures whole-call latency for CPU, GPU, and a mixed plan with one CPU worker. It keeps the mixed plan when recent medians clearly favor it. |

Direct CPU execution yields after approximately 8 ms of kernel work when more data remains. `inlineCpu: false` limits adaptive execution to workers and WebGPU. For matrices, the mixed plan aligns GPU rows to eight-row tiles and starts the worker after submitting the WebGPU command.

- `getCapabilities()` reports the configured worker count and WebGPU availability. `gpu: 'disabled'` prevents the engine from requesting an adapter.
- `profile.requestedStrategy` reports the requested strategy, `profile.strategy` reports the effective strategy, and `profile.totalMs` includes scheduling, transfers, and result assembly.
- `profile.chunks` records backend allocation, timing, and transfer bytes. `profile.matrixDispatch` identifies the plan for sufficiently large matrices. `profile.fallbackReason` explains a fallback to CPU.
- `compute({ ..., profile: 'summary' })` returns per-backend counters in `profile.chunkSummary` and an empty `profile.chunks` array. Profile memory is independent of the number of chunks. The default mode, `full`, retains per-chunk records. Chunk timings can overlap when work runs in parallel.
- `getProfile()` returns the last successfully completed compute profile. Pass `signal` to cancel a call. Submitted work may continue until its backend releases it. `close()` terminates workers and releases the GPU device and buffers.

When WebGPU reports device loss, the engine rejects active and queued GPU work with `GpuDeviceLostError`. The next call that needs the GPU attempts to reopen the device and rebuild pipelines. Compatible requests fall back to CPU if the adapter is unavailable. A subsequent request can retry a failed initialization after a one-second delay. Custom kernels can use the context's `device` as a key in a `WeakMap` of device-specific resources. [WebGPU device loss](https://www.w3.org/TR/webgpu/#dom-gpudevice-lost)

## Reproducible benchmarks

The local benchmark page compares `cpu-only`, `gpu-only`, `static`, `adaptive`, and direct CPU execution. You can select the operation, input sizes, and warm-up count. Matrix sizes are given as square side lengths, and throughput uses the `2 × side³` convention for GFLOP/s.

```sh
npm run build
npx playwright install chromium
npm run benchmark:browser -- --comparison all --operation matrixMultiply --sizes 257,513,768 --warmups 8 --samples 15 --series 4 --output matrix-results.json
```

Use `--channel chrome` or `--channel msedge` to select an installed browser, or `--load-workers 4` to add controlled CPU contention. To measure on a mobile device, open the hosted benchmark page, use the same settings, and export the JSON.

Median and p95 use the timed calls after warm-up. Each series also records the first call on a fresh engine. `vs. best CPU/GPU` divides the latency of the faster single backend by the latency of the tested strategy. A ratio above 1× means the strategy beat both single backends. `p95 loop delay` uses a 4 ms timer to measure responsiveness. Correctness checks run outside the timed section.

### Example result

A 513 × 513 matrix in Chrome on an AMD Ryzen 7 5800H with integrated Radeon graphics. No load workers were used. The comparison contains 160 timed calls per strategy.

| Metric | `cpu-only` | `gpu-only` | `adaptive` |
| --- | ---: | ---: | ---: |
| Median | 32.3 ms | 6.2 ms | **5.7 ms** |
| p95 | 37.0 ms | 8.2 ms | **7.7 ms** |

At 768 × 768, adaptive p95 was 16.3 ms and GPU-only p95 was 15.3 ms. These performance results come from one machine.

### Fractalia workload

Fractalia registers escape-time calculation as a custom operation and renders progressively through Canvas 2D. The benchmark compares the original renderer with `cpu-only`, `gpu-only`, `static`, and `adaptive`. It measures time to first preview and final frame, and compares the output image against the original worker. [Experimental integration results](docs/benchmarks.md#fractalia)

### Work estimates

`npm run benchmark:estimate-work -- --channel chrome` compares adaptive scheduling with and without `estimateWork` using the same CPU and WebGPU kernels. Add `--load-workers 8` for CPU contention, `--software` for SwiftShader, or `--mobile` for desktop Chromium with Pixel 7 emulation and CPU throttling. `--diagnostics` separately measures the main JavaScript isolate's memory and WebGPU recovery. [Method, results, and commands](docs/estimate-work.md)

## Verification and browser requirements

```sh
npm test
npx playwright install chromium
npm run test:browser
npm run test:package
```

Browser tests run against the built library and benchmark site. Set `HETEROJS_BROWSER_CHANNEL=chrome` to use installed Chrome, `HETEROJS_REQUIRE_GPU=1` to require a WebGPU adapter, or `HETEROJS_SOFTWARE_GPU=1` to verify correctness with SwiftShader.

The package check installs the archive in a clean project, type-checks the declarations, and exercises the installed library in a browser. CI runs the suites and uploads the package archive on Windows and Linux.

HeteroJS uses ES modules, Web Workers, and `AbortSignal.any`. If a Content Security Policy governs inline workers, allow `blob:` in `worker-src`. WebGPU requires a compatible browser and adapter on HTTPS or localhost. [WebGPU specification](https://www.w3.org/TR/webgpu/) · [Playwright browser configuration](https://playwright.dev/docs/browsers)

## License

[MIT](./LICENSE)
