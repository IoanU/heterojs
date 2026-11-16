# Evaluating workload estimates in HeteroJS

The benchmark verified **3,280 timed samples** on an AMD Ryzen 7 5800H with Radeon Graphics running Windows 10.0.26200. Each result was checked element by element against the CPU implementation. The optional cost hint is provided through `plan.estimateWork`.

## 1. Effect of the cost hint

At 524,288 elements on Radeon, the hint reduced median latency from **21.35 to 10.90 ms** with a fixed heavy region and from **31.45 to 13.05 ms** when the heavy region moved between calls. In both cases, `adaptive-work` beat `cpu-only`, `gpu-only`, and `static` at the median and p95.

Uniform inputs had a higher median with the hint. At 524,288 elements, the median chunk count rose from 20 to 29. A uniform cost per element and larger chunks can work well for uniform inputs.

### Chrome on Radeon, without background load

Each cell contains the median and p95 in milliseconds after warm-up.

| Work distribution | Elements | `cpu-only` | `gpu-only` | `static` | `adaptive` | `adaptive-work` |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 65,536 | 4.70 / 5.10 | 4.60 / 9.40 | 4.50 / 6.00 | 5.40 / 6.00 | 5.90 / 6.30 |
| Uniform | 524,288 | 31.50 / 32.40 | 19.40 / 21.70 | 13.90 / 16.00 | 18.80 / 21.10 | 20.65 / 25.80 |
| Fixed heavy region | 65,536 | 13.80 / 14.30 | 4.90 / 5.60 | 4.80 / 6.30 | 5.90 / 14.40 | 5.75 / 6.50 |
| Fixed heavy region | 524,288 | 106.30 / 107.70 | 20.20 / 22.00 | 30.05 / 47.40 | 21.35 / 67.00 | 10.90 / 13.40 |
| Moving heavy region | 65,536 | 13.95 / 14.50 | 4.80 / 5.80 | 4.80 / 6.00 | 8.95 / 15.00 | 5.95 / 8.50 |
| Moving heavy region | 524,288 | 106.55 / 108.70 | 19.50 / 21.90 | 23.30 / 46.70 | 31.45 / 92.30 | 13.05 / 17.30 |

### Paired comparison

- Every strategy used the same CPU and WGSL kernels. Each element runs a 32-bit integer recurrence. The uniform case uses 70 iterations per element. The nonuniform cases use 8 iterations in the light region and 256 in the heavy quarter. The moving case alternates the heavy region between the start and end of the array.
- Both adaptive variants use the same chunk scheduler. `adaptive-work` receives the total iteration count for each range as a cost hint. Each mode uses a separate engine with four CPU workers. The static strategy assigns the first 25% of elements to CPU. The mode order rotates between samples, and all modes receive identical input within a sample group.
- Timing starts at `compute` and ends when the result is available. Verification takes place outside the timed section. The raw data includes cold calls, warm-ups, measured calls, CPU and GPU allocation, estimated work, and chunk counts. p95 uses rank `ceil(0.95 × N)`.

## 2. Compatibility, contention, memory, and recovery

The machine has one integrated GPU. Chrome and Edge used the same Radeon. SwiftShader is a software adapter. The Pixel 7 profile runs on desktop Chrome with mobile emulation and CDP CPU throttling. Validation on a dedicated GPU and a physical phone is pending. [Chromium emulation](https://chromedevtools.github.io/devtools-protocol/tot/Emulation/#method-setCPUThrottlingRate)

| Configuration | Browser | Reported adapter | Timed samples |
| --- | --- | --- | --- |
| Chrome · Radeon | 157.0.8103.78 | amd gcn-5 | 1,440 |
| Chrome · Radeon · 8 background workers | 157.0.8103.78 | amd gcn-5 | 640 |
| Edge · Radeon | 157.0.4341.60 | amd gcn-5 | 480 |
| Chrome · SwiftShader | 157.0.8103.78 | google swiftshader | 360 |
| Chrome · Pixel 7 emulation and 4× CPU slowdown | 157.0.8103.78 | amd gcn-5 | 360 |

### Contention and compatibility results

With eight background workers and the moving heavy region at 524,288 elements, p95 was **91.90 ms** without the hint and **18.80 ms** with it. The benefits depend on the backend, workload size, and contention.

| Configuration | Work distribution | Elements | Adaptive median / p95 | Adaptive-work median / p95 | Median speedup |
| --- | --- | --- | --- | --- | --- |
| Chrome · Radeon · 8 background workers | Fixed heavy region | 65,536 | 7.80 / 14.30 | 5.80 / 6.30 | 1.34× |
| Chrome · Radeon · 8 background workers | Fixed heavy region | 524,288 | 25.45 / 74.80 | 11.20 / 15.60 | 2.27× |
| Chrome · Radeon · 8 background workers | Moving heavy region | 65,536 | 8.35 / 14.60 | 6.15 / 7.10 | 1.36× |
| Chrome · Radeon · 8 background workers | Moving heavy region | 524,288 | 38.35 / 91.90 | 12.55 / 18.80 | 3.06× |
| Edge · Radeon | Fixed heavy region | 65,536 | 6.45 / 41.70 | 5.60 / 7.20 | 1.15× |
| Edge · Radeon | Fixed heavy region | 524,288 | 35.65 / 179.00 | 14.35 / 17.30 | 2.48× |
| Edge · Radeon | Moving heavy region | 65,536 | 14.25 / 43.00 | 7.45 / 11.40 | 1.91× |
| Edge · Radeon | Moving heavy region | 524,288 | 35.10 / 59.80 | 15.85 / 19.50 | 2.21× |
| Chrome · SwiftShader | Uniform | 65,536 | 5.75 / 6.10 | 6.10 / 6.80 | 0.94× |
| Chrome · SwiftShader | Fixed heavy region | 65,536 | 7.80 / 14.40 | 6.10 / 6.60 | 1.28× |
| Chrome · SwiftShader | Moving heavy region | 65,536 | 8.60 / 14.80 | 6.35 / 7.30 | 1.35× |
| Chrome · Pixel 7 emulation and 4× CPU slowdown | Uniform | 65,536 | 18.15 / 28.80 | 16.40 / 42.30 | 1.11× |
| Chrome · Pixel 7 emulation and 4× CPU slowdown | Fixed heavy region | 65,536 | 13.35 / 28.90 | 19.55 / 47.70 | 0.68× |
| Chrome · Pixel 7 emulation and 4× CPU slowdown | Moving heavy region | 65,536 | 13.85 / 22.30 | 13.05 / 26.80 | 1.06× |

### Memory sampled in the main JavaScript isolate

Each profile mode ran 16 adaptive calls at 1,048,576 elements after warm-up. CDP collected the diagnostics without background load workers. These are sampled peaks and can vary with garbage collection. The counters cover the main isolate. Worker memory, GPU process memory, and total physical memory are outside their scope. [CDP memory counters](https://chromedevtools.github.io/devtools-protocol/tot/Runtime/#method-getHeapUsage)

| Profile | Samples | Heap baseline / peak | Backing storage baseline / peak |
| --- | --- | --- | --- |
| `full` | 22 | 1.13 / 3.17 MiB | 4.13 / 51.48 MiB |
| `summary` | 22 | 1.13 / 3.65 MiB | 4.13 / 51.47 MiB |

Peak backing storage was nearly identical in this run. The summary profile's records use constant space with respect to chunk count. The workload arrays account for most of the backing storage in this experiment.

### Recovery after GPU device loss

In four trials, the benchmark destroyed a real WebGPU device, simulated an unavailable adapter on the first retry, and verified CPU fallback. Once the retry interval elapsed, a new GPU call recreated the device and pipeline. The separate medians were 0.15 ms for loss notification, 17.60 ms for CPU fallback, and 26.85 ms for the first restored GPU call after the retry delay.

## Reproduction

```sh
npm ci
npm run build
npm run benchmark:estimate-work -- --channel chrome --output estimate-work-native.json
npm run benchmark:estimate-work -- --channel chrome --layouts clustered,moving --series 2 --load-workers 8 --diagnostics --output estimate-work-load.json
npm run benchmark:estimate-work -- --channel msedge --layouts clustered,moving --series 2 --warmups 6 --samples 12 --output estimate-work-edge.json
npm run benchmark:estimate-work -- --channel chrome --software --sizes 65536 --series 2 --warmups 6 --samples 12 --output estimate-work-software.json
npm run benchmark:estimate-work -- --channel chrome --mobile --sizes 65536 --series 2 --warmups 6 --samples 12 --output estimate-work-mobile.json
```

The recordings include browser versions, adapters, settings, and hashes of the benchmark code. [Raw data](https://github.com/IoanU/heterojs/releases/download/v0.2.0-beta.2/HeteroJS-estimate-work-results.json)
