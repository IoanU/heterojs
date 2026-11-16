# Cross-device benchmarks

Open the [browser suite](https://ioanu.github.io/heterojs/device-benchmarks.html) on each computer or phone. The benchmarks and report comparison run in the browser. Exports are local JSON downloads.

1. Enter a label with the device model, CPU or SoC, and GPU if known. The report also records the browser user agent, the logical core and worker counts, and the WebGPU adapter details exposed by the browser.
2. Select the same profile on each device. **Quick** runs four workloads with eight warm-ups and eight timed calls per strategy. **Standard** runs seven workloads with ten warm-ups and 20 timed calls per strategy. Each also records a first call on a fresh benchmark engine before warm-up. Use Standard for a less coarse p95.
3. Keep the tab visible and use comparable browser, power, and thermal conditions. Record any deliberate differences in the device label. Run the suite again to check whether close results repeat.
4. Export JSON on each device and move the files to the device where you want to compare them. Open **Compare devices** and import all files. Select the profile and median or p95. The page compares reports from the same suite and engine version.

## Reading the comparison

- **Adaptive latency** is an absolute time. Compare this column to see which device finished the same workload sooner.
- **Vs. CPU**, **Vs. GPU**, and **Vs. static** divide the corresponding strategy's latency by adaptive latency on that same device. **1.20×** means the baseline took 1.20 times as long as adaptive. **Vs. best CPU/GPU** uses the faster of the two single-backend strategies. Values above 1× mean adaptive beat both. A missing or invalid baseline produces a dash. Browser-reported fallback adapters are identified in the device information.
- **Median** and **p95** use only timed calls after warm-up. p95 uses the nearest-rank rule. With eight timed calls, it is the observed maximum. Benchmark order rotates within each case. Results retain the individual samples and actual backend allocation.
- The vector and matrix cases reuse the HeteroJS browser harness, which compares each output against a CPU reference. Worker counts scale with the device, so this is a comparison at each device's default settings.

## Fractalia

The suite uses the Fractalia worker, CPU and WebGPU kernels, and colorizer from integration snapshot `fb70975499166771e4f61e7a0d3800c2b5ab419c`. The modules are included in this repository under `public/fractalia/`. The fixed raster and iteration count are independent of browser window size and screen pixel density. Both previews use CPU workers in all HeteroJS strategies. Stock Fractalia uses its original worker and lane policy. The harness draws all three passes with Canvas 2D using the Fractalia colorizer. [Fractalia project](https://github.com/IoanU/fractalia)

- **Compute** measures HeteroJS's final, full-resolution pass. Stock Fractalia is included for preview and frame timings.
- **First preview** includes calculation, coloring, and submitting the first Canvas draw. **Full frame** includes these steps for all three passes. The timer stops after calling the Canvas draw API. Display compositor timing and the interactive app's other features are outside this test.
- The final timed frame for each mode is compared with the stock worker's field. The CPU-only path must match exactly. GPU-backed modes can have floating-point differences. The results show the percentage of pixels where at least one RGB channel changed by more than 8 on a 0–255 scale.

## Running from the repository

```sh
npm ci
npm run build
npm run preview
```

Open **Cross-device benchmarks and Fractalia** from the local benchmark home page. For other devices, serve `dist/site` over HTTPS or use the hosted suite. WebGPU works on supported browsers in secure contexts such as HTTPS or localhost. The suite can still measure CPU execution on a browser without a WebGPU adapter.
