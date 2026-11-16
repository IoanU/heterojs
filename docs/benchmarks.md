# HeteroJS benchmarks

These measurements were taken on an AMD Ryzen 7 5800H with integrated Radeon graphics. Latency depends on the device, operation, and contention. [Work-estimate benchmark](estimate-work.md)

## Built-in operations

For a 513 × 513 matrix in Chrome, median latency was 32.3 ms with `cpu-only`, 6.2 ms with `gpu-only`, and 5.7 ms with `adaptive`. The respective p95 values were 37.0, 8.2, and 7.7 ms. At 768 × 768, GPU-only p95 was 15.3 ms and adaptive p95 was 16.3 ms. Use the benchmark runner to reproduce the comparison.

```sh
npm run build
npm run benchmark:browser -- --channel chrome --comparison all --operation matrixMultiply --sizes 513,768
```

## GPU kernel chains

For multiply, threshold activation, and sum, keeping intermediate results on GPU gave a 2.34–3.17× median speedup over the same kernels with readback and re-upload between stages. At 1,048,576 elements per input, the medians were 5.50 ms and 17.45 ms, respectively. Every result matched the CPU reference sum.

```sh
npm run benchmark:gpu-graph -- --channel chrome
```

## Fractalia

The experimental integration measured the escape-time field for Mandelbrot, Julia, and Burning Ship. With identical previews, adaptive selection of complete plans beat both single backends in 9 of 9 cases at the final-pass median and 8 of 9 at p95. A Julia trial used a 3840 × 2307 raster. The final-pass medians were 179.4 ms with adaptive scheduling and 239.8 ms on CPU. End-to-end frame medians, including Canvas, were 709.8 ms and 688.9 ms, respectively. [Original Fractalia project](https://github.com/IoanU/fractalia)

The experimental WebGPU shader uses different arithmetic from the Float64 CPU implementation. Sensitive orbits can produce different colors. The study reports image fidelity separately from latency.
