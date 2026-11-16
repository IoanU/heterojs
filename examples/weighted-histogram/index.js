import { histogramGpu } from './gpu.js';

export function registerWeightedHistogram(engine) {
  return engine.registerOperation('weightedHistogram', {
    cpu: { module: new URL('./cpu.js', import.meta.url) },
    gpu: histogramGpu,
    plan({ keys, weights, binCount }) {
      if (!(keys instanceof Uint32Array) || !(weights instanceof Uint32Array) || keys.length !== weights.length) {
        throw new TypeError('Expected Uint32Array keys and weights of equal length.');
      }
      if (!Number.isSafeInteger(binCount) || binCount < 1 || keys.some((key) => key >= binCount)) {
        throw new RangeError('Keys must be valid indices into the histogram.');
      }
      return {
        units: keys.length,
        maxChunkUnits: 2 ** 18,
        chunk({ start, length }) {
          const end = start + length;
          return { keys: keys.slice(start, end), weights: weights.slice(start, end), binCount };
        },
        combine(parts) {
          const output = new Uint32Array(binCount);
          for (const { value: bins } of parts) {
            for (let bin = 0; bin < binCount; bin++) output[bin] += bins[bin];
          }
          return output;
        },
      };
    },
  });
}
