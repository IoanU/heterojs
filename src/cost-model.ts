import type { AdaptiveEstimate } from './types.js';

const MAX_SAMPLES = 12;
const MIN_TIME_MS = 0.05;

export class CostModel {
  private readonly recent: { elements: number; elapsedMs: number }[] = [];
  private current?: AdaptiveEstimate;

  get estimate() { return this.current; }

  observe(elements: number, elapsedMs: number) {
    this.recent.unshift({ elements, elapsedMs: Math.max(MIN_TIME_MS, elapsedMs) });
    if (this.recent.length > MAX_SAMPLES) this.recent.pop();

    let weight = 1;
    let weights = 0;
    let sizes = 0;
    let times = 0;
    let sizeSquared = 0;
    let sizeTime = 0;
    for (const sample of this.recent) {
      weights += weight;
      sizes += weight * sample.elements;
      times += weight * sample.elapsedMs;
      sizeSquared += weight * sample.elements ** 2;
      sizeTime += weight * sample.elements * sample.elapsedMs;
      weight *= 0.75;
    }
    const size = sizes / weights;
    const time = times / weights;
    const variance = sizeSquared / weights - size ** 2;
    // Similar chunk sizes cannot identify both coefficients independently.
    const slope = variance > size ** 2 * 0.04 ? (sizeTime / weights - size * time) / variance : 0.75 * time / size;
    const msPerElement = Math.min(time / size, Math.max(0.0625 * time / size, slope));
    this.current = { samples: this.recent.length, fixedMs: Math.max(0, time - msPerElement * size), msPerElement };
  }
}
