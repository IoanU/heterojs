import type { Strategy } from './types.js';

export const MAX_CHUNK_ELEMENTS = 2 ** 20;

export function validateStrategy(strategy: Strategy): Strategy {
  if (strategy !== 'cpu-only' && strategy !== 'gpu-only' && strategy !== 'static' && strategy !== 'adaptive') {
    throw new TypeError(`Unsupported strategy: ${strategy}`);
  }
  return strategy;
}

export function validateCpuShare(share: number): number {
  if (!Number.isFinite(share) || share < 0 || share > 1) {
    throw new RangeError('cpuShare must be a number between 0 and 1.');
  }
  return share;
}

export function partition(length: number, strategy: Exclude<Strategy, 'adaptive'>, share: number) {
  const cpuElements = strategy === 'cpu-only' ? length
    : strategy === 'gpu-only' ? 0
    : Math.round(length * share);
  return { cpuElements, gpuElements: length - cpuElements };
}

export function cpuChunkSize(length: number, workers: number) {
  return Math.max(1, Math.min(MAX_CHUNK_ELEMENTS, Math.ceil(length / workers)));
}
