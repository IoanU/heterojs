import { describe, expect, it } from 'vitest';
import { cpuChunkSize, MAX_CHUNK_ELEMENTS, partition, validateCpuShare, validateStrategy } from '../../src/partition.js';
import type { Strategy } from '../../src/types.js';

describe('static partitioning', () => {
  it.each<[Exclude<Strategy, 'adaptive'>, number, number, number]>([
    ['cpu-only', 16, .25, 16], ['gpu-only', 16, .25, 0], ['static', 17, .25, 4],
    ['static', 6, .25, 2], ['static', 16, 0, 0], ['static', 16, 1, 16], ['static', 0, .25, 0],
  ])('assigns %s, length %i, share %f', (strategy, length, share, cpuElements) => {
    expect(partition(length, strategy, share)).toEqual({ cpuElements, gpuElements: length - cpuElements });
  });

  it('bounds CPU chunks and spreads small workloads across workers', () => {
    expect(cpuChunkSize(0, 8)).toBe(1);
    expect(cpuChunkSize(17, 4)).toBe(5);
    expect(cpuChunkSize(MAX_CHUNK_ELEMENTS * 12, 4)).toBe(MAX_CHUNK_ELEMENTS);
  });

  it('validates strategies and shares', () => {
    expect(() => validateStrategy('invalid' as Strategy)).toThrow(TypeError);
    expect(validateStrategy('adaptive')).toBe('adaptive');
    for (const bad of [NaN, Infinity, -0.01, 1.01]) expect(() => validateCpuShare(bad)).toThrow(RangeError);
    expect(validateCpuShare(0)).toBe(0);
    expect(validateCpuShare(1)).toBe(1);
  });
});
