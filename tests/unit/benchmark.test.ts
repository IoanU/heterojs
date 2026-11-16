import { describe, expect, it } from 'vitest';
import { median, p95, parseSizes, speedup, valid } from '../../benchmark/runner.js';
import type { Row } from '../../benchmark/runner.js';

function row(strategy: Row['strategy'], times: number[]): Row {
  const requested = strategy === 'cpu-direct' ? 'cpu-only' : strategy;
  const sample = (totalMs: number) => ({
    profile: {
      totalMs, strategy: requested, requestedStrategy: requested, cpuElements: 0, gpuElements: 0,
      elements: 32, setupMs: 0, schedulingMs: 0, aggregationMs: 0, chunks: [],
    },
    mismatches: 0,
    eventLoopDelayMs: 0,
  });
  return { size: 32, strategy, cold: sample(20), warmups: [], warm: times.map(sample) };
}

describe('benchmark statistics', () => {
  it('parses distinct sizes and rejects invalid inputs', () => {
    expect(parseSizes('65 128, 65\n1024')).toEqual([65, 128, 1024]);
    for (const bad of ['', '1', '1.5', 'a', '16777217', '2 3 4 5 6 7 8 9 10']) expect(() => parseSizes(bad)).toThrow();
  });

  it('uses odd and even warm-sample medians', () => {
    expect(median([3, 10, 4])).toBe(4);
    expect(median([3, 10, 4, 5])).toBe(4.5);
  });

  it('uses nearest rank for p95, even for short series', () => {
    expect(p95([1])).toBe(1);
    expect(p95(Array.from({ length: 20 }, (_, i) => 20 - i))).toBe(19);
  });

  it('compares to the faster verified single device and excludes failed or missing samples', () => {
    const rows = [row('cpu-only', [5, 6, 7]), row('gpu-only', [10, 11, 12]), row('static', [3, 4, 5])];
    expect(speedup(rows[2], rows)).toBe(1.5);
    expect(speedup(rows[2], [...rows, row('cpu-direct', [2, 3, 4])])).toBe(1.5);
    expect(speedup(rows[2], [rows[0], rows[2]])).toBeUndefined();
    rows[2].warm[1].mismatches = 1;
    expect(valid(rows[2])).toBe(false);
    expect(speedup(rows[2], rows)).toBeUndefined();
  });
});
