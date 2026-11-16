import { describe, expect, it, vi } from 'vitest';
import { createInlineBudget, InlineCpu } from '../../src/inline-cpu.js';

describe('inline CPU', () => {
  it('yields after the turn budget, honors cancellation, and releases the lane', async () => {
    const cpu = new InlineCpu();
    const controller = new AbortController();
    const yieldTurn = vi.fn(async () => { controller.abort(new Error('cancelled at yield')); });
    vi.stubGlobal('scheduler', { postTask: yieldTurn });
    const destination = new Float32Array(3);
    const task = {
      workload: { kernel: { operation: 'vectorMultiply' as const }, left: Float32Array.of(2, 3, 4), right: Float32Array.of(5, 6, 7),
        units: 3, outputLength: 3, workPerUnit: 1 }, destination,
      start: 1, length: 2, runStarted: performance.now(), signal: controller.signal,
    };
    try {
      await expect(cpu.run(task, { ...createInlineBudget(), computeMs: 8 })).rejects.toThrow('cancelled at yield');
      expect(yieldTurn).toHaveBeenCalledOnce();
      expect(yieldTurn).toHaveBeenCalledWith(expect.any(Function), { priority: 'background' });
      expect(Array.from(destination)).toEqual([0, 0, 0]);
      expect(cpu.pendingChunks).toBe(0);
      await cpu.run({ ...task, signal: new AbortController().signal }, createInlineBudget());
      expect(Array.from(destination)).toEqual([0, 18, 28]);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
