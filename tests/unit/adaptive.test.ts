import { describe, expect, it } from 'vitest';
import { planCpuElements, runAdaptive, shouldInitializeGpu } from '../../src/adaptive.js';
import type { Device } from '../../src/adaptive.js';
import { CostModel } from '../../src/cost-model.js';
import type { Backend, ChunkProfile } from '../../src/types.js';

const length = 2 ** 20 + 7;
const signal = new AbortController().signal;

function device(kind: Backend, lanes: number, elapsed: (length: number) => number, chunks: ChunkProfile[]): Device {
  let pending = 0;
  return {
    kind, lanes, maxChunkElements: 2 ** 20, pendingChunks: () => pending,
    async dispatch(start, length) {
      pending++;
      await Promise.resolve();
      pending--;
      const profile: ChunkProfile = {
        backend: kind, start, length, startedAtMs: 0, elapsedMs: elapsed(length), queueMs: 0, dispatchMs: 0,
        inputTransferMs: 0, outputTransferMs: 0, inputBytes: length * 8, outputBytes: length * 4,
        computeMs: null, synchronizationMs: null,
      };
      chunks.push(profile);
      return profile;
    },
  };
}

describe('adaptive scheduling', () => {
  it('estimates fixed and marginal costs and responds to new samples', () => {
    const model = new CostModel();
    model.observe(100_000, 3);
    model.observe(300_000, 7);
    expect(model.estimate?.fixedMs).toBeCloseTo(1);
    expect(model.estimate?.msPerElement).toBeCloseTo(0.00002, 8);
    for (let index = 0; index < 20; index++) model.observe(300_000, 30);
    const estimate = model.estimate!;
    expect(estimate.fixedMs + estimate.msPerElement * 300_000).toBeCloseTo(30);
  });

  it('weighs a mixed split against single devices and their queue occupancy', () => {
    const idle = { lanes: 1, chunkSize: 2 ** 20, readyMs: 0, estimate: { samples: 2, fixedMs: 0.3, msPerElement: 0.00001 } };
    expect(planCpuElements(800_000, idle, idle)).toBe(400_000);
    expect(planCpuElements(800_000, idle, { ...idle, readyMs: 30 })).toBe(800_000);
    expect(planCpuElements(800_000, { ...idle, readyMs: 30 }, idle)).toBe(0);
    expect(planCpuElements(20, idle, idle)).toBe(20);
  });

  it('plans a split using additive range costs rather than counting equally sized blocks', () => {
    const idle = { lanes: 1, chunkSize: 1000, readyMs: 0, estimate: { samples: 2, fixedMs: 0.2, msPerElement: 0.02 } };
    const prefix = (length: number) => length + Math.min(length, 100) * 9;
    expect(planCpuElements(1000, idle, idle)).toBe(500);
    expect(planCpuElements(1000, idle, idle, { total: prefix(1000), prefix })).toBe(95);
  });

  it('shrinks expensive ranges and dispatches the complete selection exactly once', async () => {
    const chunks: ChunkProfile[] = [];
    const models = { cpu: new CostModel(), gpu: new CostModel() };
    for (const size of [100, 500]) models.cpu.observe(size, 0.4 + size * 0.01);
    const cpu = device('cpu', 1, (size) => 0.4 + size * 0.01, chunks);
    const estimateWork = (start: number, count: number) => count + 19 * Math.max(0, Math.min(count, 512 - start));
    await runAdaptive({ elements: 1024, cpu, models, refresh: false, signal, probeElements: 64, estimateWork });
    expect(chunks[0].length).toBeLessThan(30);
    let next = 0;
    for (const chunk of chunks) { expect(chunk.start).toBe(next); next += chunk.length; }
    expect(next).toBe(1024);
  });

  it('can initialize WebGPU sooner when useful inline samples predict an expensive job', () => {
    const model = new CostModel();
    expect(shouldInitializeGpu(2 ** 20, model)).toBe(false);
    expect(shouldInitializeGpu(2 ** 23, model)).toBe(true);
    model.observe(2 ** 18, 4);
    model.observe(2 ** 19, 8);
    expect(shouldInitializeGpu(2 ** 20, model)).toBe(true);
    expect(shouldInitializeGpu(2 ** 20)).toBe(true);
  });

  it('samples both available devices with output work, then resizes and covers every element once', async () => {
    const chunks: ChunkProfile[] = [];
    const models = { cpu: new CostModel(), gpu: new CostModel() };
    const cpu = device('cpu', 2, (size) => 0.5 + size * 0.00001, chunks);
    const gpu = device('gpu', 1, (size) => 5 + size * 0.00003, chunks);
    await runAdaptive({ elements: length, cpu, gpu: { device: gpu }, models, refresh: false, signal });
    expect(chunks.filter((chunk) => chunk.exploratory).map((chunk) => chunk.backend).sort()).toEqual(['cpu', 'gpu']);
    expect(new Set(chunks.filter((chunk) => chunk.backend === 'cpu').map((chunk) => chunk.length)).size).toBeGreaterThan(1);
    let offset = 0;
    for (const chunk of chunks.sort((a, b) => a.start - b.start)) {
      expect(chunk.start).toBe(offset);
      offset += chunk.length;
    }
    expect(offset).toBe(length);
    expect(models.cpu.estimate?.samples).toBeGreaterThan(1);
    expect(models.gpu.estimate?.samples).toBeGreaterThan(0);
  });

  it.each(['cpu', 'gpu'] as const)('can use %s exclusively when recent measurements favor it', async (fast) => {
    const chunks: ChunkProfile[] = [];
    const models = { cpu: new CostModel(), gpu: new CostModel() };
    const slow = fast === 'cpu' ? 'gpu' : 'cpu';
    const quickTime = (size: number) => 0.3 + size * 0.000001;
    const slowTime = (size: number) => 100 + size * 0.00001;
    for (const size of [65_536, 500_000]) {
      models[fast].observe(size, quickTime(size));
      models[slow].observe(size, slowTime(size));
    }
    const cpu = device('cpu', 2, fast === 'cpu' ? quickTime : slowTime, chunks);
    const gpu = device('gpu', 1, fast === 'gpu' ? quickTime : slowTime, chunks);
    await runAdaptive({ elements: length, cpu, gpu: { device: gpu }, models, refresh: false, signal });
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks.every((chunk) => chunk.backend === fast && !chunk.exploratory)).toBe(true);
  });

  it('can finish on CPU while device initialization is pending', async () => {
    const chunks: ChunkProfile[] = [];
    const cpu = device('cpu', 4, (size) => 1 + size * 0.00001, chunks);
    const delayedGpu = new Promise<never>(() => {});
    await runAdaptive({ elements: length, cpu, gpu: delayedGpu, models: { cpu: new CostModel(), gpu: new CostModel() }, refresh: false, signal });
    expect(chunks.reduce((total, chunk) => total + chunk.length, 0)).toBe(length);
  });

  it('refreshes a previously unused device when asked to explore again', async () => {
    const chunks: ChunkProfile[] = [];
    const models = { cpu: new CostModel(), gpu: new CostModel() };
    models.cpu.observe(65_536, 0.5);
    models.gpu.observe(65_536, 30);
    const cpu = device('cpu', 2, () => 0.5, chunks);
    const gpu = device('gpu', 1, () => 30, chunks);
    await runAdaptive({ elements: length, cpu, gpu: { device: gpu }, models, refresh: true, signal });
    expect(chunks.filter((chunk) => chunk.exploratory).map((chunk) => chunk.backend).sort()).toEqual(['cpu', 'gpu']);
  });

  it('rejects a failed dispatch and wakes when outstanding work is cancelled', async () => {
    const models = { cpu: new CostModel(), gpu: new CostModel() };
    const cpu = device('cpu', 1, () => 1, []);
    const gpu = { reason: 'No GPU' };
    const failed = { ...cpu, dispatch: () => { throw new Error('dispatch failed'); } };
    await expect(runAdaptive({ elements: length, cpu: failed, gpu, models, refresh: false, signal })).rejects.toThrow('dispatch failed');
    const controller = new AbortController();
    const pending = { ...cpu, pendingChunks: () => 1, dispatch: () => new Promise<never>(() => {}) };
    const running = runAdaptive({ elements: length, cpu: pending, gpu, models, refresh: false, signal: controller.signal });
    controller.abort(new Error('stop'));
    await expect(running).rejects.toThrow('stop');
  });
});
