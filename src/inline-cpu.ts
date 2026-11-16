import type { ChunkResult, ChunkTask } from './task.js';
import type { InlineTiming } from './types.js';
import { combineReduction, executeKernel, reductionIdentity, transposeMatrixRight } from './workload.js';
import type { Kernel } from './workload.js';

export const MAX_INLINE_CHUNK_ELEMENTS = 2 ** 22;
const SLICE_ELEMENTS = 2 ** 18;
const TURN_BUDGET_MS = 8;

export interface InlineBudget extends InlineTiming { computeMs: number }

export function createInlineBudget(): InlineBudget {
  return { computeMs: 0, longestTurnComputeMs: 0, yields: 0, yieldWaitMs: 0 };
}

export function inlineTiming({ longestTurnComputeMs, yields, yieldWaitMs }: InlineBudget): InlineTiming {
  return { longestTurnComputeMs, yields, yieldWaitMs };
}

function yieldToBrowser() {
  const scheduler = (globalThis as {
    scheduler?: { postTask?: (callback: () => void, options: { priority: 'background' }) => Promise<void> };
  }).scheduler;
  // A boosted scheduler.yield() continuation can keep overtaking timers in a long chain.
  if (typeof scheduler?.postTask === 'function') return scheduler.postTask(() => {}, { priority: 'background' });
  return new Promise<void>((resolve) => setTimeout(resolve, 0));
}

export class InlineCpu {
  private tail?: Promise<void>;
  private pending = 0;
  private readonly transposes = new WeakMap<Kernel, Float32Array>();

  get pendingChunks() { return this.pending; }

  run(task: ChunkTask, budget: InlineBudget): Promise<ChunkResult> {
    const queuedAt = performance.now();
    this.pending++;
    const execute = () => this.execute(task, budget, queuedAt);
    const result = (this.tail ? this.tail.then(execute) : execute()).finally(() => { this.pending--; });
    const tail = result.then(() => undefined, () => undefined);
    this.tail = tail;
    void tail.then(() => { if (this.tail === tail) this.tail = undefined; });
    return result;
  }

  private async execute(task: ChunkTask, budget: InlineBudget, queuedAt: number): Promise<ChunkResult> {
    const { workload, destination, start, length, signal } = task;
    const { kernel, left, right, reduction, workPerUnit } = workload;
    const startedAt = performance.now();
    signal.throwIfAborted();
    let computeMs = 0;
    const account = (before: number) => {
      const elapsed = performance.now() - before;
      budget.computeMs += elapsed;
      budget.longestTurnComputeMs = Math.max(budget.computeMs, budget.longestTurnComputeMs);
      computeMs += elapsed;
    };
    let cpuRight = right;
    if (kernel.operation === 'matrixMultiply') {
      cpuRight = this.transposes.get(kernel);
      if (!cpuRight) {
        const before = performance.now();
        cpuRight = transposeMatrixRight(right!, kernel.shape);
        this.transposes.set(kernel, cpuRight);
        account(before);
      }
    }
    let partial = reduction ? reductionIdentity(reduction) : undefined;
    const end = start + length;
    const slice = Math.max(1, Math.floor(SLICE_ELEMENTS / workPerUnit));
    for (let offset = start; offset < end; offset += slice) {
      signal.throwIfAborted();
      if (budget.computeMs >= TURN_BUDGET_MS) {
        const yieldStarted = performance.now();
        await yieldToBrowser();
        budget.yields++;
        budget.yieldWaitMs += performance.now() - yieldStarted;
        budget.computeMs = 0;
        signal.throwIfAborted();
      }
      const before = performance.now();
      const value = executeKernel(kernel, left, cpuRight, destination, offset, Math.min(end, offset + slice));
      if (reduction) partial = combineReduction(reduction, partial!, value!);
      account(before);
    }
    return { partial, profile: {
      backend: 'cpu', cpuExecution: 'inline', start, length, startedAtMs: startedAt - task.runStarted,
      elapsedMs: performance.now() - startedAt, queueMs: startedAt - queuedAt, dispatchMs: 0, computeMs,
      inputTransferMs: 0, outputTransferMs: 0, inputBytes: 0, outputBytes: 0, synchronizationMs: null,
    } };
  }
}
