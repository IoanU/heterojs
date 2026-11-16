import CpuWorker from './cpu.worker?worker&inline';
import type { BuiltInWorkerRequest, ChunkResult, ChunkTask, CustomChunkTask, WorkerResponse } from './task.js';
import type { ChunkProfile } from './types.js';
import { inputViews } from './workload.js';
import type { Kernel } from './workload.js';

interface Pending {
  task: ChunkTask | CustomChunkTask;
  queuedAt: number;
  resolve: (result: { response: Exclude<WorkerResponse, { kind: 'error' }>; profile: ChunkProfile }) => void;
  reject: (error: unknown) => void;
  onAbort: () => void;
}

interface Active {
  pending: Pending;
  startedAt: number;
  inputTransferMs: number;
  dispatchMs: number;
  workerStartup: boolean;
  inputBytes: number;
}

interface Slot {
  worker: Worker;
  active?: Active;
  matrix?: Kernel;
}

export class CpuPool {
  private readonly slots = new Set<Slot>();
  private readonly queue: Pending[] = [];
  private closed = false;

  constructor(readonly size: number) {}

  get pendingChunks() {
    let pending = this.queue.length;
    for (const slot of this.slots) if (slot.active) pending++;
    return pending;
  }

  async run(task: ChunkTask): Promise<ChunkResult> {
    const { response, profile } = await this.enqueue(task);
    if ('kind' in response) throw new Error('Unexpected custom CPU response.');
    return { output: response.output, partial: response.partial, profile };
  }

  async runCustom(task: CustomChunkTask) {
    const { response, profile } = await this.enqueue(task);
    if (!('kind' in response)) throw new Error('Unexpected built-in CPU response.');
    return { value: response.value, profile };
  }

  private enqueue(task: Pending['task']): Promise<Parameters<Pending['resolve']>[0]> {
    return new Promise((resolve, reject) => {
      task.signal.throwIfAborted();
      if (this.closed) throw new Error('The CPU worker pool is closed.');
      const pending: Pending = {
        task, queuedAt: performance.now(), resolve, reject,
        onAbort: () => {
          const index = this.queue.indexOf(pending);
          if (index !== -1) this.queue.splice(index, 1);
          else if (!('workload' in task) && !this.closed) {
            const slot = [...this.slots].find((candidate) => candidate.active?.pending === pending);
            if (slot) {
              this.failSlot(slot, task.signal.reason);
              return;
            }
          }
          reject(task.signal.reason);
        },
      };
      task.signal.addEventListener('abort', pending.onAbort, { once: true });
      this.queue.push(pending);
      this.dispatch();
    });
  }

  close(reason: unknown) {
    this.closed = true;
    for (const pending of this.queue.splice(0)) this.reject(pending, reason);
    for (const slot of this.slots) {
      if (slot.active) this.reject(slot.active.pending, reason);
      slot.worker.terminate();
    }
    this.slots.clear();
  }

  private dispatch() {
    while (!this.closed && this.queue.length > 0) {
      let slot = [...this.slots].find((candidate) => !candidate.active);
      if (!slot && this.slots.size === this.size) return;
      const pending = this.queue.shift()!;
      const startedAt = performance.now();
      try {
        const workerStartup = slot === undefined;
        slot ??= this.createSlot();
        const active: Active = { pending, startedAt, workerStartup, inputBytes: 0, inputTransferMs: 0, dispatchMs: 0 };
        slot.active = active;
        const { inputBytes, inputTransferMs } = this.send(slot, pending.task);
        active.inputBytes = inputBytes;
        active.inputTransferMs = inputTransferMs;
        active.dispatchMs = performance.now() - startedAt - inputTransferMs;
        if (slot.active === active && !('workload' in pending.task) && pending.task.signal.aborted) {
          this.failSlot(slot, pending.task.signal.reason);
        }
      } catch (error) {
        if (slot?.active?.pending === pending) slot.active = undefined;
        this.reject(pending, error);
      }
    }
  }

  private send(slot: Slot, task: Pending['task']) {
    const copyStarted = performance.now();
    const { start, length } = task;
    if (!('workload' in task)) {
      const { moduleUrl, exportName, transferOutput, transferInput, input, totalUnits } = task;
      const range = { start, length, totalUnits };
      const transfers = transferInput ? transferInput(input, Object.freeze(range)) : [];
      if (!Array.isArray(transfers) || transfers.some((buffer) => !(buffer instanceof ArrayBuffer))) {
        throw new TypeError('CPU transferInput must return an array of ArrayBuffers.');
      }
      const inputBytes = transfers.reduce((bytes, buffer) => bytes + buffer.byteLength, 0);
      task.signal.throwIfAborted();
      slot.worker.postMessage({ kind: 'custom', moduleUrl, exportName, transferOutput, input, range }, transfers);
      slot.matrix = undefined;
      return { inputBytes, inputTransferMs: performance.now() - copyStarted };
    }
    const { workload } = task;
    const { left, right } = inputViews(workload, start, length);
    const matrix = workload.kernel.operation === 'matrixMultiply' ? workload.kernel : undefined;
    const transferRight = !matrix || slot.matrix !== matrix ? right : undefined;
    const request: BuiltInWorkerRequest = { kernel: workload.kernel, length, left: left.slice(), right: transferRight?.slice() };
    const inputBytes = left.byteLength + (transferRight?.byteLength ?? 0);
    const inputTransferMs = performance.now() - copyStarted;
    const transfers = request.right ? [request.left.buffer, request.right.buffer] : [request.left.buffer];
    slot.worker.postMessage(request, transfers);
    slot.matrix = matrix;
    return { inputBytes, inputTransferMs };
  }

  private createSlot(): Slot {
    const worker = new CpuWorker();
    const slot: Slot = { worker };
    this.slots.add(slot);
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      const active = slot.active;
      if (!active) return;
      slot.active = undefined;
      const { pending, startedAt, inputTransferMs, dispatchMs, workerStartup, inputBytes } = active;
      const { task } = pending;
      task.signal.removeEventListener('abort', pending.onAbort);
      if ('kind' in data && data.kind === 'error') {
        const error = new Error(data.message);
        error.name = data.name;
        pending.reject(error);
        this.dispatch();
        return;
      }
      pending.resolve({
        response: data,
        profile: {
          backend: 'cpu', cpuExecution: 'worker', start: task.start, length: task.length,
          ...(workerStartup && { workerStartup }),
          startedAtMs: startedAt - task.runStarted,
          elapsedMs: performance.now() - startedAt,
          queueMs: startedAt - pending.queuedAt,
          dispatchMs, computeMs: data.computeMs, inputTransferMs, outputTransferMs: 0,
          inputBytes, outputBytes: 'kind' in data ? data.outputBytes ?? 0 : data.output?.byteLength ?? 8, synchronizationMs: null,
        },
      });
      this.dispatch();
    };
    worker.onerror = (event) => {
      event.preventDefault();
      this.failSlot(slot, new Error(event.message || 'A CPU worker failed.'));
    };
    worker.onmessageerror = () => this.failSlot(slot, new Error('A CPU worker response could not be read.'));
    return slot;
  }

  private failSlot(slot: Slot, error: unknown) {
    if (!this.slots.delete(slot)) return;
    slot.worker.terminate();
    if (slot.active) this.reject(slot.active.pending, error);
    this.dispatch();
  }

  private reject(pending: Pending, error: unknown) {
    pending.task.signal.removeEventListener('abort', pending.onAbort);
    pending.reject(error);
  }
}
