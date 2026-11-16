import { MAX_CHUNK_ELEMENTS } from './partition.js';
import { awaitAbortable } from './abort.js';
import { GpuGraphExecutor } from './gpu-graph.js';
import type { GpuGraphPlan } from './gpu-graph.js';
import type { ChunkResult, ChunkTask } from './task.js';
import type { ChunkProfile, CustomGpuKernel, CustomRange, EngineOptions, GpuCapability, Operation } from './types.js';
import { foldReduction, inputViews, outputSize } from './workload.js';
import type { Kernel, Workload } from './workload.js';
import vectorSource from './vector-multiply.wgsl?raw';
import matrixSource from './matrix-multiply.wgsl?raw';
import reduceSource from './reduce.wgsl?raw';

const WORKGROUP_SIZE = 64;
const TILE_SIZE = 8;
const TIMESTAMP_BYTES = 16;
const MAX_MATRIX_PRODUCTS_PER_DISPATCH = 2 ** 29;

interface Buffers {
  left: GPUBuffer;
  right?: GPUBuffer;
  output: GPUBuffer;
  readback: GPUBuffer;
}

type GpuInit = { backend: GpuBackend } | { reason: string };

export class GpuDeviceLostError extends Error {
  constructor(readonly reason: GPUDeviceLostReason, message: string) {
    super(message || 'The WebGPU device was lost.');
    this.name = 'GpuDeviceLostError';
  }
}

function timestampOffset(bytes: number) {
  return Math.ceil(bytes / 8) * 8;
}

async function checked<T>(device: GPUDevice, operation: () => Promise<T>, asyncCommands = false): Promise<T> {
  const filters = ['internal', 'out-of-memory', 'validation'] as const;
  for (const filter of filters) device.pushErrorScope(filter);
  let result: Promise<T>;
  try {
    result = operation();
  } catch (error) {
    result = Promise.reject(error);
  }
  if (asyncCommands) await Promise.allSettled([result]);
  const scopes = filters.map(() => device.popErrorScope());
  const [value, ...errors] = await Promise.allSettled([result, ...scopes] as const);
  for (const error of errors) {
    if (error.status === 'rejected') throw error.reason;
    if (error.value) throw new Error(`WebGPU error: ${error.value.message}`);
  }
  if (value.status === 'rejected') throw value.reason;
  return value.value;
}

export class GpuBackend {
  private readonly maxStorageBytes: number;
  private readonly maxOutputBytes: number;
  private readonly parameters: GPUBuffer;
  private readonly pipelines = new Map<Operation, Promise<GPUComputePipeline>>();
  private readonly timer?: { queries: GPUQuerySet; resolve: GPUBuffer };
  private buffers?: Buffers;
  private matrixRight?: Kernel;
  private tail: Promise<void> = Promise.resolve();
  private queuedChunks = 0;
  private readonly lifetime = new AbortController();
  private readonly graph: GpuGraphExecutor;

  private constructor(
    private readonly device: GPUDevice,
    private readonly adapterInfo: GPUAdapterInfo,
  ) {
    this.graph = new GpuGraphExecutor(device);
    if (device.features.has('timestamp-query')) {
      this.timer = {
        queries: device.createQuerySet({ type: 'timestamp', count: 2 }),
        resolve: device.createBuffer({
          size: TIMESTAMP_BYTES, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
      };
    }
    this.maxStorageBytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    this.maxOutputBytes = Math.min(this.maxStorageBytes, device.limits.maxBufferSize - (this.timer ? TIMESTAMP_BYTES + 4 : 0));
    this.parameters = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    void device.lost.then(({ reason, message }) => this.invalidate(new GpuDeviceLostError(reason, message)));
  }

  static async open(options?: EngineOptions['gpuDevice']): Promise<GpuInit> {
    if (!globalThis.navigator?.gpu) {
      return { reason: 'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.' };
    }
    let device: GPUDevice | undefined;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { reason: 'No WebGPU adapter is available.' };
      const requiredFeatures = new Set(options?.requiredFeatures);
      if (adapter.features.has('timestamp-query')) requiredFeatures.add('timestamp-query');
      device = await adapter.requestDevice({ requiredFeatures, requiredLimits: options?.requiredLimits });
      const gpu = device;
      const backend = await checked(gpu, async () => new GpuBackend(gpu, adapter.info));
      return { backend };
    } catch (error) {
      device?.destroy();
      return { reason: `WebGPU initialization failed: ${error instanceof Error ? error.message : String(error)}` };
    }
  }

  get capability(): GpuCapability {
    const { isFallbackAdapter, vendor, architecture } = this.adapterInfo;
    return this.lost
      ? { available: false, reason: this.lifetime.signal.reason.message }
      : { available: true, timestampQueries: Boolean(this.timer), fallbackAdapter: isFallbackAdapter, vendor, architecture };
  }

  get pendingChunks() { return this.queuedChunks; }
  get lost() { return this.lifetime.signal.aborted; }

  chunkLimit(workload: Workload) {
    const groups = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (workload.kernel.operation !== 'matrixMultiply') {
      return Math.min(MAX_CHUNK_ELEMENTS, Math.floor(this.maxOutputBytes / 4), groups * WORKGROUP_SIZE);
    }
    const { inner, columns } = workload.kernel.shape;
    if (workload.right!.byteLength > this.maxStorageBytes || Math.ceil(columns / TILE_SIZE) > groups) return 0;
    return Math.min(MAX_CHUNK_ELEMENTS, Math.floor(this.maxStorageBytes / (4 * inner)),
      Math.floor(this.maxOutputBytes / (4 * columns)), groups * TILE_SIZE,
      Math.max(1, Math.floor(MAX_MATRIX_PRODUCTS_PER_DISPATCH / workload.workPerUnit)));
  }

  run(task: ChunkTask): Promise<ChunkResult> {
    return this.enqueue((queuedAt) => this.execute(task, queuedAt));
  }

  runGraph(graph: GpuGraphPlan, runStarted: number, signal: AbortSignal) {
    return this.enqueue(async (queuedAt) => {
      signal.throwIfAborted();
      const result = await checked(this.device, () => this.graph.run(graph, runStarted, queuedAt, signal), true);
      signal.throwIfAborted();
      this.lifetime.signal.throwIfAborted();
      result.profile.totalMs = performance.now() - runStarted;
      return result;
    });
  }

  runCustom(kernel: CustomGpuKernel<unknown, unknown>, input: unknown, range: CustomRange,
    runStarted: number, signal: AbortSignal) {
    return this.enqueue(async (queuedAt) => {
      signal.throwIfAborted();
      const activeSignal = AbortSignal.any([signal, this.lifetime.signal]);
      const startedAt = performance.now();
      let synchronizationMs = 0;
      const value = await checked(this.device, async () => {
        try {
          return await kernel(input, { ...range, device: this.device, signal: activeSignal });
        } finally {
          const syncStarted = performance.now();
          await this.device.queue.onSubmittedWorkDone();
          synchronizationMs = performance.now() - syncStarted;
        }
      }, true);
      activeSignal.throwIfAborted();
      const completed = performance.now();
      const profile: ChunkProfile = {
        backend: 'gpu', start: range.start, length: range.length, startedAtMs: startedAt - runStarted,
        queueMs: startedAt - queuedAt, elapsedMs: completed - startedAt, computeMs: null,
        dispatchMs: 0, inputTransferMs: 0, outputTransferMs: 0, inputBytes: 0, outputBytes: 0,
        synchronizationMs,
      };
      return { value, profile };
    });
  }

  private enqueue<T>(execute: (queuedAt: number) => Promise<T>): Promise<T> {
    const queuedAt = performance.now();
    this.queuedChunks++;
    const signal = this.lifetime.signal;
    const queued = this.tail.then(() => {
      signal.throwIfAborted();
      return execute(queuedAt);
    });
    const result = awaitAbortable(queued, signal).finally(() => { this.queuedChunks--; });
    this.tail = result.then(() => undefined, () => undefined);
    return result;
  }

  close() {
    this.invalidate(new DOMException('The WebGPU backend is closed.', 'AbortError'));
    this.device.destroy();
  }

  private invalidate(reason: Error) {
    if (this.lost) return;
    this.lifetime.abort(reason);
    this.releaseBuffers();
    this.graph.clear();
    this.pipelines.clear();
    this.timer?.queries.destroy();
    this.timer?.resolve.destroy();
    this.parameters.destroy();
  }

  private async execute(task: ChunkTask, queuedAt: number): Promise<ChunkResult> {
    task.signal.throwIfAborted();
    const startedAt = performance.now();
    const { workload, start, length } = task;
    const { kernel, reduction } = workload;
    const { left, right } = inputViews(workload, start, length);
    const matrix = kernel.operation === 'matrixMultiply' ? kernel.shape : undefined;
    const [groupsX, groupsY] = matrix ? [Math.ceil(matrix.columns / TILE_SIZE), Math.ceil(length / TILE_SIZE)]
      : [Math.ceil(length / WORKGROUP_SIZE), 1];
    const count = reduction ? groupsX : outputSize(kernel, length);
    const bytes = count * 4;
    const timeOffset = timestampOffset(bytes);
    const mapSize = this.timer ? timeOffset + TIMESTAMP_BYTES : bytes;
    const { device, timer } = this;
    try {
      const gpuPipelineStartup = !this.pipelines.has(kernel.operation);
      const pipeline = await this.pipelineFor(kernel.operation);
      task.signal.throwIfAborted();
      const { buffers, inputBytes, inputTransferMs, dispatchMs, synchronizationMs } = await checked(device, async () => {
        const buffers = this.ensureBuffers(left.byteLength, right?.byteLength ?? 0, bytes);
        const uploadStarted = performance.now();
        device.queue.writeBuffer(buffers.left, 0, left.buffer, left.byteOffset, left.byteLength);
        let inputBytes = left.byteLength;
        if (right && (!matrix || this.matrixRight !== kernel)) {
          device.queue.writeBuffer(buffers.right!, 0, right.buffer, right.byteOffset, right.byteLength);
          this.matrixRight = matrix ? kernel : undefined;
          inputBytes += right.byteLength;
        }
        const parameters = matrix ? new Uint32Array([length, matrix.inner, matrix.columns, 0])
          : kernel.operation === 'scalarMultiply' ? new Float32Array([kernel.scalar, 0, 0, 0]) : undefined;
        if (parameters) {
          device.queue.writeBuffer(this.parameters, 0, parameters);
          inputBytes += parameters.byteLength;
        }
        const uploadedAt = performance.now();

        const entries: GPUBindGroupEntry[] = [
          { binding: 0, resource: { buffer: buffers.left, size: left.byteLength } },
          { binding: 2, resource: { buffer: buffers.output, size: bytes } },
        ];
        if (right) entries.push({ binding: 1, resource: { buffer: buffers.right!, size: right.byteLength } });
        if (parameters) entries.push({ binding: 3, resource: { buffer: this.parameters } });
        const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass(timer ? {
          timestampWrites: { querySet: timer.queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
        } : undefined);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(groupsX, groupsY);
        pass.end();
        encoder.copyBufferToBuffer(buffers.output, 0, buffers.readback, 0, bytes);
        if (timer) {
          encoder.resolveQuerySet(timer.queries, 0, 2, timer.resolve, 0);
          encoder.copyBufferToBuffer(timer.resolve, 0, buffers.readback, timeOffset, TIMESTAMP_BYTES);
        }
        device.queue.submit([encoder.finish()]);
        const submittedAt = performance.now();
        task.onGpuSubmitted?.();
        const completion = await Promise.allSettled([
          device.queue.onSubmittedWorkDone(), buffers.readback.mapAsync(GPUMapMode.READ, 0, mapSize),
        ]);
        for (const status of completion) {
          if (status.status === 'rejected') throw status.reason;
        }
        const synchronizationMs = performance.now() - submittedAt;
        const inputTransferMs = uploadedAt - uploadStarted;
        return { buffers, inputBytes, inputTransferMs, dispatchMs: submittedAt - startedAt - inputTransferMs, synchronizationMs };
      });

      const readStarted = performance.now();
      try {
        task.signal.throwIfAborted();
        this.lifetime.signal.throwIfAborted();
        const mapped = buffers.readback.getMappedRange(0, mapSize);
        const values = new Float32Array(mapped, 0, count);
        let partial: number | undefined;
        if (reduction) {
          partial = foldReduction(reduction, values);
          if (reduction === 'sum') partial *= WORKGROUP_SIZE;
        } else {
          task.destination.set(values, outputSize(kernel, start));
        }
        let computeMs = null;
        if (timer) {
          const [begin, end] = new BigUint64Array(mapped, timeOffset, 2);
          if (end >= begin) computeMs = Number(end - begin) / 1_000_000;
        }
        const finished = performance.now();
        return {
          partial, profile: {
            backend: 'gpu', start, length, startedAtMs: startedAt - task.runStarted,
            ...(gpuPipelineStartup && { gpuPipelineStartup }),
            elapsedMs: finished - startedAt, queueMs: startedAt - queuedAt,
            dispatchMs, computeMs, inputTransferMs, outputTransferMs: finished - readStarted,
            inputBytes, outputBytes: bytes, synchronizationMs,
          },
        };
      } finally {
        if (buffers.readback.mapState === 'mapped') buffers.readback.unmap();
      }
    } catch (error) {
      this.releaseBuffers();
      throw error;
    }
  }

  private pipelineFor(operation: Operation) {
    const existing = this.pipelines.get(operation);
    if (existing) return existing;
    const reductionModes = { sum: 0, min: 1, max: 2 } as const;
    const reduction = operation === 'sum' || operation === 'min' || operation === 'max';
    const matrix = operation === 'matrixMultiply';
    const entryPoint = reduction ? 'reduce' : matrix ? 'matrix' : operation === 'vectorMultiply' ? 'multiply'
      : operation === 'vectorAdd' ? 'add' : 'scale';
    const code = reduction ? reduceSource : matrix ? matrixSource : vectorSource;
    const constants: Record<string, number> | undefined = reduction ? { mode: reductionModes[operation] }
      : matrix ? undefined : { workgroupSize: WORKGROUP_SIZE };
    const pipeline = checked(this.device, () => this.device.createComputePipelineAsync({
      layout: 'auto', compute: { module: this.device.createShaderModule({ code }), entryPoint, constants },
    })).catch((error: unknown) => {
      this.pipelines.delete(operation);
      throw error;
    });
    this.pipelines.set(operation, pipeline);
    return pipeline;
  }

  private ensureBuffers(leftBytes: number, rightBytes: number, outputBytes: number): Buffers {
    if (this.buffers && this.buffers.left.size >= leftBytes && (this.buffers.right?.size ?? 0) >= rightBytes
        && this.buffers.output.size >= outputBytes) return this.buffers;
    this.releaseBuffers();
    const capacity = (bytes: number, max: number) => Math.min(2 ** Math.ceil(Math.log2(bytes)), Math.floor(max / 4) * 4);
    const outputCapacity = capacity(outputBytes, this.maxOutputBytes);
    const inputUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const allocated: GPUBuffer[] = [];
    const allocate = (size: number, usage: GPUBufferUsageFlags) => {
      const buffer = this.device.createBuffer({ size, usage });
      allocated.push(buffer);
      return buffer;
    };
    try {
      this.buffers = {
        left: allocate(capacity(leftBytes, this.maxStorageBytes), inputUsage),
        right: rightBytes ? allocate(capacity(rightBytes, this.maxStorageBytes), inputUsage) : undefined,
        output: allocate(outputCapacity, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
        readback: allocate(this.timer ? timestampOffset(outputCapacity) + TIMESTAMP_BYTES : outputCapacity,
          GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      };
      return this.buffers;
    } catch (error) {
      for (const buffer of allocated) buffer.destroy();
      throw error;
    }
  }

  private releaseBuffers() {
    this.matrixRight = undefined;
    if (!this.buffers) return;
    const { left, right, output, readback } = this.buffers;
    this.buffers = undefined;
    for (const buffer of [left, right, output, readback]) buffer?.destroy();
  }
}
