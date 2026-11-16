import { CpuPool } from './cpu.js';
import { awaitAbortable } from './abort.js';
import { GpuBackend } from './gpu.js';
import { buildGpuGraph } from './gpu-graph.js';
import { MIN_ADAPTIVE_ELEMENTS, runAdaptive, shouldInitializeGpu } from './adaptive.js';
import type { OptionalGpu } from './adaptive.js';
import { CostModel } from './cost-model.js';
import { ChunkRecorder } from './chunk-recorder.js';
import { CustomOperations } from './custom.js';
import { createInlineBudget, InlineCpu, inlineTiming, MAX_INLINE_CHUNK_ELEMENTS } from './inline-cpu.js';
import { MatrixPlanner } from './matrix-planner.js';
import { cpuChunkSize, MAX_CHUNK_ELEMENTS, partition, validateCpuShare, validateStrategy } from './partition.js';
import type { ChunkResult, ChunkTask } from './task.js';
import type { Capabilities, ComputeProfile, ComputeRequest, ComputeResult, CustomComputeRequest,
  CustomComputeResult, CustomOperation, CustomOperationDefinition, EngineOptions, GpuGraphBuilder, GpuGraphResult,
  MatrixDispatchProfile } from './types.js';
import { combineReduction, outputSize, prepareWorkload, reductionIdentity } from './workload.js';
import type { Workload } from './workload.js';

export type * from './types.js';
export { GpuDeviceLostError } from './gpu.js';

const GPU_RETRY_DELAY_MS = 1000;

type GpuState = Awaited<ReturnType<typeof GpuBackend.open>>;
interface History { cpu: CostModel; gpu: CostModel; inline: CostModel; runs: number; matrix?: MatrixPlanner }
type GpuExecution = { backend: GpuBackend; limit: number };

function workerCount(option: EngineOptions['cpuWorkers']) {
  if (option === undefined || option === 'auto') {
    const cores = globalThis.navigator?.hardwareConcurrency || 2;
    return Math.min(8, Math.max(1, cores - 1));
  }
  if (!Number.isSafeInteger(option) || option < 1) throw new RangeError('cpuWorkers must be a positive integer.');
  return option;
}

export class HeteroJS {
  private readonly cpu: CpuPool;
  private readonly custom: CustomOperations;
  private readonly inline = new InlineCpu();
  private readonly shutdown = new AbortController();
  private gpu?: Promise<GpuState>;
  private gpuState?: GpuState;
  private retryGpuAt = Infinity;
  private lastProfile?: ComputeProfile;
  private readonly histories = new Map<string, History>();

  private constructor(
    workers: number,
    private readonly strategy: NonNullable<EngineOptions['strategy']>,
    private readonly cpuShare: number,
    private readonly gpuEnabled: boolean,
    private readonly inlineEnabled: boolean,
    private readonly gpuDevice?: EngineOptions['gpuDevice'],
  ) {
    this.cpu = new CpuPool(workers);
    this.custom = new CustomOperations(this.cpu, () => this.getGpu());
  }

  static async create(options: EngineOptions = {}): Promise<HeteroJS> {
    if (options.gpu !== undefined && options.gpu !== 'auto' && options.gpu !== 'disabled') {
      throw new TypeError(`Unsupported GPU option: ${options.gpu}`);
    }
    const gpuDevice = options.gpuDevice && {
      requiredFeatures: Array.from(options.gpuDevice.requiredFeatures ?? []),
      requiredLimits: { ...options.gpuDevice.requiredLimits },
    };
    return new HeteroJS(workerCount(options.cpuWorkers), validateStrategy(options.strategy ?? 'static'),
      validateCpuShare(options.cpuShare ?? 0.25), options.gpu !== 'disabled', options.inlineCpu ?? true, gpuDevice);
  }

  async getCapabilities(): Promise<Capabilities> {
    this.shutdown.signal.throwIfAborted();
    const gpu = await this.getGpu();
    this.shutdown.signal.throwIfAborted();
    return {
      cpuWorkers: this.cpu.size,
      gpu: 'backend' in gpu ? gpu.backend.capability : { available: false, reason: gpu.reason },
    };
  }

  getProfile(): ComputeProfile | undefined {
    return this.lastProfile;
  }

  registerOperation<Input, Output, ChunkInput = Input, ChunkOutput = Output>(name: string,
    definition: CustomOperationDefinition<Input, Output, ChunkInput, ChunkOutput>): CustomOperation<Input, Output> {
    this.shutdown.signal.throwIfAborted();
    return this.custom.register(name, definition);
  }

  async runGpuGraph(build: (graph: GpuGraphBuilder) => void, options: { signal?: AbortSignal } = {}): Promise<GpuGraphResult> {
    const runStarted = performance.now();
    const signal = options.signal ? AbortSignal.any([this.shutdown.signal, options.signal]) : this.shutdown.signal;
    signal.throwIfAborted();
    const graph = buildGpuGraph(build);
    signal.throwIfAborted();
    const work = this.getGpu().then((state) => {
      signal.throwIfAborted();
      if ('reason' in state) throw new Error(state.reason);
      return state.backend.runGraph(graph, runStarted, signal);
    });
    return awaitAbortable(work, signal);
  }

  compute<Input, Output>(request: CustomComputeRequest<Input, Output>): Promise<CustomComputeResult<Output>>;
  compute(request: ComputeRequest): Promise<ComputeResult>;
  async compute<Input, Output>(request: ComputeRequest | CustomComputeRequest<Input, Output>):
    Promise<ComputeResult | CustomComputeResult<Output>> {
    const runStarted = performance.now();
    const run = new AbortController();
    const signals = [run.signal, this.shutdown.signal];
    if (request.signal) signals.push(request.signal);
    const signal = AbortSignal.any(signals);
    signal.throwIfAborted();
    if (request.profile !== undefined && request.profile !== 'full' && request.profile !== 'summary') {
      throw new TypeError(`Unsupported profile mode: ${request.profile}`);
    }
    const summary = request.profile === 'summary';
    if (typeof request.operation === 'object') {
      const custom = request as CustomComputeRequest<Input, Output>;
      const result = await this.custom.run(custom.operation, custom.input, validateStrategy(request.strategy ?? this.strategy),
        validateCpuShare(request.cpuShare ?? this.cpuShare), summary, runStarted, run, signal);
      signal.throwIfAborted();
      this.lastProfile = result.profile;
      return result;
    }
    const workload = prepareWorkload(request as ComputeRequest);
    const { kernel, units: elements, workPerUnit, reduction } = workload;
    const requestedStrategy = validateStrategy(request.strategy ?? this.strategy);
    const share = validateCpuShare(request.cpuShare ?? this.cpuShare);
    const adaptive = requestedStrategy === 'adaptive';
    const cachedGpu = adaptive ? this.currentGpuState() : undefined;
    const history = adaptive ? this.historyFor(workload) : undefined;
    const matrix = kernel.operation === 'matrixMultiply';
    const useInline = adaptive && this.inlineEnabled && elements > 0 && !matrix;
    const explore = history && (this.gpuEnabled || useInline) && elements * workPerUnit >= (matrix ? 2 ** 23 : MIN_ADAPTIVE_ELEMENTS);
    const cpuLimit = Math.min(MAX_CHUNK_ELEMENTS, Math.max(1, Math.floor(2 ** 22 / workPerUnit)));
    const fixed = partition(elements, adaptive ? 'cpu-only' : requestedStrategy, share);
    let fallbackReason: string | undefined;
    let gpuExecution: GpuExecution | undefined;
    if (fixed.gpuElements > 0) {
      const gpu = this.availableGpu(await this.getGpu(), workload);
      signal.throwIfAborted();
      if ('backend' in gpu) gpuExecution = gpu;
      else {
        fallbackReason = gpu.reason;
        if (requestedStrategy === 'gpu-only') throw new Error(fallbackReason);
        fixed.cpuElements = elements;
      }
    } else if (adaptive && !this.gpuEnabled && elements > 0) {
      fallbackReason = 'WebGPU is disabled for this engine.';
    }
    const output = new Float32Array(workload.outputLength);
    const partials = reduction ? new Map<number, { length: number; value: number }>() : undefined;
    let nextPartial = 0;
    let reduced = reduction ? reductionIdentity(reduction) : 0;
    const recorder = new ChunkRecorder(summary);
    const setupMs = performance.now() - runStarted;
    let schedulingMs = 0;
    let aggregationMs = 0;

    const executeChunk = async (start: number, length: number, dispatch: (task: ChunkTask) => Promise<ChunkResult>,
      onGpuSubmitted?: () => void) => {
      const scheduledAt = performance.now();
      signal.throwIfAborted();
      const task: ChunkTask = { workload, destination: output, start, length, runStarted, signal, onGpuSubmitted };
      schedulingMs += performance.now() - scheduledAt;
      const result = await dispatch(task);
      signal.throwIfAborted();
      const mergingAt = performance.now();
      if (result.output) output.set(result.output, outputSize(kernel, start));
      if (result.partial !== undefined && reduction && partials) {
        partials.set(start, { length, value: result.partial });
        for (let part = partials.get(nextPartial); part; part = partials.get(nextPartial)) {
          reduced = combineReduction(reduction, reduced, part.value);
          partials.delete(nextPartial);
          nextPartial += part.length;
        }
      }
      recorder.add(result.profile);
      aggregationMs += performance.now() - mergingAt;
      return result.profile;
    };

    const execute = (start: number, end: number, size: number, lanes: number,
      dispatch: (task: ChunkTask) => Promise<ChunkResult>, onGpuSubmitted?: () => void) => {
      let next = start;
      const loop = async () => {
        while (next < end) {
          const offset = next;
          const length = Math.min(size, end - next);
          next += length;
          await executeChunk(offset, length, dispatch, onGpuSubmitted);
        }
      };
      return Promise.all(Array.from({ length: Math.min(lanes, Math.ceil((end - start) / size)) }, loop));
    };

    const executeSplit = async (cpuRows: number, workers: number, gpu?: GpuExecution, gpuFirst = false) => {
      const runCpu = () => execute(0, cpuRows, Math.min(cpuLimit, cpuChunkSize(cpuRows, Math.max(1, workers))),
        workers, (task) => this.cpu.run(task));
      if (!gpu || cpuRows === elements) {
        await runCpu();
        return;
      }
      const runGpu = (submitted?: () => void) => execute(cpuRows, elements, gpu.limit, 1,
        (task) => gpu.backend.run(task), submitted);
      if (!gpuFirst || cpuRows === 0) {
        await Promise.all([runCpu(), runGpu()]);
        return;
      }
      let onSubmitted!: () => void;
      const submitted = new Promise<void>((resolve) => { onSubmitted = resolve; });
      const gpuWork = runGpu(onSubmitted);
      await Promise.all([gpuWork, Promise.race([submitted, gpuWork]).then(runCpu)]);
    };

    const inlineBudget = useInline ? createInlineBudget() : undefined;
    const runInline = inlineBudget && ((task: ChunkTask) => this.inline.run(task, inlineBudget));
    let matrixDispatch: MatrixDispatchProfile | undefined;
    let canLearnMatrix = false;
    try {
      if (history?.matrix && explore) {
        const state = cachedGpu;
        if (!state) void this.getGpu();
        const gpu = state && this.availableGpu(state, workload);
        if (gpu && 'reason' in gpu) fallbackReason = gpu.reason;
        const available = gpu && 'backend' in gpu ? gpu : undefined;
        canLearnMatrix = this.cpu.pendingChunks === 0 && (available?.backend.pendingChunks ?? 0) === 0;
        matrixDispatch = history.matrix.choose(Boolean(available), canLearnMatrix);
        await executeSplit(matrixDispatch.cpuRows, matrixDispatch.cpuWorkers, available, true);
      } else if (history && explore) {
        const bindGpu = (result: GpuState): OptionalGpu => {
          const available = this.availableGpu(result, workload);
          if ('reason' in available) return available;
          const { backend, limit } = available;
          return { device: {
            kind: 'gpu' as const, lanes: 1, maxChunkElements: limit,
            pendingChunks: () => backend.pendingChunks,
            dispatch: (start: number, length: number) => executeChunk(start, length, (task) => backend.run(task)),
          } };
        };
        const tryGpu = this.gpuEnabled && (cachedGpu || shouldInitializeGpu(elements, runInline ? history.inline : undefined));
        const gpu = tryGpu ? (cachedGpu ? bindGpu(cachedGpu) : this.getGpu().then(bindGpu)) : undefined;
        const result = await runAdaptive({
          elements, gpu, signal, models: history, refresh: ++history.runs % 8 === 0,
          cpu: {
            kind: 'cpu', lanes: this.cpu.size, maxChunkElements: cpuLimit,
            pendingChunks: () => this.cpu.pendingChunks,
            dispatch: (start, length) => executeChunk(start, length, (task) => this.cpu.run(task)),
          },
          inline: runInline ? { model: history.inline, device: {
            kind: 'inline', lanes: 1, maxChunkElements: Math.max(1, Math.floor(MAX_INLINE_CHUNK_ELEMENTS / workPerUnit)),
            pendingChunks: () => this.inline.pendingChunks,
            dispatch: (start, length) => executeChunk(start, length, runInline),
          } } : undefined,
        });
        fallbackReason ??= result.fallbackReason;
        schedulingMs += result.schedulingMs;
      } else if (runInline) {
        await executeChunk(0, elements, runInline);
      } else {
        await executeSplit(fixed.cpuElements, this.cpu.size, gpuExecution);
      }
    } catch (error) {
      run.abort(error);
      throw signal.reason;
    }
    signal.throwIfAborted();
    const chunks = recorder.ordered();
    if (reduction) output[0] = reduced;
    const cpuElements = recorder.cpuElements;
    const gpuElements = elements - cpuElements;
    const strategy = elements === 0 ? requestedStrategy : gpuElements === 0 ? 'cpu-only'
      : cpuElements === 0 ? 'gpu-only' : requestedStrategy;
    const profile: ComputeProfile = {
      operation: kernel.operation, requestedStrategy, strategy, elements, cpuElements, gpuElements, fallbackReason,
      setupMs, schedulingMs, aggregationMs, chunks, chunkSummary: recorder.summary, totalMs: performance.now() - runStarted,
    };
    if (inlineBudget && recorder.usedInline) {
      profile.inlineTiming = inlineTiming(inlineBudget);
    }
    if (history && !matrix) {
      const { cpu, gpu } = history;
      const inline = history.inline.estimate;
      profile.estimates = {
        ...(cpu.estimate && { cpu: { ...cpu.estimate } }),
        ...(gpu.estimate && { gpu: { ...gpu.estimate } }),
        ...(inline && { inline: { ...inline } }),
      };
    }
    if (matrixDispatch) {
      profile.matrixDispatch = matrixDispatch;
      if (canLearnMatrix) history!.matrix!.observe(matrixDispatch, profile.totalMs, recorder.startedBackend);
    }
    this.lastProfile = profile;
    return { output, profile };
  }

  close(): void {
    if (this.shutdown.signal.aborted) return;
    this.shutdown.abort(new DOMException('The HeteroJS engine is closed.', 'AbortError'));
    this.custom.clear();
    this.cpu.close(this.shutdown.signal.reason);
    void this.gpu?.then((gpu) => {
      if ('backend' in gpu) gpu.backend.close();
    });
  }

  private getGpu(): Promise<GpuState> {
    if (!this.gpuEnabled) return Promise.resolve({ reason: 'WebGPU is disabled for this engine.' });
    this.currentGpuState();
    return this.gpu ??= GpuBackend.open(this.gpuDevice).then((state) => {
      this.retryGpuAt = 'backend' in state ? Infinity : performance.now() + GPU_RETRY_DELAY_MS;
      return this.gpuState = state;
    });
  }

  private currentGpuState() {
    const state = this.gpuState;
    const lost = state && 'backend' in state && state.backend.lost;
    if (lost) {
      this.histories.clear();
      this.custom.resetGpuHistory();
    }
    if (lost || (state && 'reason' in state && performance.now() >= this.retryGpuAt)) {
      this.gpu = undefined;
      this.gpuState = undefined;
    }
    return this.gpuState;
  }

  private availableGpu(gpu: GpuState, workload: Workload): GpuExecution | { reason: string } {
    if ('reason' in gpu) return gpu;
    const capability = gpu.backend.capability;
    if (!capability.available) return { reason: capability.reason };
    const limit = gpu.backend.chunkLimit(workload);
    return limit ? { backend: gpu.backend, limit } : { reason: 'This matrix shape exceeds the WebGPU device limits.' };
  }

  private historyFor(workload: Workload) {
    const { kernel } = workload;
    const key = kernel.operation === 'matrixMultiply'
      ? `${kernel.operation}:${kernel.shape.rows}:${kernel.shape.inner}:${kernel.shape.columns}` : kernel.operation;
    const previous = this.histories.get(key);
    if (previous) this.histories.delete(key);
    else if (this.histories.size >= 32) this.histories.delete(this.histories.keys().next().value!);
    const history: History = previous ?? { cpu: new CostModel(), gpu: new CostModel(), inline: new CostModel(), runs: 0,
      matrix: kernel.operation === 'matrixMultiply' ? new MatrixPlanner(kernel.shape.rows, this.cpu.size) : undefined };
    this.histories.set(key, history);
    return history;
  }
}
