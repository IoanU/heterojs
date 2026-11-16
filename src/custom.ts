import { runAdaptive } from './adaptive.js';
import { awaitAbortable } from './abort.js';
import { ChunkRecorder } from './chunk-recorder.js';
import { CostModel } from './cost-model.js';
import type { CpuPool } from './cpu.js';
import { CustomPlanner } from './custom-planner.js';
import type { GpuBackend } from './gpu.js';
import { cpuChunkSize, MAX_CHUNK_ELEMENTS, partition } from './partition.js';
import type { CustomChunkTask } from './task.js';
import type { Backend, ComputeProfile, CustomCpuModule, CustomOperation, CustomOperationDefinition,
  CustomPartial, CustomPlan, CustomRange, Strategy, StreamingCustomPlan } from './types.js';

type Definition = CustomOperationDefinition<unknown, unknown>;
type GpuState = Awaited<ReturnType<typeof GpuBackend.open>>;
type Models = Record<Backend, CostModel>;

interface WeightedPlan {
  planner: CustomPlanner;
  models?: Models;
  runs: number;
}

interface RegisteredOperation {
  name: string;
  definition: Definition;
  cpu?: Pick<CustomChunkTask, 'moduleUrl' | 'exportName' | 'transferOutput' | 'transferInput'>;
  models: Models;
  weightedModels?: Models;
  planners: Map<string, WeightedPlan>;
  runs: number;
}

function newModels(): Models { return { cpu: new CostModel(), gpu: new CostModel() }; }

function validUnits(value: number, minimum: number, field: string) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${field} must be a ${minimum === 0 ? 'nonnegative' : 'positive'} safe integer.`);
  }
  return value;
}

function chooseWholeBackend(operation: RegisteredOperation, units: number) {
  const { models, runs } = operation;
  const cpu = models.cpu.estimate;
  const gpu = models.gpu.estimate;
  if (!cpu || runs % 16 === 0) return { backend: 'cpu' as const, exploratory: true };
  if (!gpu || runs % 8 === 0) return { backend: 'gpu' as const, exploratory: true };
  const cpuMs = cpu.fixedMs + cpu.msPerElement * units;
  const gpuMs = gpu.fixedMs + gpu.msPerElement * units;
  return { backend: cpuMs <= gpuMs ? 'cpu' as const : 'gpu' as const, exploratory: false };
}

export class CustomOperations {
  private readonly operations = new Map<object, RegisteredOperation>();
  private readonly names = new Set<string>();

  constructor(private readonly cpuPool: CpuPool, private readonly getGpu: () => Promise<GpuState>) {}

  register<Input, Output, ChunkInput, ChunkOutput>(name: string,
    definition: CustomOperationDefinition<Input, Output, ChunkInput, ChunkOutput>): CustomOperation<Input, Output> {
    if (typeof name !== 'string' || !name.trim()) throw new TypeError('A custom operation requires a nonempty name.');
    if (this.names.has(name)) throw new Error(`Custom operation "${name}" is already registered.`);
    if (!definition || (!definition.cpu && !definition.gpu)) {
      throw new TypeError('A custom operation requires a CPU module, a GPU function, or both.');
    }
    for (const [field, value] of Object.entries({ gpu: definition.gpu, plan: definition.plan,
      estimateUnits: definition.estimateUnits })) {
      if (value !== undefined && typeof value !== 'function') throw new TypeError(`${field} must be a function.`);
    }
    if (definition.plan && definition.estimateUnits) throw new TypeError('A partitioned operation supplies its units through plan.');
    let cpu: RegisteredOperation['cpu'];
    if (definition.cpu) {
      const { module, exportName = 'run', transferOutput = false, transferInput } = definition.cpu;
      if ((typeof module !== 'string' || !module.trim()) && !(module instanceof URL)) {
        throw new TypeError('A CPU implementation requires a JavaScript module URL.');
      }
      if (typeof exportName !== 'string' || !exportName.trim()) throw new TypeError('The CPU export name must be nonempty.');
      if (typeof transferOutput !== 'boolean') throw new TypeError('transferOutput must be a boolean.');
      if (transferInput !== undefined && typeof transferInput !== 'function') throw new TypeError('transferInput must be a function.');
      cpu = { moduleUrl: new URL(module, globalThis.location?.href).href, exportName, transferOutput,
        transferInput: transferInput as CustomCpuModule['transferInput'] };
    }
    const handle = Object.freeze({ name }) as CustomOperation<Input, Output>;
    // Keep the public generics on the handle; registry storage is heterogeneous.
    this.operations.set(handle, { name, definition: { ...definition } as Definition, cpu,
      models: newModels(), planners: new Map(), runs: 0 });
    this.names.add(name);
    return handle;
  }

  clear() {
    this.operations.clear();
    this.names.clear();
  }

  resetGpuHistory() {
    for (const operation of this.operations.values()) {
      operation.planners.clear();
      operation.models = { cpu: operation.models.cpu, gpu: new CostModel() };
      operation.weightedModels = undefined;
    }
  }

  async run<Input, Output>(handle: CustomOperation<Input, Output>, input: Input, requestedStrategy: Strategy,
    cpuShare: number, summary: boolean, runStarted: number, run: AbortController, signal: AbortSignal) {
    const operation = this.operations.get(handle);
    if (!operation) throw new TypeError('The custom operation must be registered on this HeteroJS engine.');
    const work = this.execute(operation, input, requestedStrategy, cpuShare, summary, runStarted, signal).catch((error: unknown) => {
      run.abort(error);
      throw signal.reason;
    });
    const result = await awaitAbortable(work, signal);
    return { output: result.output as Output, profile: result.profile };
  }

  private async execute(operation: RegisteredOperation, input: unknown, requestedStrategy: Strategy,
    cpuShare: number, summary: boolean, runStarted: number, signal: AbortSignal) {
    const { definition, cpu, name } = operation;
    const { gpu, estimateUnits } = definition;
    if (requestedStrategy === 'cpu-only' && !cpu) throw new Error(`Custom operation "${name}" has no CPU implementation.`);
    if (requestedStrategy === 'gpu-only' && !gpu) throw new Error(`Custom operation "${name}" has no GPU implementation.`);

    let plan: CustomPlan<unknown, unknown, unknown> | StreamingCustomPlan<unknown, unknown, unknown> | undefined;
    if (definition.plan) {
      plan = definition.plan(input);
      if (!plan || typeof plan.chunk !== 'function') throw new TypeError('plan must return a chunk function.');
      const { collector, combine, adaptiveKey } = plan;
      if (collector ? combine !== undefined || typeof collector.add !== 'function' || typeof collector.finish !== 'function'
        : typeof combine !== 'function') {
        throw new TypeError('plan must return a combine function or a collector with add and finish functions.');
      }
      if (adaptiveKey !== undefined && (typeof adaptiveKey !== 'string' || !adaptiveKey.length)) {
        throw new TypeError('plan.adaptiveKey must be a nonempty string.');
      }
      if (plan.estimateWork !== undefined && typeof plan.estimateWork !== 'function') {
        throw new TypeError('plan.estimateWork must be a function.');
      }
    }
    const collector = plan?.collector;
    const elements = plan ? validUnits(plan.units, 0, 'plan.units')
      : validUnits(estimateUnits ? estimateUnits(input) : 1, 1, 'estimateUnits');
    const limit = validUnits(plan?.maxChunkUnits ?? MAX_CHUNK_ELEMENTS, 1, 'plan.maxChunkUnits');
    const parts: CustomPartial<unknown>[] = [];
    const recorder = new ChunkRecorder(summary);
    let fallbackReason: string | undefined;
    let backend: GpuBackend | undefined;
    let schedulingMs = 0;
    let collectionMs = 0;
    let collectorTail: Promise<void> | undefined;
    const collecting = { cpu: 0, gpu: 0 };

    const collect = (part: CustomPartial<unknown>) => {
      const add = () => {
        signal.throwIfAborted();
        const started = performance.now();
        if (!collector) parts.push(part);
        else {
          const pending = collector.add(part, signal);
          if (pending) return awaitAbortable(Promise.resolve(pending), signal).then(() => { collectionMs += performance.now() - started; });
        }
        collectionMs += performance.now() - started;
      };
      const waiting = collectorTail ? collectorTail.then(add) : add();
      if (waiting) {
        collectorTail = waiting;
        void waiting.then(() => { if (collectorTail === waiting) collectorTail = undefined; }, () => undefined);
      }
      return waiting;
    };

    const adaptive = requestedStrategy === 'adaptive';
    const cpuUnits = requestedStrategy === 'gpu-only' || !cpu ? 0 : requestedStrategy === 'cpu-only' || adaptive
      ? elements : plan ? partition(elements, 'static', cpuShare).cpuElements : cpuShare >= 0.5 ? elements : 0;
    if (elements && requestedStrategy !== 'cpu-only' && (cpuUnits < elements || adaptive)) {
      if (gpu) {
        const state = await this.getGpu();
        signal.throwIfAborted();
        if ('reason' in state) fallbackReason = state.reason;
        else {
          const capability = state.backend.capability;
          if (capability.available) backend = state.backend;
          else fallbackReason = capability.reason;
        }
      } else fallbackReason = `Custom operation "${name}" has no GPU implementation.`;
      if (!backend && (!cpu || requestedStrategy === 'gpu-only')) throw new Error(fallbackReason);
    }
    const setupMs = performance.now() - runStarted;
    const staticSplit = partition(elements, 'static', cpuShare).cpuElements;
    const hint = plan?.estimateWork;
    const estimateWork = hint && ((start: number, length: number) =>
      validUnits(hint(Object.freeze({ start, length, totalUnits: elements })), 1, 'Estimated work'));
    const planning = adaptive && cpu && backend && plan?.adaptiveKey && elements > 1
      ? this.plannerFor(operation, plan.adaptiveKey, elements, limit, staticSplit, Boolean(hint)) : undefined;
    const planner = planning?.planner;
    const models = estimateWork ? (planning?.models ?? (operation.weightedModels ??= newModels())) : operation.models;
    const canLearn = Boolean(planner && this.cpuPool.pendingChunks === 0 && backend?.pendingChunks === 0);
    const customDispatch = planner?.choose(canLearn);

    const dispatch = async (kind: Backend, start: number, length: number) => {
      const scheduled = performance.now();
      signal.throwIfAborted();
      const range: CustomRange = Object.freeze({ start, length, totalUnits: elements });
      const payload = plan ? plan.chunk(range) : input;
      schedulingMs += performance.now() - scheduled;
      const result = kind === 'cpu' ? await this.cpuPool.runCustom({ ...cpu!, ...range, input: payload, runStarted, signal })
        : await backend!.runCustom(gpu!, payload, range, runStarted, signal);
      signal.throwIfAborted();
      const waiting = collect({ ...range, value: result.value });
      if (waiting) {
        collecting[kind]++;
        try { await waiting; } finally { collecting[kind]--; }
      }
      recorder.add(result.profile);
      return result.profile;
    };

    const dispatchRange = (kind: Backend, start: number, end: number, lanes: number, size: number) => {
      let next = start;
      const loop = async () => {
        while (next < end) {
          const offset = next;
          const length = Math.min(size, end - offset);
          next += length;
          await dispatch(kind, offset, length);
        }
      };
      return Promise.all(Array.from({ length: Math.min(lanes, Math.ceil((end - start) / size)) }, loop));
    };

    if (elements > 0) {
      if (!plan || elements === 1) {
        if (!plan && adaptive && cpu && backend) {
          operation.runs++;
          const { backend: choice, exploratory } = chooseWholeBackend(operation, elements);
          const chunk = await dispatch(choice, 0, elements);
          if (exploratory) chunk.exploratory = true;
          models[choice].observe(elements, chunk.elapsedMs + chunk.queueMs);
        } else {
          await dispatch(backend && (cpuUnits === 0 || !cpu) ? 'gpu' : 'cpu', 0, elements);
        }
      } else if (adaptive && cpu && (backend || estimateWork) && (!customDispatch || customDispatch.mode === 'adaptive-chunks')) {
        const result = await runAdaptive({
          elements, models, refresh: ++(planning ?? operation).runs % 8 === 0, signal, estimateWork,
          probeElements: Math.min(limit, 2 ** 16, Math.max(1, Math.floor(elements / 4))),
          cpu: { kind: 'cpu', lanes: this.cpuPool.size, maxChunkElements: limit,
            pendingChunks: () => this.cpuPool.pendingChunks + collecting.cpu,
            dispatch: (start, length) => dispatch('cpu', start, length) },
          gpu: backend && { device: { kind: 'gpu', lanes: 1, maxChunkElements: limit,
            pendingChunks: () => backend.pendingChunks + collecting.gpu,
            dispatch: (start, length) => dispatch('gpu', start, length) } },
        });
        schedulingMs += result.schedulingMs;
      } else {
        const split = customDispatch?.cpuUnits ?? (backend ? cpuUnits : elements);
        await Promise.all([
          dispatchRange('cpu', 0, split, this.cpuPool.size, Math.min(limit, cpuChunkSize(split, this.cpuPool.size))),
          dispatchRange('gpu', split, elements, 1, limit),
        ]);
      }
    }
    signal.throwIfAborted();
    const aggregationStarted = performance.now();
    parts.sort((a, b) => a.start - b.start);
    const chunks = recorder.ordered();
    const output = collector ? await collector.finish(signal) : plan?.combine ? await plan.combine(parts, signal) : parts[0].value;
    signal.throwIfAborted();
    const cpuElements = recorder.cpuElements;
    const gpuElements = elements - cpuElements;
    const profile: ComputeProfile = {
      customOperation: name, customDispatch: customDispatch && { ...customDispatch, cpuUnits: cpuElements },
      requestedStrategy, strategy: !elements ? requestedStrategy
        : !gpuElements ? 'cpu-only' : !cpuElements ? 'gpu-only' : requestedStrategy,
      elements, cpuElements, gpuElements, fallbackReason, setupMs, schedulingMs, chunks, chunkSummary: recorder.summary,
      aggregationMs: collectionMs + performance.now() - aggregationStarted, totalMs: performance.now() - runStarted,
    };
    if (planner && customDispatch && canLearn) planner.observe(customDispatch, profile.totalMs - setupMs, recorder.startedBackend);
    if (adaptive) profile.estimates = {
      ...(models.cpu.estimate && { cpu: { ...models.cpu.estimate } }),
      ...(models.gpu.estimate && { gpu: { ...models.gpu.estimate } }),
    };
    if (adaptive && estimateWork) profile.estimateBasis = 'work';
    return { output, profile };
  }

  private plannerFor(operation: RegisteredOperation, workload: string, units: number, limit: number, split: number, weighted: boolean) {
    const key = JSON.stringify([workload, units, limit, split, weighted]);
    const { planners } = operation;
    const previous = planners.get(key);
    if (previous) planners.delete(key);
    else if (planners.size === 32) planners.delete(planners.keys().next().value!);
    const planner = previous ?? { planner: new CustomPlanner(units, split, weighted), models: weighted ? newModels() : undefined, runs: 0 };
    planners.set(key, planner);
    return planner;
  }
}
