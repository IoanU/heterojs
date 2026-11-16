import type { CostModel } from './cost-model.js';
import type { AdaptiveEstimate, Backend, ChunkProfile } from './types.js';

export const MIN_ADAPTIVE_ELEMENTS = 2 ** 19;
const MIN_GPU_WITH_INLINE_ELEMENTS = 2 ** 23;
const GPU_PROBE_BUDGET_MS = 9;
const WORKER_PROBE_BUDGET_MS = 16;
const PROBE_ELEMENTS = 2 ** 16;
const INLINE_PROBE_ELEMENTS = 2 ** 18;
const TARGET_MS = 3;

export interface Device {
  kind: Backend | 'inline';
  lanes: number;
  maxChunkElements: number;
  pendingChunks: () => number;
  dispatch: (start: number, length: number) => Promise<ChunkProfile>;
}

interface ScheduleState {
  lanes: number;
  chunkSize: number;
  readyMs: number;
  estimate: AdaptiveEstimate;
}

export function shouldInitializeGpu(elements: number, inline?: CostModel) {
  if (!inline) return true;
  const estimate = inline.estimate;
  return elements >= MIN_GPU_WITH_INLINE_ELEMENTS || Boolean(estimate && estimate.samples >= 2
    && estimate.fixedMs + elements * estimate.msPerElement >= GPU_PROBE_BUDGET_MS);
}

function finish(state: ScheduleState, elements: number, work?: number) {
  if (elements === 0) return 0;
  const perLane = Math.ceil(elements / state.lanes);
  const waves = Math.ceil(perLane / state.chunkSize);
  return state.readyMs + (work === undefined ? perLane : work / state.lanes) * state.estimate.msPerElement
    + waves * state.estimate.fixedMs;
}

export function planCpuElements(elements: number, cpu: ScheduleState, gpu: ScheduleState,
  work?: { total: number; prefix: (length: number) => number }) {
  const cpuOnly = finish(cpu, elements, work?.total);
  const gpuOnly = finish(gpu, elements, work?.total);
  let best = Math.min(cpuOnly, gpuOnly);
  let amount = cpuOnly <= gpuOnly ? elements : 0;
  const threshold = Math.max(0.1, best * 0.05);

  // The CPU curve increases with its allocation and the GPU curve decreases.
  let low = 1;
  let high = elements - 1;
  while (low <= high) {
    const split = low + Math.floor((high - low) / 2);
    const prefixWork = work?.prefix(split);
    if (work && prefixWork! >= work.total) throw new RangeError('Estimated work must increase with the range length.');
    const cpuTime = finish(cpu, split, prefixWork);
    const gpuTime = finish(gpu, elements - split, work && work.total - prefixWork!);
    const time = Math.max(cpuTime, gpuTime);
    if (time < best - threshold || (amount !== 0 && amount !== elements && time < best)) {
      best = time;
      amount = split;
    }
    if (cpuTime < gpuTime) low = split + 1;
    else high = split - 1;
  }
  return amount;
}

function state(device: Device, model: CostModel, probeElements: number, workPerElement = 1): ScheduleState {
  const estimate = model.estimate ?? { samples: 0, fixedMs: 0.2, msPerElement: 0.8 / probeElements / workPerElement };
  const slope = estimate.msPerElement * workPerElement;
  const minimum = device.kind === 'inline' ? INLINE_PROBE_ELEMENTS
    : device.kind === 'gpu' ? probeElements : Math.max(1, Math.floor(probeElements / 4));
  const chunkSize = Math.min(device.maxChunkElements, Math.max(minimum,
    Math.floor(Math.max(0, TARGET_MS - estimate.fixedMs) / slope)));
  const pending = device.pendingChunks();
  const queuedWaves = Math.floor(pending / device.lanes);
  const readyMs = queuedWaves * (estimate.fixedMs + chunkSize * slope);
  return { lanes: device.lanes, estimate, chunkSize, readyMs };
}

function fitWork(max: number, minimum: number, estimate: AdaptiveEstimate, work: (length: number) => number) {
  if (!estimate.samples) return max;
  const budget = Math.max(TARGET_MS - estimate.fixedMs, estimate.fixedMs * 0.5, 0.2) / estimate.msPerElement;
  if (max <= minimum || work(max) <= budget) return max;
  let low = minimum;
  let high = max;
  while (low < high) {
    const middle = low + Math.ceil((high - low) / 2);
    if (work(middle) <= budget) low = middle;
    else high = middle - 1;
  }
  return low;
}

export type OptionalGpu = { device: Device } | { reason: string };

export async function runAdaptive(options: {
  elements: number;
  cpu: Device;
  gpu?: OptionalGpu | Promise<OptionalGpu>;
  inline?: { device: Device; model: CostModel };
  models: Record<Backend, CostModel>;
  refresh: boolean;
  signal: AbortSignal;
  probeElements?: number;
  estimateWork?: (start: number, length: number) => number;
}) {
  const { elements, cpu, inline, models, refresh, signal, probeElements = PROBE_ELEMENTS, estimateWork } = options;
  let next = 0;
  let schedulingMs = 0;
  let gpu: Device | undefined;
  let fallbackReason: string | undefined;
  const needsProbe = {
    cpu: refresh || !models.cpu.estimate || Boolean(inline && models.cpu.estimate.samples === 1),
    gpu: refresh || !models.gpu.estimate, inline: !inline?.model.estimate,
  };
  let wake: (() => void) | undefined;
  let active = 0;
  let failure: { reason: unknown } | undefined;
  const notify = () => {
    const resume = wake;
    wake = undefined;
    resume?.();
  };
  const useGpu = (value: OptionalGpu) => {
    if ('device' in value) gpu = value.device;
    else fallbackReason = value.reason;
  };
  let waitingForGpu = options.gpu instanceof Promise;
  if (options.gpu instanceof Promise) {
    void options.gpu.then(useGpu, (reason: unknown) => {
      fallbackReason = reason instanceof Error ? reason.message : String(reason);
    }).finally(() => {
      waitingForGpu = false;
      notify();
    });
  } else if (options.gpu) useGpu(options.gpu);

  const launch = async (device: Device, length: number, exploratory: boolean) => {
    const start = next;
    next += length;
    active++;
    try {
      const work = estimateWork ? estimateWork(start, length) : length;
      const profile = await device.dispatch(start, length);
      if (exploratory) profile.exploratory = true;
      const model = device.kind === 'inline' ? inline?.model : models[device.kind];
      // Worker startup still counts toward end-to-end latency; it cannot predict a warm worker.
      if ((!inline && !estimateWork) || !profile.workerStartup) model?.observe(work, profile.elapsedMs);
    } catch (reason) {
      failure ??= { reason };
    } finally {
      active--;
      notify();
    }
  };

  signal.addEventListener('abort', notify, { once: true });
  try {
    while (true) {
      signal.throwIfAborted();
      if (failure) throw failure.reason;
      if (next === elements && active === 0) break;
      while (next < elements && !failure) {
        const started = performance.now();
        const remaining = elements - next;
        let device: Device | undefined;
        let selectedState: ScheduleState | undefined;
        let length = 0;
        let explore = false;
        const prefix = estimateWork && ((length: number) => estimateWork(next, length));
        const work = prefix && { prefix, total: prefix(remaining) };
        const workPerElement = work ? work.total / remaining : 1;
        const workerState = state(cpu, models.cpu, probeElements, workPerElement);
        const inlineState = inline && state(inline.device, inline.model, probeElements);
        const directMs = inlineState ? finish(inlineState, remaining) : Infinity;
        if (inline && needsProbe.inline && inline.device.pendingChunks() === 0) {
          device = inline.device;
          length = Math.min(remaining, INLINE_PROBE_ELEMENTS);
          explore = true;
        } else if (gpu && needsProbe.gpu && gpu.pendingChunks() === 0 && remaining >= probeElements
          && (!inline || (inline.model.estimate && directMs >= GPU_PROBE_BUDGET_MS))) {
          device = gpu;
          if (prefix) selectedState = state(gpu, models.gpu, probeElements, workPerElement);
          length = Math.min(gpu.maxChunkElements, probeElements);
          explore = true;
        } else if (needsProbe.cpu && cpu.pendingChunks() < cpu.lanes && remaining >= probeElements
          && (!inline || (inline.model.estimate && directMs >= WORKER_PROBE_BUDGET_MS))) {
          device = cpu;
          selectedState = workerState;
          length = inline ? Math.min(cpu.maxChunkElements, Math.ceil(remaining / Math.max(2, cpu.lanes)))
            : Math.min(cpu.maxChunkElements, probeElements);
          explore = true;
        } else {
          const useInline = inline && inlineState && (!models.cpu.estimate || directMs <= finish(workerState, remaining) * 1.1);
          const cpuDevice = useInline ? inline.device : cpu;
          const cpuState = useInline ? inlineState : workerState;
          const gpuState = gpu && (!inline || models.gpu.estimate) ? state(gpu, models.gpu, probeElements, workPerElement) : undefined;
          const gpuFirst = work && gpuState && gpu?.pendingChunks() === 0;
          let onCpu = gpuState ? (gpuFirst ? remaining - planCpuElements(remaining, gpuState, cpuState, work)
            : planCpuElements(remaining, cpuState, gpuState, work)) : remaining;
          if (inline && onCpu > 0 && remaining - onCpu < PROBE_ELEMENTS) onCpu = remaining;
          const onGpu = remaining - onCpu;
          const idleCpu = Math.max(0, cpuDevice.lanes - cpuDevice.pendingChunks());
          if (onGpu > 0 && gpu && gpuState && (gpu.pendingChunks() === 0 || (onCpu === 0 && active === 0))) {
            device = gpu;
            selectedState = gpuState;
            length = Math.min(onGpu, gpuState.chunkSize);
          } else if (onCpu > 0 && (idleCpu > 0 || active === 0)) {
            device = cpuDevice;
            selectedState = cpuState;
            // Keep CPU waves short while the GPU is initializing in parallel.
            const max = waitingForGpu && !useInline ? Math.min(cpuState.chunkSize, 2 * PROBE_ELEMENTS) : cpuState.chunkSize;
            length = Math.min(max, Math.ceil(onCpu / Math.max(1, idleCpu)));
          }
        }
        if (prefix && selectedState && length) {
          length = fitWork(length, Math.max(1, Math.floor(probeElements / 32)), selectedState.estimate, prefix);
        }
        schedulingMs += performance.now() - started;
        if (!device) break;
        if (explore) needsProbe[device.kind] = false;
        void launch(device, length, explore);
      }
      if (failure || signal.aborted) continue;
      await new Promise<void>((resolve) => { wake = resolve; });
    }
  } finally {
    signal.removeEventListener('abort', notify);
  }
  return { schedulingMs, fallbackReason };
}
