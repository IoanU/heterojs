import { HeteroJS } from '../src/index.js';
import type { Capabilities, ComputeProfile, ComputeResult, Operation, Strategy } from '../src/index.js';
import { DirectCpuReference } from './cpu-direct.js';
import { startEventLoopProbe } from './event-loop.js';
import { inputsFor, workloadLabels } from './workloads.js';

export interface Settings {
  operation?: Operation;
  sizes: number[];
  samples: number;
  warmups: number;
  cpuShare: number;
  comparison?: 'all' | 'cpu';
  includeDirect?: boolean;
}

export interface Sample {
  profile: ComputeProfile;
  mismatches: number;
  eventLoopDelayMs: number;
}

export type BenchmarkStrategy = Strategy | 'cpu-direct';

export interface Row {
  operation?: Operation;
  size: number;
  strategy: BenchmarkStrategy;
  unavailable?: string;
  cold?: Sample;
  warmups: Sample[];
  warm: Sample[];
}

const strategies = ['cpu-only', 'cpu-direct', 'gpu-only', 'static', 'adaptive'] as const;
export const labels: Record<BenchmarkStrategy, string> = {
  'cpu-only': 'CPU only', 'cpu-direct': 'CPU direct', 'gpu-only': 'GPU only', static: 'Static hybrid', adaptive: 'Adaptive',
};

export function parseSizes(value: string, operation: Operation = 'vectorMultiply') {
  const tokens = value.trim().split(/[,\s]+/);
  const sizes = [...new Set(tokens.map(Number))];
  const max = operation === 'matrixMultiply' ? 1024 : 16_777_216;
  if (tokens.some((token) => !/^\d+$/.test(token)) || sizes.length > 8
      || sizes.some((size) => !Number.isSafeInteger(size) || size < 2 || size > max)) {
    throw new Error(`Enter 1–8 sizes between 2 and ${max.toLocaleString()}, separated by commas or spaces.`);
  }
  return sizes;
}

export function median(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function latency(sample: Sample) {
  return sample.profile.totalMs;
}

export function p95(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(0.95 * sorted.length) - 1];
}

export function valid(row: Row) {
  const requested = row.strategy === 'cpu-direct' ? 'cpu-only' : row.strategy;
  return Boolean(!row.unavailable && row.cold && row.warm.length && [row.cold, ...row.warmups, ...row.warm].every((sample) => {
    const { profile } = sample;
    if (sample.mismatches || profile.requestedStrategy !== requested || (row.operation && row.operation !== profile.operation)) return false;
    if (row.strategy === 'adaptive') return true;
    if (profile.strategy !== requested) return false;
    if (row.strategy === 'cpu-direct') return profile.chunks.every((chunk) => chunk.cpuExecution === 'inline');
    return true;
  }));
}

export function speedup(row: Row, rows: readonly Row[]) {
  if (!valid(row)) return undefined;
  const comparable = rows.filter((item) => item.size === row.size && item.operation === row.operation);
  const cpu = comparable.find((item) => item.strategy === 'cpu-only' && valid(item));
  const gpu = comparable.find((item) => item.strategy === 'gpu-only' && valid(item));
  const time = median(row.warm.map(latency));
  if (!cpu || !gpu || time <= 0) return undefined;
  const baseline = Math.min(median(cpu.warm.map(latency)), median(gpu.warm.map(latency)));
  return baseline > 0 ? baseline / time : undefined;
}

export function speedupVsCpu(row: Row, rows: readonly Row[]) {
  if (!valid(row)) return undefined;
  const cpu = rows.find((sample) => sample.size === row.size && sample.operation === row.operation && sample.strategy === 'cpu-only' && valid(sample));
  const current = median(row.warm.map(latency));
  if (!cpu || current <= 0) return undefined;
  const baseline = median(cpu.warm.map(latency));
  return baseline > 0 ? baseline / current : undefined;
}

export async function benchmark(
  settings: Settings, capabilities: Capabilities, signal: AbortSignal,
  progress: (message: string) => void, report: (rows: Row[]) => void,
) {
  const rows: Row[] = [];
  const operation = settings.operation ?? 'vectorMultiply';
  const selected = settings.comparison === 'cpu' ? ['cpu-only', 'cpu-direct', 'adaptive'] as const : strategies;
  for (const size of settings.sizes) {
    signal.throwIfAborted();
    progress(`Testing ${workloadLabels[operation]} at size ${size.toLocaleString()}…`);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    signal.throwIfAborted();
    const { request, reference, tolerance } = inputsFor(operation, size);
    const group: Row[] = selected.filter((strategy) => strategy !== 'cpu-direct' || settings.includeDirect !== false)
      .map((strategy) => ({ operation, size, strategy, warmups: [], warm: [] }));
    const active: { row: Row; compute: () => Promise<ComputeResult>; close?: () => void }[] = [];
    const measure = async ({ row, compute }: (typeof active)[number]) => {
      if (row.unavailable) return;
      const stopProbe = startEventLoopProbe();
      try {
        const { output, profile } = await compute();
        const eventLoopDelayMs = stopProbe();
        if (profile.fallbackReason && row.strategy !== 'adaptive') throw new Error(profile.fallbackReason);
        let mismatches = Math.abs(reference.length - output.length);
        for (let index = 0; index < Math.min(reference.length, output.length); index++) {
          const error = tolerance ? tolerance + 1e-6 * Math.abs(reference[index]) : 0;
          if (!(Math.abs(output[index] - reference[index]) <= error)) mismatches++;
        }
        return { profile, mismatches, eventLoopDelayMs };
      } catch (error) {
        stopProbe();
        signal.throwIfAborted();
        row.unavailable = error instanceof Error ? error.message : String(error);
      }
    };
    try {
      const staticCpuElements = Math.round(size * settings.cpuShare);
      for (const row of group) {
        if ((row.strategy === 'gpu-only' || row.strategy === 'static') && !capabilities.gpu.available) {
          row.unavailable = capabilities.gpu.reason;
        } else if (row.strategy === 'static' && (staticCpuElements === 0 || staticCpuElements === size)) {
          row.unavailable = 'This split assigns the entire vector to a single backend.';
        } else if (row.strategy === 'cpu-direct') {
          const direct = new DirectCpuReference();
          active.push({ row, compute: () => direct.compute(request, signal) });
        } else {
          const engine = await HeteroJS.create({
            strategy: row.strategy, cpuShare: settings.cpuShare, cpuWorkers: capabilities.cpuWorkers,
            gpu: capabilities.gpu.available ? 'auto' : 'disabled',
          });
          active.push({ row, compute: () => engine.compute({ ...request, signal }), close: () => engine.close() });
        }
      }
      for (const item of active) item.row.cold = await measure(item);
      for (let sample = 0; sample < settings.warmups + settings.samples; sample++) {
        for (let index = 0; index < active.length; index++) {
          const item = active[(sample + index) % active.length];
          const result = await measure(item);
          if (result) (sample < settings.warmups ? item.row.warmups : item.row.warm).push(result);
        }
      }
      rows.push(...group);
      report(rows);
    } finally {
      for (const { close } of active) close?.();
    }
  }
  return rows;
}
