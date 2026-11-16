import manifest from '../../package.json' with { type: 'json' };
import type { Capabilities, Operation, Strategy } from '../../src/index.js';
import { median, p95 } from '../runner.js';

export const modes = ['cpu-only', 'gpu-only', 'static', 'adaptive', 'legacy'] as const;
export type Mode = Strategy | 'legacy';
export type Metric = 'computeMs' | 'firstPreviewMs' | 'frameMs';
export type Statistic = 'median' | 'p95';
export const metrics = { computeMs: 'Compute', firstPreviewMs: 'First preview', frameMs: 'Full frame' } as const;
const suite = 'heterojs-device-benchmarks/1';
const { version } = manifest;
export const fractaliaRevision = 'fb70975499166771e4f61e7a0d3800c2b5ab419c';

type Preset = 'julia' | 'seahorse' | 'ship';
type Builtin = { operation: Operation; size: number; label: string };
export type Fractal = { preset: Preset; width: number; height: number; iterations: number; label: string };
type Case = Builtin | Fractal;
export type Profile = 'quick' | 'standard';
export const profiles: Record<Profile, { label: string; warmups: number; samples: number; cases: Case[] }> = {
  quick: { label: 'Quick', warmups: 8, samples: 8, cases: [
    { operation: 'vectorMultiply', size: 1_048_576, label: 'Vector multiply · 1,048,576 elements' },
    { operation: 'matrixMultiply', size: 257, label: 'Matrix multiply · 257 × 257' },
    { preset: 'julia', width: 480, height: 360, iterations: 400, label: 'Fractalia · Julia · 480 × 360' },
    { preset: 'ship', width: 480, height: 360, iterations: 400, label: 'Fractalia · Burning Ship · 480 × 360' },
  ] },
  standard: { label: 'Standard', warmups: 10, samples: 20, cases: [
    { operation: 'vectorMultiply', size: 1_048_576, label: 'Vector multiply · 1,048,576 elements' },
    { operation: 'vectorMultiply', size: 4_194_304, label: 'Vector multiply · 4,194,304 elements' },
    { operation: 'matrixMultiply', size: 257, label: 'Matrix multiply · 257 × 257' },
    { operation: 'matrixMultiply', size: 513, label: 'Matrix multiply · 513 × 513' },
    { preset: 'julia', width: 640, height: 480, iterations: 600, label: 'Fractalia · Julia · 640 × 480' },
    { preset: 'seahorse', width: 640, height: 480, iterations: 600, label: 'Fractalia · Mandelbrot · 640 × 480' },
    { preset: 'ship', width: 960, height: 720, iterations: 600, label: 'Fractalia · Burning Ship · 960 × 720' },
  ] },
};
export type Quality = { nonfinite: number; exactDifferences: number; pixels: number; rgbOver8: number };
export type Sample = Partial<Record<Metric, number>> & { cpuUnits?: number; gpuUnits?: number; plan?: string };
export interface Row {
  mode: Mode;
  cold?: Sample;
  warmups: Sample[];
  measured: Sample[];
  issue?: string;
  quality?: Quality;
}
export interface Result { id: string; rows: Row[] }
export interface Device {
  label: string;
  userAgent: string;
  logicalCores: number;
  deviceMemoryGiB: number | null;
  capabilities: Capabilities;
}
export interface Report {
  suite: typeof suite;
  engine: string;
  fractalia: string;
  date: string;
  profile: Profile;
  device: Device;
  results: Result[];
}

export function caseId(item: Case) {
  return 'preset' in item ? `${item.preset}/${item.width}x${item.height}/${item.iterations}` : `${item.operation}/${item.size}`;
}

export function createReport(profile: Profile, device: Device, results: Result[]): Report {
  return { suite, engine: version, fractalia: fractaliaRevision, date: new Date().toISOString(), profile, device, results };
}

export function statistic(result: Result, mode: Mode, metric: Metric, stat: Statistic) {
  const row = result.rows.find((candidate) => candidate.mode === mode);
  if (!row || row.issue || !row.measured.length) return undefined;
  const values = row.measured.map((sample) => sample[metric]);
  if (!values.every((value) => value !== undefined)) return undefined;
  return (stat === 'p95' ? p95 : median)(values);
}

export function speedups(result: Result, metric: Metric, stat: Statistic) {
  const time = statistic(result, 'adaptive', metric, stat);
  const baseline = (mode: Mode) => {
    const value = statistic(result, mode, metric, stat);
    return time && value ? value / time : undefined;
  };
  const cpu = baseline('cpu-only');
  const gpu = baseline('gpu-only');
  return { cpu, gpu, best: cpu && gpu ? Math.min(cpu, gpu) : undefined, static: baseline('static'), legacy: baseline('legacy') };
}

function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function amount(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}

function validSample(value: unknown, required: Metric[]) {
  return object(value) && required.every((metric) => amount(value[metric])) &&
    Object.keys(metrics).every((metric) => value[metric] === undefined || amount(value[metric])) &&
    ['cpuUnits', 'gpuUnits'].every((key) => value[key] === undefined || amount(value[key])) &&
    (value.plan === undefined || typeof value.plan === 'string');
}

export function readReport(text: string): Report {
  const report: unknown = JSON.parse(text);
  if (!object(report) || report.suite !== suite || report.engine !== version || report.fractalia !== fractaliaRevision) {
    throw new Error('Use a JSON report exported by the same benchmark version.');
  }
  if (report.profile !== 'quick' && report.profile !== 'standard') throw new Error('Unknown test profile.');
  const expected = profiles[report.profile];
  const device = report.device;
  if (typeof report.date !== 'string' || !Number.isFinite(Date.parse(report.date)) || !object(device) ||
      typeof device.label !== 'string' || !device.label.trim() || typeof device.userAgent !== 'string' ||
      !amount(device.logicalCores) || (device.deviceMemoryGiB !== null && !amount(device.deviceMemoryGiB)) ||
      !object(device.capabilities) || !amount(device.capabilities.cpuWorkers) || !object(device.capabilities.gpu)) {
    throw new Error('Missing device information.');
  }
  const gpu = device.capabilities.gpu;
  if (gpu.available === true ? typeof gpu.fallbackAdapter !== 'boolean' ||
    ['vendor', 'architecture'].some((key) => gpu[key] !== undefined && typeof gpu[key] !== 'string') :
    gpu.available !== false || typeof gpu.reason !== 'string') throw new Error('Invalid adapter information.');
  if (!Array.isArray(report.results) || report.results.length !== expected.cases.length) throw new Error('Incomplete result set.');
  for (const [index, item] of expected.cases.entries()) {
    const result = report.results[index];
    const fractal = 'preset' in item;
    const expectedModes = fractal ? modes : modes.slice(0, -1);
    if (!object(result) || result.id !== caseId(item) || !Array.isArray(result.rows) || result.rows.length !== expectedModes.length) {
      throw new Error('The report does not match the cases in its profile.');
    }
    const rows = result.rows;
    for (const mode of expectedModes) {
      const row: unknown = rows.find((value: unknown) => object(value) && value.mode === mode);
      if (!object(row) || !Array.isArray(row.warmups) || !Array.isArray(row.measured) ||
        (row.issue !== undefined && (typeof row.issue !== 'string' || !row.issue.trim()))) throw new Error('Invalid strategy results.');
      const required: Metric[] = fractal ? ['firstPreviewMs', 'frameMs'] : [];
      if (mode !== 'legacy') required.push('computeMs');
      if ((!row.issue && (!row.cold || row.warmups.length !== expected.warmups || row.measured.length !== expected.samples)) ||
        (row.cold !== undefined && !validSample(row.cold, required)) || row.warmups.length > expected.warmups ||
        row.measured.length > expected.samples || ![...row.warmups, ...row.measured].every((value) => validSample(value, required))) {
        throw new Error('A strategy has missing or invalid timings.');
      }
      if (row.quality !== undefined) {
        const quality = row.quality;
        if (!object(quality) || !amount(quality.pixels) || !quality.pixels ||
          !['nonfinite', 'exactDifferences', 'rgbOver8'].every((key) => amount(quality[key]) && quality[key] <= (quality.pixels as number))) {
          throw new Error('Invalid Fractalia quality data.');
        }
      }
    }
  }
  return report as unknown as Report;
}
