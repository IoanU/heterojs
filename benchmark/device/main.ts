import { HeteroJS } from '../../src/index.js';
import type { Capabilities } from '../../src/index.js';
import { benchmark, labels, valid } from '../runner.js';
import type { Sample as EngineSample } from '../runner.js';
import { fractaliaBenchmark } from './fractalia.js';
import { caseId, createReport, metrics, profiles, readReport, speedups, statistic } from './protocol.js';
import type { Device, Metric, Profile, Report, Result, Sample, Statistic } from './protocol.js';
import './style.css';

const select = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const input = select<HTMLInputElement>('device');
const setup = select<HTMLFormElement>('setup');
const profileInput = select<HTMLSelectElement>('profile');
const compareProfile = select<HTMLSelectElement>('compare-profile');
const statInput = select<HTMLSelectElement>('stat');
const runButton = select<HTMLButtonElement>('run');
const cancel = select<HTMLButtonElement>('cancel');
const exportButton = select<HTMLButtonElement>('export');
const clear = select<HTMLButtonElement>('clear');
const importInput = select<HTMLInputElement>('import');
const status = select('status');
const importStatus = select('import-status');
const canvas = select<HTMLCanvasElement>('fractal');
const reports = new Map<string, Report>();
let capability: Capabilities;
let controller: AbortController | undefined;
let latest: Report | undefined;

const number = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
const time = (value: number | undefined) => value === undefined ? '—' : `${number.format(value)} ms`;
const ratio = (value: number | undefined) => value === undefined ? '—' : `${number.format(value)}×`;
const modeNames = { ...labels, legacy: 'Stock Fractalia' };

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function table(name: string, headers: string[], rows: string[][]) {
  const region = element('div');
  region.className = 'table-scroll';
  region.tabIndex = 0;
  region.setAttribute('role', 'region');
  region.setAttribute('aria-label', name);
  const grid = element('table');
  const head = grid.createTHead().insertRow();
  for (const label of headers) {
    const cell = element('th', label);
    cell.scope = 'col';
    head.append(cell);
  }
  const body = grid.createTBody();
  for (const [title, ...values] of rows) {
    const row = body.insertRow();
    const cell = element('th', title);
    cell.scope = 'row';
    row.append(cell, ...values.map((value) => element('td', value)));
  }
  region.append(grid);
  return region;
}

function describeGpu({ gpu }: Capabilities) {
  if (!gpu.available) return `GPU unavailable (${gpu.reason})`;
  const adapter = [gpu.vendor, gpu.architecture].filter(Boolean).join(' ') || 'adapter details hidden by browser';
  return `WebGPU · ${adapter}${gpu.fallbackAdapter ? ' · fallback adapter' : ''}`;
}

function showProfile() {
  const { cases, warmups, samples } = profiles[profileInput.value as Profile];
  select('profile-info').textContent = `${cases.length} cases · ${warmups} warm-ups and ${samples} timed calls per strategy. Keep this tab visible.`;
}

function showResult(report: Report) {
  const target = select('results');
  target.replaceChildren();
  for (const [index, item] of profiles[report.profile].cases.entries()) {
    const result = report.results[index];
    const fractal = 'preset' in item;
    const measured: Metric[] = fractal ? ['computeMs', 'firstPreviewMs', 'frameMs'] : ['computeMs'];
    const rows = result.rows.map((row) => [modeNames[row.mode], time(row.cold?.[fractal ? 'frameMs' : 'computeMs']),
      ...measured.map((metric) => `${time(statistic(result, row.mode, metric, 'median'))} / ${time(statistic(result, row.mode, metric, 'p95'))}`),
      row.issue ?? (row.quality ? `${number.format(100 * row.quality.rgbOver8 / row.quality.pixels)}% of pixels with RGB change >8` : 'Output verified')]);
    target.append(element('h3', item.label), table(item.label,
      ['Strategy', 'Cold', ...measured.map((metric) => `${metrics[metric]} median / p95`), 'Check'], rows));
  }
}

function storeReport(report: Report) {
  reports.set(`${report.profile}/${report.date}/${report.device.label}/${report.device.userAgent}`, report);
}

function showComparison() {
  const profile = compareProfile.value as Profile;
  const stat = statInput.value as Statistic;
  const selected = [...reports.values()].filter((report) => report.profile === profile);
  const target = select('comparison');
  clear.disabled = reports.size === 0;
  if (selected.length === 0) {
    target.replaceChildren(element('p', 'Add reports from devices that ran this test profile.'));
    return;
  }
  const devices = element('details');
  devices.append(element('summary', `${selected.length} ${selected.length === 1 ? 'report' : 'reports'} · device details`));
  const listing = element('ol');
  for (const { date, device } of selected) {
    const row = element('li', `${device.label} · ${date} · ${device.logicalCores} logical cores · ${device.capabilities.cpuWorkers} workers · ${describeGpu(device.capabilities)}`);
    row.append(element('small', device.userAgent));
    listing.append(row);
  }
  devices.append(listing);
  target.replaceChildren(devices);
  for (const [index, item] of profiles[profile].cases.entries()) {
    const measured: Metric[] = 'preset' in item ? ['computeMs', 'firstPreviewMs', 'frameMs'] : ['computeMs'];
    const rows = measured.flatMap((metric) => selected.map((report, deviceIndex) => {
      const result = report.results[index];
      const { cpu, gpu, best, static: fixed, legacy } = speedups(result, metric, stat);
      return [`${metrics[metric]} · ${report.device.label} (${deviceIndex + 1})`, time(statistic(result, 'adaptive', metric, stat)),
        ratio(cpu), ratio(gpu), ratio(best), ratio(fixed), ...('preset' in item ? [ratio(legacy)] : [])];
    }));
    target.append(element('h3', item.label), table(`${item.label} device comparison`,
      ['Metric and device', `Adaptive ${stat}`, 'Vs. CPU', 'Vs. GPU', 'Vs. best CPU/GPU', 'Vs. static',
        ...('preset' in item ? ['Vs. stock Fractalia'] : [])], rows));
  }
}

function engineSample({ profile }: EngineSample): Sample {
  return { computeMs: profile.totalMs, cpuUnits: profile.cpuElements, gpuUnits: profile.gpuElements,
    plan: profile.matrixDispatch?.mode ?? profile.strategy };
}

async function runSuite(profile: Profile, signal: AbortSignal) {
  const settings = profiles[profile];
  const results: Result[] = [];
  const progress = (message: string) => { status.textContent = message; };
  for (const item of settings.cases) {
    signal.throwIfAborted();
    if ('preset' in item) results.push(await fractaliaBenchmark(item, settings, capability, canvas, signal, progress));
    else {
      const rows = await benchmark({ samples: settings.samples, warmups: settings.warmups, operation: item.operation,
        sizes: [item.size], cpuShare: 0.25, includeDirect: false },
        capability, signal, progress, () => {});
      results.push({ id: caseId(item), rows: rows.flatMap((row) => row.strategy === 'cpu-direct' ? [] : [{
        mode: row.strategy, cold: row.cold && engineSample(row.cold), warmups: row.warmups.map(engineSample),
        measured: row.warm.map(engineSample), issue: row.unavailable ?? (valid(row) ? undefined : 'The numeric output or backend check failed.'),
      }]) });
    }
  }
  return results;
}

async function initialize() {
  const engine = await HeteroJS.create();
  try {
    capability = await engine.getCapabilities();
    select('hardware').textContent = `${navigator.hardwareConcurrency ?? 0} logical cores · ${capability.cpuWorkers} workers · ${describeGpu(capability)}`;
    runButton.disabled = false;
  } finally { engine.close(); }
}

setup.addEventListener('submit', async (event) => {
  event.preventDefault();
  const label = input.value.trim();
  if (!label || controller || !capability) return;
  const profile = profileInput.value as Profile;
  const device: Device = { label, userAgent: navigator.userAgent, logicalCores: navigator.hardwareConcurrency ?? 0,
    deviceMemoryGiB: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null, capabilities: capability };
  const run = new AbortController();
  controller = run;
  input.disabled = profileInput.disabled = runButton.disabled = true;
  cancel.disabled = false;
  try {
    const results = await runSuite(profile, run.signal);
    run.signal.throwIfAborted();
    const report = createReport(profile, device, results);
    latest = report;
    exportButton.disabled = false;
    showResult(report);
    storeReport(report);
    compareProfile.value = profile;
    showComparison();
    const issues = results.flatMap(({ rows }) => rows).filter((row) => row.issue).length;
    status.textContent = issues ? `Complete. ${issues} strategy rows were unavailable or failed checks. You can export the report.` : 'Complete. You can export the report.';
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    controller = undefined;
    input.disabled = profileInput.disabled = runButton.disabled = false;
    cancel.disabled = true;
  }
});

profileInput.addEventListener('change', showProfile);
compareProfile.addEventListener('change', showComparison);
statInput.addEventListener('change', showComparison);
cancel.addEventListener('click', () => controller?.abort(new Error('Benchmark cancelled.')));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) controller?.abort(new Error('Benchmark stopped when the tab became hidden. Run again with this tab visible.'));
});
exportButton.addEventListener('click', () => {
  if (!latest) return;
  const href = URL.createObjectURL(new Blob([JSON.stringify(latest, null, 2)], { type: 'application/json' }));
  const link = element('a');
  link.href = href;
  const slug = latest.device.label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  link.download = `heterojs-${latest.profile}-${slug || 'device'}-${latest.date.replace(/[:.]/g, '-')}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
});
importInput.addEventListener('change', async () => {
  const messages: string[] = [];
  let imported = 0;
  const files = Array.from(importInput.files ?? []);
  importInput.value = '';
  for (const file of files) {
    try {
      if (file.size > 2_000_000) throw new Error('File exceeds the 2 MB report limit.');
      const report = readReport(await file.text());
      if (imported === 0) compareProfile.value = report.profile;
      storeReport(report);
      imported++;
    } catch (error) { messages.push(`${file.name} · ${error instanceof Error ? error.message : String(error)}`); }
  }
  importStatus.textContent = [`${imported} ${imported === 1 ? 'report' : 'reports'} loaded.`, ...messages].join(' ');
  showComparison();
});
clear.addEventListener('click', () => {
  reports.clear();
  importStatus.textContent = 'Comparison cleared.';
  showComparison();
});

showProfile();
void initialize().catch((error) => { select('hardware').textContent = `Could not initialize the benchmark. ${error}`; });
