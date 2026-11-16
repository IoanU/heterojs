import { HeteroJS } from '../src/index.js';
import type { Capabilities, ChunkProfile, Operation } from '../src/index.js';
import { benchmark, labels, latency, median, p95, parseSizes, speedup, speedupVsCpu, valid } from './runner.js';
import type { Row, Settings } from './runner.js';
import './style.css';

const form = document.querySelector<HTMLFormElement>('#settings')!;
const hardware = document.querySelector<HTMLDivElement>('#hardware')!;
const status = document.querySelector<HTMLParagraphElement>('#status')!;
const results = document.querySelector<HTMLDivElement>('#results')!;
const empty = document.querySelector<HTMLDivElement>('#empty')!;
const run = document.querySelector<HTMLButtonElement>('#run')!;
const cancel = document.querySelector<HTMLButtonElement>('#cancel')!;
const download = document.querySelector<HTMLButtonElement>('#download')!;
const comparison = document.querySelector<HTMLSelectElement>('#comparison')!;
const operationSelect = document.querySelector<HTMLSelectElement>('#operation')!;
const staticShare = document.querySelector<HTMLInputElement>('#share')!;
const input = (id: string) => document.querySelector<HTMLInputElement>(`#${id}`)!;

let capabilities: Capabilities;
let controller: AbortController | undefined;
let lastReport: { settings: Settings; capabilities: Capabilities; rows: Row[]; browser: string; logicalCores: number; date: string } | undefined;
const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const ms = (value: number) => `${number.format(value)} ms`;

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function sum(chunks: readonly ChunkProfile[], key: 'computeMs' | 'inputTransferMs' | 'outputTransferMs' | 'dispatchMs' | 'synchronizationMs' | 'queueMs') {
  return chunks.reduce((total, chunk) => total + (chunk[key] ?? 0), 0);
}

function details(group: readonly Row[]) {
  const disclosure = element('details');
  disclosure.className = 'details';
  disclosure.append(element('summary', 'Timing breakdown'));
  const list = element('ul');
  for (const row of group) {
    if (!row.warm.length) continue;
    const target = median(row.warm.map(latency));
    const nearest = row.warm.reduce((best, sample) =>
      Math.abs(latency(sample) - target) < Math.abs(latency(best) - target) ? sample : best);
    const { chunks, setupMs, schedulingMs, aggregationMs, cpuElements, gpuElements, inlineTiming } = nearest.profile;
    const cpu = chunks.filter((chunk) => chunk.backend === 'cpu');
    const inline = cpu.filter((chunk) => chunk.cpuExecution === 'inline').reduce((total, chunk) => total + chunk.length, 0);
    const gpu = chunks.filter((chunk) => chunk.backend === 'gpu');
    const compute = [
      cpu.length && `${ms(sum(cpu, 'computeMs'))} CPU kernel`,
      gpu.length && (gpu.some((chunk) => chunk.computeMs === null) ? 'GPU kernel timing unavailable' : `${ms(sum(gpu, 'computeMs'))} GPU kernel`),
    ].filter(Boolean).join(' · ');
    const transferMs = sum(chunks, 'inputTransferMs') + sum(chunks, 'outputTransferMs');
    const synchronization = gpu.length ? ` · ${ms(sum(gpu, 'synchronizationMs'))} GPU synchronization` : '';
    const probes = row.strategy === 'adaptive'
      ? ` · ${chunks.filter((chunk) => chunk.exploratory).length} exploration chunks` : '';
    const units = row.operation === 'matrixMultiply' ? 'rows' : 'elements';
    const allocation = inline > 0
      ? `${inline.toLocaleString()} inline CPU / ${(cpuElements - inline).toLocaleString()} worker / ${gpuElements.toLocaleString()} GPU ${units}`
      : `${cpuElements.toLocaleString()} CPU / ${gpuElements.toLocaleString()} GPU ${units}`;
    const turns = inlineTiming ? ` · ${ms(inlineTiming.longestTurnComputeMs)} longest inline turn · ${inlineTiming.yields} yields (${ms(inlineTiming.yieldWaitMs)} waiting)` : '';
    list.append(element('li', `${labels[row.strategy]} · ${allocation} · ${chunks.length} chunks${probes} · ${compute} · ${ms(transferMs)} host transfer work · ${ms(sum(chunks, 'dispatchMs'))} dispatch · ${ms(sum(chunks, 'queueMs'))} queue · ${ms(setupMs + schedulingMs + aggregationMs)} setup, scheduling and merge${synchronization}${turns}`));
  }
  if (list.children.length === 0) return;
  disclosure.append(list, element('p', 'Timing sums use the run closest to the median. Parallel chunks and GPU timings can overlap.'));
  return disclosure;
}

function render(rows: Row[]) {
  const content = document.createDocumentFragment();
  for (const size of new Set(rows.map((row) => row.size))) {
    const group = rows.filter((row) => row.size === size);
    const section = element('section');
    section.className = 'result-group';
    const matrix = group[0].operation === 'matrixMultiply';
    const name = size.toLocaleString();
    const title = element('h3', matrix ? `${name} × ${name} matrices` : `${name} elements`);
    const scroll = element('div');
    scroll.className = 'table-scroll';
    scroll.tabIndex = 0;
    scroll.setAttribute('role', 'region');
    scroll.setAttribute('aria-label', `${title.textContent} results`);
    const table = element('table');
    const header = table.createTHead().insertRow();
    const hasGpuBaseline = group.some((row) => row.strategy === 'gpu-only');
    const headers = ['Strategy', 'Cold', 'Median', 'p95', 'p95 loop delay', matrix ? 'GFLOP/s' : 'M elem/s', 'vs. CPU only',
      ...(hasGpuBaseline ? ['vs. best CPU/GPU'] : []), 'Check'];
    for (const label of headers) {
      const cell = element('th', label);
      cell.scope = 'col';
      if (label === 'p95 loop delay') cell.title = '95th percentile of the maximum 4 ms timer delay during each run';
      header.append(cell);
    }
    const body = table.createTBody();
    for (const row of group) {
      const tr = body.insertRow();
      const name = element('th', labels[row.strategy]);
      name.scope = 'row';
      tr.append(name);
      if (row.unavailable || !row.cold || !row.warm.length) {
        const cell = tr.insertCell();
        cell.colSpan = headers.length - 1;
        cell.className = 'unavailable';
        cell.textContent = row.unavailable ?? 'No completed measurement.';
        continue;
      }
      const times = row.warm.map(latency);
      const typical = median(times);
      const relative = speedup(row, rows);
      const relativeCpu = speedupVsCpu(row, rows);
      const correct = valid(row);
      const cells = [
        ms(latency(row.cold)), ms(typical), ms(p95(times)), ms(p95(row.warm.map((sample) => sample.eventLoopDelayMs))),
        typical > 0 ? number.format(matrix ? 2 * size ** 3 / (typical * 1_000_000) : size / (typical * 1_000)) : '—',
        relativeCpu === undefined ? '—' : `${relativeCpu.toFixed(2)}×`,
        ...(hasGpuBaseline ? [relative === undefined ? '—' : `${relative.toFixed(2)}×`] : []), correct ? 'Pass' : 'Fail',
      ];
      for (const value of cells) tr.insertCell().textContent = value;
      tr.lastElementChild!.className = correct ? 'pass' : 'fail';
    }
    scroll.append(table);
    section.append(title, scroll);
    const breakdown = details(group);
    if (breakdown) section.append(breakdown);
    content.append(section);
  }
  results.replaceChildren(content);
  empty.hidden = rows.length > 0;
}

async function initialize() {
  if (typeof Worker !== 'function' || typeof AbortSignal.any !== 'function') {
    hardware.textContent = 'Benchmark unavailable';
    status.textContent = 'Use a browser with Web Workers and AbortSignal.any.';
    return;
  }
  const probe = await HeteroJS.create();
  try {
    capabilities = await probe.getCapabilities();
  } finally {
    probe.close();
  }
  const { cpuWorkers, gpu } = capabilities;
  const device = !gpu.available ? 'WebGPU unavailable' : gpu.fallbackAdapter ? 'WebGPU fallback adapter' : 'WebGPU available';
  hardware.textContent = `${cpuWorkers} CPU ${cpuWorkers === 1 ? 'worker' : 'workers'} · ${device}`;
  status.textContent = gpu.available ? 'Ready to benchmark.' : gpu.reason;
  run.disabled = false;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (controller || !capabilities) return;
  const operation = operationSelect.value as Operation;
  let sizes: number[];
  try {
    sizes = parseSizes(input('sizes').value, operation);
  } catch (error) {
    status.textContent = (error as Error).message;
    input('sizes').focus();
    return;
  }
  const focused = comparison.value === 'cpu';
  const settings: Settings = {
    operation,
    sizes, samples: input('samples').valueAsNumber, warmups: input('warmups').valueAsNumber,
    cpuShare: focused ? 0.25 : staticShare.valueAsNumber / 100,
    comparison: focused ? 'cpu' : 'all',
  };
  const active = new AbortController();
  controller = active;
  run.disabled = download.disabled = true;
  cancel.hidden = false;
  lastReport = undefined;
  render([]);
  try {
    const rows = await benchmark(settings, capabilities, active.signal, (message) => { status.textContent = message; }, render);
    lastReport = { settings, capabilities, rows, browser: navigator.userAgent,
      logicalCores: navigator.hardwareConcurrency, date: new Date().toISOString() };
    const passed = rows.filter(valid).length;
    const unavailable = rows.filter((row) => row.unavailable).length;
    const failed = rows.length - passed - unavailable;
    status.textContent = `Complete · ${passed} passed${failed ? ` · ${failed} failed` : ''}${unavailable ? ` · ${unavailable} unavailable` : ''}`;
    download.disabled = false;
  } catch (error) {
    status.textContent = active.signal.aborted ? 'Benchmark cancelled.' : error instanceof Error ? error.message : String(error);
  } finally {
    controller = undefined;
    run.disabled = false;
    cancel.hidden = true;
  }
});

cancel.addEventListener('click', () => controller?.abort());
const updateComparison = () => { staticShare.disabled = comparison.value === 'cpu'; };
comparison.addEventListener('change', updateComparison);
updateComparison();
let matrixSelected = operationSelect.value === 'matrixMultiply';
const setSizeLabel = () => {
  document.querySelector('label[for="sizes"]')!.textContent = matrixSelected ? 'Matrix side lengths' : 'Vector lengths';
};
operationSelect.addEventListener('change', () => {
  const matrix = operationSelect.value === 'matrixMultiply';
  if (matrix !== matrixSelected) input('sizes').value = matrix ? '128, 256, 512' : '65536, 1048576, 4194304';
  matrixSelected = matrix;
  setSizeLabel();
});
setSizeLabel();
download.addEventListener('click', () => {
  if (!lastReport) return;
  const link = element('a');
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastReport, null, 2)], { type: 'application/json' }));
  link.href = url;
  link.download = 'heterojs-benchmark.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
});

void initialize().catch((error: unknown) => { status.textContent = error instanceof Error ? error.message : String(error); });
