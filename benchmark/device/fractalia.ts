import { HeteroJS } from '../../src/index.js';
import type { Capabilities, ComputeProfile, Strategy } from '../../src/index.js';
import { caseId, modes } from './protocol.js';
import type { Fractal, Mode, Quality, Result, Row, Sample } from './protocol.js';

type State = { type: 'mandelbrot' | 'julia' | 'ship'; digits: number; cx: string; cy: string; span: string;
  jr: number; ji: number; shift: number; palette: string };
type Request = { id: number; state: State; width: number; height: number; iterations: number };
type Pass = { step: number; done: boolean; width: number; height: number; values: Float64Array; profile?: ComputeProfile };
type FixtureRenderer = { engine: HeteroJS; operation: unknown };
type Colorize = (values: Float64Array, iterations: number, palette: string, shift: number) => Uint8ClampedArray<ArrayBuffer>;

const presets = {
  julia: { type: 'julia', cx: '0', cy: '0', span: '3.3', palette: 'ocean' },
  seahorse: { type: 'mandelbrot', cx: '-.7453', cy: '.1127', span: '.012', palette: 'aurora' },
  ship: { type: 'ship', cx: '-1.755', cy: '-.032', span: '.085', palette: 'ember' },
} as const;
const fixture = new URL('./fractalia/', document.baseURI);
const load = <T>(file: string): Promise<T> => import(/* @vite-ignore */ new URL(file, fixture).href);

async function loadFixture() {
  const [math, paint, renderer] = await Promise.all([
    load<{ parseDecimal: (value: string, digits: number) => bigint }>('math.js'),
    load<{ colorize: Colorize }>('colorize.js'),
    load<{
      createRenderer: (engine: typeof HeteroJS, cpuWorkers: number, gpu: 'auto' | 'disabled') => Promise<FixtureRenderer>;
      renderWithHeteroJS: (renderer: FixtureRenderer, request: Request, mode: Strategy, signal: AbortSignal,
        onPass: (pass: Pass) => void) => Promise<void>;
    }>('hetero-renderer.js'),
  ]);
  return { ...math, ...paint, ...renderer };
}

function legacy(request: Request, lanes: number, signal: AbortSignal, onPass: (pass: Pass) => void) {
  const workers: Worker[] = [];
  const passes = new Map<number, { values: Float64Array; received: number }>();
  let cleanup: (() => void) | undefined;
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) { reject(signal.reason); return; }
    const onAbort = () => reject(signal.reason);
    signal.addEventListener('abort', onAbort, { once: true });
    cleanup = () => signal.removeEventListener('abort', onAbort);
    try {
      for (let lane = 0; lane < lanes; lane++) {
        const worker = new Worker(new URL('worker.js', fixture), { type: 'module' });
        workers.push(worker);
        worker.onerror = (event) => reject(new Error(event.message || 'Fractalia worker failed.'));
        worker.onmessageerror = () => reject(new Error('Fractalia worker returned an unreadable message.'));
        worker.onmessage = ({ data }: MessageEvent<Pass & { error?: string }>) => {
          try {
            if (data.error) throw new Error(data.error);
            if (lanes === 1) onPass(data);
            else {
              const { step, width, height, values } = data;
              const pass = passes.get(step) ?? { values: new Float64Array(width * height), received: 0 };
              passes.set(step, pass);
              for (let y = lane, offset = 0; y < height; y += lanes, offset += width) {
                pass.values.set(values.subarray(offset, offset + width), y * width);
              }
              if (++pass.received !== lanes) return;
              passes.delete(step);
              onPass({ ...data, values: pass.values });
            }
            if (data.done) resolve();
          } catch (error) { reject(error); }
        };
        worker.postMessage({ ...request, lane, lanes });
      }
    } catch (error) { reject(error); }
  }).finally(() => {
    cleanup?.();
    workers.forEach((worker) => worker.terminate());
  });
}

function checkProfile(mode: Mode, profile: ComputeProfile | undefined, pixels: number) {
  if (mode === 'legacy' && !profile) return;
  if (!profile || profile.requestedStrategy !== mode || profile.elements !== pixels ||
    profile.cpuElements + profile.gpuElements !== pixels ||
    (mode === 'cpu-only' && profile.cpuElements !== pixels) ||
    (mode === 'gpu-only' && profile.gpuElements !== pixels) ||
    (mode === 'static' && (!profile.cpuElements || !profile.gpuElements))) {
    throw new Error('The render did not execute on the requested backend.');
  }
}

function compare(actual: Float64Array, expected: Float64Array | undefined, expectedColors: Uint8ClampedArray | undefined,
  colorize: Colorize, request: Request): Quality {
  const { iterations, state: { palette, shift } } = request;
  const result: Quality = { pixels: actual.length, nonfinite: 0, exactDifferences: 0, rgbOver8: 0 };
  const colors = expectedColors && colorize(actual, iterations, palette, shift);
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i])) result.nonfinite++;
    if (expected && actual[i] !== expected[i]) result.exactDifferences++;
    if (colors && expectedColors) {
      const pixel = i * 4;
      if (Math.max(Math.abs(colors[pixel] - expectedColors[pixel]), Math.abs(colors[pixel + 1] - expectedColors[pixel + 1]),
        Math.abs(colors[pixel + 2] - expectedColors[pixel + 2])) > 8) result.rgbOver8++;
    }
  }
  return result;
}

export async function fractaliaBenchmark(item: Fractal, runs: { warmups: number; samples: number }, capabilities: Capabilities,
  canvas: HTMLCanvasElement, signal: AbortSignal, progress: (message: string) => void): Promise<Result> {
  const { parseDecimal, colorize, createRenderer, renderWithHeteroJS } = await loadFixture();
  const { width, height, iterations, preset } = item;
  const seed = presets[preset];
  const digits = 40;
  const request: Request = { id: 1, width, height, iterations, state: { ...seed, digits,
    cx: parseDecimal(seed.cx, digits).toString(), cy: parseDecimal(seed.cy, digits).toString(),
    span: parseDecimal(seed.span, digits).toString(), jr: -.745, ji: .186, shift: 0 } };
  const cpuWorkers = Math.min(capabilities.cpuWorkers, 4);
  const lanes = width * height < 400_000 ? 1 : cpuWorkers;
  const scratch = document.createElement('canvas');
  const drawing = canvas.getContext('2d', { alpha: false });
  const colors = scratch.getContext('2d');
  if (!drawing || !colors) throw new Error('Canvas 2D is unavailable.');
  canvas.width = width;
  canvas.height = height;
  const rows: Row[] = modes.map((mode) => ({ mode, warmups: [], measured: [] }));
  const engines = new Map<Mode, FixtureRenderer>();
  const fields = new Map<Mode, Float64Array>();
  async function measure(row: Row) {
    const { mode } = row;
    signal.throwIfAborted();
    const started = performance.now();
    const sample: Sample = {};
    const steps: number[] = [];
    let field: Float64Array | undefined;
    const onPass = (pass: Pass) => {
      steps.push(pass.step);
      scratch.width = pass.width;
      scratch.height = pass.height;
      const pixels = colorize(pass.values, iterations, seed.palette, request.state.shift);
      colors!.putImageData(new ImageData(pixels, pass.width, pass.height), 0, 0);
      drawing!.drawImage(scratch, 0, 0, width, height);
      const elapsed = performance.now() - started;
      if (pass.step === 6) sample.firstPreviewMs = elapsed;
      if (pass.done) {
        sample.frameMs = elapsed;
        const profile = pass.profile;
        checkProfile(mode, profile, width * height);
        if (profile) {
          sample.computeMs = profile.totalMs;
          sample.cpuUnits = profile.cpuElements;
          sample.gpuUnits = profile.gpuElements;
          sample.plan = profile.customDispatch?.mode ?? profile.strategy;
        }
        field = pass.values;
      }
    };
    if (mode === 'legacy') await legacy(request, lanes, signal, onPass);
    else await renderWithHeteroJS(engines.get(mode)!, request, mode, signal, onPass);
    if (steps.join(',') !== '6,2,1' || !field || field.length !== width * height) {
      throw new Error('Fractalia returned an incomplete frame.');
    }
    if (!field.every(Number.isFinite)) throw new Error('Fractalia returned non-finite values.');
    fields.set(mode, field);
    return sample;
  }
  try {
    for (const row of rows) {
      if (row.mode === 'legacy') continue;
      if (!capabilities.gpu.available && (row.mode === 'gpu-only' || row.mode === 'static')) {
        row.issue = capabilities.gpu.reason;
        continue;
      }
      engines.set(row.mode, await createRenderer(HeteroJS, cpuWorkers, capabilities.gpu.available ? 'auto' : 'disabled'));
    }
    for (let trial = 0; trial <= runs.warmups + runs.samples; trial++) {
      const phase = trial === 0 ? 'cold call' : trial <= runs.warmups ? `warm-up ${trial}/${runs.warmups}` :
        `sample ${trial - runs.warmups}/${runs.samples}`;
      progress(`${item.label} · ${phase}`);
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      for (let i = 0; i < rows.length; i++) {
        const row = rows[(trial + i) % rows.length];
        if (row.issue) continue;
        try {
          const sample = await measure(row);
          if (trial === 0) row.cold = sample;
          else (trial <= runs.warmups ? row.warmups : row.measured).push(sample);
        } catch (error) {
          signal.throwIfAborted();
          row.issue = error instanceof Error ? error.message : String(error);
        }
      }
    }
    const reference = fields.get('legacy');
    const expectedColors = reference && colorize(reference, iterations, seed.palette, request.state.shift);
    for (const row of rows) {
      const field = fields.get(row.mode);
      if (row.issue || !field) continue;
      row.quality = compare(field, reference, expectedColors, colorize, request);
      if (!reference || rows.find(({ mode }) => mode === 'legacy')?.issue) row.issue = 'Stock Fractalia did not produce a reference frame.';
      else if (row.mode === 'cpu-only' && row.quality.exactDifferences) row.issue = 'The CPU field differs from stock Fractalia.';
    }
    return { id: caseId(item), rows };
  } finally {
    for (const { engine } of engines.values()) engine.close();
    scratch.width = scratch.height = 1;
  }
}
