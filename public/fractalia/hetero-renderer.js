import { numeric, zoomLog } from './math.js';

export async function createRenderer(HeteroJS, cpuWorkers, gpu) {
  const engine = await HeteroJS.create({ cpuWorkers, gpu });
  const operation = engine.registerOperation('fractalia.escape-raster', {
    cpu: { module: new URL('./fractal-cpu.js', import.meta.url), transferOutput: true },
    gpu: async (input, context) => (await import('./fractal-gpu.js')).renderGpu(input, context),
    plan(input) {
      const units = input.passWidth * input.passHeight;
      let output;
      return {
        units,
        maxChunkUnits: Math.min(2 ** 19, Math.max(1, Math.floor(2 ** 28 / input.iterations))),
        adaptiveKey: input.adaptiveKey,
        chunk: () => input,
        collector: {
          add({ start, value }) {
            if (start === 0 && value.length === units && value instanceof Float64Array) output = value;
            else (output ??= new Float64Array(units)).set(value, start);
          },
          finish() { return output ?? new Float64Array(units); },
        },
      };
    },
  });
  return { engine, operation };
}

function workloadBand(values, max) {
  // Sample across the raster to keep cost classification bounded at large sizes.
  const stride = Math.max(1, Math.floor(values.length / 1024));
  let steps = 0;
  let count = 0;
  for (let i = 0; i < values.length; i += stride) {
    steps += Math.max(0, Math.min(max, values[i]));
    count++;
  }
  return Math.min(7, Math.floor(8 * steps / (max * count)));
}

export async function renderWithHeteroJS({ engine, operation }, request, mode, signal, onPass) {
  const { id, state, width, height, iterations } = request;
  const base = { width, height, iterations, type: state.type, jr: state.jr, ji: state.ji,
    cx: numeric(state.cx, state.digits), cy: numeric(state.cy, state.digits), span: numeric(state.span, state.digits) };
  let band = 0;
  for (const [index, step] of [6, 2, 1].entries()) {
    signal.throwIfAborted();
    const passWidth = Math.ceil(width / step);
    const passHeight = Math.ceil(height / step);
    // Use the same preview backend in every HeteroJS strategy.
    const passStrategy = step === 1 ? mode : 'cpu-only';
    const adaptiveKey = mode === 'adaptive' ? [base.type, width, height, iterations, step, Math.floor(zoomLog(state)), band].join(':') : undefined;
    const { output: values, profile } = await engine.compute({ operation, strategy: passStrategy, signal,
      input: { ...base, passWidth, passHeight, step, adaptiveKey } });
    const done = step === 1;
    if (step === 6 && mode === 'adaptive') band = workloadBand(values, iterations);
    onPass({ id, width: passWidth, height: passHeight, values, step, done, deep: false, profile,
      progress: done ? 100 : (index + 1) * 40 });
  }
}
