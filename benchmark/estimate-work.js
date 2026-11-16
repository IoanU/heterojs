import { computeRange } from './estimate-work.cpu.js';

const states = new WeakMap();
const code = /* wgsl */ `
  struct Options {
    start: u32, count: u32, heavyStart: u32, heavyLength: u32,
    light: u32, heavy: u32, seed: u32, padding: u32,
  };
  @group(0) @binding(0) var<uniform> options: Options;
  @group(0) @binding(1) var<storage, read_write> result: array<u32>;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= options.count { return; }
    let position = options.start + id.x;
    let isHeavy = position >= options.heavyStart && position < options.heavyStart + options.heavyLength;
    let steps = select(options.light, options.heavy, isHeavy);
    var value = position ^ options.seed;
    for (var i = 0u; i < steps; i++) { value = (value ^ (value >> 13u)) * 1664525u + 1013904223u; }
    result[id.x] = value;
  }
`;

async function setup(device) {
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
  });
  const parameters = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  return { pipeline, parameters };
}

function buffersFor(device, state, bytes) {
  if (state.buffers?.output.size >= bytes) return state.buffers;
  state.buffers?.output.destroy();
  state.buffers?.readback.destroy();
  state.buffers = undefined;
  const size = 2 ** Math.ceil(Math.log2(bytes));
  const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  let readback;
  try {
    readback = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bindings = device.createBindGroup({ layout: state.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: state.parameters } }, { binding: 1, resource: { buffer: output } },
    ] });
    return state.buffers = { output, readback, bindings };
  } catch (error) {
    output.destroy();
    readback?.destroy();
    throw error;
  }
}

async function computeGpu(input, { device, signal, start, length }) {
  let pending = states.get(device);
  if (!pending) {
    pending = setup(device).catch((error) => { states.delete(device); throw error; });
    states.set(device, pending);
  }
  const state = await pending;
  signal.throwIfAborted();
  const bytes = length * Uint32Array.BYTES_PER_ELEMENT;
  const { output, readback, bindings } = buffersFor(device, state, bytes);
  device.queue.writeBuffer(state.parameters, 0, new Uint32Array([
    start, length, input.heavyStart, input.heavyLength, input.light, input.heavy, input.seed, 0,
  ]));
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(Math.ceil(length / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    signal.throwIfAborted();
    return new Uint32Array(readback.getMappedRange(0, bytes)).slice();
  } finally { readback.unmap(); }
}

export const layouts = ['uniform', 'clustered', 'moving'];
export const modes = ['cpu-only', 'gpu-only', 'static', 'adaptive', 'adaptive-work'];

export function makeInput(units, layout, step) {
  const heavyLength = layout === 'uniform' ? 0 : Math.max(1, Math.floor(units / 4));
  const heavyStart = layout === 'moving' && step % 2 ? units - heavyLength : 0;
  return { units, heavyLength, heavyStart, light: layout === 'uniform' ? 70 : 8, heavy: 256, seed: 132755 };
}

export function work(input, { start, length }) {
  const overlap = Math.max(0, Math.min(start + length, input.heavyStart + input.heavyLength) - Math.max(start, input.heavyStart));
  return length * input.light + overlap * (input.heavy - input.light);
}

export function reference(input) { return computeRange(input, { start: 0, length: input.units }); }

export function register(engine, mode) {
  return engine.registerOperation(`cost-benchmark.${mode}`, {
    cpu: { module: new URL('./estimate-work.cpu.js', import.meta.url), transferOutput: true },
    gpu: computeGpu,
    plan(input) {
      const result = new Uint32Array(input.units);
      return {
        units: input.units, maxChunkUnits: 2 ** 17,
        estimateWork: mode === 'adaptive-work' ? (range) => work(input, range) : undefined,
        chunk: () => input,
        collector: { add: ({ value, start }) => result.set(value, start), finish: () => result },
      };
    },
  });
}
