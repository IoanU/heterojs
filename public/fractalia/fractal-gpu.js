const states = new WeakMap();
const workgroupSize = 64;
const shader = /* wgsl */ `
  struct View {
    passWidth: u32, start: u32, count: u32, maxSteps: u32,
    family: u32, padding: u32,
    x0: vec2f, y0: vec2f, spacing: vec2f, jr: vec2f, ji: vec2f,
  }
  @group(0) @binding(0) var<uniform> view: View;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;

  // Each coordinate is stored as a rounded high word and its residual.
  fn normalized(high: f32, low: f32) -> vec2f {
    let sum = high + low;
    return vec2f(sum, low - (sum - high));
  }

  fn add(a: vec2f, b: vec2f) -> vec2f {
    let sum = a.x + b.x;
    let offset = sum - a.x;
    let residual = (a.x - (sum - offset)) + (b.x - offset);
    return normalized(sum, residual + a.y + b.y);
  }

  fn product(a: vec2f, b: vec2f) -> vec2f {
    let rounded = a.x * b.x;
    // Keep 12 significant bits per high part so its product is exact in f32.
    let ah = bitcast<f32>(bitcast<u32>(a.x) & 0xfffff000u);
    let bh = bitcast<f32>(bitcast<u32>(b.x) & 0xfffff000u);
    let al = a.x - ah;
    let bl = b.x - bh;
    let residual = (((ah * bh - rounded) + ah * bl) + al * bh) + al * bl;
    return normalized(rounded, residual + a.x * b.y + a.y * b.x);
  }

  fn magnitude(a: vec2f) -> vec2f {
    return select(a, -a, a.x < 0.0);
  }

  fn escapeValue(real: vec2f, imag: vec2f) -> f32 {
    var x = vec2f(0);
    var y = vec2f(0);
    var cr = real;
    var ci = imag;
    if view.family == 0u {
      let q = (real.x - 0.25) * (real.x - 0.25) + imag.x * imag.x;
      // Only skip clearly interior orbits. Points close to either boundary
      // go through the compensated recurrence.
      if q * (q + real.x - 0.25) < 0.25 * imag.x * imag.x - 0.000001
        || (real.x + 1.0) * (real.x + 1.0) + imag.x * imag.x < 0.0625 - 0.000001 {
        return f32(view.maxSteps);
      }
    } else if view.family == 1u {
      x = real;
      y = imag;
      cr = view.jr;
      ci = view.ji;
    }
    var xx = product(x, x);
    var yy = product(y, y);
    for (var n = 0u; n < view.maxSteps; n++) {
      if view.family == 2u {
        x = magnitude(x);
        y = magnitude(y);
      }
      let nx = add(add(xx, -yy), cr);
      y = add(2.0 * product(x, y), ci);
      x = nx;
      xx = product(x, x);
      yy = product(y, y);
      let mag = xx.x + yy.x;
      if mag > 256.0 { return f32(n + 1u) - log2(log2(mag) / 2.0); }
    }
    return f32(view.maxSteps);
  }

  @compute @workgroup_size(${workgroupSize})
  fn render(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= view.count { return; }
    let index = view.start + id.x;
    let px = vec2f(f32(index % view.passWidth), 0.0);
    let py = vec2f(f32(index / view.passWidth), 0.0);
    let real = add(view.x0, product(view.spacing, px));
    let imag = add(view.y0, product(view.spacing, py));
    output[id.x] = escapeValue(real, imag);
  }
`;

async function init(device) {
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module: device.createShaderModule({ code: shader }), entryPoint: 'render' },
  });
  const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  return { pipeline, uniform };
}

function ensureBuffers(device, state, bytes) {
  if (state.buffers?.output.size >= bytes) return state.buffers;
  state.buffers?.output.destroy();
  state.buffers?.readback.destroy();
  state.buffers = undefined;
  const size = 2 ** Math.ceil(Math.log2(bytes));
  const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  let readback;
  try {
    readback = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({ layout: state.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: state.uniform } },
      { binding: 1, resource: { buffer: output } },
    ] });
    return state.buffers = { output, readback, group };
  } catch (error) {
    output.destroy();
    readback?.destroy();
    throw error;
  }
}

export async function renderGpu(input, { device, signal, start, length }) {
  let pending = states.get(device);
  if (!pending) {
    pending = init(device).catch((error) => { states.delete(device); throw error; });
    states.set(device, pending);
  }
  const state = await pending;
  signal.throwIfAborted();
  const bytes = length * 4;
  const { output, readback, group } = ensureBuffers(device, state, bytes);
  const { width, height, passWidth, step, iterations, cx, cy, span, type, jr, ji } = input;
  const data = new ArrayBuffer(64);
  new Uint32Array(data).set([passWidth, start, length, iterations, type === 'mandelbrot' ? 0 : type === 'julia' ? 1 : 2]);
  const x0 = cx + (.5 * step / width - .5) * span;
  const y0 = cy + (.5 * step - height / 2) / width * span;
  const split = (v) => { const high = Math.fround(v); return [high, v - high]; };
  new Float32Array(data, 24).set([x0, y0, span / width * step, jr, ji].flatMap(split));
  device.queue.writeBuffer(state.uniform, 0, data);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(length / workgroupSize));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    signal.throwIfAborted();
    return new Float32Array(readback.getMappedRange(0, bytes)).slice();
  } finally {
    readback.unmap();
  }
}
