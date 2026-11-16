const workgroupSize = 256;

const multiply = {
  code: `
    @group(0) @binding(0) var<storage, read> a: array<u32>;
    @group(0) @binding(1) var<storage, read> b: array<u32>;
    @group(0) @binding(2) var<storage, read_write> output: array<u32>;

    @compute @workgroup_size(${workgroupSize})
    fn main(@builtin(global_invocation_id) id: vec3u) {
      if id.x < arrayLength(&output) { output[id.x] = a[id.x] * b[id.x]; }
    }
  `,
};

const activate = {
  code: `
    @group(0) @binding(0) var<storage, read> values: array<u32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;

    @compute @workgroup_size(${workgroupSize})
    fn main(@builtin(global_invocation_id) id: vec3u) {
      if id.x < arrayLength(&output) { output[id.x] = max(values[id.x], 16u) - 16u; }
    }
  `,
};

const sum = {
  code: `
    @group(0) @binding(0) var<storage, read> values: array<u32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    var<workgroup> partials: array<u32, ${workgroupSize}>;

    @compute @workgroup_size(${workgroupSize})
    fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) i: u32,
        @builtin(workgroup_id) group: vec3u) {
      partials[i] = 0u;
      if id.x < arrayLength(&values) { partials[i] = values[id.x]; }
      workgroupBarrier();
      for (var stride = ${workgroupSize / 2}u; stride > 0u; stride >>= 1u) {
        if i < stride { partials[i] += partials[i + stride]; }
        workgroupBarrier();
      }
      if i == 0u { output[group.x] = partials[0]; }
    }
  `,
};

function dispatch(graph, kernel, buffers, count) {
  graph.dispatch({ kernel, bindings: buffers.map((buffer, binding) => ({ binding, buffer })), workgroups: [count] });
}

function multiplyInto(graph, a, b) {
  const output = graph.createBuffer(a.byteLength);
  dispatch(graph, multiply, [a, b, output], Math.ceil(a.byteLength / 4 / workgroupSize));
  return output;
}

function activateInto(graph, input) {
  const output = graph.createBuffer(input.byteLength);
  dispatch(graph, activate, [input, output], Math.ceil(input.byteLength / 4 / workgroupSize));
  return output;
}

function reduceInto(graph, input) {
  while (input.byteLength > 4) {
    const groups = Math.ceil(input.byteLength / 4 / workgroupSize);
    const output = graph.createBuffer(groups * 4);
    dispatch(graph, sum, [input, output], groups);
    input = output;
  }
  return input;
}

export function makeInputs(length) {
  const a = Uint32Array.from({ length }, (_, index) => index % 23);
  const b = Uint32Array.from({ length }, (_, index) => index % 13);
  let expected = 0;
  for (let i = 0; i < length; i++) expected = (expected + Math.max(a[i] * b[i] - 16, 0)) >>> 0;
  return { a, b, expected };
}

export async function runResident(engine, a, b) {
  const { outputs, profile } = await engine.runGpuGraph((graph) => {
    const product = multiplyInto(graph, graph.upload(a), graph.upload(b));
    const activated = activateInto(graph, product);
    graph.read(reduceInto(graph, activated));
  });
  return { value: new Uint32Array(outputs[0])[0], profiles: [profile] };
}

export async function runStaged(engine, a, b) {
  const product = await engine.runGpuGraph((graph) => graph.read(multiplyInto(graph, graph.upload(a), graph.upload(b))));
  const activated = await engine.runGpuGraph((graph) => graph.read(activateInto(graph, graph.upload(product.outputs[0]))));
  const sum = await engine.runGpuGraph((graph) => graph.read(reduceInto(graph, graph.upload(activated.outputs[0]))));
  return { value: new Uint32Array(sum.outputs[0])[0], profiles: [product.profile, activated.profile, sum.profile] };
}
