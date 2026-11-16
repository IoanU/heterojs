const pipelines = new WeakMap();

const shader = `
  @group(0) @binding(0) var<storage, read> keys: array<u32>;
  @group(0) @binding(1) var<storage, read> weights: array<u32>;
  @group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

  @compute @workgroup_size(64)
  fn histogram(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < arrayLength(&keys)) {
      atomicAdd(&bins[keys[id.x]], weights[id.x]);
    }
  }
`;

function pipelineFor(device) {
  let pipeline = pipelines.get(device);
  if (!pipeline) {
    pipeline = device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: 'histogram' },
    }).catch((error) => {
      pipelines.delete(device);
      throw error;
    });
    pipelines.set(device, pipeline);
  }
  return pipeline;
}

export async function histogramGpu({ keys, weights, binCount }, { device, signal }) {
  const pipeline = await pipelineFor(device);
  signal.throwIfAborted();
  const buffers = [];
  const allocate = (size, usage) => {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer);
    return buffer;
  };
  try {
    const inputUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const keyBuffer = allocate(keys.byteLength, inputUsage);
    const weightBuffer = allocate(weights.byteLength, inputUsage);
    const binBuffer = allocate(binCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = allocate(binCount * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(keyBuffer, 0, keys);
    device.queue.writeBuffer(weightBuffer, 0, weights);
    const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [keyBuffer, weightBuffer, binBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(Math.ceil(keys.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(binBuffer, 0, readback, 0, binCount * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    signal.throwIfAborted();
    return new Uint32Array(readback.getMappedRange()).slice();
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}
