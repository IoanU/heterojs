import type { GpuGraphBuffer, GpuGraphBufferKind, GpuGraphBuilder, GpuGraphDispatch, GpuGraphKernel,
  GpuGraphRange, GpuGraphResult } from './types.js';

interface BufferNode {
  size: number;
  kind: GpuGraphBufferKind;
  data?: Uint8Array;
}

interface BufferRange {
  node: BufferNode;
  offset: number;
  size: number;
}

interface Pass {
  kernel: GpuGraphKernel;
  groups: ReadonlyMap<number, readonly (BufferRange & { binding: number })[]>;
  workgroups: readonly [number, number, number];
}

interface Readback extends BufferRange {
  destination: number;
}

export interface GpuGraphPlan {
  readonly buffers: readonly BufferNode[];
  readonly passes: readonly Pass[];
  readonly reads: readonly Readback[];
  readonly readbackBytes: number;
}

function aligned(value: number, minimum: number, name: string) {
  if (!Number.isSafeInteger(value) || value < minimum || value % 4 !== 0) {
    throw new RangeError(`${name} must be a ${minimum ? 'positive' : 'nonnegative'} multiple of four bytes.`);
  }
  return value;
}

export function buildGpuGraph(build: (graph: GpuGraphBuilder) => void): GpuGraphPlan {
  const buffers = new Map<GpuGraphBuffer, BufferNode>();
  const passes: Pass[] = [];
  const reads: Readback[] = [];
  let readbackBytes = 0;
  let open = true;

  const ensureOpen = () => {
    if (!open) throw new Error('This GPU graph has already been built.');
  };
  const addBuffer = (size: number, kind: GpuGraphBufferKind = 'storage', data?: Uint8Array) => {
    ensureOpen();
    aligned(size, 4, 'Buffer size');
    if (kind !== 'storage' && kind !== 'uniform') throw new TypeError('GPU graph buffers must be storage or uniform.');
    const handle = Object.freeze({ byteLength: size }) as GpuGraphBuffer;
    buffers.set(handle, { size, kind, data });
    return handle;
  };
  const resolveRange = (buffer: GpuGraphBuffer, { offset = 0, size }: GpuGraphRange): BufferRange => {
    const node = buffers.get(buffer);
    if (!node) throw new TypeError('The buffer must belong to this GPU graph.');
    aligned(offset, 0, 'Buffer offset');
    const length = aligned(size ?? node.size - offset, 4, 'Range size');
    if (offset > node.size - length) throw new RangeError('The range exceeds its GPU graph buffer.');
    return { node, offset, size: length };
  };

  const builder: GpuGraphBuilder = Object.freeze({
    upload(data: ArrayBuffer | ArrayBufferView, kind?: GpuGraphBufferKind) {
      const view = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : data instanceof ArrayBuffer ? new Uint8Array(data) : undefined;
      if (!view) throw new TypeError('GPU graph uploads require an ArrayBuffer or view.');
      return addBuffer(view.byteLength, kind, view);
    },
    createBuffer(size: number, kind?: GpuGraphBufferKind) { return addBuffer(size, kind); },
    dispatch({ kernel, bindings, workgroups }: GpuGraphDispatch) {
      ensureOpen();
      if (typeof kernel?.code !== 'string' || !kernel.code.trim()) throw new TypeError('A dispatch requires WGSL source.');
      if (kernel.entryPoint !== undefined && (typeof kernel.entryPoint !== 'string' || !kernel.entryPoint.trim())) {
        throw new TypeError('The WGSL entry point must be a nonempty string.');
      }
      const constants = Object.entries(kernel.constants ?? {}).sort(([a], [b]) => a.localeCompare(b));
      if (constants.some(([, value]) => !Number.isFinite(value))) throw new TypeError('WGSL overrides must be finite numbers.');
      if (!Array.isArray(workgroups) || workgroups.length < 1 || workgroups.length > 3) {
        throw new RangeError('Provide one to three workgroup counts.');
      }
      const [x, y = 1, z = 1] = workgroups;
      if (![x, y, z].every((n) => Number.isSafeInteger(n) && n > 0)) throw new RangeError('Workgroup counts must be positive integers.');
      const groups = new Map<number, (BufferRange & { binding: number })[]>();
      for (const { group = 0, binding, buffer, offset, size } of bindings) {
        if (![group, binding].every((n) => Number.isSafeInteger(n) && n >= 0)) {
          throw new RangeError('Groups and bindings must be nonnegative integers.');
        }
        const entries = groups.get(group) ?? [];
        if (entries.some((entry) => entry.binding === binding)) throw new TypeError('A bind group cannot contain duplicate bindings.');
        entries.push({ ...resolveRange(buffer, { offset, size }), binding });
        groups.set(group, entries);
      }
      passes.push({ kernel: { code: kernel.code, entryPoint: kernel.entryPoint, constants: Object.fromEntries(constants) },
        groups, workgroups: [x, y, z] });
    },
    read(buffer: GpuGraphBuffer, range: GpuGraphRange = {}) {
      ensureOpen();
      const source = resolveRange(buffer, range);
      const destination = readbackBytes;
      readbackBytes = aligned(readbackBytes + source.size, 4, 'Combined readback size');
      return reads.push({ ...source, destination }) - 1;
    },
  });
  try {
    const built: unknown = build(builder);
    if (typeof (built as PromiseLike<unknown> | null | undefined)?.then === 'function') {
      void Promise.resolve(built).catch(() => undefined);
      throw new TypeError('Build a GPU graph synchronously and schedule output with graph.read().');
    }
  } finally {
    open = false;
  }
  return { buffers: [...buffers.values()], passes, reads, readbackBytes };
}

export class GpuGraphExecutor {
  private readonly pipelines = new Map<string, Promise<GPUComputePipeline>>();

  constructor(private readonly device: GPUDevice) {}

  clear() { this.pipelines.clear(); }

  async run(graph: GpuGraphPlan, runStarted: number, queuedAt: number, signal: AbortSignal): Promise<GpuGraphResult> {
    const { device } = this;
    const started = performance.now();
    const { buffers, passes, reads, readbackBytes } = graph;
    if (readbackBytes > device.limits.maxBufferSize || buffers.some((node) => node.size > device.limits.maxBufferSize)) {
      throw new RangeError('A GPU graph buffer exceeds the device buffer size limit.');
    }
    if (passes.some((pass) => pass.workgroups.some((count) => count > device.limits.maxComputeWorkgroupsPerDimension))) {
      throw new RangeError('A dispatch exceeds the device workgroup count limit.');
    }
    const pipelines = await Promise.all(passes.map(({ kernel }) => this.pipelineFor(kernel)));
    signal.throwIfAborted();

    const resident = new Map<BufferNode, GPUBuffer>();
    let staging: GPUBuffer | undefined;
    const bufferFor = (node: BufferNode) => resident.get(node)!;
    try {
      for (const node of buffers) {
        const kind = node.kind === 'storage' ? GPUBufferUsage.STORAGE : GPUBufferUsage.UNIFORM;
        resident.set(node, device.createBuffer({ size: node.size, usage: kind | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
      }
      if (readbackBytes) staging = device.createBuffer({ size: readbackBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const uploadStarted = performance.now();
      let uploadBytes = 0;
      let uploads = 0;
      for (const node of buffers) {
        if (!node.data) continue;
        if (node.data.byteLength !== node.size) throw new RangeError('An upload changed size before the GPU graph ran.');
        device.queue.writeBuffer(bufferFor(node), 0, node.data.buffer, node.data.byteOffset, node.size);
        uploadBytes += node.size;
        uploads++;
      }

      const uploaded = performance.now();
      const submissions = passes.length || staging ? 1 : 0;
      if (submissions) {
        const encoder = device.createCommandEncoder();
        for (const [index, invocation] of passes.entries()) {
          const pipeline = pipelines[index];
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          for (const [group, bindings] of invocation.groups) {
            pass.setBindGroup(group, device.createBindGroup({ layout: pipeline.getBindGroupLayout(group),
              entries: bindings.map(({ binding, node, offset, size }) => ({ binding, resource: { buffer: bufferFor(node), offset, size } })),
            }));
          }
          pass.dispatchWorkgroups(...invocation.workgroups);
          pass.end();
        }
        if (staging) {
          for (const { node, offset, size, destination } of reads) {
            encoder.copyBufferToBuffer(bufferFor(node), offset, staging, destination, size);
          }
        }
        device.queue.submit([encoder.finish()]);
      }
      const submitted = performance.now();
      if (staging) await staging.mapAsync(GPUMapMode.READ);
      else if (submissions || uploads) await device.queue.onSubmittedWorkDone();
      const synchronized = performance.now();
      signal.throwIfAborted();
      const mapped = staging?.getMappedRange();
      const outputs = mapped ? reads.map(({ size, destination }) => mapped.slice(destination, destination + size)) : [];
      return { outputs, profile: {
        totalMs: 0, setupMs: queuedAt - runStarted + uploadStarted - started, queueMs: started - queuedAt,
        uploadMs: uploaded - uploadStarted, dispatchMs: submitted - uploaded, synchronizationMs: synchronized - submitted,
        readbackMs: performance.now() - synchronized, uploadBytes, readbackBytes, uploads, readbacks: reads.length,
        dispatches: passes.length, submissions,
      } };
    } finally {
      staging?.destroy();
      for (const buffer of resident.values()) buffer.destroy();
    }
  }

  private pipelineFor(kernel: GpuGraphKernel) {
    const key = JSON.stringify([kernel.code, kernel.entryPoint, kernel.constants]);
    const existing = this.pipelines.get(key);
    if (existing) {
      this.pipelines.delete(key);
      this.pipelines.set(key, existing);
      return existing;
    }
    const compilation = this.device.createComputePipelineAsync({ layout: 'auto', compute: {
      module: this.device.createShaderModule({ code: kernel.code }), entryPoint: kernel.entryPoint, constants: kernel.constants,
    } }).catch((error: unknown) => {
      if (this.pipelines.get(key) === compilation) this.pipelines.delete(key);
      throw error;
    });
    if (this.pipelines.size === 32) this.pipelines.delete(this.pipelines.keys().next().value!);
    this.pipelines.set(key, compilation);
    return compilation;
  }
}
