/// <reference types="@webgpu/types" preserve="true" />

export type Strategy = 'cpu-only' | 'gpu-only' | 'static' | 'adaptive';
export type Backend = 'cpu' | 'gpu';

export interface EngineOptions {
  strategy?: Strategy;
  cpuShare?: number;
  cpuWorkers?: number | 'auto';
  gpu?: 'auto' | 'disabled';
  gpuDevice?: Pick<GPUDeviceDescriptor, 'requiredFeatures' | 'requiredLimits'>;
  /** Let adaptive execution use bounded CPU slices on the calling thread. Defaults to true. */
  inlineCpu?: boolean;
}

export type VectorOperation = 'vectorMultiply' | 'vectorAdd';
export type ReductionOperation = 'sum' | 'min' | 'max';
export type Operation = VectorOperation | ReductionOperation | 'scalarMultiply' | 'matrixMultiply';

export interface MatrixShape {
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
}

interface ComputeOptions {
  strategy?: Strategy;
  cpuShare?: number;
  signal?: AbortSignal;
  profile?: 'full' | 'summary';
}

export type ComputeRequest = ComputeOptions & (
  | { operation: VectorOperation; inputs: readonly [Float32Array, Float32Array] }
  | { operation: 'scalarMultiply'; inputs: readonly [Float32Array]; scalar: number }
  | { operation: ReductionOperation; inputs: readonly [Float32Array] }
  | { operation: 'matrixMultiply'; inputs: readonly [Float32Array, Float32Array]; shape: MatrixShape }
);

export interface ComputeResult {
  output: Float32Array<ArrayBuffer>;
  profile: ComputeProfile;
}

export interface CustomRange {
  readonly start: number;
  readonly length: number;
  readonly totalUnits: number;
}

export interface CustomGpuContext extends CustomRange {
  readonly device: GPUDevice;
  readonly signal: AbortSignal;
}

export type CustomCpuKernel<Input, Output> = (input: Input, range: CustomRange) => Output | Promise<Output>;
export type CustomGpuKernel<Input, Output> = (input: Input, context: CustomGpuContext) => Output | Promise<Output>;

export interface CustomCpuModule<Input = unknown> {
  readonly module: string | URL;
  /** Named module export. Defaults to `run`. */
  readonly exportName?: string;
  /** The worker export returns a CustomCpuTransfer and relinquishes those buffers. */
  readonly transferOutput?: boolean;
  /** Called when launching a CPU chunk, with buffers the caller gives up ownership of. */
  readonly transferInput?: (input: Input, range: CustomRange) => readonly ArrayBuffer[];
}

export interface CustomCpuTransfer<Value> {
  readonly value: Value;
  readonly transfer: readonly ArrayBuffer[];
}

export interface CustomPartial<Value> extends CustomRange {
  readonly value: Value;
}

interface CustomTiling<ChunkInput> {
  /** Number of independent work units. Zero skips dispatch and finalizes an empty result. */
  readonly units: number;
  readonly maxChunkUnits?: number;
  /** Workload class for adaptive selection of complete dispatch plans across calls. */
  readonly adaptiveKey?: string;
  /** Positive, additive relative cost of a nonempty range. Called during scheduling. */
  readonly estimateWork?: (range: CustomRange) => number;
  readonly chunk: (range: CustomRange) => ChunkInput;
}

export interface CustomPlan<ChunkInput, ChunkOutput, Output> extends CustomTiling<ChunkInput> {
  readonly combine: (parts: readonly CustomPartial<ChunkOutput>[], signal: AbortSignal) => Output | Promise<Output>;
  readonly collector?: never;
}

export interface StreamingCustomPlan<ChunkInput, ChunkOutput, Output> extends CustomTiling<ChunkInput> {
  readonly combine?: never;
  readonly collector: {
    /** Called serially, in completion order. A pending promise keeps the dispatch lane occupied. */
    readonly add: (part: CustomPartial<ChunkOutput>, signal: AbortSignal) => void | Promise<void>;
    readonly finish: (signal: AbortSignal) => Output | Promise<Output>;
  };
}

type CustomBackends<Input, Output> =
  | { cpu: CustomCpuModule<Input>; gpu?: CustomGpuKernel<Input, Output> }
  | { cpu?: CustomCpuModule<Input>; gpu: CustomGpuKernel<Input, Output> };

export type CustomOperationDefinition<Input, Output, ChunkInput = Input, ChunkOutput = Output> =
  | (CustomBackends<Input, Output> & { plan?: undefined; estimateUnits?: (input: Input) => number })
  | (CustomBackends<ChunkInput, ChunkOutput> & {
    plan: (input: Input) => CustomPlan<ChunkInput, ChunkOutput, Output> | StreamingCustomPlan<ChunkInput, ChunkOutput, Output>;
    estimateUnits?: never;
  });

declare const customSignature: unique symbol;

export interface CustomOperation<Input, Output> {
  readonly name: string;
  readonly [customSignature]: (input: Input) => Output;
}

export interface CustomComputeRequest<Input, Output> extends ComputeOptions {
  operation: CustomOperation<Input, Output>;
  input: NoInfer<Input>;
}

export interface CustomComputeResult<Output> {
  output: Output;
  profile: ComputeProfile;
}

export type GpuCapability =
  | { available: false; reason: string }
  | { available: true; timestampQueries: boolean; fallbackAdapter: boolean; vendor?: string; architecture?: string };

export interface Capabilities {
  cpuWorkers: number;
  gpu: GpuCapability;
}

declare const graphBufferBrand: unique symbol;

export interface GpuGraphBuffer {
  readonly byteLength: number;
  readonly [graphBufferBrand]: true;
}

export interface GpuGraphKernel {
  readonly code: string;
  readonly entryPoint?: string;
  readonly constants?: Readonly<Record<string, number>>;
}

export interface GpuGraphRange {
  readonly offset?: number;
  readonly size?: number;
}

export interface GpuGraphBinding extends GpuGraphRange {
  readonly group?: number;
  readonly binding: number;
  readonly buffer: GpuGraphBuffer;
}

export interface GpuGraphDispatch {
  readonly kernel: GpuGraphKernel;
  readonly bindings: readonly GpuGraphBinding[];
  readonly workgroups: readonly [x: number, y?: number, z?: number];
}

export type GpuGraphBufferKind = 'storage' | 'uniform';

export interface GpuGraphBuilder {
  upload(data: ArrayBuffer | ArrayBufferView, kind?: GpuGraphBufferKind): GpuGraphBuffer;
  createBuffer(byteLength: number, kind?: GpuGraphBufferKind): GpuGraphBuffer;
  dispatch(dispatch: GpuGraphDispatch): void;
  /** Schedules a read after all dispatches and returns its index in result.outputs. */
  read(buffer: GpuGraphBuffer, range?: GpuGraphRange): number;
}

export interface GpuGraphProfile {
  totalMs: number;
  setupMs: number;
  queueMs: number;
  /** Host time spent issuing WebGPU buffer writes. */
  uploadMs: number;
  /** Host time spent encoding and submitting commands. */
  dispatchMs: number;
  /** Wall time awaiting GPU completion and mapping, which can overlap GPU compute. */
  synchronizationMs: number;
  /** Host time spent copying mapped results to independent ArrayBuffers. */
  readbackMs: number;
  uploadBytes: number;
  readbackBytes: number;
  uploads: number;
  readbacks: number;
  dispatches: number;
  submissions: number;
}

export interface GpuGraphResult {
  outputs: readonly ArrayBuffer[];
  profile: GpuGraphProfile;
}

export interface ChunkProfile {
  backend: Backend;
  cpuExecution?: 'worker' | 'inline';
  /** First chunk sent to a newly created worker. */
  workerStartup?: boolean;
  /** First chunk compiling this operation's WebGPU pipeline. */
  gpuPipelineStartup?: boolean;
  /** Inclusive scheduled offset, in rows for matrix multiplication. */
  start: number;
  length: number;
  /** Offset from the start of the compute call. */
  startedAtMs: number;
  elapsedMs: number;
  queueMs: number;
  dispatchMs: number;
  /** CPU worker loop or GPU timestamp duration, where available. */
  computeMs: number | null;
  /** CPU-side copies and API calls for inputs and output. */
  inputTransferMs: number;
  outputTransferMs: number;
  inputBytes: number;
  outputBytes: number;
  /** GPU queue and readback completion, including their overlap with GPU compute. */
  synchronizationMs: number | null;
  /** Useful work sent to refresh the adaptive model. */
  exploratory?: boolean;
}

export interface ChunkTotals {
  chunks: number;
  elements: number;
  inputBytes: number;
  outputBytes: number;
  inputTransferMs: number;
  outputTransferMs: number;
  /** Sum across chunks. Parallel work can overlap. */
  elapsedMs: number;
  queueMs: number;
}

export interface AdaptiveEstimate {
  samples: number;
  fixedMs: number;
  msPerElement: number;
}

export interface InlineTiming {
  /** Longest sum of inline kernel time between cooperative yields. */
  longestTurnComputeMs: number;
  yields: number;
  yieldWaitMs: number;
}

export interface MatrixDispatchProfile {
  mode: 'cpu' | 'gpu' | 'hybrid';
  cpuRows: number;
  cpuWorkers: number;
  exploratory: boolean;
}

export interface ComputeProfile {
  operation?: Operation;
  customOperation?: string;
  customDispatch?: { mode: 'cpu-only' | 'gpu-only' | 'static' | 'adaptive-chunks'; cpuUnits: number; exploratory: boolean };
  requestedStrategy: Strategy;
  strategy: Strategy;
  fallbackReason?: string;
  /** Vector/reduction elements, matrix rows, or the operation's own custom units. */
  elements: number;
  cpuElements: number;
  gpuElements: number;
  totalMs: number;
  setupMs: number;
  schedulingMs: number;
  aggregationMs: number;
  chunks: readonly ChunkProfile[];
  /** Aggregate counters returned with profile: 'summary'. */
  chunkSummary?: Readonly<Record<Backend, ChunkTotals>>;
  /** Recent service-time estimates for this engine, when available. */
  estimates?: Partial<Record<Backend | 'inline', AdaptiveEstimate>>;
  /** `msPerElement` uses the operation's estimated work units when set to `work`. */
  estimateBasis?: 'work';
  inlineTiming?: InlineTiming;
  matrixDispatch?: MatrixDispatchProfile;
}
