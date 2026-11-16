import type { ChunkProfile, CustomCpuModule, CustomRange } from './types.js';
import type { Kernel, Workload } from './workload.js';

export interface ChunkTask {
  readonly workload: Workload;
  readonly destination: Float32Array<ArrayBuffer>;
  readonly start: number;
  readonly length: number;
  readonly runStarted: number;
  readonly signal: AbortSignal;
  readonly onGpuSubmitted?: () => void;
}

export interface ChunkResult {
  output?: Float32Array<ArrayBuffer>;
  partial?: number;
  profile: ChunkProfile;
}

export interface BuiltInWorkerRequest {
  kernel: Kernel;
  left: Float32Array<ArrayBuffer>;
  right?: Float32Array<ArrayBuffer>;
  length: number;
}

export interface BuiltInWorkerResponse {
  output?: Float32Array<ArrayBuffer>;
  partial?: number;
  computeMs: number;
}

export interface CustomChunkTask extends CustomRange {
  readonly moduleUrl: string;
  readonly exportName: string;
  readonly transferOutput: boolean;
  readonly transferInput?: CustomCpuModule['transferInput'];
  readonly input: unknown;
  readonly runStarted: number;
  readonly signal: AbortSignal;
}

export type WorkerRequest = BuiltInWorkerRequest | {
  kind: 'custom';
  moduleUrl: string;
  exportName: string;
  transferOutput: boolean;
  input: unknown;
  range: CustomRange;
};

export type WorkerResponse = BuiltInWorkerResponse
  | { kind: 'custom'; value: unknown; computeMs: number; outputBytes?: number }
  | { kind: 'error'; name: string; message: string };
