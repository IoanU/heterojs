import type { Backend, ChunkProfile, ChunkTotals } from './types.js';

function empty(): ChunkTotals {
  return { chunks: 0, elements: 0, inputBytes: 0, outputBytes: 0,
    inputTransferMs: 0, outputTransferMs: 0, elapsedMs: 0, queueMs: 0 };
}

export class ChunkRecorder {
  readonly chunks: ChunkProfile[] = [];
  readonly summary?: Record<Backend, ChunkTotals>;
  cpuElements = 0;
  startedBackend = false;
  usedInline = false;

  constructor(summary: boolean) {
    if (summary) this.summary = { cpu: empty(), gpu: empty() };
  }

  add(chunk: ChunkProfile) {
    if (chunk.backend === 'cpu') this.cpuElements += chunk.length;
    this.startedBackend ||= Boolean(chunk.workerStartup || chunk.gpuPipelineStartup);
    this.usedInline ||= chunk.cpuExecution === 'inline';
    const summary = this.summary?.[chunk.backend];
    if (summary) {
      summary.chunks++;
      summary.elements += chunk.length;
      summary.elapsedMs += chunk.elapsedMs;
      summary.queueMs += chunk.queueMs;
      summary.inputBytes += chunk.inputBytes;
      summary.outputBytes += chunk.outputBytes;
      summary.inputTransferMs += chunk.inputTransferMs;
      summary.outputTransferMs += chunk.outputTransferMs;
    } else this.chunks.push(chunk);
  }

  ordered() {
    return this.chunks.sort((a, b) => a.start - b.start);
  }
}
