import { createInlineBudget, InlineCpu, inlineTiming } from '../src/inline-cpu.js';
import { prepareWorkload } from '../src/workload.js';
import type { ComputeProfile, ComputeRequest, ComputeResult } from '../src/types.js';

/** Same allocation, kernel and cooperative yields as adaptive CPU, with no device selection. */
export class DirectCpuReference {
  private readonly cpu = new InlineCpu();

  async compute(request: ComputeRequest, signal: AbortSignal): Promise<ComputeResult> {
    const runStarted = performance.now();
    signal.throwIfAborted();
    const workload = prepareWorkload(request);
    const length = workload.units;
    const output = new Float32Array(workload.outputLength);
    const budget = createInlineBudget();
    const setupMs = performance.now() - runStarted;
    const result = await this.cpu.run({ workload, destination: output, start: 0, length, runStarted, signal }, budget);
    signal.throwIfAborted();
    if (result.partial !== undefined) output[0] = result.partial;
    const profile: ComputeProfile = {
      operation: request.operation,
      requestedStrategy: 'cpu-only', strategy: 'cpu-only', elements: length, cpuElements: length, gpuElements: 0,
      totalMs: performance.now() - runStarted, setupMs, schedulingMs: 0, aggregationMs: 0,
      chunks: [result.profile], inlineTiming: inlineTiming(budget),
    };
    return { output, profile };
  }
}
