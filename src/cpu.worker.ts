import type { BuiltInWorkerResponse, WorkerRequest } from './task.js';
import type { CustomCpuKernel, CustomCpuTransfer } from './types.js';
import { executeKernel, outputSize, transposeMatrixRight } from './workload.js';

const worker = self as unknown as DedicatedWorkerGlobalScope;
let matrixRight: Float32Array<ArrayBuffer> | undefined;

worker.onmessage = async ({ data }: MessageEvent<WorkerRequest>) => {
  if ('kind' in data) {
    try {
      const exports = await import(/* @vite-ignore */ data.moduleUrl);
      const run = exports[data.exportName] as CustomCpuKernel<unknown, unknown> | undefined;
      if (typeof run !== 'function') throw new TypeError(`CPU module has no function export named ${data.exportName}.`);
      const started = performance.now();
      const value = await run(data.input, data.range);
      const computeMs = performance.now() - started;
      if (data.transferOutput) {
        const result = value as CustomCpuTransfer<unknown> | undefined;
        if (!result || !Array.isArray(result.transfer) || !result.transfer.every((buffer) => buffer instanceof ArrayBuffer)) {
          throw new TypeError('Expected a custom CPU result with a value and an ArrayBuffer transfer list.');
        }
        const outputBytes = result.transfer.reduce((bytes, buffer) => bytes + buffer.byteLength, 0);
        worker.postMessage({ kind: 'custom', value: result.value, computeMs, outputBytes }, [...result.transfer]);
      } else worker.postMessage({ kind: 'custom', value, computeMs });
    } catch (error) {
      worker.postMessage({ kind: 'error', name: error instanceof Error ? error.name : 'Error',
        message: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  const { kernel, left, right, length } = data;
  const started = performance.now();
  if (kernel.operation === 'matrixMultiply') {
    if (right) matrixRight = transposeMatrixRight(right, kernel.shape);
  } else {
    matrixRight = undefined;
  }
  const output = kernel.operation === 'matrixMultiply' ? new Float32Array(outputSize(kernel, length)) : left;
  const partial = executeKernel(kernel, left, matrixRight ?? right, output, 0, length);
  const computeMs = performance.now() - started;
  const response: BuiltInWorkerResponse = partial === undefined ? { output, computeMs } : { partial, computeMs };
  worker.postMessage(response, response.output ? [response.output.buffer] : []);
};
