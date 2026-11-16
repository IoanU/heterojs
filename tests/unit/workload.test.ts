import { describe, expect, it } from 'vitest';
import { executeKernel, inputViews, prepareWorkload, reductionIdentity, transposeMatrixRight } from '../../src/workload.js';

describe('operation layouts and arithmetic', () => {
  it('prepares rectangular matrices as independent output rows with a shared right operand', () => {
    const left = Float32Array.of(1, 2, 3, 4, 5, 6);
    const right = Float32Array.of(7, 8, 9, 10, 11, 12);
    const shape = { rows: 2, inner: 3, columns: 2 };
    const work = prepareWorkload({ operation: 'matrixMultiply', shape, inputs: [left, right] });
    expect(work.units).toBe(2);
    expect(work.outputLength).toBe(4);
    const chunk = inputViews(work, 1, 1);
    expect([...chunk.left]).toEqual([4, 5, 6]);
    expect(chunk.right).toBe(right);
    const output = new Float32Array(2);
    executeKernel(work.kernel, chunk.left, transposeMatrixRight(right, shape), output, 0, 1);
    expect([...output]).toEqual([139, 154]);
    expect(() => prepareWorkload({ operation: 'matrixMultiply', inputs: [left, right], shape: { ...shape, inner: 2 } })).toThrow(RangeError);
    expect(() => prepareWorkload({ operation: 'matrixMultiply', inputs: [left, right], shape: { ...shape, rows: -1 } })).toThrow(RangeError);
  });

  it('keeps scalar f32 precision and defines empty reduction identities', () => {
    const work = prepareWorkload({ operation: 'scalarMultiply', inputs: [Float32Array.of(1, -2)], scalar: 1 / 3 });
    const output = new Float32Array(2);
    executeKernel(work.kernel, work.left, work.right, output, 0, 2);
    expect(output).toEqual(Float32Array.of(Math.fround(1 / 3), -2 * Math.fround(1 / 3)));
    expect(['sum', 'min', 'max'].map((operation) => reductionIdentity(operation as 'sum' | 'min' | 'max')))
      .toEqual([0, Infinity, -Infinity]);
  });
});
