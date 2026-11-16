import { multiplyRange } from './vector-multiply.js';
import type { ComputeRequest, MatrixShape, ReductionOperation, VectorOperation } from './types.js';

export type Kernel = { operation: VectorOperation | ReductionOperation }
  | { operation: 'scalarMultiply'; scalar: number }
  | { operation: 'matrixMultiply'; shape: MatrixShape };

export interface Workload {
  kernel: Kernel;
  left: Float32Array;
  right?: Float32Array;
  units: number;
  outputLength: number;
  workPerUnit: number;
  reduction?: ReductionOperation;
}

export function prepareWorkload(request: ComputeRequest): Workload {
  const { operation, inputs } = request;
  const binary = operation === 'vectorMultiply' || operation === 'vectorAdd' || operation === 'matrixMultiply';
  const [left, right] = inputs;
  if (inputs.length !== (binary ? 2 : 1) || !(left instanceof Float32Array)
      || (binary && !(right instanceof Float32Array))) {
    throw new TypeError(`${operation} requires ${binary ? 'two' : 'one'} Float32Array input${binary ? 's' : ''}.`);
  }
  switch (operation) {
    case 'matrixMultiply': {
      const shape = { ...request.shape };
      const { rows, inner, columns } = shape;
      if (![rows, inner, columns, rows * inner, inner * columns, rows * columns].every((dim) => Number.isSafeInteger(dim) && dim >= 0)) {
        throw new RangeError('Matrix dimensions and their products must be nonnegative safe integers.');
      }
      if (left.length !== rows * inner || right!.length !== inner * columns) {
        throw new RangeError('Matrix input lengths must match rows × inner and inner × columns.');
      }
      return { kernel: { operation, shape }, left, right, units: rows && inner && columns ? rows : 0,
        outputLength: rows * columns, workPerUnit: Math.max(1, inner * columns) };
    }
    case 'vectorMultiply':
    case 'vectorAdd':
      if (left.length !== right!.length) throw new RangeError('Input vectors must have equal lengths.');
      return { kernel: { operation }, left, right, units: left.length, outputLength: left.length, workPerUnit: 1 };
    case 'scalarMultiply':
      if (typeof request.scalar !== 'number') throw new TypeError('scalar must be a number.');
      return { kernel: { operation, scalar: Math.fround(request.scalar) }, left, units: left.length, outputLength: left.length, workPerUnit: 1 };
    case 'sum':
    case 'min':
    case 'max':
      return { kernel: { operation }, left, units: left.length, outputLength: 1, workPerUnit: 1, reduction: operation };
    default:
      throw new TypeError(`Unsupported operation: ${operation}`);
  }
}

export function inputViews({ kernel, left, right }: Workload, start: number, length: number) {
  if (kernel.operation === 'matrixMultiply') {
    const { inner } = kernel.shape;
    return { left: left.subarray(start * inner, (start + length) * inner), right };
  }
  return { left: left.subarray(start, start + length), right: right?.subarray(start, start + length) };
}

export function outputSize(kernel: Kernel, units: number) {
  return kernel.operation === 'matrixMultiply' ? units * kernel.shape.columns : units;
}

export function reductionIdentity(operation: ReductionOperation) {
  return operation === 'sum' ? 0 : operation === 'min' ? Infinity : -Infinity;
}

export function foldReduction(operation: ReductionOperation, values: Iterable<number>) {
  let value = reductionIdentity(operation);
  for (const partial of values) value = combineReduction(operation, value, partial);
  return value;
}

export function combineReduction(operation: ReductionOperation, value: number, partial: number) {
  return operation === 'sum' ? value + partial : operation === 'min' ? Math.min(value, partial) : Math.max(value, partial);
}

export function transposeMatrixRight(right: Float32Array, { inner, columns }: MatrixShape) {
  const transposed = new Float32Array(right.length);
  for (let row = 0; row < inner; row++) {
    for (let col = 0; col < columns; col++) transposed[col * inner + row] = right[row * columns + col];
  }
  return transposed;
}

export function executeKernel(kernel: Kernel, left: Float32Array, right: Float32Array | undefined,
  output: Float32Array, start: number, end: number): number | undefined {
  switch (kernel.operation) {
    case 'vectorMultiply':
      multiplyRange(left, right!, output, start, end);
      return;
    case 'vectorAdd':
      for (let index = start; index < end; index++) output[index] = left[index] + right![index];
      return;
    case 'scalarMultiply':
      for (let index = start; index < end; index++) output[index] = left[index] * kernel.scalar;
      return;
    case 'sum': {
      let value = 0;
      for (let index = start; index < end; index++) value += left[index];
      return value;
    }
    case 'min': {
      let value = Infinity;
      for (let index = start; index < end; index++) value = Math.min(value, left[index]);
      return value;
    }
    case 'max': {
      let value = -Infinity;
      for (let index = start; index < end; index++) value = Math.max(value, left[index]);
      return value;
    }
    case 'matrixMultiply': {
      const { inner, columns } = kernel.shape;
      for (let row = start; row < end; row++) {
        const leftOffset = row * inner;
        for (let col = 0; col < columns; col++) {
          const rightOffset = col * inner;
          let value = 0;
          for (let k = 0; k < inner; k++) value += left[leftOffset + k] * right![rightOffset + k];
          output[row * columns + col] = value;
        }
      }
    }
  }
}
