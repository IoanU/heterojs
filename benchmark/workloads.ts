import type { ComputeRequest, Operation } from '../src/index.js';

export const workloadLabels: Record<Operation, string> = {
  vectorMultiply: 'Vector multiply', vectorAdd: 'Vector add', scalarMultiply: 'Scalar multiply',
  sum: 'Sum reduction', min: 'Minimum reduction', max: 'Maximum reduction', matrixMultiply: 'Matrix multiply',
};

export function inputsFor(operation: Operation, size: number): { request: ComputeRequest; reference: Float32Array; tolerance: number } {
  const matrix = operation === 'matrixMultiply';
  const left = new Float32Array(matrix ? size * size : size);
  let state = 0x5eed1234;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 31 - 1;
  };
  const binary = matrix || operation === 'vectorMultiply' || operation === 'vectorAdd';
  const right = binary ? new Float32Array(left.length) : undefined;
  const quantized = matrix || operation === 'sum';
  for (let index = 0; index < left.length; index++) {
    const a = next();
    left[index] = quantized ? Math.round(a * 16) / 16 : a;
    if (right) {
      const b = next();
      right[index] = quantized ? Math.round(b * 16) / 16 : b;
    }
  }
  if (operation === 'matrixMultiply') {
    const reference = new Float32Array(left.length);
    for (let row = 0; row < size; row++) {
      for (let inner = 0; inner < size; inner++) {
        const a = left[row * size + inner];
        for (let col = 0; col < size; col++) reference[row * size + col] += a * right![inner * size + col];
      }
    }
    return { request: { operation, inputs: [left, right!], shape: { rows: size, inner: size, columns: size } }, reference, tolerance: 0 };
  }
  if (operation === 'sum' || operation === 'min' || operation === 'max') {
    let value = operation === 'sum' ? 0 : operation === 'min' ? Infinity : -Infinity;
    for (const item of left) value = operation === 'sum' ? value + item : operation === 'min' ? Math.min(value, item) : Math.max(value, item);
    return { request: { operation, inputs: [left] }, reference: Float32Array.of(value), tolerance: 0 };
  }
  const scalar = 0.625;
  const reference = left.map((value, index) => operation === 'scalarMultiply' ? value * scalar
    : operation === 'vectorAdd' ? value + right![index] : value * right![index]);
  const request = operation === 'scalarMultiply' ? { operation, inputs: [left] as const, scalar }
    : { operation, inputs: [left, right!] as const };
  return { request, reference, tolerance: 1e-7 };
}
