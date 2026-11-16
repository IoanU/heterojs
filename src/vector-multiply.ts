export function multiplyRange(left: Float32Array, right: Float32Array, output: Float32Array, start: number, end: number) {
  for (let index = start; index < end; index++) output[index] = left[index] * right[index];
}
