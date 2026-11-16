import { escapeFloat } from './math.js';

export function raster(input, { start, length }) {
  const { width, height, passWidth, step, iterations, cx, cy, span, type, jr, ji } = input;
  const values = new Float64Array(length);
  let x = start % passWidth;
  let y = Math.floor(start / passWidth);
  let imag = cy + ((y + .5) * step - height / 2) / width * span;
  for (let i = 0; i < length; i++) {
    const real = cx + ((x + .5) * step / width - .5) * span;
    values[i] = escapeFloat(real, imag, iterations, type, jr, ji);
    if (++x === passWidth) {
      x = 0;
      imag = cy + ((++y + .5) * step - height / 2) / width * span;
    }
  }
  return values;
}

export function run(input, range) {
  const value = raster(input, range);
  return { value, transfer: [value.buffer] };
}
