export function computeRange({ heavyStart, heavyLength, light, heavy, seed }, { start, length }) {
  const result = new Uint32Array(length);
  for (let i = 0; i < length; i++) {
    const position = start + i;
    const steps = position >= heavyStart && position < heavyStart + heavyLength ? heavy : light;
    let value = (position ^ seed) >>> 0;
    for (let j = 0; j < steps; j++) value = (Math.imul(value ^ (value >>> 13), 1664525) + 1013904223) >>> 0;
    result[i] = value;
  }
  return result;
}

export function run(input, range) {
  const value = computeRange(input, range);
  return { value, transfer: [value.buffer] };
}
