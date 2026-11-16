export async function sort({ records, order }, range) {
  return {
    sorted: records.toSorted((a, b) => order === 'asc' ? a.score - b.score : b.score - a.score),
    values: new Map([['units', range.totalUnits]]),
    inWorker: typeof document === 'undefined',
  };
}

export async function ordered({ items }, { start }) {
  if (start === 0) await new Promise((resolve) => setTimeout(resolve, 250));
  return items.join('');
}

export function throwing() {
  throw new RangeError('Custom kernel failed.');
}

export function identity(value) {
  return value;
}

export function badOutput() {
  return () => 1;
}

export function transferRange(input, { start, length }) {
  const value = Uint32Array.from({ length }, (_, index) => input + start + index);
  return { value, transfer: [value.buffer] };
}

export function transferEcho(value) {
  return { value, transfer: [value.buffer] };
}

export async function stuck(url) {
  await fetch(url);
  return new Promise(() => {});
}
