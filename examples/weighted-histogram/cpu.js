export function run({ keys, weights, binCount }) {
  const bins = new Uint32Array(binCount);
  for (let i = 0; i < keys.length; i++) bins[keys[i]] += weights[i];
  return bins;
}
