import { expect, test } from 'vitest';
import { CustomPlanner } from '../../src/custom-planner.js';

test('explores on idle calls and retains a measured split through transient outliers', () => {
  const planner = new CustomPlanner(100, 25);
  expect(planner.choose(false)).toEqual({ mode: 'static', cpuUnits: 25, exploratory: false });
  for (const [mode, cpuUnits, elapsed] of [['static', 25, 8], ['gpu-only', 0, 10], ['cpu-only', 100, 12],
    ['static', 13, 9], ['static', 50, 11]] as const) {
    const trial = planner.choose(true);
    expect(trial).toEqual({ mode, cpuUnits, exploratory: true });
    planner.observe(trial, elapsed, false);
  }
  const settled = { mode: 'static', cpuUnits: 25, exploratory: false } as const;
  expect(planner.choose(true)).toEqual(settled);
  planner.observe(settled, 7.9, false);
  planner.observe(settled, 50, false);
  for (let call = 1; call < 31; call++) expect(planner.choose(true)).toEqual(settled);
  expect(planner.choose(true)).toEqual({ mode: 'gpu-only', cpuUnits: 0, exploratory: true });
});

test('uses a single backend when it is measurably faster', () => {
  const planner = new CustomPlanner(100, 25);
  for (const elapsed of [16, 4, 8, 6, 10]) planner.observe(planner.choose(true), elapsed, false);
  expect(planner.choose(false)).toEqual({ mode: 'gpu-only', cpuUnits: 0, exploratory: false });
});

test('can learn a weighted adaptive dispatch alongside complete fixed plans', () => {
  const planner = new CustomPlanner(100, 25, true);
  for (const elapsed of [16, 4, 8, 7, 6, 10]) {
    const choice = planner.choose(true);
    // A weighted dispatch can use any actual CPU allocation. Its history belongs to its mode.
    planner.observe(choice.mode === 'adaptive-chunks' ? { ...choice, cpuUnits: 73 } : choice, elapsed, false);
  }
  expect(planner.choose(true)).toMatchObject({ mode: 'adaptive-chunks', exploratory: false });
});
