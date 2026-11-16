import { describe, expect, it } from 'vitest';
import { MatrixPlanner } from '../../src/matrix-planner.js';

function train(hybridMs: number, cpuMs = 32) {
  const planner = new MatrixPlanner(513, 8);
  const results = [6.2, 6, hybridMs, hybridMs, cpuMs];
  const plans = results.map((elapsed) => {
    const plan = planner.choose(true, true);
    planner.observe(plan, elapsed, false);
    return plan;
  });
  return { planner, plans };
}

describe('matrix dispatch planning', () => {
  it('uses a small tile-aligned split only if it beats both measured single paths', () => {
    const { planner, plans } = train(5.3);
    expect(plans.map((plan) => plan.mode)).toEqual(['gpu', 'gpu', 'hybrid', 'hybrid', 'cpu']);
    const chosen = planner.choose(true, true);
    expect(chosen).toEqual({ mode: 'hybrid', cpuRows: 9, cpuWorkers: 1, exploratory: false });
    expect((513 - chosen.cpuRows) % 8).toBe(0);
  });

  it('uses the fastest verified single backend when the split is slower or tied', () => {
    expect(train(6.1).planner.choose(true, true).mode).toBe('gpu');
    expect(train(6.9, 4).planner.choose(true, true).mode).toBe('cpu');
  });

  it('never learns a cold pipeline measurement as warm performance and can fall back without a GPU', () => {
    const planner = new MatrixPlanner(513, 8);
    planner.observe(planner.choose(true, true), 90, true);
    planner.observe(planner.choose(true, true), 6, false);
    expect(planner.choose(true, true).mode).toBe('gpu');
    expect(planner.choose(false, true)).toMatchObject({ mode: 'cpu', cpuRows: 513 });
  });
});
