import type { MatrixDispatchProfile } from './types.js';

type Mode = MatrixDispatchProfile['mode'];

const modes = ['gpu', 'hybrid', 'cpu'] as const;
const targetSamples: Record<Mode, number> = { gpu: 2, hybrid: 2, cpu: 1 };
const HISTORY_SIZE = 7;
const RECHECK_INTERVAL = 24;

export class MatrixPlanner {
  private readonly samples: Record<Mode, number[]> = { cpu: [], gpu: [], hybrid: [] };
  private steadyCalls = 0;
  private readonly fringeRows: number;

  constructor(private readonly rows: number, private readonly workers: number) {
    // Keep the GPU range tile-aligned and give a single CPU worker at most two tiles of tail rows.
    const remainder = rows % 8;
    const fringe = rows >= 384 ? remainder + 8 : remainder || 8;
    this.fringeRows = fringe < rows ? fringe : 0;
  }

  choose(gpuAvailable: boolean, idle: boolean): MatrixDispatchProfile {
    if (!gpuAvailable) return this.plan('cpu', true);
    const eligible = this.fringeRows ? modes : ['gpu', 'cpu'] as const;
    const missing = eligible.find((mode) => this.samples[mode].length < targetSamples[mode]);
    if (idle && missing) return this.plan(missing, true);

    const cpu = this.predict('cpu');
    const gpu = this.predict('gpu');
    const hybrid = this.predict('hybrid');
    const single = Math.min(cpu, gpu);
    const winner: Mode = hybrid < single - Math.max(0.1, single * 0.02) ? 'hybrid' : cpu < gpu ? 'cpu' : 'gpu';
    if (idle && ++this.steadyCalls % RECHECK_INTERVAL === 0) {
      const winnerTime = this.predict(winner);
      const challenger = eligible.find((mode) => mode !== winner && this.predict(mode) <= winnerTime * 1.25);
      if (challenger) return this.plan(challenger, true);
    }
    return this.plan(winner, false);
  }

  observe(plan: MatrixDispatchProfile, elapsedMs: number, startedBackend: boolean) {
    if (startedBackend) return;
    const times = this.samples[plan.mode];
    times.push(elapsedMs);
    if (times.length > HISTORY_SIZE) times.shift();
  }

  private predict(mode: Mode) {
    const sorted = [...this.samples[mode]].sort((a, b) => a - b);
    if (!sorted.length) return Infinity;
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  private plan(mode: Mode, exploratory: boolean): MatrixDispatchProfile {
    const cpuRows = mode === 'cpu' ? this.rows : mode === 'hybrid' ? this.fringeRows : 0;
    return { mode, cpuRows, cpuWorkers: mode === 'cpu' ? this.workers : mode === 'hybrid' ? 1 : 0, exploratory };
  }
}
