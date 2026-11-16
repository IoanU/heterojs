import type { ComputeProfile } from './types.js';

type Dispatch = NonNullable<ComputeProfile['customDispatch']>;
type Candidate = Omit<Dispatch, 'exploratory'>;

const RECHECK_INTERVAL = 32;
const HISTORY_SIZE = 7;

function candidateKey(candidate: Candidate) {
  return candidate.mode === 'adaptive-chunks' ? candidate.mode : candidate.cpuUnits;
}

export class CustomPlanner {
  private readonly times = new Map<number | 'adaptive-chunks', number[]>();
  private steadyCalls = 0;
  private rechecks = 0;

  private readonly candidates: readonly Candidate[];

  constructor(units: number, split: number, weighted = false) {
    const splits = new Set([split, 0, units]);
    if (split > 0 && split < units) {
      splits.add(Math.round(split / 2));
      splits.add(split + Math.min(split, Math.floor((units - split) / 2)));
    }
    const candidates = [...splits].map<Candidate>((cpuUnits) => ({ cpuUnits,
      mode: cpuUnits === 0 ? 'gpu-only' : cpuUnits === units ? 'cpu-only' : 'static' }));
    if (weighted) candidates.splice(1, 0, { mode: 'adaptive-chunks', cpuUnits: 0 });
    this.candidates = candidates;
  }

  choose(idle: boolean): Dispatch {
    if (idle) {
      const unmeasured = this.candidates.find((candidate) => !this.times.has(candidateKey(candidate)));
      if (unmeasured) return { ...unmeasured, exploratory: true };
    }

    const ranked = this.candidates.filter((candidate) => this.times.has(candidateKey(candidate)))
      .map((candidate) => ({ candidate, median: this.median(candidateKey(candidate)) })).sort((a, b) => a.median - b.median);
    const fastest = ranked[0];
    if (!fastest) return { ...this.candidates[0], exploratory: false };

    // Retain the configured split when a single device's lead is within timing noise.
    const preferred = ranked.find(({ candidate, median }) => candidate === this.candidates[0]
      && median <= fastest.median + Math.max(0.1, fastest.median * 0.02));
    const winner = preferred?.candidate ?? fastest.candidate;
    if (idle && ++this.steadyCalls % RECHECK_INTERVAL === 0) {
      const challengers = this.candidates.filter((candidate) => candidate !== winner);
      return { ...challengers[this.rechecks++ % challengers.length], exploratory: true };
    }
    return { ...winner, exploratory: false };
  }

  observe(candidate: Dispatch, elapsedMs: number, startedBackend: boolean) {
    if (startedBackend) return;
    const key = candidateKey(candidate);
    const history = this.times.get(key) ?? [];
    history.push(elapsedMs);
    if (history.length > HISTORY_SIZE) history.shift();
    this.times.set(key, history);
  }

  private median(key: ReturnType<typeof candidateKey>) {
    const history = [...this.times.get(key)!].sort((a, b) => a - b);
    return (history[history.length >> 1] + history[(history.length - 1) >> 1]) / 2;
  }
}
