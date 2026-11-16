import { writeFile } from 'node:fs/promises';
import { cpus, platform, release } from 'node:os';
import { parseArgs } from 'node:util';
import { chromium } from '@playwright/test';
import { preview } from 'vite';

const { values: args } = parseArgs({ options: {
  channel: { type: 'string', default: 'chromium' },
  sizes: { type: 'string', default: '4096,262144,1048576,4194304' },
  series: { type: 'string', default: '4' },
  warmups: { type: 'string', default: '8' },
  samples: { type: 'string', default: '24' },
  output: { type: 'string', default: 'gpu-graph-results.json' },
} });

function integer(text, minimum, maximum, name) {
  const value = Number(text);
  if (!/^\d+$/.test(text) || !Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer between ${minimum} and ${maximum}.`);
  }
  return value;
}

const sizes = [...new Set(args.sizes.split(',').map((size) => integer(size.trim(), 1, 4_194_304, 'size')))];
const series = integer(args.series, 1, 20, 'series');
const warmups = integer(args.warmups, 0, 100, 'warmups');
const samples = integer(args.samples, 1, 100, 'samples');
const library = '/__gpu-graph-library.js';
const example = '/__gpu-graph-example.js';

function statistics(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return {
    median: (sorted[middle] + sorted[(sorted.length - 1) >> 1]) / 2,
    p95: sorted[Math.ceil(sorted.length * .95) - 1],
  };
}

const server = await preview({ preview: { host: '127.0.0.1', port: 0 } });
let browser;
try {
  browser = await chromium.launch({ channel: args.channel });
  const address = `http://127.0.0.1:${server.httpServer.address().port}/`;
  const runs = [];
  for (let index = 0; index < series; index++) {
    const page = await browser.newPage();
    try {
      await page.route(`**${library}`, (route) => route.fulfill({ path: 'dist/library/index.js', contentType: 'text/javascript' }));
      await page.route(`**${example}`, (route) => route.fulfill({ path: 'examples/gpu-graph/index.js', contentType: 'text/javascript' }));
      await page.goto(address);
      const report = await page.evaluate(async ({ index, sizes, warmups, samples, library, example }) => {
        const { HeteroJS } = await import(library);
        const { makeInputs, runResident, runStaged } = await import(example);
        const rows = [];
        let gpu;
        for (const [sizeIndex, length] of sizes.entries()) {
          const { a, b, expected } = makeInputs(length);
          const engines = { resident: await HeteroJS.create(), staged: await HeteroJS.create() };
          const calls = { resident: runResident, staged: runStaged };
          const runs = { resident: [], staged: [] };
          try {
            for (let sample = 0; sample < 1 + warmups + samples; sample++) {
              const order = (index + sizeIndex + sample) % 2 ? ['staged', 'resident'] : ['resident', 'staged'];
              for (const mode of order) {
                const started = performance.now();
                const { value, profiles } = await calls[mode](engines[mode], a, b);
                const elapsedMs = performance.now() - started;
                if (value !== expected) throw new Error(`${mode} produced ${value}, expected ${expected}, at ${length} elements.`);
                const sum = (key) => profiles.reduce((total, profile) => total + profile[key], 0);
                const uploadBytes = sum('uploadBytes');
                const readbackBytes = sum('readbackBytes');
                const submissions = sum('submissions');
                const baseline = mode === 'staged';
                if (uploadBytes !== length * (baseline ? 16 : 8) || readbackBytes !== (baseline ? length * 8 : 0) + 4
                    || submissions !== (baseline ? 3 : 1)) throw new Error(`Unexpected ${mode} transfer profile at ${length}.`);
                runs[mode].push({ elapsedMs, uploadBytes, readbackBytes, submissions, profiles });
              }
            }
            gpu ??= (await engines.resident.getCapabilities()).gpu;
            rows.push({ length, expected, modes: Object.fromEntries(Object.entries(runs).map(([mode, [cold, ...rest]]) =>
              [mode, { cold, warmups: rest.slice(0, warmups), timed: rest.slice(warmups) }])) });
          } finally {
            engines.resident.close();
            engines.staged.close();
          }
        }
        return { gpu, rows };
      }, { index, sizes, warmups, samples, library, example });
      runs.push(report);
      console.log(`GPU graph series ${index + 1}/${series} completed`);
    } finally { await page.close(); }
  }
  const summary = sizes.map((length) => {
    const rows = runs.map(({ rows }) => rows.find((row) => row.length === length));
    const modes = Object.fromEntries(['resident', 'staged'].map((mode) => {
      const measurements = rows.flatMap((row) => row.modes[mode].timed);
      const first = measurements[0];
      return [mode, { samples: measurements.length, elapsedMs: statistics(measurements.map((sample) => sample.elapsedMs)),
        coldMs: statistics(rows.map((row) => row.modes[mode].cold.elapsedMs)), uploadBytes: first.uploadBytes,
        readbackBytes: first.readbackBytes, submissions: first.submissions }];
    }));
    return { length, ...modes };
  });
  const metadata = { timestamp: new Date().toISOString(), browser: { channel: args.channel, version: browser.version() },
    cpu: cpus()[0]?.model.trim(), platform: platform(), osRelease: release(), series, warmups, samples, gpu: runs[0].gpu };
  await writeFile(args.output, JSON.stringify({ metadata, summary, runs }, null, 2) + '\n');
} finally {
  try { await browser?.close(); } finally { await new Promise((resolve) => server.httpServer.close(resolve)); }
}
