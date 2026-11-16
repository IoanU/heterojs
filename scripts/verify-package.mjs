import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { chromium } from '@playwright/test';

const npm = process.env.npm_execpath;
if (!npm) throw new Error('Run this check with npm run test:package.');
const testRoot = resolve(tmpdir());
const scratch = await mkdtemp(join(testRoot, 'heterojs-consumer-'));
const command = (script, args, options = {}) => execFileSync(process.execPath, [script, ...args], { encoding: 'utf8', ...options });
let browser;
let server;
try {
  const [pack] = JSON.parse(command(npm, ['pack', '--ignore-scripts', '--json', '--pack-destination', scratch]));
  const files = new Set(pack.files.map(({ path }) => path));
  for (const file of ['README.md', 'LICENSE', 'dist/library/index.js', 'dist/library/types/index.d.ts', 'examples/gpu-graph/index.js']) {
    assert(files.has(file), `Package is missing ${file}.`);
  }
  assert(![...files].some((file) => /^(src|tests|node_modules|\.github)\//.test(file)));
  await writeFile(join(scratch, 'package.json'), JSON.stringify({ name: 'heterojs-consumer-test', private: true, type: 'module' }));
  command(npm, ['install', '--no-audit', '--no-fund', '--ignore-scripts', `./${pack.filename}`], { cwd: scratch });
  await writeFile(join(scratch, 'consumer.ts'), `
    import { HeteroJS, GpuDeviceLostError } from 'heterojs';
    import type { CustomCpuTransfer, StreamingCustomPlan, GpuGraphResult } from 'heterojs';
    const engine = await HeteroJS.create({ gpu: 'disabled' });
    const computed = await engine.compute({ operation: 'sum', inputs: [Float32Array.of(5)], profile: 'summary' });
    const result: Float32Array<ArrayBuffer> = computed.output;
    const operation = engine.registerOperation<number, Uint32Array, Uint32Array<ArrayBuffer>, Uint32Array>('stream', {
      cpu: { module: '/worker.js', transferOutput: true, transferInput: (view: Uint32Array<ArrayBuffer>) => [view.buffer] },
      plan(units): StreamingCustomPlan<Uint32Array<ArrayBuffer>, Uint32Array, Uint32Array> {
        const values = new Uint32Array(units);
        return { units, estimateWork: ({ length }) => length, chunk: ({ length }) => new Uint32Array(length),
          collector: { async add({ start, value }) { values.set(value, start); }, finish: () => values } };
      },
    });
    const custom: Uint32Array = (await engine.compute({ operation, input: 4 })).output;
    const graph: Promise<GpuGraphResult> = engine.runGpuGraph((builder) => builder.read(builder.upload(Uint32Array.of(1))));
    const transfer: CustomCpuTransfer<Float32Array> = { value: result, transfer: [result.buffer] };
    void [transfer, graph, GpuDeviceLostError, custom];
  `);
  command(resolve('node_modules/typescript/bin/tsc'), ['--noEmit', '--strict', '--target', 'es2022', '--lib', 'es2023,dom',
    '--module', 'esnext', '--moduleResolution', 'bundler', 'consumer.ts'], { cwd: scratch });

  const bundle = await readFile(join(scratch, 'node_modules/heterojs/dist/library/index.js'));
  server = createServer((request, response) => {
    if (request.url === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Package check</title>'); }
    else if (request.url === '/library.js') { response.setHeader('Content-Type', 'text/javascript'); response.end(bundle); }
    else if (request.url === '/worker.js') {
      response.setHeader('Content-Type', 'text/javascript'); response.end('export function run(value) { return value * 3; }');
    } else { response.writeHead(404); response.end(); }
  });
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  browser = await chromium.launch({ channel: process.env.HETEROJS_BROWSER_CHANNEL || 'chromium' });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async () => {
    const { HeteroJS } = await import('/library.js');
    const engine = await HeteroJS.create({ gpu: 'disabled', cpuWorkers: 1 });
    try {
      const operation = engine.registerOperation('triple', { cpu: { module: '/worker.js' } });
      const builtIn = await engine.compute({ operation: 'sum', inputs: [Float32Array.of(1, 2, 3)], profile: 'summary' });
      const custom = await engine.compute({ operation, input: 11 });
      return { builtIn: builtIn.output[0], summary: builtIn.profile.chunkSummary.cpu.elements, custom: custom.output };
    } finally { engine.close(); }
  });
  assert.deepEqual(result, { builtIn: 6, summary: 3, custom: 33 });
  console.log(`Installed, type-checked and exercised ${pack.filename} in a clean consumer.`);
} finally {
  try { await browser?.close(); }
  finally {
    try { if (server) await new Promise((done) => server.close(done)); }
    finally {
      if (!scratch.startsWith(`${testRoot}${sep}heterojs-consumer-`)) throw new Error('Unexpected consumer test directory.');
      await rm(scratch, { recursive: true, force: true });
    }
  }
}
