import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist/site',
    rolldownOptions: { input: { engine: 'index.html', devices: 'device-benchmarks.html' } },
  },
});
