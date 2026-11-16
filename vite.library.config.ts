import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'dist/library',
    lib: { entry: 'src/index.ts', formats: ['es'], fileName: () => 'index.js' },
  },
});
