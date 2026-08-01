import { defineConfig } from 'astro/config';
import { integration } from '@czap/astro';

const dir = (path: string) => decodeURIComponent(new URL(path, import.meta.url).pathname);

export default defineConfig({
  vite: {
    server: {
      proxy: {
        '/api': 'http://127.0.0.1:8000',
      },
    },
  },
  integrations: [
    integration({
      detect: true,
      // Optional generated UI: `pnpm add @czap/genui`, define a catalog with
      // `defineComponentCatalog`, pass `genuiCatalog` to createLLMSession (or
      // set `data-czap-genui` on `client:llm`). See GETTING-STARTED.md.
      vite: {
        dirs: {
          boundary: dir('./src/boundaries'),
          token: dir('./src/tokens'),
        },
      },
      gpu: { enabled: false },
      llm: { enabled: false },
      stream: { enabled: true },
      inspector: false,
    }),
  ],
});
