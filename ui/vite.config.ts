import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// `/ui-next/` is the mount path on the FastAPI side. `base` makes Vite
// emit asset URLs relative to that prefix so the static bundle works
// behind the FastAPI mount as well as on `vite preview`.
const UI_BASE = "/ui-next/";

export default defineConfig({
  plugins: [svelte()],
  base: UI_BASE,
  server: {
    port: 5173,
    proxy: {
      // During `npm run dev`, forward backend calls to the running
      // FastAPI server on :3333. Avoids CORS plumbing and keeps the
      // dev experience identical to production (same paths).
      "/ui/status": "http://127.0.0.1:3333",
      "/ui/document-preview": "http://127.0.0.1:3333",
      "/api": "http://127.0.0.1:3333",
      "/mcp": "http://127.0.0.1:3333",
      "/health": "http://127.0.0.1:3333",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
});
