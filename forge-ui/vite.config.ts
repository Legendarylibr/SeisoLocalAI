import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    globals: true,
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765", "/health": "http://127.0.0.1:8765", "/v1": "http://127.0.0.1:8765" },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (
            id.includes("node_modules/react/") ||
            id.includes("node_modules/react-dom/") ||
            id.includes("node_modules/react-router/")
          ) {
            return "react";
          }
          if (id.includes("node_modules/@xyflow/")) {
            return "xyflow";
          }
        },
      },
    },
  },
});
