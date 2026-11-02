import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The data/*.json files are tracked (the CI build has no data step).
// GitHub Pages serves the site at /<repo>/; local dev and preview stay at /.
export default defineConfig({
  base: process.env.BASE_PATH || "/",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    // the "Try it" section's model API (blogs/server.py, npm run api)
    proxy: { "/api": "http://127.0.0.1:8767" },
  },
});
