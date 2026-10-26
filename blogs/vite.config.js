import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Local only: the data is real dataset rows (data/ is git-ignored).
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
