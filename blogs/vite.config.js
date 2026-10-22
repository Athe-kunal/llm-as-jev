import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Local only: the data is real dataset rows (data/ is git-ignored).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    // the "Try it" section's model API (blogs/server.py, npm run api)
    proxy: { "/api": "http://127.0.0.1:8767" },
  },
});
