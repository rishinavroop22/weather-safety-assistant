import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// In development the Vite server proxies /api and /health to FastAPI, so the browser
// talks to a single origin. Set VITE_API_BASE_URL instead to call the API directly
// (then FastAPI's FRONTEND_ORIGIN must allow this dev server's origin).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_DEV_PROXY_TARGET || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      proxy: { "/api": target, "/health": target },
    },
  };
});
