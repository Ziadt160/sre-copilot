import { defineConfig } from "@playwright/test";

// End to end against the real local stack (not CI): the web UI on :3000 built with the real
// API and Supabase URLs, the core API on :8000, mcp-ops on :8100 (simulator), local Supabase
// Auth, Redis. A test login comes from E2E_EMAIL / E2E_PASSWORD. Models are whatever the API
// runs on (the lab config uses the local 7B model), so steps get minutes, not seconds.
export default defineConfig({
  testDir: "e2e-live",
  timeout: 300_000,
  expect: { timeout: 180_000 },
  workers: 1,
  reporter: "list",
  use: { baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000", trace: "retain-on-failure" },
});
