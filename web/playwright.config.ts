import { defineConfig } from "@playwright/test";

// The smoke test runs the built app against the mock API (mock/server.mjs), which also
// answers the Supabase password login the UI uses. Build first with the NEXT_PUBLIC_* URLs
// pointing at the mock (see .github/workflows/web.yml).
export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: "http://127.0.0.1:3100", trace: "retain-on-failure" },
  webServer: [
    {
      command: "node mock/server.mjs",
      url: "http://127.0.0.1:8099/healthz",
      reuseExistingServer: !process.env.CI,
    },
    {
      command: "npx next start -p 3100",
      url: "http://127.0.0.1:3100",
      reuseExistingServer: !process.env.CI,
    },
  ],
});
