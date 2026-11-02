# web (Phase 3)

Next.js (App Router) chat UI: streamed answers with citations, the approval dialog (approve,
edit arguments, reject; 15 minute countdown), operation progress, escalation and abstain
states, a thread list, thumbs with a reason, per-source "useful", and copy / regenerate events.

Configuration (build-time, `.env.local`; see `.env.example`): `NEXT_PUBLIC_SUPABASE_URL`,
`NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL`.

```
npm ci
npm run gen:types     # src/contracts/agent.types.ts from agent.schema.json (never hand-edit)
npm run dev:mock      # mock API + Supabase login on :8099 (any email/password)
NEXT_PUBLIC_API_URL=http://127.0.0.1:8099 NEXT_PUBLIC_SUPABASE_URL=http://127.0.0.1:8099 \
  NEXT_PUBLIC_SUPABASE_ANON_KEY=mock npm run dev
npm run lint && npm run typecheck && npm run build
npx playwright install chromium && npm run test:e2e   # needs a build made with the mock URLs
```

`agent.schema.json` is exported from the Python contracts
(`uv run python -m sre_shared.contracts.export`); `npm run check:types` fails when the generated
types are stale. In the mock, a message containing "restart" proposes a write that needs
approval, "outage" escalates, "unknown" abstains; anything else streams an answer with sources.
