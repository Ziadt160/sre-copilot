# Decision log

Each decision names what it rejected, so "why not X?" has an answer on file. Decisions 1–14 come
from the architecture doc (Oct 29, 2026); later ones were made while building.

| # | Decision | Why | Rejected |
|---|---|---|---|
| 1 | Supabase Postgres for vectors, keyword search, auth and state | One system; RLS gives permission-aware retrieval | A separate vector database |
| 2 | Hybrid = pgvector + Postgres full-text, weighted RRF | No extension; works on Supabase Free | A BM25 extension |
| 3 | gte-small, 384-d, stored as halfvec | Fits 500 MB; free query embeddings inside Supabase | API embeddings (quota), larger models — *being re-tested, see 17* |
| 4 | Groq gpt-oss first, then OpenRouter, Cloudflare Workers AI and NVIDIA | $0; OpenAI's open-weight models | GitHub Models (retired), Cerebras (paid trial only), paid OpenAI API |
| 5 | Fixed LangGraph graph with a router | Bounded calls under token quotas; testable paths | An open ReAct loop, multi-agent |
| 6 | mcp-ops enforces scopes and idempotency itself | Stops confused-deputy errors and duplicate writes | Checks only inside the agent |
| 7 | Simulator backend in the public demo | Safety; free hosts cannot run Kubernetes | A real cluster in the demo |
| 8 | DeepEval for metrics, LangSmith for traces and review | One metric definition; free human review | Adding RAGAS; Confident AI |
| 9 | Local judge, calibrated against a hosted judge and humans | Volume at $0 | A hosted judge for every run |
| 10 | Primary/replica in the lab only | The free tier has no replicas | Paying for managed replicas |
| 11 | Five services | Split where scaling or failure differs | Retrieval as its own service |
| 12 | FastMCP 4 for mcp-ops; slow writes run as MCP tasks | The only Python route to the tasks extension | The official SDK (no tasks yet); a worker microservice |
| 13 | Agent runs inside the request, streamed over SSE; side work via a Postgres outbox | Free hosts sleep; runs already have IDs and checkpoints | A queue and worker for every agent run |
| 14 | core stays one service with module boundaries checked in CI (import-linter) | No extra hops or cold starts on the answer path | Splitting retrieval, routing and caching into microservices |
| 15 | Fusion (RRF) in Python over one SQL candidates query | One round trip; fusion unit-testable; legs switchable for baselines and degradation | RRF inside SQL |
| 16 | gte-small runs on ONNX Runtime (CPU), no torch in core or CI | Same export the Edge Runtime serves; light images; no CUDA wheels in CI | sentence-transformers + torch everywhere |
| 17 | Embedding-model experiment: gte-small vs bge-large-en-v1.5 vs bge-m3, one halfvec column per model | The doc's rule: a bigger model stays only if it measurably wins; per-model columns keep filtered HNSW on one table | Assuming gte-small is good enough; a join table per model |
| 18 | Gold evidence stored as (path, version, quote); matching by word-5-gram coverage | Metrics survive chunking changes | Gold chunk IDs |
| 19 | Integration tests: one shared Postgres server, a fresh cloned database per test session | Isolation for parallel work within a 3 GB WSL memory cap | A container per test session |
| 20 | BGE models run in a localhost GPU embedding server (`tools/gpu_embed_server.py`) under an existing torch Python; the project calls it over HTTP | Reuses the machine's torch + CUDA; the uv environment, images and CI stay torch-free | A `lab-gpu` extra that downloads ~4 GB of torch wheels into the project venv |
| 21 | The demo's Edge Function embeds one query per request (max 8 texts) | Measured: Supabase Edge Runtime gte-small = Python ONNX gte-small (cosine 1.000000 min over 53 texts incl. near-512-token ones); a ~500-token text costs ~0.45 s CPU and several in one request hit the runtime's CPU-time limit (request cancelled) | Batching 16–64 texts per call |
| 22 | Golden items are asked against their gold passage's version (`filters.k8s_versions`); cross-version questions get their own stratum | Most pages are near-identical across 1.35–1.37, so an unfiltered query often returns the right text from another version, which version-exact gold scores as a miss | Matching gold across versions (hides version mistakes); unfiltered questions |
| 23 | Exact-string contexts need a rare literal (≤ 3 pages in its version, ≤ 10 for error names), kinds take turns, a page is used once, `contribute/` is skipped | First pick: 39 of 45 were flags, many kubectl global flags repeated on 110–135 generated pages, which makes one gold passage arbitrary | Random literal per chunk; dropping generated reference pages entirely |
| 24 | New database connections retry (4 attempts, 0.5–2 s backoff) unless the server rejected them (`FATAL:`); queries never retry | Measured: the lab's Docker port forward drops ~1 in 25 connects; hosted networks blip the same way | Fixing it per call site; retrying queries (not idempotent in general) |
| 25 | An identical re-ingest (same label, SHAs, chunker) reuses the active corpus version in place and leaves it active; activating the active version is a no-op | Found in the lab: the re-run built a second full copy (47,151 chunks, +530 MB) because only building/ready versions were reusable; same SHAs + chunker mean the same content, and a changed page still raises | Always building a new version (doubles storage per re-run); refusing to run while a matching version is active |
| 26 | Keyword leg matches any query term (`plainto_tsquery` with `&` → `|`), ranked by `ts_rank_cd`; `keyword_match: all` (websearch_to_tsquery) stays as the `kw-english-all` experiment | Lab probe: with every term required, natural questions ("my pod is stuck in CrashLoopBackOff, how do I debug it?") matched nothing, so keyword-only and hybrid were a strawman | AND semantics; a BM25 extension (not on Supabase). Known gap: Postgres ranks without term rarity (IDF); IDF weighting is the next experiment, kept only if it beats this baseline |
