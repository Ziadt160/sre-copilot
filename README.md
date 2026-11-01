# SRE Copilot

A production-style RAG + MCP agent for Kubernetes operations, built solo on a $0 budget.
It answers Kubernetes and cloud-native questions from versioned docs with citations, and (from
Phase 3) acts on a cluster through an MCP server: inspect resources, restart or scale workloads
after approval, open incidents.

**The rule:** a technique stays only if it beats the baseline on quality, latency or cost.
Every number below comes from an eval run stamped with the git SHA, config hash, embedding
model, corpus version and golden-set version that produced it. Experiments that lose are written
up as negative results.

## Results

<!-- results:start -->
_No eval runs yet. Phase 1 fills this table with retrieval baselines: dense-only vs keyword-only
vs hybrid, per embedding model._
<!-- results:end -->

## Architecture (Phase 1 slice)

- **Corpus:** Kubernetes docs (`kubernetes/website`, CC BY 4.0) for the latest three minor
  versions, pinned by commit SHA in `config/corpus.yaml`.
- **Ingest** (`services/ingest`): parse Hugo Markdown, chunk by heading with code blocks kept
  whole (≤ 510 gte-small tokens incl. a context header), embed, upsert idempotently into a new
  blue/green corpus version.
- **Retrieval** (`services/core`, module `sre_core.retrieval`): one SQL query returns dense
  (pgvector HNSW on `halfvec`) and keyword (Postgres full-text) candidates; weighted reciprocal
  rank fusion runs in Python. Version and doc-type filters run inside SQL with pgvector's
  iterative index scan; RLS restricts rows by access group.
- **Evals** (`evals`): a hand-reviewed golden set whose evidence is stored as source passages,
  not chunk IDs; recall@k, MRR, nDCG, ANN recall and latency; a CI gate against a committed
  baseline.

The full design covers five services (core, mcp-ops, ingest, monitor, web); see
`docs/decisions.md` for the decision log.

## Quick start (lab)

```bash
uv sync                                  # Python 3.12 workspace
npx supabase start                       # local Supabase stack (Docker)
uv run pytest                            # unit + integration tests
```

GPU embedding experiments (bge-large-en-v1.5, bge-m3) run in a small localhost server under any
Python that already has torch + sentence-transformers, so the project environment never installs
torch:

```bash
HF_HUB_OFFLINE=1 python tools/gpu_embed_server.py --port 8091   # a Python with torch + CUDA
uv run pytest -m gpu packages/shared/tests
```

## Licence

Code: MIT. Kubernetes documentation: CC BY 4.0, (c) The Kubernetes Authors; see `NOTICE`.
