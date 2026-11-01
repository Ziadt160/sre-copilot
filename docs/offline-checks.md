# Offline checks

Measurements made by the tools under `tools/`, outside the eval-run tables in
[results.md](results.md) (which is generated from stored runs). Each section names the
command that reproduces it.

## halfvec vs fp32 (decision 35)

`uv run python tools/halfvec_check.py --database-url <sre_eval> --version 1.37`

Version 1.37, 17,570 chunks, re-embedded in fp32 with gte-small; exact inner-product search
with the vectors as stored (fp16) and in fp32.

| Queries | n | top-10 overlap | recall@5 fp32 | recall@5 fp16 | top-10 order changed |
|---|---|---|---|---|---|
| Golden questions of 1.37 | 23 | 0.9913 | 0.9130 | 0.9130 | 5 |
| Chunk-header queries | 200 | 0.9960 | 0.5400 | 0.5400 | 27 |

Storage (vector values, analytic): 1,544 B per row in fp32, 776 B in fp16; HNSW index
estimated at 30.6 MiB vs 17.7 MiB for this version (m = 16).

## What each answer setting hands the model (2026-11-01)

`uv run python tools/context_check.py --database-url <sre_eval> --out <report.json>`

Golden v1, 69 questions, the answer path's own retrieval (dense gte-small, top 20 reranked
with bge-reranker-base, every question reranked), `config/answer.yaml` token budget 2,500.
No model is called. "Gold in context" is the share of questions whose context still
evidences a gold passage (`match_grade >= 1`), an upper bound on citation hit.

| Setting | Context tokens, mean (median) | Chunks in prompt | Gold in context | Gold lost |
|---|---|---|---|---|
| K3 | 864 (882) | 3.00 | 0.870 | 0 |
| K3 + sentence compression | 735 (740) | 3.00 | 0.855 | 1 |
| K5 (default) | 1,455 (1,367) | 5.00 | 0.942 | 0 |
| K5 + sentence compression | 1,238 (1,178) | 5.00 | 0.928 | 1 |
| K8 | 2,055 (2,107) | 7.49 | 0.942 | 1 |
| K8 + sentence compression | 1,850 (1,820) | 7.84 | 0.942 | 1 |

Sentence compression saves 15% of the context at K3 and K5 (129 and 218 tokens per
question) and 10% at K8, and drops the gold sentence from one question. K8 adds 41% more
context than K5 with no gain in gold coverage: the token budget cuts its tail (7.49 of 8
chunks fit), once removing a gold chunk. K3 saves 41% against K5 and loses gold for 5 of
69 questions. These numbers say what each setting costs; whether faithfulness or citation
hit change is measured by the answer runs.
