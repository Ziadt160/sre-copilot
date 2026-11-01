# Golden set

The golden set is the yardstick for every retrieval number in the README. Every item is read
and approved by a human. Tooling: `uv run sre-golden --help`.

## Versioning

Items live in `evals/golden/<version>/items.jsonl` (`v1`, `v2`, ...). A released directory is
never edited in place: changes go to a new directory. Every metric is stamped with
`golden_version(path)` from `sre_shared.golden_io`, which is `<dir>-<content hash>` and changes
whenever any byte of the file changes.

## Item format

One `GoldenItem` (see `sre_shared.models`) per JSON line:

| field | meaning |
|---|---|
| `id` | stable, e.g. `fac-0001` (factual), `exs-0001` (exact string) |
| `stratum` | Phase 1: `factual` (about 40 kept) and `exact_string` (about 30 kept) |
| `question`, `expected_answer` | the query and a reference answer |
| `filters` | retrieval filters for the question; synthesized items are scoped to their gold passage's `k8s_version` (pages are near-identical across versions) |
| `gold_passages` | list of `{source_path, k8s_version, quote}`; `quote` is a verbatim span of `documents.parsed_text` for that version (never a chunk id, so results survive re-chunking) |
| `origin` | `synthetic`, `rewritten` (question changed by hand), `hand` |
| `status` | `candidate`, `accepted`, `rejected` |
| `synth_model`, `reviewer`, `reviewed_at` | provenance |
| `tags` | see below |

Tags written by the tooling (hand-written items should follow the same convention):

- `exact:<string>` (required for `exact_string` items): the literal the question must contain
  (an error name, CLI flag or field path).
- `evolutions:N`: DeepEval evolutions used; `chunk:<chunk_key>`, `url:<url>`,
  `match:<error|flag|field>`: traceability.

## Workflow

```
sre-golden pick --stratum factual --n 60 --versions 1.34 --versions 1.33 --seed 1 --out work/ctx_fac.jsonl
sre-golden pick --stratum exact_string --n 45 --seed 1 --out work/ctx_exs.jsonl
sre-golden synthesize --contexts work/ctx_fac.jsonl --model ollama:qwen2.5:7b --out work/cand_fac.jsonl
sre-golden synthesize --contexts work/ctx_exs.jsonl --model ollama:qwen2.5:7b --out work/cand_exs.jsonl --id-start 1
cat work/cand_fac.jsonl work/cand_exs.jsonl > work/candidates.jsonl
sre-golden review --in work/candidates.jsonl --out evals/golden/v1/items.jsonl --reviewer <name>
sre-golden validate --in evals/golden/v1/items.jsonl
sre-golden mirror --in evals/golden/v1/items.jsonl --dataset sre-copilot-golden-v1
```

- `pick` is deterministic for a seed and uses a page once across versions; `contribute/`
  pages are skipped. Factual: concept/task chunks spread across pages. Exact string: chunks
  containing an error name, a multi-word CLI flag or a field path that is rare in its version
  (on at most 3 pages, 10 for error names, so kubectl's global flags repeated on ~120
  generated pages are out), with the three kinds taking turns.
- `synthesize` uses DeepEval's Synthesizer with one chunk per context, so the gold passage is
  the chunk body (trim it in review). `--quality-threshold` sets the critic score below which a
  candidate is dropped (the count per stratum is printed); `--evolutions N` (default 0) adds
  DeepEval evolutions, recorded in `tags`. Hosted models are refused when the planned tokens
  exceed `budgets.golden_synthesis_max_share_of_tpd` of the model's `tpd`; Ollama is exempt.
  Synthetic questions echo the source wording, which flatters keyword search: rewrite many.
- `review` keys: `a` accept, `r` reject, `q` rewrite question (sets `origin=rewritten`),
  `e` edit answer, `t` trim quote (must be a sub-span), `s` skip, `u` undo, `x` save and exit.
  `q`, `e`, `t` edit the item on screen and `a` commits it. Every decision is saved at once;
  rerun the same command to resume. The screen shows per-stratum progress and the share of
  accepted items rewritten by hand (target at least 30%).
- `validate` exits non-zero if an accepted item has no passage, a quote is not found in the
  matching document (whitespace-normalized), an `exact_string` question lacks its string, or
  ids repeat.
- `mirror` upserts accepted items to LangSmith by stable example id; without
  `LANGSMITH_API_KEY` it prints a message and does nothing.
