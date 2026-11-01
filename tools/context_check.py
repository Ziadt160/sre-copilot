"""What does each answer setting hand the model? A deterministic check before spending quota.

For every golden question it retrieves once (the answer path's own retrieval request: dense,
reranked top 20) and builds the model's context for each K and compression mode exactly as
the answerer does (`NoCompressor` / `SentenceCompressor`, then the token budget). No model is
called. Per setting it reports:

  * context tokens (the answerer's `len // 4` estimate), mean and median
  * chunks that reached the prompt (the token budget can cut the tail at large K)
  * gold kept: share of questions with gold passages whose context still evidences one
    (`match_grade >= 1`, the citation-hit rule); an upper bound on citation hit
  * gold lost to compression: questions whose top K evidenced gold before compression and
    whose compressed context no longer does

Answer runs then measure what tokens cannot: faithfulness and citation hit. Read-only on the
database. The reranker must be reachable: a run where any question fell back to fused order
is refused, since that would change what is measured. The lead runs it, e.g.:

    uv run python tools/context_check.py --database-url postgresql://... --out report.json

The summary parts are pure functions (tested in evals/tests/tools/test_context_check.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODES = ("none", "sentence")


@dataclass(frozen=True)
class Row:
    item_id: str
    stratum: str
    k: int
    compression: str
    tokens: int
    n_chunks: int
    gold: int | None  # best match_grade of the context; None when the item has no gold
    gold_before: int | None  # the same for the top K before compression and budget


def best_grade(chunks: Iterable[Any], passages: Sequence[Any]) -> int | None:
    """Best `match_grade` of any chunk against any gold passage; None without gold."""
    from sre_evals.metrics.core import match_grade

    if not passages:
        return None
    return max((match_grade(c, p) for c in chunks for p in passages), default=0)


def summarize(rows: Sequence[Row]) -> dict[str, dict[str, Any]]:
    """Per `k<K>-<mode>` setting: token and chunk counts, gold kept and lost."""
    out: dict[str, dict[str, Any]] = {}
    for key in sorted({(r.k, r.compression) for r in rows}):
        rs = [r for r in rows if (r.k, r.compression) == key]
        gold = [r for r in rs if r.gold is not None]
        out[f"k{key[0]}-{key[1]}"] = {
            "n": len(rs),
            "tokens_mean": statistics.fmean(r.tokens for r in rs),
            "tokens_median": statistics.median(r.tokens for r in rs),
            "chunks_mean": statistics.fmean(r.n_chunks for r in rs),
            "budget_cut": sum(1 for r in rs if r.n_chunks < r.k and r.compression == "none"),
            "n_gold": len(gold),
            "gold_kept": statistics.fmean(1.0 if (r.gold or 0) >= 1 else 0.0 for r in gold)
            if gold
            else None,
            "gold_lost": sorted(
                r.item_id for r in gold if (r.gold_before or 0) >= 1 and (r.gold or 0) < 1
            ),
        }
    return out


def paired_savings(rows: Sequence[Row]) -> dict[int, dict[str, float]]:
    """Per K: tokens saved by sentence compression, per question (mean, median, share)."""
    by = {(r.item_id, r.k, r.compression): r for r in rows}
    out: dict[int, dict[str, float]] = {}
    for k in sorted({r.k for r in rows}):
        saved = [
            by[(i, k, "none")].tokens - by[(i, k, "sentence")].tokens
            for (i, kk, m) in by
            if kk == k and m == "none" and (i, k, "sentence") in by
        ]
        base = [by[(i, k, "none")].tokens for (i, kk, m) in by if kk == k and m == "none"]
        if saved:
            out[k] = {
                "saved_mean": statistics.fmean(saved),
                "saved_median": statistics.median(saved),
                "saved_share": sum(saved) / sum(base) if sum(base) else 0.0,
            }
    return out


def render(summary: Mapping[str, Mapping[str, Any]], savings: Mapping[int, Any]) -> str:
    lines = [
        "| setting | n | context tokens mean (median) | chunks in prompt | gold kept | gold lost |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        kept = "-" if s["gold_kept"] is None else f"{s['gold_kept']:.3f} ({s['n_gold']})"
        lines.append(
            f"| {name} | {s['n']} | {s['tokens_mean']:.0f} ({s['tokens_median']:.0f}) |"
            f" {s['chunks_mean']:.2f} | {kept} | {len(s['gold_lost'])} |"
        )
    for k, v in savings.items():
        lines.append(
            f"\nk{k}: sentence compression saves {v['saved_mean']:.0f} tokens per question"
            f" (median {v['saved_median']:.0f}, {v['saved_share']:.0%} of the context)"
        )
    return "\n".join(lines)


async def collect(dsn: str, golden: str, ks: Sequence[int], budget: int | None) -> list[Row]:
    from sre_core.answer.base import Compressor
    from sre_core.answer.compress import NoCompressor, SentenceCompressor, chunk_tokens
    from sre_core.answer.service import Answerer
    from sre_core.retrieval.rerank import default_reranker
    from sre_core.retrieval.service import Retriever
    from sre_shared.config import load_answer_config
    from sre_shared.db import create_pool
    from sre_shared.embed import get_embedder
    from sre_shared.golden_io import golden_path, load_golden
    from sre_shared.models import ANONYMOUS, RetrievalFilters

    cfg = load_answer_config().model_copy(update={"k": max(ks)})
    budget = budget or cfg.context_token_budget
    reranker = default_reranker()
    if reranker is None:
        raise SystemExit("no reranker configured for this environment")
    compressors: dict[str, Compressor] = {
        "none": NoCompressor(),
        "sentence": SentenceCompressor(get_embedder("gte-small")),
    }
    items = load_golden(golden_path(golden), accepted_only=True)
    rows: list[Row] = []
    pool = create_pool(dsn, min_size=1, max_size=2)
    await pool.open(wait=True)
    try:
        retriever = Retriever(pool, reranker=reranker)
        for n, item in enumerate(items, 1):
            req = Answerer._retrieval_request(
                item.question, cfg, item.filters or RetrievalFilters(), None
            )
            res = await retriever.retrieve(req, ANONYMOUS)
            if not res.reranked:
                raise SystemExit(f"{item.id}: reranker unavailable (fused order); refusing")
            for k in ks:
                top = res.chunks[:k]
                before = best_grade(top, item.gold_passages)
                for mode in MODES:
                    ctx = await compressors[mode].compress(item.question, top, budget)
                    rows.append(
                        Row(
                            item.id, item.stratum, k, mode,
                            sum(chunk_tokens(c) for c in ctx), len(ctx),
                            best_grade(ctx, item.gold_passages), before,
                        )
                    )  # fmt: skip
            if n % 10 == 0:
                print(f"  {n}/{len(items)} questions", file=sys.stderr)
    finally:
        await pool.close()
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--database-url", required=True, help="Postgres DSN (an active corpus)")
    ap.add_argument("--golden", default="v1", help="golden set version under evals/golden/")
    ap.add_argument("--ks", default="3,5,8", help="comma-separated chunk counts")
    ap.add_argument("--budget", type=int, default=None, help="default: config/answer.yaml")
    ap.add_argument("--out", type=Path, default=None, help="write rows + summary as JSON")
    args = ap.parse_args(argv)
    ks = sorted({int(x) for x in args.ks.split(",")})

    from sre_shared.aio import run

    rows = run(collect(args.database_url, args.golden, ks, args.budget))
    summary, savings = summarize(rows), paired_savings(rows)
    print(render(summary, savings))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"summary": summary, "savings": savings, "rows": [asdict(r) for r in rows]},
                indent=1,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
