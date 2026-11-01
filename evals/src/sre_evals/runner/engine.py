"""Run one experiment over the golden items and store `eval_runs` + `eval_scores`.

`eval_runs.summary` keys (flat, JSON; the report and dashboards read these):

    n_items            golden items handed to the run
    n_scored           items with >= 1 gold passage (the only ones retrieved and scored)
    n_skipped_no_gold  n_items - n_scored
    recall@K, mrr@K, ndcg@K   mean over scored items, for each K in the config's `k` list
    empty_result_rate  share of scored items whose retrieval returned no chunk
    latency_ms         {"p50", "p95", "mean"} of `timings_ms["total"]`
    by_stratum         {stratum: {"n", "recall@K", "mrr@K", "ndcg@K", ...}}
    ann_recall         (only with --ann-recall) {"<ef_search>": mean ann recall}
    ann_sample         items used for ANN recall
    k_max              the cutoff retrieval ran at

`eval_scores` rows: metrics recall / mrr / ndcg with `k` = cutoff, `latency_ms` and `n_results`
with k = 0 (the `n_results` row's `details` holds the per-chunk grade table), and per-item
`ann_recall` with k = the ef_search used.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from sre_evals.metrics import ItemScores, ann_recall, empty_result_rate, score_item
from sre_evals.runner.config import ExperimentConfig
from sre_shared.embed.base import SPECS
from sre_shared.models import (
    ANONYMOUS,
    Claims,
    GoldenItem,
    RetrievalRequest,
    RetrievalResult,
)
from sre_shared.stamps import VersionStamps, config_hash

ANN_EF_VALUES = (40, 100, 200)
ANN_TOP_N = 10
ANN_SAMPLE = 30
METRICS = ("recall", "mrr", "ndcg")


class RetrieverLike(Protocol):
    async def retrieve(
        self, req: RetrievalRequest, claims: Claims = ..., *, exact: bool = False
    ) -> RetrievalResult: ...


@dataclass
class RunOutcome:
    run_id: int
    status: str
    summary: dict[str, Any]


@dataclass
class _ItemRun:
    item: GoldenItem
    result: RetrievalResult
    scores: ItemScores


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


WARMUP_QUERIES = 3


def latency_stats(latencies: Sequence[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.asarray(latencies, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
    }


def _metric_means(runs: Sequence[_ItemRun], ks: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = _mean([r.scores.recall[k] for r in runs])
        out[f"mrr@{k}"] = _mean([r.scores.mrr[k] for r in runs])
        out[f"ndcg@{k}"] = _mean([r.scores.ndcg[k] for r in runs])
    return out


def summarize(runs: Sequence[_ItemRun], cfg: ExperimentConfig, n_items: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_items": n_items,
        "n_scored": len(runs),
        "n_skipped_no_gold": n_items - len(runs),
        "k_max": cfg.max_k,
        **_metric_means(runs, cfg.k),
        "empty_result_rate": empty_result_rate([len(r.result.chunks) for r in runs]),
        "latency_ms": latency_stats([r.result.timings_ms.get("total", 0.0) for r in runs]),
    }
    groups: dict[str, list[_ItemRun]] = defaultdict(list)
    for r in runs:
        groups[r.item.stratum].append(r)
    summary["by_stratum"] = {
        s: {"n": len(g), **_metric_means(g, cfg.k)} for s, g in sorted(groups.items())
    }
    return summary


async def _retrieve_all(
    cfg: ExperimentConfig,
    items: Sequence[GoldenItem],
    retriever: RetrieverLike,
    claims: Claims,
    concurrency: int,
) -> list[_ItemRun]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(item: GoldenItem) -> _ItemRun:
        async with sem:
            result = await retriever.retrieve(cfg.request_for(item.question, item.filters), claims)
        scores = score_item(
            result.chunks,
            item.gold_passages,
            cfg.k,
            chunk_ids=[c.chunk_id for c in result.chunks],
        )
        assert scores is not None  # items are pre-filtered to those with gold passages
        return _ItemRun(item, result, scores)

    return list(await asyncio.gather(*(one(i) for i in items)))


async def _ann_recall(
    cfg: ExperimentConfig,
    items: Sequence[GoldenItem],
    retriever: RetrieverLike,
    claims: Claims,
    sample: int,
    ef_values: Sequence[int],
    concurrency: int,
) -> dict[str, dict[int, float]]:
    """Per item and ef_search: dense-only top-N from the index vs the exact scan."""
    chosen = sorted(items, key=lambda i: i.id)[:sample]
    sem = asyncio.Semaphore(max(1, concurrency))
    n = max(ANN_TOP_N, 1)

    def req(item: GoldenItem, ef: int) -> RetrievalRequest:
        return cfg.request_for(
            item.question,
            item.filters,
            use_dense=True,
            use_keyword=False,
            k=n,
            candidates_per_leg=max(cfg.candidates_per_leg, n),
            ef_search=ef,
        )

    async def one(item: GoldenItem) -> tuple[str, dict[int, float]]:
        async with sem:
            exact = await retriever.retrieve(req(item, ef_values[0]), claims, exact=True)
            exact_ids = [c.chunk_id for c in exact.chunks]
            per_ef: dict[int, float] = {}
            for ef in ef_values:
                got = await retriever.retrieve(req(item, ef), claims)
                value = ann_recall([c.chunk_id for c in got.chunks], exact_ids)
                if value is not None:  # empty exact list: query excluded from the mean
                    per_ef[ef] = value
        return item.id, per_ef

    results = await asyncio.gather(*(one(i) for i in chosen))
    return {iid: per_ef for iid, per_ef in results if per_ef}


async def _insert_run(
    conn: AsyncConnection[Any],
    cfg: ExperimentConfig,
    loop: str,
    golden_version: str,
    stamps: VersionStamps,
    status: str,
    summary: dict[str, Any],
    kind: str = "retrieval",
) -> int:
    row = await (
        await conn.execute(
            "insert into public.eval_runs (kind, loop, experiment, config, stamps, golden_version,"
            " corpus_version_id, status, summary, finished_at)"
            " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, now()) returning id",
            [
                kind,
                loop,
                cfg.name,
                Jsonb(cfg.model_dump(mode="json")),
                Jsonb(stamps.model_dump(mode="json")),
                golden_version,
                stamps.corpus_version_id,
                status,
                Jsonb(summary),
            ],
        )
    ).fetchone()
    assert row is not None
    return int(row["id"] if isinstance(row, dict) else row[0])


async def _chunker_version(conn: AsyncConnection[Any], corpus_version_id: int | None) -> str | None:
    if corpus_version_id is None:
        return None
    row = await (
        await conn.execute(
            "select chunker_version from public.corpus_versions where id = %s",
            [corpus_version_id],
        )
    ).fetchone()
    if row is None:
        return None
    return str(row["chunker_version"] if isinstance(row, dict) else row[0])


async def run_experiment(
    cfg: ExperimentConfig,
    items: Sequence[GoldenItem],
    retriever: RetrieverLike,
    conn: AsyncConnection[Any],
    *,
    loop: str,
    golden_version: str,
    concurrency: int = 1,
    ann: bool = False,
    ann_sample: int = ANN_SAMPLE,
    ann_ef_values: Sequence[int] = ANN_EF_VALUES,
    claims: Claims = ANONYMOUS,
    kind: str = "retrieval",
    extra: Mapping[str, Any] | None = None,
) -> RunOutcome:
    """Run `cfg` on every item that has gold passages and store the run.

    `kind` other than "retrieval" (a sweep point) keeps the run out of the README table;
    `extra` is merged into the stored summary (sweep name, parameters, index build facts).
    A retrieval error stores a `failed` run (no scores) and re-raises.
    """
    scorable = [i for i in items if i.gold_passages]
    stamps = VersionStamps(
        config_hash=config_hash(cfg),
        embedding_model=cfg.embedding_model,
        embedding_revision=SPECS[cfg.embedding_model].revision,
        golden_version=golden_version,
    )
    try:
        # Untimed warm-up: loads the query embedder, fills the pool and the database caches,
        # so latency covers steady-state queries only.
        for item in scorable[:WARMUP_QUERIES]:
            await retriever.retrieve(cfg.request_for(item.question, item.filters), claims)
        runs = await _retrieve_all(cfg, scorable, retriever, claims, concurrency)
        ann_by_item: dict[str, dict[int, float]] = {}
        if ann and cfg.uses_dense and scorable:
            ann_by_item = await _ann_recall(
                cfg, scorable, retriever, claims, ann_sample, ann_ef_values, concurrency
            )
    except Exception as e:
        stamps.corpus_version_id = None
        await _insert_run(
            conn, cfg, loop, golden_version, stamps, "failed",
            {"error": f"{type(e).__name__}: {e}", **(extra or {})}, kind,
        )  # fmt: skip
        raise

    summary = summarize(runs, cfg, len(items))
    summary.update(extra or {})
    if ann_by_item:
        efs = sorted({ef for v in ann_by_item.values() for ef in v})
        summary["ann_recall"] = {
            str(ef): _mean([v[ef] for v in ann_by_item.values() if ef in v]) for ef in efs
        }
        summary["ann_sample"] = len(ann_by_item)

    corpus_ids = {r.result.corpus_version_id for r in runs}
    stamps.corpus_version_id = next(iter(corpus_ids)) if len(corpus_ids) == 1 else None
    stamps.chunker_version = await _chunker_version(conn, stamps.corpus_version_id)

    async with conn.transaction():
        run_id = await _insert_run(conn, cfg, loop, golden_version, stamps, "done", summary, kind)
        rows: list[list[Any]] = []
        for r in runs:
            iid = r.item.id
            for k in cfg.k:
                rows.append([run_id, iid, "recall", k, r.scores.recall[k], None])
                rows.append([run_id, iid, "mrr", k, r.scores.mrr[k], None])
                rows.append([run_id, iid, "ndcg", k, r.scores.ndcg[k], None])
            rows.append([run_id, iid, "latency_ms", 0, r.result.timings_ms.get("total", 0.0), None])
            details = {"matches": [asdict(m) for m in r.scores.matches]}
            rows.append([run_id, iid, "n_results", 0, float(r.scores.n_results), Jsonb(details)])
        for iid, per_ef in ann_by_item.items():
            for ef, value in per_ef.items():
                rows.append([run_id, iid, "ann_recall", ef, value, None])
        async with conn.cursor() as cur:
            await cur.executemany(
                "insert into public.eval_scores (run_id, item_id, metric, k, value, details)"
                " values (%s, %s, %s, %s, %s, %s)",
                rows,
            )
    return RunOutcome(run_id=run_id, status="done", summary=summary)
