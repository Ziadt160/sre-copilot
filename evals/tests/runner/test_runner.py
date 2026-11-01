from __future__ import annotations

from typing import Any

import pytest

from sre_evals.runner import load_experiments, run_experiment
from sre_evals.runner.config import ExperimentConfig
from sre_evals.runner.engine import WARMUP_QUERIES
from sre_evals.runner.testing import SRC, VER, FakeRetriever, W, chunk
from sre_shared.models import GoldenItem, GoldPassage, RetrievalFilters, RetrievalRequest
from sre_shared.testing.seed import create_corpus_version

pytestmark = pytest.mark.integration


def item(iid: str, stratum: str, quote: str | None, **kw: Any) -> GoldenItem:
    gold = [GoldPassage(source_path=SRC, k8s_version=VER, quote=quote)] if quote else []
    return GoldenItem(
        id=iid, stratum=stratum, question=f"question {iid}", gold_passages=gold,
        status="accepted", **kw,
    )  # fmt: skip


def script(req: RetrievalRequest, exact: bool) -> tuple[list[Any], float]:
    if req.query == "question fac-0001":  # gold at rank 1, 10 ms
        return [chunk(1, W(1, 10)), chunk(2, "noise")], 10.0
    if req.query == "question exa-0001":  # gold at rank 3, 30 ms
        return [chunk(3, "noise a"), chunk(4, "noise b"), chunk(5, W(21, 30))], 30.0
    raise AssertionError(f"unexpected query {req.query}")


ITEMS = [
    item("fac-0001", "factual", W(1, 10), filters=RetrievalFilters(k8s_versions=[VER])),
    item("exa-0001", "exact_string", W(21, 30)),
    item("unans-0001", "unanswerable", None),
]


def cfg(**kw: Any) -> ExperimentConfig:
    return ExperimentConfig(name="t", k=[3, 5], **kw)


async def test_run_writes_run_and_scores(db: Any) -> None:
    await create_corpus_version(db)
    fake = FakeRetriever(lambda r, e: script(r, e))
    out = await run_experiment(
        cfg(), ITEMS, fake, db, loop="ci", golden_version="v1-abc", concurrency=2
    )
    s = out.summary
    # Item A: recall 1, MRR 1, nDCG 1. Item B: gold at rank 3 -> recall@3 1, MRR 1/3,
    # nDCG = (3/log2(4)) / 3 = 0.5. Means: recall 1, MRR (1 + 1/3)/2, nDCG (1 + 0.5)/2.
    assert s["recall@3"] == 1.0 and s["recall@5"] == 1.0
    assert s["mrr@3"] == pytest.approx(2 / 3)
    assert s["ndcg@5"] == pytest.approx(0.75)
    # The unanswerable item has no gold: not retrieved, not scored.
    assert (s["n_items"], s["n_scored"], s["n_skipped_no_gold"]) == (3, 2, 1)
    # Untimed warm-up queries come first; the unanswerable item is never queried.
    assert len(fake.calls) == 2 + min(WARMUP_QUERIES, 2)
    assert all(req.query != "question unans-0001" for req, _ in fake.calls)
    # Latency 10 and 30 ms (warm-up excluded): p50 = 20, p95 = 10 + 0.95 * 20 = 29.
    assert s["latency_ms"]["p50"] == pytest.approx(20.0)
    assert s["latency_ms"]["p95"] == pytest.approx(29.0)
    assert s["empty_result_rate"] == 0.0
    assert s["by_stratum"]["exact_string"]["mrr@3"] == pytest.approx(1 / 3)
    assert s["by_stratum"]["factual"]["n"] == 1

    # Retrieval ran once at max(k) with the item's own filters.
    first = next(r for r, _ in fake.calls if r.query == "question fac-0001")
    assert first.k == 5 and first.filters.k8s_versions == [VER]

    run = await (
        await db.execute("select * from public.eval_runs where id = %s", [out.run_id])
    ).fetchone()
    assert run["status"] == "done" and run["kind"] == "retrieval" and run["loop"] == "ci"
    assert run["experiment"] == "t" and run["golden_version"] == "v1-abc"
    assert run["stamps"]["embedding_revision"] == "93b36ff09519291b77d6000d2e86bd8565378086"
    assert run["stamps"]["config_hash"] and run["stamps"]["git_sha"]
    assert run["stamps"]["golden_version"] == "v1-abc"
    assert run["stamps"]["chunker_version"] == "test"
    assert run["corpus_version_id"] == 1 and run["stamps"]["corpus_version_id"] == 1
    assert run["summary"]["recall@5"] == 1.0 and run["finished_at"] is not None

    rows = await (
        await db.execute(
            "select item_id, metric, k, value, details from public.eval_scores"
            " where run_id = %s order by item_id, metric, k",
            [out.run_id],
        )
    ).fetchall()
    got = {(r["item_id"], r["metric"], r["k"]): r["value"] for r in rows}
    # Per item: 3 metrics x 2 ks + latency + n_results = 8 rows, two scored items.
    assert len(rows) == 16
    assert got[("exa-0001", "mrr", 3)] == pytest.approx(1 / 3)
    assert got[("exa-0001", "latency_ms", 0)] == 30.0
    assert got[("fac-0001", "ndcg", 5)] == pytest.approx(1.0)
    detail = next(
        r["details"] for r in rows if r["item_id"] == "exa-0001" and r["metric"] == "n_results"
    )
    assert detail["matches"] == [{"rank": 3, "chunk_id": 5, "grades": [2]}]


async def test_empty_results_counted(db: Any) -> None:
    await create_corpus_version(db)
    fake = FakeRetriever(lambda r, e: ([], 5.0))
    out = await run_experiment(cfg(), ITEMS[:2], fake, db, loop="local", golden_version="v1-x")
    assert out.summary["empty_result_rate"] == 1.0
    assert out.summary["recall@5"] == 0.0 and out.summary["mrr@3"] == 0.0


async def test_failed_run_is_recorded_and_raised(db: Any) -> None:
    def boom(req: RetrievalRequest, exact: bool) -> tuple[list[Any], float]:
        raise RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        await run_experiment(
            cfg(), ITEMS[:1], FakeRetriever(boom), db, loop="local", golden_version="v1"
        )
    row = await (await db.execute("select status, summary from public.eval_runs")).fetchone()
    assert row["status"] == "failed" and "database down" in row["summary"]["error"]
    n = await (await db.execute("select count(*) as n from public.eval_scores")).fetchone()
    assert n["n"] == 0


async def test_ann_recall_vs_exact(db: Any) -> None:
    await create_corpus_version(db)

    def ann_script(req: RetrievalRequest, exact: bool) -> tuple[list[Any], float]:
        if not req.use_keyword and req.k == 10 and (exact or req.ef_search):
            ids = list(range(1, 11))
            if not exact:
                # ef 40 misses 4 of the exact 10; ef 100 misses 1; ef 200 misses none.
                missing = {40: 4, 100: 1, 200: 0}[req.ef_search]
                ids = ids[: 10 - missing] + [100 + i for i in range(missing)]
            return [chunk(i, "x") for i in ids], 1.0
        return script(req, exact)

    fake = FakeRetriever(ann_script)
    out = await run_experiment(
        cfg(legs=["dense"]), ITEMS[:2], fake, db, loop="local", golden_version="v1", ann=True
    )
    # Per item: 6/10 at ef 40, 9/10 at 100, 10/10 at 200 (same for both items).
    assert out.summary["ann_recall"] == pytest.approx({"40": 0.6, "100": 0.9, "200": 1.0})
    assert out.summary["ann_sample"] == 2
    exact_calls = [r for r, e in fake.calls if e]
    assert len(exact_calls) == 2 and all(not r.use_keyword for r in exact_calls)
    row = await (
        await db.execute(
            "select value from public.eval_scores where metric = 'ann_recall' and k = 40"
            " and item_id = 'fac-0001'"
        )
    ).fetchone()
    assert row["value"] == pytest.approx(0.6)


async def test_ann_recall_excludes_queries_with_empty_exact(db: Any) -> None:
    await create_corpus_version(db)

    def ann_script(req: RetrievalRequest, exact: bool) -> tuple[list[Any], float]:
        if not req.use_keyword and req.k == 10:
            if req.query == "question exa-0001":  # nothing passes this item's filters
                return [], 1.0
            # Exact top 4 is {1..4}; the index returns only {1, 2} -> 2/4 = 0.5.
            return [chunk(i, "x") for i in ([1, 2, 3, 4] if exact else [1, 2])], 1.0
        return script(req, exact)

    out = await run_experiment(
        cfg(legs=["dense"]),
        ITEMS[:2],
        FakeRetriever(ann_script),
        db,
        loop="local",
        golden_version="v1",
        ann=True,
    )
    # The empty-exact item is excluded (not counted as 1.0): mean = 0.5, sample = 1.
    assert out.summary["ann_recall"] == pytest.approx({"40": 0.5, "100": 0.5, "200": 0.5})
    assert out.summary["ann_sample"] == 1


def test_all_experiment_configs_load() -> None:
    exps = load_experiments()
    assert set(exps) >= {
        "dense-gte-small", "dense-bge-large", "dense-bge-m3", "kw-english", "kw-simple",
        "hybrid-gte-small", "hybrid-bge-large", "hybrid-bge-m3",
        "hybrid-gte-small-wk0p5", "hybrid-gte-small-wk2",
    }  # fmt: skip
    assert exps["kw-simple"].ts_config == "simple" and exps["kw-simple"].legs == ["keyword"]
    assert exps["hybrid-gte-small-wk2"].weights.keyword == 2.0
    req = exps["dense-bge-m3"].request_for("q", None)
    assert req.use_dense and not req.use_keyword and req.k == 10
    assert req.embedding_model == "bge-m3"


def test_config_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", legs=[])
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", k=[0])
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({"name": "x", "bogus": 1})
