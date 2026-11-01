"""Parameter grids, derived configs, HNSW DDL and an index rebuild on a real database."""

from __future__ import annotations

from typing import Any

import pytest

from sre_evals.runner import run_experiment
from sre_evals.runner.config import ExperimentConfig
from sre_evals.runner.sweep import (
    MIGRATION_HNSW,
    build_hnsw,
    check_hnsw_params,
    current_hnsw_params,
    derive,
    grid_points,
    hnsw_ddl,
    parse_grid,
    point_name,
)
from sre_evals.runner.testing import SRC, VER, FakeRetriever, W, chunk
from sre_shared.models import GoldenItem, GoldPassage, RetrievalRequest
from sre_shared.testing.seed import create_corpus_version


def test_parse_grid_and_points() -> None:
    grid = parse_grid(["w_keyword=0,0.5", "rrf_k=10,60"])
    assert grid == {"w_keyword": [0.0, 0.5], "rrf_k": [10, 60]}
    assert grid_points(grid) == [
        {"w_keyword": 0.0, "rrf_k": 10},
        {"w_keyword": 0.0, "rrf_k": 60},
        {"w_keyword": 0.5, "rrf_k": 10},
        {"w_keyword": 0.5, "rrf_k": 60},
    ]


@pytest.mark.parametrize(
    "specs", [["bogus=1"], ["rrf_k=ten"], ["rrf_k="], ["rrf_k=1", "rrf_k=2"], [], ["w_keyword"]]
)
def test_parse_grid_rejects_bad_input(specs: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_grid(specs)


def test_derive_sets_nested_weights_and_names_the_point() -> None:
    base = ExperimentConfig(name="hybrid-gte-small")
    cfg = derive(base, {"w_keyword": 0.25, "rrf_k": 10, "ef_search": 40})
    assert cfg.name == "hybrid-gte-small[w_keyword=0.25,rrf_k=10,ef_search=40]"
    assert (cfg.weights.dense, cfg.weights.keyword, cfg.rrf_k, cfg.ef_search) == (1.0, 0.25, 10, 40)
    assert base.weights.keyword == 1.0  # the base is not mutated
    req = cfg.request_for("q", None)
    assert (req.w_keyword, req.rrf_k, req.ef_search) == (0.25, 10, 40)
    assert point_name("x", {"w_keyword": 0.0}) == "x[w_keyword=0]"


@pytest.mark.parametrize(("m", "efc"), [(1, 64), (101, 300), (16, 3), (16, 1001), (32, 32)])
def test_hnsw_params_follow_pgvector_limits(m: int, efc: int) -> None:
    with pytest.raises(ValueError):
        check_hnsw_params(m, efc)


def test_hnsw_ddl_uses_the_whitelisted_column(pg_dsn: str) -> None:
    import psycopg

    drop, create = hnsw_ddl("gte-small", 8, 32)
    with psycopg.connect(pg_dsn) as conn:
        assert drop.as_string(conn) == 'drop index if exists public."chunks_emb_gte_small_hnsw"'
        assert create.as_string(conn) == (
            'create index "chunks_emb_gte_small_hnsw" on public.chunks using hnsw'
            ' ("emb_gte_small" extensions.halfvec_ip_ops) with (m = 8, ef_construction = 32)'
        )


async def _index_options(db: Any) -> list[str]:
    row = await (
        await db.execute(
            "select c.reloptions from pg_class c where c.relname = 'chunks_emb_gte_small_hnsw'"
        )
    ).fetchone()
    return list(row["reloptions"]) if row else []


@pytest.mark.integration
async def test_build_hnsw_rebuilds_and_restores(db: Any) -> None:
    assert await current_hnsw_params(db, "gte-small") == MIGRATION_HNSW["gte-small"] == (16, 128)
    info = await build_hnsw(db, "gte-small", 8, 32)
    assert (info["m"], info["ef_construction"]) == (8, 32) and info["build_seconds"] >= 0
    assert await _index_options(db) == ["m=8", "ef_construction=32"]
    assert await current_hnsw_params(db, "gte-small") == (8, 32)
    await build_hnsw(db, "gte-small", *MIGRATION_HNSW["gte-small"])
    assert await _index_options(db) == ["m=16", "ef_construction=128"]


@pytest.mark.integration
async def test_sweep_runs_stay_out_of_the_readme_table(db: Any) -> None:
    from sre_evals.report import latest_runs

    await create_corpus_version(db)

    def script(req: RetrievalRequest, exact: bool) -> tuple[list[Any], float]:
        return [chunk(1, W(1, 10))], 5.0

    item = GoldenItem(
        id="fac-0001", stratum="factual", question="q", status="accepted",
        gold_passages=[GoldPassage(source_path=SRC, k8s_version=VER, quote=W(1, 10))],
    )  # fmt: skip
    cfg = derive(ExperimentConfig(name="hybrid-gte-small"), {"w_keyword": 0.5})
    out = await run_experiment(
        cfg, [item], FakeRetriever(script), db, loop="local", golden_version="v1-x",
        kind="sweep", extra={"sweep": "fusion", "params": {"w_keyword": 0.5}},
    )  # fmt: skip
    assert out.summary["sweep"] == "fusion" and out.summary["params"] == {"w_keyword": 0.5}
    row = await (
        await db.execute("select kind from public.eval_runs where id = %s", [out.run_id])
    ).fetchone()
    assert row["kind"] == "sweep"
    assert await latest_runs(db) == []
