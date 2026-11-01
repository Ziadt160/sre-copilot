"""Parameter sweeps: a grid over one experiment config's retrieval knobs, and HNSW build
parameters (m, ef_construction) for one embedding column.

Sweep points are stored like any run, with kind `sweep` or `hnsw`, so the README table (kind
`retrieval`) never shows them; `docs/results.md` renders them in sections of their own.
Picking the best point and reporting its score on the same 69 items flatters it, so the
report also shows each point on two fixed, stratified halves of the golden set.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping, Sequence
from typing import Any

from psycopg import AsyncConnection, sql

from sre_evals.runner.config import ExperimentConfig
from sre_shared.embed.base import SPECS
from sre_shared.models import EmbeddingModelName

# Grid parameter -> (type, ExperimentConfig field it sets). Weights are nested in the config.
SWEEP_PARAMS: dict[str, type] = {
    "w_dense": float,
    "w_keyword": float,
    "rrf_k": int,
    "ef_search": int,
    "candidates_per_leg": int,
}

# The migrations' index definitions; a sweep restores whatever definition it found instead,
# and falls back to these only when the index was missing.
MIGRATION_HNSW: dict[str, tuple[int, int]] = {
    "gte-small": (16, 128),
    "bge-large": (16, 64),
    "bge-m3": (16, 64),
}
# pgvector limits: m in 2..100, ef_construction in 4..1000 and at least 2 * m.
M_RANGE = (2, 100)
EF_CONSTRUCTION_RANGE = (4, 1000)


def parse_grid(specs: Sequence[str]) -> dict[str, list[float | int]]:
    """`["w_keyword=0,0.5,1", "rrf_k=10,60"]` -> {"w_keyword": [0.0, 0.5, 1.0], "rrf_k": [10, 60]}."""
    grid: dict[str, list[float | int]] = {}
    for spec in specs:
        name, sep, values = spec.partition("=")
        name = name.strip()
        if not sep or name not in SWEEP_PARAMS:
            raise ValueError(
                f"bad grid {spec!r}; use NAME=v1,v2 with NAME in {sorted(SWEEP_PARAMS)}"
            )
        if name in grid:
            raise ValueError(f"grid parameter {name!r} given twice")
        cast = SWEEP_PARAMS[name]
        try:
            parsed = [cast(v) for v in values.split(",") if v.strip()]
        except ValueError as e:
            raise ValueError(f"bad value in {spec!r}: {e}") from e
        if not parsed:
            raise ValueError(f"no values in {spec!r}")
        grid[name] = parsed
    if not grid:
        raise ValueError("give at least one --grid NAME=v1,v2")
    return grid


def grid_points(grid: dict[str, list[float | int]]) -> list[dict[str, float | int]]:
    """Cartesian product in the order the parameters and values were given."""
    names = list(grid)
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*grid.values())]


def _fmt(v: float | int) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def point_name(base: str, params: Mapping[str, float | int]) -> str:
    return f"{base}[{','.join(f'{k}={_fmt(v)}' for k, v in params.items())}]"


def derive(base: ExperimentConfig, params: dict[str, float | int]) -> ExperimentConfig:
    """The base config with the point's parameters applied, named after the point."""
    update: dict[str, Any] = {"name": point_name(base.name, params)}
    weights = base.weights.model_copy()
    for key, value in params.items():
        if key == "w_dense":
            weights.dense = float(value)
        elif key == "w_keyword":
            weights.keyword = float(value)
        else:
            update[key] = value
    update["weights"] = weights
    return ExperimentConfig.model_validate({**base.model_dump(), **update})


def hnsw_index_name(model: EmbeddingModelName) -> str:
    return f"chunks_{SPECS[model].column}_hnsw"


def check_hnsw_params(m: int, ef_construction: int) -> None:
    if not M_RANGE[0] <= m <= M_RANGE[1]:
        raise ValueError(f"m={m} outside {M_RANGE}")
    if not EF_CONSTRUCTION_RANGE[0] <= ef_construction <= EF_CONSTRUCTION_RANGE[1]:
        raise ValueError(f"ef_construction={ef_construction} outside {EF_CONSTRUCTION_RANGE}")
    if ef_construction < 2 * m:
        raise ValueError(f"ef_construction={ef_construction} must be at least 2 * m = {2 * m}")


def hnsw_ddl(
    model: EmbeddingModelName, m: int, ef_construction: int
) -> tuple[sql.Composed, sql.Composed]:
    """(drop, create) statements for the model's HNSW index; identifiers come from SPECS and
    the parameters are checked integers, so nothing user-typed reaches the SQL text."""
    check_hnsw_params(m, ef_construction)
    index = sql.Identifier(hnsw_index_name(model))
    drop = sql.SQL("drop index if exists public.{}").format(index)
    create = sql.SQL(
        "create index {} on public.chunks using hnsw ({} extensions.halfvec_ip_ops)"
        " with (m = {}, ef_construction = {})"
    ).format(
        index, sql.Identifier(SPECS[model].column), sql.Literal(m), sql.Literal(ef_construction)
    )
    return drop, create


async def current_hnsw_params(
    conn: AsyncConnection[Any], model: EmbeddingModelName
) -> tuple[int, int]:
    """(m, ef_construction) of the model's index as it stands, or the migration's if absent."""
    row = await (
        await conn.execute(
            "select reloptions from pg_class where relname = %s", [hnsw_index_name(model)]
        )
    ).fetchone()
    opts = dict(o.split("=", 1) for o in (row["reloptions"] or [])) if row else {}
    m_default, efc_default = MIGRATION_HNSW[model]
    return int(opts.get("m", m_default)), int(opts.get("ef_construction", efc_default))


async def build_hnsw(
    conn: AsyncConnection[Any], model: EmbeddingModelName, m: int, ef_construction: int
) -> dict[str, Any]:
    """Rebuild the model's HNSW index with these parameters (autocommit connection). Returns
    the build facts stored with each sweep point."""
    drop, create = hnsw_ddl(model, m, ef_construction)
    # Single-threaded: a parallel build keeps the graph in dynamic shared memory, and the
    # container's /dev/shm is too small for it ("could not resize shared memory segment",
    # which also left the index dropped). One worker also makes build times comparable.
    await conn.execute("set max_parallel_maintenance_workers = 0")
    await conn.execute("set maintenance_work_mem = '256MB'")
    await conn.execute(drop)
    t0 = time.perf_counter()
    await conn.execute(create)
    seconds = time.perf_counter() - t0
    row = await (
        await conn.execute(
            "select pg_relation_size(%s::regclass) as bytes", [f"public.{hnsw_index_name(model)}"]
        )
    ).fetchone()
    assert row is not None
    return {
        "model": model,
        "m": m,
        "ef_construction": ef_construction,
        "build_seconds": round(seconds, 2),
        "index_mb": round(int(row["bytes"]) / 1_000_000, 1),
    }
