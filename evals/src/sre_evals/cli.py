"""`sre-eval` command line: retrieval runs, parameter sweeps, results report, baseline."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from sre_evals.report import build_reports
from sre_evals.runner import (
    check_gate,
    load_baseline,
    load_experiments,
    make_baseline,
    run_experiment,
    write_baseline,
)
from sre_evals.runner.engine import ANN_EF_VALUES, ANN_SAMPLE
from sre_evals.runner.gate import DEFAULT_BASELINE
from sre_evals.runner.sweep import (
    SWEEP_PARAMS,
    build_hnsw,
    check_hnsw_params,
    current_hnsw_params,
    derive,
    grid_points,
    parse_grid,
    point_name,
)
from sre_shared.aio import run
from sre_shared.db import connect, create_pool
from sre_shared.golden_io import golden_path, golden_version, load_golden
from sre_shared.models import EmbeddingModelName, GoldenItem
from sre_shared.settings import REPO_ROOT, get_settings

app = typer.Typer(help="Run evals and build results reports.", no_args_is_help=True)
console = Console(highlight=False)

DbOption = Annotated[
    str | None, typer.Option("--database-url", help="Postgres DSN (default: DATABASE_URL).")
]


@app.callback()
def main() -> None:
    """Run evals and build results reports."""


@app.command("retrieval")
def retrieval(
    experiment: Annotated[str | None, typer.Option("--experiment", "-e")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Run every experiment.")] = False,
    golden: Annotated[
        str, typer.Option(help="Golden set directory name under evals/golden.")
    ] = "v1",
    loop: Annotated[str, typer.Option(help="local | ci")] = "local",
    ann_recall: Annotated[
        bool, typer.Option("--ann-recall", help="Also measure ANN recall.")
    ] = False,
    ann_sample: Annotated[int, typer.Option(help="Items sampled for ANN recall.")] = ANN_SAMPLE,
    gate: Annotated[bool, typer.Option("--gate", help="Fail on a drop vs the baseline.")] = False,
    exact: Annotated[
        bool,
        typer.Option(
            "--exact",
            help="Dense leg as an exact scan: deterministic metrics (the CI gate); stored as"
            " kind retrieval-exact, so the README table keeps the index-backed runs.",
        ),
    ] = False,
    baseline: Annotated[Path, typer.Option(help="Baseline file for --gate.")] = DEFAULT_BASELINE,
    concurrency: Annotated[
        int,
        typer.Option(
            help="Concurrent retrievals; above 1 runs faster but latency then includes queueing."
        ),
    ] = 1,
    database_url: DbOption = None,
) -> None:
    """Run experiment configs through the Retriever and store scores."""
    if bool(experiment) == all_:
        raise typer.BadParameter("pass exactly one of --experiment NAME or --all")
    if loop not in {"local", "ci"}:
        raise typer.BadParameter("--loop must be local or ci")
    configs = load_experiments()
    if experiment:
        if experiment not in configs:
            raise typer.BadParameter(f"unknown experiment {experiment!r}; have {sorted(configs)}")
        selected = [configs[experiment]]
    else:
        selected = list(configs.values())

    path = golden_path(golden)
    items = load_golden(path, accepted_only=True) if path.exists() else []
    scorable = [i for i in items if i.gold_passages]
    if not scorable:
        msg = f"golden set {golden!r} has no accepted items with gold passages"
        if gate:
            console.print(f"NOTICE: {msg}; gate skipped")
            raise typer.Exit(0)
        raise typer.BadParameter(msg)
    base = load_baseline(baseline) if gate else {}
    if gate and not base:
        console.print("NOTICE: baseline is empty ({}); the run is recorded but the gate is skipped")
    gversion = golden_version(path)
    dsn = database_url or get_settings().database_url
    failed = run(_run_all(dsn, selected, scorable, items, loop, gversion, ann_recall, ann_sample,
                          concurrency, gate, base, exact))  # fmt: skip
    if failed:
        raise typer.Exit(1)


async def _run_all(
    dsn: str,
    selected: list[Any],
    scorable: list[Any],
    all_items: list[Any],
    loop: str,
    gversion: str,
    ann: bool,
    ann_sample: int,
    concurrency: int,
    gate: bool,
    base: dict[str, Any],
    exact: bool = False,
) -> bool:

    # Open every connection before timing: a connect inside a timed query (the lab's port
    # forward drops some) showed up as multi-second latency outliers.
    size = max(2, concurrency)
    pool = create_pool(dsn, min_size=size, max_size=size)
    await pool.open(wait=True)
    conn = await connect(dsn, autocommit=True)
    failed = False
    try:
        retrievers: dict[str | None, Any] = {}
        for cfg in selected:
            retriever = retriever_for(pool, cfg, retrievers)
            outcome = await run_experiment(
                cfg, all_items, retriever, conn, loop=loop, golden_version=gversion,
                concurrency=concurrency, ann=ann, ann_sample=ann_sample,
                ann_ef_values=sorted({*ANN_EF_VALUES, cfg.ef_search}),
                kind="retrieval-exact" if exact else "retrieval", exact=exact,
            )  # fmt: skip
            s = outcome.summary
            console.print(
                f"{cfg.name}: run {outcome.run_id}  recall@5 {s['recall@5']:.3f}  "
                f"MRR@10 {s['mrr@10']:.3f}  nDCG@10 {s['ndcg@10']:.3f}  "
                f"p95 {s['latency_ms']['p95']:.0f} ms  items {s['n_scored']}"
            )
            if gate and base and base.get("experiment") == cfg.name:
                result = check_gate(cfg.name, gversion, s, base)
                for m in result.messages:
                    console.print(f"  gate: {m}")
                console.print(f"  gate {result.status.upper()}")
                failed = failed or not result.ok
        if gate and base and base.get("experiment") not in {c.name for c in selected}:
            console.print(
                f"  gate FAIL: baseline experiment {base.get('experiment')!r} was not run"
            )
            failed = True
    finally:
        await conn.close()
        await pool.close()
    return failed


def retriever_for(pool: Any, cfg: Any, cache: dict[str | None, Any]) -> Any:
    """One Retriever per reranker model, so experiments without a reranker never load one."""
    from sre_core.retrieval.rerank import RemoteReranker
    from sre_core.retrieval.service import Retriever

    key = cfg.rerank.model if cfg.rerank else None
    if key not in cache:
        cache[key] = Retriever(pool, reranker=RemoteReranker(key) if key else None)
    return cache[key]


def _scorable(golden: str) -> tuple[list[GoldenItem], list[GoldenItem], str]:
    path = golden_path(golden)
    items = load_golden(path, accepted_only=True) if path.exists() else []
    scorable = [i for i in items if i.gold_passages]
    if not scorable:
        raise typer.BadParameter(f"golden set {golden!r} has no accepted items with gold passages")
    return items, scorable, golden_version(path)


def _ints(text: str, what: str) -> list[int]:
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError as e:
        raise typer.BadParameter(f"{what}: {e}") from e
    if not values:
        raise typer.BadParameter(f"{what}: no values")
    return values


@app.command("sweep")
def sweep(
    base: Annotated[str, typer.Option(help="Experiment to vary, e.g. hybrid-gte-small.")],
    grid: Annotated[
        list[str],
        typer.Option("--grid", help=f"NAME=v1,v2 (repeatable); NAME in {sorted(SWEEP_PARAMS)}"),
    ],
    name: Annotated[str | None, typer.Option(help="Sweep name stored with each point.")] = None,
    golden: Annotated[str, typer.Option(help="Golden set directory under evals/golden.")] = "v1",
    ann_recall: Annotated[
        bool, typer.Option("--ann-recall", help="ANN recall at each point's ef_search, all items.")
    ] = False,
    database_url: DbOption = None,
) -> None:
    """Run every point of a parameter grid over one experiment (stored as kind `sweep`)."""
    configs = load_experiments()
    if base not in configs:
        raise typer.BadParameter(f"unknown experiment {base!r}; have {sorted(configs)}")
    try:
        points = grid_points(parse_grid(grid))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    items, scorable, gversion = _scorable(golden)
    sweep_name = name or base
    dsn = database_url or get_settings().database_url

    async def go() -> None:

        pool = create_pool(dsn, min_size=2, max_size=2)
        await pool.open(wait=True)
        conn = await connect(dsn, autocommit=True)
        try:
            retrievers: dict[str | None, Any] = {}
            for params in points:
                cfg = derive(configs[base], params)
                retriever = retriever_for(pool, cfg, retrievers)
                outcome = await run_experiment(
                    cfg, items, retriever, conn, loop="local", golden_version=gversion,
                    ann=ann_recall, ann_sample=len(scorable), ann_ef_values=[cfg.ef_search],
                    kind="sweep", extra={"sweep": sweep_name, "params": params},
                )  # fmt: skip
                _print_point(cfg.name, outcome.run_id, outcome.summary)
        finally:
            await conn.close()
            await pool.close()

    run(go())


@app.command("hnsw")
def hnsw(
    model: Annotated[
        str, typer.Option(help="Embedding model whose index is rebuilt.")
    ] = "gte-small",
    m: Annotated[str, typer.Option("--m", help="Comma-separated m values.")] = "8,16,32",
    ef_construction: Annotated[str, typer.Option(help="Comma-separated values.")] = "32,64,128,256",
    ef_search: Annotated[str, typer.Option(help="Comma-separated values.")] = "20,40,100,200",
    repeat: Annotated[
        int,
        typer.Option(
            help="Builds per (m, ef_construction); above 1 measures build-to-build variance."
        ),
    ] = 1,
    golden: Annotated[str, typer.Option(help="Golden set directory under evals/golden.")] = "v1",
    database_url: DbOption = None,
) -> None:
    """Rebuild the model's HNSW index for each (m, ef_construction), measure quality, ANN recall
    and latency at each ef_search (kind `hnsw`), then restore the index as it was. With
    --repeat N each combination is built N times (sweep `hnsw-<model>-repeat`): HNSW builds
    are randomised, so the same parameters give slightly different graphs."""
    from sre_shared.embed.base import SPECS

    if model not in SPECS:
        raise typer.BadParameter(f"unknown model {model!r}; have {sorted(SPECS)}")
    name: EmbeddingModelName = model
    configs = load_experiments()
    base = configs.get(f"dense-{model}")
    if base is None:
        raise typer.BadParameter(f"no experiment dense-{model} to sweep")
    builds = []
    for mv in _ints(m, "--m"):
        for efc in _ints(ef_construction, "--ef-construction"):
            try:
                check_hnsw_params(mv, efc)
            except ValueError as e:
                console.print(f"skip m={mv}, ef_construction={efc}: {e}")
                continue
            builds.append((mv, efc))
    efs = _ints(ef_search, "--ef-search")
    if repeat < 1:
        raise typer.BadParameter("--repeat must be at least 1")
    items, scorable, gversion = _scorable(golden)
    dsn = database_url or get_settings().database_url

    async def go() -> None:
        from sre_core.retrieval.service import Retriever

        pool = create_pool(dsn, min_size=2, max_size=2)
        await pool.open(wait=True)
        conn = await connect(dsn, autocommit=True)
        original = await current_hnsw_params(conn, name)
        try:
            retriever = Retriever(pool)
            sweep_name = f"hnsw-{model}" if repeat == 1 else f"hnsw-{model}-repeat"
            for (mv, efc), build_no in itertools.product(builds, range(1, repeat + 1)):
                info = await build_hnsw(conn, name, mv, efc)
                console.print(
                    f"built m={mv} ef_construction={efc}: {info['build_seconds']:.1f} s,"
                    f" {info['index_mb']:.1f} MB (single-threaded)"
                )
                for ef in efs:
                    params: dict[str, int] = {"m": mv, "ef_construction": efc, "ef_search": ef}
                    if repeat > 1:
                        params["build"] = build_no
                    cfg = derive(base, {"ef_search": ef}).model_copy(
                        update={"name": point_name(base.name, params)}
                    )
                    outcome = await run_experiment(
                        cfg, items, retriever, conn, loop="local", golden_version=gversion,
                        ann=True, ann_sample=len(scorable), ann_ef_values=[ef], kind="hnsw",
                        extra={"sweep": sweep_name, "params": params, "hnsw": info},
                    )  # fmt: skip
                    _print_point(cfg.name, outcome.run_id, outcome.summary)
        finally:
            try:
                restored = await build_hnsw(conn, name, *original)
                console.print(
                    f"restored the index as it was (m={original[0]},"
                    f" ef_construction={original[1]}) in {restored['build_seconds']:.1f} s"
                )
            except Exception:
                console.print(
                    f"[red]the {model} HNSW index may be missing: restore it with"
                    f" `sre-eval hnsw --model {model} --m {original[0]}"
                    f" --ef-construction {original[1]} --ef-search 100`[/]"
                )
                raise
            finally:
                await conn.close()
                await pool.close()

    run(go())


def _print_point(label: str, run_id: int, s: dict[str, Any]) -> None:
    ann = s.get("ann_recall") or {}
    ann_txt = "  ann " + "/".join(f"{v:.3f}" for v in ann.values()) if ann else ""
    console.print(  # markup off: point names contain [brackets]
        f"{label}: run {run_id}  recall@5 {s['recall@5']:.3f}  MRR@10 {s['mrr@10']:.3f}  "
        f"p50 {s['latency_ms']['p50']:.0f} ms  p95 {s['latency_ms']['p95']:.0f} ms{ann_txt}",
        markup=False,
    )


@app.command("report")
def report(
    out: Annotated[Path, typer.Option(help="README to update (only the results block).")] = (
        REPO_ROOT / "README.md"
    ),
    details: Annotated[Path, typer.Option(help="Details page to write.")] = (
        REPO_ROOT / "docs" / "results.md"
    ),
    database_url: DbOption = None,
) -> None:
    """Rewrite the README results block and docs/results.md from the latest runs."""

    async def go() -> int:
        conn = await connect(database_url or get_settings().database_url, autocommit=True)
        try:
            return len(await build_reports(conn, out, details))
        finally:
            await conn.close()

    n = run(go())
    console.print(f"report written from {n} experiment run(s)")


@app.command("baseline")
def baseline_cmd(
    from_run: Annotated[int, typer.Option("--from-run", help="eval_runs.id to freeze.")],
    out: Annotated[Path, typer.Option(help="Baseline file.")] = DEFAULT_BASELINE,
    ann_floor: Annotated[
        float | None,
        typer.Option(help="Minimum ANN recall@10 at the run's ef_search for every gated run."),
    ] = None,
    database_url: DbOption = None,
) -> None:
    """Write the gate baseline from a finished run (an --exact run for a deterministic gate)."""

    async def go() -> dict[str, Any]:
        conn = await connect(database_url or get_settings().database_url, autocommit=True)
        try:
            row = await (
                await conn.execute(
                    "select id, experiment, config, stamps, golden_version, corpus_version_id,"
                    " summary from public.eval_runs where id = %s"
                    " and kind in ('retrieval', 'retrieval-exact') and status = 'done'",
                    [from_run],
                )
            ).fetchone()
        finally:
            await conn.close()
        if row is None:
            raise typer.BadParameter(f"no finished retrieval run with id {from_run}")
        return make_baseline(
            row["experiment"],
            row["golden_version"],
            row["summary"],
            run_id=row["id"],
            git_sha=row["stamps"].get("git_sha", "unknown"),
            corpus_version_id=row["corpus_version_id"],
            config_hash=row["stamps"].get("config_hash", ""),
            ann_floor=(int(row["config"]["ef_search"]), ann_floor) if ann_floor else None,
        )

    data = run(go())
    write_baseline(data, out)
    console.print(f"baseline written to {out}: {data['metrics']}")
