"""`sre-eval` command line: retrieval runs, results report, baseline."""

from __future__ import annotations

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
from sre_evals.runner.engine import ANN_SAMPLE
from sre_evals.runner.gate import DEFAULT_BASELINE
from sre_shared.aio import run
from sre_shared.db import connect, create_pool
from sre_shared.golden_io import golden_path, golden_version, load_golden
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
    baseline: Annotated[Path, typer.Option(help="Baseline file for --gate.")] = DEFAULT_BASELINE,
    concurrency: Annotated[int, typer.Option(help="Concurrent retrievals.")] = 4,
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
                          concurrency, gate, base))  # fmt: skip
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
) -> bool:
    from sre_core.retrieval.service import Retriever

    pool = create_pool(dsn)
    await pool.open()
    conn = await connect(dsn, autocommit=True)
    failed = False
    try:
        retriever = Retriever(pool)
        for cfg in selected:
            outcome = await run_experiment(
                cfg, all_items, retriever, conn, loop=loop, golden_version=gversion,
                concurrency=concurrency, ann=ann, ann_sample=ann_sample,
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
    database_url: DbOption = None,
) -> None:
    """Write the gate baseline from a finished run."""

    async def go() -> dict[str, Any]:
        conn = await connect(database_url or get_settings().database_url, autocommit=True)
        try:
            row = await (
                await conn.execute(
                    "select id, experiment, stamps, golden_version, corpus_version_id, summary"
                    " from public.eval_runs where id = %s and kind = 'retrieval' and status = 'done'",
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
        )

    data = run(go())
    write_baseline(data, out)
    console.print(f"baseline written to {out}: {data['metrics']}")
