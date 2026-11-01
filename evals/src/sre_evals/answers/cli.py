"""`sre-eval answers run | label | calibrate`."""

from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from sre_evals.answers.calibrate import (
    JudgeHandle,
    calibrate,
    calibration_summary,
    store_calibration,
)
from sre_evals.answers.judge import Judge, build_judge, check_judge_budget
from sre_evals.answers.judge_cache import CachedJudge
from sre_evals.answers.label import (
    LabelSession,
    labellable,
    load_details,
    read_labels,
    run_label,
    stratified_sample,
)
from sre_evals.answers.runner import plan_judge_tokens, run_answers
from sre_evals.golden.review import StdTerminal
from sre_evals.runner import ExperimentConfig, experiments_dir, load_experiment
from sre_shared.aio import run
from sre_shared.config import load_answer_config
from sre_shared.db import connect, create_pool
from sre_shared.golden_io import golden_path, golden_version, load_golden
from sre_shared.llm import ChatClient, QuotaExceeded, parse_model_spec
from sre_shared.models import AnswerConfig, GoldenItem
from sre_shared.settings import get_settings

app = typer.Typer(help="Answer evals: run, label, calibrate.", no_args_is_help=True)
console = Console(highlight=False)

DbOption = Annotated[
    str | None, typer.Option("--database-url", help="Postgres DSN (default: DATABASE_URL).")
]


def _client(spec: str) -> ChatClient:
    try:
        return ChatClient(parse_model_spec(spec))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _judge(spec: str) -> tuple[Judge, ChatClient]:
    """`provider:name` is DeepEval's faithfulness (three calls); `rubric:provider:name` is
    the one-call rubric judge on that model (relevancy still DeepEval's)."""
    from sre_evals.answers.rubric_judge import PREFIX, RubricJudge

    client = _client(spec.removeprefix(PREFIX))
    deepeval = build_judge(client)
    if spec.startswith(PREFIX):
        return RubricJudge(client, deepeval), client
    return deepeval, client


def load_answer_experiments() -> dict[str, ExperimentConfig]:
    """Answer experiments live in `config/experiments/answers/`, apart from the retrieval ones
    (the retrieval loader reads only the top-level directory, so `retrieval --all` skips them)."""
    folder = experiments_dir() / "answers"
    return {c.name: c for c in (load_experiment(p) for p in sorted(folder.glob("*.yaml")))}


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def select_items(
    items: list[GoldenItem], ids: str | None, subset: int | None, seed: int
) -> list[GoldenItem]:
    """`--items a,b` (in that order) or a seeded random `--subset N` (id order), else all."""
    if ids and subset:
        raise typer.BadParameter("pass --items or --subset, not both")
    if ids:
        by_id = {i.id: i for i in items}
        wanted = [x.strip() for x in ids.split(",") if x.strip()]
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise typer.BadParameter(f"unknown golden item ids: {missing}")
        return [by_id[w] for w in wanted]
    if subset:
        picked = random.Random(seed).sample(  # noqa: S311
            sorted(items, key=lambda i: i.id), min(subset, len(items))
        )
        return sorted(picked, key=lambda i: i.id)
    return items


def effective_answer_config(block: AnswerConfig | None) -> AnswerConfig:
    """config/answer.yaml with the experiment's own `answer:` keys on top. Unset keys keep the
    yaml value, not the model default, so a default changed there reaches every experiment."""
    base = load_answer_config()
    if block is None:
        return base
    return base.model_copy(update=block.model_dump(exclude_unset=True))


@app.command("run")
def run_cmd(
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="answer-*.yaml name.")],
    model: Annotated[str | None, typer.Option(help="Answer model, provider:name.")] = None,
    judge: Annotated[
        str | None, typer.Option(help="Judge model, provider:name (default: no LLM metrics).")
    ] = None,
    repeat: Annotated[int, typer.Option(help="Runs of the same commit (noise).")] = 1,
    items: Annotated[str | None, typer.Option(help="Comma-separated golden item ids.")] = None,
    subset: Annotated[int | None, typer.Option(help="Random subset of N items.")] = None,
    seed: Annotated[int, typer.Option(help="Seed for --subset.")] = 0,
    golden: Annotated[str, typer.Option(help="Golden set directory under evals/golden.")] = "v1",
    loop: Annotated[str, typer.Option(help="local | ci")] = "local",
    concurrency: Annotated[int, typer.Option(help="Concurrent answers.")] = 1,
    database_url: DbOption = None,
) -> None:
    """Answer the golden items with an AnswererLike and score the answers."""
    configs = load_answer_experiments()
    if experiment not in configs:
        raise typer.BadParameter(f"unknown experiment {experiment!r}; have {sorted(configs)}")
    cfg = configs[experiment]
    if repeat < 1:
        raise typer.BadParameter("--repeat must be at least 1")
    if loop not in {"local", "ci"}:
        raise typer.BadParameter("--loop must be local or ci")
    answer_cfg = effective_answer_config(cfg.answer)
    if model:
        answer_cfg = answer_cfg.model_copy(update={"model": model})
    path = golden_path(golden)
    all_items = load_golden(path, accepted_only=True) if path.exists() else []
    chosen = select_items(all_items, items, subset, seed)
    if not chosen:
        raise typer.BadParameter(f"golden set {golden!r} has no accepted items to answer")

    judge_obj: Judge | None = None
    if judge:
        judge_obj, client = _judge(judge)
        planned = plan_judge_tokens(answer_cfg, len(chosen) * repeat, judge_obj)
        try:  # refuse up front: no answer is generated if the judge cannot finish the run
            check_judge_budget(client, planned)
        except QuotaExceeded as e:
            console.print(f"refused: {e}")
            raise typer.Exit(2) from e
    gversion = golden_version(path)
    dsn = database_url or get_settings().database_url
    run(_run(dsn, cfg, chosen, judge_obj, loop, gversion, answer_cfg, repeat, concurrency))


async def _run(
    dsn: str,
    cfg: Any,
    chosen: list[GoldenItem],
    judge: Judge | None,
    loop: str,
    gversion: str,
    answer_cfg: Any,
    repeat: int,
    concurrency: int,
) -> None:
    from sre_core.retrieval.service import Retriever

    # T06's factory, imported late (and untyped here) so this module loads without it.
    build_answerer = importlib.import_module("sre_core.answer").build_answerer

    size = max(2, concurrency)
    pool = create_pool(dsn, min_size=size, max_size=size)
    await pool.open(wait=True)
    conn = await connect(dsn, autocommit=True)
    try:
        from sre_core.retrieval.rerank import default_reranker

        reranker = default_reranker()
        if reranker is not None:
            await reranker.warm()
        retriever = Retriever(pool, reranker=reranker)
        answerer = build_answerer(retriever, config=answer_cfg)
        outcomes = await run_answers(
            cfg, chosen, answerer, conn, judge=judge, loop=loop, golden_version=gversion,
            answer_cfg=answer_cfg, repeat=repeat, concurrency=concurrency, use_cache=True,
            require_rerank=reranker is not None,
        )  # fmt: skip
        for o in outcomes:
            s = o.summary
            f = s["faithfulness"]
            r = s["answer_relevancy"]
            console.print(
                f"{cfg.name} repeat {o.repeat}/{repeat}: run {o.run_id}  "
                f"faithfulness {'-' if f is None else f'{f:.3f}'}  "
                f"relevancy {'-' if r is None else f'{r:.3f}'}  "
                f"citation hit {s['citation_hit_rate'] or 0:.3f}  abstain {s['abstain_rate']:.3f}  "
                f"answered {s['n_answered']}/{s['n_items']}"
            )
    finally:
        await conn.close()
        await pool.close()


@app.command("judge")
def judge_cmd(
    run_id: Annotated[int, typer.Option("--run", help="Answer run whose answers to judge.")],
    judge: Annotated[str, typer.Option(help="Judge model, provider:name.")],
    repeat: Annotated[int, typer.Option(help="Judge the same answers N times (noise).")] = 1,
    database_url: DbOption = None,
) -> None:
    """Judge a stored run's answers again (no generation); one new run per repeat."""
    from sre_evals.answers.rejudge import StoredRunError, rejudge_run

    if repeat < 1:
        raise typer.BadParameter("--repeat must be at least 1")
    judge_obj, _ = _judge(judge)
    dsn = database_url or get_settings().database_url

    async def go() -> None:
        conn = await connect(dsn, autocommit=True)
        try:
            outcomes = await rejudge_run(conn, run_id, judge_obj, repeat=repeat)
        except StoredRunError as e:
            console.print(f"refused: {e}")
            raise typer.Exit(2) from e
        finally:
            await conn.close()
        for o in outcomes:
            s = o.summary
            f, r = s["faithfulness"], s["answer_relevancy"]
            console.print(
                f"run {run_id} judged by {judge} ({o.repeat}/{repeat}): run {o.run_id}  "
                f"faithfulness {'-' if f is None else f'{f:.3f}'}  "
                f"relevancy {'-' if r is None else f'{r:.3f}'}"
            )

    run(go())


@app.command("label")
def label_cmd(
    run_id: Annotated[int, typer.Option("--run", help="eval_runs.id of an answer run.")],
    out: Annotated[Path, typer.Option(help="Label file (JSONL); resumed if it exists.")],
    labeller: Annotated[str, typer.Option(help="Your name, stored with each label.")],
    sample: Annotated[int, typer.Option(help="Answers to label.")] = 60,
    seed: Annotated[int, typer.Option(help="Sampling seed (same seed, same sample).")] = 1,
    database_url: DbOption = None,
) -> None:
    """Label the faithfulness of a stratified sample of a run's answers (s / p / n / u / x)."""

    async def load() -> list[Any]:
        conn = await connect(database_url or get_settings().database_url, autocommit=True)
        try:
            return await load_details(conn, run_id)
        finally:
            await conn.close()

    pool = labellable(run(load()))
    if not pool:
        raise typer.BadParameter(f"run {run_id} has no answered items to label")
    session = LabelSession.open(stratified_sample(pool, sample, seed), out, labeller)
    run_label(session, StdTerminal(), console)
    console.print(f"{len(session.decided)} labels in {out}")


@app.command("calibrate")
def calibrate_cmd(
    labels: Annotated[Path, typer.Option(help="Label file from `answers label`.")],
    judges: Annotated[str, typer.Option(help="Comma-separated provider:name judges.")],
    reference: Annotated[
        str | None, typer.Option(help="Judge (one of --judges) to compare the others with.")
    ] = None,
    store: Annotated[bool, typer.Option(help="Store the result for the report.")] = True,
    concurrency: Annotated[int, typer.Option(help="Concurrent judge calls.")] = 1,
    database_url: DbOption = None,
) -> None:
    """Re-judge the labelled answers with each judge; accuracy and kappa vs the human labels."""
    specs = [s.strip() for s in judges.split(",") if s.strip()]
    if not specs:
        raise typer.BadParameter("--judges: no values")
    given = read_labels(labels)
    if not given:
        raise typer.BadParameter(f"{labels}: no labels")
    handles = []
    for spec in specs:
        judge_obj, client = _judge(spec)
        # Cached across runs: a hosted judge's quota is spent once per answer (judge_cache).
        handles.append(JudgeHandle(spec, CachedJudge(judge_obj), client))

    async def go() -> None:
        conn = await connect(database_url or get_settings().database_url, autocommit=True)
        try:
            items = [
                it
                for rid in sorted({lab.run_id for lab in given})
                for it in await load_details(
                    conn, rid, [x.item_id for x in given if x.run_id == rid]
                )
            ]
            try:
                results = await calibrate(
                    given, items, handles, reference=reference, concurrency=concurrency
                )
            except QuotaExceeded as e:
                console.print(f"refused: {e}")
                raise typer.Exit(2) from e
            summary = calibration_summary(results, given, labels.name, reference)
            table = Table(title=f"judge calibration ({len(given)} labels)")
            for col in ("judge", "scored", "acc", "kappa", "acc (binary)", "kappa (binary)"):
                table.add_column(col)
            for r in results:
                table.add_row(
                    r.name, f"{r.n}/{r.n + r.n_failed}", _fmt(r.accuracy), _fmt(r.kappa),
                    _fmt(r.accuracy_binary), _fmt(r.kappa_binary),
                )  # fmt: skip
            console.print(table)
            for name in summary["below_target"]:
                console.print(
                    f"{name}: kappa below {summary['kappa_target']}: revise the judge prompt"
                )
            if store:
                rid = await store_calibration(conn, labels.stem, summary)
                console.print(f"stored as run {rid} (kind calibration)")
        finally:
            await conn.close()

    run(go())
