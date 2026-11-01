"""`sre-golden` command line: pick, synthesize, review, validate, mirror."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from sre_evals.golden.contexts import pick_contexts, read_contexts, write_contexts
from sre_evals.golden.llm import OpenAICompatLLM, parse_model_spec
from sre_evals.golden.mirror import mirror_dataset
from sre_evals.golden.review import (
    ReviewDecision,
    ReviewSession,
    StdTerminal,
    Terminal,
    apply_decisions,
    run_review,
)
from sre_evals.golden.synth import (
    DEFAULT_QUALITY_THRESHOLD,
    QuotaError,
    a_synthesize,
    check_quota,
)
from sre_evals.golden.validate import validate_file
from sre_shared.config import load_limits
from sre_shared.db import connect
from sre_shared.golden_io import write_golden

app = typer.Typer(help="Build, review and validate the golden set.", no_args_is_help=True)
console = Console()

DsnOpt = Annotated[
    str | None, typer.Option("--dsn", help="Postgres DSN (default: SRE_DATABASE_URL setting)")
]


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    if sys.platform == "win32":
        # psycopg async needs a selector event loop on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def make_llm(model: str) -> OpenAICompatLLM:
    """Seam for tests: build the DeepEval model for a `provider:name` spec."""
    return OpenAICompatLLM(parse_model_spec(model))


def make_terminal() -> Terminal:
    """Seam for tests: the real keyboard."""
    return StdTerminal()


@app.callback()
def main() -> None:
    """Build, review and validate the golden set."""
    # DeepEval prints emoji; a redirected stdout on Windows defaults to cp1252 and would crash.
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")


@app.command()
def pick(
    stratum: Annotated[str, typer.Option(help="factual | exact_string")],
    out: Annotated[Path, typer.Option(help="contexts JSONL to write")],
    n: Annotated[int, typer.Option(help="how many contexts")] = 50,
    versions: Annotated[list[str] | None, typer.Option(help="k8s versions (repeatable)")] = None,
    seed: Annotated[int, typer.Option(help="selection is deterministic per seed")] = 0,
    dsn: DsnOpt = None,
) -> None:
    """Select context chunks from the database for synthesis."""
    if stratum not in ("factual", "exact_string"):
        raise typer.BadParameter("stratum must be factual or exact_string")

    async def go() -> list[Any]:
        conn = await connect(dsn, autocommit=True)
        try:
            return await pick_contexts(conn, stratum, n, versions, seed)  # type: ignore[arg-type]
        finally:
            await conn.close()

    rows = _run(go())
    write_contexts(out, rows)
    console.print(f"picked {len(rows)} {stratum} contexts (asked {n}) -> {out}")
    if len(rows) < n:
        console.print("[yellow]fewer contexts than requested: the corpus has no more matches[/]")


@app.command()
def synthesize(
    contexts: Annotated[Path, typer.Option(help="contexts JSONL from `pick`")],
    model: Annotated[str, typer.Option(help="ollama:qwen2.5:7b | groq:openai/gpt-oss-20b")],
    out: Annotated[Path, typer.Option(help="candidates JSONL to write")],
    evolutions: Annotated[int, typer.Option(help="DeepEval evolutions per question")] = 0,
    quality_threshold: Annotated[
        float, typer.Option(help="critic score below which a candidate is rejected")
    ] = DEFAULT_QUALITY_THRESHOLD,
    id_start: Annotated[int, typer.Option(help="first id number per stratum")] = 1,
    max_concurrent: Annotated[int, typer.Option(help="parallel model calls")] = 4,
) -> None:
    """Generate candidate questions, one per context chunk, with DeepEval's Synthesizer."""
    rows = read_contexts(contexts)
    if not rows:
        console.print("[red]no contexts to synthesize[/]")
        raise typer.Exit(1)
    spec = parse_model_spec(model)
    try:
        est = check_quota(spec, rows, load_limits(), evolutions=evolutions)
    except QuotaError as e:
        console.print(f"[red]quota guard:[/] {e}")
        raise typer.Exit(2) from e
    console.print(f"{len(rows)} contexts, ~{est} tokens planned on {spec.label}")
    llm = make_llm(model)

    async def go() -> list[tuple[str, int, Any]]:
        # One event loop for every stratum: the async HTTP client's connections belong to the
        # loop that opened them, so a second asyncio.run() leaves it talking to a closed loop.
        done: list[tuple[str, int, Any]] = []
        for stratum in sorted({r.stratum for r in rows}):
            group = [r for r in rows if r.stratum == stratum]
            res = await a_synthesize(
                group,
                llm,
                evolutions=evolutions,
                threshold=quality_threshold,
                id_start=id_start,
                max_concurrent=max_concurrent,
            )
            done.append((stratum, len(group), res))
        return done

    items = []
    for stratum, n_group, res in _run(go()):
        items.extend(res.items)
        console.print(
            f"{stratum}: {len(res.items)} kept of {n_group}; "
            f"filter rejected {res.filter_rejected[stratum]}; "
            f"exact-string repaired {res.exact_repaired}, rejected {res.exact_rejected}; "
            f"{res.llm_calls} model calls"
        )
        for key, why in res.skipped:
            console.print(f"  skipped {key[:12]}: {why}")
    write_golden(out, items)
    console.print(f"wrote {len(items)} candidates -> {out}")
    if not items:
        raise typer.Exit(1)


@app.command()
def review(
    in_: Annotated[Path, typer.Option("--in", help="candidates JSONL")],
    out: Annotated[Path, typer.Option(help="decisions file, e.g. evals/golden/v1/items.jsonl")],
    reviewer: Annotated[str, typer.Option(help="your name, stored on each decision")],
) -> None:
    """Accept, reject, rewrite, trim and edit candidates one at a time (resumable)."""
    session = ReviewSession.open(in_, out, reviewer)
    run_review(session, make_terminal(), console)
    console.print(f"saved {len(session.decided)} decisions -> {out}")


@app.command()
def apply(
    in_: Annotated[Path, typer.Option("--in", help="candidates JSONL")],
    decisions: Annotated[Path, typer.Option(help="decisions JSONL (ReviewDecision per line)")],
    out: Annotated[Path, typer.Option(help="decisions file, e.g. evals/golden/v1/items.jsonl")],
    reviewer: Annotated[str, typer.Option(help="who decided, stored on each decision")],
) -> None:
    """Apply a decisions file non-interactively, with the same rules as `review`."""
    rows = [
        ReviewDecision.model_validate_json(line)
        for line in decisions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    session = ReviewSession.open(in_, out, reviewer)
    report = apply_decisions(session, rows)
    console.print(
        f"accepted {report.accepted}, rejected {report.rejected}, "
        f"rewritten share {session.rewritten_share():.0%} -> {out}"
    )
    for err in report.errors:
        console.print(f"[red]{err}[/]")
    if report.errors:
        raise typer.Exit(1)


@app.command()
def validate(
    in_: Annotated[Path, typer.Option("--in", help="golden items JSONL")],
    dsn: DsnOpt = None,
) -> None:
    """Check every accepted item against the documents; non-zero exit on any failure."""

    async def go() -> Any:
        conn = await connect(dsn, autocommit=True)
        try:
            return await validate_file(conn, in_)
        finally:
            await conn.close()

    report = _run(go())
    table = Table(title=f"{in_}")
    for col in ("stratum", "accepted", "rejected", "candidate"):
        table.add_column(col, justify="left" if col == "stratum" else "right")
    for stratum, c in sorted(report.counts.items()):
        table.add_row(stratum, str(c["accepted"]), str(c["rejected"]), str(c["candidate"]))
    console.print(table)
    for err in report.errors:
        console.print(f"[red]FAIL[/] {err}")
    if not report.ok:
        console.print(f"[red]{len(report.errors)} problem(s)[/]")
        raise typer.Exit(1)
    console.print("[green]valid[/]")


@app.command()
def mirror(
    in_: Annotated[Path, typer.Option("--in", help="golden items JSONL")],
    dataset: Annotated[str, typer.Option(help="LangSmith dataset name")] = "sre-copilot-golden-v1",
) -> None:
    """Upsert accepted items to a LangSmith dataset (skips when no API key is set)."""
    res = mirror_dataset(in_, dataset)
    if res.skipped_reason:
        console.print(f"[yellow]{res.skipped_reason}[/]")
        return
    console.print(f"mirrored {res.mirrored} items to {dataset}")
