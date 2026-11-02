"""`sre-eval agent run | validate`."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console

from sre_evals.agent.fakes import Behavior, make_fake_rig_factory
from sre_evals.agent.items import (
    AGENT_GOLDEN_VERSION,
    item_set_version,
    items_path,
    load_items,
    validate_items,
)
from sre_evals.agent.report import render_agent_report
from sre_evals.agent.rig import FaultPlan, RigFactory
from sre_evals.agent.runner import AgentRunConfig, run_agent_evals
from sre_shared.aio import run
from sre_shared.config import load_mcp_ops_config, load_simulator_scenarios
from sre_shared.db import connect
from sre_shared.golden_io import golden_path, load_golden
from sre_shared.models import GoldenItem
from sre_shared.settings import get_settings

app = typer.Typer(help="Agent evals: run the agent-v1 task set, validate it.", no_args_is_help=True)
console = Console(highlight=False)

ItemsOption = Annotated[Path | None, typer.Option("--items", help="Task set (JSONL).")]


def _golden() -> dict[str, GoldenItem]:
    return {g.id: g for g in load_golden(golden_path("v1"))}


def _rig_factory(spec: str, golden: dict[str, GoldenItem]) -> RigFactory:
    """`fake` is the oracle agent over a fake cluster (CI). `module:function` is a callable
    returning a RigFactory; the lab wires the real agent and simulator there."""
    if spec == "fake":
        mcp = load_mcp_ops_config()
        return make_fake_rig_factory(
            load_simulator_scenarios().scenarios, mcp.policy(), golden, behavior=Behavior()
        )
    module, _, func = spec.partition(":")
    if not func:
        raise typer.BadParameter("use `fake` or `module:function`")
    factory: Any = getattr(importlib.import_module(module), func)()
    return factory  # type: ignore[no-any-return]


@app.command("validate")
def validate_cmd(items: ItemsOption = None) -> None:
    """Check the task set: schema, counts, reachable end states, golden references."""
    path = items or items_path()
    loaded = load_items(path)
    problems = validate_items(
        loaded,
        _golden(),
        load_simulator_scenarios().scenarios,
        load_mcp_ops_config().policy(),
    )
    for p in problems:
        console.print(f"[red]{p}[/red]")
    console.print(f"{len(loaded)} items, {len(problems)} problems")
    raise typer.Exit(1 if problems else 0)


@app.command("run")
def run_cmd(
    subset: Annotated[str, typer.Option(help="pr (the PR subset) or full.")] = "full",
    faults: Annotated[bool, typer.Option("--faults", help="Inject the three faults.")] = False,
    rig: Annotated[str, typer.Option(help="fake | module:function")] = "fake",
    items: ItemsOption = None,
    store: Annotated[bool, typer.Option("--store/--no-store")] = False,
    loop: Annotated[str, typer.Option(help="local | ci | nightly | hourly")] = "local",
    seed: Annotated[int, typer.Option(help="Simulator seed.")] = 7,
    report: Annotated[Path | None, typer.Option(help="Write the markdown section here.")] = None,
    min_success: Annotated[float, typer.Option(help="Exit 1 below this task success.")] = 0.0,
    database_url: Annotated[str | None, typer.Option("--database-url")] = None,
) -> None:
    """Run the task set. Exit 1 on any duplicate side effect (the Phase 3 gate), or when task
    success is below `--min-success`."""
    if subset not in ("pr", "full"):
        raise typer.BadParameter("subset is pr or full")
    path = items or items_path()
    chosen = load_items(path, subset=subset)  # type: ignore[arg-type]
    golden = _golden()
    factory = _rig_factory(rig, golden)
    plan = FaultPlan.all() if faults else FaultPlan()
    cfg = AgentRunConfig(name=AGENT_GOLDEN_VERSION, subset=subset, seed=seed, rig=rig)

    async def go() -> Any:
        conn = None
        if store:
            conn = await connect(database_url or get_settings().database_url, autocommit=True)
        try:
            return await run_agent_evals(
                chosen, factory, golden, cfg=cfg, faults=plan, conn=conn, loop=loop,
                golden_version=item_set_version(path),
            )  # fmt: skip
        finally:
            if conn is not None:
                await conn.close()

    result = run(go())
    s: dict[str, Any] = result.summary
    console.print(
        json.dumps({k: v for k, v in s.items() if k not in ("items", "item_strata")}, indent=2)
    )
    if report is not None:
        report.write_text(render_agent_report(list(result.summaries.values())), encoding="utf-8")
    failed: Literal[0, 1] = 0
    if s.get("duplicate_side_effects", 0):
        console.print("[red]duplicate side effects: the gate requires 0[/red]")
        failed = 1
    if s["task_success"] < min_success:
        console.print(f"[red]task success {s['task_success']:.3f} < {min_success}[/red]")
        failed = 1
    raise typer.Exit(failed)
