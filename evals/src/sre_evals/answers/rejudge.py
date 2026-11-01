"""Judge a stored answer run again, without generating anything.

Generation and judging run as separate passes so each local model has the GPU to itself (an
8 GB card cannot hold the generator and an 8B judge at once), and so judge noise can be
measured on identical answers. Each stored answer keeps its full `AnswerResult`
(`answer_detail.details.result`); a replay answerer hands those back to `run_answers`, which
scores, stores and counts failures exactly as a normal run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from psycopg import AsyncConnection

from sre_evals.answers.judge import Judge
from sre_evals.answers.runner import AnswerRunOutcome, run_answers
from sre_evals.runner.config import ExperimentConfig
from sre_shared.golden_io import golden_path, golden_version, load_golden
from sre_shared.models import (
    ANONYMOUS,
    AnswerConfig,
    AnswerRequest,
    AnswerResult,
    Claims,
    GoldenItem,
)


class StoredRunError(LookupError):
    """The run cannot be re-judged (missing, failed, or stored before full results were kept)."""


class ReplayAnswerer:
    """Returns the stored result for each question; never retrieves or generates."""

    def __init__(self, results: Mapping[str, AnswerResult]) -> None:
        self._by_question = dict(results)

    async def answer(self, req: AnswerRequest, claims: Claims = ANONYMOUS) -> AnswerResult:
        try:
            return self._by_question[req.question]
        except KeyError as e:
            raise StoredRunError(f"no stored answer for question {req.question!r}") from e


async def load_stored_run(
    conn: AsyncConnection[Any], run_id: int
) -> tuple[ExperimentConfig, AnswerConfig, str, dict[str, AnswerResult], dict[str, Any]]:
    """(experiment config, answer config, golden version, results by item id, summary)."""
    row = await (
        await conn.execute(
            "select kind, status, config, golden_version, summary from public.eval_runs"
            " where id = %s",
            [run_id],
        )
    ).fetchone()
    if row is None or row["kind"] != "answer":
        raise StoredRunError(f"run {run_id} is not an answer run")
    if row["status"] != "done":
        raise StoredRunError(f"run {run_id} is {row['status']}, not done")
    cfg = ExperimentConfig.model_validate(row["config"])
    summary = dict(row["summary"] or {})
    answer_cfg = (cfg.answer or AnswerConfig()).model_copy(
        update={"model": summary.get("model", (cfg.answer or AnswerConfig()).model)}
    )
    cur = await conn.execute(
        "select item_id, details from public.eval_scores where run_id = %s"
        " and metric = 'answer_detail'",
        [run_id],
    )
    results: dict[str, AnswerResult] = {}
    for r in await cur.fetchall():
        stored = (r["details"] or {}).get("result")
        if stored is None:
            raise StoredRunError(
                f"run {run_id} predates full stored results; generate it again to re-judge"
            )
        results[r["item_id"]] = AnswerResult.model_validate(stored)
    if not results:
        raise StoredRunError(f"run {run_id} has no stored answers")
    return cfg, answer_cfg, str(row["golden_version"]), results, summary


async def rejudge_run(
    conn: AsyncConnection[Any],
    run_id: int,
    judge: Judge,
    *,
    repeat: int = 1,
    loop: str = "local",
    items_by_id: Mapping[str, GoldenItem] | None = None,
) -> list[AnswerRunOutcome]:
    """Store `repeat` new answer runs: the stored answers of `run_id`, scored by `judge`.
    `items_by_id` replaces reading the run's golden set from disk (tests)."""
    cfg, answer_cfg, gversion, results, summary = await load_stored_run(conn, run_id)
    if items_by_id is None:
        golden_dir = gversion.rsplit("-", 1)[0]
        path = golden_path(golden_dir)
        if golden_version(path) != gversion:
            raise StoredRunError(
                f"golden set changed since run {run_id} ({gversion} -> {golden_version(path)})"
            )
        items_by_id = {i.id: i for i in load_golden(path, accepted_only=True)}
    by_id = dict(items_by_id)
    missing = sorted(set(results) - set(by_id))
    if missing:
        raise StoredRunError(f"items no longer in the golden set: {missing[:5]}")
    items = [by_id[i] for i in sorted(results)]
    replay = ReplayAnswerer({by_id[i].question: r for i, r in results.items()})
    return await run_answers(
        cfg, items, replay, conn, judge=judge, loop=loop, golden_version=gversion,
        answer_cfg=answer_cfg, repeat=repeat,
        extra={"judged_from_run": run_id, "generation_batch": summary.get("batch")},
    )  # fmt: skip
