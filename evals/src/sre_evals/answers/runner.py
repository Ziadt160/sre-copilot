"""Run an answer experiment over the golden items and store `kind='answer'` runs.

One `eval_runs` row per repeat (`--repeat N` stores N). `eval_runs.summary` keys:

    experiment, repeat, repeats, batch   which experiment; this repeat (1-based) of how many;
                                          `batch` ties the repeats of one invocation together
    model, judge                          answer model and judge labels ("provider:name", or
                                          "none" when judging was skipped)
    judge_prompt_version, metric_version, answer_prompt_version
    n_items, n_answered, n_abstained, judge_failures
    faithfulness, answer_relevancy        mean over answered items the judge scored
    n_faithfulness, n_relevancy           how many items those means cover
    cited_rate                            share of answered items with >= 1 valid citation
    citation_hit_rate                     share of items with gold passages whose answer cites a
                                          chunk matching a gold passage (match_grade >= 1);
                                          an abstained answer counts as a miss
    invalid_citations                     total cited ids outside the given chunks
    abstain_rate                          n_abstained / n_items
    prompt_tokens, completion_tokens, context_tokens   means per item
    latency_ms                            {"p50", "p95", "mean"} of the answer's total time
    by_stratum                            {stratum: {"n", "abstain_rate", "faithfulness", ...}}

`eval_scores` rows (k = 0): faithfulness, answer_relevancy (only when scored), cited,
citation_hit (items with gold), invalid_citations, abstained, prompt_tokens,
completion_tokens, context_tokens, latency_ms, and `answer_detail` (value 1.0) whose `details`
hold what the labeller shows: answer text, citations, invalid citation ids, and the exact
chunks given to the model (id, source, version, title, url, body), in prompt order.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from psycopg import AsyncConnection

from sre_core.answer.base import AnswererLike
from sre_evals.answers.cache import AnswerCache, cache_key
from sre_evals.answers.judge import (
    JUDGE_PROMPT_VERSION,
    Judge,
    estimate_judge_tokens,
    metric_version,
)
from sre_evals.answers.scoring import AnswerScores, score_answer
from sre_evals.runner.config import ExperimentConfig
from sre_evals.runner.engine import RunOutcome, _chunker_version, _insert_run, latency_stats
from sre_shared.embed.base import SPECS
from sre_shared.models import (
    ANONYMOUS,
    AnswerConfig,
    AnswerRequest,
    AnswerResult,
    Claims,
    GoldenItem,
    RetrievalFilters,
)
from sre_shared.stamps import VersionStamps, config_hash


@dataclass
class ItemRun:
    item: GoldenItem
    result: AnswerResult
    scores: AnswerScores
    faithfulness: float | None = None
    relevancy: float | None = None


@dataclass
class AnswerRunOutcome(RunOutcome):
    repeat: int = 1
    batch: str = ""
    items: list[ItemRun] = field(default_factory=list)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def plan_judge_tokens(answer_cfg: AnswerConfig, n_answers: int) -> int:
    """Upper-ish estimate of the judge tokens for `n_answers` answers under `answer_cfg`,
    before any answer exists: answer at max_tokens, passages at the context budget."""
    per = estimate_judge_tokens(
        "x" * (answer_cfg.max_tokens * 4), ["x" * (answer_cfg.context_token_budget * 4)]
    )
    return per * n_answers


JUDGE_COUNTERS = ("faithfulness_failures", "relevancy_failures", "no_claims", "no_statements")
MAX_JUDGE_FAILURE_SHARE = 0.2


class JudgeUnreliable(RuntimeError):
    """The judge failed to produce a parseable score for too many judged answers."""


def chunk_text(c: Any) -> str:
    """The passage as the model saw it (T06-1: header, blank line, body)."""
    return f"{c.header}\n\n{c.body}" if c.header else str(c.body)


def summarize(
    runs: Sequence[ItemRun],
    n_items: int,
    judge_failures: int = 0,
    counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    answered = [r for r in runs if not r.result.abstained]
    faith = [r.faithfulness for r in runs if r.faithfulness is not None]
    rel = [r.relevancy for r in runs if r.relevancy is not None]
    gold = [r for r in runs if r.scores.citation_hit is not None]
    out: dict[str, Any] = {
        "n_items": n_items,
        "n_answered": len(answered),
        "n_abstained": len(runs) - len(answered),
        "n_empty_answers": sum(1 for r in answered if not r.result.answer.strip()),
        "judge_failures": judge_failures,
        **{k: int((counts or {}).get(k, 0)) for k in JUDGE_COUNTERS},
        "faithfulness": _mean(faith),
        "n_faithfulness": len(faith),
        "answer_relevancy": _mean(rel),
        "n_relevancy": len(rel),
        "cited_rate": _mean([float(r.scores.cited) for r in answered]),
        "citation_hit_rate": _mean([float(r.scores.citation_hit or 0) for r in gold]),
        "invalid_citations": sum(r.scores.invalid_citations for r in runs),
        "abstain_rate": (len(runs) - len(answered)) / len(runs) if runs else 0.0,
        "prompt_tokens": _mean([r.result.prompt_tokens for r in runs]),
        "completion_tokens": _mean([r.result.completion_tokens for r in runs]),
        "context_tokens": _mean([r.result.context_tokens for r in runs]),
        "latency_ms": latency_stats([r.result.timings_ms.get("total", 0.0) for r in runs]),
    }
    out["items"] = {
        r.item.id: {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.relevancy,
            "citation_hit": None if r.scores.citation_hit is None else float(r.scores.citation_hit),
            "cited": None if r.result.abstained else float(r.scores.cited),
            "abstained": float(r.result.abstained),
        }
        for r in runs
    }
    strata: dict[str, list[ItemRun]] = {}
    for r in runs:
        strata.setdefault(r.item.stratum, []).append(r)
    out["by_stratum"] = {
        s: {
            "n": len(g),
            "abstain_rate": sum(x.result.abstained for x in g) / len(g),
            "faithfulness": _mean([x.faithfulness for x in g if x.faithfulness is not None]),
            "answer_relevancy": _mean([x.relevancy for x in g if x.relevancy is not None]),
            "citation_hit_rate": _mean(
                [float(x.scores.citation_hit or 0) for x in g if x.scores.citation_hit is not None]
            ),
        }
        for s, g in sorted(strata.items())
    }
    return out


def _score_rows(run_id: int, r: ItemRun) -> list[list[Any]]:
    from psycopg.types.json import Jsonb

    iid = r.item.id
    res = r.result
    rows: list[list[Any]] = [
        [run_id, iid, "abstained", 0, float(res.abstained), None],
        [run_id, iid, "cited", 0, float(r.scores.cited), None],
        [run_id, iid, "invalid_citations", 0, float(r.scores.invalid_citations), None],
        [run_id, iid, "prompt_tokens", 0, float(res.prompt_tokens), None],
        [run_id, iid, "completion_tokens", 0, float(res.completion_tokens), None],
        [run_id, iid, "context_tokens", 0, float(res.context_tokens), None],
        [run_id, iid, "latency_ms", 0, float(res.timings_ms.get("total", 0.0)), None],
    ]
    if r.scores.citation_hit is not None:
        rows.append([run_id, iid, "citation_hit", 0, float(r.scores.citation_hit), None])
    if r.faithfulness is not None:
        rows.append([run_id, iid, "faithfulness", 0, r.faithfulness, None])
    if r.relevancy is not None:
        rows.append([run_id, iid, "answer_relevancy", 0, r.relevancy, None])
    detail = {
        "question": res.question,
        "answer": res.answer,
        "abstained": res.abstained,
        "citations": [c.model_dump(mode="json") for c in res.citations],
        "invalid_citations": res.invalid_citations,
        "context_chunk_ids": [c.chunk_id for c in res.context],
        "context": [
            {
                "chunk_id": c.chunk_id,
                "source_path": c.source_path,
                "k8s_version": c.k8s_version,
                "title": c.title,
                "url": c.url,
                "header": c.header,
                "body": c.body,
            }
            for c in res.context
        ],
        "stratum": r.item.stratum,
        # The complete result, so a stored run can be judged again without regenerating
        # (sre_evals.answers.rejudge).
        "result": res.model_dump(mode="json"),
    }
    rows.append([run_id, iid, "answer_detail", 0, 1.0, Jsonb(detail)])
    return rows


# Transient provider errors get a few patient retries on top of the SDK's quick ones, so a
# short outage does not fail a run that has already spent most of a day's tokens.
RETRY_DELAYS_S = (10.0, 30.0, 60.0)


def _transient(exc: BaseException) -> bool:
    import openai

    return isinstance(
        exc, openai.APIConnectionError | openai.APITimeoutError | openai.RateLimitError
    )


async def _answer_with_retries(
    answerer: AnswererLike, req: AnswerRequest, claims: Claims, delays: Sequence[float]
) -> AnswerResult:
    for attempt in range(len(delays) + 1):
        try:
            return await answerer.answer(req, claims)
        except Exception as e:
            if attempt == len(delays) or not _transient(e):
                raise
            await asyncio.sleep(delays[attempt])
    raise AssertionError("unreachable")


async def _one(
    item: GoldenItem,
    cfg: ExperimentConfig,
    answer_cfg: AnswerConfig,
    answerer: AnswererLike,
    judge: Judge | None,
    claims: Claims,
    corpus_version_id: int | None,
    cache: AnswerCache | None = None,
    retry_delays: Sequence[float] = RETRY_DELAYS_S,
) -> ItemRun:
    req = AnswerRequest(
        question=item.question,
        filters=item.filters or RetrievalFilters(),
        config=answer_cfg,
        corpus_version_id=corpus_version_id,
    )
    result = cache.get(item.id, item.question) if cache is not None else None
    if result is None:
        result = await _answer_with_retries(answerer, req, claims, retry_delays)
        if cache is not None:
            cache.put(item.id, result)
    run = ItemRun(item, result, score_answer(result, item))
    if judge is not None and not result.abstained and result.answer.strip():
        contexts = [chunk_text(c) for c in result.context]
        run.faithfulness = await judge.faithfulness(item.question, result.answer, contexts)
        run.relevancy = await judge.relevancy(item.question, result.answer)
    return run


async def run_answers(
    cfg: ExperimentConfig,
    items: Sequence[GoldenItem],
    answerer: AnswererLike,
    conn: AsyncConnection[Any],
    *,
    judge: Judge | None,
    loop: str,
    golden_version: str,
    answer_cfg: AnswerConfig | None = None,
    repeat: int = 1,
    concurrency: int = 1,
    claims: Claims = ANONYMOUS,
    extra: Mapping[str, Any] | None = None,
    max_judge_failure_share: float = MAX_JUDGE_FAILURE_SHARE,
    use_cache: bool = False,
    cache_directory: Path | None = None,
    retry_delays: Sequence[float] = RETRY_DELAYS_S,
) -> list[AnswerRunOutcome]:
    """Answer and score every item `repeat` times; one stored `kind='answer'` run per repeat.
    An answerer, judge-transport or quota error stores a `failed` run (no scores) and re-raises;
    so does a judge that fails to parse more than `max_judge_failure_share` of the judged
    answers (`JudgeUnreliable`)."""
    acfg = answer_cfg or cfg.answer or AnswerConfig()
    batch = uuid.uuid4().hex[:12]
    outcomes: list[AnswerRunOutcome] = []
    sem = asyncio.Semaphore(max(1, concurrency))
    for rep in range(1, repeat + 1):
        stamps = VersionStamps(
            config_hash=config_hash({"experiment": cfg, "answer": acfg}),
            embedding_model=cfg.embedding_model,
            embedding_revision=SPECS[cfg.embedding_model].revision,
            golden_version=golden_version,
            prompt_versions={"answer": acfg.prompt_version, "judge": JUDGE_PROMPT_VERSION},
            models_per_node={"answer": acfg.model, "judge": judge.label if judge else "none"},
        )
        meta: dict[str, Any] = {
            "experiment": cfg.name,
            "repeat": rep,
            "repeats": repeat,
            "batch": batch,
            "model": acfg.model,
            "judge": judge.label if judge else "none",
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "metric_version": metric_version(),
            "answer_prompt_version": acfg.prompt_version,
            "golden_version": golden_version,
            **(extra or {}),
        }
        before = {k: int(getattr(judge, k, 0)) for k in JUDGE_COUNTERS}
        before_total = int(getattr(judge, "failures", 0))

        cache = (
            AnswerCache(cache_key(stamps.config_hash, rep, golden_version), cache_directory)
            if use_cache
            else None
        )
        if cache is not None:
            meta["cached_answers_at_start"] = len(cache)

        async def guarded(item: GoldenItem, cache: AnswerCache | None = cache) -> ItemRun:
            async with sem:
                return await _one(
                    item, cfg, acfg, answerer, judge, claims, cfg.resolved_corpus_version_id,
                    cache, retry_delays,
                )  # fmt: skip

        try:
            runs = list(await asyncio.gather(*(guarded(i) for i in items)))
            counts = {k: int(getattr(judge, k, 0)) - before[k] for k in JUDGE_COUNTERS}
            failures = int(getattr(judge, "failures", 0)) - before_total
            judged = sum(1 for r in runs if not r.result.abstained and r.result.answer.strip())
            worst = max(counts["faithfulness_failures"], counts["relevancy_failures"])
            if judged and worst / judged > max_judge_failure_share:
                raise JudgeUnreliable(
                    f"{worst} of {judged} judged answers got no parseable score from"
                    f" {meta['judge']} (limit {max_judge_failure_share:.0%})"
                )
        except Exception as e:
            stamps.corpus_version_id = None
            await _insert_run(
                conn, cfg, loop, golden_version, stamps, "failed",
                {"error": f"{type(e).__name__}: {e}", **meta}, "answer",
            )  # fmt: skip
            raise
        summary = summarize(runs, len(items), failures, counts)
        summary.update(meta)
        corpus_ids = {r.result.corpus_version_id for r in runs}
        stamps.corpus_version_id = next(iter(corpus_ids)) if len(corpus_ids) == 1 else None
        stamps.chunker_version = await _chunker_version(conn, stamps.corpus_version_id)
        async with conn.transaction():
            run_id = await _insert_run(
                conn, cfg, loop, golden_version, stamps, "done", summary, "answer"
            )
            rows = [row for r in runs for row in _score_rows(run_id, r)]
            async with conn.cursor() as cur:
                await cur.executemany(
                    "insert into public.eval_scores (run_id, item_id, metric, k, value, details)"
                    " values (%s, %s, %s, %s, %s, %s)",
                    rows,
                )
        outcomes.append(
            AnswerRunOutcome(run_id, "done", summary, repeat=rep, batch=batch, items=runs)
        )
    return outcomes
