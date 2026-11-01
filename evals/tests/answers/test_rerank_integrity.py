"""A run that asks for reranking never mixes in fused-order results (a cold reranker made the
first query fall back to the fused order without failing)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sre_evals.answers.runner import run_answers
from sre_evals.answers.testing import FakeAnswerer, W, chunk, golden, result
from sre_evals.runner.config import ExperimentConfig, RerankConfig
from sre_evals.runner.engine import RerankFallback, _retrieve_all
from sre_shared.models import (
    ANONYMOUS,
    AnswerConfig,
    AnswerRequest,
    AnswerResult,
    Claims,
    RetrievalRequest,
    RetrievalResult,
)
from sre_shared.testing.seed import create_corpus_version

ITEMS = [golden(f"fac-000{i}", "factual", W(1, 12)) for i in range(1, 4)]
CFG = ExperimentConfig(name="answer-t", legs=["dense"], answer=AnswerConfig(k=5))


def answered(req: AnswerRequest, reranked: bool) -> AnswerResult:
    res = result(req.question, "Answer [1].", [chunk(1, W(1, 12))], [1])
    return res.model_copy(update={"reranked": reranked})


@pytest.mark.integration
async def test_an_answer_run_stops_at_a_fallback_and_does_not_cache_it(
    db: Any, tmp_path: Path
) -> None:
    await create_corpus_version(db)
    cold = {"question fac-0002"}

    def script(req: AnswerRequest) -> AnswerResult:
        return answered(req, reranked=req.question not in cold)

    kw: dict[str, Any] = {
        "judge": None, "loop": "local", "golden_version": "v1-abc", "use_cache": True,
        "cache_directory": tmp_path, "answer_cfg": AnswerConfig(k=5, model="fake:m"),
        "require_rerank": True,
    }  # fmt: skip
    with pytest.raises(RerankFallback, match="fac-0002"):
        await run_answers(CFG, ITEMS, FakeAnswerer(script), db, **kw)
    cold.clear()  # the reranker is warm now
    calls: list[str] = []

    def again(req: AnswerRequest) -> AnswerResult:
        calls.append(req.question)
        return answered(req, reranked=True)

    [out] = await run_answers(CFG, ITEMS, FakeAnswerer(again), db, **kw)
    assert "question fac-0002" in calls  # the fallback answer was not kept
    assert out.summary["n_reranked"] == 3


@pytest.mark.integration
async def test_without_require_rerank_fallbacks_are_counted_not_refused(db: Any) -> None:
    await create_corpus_version(db)
    [out] = await run_answers(
        CFG, ITEMS, FakeAnswerer(lambda r: answered(r, reranked=False)), db, judge=None,
        loop="local", golden_version="v1-abc",
    )  # fmt: skip
    assert out.summary["n_reranked"] == 0 and out.summary["n_answered"] == 3


class FallbackRetriever:
    """Reranks every query except one, which keeps the fused order."""

    def __init__(self, cold: str) -> None:
        self.cold = cold

    async def retrieve(
        self, req: RetrievalRequest, claims: Claims = ANONYMOUS, *, exact: bool = False
    ) -> RetrievalResult:
        return RetrievalResult(
            request=req,
            corpus_version_id=1,
            chunks=[chunk(1, W(1, 12))],
            timings_ms={"total": 1.0},
            reranked=req.query != self.cold,
        )


async def test_a_retrieval_run_with_a_reranker_stops_at_a_fallback() -> None:
    reranked = CFG.model_copy(update={"rerank": RerankConfig(candidates=20)})
    with pytest.raises(RerankFallback, match="fac-0003"):
        await _retrieve_all(reranked, ITEMS, FallbackRetriever("question fac-0003"), ANONYMOUS, 1)
    # Without a reranker in the config, the flag is not checked.
    runs = await _retrieve_all(CFG, ITEMS, FallbackRetriever("question fac-0003"), ANONYMOUS, 1)
    assert len(runs) == 3
