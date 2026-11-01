"""Fakes for answer-eval tests: an answerer, a judge, and result builders (no network)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sre_evals.runner.testing import SRC, VER, W, chunk
from sre_shared.models import (
    ANONYMOUS,
    AnswerConfig,
    AnswerRequest,
    AnswerResult,
    Citation,
    Claims,
    GoldenItem,
    GoldPassage,
    RetrievedChunk,
)

__all__ = ["SRC", "VER", "FakeAnswerer", "FakeJudge", "W", "chunk", "golden", "result"]


def golden(iid: str, stratum: str, quote: str | None) -> GoldenItem:
    gold = [GoldPassage(source_path=SRC, k8s_version=VER, quote=quote)] if quote else []
    return GoldenItem(
        id=iid,
        stratum=stratum,
        question=f"question {iid}",
        gold_passages=gold,
        status="accepted",
    )


def result(
    question: str,
    answer: str,
    context: Sequence[RetrievedChunk],
    cited: Sequence[int] = (),
    *,
    invalid: Sequence[str] = (),
    abstained: bool = False,
    corpus_version_id: int = 1,
) -> AnswerResult:
    by_id = {c.chunk_id: c for c in context}
    return AnswerResult(
        question=question,
        answer=answer,
        citations=[
            Citation(
                chunk_id=i,
                source_path=by_id[i].source_path,
                k8s_version=by_id[i].k8s_version,
                url=by_id[i].url,
                title=by_id[i].title,
            )
            for i in cited
        ],
        invalid_citations=list(invalid),
        abstained=abstained,
        context=list(context),
        config=AnswerConfig(),
        model="fake:model",
        corpus_version_id=corpus_version_id,
        prompt_tokens=100,
        completion_tokens=20,
        context_tokens=60,
        timings_ms={"total": 50.0},
    )


@dataclass
class FakeAnswerer:
    """`script(req)` -> AnswerResult. Records every request."""

    script: Callable[[AnswerRequest], AnswerResult]
    calls: list[AnswerRequest] = field(default_factory=list)

    async def answer(self, req: AnswerRequest, claims: Claims = ANONYMOUS) -> AnswerResult:
        self.calls.append(req)
        return self.script(req)


@dataclass
class FakeJudge:
    """Scores from a table keyed by the question (faithfulness) or a constant."""

    label: str = "fake:judge"
    faith: dict[str, float | None] = field(default_factory=dict)
    default_faith: float | None = 1.0
    rel: float | None = 0.8
    seen: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    failures: int = 0

    @property
    def faithfulness_failures(self) -> int:
        return self.failures

    async def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> float | None:
        self.seen.append((question, tuple(contexts)))
        score = self.faith.get(question, self.default_faith)
        if score is None:
            self.failures += 1
        return score

    async def relevancy(self, question: str, answer: str) -> float | None:
        return self.rel
