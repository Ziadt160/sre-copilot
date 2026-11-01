"""Deterministic answer checks (no model): citations against the chunks given and the gold."""

from __future__ import annotations

from dataclasses import dataclass

from sre_evals.metrics.core import match_grade
from sre_shared.models import AnswerResult, GoldenItem


@dataclass(frozen=True)
class AnswerScores:
    cited: bool  # at least one valid citation
    citation_hit: bool | None  # a cited chunk matches a gold passage; None: item has no gold
    invalid_citations: int  # cited ids outside the chunks given to the model
    abstained: bool


def score_answer(result: AnswerResult, item: GoldenItem) -> AnswerScores:
    """`citation_hit` looks each citation up among the chunks the model was given (a citation
    names a chunk id) and asks `match_grade` whether it evidences any gold passage (grade >= 1,
    the same bar as retrieval recall). An abstained answer has no citations, hence a miss."""
    given = {c.chunk_id: c for c in result.context}
    hit: bool | None = None
    if item.gold_passages:
        hit = any(
            match_grade(given[c.chunk_id], p) >= 1
            for c in result.citations
            if c.chunk_id in given
            for p in item.gold_passages
        )
    return AnswerScores(
        cited=bool(result.citations),
        citation_hit=hit,
        invalid_citations=len(result.invalid_citations),
        abstained=result.abstained,
    )
