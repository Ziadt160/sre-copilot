"""Weighted reciprocal rank fusion. Pure functions, no I/O."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One row of `public.search_candidates`: a chunk with its rank in each leg (None if absent)."""

    chunk_id: int
    dense_rank: int | None = None
    keyword_rank: int | None = None
    dense_similarity: float | None = None
    keyword_score: float | None = None


@dataclass(frozen=True)
class Fused:
    candidate: Candidate
    score: float


_ABSENT = 1 << 62


def fuse_rrf(
    candidates: Iterable[Candidate],
    *,
    w_dense: float = 1.0,
    w_keyword: float = 1.0,
    rrf_k: int = 60,
) -> list[Fused]:
    """score = w_dense / (rrf_k + dense_rank) + w_keyword / (rrf_k + keyword_rank), a leg adding
    nothing when the chunk is absent from it.

    Sorted by score descending. Ties break on the better dense rank, then the better keyword rank,
    then the lower chunk id, so the order never depends on input order.
    """
    fused: list[Fused] = []
    for c in candidates:
        score = 0.0
        if c.dense_rank is not None:
            score += w_dense / (rrf_k + c.dense_rank)
        if c.keyword_rank is not None:
            score += w_keyword / (rrf_k + c.keyword_rank)
        fused.append(Fused(c, score))
    fused.sort(
        key=lambda f: (
            -f.score,
            f.candidate.dense_rank if f.candidate.dense_rank is not None else _ABSENT,
            f.candidate.keyword_rank if f.candidate.keyword_rank is not None else _ABSENT,
            f.candidate.chunk_id,
        )
    )
    return fused
