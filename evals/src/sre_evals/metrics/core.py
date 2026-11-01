"""Pure retrieval metric functions.

Definitions (the README numbers rest on these; tests restate each by hand calculation).

Matching. A chunk can match a gold passage only when it has the same `source_path` and
`k8s_version`. Text is compared on the chunk BODY (not the header), after whitespace
normalization (runs of whitespace become one space), case-sensitive.

coverage(body, quote)
    quote with >= 5 words: |distinct word 5-grams of the quote that occur, as contiguous words,
    in the body| / |distinct word 5-grams of the quote|.
    quote with < 5 words: 1.0 when the normalized quote is a substring of the normalized body,
    else 0.0.

match_grade
    2 when coverage >= 0.8, 1 when coverage >= 0.5, else 0.

A gold passage split across two chunks counts as matched only if some single chunk covers
>= 50% of its 5-grams; 5-grams straddling the boundary are covered by neither chunk.

Per-item metrics take the ranked chunk list (best first) and cut it to the top k. k larger than
the list is fine (the whole list is used). Items with no gold passages are not scorable:
`recall_at_k`, `mrr_at_k` and `ndcg_at_k` return None for them.

recall@k   matched gold passages / gold passages, where a passage is matched when at least one
           chunk in the top k has grade >= 1 for it. Several chunks matching one passage count once.
MRR@k      1 / rank of the first chunk in the top k that matches (grade >= 1) any gold passage;
           0 when none does. Ranks are 1-based.
nDCG@k     gain(grade) = 2**grade - 1 (1 for grade 1, 3 for grade 2), discount 1 / log2(rank + 1).
           Walk ranks 1..k in order. The chunk at each rank earns the gain of the highest grade
           among the gold passages it matches that are not yet credited (ties: lowest passage
           index); that passage is then credited and cannot pay again. A chunk matching only
           credited passages, or none, earns 0 (so a passage matched at grade 1 at rank 1 and
           grade 2 at rank 3 is credited grade 1 at rank 1). Ideal DCG: n_gold grade-2 passages
           at ranks 1..min(n_gold, k).
ann_recall |index ids ∩ exact ids| / |exact ids|; 1.0 when the exact list is empty.
empty_result_rate  share of queries that returned no chunk at all.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sre_shared.models import GoldPassage

N_GRAM = 5
GRADE2_COVERAGE = 0.8
GRADE1_COVERAGE = 0.5


class ChunkLike(Protocol):
    """What the matcher needs from a retrieved chunk (`RetrievedChunk` satisfies it)."""

    @property
    def source_path(self) -> str: ...
    @property
    def k8s_version(self) -> str: ...
    @property
    def body(self) -> str: ...


def _ngrams(words: Sequence[str], n: int = N_GRAM) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def coverage(chunk_body: str, quote: str) -> float:
    """Share of the quote's distinct word 5-grams found in the chunk body (see module doc)."""
    q_words = quote.split()
    c_words = chunk_body.split()
    if not q_words:
        return 0.0
    if len(q_words) < N_GRAM:
        return 1.0 if " ".join(q_words) in " ".join(c_words) else 0.0
    q_grams = _ngrams(q_words)
    return len(q_grams & _ngrams(c_words)) / len(q_grams)


def grade_from_coverage(cov: float) -> int:
    if cov >= GRADE2_COVERAGE:
        return 2
    if cov >= GRADE1_COVERAGE:
        return 1
    return 0


def match_grade(chunk: ChunkLike, passage: GoldPassage) -> int:
    """0, 1 or 2: how well `chunk` evidences `passage` (same source and version required)."""
    if chunk.source_path != passage.source_path or chunk.k8s_version != passage.k8s_version:
        return 0
    return grade_from_coverage(coverage(chunk.body, passage.quote))


def _grade_matrix(chunks: Sequence[ChunkLike], passages: Sequence[GoldPassage]) -> list[list[int]]:
    """grades[rank - 1][passage index]."""
    return [[match_grade(c, p) for p in passages] for c in chunks]


def _recall(grades: list[list[int]], n_gold: int, k: int) -> float:
    matched = sum(1 for j in range(n_gold) if any(row[j] >= 1 for row in grades[:k]))
    return matched / n_gold


def _mrr(grades: list[list[int]], k: int) -> float:
    for rank, row in enumerate(grades[:k], start=1):
        if any(g >= 1 for g in row):
            return 1.0 / rank
    return 0.0


def _gain(grade: int) -> float:
    return float(2**grade - 1)


def _ndcg(grades: list[list[int]], n_gold: int, k: int) -> float:
    credited: set[int] = set()
    dcg = 0.0
    for rank, row in enumerate(grades[:k], start=1):
        best_grade, best_j = 0, -1
        for j, g in enumerate(row):
            if j not in credited and g > best_grade:
                best_grade, best_j = g, j
        if best_j >= 0:
            credited.add(best_j)
            dcg += _gain(best_grade) / math.log2(rank + 1)
    ideal = sum(_gain(2) / math.log2(i + 1) for i in range(1, min(n_gold, k) + 1))
    return dcg / ideal


def recall_at_k(chunks: Sequence[ChunkLike], gold: Sequence[GoldPassage], k: int) -> float | None:
    """Share of gold passages matched by at least one of the top-k chunks (None: no gold)."""
    if not gold:
        return None
    return _recall(_grade_matrix(chunks[:k], gold), len(gold), k)


def mrr_at_k(chunks: Sequence[ChunkLike], gold: Sequence[GoldPassage], k: int) -> float | None:
    """1 / rank of the first top-k chunk matching any gold passage, else 0 (None: no gold)."""
    if not gold:
        return None
    return _mrr(_grade_matrix(chunks[:k], gold), k)


def ndcg_at_k(chunks: Sequence[ChunkLike], gold: Sequence[GoldPassage], k: int) -> float | None:
    """Graded nDCG over the top-k chunks (None: no gold). See the module doc."""
    if not gold:
        return None
    return _ndcg(_grade_matrix(chunks[:k], gold), len(gold), k)


def empty_result_rate(result_counts: Sequence[int]) -> float:
    """Share of queries that returned zero chunks (0.0 for no queries)."""
    if not result_counts:
        return 0.0
    return sum(1 for n in result_counts if n == 0) / len(result_counts)


def ann_recall(index_ids: Sequence[int], exact_ids: Sequence[int]) -> float:
    """|index ∩ exact| / |exact|: how much of the exact top-N the ANN index returned."""
    exact = set(exact_ids)
    if not exact:
        return 1.0
    return len(exact & set(index_ids)) / len(exact)


@dataclass
class MatchedChunk:
    rank: int
    chunk_id: int | None
    grades: list[int]
    """Grade per gold passage, in gold order."""


@dataclass
class ItemScores:
    """All metrics for one item at several cutoffs, plus the match table for `details`."""

    recall: dict[int, float] = field(default_factory=dict)
    mrr: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    matches: list[MatchedChunk] = field(default_factory=list)
    n_results: int = 0


def score_item(
    chunks: Sequence[ChunkLike],
    gold: Sequence[GoldPassage],
    ks: Sequence[int],
    chunk_ids: Sequence[int | None] | None = None,
) -> ItemScores | None:
    """Score one item at every cutoff in `ks`, matching each chunk once. None when no gold."""
    if not gold:
        return None
    grades = _grade_matrix(chunks, gold)
    out = ItemScores(n_results=len(chunks))
    for k in ks:
        out.recall[k] = _recall(grades, len(gold), k)
        out.mrr[k] = _mrr(grades, k)
        out.ndcg[k] = _ndcg(grades, len(gold), k)
    ids = chunk_ids if chunk_ids is not None else [None] * len(chunks)
    out.matches = [
        MatchedChunk(rank=i, chunk_id=ids[i - 1], grades=row)
        for i, row in enumerate(grades, start=1)
        if any(row)
    ]
    return out
