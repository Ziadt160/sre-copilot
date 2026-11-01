"""Metric tests. Every expected value is worked by hand in the comment above it.

Quotes are built from numbered words so the 5-gram arithmetic is easy to follow:
W(a, b) = "w{a} w{a+1} ... w{b}". A quote W(1, 10) has 10 words and 6 distinct 5-grams,
the gram starting at word i being w_i..w_(i+4), i = 1..6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from sre_evals.metrics import (
    ann_recall,
    coverage,
    empty_result_rate,
    match_grade,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    score_item,
)
from sre_shared.models import GoldPassage

SRC = "content/en/docs/a.md"
VER = "1.33"


def W(a: int, b: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(a, b + 1))


@dataclass
class Chunk:
    body: str
    source_path: str = SRC
    k8s_version: str = VER


def gold(quote: str, src: str = SRC, ver: str = VER) -> GoldPassage:
    return GoldPassage(source_path=src, k8s_version=ver, quote=quote)


NOISE = Chunk("nothing relevant here at all")


# --- coverage and grade -------------------------------------------------------------------


def test_coverage_full_and_none() -> None:
    # Chunk holds all ten words: all 6 grams present -> 6/6 = 1.0.
    assert coverage(W(1, 10), W(1, 10)) == 1.0
    # No shared words -> 0/6 = 0.
    assert coverage("zzz yyy xxx www vvv", W(1, 10)) == 0.0


def test_coverage_partial_counts_grams_not_words() -> None:
    # Chunk has w1..w8: grams starting 1,2,3,4 are inside (w4..w8 ends at w8) -> 4/6.
    assert coverage(W(1, 8), W(1, 10)) == pytest.approx(4 / 6)
    # Chunk has w1..w9: grams 1..5 -> 5/6.
    assert coverage(W(1, 9), W(1, 10)) == pytest.approx(5 / 6)


def test_coverage_is_whitespace_normalized_and_case_sensitive() -> None:
    # Newlines and double spaces collapse to single spaces: still 6/6.
    messy = W(1, 10).replace(" ", "  ", 3).replace(" w8", "\n\tw8")
    assert coverage(messy, W(1, 10)) == 1.0
    # Case differs in every word -> no gram matches -> 0.
    assert coverage(W(1, 10).upper(), W(1, 10)) == 0.0


def test_coverage_requires_contiguous_words() -> None:
    # Chunk has the same words in reverse order: no contiguous 5-gram of the quote -> 0/6.
    reversed_body = " ".join(reversed(W(1, 10).split()))
    assert coverage(reversed_body, W(1, 10)) == 0.0


def test_coverage_duplicate_grams_counted_once() -> None:
    # Quote "a b c d e a b c d e" has grams: abcde, bcdea, cdeab, deabc, eabcd, abcde(dup)
    # -> 5 distinct. Chunk "a b c d e" holds only abcde -> 1/5 = 0.2 (not 2/6).
    assert coverage("a b c d e", "a b c d e a b c d e") == pytest.approx(1 / 5)


def test_short_quote_uses_exact_substring() -> None:
    # 3-word quote: substring present (after whitespace normalization) -> 1.0.
    assert coverage("set the  max\nunavailable value to 1", "max unavailable value") == 1.0
    # Words present but not contiguous -> 0.0.
    assert coverage("max value unavailable", "max unavailable value") == 0.0
    # Four-word quote is still "short" (< 5 words).
    assert coverage("x kubectl rollout undo deployment y", "kubectl rollout undo deployment") == 1.0
    # Exactly 5 words is NOT short: one 5-gram, present -> 1/1.
    assert coverage("a b c d e f", "a b c d e") == 1.0


def test_match_grade_boundaries() -> None:
    g = gold(W(1, 10))  # 6 grams
    # w1..w7: grams 1,2,3 -> 3/6 = 0.5 exactly -> grade 1 (threshold is >=).
    assert match_grade(Chunk(W(1, 7)), g) == 1
    # w1..w6: grams 1,2 -> 2/6 = 0.333 -> grade 0.
    assert match_grade(Chunk(W(1, 6)), g) == 0
    # w1..w9: 5/6 = 0.833 -> grade 2.
    assert match_grade(Chunk(W(1, 9)), g) == 2
    # w1..w8: 4/6 = 0.667 -> grade 1.
    assert match_grade(Chunk(W(1, 8)), g) == 1


def test_grade_two_threshold_is_exactly_point_eight() -> None:
    # Quote of 9 words has 5 grams. 4 grams present = 0.8 -> grade 2.
    g = gold(W(1, 9))
    assert match_grade(Chunk(W(1, 8)), g) == 2  # grams 1..4 -> 4/5 = 0.8
    assert match_grade(Chunk(W(1, 7)), g) == 1  # grams 1..3 -> 3/5 = 0.6


def test_match_requires_same_source_and_version() -> None:
    g = gold(W(1, 10))
    assert match_grade(Chunk(W(1, 10), source_path="other.md"), g) == 0
    assert match_grade(Chunk(W(1, 10), k8s_version="1.32"), g) == 0
    assert match_grade(Chunk(W(1, 10)), g) == 2


# --- recall / MRR -------------------------------------------------------------------------


def test_recall_mrr_basic() -> None:
    p1, p2 = gold(W(1, 10)), gold(W(21, 30, "v"))
    chunks = [NOISE, Chunk(W(1, 10)), NOISE, Chunk(W(21, 30, "v"))]
    # top 3: only p1 (rank 2) matched -> recall 1/2; MRR = 1/2.
    assert recall_at_k(chunks, [p1, p2], 3) == 0.5
    assert mrr_at_k(chunks, [p1, p2], 3) == 0.5
    # top 4: both matched -> recall 2/2 = 1; MRR still 1/2 (first match at rank 2).
    assert recall_at_k(chunks, [p1, p2], 4) == 1.0
    assert mrr_at_k(chunks, [p1, p2], 4) == 0.5
    # top 1: nothing matched -> 0 and 0.
    assert recall_at_k(chunks, [p1, p2], 1) == 0.0
    assert mrr_at_k(chunks, [p1, p2], 1) == 0.0


def test_passage_split_across_two_chunks() -> None:
    p = gold(W(1, 10))  # 6 grams
    first = Chunk(W(1, 7))  # grams 1,2,3 -> 3/6 = 0.5 -> grade 1, matches
    second = Chunk(W(6, 10))  # gram 6 only (w6..w10) -> 1/6 -> no match
    # The straddling grams 4 and 5 are in neither chunk, so only `first` matches.
    assert match_grade(first, p) == 1
    assert match_grade(second, p) == 0
    # second ranked first: MRR = 1/2 (first match at rank 2); recall@2 = 1/1.
    assert mrr_at_k([second, first], [p], 5) == 0.5
    assert recall_at_k([second, first], [p], 5) == 1.0
    # With a 40/60 split neither half reaches 0.5, so the passage is NOT retrieved:
    # w1..w6 -> grams 1,2 = 2/6; w5..w10 -> grams 5,6 = 2/6. Recall 0, MRR 0.
    halves = [Chunk(W(1, 6)), Chunk(W(5, 10))]
    assert recall_at_k(halves, [p], 5) == 0.0
    assert mrr_at_k(halves, [p], 5) == 0.0


def test_duplicate_matches_count_once_for_recall() -> None:
    p = gold(W(1, 10))
    chunks = [Chunk(W(1, 10)), Chunk(W(1, 10)), Chunk(W(1, 9))]
    # Three chunks match the single passage: recall = 1/1 (not 3), MRR = 1/1.
    assert recall_at_k(chunks, [p], 5) == 1.0
    assert mrr_at_k(chunks, [p], 5) == 1.0


def test_no_gold_passage_retrieved() -> None:
    p = gold(W(1, 10))
    assert recall_at_k([NOISE, NOISE], [p], 5) == 0.0
    assert mrr_at_k([NOISE, NOISE], [p], 5) == 0.0
    assert ndcg_at_k([NOISE, NOISE], [p], 5) == 0.0


def test_k_larger_than_results_and_empty_results() -> None:
    p = gold(W(1, 10))
    # One chunk at rank 1, k = 10: recall 1, MRR 1, nDCG = 3/log2(2) / 3/log2(2) = 1.
    assert recall_at_k([Chunk(W(1, 10))], [p], 10) == 1.0
    assert mrr_at_k([Chunk(W(1, 10))], [p], 10) == 1.0
    assert ndcg_at_k([Chunk(W(1, 10))], [p], 10) == pytest.approx(1.0)
    # Empty list: all zero, no exception.
    assert recall_at_k([], [p], 10) == 0.0
    assert mrr_at_k([], [p], 10) == 0.0
    assert ndcg_at_k([], [p], 10) == 0.0


def test_no_gold_is_not_scorable() -> None:
    assert recall_at_k([NOISE], [], 5) is None
    assert mrr_at_k([NOISE], [], 5) is None
    assert ndcg_at_k([NOISE], [], 5) is None
    assert score_item([NOISE], [], [5]) is None


def test_cutoff_excludes_lower_ranks() -> None:
    p = gold(W(1, 10))
    chunks = [NOISE, NOISE, NOISE, Chunk(W(1, 10))]
    # Match at rank 4: invisible at k=3, MRR@5 = 1/4.
    assert recall_at_k(chunks, [p], 3) == 0.0
    assert mrr_at_k(chunks, [p], 3) == 0.0
    assert mrr_at_k(chunks, [p], 5) == 0.25


# --- nDCG ---------------------------------------------------------------------------------


def test_ndcg_two_passages_mixed_grades() -> None:
    p1, p2 = gold(W(1, 10)), gold(W(21, 30, "v"))
    chunks = [NOISE, Chunk(W(1, 10)), Chunk(W(21, 27, "v"))]
    # Rank 2 grade 2 for p1 -> gain 3; rank 3 grade 1 for p2 (3/6 = 0.5) -> gain 1.
    # DCG = 3/log2(3) + 1/log2(4) = 3/1.58496 + 1/2 = 1.89279 + 0.5 = 2.39279.
    # Ideal = 3/log2(2) + 3/log2(3) = 3 + 1.89279 = 4.89279. nDCG = 2.39279 / 4.89279 = 0.48907.
    expected = (3 / math.log2(3) + 1 / 2) / (3 + 3 / math.log2(3))
    assert ndcg_at_k(chunks, [p1, p2], 10) == pytest.approx(expected)
    assert expected == pytest.approx(0.48907, abs=1e-4)


def test_ndcg_perfect_ranking_is_one() -> None:
    p1, p2 = gold(W(1, 10)), gold(W(21, 30, "v"))
    chunks = [Chunk(W(1, 10)), Chunk(W(21, 30, "v")), NOISE]
    # DCG = 3 + 3/log2(3) = ideal -> 1.0.
    assert ndcg_at_k(chunks, [p1, p2], 10) == pytest.approx(1.0)


def test_ndcg_duplicates_credit_passage_once() -> None:
    p = gold(W(1, 10))
    chunks = [Chunk(W(1, 10)), Chunk(W(1, 10))]
    # Single passage: DCG = 3 (rank 1 only); ideal = 3. The second copy adds nothing -> 1.0.
    assert ndcg_at_k(chunks, [p], 10) == pytest.approx(1.0)


def test_ndcg_passage_credited_at_first_match_even_if_better_later() -> None:
    p = gold(W(1, 10))
    chunks = [Chunk(W(1, 7)), NOISE, Chunk(W(1, 10))]
    # Rank 1 is grade 1 (gain 2^1 - 1 = 1) and credits the passage. The grade-2 hit at rank 3
    # earns 0 because the passage is already credited. DCG = 1/log2(2) = 1; ideal = 3/log2(2) = 3.
    # nDCG = 1/3 = 0.33333.
    assert ndcg_at_k(chunks, [p], 10) == pytest.approx(1 / 3)


def test_ndcg_one_chunk_cannot_pay_two_passages_twice() -> None:
    # Two passages sit side by side; chunk 1 contains both completely (grade 2 for each),
    # chunk 2 contains only the second. Rank 1 credits passage 1 (tie: lowest index), rank 2
    # credits passage 2:
    # DCG = 3/1 + 3/log2(3) = ideal -> 1.0 (the naive per-passage rule would give > 1).
    p1, p2 = gold(W(1, 10)), gold(W(11, 20))
    chunks = [Chunk(W(1, 20)), Chunk(W(11, 20))]
    assert ndcg_at_k(chunks, [p1, p2], 10) == pytest.approx(1.0)
    # Recall counts passages matched by any chunk: both matched by chunk 1 -> 2/2.
    assert recall_at_k(chunks, [p1, p2], 1) == 1.0


def test_ndcg_ideal_limited_by_k() -> None:
    golds = [gold(W(100 * i + 1, 100 * i + 10)) for i in range(3)]
    chunks = [Chunk(W(1, 10)), Chunk(W(101, 110))]
    # k = 2, three passages: ideal = 3 + 3/log2(3) (only two ranks exist) = 4.89279.
    # DCG = same -> 1.0. At k = 3 ideal adds 3/log2(4) = 1.5 -> 4.89279 / 6.39279 = 0.76537.
    assert ndcg_at_k(chunks, golds, 2) == pytest.approx(1.0)
    assert ndcg_at_k(chunks, golds, 3) == pytest.approx(
        (3 + 3 / math.log2(3)) / (3 + 3 / math.log2(3) + 1.5)
    )


# --- other metrics ------------------------------------------------------------------------


def test_empty_result_rate() -> None:
    # 1 of 4 queries returned nothing -> 0.25.
    assert empty_result_rate([5, 0, 3, 1]) == 0.25
    assert empty_result_rate([]) == 0.0
    assert empty_result_rate([0, 0]) == 1.0


def test_ann_recall() -> None:
    # Exact top-4 = {1,2,3,4}; the index returned {1,2,9,4} -> 3 of 4 = 0.75.
    assert ann_recall([1, 2, 9, 4], [1, 2, 3, 4]) == 0.75
    assert ann_recall([4, 3, 2, 1], [1, 2, 3, 4]) == 1.0  # order is irrelevant
    assert ann_recall([], [1, 2]) == 0.0
    # Exact list empty (no rows pass the filter): undefined, excluded from means.
    assert ann_recall([1], []) is None
    assert ann_recall([], []) is None
    # index=[1,2,3,4] vs exact=[1,2,3,4] -> 4/4 = 1.0.
    assert ann_recall([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    # index=[1,2] vs exact=[1,2,3,4]: divides by |exact| = 4 -> 2/4 = 0.5.
    assert ann_recall([1, 2], [1, 2, 3, 4]) == 0.5


def test_score_item_matches_individual_functions() -> None:
    p1, p2 = gold(W(1, 10)), gold(W(21, 30, "v"))
    chunks = [NOISE, Chunk(W(1, 10)), Chunk(W(21, 27, "v"))]
    s = score_item(chunks, [p1, p2], [3, 5, 8, 10], chunk_ids=[11, 12, 13])
    assert s is not None
    for k in (3, 5, 8, 10):
        assert s.recall[k] == recall_at_k(chunks, [p1, p2], k)
        assert s.mrr[k] == mrr_at_k(chunks, [p1, p2], k)
        assert s.ndcg[k] == ndcg_at_k(chunks, [p1, p2], k)
    # Recall at 3: p1 grade 2, p2 grade 1 -> both matched -> 1.0; MRR = 1/2.
    assert s.recall[3] == 1.0 and s.mrr[3] == 0.5
    assert [(m.rank, m.chunk_id, m.grades) for m in s.matches] == [
        (2, 12, [2, 0]),
        (3, 13, [0, 1]),
    ]
    assert s.n_results == 3
