from __future__ import annotations

import random

import pytest

from sre_core.retrieval import Candidate, fuse_rrf


def test_hand_computed_scores() -> None:
    # rrf_k=60, w_dense=1, w_keyword=2
    cands = [
        Candidate(1, dense_rank=1, keyword_rank=3),  # 1/61 + 2/63
        Candidate(2, dense_rank=2),  # 1/62
        Candidate(3, keyword_rank=1),  # 2/61
    ]
    out = fuse_rrf(cands, w_dense=1.0, w_keyword=2.0, rrf_k=60)
    assert [f.candidate.chunk_id for f in out] == [1, 3, 2]
    assert out[0].score == pytest.approx(1 / 61 + 2 / 63)
    assert out[1].score == pytest.approx(2 / 61)
    assert out[2].score == pytest.approx(1 / 62)


def test_single_leg_score_is_that_legs_term() -> None:
    out = fuse_rrf([Candidate(7, dense_rank=4)], w_dense=0.5, rrf_k=10)
    assert out[0].score == pytest.approx(0.5 / 14)


def test_zero_weight_drops_a_leg() -> None:
    cands = [Candidate(1, dense_rank=1, keyword_rank=2), Candidate(2, dense_rank=2, keyword_rank=1)]
    out = fuse_rrf(cands, w_dense=0.0, w_keyword=1.0, rrf_k=60)
    assert [f.candidate.chunk_id for f in out] == [2, 1]


def test_tie_break_is_deterministic_and_input_order_independent() -> None:
    # Equal scores: dense rank 1 only vs keyword rank 1 only. Better dense rank wins; then id.
    a = Candidate(10, dense_rank=1)
    b = Candidate(5, keyword_rank=1)
    c = Candidate(3, keyword_rank=1)
    expected = [10, 3, 5]
    for seed in range(5):
        items = [a, b, c]
        random.Random(seed).shuffle(items)
        assert [f.candidate.chunk_id for f in fuse_rrf(items)] == expected


def test_empty() -> None:
    assert fuse_rrf([]) == []
