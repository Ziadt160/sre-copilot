"""Hand-worked cases for kappa, accuracy, noise stats, the score -> label mapping."""

from __future__ import annotations

import pytest

from sre_evals.answers.stats import (
    NOT_SUPPORTED,
    PARTLY,
    SUPPORTED,
    accuracy,
    cohens_kappa,
    noise_stats,
    paired_bootstrap_ci,
    score_to_label,
    to_binary,
)

S, P, N = SUPPORTED, PARTLY, NOT_SUPPORTED


def test_accuracy_hand_worked() -> None:
    assert accuracy([S, P, N, S], [S, P, S, S]) == 0.75
    assert accuracy([], []) is None
    with pytest.raises(ValueError):
        accuracy([S], [S, N])


def test_kappa_perfect_agreement_with_variation_is_one() -> None:
    a = [S, P, N, S]
    assert cohens_kappa(a, a) == pytest.approx(1.0)


def test_kappa_chance_agreement_is_zero() -> None:
    # po = 2/4. Marginals: A says S,S,N,N; B says S,N,S,N -> pe = .5*.5 + .5*.5 = .5.
    assert cohens_kappa([S, S, N, N], [S, N, S, N]) == pytest.approx(0.0)


def test_kappa_below_chance_is_negative() -> None:
    # Complete disagreement on a balanced pair: po = 0, pe = .5 -> kappa = -1.
    assert cohens_kappa([S, S, N, N], [N, N, S, S]) == pytest.approx(-1.0)


def test_kappa_all_one_class_is_undefined_not_one() -> None:
    assert cohens_kappa([S, S, S], [S, S, S]) is None
    assert cohens_kappa([], []) is None


def test_kappa_all_one_class_for_one_rater_only() -> None:
    # B always says S: pe = P(A=S)*1 = .75, po = .75 -> kappa = 0.
    assert cohens_kappa([S, S, S, N], [S, S, S, S]) == pytest.approx(0.0)


def test_kappa_three_class_hand_worked() -> None:
    # 10 items. Confusion: S/S 4, P/P 1, N/N 2, S/P 1, P/N 1, N/S 1.
    human = [S] * 4 + [P] + [N] * 2 + [S, P, N]
    judge = [S] * 4 + [P] + [N] * 2 + [P, N, S]
    # po = 7/10. human marginals S .5 P .2 N .3 ; judge S .5 P .2 N .3? count judge:
    # S: 4 + 1 (N/S) = 5, P: 1 + 1 = 2, N: 2 + 1 = 3 -> same marginals as human (S5 P2 N3).
    # pe = .25 + .04 + .09 = .38 -> kappa = (.7 - .38) / .62.
    assert cohens_kappa(human, judge) == pytest.approx(0.32 / 0.62)
    # Binary view: supported vs not. human S5 / not5, judge S5 / not5; agree on 4 + 4 + ...
    hb, jb = [to_binary(x) for x in human], [to_binary(x) for x in judge]
    # pairs (S,S)x4, (P,P)->not/not 1, (N,N)x2 -> 3 not/not, (S,P) S/not, (P,N) not/not, (N,S) not/S
    # agree = 4 + 3 + 1 = 8 -> po .8; pe = .5*.5 + .5*.5 = .5 -> kappa .6
    assert accuracy(hb, jb) == pytest.approx(0.8)
    assert cohens_kappa(hb, jb) == pytest.approx(0.6)


def test_score_to_label_thresholds() -> None:
    assert score_to_label(1.0) == S
    assert score_to_label(0.9) == S
    assert score_to_label(0.8999) == P
    assert score_to_label(0.5) == P
    assert score_to_label(0.4999) == N
    assert score_to_label(0.0) == N
    assert [to_binary(x) for x in (S, P, N)] == [S, N, N]


def test_noise_stats_hand_worked() -> None:
    runs = [{"f": 0.6, "r": 1.0}, {"f": 0.8, "r": 1.0}, {"f": 0.7}]
    s = noise_stats(runs)
    assert s["f"].n == 3 and s["f"].mean == pytest.approx(0.7)
    assert (s["f"].min, s["f"].max) == (0.6, 0.8)
    assert s["f"].stdev == pytest.approx(0.1)  # sqrt((0.01 + 0.01 + 0) / 2)
    assert s["f"].spread == pytest.approx(0.2)
    assert s["r"].n == 2 and s["r"].stdev == 0.0  # identical values
    assert noise_stats([{"f": 0.5}])["f"].stdev == 0.0  # one repeat: no spread
    assert noise_stats([]) == {}


def test_paired_bootstrap_ci() -> None:
    assert paired_bootstrap_ci([]) is None
    mean, lo, hi = paired_bootstrap_ci([0.1] * 20) or (0, 0, 0)
    assert mean == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    mean, lo, hi = paired_bootstrap_ci([1.0, -1.0] * 20, seed=3) or (0, 0, 0)
    assert mean == pytest.approx(0.0) and lo < 0 < hi
    assert paired_bootstrap_ci([0.0, 0.3, -0.1], seed=1) == paired_bootstrap_ci(
        [0.0, 0.3, -0.1], seed=1
    )
