"""Paired deltas, the two-halves check and the verdict rule (no DB)."""

from __future__ import annotations

from sre_evals.report.compare import Delta, paired_delta, render, verdict
from sre_evals.report.details import fold_of

IDS = [f"fac-{i:04d}" for i in range(1, 21)]


def test_a_uniform_gain_is_a_win_with_a_tight_interval() -> None:
    base = {i: 0.5 for i in IDS}
    exp = {i: 0.6 for i in IDS}
    d = paired_delta("recall@5", exp, base)
    assert d is not None and d.n == 20
    assert abs(d.mean - 0.1) < 1e-9 and abs(d.low - 0.1) < 1e-9 and abs(d.high - 0.1) < 1e-9
    assert d.both_halves and verdict(d) == "win"


def test_a_gain_on_one_half_only_is_a_tie_even_with_an_interval_above_zero() -> None:
    folds = fold_of(IDS)
    base = {i: 0.0 for i in IDS}
    exp = {i: (1.0 if folds[i] == "A" else 0.0) for i in IDS}
    d = paired_delta("recall@5", exp, base)
    assert d is not None and d.fold_a == 1.0 and d.fold_b == 0.0
    assert d.low > 0  # the pooled interval alone would call it a win
    assert verdict(d) == "tie"


def test_only_items_both_runs_scored_are_paired() -> None:
    d = paired_delta("recall@5", {"fac-0001": 1.0, "fac-0002": 1.0}, {"fac-0001": 0.0})
    assert d is not None and d.n == 1
    assert paired_delta("recall@5", {"fac-0001": 1.0}, {"exs-0001": 1.0}) is None


def test_the_interval_does_not_depend_on_what_else_was_compared() -> None:
    base = {i: 0.0 for i in IDS}
    exp = {i: float(n % 3 == 0) for n, i in enumerate(IDS)}
    first = paired_delta("recall@5", exp, base)
    paired_delta("recall@5", {i: 1.0 for i in IDS}, base)  # another comparison in between
    assert paired_delta("recall@5", exp, base) == first


def test_verdicts_and_render() -> None:
    loss = Delta("recall@5", 69, -0.087, -0.174, -0.014, -0.143, -0.029)
    tie = Delta("recall@5", 69, 0.014, -0.029, 0.058, 0.0, 0.029)
    assert verdict(loss) == "loss" and verdict(tie) == "tie" and verdict(None) == "no data"
    lines = render("base", {"x": [loss, None, None]})
    assert lines[-1].startswith("| `x` | 69 | -0.087 [-0.174, -0.014] (-0.143 / -0.029) | - |")
    assert lines[-1].endswith("| loss |")
