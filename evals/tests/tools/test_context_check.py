"""Summary functions of tools/context_check.py, on hand-made rows (no DB, no model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[3] / "tools" / "context_check.py"
_spec = importlib.util.spec_from_file_location("context_check", _PATH)
assert _spec and _spec.loader
cc = importlib.util.module_from_spec(_spec)
sys.modules["context_check"] = cc
_spec.loader.exec_module(cc)

Row = cc.Row


def rows() -> list[object]:
    return [
        # q1: gold survives compression; q2: compression drops it; q3: no gold (unanswerable)
        Row("q1", "factual", 5, "none", 1000, 5, 2, 2),
        Row("q1", "factual", 5, "sentence", 600, 4, 1, 2),
        Row("q2", "factual", 5, "none", 800, 4, 1, 1),  # budget cut: 4 of 5 chunks fit
        Row("q2", "factual", 5, "sentence", 400, 3, 0, 1),
        Row("q3", "unanswerable", 5, "none", 900, 5, None, None),
        Row("q3", "unanswerable", 5, "sentence", 500, 5, None, None),
    ]


def test_gold_kept_counts_only_items_with_gold_and_names_what_compression_lost() -> None:
    s = cc.summarize(rows())
    assert s["k5-none"]["n_gold"] == 2 and s["k5-none"]["gold_kept"] == 1.0
    assert s["k5-sentence"]["gold_kept"] == 0.5
    assert s["k5-sentence"]["gold_lost"] == ["q2"] and s["k5-none"]["gold_lost"] == []
    assert s["k5-none"]["budget_cut"] == 1
    assert s["k5-none"]["tokens_mean"] == 900 and s["k5-sentence"]["tokens_median"] == 500


def test_paired_savings_are_per_question() -> None:
    [(k, v)] = cc.paired_savings(rows()).items()
    assert k == 5
    assert v["saved_mean"] == 400 and v["saved_median"] == 400
    assert abs(v["saved_share"] - 1200 / 2700) < 1e-9


def test_best_grade_is_none_without_gold() -> None:
    assert cc.best_grade([], []) is None
    assert cc.best_grade([], [object()]) == 0


def test_render_lists_every_setting() -> None:
    text = cc.render(cc.summarize(rows()), cc.paired_savings(rows()))
    assert "| k5-none | 3 |" in text and "| k5-sentence | 3 |" in text
    assert "saves 400 tokens per question" in text
