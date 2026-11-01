"""Round 2 additions: comparability of paired differences, per-half differences, passage text."""

from __future__ import annotations

from typing import Any

from sre_evals.answers.label import Passage
from sre_evals.answers.report import AnswerData, AnswerExperiment, render_answer_sections


def run(judge: str, golden_version: str = "g1") -> dict[str, Any]:
    return {
        "experiment": "x", "repeat": 1, "model": "m", "judge": judge,
        "judge_prompt_version": "judge-v2", "metric_version": "deepeval-4", "n_items": 4,
        "golden_version": golden_version, "n_answered": 4, "abstain_rate": 0.0,
        "latency_ms": {"p50": 1.0},
    }  # fmt: skip


def exp(name: str, scores: list[float], **kw: Any) -> AnswerExperiment:
    per = {f"fac-{i + 1:04d}": s for i, s in enumerate(scores)}
    return AnswerExperiment(name, [run(**kw)], {"faithfulness": per})


def test_paired_difference_refuses_different_judges_and_golden_versions() -> None:
    base = exp("answer-k5", [1.0, 1.0, 1.0, 1.0], judge="ollama:a")
    same = exp("answer-k8", [0.5, 0.5, 0.5, 0.5], judge="ollama:a")
    other_judge = exp("answer-k3", [0.5] * 4, judge="groq:b")
    other_golden = exp("answer-k4", [0.5] * 4, judge="ollama:a", golden_version="g2")
    text = "\n".join(render_answer_sections(AnswerData([base, same, other_judge, other_golden])))
    assert "-0.500 [-0.500, -0.500]*" in text  # the comparable one gets a number
    assert text.count("not comparable (judge differs)") == 3  # one row, three metrics
    assert "not comparable (golden_version differs)" in text


def test_paired_difference_per_half_and_missing_baseline_note() -> None:
    base = exp("answer-a", [1.0, 1.0, 1.0, 1.0], judge="j")
    other = exp("answer-b", [1.0, 0.0, 1.0, 0.0], judge="j")  # B-half items (even) drop
    text = "\n".join(render_answer_sections(AnswerData([base, other])))
    assert "| `answer-b` | +0.000 (n=2) | -1.000 (n=2) |" in text
    assert "`answer-k5` has no run: the baseline is `answer-a`" in text


def test_passage_text_is_header_then_body() -> None:
    assert Passage(1, "s", "1.33", "t", "u", "body", "head").text == "head\n\nbody"
    assert Passage(1, "s", "1.33", "t", "u", "body").text == "body"
