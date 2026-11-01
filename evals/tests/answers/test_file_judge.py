"""Subagent judgements are scored by the rubric judge's code (decision 39)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sre_evals.answers.file_judge import FileJudge, read_judgements

SRC = "The container that is terminated does not clean up the packet filtering rules."


def line(item: str, claims: list[dict[str, Any]], statements: list[dict[str, Any]] | None = None,
         judge: str = "claude-opus-5.5") -> str:  # fmt: skip
    return json.dumps({"item_id": item, "judge": judge, "claims": claims,
                       "statements": statements or []})  # fmt: skip


def write(tmp_path: Path, *lines: str) -> Path:
    p = tmp_path / "judgements.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


async def test_scores_come_from_the_rubric_code_with_the_evidence_check(tmp_path: Path) -> None:
    claims = [
        {"claim": "rules stay", "evidence": "does not clean up the packet filtering rules",
         "verdict": "supported", "answers_question": True},
        {"claim": "rules come back", "evidence": "the rules are recreated", "verdict": "supported"},
    ]  # fmt: skip
    statements = [{"statement": "a", "relevant": True}, {"statement": "b", "relevant": False}]
    path = write(tmp_path, line("exs-0026", claims, statements))
    j = FileJudge(path, {"why does DNS fail?": "exs-0026"})
    assert j.label == "subagent:claude-opus-5.5"
    faith = await j.faithfulness("why does DNS fail?", "answer", [SRC])
    # The second quote is not in the source: the claim is turned unsupported.
    assert faith is not None and faith.score == 0.5 and j.unverified_evidence == 1
    assert faith.explanation is not None and "1 quoted evidence not found" in faith.explanation
    rel = await j.relevancy("why does DNS fail?", "answer")
    assert rel is not None and rel.score == 0.5 and "off the question" in (rel.explanation or "")


async def test_an_answer_without_a_judgement_is_a_failure_not_a_score(tmp_path: Path) -> None:
    path = write(
        tmp_path, line("fac-0001", [{"claim": "c", "evidence": "x", "verdict": "supported"}])
    )
    j = FileJudge(path, {"q1": "fac-0001", "q2": "fac-0002"})
    assert await j.faithfulness("q2", "a", [SRC]) is None and j.faithfulness_failures == 1
    assert await j.relevancy("q1", "a") is None and j.no_statements == 1


def test_a_file_with_two_judges_or_a_repeated_item_is_refused(tmp_path: Path) -> None:
    c = [{"claim": "c", "evidence": "", "verdict": "unsupported"}]
    with pytest.raises(ValueError, match="more than one judge"):
        read_judgements(write(tmp_path, line("a", c), line("b", c, judge="claude-sonnet-5.5")))
    with pytest.raises(ValueError, match="judged twice"):
        read_judgements(write(tmp_path, line("a", c), line("a", c)))
    with pytest.raises(ValueError, match=":1:"):
        read_judgements(write(tmp_path, json.dumps({"item_id": "a", "claims": "nope"})))
