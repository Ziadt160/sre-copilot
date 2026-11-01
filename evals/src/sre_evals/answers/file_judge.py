"""A `Judge` that serves judgements written by a Claude Code subagent (decision 39).

The subagent follows `evals/judging/claude-judge.md` and writes one JSON line per answer: the
answer's claims (verdict, quoted evidence, whether it answers the question) and its statements
(relevant or not). Scoring is not left to the subagent: faithfulness goes through the rubric
judge's own code (`checked`: evidence must be in the sources; `score_reply`: share supported,
capped when the answering claim is not; `explain`), and relevancy is the share of relevant
statements. So a subagent judgement and a rubric-judge call differ only in the model.

Selected with `--judge file:<path>` on `sre-eval answers judge`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from sre_evals.answers.judge import Judgement
from sre_evals.answers.rubric_judge import (
    RUBRIC_PROMPT_VERSION,
    ClaimVerdict,
    RubricReply,
    checked,
    explain,
    score_reply,
)

PREFIX = "file:"


class Statement(BaseModel):
    statement: str
    relevant: bool


class FileJudgement(BaseModel):
    item_id: str
    judge: str
    claims: list[ClaimVerdict]
    statements: list[Statement] = []


def read_judgements(path: Path) -> dict[str, FileJudgement]:
    """item_id -> judgement; one judge per file, every line valid, no item twice."""
    out: dict[str, FileJudgement] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            j = FileJudgement.model_validate(json.loads(line))
        except (ValueError, ValidationError) as e:
            raise ValueError(f"{path}:{n}: {e}") from e
        if j.item_id in out:
            raise ValueError(f"{path}:{n}: {j.item_id} judged twice")
        out[j.item_id] = j
    judges = {j.judge for j in out.values()}
    if len(judges) > 1:
        raise ValueError(f"{path}: more than one judge {sorted(judges)}")
    return out


def relevancy_of(statements: Sequence[Statement]) -> Judgement | None:
    if not statements:
        return None
    n_ok = sum(s.relevant for s in statements)
    lines = [f"{n_ok} of {len(statements)} statements relevant."]
    lines += [
        f"  [{'relevant' if s.relevant else 'off the question'}] {s.statement}" for s in statements
    ]
    return Judgement(n_ok / len(statements), "\n".join(lines))


@dataclass
class FileJudge:
    """Serves the judgements in `path`; `item_by_question` maps the run's questions to ids."""

    path: Path
    item_by_question: Mapping[str, str]
    prompt_version: str = f"{RUBRIC_PROMPT_VERSION}+subagent"
    faithfulness_failures: int = 0  # an answer the file has no judgement for
    relevancy_failures: int = 0
    no_claims: int = 0
    no_statements: int = 0
    unverified_evidence: int = 0
    label: str = field(init=False)
    _by_item: dict[str, FileJudgement] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_item = read_judgements(self.path)
        if not self._by_item:
            raise ValueError(f"{self.path}: no judgements")
        self.label = f"subagent:{next(iter(self._by_item.values())).judge}"

    @property
    def failures(self) -> int:
        return self.faithfulness_failures + self.relevancy_failures

    def _find(self, question: str) -> FileJudgement | None:
        item = self.item_by_question.get(question)
        return None if item is None else self._by_item.get(item)

    async def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> Judgement | None:
        j = self._find(question)
        if j is None:
            self.faithfulness_failures += 1
            return None
        reply, turned = checked(RubricReply(claims=j.claims), contexts)
        self.unverified_evidence += turned
        score = score_reply(reply)
        if score is None:
            self.no_claims += 1
            return None
        return Judgement(score, explain(reply, turned))

    async def relevancy(self, question: str, answer: str) -> Judgement | None:
        j = self._find(question)
        if j is None:
            self.relevancy_failures += 1
            return None
        got = relevancy_of(j.statements)
        if got is None:
            self.no_statements += 1
        return got
