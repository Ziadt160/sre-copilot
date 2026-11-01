"""Calibration's judge cache: a hosted judge's quota is spent once per answer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sre_evals.answers.calibrate import plan_tokens
from sre_evals.answers.judge import Judgement
from sre_evals.answers.judge_cache import CachedJudge
from sre_evals.answers.label import LabelItem, Passage


class CountingJudge:
    label = "groq:openai/gpt-oss-120b"
    prompt_version = "judge-rubric-v1"
    faithfulness_calls = 1
    reply_tokens = 1536

    def __init__(self, score: float | None = 0.75) -> None:
        self.score = score
        self.calls = 0

    async def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> Judgement | None:
        self.calls += 1
        return None if self.score is None else Judgement(self.score, "because")

    async def relevancy(self, question: str, answer: str) -> Judgement | None:
        self.calls += 1
        return None if self.score is None else Judgement(self.score, "because")


def item(n: int) -> LabelItem:
    return LabelItem(1, f"fac-{n:04d}", "factual", f"q{n}", "answer",
                     (Passage(n, "p.md", "1.37", "t", "u", "body " * 200),))  # fmt: skip


async def test_scores_survive_a_new_run_and_failures_are_retried(tmp_path: Path) -> None:
    inner = CountingJudge()
    first = CachedJudge(inner, tmp_path)
    assert await first.faithfulness("q", "a", ["s"]) == Judgement(0.75, "because")
    assert (
        await first.faithfulness("q", "a", ["s"]) == Judgement(0.75, "because") and inner.calls == 1
    )
    again = CachedJudge(CountingJudge(), tmp_path)  # a later calibration run
    assert (
        await again.faithfulness("q", "a", ["s"]) == Judgement(0.75, "because")
        and again.judge.calls == 0
    )  # type: ignore[attr-defined]
    assert await again.faithfulness("q", "a different answer", ["s"]) == Judgement(0.75, "because")
    assert again.judge.calls == 1  # type: ignore[attr-defined]
    failing = CachedJudge(CountingJudge(score=None), tmp_path / "other")
    assert await failing.faithfulness("q", "a", ["s"]) is None
    assert await failing.faithfulness("q", "a", ["s"]) is None
    assert failing.judge.calls == 2  # type: ignore[attr-defined]


async def test_each_judge_and_prompt_version_has_its_own_file(tmp_path: Path) -> None:
    a = CachedJudge(CountingJudge(), tmp_path)
    other = CountingJudge(score=0.1)
    other.prompt_version = "judge-rubric-v2"
    b = CachedJudge(other, tmp_path)
    await a.faithfulness("q", "a", ["s"])
    assert await b.faithfulness("q", "a", ["s"]) == Judgement(0.1, "because") and a.path != b.path


async def test_the_quota_plan_counts_only_uncached_answers(tmp_path: Path) -> None:
    j = CachedJudge(CountingJudge(), tmp_path)
    items = [item(1), item(2)]
    full = plan_tokens(items, j)
    await j.faithfulness(items[0].question, items[0].answer, [p.text for p in items[0].passages])
    assert plan_tokens(items, j) == full // 2
    assert j.faithfulness_calls == 1  # planning attributes come from the wrapped judge
