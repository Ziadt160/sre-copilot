"""Answers are cached as generated, so a failed run never re-spends tokens; transient provider
errors are retried."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from sre_evals.answers.cache import AnswerCache
from sre_evals.answers.runner import run_answers
from sre_evals.answers.testing import FakeAnswerer, W, chunk, golden, result
from sre_evals.runner.config import ExperimentConfig
from sre_shared.models import AnswerConfig, AnswerRequest, AnswerResult
from sre_shared.testing.seed import create_corpus_version

pytestmark = pytest.mark.integration

ITEMS = [golden(f"fac-000{i}", "factual", W(1, 12)) for i in range(1, 5)]
CFG = ExperimentConfig(name="answer-t", legs=["dense"], answer=AnswerConfig(k=5))


def answer_for(req: AnswerRequest) -> AnswerResult:
    return result(req.question, "Answer [1].", [chunk(1, W(1, 12))], [1])


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://x/v1"))


async def test_a_failed_run_keeps_its_answers_and_a_rerun_reuses_them(
    db: Any, tmp_path: Path
) -> None:
    await create_corpus_version(db)
    calls: list[str] = []

    def flaky(req: AnswerRequest) -> AnswerResult:
        calls.append(req.question)
        if req.question == "question fac-0003" and calls.count(req.question) == 1:
            raise RuntimeError("provider dropped the connection")  # not transient: run fails
        return answer_for(req)

    kw: dict[str, Any] = {
        "judge": None, "loop": "local", "golden_version": "v1-abc", "use_cache": True,
        "cache_directory": tmp_path, "answer_cfg": AnswerConfig(k=5, model="fake:m"),
    }  # fmt: skip
    with pytest.raises(RuntimeError):
        await run_answers(CFG, ITEMS, FakeAnswerer(flaky), db, **kw)
    first = list(calls)
    assert "question fac-0003" in first

    [out] = await run_answers(CFG, ITEMS, FakeAnswerer(flaky), db, **kw)
    regenerated = calls[len(first) :]
    # Only what the failed run never finished is generated again.
    assert set(regenerated) == {"question fac-0003"} | (
        {q for q in (f"question {i.id}" for i in ITEMS)} - set(first)
    )
    assert out.summary["n_answered"] == 4
    assert out.summary["cached_answers_at_start"] == len(set(first)) - 1


async def test_transient_errors_are_retried(db: Any, tmp_path: Path) -> None:
    await create_corpus_version(db)
    failures = {"left": 2}

    def sometimes(req: AnswerRequest) -> AnswerResult:
        if req.question == "question fac-0002" and failures["left"]:
            failures["left"] -= 1
            raise connection_error()
        return answer_for(req)

    [out] = await run_answers(
        CFG, ITEMS, FakeAnswerer(sometimes), db, judge=None, loop="local",
        golden_version="v1-abc", retry_delays=(0.0, 0.0, 0.0),
    )  # fmt: skip
    assert out.summary["n_answered"] == 4 and failures["left"] == 0


async def test_non_transient_errors_are_not_retried(db: Any) -> None:
    await create_corpus_version(db)
    calls = {"n": 0}

    def broken(req: AnswerRequest) -> AnswerResult:
        calls["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        await run_answers(
            CFG, ITEMS[:1], FakeAnswerer(broken), db, judge=None, loop="local",
            golden_version="v1-abc", retry_delays=(0.0, 0.0, 0.0),
        )  # fmt: skip
    assert calls["n"] == 1


def test_cache_ignores_a_cut_off_line_and_a_changed_question(tmp_path: Path) -> None:
    cache = AnswerCache("k", tmp_path)
    r = answer_for(AnswerRequest(question="question fac-0001"))
    cache.put("fac-0001", r)
    with cache.path.open("a", encoding="utf-8") as f:
        f.write('{"item_id": "fac-0002", "result": {"quest')  # crash mid-write
    again = AnswerCache("k", tmp_path)
    assert len(again) == 1 and again.get("fac-0001", "question fac-0001") is not None
    assert again.get("fac-0001", "a different question") is None
