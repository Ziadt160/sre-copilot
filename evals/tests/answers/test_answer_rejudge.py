"""Judging a stored answer run again: same answers, new judge scores, nothing regenerated."""

from __future__ import annotations

from typing import Any

import pytest

from sre_evals.answers.rejudge import StoredRunError, rejudge_run
from sre_evals.answers.runner import run_answers
from sre_evals.answers.testing import FakeAnswerer, FakeJudge, W, chunk, golden, result
from sre_evals.runner.config import ExperimentConfig
from sre_shared.models import AnswerConfig, AnswerRequest, AnswerResult
from sre_shared.testing.seed import create_corpus_version

pytestmark = pytest.mark.integration

ITEMS = [golden("fac-0001", "factual", W(1, 12)), golden("fac-0002", "factual", W(21, 32))]


def script(req: AnswerRequest) -> AnswerResult:
    ctx = [chunk(1, W(1, 12)), chunk(2, W(21, 32))]
    return result(req.question, "Answer [1].", ctx, [1])


async def test_rejudge_scores_the_stored_answers_without_answering(db: Any) -> None:
    await create_corpus_version(db)
    cfg = ExperimentConfig(name="answer-t", legs=["dense"], answer=AnswerConfig(k=5))
    answerer = FakeAnswerer(script)
    [gen] = await run_answers(
        cfg, ITEMS, answerer, db, judge=None, loop="local", golden_version="v1-abc",
        answer_cfg=AnswerConfig(k=5, model="fake:gen"),
    )  # fmt: skip
    assert gen.summary["faithfulness"] is None and len(answerer.calls) == 2

    judge = FakeJudge(faith={"question fac-0001": 0.5}, default_faith=1.0, rel=0.9)
    outs = await rejudge_run(db, gen.run_id, judge, repeat=2, items_by_id={i.id: i for i in ITEMS})
    assert len(outs) == 2 and len(answerer.calls) == 2  # nothing regenerated
    s = outs[0].summary
    assert s["faithfulness"] == pytest.approx(0.75) and s["answer_relevancy"] == pytest.approx(0.9)
    assert s["judged_from_run"] == gen.run_id and s["model"] == "fake:gen"
    assert s["judge"] == "fake:judge" and s["batch"] == outs[1].summary["batch"]
    # The judge saw the stored answers and passages exactly.
    assert {q for q, _ in judge.seen} == {"question fac-0001", "question fac-0002"}


async def test_rejudge_refuses_failed_and_non_answer_runs(db: Any) -> None:
    await create_corpus_version(db)
    row = await (
        await db.execute(
            "insert into public.eval_runs (kind, loop, experiment, config, stamps, golden_version,"
            " status, summary) values ('retrieval', 'local', 'x', '{}', '{}', 'v1-abc', 'done',"
            " '{}') returning id"
        )
    ).fetchone()
    with pytest.raises(StoredRunError, match="not an answer run"):
        await rejudge_run(db, row["id"], FakeJudge(), items_by_id={})
    with pytest.raises(StoredRunError, match="not an answer run"):
        await rejudge_run(db, 999999, FakeJudge(), items_by_id={})
