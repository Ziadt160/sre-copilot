"""The answer runner with a fake answerer and a fake judge: stored rows, repeats, report."""

from __future__ import annotations

from typing import Any

import pytest

from sre_evals.answers.report import fetch_answer_data, render_answer_sections
from sre_evals.answers.runner import JudgeUnreliable, plan_judge_tokens, run_answers
from sre_evals.answers.testing import SRC, VER, FakeAnswerer, FakeJudge, W, chunk, golden, result
from sre_evals.report import build_reports
from sre_evals.runner.config import ExperimentConfig
from sre_shared.models import AnswerConfig, AnswerRequest, AnswerResult
from sre_shared.testing.seed import create_corpus_version

pytestmark = pytest.mark.integration

GOLD_A = W(1, 12)
GOLD_B = W(21, 32)
ITEMS = [
    golden("fac-0001", "factual", GOLD_A),
    golden("exs-0001", "exact_string", GOLD_B),
    golden("unans-0001", "unanswerable", None),
    golden("fac-0002", "factual", W(41, 52)),
]


def script(req: AnswerRequest) -> AnswerResult:
    q = req.question
    ctx = [chunk(1, GOLD_A), chunk(2, "noise noise"), chunk(3, GOLD_B)]
    if q == "question fac-0001":  # cites the gold chunk
        return result(q, "Answer [1].", ctx, [1])
    if q == "question exs-0001":  # cites a non-gold chunk and an id outside the context
        return result(q, "Answer [2] [99].", ctx, [2], invalid=["99"])
    if q == "question unans-0001":  # abstains
        return result(q, "I cannot answer this.", ctx, [], abstained=True)
    return result(q, "Answer without citation.", ctx, [])


def cfg() -> ExperimentConfig:
    return ExperimentConfig(name="answer-t", legs=["dense"], answer=AnswerConfig(k=5))


async def test_run_stores_answer_run_and_scores(db: Any) -> None:
    await create_corpus_version(db)
    answerer = FakeAnswerer(script)
    judge = FakeJudge(faith={"question fac-0001": 1.0, "question exs-0001": 0.5}, rel=0.75)
    [out] = await run_answers(
        cfg(), ITEMS, answerer, db, judge=judge, loop="ci", golden_version="v1-abc",
        answer_cfg=AnswerConfig(k=5, model="fake:m"), concurrency=2,
    )  # fmt: skip
    s = out.summary
    assert (s["n_items"], s["n_answered"], s["n_abstained"]) == (4, 3, 1)
    # Faithfulness: fac-0001 1.0, exs-0001 0.5, fac-0002 default 1.0; abstained item not judged.
    assert s["faithfulness"] == pytest.approx(2.5 / 3) and s["n_faithfulness"] == 3
    assert s["answer_relevancy"] == pytest.approx(0.75)
    assert len(judge.seen) == 3 and all(q != "question unans-0001" for q, _ in judge.seen)
    # The judge is given exactly the chunk bodies the model was given, in order.
    # ...with the header the model saw (the test chunks have header "h").
    expected = (f"h\n\n{GOLD_A}", "h\n\nnoise noise", f"h\n\n{GOLD_B}")
    assert all(c == expected for _, c in judge.seen)
    # Cited: 2 of 3 answered items cite something. Citation hit: only fac-0001 cites a gold chunk
    # among the three items with gold (fac-0002 cites nothing, exs-0001 cites noise).
    assert s["cited_rate"] == pytest.approx(2 / 3)
    assert s["citation_hit_rate"] == pytest.approx(1 / 3)
    assert s["invalid_citations"] == 1
    assert s["abstain_rate"] == pytest.approx(0.25)
    assert s["by_stratum"]["unanswerable"]["abstain_rate"] == 1.0
    assert s["prompt_tokens"] == 100 and s["context_tokens"] == 60
    assert s["latency_ms"]["p50"] == pytest.approx(50.0)
    # Requests carry the item's question and the answer config of the run.
    assert {r.question for r in answerer.calls} == {f"question {i.id}" for i in ITEMS}
    assert all(r.config is not None and r.config.model == "fake:m" for r in answerer.calls)

    run = await (
        await db.execute("select * from public.eval_runs where id = %s", [out.run_id])
    ).fetchone()
    assert (run["kind"], run["status"], run["loop"]) == ("answer", "done", "ci")
    assert run["experiment"] == "answer-t" and run["golden_version"] == "v1-abc"
    ex = run["summary"]
    assert (ex["repeat"], ex["repeats"], ex["model"], ex["judge"]) == (1, 1, "fake:m", "fake:judge")
    assert ex["judge_prompt_version"] and ex["metric_version"].startswith("deepeval-")
    assert ex["answer_prompt_version"] == "answer-v1"
    assert run["stamps"]["prompt_versions"]["judge"] == ex["judge_prompt_version"]

    rows = await (
        await db.execute(
            "select item_id, metric, k, value, details from public.eval_scores where run_id = %s",
            [out.run_id],
        )
    ).fetchall()
    by = {(r["item_id"], r["metric"]): r for r in rows}
    assert by[("fac-0001", "faithfulness")]["value"] == 1.0
    assert by[("exs-0001", "faithfulness")]["value"] == 0.5
    assert ("unans-0001", "faithfulness") not in by  # abstained: nothing to judge
    assert by[("fac-0001", "citation_hit")]["value"] == 1.0
    assert by[("exs-0001", "citation_hit")]["value"] == 0.0
    assert ("unans-0001", "citation_hit") not in by  # no gold passages
    assert by[("exs-0001", "invalid_citations")]["value"] == 1.0
    assert by[("unans-0001", "abstained")]["value"] == 1.0
    assert all(r["k"] == 0 for r in rows)
    detail = by[("fac-0001", "answer_detail")]["details"]
    assert detail["answer"] == "Answer [1]." and detail["context_chunk_ids"] == [1, 2, 3]
    assert [p["body"] for p in detail["context"]] == [GOLD_A, "noise noise", GOLD_B]
    assert detail["citations"][0]["chunk_id"] == 1 and detail["stratum"] == "factual"
    assert detail["context"][0]["source_path"] == SRC and detail["context"][0]["k8s_version"] == VER


async def test_repeat_two_stores_two_runs_sharing_a_batch(db: Any) -> None:
    await create_corpus_version(db)
    judge = FakeJudge()
    outs = await run_answers(
        cfg(), ITEMS, FakeAnswerer(script), db, judge=judge, loop="local",
        golden_version="v1", repeat=2,
    )  # fmt: skip
    assert [o.repeat for o in outs] == [1, 2] and outs[0].batch == outs[1].batch
    assert outs[0].run_id != outs[1].run_id
    n = await (
        await db.execute(
            "select count(*) as n from public.eval_runs where kind = 'answer' and"
            " summary->>'batch' = %s",
            [outs[0].batch],
        )
    ).fetchone()
    assert n["n"] == 2
    assert len(judge.seen) == 6  # 3 answered items, twice


async def test_no_judge_stores_deterministic_metrics_only(db: Any) -> None:
    await create_corpus_version(db)
    [out] = await run_answers(
        cfg(), ITEMS, FakeAnswerer(script), db, judge=None, loop="local", golden_version="v1"
    )
    assert out.summary["faithfulness"] is None and out.summary["judge"] == "none"
    n = await (
        await db.execute(
            "select count(*) as n from public.eval_scores where run_id = %s"
            " and metric in ('faithfulness', 'answer_relevancy')",
            [out.run_id],
        )
    ).fetchone()
    assert n["n"] == 0


async def test_judge_failure_is_counted_and_skipped(db: Any) -> None:
    await create_corpus_version(db)
    judge = FakeJudge(faith={"question fac-0001": None})
    [out] = await run_answers(
        cfg(),
        ITEMS,
        FakeAnswerer(script),
        db,
        judge=judge,
        loop="local",
        golden_version="v1",
        max_judge_failure_share=0.5,
    )
    assert out.summary["judge_failures"] == 1 and out.summary["n_faithfulness"] == 2
    assert out.summary["faithfulness_failures"] == 1 and out.summary["relevancy_failures"] == 0


async def test_too_many_judge_failures_store_a_failed_run(db: Any) -> None:
    await create_corpus_version(db)
    judge = FakeJudge(faith={"question fac-0001": None})  # 1 of 3 judged answers = 33% > 20%
    with pytest.raises(JudgeUnreliable):
        await run_answers(
            cfg(), ITEMS, FakeAnswerer(script), db, judge=judge, loop="local", golden_version="v1"
        )
    row = await (
        await db.execute(
            "select status, summary from public.eval_runs where kind = 'answer' order by id desc"
        )
    ).fetchone()
    assert row["status"] == "failed" and "JudgeUnreliable" in row["summary"]["error"]


async def test_answerer_error_stores_failed_run_and_reraises(db: Any) -> None:
    await create_corpus_version(db)

    def boom(req: AnswerRequest) -> AnswerResult:
        raise RuntimeError("model down")

    with pytest.raises(RuntimeError, match="model down"):
        await run_answers(
            cfg(), ITEMS, FakeAnswerer(boom), db, judge=None, loop="local", golden_version="v1"
        )
    row = await (
        await db.execute(
            "select status, summary from public.eval_runs where kind = 'answer' order by id desc"
        )
    ).fetchone()
    assert row["status"] == "failed" and "model down" in row["summary"]["error"]


async def test_report_sections_from_stored_runs(db: Any, tmp_path: Any) -> None:
    await create_corpus_version(db)
    for name in ("answer-k5", "answer-k8"):
        c = ExperimentConfig(name=name, legs=["dense"], answer=AnswerConfig(k=5))
        judge = FakeJudge(default_faith=1.0 if name == "answer-k5" else 0.5)
        await run_answers(
            c, ITEMS, FakeAnswerer(script), db, judge=judge, loop="local",
            golden_version="v1", repeat=2,
        )  # fmt: skip
    data = await fetch_answer_data(db)
    assert [e.experiment for e in data.experiments] == ["answer-k5", "answer-k8"]
    assert all(len(e.runs) == 2 for e in data.experiments)
    text = "\n".join(render_answer_sections(data))
    assert "## Answers" in text and "`answer-k8`" in text
    assert "Paired differences vs `answer-k5`" in text
    assert "-0.500 [-0.500, -0.500]* (n=3)" in text  # faithfulness: 0.5 - 1.0 on 3 items
    assert "### Noise" in text and "| `answer-k5` | faithfulness | 2 |" in text
    assert "Not calibrated yet" in text

    readme = tmp_path / "README.md"
    readme.write_text("# x\n<!-- results:start -->\n<!-- results:end -->\n", encoding="utf-8")
    details = tmp_path / "results.md"
    await build_reports(db, readme, details)
    assert "## Answers" in details.read_text(encoding="utf-8")


def test_plan_judge_tokens_scales_with_items() -> None:
    cfg_ = AnswerConfig()
    assert plan_judge_tokens(cfg_, 10) == 10 * plan_judge_tokens(cfg_, 1) > 0
