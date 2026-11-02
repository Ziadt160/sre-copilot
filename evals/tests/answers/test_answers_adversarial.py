"""V05 adversarial checks of the answer evals (T07). No model or network calls.

Tests marked `xfail(strict=True)` assert the behaviour the contract asks for and fail today; each
names its finding in .orchestra/20261102-phase2-retrieval-scale/reports/V05.md. Run with
`--runxfail` to see the failures. When a finding is fixed the test XPASSes, which strict mode
reports as a failure: drop the marker then.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sre_evals.answers import report as answers_report
from sre_evals.answers.calibrate import JudgeHandle, calibrate, calibration_summary
from sre_evals.answers.judge import build_judge, check_judge_budget
from sre_evals.answers.label import Label, LabelItem, LabelSession, Passage, stratified_sample
from sre_evals.answers.report import AnswerData, AnswerExperiment, render_answer_sections
from sre_evals.answers.runner import ItemRun, _one, _score_rows, run_answers, summarize
from sre_evals.answers.scoring import score_answer
from sre_evals.answers.stats import (
    NOT_SUPPORTED,
    PARTLY,
    SUPPORTED,
    accuracy,
    cohens_kappa,
    noise_stats,
    score_to_label,
    to_binary,
)
from sre_evals.answers.testing import FakeAnswerer, FakeJudge, W, chunk, golden, result
from sre_evals.runner.config import ExperimentConfig
from sre_shared.llm import ChatClient, MemoryLedger, ModelSpec, QuotaExceeded
from sre_shared.models import ANONYMOUS, AnswerConfig, AnswerRequest, AnswerResult

# -- 1. kappa and accuracy against an independent confusion-matrix computation ------------------


def _kappa_by_hand(a: list[str], b: list[str]) -> float | None:
    """sklearn's cohen_kappa_score formula, written out over a confusion matrix."""
    classes = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for x, y in zip(a, b, strict=True):
        m[idx[x]][idx[y]] += 1
    n = len(a)
    po = sum(m[i][i] for i in range(len(classes))) / n
    rows = [sum(r) for r in m]
    cols = [sum(m[i][j] for i in range(len(classes))) for j in range(len(classes))]
    pe = sum(rows[i] * cols[i] for i in range(len(classes))) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_kappa_matches_confusion_matrix_formula_on_random_cases(n_classes: int) -> None:
    rng = random.Random(n_classes)
    pool = [SUPPORTED, PARTLY, NOT_SUPPORTED][:n_classes]
    for _ in range(2000):
        n = rng.randint(1, 25)
        # Skewed draws so single-class and unseen-label cases come up often.
        wa = [rng.random() ** 3 for _ in pool]
        wb = [rng.random() ** 3 for _ in pool]
        a = rng.choices(pool, weights=wa, k=n)
        b = rng.choices(pool, weights=wb, k=n)
        want = _kappa_by_hand(a, b)
        got = cohens_kappa(a, b)
        if want is None:
            assert got is None, (a, b)
        else:
            assert got is not None and math.isclose(got, want, abs_tol=1e-12), (a, b, got, want)
        assert accuracy(a, b) == pytest.approx(sum(x == y for x, y in zip(a, b)) / n)
        # Label order does not matter: relabelling the classes leaves kappa unchanged.
        perm = dict(zip(pool, rng.sample(pool, len(pool)), strict=True))
        k2 = cohens_kappa([perm[x] for x in a], [perm[x] for x in b])
        assert (k2 is None) == (got is None)
        if got is not None and k2 is not None:
            assert math.isclose(k2, got, abs_tol=1e-12)


def test_kappa_degenerate_inputs() -> None:
    assert cohens_kappa([], []) is None and accuracy([], []) is None
    assert cohens_kappa([SUPPORTED] * 5, [SUPPORTED] * 5) is None  # 0/0, not 1.0
    # Two raters, each one class, but different classes: po = 0, pe = 0 -> 0.0 (as sklearn).
    assert cohens_kappa([SUPPORTED] * 4, [PARTLY] * 4) == 0.0
    # A label only one rater uses.
    assert cohens_kappa([SUPPORTED, SUPPORTED, PARTLY], [SUPPORTED, NOT_SUPPORTED, PARTLY]) == (
        pytest.approx(_kappa_by_hand(
            [SUPPORTED, SUPPORTED, PARTLY], [SUPPORTED, NOT_SUPPORTED, PARTLY]
        ))
    )  # fmt: skip
    with pytest.raises(ValueError):
        accuracy([SUPPORTED], [])


# -- 2. score -> label boundaries ---------------------------------------------------------------


def test_score_to_label_boundaries_and_ratio_scores() -> None:
    assert score_to_label(0.9) == SUPPORTED and score_to_label(1.0) == SUPPORTED
    assert score_to_label(math.nextafter(0.9, 0)) == PARTLY
    assert score_to_label(0.5) == PARTLY
    assert score_to_label(math.nextafter(0.5, 0)) == NOT_SUPPORTED
    assert score_to_label(0.0) == NOT_SUPPORTED
    # DeepEval scores are ratios passed / verdicts: 9 of 10 is exactly the boundary.
    for k in range(1, 40):
        assert score_to_label(9 * k / (10 * k)) == SUPPORTED
        assert score_to_label(k / (2 * k)) == PARTLY
    assert to_binary(PARTLY) == NOT_SUPPORTED and to_binary(SUPPORTED) == SUPPORTED


# -- 3. noise ------------------------------------------------------------------------------------


def test_noise_stats_is_sample_stdev() -> None:
    s = noise_stats([{"m": 0.5}, {"m": 0.7}, {"m": 0.9}])["m"]
    assert (s.n, s.min, s.max) == (3, 0.5, 0.9) and s.mean == pytest.approx(0.7)
    assert s.stdev == pytest.approx(0.2)  # ddof 1; population stdev would be 0.163
    assert noise_stats([{"m": 0.4}])["m"].stdev == 0.0


ITEMS4 = [golden(f"fac-000{i}", "factual", W(10 * i, 10 * i + 11)) for i in range(1, 5)]


def _item_run(item: Any, faith: float | None, *, abstained: bool = False) -> ItemRun:
    ctx = [chunk(1, item.gold_passages[0].quote)]
    res = result(item.question, "" if abstained else "A [1].", ctx, [] if abstained else [1],
                 abstained=abstained)  # fmt: skip
    return ItemRun(item, res, score_answer(res, item), faithfulness=faith, relevancy=faith)


def test_noise_spread_uses_the_same_items_in_every_repeat() -> None:
    # The judge gives every item the same score in both repeats; in repeat 2 it fails to score
    # one item (None). Nothing about the judged answers varied, yet the report shows a spread.
    rep1 = summarize([_item_run(i, f) for i, f in zip(ITEMS4, [1.0, 0.0, 1.0, 1.0])], 4, 0)
    rep2 = summarize([_item_run(i, f) for i, f in zip(ITEMS4, [1.0, None, 1.0, 1.0])], 4, 1)
    for rep, s in ((1, rep1), (2, rep2)):
        s.update(experiment="answer-x", repeat=rep, model="m", judge="j")
    lines = render_answer_sections(AnswerData([AnswerExperiment("answer-x", [rep1, rep2])]))
    [row] = [x for x in lines if x.startswith("| `answer-x` | faithfulness |")]
    assert row.split(" | ")[-1].strip(" |") == "0.000", row  # today: 0.177 (0.75 vs 1.0)


# -- 4. citation / abstention accounting ---------------------------------------------------------


def test_citation_hit_cited_invalid_abstained_from_answer_result() -> None:
    item = golden("fac-0009", "factual", W(1, 12))
    gold_chunk, noise = chunk(1, W(1, 12)), chunk(2, "noise noise")
    q = item.question
    # Citing a non-gold chunk: cited, no hit. An invalid id never counts.
    s = score_answer(result(q, "x [2] [9]", [gold_chunk, noise], [2], invalid=["9"]), item)
    assert (s.cited, s.citation_hit, s.invalid_citations, s.abstained) == (True, False, 1, False)
    # The gold chunk in context but not cited: no hit (retrieval found it, the answer did not).
    s = score_answer(result(q, "x", [gold_chunk, noise], []), item)
    assert (s.cited, s.citation_hit) == (False, False)
    # Abstained on an item with gold: a miss; on an item without gold: no row (None).
    assert score_answer(result(q, "", [gold_chunk], abstained=True), item).citation_hit is False
    nogold = golden("unans-0009", "unanswerable", None)
    assert score_answer(result(q, "", [gold_chunk], abstained=True), nogold).citation_hit is None
    # Same body under another version is not a hit (match_grade needs source and version).
    other = chunk(3, W(1, 12), ver="1.32")
    assert score_answer(result(q, "x [1]", [other], [3]), item).citation_hit is False


async def test_every_answered_item_left_out_of_the_judged_mean_is_counted() -> None:
    def script(req: AnswerRequest) -> AnswerResult:
        ctx = [chunk(1, W(1, 12))]
        if req.question.endswith("0001"):
            return result(req.question, "   ", ctx, [])  # empty, not flagged as abstained
        return result(req.question, "A [1].", ctx, [1])

    answerer, judge = FakeAnswerer(script), FakeJudge(default_faith=1.0)
    cfg = ExperimentConfig(name="answer-x", legs=["dense"], answer=AnswerConfig())
    runs = [
        await _one(i, cfg, AnswerConfig(), answerer, judge, ANONYMOUS, None) for i in ITEMS4[:2]
    ]
    s = summarize(runs, 2, judge.failures)
    assert (s["n_answered"], s["n_faithfulness"], s["judge_failures"]) == (2, 1, 0)
    # The answered item that was left out of the judged mean is counted as an empty answer.
    # (Amended: the original assertion `n_answered == n_faithfulness + judge_failures` said
    # 2 == 1 + 0 right after asserting those very values, so it could not hold.)
    assert s["n_empty_answers"] == 1
    assert s["n_answered"] == s["n_faithfulness"] + s["judge_failures"] + s["n_empty_answers"]


# -- 5. storage ----------------------------------------------------------------------------------


def test_stored_context_is_what_the_model_saw() -> None:
    item = ITEMS4[0]
    run = _item_run(item, 1.0)
    detail = _score_rows(1, run)[-1][-1].obj
    # T06-1: the model sees `<source title=.. version=..>header\n\nbody</source>`.
    assert "header" in detail["context"][0]


def _replying_client(replies: dict[str, Any], *, hosted: bool = False, used: int = 0,
                     error: Exception | None = None) -> ChatClient:  # fmt: skip
    """A ChatClient whose fake transport answers each DeepEval step from `replies`."""

    class Completions:
        async def create(self, messages: list[dict[str, str]], **kw: Any) -> Any:
            if error is not None:
                raise error
            schema = json.loads(messages[0]["content"].split("JSON schema:\n", 1)[1])
            (fld,) = schema["properties"]
            text = json.dumps(replies[fld])
            return SimpleNamespace(
                model="m",
                choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                         finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )  # fmt: skip

    if hosted:
        spec = ModelSpec("groq", "openai/gpt-oss-120b", "http://127.0.0.1:9", "unused")
    else:
        spec = ModelSpec("ollama", "fake-8b", "http://127.0.0.1:9/v1", "ollama")
    ledger = MemoryLedger()
    ledger.add(spec.label, used)
    return ChatClient(spec, ledger=ledger, client=SimpleNamespace(),
                      aclient=SimpleNamespace(chat=SimpleNamespace(completions=Completions())))  # fmt: skip


BASE_REPLIES: dict[str, Any] = {
    "truths": {"truths": ["t1"]},
    "claims": {"claims": ["c1", "c2"]},
    "statements": {"statements": ["s1"]},
    "verdicts": {"verdicts": [{"verdict": "yes"}, {"verdict": "yes"}]},
    "reason": {"reason": "fake reason"},
}


@pytest.mark.integration
async def test_quota_hit_mid_run_stores_a_failed_answer_run(db: Any) -> None:
    from sre_shared.testing.seed import create_corpus_version

    await create_corpus_version(db)
    judge = build_judge(_replying_client(BASE_REPLIES, hosted=True, used=179_995))
    answerer = FakeAnswerer(lambda req: result(req.question, "A [1].", [chunk(1, "b")], [1]))
    cfg = ExperimentConfig(name="answer-v05-quota", legs=["dense"], answer=AnswerConfig())
    with pytest.raises(QuotaExceeded):
        await run_answers(cfg, ITEMS4, answerer, db, judge=judge, loop="ci",
                          golden_version="v1-x")  # fmt: skip
    rows = await (
        await db.execute(
            "select kind, status, summary from public.eval_runs where experiment = %s",
            [cfg.name],
        )
    ).fetchall()
    assert [(r["kind"], r["status"]) for r in rows] == [("answer", "failed")]
    assert "QuotaExceeded" in rows[0]["summary"]["error"]


async def test_an_unreachable_judge_is_an_error_not_a_score_gap() -> None:
    judge = build_judge(_replying_client({}, error=ConnectionError("connection refused")))
    with pytest.raises(ConnectionError):
        await judge.faithfulness("q", "a", ["p"])


async def test_an_answer_with_no_extracted_claims_is_not_scored_fully_faithful() -> None:
    judge = build_judge(_replying_client({**BASE_REPLIES, "claims": {"claims": []}}))
    score = await judge.faithfulness("q", "Some answer with claims.", ["p"])
    assert score is None, score  # today: 1.0 -> 'supported' in calibration


async def test_claims_the_passages_do_not_support_are_not_scored_supported() -> None:
    verdicts = {"verdicts": [{"verdict": "borderline"}, {"verdict": "borderline"}]}
    judge = build_judge(_replying_client({**BASE_REPLIES, "verdicts": verdicts}))
    got = await judge.faithfulness("q", "a", ["p"])
    # The labeller's 's' means "every claim supported by the passages"; DeepEval's default
    # (penalize_ambiguous_claims=False) counts "not in the passages" as supported.
    assert got is not None and score_to_label(got.score) != SUPPORTED, got


# -- 6. quota guard ------------------------------------------------------------------------------


def test_local_judge_is_never_blocked_and_hosted_is_refused_before_a_call() -> None:
    local = _replying_client(BASE_REPLIES, used=10**12)
    check_judge_budget(local, 10**12)  # no daily limit for ollama
    hosted = _replying_client({}, hosted=True, used=179_990,
                              error=AssertionError("a model call was made"))  # fmt: skip
    with pytest.raises(QuotaExceeded):
        check_judge_budget(hosted, 11)


# -- 7. report -----------------------------------------------------------------------------------


def test_two_halves_are_fixed_across_experiments() -> None:
    full = {"fac-0001": 1.0, "fac-0002": 0.0, "fac-0003": 1.0, "fac-0004": 0.0}
    x = AnswerExperiment("answer-x", [], {"faithfulness": full})
    # Same scores, but fac-0001 abstained (no faithfulness row).
    y = AnswerExperiment("answer-y", [], {"faithfulness": {k: v for k, v in full.items()
                                                           if k != "fac-0001"}})  # fmt: skip
    assert answers_report._halves(x, "faithfulness") == "1.000 / 0.000"
    # Fixed halves: A = {fac-0001, fac-0003}, B = {fac-0002, fac-0004} -> A = 1.0, B = 0.0.
    assert answers_report._halves(y, "faithfulness") == "1.000 / 0.000"  # today "0.000 / 1.000"


def test_report_answered_column_is_from_repeat_one_only() -> None:
    """F8: `Answered` covers all repeats (a range), not repeat 1 only."""
    runs = []
    for rep, n_ans in ((1, 4), (2, 2)):
        runs.append({"experiment": "answer-x", "repeat": rep, "model": "m", "judge": "j",
                     "n_answered": n_ans, "n_items": 4, "abstain_rate": 1 - n_ans / 4,
                     "latency_ms": {"p50": 1.0}})  # fmt: skip
    text = "\n".join(render_answer_sections(AnswerData([AnswerExperiment("answer-x", runs)])))
    assert "| 2-4/4 |" in text and "| 0.250 |" in text


async def test_calibration_counts_labels_it_could_not_join() -> None:
    p = (Passage(1, "s", "1.33", "t", "u", "body"),)
    items = [LabelItem(7, "fac-0001", "factual", "q", "a", p)]
    labels = [Label(7, f"fac-000{i}", "factual", SUPPORTED, "z", "t") for i in (1, 2)]
    res = await calibrate(labels, items, [JudgeHandle("j", FakeJudge(default_faith=1.0))])
    summary = calibration_summary(res, labels, "f.jsonl", None)
    j = summary["judges"]["j"]
    assert j["n"] + j["n_failed"] == summary["n_labels"]  # today 1 vs 2


# -- 8. label tool -------------------------------------------------------------------------------


def _li(iid: str, stratum: str, run_id: int = 7) -> LabelItem:
    return LabelItem(run_id, iid, stratum, "q", "a", (Passage(1, "s", "1.33", "t", "u", "b"),))


def test_stratified_sample_size_determinism_and_coverage() -> None:
    rng = random.Random(5)
    for _ in range(300):
        sizes = [rng.randint(1, 30) for _ in range(rng.randint(1, 6))]
        pool = [_li(f"s{s}-{i:03d}", f"s{s}") for s, size in enumerate(sizes) for i in range(size)]
        for n in (0, 1, len(sizes), len(pool) // 2, len(pool) - 1, len(pool), len(pool) + 3):
            seed = rng.randint(0, 9)
            got = stratified_sample(pool, n, seed)
            assert len(got) == min(n, len(pool)), (sizes, n)
            assert len({i.key for i in got}) == len(got)
            assert [i.key for i in got] == [i.key for i in stratified_sample(pool, n, seed)]
            rev = [i.key for i in stratified_sample(pool[::-1], n, seed)]
            if n < len(pool):  # sorted internally: independent of the pool's order
                assert [i.key for i in got] == rev
            else:  # n >= pool: shuffles the pool as given (order-dependent; note N1)
                assert sorted(i.key for i in got) == sorted(rev)
            if len(sizes) <= n:
                assert set(Counter(i.stratum for i in got)) == {f"s{s}" for s in range(len(sizes))}


def test_labelling_one_run_keeps_the_labels_of_another_run(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    first = LabelSession.open([_li("fac-0001", "factual", run_id=12)], out, "ziad")
    first.label(SUPPORTED)
    second = LabelSession.open([_li("fac-0001", "factual", run_id=15)], out, "ziad")
    second.label(PARTLY)  # rewrites the whole file from this session's sample only
    keys = {
        (json.loads(x)["run_id"], json.loads(x)["item_id"]) for x in out.read_text().splitlines()
    }
    assert keys == {(12, "fac-0001"), (15, "fac-0001")}, keys  # today: run 12's label is gone


def test_undo_survives_a_duplicated_label_line(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    line = {"run_id": 7, "item_id": "fac-0001", "stratum": "factual", "labeller": "z",
            "labelled_at": "t"}  # fmt: skip
    out.write_text(json.dumps({**line, "label": SUPPORTED}) + "\n"
                   + json.dumps({**line, "label": PARTLY}) + "\n")  # fmt: skip
    s = LabelSession.open([_li("fac-0001", "factual")], out, "z")
    s.undo()
    assert s.current is not None and s.current.item_id == "fac-0001"


def test_label_resume_and_undo_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    items = [_li(f"fac-000{i}", "factual") for i in range(1, 4)]
    s = LabelSession.open(items, out, "z")
    s.label(SUPPORTED)
    s.label(NOT_SUPPORTED)
    assert len(out.read_text().splitlines()) == 2  # saved at once
    r = LabelSession.open(items, out, "z")
    assert [i.item_id for i in r.queue] == ["fac-0003"]  # resume skips decided
    r.undo()  # undoes fac-0002 from the previous session
    assert [i.item_id for i in r.queue] == ["fac-0002", "fac-0003"]
    assert [json.loads(x)["item_id"] for x in out.read_text().splitlines()] == ["fac-0001"]
