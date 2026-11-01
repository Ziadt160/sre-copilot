"""Label sampling, the label TUI (scripted keys), calibration with fake judges, quota guard."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from sre_evals.answers.calibrate import (
    JudgeHandle,
    calibrate,
    calibration_summary,
    store_calibration,
)
from sre_evals.answers.label import (
    Label,
    LabelItem,
    LabelSession,
    Passage,
    labellable,
    read_labels,
    run_label,
    stratified_sample,
)
from sre_evals.answers.testing import FakeJudge
from sre_shared.llm import ChatClient, MemoryLedger, ModelSpec, QuotaExceeded, today

FIXED = datetime(2026, 11, 3, 10, 0, tzinfo=UTC)


class ScriptedTerminal:
    def __init__(self, keys: str) -> None:
        self.keys = list(keys)

    def read_key(self) -> str:
        return self.keys.pop(0)

    def read_line(self, prompt: str) -> str:
        raise AssertionError("the label UI never asks for a line")


def li(iid: str, stratum: str = "factual", run_id: int = 7, answer: str | None = None) -> LabelItem:
    return LabelItem(
        run_id=run_id,
        item_id=iid,
        stratum=stratum,
        question=f"question {iid}",
        answer=answer if answer is not None else f"answer {iid} [1]",
        passages=(
            Passage(11, "a.md", "1.33", "Pods", "https://x/a", f"passage one of {iid}"),
            Passage(12, "b.md", "1.33", "Jobs", "https://x/b", f"passage two of {iid}"),
        ),
    )


def con() -> Console:
    return Console(record=True, width=120, force_terminal=False)


# -- sampling ------------------------------------------------------------------------------


def test_stratified_sample_is_proportional_deterministic_and_covers_strata() -> None:
    pool = (
        [li(f"fac-{i:04d}", "factual") for i in range(30)]
        + [li(f"exs-{i:04d}", "exact_string") for i in range(8)]
        + [li(f"act-{i:04d}", "action") for i in range(2)]
    )
    a = stratified_sample(pool, 20, seed=1)
    assert len(a) == 20 and len({i.key for i in a}) == 20
    counts = {s: sum(i.stratum == s for i in a) for s in ("factual", "exact_string", "action")}
    assert counts == {"factual": 15, "exact_string": 4, "action": 1}  # 30:8:2 of 20 = 15:4:1
    assert [i.key for i in a] == [i.key for i in stratified_sample(pool, 20, seed=1)]
    assert [i.key for i in a] != [i.key for i in stratified_sample(pool, 20, seed=2)]
    # Asking for at least as many as exist returns everything; a tiny sample still spans strata.
    assert len(stratified_sample(pool, 99, seed=1)) == 40
    assert {i.stratum for i in stratified_sample(pool, 3, seed=1)} == {
        "factual",
        "exact_string",
        "action",
    }


def test_labellable_drops_abstentions_and_empty_answers() -> None:
    abst = li("a-1")
    abst = LabelItem(**{**abst.__dict__, "abstained": True})
    assert labellable([li("a-2"), abst, li("a-3", answer="  ")]) == [li("a-2")]


# -- TUI -----------------------------------------------------------------------------------


def test_label_round_trip_resume_and_undo(tmp_path: Path) -> None:
    out = tmp_path / "labels" / "mine.jsonl"
    items = [li("fac-0001"), li("fac-0002"), li("exs-0001", "exact_string"), li("fac-0003")]
    s = LabelSession.open(items, out, "ziad", now=lambda: FIXED)
    # supported, partly, "z" (unknown key), then not supported on item 3, undo it, then exit.
    c = con()
    run_label(s, ScriptedTerminal("spznux"), c)
    text = c.export_text()
    assert "question fac-0001" in text and "passage two of fac-0001" in text
    assert "unknown key 'z'" in text and "labelled 0 of 4" in text

    saved = read_labels(out)  # saved at once, undo included
    assert [(x.item_id, x.label) for x in saved] == [
        ("fac-0001", "supported"),
        ("fac-0002", "partly"),
    ]
    assert saved[0] == Label(7, "fac-0001", "factual", "supported", "ziad", FIXED.isoformat())

    # Resume: a new session skips what is decided and carries on with the rest.
    s2 = LabelSession.open(items, out, "ziad", now=lambda: FIXED)
    assert [i.item_id for i in s2.queue] == ["exs-0001", "fac-0003"]
    run_label(s2, ScriptedTerminal("nsx"), con())
    final = {x.item_id: x.label for x in read_labels(out)}
    assert final == {
        "fac-0001": "supported",
        "fac-0002": "partly",
        "exs-0001": "not_supported",
        "fac-0003": "supported",
    }
    assert s2.finished and s2.progress()["factual"]["supported"] == 2

    # Undo after resume removes the last decision on disk and returns it to the queue front.
    s3 = LabelSession.open(items, out, "ziad")
    s3.undo()
    assert [x.item_id for x in read_labels(out)][-1] == "exs-0001"
    assert s3.current is not None and s3.current.item_id == "fac-0003"


def test_label_undo_with_nothing_decided(tmp_path: Path) -> None:
    s = LabelSession.open([li("fac-0001")], tmp_path / "l.jsonl", "ziad")
    s.undo()
    assert s.message == "nothing to undo" and not (tmp_path / "l.jsonl").exists()


# -- calibration ---------------------------------------------------------------------------

# Human labels for 8 answers (item i has question "question fac-000i"):
HUMAN = [
    "supported",
    "supported",
    "supported",
    "supported",
    "partly",
    "partly",
    "not_supported",
    "not_supported",
]
IDS = [f"fac-{i:04d}" for i in range(8)]
LABELS = [Label(7, iid, "factual", lab, "ziad", "t") for iid, lab in zip(IDS, HUMAN, strict=True)]
ITEMS = [li(iid) for iid in IDS]


def judge_with(scores: list[float | None], label: str) -> JudgeHandle:
    table = {f"question {iid}": s for iid, s in zip(IDS, scores, strict=True)}
    return JudgeHandle(label, FakeJudge(label=label, faith=table, default_faith=None))


async def test_calibrate_perfect_and_imperfect_judges_and_reference() -> None:
    perfect = judge_with([1.0, 0.95, 0.9, 1.0, 0.7, 0.5, 0.3, 0.0], "perfect")
    # Imperfect: item 3 supported -> partly (0.6), item 6 not_supported -> partly (0.6), rest right.
    # 3-class: 6/8 correct. Human S4 P2 N2; judge S3 P4 N1 -> pe = (4*3 + 2*4 + 2*1) / 64 = 22/64.
    # kappa = (0.75 - 0.34375) / (1 - 0.34375) = 0.40625 / 0.65625.
    # Binary: human S4/not4; judge S3/not5; agree 7/8 (item 3 flips), pe = (4*3 + 4*5) / 64 = 0.5.
    # kappa_binary = (0.875 - 0.5) / 0.5 = 0.75.
    sloppy = judge_with([1.0, 0.95, 0.9, 0.6, 0.7, 0.5, 0.6, 0.0], "sloppy")
    res = await calibrate(LABELS, ITEMS, [perfect, sloppy], reference="perfect")
    p, q = res
    assert (p.n, p.n_failed, p.accuracy, p.kappa) == (8, 0, 1.0, pytest.approx(1.0))
    assert q.accuracy == pytest.approx(0.75)
    assert q.kappa == pytest.approx(0.40625 / 0.65625)
    assert q.accuracy_binary == pytest.approx(0.875)
    assert q.kappa_binary == pytest.approx(0.75)
    # Against the (perfect) reference, the sloppy judge's kappa equals its kappa vs the humans.
    assert q.vs_reference["kappa"] == pytest.approx(0.40625 / 0.65625)
    assert p.vs_reference == {}

    summary = calibration_summary(res, LABELS, "mine.jsonl", "perfect")
    assert summary["below_target"] == ["sloppy"]
    assert summary["human_distribution"] == {"supported": 4, "partly": 2, "not_supported": 2}
    assert summary["thresholds"] == {"supported_min": 0.9, "partly_min": 0.5}


async def test_calibrate_failed_scores_are_excluded_and_counted() -> None:
    j = judge_with([1.0, None, 0.9, 1.0, 0.7, 0.5, 0.3, 0.0], "flaky")
    [r] = await calibrate(LABELS, ITEMS, [j])
    assert (r.n, r.n_failed, r.accuracy) == (7, 1, 1.0)


async def test_calibrate_all_one_class_gives_undefined_kappa() -> None:
    labels = [Label(7, iid, "factual", "supported", "ziad", "t") for iid in IDS]
    j = judge_with([1.0] * 8, "same")
    [r] = await calibrate(labels, ITEMS, [j])
    assert r.accuracy == 1.0 and r.kappa is None and r.kappa_binary is None
    assert calibration_summary([r], labels, "f", None)["below_target"] == ["same"]


async def test_calibrate_unknown_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="reference"):
        await calibrate(LABELS, ITEMS, [judge_with([1.0] * 8, "a")], reference="b")


class _Exploding:
    """A client that fails the test if any request is made."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError("a model call was made")


def hosted_client(used: int) -> ChatClient:
    spec = ModelSpec("groq", "openai/gpt-oss-120b", "http://127.0.0.1:9", "unused")
    ledger = MemoryLedger()
    ledger.add(spec.label, today(), used)
    return ChatClient(spec, ledger=ledger, client=_Exploding(), aclient=_Exploding())


async def test_quota_guard_refuses_before_any_call() -> None:
    local = judge_with([1.0] * 8, "local")
    hosted_judge = FakeJudge(label="groq:openai/gpt-oss-120b")
    # 90% of the 200000 tokens/day limit is 180000; 179990 are used, so 10 tokens are left.
    hosted = JudgeHandle("groq", hosted_judge, hosted_client(used=179_990))
    with pytest.raises(QuotaExceeded):
        await calibrate(LABELS, ITEMS, [local, hosted])
    assert hosted_judge.seen == [] and local.judge.seen == []  # type: ignore[attr-defined]


async def test_quota_guard_allows_a_budget_that_fits() -> None:
    hosted_judge = FakeJudge(label="groq:x", default_faith=1.0)
    hosted = JudgeHandle("groq", hosted_judge, hosted_client(used=0))
    res = await calibrate(LABELS, ITEMS, [hosted])
    assert res[0].n == 8 and len(hosted_judge.seen) == 8


@pytest.mark.integration
async def test_store_calibration_row_feeds_the_report(db: Any) -> None:
    from sre_evals.answers.report import AnswerData, fetch_answer_data, render_answer_sections

    j = judge_with([1.0, 0.95, 0.9, 0.6, 0.7, 0.5, 0.6, 0.0], "ollama:qwen2.5:7b")
    res = await calibrate(LABELS, ITEMS, [j])
    summary = calibration_summary(res, LABELS, "mine.jsonl", None)
    rid = await store_calibration(db, "mine", summary)
    assert rid > 0
    data = await fetch_answer_data(db)
    assert data.calibration is not None and data.calibration["n_labels"] == 8
    text = "\n".join(render_answer_sections(AnswerData(calibration=data.calibration)))
    assert "`ollama:qwen2.5:7b`" in text and "0.750" in text and "0.619" in text
    assert "below target" in text
