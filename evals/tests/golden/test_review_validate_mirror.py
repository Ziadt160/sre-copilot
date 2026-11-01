from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from sre_evals.golden import cli
from sre_evals.golden.contexts import write_contexts
from sre_evals.golden.mirror import example_id, mirror_dataset
from sre_evals.golden.review import ReviewSession, run_review
from sre_evals.golden.validate import normalize_ws, validate_file
from sre_shared.golden_io import load_golden, write_golden
from sre_shared.models import GoldenItem, GoldPassage

FIXED = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
DEP_PATH = "content/en/docs/concepts/workloads/controllers/deployment.md"
QUOTE = "Use --max-surge to bound extra Pods during a rolling update."


class ScriptedTerminal:
    def __init__(self, keys: str, lines: list[str] | None = None) -> None:
        self.keys = list(keys)
        self.lines = list(lines or [])
        self.prompts: list[str] = []

    def read_key(self) -> str:
        return self.keys.pop(0)

    def read_line(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.lines.pop(0)


def _item(
    id_: str,
    question: str,
    *,
    stratum: str = "factual",
    quote: str = QUOTE + " Extra words follow after the surge sentence.",
    tags: list[str] | None = None,
    **kw: Any,
) -> GoldenItem:
    return GoldenItem(
        id=id_,
        stratum=stratum,  # type: ignore[arg-type]
        question=question,
        expected_answer=f"answer to {id_}",
        gold_passages=[GoldPassage(source_path=DEP_PATH, k8s_version="1.34", quote=quote)],
        origin="synthetic",
        status="candidate",
        synth_model="ollama:fake",
        tags=tags or [],
        **kw,
    )


def _candidates() -> list[GoldenItem]:
    return [
        _item("fac-0001", "What bounds extra pods?"),
        _item("fac-0002", "How are surge pods limited?"),
        _item(
            "exs-0001",
            "Which status shows restarts?",
            stratum="exact_string",
            tags=["exact:CrashLoopBackOff"],
        ),
        _item("fac-0003", "What else follows?"),
    ]


def _session(tmp_path: Path) -> ReviewSession:
    src = tmp_path / "candidates.jsonl"
    if not src.exists():
        write_golden(src, _candidates())
    return ReviewSession.open(src, tmp_path / "items.jsonl", "ziad", now=lambda: FIXED)


def _console() -> Console:
    return Console(record=True, width=120, force_terminal=False)


def test_review_round_trip(tmp_path: Path) -> None:
    s = _session(tmp_path)
    term = ScriptedTerminal(
        "aqtarsuqrx",
        [
            "How is the number of surge Pods limited?",  # rewrite fac-0002
            "Use --max-surge to bound   extra Pods",  # trim (different whitespace)
            "Which state is shown?",  # rewrite of exs-0001 without its string: refused
        ],
    )
    run_review(s, term, _console())

    out = load_golden(tmp_path / "items.jsonl")
    by_id = {i.id: i for i in out}
    assert [i.id for i in out] == ["fac-0001", "fac-0002", "exs-0001"]
    a, b, c = by_id["fac-0001"], by_id["fac-0002"], by_id["exs-0001"]
    assert (a.status, a.origin, a.reviewer, a.reviewed_at) == (
        "accepted",
        "synthetic",
        "ziad",
        FIXED,
    )
    assert (b.status, b.origin) == ("accepted", "rewritten")
    assert b.question == "How is the number of surge Pods limited?"
    assert b.gold_passages[0].quote == "Use --max-surge to bound extra Pods"
    assert b.expected_answer == "answer to fac-0002" and b.synth_model == "ollama:fake"
    assert (c.status, c.question) == ("rejected", "Which status shows restarts?")
    assert "fac-0003" not in by_id  # skipped item stays undecided

    # resume: only the skipped item is left; accepting finishes the session
    s2 = _session(tmp_path)
    assert s2.queue == ["fac-0003"]
    run_review(s2, ScriptedTerminal("a"), _console())
    assert s2.finished
    final = load_golden(tmp_path / "items.jsonl")
    assert [i.status for i in final] == ["accepted", "accepted", "rejected", "accepted"]
    assert s2.rewritten_share() == pytest.approx(1 / 3)


def test_review_undo_restores_and_rewrite_guard(tmp_path: Path) -> None:
    s = _session(tmp_path)
    s.accept()
    s.undo()
    assert s.current is not None and s.current.id == "fac-0001"
    assert not (tmp_path / "items.jsonl").read_text()  # decision removed from disk
    s.reject()
    assert load_golden(tmp_path / "items.jsonl")[0].status == "rejected"
    s.skip()  # fac-0002 goes to the back
    assert s.current is not None and s.current.id == "exs-0001"
    assert not s.set_question("No string here?") and "exact string" in s.message
    assert s.set_question("What does CrashLoopBackOff mean?")
    assert s.current.origin == "rewritten"
    assert not s.trim_quote("words not in the quote")
    assert "sub-span" in s.message
    s.undo()  # undo the earlier reject of fac-0001 while another item is open
    assert s.current is not None and s.current.id == "fac-0001"


def test_review_renders_progress_and_quote(tmp_path: Path) -> None:
    s = _session(tmp_path)
    con = _console()
    run_review(s, ScriptedTerminal("x"), con)
    text = con.export_text()
    assert "fac-0001" in text and "What bounds extra pods?" in text
    assert DEP_PATH in text and "v1.34" in text
    assert "below 30% target" in text
    assert "pending" in text


# --- validate + mirror ---------------------------------------------------------------


def _accepted(id_: str, question: str, quote: str, **kw: Any) -> GoldenItem:
    item = _item(id_, question, quote=quote, **kw)
    item.status = "accepted"
    item.reviewer = "ziad"
    return item


async def test_validate_good_and_bad(db: Any, corpus: int, tmp_path: Path) -> None:
    good = [
        _accepted("fac-0001", "q1?", QUOTE),
        # whitespace differs from the stored text
        _accepted("fac-0002", "q2?", "Use --max-surge\nto bound   extra Pods"),
        _accepted(
            "exs-0001",
            "What is --max-surge for?",
            QUOTE,
            stratum="exact_string",
            tags=["exact:--max-surge"],
        ),
        _item("fac-0009", "still a candidate"),  # not checked
    ]
    p = tmp_path / "good.jsonl"
    write_golden(p, good)
    rep = await validate_file(db, p)
    assert rep.ok, rep.errors
    assert rep.counts["factual"]["accepted"] == 2 and rep.counts["exact_string"]["accepted"] == 1

    bad = [
        _accepted("fac-0001", "q1?", "A sentence that appears nowhere in the document."),
        _accepted(
            "exs-0001", "Which flag?", QUOTE, stratum="exact_string", tags=["exact:--max-surge"]
        ),
        _accepted("exs-0002", "No tag?", QUOTE, stratum="exact_string"),
        _accepted(
            "fac-0004",
            "wrong doc?",
            QUOTE,
        ),
    ]
    bad[3].gold_passages[0].source_path = "content/en/docs/missing.md"
    bad.append(_accepted("fac-0001", "dup id?", QUOTE))
    bp = tmp_path / "bad.jsonl"
    # write_golden does not check ids, so duplicates can be planted
    write_golden(bp, bad)
    rep = await validate_file(db, bp)
    joined = "\n".join(rep.errors)
    assert not rep.ok
    assert "fac-0001: passage 0" in joined and "quote not found" in joined
    assert "exs-0001: question does not contain '--max-surge'" in joined
    assert "exs-0002: exact_string item has no" in joined
    assert "no such document" in joined
    assert "duplicate id fac-0001" in joined


async def test_validate_cli_exit_codes(db: Any, corpus: int, pg_dsn: str, tmp_path: Path) -> None:
    runner = CliRunner()
    good = tmp_path / "good.jsonl"
    write_golden(good, [_accepted("fac-0001", "q1?", QUOTE)])
    bad = tmp_path / "bad.jsonl"
    write_golden(bad, [_accepted("fac-0001", "q1?", "nowhere in the docs")])
    r_good = await asyncio.to_thread(
        runner.invoke, cli.app, ["validate", "--in", str(good), "--dsn", pg_dsn]
    )
    r_bad = await asyncio.to_thread(
        runner.invoke, cli.app, ["validate", "--in", str(bad), "--dsn", pg_dsn]
    )
    assert r_good.exit_code == 0, r_good.output
    assert "valid" in r_good.output
    assert r_bad.exit_code == 1 and "FAIL" in r_bad.output


def test_normalize_ws() -> None:
    assert normalize_ws("a \n  b\t c ") == "a b c"


class FakeLangSmith:
    def __init__(self, exists: bool = False) -> None:
        self.exists = exists
        self.created: list[str] = []
        self.examples: list[dict[str, Any]] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return self.exists

    def create_dataset(self, name: str, **_: Any) -> None:
        self.created.append(name)

    def create_examples(self, *, dataset_name: str, examples: list[dict[str, Any]]) -> None:
        assert dataset_name == "ds"
        self.examples = examples


def test_mirror_upserts_accepted_only(tmp_path: Path) -> None:
    p = tmp_path / "items.jsonl"
    items = [
        _accepted("fac-0001", "q1?", QUOTE),
        _item("fac-0002", "cand"),
        _accepted("fac-0003", "q3?", QUOTE),
    ]
    items[2].status = "rejected"
    write_golden(p, items)
    fake = FakeLangSmith()
    res = mirror_dataset(p, "ds", api_key="k", client_factory=lambda key: fake)
    assert res.mirrored == 1 and fake.created == ["ds"]
    ex = fake.examples[0]
    assert ex["id"] == example_id("ds", "fac-0001")
    assert ex["inputs"]["question"] == "q1?"
    assert ex["outputs"]["gold_passages"][0]["quote"] == QUOTE
    assert ex["metadata"]["golden_version"].startswith(f"{tmp_path.name}-")
    # existing dataset is not recreated
    fake2 = FakeLangSmith(exists=True)
    mirror_dataset(p, "ds", api_key="k", client_factory=lambda key: fake2)
    assert fake2.created == []


def test_mirror_skips_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(
        "sre_evals.golden.mirror.get_settings", lambda: type("S", (), {"langsmith_api_key": None})()
    )
    p = tmp_path / "items.jsonl"
    write_golden(p, [_accepted("fac-0001", "q?", QUOTE)])

    def boom(key: str) -> Any:
        raise AssertionError("client must not be created")

    res = mirror_dataset(p, "ds", client_factory=boom)
    assert res.skipped_reason and "no LangSmith API key" in res.skipped_reason
    r = CliRunner().invoke(cli.app, ["mirror", "--in", str(p)])
    assert r.exit_code == 0 and "skipped" in r.output


# --- pick + synthesize through the CLI ---------------------------------------------------


async def test_cli_pick_then_synthesize(
    db: Any,
    corpus: int,
    pg_dsn: str,
    tmp_path: Path,
    make_llm: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    ctx = tmp_path / "ctx.jsonl"
    r = await asyncio.to_thread(
        runner.invoke,
        cli.app,
        [
            "pick",
            "--stratum",
            "factual",
            "--n",
            "3",
            "--seed",
            "5",
            "--out",
            str(ctx),
            "--dsn",
            pg_dsn,
        ],
    )
    assert r.exit_code == 0, r.output
    llm, _ = make_llm()
    monkeypatch.setattr(cli, "make_llm", lambda model: llm)
    out = tmp_path / "cand.jsonl"
    r = await asyncio.to_thread(
        runner.invoke,
        cli.app,
        ["synthesize", "--contexts", str(ctx), "--model", "ollama:qwen2.5:7b", "--out", str(out)],
    )
    assert r.exit_code == 0, r.output
    items = load_golden(out)
    assert len(items) == 3 and all(i.status == "candidate" for i in items)
    assert "filter rejected 0" in r.output


def test_cli_synthesize_quota_refusal(tmp_path: Path) -> None:
    from sre_evals.golden.contexts import ContextRow

    rows = [
        ContextRow(
            chunk_key=f"k{i}",
            source_path="p.md",
            k8s_version="1.34",
            url="u",
            title="t",
            body="x" * 2000,
            stratum="factual",
        )
        for i in range(300)
    ]
    ctx = tmp_path / "ctx.jsonl"
    write_contexts(ctx, rows)
    r = CliRunner().invoke(
        cli.app,
        [
            "synthesize",
            "--contexts",
            str(ctx),
            "--model",
            "groq:openai/gpt-oss-20b",
            "--out",
            str(tmp_path / "o.jsonl"),
        ],
    )
    assert r.exit_code == 2 and "quota guard" in r.output
