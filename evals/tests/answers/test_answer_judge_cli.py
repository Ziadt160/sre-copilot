"""The DeepEval judge over a ChatClient with a scripted fake model, and the CLI wiring."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from sre_evals.answers import cli as answers_cli
from sre_evals.answers.judge import (
    ChatClientLLM,
    build_judge,
    check_judge_budget,
    estimate_judge_tokens,
    metric_version,
)
from sre_evals.cli import app
from sre_evals.runner.testing import W
from sre_shared.llm import ChatClient, MemoryLedger, ModelSpec, QuotaExceeded
from sre_shared.models import GoldenItem

SPEC = ModelSpec("ollama", "fake-8b", "http://127.0.0.1:9/v1", "ollama")

REPLIES: dict[str, Any] = {
    "truths": {"truths": ["t1", "t2"]},
    "claims": {"claims": ["c1", "c2"]},
    "statements": {"statements": ["s1", "s2"]},
    "verdicts": {"verdicts": [{"verdict": "yes"}, {"verdict": "no", "reason": "r"}]},
    "reason": {"reason": "one of two claims is contradicted"},
}


def _reply_for(prompt: str) -> str:
    """The reply for whichever DeepEval step this is, read from the schema we append."""
    schema = json.loads(prompt.split("matches this JSON schema:\n", 1)[1])
    (field,) = schema["properties"]
    return json.dumps(REPLIES[field])


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _resp(self, messages: list[dict[str, str]]) -> Any:
        text = (
            _reply_for(messages[0]["content"]) if "JSON schema" in messages[0]["content"] else "ok"
        )
        return SimpleNamespace(
            model="fake-8b",
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    def create(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        self.calls.append(kw)
        return self._resp(messages)


class FakeAsyncCompletions(FakeCompletions):
    async def create(self, messages: list[dict[str, str]], **kw: Any) -> Any:  # type: ignore[override]
        self.calls.append(kw)
        return self._resp(messages)


def fake_client() -> tuple[ChatClient, FakeCompletions, FakeAsyncCompletions]:
    sync, asyn = FakeCompletions(), FakeAsyncCompletions()
    client = ChatClient(
        SPEC,
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        aclient=SimpleNamespace(chat=SimpleNamespace(completions=asyn)),
    )
    return client, sync, asyn


async def test_deepeval_judge_scores_with_a_custom_model_at_temperature_zero() -> None:
    client, _, asyn = fake_client()
    judge = build_judge(client)
    assert judge.label == "ollama:fake-8b"
    # Two verdicts, one "yes": half of the claims are supported / half the statements relevant.
    faith = await judge.faithfulness("What is a Pod?", "A Pod runs containers.", ["Pods run."])
    rel = await judge.relevancy("What is a Pod?", "A Pod runs containers.")
    assert faith is not None and faith.score == 0.5
    assert faith.explanation == "one of two claims is contradicted"  # DeepEval's reason, kept
    assert rel is not None and rel.score == 0.5 and rel.explanation
    assert judge.failures == 0
    assert asyn.calls and all(c["temperature"] == 0.0 for c in asyn.calls)
    assert all(c["response_format"] == {"type": "json_object"} for c in asyn.calls)
    assert judge.llm.calls >= 7  # 5 scoring calls + a reason per metric  # type: ignore[attr-defined]
    assert judge.llm.prompt_tokens == 10 * judge.llm.calls  # type: ignore[attr-defined]
    assert metric_version().startswith("deepeval-")


async def test_judge_counts_unparseable_replies_as_failures() -> None:
    sync, asyn = FakeCompletions(), FakeAsyncCompletions()
    asyn._resp = lambda messages: SimpleNamespace(  # type: ignore[method-assign]
        model="m",
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="not json"), finish_reason="stop")
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    client = ChatClient(
        SPEC,
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        aclient=SimpleNamespace(chat=SimpleNamespace(completions=asyn)),
    )
    judge = build_judge(client)
    assert await judge.faithfulness("q", "a", ["c"]) is None
    assert judge.failures == 1


def test_chat_client_llm_sync_generate_parses_schema_and_plain_text() -> None:
    from pydantic import BaseModel

    class Out(BaseModel):
        claims: list[str]

    client, sync, _ = fake_client()
    llm = ChatClientLLM(client)
    got = llm.generate("prompt", schema=Out)
    assert got.claims == ["c1", "c2"] and sync.calls[0]["temperature"] == 0.0
    assert llm.generate("plain") == "ok"
    assert llm.get_model_name() == "ollama:fake-8b"


def test_check_judge_budget_refuses_over_budget_hosted_judge() -> None:
    spec = ModelSpec("groq", "openai/gpt-oss-120b", "http://127.0.0.1:9", "k")
    ledger = MemoryLedger()
    ledger.add(spec.label, 170_000)  # 180000 usable -> 10000 left
    client = ChatClient(spec, ledger=ledger, client=object(), aclient=object())
    check_judge_budget(client, 9_000)
    with pytest.raises(QuotaExceeded):
        check_judge_budget(client, 11_000)
    local, _, _ = fake_client()
    check_judge_budget(local, 10**9)  # local models have no daily limit
    assert estimate_judge_tokens("a" * 400, ["b" * 4000]) > estimate_judge_tokens("a", ["b"])


# -- CLI -----------------------------------------------------------------------------------


def test_select_items_by_ids_and_seeded_subset() -> None:
    items = [
        GoldenItem(id=f"fac-{i:04d}", stratum="factual", question="q", status="accepted")
        for i in range(10)
    ]
    assert [i.id for i in answers_cli.select_items(items, "fac-0003,fac-0001", None, 0)] == [
        "fac-0003",
        "fac-0001",
    ]
    a = answers_cli.select_items(items, None, 4, 5)
    assert len(a) == 4 and [i.id for i in a] == sorted(i.id for i in a)
    assert a == answers_cli.select_items(items, None, 4, 5)
    assert a != answers_cli.select_items(items, None, 4, 6)
    assert answers_cli.select_items(items, None, None, 0) == items
    with pytest.raises(typer.BadParameter):
        answers_cli.select_items(items, "nope", None, 0)
    with pytest.raises(typer.BadParameter):
        answers_cli.select_items(items, "fac-0001", 3, 0)


def _help(*args: str) -> str:
    # Typer forces a colour terminal when GITHUB_ACTIONS is set, so CI's help text carries ANSI
    # codes inside option names; compare the plain text.
    out = CliRunner().invoke(app, [*args, "--help"], env={"COLUMNS": "200"}).output
    return re.sub(r"\x1b\[[0-9;]*m", "", out)


def test_answers_commands_are_registered() -> None:
    out = _help("answers")
    for cmd in ("run", "label", "calibrate"):
        assert cmd in out
    run_help = _help("answers", "run")
    for flag in ("--experiment", "--model", "--judge", "--repeat", "--items", "--subset", "--seed"):
        assert flag in run_help


def test_run_refuses_an_over_budget_hosted_judge_before_any_database_or_model_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    spec = ModelSpec("groq", "openai/gpt-oss-120b", "http://127.0.0.1:9", "k")
    ledger = MemoryLedger()
    ledger.add(spec.label, 179_990)
    client = ChatClient(spec, ledger=ledger, client=object(), aclient=object())
    monkeypatch.setattr(answers_cli, "_client", lambda s: client)
    items = [
        GoldenItem(id="fac-0001", stratum="factual", question="q", status="accepted"),
    ]
    monkeypatch.setattr(answers_cli, "golden_path", lambda name: tmp_path / "items.jsonl")
    (tmp_path / "items.jsonl").write_text("x", encoding="utf-8")
    monkeypatch.setattr(answers_cli, "load_golden", lambda path, accepted_only=True: items)
    monkeypatch.setattr(answers_cli, "golden_version", lambda path: "v1")

    def no_db(*a: Any, **k: Any) -> Any:
        raise AssertionError("database or answerer reached")

    monkeypatch.setattr(answers_cli, "run", no_db)
    res = CliRunner().invoke(
        app, ["answers", "run", "-e", "answer-k5", "--judge", "groq:openai/gpt-oss-120b"]
    )
    assert res.exit_code == 2 and "refused" in res.output and "planned tokens" in res.output


def test_baseline_experiment_loads_as_an_answer_config() -> None:
    from sre_evals.answers.cli import load_answer_experiments
    from sre_evals.runner import load_experiments

    assert "answer-k5" not in load_experiments()  # `retrieval --all` never sees answer configs
    cfg = load_answer_experiments()["answer-k5"]
    assert cfg.legs == ["dense"] and cfg.answer is not None
    assert (cfg.answer.k, cfg.answer.compression) == (5, "none")
    assert W(1, 2) == "w1 w2"
