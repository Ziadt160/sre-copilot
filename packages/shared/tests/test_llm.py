"""ChatClient: quota guard, usage ledger, streaming, with fake OpenAI clients."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sre_shared.config import Limits, ModelLimits, Provider
from sre_shared.llm import (
    ChatClient,
    FileLedger,
    MemoryLedger,
    ModelSpec,
    QuotaExceeded,
    UnknownModelError,
    estimate_tokens,
    today,
)

LIMITS = Limits(
    providers={
        "groq": Provider(
            order=1, base_url="https://x/v1", key_env="GROQ_API_KEY",
            models={"m": ModelLimits(tpd=1000)},
        ),
    },
    budgets={"stop_routing_at_share_of_tpd": 0.9},
)  # fmt: skip
GROQ = ModelSpec("groq", "m", "https://x/v1", "k")
OLLAMA = ModelSpec("ollama", "qwen", "http://127.0.0.1:11434/v1", "ollama")


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        msg = SimpleNamespace(content="hello")
        return SimpleNamespace(
            model=kw["model"],
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=10),
        )


class FakeAsyncCompletions(FakeCompletions):
    async def create(self, **kw: Any) -> Any:  # type: ignore[override]
        if kw.get("stream"):
            self.calls.append(kw)
            return self._stream()
        return super().create(**kw)

    async def _stream(self) -> Any:
        for piece in ("hel", "lo", ""):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])


def client(spec: ModelSpec, ledger: Any = None) -> tuple[ChatClient, FakeCompletions]:
    sync, asy = FakeCompletions(), FakeAsyncCompletions()
    c = ChatClient(
        spec,
        limits=LIMITS,
        ledger=ledger or MemoryLedger(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        aclient=SimpleNamespace(chat=SimpleNamespace(completions=asy)),
    )
    return c, sync


def test_complete_returns_usage_and_records_it() -> None:
    c, fake = client(GROQ)
    r = c.complete([{"role": "user", "content": "hi"}], max_tokens=20)
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.total_tokens) == ("hello", 40, 10, 50)
    assert fake.calls[0]["max_tokens"] == 20 and fake.calls[0]["temperature"] == 0.0
    assert c.remaining_tokens() == 900 - 50


async def test_reasoning_effort_is_sent_only_when_set() -> None:
    sync, asy = FakeCompletions(), FakeAsyncCompletions()
    c = ChatClient(
        GROQ, limits=LIMITS, ledger=MemoryLedger(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        aclient=SimpleNamespace(chat=SimpleNamespace(completions=asy)),
    )  # fmt: skip
    msgs = [{"role": "user", "content": "hi"}]
    c.complete(msgs)
    c.complete(msgs, reasoning_effort="low")
    await c.acomplete(msgs, reasoning_effort="low")
    _ = [d async for d in c.astream(msgs, reasoning_effort="low")]
    # Unset, the key is absent: providers without reasoning models reject unknown params.
    assert "reasoning_effort" not in sync.calls[0]
    assert sync.calls[1]["reasoning_effort"] == "low"
    assert [kw["reasoning_effort"] for kw in asy.calls] == ["low", "low"]


def test_quota_guard_refuses_before_sending() -> None:
    ledger = MemoryLedger()
    ledger.add("groq:m", today(), 880)
    c, fake = client(GROQ, ledger)
    with pytest.raises(QuotaExceeded, match="left today"):
        c.complete([{"role": "user", "content": "x" * 400}], max_tokens=10)
    assert fake.calls == []  # nothing was sent


def test_local_models_are_not_guarded() -> None:
    c, _ = client(OLLAMA)
    assert c.remaining_tokens() is None
    c.complete([{"role": "user", "content": "x" * 100_000}], max_tokens=10)


async def test_stream_yields_deltas_and_records_an_estimate() -> None:
    c, _ = client(GROQ)
    got = [d async for d in c.astream([{"role": "user", "content": "x" * 40}], max_tokens=5)]
    assert got == ["hel", "lo"]
    assert c.remaining_tokens() == 900 - (10 + 1)  # 40 chars of prompt + 5 of output, / 4


def test_estimate_tokens() -> None:
    assert estimate_tokens([{"role": "user", "content": "abcd" * 10}], 7) == 17


def test_file_ledger_keeps_only_today(tmp_path: Path) -> None:
    path = tmp_path / "q" / "tokens.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"2000-01-01": {"groq:m": 5}}), encoding="utf-8")
    ledger = FileLedger(path)
    ledger.add("groq:m", today(), 30)
    ledger.add("groq:m", today(), 12)
    assert ledger.used("groq:m", today()) == 42
    assert list(json.loads(path.read_text(encoding="utf-8"))) == [today()]
    assert FileLedger(tmp_path / "missing.json").used("groq:m", today()) == 0


def test_hosted_model_without_a_daily_limit_is_refused() -> None:
    with pytest.raises(UnknownModelError, match="no tokens-per-day limit"):
        client(ModelSpec("groq", "not-listed", "https://x/v1", "k"))


def test_model_spec_repr_hides_the_key() -> None:
    assert "secret-key" not in repr(ModelSpec("groq", "m", "https://x/v1", "secret-key"))


def test_local_models_get_a_longer_timeout() -> None:
    from sre_shared.llm import DEFAULT_TIMEOUT_S, LOCAL_TIMEOUT_S

    hosted = ChatClient(GROQ, limits=LIMITS, ledger=MemoryLedger())
    local = ChatClient(OLLAMA, limits=LIMITS, ledger=MemoryLedger())
    assert hosted._client.timeout == DEFAULT_TIMEOUT_S
    assert local._client.timeout == LOCAL_TIMEOUT_S > DEFAULT_TIMEOUT_S


def test_a_provider_can_set_its_own_timeout() -> None:
    limits = LIMITS.model_copy(deep=True)
    setattr(limits.providers["groq"], "timeout_s", 240)  # extra="allow" on Provider
    c = ChatClient(GROQ, limits=limits, ledger=MemoryLedger())
    assert c._client.timeout == 240.0 and c._aclient.timeout == 240.0
    assert ChatClient(GROQ, limits=LIMITS, ledger=MemoryLedger())._client.timeout == 60.0
