"""Tracing switches and the wrapper, with a fake LangSmith transport (nothing leaves the process)."""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Any

import httpx
import pytest
import requests
from langsmith import Client
from openai import AsyncOpenAI
from pydantic import SecretStr

from sre_core.review.tracing import client_wrapper, tracing_enabled, turn_trace
from sre_shared.settings import Settings


def _settings(key: str | None = "test-key") -> Settings:
    return Settings(langsmith_api_key=SecretStr(key) if key else None)


class RecordingSession(requests.Session):
    """Answers every LangSmith request with 200 and keeps the bodies."""

    def __init__(self) -> None:
        super().__init__()
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        self.urls.append(url)
        data = kwargs.get("data")
        if isinstance(data, (bytes, str)) and data:
            # multipart batch bodies are not JSON and not needed for these checks
            with contextlib.suppress(ValueError):
                self.bodies.append(json.loads(data))
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b"{}"
        return resp


def _ls_client(session: RecordingSession) -> Client:
    return Client(
        api_url="http://langsmith.invalid",
        api_key="test-key",
        session=session,
        auto_batch_tracing=False,
    )


def _openai_stub() -> AsyncOpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return AsyncOpenAI(
        api_key="x",
        base_url="http://llm.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_off_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SRE_TRACING", raising=False)
    assert not tracing_enabled(_settings(None))
    assert client_wrapper(_settings(None)) is None


def test_off_when_sre_tracing_is_off_even_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SRE_TRACING", "off")
    assert client_wrapper(_settings()) is None
    with turn_trace(_settings(), thread_id=uuid.uuid4(), turn=1, user_sub="u") as t:
        assert t.run_id is None


def test_on_with_a_key_and_the_wrapper_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SRE_TRACING", raising=False)
    session = RecordingSession()
    wrap = client_wrapper(_settings(), client_factory=lambda _k: _ls_client(session))
    assert wrap is not None
    plain = _openai_stub()
    wrapped = wrap(plain)
    assert wrapped is plain  # wrap_openai patches in place and returns the same client
    assert hasattr(wrapped.chat.completions.create, "__wrapped__")


async def test_a_turn_records_a_root_run_with_thread_metadata_and_child_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SRE_TRACING", raising=False)
    session = RecordingSession()
    factory = lambda _k: _ls_client(session)  # noqa: E731
    wrap = client_wrapper(_settings(), client_factory=factory)
    assert wrap is not None
    client = wrap(_openai_stub())
    thread = uuid.uuid4()

    with turn_trace(
        _settings(), thread_id=thread, turn=3, user_sub="alice", client_factory=factory
    ) as trace:
        assert trace.run_id is not None
        await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        trace.finish(outputs={"answer": "hi"})

    sent = json.dumps(session.bodies)
    assert str(thread) in sent and "alice" in sent and trace.run_id in sent
    assert all(u.startswith("http://langsmith.invalid") for u in session.urls)


async def test_outside_a_turn_a_wrapped_client_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SRE_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    session = RecordingSession()
    wrap = client_wrapper(_settings(), client_factory=lambda _k: _ls_client(session))
    assert wrap is not None
    client = wrap(_openai_stub())
    await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert session.urls == []
