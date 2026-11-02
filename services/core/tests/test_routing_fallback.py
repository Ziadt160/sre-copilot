"""Model router: the fallback chain, driven through the real OpenAI SDK over fake transports."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from sre_core.agent.models import (
    AllProvidersFailed,
    extractive_answer,
    make_model_router,
    should_advance,
)
from sre_core.answer.testing import chunk
from sre_shared.llm import MemoryLedger, ModelSpec, QuotaExceeded, parse_model_spec
from sre_shared.settings import Settings

MESSAGES = [{"role": "user", "content": "why is my pod pending?"}]


def settings(**kw: object) -> Settings:
    base: dict[str, object] = {
        "groq_api_key": "g",
        "openrouter_api_key": "o",
        "nvidia_api_key": "n",
        "redis_url": None,
    }
    return Settings(_env_file=None, **{**base, **kw})  # type: ignore[arg-type]


def ok_body(model: str = "m", text: str = "answer") -> dict[str, object]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class Fleet:
    """Fake providers keyed by host; records every request."""

    def __init__(self, behaviour: dict[str, Callable[[httpx2.Request], httpx2.Response]]) -> None:
        self.behaviour = behaviour
        self.calls: list[tuple[str, str]] = []

    def factory(self, spec: ModelSpec) -> httpx2.AsyncClient:
        def handler(request: httpx2.Request) -> httpx2.Response:
            model = json.loads(request.content)["model"]
            self.calls.append((request.url.host, model))
            return self.behaviour[request.url.host](request)

        return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    def hosts(self) -> list[str]:
        return [h for h, _ in self.calls]


def status(code: int) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda r: httpx2.Response(code, json={"error": {"message": f"http {code}"}})


def answers(text: str) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda r: httpx2.Response(200, json=ok_body(text=text))


GROQ, OPENROUTER, NVIDIA = "api.groq.com", "openrouter.ai", "integrate.api.nvidia.com"


def router(fleet: Fleet, **kw: object):
    return make_model_router(
        settings(**{k: v for k, v in kw.items() if k in ("cloudflare_account_id",)}),
        ledger=kw.get("ledger", MemoryLedger()),  # type: ignore[arg-type]
        http_client_factory=fleet.factory,
        client_wrapper=kw.get("client_wrapper"),  # type: ignore[arg-type]
    )


async def test_first_choice_answers_without_touching_others() -> None:
    fleet = Fleet({GROQ: answers("from groq")})
    res = await router(fleet).complete("answer", MESSAGES, max_tokens=50)
    assert res.text == "from groq"
    assert fleet.calls == [(GROQ, "openai/gpt-oss-20b")]


async def test_429_on_groq_falls_to_openrouter() -> None:
    fleet = Fleet({GROQ: status(429), OPENROUTER: answers("from openrouter")})
    res = await router(fleet).complete("answer", MESSAGES, max_tokens=50)
    assert res.text == "from openrouter"
    # Every Groq model is tried before the next provider.
    assert fleet.hosts() == [GROQ, GROQ, GROQ, OPENROUTER]


async def test_quota_stop_skips_the_model_without_a_request() -> None:
    ledger = MemoryLedger()
    ledger.add("groq:openai/gpt-oss-20b", 190_000)  # past the 90% stop of 200000
    fleet = Fleet({GROQ: answers("from groq"), OPENROUTER: answers("x")})
    res = await router(fleet, ledger=ledger).complete("answer", MESSAGES, max_tokens=50)
    # The 20b is refused locally; Groq's other model answers.
    assert fleet.calls == [(GROQ, "openai/gpt-oss-120b")]
    assert res.text == "from groq"


async def test_every_groq_model_at_quota_goes_to_openrouter() -> None:
    ledger = MemoryLedger()
    for m in ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"):
        ledger.add(f"groq:{m}", 190_000)
    fleet = Fleet({OPENROUTER: answers("or")})
    res = await router(fleet, ledger=ledger).complete("answer", MESSAGES, max_tokens=50)
    assert res.text == "or"
    assert GROQ not in fleet.hosts()


@pytest.mark.parametrize("code", [429, 500, 502, 503, 401, 408])
async def test_provider_side_errors_advance(code: int) -> None:
    fleet = Fleet({GROQ: status(code), OPENROUTER: answers("fallback")})
    assert (await router(fleet).complete("router", MESSAGES, max_tokens=5)).text == "fallback"


@pytest.mark.parametrize("code", [400, 413, 422])
async def test_request_errors_do_not_advance(code: int) -> None:
    fleet = Fleet({GROQ: status(code), OPENROUTER: answers("must not be reached")})
    with pytest.raises(openai.APIStatusError) as err:
        await router(fleet).complete("answer", MESSAGES, max_tokens=50)
    assert err.value.status_code == code
    assert fleet.hosts() == [GROQ]  # one call, nothing after it


async def test_timeout_and_connection_error_advance() -> None:
    def boom(kind: type[Exception]) -> Callable[[httpx2.Request], httpx2.Response]:
        def raise_(r: httpx2.Request) -> httpx2.Response:
            raise kind("down", request=r)

        return raise_

    fleet = Fleet(
        {
            GROQ: boom(httpx2.ReadTimeout),
            OPENROUTER: boom(httpx2.ConnectError),
            NVIDIA: answers("nv"),
        }
    )
    res = await router(fleet).complete("answer", MESSAGES, max_tokens=50)
    assert res.text == "nv"


async def test_all_fail_raises_all_providers_failed() -> None:
    fleet = Fleet({GROQ: status(429), OPENROUTER: status(500), NVIDIA: status(503)})
    with pytest.raises(AllProvidersFailed) as err:
        await router(fleet).complete("answer", MESSAGES, max_tokens=50)
    labels = [label for label, _ in err.value.attempts]
    assert labels[0] == "groq:openai/gpt-oss-20b"
    assert labels[-1].startswith("nvidia:")
    assert {why for _, why in err.value.attempts} == {
        "RateLimitError",
        "InternalServerError",
    }


async def test_workers_ai_is_skipped_until_configured_then_substituted() -> None:
    r = router(Fleet({}))
    assert not any(c.startswith("cloudflare") for c in r.chain("answer"))
    s = settings(cloudflare_account_id="acct123", cloudflare_api_token="t")
    spec = parse_model_spec("cloudflare_workers_ai:@cf/openai/gpt-oss-20b", s)
    assert spec.base_url == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/v1"
    r2 = make_model_router(s, ledger=MemoryLedger(), http_client_factory=Fleet({}).factory)
    chain = r2.chain("answer")
    assert "cloudflare_workers_ai:@cf/openai/gpt-oss-20b" in chain
    assert chain.index("cloudflare_workers_ai:@cf/openai/gpt-oss-20b") > chain.index(
        "openrouter:google/gemma-4-31b-it:free"
    )


async def test_provider_without_a_key_is_skipped() -> None:
    r = make_model_router(
        settings(openrouter_api_key=None),
        ledger=MemoryLedger(),
        http_client_factory=Fleet({}).factory,
    )
    assert not any(c.startswith("openrouter:") for c in r.chain("answer"))
    assert any(c.startswith("nvidia:") for c in r.chain("answer"))


async def test_client_wrapper_is_applied_to_each_client_once() -> None:
    seen: list[str] = []

    def wrap(client: AsyncOpenAI) -> AsyncOpenAI:
        seen.append(str(client.base_url))
        return client

    fleet = Fleet({GROQ: status(429), OPENROUTER: answers("wrapped")})
    r = router(fleet, client_wrapper=wrap)
    labels = {c for role in ("answer", "router", "synthesize") for c in r.chain(role)}  # type: ignore[arg-type]
    assert len(seen) == len(labels)  # one wrapped client per model, shared across roles
    assert (await r.complete("answer", MESSAGES, max_tokens=50)).text == "wrapped"


def test_model_for_and_chain_order() -> None:
    r = make_model_router(settings(), ledger=MemoryLedger(), http_client_factory=Fleet({}).factory)
    assert r.model_for("synthesize") == "groq:openai/gpt-oss-120b"
    chain = r.chain("answer")
    assert chain[0] == "groq:openai/gpt-oss-20b"
    assert chain.index("groq:openai/gpt-oss-120b") < chain.index(
        "openrouter:google/gemma-4-31b-it:free"
    )
    assert len(chain) == len(set(chain))
    assert chain.index("nvidia:openai/gpt-oss-20b") < chain.index("nvidia:moonshotai/kimi-k2.6")


def test_should_advance_classification() -> None:
    assert should_advance(QuotaExceeded("x"))
    assert not should_advance(ValueError("x"))
    assert not should_advance(KeyError("x"))


# --- extractive answer ----------------------------------------------------------------


def test_extractive_answer_quotes_top_passages_with_markers() -> None:
    chunks = [
        chunk(1, "Pods are the smallest deployable units.", title="Pods"),
        chunk(2, "A Service exposes pods on a stable address.", title="Service"),
        chunk(3, "Third.", title="Three"),
        chunk(4, "Fourth passage is not used.", title="Four"),
    ]
    text, used = extractive_answer("what is a pod?", chunks)
    assert used == [1, 2, 3]
    assert "[1]" in text and "[2]" in text and "[3]" in text and "[4]" not in text
    assert "smallest deployable units" in text


def test_extractive_answer_bounds_long_passages_and_handles_empty() -> None:
    long = "Sentence number one is here. " * 100
    text, used = extractive_answer("q", [chunk(1, long, title="Long")])
    assert used == [1]
    assert len(text) < 900
    assert extractive_answer("q", []) == ("", [])
