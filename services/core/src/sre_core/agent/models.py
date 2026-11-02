"""Model routing with a fallback chain (Phase 3, T14).

For each role the chain is the config's first-choice model, then the models of each provider in
`fallback.providers` order (Groq's other models, OpenRouter free, Workers AI, NVIDIA). A model
is left out of the chain when its provider has no key or an unresolved base_url, or when the
provider is not allowed for user traffic. The router advances on anything that is the
provider's or the quota's problem (the 90% daily stop, 429, timeouts, 5xx, connection errors,
an unusable key or retired model) and stops on an error that is the request's own (400, 413,
422): another provider would refuse the same request, and retrying it would burn quota.
When every model fails, `AllProvidersFailed` is raised; the caller (the graph) then answers
with `extractive_answer`, which needs no model.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import get_args

import httpx2
import openai
from openai import AsyncOpenAI

from sre_core.agent.ports import ModelRole
from sre_core.answer.prompts import group_sources
from sre_shared.config import AgentConfig, Limits, load_agent_config, load_limits
from sre_shared.llm import (
    DEFAULT_TIMEOUT_S,
    LOCAL_TIMEOUT_S,
    ChatClient,
    ChatResult,
    Message,
    ModelSpec,
    QuotaExceeded,
    RedisLedger,
    TokenLedger,
    parse_model_spec,
)
from sre_shared.models import RetrievedChunk
from sre_shared.settings import Settings, get_settings

log = logging.getLogger(__name__)

TRAFFIC = "user_traffic"
# The chain itself is the retry: a provider that answers 429 should not be hammered by the SDK's
# own backoff while the next provider is ready.
CHAIN_SDK_RETRIES = 0
# Status codes that mean "this provider cannot serve us" rather than "this request is wrong":
# bad or revoked key, no credit, retired model, request timeout, rate limit.
_ADVANCE_STATUS = frozenset({401, 402, 403, 404, 408, 429})

ClientWrapper = Callable[[AsyncOpenAI], AsyncOpenAI]
HttpClientFactory = Callable[[ModelSpec], httpx2.AsyncClient | None]


class AllProvidersFailed(RuntimeError):
    """Every model in the chain failed or was out of quota. `attempts` lists
    (model label, error class) in order, never keys or request text."""

    def __init__(self, role: str, attempts: Sequence[tuple[str, str]]) -> None:
        self.role = role
        self.attempts = list(attempts)
        tried = ", ".join(f"{label} ({why})" for label, why in self.attempts) or "no model"
        super().__init__(f"no model answered for role {role!r}: {tried}")


def should_advance(exc: BaseException) -> bool:
    """True when the next model may succeed where this one failed."""
    if isinstance(
        exc,
        (QuotaExceeded, openai.APITimeoutError, openai.APIConnectionError, asyncio.TimeoutError),
    ):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500 or exc.status_code in _ADVANCE_STATUS
    return False


@dataclass
class _Link:
    label: str
    client: ChatClient


def _usable(spec: ModelSpec) -> bool:
    if "{" in spec.base_url:  # unresolved placeholder (Workers AI without an account id)
        return False
    return not spec.hosted or bool(spec.api_key)


def _allowed(limits: Limits, provider: str) -> bool:
    prov = limits.providers.get(provider)
    return prov is not None and TRAFFIC in prov.allowed_for


def _provider_models(limits: Limits, provider: str, prefer: str) -> list[str]:
    models = limits.providers[provider].models
    names = list(models)
    # The same model name as the first choice goes first (the 20b role falls back to the 20b
    # elsewhere before it tries a different family).
    names.sort(key=lambda m: m != prefer)
    return names


class FallbackRouter:
    """Implements `ModelRouter`; see the module docstring."""

    def __init__(self, chains: dict[ModelRole, list[_Link]], firsts: dict[ModelRole, str]) -> None:
        self._chains = chains
        self._firsts = firsts

    def model_for(self, role: ModelRole) -> str:
        return self._firsts[role]

    def chain(self, role: ModelRole) -> list[str]:
        return [link.label for link in self._chains[role]]

    async def complete(
        self,
        role: ModelRole,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        attempts: list[tuple[str, str]] = []
        for link in self._chains[role]:
            try:
                return await link.client.acomplete(
                    messages, max_tokens=max_tokens, response_format=response_format
                )
            except Exception as exc:
                if not should_advance(exc):
                    raise
                log.warning("model %s failed (%s); trying the next", link.label, type(exc).__name__)
                attempts.append((link.label, type(exc).__name__))
        raise AllProvidersFailed(role, attempts)


def make_model_router(
    settings: Settings | None = None,
    *,
    client_wrapper: ClientWrapper | None = None,
    config: AgentConfig | None = None,
    limits: Limits | None = None,
    ledger: TokenLedger | None = None,
    http_client_factory: HttpClientFactory | None = None,
) -> FallbackRouter:
    """`client_wrapper` is applied to every AsyncOpenAI client (T16 adds tracing with it).
    `ledger` defaults to Redis when `settings.redis_url` is set, otherwise to the file ledger.
    `http_client_factory` lets tests put a fake transport under each client."""
    settings = settings or get_settings()
    config = config or load_agent_config(settings.config_dir)
    limits = limits or load_limits(settings.config_dir)
    if ledger is None and settings.redis_url:
        ledger = RedisLedger.from_url(settings.redis_url)

    clients: dict[str, ChatClient] = {}  # one per model label, shared by every role

    def client_for(spec: ModelSpec) -> ChatClient:
        if spec.label not in clients:
            raw = AsyncOpenAI(
                base_url=spec.base_url,
                api_key=spec.api_key or "none",
                max_retries=CHAIN_SDK_RETRIES,
                http_client=http_client_factory(spec) if http_client_factory else None,
                timeout=_timeout(limits, spec),
            )
            aclient = client_wrapper(raw) if client_wrapper else raw
            clients[spec.label] = ChatClient(
                spec, ledger=ledger, limits=limits, aclient=aclient, max_retries=0
            )
        return clients[spec.label]

    chains: dict[ModelRole, list[_Link]] = {}
    firsts: dict[ModelRole, str] = {}
    roles: tuple[ModelRole, ...] = get_args(ModelRole)
    for role in roles:
        first = getattr(config.models, role)
        firsts[role] = first
        specs = [parse_model_spec(first, settings, limits)]
        prefer = specs[0].model
        for provider in config.fallback.providers:
            if not _allowed(limits, provider):
                continue
            for model in _provider_models(limits, provider, prefer):
                specs.append(parse_model_spec(f"{provider}:{model}", settings, limits))
        seen: set[str] = set()
        links: list[_Link] = []
        for spec in specs:
            if spec.label in seen or not _usable(spec):
                continue
            seen.add(spec.label)
            links.append(_Link(spec.label, client_for(spec)))
        chains[role] = links
    return FallbackRouter(chains, firsts)


def _timeout(limits: Limits, spec: ModelSpec) -> float:
    prov = limits.providers.get(spec.provider)
    own = getattr(prov, "timeout_s", None) if prov is not None else None
    if own is not None:
        return float(own)
    return DEFAULT_TIMEOUT_S if spec.hosted else LOCAL_TIMEOUT_S


# --- the last fallback: no model -------------------------------------------------------

EXTRACT_MAX_PASSAGES = 3
EXTRACT_MAX_CHARS = 600  # per passage
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)


def _quote(body: str) -> str:
    text = " ".join(_HEADING.sub("", body).split())
    if len(text) <= EXTRACT_MAX_CHARS:
        return text
    cut = text[:EXTRACT_MAX_CHARS]
    # End on a sentence or word boundary, not mid-word.
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    cut = cut[: end + 1] if end > EXTRACT_MAX_CHARS // 2 else cut.rsplit(" ", 1)[0]
    return cut + " ..."


def extractive_answer(question: str, context: Sequence[RetrievedChunk]) -> tuple[str, list[int]]:
    """The answer when no model is available: the top passages quoted, each tagged with the
    `[n]` of the source it came from, plus the source numbers used (1-based, in the same
    numbering the prompt gives the model, so `resolve_citations` maps them to citations).
    Nothing is generated, so nothing can be ungrounded. Empty context -> ("", [])."""
    del question  # the retriever already ranked the passages for it
    groups = group_sources(context)
    parts: list[str] = []
    used: list[int] = []
    for n, members in enumerate(groups[:EXTRACT_MAX_PASSAGES], start=1):
        body = "\n\n".join(m.body for m in members if m.body)
        quote = _quote(body)
        if not quote:
            continue
        title = members[0].title
        parts.append(f'From {title}: "{quote}" [{n}]')
        used.append(n)
    if not parts:
        return "", []
    intro = "The language models are unavailable right now. Relevant passages from the docs:"
    return intro + "\n\n" + "\n\n".join(parts), used


__all__ = [
    "AllProvidersFailed",
    "FallbackRouter",
    "extractive_answer",
    "make_model_router",
    "should_advance",
]
