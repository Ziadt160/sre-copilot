"""Chat completions over any OpenAI-compatible endpoint (Ollama, Groq, OpenRouter, ...).

Provider endpoints and per-model daily limits come from `config/limits.yaml`; API keys are
read from settings by name and never logged. Every call has a timeout, and the OpenAI SDK
retries 429s, 5xx and connection errors with backoff that honours `retry-after` (chat calls
have no side effects, so retrying is safe). Hosted models are guarded by a token ledger over a
rolling 24 hours: a call that would take a model past `stop_share` of its tokens-per-day is
refused before it is sent, because the binding free-tier limit is tokens per model per day.
Groq counts that day as a rolling window, not a calendar day (2026-11-02: tokens used the
evening before still counted the next morning, and its count fell minute by minute), so a
ledger that reset at midnight UTC let runs start that Groq then refused with 429.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI, OpenAI

from sre_shared.config import Limits, load_limits
from sre_shared.settings import Settings, get_settings

DEFAULT_TIMEOUT_S = 60.0
# A local model shares the lab machine (GPU memory, CPU offload under load): a 7B answer took
# over 60 s while an ingest held the CPU, so local calls get longer before timing out.
LOCAL_TIMEOUT_S = 300.0
DEFAULT_RETRIES = 3
CHARS_PER_TOKEN = 4  # estimate for the quota guard; actual usage is recorded after the call

Message = dict[str, str]


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    base_url: str
    api_key: str | None = field(repr=False)  # never in logs or tracebacks

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def hosted(self) -> bool:
        return self.provider != "ollama"


def parse_model_spec(
    spec: str, settings: Settings | None = None, limits: Limits | None = None
) -> ModelSpec:
    """`ollama:qwen2.5:7b` or `groq:openai/gpt-oss-20b` -> endpoint and key (key by name only)."""
    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"model must look like provider:name, got {spec!r}")
    settings = settings or get_settings()
    limits = limits or load_limits()
    if provider == "ollama":
        return ModelSpec(provider, model, settings.ollama_base_url.rstrip("/") + "/v1", "ollama")
    if provider not in limits.providers or not limits.providers[provider].base_url:
        raise ValueError(
            f"unknown provider {provider!r}; known: ollama, {sorted(limits.providers)}"
        )
    prov = limits.providers[provider]
    key_attr = (prov.key_env or "").lower()
    secret = getattr(settings, key_attr, None)
    key = secret.get_secret_value() if secret is not None else None
    assert prov.base_url is not None
    return ModelSpec(provider, model, prov.base_url, key)


class UnknownModelError(ValueError):
    """A hosted model with no daily token limit configured: refused, because it would be
    unguarded."""


class QuotaExceeded(RuntimeError):
    """The call would take a hosted model past its share of the daily token limit."""


WINDOW_S = 24 * 3600  # tokens-per-day is counted over the last 24 hours (module doc)
_MERGE_S = 60  # calls within a minute share one ledger entry, so the file stays small


class TokenLedger(Protocol):
    def used(self, label: str) -> int: ...
    def add(self, label: str, tokens: int) -> None: ...


def _in_window(entries: Sequence[Sequence[float]], now: float) -> int:
    return int(sum(n for t, n in entries if now - t < WINDOW_S))


class MemoryLedger:
    """Per-process counts (tests, short scripts), over the same rolling window."""

    def __init__(self, clock: Any = time.time) -> None:
        self._entries: dict[str, list[list[float]]] = {}
        self._clock = clock

    def used(self, label: str) -> int:
        return _in_window(self._entries.get(label, []), self._clock())

    def add(self, label: str, tokens: int) -> None:
        self._entries.setdefault(label, []).append([self._clock(), float(tokens)])


class FileLedger:
    """Counts in a JSON file shared by lab processes (`data/quota/tokens.json`):
    `{"window_s": 86400, "entries": {label: [[unix_time, tokens], ...]}}`, entries older than the
    window dropped on every write. Writes are atomic replaces; concurrent processes can
    undercount slightly, which the 90% stop absorbs. The running API will use Redis counters
    instead (design doc, budgets). A file in the old per-calendar-day format is converted on
    first read, conservatively: today's totals dated then, an earlier day's dated at its end."""

    def __init__(self, path: Path | None = None, clock: Any = time.time) -> None:
        self.path = path or get_settings().data_dir / "quota" / "tokens.json"
        self._clock = clock

    def _read(self) -> dict[str, list[list[float]]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        if "entries" in data:
            return {k: [list(e) for e in v] for k, v in data["entries"].items()}
        now = self._clock()
        day_now = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
        out: dict[str, list[list[float]]] = {}
        for day, counts in data.items():  # old format: {"YYYY-MM-DD": {label: tokens}}
            try:
                end = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() + 86400
            except (TypeError, ValueError):
                continue
            at = now if day == day_now else min(end, now)
            for label, n in (counts or {}).items():
                out.setdefault(label, []).append([at, float(n)])
        self._write(out)  # dated once: a re-read must not move "now" forward
        return out

    def used(self, label: str) -> int:
        return _in_window(self._read().get(label, []), self._clock())

    def add(self, label: str, tokens: int) -> None:
        now = self._clock()
        entries = {k: [e for e in v if now - e[0] < WINDOW_S] for k, v in self._read().items()}
        mine = entries.setdefault(label, [])
        if mine and now - mine[-1][0] < _MERGE_S:
            mine[-1][1] += tokens
        else:
            mine.append([now, float(tokens)])
        self._write(entries)

    def _write(self, entries: dict[str, list[list[float]]]) -> None:
        data = {"window_s": WINDOW_S, "entries": {k: v for k, v in entries.items() if v}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def estimate_tokens(messages: Sequence[Message], max_tokens: int) -> int:
    return sum(len(m.get("content", "")) for m in messages) // CHARS_PER_TOKEN + max_tokens


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatClient:
    """One model behind an OpenAI-compatible endpoint. `client` / `aclient` may be injected
    (tests pass fakes with the `chat.completions.create` surface)."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        timeout: float | None = None,
        max_retries: int = DEFAULT_RETRIES,
        ledger: TokenLedger | None = None,
        limits: Limits | None = None,
        stop_share: float | None = None,
        client: Any = None,
        aclient: Any = None,
    ) -> None:
        self.spec = spec
        limits = limits or load_limits()
        self.tpd = limits.model_limits(spec.provider, spec.model).tpd if spec.hosted else None
        if spec.hosted and self.tpd is None:
            # Found by V04: an unlisted hosted model had no daily guard at all.
            raise UnknownModelError(
                f"{spec.label}: no tokens-per-day limit in config/limits.yaml; list the model "
                "with its limits before using it"
            )
        self.stop_share = (
            stop_share
            if stop_share is not None
            else float(limits.budgets.get("stop_routing_at_share_of_tpd", 0.9))
        )
        self.ledger: TokenLedger = ledger or (FileLedger() if spec.hosted else MemoryLedger())
        key = spec.api_key or "none"
        if timeout is None:
            # A provider may set its own (`timeout_s` in limits.yaml): large models on free
            # endpoints can take longer than 60 s for a long reply.
            prov = limits.providers.get(spec.provider)
            own = getattr(prov, "timeout_s", None) if prov is not None else None
            timeout = (
                float(own)
                if own is not None
                else (DEFAULT_TIMEOUT_S if spec.hosted else LOCAL_TIMEOUT_S)
            )
        # Duck-typed on purpose: tests inject fakes with the `chat.completions.create` surface,
        # and messages are plain role/content dicts.
        self._client: Any = client or OpenAI(
            base_url=spec.base_url, api_key=key, timeout=timeout, max_retries=max_retries
        )
        self._aclient: Any = aclient or AsyncOpenAI(
            base_url=spec.base_url, api_key=key, timeout=timeout, max_retries=max_retries
        )

    # -- quota --------------------------------------------------------------------------

    def remaining_tokens(self) -> int | None:
        """Tokens left before the stop share, or None when the model has no daily limit."""
        if self.tpd is None:
            return None
        return int(self.tpd * self.stop_share) - self.ledger.used(self.spec.label)

    def check_quota(self, planned_tokens: int) -> None:
        left = self.remaining_tokens()
        if left is not None and planned_tokens > left:
            raise QuotaExceeded(
                f"{self.spec.label}: {planned_tokens} planned tokens > {left} left in the last 24 h "
                f"({self.stop_share:.0%} of {self.tpd} tokens/day)"
            )

    def _record(self, result: ChatResult) -> None:
        if self.spec.hosted:
            self.ledger.add(self.spec.label, result.total_tokens)

    # -- calls --------------------------------------------------------------------------

    def _kwargs(
        self,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, str] | None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            kw["response_format"] = response_format
        if reasoning_effort is not None:  # reasoning models (gpt-oss); sent only when set
            kw["reasoning_effort"] = reasoning_effort
        return kw

    def _result(self, resp: Any, started: float) -> ChatResult:
        usage = getattr(resp, "usage", None)
        choice = resp.choices[0]
        return ChatResult(
            text=str(choice.message.content or ""),
            model=str(getattr(resp, "model", None) or self.spec.model),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        self.check_quota(estimate_tokens(messages, max_tokens))
        started = time.perf_counter()
        resp = self._client.chat.completions.create(
            messages=list(messages),
            **self._kwargs(max_tokens, temperature, response_format, reasoning_effort),
        )
        result = self._result(resp, started)
        self._record(result)
        return result

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        self.check_quota(estimate_tokens(messages, max_tokens))
        started = time.perf_counter()
        resp = await self._aclient.chat.completions.create(
            messages=list(messages),
            **self._kwargs(max_tokens, temperature, response_format, reasoning_effort),
        )
        result = self._result(resp, started)
        self._record(result)
        return result

    async def astream(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Usage is estimated from the streamed text (providers differ on
        whether they report usage in a stream)."""
        self.check_quota(estimate_tokens(messages, max_tokens))
        stream = await self._aclient.chat.completions.create(
            messages=list(messages),
            stream=True,
            **self._kwargs(max_tokens, temperature, None, reasoning_effort),
        )
        out: list[str] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                out.append(delta)
                yield delta
        if self.spec.hosted:
            prompt = sum(len(m.get("content", "")) for m in messages) // CHARS_PER_TOKEN
            self.ledger.add(self.spec.label, prompt + len("".join(out)) // CHARS_PER_TOKEN)
