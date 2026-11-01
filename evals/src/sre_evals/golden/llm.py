"""A DeepEval custom LLM over any OpenAI-compatible chat endpoint (Ollama, Groq, ...).

`ModelSpec` and `parse_model_spec` live in `sre_shared.llm`; they are re-exported here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

from sre_shared.llm import ModelSpec as ModelSpec
from sre_shared.llm import parse_model_spec as parse_model_spec

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class JsonReplyError(ValueError):
    """The model's reply could not be parsed against the requested schema."""


def parse_reply(text: str, schema: type[BaseModel]) -> BaseModel:
    body = _FENCE.sub("", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise JsonReplyError(f"no JSON object in reply: {text[:120]!r}")
    try:
        return schema.model_validate(json.loads(body[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as e:
        raise JsonReplyError(str(e)[:300]) from e


@dataclass
class CallStats:
    calls: int = 0
    critic_scores: list[float] = field(default_factory=list)


class OpenAICompatLLM(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """DeepEval model: `generate(prompt, schema=...)` returns a parsed pydantic instance.

    `client` / `aclient` may be injected (tests pass fakes with the `chat.completions.create`
    surface); otherwise they are built from the spec.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        client: Any = None,
        aclient: Any = None,
        temperature: float = 0.2,
        max_repairs: int = 1,
    ) -> None:
        self.spec = spec
        self.name = spec.model
        self.temperature = temperature
        self.max_repairs = max_repairs
        self.stats = CallStats()
        key = spec.api_key or "none"
        self._client = client or OpenAI(base_url=spec.base_url, api_key=key)
        self._aclient = aclient or AsyncOpenAI(base_url=spec.base_url, api_key=key)
        self.model = self._client  # type: ignore[assignment]

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        return self._client

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self.spec.label

    # -- request building -------------------------------------------------------------

    def _messages(self, prompt: str, schema: type[BaseModel] | None) -> list[Any]:
        if schema is not None:
            prompt = (
                f"{prompt}\n\nReply with ONLY a JSON object (no prose, no code fence) that "
                f"matches this JSON schema:\n{json.dumps(schema.model_json_schema())}"
            )
        return [{"role": "user", "content": prompt}]

    def _kwargs(self, schema: type[BaseModel] | None) -> dict[str, Any]:
        kw: dict[str, Any] = {"model": self.spec.model, "temperature": self.temperature}
        if schema is not None:
            kw["response_format"] = {"type": "json_object"}
        return kw

    @staticmethod
    def _text(resp: Any) -> str:
        return str(resp.choices[0].message.content or "")

    def _record(self, parsed: BaseModel) -> None:
        score = getattr(parsed, "score", None)
        if type(parsed).__name__ == "InputFeedback" and isinstance(score, int | float):
            self.stats.critic_scores.append(float(score))

    # -- DeepEval surface -------------------------------------------------------------

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        messages = self._messages(prompt, schema)
        for attempt in range(self.max_repairs + 1):
            self.stats.calls += 1
            text = self._text(
                self._client.chat.completions.create(messages=messages, **self._kwargs(schema))
            )
            if schema is None:
                return text
            try:
                parsed = parse_reply(text, schema)
            except JsonReplyError as e:
                if attempt >= self.max_repairs:
                    raise
                messages = [*messages, *self._repair(text, e)]
                continue
            self._record(parsed)
            return parsed
        raise AssertionError("unreachable")

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        messages = self._messages(prompt, schema)
        for attempt in range(self.max_repairs + 1):
            self.stats.calls += 1
            text = self._text(
                await self._aclient.chat.completions.create(
                    messages=messages, **self._kwargs(schema)
                )
            )
            if schema is None:
                return text
            try:
                parsed = parse_reply(text, schema)
            except JsonReplyError as e:
                if attempt >= self.max_repairs:
                    raise
                messages = [*messages, *self._repair(text, e)]
                continue
            self._record(parsed)
            return parsed
        raise AssertionError("unreachable")

    @staticmethod
    def _repair(text: str, err: Exception) -> list[Any]:
        return [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": f"That reply was not valid ({err}). Reply with only the JSON object.",
            },
        ]

    # -- plain helper for the exact-string repair prompt ------------------------------

    def complete(self, prompt: str) -> str:
        self.stats.calls += 1
        return self._text(
            self._client.chat.completions.create(
                messages=self._messages(prompt, None), **self._kwargs(None)
            )
        )
