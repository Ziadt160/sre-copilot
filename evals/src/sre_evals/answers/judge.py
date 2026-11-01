"""Judges for answer metrics: DeepEval faithfulness and answer relevancy over any chat model.

`Judge` is the small surface the runner and the calibration use; tests inject fakes. The real
one, `DeepEvalJudge`, drives DeepEval's `FaithfulnessMetric` and `AnswerRelevancyMetric` with
`ChatClientLLM`, a `DeepEvalBaseLLM` over `sre_shared.llm.ChatClient` (so hosted judges go
through the quota guard and the token ledger). Temperature is 0 and `include_reason` is off.

Versions recorded with every run: `JUDGE_PROMPT_VERSION` (our JSON-reply wrapper around
DeepEval's own prompts; bump it when `ChatClientLLM`'s instructions change) and
`metric_version()` (the DeepEval release, whose metric prompts and scoring we do not control).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Protocol

import httpx
import openai
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from sre_evals.golden.llm import JsonReplyError, parse_reply
from sre_shared.llm import ChatClient, QuotaExceeded

JUDGE_PROMPT_VERSION = "judge-v2"  # v2: ambiguous claims penalised, empty claims -> None
JUDGE_MAX_TOKENS = 1024
# Faithfulness makes about three calls per answer (truths, claims, verdicts), relevancy two.
FAITHFULNESS_CALLS = 3
RELEVANCY_CALLS = 2


def metric_version() -> str:
    return f"deepeval-{version('deepeval')}"


class Judge(Protocol):
    """Scores one answer. `None` means the judge could not produce a score (bad JSON, ...)."""

    label: str

    async def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> float | None: ...

    async def relevancy(self, question: str, answer: str) -> float | None: ...


class ChatClientLLM(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """DeepEval model over a `ChatClient`: JSON-mode replies parsed into DeepEval's schemas,
    one repair retry, token counts kept for budgeting."""

    def __init__(
        self, client: ChatClient, *, max_tokens: int = JUDGE_MAX_TOKENS, max_repairs: int = 1
    ) -> None:
        self.client = client
        self.name = client.spec.model
        self.max_tokens = max_tokens
        self.max_repairs = max_repairs
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = client  # type: ignore[assignment]  # DeepEval reads `.model` on some paths

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        return self.client

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self.client.spec.label

    @staticmethod
    def _messages(prompt: str, schema: type[BaseModel] | None) -> list[dict[str, str]]:
        if schema is not None:
            prompt = (
                f"{prompt}\n\nReply with ONLY a JSON object (no prose, no code fence) that "
                f"matches this JSON schema:\n{json.dumps(schema.model_json_schema())}"
            )
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _repair(text: str, err: Exception) -> list[dict[str, str]]:
        return [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"Not valid JSON ({err}). Reply with only the object."},
        ]

    def _tally(self, result: Any) -> None:
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        messages = self._messages(prompt, schema)
        fmt = {"type": "json_object"} if schema is not None else None
        for attempt in range(self.max_repairs + 1):
            res = self.client.complete(
                messages, max_tokens=self.max_tokens, temperature=0.0, response_format=fmt
            )
            self._tally(res)
            if schema is None:
                return res.text
            try:
                return parse_reply(res.text, schema)
            except JsonReplyError as e:
                if attempt >= self.max_repairs:
                    raise
                messages = [*messages, *self._repair(res.text, e)]
        raise AssertionError("unreachable")

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        messages = self._messages(prompt, schema)
        fmt = {"type": "json_object"} if schema is not None else None
        for attempt in range(self.max_repairs + 1):
            res = await self.client.acomplete(
                messages, max_tokens=self.max_tokens, temperature=0.0, response_format=fmt
            )
            self._tally(res)
            if schema is None:
                return res.text
            try:
                return parse_reply(res.text, schema)
            except JsonReplyError as e:
                if attempt >= self.max_repairs:
                    raise
                messages = [*messages, *self._repair(res.text, e)]
        raise AssertionError("unreachable")


@dataclass
class DeepEvalJudge:
    """`Judge` over DeepEval metrics. A fresh metric per call (metrics keep per-call state)."""

    llm: DeepEvalBaseLLM
    label: str
    faithfulness_failures: int = 0  # reply not parseable against the schema
    relevancy_failures: int = 0
    no_claims: int = 0  # the judge extracted no claim from a non-empty answer
    no_statements: int = 0

    @property
    def failures(self) -> int:
        return self.faithfulness_failures + self.relevancy_failures

    async def _score(self, metric: Any, case: LLMTestCase, *, extracted: str) -> float | None:
        """None when the judge's reply could not be parsed (counted as a failure) or it
        extracted nothing to judge (counted in `no_claims` / `no_statements`). Transport
        problems (connection, auth, timeout) are not scores: they propagate."""
        try:
            await metric.a_measure(case, _show_indicator=False)
        except (QuotaExceeded, openai.APIError, httpx.HTTPError, OSError):
            raise
        except (ValueError, KeyError, TypeError):  # JsonReplyError, ValidationError, bad JSON
            if extracted == "claims":
                self.faithfulness_failures += 1
            else:
                self.relevancy_failures += 1
            return None
        if not getattr(metric, extracted, None):
            if extracted == "claims":
                self.no_claims += 1
            else:
                self.no_statements += 1
            return None  # DeepEval scores "nothing to check" as 1.0; that is not faithfulness
        score = metric.score
        return None if score is None else float(score)

    async def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> float | None:
        # "The passages do not say" is not support: the labeller's `s` needs every claim backed.
        metric = FaithfulnessMetric(
            model=self.llm,
            include_reason=False,
            async_mode=True,
            verbose_mode=False,
            penalize_ambiguous_claims=True,
        )
        case = LLMTestCase(input=question, actual_output=answer, retrieval_context=list(contexts))
        return await self._score(metric, case, extracted="claims")

    async def relevancy(self, question: str, answer: str) -> float | None:
        metric = AnswerRelevancyMetric(
            model=self.llm, include_reason=False, async_mode=True, verbose_mode=False
        )
        case = LLMTestCase(input=question, actual_output=answer)
        return await self._score(metric, case, extracted="statements")


def build_judge(client: ChatClient) -> DeepEvalJudge:
    return DeepEvalJudge(ChatClientLLM(client), client.spec.label)


def estimate_judge_tokens(answer: str, contexts: Sequence[str], *, relevancy: bool = True) -> int:
    """Rough tokens one judged answer will cost (4 chars per token, every call carries the
    answer and the passages, plus the reply budget). Used for the quota guard, so it errs high."""
    chars = len(answer) + sum(len(c) for c in contexts)
    calls = FAITHFULNESS_CALLS + (RELEVANCY_CALLS if relevancy else 0)
    return calls * (chars // 4 + 400 + JUDGE_MAX_TOKENS // 2)


def check_judge_budget(client: ChatClient, planned_tokens: int) -> None:
    """Refuse up front when a hosted judge cannot cover the planned tokens (QuotaExceeded)."""
    client.check_quota(planned_tokens)


__all__ = [
    "JUDGE_PROMPT_VERSION",
    "ChatClientLLM",
    "DeepEvalJudge",
    "JsonReplyError",
    "Judge",
    "build_judge",
    "check_judge_budget",
    "estimate_judge_tokens",
    "metric_version",
]
