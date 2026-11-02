"""What every node shares: the dependencies, the per-turn meter, and the guarded model call.

All limits live here so no node can forget one: a model call goes through `Ctx.complete`, which
refuses to start past the decision limit or the turn's token budget, asks the `BudgetGuard`
first and records the tokens after. A refusal raises `LimitReached`; `Ctx.run` turns it into a
state update that ends the turn with an abstain (every conditional edge checks `stop`).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from pydantic import BaseModel, ValidationError

from sre_core.agent.ports import (
    AnswerCache,
    ApprovalStore,
    BudgetExceeded,
    BudgetGuard,
    ModelRole,
    ModelRouter,
    OpsClient,
    Outbox,
    ThreadStore,
)
from sre_core.agent.prompts import REPAIR_USER
from sre_core.agent.state import AgentState
from sre_core.retrieval import Retriever
from sre_shared.config import AgentConfig, load_answer_config, load_retrieval_defaults
from sre_shared.contracts import ops
from sre_shared.contracts.agent import AgentEvent
from sre_shared.llm import ChatResult, Message
from sre_shared.models import (
    AnswerConfig,
    Claims,
    EmbeddingModelName,
    RetrievalRequest,
    RetrievalResult,
    Scope,
)
from sre_shared.settings import Settings

log = logging.getLogger("sre_core.agent")

MAX_REPAIRS = 2  # a malformed structured reply is re-asked at most twice (each counts as a step)

T = TypeVar("T", bound=BaseModel)
Update = dict[str, Any]
Node = Callable[[AgentState, RunnableConfig], Awaitable[Update]]


class LimitReached(Exception):
    """The turn must end: `code` says why, `text` is what the user is told."""

    def __init__(self, code: str, text: str) -> None:
        super().__init__(text)
        self.code = code
        self.text = text


class ParseFailure(ValueError):
    """The model did not return JSON matching the expected schema."""


@dataclass
class Meter:
    decisions: int
    tokens: int
    prompt: int
    completion: int

    @classmethod
    def of(cls, state: AgentState) -> Meter:
        return cls(
            state.get("decisions", 0),
            state.get("tokens_used", 0),
            state.get("prompt_tokens", 0),
            state.get("completion_tokens", 0),
        )

    def update(self) -> Update:
        return {
            "decisions": self.decisions,
            "tokens_used": self.tokens,
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
        }


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_model[T: BaseModel](text: str, schema: type[T]) -> T:
    """The first JSON object in `text` (code fences tolerated), validated against `schema`."""
    s = _FENCE.sub("", text.strip())
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ParseFailure("no JSON object in the reply")
    try:
        return schema.model_validate_json(s[i : j + 1])
    except (ValidationError, ValueError) as exc:
        raise ParseFailure(str(exc)[:200]) from exc


class Ctx:
    """Dependencies and helpers; nodes are closures over one of these."""

    def __init__(
        self,
        *,
        threads: ThreadStore,
        approvals: ApprovalStore,
        ops_client: OpsClient,
        cache: AnswerCache,
        budget: BudgetGuard,
        models: ModelRouter,
        outbox: Outbox,
        policy: ops.NamespacePolicy,
        retriever: Retriever,
        answerer: Any,
        config: AgentConfig,
        settings: Settings,
    ) -> None:
        self.threads = threads
        self.approvals = approvals
        self.ops = ops_client
        self.cache = cache
        self.budget = budget
        self.models = models
        self.outbox = outbox
        self.policy = policy
        self.retriever = retriever
        self.answerer = answerer
        self.cfg = config
        self.settings = settings
        self.answer_cfg: AnswerConfig = load_answer_config(settings.config_dir)
        self._retrieval_defaults = load_retrieval_defaults(settings.config_dir)
        # The cache is keyed by corpus version, which the Retriever only reports with a result:
        # remember the last one seen (None until the first retrieval, which skips the lookup).
        self.corpus_version_id: int | None = None

    # -- events and identity -------------------------------------------------------------

    @staticmethod
    def emit(event: AgentEvent) -> None:
        get_stream_writer()(event)

    @staticmethod
    def claims(config: RunnableConfig) -> Claims:
        claims = config.get("configurable", {}).get("claims")
        if not isinstance(claims, Claims):
            raise RuntimeError("the graph was invoked without claims")
        return claims

    # -- guarded node body ---------------------------------------------------------------

    async def run(
        self,
        state: AgentState,
        config: RunnableConfig,
        body: Callable[[Meter, Claims], Awaitable[Update]],
    ) -> Update:
        meter = Meter.of(state)
        try:
            out = await body(meter, self.claims(config))
        except LimitReached as stop:
            out = self.stop_update(state, stop.code, stop.text)
        except BudgetExceeded:
            out = self.stop_update(state, "daily_budget", "Your daily token allowance is used up.")
        return {**out, **meter.update()}

    async def run_unguarded(
        self,
        state: AgentState,
        config: RunnableConfig,
        body: Callable[[Meter, Claims], Awaitable[Update]],
    ) -> Update:
        """Like `run`, but limit errors reach the caller (which has its own fallback)."""
        meter = Meter.of(state)
        out = await body(meter, self.claims(config))
        return {**out, **meter.update()}

    @staticmethod
    def stop_update(state: AgentState, code: str, text: str) -> Update:
        return {
            "stop": code,
            "final_kind": "abstain",
            "final_route": state.get("route") or "simple",
            "abstain_reason": text,
        }

    # -- model calls ---------------------------------------------------------------------

    def check_limits(self, meter: Meter, planned: int) -> None:
        lim = self.cfg.limits
        if meter.decisions >= lim.llm_decisions_per_turn:
            raise LimitReached(
                "step_limit",
                f"I stopped after {lim.llm_decisions_per_turn} steps without a reliable answer.",
            )
        if meter.tokens + planned > lim.answer_token_budget:
            raise LimitReached("token_budget", "This turn used up its token budget.")

    async def charge(self, meter: Meter, claims: Claims, planned: int) -> None:
        self.check_limits(meter, planned)
        await self.budget.check(claims, planned)

    async def record(self, meter: Meter, claims: Claims, prompt: int, completion: int) -> None:
        meter.decisions += 1
        meter.prompt += prompt
        meter.completion += completion
        meter.tokens += prompt + completion
        await self.budget.record(claims, prompt + completion)

    async def complete(
        self,
        meter: Meter,
        claims: Claims,
        role: ModelRole,
        messages: list[Message],
        *,
        max_tokens: int,
        json_mode: bool = False,
    ) -> ChatResult:
        await self.charge(meter, claims, max_tokens)
        res = await self.models.complete(
            role,
            messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if json_mode else None,
        )
        await self.record(meter, claims, res.prompt_tokens, res.completion_tokens)
        return res

    async def structured(
        self,
        meter: Meter,
        claims: Claims,
        role: ModelRole,
        messages: list[Message],
        schema: type[T],
        *,
        max_tokens: int = 400,
    ) -> T:
        """Ask for JSON and parse it; a bad reply is re-asked up to MAX_REPAIRS times. Every
        attempt is a counted step, so a model that never answers properly hits the step limit."""
        msgs = list(messages)
        last: ParseFailure | None = None
        for _ in range(1 + MAX_REPAIRS):
            res = await self.complete(
                meter, claims, role, msgs, max_tokens=max_tokens, json_mode=True
            )
            try:
                return parse_model(res.text, schema)
            except ParseFailure as exc:
                last = exc
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": res.text[:1500]},
                    {"role": "user", "content": REPAIR_USER},
                ]
        raise last or ParseFailure("no reply")

    # -- retrieval -----------------------------------------------------------------------

    def retrieval_request(self, question: str, k: int | None = None) -> RetrievalRequest:
        """The same request the answer path builds, so both routes retrieve alike."""
        d = self._retrieval_defaults
        model: EmbeddingModelName = d.embedding_model  # type: ignore[assignment]
        return RetrievalRequest(
            query=question,
            embedding_model=model,
            use_dense=d.mode in ("hybrid", "dense"),
            use_keyword=d.mode in ("hybrid", "keyword"),
            ts_config=d.ts_config,
            keyword_match=d.keyword_match,
            candidates_per_leg=d.candidates_per_leg,
            k=k or self.answer_cfg.k,
            w_dense=d.weights.get("dense", 1.0),
            w_keyword=d.weights.get("keyword", 1.0),
            rrf_k=d.rrf_k,
            ef_search=d.hnsw.ef_search,
            iterative_scan=d.hnsw.iterative_scan,
            rerank_candidates=d.rerank.candidates if d.rerank else None,
        )

    async def retrieve(
        self, question: str, claims: Claims, k: int | None = None
    ) -> RetrievalResult:
        res = await self.retriever.retrieve(self.retrieval_request(question, k), claims)
        self.corpus_version_id = res.corpus_version_id
        return res

    @staticmethod
    def claims_of_requester(sub: str) -> Claims:
        """Only the subject matters: `mint` is given explicit scopes and namespaces."""
        return Claims(sub=sub)

    # -- mcp-ops tokens ------------------------------------------------------------------

    def mint(
        self,
        claims: Claims,
        *,
        thread_id: uuid.UUID,
        turn: int,
        approval_id: uuid.UUID | None = None,
        scopes: list[Scope] | None = None,
        sub: str | None = None,
        namespaces: list[str] | None = None,
        max_calls: int = 1,
    ) -> str:
        """A token for one or two mcp-ops calls of this turn (never the user's own token)."""
        secret = self.settings.mcp_token_secret
        if secret is None:
            raise RuntimeError("mcp_token_secret is not configured")
        return ops.mint_mcp_token(
            secret=secret.get_secret_value(),
            sub=sub or claims.sub,
            scopes=list(claims.scopes) if scopes is None else scopes,
            namespaces=list(claims.namespaces) if namespaces is None else namespaces,
            thread_id=thread_id,
            turn=turn,
            approval_id=approval_id,
            max_calls=max_calls,
            ttl_s=self.cfg.mcp.token_ttl_s,
        )


def to_json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)
