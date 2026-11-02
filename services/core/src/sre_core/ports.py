"""Ports the agent graph and the chat API depend on. Fixed in Wave 0 of Phase 3.

This module sits outside the core layers (it imports only sre_shared), so the implementations
below the agent (store, cache, budget) and the agent above them can all import it. It began as
`sre_core.agent.ports`, which made the lower layers import the agent layer (T14-2); that path
remains as a re-export for the agent.

Each port has one owner that implements it and in-memory fakes in `sre_core.agent.testing`, so
the graph (T12), the API (T13), caches, budgets and routing (T14) and mcp-ops (T11) are built in
parallel and wired together only in `sre_core.api.runtime`. Implementations are exported by
factories at fixed paths, which `runtime.build_runtime` calls (T13 owns runtime.py):

    sre_core.store.factory.make_thread_store(pool) -> ThreadStore             (T13)
    sre_core.store.factory.make_approval_store(pool) -> ApprovalStore         (T13)
    sre_core.store.factory.make_outbox(pool) -> Outbox                        (T13)
    sre_core.cache.factory.make_answer_cache(pool, settings) -> AnswerCache   (T14)
    sre_core.budget.factory.make_budget_guard(settings) -> BudgetGuard        (T14)
    sre_core.agent.models.make_model_router(settings, *, client_wrapper=None) -> ModelRouter (T14)
        `client_wrapper(AsyncOpenAI) -> AsyncOpenAI` is how tracing (T16) is added; the module
        also exports `AllProvidersFailed(RuntimeError)` and `extractive_answer(question, context)
        -> tuple[str, list[int]]` (quoted passages with [n] markers; the last fallback).
    sre_core.agent.ops_client.make_ops_client(settings) -> OpsClient          (T12)
    sre_core.agent.factory.make_chat_agent(*, threads, approvals, ops, cache, budget, models,
        outbox, policy, retriever, answerer, checkpointer, config, settings) -> ChatAgent (T12)
        Explicit arguments, not the API's Runtime: the agent layer must not import the API.

Rules every implementation keeps:
* a turn number comes only from `ThreadStore.claim_turn` (an `update ... returning` in the
  database), never from the graph, because LangGraph re-runs nodes;
* an approval is decided only through `ApprovalStore.decide`, a conditional update that succeeds
  once, and only while the approval is pending and unexpired;
* `OpsClient` sends a token minted for the turn (`sre_shared.contracts.ops.mint_mcp_token`); it
  never forwards the user's own token, and never invents an idempotency key (mcp-ops builds it).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from sre_shared.contracts.agent import AgentEvent, Route
from sre_shared.contracts.ops import ApprovalRecord, OperationRecord
from sre_shared.llm import ChatResult, Message
from sre_shared.models import Citation, Claims

ThreadStatus = Literal["active", "awaiting_approval", "escalated", "closed"]
MessageRole = Literal["user", "assistant", "human"]
ModelRole = Literal["rewrite", "router", "answer", "synthesize", "grounding", "summary"]


class ThreadRecord(BaseModel):
    id: uuid.UUID
    owner_sub: str
    title: str | None = None
    status: ThreadStatus = "active"
    next_turn: int = 1
    summary: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MessageRecord(BaseModel):
    thread_id: uuid.UUID
    turn: int
    role: MessageRole
    content: str
    route: Route | Literal["cache"] | None = None
    citations: list[Citation] = Field(default_factory=list)
    trace_run_id: str | None = None


class CachedAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: Literal["simple", "complex"]
    cache: Literal["exact", "semantic"] = "exact"


class ThreadBusy(RuntimeError):
    """The thread waits on an approval (or a human after escalation): no new turn yet (409)."""


class ThreadNotFound(LookupError):
    """No such thread for this caller (RLS hides other users' threads): 404."""


class BudgetExceeded(RuntimeError):
    """A per-user daily or per-request budget would be exceeded: refuse cleanly (429)."""


@runtime_checkable
class ThreadStore(Protocol):
    async def create(self, claims: Claims, title: str | None = None) -> ThreadRecord: ...

    async def get(self, claims: Claims, thread_id: uuid.UUID) -> ThreadRecord | None:
        """None when the thread does not exist or is not the caller's."""
        ...

    async def list_threads(self, claims: Claims, limit: int = 50) -> list[ThreadRecord]: ...

    async def claim_turn(self, claims: Claims, thread_id: uuid.UUID) -> int:
        """Next turn number (atomic). ThreadNotFound / ThreadBusy as above."""
        ...

    async def add_message(self, msg: MessageRecord) -> None: ...

    async def messages(
        self, claims: Claims, thread_id: uuid.UUID, *, last: int = 20
    ) -> list[MessageRecord]: ...

    async def set_status(self, thread_id: uuid.UUID, status: ThreadStatus) -> None: ...

    async def set_summary(self, thread_id: uuid.UUID, summary: str) -> None: ...


@runtime_checkable
class ApprovalStore(Protocol):
    async def create(self, approval: ApprovalRecord) -> ApprovalRecord: ...

    async def get(self, approval_id: uuid.UUID) -> ApprovalRecord | None: ...

    async def decide(
        self,
        approval_id: uuid.UUID,
        *,
        decision: Literal["approved", "rejected"],
        decided_by: str,
        args: Mapping[str, Any] | None = None,
    ) -> ApprovalRecord | None:
        """The decided record, or None when it was not pending or has expired (409)."""
        ...

    async def claim_execution(self, approval_id: uuid.UUID) -> ApprovalRecord | None:
        """Claim an approved, unexpired approval for its one execution, atomically: the claimed
        record, or None when it is not approved, has expired, or another resume claimed it."""
        ...

    async def mark_executed(self, approval_id: uuid.UUID) -> None: ...

    async def expire_due(self) -> list[ApprovalRecord]:
        """Mark expired every approval past `expires_at` that is pending, or approved but never
        claimed for execution (a crash between decision and resume); return them (the sweeper,
        which then resumes their threads)."""
        ...


@runtime_checkable
class OpsClient(Protocol):
    async def read(self, tool: str, args: Mapping[str, Any], token: str) -> dict[str, Any]: ...

    async def write(self, tool: str, args: Mapping[str, Any], token: str) -> OperationRecord:
        """Submit an approved write (an MCP task); returns its operation row at once."""
        ...

    async def escalate(self, tool: str, args: Mapping[str, Any], token: str) -> OperationRecord: ...

    async def operation(self, operation_id: uuid.UUID, token: str) -> OperationRecord:
        """Poll an operation (Postgres, through get_operation; survives a task-store restart)."""
        ...


@runtime_checkable
class AnswerCache(Protocol):
    async def lookup(
        self, question: str, *, claims: Claims, corpus_version_id: int, prompt_version: str
    ) -> CachedAnswer | None: ...

    async def store(
        self,
        answer: CachedAnswer,
        *,
        claims: Claims,
        corpus_version_id: int,
        prompt_version: str,
    ) -> None:
        """Only simple/complex answers built from retrieved documents; never tool output."""
        ...


@runtime_checkable
class BudgetGuard(Protocol):
    async def check(self, claims: Claims, planned_tokens: int) -> None:
        """Raise BudgetExceeded if `planned_tokens` would pass the caller's daily allowance."""
        ...

    async def record(self, claims: Claims, tokens: int) -> None: ...


@runtime_checkable
class ModelRouter(Protocol):
    async def complete(
        self,
        role: ModelRole,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        """Walk the fallback chain for `role` (first model, then providers by `order`)."""
        ...

    def model_for(self, role: ModelRole) -> str:
        """The first-choice model label for `role` (for stamps and traces)."""
        ...


@runtime_checkable
class Outbox(Protocol):
    async def enqueue(self, kind: str, dedupe_key: str, payload: Mapping[str, Any]) -> None:
        """Idempotent on `dedupe_key`."""
        ...


@runtime_checkable
class ChatAgent(Protocol):
    def stream_turn(
        self, claims: Claims, thread_id: uuid.UUID, turn: int, message: str
    ) -> AsyncIterator[AgentEvent]: ...

    def resume(
        self, claims: Claims, thread_id: uuid.UUID, approval: ApprovalRecord
    ) -> AsyncIterator[AgentEvent]:
        """Continue a run paused at the approval gate; the decision is read from `approval`
        (the database), never from the request body."""
        ...
