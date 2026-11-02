"""`make_chat_agent`: the LangGraph agent behind the `ChatAgent` port.

`stream_turn` runs a turn; at the approval gate the graph pauses (a checkpoint holds its state)
and the stream ends after `approval_required`. `resume` continues it once the decision is in
the database. The decision is never taken from the caller: see `resume`.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from sre_core.agent.graph import build_graph
from sre_core.agent.nodes.common import Ctx
from sre_core.agent.ports import (
    AnswerCache,
    ApprovalStore,
    BudgetGuard,
    ChatAgent,
    ModelRouter,
    OpsClient,
    Outbox,
    ThreadStore,
)
from sre_core.agent.state import new_turn_input
from sre_core.retrieval import Retriever
from sre_shared.config import AgentConfig
from sre_shared.contracts.agent import AGENT_EVENT, AgentEvent, ErrorEvent, TurnEvent
from sre_shared.contracts.ops import ApprovalRecord, NamespacePolicy
from sre_shared.models import Claims
from sre_shared.settings import Settings

log = logging.getLogger("sre_core.agent")


class LangGraphChatAgent:
    def __init__(self, ctx: Ctx, checkpointer: BaseCheckpointSaver[Any]) -> None:
        self._ctx = ctx
        self._graph = build_graph(ctx, checkpointer)
        self._locks: defaultdict[uuid.UUID, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _config(self, claims: Claims, thread_id: uuid.UUID, approver: Claims | None = None) -> Any:
        """`claims` is who the turn runs as (the requester); `approver` is used only to check an
        approval."""
        return {
            "configurable": {
                "thread_id": str(thread_id),
                "claims": claims,
                "approver": approver or claims,
            },
            "recursion_limit": self._ctx.cfg.limits.recursion_limit,
        }

    async def _drive(self, graph_input: Any, config: Any) -> AsyncIterator[AgentEvent]:
        try:
            async for event in cast("Any", self._graph).astream(
                graph_input, config, stream_mode="custom"
            ):
                yield AGENT_EVENT.validate_python(event)
        except GraphRecursionError:
            yield ErrorEvent(type="RecursionLimit", detail="the turn exceeded its step limit")
        except Exception as exc:
            log.exception("agent turn failed")
            yield ErrorEvent(type=type(exc).__name__, detail="the turn failed")

    async def stream_turn(
        self, claims: Claims, thread_id: uuid.UUID, turn: int, message: str
    ) -> AsyncIterator[AgentEvent]:
        # Refuse before any event if the caller is already out of budget (the API maps it to 429).
        await self._ctx.budget.check(claims, self._ctx.answer_cfg.max_tokens)
        yield TurnEvent(thread_id=thread_id, turn=turn)
        async with self._locks[thread_id]:
            graph_input = new_turn_input(str(thread_id), turn, message)
            graph_input["requester"] = claims.model_dump(mode="json")
            async for event in self._drive(graph_input, self._config(claims, thread_id)):
                yield event

    async def resume(
        self, claims: Claims, thread_id: uuid.UUID, approval: ApprovalRecord
    ) -> AsyncIterator[AgentEvent]:
        """Continue a run paused at the gate. Only `approval.id` is used from the argument: the
        graph must be waiting on that exact approval, and the decision is the one stored in the
        ApprovalStore (a forged status in `approval` changes nothing)."""
        async with self._locks[thread_id]:
            snapshot = await self._graph.aget_state(self._config(claims, thread_id))
            saved = snapshot.values.get("requester")
            # Run the rest of the turn as the user who asked, not as the approver.
            requester = Claims.model_validate(saved) if saved else claims
            config = self._config(requester, thread_id, approver=claims)
            waiting = [i.value for t in snapshot.tasks for i in t.interrupts]
            if not any(
                isinstance(v, dict) and v.get("approval_id") == str(approval.id) for v in waiting
            ):
                yield ErrorEvent(
                    type="NothingToResume", detail="no run is waiting on this approval"
                )
                return
            stored = await self._ctx.approvals.get(approval.id)
            if stored is None or stored.status == "pending":
                yield ErrorEvent(type="ApprovalPending", detail="the approval is not decided yet")
                return
            async for event in self._drive(
                Command(resume={"approval_id": str(approval.id)}), config
            ):
                yield event


def make_chat_agent(
    *,
    threads: ThreadStore,
    approvals: ApprovalStore,
    ops: OpsClient,
    cache: AnswerCache,
    budget: BudgetGuard,
    models: ModelRouter,
    outbox: Outbox,
    policy: NamespacePolicy,
    retriever: Retriever,
    answerer: Any,
    checkpointer: BaseCheckpointSaver[Any],
    config: AgentConfig,
    settings: Settings,
) -> ChatAgent:
    ctx = Ctx(
        threads=threads, approvals=approvals, ops_client=ops, cache=cache, budget=budget,
        models=models, outbox=outbox, policy=policy, retriever=retriever, answerer=answerer,
        config=config, settings=settings,
    )  # fmt: skip
    return LangGraphChatAgent(ctx, checkpointer)
