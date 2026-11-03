"""Escalation (a human takes the thread) and the single place a turn ends."""

from __future__ import annotations

import logging
import uuid

from langchain_core.runnables import RunnableConfig

from sre_core.agent.nodes.common import Ctx, Node, Update
from sre_core.agent.ports import MessageRecord
from sre_core.agent.state import AgentState
from sre_shared.contracts import ops
from sre_shared.contracts.agent import (
    AbstainEvent,
    DoneEvent,
    EscalatedEvent,
    TokenEvent,
)
from sre_shared.models import Citation

log = logging.getLogger("sre_core.agent")


def make_closing_nodes(ctx: Ctx) -> dict[str, Node]:
    async def escalate(state: AgentState, config: RunnableConfig) -> Update:
        """Open an incident and page on-call; no approval (it must not wait on the user)."""
        claims = ctx.claims(config)
        thread_id, turn = uuid.UUID(state["thread_id"]), state["turn"]
        reason = state.get("escalate_reason") or "a person needs to look at this"
        # The token carries the minimum scope escalation needs; both calls share it.
        token = ctx.mint(
            claims, thread_id=thread_id, turn=turn, scopes=["viewer"],
            tools=["create_incident", "page_oncall"], max_calls=2,
        )  # fmt: skip
        args = ops.CreateIncidentArgs(
            severity="sev3",
            title=f"Escalated: {state['standalone']}"[:200],
            summary=f"{reason}\n\nUser request: {state['message']}"[:4000],
        ).model_dump(mode="json")
        created = await ctx.ops.escalate("create_incident", args, token)
        raw_id = (created.result or {}).get("incident_id")
        if created.status != "succeeded" or raw_id is None:
            raise RuntimeError("the incident could not be created")
        incident_id = uuid.UUID(str(raw_id))
        paged = False
        try:
            page = await ctx.ops.escalate("page_oncall", {"incident_id": str(incident_id)}, token)
            paged = page.status == "succeeded"
        except Exception as exc:  # the incident exists; say the page did not go out
            log.warning("page_oncall failed: %s", type(exc).__name__)
        await ctx.threads.set_status(thread_id, "escalated")
        await ctx.outbox.enqueue(
            "escalation",
            f"escalation:{thread_id}:{turn}",
            {"incident_id": str(incident_id), "thread_id": str(thread_id), "reason": reason},
        )
        ctx.emit(EscalatedEvent(incident_id=incident_id, paged=paged, reason=reason))
        paged_text = "paged the on-call engineer" if paged else "could not page on-call"
        answer = (
            f"I am handing this over ({reason}). I opened incident {incident_id} and "
            f"{paged_text}; a person now owns this thread."
        )
        return {"answer": answer, "final_kind": "done", "final_route": "escalate"}

    async def finish(state: AgentState, config: RunnableConfig) -> Update:
        """Save the assistant message, then emit the terminal event (`done` or `abstain`)."""
        thread_id, turn = uuid.UUID(state["thread_id"]), state["turn"]
        route = state["final_route"]
        if state["final_kind"] == "abstain":
            reason = state.get("abstain_reason") or "I could not answer this."
            sources = [Citation.model_validate(c) for c in state["context_citations"]]
            content, citations = reason, []
        else:
            content = state["answer"]
            citations = [Citation.model_validate(c) for c in state["citations"]]
        trace_run_id = config.get("configurable", {}).get("trace_run_id")
        await ctx.threads.add_message(
            _message(thread_id, turn, route, content, citations, trace_run_id)
        )
        if state["final_kind"] == "abstain":
            ctx.emit(AbstainEvent(reason=content, sources=sources))
        else:
            ctx.emit(TokenEvent(text=content))
            ctx.emit(
                DoneEvent(
                    thread_id=thread_id,
                    turn=turn,
                    route=route,
                    answer=content,
                    citations=citations,
                    prompt_tokens=state["prompt_tokens"],
                    completion_tokens=state["completion_tokens"],
                )
            )
        return {}

    return {"escalate": escalate, "finish": finish}


def _message(
    thread_id: uuid.UUID,
    turn: int,
    route: str,
    content: str,
    citations: list[Citation],
    trace_run_id: str | None,
) -> MessageRecord:
    return MessageRecord.model_validate(
        {
            "thread_id": thread_id,
            "turn": turn,
            "role": "assistant",
            "content": content,
            "route": route,
            "citations": citations,
            "trace_run_id": trace_run_id,
        }
    )
