"""The start of every turn: load context, rewrite the follow-up, check the cache, route."""

from __future__ import annotations

import logging
import uuid

from langchain_core.runnables import RunnableConfig

from sre_core.agent.models import AllProvidersFailed
from sre_core.agent.nodes.common import Ctx, Meter, Node, ParseFailure, Update
from sre_core.agent.prompts import (
    AGENT_PROMPT_VERSION,
    rewrite_messages,
    router_messages,
    summary_messages,
)
from sre_core.agent.schemas import RewriteOut, RouteOut
from sre_core.agent.state import AgentState
from sre_shared.contracts.agent import CacheHitEvent, RouteEvent
from sre_shared.models import Claims

log = logging.getLogger("sre_core.agent")


def make_front_nodes(ctx: Ctx) -> dict[str, Node]:
    cfg = ctx.cfg

    async def memory(state: AgentState, config: RunnableConfig) -> Update:
        """Recent turns verbatim, older ones folded into the thread's rolling summary."""

        async def body(meter: Meter, claims: Claims) -> Update:
            thread_id, turn = uuid.UUID(state["thread_id"]), state["turn"]
            keep = cfg.memory.recent_turns
            record = await ctx.threads.get(claims, thread_id)
            summary = record.summary if record else None
            msgs = await ctx.threads.messages(claims, thread_id, last=2 * (keep + 3))
            prior = [m for m in msgs if m.turn < turn]  # the API saved this turn's message already
            recent = set(sorted({m.turn for m in prior})[-keep:])
            through = state.get("summarized_through", 0)
            old = [m for m in prior if m.turn not in recent and m.turn > through]
            update: Update = {}
            if old:
                folded = [{"role": m.role, "content": m.content} for m in old]
                try:
                    res = await ctx.complete(
                        meter,
                        claims,
                        "summary",
                        summary_messages(summary, folded, cfg.memory.summary_max_tokens * 3 // 4),
                        max_tokens=cfg.memory.summary_max_tokens,
                    )
                except AllProvidersFailed:
                    log.warning("summary model unavailable: keeping the old summary")
                else:
                    summary = res.text.strip() or summary
                    if summary:
                        await ctx.threads.set_summary(thread_id, summary)
                    update["summarized_through"] = max(m.turn for m in old)
            history = [
                {"turn": m.turn, "role": m.role, "content": m.content}
                for m in prior
                if m.turn in recent
            ]
            return {**update, "history": history, "summary": summary}

        return await ctx.run(state, config, body)

    async def rewrite(state: AgentState, config: RunnableConfig) -> Update:
        """Follow-ups become standalone questions; `intent` keeps actions out of the cache."""

        async def body(meter: Meter, claims: Claims) -> Update:
            msgs = rewrite_messages(state.get("summary"), state["history"], state["message"])
            try:
                out = await ctx.structured(
                    meter, claims, "rewrite", msgs, RewriteOut, max_tokens=300
                )
            except (ParseFailure, AllProvidersFailed):
                return {"standalone": state["message"], "intent": "question"}
            return {"standalone": out.standalone, "intent": out.intent}

        return await ctx.run(state, config, body)

    async def cache_check(state: AgentState, config: RunnableConfig) -> Update:
        async def body(meter: Meter, claims: Claims) -> Update:
            version = ctx.corpus_version_id
            if state["intent"] == "action" or version is None:
                return {}
            try:
                hit = await ctx.cache.lookup(
                    state["standalone"],
                    claims=claims,
                    corpus_version_id=version,
                    prompt_version=AGENT_PROMPT_VERSION,
                )
            except Exception:  # a cache outage must not stop answers
                log.warning("answer cache lookup failed", exc_info=True)
                return {}
            if hit is None:
                return {}
            ctx.emit(CacheHitEvent(cache=hit.cache))
            return {
                "cache": hit.cache,
                "answer": hit.answer,
                "citations": [c.model_dump() for c in hit.citations],
                "final_kind": "done",
                "final_route": "cache",
            }

        return await ctx.run(state, config, body)

    async def router(state: AgentState, config: RunnableConfig) -> Update:
        async def body(meter: Meter, claims: Claims) -> Update:
            try:
                out = await ctx.structured(
                    meter,
                    claims,
                    "router",
                    router_messages(state["standalone"]),
                    RouteOut,
                    max_tokens=200,
                )
            except (ParseFailure, AllProvidersFailed):
                # No decision: the documentation path is the safe default, not an escalation.
                out = RouteOut(
                    route="simple",
                    confidence=cfg.confidence_floor,
                    reason="router unavailable; answering from the documentation",
                )
            route, reason = out.route, out.reason
            if out.confidence < cfg.confidence_floor and route != "escalate":
                route = "escalate"
                reason = f"low routing confidence ({out.confidence:.2f}): {reason}".strip(": ")
            ctx.emit(RouteEvent(route=route, standalone_question=state["standalone"]))
            update: Update = {
                "route": route,
                "route_confidence": out.confidence,
                "route_reason": reason,
            }
            if route == "escalate":
                update["escalate_reason"] = reason or "outside what this assistant can do"
            return update

        return await ctx.run(state, config, body)

    return {"memory": memory, "rewrite": rewrite, "cache_check": cache_check, "router": router}
