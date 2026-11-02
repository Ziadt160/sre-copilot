"""The fixed graph (design doc p7-9). No node chooses the next node freely: edges are the
routes below, and every edge after a model call first checks `stop` (limit or budget reached),
which sends the turn to `finish` with an abstain.

    memory -> rewrite -> cache_check -> router -+-> simple -> grounding -> finish
                                 | (hit)        +-> decompose -> complex -> grounding
                                 v                +-> propose -> next_call <-+- run_read
                               finish             |              |  request_approval
                                                  |              |  -> await_approval -> apply_decision
                                                  |              |  -> submit_write -> poll_write -+
                                                  |              v (queue empty)
                                                  |           summarize -> finish
                                                  +-> escalate -> finish
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sre_core.agent.nodes.action import make_action_nodes
from sre_core.agent.nodes.answering import make_answer_nodes
from sre_core.agent.nodes.closing import make_closing_nodes
from sre_core.agent.nodes.common import Ctx
from sre_core.agent.nodes.front import make_front_nodes
from sre_core.agent.state import AgentState

Edge = Callable[[AgentState], str]


def _unless_stopped(next_node: str) -> Edge:
    return lambda s: "finish" if s.get("stop") else next_node


def build_graph(
    ctx: Ctx, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[Any, Any, Any, Any]:
    g: StateGraph[Any, Any, Any, Any] = StateGraph(AgentState)
    nodes = {
        **make_front_nodes(ctx),
        **make_answer_nodes(ctx),
        **make_action_nodes(ctx),
        **make_closing_nodes(ctx),
    }
    for name, fn in nodes.items():
        g.add_node(name, cast("Any", fn))

    g.add_edge(START, "memory")

    def chain(src: str, dst: str) -> None:
        g.add_conditional_edges(src, _unless_stopped(dst), [dst, "finish"])

    chain("memory", "rewrite")
    chain("rewrite", "cache_check")
    g.add_conditional_edges(
        "cache_check",
        lambda s: "finish" if s.get("stop") or s.get("cache") else "router",
        ["router", "finish"],
    )
    g.add_conditional_edges(
        "router",
        lambda s: "finish" if s.get("stop") else {
            "simple": "simple", "complex": "decompose", "action": "propose",
        }.get(s["route"], "escalate"),
        ["simple", "decompose", "propose", "escalate", "finish"],
    )  # fmt: skip
    chain("simple", "grounding")
    chain("decompose", "complex")
    chain("complex", "grounding")
    g.add_edge("grounding", "finish")

    g.add_conditional_edges(
        "propose",
        lambda s: "next_call" if s["calls"] and not s.get("stop") else "finish",
        ["next_call", "finish"],
    )

    def after_next_call(s: AgentState) -> str:
        call = s["current"]
        if call is None:
            return "summarize"
        return "run_read" if call["kind"] == "read" else "request_approval"

    g.add_conditional_edges(
        "next_call", after_next_call, ["summarize", "run_read", "request_approval"]
    )
    g.add_edge("run_read", "next_call")
    g.add_edge("request_approval", "await_approval")
    g.add_edge("await_approval", "apply_decision")
    g.add_conditional_edges(
        "apply_decision",
        lambda s: "submit_write" if s["exec_args"] else "finish",
        ["submit_write", "finish"],
    )
    g.add_edge("submit_write", "poll_write")
    g.add_conditional_edges(
        "poll_write",
        lambda s: "escalate" if s.get("escalate_reason") and s["operation_status"] == "failed"
        and s.get("failed_writes", 0) >= 2 else "next_call",
        ["escalate", "next_call"],
    )  # fmt: skip
    g.add_edge("summarize", "finish")
    g.add_edge("escalate", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)
