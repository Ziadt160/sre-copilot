"""Graph state: JSON values only, so any LangGraph checkpointer can store it.

Per-turn fields are reset by `new_turn_input`; `failed_writes` and `summarized_through` are
deliberately left out of it, so they persist across turns in the thread's checkpoint.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # identity of the turn
    thread_id: str
    turn: int
    message: str  # the user's raw message
    # conversation context
    history: list[dict[str, Any]]  # earlier turns, oldest first: {turn, role, content}
    summary: str | None
    summarized_through: int  # thread-persistent: last turn folded into the summary
    # understanding the request
    standalone: str
    intent: str  # question | action
    route: str  # simple | complex | action | escalate
    route_confidence: float
    route_reason: str
    subquestions: list[str]
    # answering
    answer: str
    citations: list[dict[str, Any]]
    invalid_citations: list[str]
    confidence: float
    answer_route: str  # simple | complex, for the cache entry
    context_citations: list[dict[str, Any]]  # what was retrieved, shown when abstaining
    corpus_version_id: int | None
    abstained: bool
    degraded: bool  # answered by the extractive fallback: not cached
    cache: str | None  # exact | semantic when served from the cache
    # acting
    calls: list[dict[str, Any]]  # queue of accepted tool calls: {tool, args, kind, level}
    current: dict[str, Any] | None  # the call being run
    notes: list[str]  # refusals and failures to tell the user
    tool_results: list[dict[str, Any]]  # for the summary only; never seen by a proposal
    pending_approval_id: str | None
    operation_id: str | None
    operation_status: str | None
    exec_args: dict[str, Any] | None
    exec_approval_id: str | None
    failed_writes: int  # thread-persistent
    escalate_reason: str | None
    # limits (per turn)
    decisions: int
    tool_calls: int
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    stop: str | None  # why the turn ended early: step_limit | token_budget | ...
    # outcome
    final_kind: str  # done | abstain
    final_route: str  # simple | complex | action | escalate | cache
    abstain_reason: str | None


def new_turn_input(thread_id: str, turn: int, message: str) -> AgentState:
    """Input for a new turn: every per-turn field starts empty."""
    return {
        "thread_id": thread_id,
        "turn": turn,
        "message": message,
        "history": [],
        "summary": None,
        "standalone": message,
        "intent": "question",
        "route": "simple",
        "route_confidence": 0.0,
        "route_reason": "",
        "subquestions": [],
        "answer": "",
        "citations": [],
        "invalid_citations": [],
        "confidence": 0.0,
        "answer_route": "simple",
        "context_citations": [],
        "corpus_version_id": None,
        "abstained": False,
        "degraded": False,
        "cache": None,
        "calls": [],
        "current": None,
        "notes": [],
        "tool_results": [],
        "pending_approval_id": None,
        "operation_id": None,
        "operation_status": None,
        "exec_args": None,
        "exec_approval_id": None,
        "escalate_reason": None,
        "decisions": 0,
        "tool_calls": 0,
        "tokens_used": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "stop": None,
        "final_kind": "done",
        "final_route": "simple",
        "abstain_reason": None,
    }
