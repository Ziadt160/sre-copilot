"""The chat API's contract: requests, the approval decision, and the SSE events of a turn.

`POST /chat` streams one turn as Server-Sent Events, framed like `/answer`
(`event: <name>` / `data: <json>`). The web UI generates its TypeScript types from the JSON
Schema this module exports (`python -m sre_shared.contracts.export`), so a field added here
shows up in the UI's types.

A turn's events, in order: `turn` first; then `route` (or `cache_hit` when a cached answer is
served); then, depending on the route, `sources` and `token`s, or `tool_proposed` followed by
`approval_required` (the stream ends there; the decision arrives on `POST /approvals/{id}` and
the run resumes server-side), or `operation` updates, or `escalated`; then `abstain` or `done`.
`error` can end any stream.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from sre_shared.contracts.ops import OperationStatus, Risk
from sre_shared.models import Citation

Route = Literal["simple", "complex", "action", "escalate"]


class ChatRequest(BaseModel):
    """One user message. No `thread_id` starts a new thread. `client_message_id` lets the UI
    retry a send without creating a second turn."""

    model_config = ConfigDict(extra="forbid")

    thread_id: uuid.UUID | None = None
    message: str = Field(min_length=1, max_length=4000)
    client_message_id: str | None = Field(default=None, max_length=100)


class ApprovalDecision(BaseModel):
    """`POST /approvals/{id}`. `edit` approves with changed arguments, which are validated and
    re-checked against the approval rules like a new proposal."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "edit", "reject"]
    args: dict[str, Any] | None = None
    reason: str | None = Field(default=None, max_length=500)


class FeedbackRequest(BaseModel):
    """`POST /feedback`: thumbs on a turn, optionally per cited chunk."""

    model_config = ConfigDict(extra="forbid")

    thread_id: uuid.UUID
    turn: int = Field(ge=1)
    rating: Literal[-1, 1] | None = None
    reason: str | None = Field(default=None, max_length=1000)
    sources: dict[int, bool] = Field(
        default_factory=dict, description="chunk_id -> was this source useful?"
    )


class UiEventRequest(BaseModel):
    """`POST /events`: implicit signals."""

    model_config = ConfigDict(extra="forbid")

    thread_id: uuid.UUID
    turn: int = Field(ge=1)
    kind: Literal["regenerate", "copy"]


# --- SSE events -------------------------------------------------------------------------------


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TurnEvent(_Event):
    event: Literal["turn"] = "turn"
    thread_id: uuid.UUID
    turn: int


class RouteEvent(_Event):
    event: Literal["route"] = "route"
    route: Route
    standalone_question: str = Field(description="the follow-up rewritten as a standalone question")


class CacheHitEvent(_Event):
    event: Literal["cache_hit"] = "cache_hit"
    cache: Literal["exact", "semantic"]


class SourcesEvent(_Event):
    event: Literal["sources"] = "sources"
    sources: list[Citation]


class TokenEvent(_Event):
    event: Literal["token"] = "token"
    text: str


class ToolProposedEvent(_Event):
    event: Literal["tool_proposed"] = "tool_proposed"
    tool: str
    args: dict[str, Any]
    kind: Literal["read", "write", "escalation"]


class ApprovalRequiredEvent(_Event):
    event: Literal["approval_required"] = "approval_required"
    approval_id: uuid.UUID
    tool: str
    args: dict[str, Any]
    namespace: str
    target: str
    risk: Risk
    required_scope: Literal["operator", "admin"]
    expected_effect: str
    expires_at: datetime


class OperationEvent(_Event):
    event: Literal["operation"] = "operation"
    operation_id: uuid.UUID
    tool: str
    status: OperationStatus
    progress: str | None = None
    result: dict[str, Any] | None = None


class EscalatedEvent(_Event):
    event: Literal["escalated"] = "escalated"
    incident_id: uuid.UUID
    paged: bool
    reason: str


class AbstainEvent(_Event):
    event: Literal["abstain"] = "abstain"
    reason: str
    sources: list[Citation] = Field(default_factory=list)


class DoneEvent(_Event):
    event: Literal["done"] = "done"
    thread_id: uuid.UUID
    turn: int
    route: Route | Literal["cache"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ErrorEvent(_Event):
    event: Literal["error"] = "error"
    type: str
    detail: str


AgentEvent = Annotated[
    TurnEvent
    | RouteEvent
    | CacheHitEvent
    | SourcesEvent
    | TokenEvent
    | ToolProposedEvent
    | ApprovalRequiredEvent
    | OperationEvent
    | EscalatedEvent
    | AbstainEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="event"),
]
AGENT_EVENT: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)


def sse_payload(event: BaseModel) -> tuple[str, dict[str, Any]]:
    """(event name, data) for the SSE frame; the `event` field becomes the frame's name."""
    data = event.model_dump(mode="json")
    return str(data.pop("event")), data


def json_schemas() -> dict[str, Any]:
    """JSON Schemas the web UI generates its types from."""
    return {
        "AgentEvent": AGENT_EVENT.json_schema(),
        "ChatRequest": ChatRequest.model_json_schema(),
        "ApprovalDecision": ApprovalDecision.model_json_schema(),
        "FeedbackRequest": FeedbackRequest.model_json_schema(),
        "UiEventRequest": UiEventRequest.model_json_schema(),
    }
