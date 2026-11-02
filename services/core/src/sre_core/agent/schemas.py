"""Structured outputs the model must produce (parsed from JSON; anything else is repaired or
replaced by a safe default)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Out(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RewriteOut(_Out):
    standalone: str = Field(min_length=1, max_length=4000)
    intent: Literal["question", "action"] = "question"


class RouteOut(_Out):
    route: Literal["simple", "complex", "action", "escalate"]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    reason: str = Field(default="", max_length=500)


class PlanOut(_Out):
    subquestions: list[str] = Field(min_length=1)


class ToolCallOut(_Out):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ProposalOut(_Out):
    calls: list[ToolCallOut] = Field(default_factory=list)
    clarification: str | None = Field(default=None, max_length=500)


class GroundedOut(_Out):
    answer: str = Field(min_length=1)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
