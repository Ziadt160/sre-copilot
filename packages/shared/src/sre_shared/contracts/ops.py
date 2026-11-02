"""Cluster tools: names, arguments, approval rules, idempotency and the mcp-ops token.

The agent (sre_core) proposes tool calls and mcp-ops (sre_mcp_ops) executes them; both import
this module so they cannot disagree on what a call needs (design doc p9-11):

* every argument model validates Kubernetes names, so a model cannot smuggle odd strings;
* `idempotency_key` is `sha256(thread, turn, tool, normalized args)`, and mcp-ops builds it from
  the token's thread and turn, never from anything the model wrote;
* `required_approval` decides who must approve a call: reads need nobody, writes need the user's
  confirmation (operator), and an admin in a protected namespace or when scaling to 0;
* rollback takes an explicit revision, never "undo": repeating an undo would flip the rollout
  back and forth, repeating "go to revision 3" is harmless.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sre_shared.models import Scope

# --- tools ----------------------------------------------------------------------------------

ToolName = Literal[
    "list_pods",
    "describe",
    "get_logs",
    "get_events",
    "rollout_status",
    "get_operation",
    "restart_deployment",
    "scale_deployment",
    "rollback_deployment",
    "create_incident",
    "page_oncall",
]
ToolKind = Literal["read", "write", "escalation"]

READ_TOOLS: frozenset[str] = frozenset(
    {"list_pods", "describe", "get_logs", "get_events", "rollout_status", "get_operation"}
)
WRITE_TOOLS: frozenset[str] = frozenset(
    {"restart_deployment", "scale_deployment", "rollback_deployment"}
)
ESCALATION_TOOLS: frozenset[str] = frozenset({"create_incident", "page_oncall"})


def tool_kind(tool: str) -> ToolKind:
    if tool in READ_TOOLS:
        return "read"
    if tool in WRITE_TOOLS:
        return "write"
    if tool in ESCALATION_TOOLS:
        return "escalation"
    raise ValueError(f"unknown tool {tool!r}")


def required_scope(tool: str) -> Scope:
    """The scope a caller needs to call `tool` at all (approval is a separate question)."""
    return "operator" if tool_kind(tool) == "write" else "viewer"


# Kubernetes DNS-1123 label: lower-case alphanumerics and '-', at most 63 characters.
K8sName = Annotated[
    str, Field(min_length=1, max_length=63, pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListPodsArgs(_Args):
    namespace: K8sName
    label_selector: str | None = Field(default=None, max_length=200)


class DescribeArgs(_Args):
    namespace: K8sName
    kind: Literal["pod", "deployment", "service"]
    name: K8sName


class GetLogsArgs(_Args):
    namespace: K8sName
    # A pod by name, or a deployment whose pod mcp-ops picks (as `kubectl logs deploy/<name>`
    # does): the agent never reads tool output when it proposes calls, so it cannot copy a pod
    # name out of an earlier list_pods.
    pod: str | None = Field(
        default=None, min_length=1, max_length=253, pattern=r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$"
    )
    deployment: K8sName | None = None
    container: K8sName | None = None
    tail_lines: int = Field(default=200, ge=1, le=2000)
    previous: bool = False

    @model_validator(mode="after")
    def _one_target(self) -> GetLogsArgs:
        if (self.pod is None) == (self.deployment is None):
            raise ValueError("give either pod or deployment")
        return self


class GetEventsArgs(_Args):
    namespace: K8sName
    involved_object: K8sName | None = None


class RolloutStatusArgs(_Args):
    namespace: K8sName
    deployment: K8sName


class GetOperationArgs(_Args):
    operation_id: uuid.UUID


class RestartArgs(_Args):
    namespace: K8sName
    deployment: K8sName


class ScaleArgs(_Args):
    namespace: K8sName
    deployment: K8sName
    replicas: int = Field(ge=0, le=50)


class RollbackArgs(_Args):
    namespace: K8sName
    deployment: K8sName
    to_revision: int = Field(ge=1, description="explicit target revision, resolved at proposal")


class CreateIncidentArgs(_Args):
    severity: Literal["sev1", "sev2", "sev3", "sev4"]
    title: str = Field(min_length=3, max_length=200)
    summary: str = Field(min_length=3, max_length=4000)
    namespace: K8sName | None = None


class PageOncallArgs(_Args):
    incident_id: uuid.UUID
    rotation: K8sName = "sre-primary"


TOOL_ARGS: dict[str, type[_Args]] = {
    "list_pods": ListPodsArgs,
    "describe": DescribeArgs,
    "get_logs": GetLogsArgs,
    "get_events": GetEventsArgs,
    "rollout_status": RolloutStatusArgs,
    "get_operation": GetOperationArgs,
    "restart_deployment": RestartArgs,
    "scale_deployment": ScaleArgs,
    "rollback_deployment": RollbackArgs,
    "create_incident": CreateIncidentArgs,
    "page_oncall": PageOncallArgs,
}


def parse_args(tool: str, args: Mapping[str, Any]) -> _Args:
    """Validate `args` for `tool` (pydantic.ValidationError on anything unexpected)."""
    if tool not in TOOL_ARGS:
        raise ValueError(f"unknown tool {tool!r}")
    return TOOL_ARGS[tool].model_validate(dict(args))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def normalize_args(tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Validated args with defaults filled in, as plain JSON values (stable for hashing)."""
    return parse_args(tool, args).model_dump(mode="json")


def idempotency_key(
    thread_id: uuid.UUID | str | None, turn: int | None, tool: str, args: Mapping[str, Any]
) -> str:
    """sha256(thread, turn, tool, normalized args). mcp-ops takes thread and turn from the token."""
    raw = canonical_json(
        [str(thread_id) if thread_id else "", turn or 0, tool, normalize_args(tool, args)]
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def args_hash(tool: str, args: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json([tool, normalize_args(tool, args)]).encode()).hexdigest()


# --- approval policy ------------------------------------------------------------------------

ApprovalLevel = Literal["none", "operator", "admin"]
Risk = Literal["low", "medium", "high"]


class NamespacePolicy(BaseModel):
    """From config/mcp_ops.yaml: where tools may act at all, and where writes need an admin."""

    model_config = ConfigDict(frozen=True)

    allowed: frozenset[str]
    protected: frozenset[str] = frozenset()


def target_of(tool: str, args: _Args) -> tuple[str | None, str | None]:
    """(namespace, object name) a call acts on, for the policy, the audit log and the UI."""
    ns = getattr(args, "namespace", None)
    name = (
        getattr(args, "deployment", None)
        or getattr(args, "name", None)
        or getattr(args, "pod", None)
    )
    return ns, name


def required_approval(
    tool: str, args: _Args, policy: NamespacePolicy, *, route: Literal["action", "escalate"]
) -> ApprovalLevel:
    """Who must approve this call before it runs.

    Reads: nobody. Writes: the user confirms (operator); an admin approves in a protected
    namespace or when scaling to 0. Escalation tools: on the escalate route they run without
    approval (escalation must not wait on the user it is escalating for); when a user asks for
    an incident on the action route, they confirm it like a write. The design doc lists
    `create_incident` as "user confirms" and also automatic on escalate; this resolves both.
    """
    kind = tool_kind(tool)
    if kind == "read":
        return "none"
    if kind == "escalation":
        return "none" if route == "escalate" else "operator"
    ns, _ = target_of(tool, args)
    if ns in policy.protected:
        return "admin"
    if isinstance(args, ScaleArgs) and args.replicas == 0:
        return "admin"
    return "operator"


def risk_of(tool: str, args: _Args, policy: NamespacePolicy) -> Risk:
    kind = tool_kind(tool)
    if kind == "read":
        return "low"
    ns, _ = target_of(tool, args)
    if ns in policy.protected or isinstance(args, RollbackArgs):
        return "high"
    if isinstance(args, ScaleArgs) and args.replicas == 0:
        return "high"
    return "medium"


def expected_effect(tool: str, args: _Args) -> str:
    """One line for the approval dialog."""
    if isinstance(args, RestartArgs):
        return (
            f"Rolling restart of deployment {args.deployment} in {args.namespace}: pods are "
            "replaced one at a time."
        )
    if isinstance(args, ScaleArgs):
        return f"Set deployment {args.deployment} in {args.namespace} to {args.replicas} replicas."
    if isinstance(args, RollbackArgs):
        return (
            f"Roll deployment {args.deployment} in {args.namespace} back to revision "
            f"{args.to_revision}."
        )
    if isinstance(args, CreateIncidentArgs):
        return f"Open a {args.severity} incident: {args.title}"
    if isinstance(args, PageOncallArgs):
        return f"Page the {args.rotation} on-call rotation."
    return f"Read-only call to {tool}."


def allowed(
    tool: str, args: _Args, scopes: list[str], namespaces: list[str], policy: NamespacePolicy
) -> str | None:
    """None when the caller may call `tool` on this target; otherwise the reason it may not.
    Used by both sides: the agent to avoid proposing what will be refused, mcp-ops to refuse."""
    need = required_scope(tool)
    if need not in scopes:
        return f"{tool} needs the {need} scope"
    ns, _ = target_of(tool, args)
    if ns is not None:
        if ns not in policy.allowed:
            return f"namespace {ns} is not on the mcp-ops allow-list"
        if ns not in namespaces:
            return f"the caller may not act in namespace {ns}"
    return None


# --- records --------------------------------------------------------------------------------

OperationStatus = Literal[
    "pending", "approved", "running", "succeeded", "failed", "unknown", "refused"
]
ApprovalStatus = Literal["pending", "approved", "rejected", "expired", "executed"]
TERMINAL_OPERATION: frozenset[str] = frozenset({"succeeded", "failed", "refused"})


class OperationRecord(BaseModel):
    """One row of `operations`: every mcp-ops call (the audit log)."""

    id: uuid.UUID
    idempotency_key: str
    kind: ToolKind
    tool: str
    args: dict[str, Any]
    namespace: str | None = None
    target: str | None = None
    subject: str
    scopes: list[str]
    thread_id: uuid.UUID | None = None
    turn: int | None = None
    approval_id: uuid.UUID | None = None
    status: OperationStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    task_id: str | None = None
    latency_ms: float | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ApprovalRecord(BaseModel):
    """One row of `approvals`."""

    id: uuid.UUID
    thread_id: uuid.UUID
    turn: int
    tool: str
    args: dict[str, Any]
    args_hash: str
    namespace: str
    target: str
    risk: Risk
    required_scope: Literal["operator", "admin"]
    expected_effect: str
    status: ApprovalStatus
    requested_by: str
    decided_by: str | None = None
    decided_at: datetime | None = None
    expires_at: datetime
    created_at: datetime | None = None
    # Set once, by the resume that executes the approval (V07-1); mcp-ops still requires
    # status 'approved', so the claim is a column, not a status.
    execution_claimed_at: datetime | None = None


# --- the core -> mcp-ops token ---------------------------------------------------------------

MCP_TOKEN_AUDIENCE: Final = "mcp-ops"  # noqa: S105  (a public claim value, not a secret)
MCP_TOKEN_ISSUER: Final = "sre-core"  # noqa: S105  (a public claim value, not a secret)
MCP_TOKEN_MAX_TTL_S = 300


class McpTokenClaims(BaseModel):
    """What core vouches for when it calls mcp-ops for one user turn. The user's own Supabase
    token is never forwarded (design doc p10)."""

    model_config = ConfigDict(frozen=True)

    iss: Literal["sre-core"] = MCP_TOKEN_ISSUER
    aud: Literal["mcp-ops"] = MCP_TOKEN_AUDIENCE
    sub: str = Field(min_length=1)
    scopes: list[Scope]
    namespaces: list[str]
    thread_id: uuid.UUID | None = None
    turn: int | None = Field(default=None, ge=1)
    approval_id: uuid.UUID | None = None
    # The tools this token may call (None: any). An escalation tool used without an approval
    # (the escalate route) must be named here, so a token minted for a read cannot page anyone.
    tools: list[ToolName] | None = None
    max_calls: int = Field(default=2, ge=1, le=10)
    iat: int
    exp: int
    jti: str

    @model_validator(mode="after")
    def _short_lived(self) -> McpTokenClaims:
        if not 0 < self.exp - self.iat <= MCP_TOKEN_MAX_TTL_S:
            raise ValueError(f"token lifetime must be 1-{MCP_TOKEN_MAX_TTL_S} s")
        return self


def mint_mcp_token(
    *,
    secret: str,
    sub: str,
    scopes: list[Scope],
    namespaces: list[str],
    thread_id: uuid.UUID | None = None,
    turn: int | None = None,
    approval_id: uuid.UUID | None = None,
    tools: list[ToolName] | None = None,
    max_calls: int = 2,
    ttl_s: int = 120,
    now: float | None = None,
) -> str:
    """HS256 token for one turn's calls (core signs; mcp-ops verifies with the same secret)."""
    import jwt

    t = int(now if now is not None else time.time())
    claims = McpTokenClaims(
        sub=sub, scopes=scopes, namespaces=namespaces, thread_id=thread_id, turn=turn,
        approval_id=approval_id, tools=tools, max_calls=max_calls, iat=t, exp=t + ttl_s,
        jti=uuid.uuid4().hex,
    )  # fmt: skip
    return jwt.encode(claims.model_dump(mode="json"), secret, algorithm="HS256")


def verify_mcp_token(token: str, secret: str, *, leeway_s: int = 5) -> McpTokenClaims:
    """Verified claims, or jwt.InvalidTokenError / pydantic.ValidationError."""
    import jwt

    payload = jwt.decode(
        token, secret, algorithms=["HS256"], audience=MCP_TOKEN_AUDIENCE,
        issuer=MCP_TOKEN_ISSUER, leeway=leeway_s,
        options={"require": ["exp", "iat", "aud", "iss", "sub", "jti"]},
    )  # fmt: skip
    return McpTokenClaims.model_validate(payload)
