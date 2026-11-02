"""mcp-ops policy and execution: admit a call, run it once, record what happened.

Nothing here trusts the agent. Scope, namespace, approval and the call budget are checked from
the signed token and the database, using the same rules core uses (`sre_shared.contracts.ops`).
Every authenticated call leaves one `operations` row, allowed or refused.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from sre_mcp_ops.backends.base import BackendError, ClusterBackend
from sre_mcp_ops.store import REFUSED_SUFFIX, OpsStore, Row
from sre_shared.config import McpOpsConfig
from sre_shared.contracts import ops
from sre_shared.contracts.ops import McpTokenClaims

log = logging.getLogger(__name__)

TRUST = "untrusted_data"
# Audit rows keep the arguments a caller sent, but a refused call may carry anything: cap it.
MAX_AUDIT_ARGS_BYTES = 8192
# A replay that arrives while the first run is still going waits this long for its result.
REPLAY_WAIT_S = 5.0


class CallBudget:
    """`max_calls` per token id, counted in this process.

    The operations table has no token-id column and the schema is fixed, so the count lives in
    memory; a token lives at most 300 s, so the map stays small. One mcp-ops instance only:
    a second instance would give each token a second budget.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._seen: dict[str, tuple[int, int]] = {}

    def hit(self, claims: McpTokenClaims) -> bool:
        """Count one call; False once the token has used up `max_calls`."""
        now = self._clock()
        if len(self._seen) > 512:
            self._seen = {j: v for j, v in self._seen.items() if v[1] + 10 > now}
        count, _ = self._seen.get(claims.jti, (0, claims.exp))
        self._seen[claims.jti] = (count + 1, claims.exp)
        return count + 1 <= claims.max_calls


@dataclass
class Admission:
    tool: str
    kind: ops.ToolKind
    args: Any  # a validated ops argument model
    started: float
    op: Row | None = None  # writes and escalations: the stored operation
    read_id: uuid.UUID | None = None  # reads: the audit row to finish


def _view(row: Row, *, replayed: bool = False) -> dict[str, Any]:
    """The operation as callers see it (also what a write's task returns): the shared
    `OperationRecord` contract, so the agent parses it with the same model, plus `replayed`."""
    fields = ops.OperationRecord.model_fields
    record = ops.OperationRecord.model_validate({k: row[k] for k in fields if k in row})
    return {**record.model_dump(mode="json"), "replayed": replayed}


def _validation_reason(exc: ValidationError) -> str:
    # Field names and error types only: the message must not echo what the caller sent.
    bad = sorted({".".join(str(p) for p in e["loc"]) or "arguments" for e in exc.errors()})
    return f"invalid arguments: {', '.join(bad)}"


def _audit_args(raw: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(raw, default=str)
    if len(text) <= MAX_AUDIT_ARGS_BYTES:
        return raw
    return {"truncated": True, "size": len(text)}


class OpsService:
    def __init__(
        self,
        *,
        config: McpOpsConfig,
        store: OpsStore,
        backend: ClusterBackend,
        budget: CallBudget | None = None,
    ) -> None:
        self.config = config
        self.policy = config.policy()
        self.store = store
        self.backend = backend
        self.budget = budget or CallBudget()
        self.max_log_bytes = int(config.logs["max_bytes"])

    # --- admission ----------------------------------------------------------------------------

    async def _refuse(
        self,
        claims: McpTokenClaims,
        tool: str,
        raw: dict[str, Any],
        reason: str,
        started: float,
        key: str | None = None,
    ) -> ToolError:
        op_id = uuid.uuid4()
        known = tool in ops.TOOL_ARGS
        await self.store.insert_audit(
            {
                "id": op_id,
                "idempotency_key": f"{key or 'invalid'}{REFUSED_SUFFIX}{op_id}",
                "kind": ops.tool_kind(tool) if known else "read",
                "tool": tool[:100],
                "args": _audit_args(raw),
                "namespace": raw.get("namespace")
                if isinstance(raw.get("namespace"), str)
                else None,
                "target": None,
                "subject": claims.sub,
                "scopes": list(claims.scopes),
                "thread_id": claims.thread_id,
                "turn": claims.turn,
                "approval_id": claims.approval_id,
                "status": "refused",
                "error": reason,
                "latency_ms": (time.monotonic() - started) * 1000,
            }
        )
        return ToolError(f"refused: {reason}")

    async def admit(self, claims: McpTokenClaims, tool: str, raw: dict[str, Any]) -> Admission:
        started = time.monotonic()
        if not self.budget.hit(claims):
            raise await self._refuse(
                claims, tool, raw, "max_calls for this token is used up", started
            )
        if tool not in ops.TOOL_ARGS:
            raise await self._refuse(claims, tool, raw, "unknown tool", started)
        try:
            args = ops.parse_args(tool, raw)
        except ValidationError as exc:
            raise await self._refuse(claims, tool, raw, _validation_reason(exc), started) from None
        reason = ops.allowed(tool, args, list(claims.scopes), list(claims.namespaces), self.policy)
        if reason is not None:
            raise await self._refuse(claims, tool, raw, reason, started)
        kind = ops.tool_kind(tool)
        ns, target = ops.target_of(tool, args)
        norm = ops.normalize_args(tool, raw)
        if kind == "read":
            row = await self.store.insert_audit(
                {
                    "id": uuid.uuid4(),
                    # Reads share keys on purpose (the unique index skips them): this lets the
                    # log show how often the same read was made in a turn.
                    "idempotency_key": ops.idempotency_key(
                        claims.thread_id, claims.turn, tool, norm
                    ),
                    "kind": "read",
                    "tool": tool,
                    "args": norm,
                    "namespace": ns,
                    "target": target,
                    "subject": claims.sub,
                    "scopes": list(claims.scopes),
                    "thread_id": claims.thread_id,
                    "turn": claims.turn,
                    "approval_id": None,
                    "status": "running",
                    "error": None,
                    "latency_ms": None,
                }
            )
            return Admission(tool, kind, args, started, read_id=row["id"])

        if claims.thread_id is None or claims.turn is None:
            raise await self._refuse(
                claims, tool, raw, f"{tool} needs a token bound to a turn", started
            )
        extra = await self._escalation_checks(claims, tool, args)
        if extra is not None:
            raise await self._refuse(claims, tool, raw, extra, started)
        key = ops.idempotency_key(claims.thread_id, claims.turn, tool, norm)
        # The route is decided by the token: an escalation tool with no approval is the escalate
        # route (nobody must approve escalating); anything with an approval id is checked as an
        # action. A write can never take the escalate route.
        route: Literal["action", "escalate"] = (
            "escalate" if kind == "escalation" and claims.approval_id is None else "action"
        )
        level = ops.required_approval(tool, args, self.policy, route=route)

        stored = await self.store.get_by_key(key)
        if stored is not None:
            if stored["subject"] != claims.sub:
                raise await self._refuse(
                    claims, tool, raw, "operation belongs to another caller", started, key
                )
            if stored["status"] != "approved":
                return Admission(tool, kind, args, started, op=stored)  # a replay: nothing to check
            # Stored but never started (a lost task): it may only start again if still approved.

        if level != "none":
            reason = await self._check_approval(claims, tool, args, level)
            if reason is not None:
                raise await self._refuse(claims, tool, raw, reason, started, key)
        row, _created = await self.store.insert_first(
            {
                "id": uuid.uuid4(),
                "idempotency_key": key,
                "kind": kind,
                "tool": tool,
                "args": norm,
                "namespace": ns,
                "target": target,
                "subject": claims.sub,
                "scopes": list(claims.scopes),
                "thread_id": claims.thread_id,
                "turn": claims.turn,
                "approval_id": claims.approval_id if level != "none" else None,
                "status": "approved",
            }
        )
        if row["subject"] != claims.sub:  # lost a race to another caller with the same key
            raise await self._refuse(
                claims, tool, raw, "operation belongs to another caller", started, key
            )
        return Admission(tool, kind, args, started, op=row)

    async def _escalation_checks(self, claims: McpTokenClaims, tool: str, args: Any) -> str | None:
        if tool == "page_oncall":
            if args.rotation not in self.config.rotations:
                return f"rotation {args.rotation} is not configured"
            incident = await self.store.get_incident(args.incident_id)
            if incident is None or (
                incident["opened_by"] != claims.sub and "admin" not in claims.scopes
            ):
                return "incident not found"
        return None

    async def _check_approval(
        self, claims: McpTokenClaims, tool: str, args: Any, level: str
    ) -> str | None:
        """None when the token's approval covers exactly this call; otherwise why not."""
        # Approved writes run on behalf of the requester (O-2): the token never carries admin.
        # An admin-level call is proven by the approval row (required_scope, decided_by), which
        # core only sets after checking the approver held admin.
        if "operator" not in claims.scopes:
            return f"{tool} needs the operator scope"
        if claims.approval_id is None:
            return f"{tool} needs an approval"
        appr = await self.store.get_approval(claims.approval_id)
        if appr is None:
            return "approval not found"
        if appr["status"] != "approved":
            return f"approval is {appr['status']}"
        if not appr["live"]:
            return "approval has expired"
        if appr["tool"] != tool or appr["args_hash"] != ops.args_hash(tool, args.model_dump()):
            return "approval is for a different call"
        if appr["thread_id"] != claims.thread_id or appr["turn"] != claims.turn:
            return "approval belongs to another turn"
        if appr["requested_by"] != claims.sub:
            return "approval was requested by someone else"
        if level == "admin" and (appr["required_scope"] != "admin" or not appr["decided_by"]):
            return "this call needs an admin's approval"
        return None

    # --- execution ----------------------------------------------------------------------------

    async def execute(
        self, claims: McpTokenClaims, tool: str, raw: dict[str, Any], task_id: str | None
    ) -> dict[str, Any]:
        """Run an admitted write or escalation (the task body). Safe to run any number of times
        for one operation: only the call that wins the `approved -> running` claim does the work."""
        try:
            args = ops.parse_args(tool, raw)
            key = ops.idempotency_key(
                claims.thread_id, claims.turn, tool, ops.normalize_args(tool, raw)
            )
        except (ValidationError, ValueError):
            raise ToolError("refused: invalid call") from None
        row = await self.store.get_by_key(key)
        if row is None or row["subject"] != claims.sub or row["tool"] != tool:
            raise ToolError("refused: this call was not admitted")
        won = await self.store.claim(row["id"], task_id)
        if won is None:
            return await self._replayed(row["id"])
        return _view(await self._perform(won, args))

    async def _replayed(self, op_id: uuid.UUID) -> dict[str, Any]:
        deadline = time.monotonic() + REPLAY_WAIT_S
        while True:
            row = await self.store.get(op_id)
            assert row is not None
            if row["status"] == "approved":  # never started and cannot start: approval lapsed
                row = (
                    await self.store.refuse_unclaimed(row, "approval expired before execution")
                    or row
                )
            if row["status"] not in ("running", "approved") or time.monotonic() >= deadline:
                return _view(row, replayed=True)
            await asyncio.sleep(0.05)

    async def _perform(self, op: Row, args: Any) -> Row:
        tool = op["tool"]
        at = op["created_at"].isoformat()  # the operation's own time: a retry stamps the same
        key = op["idempotency_key"]
        extra: tuple[str, list[Any]] | None = None
        try:
            if tool == "restart_deployment":
                result = await self.backend.restart(args.namespace, args.deployment, key=key, at=at)
            elif tool == "scale_deployment":
                result = await self.backend.scale(
                    args.namespace, args.deployment, args.replicas, key=key
                )
            elif tool == "rollback_deployment":
                result = await self.backend.rollback(
                    args.namespace, args.deployment, args.to_revision, key=key
                )
            elif tool == "create_incident":
                incident_id = uuid.uuid4()
                extra = (
                    "insert into public.incidents (id, thread_id, operation_id, opened_by, "
                    "severity, title, summary) values (%s, %s, %s, %s, %s, %s, %s)",
                    [incident_id, op["thread_id"], op["id"], op["subject"], args.severity,
                     args.title, args.summary],
                )  # fmt: skip
                result = {
                    "incident_id": str(incident_id),
                    "severity": args.severity,
                    "status": "open",
                }
            elif tool == "page_oncall":
                extra = (
                    "insert into public.pages (incident_id, operation_id, target) "
                    "values (%s, %s, %s)",
                    [args.incident_id, op["id"], args.rotation],
                )
                result = {"incident_id": str(args.incident_id), "rotation": args.rotation,
                          "status": "sent"}  # fmt: skip
            else:  # admission only lets known write tools through
                raise BackendError("unsupported", f"{tool} is not executable")
            status, error = "succeeded", None
        except BackendError as exc:
            result, status, error, extra = None, "failed", f"{exc.code}: {exc.message}", None
        except Exception:
            log.exception("operation %s failed", op["id"])
            result, status, error, extra = None, "failed", "internal error", None
        try:
            done = await self.store.finish(op["id"], status, result, error, extra=extra)
        except Exception:
            # The change may be in place but unrecorded (a unique/FK violation on the incident
            # row, a dropped connection). Leave the row `running`: the reconciler settles it.
            log.exception("could not record the result of operation %s", op["id"])
            raise ToolError("the result could not be recorded; check get_operation") from None
        if done is None:  # not `running` any more: the reconciler got there first
            done = await self.store.get(op["id"])
        assert done is not None
        if status == "succeeded" and tool in ops.ESCALATION_TOOLS:
            await self.backend.record_escalation(tool, args.model_dump(mode="json"), key=key)
        return done

    # --- reads --------------------------------------------------------------------------------

    async def read(self, tool: str, args: Any) -> dict[str, Any]:
        """A backend read, wrapped so a consumer can tell data from instructions."""
        try:
            data = await self._read(tool, args)
        except BackendError as exc:
            raise ToolError(f"{exc.code}: {exc.message}") from None
        return {"tool": tool, "trust": TRUST, "data": data}

    async def _read(self, tool: str, a: Any) -> dict[str, Any]:
        b = self.backend
        if tool == "list_pods":
            return {
                "namespace": a.namespace,
                "pods": await b.list_pods(a.namespace, a.label_selector),
            }
        if tool == "describe":
            return await b.describe(a.namespace, a.kind, a.name)
        if tool == "get_events":
            return {
                "namespace": a.namespace,
                "events": await b.get_events(a.namespace, a.involved_object),
            }
        if tool == "rollout_status":
            return await b.rollout_status(a.namespace, a.deployment)
        if tool == "get_logs":
            chunk = await b.get_logs(a.namespace, a.pod, a.container, a.tail_lines, a.previous)
            raw = chunk.text.encode()
            cut = len(raw) > self.max_log_bytes
            if cut:  # keep the end of the log: that is where the failure is
                raw = raw[-self.max_log_bytes :]
            return {
                "namespace": a.namespace,
                "pod": a.pod,
                "container": a.container,
                "previous": a.previous,
                "text": raw.decode(errors="ignore"),
                "truncated": chunk.truncated or cut,
                "bytes": len(raw),
            }
        raise ToolError("unknown read tool")

    async def get_operation(
        self, claims: McpTokenClaims, operation_id: uuid.UUID
    ) -> dict[str, Any]:
        row = await self.store.get(operation_id)
        # A caller sees their own operations (an admin sees all); anything else looks absent.
        if row is None or (row["subject"] != claims.sub and "admin" not in claims.scopes):
            raise ToolError("operation not found")
        return _view(row)

    async def finish_read(self, adm: Admission, result: Any, error: str | None) -> None:
        assert adm.read_id is not None
        summary = None
        if result is not None:
            structured = getattr(result, "structured_content", None)
            summary = {"bytes": len(json.dumps(structured, default=str))}
        await self.store.finish_read(
            adm.read_id,
            "succeeded" if error is None else "failed",
            summary,
            error,
            (time.monotonic() - adm.started) * 1000,
        )
