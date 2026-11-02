"""The action route: propose, check, approve, run, summarise.

Safety rules this module keeps (tests attack each):
* `propose` is the only node that turns the user's words into a tool call, and its prompt holds
  only the user's own messages. Tool output goes to `summarize`, which has no way to call a tool.
* A write runs only from an `ApprovalRecord` read back from the `ApprovalStore` with status
  `approved`; the resume request's own content is never trusted (see `apply_decision`).
* `await_approval` holds nothing but `interrupt()`: LangGraph re-runs an interrupted node on
  resume, so anything with a side effect lives in the nodes before and after it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import ValidationError

from sre_core.agent.models import AllProvidersFailed
from sre_core.agent.nodes.common import (
    Ctx,
    LimitReached,
    Meter,
    Node,
    ParseFailure,
    Update,
    to_json,
)
from sre_core.agent.ports import BudgetExceeded
from sre_core.agent.prompts import PROPOSABLE, action_summary_messages, proposal_messages
from sre_core.agent.schemas import ProposalOut
from sre_core.agent.state import AgentState
from sre_shared.contracts import ops
from sre_shared.contracts.agent import (
    ApprovalRequiredEvent,
    OperationEvent,
    ToolProposedEvent,
)
from sre_shared.models import Claims

log = logging.getLogger("sre_core.agent")

CLARIFY_DEFAULT = (
    "I could not tell which cluster operation you want. Name the namespace and the deployment "
    "(or pod) and say what to do."
)
MAX_USER_TURNS_IN_PROPOSAL = 3


def _ids(state: AgentState) -> tuple[uuid.UUID, int]:
    return uuid.UUID(state["thread_id"]), state["turn"]


def check_approved_execution(
    rec: ops.ApprovalRecord | None,
    call: dict[str, Any] | None,
    state: AgentState,
    claims: Claims,
    policy: ops.NamespacePolicy,
) -> str | None:
    """None when the stored approval permits running; otherwise what to tell the user.

    Everything is re-derived from the stored record and the approver's own claims, so a forged
    resume request, a stale record or an edit that raised the risk cannot run a write."""
    thread_id, turn = _ids(state)
    if rec is None:
        return "Not run: the approval record could not be found."
    if rec.status != "approved":
        return f"Not run: the approval was {rec.status}."
    if call is None or (rec.tool, rec.thread_id, rec.turn) != (call["tool"], thread_id, turn):
        return "Not run: the approval does not match the proposed operation."
    try:
        if rec.args_hash != ops.args_hash(rec.tool, rec.args):
            return "Not run: the approved arguments do not match their hash."
        parsed = ops.parse_args(rec.tool, rec.args)
    except (ValidationError, ValueError):
        return "Not run: the approved arguments are not valid."
    refusal = ops.allowed(rec.tool, parsed, list(claims.scopes), list(claims.namespaces), policy)
    if refusal:
        return f"Not run: {refusal}."
    level = ops.required_approval(rec.tool, parsed, policy, route="action")
    if level == "none":
        return "Not run: this call needs no approval and should not be here."
    if level == "admin" and rec.required_scope != "admin":
        return "Not run: the edited arguments need an admin approval."
    if rec.required_scope not in claims.scopes:
        return f"Not run: the {rec.required_scope} scope is needed to approve this."
    return None


def make_action_nodes(ctx: Ctx) -> dict[str, Node]:
    cfg = ctx.cfg
    limit = cfg.limits.tool_calls_per_turn

    # -- proposal --------------------------------------------------------------------------

    async def propose(state: AgentState, config: RunnableConfig) -> Update:
        async def body(meter: Meter, claims: Claims) -> Update:
            earlier = [h["content"] for h in state["history"] if h["role"] == "user"]
            messages = proposal_messages(
                earlier[-MAX_USER_TURNS_IN_PROPOSAL:], state["message"], limit
            )
            try:
                out = await ctx.structured(
                    meter, claims, "router", messages, ProposalOut, max_tokens=500
                )
            except (ParseFailure, AllProvidersFailed):
                return _final(CLARIFY_DEFAULT)
            accepted: list[dict[str, Any]] = []
            notes: list[str] = []
            for raw in out.calls:
                tool = raw.tool
                if len(accepted) >= limit:
                    notes.append(f"Not run: {tool} (at most {limit} tool calls per turn).")
                elif tool not in PROPOSABLE:
                    notes.append(f"Not run: {tool!r} is not a tool I can use.")
                else:
                    try:
                        parsed = ops.parse_args(tool, raw.args)
                    except (ValidationError, ValueError):
                        notes.append(f"Not run: the arguments for {tool} were not valid.")
                        continue
                    refusal = ops.allowed(
                        tool, parsed, list(claims.scopes), list(claims.namespaces), ctx.policy
                    )
                    if refusal:
                        notes.append(f"I can't run {tool}: {refusal}.")
                        continue
                    accepted.append(
                        {
                            "tool": tool,
                            "args": ops.normalize_args(tool, raw.args),
                            "kind": ops.tool_kind(tool),
                        }
                    )
            if not accepted:
                text = " ".join(notes) or out.clarification or CLARIFY_DEFAULT
                return _final(text)
            return {"calls": accepted, "notes": notes}

        return await ctx.run(state, config, body)

    def _final(text: str, **extra: Any) -> Update:
        return {
            "answer": text,
            "final_kind": "done",
            "final_route": "action",
            "calls": [],
            "current": None,
            **extra,
        }

    async def next_call(state: AgentState, config: RunnableConfig) -> Update:
        calls = list(state["calls"])
        if not calls:
            return {"current": None}
        call = calls.pop(0)
        if state["tool_calls"] >= limit:  # backstop: the proposal already honoured the limit
            note = f"Not run: {call['tool']} (at most {limit} tool calls per turn)."
            return {"calls": [], "current": None, "notes": [*state["notes"], note]}
        ctx.emit(ToolProposedEvent(tool=call["tool"], args=call["args"], kind=call["kind"]))
        return {"calls": calls, "current": call, "tool_calls": state["tool_calls"] + 1}

    # -- reads -----------------------------------------------------------------------------

    async def run_read(state: AgentState, config: RunnableConfig) -> Update:
        claims = ctx.claims(config)
        thread_id, turn = _ids(state)
        call = state["current"]
        assert call is not None
        results = list(state["tool_results"])
        try:
            out = await ctx.ops.read(
                call["tool"],
                call["args"],
                ctx.mint(claims, thread_id=thread_id, turn=turn, tools=[call["tool"]]),
            )
            results.append({"tool": call["tool"], "status": "ok", "output": out})
            return {"tool_results": results}
        except Exception as exc:  # the user hears that the read failed, not why in detail
            log.warning("read %s failed: %s", call["tool"], type(exc).__name__)
            note = f"{call['tool']} failed ({type(exc).__name__})."
            return {"tool_results": results, "notes": [*state["notes"], note]}

    # -- approval gate ---------------------------------------------------------------------

    async def request_approval(state: AgentState, config: RunnableConfig) -> Update:
        claims = ctx.claims(config)
        thread_id, turn = _ids(state)
        call = state["current"]
        assert call is not None
        tool = call["tool"]
        parsed = ops.parse_args(tool, call["args"])
        level = ops.required_approval(tool, parsed, ctx.policy, route="action")
        if level == "none":
            raise RuntimeError(f"{tool} needs no approval; it must not reach the gate")
        ns, target = ops.target_of(tool, parsed)
        now = datetime.now(UTC)
        rec = ops.ApprovalRecord(
            id=uuid.uuid4(), thread_id=thread_id, turn=turn, tool=tool,
            args=ops.normalize_args(tool, call["args"]),
            args_hash=ops.args_hash(tool, call["args"]),
            namespace=ns or "", target=target or "",
            risk=ops.risk_of(tool, parsed, ctx.policy), required_scope=level,
            expected_effect=ops.expected_effect(tool, parsed), status="pending",
            requested_by=claims.sub,
            expires_at=now + timedelta(minutes=cfg.approval.ttl_minutes), created_at=now,
        )  # fmt: skip
        await ctx.approvals.create(rec)
        await ctx.threads.set_status(thread_id, "awaiting_approval")
        ctx.emit(
            ApprovalRequiredEvent(
                approval_id=rec.id,
                tool=rec.tool,
                args=rec.args,
                namespace=rec.namespace,
                target=rec.target,
                risk=rec.risk,
                required_scope=level,
                expected_effect=rec.expected_effect,
                expires_at=rec.expires_at,
            )
        )
        return {"pending_approval_id": str(rec.id), "exec_args": None, "exec_approval_id": None}

    async def await_approval(state: AgentState, config: RunnableConfig) -> Update:
        # Only interrupt() lives here (see the module docstring). The resume value is not used:
        # the decision is read from the ApprovalStore by `apply_decision`.
        interrupt({"approval_id": state["pending_approval_id"]})
        return {}

    async def apply_decision(state: AgentState, config: RunnableConfig) -> Update:
        claims = ctx.approver(config)  # the approval check uses the approver, nothing else
        thread_id, _ = _ids(state)
        pending = state["pending_approval_id"]
        rec = await ctx.approvals.get(uuid.UUID(pending)) if pending else None
        refusal = check_approved_execution(rec, state["current"], state, claims, ctx.policy)
        await ctx.threads.set_status(thread_id, "active")
        if refusal is not None or rec is None:
            return _final(refusal or "Not run.", exec_args=None, exec_approval_id=None)
        # One atomic claim per approval: a second resume, or an approval that has expired since
        # it was decided, gets None and submits nothing.
        claimed = await ctx.approvals.claim_execution(rec.id)
        if claimed is None:
            return _final(
                "Not run: the approval was already used or has expired.",
                exec_args=None,
                exec_approval_id=None,
            )
        return {"exec_args": dict(claimed.args), "exec_approval_id": str(claimed.id)}

    # -- running an approved write ---------------------------------------------------------

    def requester_token(
        rec: ops.ApprovalRecord,
        thread_id: uuid.UUID,
        turn: int,
        *,
        tool: ops.ToolName,
        approval_id: uuid.UUID | None = None,
    ) -> str:
        """The token acts as the requester (never the approver) with viewer+operator scopes and
        the approval's namespace; admin is never put in a token (mcp-ops checks admin
        requirements against the approval record)."""
        return ctx.mint(
            ctx.claims_of_requester(rec.requested_by),
            thread_id=thread_id,
            turn=turn,
            approval_id=approval_id,
            scopes=["viewer", "operator"],
            namespaces=[rec.namespace] if rec.namespace else [],
            tools=[tool],
        )

    async def submit_write(state: AgentState, config: RunnableConfig) -> Update:
        thread_id, turn = _ids(state)
        call = state["current"]
        assert call is not None and state["exec_args"] is not None
        approval_id = uuid.UUID(str(state["exec_approval_id"]))
        stored = await ctx.approvals.get(approval_id)
        if stored is None:
            return _write_failed(state, call["tool"], "approval record missing")
        token = requester_token(stored, thread_id, turn, tool=call["tool"], approval_id=approval_id)
        tool, args = call["tool"], state["exec_args"]
        try:
            submit = ctx.ops.escalate if call["kind"] == "escalation" else ctx.ops.write
            rec = await submit(tool, args, token)
        except Exception as exc:
            log.warning("submitting %s failed: %s", tool, type(exc).__name__)
            return _write_failed(state, tool, f"{type(exc).__name__}")
        if rec.status != "refused":  # a refused operation never ran: leave the approval as is
            await ctx.approvals.mark_executed(approval_id)
        ctx.emit(OperationEvent(operation_id=rec.id, tool=tool, status=rec.status))
        return {"operation_id": str(rec.id), "operation_status": rec.status}

    def _write_failed(state: AgentState, tool: str, why: str) -> Update:
        failed = state.get("failed_writes", 0) + 1
        update: Update = {
            "operation_id": None,
            "operation_status": "failed",
            "failed_writes": failed,
            "notes": [*state["notes"], f"{tool} failed ({why})."],
            "tool_results": [
                *state["tool_results"],
                {"tool": tool, "status": "failed", "output": {"error": why}},
            ],
        }
        if failed >= 2:
            update["escalate_reason"] = f"{failed} writes failed in this thread"
        return update

    async def poll_write(state: AgentState, config: RunnableConfig) -> Update:
        """Poll the operation (Postgres, through get_operation) until it ends or times out."""
        thread_id, turn = _ids(state)
        call = state["current"]
        assert call is not None
        tool = call["tool"]
        if state["operation_id"] is None:  # submission failed: nothing to poll
            return {}
        stored = await ctx.approvals.get(uuid.UUID(str(state["exec_approval_id"])))
        if stored is None:
            return {"operation_status": "unknown"}
        op_id = uuid.UUID(state["operation_id"])
        status = state["operation_status"] or "running"
        record: ops.OperationRecord | None = None
        deadline = time.monotonic() + cfg.mcp.operation_timeout_s
        try:
            while status not in ops.TERMINAL_OPERATION:
                if time.monotonic() >= deadline:
                    status = "unknown"
                    break
                await asyncio.sleep(cfg.mcp.poll_interval_s)
                token = requester_token(
                    stored, thread_id, turn, tool="get_operation"
                )  # one call each
                record = await ctx.ops.operation(op_id, token)
                if record.status != status:
                    status = record.status
                    ctx.emit(
                        OperationEvent(
                            operation_id=op_id, tool=tool, status=status, result=record.result
                        )
                    )
        except Exception as exc:
            log.warning("polling %s failed: %s", tool, type(exc).__name__)
            status = "unknown"
        if status == "succeeded":
            output = record.result if record else None
            return {
                "operation_status": status,
                "tool_results": [
                    *state["tool_results"],
                    {"tool": tool, "status": status, "output": output},
                ],
            }
        if status == "unknown":
            note = f"{tool} is still running or its outcome is unknown; check it before retrying."
            return {"operation_status": status, "notes": [*state["notes"], note]}
        why = (record.error if record and record.error else status) or status
        return {**_write_failed(state, tool, str(why)), "operation_status": "failed"}

    # -- the report ------------------------------------------------------------------------

    async def summarize(state: AgentState, config: RunnableConfig) -> Update:
        results, notes = state["tool_results"], state["notes"]
        if not results:
            return _final("\n".join(notes) or "Nothing was run.")

        async def body(meter: Meter, claims: Claims) -> Update:
            msgs = action_summary_messages(state["standalone"], results, notes)
            res = await ctx.complete(meter, claims, "answer", msgs, max_tokens=400)
            text = res.text.strip()
            if notes:
                text += "\n\n" + "\n".join(notes)
            return _final(text)

        def plain() -> str:
            lines = [
                f"{r['tool']}: {r['status']}. {to_json(r.get('output'))[:500]}" for r in results
            ]
            return "\n".join([*lines, *notes])

        try:
            return await ctx.run_unguarded(state, config, body)
        except (AllProvidersFailed, LimitReached, BudgetExceeded):
            # The operations ran: report them even when no model call is possible now.
            return _final(plain())

    return {
        "propose": propose,
        "next_call": next_call,
        "run_read": run_read,
        "request_approval": request_approval,
        "await_approval": await_approval,
        "apply_decision": apply_decision,
        "submit_write": submit_write,
        "poll_write": poll_write,
        "summarize": summarize,
    }
