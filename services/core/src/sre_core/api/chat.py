"""Chat API (Phase 3): `/chat` SSE, threads, approvals, feedback and UI events.

Every turn and every approval decision passes through here, so the approval gate's guarantees
live in this module and the stores behind it: an approval is decided once (a conditional update),
never after it expired (the database clock), and only by someone allowed. The agent resumes as a
server-side task the runtime holds (`runs.RunManager`), with the approval record read back from
the database, never from the request body.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from sre_core.agent.ports import (
    BudgetExceeded,
    ChatAgent,
    MessageRecord,
    ThreadBusy,
    ThreadNotFound,
    ThreadRecord,
)
from sre_core.answer.service import ConfigError
from sre_core.api.auth import claims_from_authorization
from sre_core.api.runs import RunManager, resume_source, sweep_once
from sre_core.api.runtime import Runtime, register_default_handlers
from sre_core.retrieval import NoActiveCorpusError
from sre_core.store.approvals import PgApprovalStore
from sre_core.store.feedback import PgFeedbackStore
from sre_core.store.outbox import OutboxDrainer, PgOutbox
from sre_core.store.threads import PgThreadStore
from sre_shared.config import load_agent_config
from sre_shared.contracts import agent as ev
from sre_shared.contracts import ops
from sre_shared.llm import QuotaExceeded
from sre_shared.models import Claims

log = logging.getLogger(__name__)
router = APIRouter()

_RANK = {"none": 0, "operator": 1, "admin": 2}


# --- per-runtime state ------------------------------------------------------------------------


@dataclass
class ChatServices:
    """Kept in `Runtime.extras["chat"]`; created on first use so a Runtime built from fakes in a
    test needs no setup."""

    manager: RunManager
    planned_tokens: int
    drainer: OutboxDrainer | None = None
    sweep_interval_s: float = 60.0
    seen_sends: dict[tuple[str, str], tuple[uuid.UUID, int]] = field(default_factory=dict)
    background: list[asyncio.Task[None]] = field(default_factory=list)

    def drain(self) -> None:
        if self.drainer is not None:
            self.drainer.kick()


def services(rt: Runtime) -> ChatServices:
    if rt.extras is None:
        rt.extras = {}
    svc: ChatServices | None = rt.extras.get("chat")
    if svc is None:
        cfg = load_agent_config()
        register_default_handlers()
        svc = ChatServices(
            manager=RunManager(rt.threads),
            planned_tokens=cfg.limits.answer_token_budget,
            drainer=OutboxDrainer(rt.outbox) if isinstance(rt.outbox, PgOutbox) else None,
            sweep_interval_s=float(cfg.approval.sweep_interval_s),
        )
        rt.extras["chat"] = svc
    return svc


async def _sweep_loop(rt: Runtime, svc: ChatServices) -> None:
    while True:
        await asyncio.sleep(svc.sweep_interval_s)
        try:
            if rt.agent is not None:
                await sweep_once(svc.manager, rt.approvals, rt.agent)
            svc.drain()
        except Exception:
            log.exception("approval sweep failed")


def start_background(rt: Runtime) -> None:
    """Start the approval sweeper (and periodic outbox drain) as tasks held by the runtime."""
    svc = services(rt)
    if not svc.background:
        svc.background.append(asyncio.get_running_loop().create_task(_sweep_loop(rt, svc)))


async def stop_background(rt: Runtime) -> None:
    svc = services(rt)
    for task in svc.background:
        task.cancel()
    await asyncio.gather(*svc.background, return_exceptions=True)
    svc.background.clear()
    await svc.manager.close()
    if svc.drainer is not None:
        await svc.drainer.close()


# --- dependencies -----------------------------------------------------------------------------


def get_runtime(request: Request) -> Runtime:
    rt: Runtime | None = getattr(request.app.state, "runtime", None)
    if rt is None:
        raise HTTPException(status_code=503, detail="chat is not configured")
    return rt


def user_claims(authorization: Annotated[str | None, Header()] = None) -> Claims:
    claims = claims_from_authorization(authorization)
    if claims.role != "authenticated":
        raise HTTPException(status_code=401, detail="sign in to use the chat")
    return claims


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
UserDep = Annotated[Claims, Depends(user_claims)]


def to_http(exc: BaseException) -> HTTPException:
    if isinstance(exc, ThreadNotFound):
        return HTTPException(status_code=404, detail="thread not found")
    if isinstance(exc, ThreadBusy):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, BudgetExceeded):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, NoActiveCorpusError | QuotaExceeded):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ConfigError):
        return HTTPException(status_code=422, detail=str(exc))
    log.exception("chat turn failed before its first event", exc_info=exc)
    return HTTPException(status_code=500, detail="the turn failed")


def sse_frame(event: ev.AgentEvent) -> str:
    name, data = ev.sse_payload(event)
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def sse_response(events: AsyncIterator[ev.AgentEvent]) -> StreamingResponse:
    """Await the first event so errors raised before it become HTTP statuses; errors after the
    200 are already `error` events (the run manager publishes them)."""
    try:
        first = await anext(events)
    except StopAsyncIteration:
        raise HTTPException(status_code=502, detail="the agent produced no events") from None
    except Exception as exc:
        raise to_http(exc) from exc

    async def frames() -> AsyncIterator[str]:
        yield sse_frame(first)
        async for event in events:
            yield sse_frame(event)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- visibility helpers (RLS when the stores are Postgres) --------------------------------------


async def _visible_approval(
    rt: Runtime, claims: Claims, approval_id: uuid.UUID
) -> ops.ApprovalRecord | None:
    if isinstance(rt.approvals, PgApprovalStore):
        return await rt.approvals.get_visible(claims, approval_id)
    rec = await rt.approvals.get(approval_id)  # in-memory fakes: apply the policy in code
    if rec is None:
        return None
    if await rt.threads.get(claims, rec.thread_id) is not None:
        return rec
    return rec if rec.required_scope == "admin" and "admin" in claims.scopes else None


async def _pending_approvals(
    rt: Runtime, claims: Claims, thread_id: uuid.UUID | None = None
) -> list[ops.ApprovalRecord]:
    if isinstance(rt.approvals, PgApprovalStore):
        return await rt.approvals.list_visible(claims, status="pending", thread_id=thread_id)
    items: dict[uuid.UUID, ops.ApprovalRecord] = getattr(rt.approvals, "items", {})
    out = []
    for rec in items.values():
        if rec.status != "pending" or (thread_id and rec.thread_id != thread_id):
            continue
        if await _visible_approval(rt, claims, rec.id) is not None:
            out.append(rec)
    return out


# --- POST /chat -------------------------------------------------------------------------------


@router.post("/chat")
async def chat(body: ev.ChatRequest, rt: RuntimeDep, claims: UserDep) -> StreamingResponse:
    """One turn as SSE (`AgentEvent`s). No `thread_id` starts a thread; a retried
    `client_message_id` gets the same turn back instead of a new one."""
    agent = _agent(rt)
    svc = services(rt)
    try:
        await rt.budget.check(claims, svc.planned_tokens)
        thread, turn, replay = await _claim(rt, svc, claims, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise to_http(exc) from exc
    if replay:
        return await sse_response(_replay(rt, svc, claims, thread, turn))
    run = svc.manager.start(
        thread.id, lambda: agent.stream_turn(claims, thread.id, turn, body.message), turn=turn
    )
    svc.drain()
    return await sse_response(run.subscribe())


def _agent(rt: Runtime) -> ChatAgent:
    if rt.agent is None:
        raise HTTPException(status_code=503, detail="the agent is not available")
    return rt.agent


async def _claim(
    rt: Runtime, svc: ChatServices, claims: Claims, body: ev.ChatRequest
) -> tuple[ThreadRecord, int, bool]:
    title = body.message[:80]
    if isinstance(rt.threads, PgThreadStore):
        c = await rt.threads.claim_message(
            claims,
            body.thread_id,
            body.message,
            client_message_id=body.client_message_id,
            title=title,
        )
        return c.thread, c.turn, c.replay
    # Any other ThreadStore (the in-memory fake): the same steps through the port, with the
    # retry memory kept in process.
    key = (claims.sub, body.client_message_id or "")
    if body.client_message_id and key in svc.seen_sends:
        thread_id, turn = svc.seen_sends[key]
        found = await rt.threads.get(claims, thread_id)
        if found is None:
            raise ThreadNotFound(str(thread_id))
        return found, turn, True
    thread = (
        await rt.threads.create(claims, title)
        if body.thread_id is None
        else await rt.threads.get(claims, body.thread_id)
    )
    if thread is None:
        raise ThreadNotFound(str(body.thread_id))
    turn = await rt.threads.claim_turn(claims, thread.id)
    await rt.threads.add_message(
        MessageRecord(thread_id=thread.id, turn=turn, role="user", content=body.message)
    )
    if body.client_message_id:
        svc.seen_sends[key] = (thread.id, turn)
    return thread, turn, False


async def _replay(
    rt: Runtime, svc: ChatServices, claims: Claims, thread: ThreadRecord, turn: int
) -> AsyncIterator[ev.AgentEvent]:
    """A retried send: follow the run if it is still around, else say what the turn became."""
    run = svc.manager.current(thread.id)
    if run is not None and run.turn == turn:
        async for event in run.subscribe():
            yield event
        return
    yield ev.TurnEvent(thread_id=thread.id, turn=turn)
    for msg in await rt.threads.messages(claims, thread.id, last=50):
        if msg.turn == turn and msg.role == "assistant":
            yield ev.DoneEvent(
                thread_id=thread.id,
                turn=turn,
                route=msg.route or "simple",
                answer=msg.content,
                citations=msg.citations,
            )
            return
    for a in await _pending_approvals(rt, claims, thread.id):
        if a.turn == turn:
            yield ev.ApprovalRequiredEvent(
                approval_id=a.id, tool=a.tool, args=a.args, namespace=a.namespace,
                target=a.target, risk=a.risk, required_scope=a.required_scope,
                expected_effect=a.expected_effect, expires_at=a.expires_at,
            )  # fmt: skip
            return
    yield ev.ErrorEvent(type="TurnInterrupted", detail="this turn did not finish; send it again")


# --- threads ----------------------------------------------------------------------------------


class ThreadView(BaseModel):
    thread: ThreadRecord
    messages: list[MessageRecord]
    pending_approvals: list[ops.ApprovalRecord]


@router.get("/threads", response_model=list[ThreadRecord])
async def list_threads(
    rt: RuntimeDep, claims: UserDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[ThreadRecord]:
    return await rt.threads.list_threads(claims, limit)


@router.get("/threads/{thread_id}", response_model=ThreadView)
async def get_thread(
    thread_id: uuid.UUID,
    rt: RuntimeDep,
    claims: UserDep,
    last: Annotated[int, Query(ge=1, le=500)] = 100,
) -> ThreadView:
    thread = await rt.threads.get(claims, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    try:
        messages = await rt.threads.messages(claims, thread_id, last=last)
    except ThreadNotFound as exc:
        raise to_http(exc) from exc
    return ThreadView(
        thread=thread,
        messages=messages,
        pending_approvals=await _pending_approvals(rt, claims, thread_id),
    )


@router.get("/threads/{thread_id}/live")
async def live(thread_id: uuid.UUID, rt: RuntimeDep, claims: UserDep) -> StreamingResponse:
    """The events of the run on this thread: the buffered ones first, then the live tail, ending
    at done / abstain / error / approval_required / escalated. After an approval the resumed run
    replaces the buffer, so reconnect to follow it."""
    if await rt.threads.get(claims, thread_id) is None:
        raise HTTPException(status_code=404, detail="thread not found")
    run = services(rt).manager.current(thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no run on this thread")
    return await sse_response(run.subscribe(stop_at_terminal=True))


# --- approvals --------------------------------------------------------------------------------


@router.get("/approvals", response_model=list[ops.ApprovalRecord])
async def list_approvals(
    rt: RuntimeDep, claims: UserDep, status: Literal["pending"] = "pending"
) -> list[ops.ApprovalRecord]:
    """The caller's own pending approvals, plus the admin-required ones for admins (RLS)."""
    return await _pending_approvals(rt, claims)


class ApprovalResult(BaseModel):
    approval: ops.ApprovalRecord


@router.post("/approvals/{approval_id}", response_model=ApprovalResult)
async def decide_approval(
    approval_id: uuid.UUID, body: ev.ApprovalDecision, rt: RuntimeDep, claims: UserDep
) -> ApprovalResult:
    agent = _agent(rt)
    svc = services(rt)
    rec = await _visible_approval(rt, claims, approval_id)
    if rec is None:  # not the owner's, and not an admin's: indistinguishable from missing
        raise HTTPException(status_code=404, detail="approval not found")
    is_admin = "admin" in claims.scopes
    if body.decision != "reject" and rec.required_scope == "admin" and not is_admin:
        raise HTTPException(status_code=403, detail="this approval needs an admin")

    new_args: dict[str, Any] | None = None
    if body.decision == "edit":
        new_args = _checked_edit(rt, rec, body.args, is_admin)
    elif body.args is not None:
        raise HTTPException(status_code=422, detail="args are accepted only with decision=edit")

    decided = await rt.approvals.decide(
        approval_id,
        decision="rejected" if body.decision == "reject" else "approved",
        decided_by=claims.sub,
        args=new_args,
    )
    if decided is None:
        await _conflict(rt, svc, agent, approval_id)
        raise HTTPException(status_code=409, detail="unreachable")  # _conflict always raises
    # The run continues from the database row, not from what the client sent, and as a task the
    # runtime holds: the approver may close the tab.
    fresh = await rt.approvals.get(approval_id) or decided
    svc.manager.start(
        fresh.thread_id, resume_source(agent, claims, fresh), turn=fresh.turn, resume=True
    )
    svc.drain()
    return ApprovalResult(approval=fresh)


def _checked_edit(
    rt: Runtime, rec: ops.ApprovalRecord, args: dict[str, Any] | None, is_admin: bool
) -> dict[str, Any]:
    """Validate edited arguments like a new proposal: the schema, the namespace allow-list, and
    the approval level they now need (a higher level than the record's needs an admin)."""
    if not args:
        raise HTTPException(status_code=422, detail="decision=edit needs args")
    try:
        parsed = ops.parse_args(rec.tool, args)
    except ValidationError as exc:
        problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        raise HTTPException(status_code=422, detail="; ".join(problems)) from exc
    ns, _ = ops.target_of(rec.tool, parsed)
    if ns is not None and ns not in rt.policy.allowed:
        raise HTTPException(status_code=403, detail=f"namespace {ns} is not allowed")
    need = ops.required_approval(rec.tool, parsed, rt.policy, route="action")
    if _RANK[need] > _RANK[rec.required_scope] and not is_admin:
        raise HTTPException(status_code=403, detail="the edited call needs an admin's approval")
    return ops.normalize_args(rec.tool, args)


async def _conflict(
    rt: Runtime, svc: ChatServices, agent: ChatAgent, approval_id: uuid.UUID
) -> None:
    """The decision lost: already decided, or expired. Run the sweeper step now so an expired
    approval releases its thread without waiting for the interval."""
    try:
        await sweep_once(svc.manager, rt.approvals, agent)
    except Exception:
        log.exception("sweep after a refused decision failed")
    now = await rt.approvals.get(approval_id)
    state = "expired" if now is None or now.status in ("pending", "expired") else now.status
    raise HTTPException(status_code=409, detail=f"approval is {state}")


# --- feedback and UI events -------------------------------------------------------------------


def _feedback_store(rt: Runtime) -> PgFeedbackStore:
    store = (rt.extras or {}).get("feedback")
    if not isinstance(store, PgFeedbackStore):
        raise HTTPException(status_code=501, detail="feedback is not configured")
    return store


async def _owned_turn(rt: Runtime, claims: Claims, thread_id: uuid.UUID, turn: int) -> None:
    thread = await rt.threads.get(claims, thread_id)
    if thread is None or turn >= thread.next_turn:
        raise HTTPException(status_code=404, detail="no such turn")


@router.post("/feedback")
async def post_feedback(
    body: ev.FeedbackRequest, rt: RuntimeDep, claims: UserDep
) -> dict[str, str]:
    if body.rating is None and not body.sources:
        raise HTTPException(status_code=422, detail="send a rating or source marks")
    store = _feedback_store(rt)
    await _owned_turn(rt, claims, body.thread_id, body.turn)
    await store.save_feedback(claims, body)
    if body.rating is not None:
        trace_run_id = None
        for msg in await rt.threads.messages(claims, body.thread_id, last=50):
            if msg.turn == body.turn and msg.role == "assistant":
                trace_run_id = msg.trace_run_id
        # Once per (thread, turn, user), the payload `feedback_sync` (T16) documents.
        await rt.outbox.enqueue(
            "feedback_sync",
            f"feedback:{body.thread_id}:{body.turn}:{claims.sub}",
            {
                "trace_run_id": trace_run_id,
                "owner_sub": claims.sub,
                "rating": body.rating,
                "reason": body.reason,
            },
        )
    services(rt).drain()
    return {"status": "stored"}


@router.post("/events")
async def post_event(body: ev.UiEventRequest, rt: RuntimeDep, claims: UserDep) -> dict[str, str]:
    store = _feedback_store(rt)
    await _owned_turn(rt, claims, body.thread_id, body.turn)
    await store.save_event(claims, body)
    return {"status": "stored"}
