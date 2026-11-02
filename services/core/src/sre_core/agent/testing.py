"""In-memory fakes of every port in `sre_core.agent.ports`, for tests across Phase 3 tasks.

They keep the same rules as the real implementations (owner-only threads, a turn claimed while
an approval is pending is refused, an approval is decided once and never after expiry), so a test
written against a fake stays meaningful against the database.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sre_core.ports import (
    BudgetExceeded,
    CachedAnswer,
    MessageRecord,
    ModelRole,
    ThreadBusy,
    ThreadNotFound,
    ThreadRecord,
    ThreadStatus,
)
from sre_shared.contracts import agent as ev
from sre_shared.contracts import ops
from sre_shared.llm import ChatResult, Message
from sre_shared.models import Claims

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryThreadStore:
    def __init__(self) -> None:
        self.threads: dict[uuid.UUID, ThreadRecord] = {}
        self.msgs: list[MessageRecord] = []

    def _own(self, claims: Claims, thread_id: uuid.UUID) -> ThreadRecord:
        t = self.threads.get(thread_id)
        if t is None or t.owner_sub != claims.sub:
            raise ThreadNotFound(str(thread_id))
        return t

    async def create(self, claims: Claims, title: str | None = None) -> ThreadRecord:
        t = ThreadRecord(id=uuid.uuid4(), owner_sub=claims.sub, title=title, created_at=utcnow())
        self.threads[t.id] = t
        return t

    async def get(self, claims: Claims, thread_id: uuid.UUID) -> ThreadRecord | None:
        try:
            return self._own(claims, thread_id)
        except ThreadNotFound:
            return None

    async def list_threads(self, claims: Claims, limit: int = 50) -> list[ThreadRecord]:
        return [t for t in self.threads.values() if t.owner_sub == claims.sub][:limit]

    async def claim_turn(self, claims: Claims, thread_id: uuid.UUID) -> int:
        t = self._own(claims, thread_id)
        if t.status in ("awaiting_approval", "escalated"):
            raise ThreadBusy(f"thread {thread_id} is {t.status}")
        turn = t.next_turn
        self.threads[thread_id] = t.model_copy(update={"next_turn": turn + 1})
        return turn

    async def add_message(self, msg: MessageRecord) -> None:
        if any(
            (m.thread_id, m.turn, m.role) == (msg.thread_id, msg.turn, msg.role) for m in self.msgs
        ):
            raise ValueError("duplicate message for thread, turn and role")
        self.msgs.append(msg)

    async def messages(
        self, claims: Claims, thread_id: uuid.UUID, *, last: int = 20
    ) -> list[MessageRecord]:
        self._own(claims, thread_id)
        return [m for m in self.msgs if m.thread_id == thread_id][-last:]

    async def set_status(self, thread_id: uuid.UUID, status: ThreadStatus) -> None:
        self.threads[thread_id] = self.threads[thread_id].model_copy(update={"status": status})

    async def set_summary(self, thread_id: uuid.UUID, summary: str) -> None:
        self.threads[thread_id] = self.threads[thread_id].model_copy(update={"summary": summary})


class MemoryApprovalStore:
    def __init__(self, clock: Clock = utcnow) -> None:
        self.items: dict[uuid.UUID, ops.ApprovalRecord] = {}
        self.clock = clock

    async def create(self, approval: ops.ApprovalRecord) -> ops.ApprovalRecord:
        self.items[approval.id] = approval
        return approval

    async def get(self, approval_id: uuid.UUID) -> ops.ApprovalRecord | None:
        return self.items.get(approval_id)

    async def decide(
        self,
        approval_id: uuid.UUID,
        *,
        decision: Literal["approved", "rejected"],
        decided_by: str,
        args: Mapping[str, Any] | None = None,
    ) -> ops.ApprovalRecord | None:
        a = self.items.get(approval_id)
        if a is None or a.status != "pending" or a.expires_at <= self.clock():
            return None
        update: dict[str, Any] = {
            "status": decision,
            "decided_by": decided_by,
            "decided_at": self.clock(),
        }
        if args is not None:
            update["args"] = dict(args)
            update["args_hash"] = ops.args_hash(a.tool, args)
        self.items[approval_id] = a.model_copy(update=update)
        return self.items[approval_id]

    async def claim_execution(self, approval_id: uuid.UUID) -> ops.ApprovalRecord | None:
        a = self.items.get(approval_id)
        if (
            a is None
            or a.status != "approved"
            or a.execution_claimed_at is not None
            or a.expires_at <= self.clock()
        ):
            return None
        self.items[approval_id] = a.model_copy(update={"execution_claimed_at": self.clock()})
        return self.items[approval_id]

    async def mark_executed(self, approval_id: uuid.UUID) -> None:
        a = self.items[approval_id]
        self.items[approval_id] = a.model_copy(update={"status": "executed"})

    async def expire_due(self) -> list[ops.ApprovalRecord]:
        now, out = self.clock(), []
        for k, a in self.items.items():
            unclaimed = a.status == "approved" and a.execution_claimed_at is None
            if (a.status == "pending" or unclaimed) and a.expires_at <= now:
                self.items[k] = a.model_copy(update={"status": "expired"})
                out.append(self.items[k])
        return out


def new_approval(
    thread_id: uuid.UUID,
    turn: int,
    tool: str,
    args: Mapping[str, Any],
    *,
    policy: ops.NamespacePolicy,
    requested_by: str,
    ttl: timedelta = timedelta(minutes=15),
    now: datetime | None = None,
) -> ops.ApprovalRecord:
    """An approval record for a write, built with the shared rules (tests and fakes)."""
    parsed = ops.parse_args(tool, args)
    ns, target = ops.target_of(tool, parsed)
    level = ops.required_approval(tool, parsed, policy, route="action")
    if level == "none":
        raise ValueError(f"{tool} needs no approval")
    t0 = now or utcnow()
    return ops.ApprovalRecord(
        id=uuid.uuid4(), thread_id=thread_id, turn=turn, tool=tool,
        args=ops.normalize_args(tool, args), args_hash=ops.args_hash(tool, args),
        namespace=ns or "", target=target or "", risk=ops.risk_of(tool, parsed, policy),
        required_scope=level, expected_effect=ops.expected_effect(tool, parsed),
        status="pending", requested_by=requested_by, expires_at=t0 + ttl, created_at=t0,
    )  # fmt: skip


@dataclass
class FakeOpsClient:
    """Records every call and its token; reads answer from `reads`, writes succeed after
    `polls_until_done` polls. Tests assert on `calls` (e.g. that no user token is forwarded)."""

    reads: dict[str, dict[str, Any]] = field(default_factory=dict)
    polls_until_done: int = 1
    fail_writes: bool = False
    calls: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)
    ops_by_id: dict[uuid.UUID, ops.OperationRecord] = field(default_factory=dict)
    _polls: dict[uuid.UUID, int] = field(default_factory=dict)

    async def read(self, tool: str, args: Mapping[str, Any], token: str) -> dict[str, Any]:
        self.calls.append((tool, dict(args), token))
        return self.reads.get(tool, {"items": []})

    def _op(self, tool: str, args: Mapping[str, Any], kind: ops.ToolKind) -> ops.OperationRecord:
        parsed = ops.parse_args(tool, args)
        ns, target = ops.target_of(tool, parsed)
        rec = ops.OperationRecord(
            id=uuid.uuid4(), idempotency_key=ops.idempotency_key(None, None, tool, args),
            kind=kind, tool=tool, args=ops.normalize_args(tool, args), namespace=ns,
            target=target, subject="fake", scopes=["operator"], status="running",
        )  # fmt: skip
        self.ops_by_id[rec.id] = rec
        return rec

    async def write(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        self.calls.append((tool, dict(args), token))
        return self._op(tool, args, "write")

    async def escalate(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        self.calls.append((tool, dict(args), token))
        rec = self._op(tool, args, "escalation").model_copy(
            update={"status": "succeeded", "result": {"incident_id": str(uuid.uuid4())}}
        )
        self.ops_by_id[rec.id] = rec
        return rec

    async def operation(self, operation_id: uuid.UUID, token: str) -> ops.OperationRecord:
        n = self._polls[operation_id] = self._polls.get(operation_id, 0) + 1
        rec = self.ops_by_id[operation_id]
        if rec.status == "running" and n >= self.polls_until_done:
            status: ops.OperationStatus = "failed" if self.fail_writes else "succeeded"
            rec = rec.model_copy(update={"status": status, "result": {"ok": not self.fail_writes}})
            self.ops_by_id[operation_id] = rec
        return rec


class MemoryAnswerCache:
    """Exact-match only, keyed by the caller's groups: a fake never shares across groups."""

    def __init__(self) -> None:
        self.items: dict[tuple[tuple[str, ...], int, str, str], CachedAnswer] = {}

    @staticmethod
    def _key(
        question: str, claims: Claims, cv: int, pv: str
    ) -> tuple[tuple[str, ...], int, str, str]:
        return tuple(sorted(claims.groups)), cv, pv, " ".join(question.lower().split())

    async def lookup(
        self, question: str, *, claims: Claims, corpus_version_id: int, prompt_version: str
    ) -> CachedAnswer | None:
        return self.items.get(self._key(question, claims, corpus_version_id, prompt_version))

    async def store(
        self, answer: CachedAnswer, *, claims: Claims, corpus_version_id: int, prompt_version: str
    ) -> None:
        self.items[self._key(answer.question, claims, corpus_version_id, prompt_version)] = answer


@dataclass
class FakeBudgetGuard:
    """Per-user daily allowance in memory; None means unlimited."""

    daily_tokens: int | None = None
    used: dict[str, int] = field(default_factory=dict)

    async def check(self, claims: Claims, planned_tokens: int) -> None:
        if (
            self.daily_tokens is not None
            and self.used.get(claims.sub, 0) + planned_tokens > self.daily_tokens
        ):
            raise BudgetExceeded(f"{claims.sub}: daily token budget reached")

    async def record(self, claims: Claims, tokens: int) -> None:
        self.used[claims.sub] = self.used.get(claims.sub, 0) + tokens


@dataclass
class ScriptedModelRouter:
    """Replies per role, in order; an exhausted role repeats its last reply."""

    replies: dict[str, list[str]] = field(default_factory=dict)
    calls: list[tuple[str, list[Message]]] = field(default_factory=list)

    async def complete(
        self,
        role: ModelRole,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        self.calls.append((role, list(messages)))
        queue = self.replies.get(role) or [""]
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return ChatResult(
            text=text, model=f"fake:{role}", prompt_tokens=10, completion_tokens=5, latency_ms=1.0
        )

    def model_for(self, role: ModelRole) -> str:
        return f"fake:{role}"


@dataclass
class MemoryOutbox:
    items: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)

    async def enqueue(self, kind: str, dedupe_key: str, payload: Mapping[str, Any]) -> None:
        self.items.setdefault(dedupe_key, (kind, dict(payload)))


@dataclass
class FakeChatAgent:
    """A scripted agent for API and UI tests: a message containing "restart" proposes a write
    and pauses for approval (creating the approval in `approvals`); anything else answers."""

    approvals: MemoryApprovalStore
    policy: ops.NamespacePolicy = field(
        default_factory=lambda: ops.NamespacePolicy(allowed=frozenset({"shop"}))
    )

    async def stream_turn(
        self, claims: Claims, thread_id: uuid.UUID, turn: int, message: str
    ) -> AsyncIterator[ev.AgentEvent]:
        yield ev.TurnEvent(thread_id=thread_id, turn=turn)
        if "restart" in message.lower():
            yield ev.RouteEvent(route="action", standalone_question=message)
            args = {"namespace": "shop", "deployment": "checkout"}
            yield ev.ToolProposedEvent(tool="restart_deployment", args=args, kind="write")
            a = await self.approvals.create(
                new_approval(
                    thread_id,
                    turn,
                    "restart_deployment",
                    args,
                    policy=self.policy,
                    requested_by=claims.sub,
                    now=self.approvals.clock(),
                )
            )
            yield ev.ApprovalRequiredEvent(
                approval_id=a.id, tool=a.tool, args=a.args, namespace=a.namespace, target=a.target,
                risk=a.risk, required_scope=a.required_scope, expected_effect=a.expected_effect,
                expires_at=a.expires_at,
            )  # fmt: skip
            return
        yield ev.RouteEvent(route="simple", standalone_question=message)
        yield ev.TokenEvent(text="Pods are the smallest deployable units.")
        yield ev.DoneEvent(thread_id=thread_id, turn=turn, route="simple",
                           answer="Pods are the smallest deployable units.")  # fmt: skip

    async def resume(
        self, claims: Claims, thread_id: uuid.UUID, approval: ops.ApprovalRecord
    ) -> AsyncIterator[ev.AgentEvent]:
        if approval.status != "approved":
            yield ev.DoneEvent(thread_id=thread_id, turn=approval.turn, route="action",
                               answer=f"Not run: the approval was {approval.status}.")  # fmt: skip
            return
        op_id = uuid.uuid4()
        yield ev.OperationEvent(operation_id=op_id, tool=approval.tool, status="running")
        yield ev.OperationEvent(operation_id=op_id, tool=approval.tool, status="succeeded")
        yield ev.DoneEvent(thread_id=thread_id, turn=approval.turn, route="action",
                           answer=f"Done: {approval.expected_effect}")  # fmt: skip
