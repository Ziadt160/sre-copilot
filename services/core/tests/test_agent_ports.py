"""The Wave 0 fakes satisfy the ports and keep the rules real implementations must keep."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from sre_core import ports
from sre_core.agent import testing as fk
from sre_shared.contracts import agent as ev
from sre_shared.contracts import ops
from sre_shared.models import Claims

ALICE = Claims(sub="alice", scopes=["viewer", "operator"], namespaces=["shop"])
BOB = Claims(sub="bob")
POLICY = ops.NamespacePolicy(allowed=frozenset({"shop"}))


def test_every_fake_satisfies_its_port() -> None:
    approvals = fk.MemoryApprovalStore()
    pairs = [
        (fk.MemoryThreadStore(), ports.ThreadStore),
        (approvals, ports.ApprovalStore),
        (fk.FakeOpsClient(), ports.OpsClient),
        (fk.MemoryAnswerCache(), ports.AnswerCache),
        (fk.FakeBudgetGuard(), ports.BudgetGuard),
        (fk.ScriptedModelRouter(), ports.ModelRouter),
        (fk.MemoryOutbox(), ports.Outbox),
        (fk.FakeChatAgent(approvals), ports.ChatAgent),
    ]
    for impl, port in pairs:
        assert isinstance(impl, port), (impl, port)


async def test_turns_are_claimed_in_order_and_refused_while_an_approval_waits() -> None:
    store = fk.MemoryThreadStore()
    t = await store.create(ALICE)
    assert [await store.claim_turn(ALICE, t.id) for _ in range(3)] == [1, 2, 3]
    with pytest.raises(ports.ThreadNotFound):
        await store.claim_turn(BOB, t.id)  # another user's thread does not exist for them
    await store.set_status(t.id, "awaiting_approval")
    with pytest.raises(ports.ThreadBusy):
        await store.claim_turn(ALICE, t.id)


async def test_an_approval_is_decided_once_and_never_after_it_expires() -> None:
    now = datetime(2026, 11, 2, 12, tzinfo=UTC)
    clock = [now]
    store = fk.MemoryApprovalStore(clock=lambda: clock[0])
    args = {"namespace": "shop", "deployment": "checkout"}
    a = await store.create(
        fk.new_approval(
            uuid.uuid4(),
            1,
            "restart_deployment",
            args,
            policy=POLICY,
            requested_by="alice",
            now=now,
        )
    )
    assert a.required_scope == "operator" and a.expires_at == now + timedelta(minutes=15)
    assert (await store.decide(a.id, decision="approved", decided_by="alice")) is not None
    assert (await store.decide(a.id, decision="approved", decided_by="alice")) is None  # 409
    late = await store.create(
        fk.new_approval(
            uuid.uuid4(),
            1,
            "restart_deployment",
            args,
            policy=POLICY,
            requested_by="alice",
            now=now,
        )
    )
    clock[0] = now + timedelta(minutes=16)
    assert (await store.decide(late.id, decision="approved", decided_by="alice")) is None
    assert [x.id for x in await store.expire_due()] == [late.id]


async def test_the_fake_agent_pauses_for_approval_and_resumes_from_the_record() -> None:
    approvals = fk.MemoryApprovalStore()
    agent = fk.FakeChatAgent(approvals)
    thread = uuid.uuid4()
    events = [e async for e in agent.stream_turn(ALICE, thread, 1, "please restart checkout")]
    assert [e.event for e in events] == ["turn", "route", "tool_proposed", "approval_required"]
    pending = events[-1]
    assert isinstance(pending, ev.ApprovalRequiredEvent)
    decided = await approvals.decide(pending.approval_id, decision="approved", decided_by="alice")
    assert decided is not None
    done = [e async for e in agent.resume(ALICE, thread, decided)]
    assert [e.event for e in done] == ["operation", "operation", "done"]
