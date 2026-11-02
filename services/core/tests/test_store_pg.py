"""Postgres stores: turn claims, the approval gate's conditional updates, the outbox, checkpoints."""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import pytest
from psycopg_pool import AsyncConnectionPool

from sre_core.agent.ports import MessageRecord, ThreadBusy, ThreadNotFound
from sre_core.agent.testing import new_approval, utcnow
from sre_core.store import outbox as ob
from sre_core.store.approvals import PgApprovalStore
from sre_core.store.checkpoints import make_checkpoint_pool, make_checkpointer, setup_checkpointer
from sre_core.store.outbox import PgOutbox, register_outbox_handler
from sre_core.store.threads import PgThreadStore
from sre_shared.contracts import ops
from sre_shared.models import Claims

pytestmark = pytest.mark.integration

ALICE = Claims(sub="alice", scopes=["operator"], namespaces=["shop"])
BOB = Claims(sub="bob")
POLICY = ops.NamespacePolicy(allowed=frozenset({"shop"}))


async def test_turns_claimed_in_order_under_concurrency(db: object, pool: AsyncConnectionPool) -> None:
    threads = PgThreadStore(pool)
    t = await threads.create(ALICE, "x")
    turns = await asyncio.gather(*[threads.claim_turn(ALICE, t.id) for _ in range(8)])
    assert sorted(turns) == list(range(1, 9))
    got = await threads.get(ALICE, t.id)
    assert got is not None and got.next_turn == 9


async def test_busy_thread_refused_and_other_user_sees_nothing(
    db: object, pool: AsyncConnectionPool
) -> None:
    threads = PgThreadStore(pool)
    t = await threads.create(ALICE)
    await threads.set_status(t.id, "awaiting_approval")
    with pytest.raises(ThreadBusy):
        await threads.claim_turn(ALICE, t.id)
    await threads.set_status(t.id, "active")
    await threads.claim_turn(ALICE, t.id)
    await threads.add_message(MessageRecord(thread_id=t.id, turn=1, role="user", content="hi"))
    assert await threads.get(BOB, t.id) is None
    assert await threads.list_threads(BOB) == []
    with pytest.raises(ThreadNotFound):
        await threads.claim_turn(BOB, t.id)
    with pytest.raises(ThreadNotFound):
        await threads.messages(BOB, t.id)
    with pytest.raises(ThreadNotFound):
        await threads.claim_message(BOB, t.id, "hello")
    assert [m.content for m in await threads.messages(ALICE, t.id)] == ["hi"]
    with pytest.raises(ThreadNotFound):
        await threads.claim_turn(Claims(sub="a", role="anon"), t.id)


async def test_client_message_id_returns_the_same_turn(db: object, pool: AsyncConnectionPool) -> None:
    threads = PgThreadStore(pool)
    a, b = await asyncio.gather(
        *[threads.claim_message(ALICE, None, "hi", client_message_id="m1") for _ in range(2)]
    )
    assert (a.thread.id, a.turn) == (b.thread.id, b.turn) and {a.replay, b.replay} == {True, False}
    again = await threads.claim_message(ALICE, a.thread.id, "hi", client_message_id="m1")
    assert again.replay and again.turn == 1
    assert len(await threads.messages(ALICE, a.thread.id)) == 1
    nxt = await threads.claim_message(ALICE, a.thread.id, "more", client_message_id="m2")
    assert nxt.turn == 2 and not nxt.replay


def _approval(thread_id: uuid.UUID, ttl: timedelta = timedelta(minutes=15)) -> ops.ApprovalRecord:
    return new_approval(
        thread_id, 1, "restart_deployment", {"namespace": "shop", "deployment": "web"},
        policy=POLICY, requested_by="alice", ttl=ttl,
    )  # fmt: skip


async def test_decide_once_never_after_expiry(db: object, pool: AsyncConnectionPool) -> None:
    threads, approvals = PgThreadStore(pool), PgApprovalStore(pool)
    t = await threads.create(ALICE)
    a = await approvals.create(_approval(t.id))
    results = await asyncio.gather(
        *[approvals.decide(a.id, decision="approved", decided_by="alice") for _ in range(5)]
    )
    assert sum(r is not None for r in results) == 1
    assert (await approvals.get(a.id)).status == "approved"  # type: ignore[union-attr]
    await approvals.mark_executed(a.id)
    assert (await approvals.get(a.id)).status == "executed"  # type: ignore[union-attr]

    old = await approvals.create(_approval(t.id, ttl=timedelta(seconds=-1)))
    assert await approvals.decide(old.id, decision="approved", decided_by="alice") is None
    assert [x.id for x in await approvals.expire_due()] == [old.id]
    assert await approvals.expire_due() == []
    assert (await approvals.get(old.id)).status == "expired"  # type: ignore[union-attr]


async def test_edit_updates_args_hash_and_visibility_follows_rls(
    db: object, pool: AsyncConnectionPool
) -> None:
    threads, approvals = PgThreadStore(pool), PgApprovalStore(pool)
    t = await threads.create(ALICE)
    scale = new_approval(
        t.id, 1, "scale_deployment", {"namespace": "shop", "deployment": "web", "replicas": 2},
        policy=POLICY, requested_by="alice",
    )  # fmt: skip
    a = await approvals.create(scale)
    edited = {"namespace": "shop", "deployment": "web", "replicas": 3}
    d = await approvals.decide(a.id, decision="approved", decided_by="alice", args=edited)
    assert d is not None and d.args["replicas"] == 3
    assert d.args_hash == ops.args_hash("scale_deployment", edited)
    assert await approvals.get_visible(ALICE, a.id) is not None
    assert await approvals.get_visible(BOB, a.id) is None
    assert await approvals.list_visible(BOB) == []


async def test_outbox_dedupe_drain_and_retry(db: object, pool: AsyncConnectionPool) -> None:
    box = PgOutbox(pool)
    seen: list[dict[str, object]] = []
    calls = {"flaky": 0}

    async def ok(payload: object) -> None:
        seen.append(dict(payload))  # type: ignore[call-overload]

    async def flaky(payload: object) -> None:
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise RuntimeError("boom")

    register_outbox_handler("t13_ok", ok)
    register_outbox_handler("t13_flaky", flaky)
    await box.enqueue("t13_ok", "k1", {"n": 1})
    await box.enqueue("t13_ok", "k1", {"n": 2})  # same key: ignored
    await box.enqueue("t13_flaky", "k2", {})
    await box.enqueue("t13_unregistered", "k3", {})
    assert await box.drain() == 1 and seen == [{"n": 1}]
    assert await box.drain() == 1 and calls["flaky"] == 2  # retried on the next drain
    assert await box.drain() == 0
    async with pool.connection() as conn:
        cur = await conn.execute("select kind, status from public.outbox order by id")
        rows = {r["kind"]: r["status"] for r in await cur.fetchall()}
    assert rows == {"t13_ok": "done", "t13_flaky": "done", "t13_unregistered": "pending"}
    ob._HANDLERS.pop("t13_ok"), ob._HANDLERS.pop("t13_flaky")


async def test_checkpointer_setup_and_roundtrip(pg_dsn: str) -> None:
    await setup_checkpointer(pg_dsn)
    await setup_checkpointer(pg_dsn)  # idempotent
    cp = make_checkpoint_pool(pg_dsn, max_size=2)
    await cp.open()
    try:
        saver = make_checkpointer(cp)
        assert await saver.aget_tuple({"configurable": {"thread_id": "none", "checkpoint_ns": ""}}) is None
        async with cp.connection() as conn:
            cur = await conn.execute(
                "select count(*) as n from information_schema.tables "
                "where table_schema = 'agent_ckpt' and table_name = 'checkpoints'"
            )
            row = await cur.fetchone()
        assert row is not None and row["n"] == 1
    finally:
        await cp.close()


def test_now_is_timezone_aware() -> None:
    assert utcnow().tzinfo is not None
