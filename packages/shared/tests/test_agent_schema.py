"""Phase 3 schema against a real Postgres: owner-only threads, approval visibility, the
idempotency index and the mcp_ops role's narrow grants."""

from typing import Any

import psycopg
import pytest

from sre_shared.db import as_user
from sre_shared.models import Claims

pytestmark = pytest.mark.integration

ALICE = Claims(sub="alice", groups=["team-a"])
BOB = Claims(sub="bob")
ADMIN = Claims(sub="carol", scopes=["viewer", "operator", "admin"])


async def _thread(db: Any, owner: str) -> str:
    row = await (
        await db.execute("insert into public.threads (owner_sub) values (%s) returning id", [owner])
    ).fetchone()
    return str(row["id"])


async def _approval(db: Any, thread_id: str, scope: str) -> None:
    await db.execute(
        "insert into public.approvals (thread_id, turn, tool, args, args_hash, namespace,"
        " target, risk, required_scope, expected_effect, requested_by, expires_at)"
        " values (%s, 1, 'restart_deployment', '{}', 'h', 'shop', 'checkout', 'medium', %s,"
        " 'rolling restart', 'alice', now() + interval '15 minutes')",
        [thread_id, scope],
    )


async def _count(db: Any, claims: Claims, table: str) -> int:
    async with as_user(db, claims) as conn:
        row = await (await conn.execute(f"select count(*) as n from public.{table}")).fetchone()
    return int(row["n"])


async def test_threads_and_messages_are_private_to_their_owner(db: Any) -> None:
    t = await _thread(db, "alice")
    await db.execute(
        "insert into public.messages (thread_id, turn, role, content) values (%s, 1, 'user', 'hi')",
        [t],
    )
    assert await _count(db, ALICE, "threads") == 1 and await _count(db, ALICE, "messages") == 1
    assert await _count(db, BOB, "threads") == 0 and await _count(db, BOB, "messages") == 0
    # Bob cannot create a thread in Alice's name.
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with as_user(db, BOB) as conn:
            await conn.execute("insert into public.threads (owner_sub) values ('alice')")


async def test_approvals_are_seen_by_the_owner_and_by_admins_when_an_admin_must_decide(
    db: Any,
) -> None:
    t = await _thread(db, "alice")
    await _approval(db, t, "operator")
    await _approval(db, t, "admin")
    assert await _count(db, ALICE, "approvals") == 2
    assert await _count(db, ADMIN, "approvals") == 1  # only the one needing an admin
    assert await _count(db, BOB, "approvals") == 0
    # Users can read approvals but never decide them directly (core does, after its checks).
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with as_user(db, ALICE) as conn:
            await conn.execute("update public.approvals set status = 'approved'")


async def test_a_write_idempotency_key_is_unique_but_reads_repeat(db: Any) -> None:
    ins = (
        "insert into public.operations (idempotency_key, kind, tool, args, subject, scopes, status)"
        " values (%s, %s, %s, '{}', 'alice', '{operator}', 'pending')"
    )
    await db.execute(ins, ["k1", "read", "list_pods"])
    await db.execute(ins, ["k1", "read", "list_pods"])  # reads are logged, never deduplicated
    await db.execute(ins, ["k2", "write", "restart_deployment"])
    with pytest.raises(psycopg.errors.UniqueViolation):
        await db.execute(ins, ["k2", "write", "restart_deployment"])


async def test_mcp_ops_role_reaches_operations_only(db: Any) -> None:
    await _thread(db, "alice")
    async with db.transaction():
        await db.execute("set local role mcp_ops")
        await db.execute(
            "insert into public.operations (idempotency_key, kind, tool, args, subject, scopes,"
            " status) values ('k', 'write', 'scale_deployment', '{}', 'alice', '{operator}',"
            " 'pending')"
        )
        row = await (await db.execute("select count(*) as n from public.operations")).fetchone()
        assert row["n"] == 1
    for table in ("threads", "messages", "feedback", "semantic_cache", "outbox"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with db.transaction():
                await db.execute("set local role mcp_ops")
                await db.execute(f"select 1 from public.{table}")


async def test_users_cannot_reach_operations_or_checkpoints(db: Any) -> None:
    for table in ("operations", "outbox", "semantic_cache", "review_labels"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with as_user(db, ALICE) as conn:
                await conn.execute(f"select 1 from public.{table}")
    row = await (
        await db.execute("select has_schema_privilege('authenticated', 'agent_ckpt', 'usage') as u")
    ).fetchone()
    assert row["u"] is False


async def test_user_scopes_come_from_app_metadata(db: Any) -> None:
    async with as_user(db, ADMIN) as conn:
        row = await (await conn.execute("select sre.user_scopes() as s")).fetchone()
    assert row["s"] == ["viewer", "operator", "admin"]
    async with as_user(db, BOB) as conn:
        row = await (await conn.execute("select sre.user_scopes() as s")).fetchone()
    assert row["s"] == []
