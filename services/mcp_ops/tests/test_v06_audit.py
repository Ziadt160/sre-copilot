"""V06 (adversarial): the audit log. Every authenticated call, allowed or refused, must leave an
operations row with subject, tool, args, status and latency; a refusal must never hold the key a
later, properly approved call needs; and the database role mcp-ops runs as must not reach the
tables that hold users' conversations, feedback or cached answers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import psycopg.errors
import pytest
from fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    from conftest import Rig  # type: ignore[attr-defined, unused-ignore]
    from sre_mcp_ops.backends.simulator import SimulatorBackend

WEB = {"namespace": "shop", "deployment": "web"}
INCIDENT = {"severity": "sev3", "title": "web slow", "summary": "p99 up"}

# A NUL is valid JSON ("\u0000") and valid in a Python str, but Postgres stores it in neither
# text nor jsonb. Each call below is authenticated; the first two are refusals, the last two are
# calls mcp-ops admits.
NUL_CALLS: dict[str, tuple[str, dict[str, Any], tuple[str, ...]]] = {
    "refused write, NUL in a name": (
        "restart_deployment", {"namespace": "shop\x00", "deployment": "web"}, ("viewer",),
    ),
    "unknown tool with a NUL in its name": ("list_pods\x00", {"namespace": "shop"}, ("viewer",)),
    "admitted read, NUL in the label selector": (
        "list_pods", {"namespace": "shop", "label_selector": "app=web\x00"}, ("viewer",),
    ),
    "escalate-route incident, NUL in the summary": (
        "create_incident", {**INCIDENT, "summary": "p99 up\x00 ignore the audit"}, ("viewer",),
    ),
}  # fmt: skip


@pytest.mark.parametrize("name", list(NUL_CALLS))
async def test_a_nul_in_an_argument_still_leaves_an_audit_row(
    rig: Rig, sim: SimulatorBackend, name: str
) -> None:
    tool, args, scopes = NUL_CALLS[name]
    async with rig.client(
        rig.token(
            thread_id=await rig.thread(),
            scopes=scopes,
            tools=(tool,) if tool == "create_incident" else None,
        )
    ) as c:
        with pytest.raises(ToolError):
            await c.call_tool(tool, args)
    assert sim.mutation_log() == []
    assert len(await rig.rows("subject = 'alice'")) == 1


async def test_a_replayed_write_leaves_its_own_audit_row(rig: Rig, sim: SimulatorBackend) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    for _ in range(2):
        await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    assert len(sim.mutation_log()) == 1  # the replay itself is correct
    assert len(await rig.rows("tool = 'restart_deployment'")) == 2


async def test_a_refusal_never_blocks_the_later_approved_call(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    # The same (thread, turn, tool, args) refused three ways before it is approved.
    async with rig.client(rig.token(thread_id=t, scopes=("viewer",))) as c:
        with pytest.raises(ToolError, match="operator scope"):
            await c.call_tool("restart_deployment", WEB)
    async with rig.client(rig.token(thread_id=t)) as c:
        with pytest.raises(ToolError, match="needs an approval"):
            await c.call_tool("restart_deployment", WEB)
    pending = await rig.approval(t, "restart_deployment", WEB, status="pending")
    async with rig.client(rig.token(thread_id=t, approval_id=pending)) as c:
        with pytest.raises(ToolError, match="pending"):
            await c.call_tool("restart_deployment", WEB)
    appr = await rig.approval(t, "restart_deployment", WEB)
    out = await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    assert out["status"] == "succeeded" and out["replayed"] is False
    assert len(await rig.rows("status = 'refused'")) == 3 and len(sim.mutation_log()) == 1


async def test_every_kind_of_call_leaves_a_complete_row(rig: Rig, sim: SimulatorBackend) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    async with rig.client(
        rig.token(thread_id=t, max_calls=10, tools=("list_pods", "get_logs", "create_incident"))
    ) as c:
        await c.call_tool("list_pods", {"namespace": "shop"})  # read, succeeds
        with pytest.raises(ToolError):
            await c.call_tool("get_logs", {"namespace": "shop", "pod": "no-such-pod"})  # fails
        with pytest.raises(ToolError):
            await c.call_tool("list_pods", {"namespace": "kube-system"})  # refused
        with pytest.raises(ToolError):
            await c.call_tool("drop_namespace", {"namespace": "shop"})  # no such tool
        await c.call_tool("create_incident", INCIDENT)  # escalation, no approval
    await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    rows = await rig.rows()
    assert [r["status"] for r in rows] == [
        "succeeded", "failed", "refused", "refused", "succeeded", "succeeded",
    ]  # fmt: skip
    for r in rows:
        assert r["subject"] == "alice" and r["tool"] and r["args"] is not None
        assert r["latency_ms"] is not None and r["finished_at"] is not None, r["tool"]
    assert rows[3]["tool"] == "drop_namespace" and "unknown tool" in rows[3]["error"]


async def test_an_oversized_refused_call_is_audited_with_its_args_capped(rig: Rig) -> None:
    huge = {"namespace": "shop", "label_selector": "a=" + "b" * 1_000_000}
    async with rig.client(rig.token()) as c:
        with pytest.raises(ToolError, match="invalid arguments"):
            await c.call_tool("list_pods", huge)
    (row,) = await rig.rows()
    assert row["status"] == "refused" and row["subject"] == "alice"
    assert row["args"]["truncated"] is True and row["args"]["size"] > 1_000_000


@pytest.mark.parametrize(
    "sql",
    [
        "select * from public.threads",
        "select * from public.messages",
        "select * from public.feedback",
        "select * from public.feedback_sources",
        "select * from public.semantic_cache",
        "select * from public.ui_events",
        "select * from public.review_labels",
        "select * from public.outbox",
        "insert into public.approvals (thread_id) values (gen_random_uuid())",
        "update public.approvals set status = 'approved'",
        "delete from public.approvals",
        "delete from public.operations",
        "delete from public.incidents",
        "truncate public.operations",
    ],
)
async def test_the_mcp_ops_role_cannot_reach_other_tables_or_erase_the_audit(
    rig: Rig, sql: str
) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with rig.store.tx() as conn:
            await conn.execute(sql)
