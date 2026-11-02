"""V06 (adversarial): approvals. An approval admits exactly one call (its tool, its normalized
args, its thread and turn, its requester) while it is approved and live, and stops admitting the
moment core withdraws it, including after admission and before the task claims the row. An
admin-level call needs an approval that core recorded as admin-decided (O-2), never a token
claim."""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.exceptions import ToolError

from sre_shared.contracts.ops import McpTokenClaims, verify_mcp_token

if TYPE_CHECKING:
    from conftest import Rig  # type: ignore[attr-defined, unused-ignore]
    from sre_mcp_ops.backends.simulator import SimulatorBackend

SECRET = "mcp-test-secret-0123456789abcdef0123456789"
WEB = {"namespace": "shop", "deployment": "web"}
PAY = {"namespace": "payments", "deployment": "api"}


async def refused(rig: Rig, token: str, tool: str, args: dict[str, Any], why: str) -> None:
    async with rig.client(token) as c:
        with pytest.raises(ToolError, match=why):
            await c.call_tool(tool, args)


@pytest.mark.parametrize("status", ["pending", "rejected", "expired", "executed"])
async def test_only_an_approved_approval_admits(
    rig: Rig, sim: SimulatorBackend, status: str
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB, status=status)  # not yet lapsed
    await refused(
        rig, rig.token(thread_id=t, approval_id=appr), "restart_deployment", WEB,
        f"approval is {status}",
    )  # fmt: skip
    assert sim.mutation_log() == []


@pytest.mark.xfail(
    strict=True,
    reason="V06-1: the refusal row copies the token's approval_id into operations.approval_id, "
    "a foreign key to approvals; an id with no row (a deleted thread cascades its approvals) "
    "fails the audit insert, so the call is refused as 'could not process' with no audit row",
)
async def test_an_approval_id_that_does_not_exist_is_refused_and_audited(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    token = rig.token(thread_id=t, approval_id=uuid.uuid4())
    try:
        await refused(rig, token, "restart_deployment", WEB, "approval not found")
    finally:
        assert sim.mutation_log() == []  # whatever the audit does, nothing ran
    assert len(await rig.rows("status = 'refused'")) == 1


@pytest.mark.parametrize("where", ["another thread", "another turn"])
async def test_an_approval_cannot_buy_a_second_idempotency_key(
    rig: Rig, sim: SimulatorBackend, where: str
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    done = await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    assert done["status"] == "succeeded"
    # Thread A's token carrying thread B's approval (or the same approval in a later turn): the
    # idempotency key would be new, so only the approval check stands between it and a rerun.
    other = (
        rig.token(thread_id=await rig.thread(), approval_id=appr)
        if where == "another thread"
        else rig.token(thread_id=t, turn=2, approval_id=appr)
    )
    await refused(rig, other, "restart_deployment", WEB, "another turn")
    assert len(sim.mutation_log()) == 1


async def test_a_thread_a_token_cannot_use_a_fresh_thread_b_approval(
    rig: Rig, sim: SimulatorBackend
) -> None:
    a, b = await rig.thread(), await rig.thread()
    appr_b = await rig.approval(b, "restart_deployment", WEB)
    await refused(
        rig, rig.token(thread_id=a, approval_id=appr_b), "restart_deployment", WEB, "another turn"
    )
    assert sim.mutation_log() == []


async def test_an_admin_scope_approval_without_a_recorded_decider_is_refused(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", PAY, required_scope="admin", decided_by=None)
    token = rig.token(thread_id=t, approval_id=appr, namespaces=("payments",))
    await refused(rig, token, "restart_deployment", PAY, "admin's approval")
    assert sim.mutation_log() == []


async def test_an_admin_scope_in_the_token_does_not_replace_an_admin_approval(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", PAY, required_scope="operator")
    token = rig.token(
        thread_id=t, approval_id=appr, namespaces=("payments",),
        scopes=("viewer", "operator", "admin"),
    )  # fmt: skip
    await refused(rig, token, "restart_deployment", PAY, "admin's approval")
    assert sim.mutation_log() == []


async def test_spellings_that_normalize_to_the_approved_args_act_as_the_approved_args(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "scale_deployment", {**WEB, "replicas": 3})
    for replicas in ("3", 3.0, 3):
        # Refusing an odd spelling is fine; acting differently on it is not.
        with contextlib.suppress(ToolError):
            await rig.write(
                "scale_deployment", {**WEB, "replicas": replicas},
                rig.token(thread_id=t, approval_id=appr),
            )  # fmt: skip
    log = sim.mutation_log()
    assert len(log) == 1 and log[0]["args"]["replicas"] == 3
    assert len(await rig.rows("kind = 'write' and status <> 'refused'")) == 1


async def test_page_oncall_spelled_two_ways_pages_once(rig: Rig, sim: SimulatorBackend) -> None:
    t = await rig.thread()
    inc = await rig.write(
        "create_incident", {"severity": "sev2", "title": "down", "summary": "crash loop"},
        rig.token(thread_id=t),
    )  # fmt: skip
    incident_id = inc["result"]["incident_id"]
    await rig.write("page_oncall", {"incident_id": incident_id}, rig.token(thread_id=t))
    again = await rig.write(
        "page_oncall",
        {"incident_id": incident_id.upper(), "rotation": "sre-primary"},
        rig.token(thread_id=t),
    )
    assert again["replayed"] is True
    cur = await rig.db.execute("select count(*) as n from public.pages")
    assert (await cur.fetchone())["n"] == 1  # type: ignore[index, unused-ignore]
    assert [e["tool"] for e in sim.mutation_log()] == ["create_incident", "page_oncall"]


async def admitted(rig: Rig) -> tuple[McpTokenClaims, uuid.UUID]:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    claims = verify_mcp_token(rig.token(thread_id=t, approval_id=appr, max_calls=10), SECRET)
    await rig.service.admit(claims, "restart_deployment", dict(WEB))
    return claims, appr


@pytest.mark.parametrize(
    "change",
    ["status = 'rejected'", "status = 'executed'", "status = 'expired'",
     "expires_at = now() - interval '1 second'"],
)  # fmt: skip
async def test_an_approval_withdrawn_between_admission_and_the_claim_never_runs(
    rig: Rig, sim: SimulatorBackend, change: str
) -> None:
    claims, appr = await admitted(rig)
    await rig.db.execute(f"update public.approvals set {change} where id = %s", [appr])
    out = await rig.service.execute(claims, "restart_deployment", dict(WEB), "late-task")
    assert out["status"] == "refused" and sim.mutation_log() == []


async def test_a_retry_after_core_marks_the_approval_executed_gets_the_stored_result(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    first = await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    await rig.db.execute("update public.approvals set status = 'executed' where id = %s", [appr])
    again = await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    assert again["id"] == first["id"] and again["replayed"] is True
    assert again["status"] == "succeeded" and len(sim.mutation_log()) == 1
