"""The Phase 3 gate in miniature: zero duplicate writes under replay, a dropped first reply,
concurrent identical calls, a redelivered or killed task and a double approval. Duplicates are
counted from the simulator's mutation log: backend applications per idempotency key beyond one,
counting no-op applications too."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.exceptions import ToolError
from fastmcp_tasks import call_tool_task

from sre_mcp_ops.backends.simulator import SimulatorBackend
from sre_mcp_ops.reconcile import reconcile
from sre_shared.contracts.ops import McpTokenClaims, verify_mcp_token

if TYPE_CHECKING:
    from conftest import Rig

SECRET = "mcp-test-secret-0123456789abcdef0123456789"
WEB = {"namespace": "shop", "deployment": "web"}


def duplicates(sim: SimulatorBackend) -> int:
    per_key = Counter(e["idempotency_key"] for e in sim.mutation_log())
    return sum(n - 1 for n in per_key.values())


async def approved(
    rig: Rig, tool: str = "restart_deployment", args: dict[str, Any] = WEB
) -> tuple[Any, Any]:
    t = await rig.thread()
    return t, await rig.approval(t, tool, args)


async def test_the_metric_catches_a_deliberate_duplicate(sim: SimulatorBackend) -> None:
    await sim.restart("shop", "web", key="k", at="t")
    await sim.restart("shop", "web", key="k", at="t")  # a no-op, but still an application
    assert len(sim.mutation_log()) == 2 and duplicates(sim) == 1


async def test_replay_returns_the_same_operation_and_changes_state_once(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t, appr = await approved(rig)
    first = await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    second = await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    assert first["id"] == second["id"]
    assert first["replayed"] is False and second["replayed"] is True
    assert second["result"] == first["result"] and second["status"] == "succeeded"
    assert len(sim.mutation_log()) == 1 and duplicates(sim) == 0
    assert len(await rig.rows("kind = 'write'")) == 1


async def test_first_reply_dropped_then_the_client_retries(rig: Rig, sim: SimulatorBackend) -> None:
    t, appr = await approved(rig)
    async with rig.client(rig.token(thread_id=t, approval_id=appr)) as c:
        await call_tool_task(c, "restart_deployment", WEB)  # accepted; the client never reads on
    for _ in range(100):  # the write finishes without anyone watching
        rows = await rig.rows("kind = 'write' and status = 'succeeded'")
        if rows:
            break
        await asyncio.sleep(0.05)
    retry = await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    assert retry["replayed"] is True and retry["id"] == str(rows[0]["id"])
    assert len(sim.mutation_log()) == 1


async def test_two_concurrent_identical_calls(rig: Rig, sim: SimulatorBackend) -> None:
    t, appr = await approved(rig)
    a, b = await asyncio.gather(
        *(rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
          for _ in range(2))
    )  # fmt: skip
    assert a["id"] == b["id"]
    assert sorted([a["replayed"], b["replayed"]]) == [False, True]
    assert a["status"] == b["status"] == "succeeded"
    assert len(sim.mutation_log()) == 1 and len(await rig.rows("kind = 'write'")) == 1


async def admitted(rig: Rig) -> tuple[McpTokenClaims, dict[str, Any]]:
    t, appr = await approved(rig)
    claims = verify_mcp_token(rig.token(thread_id=t, approval_id=appr, max_calls=10), SECRET)
    await rig.service.admit(claims, "restart_deployment", dict(WEB))
    return claims, dict(WEB)


async def test_a_redelivered_task_body_changes_state_once(rig: Rig, sim: SimulatorBackend) -> None:
    claims, args = await admitted(rig)
    views = await asyncio.gather(
        *(rig.service.execute(claims, "restart_deployment", args, f"task-{i}") for i in range(3))
    )
    assert len({v["id"] for v in views}) == 1
    assert [v["status"] for v in views] == ["succeeded"] * 3
    again = await rig.service.execute(claims, "restart_deployment", args, "task-late")
    assert again["replayed"] is True
    assert len(sim.mutation_log()) == 1


async def test_a_worker_killed_after_the_change_is_settled_by_the_reconciler(
    rig: Rig, sim: SimulatorBackend
) -> None:
    claims, args = await admitted(rig)
    (row,) = await rig.rows("kind = 'write'")
    await rig.store.claim(row["id"], "dead-task")  # claimed ... and the process dies:
    await sim.restart("shop", "web", key=row["idempotency_key"], at=row["created_at"].isoformat())
    assert await rig.rows("status = 'running'")
    assert await reconcile(rig.store, sim, min_age_s=0) == [
        {"id": str(row["id"]), "status": "succeeded"}
    ]
    # The client's retry now gets the stored result, and nothing runs again.
    done = await rig.service.execute(claims, "restart_deployment", args, "retry")
    assert done["status"] == "succeeded" and done["replayed"] is True
    assert len(sim.mutation_log()) == 1


async def test_a_worker_killed_before_the_change_ends_failed_and_never_reruns(
    rig: Rig, sim: SimulatorBackend
) -> None:
    claims, args = await admitted(rig)
    (row,) = await rig.rows("kind = 'write'")
    await rig.store.claim(row["id"], "dead-task")
    await reconcile(rig.store, sim, min_age_s=0)
    (after,) = await rig.rows("kind = 'write'")
    assert after["status"] == "failed" and "interrupted" in after["error"]
    again = await rig.service.execute(claims, "restart_deployment", args, "retry")
    assert again["status"] == "failed" and len(sim.mutation_log()) == 0


async def test_double_approval_click_runs_once(rig: Rig, sim: SimulatorBackend) -> None:
    t, appr = await approved(rig)
    one = await rig.write(
        "restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr, max_calls=1)
    )
    two = await rig.write(
        "restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr, max_calls=1)
    )
    assert one["id"] == two["id"] and len(sim.mutation_log()) == 1
    # The same approval cannot buy a second change in another turn.
    async with rig.client(rig.token(thread_id=t, approval_id=appr, turn=2)) as c:
        with pytest.raises(ToolError, match="another turn"):
            await c.call_tool("restart_deployment", WEB)
    assert len(sim.mutation_log()) == 1


async def test_an_approval_that_lapses_before_execution_never_runs(
    rig: Rig, sim: SimulatorBackend
) -> None:
    claims, args = await admitted(rig)
    await rig.db.execute("update public.approvals set expires_at = now() - interval '1 second'")
    out = await rig.service.execute(claims, "restart_deployment", args, "late")
    assert out["status"] == "refused" and "expired" in out["error"]
    assert len(sim.mutation_log()) == 0
    # The refusal released the key, so a fresh approval for the same call can still run it.
    appr2 = await rig.approval(claims.thread_id, "restart_deployment", WEB)  # type: ignore[arg-type]
    retry = await rig.write(
        "restart_deployment", WEB, rig.token(thread_id=claims.thread_id, approval_id=appr2)
    )
    assert retry["status"] == "succeeded" and len(sim.mutation_log()) == 1


async def test_escalation_is_idempotent_too(rig: Rig, sim: SimulatorBackend) -> None:
    t = await rig.thread()
    args = {"severity": "sev2", "title": "checkout down", "summary": "crash loop"}
    one = await rig.write(
        "create_incident", args, rig.token(thread_id=t, tools=("create_incident", "page_oncall"))
    )
    two = await rig.write(
        "create_incident", args, rig.token(thread_id=t, tools=("create_incident", "page_oncall"))
    )
    assert one["id"] == two["id"] and two["replayed"] is True
    cur = await rig.db.execute("select count(*) as n from public.incidents")
    assert (await cur.fetchone())["n"] == 1  # type: ignore[index]
    assert [e["tool"] for e in sim.mutation_log()] == ["create_incident"]
