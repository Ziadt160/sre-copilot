"""V06 (adversarial): idempotency under faults the T11 suite does not stage. Duplicates are counted
the way the gate counts them (O-2): backend applications per idempotency key beyond one, from the
simulator's mutation log, no-op applications included.

The interesting faults need a write that is still inside the backend call when something else
happens (a redelivery, a cancel, a second instance starting), so `GatedSim` holds every restart
until the test lets it go.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.exceptions import ToolError
from fastmcp_tasks import call_tool_task

from sre_mcp_ops.backends.simulator import SimulatorBackend
from sre_mcp_ops.reconcile import reconcile
from sre_mcp_ops.service import OpsService
from sre_shared.contracts.ops import McpTokenClaims, verify_mcp_token

if TYPE_CHECKING:
    from conftest import Rig  # type: ignore[attr-defined, unused-ignore]
    from sre_shared.config import McpOpsConfig

SECRET = "mcp-test-secret-0123456789abcdef0123456789"
WEB = {"namespace": "shop", "deployment": "web"}


def duplicates(sim: SimulatorBackend) -> int:
    per_key = Counter(e["idempotency_key"] for e in sim.mutation_log())
    return sum(n - 1 for n in per_key.values())


class GatedSim(SimulatorBackend):
    """A simulator whose restarts wait inside the backend call until `go` is set."""

    def __init__(self) -> None:
        super().__init__("default", 7)
        self.inside = asyncio.Event()
        self.go = asyncio.Event()

    async def restart(
        self, namespace: str, deployment: str, *, key: str, at: str
    ) -> dict[str, Any]:
        self.inside.set()
        await self.go.wait()
        return await super().restart(namespace, deployment, key=key, at=at)


async def admitted(rig: Rig, svc: OpsService) -> McpTokenClaims:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    claims = verify_mcp_token(rig.token(thread_id=t, approval_id=appr, max_calls=10), SECRET)
    await svc.admit(claims, "restart_deployment", dict(WEB))
    return claims


@pytest.fixture
def gated(rig: Rig, config: McpOpsConfig) -> tuple[GatedSim, OpsService]:
    sim = GatedSim()
    return sim, OpsService(config=config, store=rig.store, backend=sim)


async def test_five_concurrent_identical_calls_on_five_tokens(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    views = await asyncio.gather(
        *(rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
          for _ in range(5))
    )  # fmt: skip
    assert len({v["id"] for v in views}) == 1
    assert sum(1 for v in views if not v["replayed"]) == 1
    assert len(sim.mutation_log()) == 1 and duplicates(sim) == 0


async def test_a_plain_call_racing_a_task_call_applies_once(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)

    async def plain() -> Any:
        async with rig.client(rig.token(thread_id=t, approval_id=appr)) as c:
            return (await c.call_tool("restart_deployment", WEB)).structured_content

    async def as_task() -> Any:
        async with rig.client(rig.token(thread_id=t, approval_id=appr)) as c:
            return (
                await (await call_tool_task(c, "restart_deployment", WEB)).result()
            ).structured_content

    a, b = await asyncio.gather(plain(), as_task())
    assert a["id"] == b["id"] and len(sim.mutation_log()) == 1


async def test_a_redelivery_while_the_first_body_is_inside_the_backend(
    rig: Rig, gated: tuple[GatedSim, OpsService]
) -> None:
    sim, svc = gated
    claims = await admitted(rig, svc)
    first = asyncio.create_task(svc.execute(claims, "restart_deployment", dict(WEB), "task-1"))
    await sim.inside.wait()
    second = asyncio.create_task(svc.execute(claims, "restart_deployment", dict(WEB), "task-1"))
    await asyncio.sleep(0.2)  # the redelivered body has lost the claim and is waiting
    sim.go.set()
    one, two = await asyncio.gather(first, second)
    assert one["id"] == two["id"] and two["replayed"] is True
    assert one["status"] == two["status"] == "succeeded"
    assert len(sim.mutation_log()) == 1


async def test_a_cancelled_task_body_does_not_strand_the_operation(
    rig: Rig, gated: tuple[GatedSim, OpsService]
) -> None:
    sim, svc = gated
    claims = await admitted(rig, svc)
    body = asyncio.create_task(svc.execute(claims, "restart_deployment", dict(WEB), "task-1"))
    await sim.inside.wait()
    body.cancel()
    with pytest.raises(asyncio.CancelledError):
        await body
    assert sim.mutation_log() == []  # cancelled before the change: nothing applied
    (row,) = await rig.rows("kind = 'write'")
    assert row["status"] != "running"


async def test_a_reconciler_on_a_new_instance_does_not_misrecord_a_live_write(
    rig: Rig, gated: tuple[GatedSim, OpsService]
) -> None:
    sim, svc = gated
    claims = await admitted(rig, svc)
    old_pod = asyncio.create_task(svc.execute(claims, "restart_deployment", dict(WEB), "task-1"))
    await sim.inside.wait()
    await reconcile(rig.store, sim)  # the new pod's lifespan
    sim.go.set()
    view = await old_pod
    assert len(sim.mutation_log()) == 1 and duplicates(sim) == 0  # applied exactly once
    (row,) = await rig.rows("kind = 'write'")
    assert row["status"] == view["status"] == "succeeded"


async def test_an_incident_for_a_vanished_thread_ends_failed_not_running(
    rig: Rig, sim: SimulatorBackend
) -> None:
    args = {"severity": "sev3", "title": "web slow", "summary": "p99 up"}
    with pytest.raises(ToolError):
        await rig.write(
            "create_incident",
            args,
            rig.token(thread_id=uuid.uuid4(), tools=("create_incident", "page_oncall")),
        )
    (row,) = await rig.rows("kind = 'escalation'")
    assert sim.mutation_log() == []
    assert row["status"] == "failed"


async def test_the_reconciler_never_reruns_and_a_retry_after_it_never_reruns(
    rig: Rig, sim: SimulatorBackend
) -> None:
    claims = await admitted(rig, rig.service)
    (row,) = await rig.rows("kind = 'write'")
    await rig.store.claim(row["id"], "dead")
    for _ in range(3):  # several restarts in a row
        await reconcile(rig.store, sim, min_age_s=0)
    for _ in range(2):
        out = await rig.service.execute(claims, "restart_deployment", dict(WEB), "retry")
        assert out["status"] == "failed" and out["replayed"] is True
    assert sim.mutation_log() == []
