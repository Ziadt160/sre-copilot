"""V06 (adversarial): V06-4 reached through the wire. A client cancels its own write task
(`ToolTask.cancel()`, MCP tasks/cancel) while the body is inside the backend call."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp_tasks import call_tool_task

from sre_mcp_ops.backends.simulator import SimulatorBackend

if TYPE_CHECKING:
    from conftest import Rig  # type: ignore[attr-defined, unused-ignore]

WEB = {"namespace": "shop", "deployment": "web"}


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


@pytest.fixture
def backend() -> GatedSim:
    return GatedSim()


@pytest.mark.xfail(
    strict=True,
    reason="V06-4: tasks/cancel on a write task that already claimed its row cancels the body "
    "and leaves the operation 'running' until a process restart",
)
async def test_a_client_cancel_mid_write_does_not_strand_the_operation(
    rig: Rig, backend: GatedSim
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    async with rig.client(rig.token(thread_id=t, approval_id=appr)) as c:
        task = await call_tool_task(c, "restart_deployment", WEB)
        await asyncio.wait_for(backend.inside.wait(), 10)
        await task.cancel()
        await asyncio.sleep(0.5)
    backend.go.set()  # a body that survived the cancel finishes now
    await asyncio.sleep(0.5)
    (row,) = await rig.rows("kind = 'write'")
    assert len(backend.mutation_log()) <= 1
    assert row["status"] != "running"
