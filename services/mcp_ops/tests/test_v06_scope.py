"""V06 (adversarial): scope and namespace. Names that look like an allowed or protected target
but are spelled differently (case, whitespace, unicode look-alikes, path tricks), label
selectors that reach for another namespace, and a scale to 0 spelled so it might slip past the
admin rule. Each refusal must leave an audit row and change nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    from conftest import Rig  # type: ignore[attr-defined, unused-ignore]
    from sre_mcp_ops.backends.simulator import SimulatorBackend

WEB = {"namespace": "shop", "deployment": "web"}

# Spellings that could normalize (in a lax parser, a shell, kubectl or a URL) into shop or payments.
HOSTILE_NAMESPACES = [
    "Shop", " shop", "shop ", "shop\n", "shop\t", "\N{FULLWIDTH LATIN SMALL LETTER S}hop",  # fullwidth s
    "shop\N{ZERO WIDTH SPACE}",  # zero-width space
    "p\N{CYRILLIC SMALL LETTER A}yments",  # Cyrillic a
    "PAYMENTS", "payments\n", "../payments", "shop/../payments", "shop%2fpayments", "shop,payments",
    "shop/web", "", "-shop", "shop-", "a" * 64,
]  # fmt: skip
HOSTILE_DEPLOYMENTS = [
    "Web",
    "web\n",
    "web ",
    "../api",
    "web/../api",
    "w\N{CYRILLIC SMALL LETTER IE}b",
    "*",
    "web,api",
]


@pytest.mark.parametrize("ns", HOSTILE_NAMESPACES, ids=repr)
async def test_a_namespace_spelled_differently_is_refused_and_audited(
    rig: Rig, sim: SimulatorBackend, ns: str
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)  # a real approval for shop/web
    token = rig.token(thread_id=t, approval_id=appr, namespaces=("shop", "payments", ns))
    async with rig.client(token) as c:
        with pytest.raises(ToolError, match="refused"):
            await c.call_tool("restart_deployment", {"namespace": ns, "deployment": "web"})
        with pytest.raises(ToolError, match="refused"):
            await c.call_tool("list_pods", {"namespace": ns})
    rows = await rig.rows("status = 'refused'")
    assert len(rows) == 2 and all(r["subject"] == "alice" for r in rows)
    assert sim.mutation_log() == []
    assert not await rig.rows("status <> 'refused'")


@pytest.mark.parametrize("dep", HOSTILE_DEPLOYMENTS, ids=repr)
async def test_a_deployment_spelled_differently_is_refused_and_audited(
    rig: Rig, sim: SimulatorBackend, dep: str
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    async with rig.client(rig.token(thread_id=t, approval_id=appr)) as c:
        with pytest.raises(ToolError, match="refused"):
            await c.call_tool("restart_deployment", {"namespace": "shop", "deployment": dep})
    assert len(await rig.rows("status = 'refused'")) == 1
    assert sim.mutation_log() == []


@pytest.mark.parametrize(
    "selector",
    ["app=api", "app=api,namespace=payments", "namespace=payments", "app==api", "app in (api)",
     "app=web,app=api", "app!=web"],
)  # fmt: skip
async def test_a_label_selector_never_reaches_another_namespace(rig: Rig, selector: str) -> None:
    token = rig.token(namespaces=("shop",))
    async with rig.client(token) as c:
        try:
            res = await c.call_tool("list_pods", {"namespace": "shop", "label_selector": selector})
        except ToolError:
            return  # a selector the backend cannot parse is refused: also fine
    assert res.structured_content is not None
    pods = res.structured_content["data"]["pods"]
    assert all(p["namespace"] == "shop" for p in pods)
    assert not any(p["deployment"] == "api" for p in pods)  # api lives in payments


async def test_a_pod_from_a_namespace_outside_the_token_cannot_be_read_through_another(
    rig: Rig, sim: SimulatorBackend
) -> None:
    pay_pod = (await sim.list_pods("payments", None))[0]["name"]
    token = rig.token(namespaces=("shop",))
    async with rig.client(token) as c:
        for tool, args in (
            ("get_logs", {"namespace": "shop", "pod": pay_pod}),
            ("describe", {"namespace": "shop", "kind": "pod", "name": pay_pod}),
            ("get_logs", {"namespace": "payments", "pod": pay_pod}),
            ("get_events", {"namespace": "payments"}),
        ):
            with pytest.raises(ToolError):
                await c.call_tool(tool, args)
    rows = await rig.rows()
    assert len(rows) == 4 and all(r["status"] in ("failed", "refused") for r in rows)


@pytest.mark.parametrize("replicas", ["0", 0.0, "00"], ids=repr)
async def test_scale_to_zero_spelled_another_way_still_needs_an_admin(
    rig: Rig, sim: SimulatorBackend, replicas: Any
) -> None:
    zero = {**WEB, "replicas": 0}
    t = await rig.thread()
    appr = await rig.approval(t, "scale_deployment", zero, required_scope="operator")
    async with rig.client(rig.token(thread_id=t, approval_id=appr)) as c:
        with pytest.raises(ToolError, match="admin's approval"):
            await c.call_tool("scale_deployment", {**WEB, "replicas": replicas})
    assert sim.mutation_log() == []


async def test_a_viewer_cannot_write_even_with_an_admin_approval(
    rig: Rig, sim: SimulatorBackend
) -> None:
    pay = {"namespace": "payments", "deployment": "api"}
    t = await rig.thread()
    appr = await rig.approval(
        t, "restart_deployment", pay, required_scope="admin", decided_by="root"
    )
    async with rig.client(rig.token(thread_id=t, approval_id=appr, scopes=("viewer",))) as c:
        with pytest.raises(ToolError, match="operator scope"):
            await c.call_tool("restart_deployment", pay)
    assert sim.mutation_log() == []
