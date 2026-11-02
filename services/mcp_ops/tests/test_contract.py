"""Contract tests: what any backend must do through the real MCP surface. They run against the
simulator locally and against a kind cluster in CI (`--backend kind`, see mcp-ops-kind.yml).
Names used (shop/web with revisions 1 and 2, payments/api) exist in both fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    from conftest import Rig

WEB = {"namespace": "shop", "deployment": "web"}


async def approved_token(rig: Rig, tool: str, args: dict[str, object], **kw: object) -> str:
    t = await rig.thread()
    appr = await rig.approval(t, tool, args)  # type: ignore[arg-type]
    return rig.token(thread_id=t, approval_id=appr, **kw)  # type: ignore[arg-type]


async def test_reads_are_wrapped_as_untrusted_data(rig: Rig) -> None:
    tok = lambda: rig.token()  # noqa: E731  (a fresh token per call: max_calls counts per token)
    pods = await rig.read("list_pods", {"namespace": "shop"}, tok())
    assert pods["trust"] == "untrusted_data" and pods["tool"] == "list_pods"
    assert {p["deployment"] for p in pods["data"]["pods"]} >= {"web"}
    dep = await rig.read(
        "describe", {"namespace": "shop", "kind": "deployment", "name": "web"}, tok()
    )
    assert dep["data"]["replicas"] >= 1
    rs = await rig.read("rollout_status", WEB, tok())
    assert rs["data"]["status"] in {"complete", "progressing", "failed"}
    ev = await rig.read("get_events", {"namespace": "shop"}, tok())
    assert isinstance(ev["data"]["events"], list)
    pod = pods["data"]["pods"][0]["name"]
    logs = await rig.read("get_logs", {"namespace": "shop", "pod": pod}, tok())
    assert isinstance(logs["data"]["text"], str) and logs["data"]["truncated"] in (True, False)


async def test_reads_leave_audit_rows(rig: Rig) -> None:
    await rig.read("list_pods", {"namespace": "shop"}, rig.token())
    (row,) = await rig.rows("kind = 'read'")
    assert row["subject"] == "alice" and row["tool"] == "list_pods"
    assert row["status"] == "succeeded" and row["latency_ms"] is not None
    assert row["args"] == {"namespace": "shop", "label_selector": None}


async def test_restart_runs_as_a_task_and_is_recorded(rig: Rig) -> None:
    tok = await approved_token(rig, "restart_deployment", WEB)
    out = await rig.write("restart_deployment", WEB, tok)
    assert out["status"] == "succeeded" and out["result"]["changed"] is True
    assert out["task_id"] and out["replayed"] is False
    dep = await rig.read(
        "describe", {"namespace": "shop", "kind": "deployment", "name": "web"}, rig.token()
    )
    assert dep["data"]["restarted_at"] == out["result"]["restarted_at"]
    (row,) = await rig.rows("kind = 'write'")
    assert row["status"] == "succeeded" and row["approval_id"] is not None
    assert row["latency_ms"] is not None and row["started_at"] and row["finished_at"]


async def test_scale_sets_an_absolute_count(rig: Rig) -> None:
    args = {**WEB, "replicas": 3}
    out = await rig.write(
        "scale_deployment", args, await approved_token(rig, "scale_deployment", args)
    )
    assert out["status"] == "succeeded"
    dep = await rig.read(
        "describe", {"namespace": "shop", "kind": "deployment", "name": "web"}, rig.token()
    )
    assert dep["data"]["replicas"] == 3


async def test_rollback_to_an_explicit_revision(rig: Rig) -> None:
    args = {**WEB, "to_revision": 1}
    out = await rig.write(
        "rollback_deployment", args, await approved_token(rig, "rollback_deployment", args)
    )
    assert out["status"] == "succeeded" and out["result"]["changed"] is True


async def test_rollback_to_a_missing_revision_fails_cleanly(rig: Rig) -> None:
    args = {**WEB, "to_revision": 99}
    out = await rig.write(
        "rollback_deployment", args, await approved_token(rig, "rollback_deployment", args)
    )
    assert out["status"] == "failed" and "invalid_revision" in out["error"]


async def test_incident_then_page_on_the_escalate_route(rig: Rig) -> None:
    t = await rig.thread()
    inc = await rig.write(
        "create_incident",
        {"severity": "sev2", "title": "checkout down", "summary": "crash loop after 1.4.2"},
        rig.token(thread_id=t, turn=1, tools=("create_incident", "page_oncall")),
    )
    assert (
        inc["status"] == "succeeded" and inc["kind"] == "escalation" and inc["approval_id"] is None
    )
    iid = inc["result"]["incident_id"]
    page = await rig.write(
        "page_oncall",
        {"incident_id": iid},
        rig.token(thread_id=t, turn=2, tools=("create_incident", "page_oncall")),
    )
    assert page["status"] == "succeeded"
    cur = await rig.db.execute(
        "select count(*) as n from public.pages where target = 'sre-primary'"
    )
    assert (await cur.fetchone())["n"] == 1  # type: ignore[index]


async def test_page_oncall_for_someone_elses_incident_is_refused(rig: Rig) -> None:
    t = await rig.thread()
    inc = await rig.write(
        "create_incident", {"severity": "sev3", "title": "abc", "summary": "abc"},
        rig.token(thread_id=t, tools=("create_incident", "page_oncall")),
    )  # fmt: skip
    mallory = rig.token(
        sub="mallory",
        thread_id=await rig.thread("mallory"),
        tools=("create_incident", "page_oncall"),
    )
    async with rig.client(mallory) as c:
        with pytest.raises(ToolError, match="incident not found"):
            await c.call_tool("page_oncall", {"incident_id": inc["result"]["incident_id"]})


async def test_get_operation_is_private_to_its_subject(rig: Rig) -> None:
    args = {**WEB, "replicas": 2}
    out = await rig.write(
        "scale_deployment", args, await approved_token(rig, "scale_deployment", args)
    )
    op = out["id"]
    mine = await rig.read("get_operation", {"operation_id": op}, rig.token())
    assert mine["status"] == "succeeded"
    async with rig.client(rig.token(sub="mallory")) as c:
        with pytest.raises(ToolError, match="not found"):
            await c.call_tool("get_operation", {"operation_id": op})
    admin = await rig.read(
        "get_operation", {"operation_id": op}, rig.token(sub="root", scopes=("viewer", "admin"))
    )
    assert admin["id"] == op
