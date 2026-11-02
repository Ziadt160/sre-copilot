"""The chat API over HTTP with Postgres stores and a scripted agent: SSE, threads and RLS, the
approval gate (once, never after expiry, only by someone allowed), server-side resume, feedback."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from typing import Any, cast

import httpx
import jwt
import pytest
from psycopg_pool import AsyncConnectionPool

from sre_core.agent.testing import (
    FakeBudgetGuard,
    FakeChatAgent,
    FakeOpsClient,
    MemoryAnswerCache,
    MemoryApprovalStore,
    ScriptedModelRouter,
    new_approval,
    utcnow,
)
from sre_core.api.chat import services, sweep_once
from sre_core.api.main import create_app
from sre_core.api.runtime import Runtime
from sre_core.store.approvals import PgApprovalStore
from sre_core.store.feedback import PgFeedbackStore
from sre_core.store.outbox import PgOutbox
from sre_core.store.threads import PgThreadStore
from sre_shared.contracts import agent as ev
from sre_shared.contracts import ops
from sre_shared.models import Claims
from sre_shared.settings import get_settings

pytestmark = pytest.mark.integration

SECRET = "test-secret-test-secret-test-secret-0123"
POLICY = ops.NamespacePolicy(
    allowed=frozenset({"shop", "payments"}), protected=frozenset({"payments"})
)


class ClockedApprovals(PgApprovalStore):
    """FakeChatAgent reads `approvals.clock()`; tests move it to create old approvals."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        super().__init__(pool, POLICY)
        self.clock = utcnow


class RecordingAgent(FakeChatAgent):
    resumed: list[ops.ApprovalRecord]
    gate: asyncio.Event | None = None

    async def resume(  # type: ignore[override]
        self, claims: Claims, thread_id: uuid.UUID, approval: ops.ApprovalRecord
    ) -> AsyncIterator[ev.AgentEvent]:
        if not hasattr(self, "resumed"):
            self.resumed = []
        self.resumed.append(approval)
        if self.gate is not None:
            await self.gate.wait()
        async for e in super().resume(claims, thread_id, approval):
            yield e

    async def stream_turn(  # type: ignore[override]
        self, claims: Claims, thread_id: uuid.UUID, turn: int, message: str
    ) -> AsyncIterator[ev.AgentEvent]:
        if message == "boom-first":
            raise RuntimeError("secret internals")
        if message == "boom-later":
            yield ev.TurnEvent(thread_id=thread_id, turn=turn)
            raise RuntimeError("secret internals")
        if message == "slow":
            yield ev.TurnEvent(thread_id=thread_id, turn=turn)
            assert self.gate is not None
            await self.gate.wait()
            yield ev.DoneEvent(thread_id=thread_id, turn=turn, route="simple", answer="slow done")
            return
        async for e in super().stream_turn(claims, thread_id, turn, message):
            yield e


@pytest.fixture(autouse=True)
def _jwt(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def hdr(sub: str, *scopes: str) -> dict[str, str]:
    payload = {
        "sub": sub, "role": "authenticated", "exp": int(time.time()) + 600,
        "app_metadata": {"scopes": list(scopes), "namespaces": ["shop"]},
    }  # fmt: skip
    return {"Authorization": f"Bearer {jwt.encode(payload, SECRET, algorithm='HS256')}"}


ALICE = hdr("alice", "viewer", "operator")
BOB = hdr("bob", "viewer", "operator")
ADMIN = hdr("root", "viewer", "operator", "admin")


@pytest.fixture
async def rt(db: object, pool: AsyncConnectionPool) -> Runtime:
    approvals = ClockedApprovals(pool)
    agent = RecordingAgent(approvals=cast(MemoryApprovalStore, approvals), policy=POLICY)
    return Runtime(
        threads=PgThreadStore(pool), approvals=approvals, ops=FakeOpsClient(),
        cache=MemoryAnswerCache(), budget=FakeBudgetGuard(), models=ScriptedModelRouter(),
        outbox=PgOutbox(pool), policy=POLICY, agent=agent,
        extras={"feedback": PgFeedbackStore(pool)},
    )  # fmt: skip


@pytest.fixture
async def client(rt: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(runtime=rt)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c
    await services(rt).manager.close()


def agent_of(rt: Runtime) -> RecordingAgent:
    return cast(RecordingAgent, rt.agent)


def parse(text: str) -> list[ev.AgentEvent]:
    out = []
    for frame in [f for f in text.split("\n\n") if f]:
        name_line, data_line = frame.split("\n")
        data = json.loads(data_line.removeprefix("data: "))
        data["event"] = name_line.removeprefix("event: ")
        out.append(ev.AGENT_EVENT.validate_python(data))
    return out


async def say(c: httpx.AsyncClient, who: dict[str, str], msg: str, **extra: Any) -> httpx.Response:
    return await c.post("/chat", json={"message": msg, **extra}, headers=who)


async def settle(rt: Runtime) -> None:
    await services(rt).manager.idle()


async def propose(
    c: httpx.AsyncClient, rt: Runtime, who: dict[str, str] = ALICE
) -> ev.ApprovalRequiredEvent:
    r = await say(c, who, "please restart checkout")
    assert r.status_code == 200
    last = parse(r.text)[-1]
    assert isinstance(last, ev.ApprovalRequiredEvent)
    await settle(rt)
    return last


async def test_anonymous_gets_401_everywhere(client: httpx.AsyncClient) -> None:
    u = uuid.uuid4()
    for method, path in [
        ("POST", "/chat"), ("GET", "/threads"), ("GET", f"/threads/{u}"),
        ("GET", f"/threads/{u}/live"), ("POST", f"/approvals/{u}"), ("GET", "/approvals"),
        ("POST", "/feedback"), ("POST", "/events"),
    ]:  # fmt: skip
        r = await client.request(method, path, json={})
        assert r.status_code == 401, path


async def test_chat_streams_valid_events_and_saves_the_thread(client: httpx.AsyncClient) -> None:
    r = await say(client, ALICE, "what is a pod")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse(r.text)
    assert [e.event for e in events] == ["turn", "route", "token", "done"]
    tid = cast(ev.TurnEvent, events[0]).thread_id
    listed = (await client.get("/threads", headers=ALICE)).json()
    assert [t["id"] for t in listed] == [str(tid)]
    view = (await client.get(f"/threads/{tid}", headers=ALICE)).json()
    assert [(m["role"], m["turn"]) for m in view["messages"]] == [("user", 1), ("assistant", 1)]
    # a second turn on the same thread
    r2 = await say(client, ALICE, "and a service", thread_id=str(tid))
    assert cast(ev.TurnEvent, parse(r2.text)[0]).turn == 2


async def test_other_users_cannot_read_post_or_follow(client: httpx.AsyncClient) -> None:
    r = await say(client, ALICE, "hi")
    tid = cast(ev.TurnEvent, parse(r.text)[0]).thread_id
    assert (await client.get(f"/threads/{tid}", headers=BOB)).status_code == 404
    assert (await client.get(f"/threads/{tid}/live", headers=BOB)).status_code == 404
    assert (await say(client, BOB, "hijack", thread_id=str(tid))).status_code == 404
    assert (await client.get("/threads", headers=BOB)).json() == []


async def test_retried_client_message_id_returns_the_same_turn(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    a = await say(client, ALICE, "hi", client_message_id="c1")
    tid = cast(ev.TurnEvent, parse(a.text)[0]).thread_id
    await settle(rt)
    b = await say(client, ALICE, "hi", client_message_id="c1", thread_id=str(tid))
    assert [(e.event) for e in parse(b.text)] == [e.event for e in parse(a.text)]
    assert cast(ev.TurnEvent, parse(b.text)[0]).turn == 1
    view = (await client.get(f"/threads/{tid}", headers=ALICE)).json()
    assert len([m for m in view["messages"] if m["role"] == "user"]) == 1


async def test_first_event_errors_map_to_http_and_later_ones_to_error_events(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    assert (await say(client, ALICE, "boom-first")).status_code == 500
    r = await say(client, ALICE, "boom-later")
    assert r.status_code == 200
    last = parse(r.text)[-1]
    assert isinstance(last, ev.ErrorEvent) and "secret" not in last.detail
    rt.budget = FakeBudgetGuard(daily_tokens=1)
    r = await say(client, ALICE, "hi")
    assert r.status_code == 429


async def test_busy_thread_is_409_until_the_approval_is_decided(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    ap = await propose(client, rt)
    tid = (await client.get("/approvals", headers=ALICE)).json()[0]["thread_id"]
    assert (await say(client, ALICE, "more", thread_id=tid)).status_code == 409
    r = await client.post(
        f"/approvals/{ap.approval_id}", json={"decision": "approve"}, headers=ALICE
    )
    assert r.status_code == 200 and r.json()["approval"]["status"] == "approved"
    await settle(rt)
    assert (await say(client, ALICE, "more", thread_id=tid)).status_code == 200
    assert [a.status for a in agent_of(rt).resumed] == ["approved"]  # read back from the database


async def test_two_concurrent_approvals_one_wins(client: httpx.AsyncClient, rt: Runtime) -> None:
    ap = await propose(client, rt)
    rs = await asyncio.gather(
        *[
            client.post(f"/approvals/{ap.approval_id}", json={"decision": "approve"}, headers=ALICE)
            for _ in range(2)
        ]
    )
    assert sorted(r.status_code for r in rs) == [200, 409]
    await settle(rt)
    assert len(agent_of(rt).resumed) == 1


async def test_approve_after_expiry_is_409_and_nothing_runs(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    cast(ClockedApprovals, rt.approvals).clock = lambda: utcnow() - timedelta(minutes=20)
    ap = await propose(client, rt)
    r = await client.post(
        f"/approvals/{ap.approval_id}", json={"decision": "approve"}, headers=ALICE
    )
    assert r.status_code == 409
    await settle(rt)
    # the sweep step released the thread with an "expired" resume, never an approved one
    assert [a.status for a in agent_of(rt).resumed] == ["expired"]
    tid = (await client.get("/threads", headers=ALICE)).json()[0]["id"]
    assert (await client.get(f"/threads/{tid}", headers=ALICE)).json()["thread"][
        "status"
    ] == "active"


async def test_sweeper_expires_and_resumes(client: httpx.AsyncClient, rt: Runtime) -> None:
    cast(ClockedApprovals, rt.approvals).clock = lambda: utcnow() - timedelta(minutes=20)
    await propose(client, rt)
    assert rt.agent is not None
    assert await sweep_once(services(rt).manager, rt.approvals, rt.agent) == 1
    await settle(rt)
    assert [a.status for a in agent_of(rt).resumed] == ["expired"]


async def _admin_approval(rt: Runtime, owner: str, ns: str) -> uuid.UUID:
    t = await rt.threads.create(Claims(sub=owner))
    await rt.threads.set_status(t.id, "awaiting_approval")
    a = new_approval(
        t.id, 1, "restart_deployment", {"namespace": ns, "deployment": "web"},
        policy=POLICY, requested_by=owner,
    )  # fmt: skip
    return (await rt.approvals.create(a)).id


async def test_operator_cannot_approve_admin_required_but_admin_can(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    aid = await _admin_approval(rt, "alice", "payments")
    r = await client.post(f"/approvals/{aid}", json={"decision": "approve"}, headers=ALICE)
    assert r.status_code == 403
    assert (await client.get("/approvals", headers=ADMIN)).json()[0]["id"] == str(aid)
    r = await client.post(f"/approvals/{aid}", json={"decision": "approve"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["approval"]["decided_by"] == "root"


async def test_non_owner_cannot_decide(client: httpx.AsyncClient, rt: Runtime) -> None:
    aid = await _admin_approval(rt, "alice", "shop")
    r = await client.post(f"/approvals/{aid}", json={"decision": "approve"}, headers=BOB)
    assert r.status_code == 404
    assert (await client.get("/approvals", headers=BOB)).json() == []
    assert (await rt.approvals.get(aid)).status == "pending"  # type: ignore[union-attr]


async def _scale_approval(rt: Runtime, owner: str) -> uuid.UUID:
    t = await rt.threads.create(Claims(sub=owner))
    a = new_approval(
        t.id, 1, "scale_deployment", {"namespace": "shop", "deployment": "web", "replicas": 2},
        policy=POLICY, requested_by=owner,
    )  # fmt: skip
    return (await rt.approvals.create(a)).id


async def test_edit_is_revalidated(client: httpx.AsyncClient, rt: Runtime) -> None:
    base = {"namespace": "shop", "deployment": "web"}
    aid = await _scale_approval(rt, "alice")

    def edit(args: dict[str, Any], who: dict[str, str] = ALICE) -> Any:
        return client.post(
            f"/approvals/{aid}", json={"decision": "edit", "args": args}, headers=who
        )

    assert (await edit({**base, "replicas": 0})).status_code == 403  # now needs an admin
    assert (await edit({**base, "replicas": 2, "extra": 1})).status_code == 422
    assert (
        await edit({"namespace": "kube-system", "deployment": "web", "replicas": 1})
    ).status_code == 403
    r = await edit({**base, "replicas": 5})
    assert r.status_code == 200 and r.json()["approval"]["args"]["replicas"] == 5
    assert (
        await client.post(
            f"/approvals/{aid}", json={"decision": "approve", "args": base}, headers=ALICE
        )
    ).status_code in (409, 422)

    aid = await _scale_approval(rt, "root")
    r = await edit({**base, "replicas": 0}, ADMIN)
    assert r.status_code == 200 and r.json()["approval"]["args"]["replicas"] == 0


async def test_resume_continues_when_nobody_is_listening(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    ap = await propose(client, rt)
    gate = agent_of(rt).gate = asyncio.Event()
    r = await client.post(
        f"/approvals/{ap.approval_id}", json={"decision": "approve"}, headers=ALICE
    )
    assert r.status_code == 200  # the request is over; the run is still waiting on the gate
    tid = (await client.get("/approvals?status=pending", headers=ALICE)).json()
    assert tid == []
    gate.set()
    await settle(rt)
    threads = (await client.get("/threads", headers=ALICE)).json()
    view = (await client.get(f"/threads/{threads[0]['id']}", headers=ALICE)).json()
    assert view["thread"]["status"] == "active"
    assert view["messages"][-1]["content"].startswith("Done:")
    live = await client.get(f"/threads/{threads[0]['id']}/live", headers=ALICE)
    names = [e.event for e in parse(live.text)]
    assert names == ["operation", "operation", "done"]


async def test_abandoned_chat_stream_still_finishes(client: httpx.AsyncClient, rt: Runtime) -> None:
    gate = agent_of(rt).gate = asyncio.Event()
    req = asyncio.create_task(say(client, ALICE, "slow"))
    for _ in range(100):  # wait until the run exists, then drop the client
        await asyncio.sleep(0.02)
        if services(rt).manager._tasks:
            break
    req.cancel()
    await asyncio.gather(req, return_exceptions=True)
    gate.set()
    await settle(rt)
    tid = (await client.get("/threads", headers=ALICE)).json()[0]["id"]
    view = (await client.get(f"/threads/{tid}", headers=ALICE)).json()
    assert view["messages"][-1]["content"] == "slow done"


async def test_live_relays_a_running_turn_then_ends(client: httpx.AsyncClient, rt: Runtime) -> None:
    gate = agent_of(rt).gate = asyncio.Event()
    chat = asyncio.create_task(say(client, ALICE, "slow"))
    for _ in range(100):
        await asyncio.sleep(0.02)
        tids = (await client.get("/threads", headers=ALICE)).json()
        if tids and services(rt).manager.current(uuid.UUID(tids[0]["id"])):
            break
    live = asyncio.create_task(client.get(f"/threads/{tids[0]['id']}/live", headers=ALICE))
    await asyncio.sleep(0.1)
    gate.set()
    assert [e.event for e in parse((await live).text)] == ["turn", "done"]
    assert [e.event for e in parse((await chat).text)] == ["turn", "done"]


async def test_feedback_and_events_are_the_users_and_sync_is_enqueued_once(
    client: httpx.AsyncClient, rt: Runtime, pool: AsyncConnectionPool
) -> None:
    r = await say(client, ALICE, "hi")
    tid = str(cast(ev.TurnEvent, parse(r.text)[0]).thread_id)
    await settle(rt)
    body = {"thread_id": tid, "turn": 1, "rating": -1, "reason": "wrong", "sources": {"7": True}}
    for _ in range(2):
        assert (await client.post("/feedback", json=body, headers=ALICE)).status_code == 200
    assert (
        await client.post(
            "/events", json={"thread_id": tid, "turn": 1, "kind": "copy"}, headers=ALICE
        )
    ).status_code == 200
    assert (await client.post("/feedback", json=body, headers=BOB)).status_code == 404
    assert (
        await client.post("/feedback", json={**body, "turn": 9}, headers=ALICE)
    ).status_code == 404
    assert (
        await client.post(
            "/events", json={"thread_id": tid, "turn": 1, "kind": "copy"}, headers=BOB
        )
    ).status_code == 404
    async with pool.connection() as conn:
        fb = await (await conn.execute("select owner_sub, rating from public.feedback")).fetchall()
        ob = await (
            await conn.execute(
                "select kind, payload from public.outbox where kind = 'feedback_sync'"
            )
        ).fetchall()
        n_events = await (await conn.execute("select owner_sub from public.ui_events")).fetchall()
    assert fb == [{"owner_sub": "alice", "rating": -1}] and len(n_events) == 1
    assert (
        len(ob) == 1
        and ob[0]["payload"]["owner_sub"] == "alice"
        and ob[0]["payload"]["rating"] == -1
    )


async def test_build_runtime_wires_real_stores(
    pg_dsn: str, monkeypatch: pytest.MonkeyPatch, db: object
) -> None:
    from sre_core.api import runtime as rtm

    seen: dict[str, Any] = {}

    def fake_factory(module: str, name: str) -> Any:
        def make(*args: Any, **kw: Any) -> Any:
            seen[name] = (args, kw)
            return object() if name != "make_chat_agent" else "agent"

        return make

    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    get_settings.cache_clear()
    monkeypatch.setattr(rtm, "_factory", fake_factory)
    built = await rtm.build_runtime(get_settings(), retriever=object(), answerer=object())
    try:
        assert isinstance(built.threads, PgThreadStore) and built.agent == "agent"
        assert isinstance(built.approvals, PgApprovalStore) and isinstance(built.outbox, PgOutbox)
        kw = seen["make_chat_agent"][1]
        assert kw["checkpointer"] is built.extras["checkpointer"]  # type: ignore[index]
        assert kw["threads"] is built.threads and kw["policy"] == built.policy
    finally:
        await rtm.close_runtime(built)
