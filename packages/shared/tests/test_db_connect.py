"""Connect retries: transient connect failures are retried, server rejections are not."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from sre_shared import db


class _FakeConn:
    pass


def _flaky(failures: list[BaseException]) -> Any:
    calls: list[dict[str, Any]] = []

    async def fake_connect(*args: Any, **kwargs: Any) -> _FakeConn:
        calls.append(kwargs)
        if failures:
            raise failures.pop(0)
        return _FakeConn()

    fake_connect.calls = calls  # type: ignore[attr-defined]
    return fake_connect


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_s: float) -> None:
        return None

    async def configure(_conn: Any) -> None:
        return None

    monkeypatch.setattr(db.asyncio, "sleep", instant)
    monkeypatch.setattr(db, "_configure", configure)


def test_transient_classification() -> None:
    dropped = psycopg.OperationalError(
        "connection failed: server closed the connection unexpectedly"
    )
    timeout = psycopg.errors.ConnectionTimeout("connection timeout expired")
    rejected = psycopg.OperationalError("connection failed: FATAL:  password authentication failed")
    assert db.is_transient_connect_error(dropped)
    assert db.is_transient_connect_error(timeout)
    assert not db.is_transient_connect_error(rejected)


async def test_connect_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _flaky([psycopg.OperationalError("server closed the connection unexpectedly")] * 2)
    monkeypatch.setattr(db.AsyncConnection, "connect", fake)
    conn = await db.connect("postgresql://x@127.0.0.1:1/db")
    assert isinstance(conn, _FakeConn)
    assert len(fake.calls) == 3
    assert all(c["connect_timeout"] == db.CONNECT_TIMEOUT_S for c in fake.calls)


async def test_connect_gives_up_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    errors = [psycopg.OperationalError("timeout")] * db.CONNECT_ATTEMPTS
    fake = _flaky(list(errors))
    monkeypatch.setattr(db.AsyncConnection, "connect", fake)
    with pytest.raises(psycopg.OperationalError):
        await db.connect("postgresql://x@127.0.0.1:1/db")
    assert len(fake.calls) == db.CONNECT_ATTEMPTS


async def test_connect_does_not_retry_a_server_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _flaky([psycopg.OperationalError('FATAL:  database "nope" does not exist')])
    monkeypatch.setattr(db.AsyncConnection, "connect", fake)
    with pytest.raises(psycopg.OperationalError):
        await db.connect("postgresql://x@127.0.0.1:1/nope")
    assert len(fake.calls) == 1


def test_pool_connections_have_a_connect_timeout() -> None:
    pool = db.create_pool("postgresql://x@127.0.0.1:1/db", min_size=1, max_size=1)
    assert pool.kwargs["connect_timeout"] == db.CONNECT_TIMEOUT_S
