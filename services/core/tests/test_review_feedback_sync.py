"""feedback_sync: idempotent delivery to LangSmith, no network (fake client)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from sre_core.review.feedback_sync import HANDLERS, feedback_id, feedback_sync
from sre_shared.settings import Settings

RUN = str(uuid.uuid4())


class FakeLangSmith:
    def __init__(self) -> None:
        self.rows: dict[str, SimpleNamespace] = {}
        self.calls: list[str] = []
        self.create_kwargs: list[dict[str, Any]] = []

    def list_feedback(self, *, run_ids: list[str], feedback_key: list[str]) -> Any:
        return iter(self.rows.values())

    def create_feedback(self, run_id: str, key: str, **kw: Any) -> None:
        self.calls.append("create")
        self.create_kwargs.append(kw)
        fid = str(kw["feedback_id"])
        assert fid not in self.rows, "duplicate feedback id"
        self.rows[fid] = SimpleNamespace(id=fid, score=kw["score"], comment=kw["comment"])

    def update_feedback(self, fid: Any, *, score: int, comment: str | None) -> None:
        self.calls.append("update")
        self.rows[str(fid)] = SimpleNamespace(id=str(fid), score=score, comment=comment)


def _settings(key: str | None = "k") -> Settings:
    return Settings(langsmith_api_key=SecretStr(key) if key else None)


def _payload(rating: int = -1, reason: str | None = "wrong namespace") -> dict[str, Any]:
    return {"trace_run_id": RUN, "owner_sub": "alice", "rating": rating, "reason": reason}


async def test_sending_twice_creates_one_feedback_row() -> None:
    fake = FakeLangSmith()
    first = await feedback_sync(_payload(), settings=_settings(), client_factory=lambda _k: fake)
    second = await feedback_sync(_payload(), settings=_settings(), client_factory=lambda _k: fake)
    assert (first, second) == ("created", "unchanged")
    assert fake.calls == ["create"] and len(fake.rows) == 1
    # Extended retention is billed; the sync must never ask for it.
    assert fake.create_kwargs[0]["extend_trace_retention"] is False
    assert fake.create_kwargs[0]["score"] == 0  # thumbs-down


async def test_a_changed_rating_updates_in_place() -> None:
    fake = FakeLangSmith()
    await feedback_sync(_payload(-1), settings=_settings(), client_factory=lambda _k: fake)
    outcome = await feedback_sync(
        _payload(1, None), settings=_settings(), client_factory=lambda _k: fake
    )
    assert outcome == "updated" and len(fake.rows) == 1
    assert next(iter(fake.rows.values())).score == 1


def test_the_id_depends_on_trace_and_user_only() -> None:
    assert feedback_id(RUN, "alice") == feedback_id(RUN, "alice")
    assert feedback_id(RUN, "alice") != feedback_id(RUN, "bob")


async def test_no_trace_or_no_key_is_a_skip_not_a_call() -> None:
    fake = FakeLangSmith()

    def boom(_k: str) -> Any:
        raise AssertionError("client must not be built")

    no_trace = {**_payload(), "trace_run_id": None}
    assert (await feedback_sync(no_trace, settings=_settings(), client_factory=boom)).startswith(
        "skipped"
    )
    assert (
        await feedback_sync(_payload(), settings=_settings(None), client_factory=boom)
    ).startswith("skipped")
    assert fake.calls == []


async def test_tracing_off_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SRE_TRACING", "off")
    out = await feedback_sync(_payload(), settings=_settings(), client_factory=lambda _k: 1 / 0)
    assert out.startswith("skipped")


def test_registered_under_the_outbox_kind() -> None:
    assert HANDLERS["feedback_sync"] is feedback_sync
