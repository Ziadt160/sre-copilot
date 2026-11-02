"""Label mirror into review_labels: idempotent, links labels to the thread and turn."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from sre_core.review.mirror import mirror_labels

pytestmark = pytest.mark.integration

RUN = str(uuid.uuid4())
OTHER_RUN = str(uuid.uuid4())  # a traced run that is not one of our messages (an eval, say)


class FakeLangSmith:
    def __init__(self) -> None:
        self.queue = SimpleNamespace(id="q1", name="sre-copilot-review")
        self.comment = "right tool, wrong namespace"

    def list_annotation_queues(self, *, name: str, **_: Any) -> Any:
        return iter([self.queue] if name == self.queue.name else [])

    def list_runs_from_annotation_queue(self, queue_id: str, *, status: str) -> Any:
        assert queue_id == "q1" and status == "completed"
        return iter([SimpleNamespace(id=RUN), SimpleNamespace(id=OTHER_RUN)])

    def list_feedback(self, *, run_ids: list[str], feedback_source_type: list[str]) -> Any:
        assert feedback_source_type == ["app"]
        if str(run_ids[0]) != RUN:
            return iter([])
        return iter(
            [
                SimpleNamespace(
                    id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                    key="tool_args",
                    score=0,
                    value="wrong args",
                    comment=self.comment,
                    correction=None,
                    feedback_source=SimpleNamespace(metadata={"user_id": "reviewer-1"}),
                )
            ]
        )


async def _seed(db: Any) -> uuid.UUID:
    thread = uuid.uuid4()
    await db.execute("insert into public.threads (id, owner_sub) values (%s, 'alice')", (thread,))
    await db.execute(
        "insert into public.messages (thread_id, turn, role, content, trace_run_id) "
        "values (%s, 2, 'assistant', 'a', %s)",
        (thread, RUN),
    )
    return thread


async def test_mirror_twice_gives_the_same_rows(db: Any) -> None:
    thread = await _seed(db)
    fake = FakeLangSmith()

    first = await mirror_labels(db, fake)
    second = await mirror_labels(db, fake)

    assert (first.inserted, first.updated) == (1, 0)
    assert (second.inserted, second.updated) == (0, 1)
    rows = await (await db.execute("select * from public.review_labels")).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["thread_id"] == thread and row["turn"] == 2 and row["run_id"] == RUN
    assert row["reviewer"] == "reviewer-1" and row["source"] == "langsmith"
    assert row["label"]["key"] == "tool_args" and row["label"]["score"] == 0


async def test_an_edited_label_updates_the_row_instead_of_adding_one(db: Any) -> None:
    await _seed(db)
    fake = FakeLangSmith()
    await mirror_labels(db, fake)
    fake.comment = "actually correct"
    await mirror_labels(db, fake)
    rows = await (await db.execute("select label from public.review_labels")).fetchall()
    assert len(rows) == 1 and rows[0]["label"]["comment"] == "actually correct"


async def test_a_missing_queue_is_an_error_not_an_empty_success(db: Any) -> None:
    with pytest.raises(SystemExit, match="not found"):
        await mirror_labels(db, FakeLangSmith(), queue_name="nope")
