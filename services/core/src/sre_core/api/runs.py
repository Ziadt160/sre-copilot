"""Server-side runs: the agent's turns and resumes run as tasks the runtime holds, never as part
of a request. A client that disconnects (the approver closing the tab, a dropped SSE stream)
cannot cancel a run half way through a write.

Each thread has one current `Run` with a buffer of its events. Subscribers (the `/chat` request
that started it, `GET /threads/{id}/live`) replay the buffer and then follow the live tail. The
last run of a thread stays in memory for a while so a late or reconnecting subscriber still gets
its events; the broker is in-process, so with several API workers `/live` only sees runs started
on its own worker (single-worker deployment in Phase 3).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from functools import partial

from sre_core.agent.ports import (
    ApprovalStore,
    BudgetExceeded,
    ChatAgent,
    MessageRecord,
    ThreadBusy,
    ThreadNotFound,
    ThreadStore,
)
from sre_shared.contracts import agent as ev
from sre_shared.contracts.ops import ApprovalRecord
from sre_shared.llm import QuotaExceeded
from sre_shared.models import Claims

log = logging.getLogger(__name__)

# /live stops after the first of these; the agent may still be finishing bookkeeping.
TERMINAL = (
    ev.DoneEvent,
    ev.AbstainEvent,
    ev.ErrorEvent,
    ev.ApprovalRequiredEvent,
    ev.EscalatedEvent,
)
KEEP_FINISHED_S = 900.0

EventSource = Callable[[], AsyncIterator[ev.AgentEvent]]


def error_detail(exc: BaseException) -> str:
    """Text safe to show a client: our own refusals say why, anything else stays generic."""
    if isinstance(exc, BudgetExceeded | ThreadBusy | ThreadNotFound | QuotaExceeded):
        return str(exc)
    return "the turn failed"


class Run:
    def __init__(self, thread_id: uuid.UUID, turn: int | None, *, resume: bool) -> None:
        self.thread_id = thread_id
        self.turn = turn
        self.resume = resume
        self.events: list[ev.AgentEvent] = []
        self.first_error: BaseException | None = None  # raised before any event was produced
        self.finished = False
        self.finished_at = 0.0
        self._cond = asyncio.Condition()

    async def publish(self, event: ev.AgentEvent) -> None:
        async with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    async def finish(self) -> None:
        async with self._cond:
            self.finished = True
            self.finished_at = time.monotonic()
            self._cond.notify_all()

    def _ready(self, pos: int) -> bool:
        return pos < len(self.events) or self.finished

    async def subscribe(self, *, stop_at_terminal: bool = False) -> AsyncIterator[ev.AgentEvent]:
        """Replay the buffer, then follow. Raises the run's first error when it failed before
        producing anything (so a caller can map it to an HTTP status)."""
        i = 0
        while True:
            async with self._cond:
                await self._cond.wait_for(partial(self._ready, i))
                if i >= len(self.events):
                    if self.first_error is not None and i == 0:
                        raise self.first_error
                    return
                event = self.events[i]
            i += 1
            yield event
            if stop_at_terminal and isinstance(event, TERMINAL):
                return


class RunManager:
    def __init__(self, threads: ThreadStore) -> None:
        self.threads = threads
        self._runs: dict[uuid.UUID, Run] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def current(self, thread_id: uuid.UUID) -> Run | None:
        self._prune()
        return self._runs.get(thread_id)

    def start(
        self, thread_id: uuid.UUID, source: EventSource, *, turn: int | None, resume: bool = False
    ) -> Run:
        """Start a run as a task held here. A run that starts while the thread's previous one is
        still finishing (an approval clicked the instant its event went out) waits for it."""
        self._prune()
        run = Run(thread_id, turn, resume=resume)
        previous = self._runs.get(thread_id)
        self._runs[thread_id] = run
        task = asyncio.get_running_loop().create_task(self._drive(run, source, previous))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run

    async def _drive(self, run: Run, source: EventSource, previous: Run | None) -> None:
        saw: str | None = None
        try:
            if previous is not None and not previous.finished:
                async with previous._cond:
                    await previous._cond.wait_for(lambda: previous.finished)
            async for event in source():
                if isinstance(event, ev.ApprovalRequiredEvent):
                    saw = "awaiting_approval"
                    await self.threads.set_status(run.thread_id, "awaiting_approval")
                elif isinstance(event, ev.EscalatedEvent):
                    saw = "escalated"
                    await self.threads.set_status(run.thread_id, "escalated")
                elif isinstance(event, ev.DoneEvent) and run.turn is not None:
                    await self._save_answer(run, event)
                await run.publish(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if run.events:
                await run.publish(ev.ErrorEvent(type=type(exc).__name__, detail=error_detail(exc)))
            else:
                run.first_error = exc
            log.warning("run on thread %s failed: %s", run.thread_id, type(exc).__name__)
        finally:
            if run.resume and saw is None:
                # The approval was decided and the run is over: do not strand the thread.
                try:
                    await self.threads.set_status(run.thread_id, "active")
                except Exception:
                    log.exception("could not reactivate thread %s", run.thread_id)
            await run.finish()

    async def _save_answer(self, run: Run, event: ev.DoneEvent) -> None:
        """The agent normally persists the assistant message (it knows the trace id); this makes
        sure the thread history has it even when it did not. The store upserts."""
        try:
            await self.threads.add_message(
                MessageRecord(
                    thread_id=run.thread_id,
                    turn=event.turn,
                    role="assistant",
                    content=event.answer,
                    route=event.route,
                    citations=event.citations,
                )
            )
        except Exception:
            log.exception("could not save the answer of thread %s", run.thread_id)

    def _prune(self) -> None:
        now = time.monotonic()
        for tid in [
            t for t, r in self._runs.items() if r.finished and now - r.finished_at > KEEP_FINISHED_S
        ]:
            del self._runs[tid]

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


def resume_source(agent: ChatAgent, claims: Claims, approval: ApprovalRecord) -> EventSource:
    return lambda: agent.resume(claims, approval.thread_id, approval)


async def sweep_once(manager: RunManager, approvals: ApprovalStore, agent: ChatAgent) -> int:
    """Expire overdue approvals, then resume each run so the thread says so and returns to
    `active`. The resume sees an `expired` record: the agent must not execute it."""
    expired = await approvals.expire_due()
    for approval in expired:
        owner = Claims(sub=approval.requested_by)
        manager.start(
            approval.thread_id,
            resume_source(agent, owner, approval),
            turn=approval.turn,
            resume=True,
        )
    return len(expired)
