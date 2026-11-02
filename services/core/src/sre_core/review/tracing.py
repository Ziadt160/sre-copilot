"""LangSmith tracing for chat turns (opt-in, free tier: 5k traces a month, 14-day retention).

Two pieces:

* `client_wrapper(settings)` is the hook `make_model_router(..., client_wrapper=...)` takes. It
  wraps each `AsyncOpenAI` client so its calls become child runs, but only while a trace is open.
* `turn_trace(...)` opens one root run per chat turn, with thread, turn and user metadata
  (`thread_id` is the key LangSmith groups a conversation by), and yields the run id to store in
  `messages.trace_run_id` so feedback and review can find the trace again.

Tracing is off, and these functions do nothing, when `LANGSMITH_API_KEY` is unset or the
environment variable `SRE_TRACING` is `off`. Evaluation runs must not trace (they would burn the
5k monthly traces on batch work): the eval runner sets `SRE_TRACING=off` before building anything
(`SRE_TRACING=off uv run python -m sre_evals ...`). As a second guard a wrapped client records
nothing outside `turn_trace`, because the LangSmith SDK traces only inside an enabled context and
the environment variable `LANGSMITH_TRACING` is never set by this code.

`langsmith` is imported inside the functions: core does not declare it as a dependency yet, and
an install without it must keep working with tracing off.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from sre_shared.settings import Settings

if TYPE_CHECKING:
    from langsmith import Client

ClientFactory = Callable[[str], "Client"]

TRACING_ENV = "SRE_TRACING"


def tracing_enabled(settings: Settings) -> bool:
    if os.environ.get(TRACING_ENV, "").strip().lower() == "off":
        return False
    key = settings.langsmith_api_key
    return bool(key and key.get_secret_value())


def default_client(api_key: str) -> Client:
    from langsmith import Client

    return Client(api_key=api_key)


def _client(settings: Settings, factory: ClientFactory | None) -> Client:
    key = settings.langsmith_api_key
    assert key is not None  # callers check tracing_enabled first
    return (factory or default_client)(key.get_secret_value())


def client_wrapper(
    settings: Settings, *, client_factory: ClientFactory | None = None
) -> Callable[[AsyncOpenAI], AsyncOpenAI] | None:
    """The wrapper for `make_model_router`, or None when tracing is off."""
    if not tracing_enabled(settings):
        return None
    ls_client = _client(settings, client_factory)

    def wrap(client: AsyncOpenAI) -> AsyncOpenAI:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client, tracing_extra={"client": ls_client})

    return wrap


@dataclass
class TurnTrace:
    """Handle on the open root run. `run_id` is None when tracing is off."""

    run_id: str | None = None
    _run: Any = None

    def finish(self, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        """Record the turn's outcome; the context manager ends the run either way."""
        if self._run is not None:
            self._run.end(outputs=outputs, error=error)


@contextmanager
def turn_trace(
    settings: Settings,
    *,
    thread_id: uuid.UUID | str,
    turn: int,
    user_sub: str,
    route: str | None = None,
    client_factory: ClientFactory | None = None,
) -> Iterator[TurnTrace]:
    """One root run for a chat turn; model calls made inside become its children."""
    if not tracing_enabled(settings):
        yield TurnTrace()
        return

    from langsmith import RunTree, tracing_context

    ls_client = _client(settings, client_factory)
    metadata: dict[str, Any] = {
        "thread_id": str(thread_id),  # LangSmith's thread grouping key
        "turn": turn,
        "user_sub": user_sub,
    }
    if route:
        metadata["route"] = route
    run = RunTree(
        name="chat_turn",
        run_type="chain",
        id=uuid.uuid4(),
        project_name=settings.langsmith_project,
        extra={"metadata": metadata},
        tags=["sre-copilot"],
        ls_client=ls_client,
    )
    handle = TurnTrace(run_id=str(run.id), _run=run)
    run.post()
    try:
        with tracing_context(enabled=True, parent=run, client=ls_client):
            yield handle
    except BaseException as exc:
        run.end(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if run.end_time is None:
            run.end()
        run.patch()
