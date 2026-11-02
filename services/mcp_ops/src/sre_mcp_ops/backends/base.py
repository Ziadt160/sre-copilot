"""The one interface mcp-ops needs from a cluster.

Writes are idempotent by construction (design doc p10): restart stamps `restartedAt` with the
operation's own creation time, scale sets an absolute count, rollback names an explicit revision.
Every write also receives the operation's idempotency key so a backend can tell a repeat apart
and so the simulator's mutation log can prove how often a change was applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class BackendError(Exception):
    """A cluster call failed in a way the caller should see (message is safe to show)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LogChunk:
    text: str
    truncated: bool


class ClusterBackend(Protocol):
    """Reads return plain JSON-able dicts; writes return what changed."""

    async def list_pods(
        self, namespace: str, label_selector: str | None
    ) -> list[dict[str, Any]]: ...

    async def describe(self, namespace: str, kind: str, name: str) -> dict[str, Any]: ...

    async def get_logs(
        self, namespace: str, pod: str, container: str | None, tail_lines: int, previous: bool
    ) -> LogChunk:
        """Raw log text (the tail), uncapped in bytes: the service applies `logs.max_bytes`."""
        ...

    async def get_events(
        self, namespace: str, involved_object: str | None
    ) -> list[dict[str, Any]]: ...

    async def rollout_status(self, namespace: str, deployment: str) -> dict[str, Any]: ...

    async def restart(
        self, namespace: str, deployment: str, *, key: str, at: str
    ) -> dict[str, Any]: ...

    async def scale(
        self, namespace: str, deployment: str, replicas: int, *, key: str
    ) -> dict[str, Any]: ...

    async def rollback(
        self, namespace: str, deployment: str, to_revision: int, *, key: str
    ) -> dict[str, Any]: ...

    async def applied(self, tool: str, args: dict[str, Any], *, key: str, at: str) -> bool:
        """Is the effect of this operation in place? The startup reconciler asks after a crash."""
        ...

    async def record_escalation(self, tool: str, args: dict[str, Any], *, key: str) -> None:
        """Escalations are database rows, not cluster changes; the simulator logs them next to
        its mutations so one log shows every side effect of a run. Other backends ignore it."""
        ...
