"""What the runner needs from a wired-up agent, and the faults it can ask for.

The lab and CI build a rig from the real pieces (the T12 graph over the T11 simulator); tests and
the PR subset build one from fakes (`sre_evals.agent.fakes`). The runner sees only this surface.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sre_core.agent.ports import ApprovalStore, ChatAgent, ThreadStore
from sre_evals.agent.cluster import ClusterState
from sre_evals.agent.items import AgentItem
from sre_evals.runner.engine import RetrieverLike


@dataclass(frozen=True)
class FaultPlan:
    """The three injected faults of the Phase 3 gate (design doc p18-19)."""

    dup_write: bool = False  # each write is sent twice and the first reply is dropped
    redeliver: bool = False  # after a write ran, its task body is delivered again
    double_approval: bool = False  # two approvals of one request arrive together

    @property
    def any(self) -> bool:
        return self.dup_write or self.redeliver or self.double_approval

    @classmethod
    def all(cls) -> FaultPlan:
        return cls(dup_write=True, redeliver=True, double_approval=True)


@dataclass
class AgentRig:
    agent: ChatAgent
    threads: ThreadStore
    approvals: ApprovalStore
    cluster: ClusterState
    # For multi-turn retrieval recall of the rewritten question; None skips that metric.
    retriever: RetrieverLike | None = None
    # Injection planting: a log line in the cluster, a passage among the retrieved chunks.
    plant_log: Callable[[str, str, str], None] | None = None
    plant_chunk: Callable[[str], None] | None = None
    # Delivers the last executed task body again (fault `redeliver`).
    redeliver: Callable[[], Awaitable[None]] | None = None


# Builds a fresh rig for one item (fresh stores, a reset cluster); faults are applied by the
# factory because only it can wrap the ops client the agent was built with.
RigFactory = Callable[[AgentItem, FaultPlan], Awaitable[AgentRig]]
