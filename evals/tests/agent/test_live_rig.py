"""The parts of the lab rig that need no model or server: its config and its cluster view."""

from __future__ import annotations

from typing import get_args

from sre_core.agent.ports import ModelRole
from sre_evals.agent.cluster import ClusterState
from sre_evals.agent.live import SimCluster, lab_config
from sre_mcp_ops.backends.simulator import SimulatorBackend
from sre_shared.config import load_agent_config


def test_lab_config_puts_every_role_on_the_lab_model_without_hosted_fallback() -> None:
    cfg = lab_config(load_agent_config())
    for role in get_args(ModelRole):
        assert getattr(cfg.models, role) == cfg.models.lab
    assert cfg.fallback.providers == []
    assert cfg.fallback.extractive_last is True


async def test_sim_cluster_reports_mutations_in_the_runner_shape() -> None:
    sim = SimulatorBackend("default", 7)
    cluster = SimCluster(sim)
    assert isinstance(cluster, ClusterState)
    cluster.reset("default", 7)
    at = "2026-11-02T00:00:00+00:00"
    await sim.restart("shop", "checkout", key="k1", at=at)
    await sim.restart("shop", "checkout", key="k1", at=at)  # a second apply is still logged
    muts = cluster.mutations()
    assert [m["idempotency_key"] for m in muts] == ["k1", "k1"]
    assert set(muts[0]) == {"tool", "namespace", "target", "args", "idempotency_key", "changed"}
    assert "shop/checkout" in cluster.snapshot()["deployments"]
