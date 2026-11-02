from pathlib import Path

import pytest

from sre_shared.config import load_corpus_config, load_limits, load_retrieval_defaults


def test_limits_load() -> None:
    limits = load_limits()
    groq = limits.model_limits("groq", "openai/gpt-oss-20b")
    assert groq.tpd == 200_000
    assert limits.model_limits("groq", "nope").tpd is None


def test_retrieval_defaults_load() -> None:
    d = load_retrieval_defaults()
    assert d.rrf_k == 60
    assert d.hnsw.ef_search >= 1


def test_corpus_config_load() -> None:
    c = load_corpus_config()
    assert c.license == "CC-BY-4.0"


def test_phase3_config_files_load_and_cross_check(tmp_path: Path) -> None:
    from sre_shared.config import load_agent_config, load_mcp_ops_config, load_simulator_scenarios

    agent = load_agent_config()
    assert (agent.limits.llm_decisions_per_turn, agent.limits.tool_calls_per_turn) == (6, 2)
    assert agent.approval.ttl_minutes == 15 and agent.complex.max_subquestions == 3
    ops = load_mcp_ops_config()
    policy = ops.policy()
    assert policy.protected <= policy.allowed
    scenarios = load_simulator_scenarios()
    default = scenarios.scenarios[ops.simulator.scenario]
    assert set(default.namespaces) <= policy.allowed  # every simulated namespace is reachable
    # A protected namespace that is not allowed is a config error, not a silent gap.
    bad = tmp_path / "mcp_ops.yaml"
    bad.write_text(
        "namespaces: {allowed: [shop], protected: [payments]}\nbackend: simulator\n"
        "simulator: {scenario: default, seed: 1}\nlogs: {max_bytes: 10}\ntasks: {store: memory://}\n"
        "rotations: [sre-primary]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protected namespaces not on the allow-list"):
        load_mcp_ops_config(tmp_path)
