"""The Phase 2 retrieval experiment configs load and build the intended requests."""

from __future__ import annotations

import pytest

from sre_evals.runner import load_experiments
from sre_shared.models import RetrievalFilters

FILTERS = RetrievalFilters(k8s_versions=["1.37"])


def test_router_config() -> None:
    cfg = load_experiments()["dense-gte-small-router"]
    req = cfg.request_for("q", FILTERS)
    assert req.route_by_query_type is True
    assert req.rerank_candidates is None
    assert cfg.k == [3, 5, 8, 10]
    assert req.filters == FILTERS


@pytest.mark.parametrize("n", [20, 50])
def test_rerank_configs(n: int) -> None:
    cfg = load_experiments()[f"dense-gte-small-rerank{n}"]
    req = cfg.request_for("q", FILTERS)
    assert req.rerank_candidates == n
    assert req.candidates_per_leg >= n
    assert req.route_by_query_type is False
    assert cfg.rerank is not None and cfg.rerank.model == "BAAI/bge-reranker-base"
    assert cfg.k == [3, 5, 8, 10]


def test_iterscan_off_config() -> None:
    cfg = load_experiments()["dense-gte-small-iterscan-off"]
    req = cfg.request_for("q", FILTERS)
    assert req.iterative_scan == "off"
    assert req.ef_search == 40
    assert req.candidates_per_leg == 50
    assert req.use_dense and not req.use_keyword
    assert cfg.k == [3, 5, 8, 10]
