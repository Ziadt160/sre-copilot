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


@pytest.mark.parametrize("variant", ["t256", "o64", "nohdr"])
def test_chunking_configs_query_their_variant_with_and_without_the_reranker(variant: str) -> None:
    exps = load_experiments()
    dense, reranked = (
        exps[f"dense-gte-small-chunk-{variant}"],
        exps[f"dense-gte-small-rerank20-chunk-{variant}"],
    )
    label = f"k8s-1.37+1.36+1.35-chunker1-{variant}"
    assert dense.corpus_version == label and reranked.corpus_version == label
    assert dense.rerank is None
    assert reranked.request_for("q", FILTERS).rerank_candidates == 20
    # Apart from the reranker, the pair is the same experiment as its cv1 baselines.
    base, base_rr = exps["dense-gte-small"], exps["dense-gte-small-rerank20"]
    skip = {"name", "description", "corpus_version"}
    assert dense.model_dump(exclude=skip) == base.model_dump(exclude=skip)
    assert reranked.model_dump(exclude=skip) == base_rr.model_dump(exclude=skip)
