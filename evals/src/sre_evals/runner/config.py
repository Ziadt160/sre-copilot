"""Experiment configs: `config/experiments/<name>.yaml`.

Schema (all keys optional except `name`):

    name: hybrid-gte-small          # must equal the file stem
    description: free text
    embedding_model: gte-small      # gte-small | bge-large | bge-m3
    legs: [dense, keyword]          # any non-empty subset
    ts_config: english              # english | simple
    keyword_match: any              # any | all query terms
    weights: {dense: 1.0, keyword: 1.0}
    k: [3, 5, 8, 10]                # reporting cutoffs; retrieval runs once at max(k)
    candidates_per_leg: 50
    rrf_k: 60
    ef_search: 100
    iterative_scan: relaxed_order   # off | relaxed_order | strict_order

Filters policy is fixed: each golden item's own `filters` are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sre_shared.models import (
    AnswerConfig,
    EmbeddingModelName,
    IterativeScan,
    KeywordMatch,
    RetrievalFilters,
    RetrievalRequest,
    TsConfig,
)
from sre_shared.settings import get_settings

DEFAULT_KS = (3, 5, 8, 10)
Leg = Literal["dense", "keyword"]


def _both_legs() -> list[Leg]:
    return ["dense", "keyword"]


class Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense: float = Field(default=1.0, ge=0)
    keyword: float = Field(default=1.0, ge=0)


class RerankConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "BAAI/bge-reranker-base"
    # Pinned: the GPU server loads from the local cache only, where this snapshot is the one
    # downloaded (2026-11-02); "main" does not resolve offline.
    revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    candidates: int = Field(default=20, ge=1, le=200, description="fused candidates reranked")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    embedding_model: EmbeddingModelName = "gte-small"
    legs: list[Leg] = Field(default_factory=_both_legs)
    ts_config: TsConfig = "english"
    keyword_match: KeywordMatch = "any"
    weights: Weights = Field(default_factory=Weights)
    k: list[int] = Field(default_factory=lambda: list(DEFAULT_KS))
    candidates_per_leg: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1)
    ef_search: int = Field(default=100, ge=1, le=1000)
    iterative_scan: IterativeScan = "relaxed_order"

    @field_validator("k")
    @classmethod
    def _ks(cls, v: list[int]) -> list[int]:
        if not v or any(k < 1 or k > 100 for k in v):
            raise ValueError("k must be a non-empty list of cutoffs in 1..100")
        return sorted(set(v))

    @model_validator(mode="after")
    def _legs(self) -> ExperimentConfig:
        if not self.legs or len(set(self.legs)) != len(self.legs):
            raise ValueError("legs must be a non-empty list without duplicates")
        return self

    @property
    def uses_dense(self) -> bool:
        return "dense" in self.legs

    @property
    def uses_keyword(self) -> bool:
        return "keyword" in self.legs

    @property
    def max_k(self) -> int:
        return max(self.k)

    # Phase 2 knobs. `corpus_version` is a corpus label (e.g. a chunking variant); the
    # runner resolves it to `resolved_corpus_version_id` so variants need no activation.
    corpus_version: str | None = None
    resolved_corpus_version_id: int | None = Field(default=None, exclude=True)
    route_by_query_type: bool = False
    rerank: RerankConfig | None = None
    answer: AnswerConfig | None = None

    def request_for(
        self, query: str, filters: RetrievalFilters | None, **overrides: object
    ) -> RetrievalRequest:
        """The retrieval request for one query: this config's knobs plus the item's filters."""
        fields: dict[str, object] = {
            "query": query,
            "filters": filters or RetrievalFilters(),
            "embedding_model": self.embedding_model,
            "use_dense": self.uses_dense,
            "use_keyword": self.uses_keyword,
            "ts_config": self.ts_config,
            "keyword_match": self.keyword_match,
            "candidates_per_leg": self.candidates_per_leg,
            "k": self.max_k,
            "w_dense": self.weights.dense,
            "w_keyword": self.weights.keyword,
            "rrf_k": self.rrf_k,
            "ef_search": self.ef_search,
            "iterative_scan": self.iterative_scan,
            "route_by_query_type": self.route_by_query_type,
            "rerank_candidates": self.rerank.candidates if self.rerank else None,
            "corpus_version_id": self.resolved_corpus_version_id,
        }
        fields.update(overrides)
        return RetrievalRequest.model_validate(fields)


def experiments_dir(config_dir: Path | None = None) -> Path:
    return (config_dir or get_settings().config_dir) / "experiments"


def load_experiment(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = ExperimentConfig.model_validate(data)
    if cfg.name != path.stem:
        raise ValueError(f"{path}: name {cfg.name!r} must equal the file stem {path.stem!r}")
    return cfg


def load_experiments(config_dir: Path | None = None) -> dict[str, ExperimentConfig]:
    """All experiments, sorted by name."""
    out: dict[str, ExperimentConfig] = {}
    for path in sorted(experiments_dir(config_dir).glob("*.yaml")):
        cfg = load_experiment(path)
        out[cfg.name] = cfg
    return out
