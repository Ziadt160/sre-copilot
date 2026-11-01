"""Typed loaders for the YAML files in `config/`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sre_shared.models import AnswerConfig
from sre_shared.settings import get_settings


def _load_yaml(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    path = (config_dir or get_settings().config_dir) / name
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping at the top level")
    return data


# --- limits.yaml ---------------------------------------------------------------


class ModelLimits(BaseModel):
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None


class Provider(BaseModel):
    model_config = ConfigDict(extra="allow")

    order: int | str
    base_url: str | None = None
    key_env: str | None = None
    allowed_for: list[str] = Field(default_factory=list)
    account: dict[str, int] = Field(default_factory=dict)
    models: dict[str, ModelLimits] | list[str] = Field(default_factory=list)


class Limits(BaseModel):
    providers: dict[str, Provider]
    services: dict[str, dict[str, int]] = Field(default_factory=dict)
    budgets: dict[str, float] = Field(default_factory=dict)

    def model_limits(self, provider: str, model: str) -> ModelLimits:
        models = self.providers[provider].models
        if isinstance(models, dict) and model in models:
            return models[model]
        return ModelLimits()


def load_limits(config_dir: Path | None = None) -> Limits:
    return Limits.model_validate(_load_yaml("limits.yaml", config_dir))


# --- retrieval.yaml -------------------------------------------------------------


class HnswConfig(BaseModel):
    ef_search: int = 100
    iterative_scan: Literal["off", "relaxed_order", "strict_order"] = "relaxed_order"
    max_scan_tuples: int = 20000


class RouteConfig(BaseModel):
    use_dense: bool = True
    use_keyword: bool = False
    w_dense: float = 1.0
    w_keyword: float = 1.0
    rrf_k: int = 60


class RerankDefaults(BaseModel):
    """Cross-encoder reranking of the top fused candidates (decision 33). Enabled only in the
    environments listed: the lab runs it on the GPU server; the demo needs the Workers AI
    parity check first (design doc, open questions)."""

    model: str = "BAAI/bge-reranker-base"
    revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    candidates: int = 20
    environments: list[str] = Field(default_factory=lambda: ["lab"])

    def enabled_for(self, env: str) -> bool:
        return env in self.environments


class RouterConfig(BaseModel):
    """Per-query-type retrieval (design doc: "the router flags exact-string questions and shifts
    weight to the keyword leg; conceptual questions lean dense")."""

    literal: RouteConfig = Field(
        default_factory=lambda: RouteConfig(use_keyword=True, w_keyword=0.5, rrf_k=10)
    )
    default: RouteConfig = Field(default_factory=RouteConfig)


class RetrievalDefaults(BaseModel):
    mode: Literal["dense", "hybrid", "keyword"] = "dense"
    embedding_model: str = "gte-small"
    ts_config: Literal["english", "simple"] = "english"
    keyword_match: Literal["any", "all"] = "any"
    candidates_per_leg: int = 50
    k: int = 5
    rrf_k: int = 60
    weights: dict[str, float] = Field(default_factory=lambda: {"dense": 1.0, "keyword": 1.0})
    hnsw: HnswConfig = Field(default_factory=HnswConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    rerank: RerankDefaults | None = None


def load_retrieval_defaults(config_dir: Path | None = None) -> RetrievalDefaults:
    return RetrievalDefaults.model_validate(_load_yaml("retrieval.yaml", config_dir))


# --- corpus.yaml ------------------------------------------------------------------


class CorpusSource(BaseModel):
    k8s_version: str = Field(description='Minor version, e.g. "1.34"')
    branch: str = Field(description='e.g. "release-1.34", or "main" for the current release')
    commit_sha: str | None = None
    url_base: str = Field(
        description="Site root for this version, e.g. https://v1-34.docs.kubernetes.io"
    )


class CorpusConfig(BaseModel):
    project: str = "kubernetes"
    repo: str
    docs_root: str = "content/en/docs"
    language: str = "en"
    license: str
    attribution: str
    versions: list[CorpusSource] = Field(default_factory=list)


def load_corpus_config(config_dir: Path | None = None) -> CorpusConfig:
    return CorpusConfig.model_validate(_load_yaml("corpus.yaml", config_dir))


# --- answer.yaml ------------------------------------------------------------------


def load_answer_config(config_dir: Path | None = None) -> AnswerConfig:
    return AnswerConfig.model_validate(_load_yaml("answer.yaml", config_dir))
