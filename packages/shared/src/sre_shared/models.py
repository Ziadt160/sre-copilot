"""Data contracts shared across services.

These shapes are the interfaces between ingest, retrieval and evals. Change them only in the
shared package, in one place, with every consumer updated in the same change.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EmbeddingModelName = Literal["gte-small", "bge-large", "bge-m3"]
TsConfig = Literal["english", "simple"]
KeywordMatch = Literal["any", "all"]
IterativeScan = Literal["off", "relaxed_order", "strict_order"]


class DocType(StrEnum):
    CONCEPT = "concept"
    TASK = "task"
    REFERENCE = "reference"
    TUTORIAL = "tutorial"
    OTHER = "other"


class CorpusStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    RETIRED = "retired"


# --- identity -----------------------------------------------------------------------


class Claims(BaseModel):
    """The caller's identity as RLS sees it (`request.jwt.claims`).

    `groups` lands in `app_metadata.groups`; every caller can also read the `public` group.
    """

    sub: str
    role: Literal["anon", "authenticated"] = "authenticated"
    groups: list[str] = Field(default_factory=list)

    def to_jwt_claims(self) -> dict[str, Any]:
        return {"sub": self.sub, "role": self.role, "app_metadata": {"groups": self.groups}}


ANONYMOUS = Claims(sub="anonymous", role="anon")


# --- ingest -> database -----------------------------------------------------------------


class DocumentRecord(BaseModel):
    """One parsed source page for one Kubernetes version (table `documents`)."""

    project: str = "kubernetes"
    k8s_version: str
    source_path: str = Field(
        description="Path in the docs repo, POSIX style, e.g. content/en/docs/concepts/workloads/pods/_index.md"
    )
    url: str
    title: str
    doc_type: DocType
    language: str = "en"
    last_modified: datetime | None = None
    access_group: str = "public"
    content_hash: str
    parsed_text: str = Field(
        description="Normalized text after shortcode handling; gold quotes are validated against it"
    )


class ChunkRecord(BaseModel):
    """One retrievable chunk (table `chunks`). Embeddings are written separately, per model."""

    chunk_key: str = Field(description="sha256(source_path, k8s_version, normalized text)")
    source_path: str
    k8s_version: str
    ordinal: int = Field(description="Position of the chunk within its document")
    project: str = "kubernetes"
    doc_type: DocType
    title: str
    section_path: list[str] = Field(description="Heading trail, outermost first")
    url: str = Field(description="URL including the #anchor of the nearest heading when known")
    language: str = "en"
    access_group: str = "public"
    header: str = Field(
        description="Context header prepended for embedding: title, section path, version"
    )
    body: str
    char_start: int = Field(description="Offset of the body in DocumentRecord.parsed_text")
    char_end: int
    token_count: int = Field(
        description="gte-small tokens of header + body, special tokens excluded"
    )

    @property
    def embedding_text(self) -> str:
        return f"{self.header}\n\n{self.body}"


# --- retrieval --------------------------------------------------------------------------


class RetrievalFilters(BaseModel):
    """Filters applied inside SQL. Access groups are never a request field: they come from Claims."""

    k8s_versions: list[str] | None = None
    doc_types: list[DocType] | None = None


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    embedding_model: EmbeddingModelName = "gte-small"
    use_dense: bool = True
    use_keyword: bool = True
    ts_config: TsConfig = "english"
    route_by_query_type: bool = Field(
        default=False,
        description="pick legs/weights per query: a literal (error, flag, field path) uses the"
        " `router.literal` route in config/retrieval.yaml, anything else `router.default`",
    )
    rerank_candidates: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="rerank this many fused candidates before keeping k (None = no reranker)",
    )
    keyword_match: KeywordMatch = Field(
        default="any", description="any (OR of query terms, ranked) | all (every term must match)"
    )
    candidates_per_leg: int = Field(default=50, ge=1, le=500)
    k: int = Field(default=5, ge=1, le=100)
    w_dense: float = Field(default=1.0, ge=0)
    w_keyword: float = Field(default=1.0, ge=0)
    rrf_k: int = Field(default=60, ge=1)
    ef_search: int = Field(default=100, ge=1, le=1000)
    iterative_scan: IterativeScan = "relaxed_order"
    corpus_version_id: int | None = Field(default=None, description="None means the active version")

    @model_validator(mode="after")
    def _one_leg(self) -> RetrievalRequest:
        if not (self.use_dense or self.use_keyword):
            raise ValueError("at least one retrieval leg must be enabled")
        return self


class RetrievedChunk(BaseModel):
    chunk_id: int
    chunk_key: str
    source_path: str
    k8s_version: str
    doc_type: DocType
    title: str
    section_path: list[str]
    url: str
    header: str
    body: str
    score: float = Field(
        description="Fused RRF score; equals the single leg's RRF term when one leg is off"
    )
    dense_rank: int | None = Field(
        default=None, description="1-based rank in the dense leg, None if absent"
    )
    keyword_rank: int | None = Field(
        default=None, description="1-based rank in the keyword leg, None if absent"
    )
    dense_similarity: float | None = Field(
        default=None, description="Inner product of normalized vectors"
    )
    keyword_score: float | None = Field(default=None, description="ts_rank_cd")


class RetrievalResult(BaseModel):
    request: RetrievalRequest
    corpus_version_id: int
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = Field(description="embed, sql, fuse, total (+ rerank)")
    route: str | None = Field(default=None, description="router decision when routing was on")
    reranked: bool = False


# --- golden set --------------------------------------------------------------------------

# --- answers -----------------------------------------------------------------------------

CompressionMode = Literal["none", "sentence"]


class AnswerConfig(BaseModel):
    """Answer-path knobs. Defaults live in config/answer.yaml (`load_answer_config`); every
    value is to be set by an experiment, and eval runs stamp the config they used."""

    model: str = Field(
        default="groq:openai/gpt-oss-20b",
        description="provider:name, see sre_shared.llm.parse_model_spec",
    )
    k: int = Field(default=5, ge=1, le=20, description="chunks given to the model")
    relevance_floor: float = Field(
        default=0.0,
        description="abstain when the best chunk's dense similarity is below this",
    )
    max_tokens: int = Field(default=512, ge=16, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    compression: CompressionMode = "none"
    context_token_budget: int = Field(default=2500, ge=100, description="tokens of chunk text")
    prompt_version: str = "answer-v1"


class AnswerRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1, max_length=4000)
    filters: RetrievalFilters = Field(default_factory=lambda: RetrievalFilters())
    config: AnswerConfig | None = Field(default=None, description="None = config/answer.yaml")
    corpus_version_id: int | None = None


class Citation(BaseModel):
    """A source the answer cites: one of the chunks given to the model, by id."""

    chunk_id: int
    source_path: str
    k8s_version: str
    url: str
    title: str


class AnswerResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    invalid_citations: list[str] = Field(
        default_factory=list, description="cited ids outside the given chunks (rejected)"
    )
    abstained: bool = False
    context: list[RetrievedChunk] = Field(description="chunks given to the model, in order")
    config: AnswerConfig
    model: str
    corpus_version_id: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_tokens: int = 0
    timings_ms: dict[str, float] = Field(description="retrieve, compress, generate, total")


Stratum = Literal[
    "factual",
    "exact_string",
    "cross_version",
    "multi_turn",
    "action",
    "unanswerable",
    "permission",
    "injection",
]


class GoldPassage(BaseModel):
    """Evidence as a source passage, never a chunk id, so metrics survive chunking changes."""

    source_path: str
    k8s_version: str
    quote: str = Field(min_length=1, description="Verbatim span of DocumentRecord.parsed_text")


class GoldenItem(BaseModel):
    id: str = Field(description='Stable id, e.g. "fac-0001"')
    stratum: Stratum
    question: str
    expected_answer: str | None = None
    gold_passages: list[GoldPassage] = Field(default_factory=list)
    filters: RetrievalFilters | None = None
    origin: Literal["synthetic", "rewritten", "hand"] = "synthetic"
    status: Literal["candidate", "accepted", "rejected"] = "candidate"
    synth_model: str | None = None
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
