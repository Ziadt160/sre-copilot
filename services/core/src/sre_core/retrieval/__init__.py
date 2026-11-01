"""Hybrid retrieval: dense (pgvector HNSW) + keyword (Postgres full-text), fused with weighted RRF."""

from sre_core.retrieval.edge import EdgeFunctionEmbedder
from sre_core.retrieval.fusion import Candidate, Fused, fuse_rrf
from sre_core.retrieval.service import NoActiveCorpusError, Retriever

__all__ = [
    "Candidate",
    "EdgeFunctionEmbedder",
    "Fused",
    "NoActiveCorpusError",
    "Retriever",
    "fuse_rrf",
]
