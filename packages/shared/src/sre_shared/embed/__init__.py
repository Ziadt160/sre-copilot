"""Embedding models: gte-small on ONNX (production) and the lab-only BGE challengers,
which run in the GPU embedding server (tools/gpu_embed_server.py) under a separate torch Python."""

from __future__ import annotations

from functools import lru_cache

from sre_shared.embed.base import EMBEDDING_COLUMNS, SPECS, Embedder, EmbeddingSpec, Vectors
from sre_shared.models import EmbeddingModelName

__all__ = ["EMBEDDING_COLUMNS", "SPECS", "Embedder", "EmbeddingSpec", "Vectors", "get_embedder"]


@lru_cache(maxsize=4)
def get_embedder(name: EmbeddingModelName) -> Embedder:
    """Load (once per process) the embedder for `name`."""
    spec = SPECS[name]
    if spec.backend == "onnx":
        from sre_shared.embed.onnx_gte import OnnxGteSmall

        return OnnxGteSmall(spec)
    from sre_shared.embed.remote import RemoteSTEmbedder

    return RemoteSTEmbedder(spec)
