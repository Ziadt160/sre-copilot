"""Cross-encoder reranking (Phase 2 experiment, design doc "Reranker").

In the lab the cross-encoder runs in the GPU server (`tools/gpu_embed_server.py`, `/rerank`),
so core stays torch-free; the demo would call Cloudflare Workers AI's bge-reranker-base, and
only if the lab shows a gain worth its latency.
"""

from __future__ import annotations

from collections.abc import Sequence

from sre_core.retrieval.base import RerankerUnavailable
from sre_shared.models import RetrievedChunk
from sre_shared.settings import get_settings

DEFAULT_MODEL = "BAAI/bge-reranker-base"


class RemoteReranker:
    """Scores (query, header + body) pairs with a cross-encoder on the GPU server."""

    def __init__(
        self, model: str = DEFAULT_MODEL, base_url: str | None = None, timeout: float = 10.0
    ) -> None:
        self.name = model
        self.model = model
        self.base_url = (base_url or get_settings().gpu_embed_url).rstrip("/")
        self.timeout = timeout

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        raise RerankerUnavailable("RemoteReranker is not implemented yet (Phase 2, T08)")
