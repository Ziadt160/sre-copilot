"""Cross-encoder reranking (Phase 2 experiment, design doc "Reranker").

In the lab the cross-encoder runs in the GPU server (`tools/gpu_embed_server.py`, `/rerank`),
so core stays torch-free; the demo would call Cloudflare Workers AI's bge-reranker-base, and
only if the lab shows a gain worth its latency.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from sre_core.retrieval.base import RerankerUnavailable
from sre_shared.config import load_retrieval_defaults
from sre_shared.models import RetrievedChunk
from sre_shared.settings import get_settings


def default_reranker() -> RemoteReranker | None:
    """The reranker `config/retrieval.yaml` enables for this environment, or None."""
    rerank = load_retrieval_defaults().rerank
    if rerank is None or not rerank.enabled_for(get_settings().sre_env):
        return None
    return RemoteReranker(rerank.model, revision=rerank.revision)


DEFAULT_MODEL = "BAAI/bge-reranker-base"
MAX_LENGTH = 512


class RemoteReranker:
    """Scores (query, header + body) pairs with a cross-encoder on the GPU server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = 10.0,
        revision: str = "main",
        batch_size: int = 32,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.revision = revision
        self.batch_size = batch_size
        self.base_url = (base_url or get_settings().gpu_embed_url).rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        if not chunks:
            return []
        payload = {
            "hf_id": self.model,
            "revision": self.revision,
            "query": query,
            "texts": [f"{c.header}\n\n{c.body}" for c in chunks],
            "max_length": MAX_LENGTH,
            "batch_size": self.batch_size,
        }
        try:
            resp = await self._client.post(
                f"{self.base_url}/rerank", json=payload, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise RerankerUnavailable(f"rerank request failed: {exc!r}") from exc
        if resp.status_code != 200:
            raise RerankerUnavailable(f"rerank returned HTTP {resp.status_code}")
        try:
            out = [float(s) for s in resp.json()["scores"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise RerankerUnavailable(f"rerank returned a malformed body: {exc!r}") from exc
        if len(out) != len(chunks):
            raise RerankerUnavailable(f"rerank returned {len(out)} scores for {len(chunks)} texts")
        return out
