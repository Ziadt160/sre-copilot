"""Query embedder backed by the Supabase Edge Function `embed` (gte-small), used in the demo
environment where the API host cannot run ONNX."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import numpy as np

from sre_shared.embed.base import SPECS, EmbeddingSpec, Vectors, l2_normalize
from sre_shared.settings import get_settings


class EdgeFunctionEmbedder:
    """Implements the `Embedder` protocol for queries only (gte-small, no query prefix)."""

    def __init__(
        self,
        url: str | None = None,
        anon_key: str | None = None,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        self.spec: EmbeddingSpec = SPECS["gte-small"]
        self._url = url or f"{settings.supabase_url.rstrip('/')}/functions/v1/embed"
        key = anon_key
        if key is None and settings.supabase_anon_key is not None:
            key = settings.supabase_anon_key.get_secret_value()
        headers = {"Authorization": f"Bearer {key}", "apikey": key} if key else {}
        self._client = client or httpx.Client(timeout=timeout, headers=headers)

    @property
    def revision(self) -> str:
        return "supabase-edge-gte-small"

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError("the edge embedder only embeds queries")

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        raise NotImplementedError("the edge embedder only embeds queries")

    def embed_queries(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        if not texts:
            return np.zeros((0, self.spec.dims), dtype=np.float32)
        payload = {"texts": [self.spec.query_prefix + t for t in texts]}
        resp = self._client.post(self._url, json=payload)
        resp.raise_for_status()
        arr = np.asarray(resp.json()["embeddings"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.spec.dims or arr.shape[0] != len(texts):
            raise ValueError(f"edge embed returned shape {arr.shape}")
        return l2_normalize(arr)
