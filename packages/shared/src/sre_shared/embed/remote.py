"""Client for the lab GPU embedding server (`tools/gpu_embed_server.py`).

The BGE challengers run in a separate Python that already has torch; this process only speaks
HTTP to it, so no service, test or CI job here ever installs torch.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from sre_shared.embed.base import EmbeddingSpec, Vectors
from sre_shared.settings import get_settings

# bge-m3 accepts 8192 tokens, but chunks are capped at 510 gte-small tokens; 1024 leaves room
# for its different tokenizer while keeping activations inside 8 GB of VRAM.
DEFAULT_MAX_SEQ = {"bge-large": 512, "bge-m3": 1024}


class GpuServerUnavailable(RuntimeError):
    pass


class RemoteSTEmbedder:
    """Embedder protocol over HTTP. `truncated_total` counts inputs cut at max_seq_length."""

    def __init__(
        self,
        spec: EmbeddingSpec,
        base_url: str | None = None,
        max_seq_length: int | None = None,
        timeout: float = 600.0,
    ) -> None:
        if spec.revision is None:
            raise ValueError(f"{spec.name}: pin a revision in SPECS before using the GPU server")
        self.spec = spec
        self.max_seq_length = max_seq_length or DEFAULT_MAX_SEQ.get(spec.name, spec.max_tokens)
        self.truncated_total = 0
        self._client = httpx.Client(
            base_url=base_url or get_settings().gpu_embed_url, timeout=timeout
        )

    @property
    def revision(self) -> str:
        assert self.spec.revision is not None
        return self.spec.revision

    def health(self) -> dict[str, Any]:
        try:
            resp = self._client.get("/health", timeout=5.0)
        except httpx.TransportError as e:
            raise GpuServerUnavailable(
                f"GPU embed server not reachable at {self._client.base_url}; start it with "
                r"`C:\Python314\python.exe tools\gpu_embed_server.py`"
            ) from e
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        return body

    def count_tokens(self, text: str) -> int:
        body = self._post(
            "/count",
            {"texts": [text], "max_seq_length": self.max_seq_length},
        )
        return int(body["counts"][0])

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        return self._embed(list(texts), "document", batch_size)

    def embed_queries(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        return self._embed(list(texts), "query", batch_size)

    def _embed(self, texts: list[str], kind: str, batch_size: int) -> Vectors:
        if not texts:
            return np.zeros((0, self.spec.dims), dtype=np.float32)
        body = self._post(
            "/embed",
            {
                "texts": texts,
                "kind": kind,
                "query_prefix": self.spec.query_prefix,
                "max_seq_length": self.max_seq_length,
                "batch_size": batch_size,
            },
        )
        arr = np.frombuffer(base64.b64decode(body["data"]), dtype="<f4")
        vecs = arr.reshape(int(body["count"]), int(body["dims"])).astype(np.float32)
        if vecs.shape[1] != self.spec.dims:
            raise ValueError(f"{self.spec.name}: server returned {vecs.shape[1]} dims")
        self.truncated_total += int(body.get("truncated", 0))
        return vecs

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        full = {"hf_id": self.spec.hf_id, "revision": self.revision, **payload}
        try:
            resp = self._client.post(path, json=full)
        except httpx.TransportError as e:
            raise GpuServerUnavailable(
                f"GPU embed server not reachable at {self._client.base_url}"
            ) from e
        if resp.status_code != 200:
            raise RuntimeError(f"GPU embed server {path}: {resp.status_code} {resp.text[:300]}")
        body: dict[str, Any] = resp.json()
        return body
