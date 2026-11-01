"""Retriever interface. Phase 1 contract between retrieval (T02) and the eval runner (T04).

The signature below is fixed; the implementation fills in the body.
"""

from __future__ import annotations

from collections.abc import Mapping

from psycopg_pool import AsyncConnectionPool

from sre_shared.embed import Embedder
from sre_shared.models import (
    ANONYMOUS,
    Claims,
    EmbeddingModelName,
    RetrievalRequest,
    RetrievalResult,
)


class Retriever:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedders: Mapping[EmbeddingModelName, Embedder] | None = None,
    ) -> None:
        """`embedders` maps model names to query embedders; missing ones load on first use
        via `sre_shared.embed.get_embedder`."""
        self._pool = pool
        self._embedders = dict(embedders or {})

    async def retrieve(
        self,
        req: RetrievalRequest,
        claims: Claims = ANONYMOUS,
        *,
        exact: bool = False,
    ) -> RetrievalResult:
        """Hybrid retrieval under the caller's RLS context.

        `exact=True` runs the dense leg as an exact scan (index scans off) so evals can measure
        ANN recall against it; production never sets it.
        """
        raise NotImplementedError
