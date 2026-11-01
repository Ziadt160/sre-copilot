from __future__ import annotations

import httpx
import numpy as np

from sre_core.retrieval import EdgeFunctionEmbedder


def test_edge_embedder_posts_texts_and_normalizes() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={"embeddings": [[2.0] + [0.0] * 383]})

    emb = EdgeFunctionEmbedder(
        url="http://edge/embed", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    out = emb.embed_queries(["hello"])
    assert seen["body"] == b'{"texts":["hello"]}'
    assert out.shape == (1, 384)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
