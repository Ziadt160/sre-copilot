r"""BGE challengers through the lab GPU embedding server (marker: gpu).

Start the server first, with the Python that has torch:
    set HF_HUB_OFFLINE=1 && C:\Python314\python.exe tools\gpu_embed_server.py
then: uv run pytest -m gpu packages/shared/tests
"""

import numpy as np
import pytest

from sre_shared.embed import Embedder
from sre_shared.embed.base import SPECS
from sre_shared.embed.remote import GpuServerUnavailable, RemoteSTEmbedder

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", params=["bge-large", "bge-m3"])
def bge(request: pytest.FixtureRequest) -> RemoteSTEmbedder:
    e = RemoteSTEmbedder(SPECS[request.param])
    try:
        health = e.health()
    except GpuServerUnavailable as err:
        pytest.skip(str(err))
    assert health["device"] == "cuda"
    return e


def test_contract_and_semantics(bge: RemoteSTEmbedder) -> None:
    assert isinstance(bge, Embedder)
    docs = bge.embed_documents(
        ["A container repeatedly crashes and is restarted (CrashLoopBackOff).", "Cake recipe."]
    )
    (query,) = bge.embed_queries(["why does my pod keep restarting"])
    assert docs.shape == (2, 1024) and docs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-4)
    assert float(docs[0] @ query) > float(docs[1] @ query)


def test_batch_equals_single(bge: RemoteSTEmbedder) -> None:
    texts = ["kubectl rollout status deployment/web", "Pods are the smallest deployable units."]
    batch = bge.embed_documents(texts, batch_size=2)
    single = np.vstack([bge.embed_documents([t]) for t in texts])
    np.testing.assert_allclose(batch, single, atol=2e-3)  # fp16 inference


def test_tokens_and_truncation(bge: RemoteSTEmbedder) -> None:
    assert bge.count_tokens("token " * 600) >= 600
    before = bge.truncated_total
    bge.embed_documents(["token " * 3000])
    assert bge.truncated_total == before + 1


def test_empty_input(bge: RemoteSTEmbedder) -> None:
    assert bge.embed_documents([]).shape == (0, 1024)
