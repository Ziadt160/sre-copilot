"""gte-small on ONNX: downloads ~130 MB from the Hub on first run (marker: models)."""

import numpy as np
import pytest

from sre_shared.embed import Embedder, get_embedder

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def gte() -> Embedder:
    return get_embedder("gte-small")


def test_shape_and_normalization(gte: Embedder) -> None:
    v = gte.embed_documents(["What does CrashLoopBackOff mean?", "kubectl rollout status"])
    assert v.shape == (2, 384) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)


def test_batch_equals_single(gte: Embedder) -> None:
    texts = ["short", "a somewhat longer sentence about Deployments and ReplicaSets " * 5]
    batch = gte.embed_documents(texts, batch_size=2)
    single = np.vstack([gte.embed_documents([t]) for t in texts])
    np.testing.assert_allclose(batch, single, atol=1e-5)


def test_output_order_follows_input_with_mixed_lengths(gte: Embedder) -> None:
    """Texts are embedded in length order internally; rows must come back in input order."""
    texts = ["x " * 300, "pod", "a medium sentence about Services " * 4, "node", "y " * 120]
    batch = gte.embed_documents(texts, batch_size=2)
    single = np.vstack([gte.embed_documents([t]) for t in texts])
    np.testing.assert_allclose(batch, single, atol=1e-5)


def test_semantics_are_sane(gte: Embedder) -> None:
    q, near, far = gte.embed_documents(
        [
            "Why does my pod keep restarting?",
            "A container in the pod repeatedly crashes and Kubernetes restarts it (CrashLoopBackOff).",
            "Chocolate cake recipe with buttercream frosting.",
        ]
    )
    assert float(q @ near) > float(q @ far)


def test_token_count_is_untruncated(gte: Embedder) -> None:
    long_text = "token " * 2000
    assert gte.count_tokens(long_text) == 2000
    assert gte.count_tokens("") == 0


def test_long_inputs_are_truncated_not_rejected(gte: Embedder) -> None:
    v = gte.embed_documents(["word " * 3000])
    assert v.shape == (1, 384)
