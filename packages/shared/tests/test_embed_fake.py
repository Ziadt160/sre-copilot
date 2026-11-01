import numpy as np

from sre_shared.embed import Embedder
from sre_shared.embed.fake import FakeEmbedder


def test_fake_embedder_contract() -> None:
    e = FakeEmbedder("gte-small")
    assert isinstance(e, Embedder)
    v = e.embed_documents(["pod restarts with CrashLoopBackOff", "", "kube-proxy flags"])
    assert v.shape == (3, 384) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)


def test_fake_embedder_similarity_tracks_shared_words() -> None:
    e = FakeEmbedder("gte-small")
    a, b, c = e.embed_documents(
        ["pod restarts CrashLoopBackOff", "pod restarts CrashLoopBackOff", "service mesh ingress"]
    )
    assert float(a @ b) > 0.999
    assert float(a @ c) < 0.5
