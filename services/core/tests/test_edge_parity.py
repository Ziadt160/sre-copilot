"""Edge Function `embed` vs the Python ONNX embedder (lab only: needs `npx supabase start` with
functions served, and `supabase_anon_key` in the environment). Written to be run by the
orchestrator after merge, because the running stack serves functions from the main checkout."""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from sre_core.retrieval import EdgeFunctionEmbedder
from sre_shared.settings import get_settings

pytestmark = [pytest.mark.lab, pytest.mark.models]

WORDS = [
    "pod",
    "container",
    "kubelet",
    "scheduler",
    "node",
    "cluster",
    "service",
    "ingress",
    "volume",
    "secret",
    "namespace",
    "deployment",
    "replica",
    "statefulset",
    "daemonset",
    "probe",
    "liveness",
    "readiness",
    "taint",
    "toleration",
    "affinity",
    "etcd",
    "apiserver",
    "controller",
    "manager",
    "certificate",
    "rotation",
    "admission",
    "webhook",
    "quota",
    "limit",
]


def make_texts(onnx_count: object) -> list[str]:
    rng = np.random.default_rng(7)
    texts = [
        "How do I restart a pod?",
        "kubectl rollout status deployment/nginx",
        "ImagePullBackOff",
        "Kubernetes v1.31 | Pods > Lifecycle\n\nA Pod's status field is a PodStatus object.",
    ]
    while len(texts) < 45:
        n = int(rng.integers(3, 120))
        texts.append(" ".join(rng.choice(WORDS, size=n)))
    # Near-limit texts: grow until just under 512 tokens (special tokens included).
    for target in (380, 460, 490, 505, 509):
        words: list[str] = []
        while onnx_count(" ".join(words)) < target:  # type: ignore[operator]
            words.append(str(rng.choice(WORDS)))
        texts.append(" ".join(words))
    texts += [
        "x " * 700,  # over the limit: both sides truncate
        "naive café — ünïcode 日本語 テスト",
        "line one\nline two\n\n```yaml\nkind: Pod\n```",
    ]
    return texts


def test_edge_function_matches_onnx() -> None:
    from sre_shared.embed.onnx_gte import OnnxGteSmall

    settings = get_settings()
    assert settings.supabase_anon_key is not None, "set SUPABASE_ANON_KEY for the lab stack"
    onnx = OnnxGteSmall()
    texts = make_texts(onnx.count_tokens)
    assert len(texts) >= 50

    edge = EdgeFunctionEmbedder(timeout=120.0)
    try:
        got = np.concatenate(
            [edge.embed_queries(texts[i : i + 16]) for i in range(0, len(texts), 16)]
        )
    except httpx.HTTPError as exc:  # pragma: no cover - lab only
        pytest.fail(f"edge function unreachable or failing: {exc}")
    want = onnx.embed_queries(texts)

    cos = np.sum(got * want, axis=1)  # both L2-normalized
    worst = int(np.argmin(cos))
    assert cos.min() >= 0.99, (
        f"worst cosine {cos.min():.4f} on text #{worst}: {texts[worst][:80]!r}"
    )
