"""Selection rules for exact-string and factual contexts (pure, no database)."""

from __future__ import annotations

from typing import Any

from sre_evals.golden.contexts import MAX_DOC_FREQ, select_contexts


def _chunk(
    path: str, body: str, version: str = "1.37", doc_type: str = "reference"
) -> dict[str, Any]:
    return {
        "chunk_key": f"{path}@{version}:{abs(hash(body)) % 10**8}",
        "source_path": f"content/en/docs/{path}.md",
        "k8s_version": version,
        "url": f"https://kubernetes.io/docs/{path}/",
        "title": path,
        "body": body,
        "doc_type": doc_type,
    }


def test_boilerplate_literals_are_excluded() -> None:
    common = [_chunk(f"kubectl/cmd{i}", "Inherited: --client-key string") for i in range(5)]
    rare = _chunk("kubectl/drain", "Use --ignore-daemonsets to skip DaemonSet Pods.")
    rows = select_contexts([*common, rare], "exact_string", 10, seed=0)
    assert [r.matched for r in rows] == ["--ignore-daemonsets"]
    assert MAX_DOC_FREQ["flag"] < 5


def test_doc_frequency_is_counted_per_version() -> None:
    """The same page in three versions is one page per version, not three pages."""
    chunks = [
        _chunk("kubeadm/init", "Pass --cert-dir to move certificates.", v)
        for v in ("1.35", "1.36", "1.37")
    ]
    rows = select_contexts(chunks, "exact_string", 3, seed=0)
    assert [r.matched for r in rows] == ["--cert-dir"]  # and the page is used once


def test_kinds_take_turns() -> None:
    flags = [_chunk(f"flags/p{i}", f"Set --opt-number-{chr(97 + i)} for this.") for i in range(10)]
    field = _chunk("fields/p", "Read status.containerStatuses.restartCount for restarts.")
    error = _chunk("errors/p", "The Pod shows CrashLoopBackOff after repeated failures.")
    rows = select_contexts([*flags, field, error], "exact_string", 4, seed=3)
    kinds = sorted(r.match_kind or "" for r in rows)
    assert kinds == ["error", "field", "flag", "flag"]


def test_contribute_pages_are_skipped() -> None:
    chunks = [
        _chunk(
            "contribute/style",
            "Run git push --force-with-lease to update a PR.",
            doc_type="concept",
        ),
        _chunk("concepts/pods", "A Pod is the smallest deployable unit. " * 8, doc_type="concept"),
    ]
    assert {r.source_path for r in select_contexts(chunks, "factual", 5, seed=0)} == {
        "content/en/docs/concepts/pods.md"
    }
    assert select_contexts(chunks, "exact_string", 5, seed=0) == []


def test_factual_uses_a_page_once_across_versions_before_repeating() -> None:
    chunks = [
        _chunk(f"concepts/p{i}", f"Body {i} " * 30, v, doc_type="concept")
        for i in range(3)
        for v in ("1.35", "1.36", "1.37")
    ]
    rows = select_contexts(chunks, "factual", 3, seed=5)
    assert len({r.source_path for r in rows}) == 3
