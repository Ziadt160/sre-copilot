from __future__ import annotations

import pytest

from sre_ingest.chunker import chunk_document, chunk_key, make_header
from sre_ingest.textutil import scan_lines
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import ChunkRecord, DocType, DocumentRecord

COUNT = FakeEmbedder("gte-small").count_tokens


def doc_of(text: str, path: str = "content/en/docs/concepts/x.md") -> DocumentRecord:
    return DocumentRecord(
        k8s_version="1.36",
        source_path=path,
        url="https://example.test/docs/concepts/x/",
        title="Title",
        doc_type=DocType.CONCEPT,
        content_hash="h",
        parsed_text=text,
    )


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def check_invariants(doc: DocumentRecord, chunks: list[ChunkRecord], limit: int) -> None:
    assert len({c.chunk_key for c in chunks}) == len(chunks)
    for i, c in enumerate(chunks):
        assert c.ordinal == i
        assert c.body == doc.parsed_text[c.char_start : c.char_end]
        assert COUNT(c.embedding_text) <= limit
        assert c.token_count == COUNT(c.embedding_text)
        assert c.header.startswith("Kubernetes v1.36 | Title")
    assert [c.char_start for c in chunks] == sorted(c.char_start for c in chunks)


def test_splits_by_heading_with_section_paths_and_anchors() -> None:
    text = (
        f"intro {words(60, 'i')}\n\n## Alpha\n\n{words(60, 'a')}\n\n### Beta {{#custom}}\n\n"
        f"{words(60, 'b')}\n\n## Gamma Delta\n\n{words(60, 'g')}"
    )
    doc = doc_of(text)
    chunks, _ = chunk_document(doc, COUNT, max_tokens=200, min_body_tokens=10)
    assert [c.section_path for c in chunks] == [[], ["Alpha"], ["Alpha", "Beta"], ["Gamma Delta"]]
    assert chunks[2].header == make_header("1.36", "Title", ["Alpha", "Beta"])
    assert chunks[2].header == "Kubernetes v1.36 | Title > Alpha > Beta"
    assert chunks[0].url == doc.url
    assert chunks[2].url.endswith("#custom")
    assert chunks[3].url.endswith("#gamma-delta")
    check_invariants(doc, chunks, 200)


def test_code_blocks_stay_whole_and_headings_in_fences_are_ignored() -> None:
    code = "```yaml\n# not a heading\nkey: value\n\nother: 1\n```"
    text = f"## Sec\n\n{words(30)}\n\n{code}\n\n{words(30, 'z')}"
    doc = doc_of(text)
    chunks, stats = chunk_document(doc, COUNT, max_tokens=60, min_body_tokens=1)
    assert stats.code_block_splits == 0
    assert [c.section_path for c in chunks] == [["Sec"]] * len(chunks)
    assert sum(code in c.body for c in chunks) == 1
    check_invariants(doc, chunks, 60)


def test_oversized_code_block_splits_at_line_boundaries_and_is_flagged() -> None:
    lines = [f"line{i}: value{i}" for i in range(40)]
    text = "## Sec\n\n```text\n" + "\n".join(lines) + "\n```"
    doc = doc_of(text)
    chunks, stats = chunk_document(doc, COUNT, max_tokens=40, min_body_tokens=1)
    assert stats.code_block_splits == 1
    assert len(chunks) > 1
    for c in chunks:
        for ln in c.body.splitlines():
            assert ln in lines or ln in {"```text", "```"}
    check_invariants(doc, chunks, 40)


def test_merge_forward_keeps_first_path_and_never_splits_blocks() -> None:
    code = "```sh\nkubectl get pods\n```"
    text = f"## A\n\nshort.\n\n## B\n\n{code}\n\n## C\n\n{words(50, 'c')}"
    doc = doc_of(text)
    chunks, stats = chunk_document(doc, COUNT, max_tokens=120, min_body_tokens=10)
    assert stats.merged >= 1
    assert chunks[0].section_path == ["A"]
    assert "## B" in chunks[0].body and code in chunks[0].body
    for c in chunks:
        assert c.body.count("```") % 2 == 0
    check_invariants(doc, chunks, 120)


def test_merge_never_crosses_documents_or_split_pieces() -> None:
    text = (
        "## A\n\n```text\n" + "\n".join(f"l{i} x{i}" for i in range(30)) + "\n```\n\n## B\n\ntiny"
    )
    doc = doc_of(text)
    chunks, _ = chunk_document(doc, COUNT, max_tokens=30, min_body_tokens=25)
    assert chunks[-1].body.strip().endswith("tiny")
    assert chunks[-1].section_path == ["B"]
    check_invariants(doc, chunks, 30)


def test_oversized_prose_is_split() -> None:
    text = "## Sec\n\n" + " ".join(f"word{i}" for i in range(400))
    doc = doc_of(text)
    chunks, stats = chunk_document(doc, COUNT, max_tokens=50, min_body_tokens=1)
    assert stats.prose_splits == 1
    assert len(chunks) >= 8
    check_invariants(doc, chunks, 50)


def test_duplicate_bodies_get_distinct_keys() -> None:
    block = words(40)
    text = f"## A\n\n{block}\n\n## B\n\n{block}"
    chunks, _ = chunk_document(doc_of(text), COUNT, max_tokens=80, min_body_tokens=1)
    assert len(chunks) == 2
    assert chunks[0].chunk_key != chunks[1].chunk_key


def test_chunk_key_normalizes_whitespace_and_depends_on_inputs() -> None:
    a = chunk_key("p.md", "1.36", "a  b\nc")
    assert a == chunk_key("p.md", "1.36", "a b c")
    assert a != chunk_key("q.md", "1.36", "a b c")
    assert a != chunk_key("p.md", "1.35", "a b c")


def test_scan_lines_fence_rules() -> None:
    text = "~~~\n```\n# inside\n~~~\n# real\n"
    lines = scan_lines(text)
    assert [ln.level for ln in lines] == [0, 0, 0, 0, 1]


@pytest.mark.models
def test_real_gte_small_budget_on_long_sections() -> None:
    from sre_shared.embed import get_embedder

    emb = get_embedder("gte-small")
    text = "## Sec\n\n" + " ".join(f"kubernetes pod container {i}" for i in range(900))
    doc = doc_of(text)
    chunks, _ = chunk_document(doc, emb.count_tokens)
    assert all(emb.count_tokens(c.embedding_text) <= 510 for c in chunks)
    assert all(c.body == doc.parsed_text[c.char_start : c.char_end] for c in chunks)
