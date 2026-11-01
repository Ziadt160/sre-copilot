"""Chunking variants (config/chunking/*.yaml): the default must reproduce version 1 exactly."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from sre_ingest.chunker import CHUNKER_VERSION, MAX_TOKENS, chunk_document
from sre_ingest.chunking_config import load_chunking
from sre_ingest.parser import ParseContext, parse_document
from sre_ingest.pipeline import version_label
from sre_ingest.testsite import make_site
from sre_shared.config import CorpusSource
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import ChunkRecord, DocType, DocumentRecord

COUNT = FakeEmbedder("gte-small").count_tokens
SRC = CorpusSource(
    k8s_version="1.36",
    branch="release-1.36",
    commit_sha="1" * 40,
    url_base="https://v1-36.docs.kubernetes.io",
)
GOLDEN = Path(__file__).parent / "golden" / "default_chunks_golden.json"


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def doc_of(text: str) -> DocumentRecord:
    return DocumentRecord(
        k8s_version="1.36",
        source_path="content/en/docs/concepts/x.md",
        url="https://example.test/docs/concepts/x/",
        title="Title",
        doc_type=DocType.CONCEPT,
        content_hash="h",
        parsed_text=text,
    )


def long_doc() -> DocumentRecord:
    """Same text as the script that captured the golden file from the pre-variant chunker."""
    parts = ["intro " + " ".join(f"i{i}" for i in range(700))]
    parts.append(
        "## Alpha\n\n" + "\n\n".join(" ".join(f"a{j}x{i}" for i in range(90)) for j in range(12))
    )
    parts.append(
        "## Code\n\n```yaml\n"
        + "\n".join(f"key{i}: value{i} more words here" for i in range(300))
        + "\n```"
    )
    parts.append(
        "## Tiny\n\nshort.\n\n## Tiny2\n\nalso short.\n\n### Deep\n\n"
        + " ".join(f"d{i}" for i in range(40))
    )
    doc = doc_of("\n\n".join(parts))
    return doc.model_copy(
        update={
            "source_path": "content/en/docs/concepts/long.md",
            "url": "https://example.test/docs/concepts/long/",
            "title": "Long",
        }
    )


def as_golden(chunks: list[ChunkRecord]) -> list[dict[str, object]]:
    return [
        {
            "chunk_key": c.chunk_key,
            "ordinal": c.ordinal,
            "header": c.header,
            "body": c.body,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "token_count": c.token_count,
            "section_path": c.section_path,
            "url": c.url,
        }
        for c in chunks
    ]


def check_budget(doc: DocumentRecord, chunks: list[ChunkRecord], limit: int) -> None:
    assert len({c.chunk_key for c in chunks}) == len(chunks)
    for i, c in enumerate(chunks):
        assert c.ordinal == i
        assert c.body == doc.parsed_text[c.char_start : c.char_end]
        assert c.token_count == COUNT(c.embedding_text) <= limit


def test_default_config_reproduces_the_pre_variant_chunks(tmp_path: Path) -> None:
    """The golden file was captured from chunk_document before chunking variants existed:
    the fixture site's pages plus a long synthetic page (splits, a big code block, merges)."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cfg = load_chunking("default")
    assert cfg.version == CHUNKER_VERSION and cfg.max_tokens == MAX_TOKENS
    assert cfg.overlap_tokens == 0 and cfg.header_in_embedding

    def run(doc: DocumentRecord) -> list[dict[str, object]]:
        chunks, _ = chunk_document(
            doc, COUNT, max_tokens=cfg.max_tokens, overlap_tokens=cfg.overlap_tokens
        )
        return as_golden(chunks)

    make_site(tmp_path / "s")
    ctx = ParseContext.load(tmp_path / "s", "1.36", SRC.url_base)
    pages = 0
    for rel, expected in golden.items():
        if rel == "synthetic-long":
            assert run(long_doc()) == expected
            continue
        raw = (tmp_path / "s" / rel).read_text(encoding="utf-8")
        parsed = parse_document(rel, raw, ctx, "content/en/docs")
        assert parsed is not None
        assert run(parsed) == expected
        pages += 1
    assert pages == 4 and sum(len(v) for v in golden.values()) >= 10
    # The comparison is sensitive: another budget gives other chunks.
    assert len(chunk_document(long_doc(), COUNT, max_tokens=256)[0]) > len(golden["synthetic-long"])


def test_variant_configs_have_distinct_versions_and_labels() -> None:
    cfgs = {n: load_chunking(n) for n in ("default", "t256", "o64", "nohdr")}
    assert {n: c.version for n, c in cfgs.items()} == {
        "default": "1",
        "t256": "1-t256",
        "o64": "1-o64",
        "nohdr": "1-nohdr",
    }
    assert cfgs["t256"].max_tokens == 256 and cfgs["o64"].overlap_tokens == 64
    assert not cfgs["nohdr"].header_in_embedding
    assert version_label([SRC]) == "k8s-1.36-chunker1"
    assert version_label([SRC], cfgs["t256"].version) == "k8s-1.36-chunker1-t256"
    assert len({version_label([SRC], c.version) for c in cfgs.values()}) == 4
    with pytest.raises(ValueError, match="invalid chunking name"):
        load_chunking("../corpus")
    with pytest.raises(FileNotFoundError):
        load_chunking("nope")


def test_t256_never_exceeds_256_tokens_with_header() -> None:
    cfg = load_chunking("t256")
    doc = long_doc()
    chunks, _ = chunk_document(doc, COUNT, max_tokens=cfg.max_tokens)
    assert len(chunks) > len(chunk_document(doc, COUNT)[0])
    assert max(c.token_count for c in chunks) <= 256
    check_budget(doc, chunks, 256)


def test_o64_chunks_overlap_by_about_64_tokens_and_spans_stay_true() -> None:
    cfg = load_chunking("o64")
    doc = long_doc()
    base, _ = chunk_document(doc, COUNT)
    chunks, _ = chunk_document(doc, COUNT, overlap_tokens=cfg.overlap_tokens)
    check_budget(doc, chunks, 510)
    overlaps = []
    for prev, cur in itertools.pairwise(chunks):
        if cur.char_start < prev.char_end:
            shared = doc.parsed_text[cur.char_start : prev.char_end]
            assert prev.body.endswith(shared) and cur.body.startswith(shared)
            assert prev.section_path == cur.section_path
            n = COUNT(shared)
            assert n <= 64
            # Either about 64 tokens, or the whole (short) previous chunk.
            assert n >= 56 or cur.char_start == prev.char_start, (n, prev.ordinal)
            overlaps.append(n)
    assert len(overlaps) >= 3 and max(overlaps) == 64
    assert sum(n >= 56 for n in overlaps) >= len(overlaps) - 2
    assert len(chunks) >= len(base)  # room is reserved for the shared text
    assert chunks[0].char_start == base[0].char_start


def test_overlap_never_crosses_a_section_and_respects_the_budget() -> None:
    text = f"## A\n\n{words(70, 'a')}\n\n## B\n\n{words(70, 'b')}\n\n{words(70, 'c')}"
    doc = doc_of(text)
    chunks, _ = chunk_document(doc, COUNT, max_tokens=100, min_body_tokens=1, overlap_tokens=20)
    check_budget(doc, chunks, 100)
    a = [c for c in chunks if c.section_path == ["A"]]
    b = [c for c in chunks if c.section_path == ["B"]]
    assert a and len(b) == 2
    assert a[-1].char_end <= b[0].char_start
    assert b[1].char_start < b[0].char_end
