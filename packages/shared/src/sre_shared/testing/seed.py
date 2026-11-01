"""Insert fixture documents and chunks for tests, without the ingest pipeline.

Writes the same tables and columns ingest does, so retrieval, golden and eval tests can run
against realistic rows. Not for production use: no idempotency, no blue/green.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from pgvector import HalfVector
from psycopg import AsyncConnection

from sre_shared.embed.base import SPECS, Embedder
from sre_shared.models import DocType, EmbeddingModelName


@dataclass
class FixtureDoc:
    source_path: str
    k8s_version: str
    title: str
    sections: Sequence[tuple[Sequence[str], str]]
    """(section_path, body) per chunk, in order."""
    doc_type: DocType = DocType.CONCEPT
    access_group: str = "public"
    url: str | None = None
    project: str = "kubernetes"
    extra: dict[str, str] = field(default_factory=dict)


def header_for(title: str, section_path: Sequence[str], k8s_version: str) -> str:
    trail = " > ".join([title, *section_path]) if section_path else title
    return f"Kubernetes v{k8s_version} | {trail}"


async def create_corpus_version(
    conn: AsyncConnection, *, label: str = "test", status: str = "active"
) -> int:
    row = await (
        await conn.execute(
            "insert into public.corpus_versions (label, status, chunker_version, embedding_models)"
            " values (%s, %s, 'test', '{}') returning id",
            [label, status],
        )
    ).fetchone()
    assert row is not None
    return int(row["id"] if isinstance(row, dict) else row[0])


async def insert_docs(
    conn: AsyncConnection,
    docs: Sequence[FixtureDoc],
    *,
    corpus_version_id: int,
    embedders: Sequence[Embedder] = (),
) -> dict[str, list[int]]:
    """Insert documents and their chunks; embed with each embedder into its column.

    Returns {source_path@version: [chunk ids in order]}.
    """
    ids: dict[str, list[int]] = {}
    for doc in docs:
        bodies = [body for _, body in doc.sections]
        parsed_text = "\n\n".join(bodies)
        url = doc.url or f"https://kubernetes.io/{doc.source_path.removesuffix('.md')}/"
        drow = await (
            await conn.execute(
                "insert into public.documents (corpus_version_id, project, k8s_version, source_path,"
                " url, title, doc_type, access_group, content_hash, parsed_text)"
                " values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id",
                [
                    corpus_version_id,
                    doc.project,
                    doc.k8s_version,
                    doc.source_path,
                    url,
                    doc.title,
                    doc.doc_type.value,
                    doc.access_group,
                    hashlib.sha256(parsed_text.encode()).hexdigest(),
                    parsed_text,
                ],
            )
        ).fetchone()
        assert drow is not None
        document_id = int(drow["id"] if isinstance(drow, dict) else drow[0])

        headers = [header_for(doc.title, sp, doc.k8s_version) for sp, _ in doc.sections]
        texts = [f"{h}\n\n{b}" for h, b in zip(headers, bodies, strict=True)]
        vectors: dict[EmbeddingModelName, list[HalfVector]] = {
            e.spec.name: [HalfVector(v) for v in e.embed_documents(texts)] for e in embedders
        }
        chunk_ids: list[int] = []
        offset = 0
        for i, ((section_path, body), header) in enumerate(zip(doc.sections, headers, strict=True)):
            start = parsed_text.index(body, offset)
            offset = start + len(body)
            key = hashlib.sha256(f"{doc.source_path}|{doc.k8s_version}|{body}".encode()).hexdigest()
            cols = {
                "corpus_version_id": corpus_version_id,
                "document_id": document_id,
                "chunk_key": key,
                "ordinal": i,
                "project": doc.project,
                "k8s_version": doc.k8s_version,
                "doc_type": doc.doc_type.value,
                "source_path": doc.source_path,
                "url": url,
                "title": doc.title,
                "section_path": list(section_path),
                "access_group": doc.access_group,
                "header": header,
                "body": body,
                "char_start": start,
                "char_end": offset,
                "token_count": len(body.split()),
            }
            for name, vecs in vectors.items():
                cols[SPECS[name].column] = vecs[i]
            names = ", ".join(cols)
            marks = ", ".join(["%s"] * len(cols))
            crow = await (
                await conn.execute(
                    f"insert into public.chunks ({names}) values ({marks}) returning id",
                    list(cols.values()),
                )
            ).fetchone()
            assert crow is not None
            chunk_ids.append(int(crow["id"] if isinstance(crow, dict) else crow[0]))
        ids[f"{doc.source_path}@{doc.k8s_version}"] = chunk_ids
    return ids
