"""Postgres access for ingest: corpus versions, documents, chunks, embeddings, state machine.

Synchronous psycopg: ingest is a batch job and does not need the async pool.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector import HalfVector
from psycopg import sql

from sre_shared.config import CorpusSource
from sre_shared.db import connect_sync
from sre_shared.embed.base import SPECS
from sre_shared.models import ChunkRecord, DocumentRecord, EmbeddingModelName

Conn = psycopg.Connection[Any]


class ContentChanged(RuntimeError):
    """A document exists in this corpus version with a different content_hash."""


@contextmanager
def open_conn(dsn: str | None = None) -> Iterator[Conn]:
    conn: Conn = connect_sync(dsn)
    try:
        yield conn
    finally:
        conn.close()


def sources_json(sources: Sequence[CorpusSource]) -> str:
    return json.dumps([s.model_dump(mode="json") for s in sources], sort_keys=True)


def get_or_create_version(
    conn: Conn, label: str, sources: Sequence[CorpusSource], chunker_version: str
) -> tuple[int, bool]:
    """Reuse a building or ready version only when label, sources (SHAs) and chunker_version all
    match; otherwise create one. Returns (id, created). A reused ready version goes back to
    building until the run marks it ready again."""
    blob = sources_json(sources)
    row = conn.execute(
        "select id from public.corpus_versions where status in ('building','ready')"
        " and label = %s and chunker_version = %s and sources = %s::jsonb order by id desc limit 1",
        [label, chunker_version, blob],
    ).fetchone()
    if row is not None:
        cv = int(row["id"])
        conn.execute("update public.corpus_versions set status='building' where id=%s", [cv])
        conn.commit()
        return cv, False
    row = conn.execute(
        "insert into public.corpus_versions (label, sources, chunker_version)"
        " values (%s, %s::jsonb, %s) returning id",
        [label, blob, chunker_version],
    ).fetchone()
    conn.commit()
    assert row is not None
    return int(row["id"]), True


def upsert_document(conn: Conn, cv: int, doc: DocumentRecord) -> tuple[int, bool]:
    """Insert, or return the existing row when the hash matches. Raises on a changed hash."""
    row = conn.execute(
        "select id, content_hash from public.documents where corpus_version_id=%s"
        " and project=%s and k8s_version=%s and source_path=%s",
        [cv, doc.project, doc.k8s_version, doc.source_path],
    ).fetchone()
    if row is not None:
        if row["content_hash"] != doc.content_hash:
            raise ContentChanged(
                f"{doc.k8s_version}:{doc.source_path} changed inside corpus version {cv}; "
                "bump CHUNKER_VERSION (or the parser changed without a version bump)"
            )
        return int(row["id"]), False
    row = conn.execute(
        "insert into public.documents (corpus_version_id, project, k8s_version, source_path, url,"
        " title, doc_type, language, last_modified, access_group, content_hash, parsed_text)"
        " values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id",
        [
            cv, doc.project, doc.k8s_version, doc.source_path, doc.url, doc.title,
            doc.doc_type.value, doc.language, doc.last_modified, doc.access_group,
            doc.content_hash, doc.parsed_text,
        ],
    ).fetchone()  # fmt: skip
    assert row is not None
    return int(row["id"]), True


def upsert_chunks(conn: Conn, cv: int, document_id: int, chunks: Sequence[ChunkRecord]) -> int:
    inserted = 0
    for c in chunks:
        cur = conn.execute(
            "insert into public.chunks (corpus_version_id, document_id, chunk_key, ordinal, project,"
            " k8s_version, doc_type, source_path, url, title, section_path, language, access_group,"
            " header, body, char_start, char_end, token_count)"
            " values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            " on conflict (corpus_version_id, chunk_key) do nothing",
            [
                cv, document_id, c.chunk_key, c.ordinal, c.project, c.k8s_version,
                c.doc_type.value, c.source_path, c.url, c.title, c.section_path, c.language,
                c.access_group, c.header, c.body, c.char_start, c.char_end, c.token_count,
            ],
        )  # fmt: skip
        inserted += cur.rowcount
    return inserted


def pending_embeddings(conn: Conn, cv: int, model: EmbeddingModelName) -> list[tuple[int, str]]:
    """(chunk id, embedding_text) of chunks with no vector for `model` yet."""
    col = sql.Identifier(SPECS[model].column)
    rows = conn.execute(
        sql.SQL(
            "select id, header, body from public.chunks"
            " where corpus_version_id = %s and {} is null order by id"
        ).format(col),
        [cv],
    ).fetchall()
    return [(int(r["id"]), f"{r['header']}\n\n{r['body']}") for r in rows]


def cache_lookup(
    conn: Conn, model: str, revision: str, hashes: Sequence[str]
) -> dict[str, HalfVector]:
    out: dict[str, HalfVector] = {}
    for i in range(0, len(hashes), 2000):
        rows = conn.execute(
            "select text_hash, embedding from public.embedding_cache"
            " where model=%s and model_revision=%s and text_hash = any(%s)",
            [model, revision, list(hashes[i : i + 2000])],
        ).fetchall()
        for r in rows:
            out[r["text_hash"]] = r["embedding"]
    return out


def cache_store(
    conn: Conn, model: str, revision: str, items: Sequence[tuple[str, np.ndarray[Any, Any]]]
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "insert into public.embedding_cache (model, model_revision, text_hash, embedding)"
            " values (%s,%s,%s,%s) on conflict do nothing",
            [(model, revision, h, HalfVector(v)) for h, v in items],
        )


def set_embeddings(
    conn: Conn, model: EmbeddingModelName, items: Sequence[tuple[int, HalfVector]]
) -> None:
    q = sql.SQL("update public.chunks set {} = %s where id = %s").format(
        sql.Identifier(SPECS[model].column)
    )
    with conn.cursor() as cur:
        cur.executemany(q, [(vec, cid) for cid, vec in items])


def mark_ready(
    conn: Conn, cv: int, models: Sequence[str], stats: dict[str, Any], sources: str | None = None
) -> None:
    conn.execute(
        "update public.corpus_versions set status='ready', embedding_models=%s, stats=%s::jsonb"
        " where id=%s and status in ('building','ready')",
        [list(models), json.dumps(stats, default=str), cv],
    )
    conn.commit()


def activate(conn: Conn, cv: int) -> None:
    """ready -> active; the previous active version becomes retired, in one transaction."""
    row = conn.execute(
        "select status from public.corpus_versions where id=%s for update", [cv]
    ).fetchone()
    if row is None:
        raise LookupError(f"corpus version {cv} does not exist")
    if row["status"] != "ready":
        raise ValueError(f"corpus version {cv} is {row['status']}; only 'ready' can be activated")
    conn.execute(
        "update public.corpus_versions set status='retired', retired_at=now() where status='active'"
    )
    conn.execute(
        "update public.corpus_versions set status='active', activated_at=now() where id=%s", [cv]
    )
    conn.commit()


def prune(conn: Conn) -> int:
    cur = conn.execute("delete from public.corpus_versions where status='retired'")
    conn.commit()
    return cur.rowcount


def resolve_version(conn: Conn, ref: str) -> int:
    if ref == "active":
        row = conn.execute("select id from public.corpus_versions where status='active'").fetchone()
        if row is None:
            raise LookupError("no active corpus version")
        return int(row["id"])
    return int(ref)
