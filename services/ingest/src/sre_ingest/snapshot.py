"""Corpus snapshots: one Parquet file with documents, chunks and every embedding column.

Layout (format 1): a single table with a `kind` column ("document" or "chunk"); document rows
come first, so a reader can stream the file. Vectors are stored as float16 lists, exactly the
precision of the `halfvec` columns, so a load reproduces them bit for bit. The corpus version's
label, sources, chunker version and embedding models live in the Parquet key-value metadata
under `sre_snapshot`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pgvector import HalfVector

from sre_ingest import store
from sre_shared.embed.base import SPECS

FORMAT = 1
BATCH = 500
VECTOR_COLUMNS = [s.column for s in SPECS.values()]

SCHEMA = pa.schema(
    [
        ("kind", pa.string()),
        ("project", pa.string()),
        ("k8s_version", pa.string()),
        ("source_path", pa.string()),
        ("url", pa.string()),
        ("title", pa.string()),
        ("doc_type", pa.string()),
        ("language", pa.string()),
        ("access_group", pa.string()),
        ("last_modified", pa.timestamp("us", tz="UTC")),
        ("content_hash", pa.string()),
        ("parsed_text", pa.string()),
        ("chunk_key", pa.string()),
        ("ordinal", pa.int32()),
        ("section_path", pa.list_(pa.string())),
        ("header", pa.string()),
        ("body", pa.string()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("token_count", pa.int32()),
        *[(c, pa.list_(pa.float16())) for c in VECTOR_COLUMNS],
    ]
)
DOC_COLS = [
    "project", "k8s_version", "source_path", "url", "title", "doc_type", "language",
    "access_group", "last_modified", "content_hash", "parsed_text",
]  # fmt: skip
CHUNK_COLS = [
    "project", "k8s_version", "source_path", "url", "title", "doc_type", "language",
    "access_group", "chunk_key", "ordinal", "section_path", "header", "body", "char_start",
    "char_end", "token_count",
]  # fmt: skip


def _vec(v: HalfVector | None) -> list[float] | None:
    return None if v is None else [float(x) for x in np.asarray(v.to_numpy(), dtype=np.float16)]


def _table(rows: list[dict[str, Any]]) -> pa.Table:
    cols: dict[str, list[Any]] = {f.name: [r.get(f.name) for r in rows] for f in SCHEMA}
    return pa.table(cols, schema=SCHEMA)


def export_snapshot(
    conn: store.Conn, cv: int, out: Path, models: Sequence[str] | None = None
) -> dict[str, int]:
    """Write the corpus version to `out`. `models` limits the vector columns to those models
    (the others are written as nulls); by default every model the version holds is kept."""
    meta = conn.execute(
        "select label, sources, chunker_version, embedding_models from public.corpus_versions"
        " where id=%s",
        [cv],
    ).fetchone()
    if meta is None:
        raise LookupError(f"corpus version {cv} does not exist")
    held = list(meta["embedding_models"])
    keep = held if models is None else list(models)
    missing = [m for m in keep if m not in held]
    if missing:
        raise ValueError(f"corpus version {cv} has no embeddings for {missing}; it holds {held}")
    kept_columns = {SPECS[m].column for m in keep}  # type: ignore[index]
    info = {
        "format": FORMAT,
        "label": meta["label"],
        "sources": meta["sources"],
        "chunker_version": meta["chunker_version"],
        "embedding_models": keep,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA.with_metadata({"sre_snapshot": json.dumps(info)})
    counts = {"documents": 0, "chunks": 0}
    with pq.ParquetWriter(out, schema, compression="zstd") as writer:
        with conn.cursor(name="snap_docs") as cur:
            cur.execute(
                f"select {', '.join(DOC_COLS)} from public.documents"
                " where corpus_version_id=%s order by id",
                [cv],
            )
            while rows := cur.fetchmany(BATCH):
                writer.write_table(_table([{"kind": "document", **r} for r in rows]).cast(schema))
                counts["documents"] += len(rows)
        with conn.cursor(name="snap_chunks") as cur:
            cur.execute(
                f"select {', '.join(CHUNK_COLS + VECTOR_COLUMNS)} from public.chunks"
                " where corpus_version_id=%s order by id",
                [cv],
            )
            while rows := cur.fetchmany(BATCH):
                batch = [
                    {
                        "kind": "chunk",
                        **{
                            k: (_vec(v) if k in kept_columns else None)
                            if k in VECTOR_COLUMNS
                            else v
                            for k, v in r.items()
                        },
                    }
                    for r in rows
                ]
                writer.write_table(_table(batch).cast(schema))
                counts["chunks"] += len(rows)
    conn.rollback()
    return counts


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH):
        yield from batch.to_pylist()


def load_snapshot(
    conn: store.Conn, path: Path, *, activate: bool = False
) -> tuple[int, dict[str, int]]:
    """Restore a snapshot into a new corpus version (ready, or active with `activate`)."""
    raw = pq.read_schema(path).metadata or {}
    if b"sre_snapshot" not in raw:
        raise ValueError(f"{path} is not an sre-ingest snapshot")
    info = json.loads(raw[b"sre_snapshot"])
    if info["format"] != FORMAT:
        raise ValueError(f"unsupported snapshot format {info['format']}")
    row = conn.execute(
        "insert into public.corpus_versions (label, sources, chunker_version)"
        " values (%s, %s::jsonb, %s) returning id",
        [info["label"], json.dumps(info["sources"]), info["chunker_version"]],
    ).fetchone()
    assert row is not None
    cv = int(row["id"])
    doc_ids: dict[tuple[str, str], int] = {}
    counts = {"documents": 0, "chunks": 0}
    for r in _iter_rows(path):
        if r["kind"] == "document":
            drow = conn.execute(
                "insert into public.documents (corpus_version_id, project, k8s_version,"
                " source_path, url, title, doc_type, language, access_group, last_modified,"
                " content_hash, parsed_text) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " returning id",
                [cv, *[r[c] for c in DOC_COLS]],
            ).fetchone()
            assert drow is not None
            doc_ids[(r["k8s_version"], r["source_path"])] = int(drow["id"])
            counts["documents"] += 1
            continue
        cols = [*CHUNK_COLS, *[c for c in VECTOR_COLUMNS if r[c] is not None]]
        values = [
            HalfVector(np.asarray(r[c], dtype=np.float32)) if c in VECTOR_COLUMNS else r[c]
            for c in cols
        ]
        conn.execute(
            f"insert into public.chunks (corpus_version_id, document_id, {', '.join(cols)})"
            f" values (%s, %s, {', '.join(['%s'] * len(cols))})",
            [cv, doc_ids[(r["k8s_version"], r["source_path"])], *values],
        )
        counts["chunks"] += 1
    conn.execute(
        "update public.corpus_versions set status='ready', embedding_models=%s,"
        " stats=%s::jsonb where id=%s",
        [info["embedding_models"], json.dumps({"snapshot_of": info["label"]}), cv],
    )
    conn.commit()
    if activate:
        store.activate(conn, cv)
    return cv, counts
