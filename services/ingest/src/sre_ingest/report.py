"""Ingest report: counts, token distribution, shortcodes, sizes, timings (JSON under data/reports)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sre_ingest import store
from sre_shared.embed.base import SPECS
from sre_shared.settings import get_settings


def db_stats(conn: store.Conn, cv: int) -> dict[str, Any]:
    docs = conn.execute(
        "select count(*) as n from public.documents where corpus_version_id=%s", [cv]
    ).fetchone()
    toks = conn.execute(
        "select token_count from public.chunks where corpus_version_id=%s", [cv]
    ).fetchall()
    arr = np.array([int(r["token_count"]) for r in toks], dtype=np.int64)
    per_version = conn.execute(
        "select k8s_version, count(distinct document_id) as docs, count(*) as chunks"
        " from public.chunks where corpus_version_id=%s group by k8s_version order by 1",
        [cv],
    ).fetchall()
    sizes: dict[str, Any] = {}
    for spec in SPECS.values():
        row = conn.execute(
            f"select count({spec.column}) as n, coalesce(sum(pg_column_size({spec.column})), 0) as b"
            " from public.chunks where corpus_version_id=%s",
            [cv],
        ).fetchone()
        assert row is not None
        sizes[spec.column] = {"vectors": int(row["n"]), "bytes": int(row["b"])}
    return {
        "documents": int(docs["n"]) if docs else 0,
        "chunks": int(arr.size),
        "per_version": [dict(r) for r in per_version],
        "tokens": {
            "p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
            "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "max": int(arr.max()) if arr.size else 0,
        },
        "embedding_column_bytes": sizes,
    }


def build_report(conn: store.Conn, cv: int) -> dict[str, Any]:
    """DB-derived figures plus the parse/timing figures saved in corpus_versions.stats."""
    row = conn.execute(
        "select label, status, chunker_version, sources, embedding_models, stats"
        " from public.corpus_versions where id=%s",
        [cv],
    ).fetchone()
    if row is None:
        raise LookupError(f"corpus version {cv} does not exist")
    saved: dict[str, Any] = dict(row["stats"] or {})
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_version_id": cv,
        "label": row["label"],
        "status": row["status"],
        "chunker_version": row["chunker_version"],
        "sources": row["sources"],
        "embedding_models": row["embedding_models"],
        **db_stats(conn, cv),
        "oversized_splits": saved.get("oversized_splits", {}),
        "merged_chunks": saved.get("merged_chunks", 0),
        "shortcodes": saved.get("shortcodes", {}),
        "unknown_shortcodes": saved.get("unknown_shortcodes", {}),
        "parse_problems": saved.get("parse_problems", {}),
        "embedding": saved.get("embedding", {}),
        "stage_seconds": saved.get("stage_seconds", {}),
        "last_run": saved.get("last_run", {}),
    }


def write_report(report: dict[str, Any], data_dir: Path | None = None) -> Path:
    out_dir = (data_dir or get_settings().data_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"ingest-cv{report['corpus_version_id']}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
