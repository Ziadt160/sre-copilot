"""fetch -> parse -> chunk -> embed -> upsert -> ready (-> activate)."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sre_ingest import sources, store
from sre_ingest.chunker import CHUNKER_VERSION, ChunkStats, chunk_document
from sre_ingest.embedding import EmbedStats, embed_pending
from sre_ingest.parser import ParseContext, ParseStats, parse_document
from sre_shared.config import CorpusConfig, CorpusSource
from sre_shared.embed import get_embedder
from sre_shared.embed.base import SPECS, Embedder
from sre_shared.embed.remote import GpuServerUnavailable
from sre_shared.models import EmbeddingModelName

Log = Callable[[str], None]


@dataclass
class RunResult:
    corpus_version_id: int
    created_version: bool
    documents_seen: int = 0
    documents_inserted: int = 0
    chunks_inserted: int = 0
    chunks_total: int = 0
    skipped_models: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def list_markdown(root: Path, docs_root: str) -> list[str]:
    base = root / docs_root
    return sorted(p.relative_to(root).as_posix() for p in base.rglob("*.md") if p.is_file())


def sample_stride(paths: list[str], limit: int | None) -> list[str]:
    """Deterministic, evenly spread subset (so a small run covers concepts, tasks and reference)."""
    if limit is None or limit >= len(paths):
        return paths
    return [paths[(i * len(paths)) // limit] for i in range(limit)]


def version_label(sources_: Sequence[CorpusSource]) -> str:
    return "k8s-" + "+".join(s.k8s_version for s in sources_) + f"-chunker{CHUNKER_VERSION}"


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - t0


def load_embedders(
    models: Sequence[EmbeddingModelName], log: Log
) -> tuple[dict[EmbeddingModelName, Embedder], dict[str, str]]:
    out: dict[EmbeddingModelName, Embedder] = {}
    skipped: dict[str, str] = {}
    for m in models:
        try:
            emb = get_embedder(m)
            health = getattr(emb, "health", None)
            if callable(health):
                health()
            out[m] = emb
        except GpuServerUnavailable as e:
            skipped[m] = str(e)
            log(f"SKIP {m}: {e}")
    return out, skipped


def run_ingest(
    cfg: CorpusConfig,
    *,
    dsn: str | None = None,
    versions: Sequence[str] | None = None,
    models: Sequence[EmbeddingModelName] = ("gte-small",),
    limit_docs: int | None = None,
    activate: bool = False,
    do_fetch: bool = True,
    data_dir: Path | None = None,
    embedders: Mapping[EmbeddingModelName, Embedder] | None = None,
    token_counter: Embedder | None = None,
    log: Log = print,
) -> RunResult:
    selected = [s for s in cfg.versions if versions is None or s.k8s_version in versions]
    if not selected:
        raise ValueError(f"no pinned versions match {versions}; run `sre-ingest pin` first")
    timer = Timer()
    skipped: dict[str, str] = {}
    if embedders is None:
        with timer.stage("load_models"):
            embedders, skipped = load_embedders(models, log)
    counter = token_counter or (embedders.get("gte-small") if embedders else None)
    if counter is None:
        counter = get_embedder("gte-small")

    parse_stats = ParseStats()
    chunk_totals = ChunkStats()
    split_counts = {"code_blocks": 0, "prose": 0}
    with store.open_conn(dsn) as conn:
        cv, created = store.get_or_create_version(
            conn, version_label(selected), selected, CHUNKER_VERSION
        )
        result = RunResult(cv, created, skipped_models=skipped)
        for src in selected:
            with timer.stage("fetch"):
                root = (
                    sources.fetch(cfg, src, data_dir)
                    if do_fetch
                    else sources.source_dir(src, data_dir)
                )
            ctx = ParseContext.load(root, src.k8s_version, src.url_base)
            paths = sample_stride(list_markdown(root, cfg.docs_root), limit_docs)
            log(f"v{src.k8s_version}: {len(paths)} pages from {root}")
            for rel in paths:
                with timer.stage("parse"):
                    raw = (root / rel).read_text(encoding="utf-8", errors="replace")
                    doc = parse_document(rel, raw, ctx, cfg.docs_root)
                if doc is None:
                    continue
                with timer.stage("chunk"):
                    chunks, cstats = chunk_document(doc, counter.count_tokens)
                with timer.stage("upsert"):
                    doc_id, new_doc = store.upsert_document(conn, cv, doc)
                    result.chunks_inserted += store.upsert_chunks(conn, cv, doc_id, chunks)
                    conn.commit()
                result.documents_seen += 1
                result.documents_inserted += int(new_doc)
                split_counts["code_blocks"] += cstats.code_block_splits
                split_counts["prose"] += cstats.prose_splits
                chunk_totals.merged += cstats.merged
            parse_stats.merge(ctx.stats)

        embed_stats: dict[str, EmbedStats] = {}
        for name, emb in embedders.items():
            with timer.stage(f"embed_{name}"):
                embed_stats[name] = embed_pending(conn, cv, name, emb)
            log(f"embedded {name}: {embed_stats[name].as_dict()}")

        row = conn.execute(
            "select count(*) as n from public.chunks where corpus_version_id=%s", [cv]
        ).fetchone()
        result.chunks_total = int(row["n"]) if row else 0
        done_models = (
            [n for n in SPECS if not store.pending_embeddings(conn, cv, n)]
            if result.chunks_total
            else []
        )
        result.stats = {
            "oversized_splits": split_counts,
            "merged_chunks": chunk_totals.merged,
            "shortcodes": {
                "handled": dict(parse_stats.handled),
                "generated_dropped": dict(parse_stats.generated),
            },
            "unknown_shortcodes": dict(parse_stats.unknown),
            "parse_problems": dict(parse_stats.problems),
            "embedding": {k: v.as_dict() for k, v in embed_stats.items()},
            "skipped_models": skipped,
            "stage_seconds": {k: round(v, 2) for k, v in timer.seconds.items()},
            "last_run": {
                "documents_seen": result.documents_seen,
                "documents_inserted": result.documents_inserted,
                "chunks_inserted": result.chunks_inserted,
                "limit_docs": limit_docs,
            },
        }
        store.mark_ready(conn, cv, done_models, result.stats)
        if activate:
            store.activate(conn, cv)
    return result
